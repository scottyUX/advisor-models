"""Evaluation script for SWE-Smith advisor setup.

Evaluates the advisor training setup from env.py where advisor provides guidance
every 5 agent turns. Supports both API models and local models served via vLLM.

Key differences from training:
- Training: Initial advisor prompt is built in dataset, first step() gets policy response
- Eval: Explicitly calls advisor with initial prompt before loop for debugging visibility

Both approaches are functionally equivalent - the advisor sees the same initial prompt
and provides guidance before agent execution begins.

Local Model Support:
- Automatically detects local model paths and starts vLLM server
- Supports HuggingFace format models (served directly)
- Supports training checkpoints (automatically processes FSDP shards to HF format)
- Cleans up vLLM server and temporary files after evaluation

Usage:
    # API model (default)
    python -m advisor_models.swe_smith.eval_advisor \
        --data_file data/swe_smith/validation_jd__tenacity.json \
        --num_samples 1 \
        --advisor_model anthropic/claude-sonnet-4-5-20250929 \
        --agent_model gemini/gemini-2.5-flash-lite \
        --max_workers 20 \
        --session_mapping_file session_mappings/advisor_test.json

    # Local HuggingFace model
    python -m advisor_models.swe_smith.eval_advisor \
        --data_file data/swe_smith/validation_jd__tenacity.json \
        --num_samples 49 \
        --advisor_model Qwen/Qwen3-8B \
        --is_local \
        --is_hf_model \
        --agent_model gemini/gemini-2.5-flash \
        --tensor_parallel_size 4 \
        --max_model_len 32768 \
        --max_workers 20 \
        --no_think

    # Local training checkpoint (automatically processes FSDP shards)
    python -m advisor_models.swe_smith.eval_advisor \
        --data_file data/swe_smith/validation_jd__tenacity.json \
        --num_samples 100 \
        --advisor_model ~/checkpoints/advisor_model_step_200/ \
        --is_local \
        --agent_model gemini/gemini-3-pro-preview \
        --tensor_parallel_size 4 \
        --max_model_len 32768 \
        --max_workers 50 \
        --step_num 200 \
        --num_runs 5

    # Hosted vLLM server
    python -m advisor_models.swe_smith.eval_advisor \
        --data_file data/swe_smith/validation_jd__tenacity.json \
        --num_samples 49 \
        --advisor_model hosted_vllm/Qwen/Qwen2.5-7B-Instruct \
        --agent_model gemini/gemini-2.5-flash \
        --max_workers 20 \
        --session_mapping_file session_mappings/advisor_eval.json \
        --api_base http://your-server:8000/v1/
"""

import argparse
import json
import sys
import os
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

from .config import compute_score, ADVISOR_TEMPLATE
from .remote_agent_client import RemoteAgentClient
from utils.eval_utils import compute_multi_run_statistics, format_ci_string

litellm.drop_params = True


def prepare_local_model(
    model_path: str, step_num: Optional[int] = None, is_hf_model: bool = False
) -> tuple[str, Optional[str]]:
    """Prepare local model for serving with vLLM.

    Args:
        model_path: Path to the model (HF format or training checkpoint) or HF model identifier
        step_num: Optional specific step number for training checkpoints
        is_hf_model: If True, model_path is a HuggingFace model identifier (e.g., 'Qwen/Qwen3-8B')

    Returns:
        Tuple of (path_to_serve, temp_dir_to_cleanup)
        temp_dir_to_cleanup is None if no cleanup needed
    """
    from utils.upload_model_to_gcp import process_skyrl_model, is_skyrl_checkpoint

    # If it's a HuggingFace model identifier, vLLM will download it automatically
    if is_hf_model:
        print(f"Using HuggingFace model identifier: {model_path}")
        print("vLLM will download the model if not already cached")
        return model_path, None

    # For local paths, expand and check existence
    model_path = os.path.abspath(os.path.expanduser(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Check if this is a training checkpoint
    if is_skyrl_checkpoint(model_path):
        print(f"Detected training checkpoint at {model_path}")
        if step_num is not None:
            print(
                f"Processing FSDP shards for step {step_num} to HuggingFace format..."
            )
        else:
            print("Processing FSDP shards to HuggingFace format...")
        temp_dir = process_skyrl_model(model_path, step_num=step_num)
        print(f"Processed model saved to temporary directory: {temp_dir}")
        return temp_dir, temp_dir
    else:
        print(f"Using local HuggingFace model at {model_path}")
        return model_path, None


def clean_error_message(error: Exception, max_length: int = 300) -> str:
    """Clean and truncate error messages."""
    error_str = str(error)

    # Remove repeated patterns
    lines = error_str.split("\n")
    cleaned_lines = []
    prev_line = None
    repeat_count = 0

    for line in lines:
        if line == prev_line:
            repeat_count += 1
        else:
            if repeat_count > 0 and repeat_count < 3:
                cleaned_lines.extend([prev_line] * repeat_count)
            elif repeat_count >= 3:
                cleaned_lines.append(f"... (repeated {repeat_count} times)")
            repeat_count = 0
            prev_line = line
            cleaned_lines.append(line)

    # Add final repeat count if needed
    if repeat_count > 0 and repeat_count < 3:
        cleaned_lines.extend([prev_line] * repeat_count)
    elif repeat_count >= 3:
        cleaned_lines.append(f"... (repeated {repeat_count} times)")

    cleaned_str = "\n".join(cleaned_lines)

    # Truncate if too long
    if len(cleaned_str) > max_length:
        cleaned_str = cleaned_str[:max_length] + "\n... (truncated)"

    error_type = type(error).__name__
    return f"[{error_type}] {cleaned_str}"


# The served advisor context is 32768. A 4096-token reply reserve plus the
# stacked transcript was enough to overflow it (34887 requested). Advice is a
# short comment, and each turn only needs the latest observation.
ADVISOR_MAX_TOKENS = 1024
ADVISOR_OBSERVATION_CHARS = 60_000


def _tail_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"[earlier context omitted, {omitted} characters]\n{text[-limit:]}"


@dataclass
class AdvisorEvalResult:
    """Single evaluation result with advisor."""

    instance_id: str
    problem_statement: str
    repo: str
    generated_patch: str
    ground_truth_patch: str
    resolved: bool
    status: str
    total_steps: int
    advisor_calls: int
    advisor_guidance_history: List[str]
    agent_cost: float
    advisor_cost: float
    session_id: str = None
    eval_id: str = None
    error: str = None


class AdvisorDebugger:
    """Debug evaluator that mimics advisor training setup using remote agent."""

    def __init__(
        self,
        advisor_model: str,
        agent_model: str,
        is_local: bool = False,
        tensor_parallel_size: int = 4,
        max_model_len: int = 32768,
        step_num: Optional[int] = None,
        no_think: bool = False,
        api_base: Optional[str] = None,
        is_hf_model: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        truncate_student_thinking: bool = False,
    ):
        """Initialize the debugger with advisor and agent models.

        Args:
            advisor_model: Model to use for advisor (e.g., gpt-4o-mini or local path)
            agent_model: Model to use for agent (e.g., gpt-4.1-mini)
            is_local: Whether the advisor model is a local model that needs vLLM serving
            tensor_parallel_size: Number of GPUs for tensor parallelism (for vLLM)
            max_model_len: Maximum model length for vLLM server
            step_num: Optional specific step number for training checkpoints
            no_think: Whether to disable thinking mode in chat template
            api_base: Optional API base URL for remote vllm server. Overridden if is_local is True
            is_hf_model: Whether advisor_model is a HuggingFace model identifier (not a local path)
            temperature: Temperature for advisor model sampling
            top_p: Top-p for advisor model sampling
            top_k: Top-k for advisor model sampling
            min_p: Min-p for advisor model sampling
            truncate_student_thinking: Whether to strip student thinking from observations
        """
        print(f"Advisor model: {advisor_model}")
        print(f"Agent model: {agent_model}")
        sys.stdout.flush()

        self.original_advisor_model = advisor_model
        self.agent_model = agent_model
        self.summarizer_model = "gpt-4o-mini"
        self.no_think = no_think
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.truncate_student_thinking = truncate_student_thinking
        litellm.drop_params = True
        if api_base:
            litellm.api_base = api_base

        # vLLM server management
        self.vllm_process: Optional[subprocess.Popen] = None
        self.temp_model_dir: Optional[str] = None

        # Check if advisor model is local and needs vLLM
        if is_local:
            print(f"\nDetected local advisor model: {advisor_model}")
            print("Preparing model for vLLM serving...")

            # Prepare the model (process training checkpoint if needed)
            model_path_to_serve, temp_dir = prepare_local_model(
                advisor_model, step_num=step_num, is_hf_model=is_hf_model
            )
            self.temp_model_dir = temp_dir

            # utils.vllm is not in the repo; import only on the --is_local path
            # so the hosted_vllm and student-only paths load without it.
            from utils.vllm import start_vllm_server

            # Start vLLM server
            print(f"\nStarting vLLM server for model: {model_path_to_serve}")
            served_model_name = "local-advisor-model"
            self.vllm_process = start_vllm_server(
                model_to_serve_name=model_path_to_serve,
                served_model_name=served_model_name,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
            )

            # Update advisor_model to point to the vLLM server
            self.advisor_model = f"openai/{served_model_name}"

            # Configure litellm to use local vLLM server
            litellm.api_base = "http://127.0.0.1:8000/v1"
            print(f"vLLM server started. Using model name: {self.advisor_model}")
        else:
            print(f"Using API model for advisor: {advisor_model}")
            self.advisor_model = advisor_model

        # Agent configuration (matching env.py)
        self.max_steps_per_advisor_call = 5

        # Remote agent client (matching env.py)
        self.remote_agent = RemoteAgentClient()

    def _complete_advisor(self, messages: List[Dict[str, str]]) -> tuple[str, float]:
        """One advisor call. The reply reserve stays small so the 32768 window fits."""
        extra_body = {}
        if self.no_think:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        response = litellm.completion(
            model=self.advisor_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=ADVISOR_MAX_TOKENS,
            extra_body=extra_body,
        )
        guidance = response.choices[0].message.content
        advisor_cost = response._hidden_params.get("response_cost") or 0.0
        return guidance, advisor_cost

    def get_initial_advisor_guidance(
        self,
        problem_statement: str,
        repo: str,
        advisor_messages: List[Dict[str, str]],
    ) -> tuple[str, float]:
        """Get initial guidance from advisor model.

        Args:
            problem_statement: The problem to solve
            repo: Repository name
            advisor_messages: Holds the task prompt only. Later turns do not
                append observations onto this list.

        Returns:
            Tuple of (guidance_text, cost)
        """
        prompt_text = ADVISOR_TEMPLATE.format(
            problem_statement=problem_statement,
            repo=repo,
        )
        advisor_messages.append({"role": "user", "content": prompt_text})
        try:
            return self._complete_advisor(advisor_messages)
        except Exception as e:
            raise Exception(f"Error getting initial advisor guidance: {e}")

    def run_agent_with_advisor(
        self, instance: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Run agent with advisor guidance every 5 steps using remote agent.

        Args:
            instance: Full SWE-Smith instance dict

        Returns:
            Tuple of (patch_string, info_dict)
        """
        session_id = None
        try:
            # Extract required fields
            repo_name = instance["repo"]
            problem_statement = instance["problem_statement"]
            instance_id = instance["instance_id"]

            print(f"\n{'=' * 60}")
            print("Running agent with advisor (remote)")
            print(f"Instance: {instance_id}")
            print(f"Repo: {repo_name}")
            print(f"{'=' * 60}\n")

            # Create remote agent session with specified agent model (matching env.py init)
            session_id = self.remote_agent.create_session(
                instance, model_name=self.agent_model
            )
            print(
                f"Created remote session: {session_id} with agent model {self.agent_model}"
            )

            # Track state
            agent_terminated = False
            total_steps = 0
            advisor_calls = 0
            advisor_guidance_history = []
            advisor_total_cost = 0.0

            # Local state for this problem (thread-safe)
            advisor_messages = []

            # Initial advisor guidance
            sys.stdout.flush()
            guidance, advisor_cost = self.get_initial_advisor_guidance(
                problem_statement, repo_name, advisor_messages
            )
            if advisor_cost is not None:
                advisor_total_cost += advisor_cost
            advisor_calls += 1
            advisor_guidance_history.append(guidance)

            sys.stdout.flush()

            # Main loop: run agent with periodic advisor guidance
            while not agent_terminated:
                # Parse out think block if present (matching env.py step)
                advisor_feedback = guidance
                if "</think>" in advisor_feedback:
                    advisor_feedback = advisor_feedback.split("</think>")[1].strip()

                # Execute remotely (matching env.py step)
                result = self.remote_agent.execute_step(
                    session_id,
                    advisor_feedback,
                    max_steps=self.max_steps_per_advisor_call,
                )

                # A failed session (dead key, empty balance, rate limit) is an
                # error, not an unresolved attempt; same as eval_swe_smith.
                if result.get("error"):
                    raise RuntimeError(f"Agent session failed: {result['error']}")

                # Parse result (matching env.py step)
                if result.get("terminated"):
                    agent_terminated = True

                total_steps = result.get("total_steps", 0)

                # If not terminated, get next advisor guidance
                if not agent_terminated:
                    # Get observation from remote agent (already formatted with rolling summary)
                    observation_text = result.get("observation", "")

                    # Optionally truncate student thinking from observation
                    if self.truncate_student_thinking:
                        observation_text = self._truncate_thinking_from_observation(
                            observation_text
                        )

                    # Task prompt, the previous advice, and this turn's observation.
                    # Earlier observations stay out so the transcript cannot grow
                    # past the 32768 context.
                    request_messages = [
                        advisor_messages[0],
                        {"role": "assistant", "content": guidance},
                        {
                            "role": "user",
                            "content": _tail_text(
                                observation_text, ADVISOR_OBSERVATION_CHARS
                            ),
                        },
                    ]
                    try:
                        guidance, advisor_cost = self._complete_advisor(request_messages)
                    except Exception as e:
                        raise Exception(f"Error getting advisor guidance: {e}")

                    if advisor_cost is not None:
                        advisor_total_cost += advisor_cost
                    advisor_calls += 1
                    advisor_guidance_history.append(guidance)

            # Get patch from remote session (matching env.py _return_final_result)
            patch = self.remote_agent.get_patch(session_id)

            # Save session_id before cleanup
            completed_session_id = session_id

            # Cleanup remote session
            print(f"Cleaning up remote session: {session_id}")
            sys.stdout.flush()
            self.remote_agent.cleanup_session(session_id)
            session_id = None

            info = {
                "status": "completed",
                "total_steps": total_steps,
                "advisor_calls": advisor_calls,
                "agent_cost": result.get("cost", 0.0),
                "advisor_cost": advisor_total_cost,
                "total_cost": result.get("cost", 0.0) + advisor_total_cost,
                "advisor_guidance_history": advisor_guidance_history,
                "session_id": completed_session_id,
            }

            return patch, info

        except Exception as e:
            print(f"Error running agent with advisor: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()

            # Cleanup on error
            if session_id:
                try:
                    self.remote_agent.cleanup_session(session_id)
                except Exception:
                    pass

            return "", {
                "status": "error",
                "message": str(e),
                "total_steps": 0,
                "advisor_calls": 0,
                "agent_cost": 0.0,
                "advisor_cost": 0.0,
                "total_cost": 0.0,
                "advisor_guidance_history": [],
                "session_id": session_id,
            }

    def evaluate_single(self, instance: Dict[str, Any]) -> AdvisorEvalResult:
        """Evaluate a single instance with advisor."""
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        repo = instance["repo"]
        ground_truth_patch = instance["patch"]

        print(f"\n{'=' * 80}")
        print(f"Evaluating with Advisor: {instance_id}")
        print(f"Repo: {repo}")
        print(f"{'=' * 80}")
        sys.stdout.flush()

        try:
            # Run agent with advisor
            generated_patch, info = self.run_agent_with_advisor(instance)

            if info.get("status", "error") == "error":
                return AdvisorEvalResult(
                    instance_id=instance_id,
                    problem_statement=problem_statement,
                    repo=repo,
                    generated_patch="",
                    ground_truth_patch=ground_truth_patch,
                    resolved=False,
                    status="error",
                    total_steps=0,
                    advisor_calls=0,
                    advisor_guidance_history=[],
                    agent_cost=0.0,
                    advisor_cost=0.0,
                    session_id=None,
                    eval_id=None,
                    error=info.get("message", "Unknown advisor error"),
                )

            # Evaluate patch
            reward, eval_id, eval_info = compute_score(generated_patch, instance)

            print(f"\nEvaluation: {eval_info}")
            sys.stdout.flush()

            # Parse results
            resolved = reward > 0.5
            status = (
                eval_info.split("Status: ")[1].split(",")[0]
                if "Status:" in eval_info
                else "unknown"
            )

            result_str = "RESOLVED" if resolved else "NOT RESOLVED"
            print(f"Result: {result_str} (status: {status})")
            print(
                f"Steps: {info['total_steps']}, Advisor calls: {info['advisor_calls']}"
            )
            print(
                f"Agent cost: ${info['agent_cost']:.4f}, Advisor cost: ${info['advisor_cost']:.4f}, Total: ${info['total_cost']:.4f}"
            )
            sys.stdout.flush()

            return AdvisorEvalResult(
                instance_id=instance_id,
                problem_statement=problem_statement,
                repo=repo,
                generated_patch=generated_patch,
                ground_truth_patch=ground_truth_patch,
                resolved=resolved,
                status=f"{status} (steps: {info['total_steps']}, advisor_calls: {info['advisor_calls']})",
                total_steps=info["total_steps"],
                advisor_calls=info["advisor_calls"],
                advisor_guidance_history=info["advisor_guidance_history"],
                agent_cost=info["agent_cost"],
                advisor_cost=info["advisor_cost"],
                session_id=info.get("session_id"),
                eval_id=eval_id,
            )

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()

            return AdvisorEvalResult(
                instance_id=instance_id,
                problem_statement=problem_statement,
                repo=repo,
                generated_patch="",
                ground_truth_patch=ground_truth_patch,
                resolved=False,
                status="error",
                total_steps=0,
                advisor_calls=0,
                advisor_guidance_history=[],
                agent_cost=0.0,
                advisor_cost=0.0,
                session_id=None,
                eval_id=None,
                error=str(e),
            )

    def _truncate_thinking_from_observation(self, observation_text: str) -> str:
        """Strip student thinking from observation text.

        The mini-swe-agent returns messages in format:
        [assistant]: THOUGHT: reasoning here\n\n```bash\ncommand\n```

        This function removes the THOUGHT section and keeps only the bash command.

        Args:
            observation_text: Raw observation text from agent

        Returns:
            Observation text with thinking removed
        """
        import re

        # Pattern to match [assistant]: followed by content that may span multiple lines
        # until we hit the next [role]: or end of string
        def replace_assistant_message(match):
            full_match = match.group(0)
            content = match.group(1)

            # Extract just the bash code block from content
            if "```bash" in content:
                bash_start = content.find("```bash")
                bash_end = content.find("```", bash_start + 7)

                if bash_start != -1 and bash_end != -1:
                    # Extract just the bash code block
                    bash_block = content[bash_start : bash_end + 3]
                    return f"[assistant]: {bash_block}"

            # If no bash block found or malformed, keep original
            return full_match

        # Pattern matches [assistant]: followed by everything until next [role]: or end
        pattern = r"\[assistant\]:\s*(.*?)(?=\n\n\[|\n\nLook at the prior|$)"

        result = re.sub(
            pattern, replace_assistant_message, observation_text, flags=re.DOTALL
        )

        return result

    def cleanup(self):
        """Cleanup resources (vLLM server and temporary directories)."""
        if self.vllm_process:
            print("\nStopping vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
                print("vLLM server stopped successfully")
            except subprocess.TimeoutExpired:
                print("vLLM server did not stop gracefully, killing...")
                self.vllm_process.kill()
                self.vllm_process.wait()
            self.vllm_process = None

        if self.temp_model_dir and os.path.exists(self.temp_model_dir):
            print(f"Cleaning up temporary model directory: {self.temp_model_dir}")
            shutil.rmtree(self.temp_model_dir)
            self.temp_model_dir = None

    def evaluate_dataset(
        self, instances: List[Dict[str, Any]], max_workers: int = 1
    ) -> List[AdvisorEvalResult]:
        """Evaluate multiple instances with optional parallelization.

        Args:
            instances: List of problem instances to evaluate
            max_workers: Number of parallel workers (1 = sequential)

        Returns:
            List of evaluation results
        """
        if max_workers == 1:
            # Sequential execution
            results = []
            for instance in tqdm(instances, desc="Evaluating with advisor"):
                result = self.evaluate_single(instance)
                results.append(result)
            return results

        # Parallel execution
        results = [None] * len(instances)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.evaluate_single, instance): idx
                for idx, instance in enumerate(instances)
            }

            # Collect results with progress bar
            with tqdm(total=len(instances), desc="Evaluating with advisor") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"\nError evaluating instance {idx}: {e}")
                        import traceback

                        traceback.print_exc()
                        # Create error result
                        instance = instances[idx]
                        results[idx] = AdvisorEvalResult(
                            instance_id=instance.get("instance_id", "unknown"),
                            problem_statement=instance.get("problem_statement", ""),
                            repo=instance.get("repo", ""),
                            generated_patch="",
                            ground_truth_patch=instance.get("patch", ""),
                            resolved=False,
                            status="error",
                            total_steps=0,
                            advisor_calls=0,
                            advisor_guidance_history=[],
                            agent_cost=0.0,
                            advisor_cost=0.0,
                            session_id=None,
                            error=str(e),
                        )
                    pbar.update(1)

        return results


def save_session_mapping(results: List[AdvisorEvalResult], output_file: str):
    """Save per-instance results, keyed by instance_id.

    Every instance gets a row, including errored ones with no session_id,
    so a failed session is visible to later analysis instead of missing.

    Args:
        results: List of evaluation results
        output_file: Path to save the mapping JSON file
    """
    mapping = {}
    for r in results:
        mapping[r.instance_id] = {
            "instance_id": r.instance_id,
            "session_id": r.session_id,
            "eval_id": r.eval_id,
            "diff": r.generated_patch,
            "resolved": r.resolved,
            "status": r.status,
            "total_steps": r.total_steps,
            "agent_cost": r.agent_cost,
            "advisor_cost": r.advisor_cost,
            "advisor_calls": r.advisor_calls,
            "advisor_guidance_history": r.advisor_guidance_history,
            "error": r.error,
        }

    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)

    errors = sum(1 for row in mapping.values() if row["error"])
    print(f"\nSession mapping saved to: {output_file}")
    print(f"Total instances tracked: {len(mapping)} ({errors} errored)")


def print_summary(
    results: List[AdvisorEvalResult],
    aggregate_stats: Optional[Dict[str, Any]] = None,
):
    """Print evaluation summary."""
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    errors = sum(1 for r in results if r.error)

    # Calculate overall averages (for single-run mode or cost reporting)
    avg_steps = sum(r.total_steps for r in results) / total if total > 0 else 0
    avg_advisor_calls = (
        sum(r.advisor_calls for r in results) / total if total > 0 else 0
    )
    avg_agent_cost = sum(r.agent_cost for r in results) / total if total > 0 else 0
    avg_advisor_cost = sum(r.advisor_cost for r in results) / total if total > 0 else 0
    total_cost = sum(r.agent_cost + r.advisor_cost for r in results)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY (WITH ADVISOR)")
    print("=" * 80)

    # Print multi-run statistics if available
    if aggregate_stats:
        print(f"\nNumber of runs: {aggregate_stats['resolve_rate']['n']}")
        print(f"Total instances per run: {total}")
        print(f"{format_ci_string(aggregate_stats['resolve_rate'], 'Resolve Rate')}")
        print(f"{format_ci_string(aggregate_stats['step_count'], 'Average Steps')}")
        print(
            f"{format_ci_string(aggregate_stats['correct_step_count'], 'Average Steps (Correct)')}"
        )
        print(
            f"{format_ci_string(aggregate_stats['incorrect_step_count'], 'Average Steps (Incorrect)')}"
        )
        print(
            f"{format_ci_string(aggregate_stats['advisor_calls'], 'Average Advisor Calls')}"
        )
        print(
            f"{format_ci_string(aggregate_stats['correct_advisor_calls'], 'Average Advisor Calls (Correct)')}"
        )
        print(
            f"{format_ci_string(aggregate_stats['incorrect_advisor_calls'], 'Average Advisor Calls (Incorrect)')}"
        )
    else:
        print(f"Total instances: {total}")
        print(f"Resolved: {resolved} ({resolved / total * 100:.1f}%)")
        print(f"\nAverage steps (overall): {avg_steps:.1f}")
        print(f"Average advisor calls (overall): {avg_advisor_calls:.1f}")

    print(f"\nErrors (last run): {errors} ({errors / total * 100:.1f}%)")

    print(f"\nAverage agent cost: ${avg_agent_cost:.4f}")
    print(f"Average advisor cost: ${avg_advisor_cost:.4f}")
    print(f"Total cost: ${total_cost:.4f}")
    print("=" * 80)

    # Print per-instance breakdown
    print("\nPer-instance results (last run):")
    for r in results:
        status_icon = "+" if r.resolved else "-"
        print(
            f"  {status_icon} {r.instance_id}: {r.status} (agent: ${r.agent_cost:.2f}, advisor: ${r.advisor_cost:.2f})"
        )


def run_multi_evaluation(
    debugger: "AdvisorDebugger",
    instances: List[Dict[str, Any]],
    num_runs: int = 1,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics.

    Args:
        debugger: AdvisorDebugger instance
        instances: List of problem instances
        num_runs: Number of evaluation runs
        max_workers: Number of parallel workers

    Returns:
        Dictionary with all run results and aggregate statistics
    """
    # List of lists: all individual scores from each run
    all_run_resolve_scores = []
    all_run_step_counts = []
    all_run_advisor_calls = []
    all_run_correct_step_counts = []
    all_run_incorrect_step_counts = []
    all_run_correct_advisor_calls = []
    all_run_incorrect_advisor_calls = []
    all_run_results = []

    for run_idx in range(num_runs):
        print(f"\n{'=' * 80}")
        print(f"EVALUATION RUN {run_idx + 1}/{num_runs}")
        print(f"{'=' * 80}")

        results = debugger.evaluate_dataset(instances, max_workers=max_workers)
        all_run_results.append(results)

        # Collect all individual scores from this run
        run_resolve_scores = [1.0 if r.resolved else 0.0 for r in results]
        run_step_counts = [float(r.total_steps) for r in results]
        run_advisor_calls = [float(r.advisor_calls) for r in results]

        # Collect correct/incorrect breakdowns
        correct_results = [r for r in results if r.resolved]
        incorrect_results = [r for r in results if not r.resolved]
        run_correct_step_counts = [float(r.total_steps) for r in correct_results]
        run_incorrect_step_counts = [float(r.total_steps) for r in incorrect_results]
        run_correct_advisor_calls = [float(r.advisor_calls) for r in correct_results]
        run_incorrect_advisor_calls = [
            float(r.advisor_calls) for r in incorrect_results
        ]

        all_run_resolve_scores.append(run_resolve_scores)
        all_run_step_counts.append(run_step_counts)
        all_run_advisor_calls.append(run_advisor_calls)
        all_run_correct_step_counts.append(run_correct_step_counts)
        all_run_incorrect_step_counts.append(run_incorrect_step_counts)
        all_run_correct_advisor_calls.append(run_correct_advisor_calls)
        all_run_incorrect_advisor_calls.append(run_incorrect_advisor_calls)

        # Print run summary
        resolved = sum(1 for r in results if r.resolved)
        resolve_rate = resolved / len(results) if results else 0.0
        avg_steps = (
            sum(run_step_counts) / len(run_step_counts) if run_step_counts else 0.0
        )
        avg_advisor_calls = (
            sum(run_advisor_calls) / len(run_advisor_calls)
            if run_advisor_calls
            else 0.0
        )

        print(
            f"Run {run_idx + 1} resolve rate: {resolve_rate:.4f} ({resolved}/{len(results)})"
        )
        print(f"Run {run_idx + 1} avg steps: {avg_steps:.2f}")
        print(f"Run {run_idx + 1} avg advisor calls: {avg_advisor_calls:.2f}")

    # Compute aggregate statistics across runs using all individual scores
    aggregate_stats = {
        "resolve_rate": compute_multi_run_statistics(all_run_resolve_scores),
        "step_count": compute_multi_run_statistics(all_run_step_counts),
        "advisor_calls": compute_multi_run_statistics(all_run_advisor_calls),
        "correct_step_count": compute_multi_run_statistics(all_run_correct_step_counts),
        "incorrect_step_count": compute_multi_run_statistics(
            all_run_incorrect_step_counts
        ),
        "correct_advisor_calls": compute_multi_run_statistics(
            all_run_correct_advisor_calls
        ),
        "incorrect_advisor_calls": compute_multi_run_statistics(
            all_run_incorrect_advisor_calls
        ),
    }

    return {
        "run_results": all_run_results,
        "run_scores": all_run_resolve_scores,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate advisor setup on SWE-Smith")
    parser.add_argument(
        "--advisor_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for advisor (API model name or local path)",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Flag to indicate advisor_model is a local model that needs vLLM serving",
    )
    parser.add_argument(
        "--agent_model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use for agent",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSONL file with evaluation data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--session_mapping_file",
        type=str,
        default="session_mapping.json",
        help="Output file for session ID mapping",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism (for vLLM local models)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model length for vLLM server (for local models)",
    )
    parser.add_argument(
        "--step_num",
        type=int,
        default=None,
        help="Specific step number to use for training checkpoints (default: use latest)",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="Disable thinking mode in chat template for advisor model",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="API base URL for remote vLLM server. Overridden if is_local is set",
    )
    parser.add_argument(
        "--is_hf_model",
        action="store_true",
        help="Flag to indicate advisor_model is a HuggingFace model identifier (e.g., 'Qwen/Qwen3-8B'), not a local path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for advisor model sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p for advisor model sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k for advisor model sampling",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p for advisor model sampling",
    )
    parser.add_argument(
        "--truncate_student_thinking",
        action="store_true",
        help="Strip student thinking from observations shown to advisor to reduce context length",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of evaluation runs for confidence intervals (default: 1)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.data_file}...")
    instances = []
    with open(args.data_file, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            reward_spec = row["reward_spec"]
            # Handle both formats
            if "ground_truth_json" in reward_spec:
                ground_truth = json.loads(reward_spec["ground_truth_json"])
            else:
                ground_truth = reward_spec["ground_truth"]
            instances.append(ground_truth)

    # Limit samples if specified
    if args.num_samples and len(instances) > args.num_samples:
        import random

        random.seed(42)
        instances = random.sample(instances, args.num_samples)

    print(f"Evaluating {len(instances)} instances with advisor...")
    if args.max_workers > 1:
        print(f"Using {args.max_workers} parallel workers")
    if args.num_runs > 1:
        print(f"Running {args.num_runs} evaluation runs for confidence intervals")

    # Run evaluation with proper cleanup
    debugger = AdvisorDebugger(
        args.advisor_model,
        args.agent_model,
        is_local=args.is_local,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        step_num=args.step_num,
        no_think=args.no_think,
        api_base=args.api_base,
        is_hf_model=args.is_hf_model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        truncate_student_thinking=args.truncate_student_thinking,
    )

    try:
        # Run multi-evaluation
        multi_results = run_multi_evaluation(
            debugger=debugger,
            instances=instances,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
        )

        # Save session mapping (from last run)
        save_session_mapping(
            multi_results["run_results"][-1], args.session_mapping_file
        )

        # Print summary
        print_summary(
            multi_results["run_results"][-1],
            aggregate_stats=multi_results["aggregate_stats"],
        )
    finally:
        # Always cleanup vLLM server and temp directories
        debugger.cleanup()


if __name__ == "__main__":
    main()
