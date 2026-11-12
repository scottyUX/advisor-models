"""Configuration for SWE-Smith domain.

Contains system prompts and templates for software engineering patch generation with advisor feedback.
Defines student and advisor model prompts for generating git patches to fix bugs.
"""

from typing import Tuple
import uuid

# Optional student-only instruction. Appended after the rendered instance
# template when eval_swe_smith.py is run with --efficiency_prompt.
# Training and the untrained-advisor arm load default.yaml and do not use this.
EFFICIENCY_INSTRUCTION = (
    "Work in as few steps as possible. As soon as the original issue is fixed "
    "and a minimal check passes, submit. Do not add tests, refactors, or extra "
    "exploration unless they unblock that fix."
)

# Advisor prompt template
ADVISOR_TEMPLATE = """You are an expert software engineering advisor that will provide advice to an autonomous agent working to fix a bug in a codebase. Your role is to provide strategic, concise guidance that helps the agent solve the problem and submit a fix as quickly as possible. You will NOT be able to interact with the codebase, but the agent will have such access.

The agent can:
- Navigate the codebase
- Read and edit files (sed, cat, python scripts, etc.)
- Run tests or minimal reproduction scripts

In short, the agent has command line access. Your job is to help the agent implement and submit a fix in as few interactions with the console as possible. Do NOT suggest unnecessary steps (e.g., refactors, additional tests, cleanup) unless they directly unblock the bug. In many cases, you may need to discourage the agent from performing additional tests. The goal is to submit a fix as fast as possible.

Suggestions and guidance for advice:
- Assist the agent in rapidly localizing the error location.
- If the agent is repeatedly fighting tooling (e.g., sed), suggest a safer alternative (manual edit, Python script, direct inspection).
- Focus the agent on the error at hand: if the agent fixes adjacent issues without reducing the original symptom, point that out.
- If the agent is stuck or going in circles, help them refocus on the original problem or suggest alternatives to break out of the loop.
- Identify when extraneous tests are unnecessary: fixes ought to be tested to verify correctness, but the goal is to submit a working fix as quickly as possible. Not every edge case needs to be tested if you're confident in the fix.
- Be concise and direct. Explicitly state your advice so the agent can act on it effectively.
- Frame your advice as advisory, not authoritative, particularly when it comes to assumptions about the codebase. The agent has access to the codebase, not you. Make sure the agent understands this.

Here is the current problem statement:
{problem_statement}

Remember: Your goal is to help the agent fix the bug as quickly as possible.
"""

ADVISOR_FOLLOWUP_TEMPLATE = """## Most Recent Agent Actions and Observations (last {len_messages} messages)

{recent_text}

Continue to guide the agent to fix the issue as efficiently as possible.

Identify and address failure patterns. Actively watch for:
- Repeated errors after "fixes"
- Repeated failed edits using the same tool or method
- Drift away from the original failing behavior
- Extraneous testing when there is already high confidence in the fix

If you see a pattern that the agent doesn't seem to see or is not addressing:
- Call it out explicitly
- Recommend specific different approach (e.g., use alternative editing tools, return to work on the original issue, stop testing and submit the fix immediately)

Now, provide more advice to the agent with the goal of submitting a working fix as quickly as possible.
"""


def build_prompt(info_dict: dict) -> str:
    """Build the advisor prompt for a given problem instance.

    Args:
        info_dict: Dictionary containing problem_statement and repo

    Returns:
        Formatted prompt string
    """
    return ADVISOR_TEMPLATE.format(
        problem_statement=info_dict.get("problem_statement", ""),
    )


def build_advisor_prompt(info_dict: dict) -> str:
    """Build the advisor prompt for a given problem instance.

    Args:
        info_dict: Dictionary containing problem_statement and repo

    Returns:
        Formatted advisor prompt string
    """
    return ADVISOR_TEMPLATE.format(
        problem_statement=info_dict.get("problem_statement", ""),
    )


def summarize_agent_progress(
    new_messages: list,
    last_summarized_step: int,
    current_step: int,
    summarizer_model: str = "gpt-4.1-mini",
) -> Tuple[str, float]:
    """Summarize new agent messages since last summary.

    Args:
        new_messages: New messages since last summarization
        last_summarized_step: Step number of last summarization
        current_step: Current step number
        summarizer_model: Model to use for summarization

    Returns:
        Tuple of (summary_text, cost)
    """
    import litellm

    if not new_messages:
        return "", 0.0

    # Format messages for summarization
    message_text = []
    for msg in new_messages:
        role = msg["role"]
        content = msg["content"]
        # Truncate very long content
        if len(content) > 1000:
            content = content[:500] + "\n... [truncated] ...\n" + content[-500:]
        message_text.append(f"[{role}]: {content}")

    messages_str = "\n\n".join(message_text)

    # Summarization prompt
    prompt = f"""Summarize the following agent interaction concisely. Focus on:
1. The approach the agent is taking
2. What errors or issues occurred
3. Any patterns (e.g., repeated errors, stuck loops)
4. Current state of progress

Be concise but capture key details that would help an advisor understand what's happening. Discuss the entire trace but focus primarily on the more recent messages.

Agent messages (steps {last_summarized_step + 1} to {current_step}):
{messages_str}

Provide a concise summary (max 100 words):"""

    try:
        response = litellm.completion(
            model=summarizer_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        summary = response.choices[0].message.content
        cost = litellm.completion_cost(completion_response=response)

        return summary, cost

    except Exception as e:
        print(f"Error summarizing agent progress: {e}")
        # Fallback to simple truncation
        return messages_str[:500] + "...", 0.0


def build_advisor_observation(
    agent_history: list,
    rolling_summary: str,
    last_summarized_step: int,
    current_step: int,
    summarizer_model: str = "gpt-4.1-mini",
) -> Tuple[str, str, int, float]:
    """Build observation for advisor with rolling summary + recent messages.

    Args:
        agent_history: Full agent message history
        rolling_summary: Current rolling summary
        last_summarized_step: Step number of last summarization
        current_step: Current step number
        summarizer_model: Model to use for summarization

    Returns:
        Tuple of (observation_text, updated_rolling_summary, updated_last_summarized_step, cost)
    """
    summary_cost = 0.0
    # Get most recent messages for immediate context (last 10 messages)
    recent_messages = agent_history[-10:] if len(agent_history) >= 10 else agent_history

    recent_parts = []
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"]
        recent_parts.append(f"[{role}]: {content}")

    recent_text = "\n\n".join(recent_parts)

    # Combine rolling summary with recent messages
    observation_text = ADVISOR_FOLLOWUP_TEMPLATE.format(
        len_messages=len(recent_messages), recent_text=recent_text
    )

    return observation_text, rolling_summary, last_summarized_step, summary_cost


def compute_score(patch: str, instance: dict) -> Tuple[float, str, str]:
    """Compute score by evaluating patch using SWE-Smith harness.

    This function automatically detects whether to use local Docker evaluation
    or remote evaluation based on the EVAL_SERVER_URL environment variable.

    Args:
        patch: The generated patch string
        instance: The problem instance dict from SWE-Smith dataset

    Returns:
        Tuple of (reward, run_id, info_string)
        - reward: 1.0 if patch resolves the issue, 0.0 otherwise
        - run_id: Unique ID for this evaluation
        - info_string: Status information from evaluation
    """
    import os

    # Check if remote evaluation is configured
    eval_server_url = os.environ.get("EVAL_SERVER_URL")

    if eval_server_url:
        # Use remote evaluation
        try:
            from .remote_eval_client import compute_score_remote

            return compute_score_remote(patch, instance)
        except Exception as e:
            return 0.0, "ERROR", f"Remote evaluation failed: {str(e)}"

    return compute_score_local(patch, instance)


def compute_score_local(patch: str, instance: dict) -> Tuple[float, str, str]:
    """Evaluate a patch with the local SWE-Smith Docker harness.

    Ignores EVAL_SERVER_URL. eval_server must call this, not compute_score,
    or it sends the request back to itself when EVAL_SERVER_URL is set.
    """
    try:
        from swesmith.harness.eval import run_evaluation
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )

        # Create prediction dict
        pred = {
            KEY_INSTANCE_ID: instance["instance_id"],
            KEY_MODEL: "advisor_model",
            KEY_PREDICTION: patch,
        }

        # Run evaluation using SWE-Smith harness
        # Use a temporary run_id to avoid conflicts
        run_id = f"eval_{uuid.uuid4().hex[:8]}"

        result = run_evaluation(
            pred=pred, instance=instance, run_id=run_id, f2p_only=False, is_gold=False
        )

        # Check if patch resolved the issue
        resolved = result.get("resolved", False)
        status = result.get("status", "unknown")

        reward = 1.0 if resolved else 0.0
        info = f"Status: {status}, Resolved: {resolved}"

        return reward, run_id, info

    except Exception as e:
        return 0.0, "ERROR", f"Evaluation failed: {str(e)}"
