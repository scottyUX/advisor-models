"""SWE-Smith environment for RL training with remote agent execution.

This environment integrates the advisor model with a remote agent server:
1. Init: Create remote agent session with Docker container
2. Step: Send advisor feedback to remote agent, execute steps, get observation
3. If not terminated: Return done=False with observation
4. If terminated: Get patch from remote, evaluate, return done=True with score

To adjust the level of trajectory logging on wandb, adjust add_to_log metadata
field as desired.
"""

from typing import Dict, Any, List, Tuple
import sys
import json
import os

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .remote_agent_client import RemoteAgentClient
from .config import compute_score
from .reward import e7_reward


class SWESmithEnv(BaseTextEnv):
    """Environment for SWE-Smith with remote agent execution."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        """Initialize the environment.

        Args:
            env_config: The environment configuration (unused). E7's SkyRL fork
                passes an omegaconf DictConfig, upstream SkyRL a dict.
            extras: Additional metadata including:
                - reward_spec: Contains ground_truth instance
                - original_question: Problem statement
        """
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth_json" in extras["reward_spec"], (
            "ground_truth_json is required in reward_spec"
        )
        assert "original_question" in extras, "original_question is required"

        # Deserialize the ground truth instance from JSON
        self.ground_truth = json.loads(extras["reward_spec"]["ground_truth_json"])
        self.original_question = extras["original_question"]

        # Agent configuration
        self.max_steps_per_call = 5  # Run 5 agent steps per RL step
        self.agent_model = os.environ.get(
            "AGENT_MODEL", "gpt-4.1-mini"
        )  # Read from environment

        # Remote agent client
        self.remote_agent = RemoteAgentClient()
        self.session_id = None

        # State variables
        self.agent_terminated = False
        self.total_steps_taken = 0
        self.advisor_prompt_to_log = None
        self.action = None
        self.previous_observation = ""

        # Rolling summary state (maintained by remote agent)
        self.rolling_summary = ""
        self.last_summarized_step = 0

    def init(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Initialize the environment by creating remote agent session.

        Args:
            prompt: The advisor prompt from dataset

        Returns:
            Tuple of (prompt, metadata)
        """
        # Log the advisor prompt
        self.advisor_prompt_to_log = prompt[-1]["content"] if prompt else ""
        self.previous_observation = self.advisor_prompt_to_log

        try:
            # Create remote agent session with configured agent model
            self.session_id = self.remote_agent.create_session(
                self.ground_truth, model_name=self.agent_model
            )
        except Exception as e:
            print(f"Error creating remote session: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            # Set terminated flag so step() will return done=True with 0 reward
            self.agent_terminated = True

        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one RL step by sending advisor feedback to remote agent.

        Args:
            action: The advisor's advice

        Returns:
            BaseTextEnvStepOutput with reward, done flag, and metadata
        """
        self.action = action

        # Handle initialization failure
        if self.agent_terminated:
            return self._return_failure()

        # Parse out think block if present
        advisor_feedback = action
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>")[1].strip()

        # Execute remotely
        try:
            result = self.remote_agent.execute_step(
                self.session_id,
                advisor_feedback,
                max_steps=self.max_steps_per_call,
            )

            # Parse result
            if result.get("terminated"):
                self.agent_terminated = True

            self.total_steps_taken = result.get("total_steps", 0)

            # Update rolling summary from remote
            self.rolling_summary = result.get("rolling_summary", "")
            self.last_summarized_step = result.get("last_summarized_step", 0)

            if self.agent_terminated:
                return self._return_final_result()
            else:
                return self._return_intermediate_observation_remote(result)

        except Exception as e:
            print(f"Error in remote agent step: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            return self._return_failure()

    def _return_intermediate_observation_remote(
        self, result: Dict[str, Any]
    ) -> BaseTextEnvStepOutput:
        """Return intermediate observation from remote agent result."""
        observation_text = result.get("observation", "")

        observation_message = {
            "role": "user",
            "content": observation_text,
        }

        metadata = {
            "add_to_log": True,
            "original_question": self.advisor_prompt_to_log,
            "advisor_prompt": self.previous_observation,
            "advisor_response": self.action,
            "ground_truth": "",
            "final_reward": 0.0,
            "other_info": f"In-progress: Steps: {self.total_steps_taken}, Instance ID: {self.ground_truth['instance_id']}, Remote session ID: {self.session_id}",
        }

        self.previous_observation = observation_text

        return BaseTextEnvStepOutput(
            observations=[observation_message],
            reward=0.0,  # No reward yet
            done=False,  # Not done yet
            metadata=metadata,
        )

    def _return_final_result(self) -> BaseTextEnvStepOutput:
        """Return final result with evaluation score."""
        try:
            # Get patch from remote session
            patch = self.remote_agent.get_patch(self.session_id)

            # Evaluate patch
            score, eval_id, _ = compute_score(patch, self.ground_truth)
            reward = e7_reward(score, self.total_steps_taken)
            print(f"Reward: {reward}")
            sys.stdout.flush()

            # Combine evaluation info with agent stats for logging
            other_info = f"COMPLETED: Steps: {self.total_steps_taken}, Instance ID: {self.ground_truth['instance_id']}, Remote session ID: {self.session_id}, Eval ID: {eval_id}"

            metadata = {
                "add_to_log": True,
                "original_question": self.advisor_prompt_to_log,
                "advisor_prompt": self.previous_observation,
                "advisor_response": self.action,
                "ground_truth": "",
                "final_reward": reward,
                "other_info": other_info,
            }

            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata=metadata,
            )

        except Exception:
            import traceback

            traceback.print_exc()
            return self._return_failure()

        finally:
            # Cleanup remote session
            self._cleanup()

    def _return_failure(self) -> BaseTextEnvStepOutput:
        """Return failure result."""
        other_info = f"FAILED: Steps: {self.total_steps_taken}, Instance ID: {self.ground_truth['instance_id']}, Remote session ID: {self.session_id}"
        metadata = {
            "add_to_log": True,
            "original_question": self.advisor_prompt_to_log,
            "advisor_prompt": self.previous_observation,
            "advisor_response": self.action,
            "ground_truth": "",
            "final_reward": 0.0,
            "other_info": other_info,
        }

        self._cleanup()

        return BaseTextEnvStepOutput(
            observations=[],
            reward=0.0,
            done=True,
            metadata=metadata,
        )

    def _cleanup(self):
        """Cleanup remote session."""
        if self.session_id:
            try:
                self.remote_agent.cleanup_session(self.session_id)
                self.session_id = None  # Mark as cleaned up
            except Exception as e:
                print(f"Warning: Failed to cleanup remote session: {e}")
                sys.stdout.flush()

    def __del__(self):
        """Ensure cleanup on environment destruction."""
        try:
            self._cleanup()
        except Exception:
            pass  # Best effort cleanup
