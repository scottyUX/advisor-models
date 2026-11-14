"""E10 entrypoint: E7's SWE-smith environment on upstream SkyRL (commit 3e8700f).

Same as ``main_swe_smith.py`` (E7, SkyRL fork with Hydra) except for upstream's
API: the config is a ``SkyRLTrainConfig`` built from CLI overrides, and the
trainer lives in ``skyrl.train`` instead of ``skyrl_train``.

    python -m advisor_models.swe_smith.main_swe_smith_upstream <overrides>

See scripts/e10_train_swe_smith_grpo.sh. Do not start a training run until a
spec amendment authorizes it.
"""

import sys

import ray
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


class E10SWESmithExp(BasePPOExp):
    """Upstream's experiment with the E10 generator (per-step rollout counts).

    The generator adds metrics and a log line only; generation and rewards are
    upstream's (see generator_upstream.py).
    """

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if cfg.generator.vision_language_generator:
            raise ValueError("E10 uses the text generator only.")
        from .generator_upstream import E10SkyRLGymGenerator

        return E10SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # Register the swe_smith environment inside the entrypoint task.
    register(
        id="swe_smith",
        entry_point="advisor_models.swe_smith.env:SWESmithEnv",
    )

    # Make sure the training loop is not run on the head node.
    exp = E10SWESmithExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
