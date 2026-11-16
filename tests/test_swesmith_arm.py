"""Offline tests of swesmith_arm (E10): no Docker daemon, GPU, network or model.

install() patches swesmith for the whole process, so each case runs in a fresh
Python subprocess. Docker is faked: the tests check what swesmith would ask
Docker to do, not that Docker does it (scripts/e10_check_swesmith_arm.py does
that on the Spark).

Needs swesmith 0.0.9 and swebench 4.1.0; skipped if swesmith is not installed.

Run: python -m unittest discover -s tests -p "test_swesmith_arm.py"
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = "hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10"
HAVE_SWESMITH = importlib.util.find_spec("swesmith") is not None

PRELUDE = textwrap.dedent(
    f"""
    import json, sys
    from unittest import mock
    sys.path.insert(0, {str(ROOT)!r})
    line = open({str(ROOT / "data/swe_smith/train_jd__tenacity.json")!r}).readline()
    INST = json.loads(json.loads(line)["reward_spec"]["ground_truth_json"])
    """
)


def run_case(body: str, env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SWESMITH_")}
    env.update(env_extra)
    code = PRELUDE + textwrap.dedent(body)
    # swesmith writes logs/ under the working directory: keep it out of the repo.
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=env, cwd=cwd, timeout=120)
    if p.returncode != 0:
        raise AssertionError(f"case failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HAVE_SWESMITH, "swesmith not installed")
class SwesmithArm(unittest.TestCase):
    def test_unset_is_a_no_op(self):
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install
            import swesmith.harness.utils as u
            from swesmith.profiles import registry
            active = install()
            rp = registry.get_from_inst(INST)
            print(json.dumps({"active": active, "image": rp.image_name,
                              "shim": getattr(u.docker, "__e10_arm_shim__", False)}))
            """,
            {},
        )
        self.assertFalse(out["active"])
        self.assertFalse(out["shim"])
        self.assertNotEqual(out["image"], ARM)

    def test_image_name_points_at_arm_image(self):
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install
            from swesmith.profiles import registry
            install(); install()  # idempotent
            print(json.dumps({"image": registry.get_from_inst(INST).image_name}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM},
        )
        self.assertEqual(out["image"], ARM)

    def test_other_repo_refuses(self):
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install, SwesmithArmError
            from swesmith.profiles import registry
            install()
            other = next(p for p in registry.values() if p.repo != "tenacity")
            try:
                other.image_name
                print(json.dumps({"raised": False}))
            except SwesmithArmError:
                print(json.dumps({"raised": True}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM},
        )
        self.assertTrue(out["raised"])

    def test_never_pulls_and_no_github_call(self):
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install, SwesmithArmError
            from swesmith.profiles import registry
            install()
            rp = registry.get_from_inst(INST)
            calls = []
            def fake_run(cmd, **kw):
                calls.append(cmd)
                class R: returncode = 1
                return R()
            with mock.patch("subprocess.run", fake_run), \\
                 mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                private = rp._is_repo_private()
                try:
                    rp.pull_image(); raised = False
                except SwesmithArmError:
                    raised = True
            print(json.dumps({"private": private, "raised": raised, "calls": calls}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM},
        )
        self.assertFalse(out["private"])
        self.assertTrue(out["raised"])
        self.assertEqual(out["calls"], [["docker", "image", "inspect", ARM]])
        self.assertFalse(any("pull" in c for c in out["calls"][0]))

    def test_run_patch_in_container_creates_arm_container(self):
        # Drive swesmith's real run_patch_in_container with a fake Docker client.
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install
            install()
            import docker
            import swesmith.harness.utils as u
            from swebench.harness.constants import RUN_EVALUATION_LOG_DIR
            client = mock.MagicMock()
            container = client.containers.create.return_value
            container.exec_run.return_value = mock.MagicMock(exit_code=0, output=b"")
            ok = mock.MagicMock(returncode=0)
            with mock.patch.object(docker, "from_env", return_value=client), \\
                 mock.patch.object(u, "copy_to_container"), \\
                 mock.patch.object(u, "cleanup_container"), \\
                 mock.patch.object(u, "exec_run_with_timeout", return_value=("", False, 0.1)), \\
                 mock.patch("subprocess.run", return_value=ok):
                u.run_patch_in_container(INST, "e10-test", RUN_EVALUATION_LOG_DIR, 90,
                                         patch=INST["patch"], commit=INST["instance_id"],
                                         is_gold=True)
            kw = client.containers.create.call_args.kwargs
            print(json.dumps({"image": kw["image"], "platform": kw["platform"]}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM},
        )
        self.assertEqual(out, {"image": ARM, "platform": "linux/arm64"})

    def test_platform_can_be_set(self):
        out = run_case(
            """
            from advisor_models.swe_smith.swesmith_arm import install
            install()
            import docker
            import swesmith.harness.utils as u
            client = mock.MagicMock()
            with mock.patch.object(docker, "from_env", return_value=client):
                u.docker.from_env().containers.create(image="x", platform="linux/x86_64")
            print(json.dumps({"platform": client.containers.create.call_args.kwargs["platform"]}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM, "SWESMITH_PLATFORM": "linux/arm64/v8"},
        )
        self.assertEqual(out["platform"], "linux/arm64/v8")

    def test_wrong_swesmith_version_refuses(self):
        out = run_case(
            """
            from advisor_models.swe_smith import swesmith_arm
            with mock.patch("importlib.metadata.version", return_value="0.0.8"):
                try:
                    swesmith_arm.install(); raised = False
                except swesmith_arm.SwesmithArmError:
                    raised = True
            print(json.dumps({"raised": raised}))
            """,
            {"SWESMITH_IMAGE_OVERRIDE": ARM},
        )
        self.assertTrue(out["raised"])


if __name__ == "__main__":
    unittest.main()
