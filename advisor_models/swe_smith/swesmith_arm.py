"""Run swesmith's grader against the ARM Tenacity image on the DGX Spark (E10).

swesmith 0.0.9 cannot grade on the Spark as installed:

1. ``RepoProfile.image_name`` derives the image from the host architecture
   (``swebench/swesmith.arm64.jd_1776_tenacity.0d40e76f``), which does not
   exist, so ``pull_image`` tries to pull it.
2. ``run_patch_in_container`` (``swesmith/harness/utils.py``) and
   ``RepoProfile.get_container`` (``swesmith/profiles/base.py``) create
   containers with ``platform="linux/x86_64"`` hard-coded, which Docker
   refuses for an arm64 image.

This module fixes both from outside the installed package: nothing in
site-packages is edited. ``install()`` patches three things at runtime, and
only when ``SWESMITH_IMAGE_OVERRIDE`` is set (the same variable the agent
server reads, see ``images.py``):

- ``image_name`` returns the override, for the one repo whose name the
  override image carries (``<owner>_1776_<repo>.<commit[:8]>``). Any other
  repo raises instead of silently grading in the wrong image.
- ``pull_image`` never pulls: the override image must already be on the host.
- the ``docker`` client used by those two modules creates containers with
  ``SWESMITH_PLATFORM`` (default ``linux/arm64``) instead of x86.

The repo's privacy check (a GitHub API call) is answered "public" for the
overridden repo: Tenacity is public, and the image is local, so grading does
not depend on GitHub being reachable or on its rate limit.

Grading itself (``run_evaluation``, ``get_eval_report``, the test command, the
log parser) is swesmith's own code, unchanged. With the override unset,
``install()`` does nothing and swesmith behaves as in E7.

Written against swesmith 0.0.9 with swebench 4.1.0. ``install()`` refuses any
other swesmith version, and checks that the code it patches still looks as
expected, so a version change fails loudly instead of grading wrongly.
"""

import inspect
import os
import subprocess
import threading
import types
from typing import Any, Optional

from .images import IMAGE_OVERRIDE_ENV

PLATFORM_ENV = "SWESMITH_PLATFORM"
DEFAULT_PLATFORM = "linux/arm64"
SUPPORTED_SWESMITH = "0.0.9"

_lock = threading.Lock()
_installed = False


class SwesmithArmError(RuntimeError):
    pass


def override_image() -> str:
    return os.environ.get(IMAGE_OVERRIDE_ENV, "").strip()


def override_platform() -> str:
    return os.environ.get(PLATFORM_ENV, "").strip() or DEFAULT_PLATFORM


def image_key(owner: str, repo: str, commit: str) -> str:
    """The part of a SWE-smith image name that identifies the repo snapshot."""
    return f"{owner}_1776_{repo}.{commit[:8]}".lower()


def _image_matches(image: str, owner: str, repo: str, commit: str) -> bool:
    # hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10 -> ...jd_1776_tenacity.0d40e76f
    name = image.rsplit("/", 1)[-1].split(":", 1)[0].lower()
    return name.endswith("." + image_key(owner, repo, commit))


class _ContainersShim:
    def __init__(self, containers: Any, platform: str):
        self._containers = containers
        self._platform = platform

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if "platform" in kwargs:
            kwargs["platform"] = self._platform
        return self._containers.create(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if "platform" in kwargs:
            kwargs["platform"] = self._platform
        return self._containers.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._containers, name)


class _ClientShim:
    def __init__(self, client: Any, platform: str):
        self._client = client
        self.containers = _ContainersShim(client.containers, platform)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _docker_module_shim(docker_module: types.ModuleType, platform: str) -> types.ModuleType:
    """A stand-in for the ``docker`` module whose ``from_env`` client uses ``platform``."""
    shim = types.ModuleType("docker")
    shim.__dict__.update(
        {k: v for k, v in docker_module.__dict__.items() if not k.startswith("__")}
    )

    def from_env(*args: Any, **kwargs: Any) -> Any:
        return _ClientShim(docker_module.from_env(*args, **kwargs), platform)

    shim.from_env = from_env
    shim.__e10_arm_shim__ = True
    return shim


def _check_version_and_code(sw_version: str, utils_mod: Any, base_mod: Any) -> None:
    if sw_version != SUPPORTED_SWESMITH:
        raise SwesmithArmError(
            f"swesmith {sw_version} is installed; swesmith_arm supports only "
            f"{SUPPORTED_SWESMITH}. Re-check the patched code before changing this."
        )
    expected = [
        (utils_mod.run_patch_in_container, 'platform="linux/x86_64"'),
        (utils_mod.run_patch_in_container, "docker.from_env()"),
        (utils_mod.run_patch_in_container, "rp.image_name"),
        (utils_mod.run_patch_in_container, "rp.pull_image()"),
        (base_mod.RepoProfile.pull_image, "docker pull"),
    ]
    for fn, snippet in expected:
        if snippet not in inspect.getsource(fn):
            raise SwesmithArmError(
                f"{fn.__qualname__} no longer contains {snippet!r}; swesmith_arm "
                "would not do what it says. Stop and re-check."
            )


def install() -> bool:
    """Patch swesmith for the override image. Returns True if patches are active.

    Safe to call more than once and from several threads.
    """
    global _installed
    image = override_image()
    if not image:
        return False
    with _lock:
        if _installed:
            return True

        import docker
        import swesmith
        import swesmith.harness.utils as utils_mod
        import swesmith.profiles.base as base_mod
        from importlib.metadata import version

        _check_version_and_code(version("swesmith"), utils_mod, base_mod)
        platform = override_platform()
        RepoProfile = base_mod.RepoProfile

        original_image_name = RepoProfile.image_name
        original_is_private = RepoProfile._is_repo_private

        def image_name(self: Any) -> str:
            if _image_matches(image, self.owner, self.repo, self.commit):
                return image
            raise SwesmithArmError(
                f"{IMAGE_OVERRIDE_ENV}={image} is for another repo; refusing to "
                f"grade {self.owner}/{self.repo}@{self.commit[:8]} in it."
            )

        def pull_image(self: Any) -> None:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise SwesmithArmError(
                    f"Image {self.image_name} is not on this host; swesmith_arm never pulls."
                )

        def _is_repo_private(self: Any) -> bool:
            if _image_matches(image, self.owner, self.repo, self.commit):
                return False
            return original_is_private(self)

        RepoProfile.image_name = property(image_name)
        RepoProfile.pull_image = pull_image
        RepoProfile._is_repo_private = _is_repo_private
        RepoProfile.__e10_original_image_name__ = original_image_name

        shim = _docker_module_shim(docker, platform)
        utils_mod.docker = shim
        base_mod.docker = shim

        swesmith.__e10_arm__ = {"image": image, "platform": platform}
        _installed = True
        return True


def describe() -> Optional[dict]:
    """What install() set, for run manifests and logs; None if not installed."""
    if not _installed:
        return None
    import swesmith

    return dict(swesmith.__e10_arm__)
