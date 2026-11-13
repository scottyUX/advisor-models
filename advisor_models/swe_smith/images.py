"""Which Docker image a SWE-smith session runs in.

E7's instances name the x86 image (``jyangballin/swesmith.x86_64...``). On the
DGX Spark (aarch64) E10 uses the ARM rebuild checked in G0b, so the agent
server reads the image from ``SWESMITH_IMAGE_OVERRIDE`` when it is set.
Unset, behaviour is E7's: the instance's own ``image_name``.
"""

import os
from typing import Any, Dict

IMAGE_OVERRIDE_ENV = "SWESMITH_IMAGE_OVERRIDE"


def resolve_image_name(instance: Dict[str, Any]) -> str:
    override = os.environ.get(IMAGE_OVERRIDE_ENV, "").strip()
    return override or instance.get("image_name", "")
