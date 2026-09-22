"""What only a real module process can prove.

Every other suite builds resources directly from fakes. That misses two whole
classes of defect, both of which reached the GPU and cost a run each.

A dependency API must be REGISTERED in the process, or viam-server cannot hand
the module a client for it. Registration happens as an import side effect
inside the SDK, so a module that never imports the package gets
`No world_state_store with name "pack-sequencer" found in the registry` and the
resource never constructs. A fake dependency in a unit test needs no registry
at all, so nothing there notices.

And a dependency client is bound to the module's own event loop. Awaiting one
from a private loop on another thread never completes, because the loop that
owns its channel is not the one driving the await. A fake has no channel, so
again nothing notices.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from viam.proto.app.robot import ComponentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every API this module names as a dependency of one of its own models, as
# namespace/type/subtype triples. Written as strings on purpose: importing the
# SDK package to read its `.API` attribute is what REGISTERS it, so a check
# that imports it passes whether or not the module under test does.
DEPENDENCY_APIS = (
    ("rdk", "service", "world_state_store"),
    ("rdk", "service", "motion"),
    ("rdk", "component", "arm"),
    ("rdk", "component", "gripper"),
    ("rdk", "component", "camera"),
    ("rdk", "component", "generic"),
)


def _run_in_module_process(source: str) -> subprocess.CompletedProcess[str]:
    """Run `source` in a fresh interpreter that imports only what this module
    imports, which is the whole point: an API registered by some other test's
    import is not registered in the module process on the machine."""
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


def test_every_dependency_api_is_registered_by_importing_the_models():
    checks = "\n".join(
        f"assert Registry.lookup_api(API({namespace!r}, {rtype!r}, {subtype!r})) is not None, "
        f"'{namespace}:{rtype}:{subtype} is not registered'"
        for namespace, rtype, subtype in DEPENDENCY_APIS
    )
    source = (
        "import isaac_module.models.palletizer\n"
        "import isaac_module.models.scene_finalizer\n"
        "import isaac_module.models.arm\n"
        "import isaac_module.models.camera\n"
        "import isaac_module.models.vacuum\n"
        "import isaac_module.models.world\n"
        "from viam.resource.registry import Registry\n"
        "from viam.resource.types import API\n" + checks
    )
    result = _run_in_module_process(source)
    assert result.returncode == 0, result.stderr


def test_the_scene_finalizer_hands_the_materialise_the_running_loop():
    """A gather scheduled onto a private loop hangs on its first probe and
    never returns, which parks the world at its scene gate forever. Assert the
    loop reaches the gather, not merely that reconfigure was called."""
    from isaac_module.models import scene_finalizer

    captured: dict[str, Any] = {}

    def fake_materialise_workcell(resources: Any, *, logger: Any, loop: Any = None) -> dict:
        captured["loop"] = loop
        return {}

    original = scene_finalizer.materialise_workcell
    scene_finalizer.materialise_workcell = fake_materialise_workcell  # type: ignore[assignment]
    try:

        async def build() -> asyncio.AbstractEventLoop:
            config = ComponentConfig(name="scene-ready")
            config.depends_on.append("isaac-world")
            finalizer = scene_finalizer.IsaacSceneFinalizer.new(config, {})
            finalizer.wait_until_materialised(10.0)
            return asyncio.get_running_loop()

        running_loop = asyncio.run(build())
    finally:
        scene_finalizer.materialise_workcell = original  # type: ignore[assignment]

    assert captured["loop"] is running_loop
