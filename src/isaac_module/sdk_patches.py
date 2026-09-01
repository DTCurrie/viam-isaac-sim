"""Patches for gaps in the installed viam-sdk.

MoveThroughJointPositions: RDK's motion service executes its planned arm
trajectories through this RPC, but viam-sdk (through at least 0.80.0) never
implemented the server-side handler - the generated routing exists and falls
through to the UNIMPLEMENTED default, which breaks motion for every python
modular arm. We add the missing handler to ArmRPCService, mirroring the
SDK's own handler conventions; it dispatches to the resource's
move_through_joint_positions method (which IsaacArm implements).

Remove once the SDK ships its own handler (it will simply override ours).
"""

from viam.components.arm.service import ArmRPCService
from viam.logging import getLogger
from viam.proto.component.arm import (
    MoveThroughJointPositionsRequest,
    MoveThroughJointPositionsResponse,
)
from viam.utils import struct_to_dict

LOGGER = getLogger("viam-isaac-sim.patches")


async def _move_through_joint_positions(self, stream) -> None:
    request: MoveThroughJointPositionsRequest = await stream.recv_message()
    assert request is not None
    arm = self.get_resource(request.name)
    timeout = stream.deadline.time_remaining() if stream.deadline else None
    await arm.move_through_joint_positions(
        list(request.positions),
        options=request.options if request.HasField("options") else None,
        extra=struct_to_dict(request.extra),
        timeout=timeout,
        metadata=stream.metadata,
    )
    await stream.send_message(MoveThroughJointPositionsResponse())


def apply() -> None:
    existing = getattr(ArmRPCService, "MoveThroughJointPositions", None)
    if existing is not None and not getattr(existing, "__isabstractmethod__", False):
        # only patch if the SDK's handler is the raising placeholder
        qualname = getattr(existing, "__qualname__", "")
        if not qualname.startswith(("ArmServiceBase", "UnimplementedArmServiceBase")):
            LOGGER.info("sdk provides MoveThroughJointPositions; not patching")
            return
    ArmRPCService.MoveThroughJointPositions = _move_through_joint_positions  # type: ignore[method-assign]  # deliberate monkeypatch (see module docstring)
    LOGGER.info("patched ArmRPCService with MoveThroughJointPositions handler")
