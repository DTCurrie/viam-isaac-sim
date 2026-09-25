"""What the world state stores publish, relayed into the planner's WorldState.

RDK's builtin motion service plans against the obstacles in the Move request
and the frame system's own geometries, and never reads a
``rdk:service:world_state_store`` (``services/motion/builtin/builtin.go`` on
main, checked 2026-09-24). The stores feed the 3D viewer. So a store's
transforms reach the planner only if this module puts them in the request,
which is what this relay does: every transform a store publishes in the
world frame, with a physical object, becomes one obstacle geometry in the
world frame.

Transforms parented to any other frame are skipped and counted, not resolved:
the scene store publishes its fixtures under its own frame tree
(``pallet/group`` and the like), and those fixtures already reach the planner
as colliders through the fragment's ``frame.geometry``. The sequencer's store
publishes the placed boxes in the world frame, one per placed seq, with the
box's geometry centred on the transform's pose, which is why the palletizer
feeds ``set_box_transform`` the box's CENTRE and keeps the top face only for
its own records (read off the running store 2026-09-24: ``box-2-v1`` at the
pose fed, geometry centre (0, 0, 0), dims 200 x 150 x 100).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from viam.logging import getLogger
from viam.proto.common import Geometry, Pose, Transform

from .spatial import Quat, compose_pose, ov_to_quat, quat_to_ov

LOGGER = getLogger(__name__)

WORLD_FRAME = "world"

_SHAPE_FIELDS = ("box", "sphere", "capsule", "mesh")


class WorldStateStoreLike(Protocol):
    """The two calls the relay makes, as the installed SDK's
    ``WorldStateStoreClient`` serves them."""

    async def list_uuids(self, *, extra: Any = None, timeout: float | None = None) -> Any: ...

    async def get_transform(
        self, uuid: Any, *, extra: Any = None, timeout: float | None = None
    ) -> Transform: ...


@dataclass(frozen=True)
class RelayReading:
    """One read of one or more stores: the obstacles to hand the planner and
    the counts that say what was and was not relayed, so a run's records can
    show the planner was given the stack rather than assumed to have it."""

    obstacles: list[Geometry] = field(default_factory=list)
    relayed: int = 0
    skipped_not_in_world: int = 0
    skipped_no_geometry: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "relayed": self.relayed,
            "skipped_not_in_world": self.skipped_not_in_world,
            "skipped_no_geometry": self.skipped_no_geometry,
        }


def _orientation_quat(pose: Pose) -> Quat:
    """The ``(w, x, y, z)`` quaternion of a proto pose's orientation vector.
    An all-zero vector is the proto's default for a pose that was never given
    one, which means unrotated rather than pointing nowhere."""
    if not (pose.o_x or pose.o_y or pose.o_z):
        return (1.0, 0.0, 0.0, 0.0)
    return ov_to_quat(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta))


def geometry_in_world(transform: Transform) -> Geometry | None:
    """The transform's physical object as a geometry in the world frame, or
    None when the transform is not observed in the world frame or carries no
    shape (a box, sphere or capsule; a mesh is a shape too).

    The geometry's own ``center`` is an offset from the transform's pose, so
    the two are composed (``isaac_module.spatial.compose_pose``): a store may
    publish a box whose centre sits half a height under the pose it was given.
    The result's ``label`` is the transform's ``reference_frame`` (the store's
    name for it, ``box-2-v1`` and the like) so a planner refusal names what
    it hit."""
    if transform.pose_in_observer_frame.reference_frame != WORLD_FRAME:
        return None
    shape = transform.physical_object
    kind = shape.WhichOneof("geometry_type")
    if kind is None:
        return None
    pose = transform.pose_in_observer_frame.pose
    centre = shape.center
    position, orientation = compose_pose(
        (pose.x, pose.y, pose.z),
        _orientation_quat(pose),
        (centre.x, centre.y, centre.z),
        _orientation_quat(centre),
    )
    ox, oy, oz, theta = quat_to_ov(orientation)
    geometry = Geometry(
        center=Pose(
            x=position[0],
            y=position[1],
            z=position[2],
            o_x=ox,
            o_y=oy,
            o_z=oz,
            theta=math.degrees(theta),
        ),
        label=transform.reference_frame,
    )
    getattr(geometry, kind).CopyFrom(getattr(shape, kind))
    return geometry


# what RDK's gRPC layer answers when a store's ListUUIDs hands back a nil
# slice, which viam-labs/pack-sequencer 0.3.0 does for a store with nothing in
# it: the first read of every pack, after the checklist's reset has cleared
# each placed box's transform, or on a fresh machine (GPU machine, 2026-09-25).
# An empty store is a valid answer, not a failure of the read.
EMPTY_STORE_MARKER = "returned a nil response"


async def _list_uuids_or_empty(name: str, store: WorldStateStoreLike) -> list[Any]:
    """The store's uuid list, or an empty one when the store answered nil for
    having nothing to list. Any other error propagates."""
    try:
        return list(await store.list_uuids())
    except Exception as error:
        if EMPTY_STORE_MARKER not in str(error):
            raise
        LOGGER.info("store %r has nothing to list (nil response); relaying nothing from it", name)
        return []


async def read_store_obstacles(
    stores: Iterable[tuple[str, WorldStateStoreLike]],
) -> RelayReading:
    """Every world-frame geometry the named stores publish right now, with
    the counts. ``stores`` pairs each store's configured name with its client;
    the name only appears in logs. A store that raises makes the whole read
    raise, a planner run with half the stack being worse than a run that
    stops, with one exception: a store that answers nil for an empty list is
    read as empty (EMPTY_STORE_MARKER)."""
    obstacles: list[Geometry] = []
    relayed = 0
    skipped_not_in_world = 0
    skipped_no_geometry = 0
    for name, store in stores:
        for uuid in await _list_uuids_or_empty(name, store):
            transform = await store.get_transform(uuid)
            geometry = geometry_in_world(transform)
            if geometry is None:
                if transform.pose_in_observer_frame.reference_frame != WORLD_FRAME:
                    skipped_not_in_world += 1
                else:
                    skipped_no_geometry += 1
                continue
            obstacles.append(geometry)
            relayed += 1
    return RelayReading(
        obstacles=obstacles,
        relayed=relayed,
        skipped_not_in_world=skipped_not_in_world,
        skipped_no_geometry=skipped_no_geometry,
    )
