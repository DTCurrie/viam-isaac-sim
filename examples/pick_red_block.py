"""Pick-red-block client for the mock/real Isaac Sim pick-and-place cell.

Orchestration lives OUTSIDE the module (DEC-4): this script does the sequencing
a real pick needs - detect, open, approach, grasp, lift, release - talking to
either a live Viam machine or, with ``--mock``, an in-process mock boot of the
module so the sequencing logic can be exercised without a GPU. ``MoveToPosition``
is never used (DEC-13); every move drives joint positions (mock) or the motion
service's ``move`` (real). Depends only on the stdlib, viam-sdk and numpy;
``isaac_module`` is imported lazily, only inside the ``--mock`` code path.

Usage (real machine, W36/XC-8)::

    python examples/pick_red_block.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> \\
        --camera wrist-cam --arm pick-arm --gripper pick-grip \\
        --vision red-segmenter --motion builtin \\
        --block pick_cube --block-size-mm 60

Usage (in-process mock, no GPU, no running machine)::

    PYTHONPATH=src python examples/pick_red_block.py --mock

Real mode assumes the machine config carries the vision pipeline (color
detector -> detections-to-segments) and motion service from
``fragments/pick-and-place.json``, plus a gripper riding the arm's flange
(DEC-20 target block name is ``pick_cube``, not upstream's ``block_red``)::

    {
      "name": "pick-grip",
      "namespace": "rdk",
      "type": "gripper",
      "model": "viam:isaac-sim-devin:gripper",
      "frame": {"parent": "pick-arm", "translation": {"x": 0, "y": 0, "z": 115}},
      "attributes": {"world": "sim-world", "arm": "pick-arm"}
    },
    {
      "name": "builtin",
      "api": "rdk:service:motion",
      "model": "rdk:builtin:builtin",
      "attributes": {}
    },
    {
      "name": "red-detector",
      "api": "rdk:service:vision",
      "model": "color_detector",
      "attributes": {
        "detect_color": "#EA8D8D",
        "hue_tolerance_pct": 0.1,
        "segment_size_px": 100,
        "value_cutoff_pct": 0.15,
        "camera_name": "wrist-cam"
      }
    },
    {
      "name": "red-segmenter",
      "api": "rdk:service:vision",
      "model": "viam:vision:detections-to-segments",
      "attributes": {
        "detector_name": "red-detector",
        "camera_name": "wrist-cam",
        "mean_k": 5,
        "sigma": 1.25,
        "confidence_threshold_pct": 0.5,
        "infer_minimum_depth": true
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame, RectangularPrism, Transform, Vector3
from viam.proto.common import Geometry, WorldState
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

# python examples/pick_red_block.py (standalone, no PYTHONPATH set) needs the
# repo's src/ on sys.path before pickcell is importable; pytest already adds
# it (pyproject pythonpath = ["src"]), so this is a no-op there.
try:
    import pickcell  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pickcell.measurement
import pickcell.pipeline
import pickcell.poses
from pickcell.detector import RealDetector
from pickcell.measurement import (
    DEPTH_PROBE_RADIUS_MM,
    carry_clear_above_support_mm,
    centre_depth_mm,
    footprint_extents_mm,
    keepout_height_mm,
    measured_block_size_mm,
    parse_pcd,
    red_centroid_m,
    segment_stats,
    tallest_in_region_mm,
    top_face_centre_m,
)
from pickcell.movers import RealMover
from pickcell.obstacles import (
    KEEPOUT_HEIGHT_MM,
    REACHABLE_REGION_X_MM,
    REACHABLE_REGION_Y_MM,
    held_block_transform,
    obstacles_from_prop_geometries,
    pad_top_centre_mm,
    pick_area_keepout,
    reachable_region_mm,
    support_obstacle,
    table_recipe_unless_served,
    world_state,
)
from pickcell.pipeline import (
    DETECTED_BLOCK_POSE_MARKER,
    GRAB_DIAGNOSTICS_MARKER,
    HELD_BLOCK_TRANSFORM_MARKER,
    HOLD_SAMPLE_S,
    HOLD_SAMPLES_MARKER,
    JAW_MAX_BLOCK_MM,
    MEASURED_BLOCK_MARKER,
    MEASURED_TALLEST_MARKER,
    PLACE_SETTLE_S,
    PLACED_BLOCK_MARKER,
    RESET_MID_HOLD_MARKER,
    PickPipeline,
    TallestEstimate,
)
from pickcell.poses import (
    FINGERTIP_CLEARANCE_MM,
    FINGERTIP_OVERHANG_MM,
    POINTING_DOWN_O_Z,
    PRE_GRASP_STANDOFF_MM,
    SCAN_ATTEMPTS,
    TCP_CORRECTION_CAP_MM,
    _pointing_down,
    _pose_to_dict,
    _poses_match_mm,
    corrected_pose,
    grasp_height_mm,
    grasp_pose,
    look_pose_from,
    pre_grasp_pose,
    tallest_sweep_attempts,
    with_z,
)
from pickcell.scanners import FixedCameraScanner, _camera_client, camera_world_transform_mm

# tests/test_pick_red_block.py monkeypatches this module's HOLD_SAMPLE_S,
# PLACE_SETTLE_S, tallest_in_region_mm and tallest_sweep_attempts and expects
# the pipeline's OWN methods (pickcell.pipeline/measurement/poses, which read
# these as bare globals in the module they are defined in) to see the patched
# value - a plain re-export here would only rebind this module's copy. Give
# this module a __setattr__ that mirrors an assignment of one of those names
# onto the pickcell submodule that actually reads it, so
# ``monkeypatch.setattr(pick_red_block, "HOLD_SAMPLE_S", 0.05)`` (and its
# teardown, which restores the original value the same way) reaches the code
# that runs the pick.
_FORWARDED_ATTRS: dict[str, Any] = {
    # PickPipeline's methods live in pickcell.pipeline and read these names as
    # bare globals resolved against THAT module's own namespace - including
    # tallest_in_region_mm/tallest_sweep_attempts, which pipeline.py imports
    # from measurement.py/poses.py (an ``import as`` copies the reference, so
    # forwarding to measurement.py/poses.py would not reach pipeline.py's copy)
    "HOLD_SAMPLE_S": pickcell.pipeline,
    "PLACE_SETTLE_S": pickcell.pipeline,
    "tallest_in_region_mm": pickcell.pipeline,
    "tallest_sweep_attempts": pickcell.pipeline,
}


class _ForwardingModule(type(sys.modules[__name__])):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        target = _FORWARDED_ATTRS.get(name)
        if target is not None:
            setattr(target, name, value)


sys.modules[__name__].__class__ = _ForwardingModule

# ----------------------------------------------------------------------
# cell-default constants - the phase-1 seam cross-check asserts these against
# cell_layout (test_client_defaults_match_the_cell_layout_seam); they live
# here, not in pickcell, so the library stays cell-agnostic (phase 4's
# conductor passes zones straight from cell_layout instead)
# ----------------------------------------------------------------------

TABLE_DIMS_MM: tuple[float, float, float] = (1200.0, 800.0, 750.0)
TABLE_CENTER_MM: tuple[float, float, float] = (-1200.0, 0.0, 375.0)

# default scan spot: the source table's scatter-zone centre, inside ur20
# reach with the shipped cell's blocks in the 90 deg field of view; the
# height is ABOVE THE SUPPORT, because an absolute scan z is a floor-cell
# assumption (GPU run 13: the P5 table cell sent the camera 400 mm below the
# table top)
DEFAULT_LOOK_XY_MM = (-1025.0, 0.0)
# the wide scan height above measures poorly; a detection focuses down to
# this height instead (checklist item 3)
SCAN_HEIGHT_ABOVE_SUPPORT_MM = 650.0


def default_scan_pose(support_z_mm: float) -> Pose:
    x, y = DEFAULT_LOOK_XY_MM
    return _pointing_down(x, y, support_z_mm + SCAN_HEIGHT_ABOVE_SUPPORT_MM)


def table_obstacle() -> Geometry:
    """W4/README "Table recipe": 1200 x 800 x 750 mm box centred at
    (-1200, 0, 375) mm in the world frame, the motion-planner obstacle for the
    source table in the three-table sorting cell (arm base at the origin)."""
    x, y, z = TABLE_CENTER_MM
    dim_x, dim_y, dim_z = TABLE_DIMS_MM
    return Geometry(
        center=Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        box=RectangularPrism(dims_mm=Vector3(x=dim_x, y=dim_y, z=dim_z)),
        label="table",
    )


RANDOMIZE_REGION_MARGIN_MM = 50.0


def randomize_region_mm(
    margin_mm: float = RANDOMIZE_REGION_MARGIN_MM,
    face_z_mm: float = 0.0,
) -> tuple[list[float], list[float]]:
    """The table-footprint rectangle (table_obstacle's own x/y constants),
    inset by ``margin_mm`` so a randomized block cannot land hanging off the
    edge, at ``face_z_mm``: the surface the blocks rest on - the floor in the
    current fragment (``support_z_mm``), the table top in the P5 cell. The
    region ``randomize_props`` scatters the movable blocks within (checklist
    item 1: two consecutive picks with re-randomised blocks)."""
    center_x, center_y, _center_z = TABLE_CENTER_MM
    dim_x, dim_y, _dim_z = TABLE_DIMS_MM
    half_x = dim_x / 2.0 - margin_mm
    half_y = dim_y / 2.0 - margin_mm
    return (
        [center_x - half_x, center_y - half_y, face_z_mm],
        [center_x + half_x, center_y + half_y, face_z_mm],
    )


def _movable_prop_names_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.randomize_names:
        return ()
    return tuple(name.strip() for name in args.randomize_names.split(","))


def _randomize_region_mm_from_args(
    args: argparse.Namespace,
) -> tuple[list[float], list[float]] | None:
    if args.randomize_region is None:
        return None
    x0, y0, x1, y1 = args.randomize_region
    return ([x0, y0, args.support_z_mm], [x1, y1, args.support_z_mm])


def _look_pose_from_args(args: argparse.Namespace) -> Pose | None:
    if args.no_look:
        return None
    if args.look_at:
        return look_pose_from(args.look_at)
    return default_scan_pose(args.support_z_mm)


# checklist item 5 wants the block held 100 mm up for 5 s; item 6 resets the
# world mid-hold and expects the post-reset hooks (ARM-15/XC-5) to keep it held
DEFAULT_HOLD_S = 5.0
RESET_SETTLE_S = 2.0


# ----------------------------------------------------------------------
# real-mode collaborators kept at the script layer: RobotClient/argparse
# wiring, and the diagnostics/TCP-correction/reset-probe helpers that ride it
# ----------------------------------------------------------------------


async def _grab_diagnostics(arm: Arm, gripper: Gripper, block_name: str) -> dict[str, Any]:
    """After a failed grab: jaw angle + pad poses (gripper `tcp_pose`), the
    holding predicate's meta, and where the block actually is."""
    report: dict[str, Any] = {}
    try:
        report["arm_joint_state"] = dict(await arm.do_command({"command": "joint_state"}))
    except Exception as exc:  # noqa: BLE001 - diagnostics never mask the failure
        report["arm_joint_state_error"] = repr(exc)
    try:
        report["tcp_pose"] = dict(await gripper.do_command({"command": "tcp_pose"}))
    except Exception as exc:  # noqa: BLE001 - diagnostics never mask the grab failure
        report["tcp_pose_error"] = repr(exc)
    try:
        status = await gripper.is_holding_something()
        report["holding"] = {
            "is_holding_something": status.is_holding_something,
            "meta": status.meta,
        }
    except Exception as exc:  # noqa: BLE001
        report["holding_error"] = repr(exc)
    try:
        prim_path = f"/World/{block_name.replace('-', '_')}"
        report["block_prim_pose"] = dict(
            await arm.do_command({"command": "prim_world_pose", "prim_path": prim_path})
        )
    except Exception as exc:  # noqa: BLE001
        report["block_prim_pose_error"] = repr(exc)
    return report


async def _reset_mid_hold_report(
    world: Any,
    gripper: Any,
    diagnose: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Checklist item 6 (ARM-15/XC-5): reset the world while holding, give the
    post-reset hooks time to re-apply gains and re-command the jaw, and report
    whether the grip survived. ``world`` is anything with the world model's
    ``do_command`` - a Generic client in real mode, the model itself in mock."""
    before = await gripper.is_holding_something()
    await world.do_command({"command": "reset"})
    await asyncio.sleep(RESET_SETTLE_S)
    after = await gripper.is_holding_something()
    report: dict[str, Any] = {
        "holding_before_reset": bool(before.is_holding_something),
        "holding_after_reset": bool(after.is_holding_something),
        "meta_after": getattr(after, "meta", None),
    }
    if diagnose is not None:
        report["diagnostics"] = await diagnose()
    return report


async def _probe_depth(robot: RobotClient, args: argparse.Namespace, look_pose: Pose) -> None:
    """Move to the look pose, then compare the depth straight below the camera
    with the camera's commanded height above the support."""

    motion = MotionClient.from_robot(robot, args.motion)
    mover = RealMover(motion, args.gripper, args.camera)
    await mover.look_from(look_pose, world_state(None))
    camera = await _camera_client(robot, args.camera)
    pcd_bytes, _mime = await camera.get_point_cloud()
    xyz, _rgb = parse_pcd(pcd_bytes)
    measured = centre_depth_mm(xyz)
    expected = look_pose.z - args.support_z_mm
    if measured is None:
        print("depth probe: no points within the probe radius of the optical axis")
        return
    print(
        f"depth probe: centre depth {measured:.1f} mm, expected {expected:.1f} mm, "
        f"ratio {measured / expected:.3f} ({len(xyz)} points)"
    )


async def _tcp_correction(
    robot: RobotClient, gripper: Gripper, gripper_name: str
) -> tuple[float, float, float]:
    """(believed - physical) TCP position in mm: where the frame system says
    the gripper frame is, minus where the pads actually are (module
    `tcp_pose`). Positive z = the physical gripper hangs lower than believed."""
    identity = Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    believed = (
        await robot.transform_pose(
            PoseInFrame(reference_frame=gripper_name, pose=identity), "world"
        )
    ).pose
    physical = (await gripper.do_command({"command": "tcp_pose"}))["pad_center_midpoint_mm"]
    return (
        believed.x - float(physical[0]),
        believed.y - float(physical[1]),
        believed.z - float(physical[2]),
    )


async def _run_real(args: argparse.Namespace) -> Transform:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    robot = await RobotClient.at_address(args.address, opts)
    try:
        if args.probe_depth:
            probe_pose = (
                look_pose_from(args.look_at)
                if args.look_at
                else default_scan_pose(args.support_z_mm)
            )
            await _probe_depth(robot, args, probe_pose)
            probe_size_mm = args.block_size_mm if args.block_size_mm is not None else 0.0
            return held_block_transform(args.block, probe_size_mm, args.gripper)
        await robot.refresh()  # the resource list is a snapshot from connect time
        vision = VisionClient.from_robot(robot, args.vision)
        motion = MotionClient.from_robot(robot, args.motion)
        gripper = Gripper.from_robot(robot, args.gripper)
        arm = Arm.from_robot(robot, args.arm)
        world = Generic.from_robot(robot, args.world)

        # the wrist sweep must survive a disabled/missing side camera (GPU
        # item 3: --tallest-camera "" went straight to the ceiling fallback)
        side_scanner = (
            FixedCameraScanner(robot, args.tallest_camera) if args.tallest_camera else None
        )
        wrist_scanner = FixedCameraScanner(robot, args.camera)

        pipeline = PickPipeline(
            detector=RealDetector(
                robot, vision, args.camera, args.block_size_mm, args.support_z_mm
            ),
            mover=RealMover(motion, args.gripper, args.camera),
            gripper=gripper,
            block_name=args.block,
            block_size_mm=args.block_size_mm,
            gripper_name=args.gripper,
            look_pose=_look_pose_from_args(args),
            table=table_obstacle() if args.table else None,
            support_z_mm=args.support_z_mm,
            fingertip_overhang_mm=args.fingertip_overhang_mm,
            world=world,
            target_prop_name=args.block,
            movable_prop_names=_movable_prop_names_from_args(args),
            randomize_seed=args.randomize_seed,
            randomize_region_mm=_randomize_region_mm_from_args(args),
            randomize_size_range_mm=args.randomize_size_mm,
            side_scanner=side_scanner,
            wrist_scanner=wrist_scanner,
            place_prop_name=None if args.no_place else args.place_pad,
            diagnose=lambda: _grab_diagnostics(arm, gripper, args.block),
            tcp_correction=(
                None
                if args.no_tcp_correction
                else (lambda: _tcp_correction(robot, gripper, args.gripper))
            ),
            hold_s=args.hold_s,
            mid_hold_reset=(
                (
                    lambda: _reset_mid_hold_report(
                        world, gripper, lambda: _grab_diagnostics(arm, gripper, args.block)
                    )
                )
                if args.reset_mid_hold
                else None
            ),
        )
        return await pipeline.run()
    finally:
        await robot.close()


# ----------------------------------------------------------------------
# mock-mode collaborators (ARM-8, CAM-14) - lazy isaac_module import
# ----------------------------------------------------------------------

# Four distinct canned joint sets (degrees, 6 DOF) so look/pre-grasp/grasp/lift
# are visibly different moves even though the mock arm ignores IK entirely.
_MOCK_LOOK_JOINTS_DEG = [-90.0, -100.0, -80.0, -90.0, 90.0, 0.0]
_MOCK_PRE_GRASP_JOINTS_DEG = [-90.0, -90.0, -90.0, -90.0, 90.0, 0.0]
_MOCK_GRASP_JOINTS_DEG = [-90.0, -70.0, -100.0, -100.0, 90.0, 0.0]
_MOCK_LIFT_JOINTS_DEG = [-90.0, -95.0, -85.0, -90.0, 90.0, 0.0]
_MOCK_WAYPOINT_JOINTS_DEG = [-90.0, -95.0, -87.0, -93.0, 90.0, 0.0]
# the mock detector reports the fabricated pixel centroid straight through as
# world frame (no frame system in mock mode), so its (x, y) never lands on
# the scan pose's - the free block-airspace waypoint move always fires here
_MOCK_JOINT_SETS_DEG = [
    _MOCK_LOOK_JOINTS_DEG,
    _MOCK_WAYPOINT_JOINTS_DEG,
    _MOCK_PRE_GRASP_JOINTS_DEG,
    _MOCK_GRASP_JOINTS_DEG,
    _MOCK_LIFT_JOINTS_DEG,
]


class MockDetector:
    """``block_size_mm`` None measures the size from the mock camera's own
    red pixels (footprint x/y only - the mock scene is a flat depth plane
    with no independent height axis to cross-check, unlike RealDetector)."""

    def __init__(self, camera: Any, block_size_mm: float | None = None) -> None:
        self._camera = camera
        self._block_size_mm = block_size_mm
        self._last_measurement: dict[str, Any] | None = None

    def last_measurement(self) -> dict[str, Any] | None:
        return self._last_measurement

    async def block_pose_world(self) -> Pose:
        pcd_bytes, _ = await self._camera.get_point_cloud()
        xyz_m, rgb = parse_pcd(pcd_bytes)
        if rgb is None:
            raise RuntimeError("mock point cloud carries no colour channel")
        x_m, y_m, z_m = red_centroid_m(xyz_m, rgb)
        pose = Pose(
            x=x_m * 1000.0,
            y=y_m * 1000.0,
            z=z_m * 1000.0,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        )
        print(
            "  mock detector: no frame system in mock mode - the camera-frame "
            f"centroid is treated as world frame: {_pose_to_dict(pose)}"
        )
        if self._block_size_mm is not None:
            self._last_measurement = None
        else:
            footprint_mm = footprint_extents_mm(xyz_m)
            measured = (
                measured_block_size_mm([footprint_mm[0], footprint_mm[1]])
                if footprint_mm is not None
                else None
            )
            self._last_measurement = (
                {
                    "footprint_mm": [footprint_mm[0], footprint_mm[1]],
                    "height_mm": None,
                    "size_mm": measured[0],
                }
                if measured is not None
                else None
            )
        return pose


class MockSideScanner:
    """A ``TallestScanner`` over the mock side camera (seam's mock world
    mapping, no frame system in mock mode):
    ``xyz_world_mm = (x_cam, z_cam, -y_cam) * 1000``, NaN rows (no hit)
    dropped."""

    def __init__(self, camera: Any) -> None:
        self._camera = camera

    async def scan_world_mm(self) -> Any:
        import numpy as np

        pcd_bytes, _mime = await self._camera.get_point_cloud()
        xyz_cam_m, _rgb = parse_pcd(pcd_bytes)
        x_cam, y_cam, z_cam = xyz_cam_m[:, 0], xyz_cam_m[:, 1], xyz_cam_m[:, 2]
        world_mm = np.column_stack([x_cam, z_cam, -y_cam]) * 1000.0
        valid = ~np.isnan(world_mm).any(axis=1)
        return world_mm[valid]


class MockMover:
    def __init__(self, arm: Any, joint_sets_deg: Sequence[Sequence[float]]) -> None:
        self._arm = arm
        self._joint_sets_deg = list(joint_sets_deg)
        self._call_count = 0

    async def look_from(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        await self.move_to(pose, world_state, linear)

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        if self._call_count >= len(self._joint_sets_deg):
            raise RuntimeError("mock mover: no more canned joint sets for this pick sequence")
        joints = self._joint_sets_deg[self._call_count]
        self._call_count += 1
        print(f"  mock mover: requested pose {_pose_to_dict(pose)} -> canned joints {joints}")
        await self._arm.move_to_joint_positions(JointPositions(values=list(joints)))


def _ensure_mock_sim_booted() -> Any:
    from isaac_module.sim_manager import SimConfig, SimManager

    manager = SimManager.get()
    if not manager._booted.is_set():
        sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
        sim_thread.start()
        manager.ensure_booted(SimConfig(mock=True))
    return manager


async def _run_mock(args: argparse.Namespace) -> Transform:
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from isaac_module.models.arm import IsaacArm
    from isaac_module.models.camera import IsaacCamera
    from isaac_module.models.gripper import IsaacGripper
    from isaac_module.models.world import IsaacWorld

    def config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
        return ComponentConfig(name=name, attributes=dict_to_struct(attrs))

    _ensure_mock_sim_booted()

    world_name = "mock-pick-world"
    arm_name = "mock-pick-arm"
    camera_name = "mock-wrist-cam"
    side_camera_name = "mock-side-cam"
    gripper_name = "mock-pick-grip"

    world = IsaacWorld.new(config(world_name, {"mock": True}), {})
    arm = IsaacArm.new(config(arm_name, {"world": world_name, "asset": "ur5e"}), {})
    camera_attrs: dict[str, Any] = {"world": world_name, "depth": True}
    if args.mock_block_size_mm is not None:
        camera_attrs["block_size_mm"] = args.mock_block_size_mm
    camera = IsaacCamera.new(config(camera_name, camera_attrs), {})
    # three distractors at distinct heights (45/90/60 mm), distinct columns
    # and staggered depths so nothing overlaps in the fabricated side view
    side_blocks = [
        {
            "rgb": [200, 60, 60],
            "size_mm": 60.0,
            "height_mm": 45.0,
            "column_offset_px": -220,
            "depth_m": 0.80,
        },
        {
            "rgb": [60, 200, 60],
            "size_mm": 60.0,
            "height_mm": 90.0,
            "column_offset_px": 0,
            "depth_m": 0.90,
        },
        {
            "rgb": [60, 60, 200],
            "size_mm": 60.0,
            "height_mm": 60.0,
            "column_offset_px": 220,
            "depth_m": 1.00,
        },
    ]
    side_camera = IsaacCamera.new(
        config(
            side_camera_name,
            {"world": world_name, "depth": True, "view": "side", "blocks": side_blocks},
        ),
        {},
    )
    gripper = IsaacGripper.new(
        config(gripper_name, {"world": world_name, "arm": arm_name, "mock_object_width_m": 0.05}),
        {},
    )

    pipeline = PickPipeline(
        detector=MockDetector(camera, args.block_size_mm),
        mover=MockMover(arm, _MOCK_JOINT_SETS_DEG),
        gripper=gripper,
        verify_detection_height=False,
        block_name=args.block,
        block_size_mm=args.block_size_mm,
        gripper_name=gripper_name,
        look_pose=_look_pose_from_args(args),
        world=world,
        target_prop_name=args.block,
        movable_prop_names=_movable_prop_names_from_args(args),
        randomize_seed=args.randomize_seed,
        randomize_region_mm=_randomize_region_mm_from_args(args),
        randomize_size_range_mm=args.randomize_size_mm,
        side_scanner=MockSideScanner(side_camera),
        place_prop_name=None if args.no_place else args.place_pad,
        hold_s=args.hold_s,
        mid_hold_reset=(
            (lambda: _reset_mid_hold_report(world, gripper)) if args.reset_mid_hold else None
        ),
    )
    return await pipeline.run()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_randomize_size_mm(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--randomize-size-mm wants lo,hi, got {value!r}")
    try:
        lo, hi = (float(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--randomize-size-mm wants two numbers, got {value!r}")
    if not (lo > 0 and hi > 0 and lo <= hi):
        raise argparse.ArgumentTypeError(f"--randomize-size-mm wants 0 < lo <= hi, got {value!r}")
    return (lo, hi)


def _parse_randomize_region_mm(value: str) -> tuple[float, float, float, float]:
    """x0,y0,x1,y1 (mm, world, table-top plane) -> the flat corner tuple;
    ``main`` adds the z at --support-z-mm to build ``randomize_region_mm``."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--randomize-region wants x0,y0,x1,y1, got {value!r}")
    try:
        x0, y0, x1, y1 = (float(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--randomize-region wants four numbers, got {value!r}")
    if not (x0 < x1 and y0 < y1):
        raise argparse.ArgumentTypeError(
            f"--randomize-region wants x0 < x1 and y0 < y1, got {value!r}"
        )
    return (x0, y0, x1, y1)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true", help="run in-process against the module's mock backend"
    )
    parser.add_argument("--address", help="machine address (required unless --mock)")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--camera", default="wrist-cam")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--gripper", default="pick-grip")
    # renamed from block-segmenter when the six-color cell made segmenters per-color
    parser.add_argument("--vision", default="red-segmenter")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--block", default="block_red_1")
    parser.add_argument(
        "--block-size-mm",
        type=float,
        default=None,
        help="override the target block's size (mm) instead of measuring it from the "
        "focused detection's point cloud; omit to measure (the default)",
    )
    parser.add_argument(
        "--randomize-size-mm",
        type=_parse_randomize_size_mm,
        default=None,
        metavar="LO,HI",
        help="lo,hi (mm) size range added to the --randomize-seed randomize_props call as "
        "size_range_mm, for every movable name; warns (never fails) if the measured size "
        "falls outside it (default off, byte-identical randomize_props payload)",
    )
    parser.add_argument(
        "--mock-block-size-mm",
        type=float,
        default=None,
        help="test-only: fabricate the --mock wrist camera's red block at this metric "
        "size (mm) instead of the default fixed pixel rectangle",
    )
    parser.add_argument(
        "--place-pad", default="place_pad_red", help="fixed prop to set the block down on"
    )
    parser.add_argument(
        "--no-place", action="store_true", help="release at the lift pose instead of placing"
    )
    parser.add_argument("--world", default="sim-world", help="the isaac-sim world component name")
    parser.add_argument(
        "--hold-s",
        type=float,
        default=DEFAULT_HOLD_S,
        help="seconds to hold at the lift pose sampling IsHoldingSomething at 1 Hz "
        "(checklist item 5; 0 = release immediately)",
    )
    parser.add_argument(
        "--reset-mid-hold",
        action="store_true",
        help='send the world {"command": "reset"} while holding and require the grip to '
        "survive the post-reset re-tune (checklist item 6, ARM-15/XC-5)",
    )
    parser.add_argument(
        "--look-at",
        default=None,
        help="x,y,z (mm, world) the wrist camera is moved to, pointing down, before detecting "
        f"(default {DEFAULT_LOOK_XY_MM[0]:.0f},{DEFAULT_LOOK_XY_MM[1]:.0f},"
        f"<--support-z-mm + {SCAN_HEIGHT_ABOVE_SUPPORT_MM:.0f}>: within ur20 reach, with the "
        "fragment's blocks inside the 90 deg field of view)",
    )
    parser.add_argument(
        "--no-look", action="store_true", help="detect from wherever the arm already is"
    )
    parser.add_argument(
        "--support-z-mm",
        type=float,
        default=750.0,
        help="height of the surface the block rests on (the three-table cell's table top; "
        "0 was the old floor cell)",
    )
    parser.add_argument(
        "--fingertip-overhang-mm",
        type=float,
        default=FINGERTIP_OVERHANG_MM,
        help="how far the fingertips extend past the TCP (measured 19 mm on the 2F-85)",
    )
    parser.add_argument(
        "--probe-depth",
        action="store_true",
        help="only move to the look pose and report the wrist camera's depth straight down "
        "against its commanded height (a depth-scale check); no pick",
    )
    parser.add_argument(
        "--no-tcp-correction",
        action="store_true",
        help="skip measuring the believed-vs-physical TCP offset at pre-grasp",
    )
    parser.add_argument(
        "--randomize-seed",
        type=int,
        default=None,
        help="re-randomise the movable blocks' positions (world DoCommand randomize_props) "
        "before this pick, within the table-top region (checklist item 1: two consecutive "
        "picks with re-randomised blocks); default off",
    )
    parser.add_argument(
        "--randomize-names",
        default=None,
        metavar="NAME,NAME,...",
        help="comma-separated prop names to randomize (movable_prop_names); default off "
        "keeps today's behavior of deriving every non-fixed prop with known dims",
    )
    parser.add_argument(
        "--randomize-region",
        type=_parse_randomize_region_mm,
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="x0,y0,x1,y1 (mm, world) randomize region at --support-z-mm (randomize_region_mm); "
        "default off keeps today's reachable_region_mm(support) fallback",
    )
    parser.add_argument(
        "--tallest-camera",
        default="side-cam",
        help="fixed side camera measuring the tallest scattered object (phase 4), primary "
        "source for the dynamic keep-out/carry heights; empty string disables it, falling "
        "back to the wrist sweep then the --randomize-size-mm range max",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="add the W4 table box (README recipe) as a motion obstacle - only for a scene "
        "whose table is NOT served live; the shipped fragment serves its table via "
        "prop_geometries, and the flag is dropped automatically when the live box is present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.mock and not args.address:
        print("FAILED: --address is required unless --mock is set")
        return 1

    try:
        if args.mock:
            asyncio.run(_run_mock(args))
        else:
            asyncio.run(_run_real(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean exit code
        print(f"FAILED: {exc!r}")
        print(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
