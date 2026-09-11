"""The catalog of Isaac Sim assets addressable by short name in component
config (arm/gripper/base ``model`` and ``kind``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# the module archive ships these files (see kinematics_files/README.md for
# their upstream source and how to refresh them), so a known asset's
# kinematics never depends on a network fetch at RPC time.
_KINEMATICS_FILES_DIR = Path(__file__).resolve().parent / "kinematics_files"


def _packaged_kinematics_uri(filename: str) -> str:
    return (_KINEMATICS_FILES_DIR / filename).as_uri()


# the 6 UR joints in SVA (spatial vector algebra) order - the order the arm
# component's kinematics/motion planning expects, which need not match the
# articulation's PhysX dof order.
UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Assets shipped on the Isaac Sim nucleus/content server, addressable by a
# short name in component config. Paths are relative to the assets root.
# Where Isaac 5.0 moved an asset, the 5.0 path is listed first with the 4.x
# path as a fallback, and the first candidate that exists is used.
KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "ur3e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"],
        "kinematics": _packaged_kinematics_uri("ur3e.json"),
        "joint_names": UR_JOINT_NAMES,
        "ee_prim": "wrist_3_link",
    },
    "ur5e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": _packaged_kinematics_uri("ur5e.json"),
        # verified: the usd asset's own base link is rotated 180deg about Z
        # relative to the kinematics frame.
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
        "ee_prim": "wrist_3_link",
    },
    # No distinct ur7e USD ships in Isaac Sim 5.0: the assets root's
    # UniversalRobots folder lists ur10, ur10e, ur16e, ur20, ur3, ur30, ur3e,
    # ur5, ur5e only (checked against the asset server directly, 2026-09-10).
    # Spawns the ur5e mesh as a geometry stand-in, carrying ur5e's own
    # measured base_frame_correction since it is the same mesh file, paired
    # with the real ur7e's kinematics.
    "ur7e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": _packaged_kinematics_uri("ur7e.json"),
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
        "ee_prim": "wrist_3_link",
    },
    # ur3e correction is unchecked. Deliberately no entry (identity) until
    # verified.
    "ur20": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"],
        "kinematics": _packaged_kinematics_uri("ur20.json"),
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
        "ee_prim": "wrist_3_link",
    },
    # No kinematics: viam-modules/viam-franka-arm (the real driver) publishes
    # no kinematics file the way the UR module does, so GetKinematics /
    # GetEndPosition / MoveToPosition are unavailable for a resolved Franka.
    # The simulates.json row is verified: false for this reason. No ee_prim
    # either, since the flange link name is unconfirmed without one.
    "franka": {
        "usd": [
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ]
    },
    # A gripper asset, only ever referenced UNDER an arm prim by
    # create_gripper (never spawned free-standing). closed_deg is per Isaac
    # release, compat.caps().gripper_closed_deg. Never use the ur5e.usd
    # "Gripper" variant, and do not hard-code sibling filenames beyond
    # Robotiq_2F_85_edit.usd.
    "robotiq_2f_85": {
        "kind": "gripper",
        "usd": ["/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"],
        "drive_joint": "finger_joint",
        # the nominal spec value. At attach the drive joint's authored lower
        # limit wins when readable (~7.8 deg on the 5.0 asset)
        "open_deg": 0.0,
        # flange -> TCP along tool +Z: the fingertip PAD CENTER, measured on
        # the GPU (2026-08-28): pads span 115-153 mm, center 134. The earlier
        # spec value was 115.
        "tcp_offset_m": 0.134,
        # the single box GetGeometries / the SVA carry, spanning flange ->
        # fingertips. Measured on the GPU: pads reach 0.153 m. The earlier
        # 150 mm box, "centered on the TCP", put 75 mm of virtual gripper
        # below the fingertips.
        "jaw_box_mm": (36.0, 146.0, 153.0),
        "fingertip_reach_m": 0.153,
    },
    "jetbot": {
        "usd": [
            "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            "/Isaac/Robots/Jetbot/jetbot.usd",
        ],
        "wheel_joints": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_radius": 0.03,
        "wheel_base": 0.1125,
        # the asset's root is a plain Xform that physics never moves; the
        # chassis rigid body is what drives, so poses are read from it
        "body_prim": "chassis",
    },
}
