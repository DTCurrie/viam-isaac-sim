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

# ---------------------------------------------------------------------------
# The Robotiq EPick, the vacuum gripper the palletizing demo runs.
#
# Source: viam-labs/robotiq-epick, epick/geometry.go and epick/epick_model.json
# at commit 81aa5c75, read 2026-09-22. That repo measured every number off the
# full-resolution CAD exports under epick/meshes with its own fit script, and
# epick_model.json (vendored under kinematics_files/) is what its driver serves
# from GetKinematics. Ratings are from the EPick instruction manual, e-Series
# edition of 2021-07-09, section 6.2 (mass, payload, gripping and release
# times) and section 7 (the automatic mode's retry window).
#
# Every length is in the GRIPPER FRAME the driver's kinematics use: millimetres,
# z = 0 at the TCP, the suction plane a machine config's frame
# `translation.z: 196` puts there, and -z back toward the arm flange at
# z = -196. The rendered cups stop 10 mm short of the TCP and their colliders
# 26 mm short, so a grab approach that drives the TCP onto a box is never
# refused by the planner. Each round part is drawn as a cylinder and collided
# as the box epick_model.json carries: Viam's wire has no cylinder geometry, so
# those boxes are what the planner on a real machine sees.
EPICK: dict[str, Any] = {
    "kind": "gripper",
    "tcp_offset_m": 0.196,
    # gripper mass including the coupling, manual section 6.2
    "mass_kg": 0.706,
    "kinematics_path": _KINEMATICS_FILES_DIR / "epick_model.json",
    "body": {
        "radius_mm": 35.5,
        # drawn 129 mm long: its rear boss reaches 3 mm past the flange, into
        # the arm's own end-effector space. Collision stops at the flange plane
        "visual_length_mm": 129.0,
        "visual_center_z_mm": -134.5,
        "collision_mm": (71.0, 71.0, 126.0),
        "collision_center_z_mm": -133.0,
    },
    "plate": {
        "size_mm": (204.5, 126.3, 3.2),
        "center_z_mm": -68.4,
    },
    "cups": {
        "radius_mm": 24.5,
        # a 159.5 x 81.3 mm rectangular pattern, one cup per quadrant, named
        # as epick_model.json names its links
        "names": ("cup-xp-yp", "cup-xp-yn", "cup-xn-yp", "cup-xn-yn"),
        "offsets_mm": ((79.75, 40.65), (79.75, -40.65), (-79.75, 40.65), (-79.75, -40.65)),
        "visual_length_mm": 60.0,
        "visual_center_z_mm": -40.0,
        "tip_z_mm": -10.0,
        "collision_mm": (49.0, 49.0, 44.0),
        "collision_center_z_mm": -48.0,
        "tcp_clearance_z_mm": -26.0,
    },
    # "exceeding 4.5 kg per air node could induce damage", manual section 6.2.
    # The number Robotiq stands behind, used instead of the area formula since
    # the CAD's 49 mm cup matches neither the 40 nor the 55 mm cup it rates
    "payload_per_cup_kg": 4.5,
    # the holding force is the cup's inside area times the vacuum, manual
    # section 6.2.1, with 1 % of vacuum worth 1.013 kPa and 80 % the maximum
    # the gripper regulates to (section 6.2). At 80 % a 49 mm cup develops
    # 153 N, and that is the load that breaks an attachment in the sim
    "max_vacuum_pct": 80.0,
    "kpa_per_vacuum_pct": 1.013,
    # gripping and release times for one 40 mm cup, manual section 6.2
    "grip_time_ms": 150,
    "release_time_ms": 180,
}
# the prim name the EPick's body is authored at, under the arm link it rides
EPICK_PRIM = "EPick"

# The cup stops this far above a payload's top face rather than on it, and the
# attachment points' clearance offset equals it, so Isaac's surface gripper
# starts its raycast at the box's top face and draws the box up to the cups.
# Driving a rigid tool onto a rigid box makes contact the arm cannot push
# through, so it stalls short of its commanded pose; and the plugin displaces a
# gripped object by twice the distance its ray travels past the clearance
# before it hits, so the gap and the clearance have to be the same number. A
# vacuum mechanism constant, not packing geometry: it holds regardless of which
# box or which cell this runs in. Measured on the GPU machine 2026-09-22: with
# both at 5 mm a box is pulled 5.0 mm to the cups on close.
CUP_APPROACH_GAP_MM = 5.0
