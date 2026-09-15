import time

import pytest

from isaac_module.handles.vacuum import (
    DEFAULT_GRAB_DELAY_MS,
    author_suction_joint,
    remove_suction_joint,
    select_payload_for_cup,
)

# --- protocol conformance against the mock ---------------------------------


def test_create_vacuum_gripper_unknown_arm_raises(sim):
    with pytest.raises(ValueError, match="not attached to the sim"):
        sim.create_vacuum_gripper("vacuum-bad-arm", {"world": "isaac-world", "arm": "no-such-arm"})


def test_grab_with_attach_prop_holds(sim):
    sim.create_arm("vacuum-arm-a", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-a",
        {"world": "isaac-world", "arm": "vacuum-arm-a", "mock_attach_prop": "box-1"},
    )

    assert vacuum.is_holding() is False
    vacuum.grab()
    assert vacuum.is_holding() is True


def test_grab_with_no_attach_prop_never_holds(sim):
    sim.create_arm("vacuum-arm-b", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper("vacuum-b", {"world": "isaac-world", "arm": "vacuum-arm-b"})

    vacuum.grab()
    assert vacuum.is_holding() is False


def test_open_after_grab_releases(sim):
    sim.create_arm("vacuum-arm-c", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-c",
        {"world": "isaac-world", "arm": "vacuum-arm-c", "mock_attach_prop": "box-1"},
    )

    vacuum.grab()
    assert vacuum.is_holding() is True
    vacuum.open()
    assert vacuum.is_holding() is False


def test_is_moving_false_before_grab_and_after_zero_delay_grab(sim):
    sim.create_arm("vacuum-arm-d", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-d",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-d",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 0,
        },
    )

    assert vacuum.is_moving() is False
    vacuum.grab()
    assert vacuum.is_moving() is False


def test_is_moving_true_during_grab_delay_window_then_false(sim):
    sim.create_arm("vacuum-arm-dd", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-dd",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-dd",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 50,
        },
    )

    assert vacuum.is_moving() is False
    vacuum.grab()
    assert vacuum.is_moving() is True
    time.sleep(0.1)
    assert vacuum.is_moving() is False
    # the weld itself is instant, so holding is already reportable during the window
    assert vacuum.is_holding() is True


def test_grab_delay_defaults_to_the_epick_default(sim):
    sim.create_arm("vacuum-arm-dg", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-dg",
        {"world": "isaac-world", "arm": "vacuum-arm-dg", "mock_attach_prop": "box-1"},
    )

    assert DEFAULT_GRAB_DELAY_MS == 1000
    vacuum.grab()
    assert vacuum.is_moving() is True
    assert vacuum._grab_delay_s == pytest.approx(1.0)


def test_open_clears_the_grab_window(sim):
    sim.create_arm("vacuum-arm-do", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-do",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-do",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 5000,
        },
    )

    vacuum.grab()
    assert vacuum.is_moving() is True
    vacuum.open()
    assert vacuum.is_moving() is False


def test_dof_names_is_empty(sim):
    sim.create_arm("vacuum-arm-e", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper("vacuum-e", {"world": "isaac-world", "arm": "vacuum-arm-e"})

    assert vacuum.dof_names() == []


def test_poll_state_matches_is_moving_and_is_holding(sim):
    sim.create_arm("vacuum-arm-f", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-f",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-f",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 0,
        },
    )

    vacuum.grab()
    moving, holding = vacuum.poll_state()
    assert (moving, holding) == (vacuum.is_moving(), vacuum.is_holding())
    assert moving is False
    assert holding is True


# --- select_payload_for_cup: a pure decision --------------------------------


def test_select_payload_accepts_a_box_within_gap_and_overlap():
    cup_position = (0.0, 0.0, 0.20)
    candidates = [("box-1", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05))]
    # top face at 0.125, gap to cup face (0.20) is 0.075
    assert select_payload_for_cup(cup_position, 0.07, 0.1, candidates) == "box-1"


def test_select_payload_rejects_too_far_vertically():
    cup_position = (0.0, 0.0, 0.50)
    candidates = [("box-1", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05))]
    # top face at 0.125, gap to cup face (0.50) is 0.375, past max_gap_m
    assert select_payload_for_cup(cup_position, 0.07, 0.1, candidates) is None


def test_select_payload_rejects_footprint_clear_of_the_cup():
    cup_position = (0.0, 0.0, 0.20)
    # centered a metre away in X: same height, but nowhere near the cup's footprint
    candidates = [("box-1", (1.0, 0.0, 0.10), (0.05, 0.05, 0.05))]
    assert select_payload_for_cup(cup_position, 0.07, 0.1, candidates) is None


def test_select_payload_tie_breaks_by_smallest_gap_then_name():
    cup_position = (0.0, 0.0, 0.20)
    candidates = [
        ("box-far", (0.0, 0.0, 0.08), (0.05, 0.05, 0.05)),  # top 0.105, gap 0.095
        ("box-near", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05)),  # top 0.125, gap 0.075
        ("box-tie-b", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05)),  # same gap as box-near
    ]
    assert select_payload_for_cup(cup_position, 0.07, 0.1, candidates) == "box-near"

    # box-near removed: the two remaining candidates tie on gap, so the name
    # ordering ("box-near" < "box-tie-b" doesn't apply here) breaks the tie
    tied = [
        ("box-tie-b", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05)),
        ("box-tie-a", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05)),
    ]
    assert select_payload_for_cup(cup_position, 0.07, 0.1, tied) == "box-tie-a"


def test_select_payload_accepts_a_penetrating_cup():
    # cup face at 0.124, a millimetre BELOW the box's top face (0.125) - the
    # cup pressed firmly onto the box, not hovering above it
    cup_position = (0.0, 0.0, 0.124)
    candidates = [("box-1", (0.0, 0.0, 0.10), (0.05, 0.05, 0.05))]
    assert select_payload_for_cup(cup_position, 0.07, 0.01, candidates) == "box-1"


# --- IsaacVacuumHandle._payload_candidates: fixed props never qualify ------


class _FakeXformPrim:
    def __init__(self, pose):
        self._pose = pose

    def get_world_pose(self):
        return self._pose


class _FakeIsaacNamespace:
    def __init__(self, world_poses: dict[str, tuple]):
        self._world_poses = world_poses

    def SingleXFormPrim(self, path):
        return _FakeXformPrim(self._world_poses[path])


class _FakeSim:
    def __init__(self, prop_specs: dict, world_poses: dict[str, tuple]):
        self._prop_specs = prop_specs
        self._isaac = _FakeIsaacNamespace(world_poses)


def test_payload_candidates_excludes_fixed_props():
    from isaac_module.handles.vacuum import IsaacVacuumHandle

    prop_specs = {
        "pallet": {"name": "pallet", "fixed": True, "size": 0.4, "scale": (1.0, 1.0, 0.1)},
        "box-1": {"name": "box-1", "size": 0.05, "scale": (1.0, 1.0, 1.0)},
    }
    world_poses = {
        "/World/pallet": ((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)),
        "/World/box-1": ((0.0, 0.0, 0.10), (1.0, 0.0, 0.0, 0.0)),
    }
    fake_sim = _FakeSim(prop_specs, world_poses)
    handle = IsaacVacuumHandle(fake_sim, "vacuum-x", "/World/Arm/VacuumTool", 0.07, 0.1, 0.0)

    candidates = handle._payload_candidates()

    names = {name for name, _pos, _dims in candidates}
    assert names == {"box-1"}


def test_fixed_prop_under_the_cup_is_never_the_only_candidate_selected():
    from isaac_module.handles.vacuum import IsaacVacuumHandle

    # the pallet deck is directly under the cup and is the ONLY prop in the
    # scene - a fixed-prop-only scene must still select nothing
    prop_specs = {
        "pallet": {"name": "pallet", "fixed": True, "size": 0.4, "scale": (1.0, 1.0, 0.1)},
    }
    world_poses = {
        "/World/pallet": ((0.0, 0.0, 0.19), (1.0, 0.0, 0.0, 0.0)),
    }
    fake_sim = _FakeSim(prop_specs, world_poses)
    handle = IsaacVacuumHandle(fake_sim, "vacuum-y", "/World/Arm/VacuumTool", 0.07, 0.1, 0.0)

    cup_position = (0.0, 0.0, 0.20)
    chosen = select_payload_for_cup(
        cup_position, handle._cup_side_m, handle._max_payload_gap_m, handle._payload_candidates()
    )

    assert chosen is None


# --- author_suction_joint / remove_suction_joint: pure decisions over fakes -


class _FakeAttr:
    def __init__(self) -> None:
        self.value = None

    def Set(self, value):
        self.value = value


class _FakeRel:
    def __init__(self) -> None:
        self.targets = None

    def SetTargets(self, targets):
        self.targets = targets


class _FakeJoint:
    def __init__(self, path: str) -> None:
        self.path = path
        self.body0 = _FakeRel()
        self.body1 = _FakeRel()
        self.attrs: dict[str, object] = {}

    def CreateBody0Rel(self):
        return self.body0

    def CreateBody1Rel(self):
        return self.body1

    def CreateLocalPos0Attr(self, value):
        self.attrs["local_pos0"] = value

    def CreateLocalRot0Attr(self, value):
        self.attrs["local_rot0"] = value

    def CreateLocalPos1Attr(self, value):
        self.attrs["local_pos1"] = value

    def CreateLocalRot1Attr(self, value):
        self.attrs["local_rot1"] = value


class _FakeUsdPhysics:
    class FixedJoint:
        @staticmethod
        def Define(stage, path):
            joint = _FakeJoint(path)
            stage.joints[path] = joint
            return joint


class _FakeSdf:
    class Path(str):
        pass


class _FakeGf:
    class Vec3f(tuple):
        def __new__(cls, x, y, z):
            return super().__new__(cls, (x, y, z))

    class Quatf(tuple):
        def __new__(cls, w, xyz):
            return super().__new__(cls, (w, *xyz))


class _FakePrim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self):
        return self._valid


class _FakeStage:
    def __init__(self) -> None:
        self.joints: dict[str, _FakeJoint] = {}
        self.removed: list[str] = []

    def GetPrimAtPath(self, path):
        return _FakePrim(path in self.joints)

    def RemovePrim(self, path):
        self.removed.append(path)
        self.joints.pop(path, None)


def test_author_suction_joint_wires_body0_body1_and_identity_frames():
    stage = _FakeStage()
    author_suction_joint(
        _FakeUsdPhysics, _FakeSdf, _FakeGf, stage, "/World/Tool/Joint", "/World/Tool", "/World/Box"
    )

    joint = stage.joints["/World/Tool/Joint"]
    assert joint.body0.targets == ["/World/Tool"]
    assert joint.body1.targets == ["/World/Box"]
    assert joint.attrs["local_pos0"] == (0.0, 0.0, 0.0)
    assert joint.attrs["local_pos1"] == (0.0, 0.0, 0.0)
    assert joint.attrs["local_rot0"] == (1.0, 0.0, 0.0, 0.0)
    assert joint.attrs["local_rot1"] == (1.0, 0.0, 0.0, 0.0)


def test_remove_suction_joint_removes_and_reports_present():
    stage = _FakeStage()
    author_suction_joint(
        _FakeUsdPhysics, _FakeSdf, _FakeGf, stage, "/World/Tool/Joint", "/World/Tool", "/World/Box"
    )

    assert remove_suction_joint(stage, "/World/Tool/Joint") is True
    assert "/World/Tool/Joint" not in stage.joints


def test_remove_suction_joint_reports_absent():
    stage = _FakeStage()
    assert remove_suction_joint(stage, "/World/Tool/Joint") is False
    assert stage.removed == []


def test_a_weld_records_the_pose_the_cup_and_payload_are_already_in():
    """A FixedJoint pulls body0's joint frame onto body1's, so identity local
    frames on both sides ask PhysX to make the payload's origin coincide with
    the tool's, and it drags the box through the cup to get there. Recording
    the relative pose is what leaves the box where it was picked up."""
    from isaac_module.handles.vacuum import suction_joint_local_frame

    identity = (1.0, 0.0, 0.0, 0.0)
    tool_pose = ((0.0, 0.0, 1.0), identity)
    payload_pose = ((0.0, 0.0, 0.9), identity)

    local_pos, local_rot = suction_joint_local_frame(tool_pose, payload_pose)

    # the payload sits 100 mm below the anchor's origin, and that is exactly
    # what the anchor side of the weld has to carry. It goes on the anchor
    # because a joint frame is read in its own body's local space and the
    # payload is a cube prim with a non-uniform scale, which would multiply it
    assert local_pos == pytest.approx((0.0, 0.0, -0.1))
    assert local_rot == pytest.approx(identity)
    # the defect this replaces would have recorded the origin
    assert local_pos != (0.0, 0.0, 0.0)


def test_the_weld_anchors_on_the_arm_link_not_the_tool_body():
    """The tool is a free rigid body bolted to the wrist by its own fixed
    joint, not a link of the arm's articulation. Welding a payload to it puts
    two maximal-coordinate joints in series off an articulation, which PhysX
    resolves with an impulse that throws the arm across the cell. The payload
    has to attach to a link the articulation solver already owns."""
    from isaac_module.handles.vacuum import IsaacVacuumHandle

    handle = IsaacVacuumHandle.__new__(IsaacVacuumHandle)
    handle._tool_prim_path = "/World/pallet_arm/VacuumTool"

    handle.parent_prim_path = "/World/pallet_arm/wrist_3_link"
    assert handle._weld_anchor_prim_path() == "/World/pallet_arm/wrist_3_link"

    # with no mount link recorded there is nothing better to anchor on
    handle.parent_prim_path = None
    assert handle._weld_anchor_prim_path() == "/World/pallet_arm/VacuumTool"
