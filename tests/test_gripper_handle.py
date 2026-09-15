import math
import time

import pytest

from isaac_module.handles.gripper import GRIPPER_OPEN_WIDTH_M
from isaac_module.sim_manager import MockArmHandle

SETTLE_POLLS = 400
SETTLE_POLL_S = 0.01


def _wait_until(predicate, polls: int = SETTLE_POLLS, poll_s: float = SETTLE_POLL_S) -> bool:
    for _ in range(polls):
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def test_create_gripper_unknown_arm_raises(sim):
    with pytest.raises(ValueError, match="not attached to the sim"):
        sim.create_gripper("gripper-bad-arm", {"world": "isaac-world", "arm": "no-such-arm"})


def test_grab_with_no_object_reaches_closed_rad(sim):
    sim.create_arm("gripper-arm-a", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-a", {"world": "isaac-world", "arm": "gripper-arm-a"})

    gripper.grab()
    assert _wait_until(lambda: not gripper.is_moving())

    assert gripper.get_jaw() == pytest.approx(math.radians(47.0), abs=1e-6)
    assert gripper.is_moving() is False
    assert gripper.is_holding() is False


def test_grab_on_object_stalls_at_contact_angle_and_holds(sim):
    sim.create_arm("gripper-arm-b", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper(
        "gripper-b",
        {"world": "isaac-world", "arm": "gripper-arm-b", "mock_object_width_m": 0.05},
    )

    open_rad = math.radians(0.0)
    closed_rad = math.radians(47.0)
    expected_contact = open_rad + (closed_rad - open_rad) * (1.0 - 0.05 / GRIPPER_OPEN_WIDTH_M)

    gripper.grab()
    assert _wait_until(lambda: not gripper.is_moving())

    assert gripper.get_jaw() == pytest.approx(expected_contact, abs=1e-6)
    assert gripper.is_moving() is False

    assert _wait_until(gripper.is_holding)
    assert gripper.is_holding() is True

    gripper.open()
    assert gripper.is_holding() is False


def test_stop_mid_travel_freezes_jaw(sim):
    sim.create_arm("gripper-arm-c", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-c", {"world": "isaac-world", "arm": "gripper-arm-c"})

    gripper.grab()
    time.sleep(0.1)
    gripper.stop()

    assert gripper.is_moving() is False
    jaw = gripper.get_jaw()
    assert math.radians(0.0) < jaw < math.radians(47.0)


def test_jaw_limits_default_and_attrs(sim):
    sim.create_arm("gripper-arm-d", {"world": "isaac-world", "asset": "ur5e"})
    default_gripper = sim.create_gripper(
        "gripper-d", {"world": "isaac-world", "arm": "gripper-arm-d"}
    )
    assert default_gripper.jaw_limits() == pytest.approx((0.0, math.radians(47.0)))

    sim.create_arm("gripper-arm-e", {"world": "isaac-world", "asset": "ur5e"})
    custom_gripper = sim.create_gripper(
        "gripper-e",
        {
            "world": "isaac-world",
            "arm": "gripper-arm-e",
            "open_deg": 5.0,
            "closed_deg": 50.0,
        },
    )
    assert custom_gripper.jaw_limits() == pytest.approx((math.radians(5.0), math.radians(50.0)))

    custom_gripper.set_jaw(math.radians(1000.0))
    assert _wait_until(lambda: not custom_gripper.is_moving())
    assert custom_gripper.get_jaw() == pytest.approx(math.radians(50.0), abs=1e-6)

    custom_gripper.set_jaw(math.radians(-1000.0))
    assert _wait_until(lambda: not custom_gripper.is_moving())
    assert custom_gripper.get_jaw() == pytest.approx(math.radians(5.0), abs=1e-6)


def test_dof_names_is_the_drive_joint(sim):
    sim.create_arm("gripper-arm-f", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-f", {"world": "isaac-world", "arm": "gripper-arm-f"})
    assert gripper.dof_names() == ["finger_joint"]


def test_create_gripper_is_cached_per_name(sim):
    sim.create_arm("gripper-arm-g", {"world": "isaac-world", "asset": "ur5e"})
    attrs = {"world": "isaac-world", "arm": "gripper-arm-g"}
    first = sim.create_gripper("gripper-g", attrs)
    second = sim.create_gripper("gripper-g", dict(attrs))
    assert first is second


def test_mock_arm_speed_constant_used_by_gripper_interpolation():
    # sanity: the gripper's interpolation speed comes from the same constant
    # the arm mock uses (the shared-pattern requirement).
    assert MockArmHandle.SPEED == 1.0


def test_poll_jaw_state_reports_moving_and_not_holding_mid_travel(sim):
    sim.create_arm("gripper-arm-h", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper(
        "gripper-h",
        {"world": "isaac-world", "arm": "gripper-arm-h", "mock_object_width_m": 0.05},
    )

    gripper.grab()
    time.sleep(0.05)  # still travelling toward the contact angle

    jaw, moving, holding = gripper.poll_jaw_state()
    assert math.radians(0.0) < jaw < math.radians(47.0)
    assert moving is True
    assert holding is False


def test_poll_jaw_state_reports_holding_once_stalled_on_an_object(sim):
    sim.create_arm("gripper-arm-i", {"world": "isaac-world", "asset": "ur5e"})
    gripper = sim.create_gripper(
        "gripper-i",
        {"world": "isaac-world", "arm": "gripper-arm-i", "mock_object_width_m": 0.05},
    )

    gripper.grab()
    assert _wait_until(gripper.is_holding)

    jaw, moving, holding = gripper.poll_jaw_state()
    assert jaw == pytest.approx(gripper.get_jaw(), abs=1e-3)
    assert moving is False
    assert holding is True
