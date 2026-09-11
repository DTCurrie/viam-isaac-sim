import json
import math
from pathlib import Path

import numpy as np
import pytest

from isaac_module.kinematics import (
    IK_ORIENTATION_TOL_RAD,
    IK_POSITION_TOL_M,
    Chain,
    JointLimitError,
    UnreachablePoseError,
)
from isaac_module.spatial import ov_to_quat

FIXTURES = Path(__file__).parent / "fixtures" / "kinematics"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _ur5e() -> Chain:
    return Chain.from_sva(_load("ur5e.json"))


def _ur20() -> Chain:
    return Chain.from_sva(_load("ur20.json"))


def _xarm7() -> Chain:
    return Chain.from_sva(_load("xarm7.json"))


def test_ur5e_fk_at_zero_matches_dh_parameters():
    # UR5e Denavit-Hartenberg parameters (meters): d1=0.1625, a2=-0.425,
    # a3=-0.3922, d4=0.1333, d5=0.0997, d6=0.0996. At q=0 the standard DH
    # chain places the tool at x = a2+a3, y = -(d4+d6), z = d1-d5.
    d1, a2, a3, d4, d5, d6 = 0.1625, -0.425, -0.3922, 0.1333, 0.0997, 0.0996
    expected_x = abs(a2 + a3)
    expected_y = abs(d4 + d6)
    expected_z = abs(d1 - d5)

    chain = _ur5e()
    pos, _ = chain.fk([0.0] * chain.dof)

    # Observed from this SVA file's own frame conventions: x and y come out
    # negative, z positive, at q=0.
    assert pos[0] == pytest.approx(-expected_x, abs=1e-3)
    assert pos[1] == pytest.approx(-expected_y, abs=1e-3)
    assert pos[2] == pytest.approx(expected_z, abs=1e-3)
    assert abs(pos[0]) == pytest.approx(0.8172, abs=1e-3)
    assert abs(pos[1]) == pytest.approx(0.2329, abs=1e-3)
    assert abs(pos[2]) == pytest.approx(0.0628, abs=1e-3)


_JACOBIAN_CONDITION_LIMIT = 30
_SINGULARITY_RETRY_CAP = 500


def _jacobian_condition_number(chain, q):
    singular_values = np.linalg.svd(
        chain._numeric_jacobian(np.asarray(q, dtype=float)), compute_uv=False
    )
    return singular_values[0] / singular_values[-1]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("chain_factory", [_ur5e, _ur20])
def test_ik_round_trip_random_poses(chain_factory, seed):
    rng = np.random.default_rng(seed)
    chain = chain_factory()
    elbow_index = [j.id for j in chain.joints].index("elbow_joint")
    for _ in range(20):
        for _retry in range(_SINGULARITY_RETRY_CAP):
            q = rng.uniform(math.radians(-90), math.radians(90), size=chain.dof)
            # Exclude [-10, 10] degrees on the elbow: that band straddles the
            # elbow-straight singularity, where the Jacobian is near-singular
            # and a tiny pose-space error blows up into joint-space error.
            band_low, band_high = (-90, -10) if rng.integers(0, 2) == 0 else (10, 90)
            q[elbow_index] = math.radians(rng.uniform(band_low, band_high))
            noise = rng.uniform(math.radians(-5), math.radians(5), size=chain.dof)
            q0 = q + noise
            # A 6-DOF arm also has a shoulder singularity, unrelated to the
            # elbow, whenever the wrist center nears the shoulder's rotation
            # axis. Skip draws landing near either the target or the start
            # near that or any other near-singular configuration.
            if _jacobian_condition_number(chain, q) <= _JACOBIAN_CONDITION_LIMIT and (
                _jacobian_condition_number(chain, q0) <= _JACOBIAN_CONDITION_LIMIT
            ):
                break
        else:
            pytest.fail("could not draw a well-conditioned sample within the retry cap")

        pos, quat = chain.fk(q)
        solution = chain.ik(pos, quat, q0_rad=q0)
        assert np.allclose(solution, q, atol=1e-3)


def test_ik_pose_match_for_7dof_xarm7():
    rng = np.random.default_rng(0)
    chain = _xarm7()
    # Clip the sampled range to each joint's own declared limits so the
    # generated target poses are ones the arm's real joints can reach.
    lower = np.array([max(math.radians(-60), j.min_rad) for j in chain.joints])
    upper = np.array([min(math.radians(60), j.max_rad) for j in chain.joints])
    for _ in range(10):
        q = rng.uniform(lower, upper)
        noise = rng.uniform(math.radians(-5), math.radians(5), size=chain.dof)
        target_pos, target_quat = chain.fk(q)
        solution = chain.ik(target_pos, target_quat, q0_rad=q + noise)
        result_pos, result_quat = chain.fk(solution)

        assert np.linalg.norm(np.array(result_pos) - np.array(target_pos)) < IK_POSITION_TOL_M * 10
        dot = sum(a * b for a, b in zip(result_quat, target_quat, strict=True))
        angle_between = 2 * math.acos(min(1.0, abs(dot)))
        assert angle_between < IK_ORIENTATION_TOL_RAD * 10


def test_ik_returns_nearest_solution_not_elbow_flip():
    chain = _ur5e()
    rng = np.random.default_rng(2)
    q = rng.uniform(math.radians(-90), math.radians(90), size=chain.dof)
    pos, quat = chain.fk(q)

    solution = chain.ik(pos, quat, q0_rad=q)

    elbow_index = [j.id for j in chain.joints].index("elbow_joint")
    assert np.sign(solution[elbow_index]) == np.sign(q[elbow_index])
    assert np.allclose(solution, q, atol=1e-3)


def test_ik_unreachable_pose_raises():
    chain = _ur5e()
    q0 = [0.0] * chain.dof
    with pytest.raises(UnreachablePoseError):
        chain.ik((3.0, 3.0, 3.0), (1.0, 0.0, 0.0, 0.0), q0_rad=q0)


def test_ik_joint_limit_raises_naming_the_joint():
    raw = json.loads(_load("ur5e.json"))
    for joint in raw["joints"]:
        if joint["id"] == "elbow_joint":
            joint["min"] = -10
            joint["max"] = 10

    chain = Chain.from_sva(json.dumps(raw).encode())
    elbow_index = [j.id for j in chain.joints].index("elbow_joint")

    q = [0.0] * chain.dof
    q[elbow_index] = math.radians(90)
    target_pos, target_quat = chain.fk(q)

    with pytest.raises(JointLimitError, match="elbow_joint"):
        chain.ik(target_pos, target_quat, q0_rad=q)


def test_dof_matches_file():
    assert _ur5e().dof == 6
    assert _ur20().dof == 6
    assert _xarm7().dof == 7


def test_branching_sva_raises_value_error():
    raw = json.loads(_load("ur5e.json"))
    # Add a second link claiming the same parent joint as an existing link,
    # creating a branch the chain-walk cannot resolve.
    branch_link = dict(raw["links"][2])
    branch_link["id"] = branch_link["id"] + "_branch"
    raw["links"].append(branch_link)

    with pytest.raises(ValueError):
        Chain.from_sva(json.dumps(raw).encode())


def test_ik_from_the_stretched_zero_pose_stays_inside_joint_limits():
    """The GPU check on 2026-09-08: from zero joints, a UR5e asked for a pose 0.55 m away
    converged on shoulder_lift at 16.9 rad and was rejected against the two-pi limits."""
    chain = _ur5e()
    target_quat = ov_to_quat(0.0, 0.0, -1.0, 0.0)
    q = chain.ik((0.35, 0.10, 0.40), target_quat, [0.0] * 6)
    assert all(abs(angle) <= 2 * math.pi for angle in q)
    pos, quat = chain.fk(q)
    assert max(abs(a - b) for a, b in zip(pos, (0.35, 0.10, 0.40), strict=True)) < 1e-3
    assert abs(sum(a * b for a, b in zip(quat, target_quat, strict=True))) > 0.9999


def test_ik_picks_the_in_limit_equivalent_nearest_the_start():
    chain = _ur5e()
    start = [5.8, -1.2, 1.0, -1.0, 1.5, 0.3]
    goal = list(start)
    goal[0] = 6.8  # one radian further, past the two-pi limit
    pos, quat = chain.fk(goal)
    q = chain.ik(pos, quat, start)
    assert -2 * math.pi <= q[0] <= 2 * math.pi
    assert (
        abs((q[0] - goal[0]) % (2 * math.pi)) < 1e-3 or abs((goal[0] - q[0]) % (2 * math.pi)) < 1e-3
    )
