"""Set-equality check: the README's per-model attribute tables (world / arm /
gripper / camera) must list exactly the attribute keys the model's own code
reads - no code key missing from the README, no README key no code reads.

Code keys are found by static regex over ``attrs.get(...)`` / ``attrs[...]``
/ ``attrs.setdefault(...)`` (and ``self._attrs`` equivalents), scoped to the
named source files/functions that implement each model's spawn contract:
SimConfig's own construction (models/world.py), the arm/gripper/camera spawn
+ mock functions in sim_manager.py, and the shared frame/dependency
validation in models/utils.py. This is a static approximation, not an
interpreter, so a few known gaps are carried in ALLOWLIST below rather than
silenced.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SRC = REPO_ROOT / "src" / "isaac_module"

WORLD_MODEL_FILE = SRC / "models" / "world.py"
ARM_MODEL_FILE = SRC / "models" / "arm.py"
GRIPPER_MODEL_FILE = SRC / "models" / "gripper.py"
CAMERA_MODEL_FILE = SRC / "models" / "camera.py"
SIM_MANAGER_FILE = SRC / "sim_manager.py"
UTILS_FILE = SRC / "models" / "utils.py"
MOCK_CAMERA_FILE = SRC / "mock_camera.py"
CONDUCTOR_MODEL_FILE = SRC / "models" / "conductor.py"
SORTER_SENSOR_MODEL_FILE = SRC / "models" / "sorter_sensor.py"
META_JSON = REPO_ROOT / "meta.json"

# Matches attrs.get("key"), attrs["key"], attrs.setdefault("key", ...) and
# the same on self._attrs (models/arm.py, models/gripper.py store the config
# dict as self._attrs after reconfigure).
ATTR_READ_RE = re.compile(
    r"""
    (?:attrs|self\._attrs)
    (?:
        \.get\(\s*["'](?P<get_key>\w+)["']
      | \.setdefault\(\s*["'](?P<setdefault_key>\w+)["']
      | \[\s*["'](?P<index_key>\w+)["']\s*\]
    )
    """,
    re.VERBOSE,
)


def _extract_keys(source: str) -> set[str]:
    keys: set[str] = set()
    for m in ATTR_READ_RE.finditer(source):
        keys.add(m.group("get_key") or m.group("setdefault_key") or m.group("index_key"))
    return keys


def _named_defs(file_path: Path) -> dict[str, str]:
    """{name: source text} for every function/class def in the file, keyed
    by its own (non-qualified) name - fine here since the names we ask for
    are unambiguous within a file."""
    source = file_path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return out


def _keys_from_whole_file(file_path: Path) -> set[str]:
    return _extract_keys(file_path.read_text())


def _keys_from_defs(file_path: Path, names: list[str]) -> set[str]:
    defs = _named_defs(file_path)
    keys: set[str] = set()
    for name in names:
        assert name in defs, f"{name!r} not found in {file_path}"
        keys |= _extract_keys(defs[name])
    return keys


# ---------------------------------------------------------------------------
# code-side key sets, one per model with a README attribute table
# ---------------------------------------------------------------------------

WORLD_CODE_KEYS = _keys_from_whole_file(WORLD_MODEL_FILE)

ARM_CODE_KEYS = (
    _keys_from_whole_file(ARM_MODEL_FILE)
    | _keys_from_defs(
        SIM_MANAGER_FILE,
        ["_create_arm_isaac", "spawn_orientation", "_resolve_usd", "MockArmHandle"],
    )
    # arm.py calls validate_sim_component(config) with the default
    # needs_source=True, so every key that function reads (world, asset,
    # usd_path, prim_path, parent_prim) is really read for an arm.
    | _keys_from_defs(UTILS_FILE, ["validate_sim_component"])
)

GRIPPER_CODE_KEYS = _keys_from_whole_file(GRIPPER_MODEL_FILE) | _keys_from_defs(
    SIM_MANAGER_FILE, ["create_gripper", "_create_gripper_isaac", "MockGripperHandle"]
)

CAMERA_CODE_KEYS = (
    _keys_from_whole_file(CAMERA_MODEL_FILE)
    | _keys_from_defs(
        SIM_MANAGER_FILE,
        ["_camera_prim_path", "_place_camera", "_configure_camera_optics", "_create_camera_isaac"],
    )
    | _keys_from_whole_file(MOCK_CAMERA_FILE)
    # camera.py calls validate_sim_component(config, needs_source=False), so
    # only "world" and "parent_prim" (read unconditionally in that function)
    # are really read for a camera - "asset"/"usd_path"/"prim_path" also
    # appear in that function's source but are gated behind
    # `needs_source and (...)`, which short-circuits to False and never
    # executes for a camera. A static regex scan can't see that
    # short-circuit, so these two keys are added by hand instead of pulling
    # in the whole function.
    | {"world", "parent_prim"}
)

CONDUCTOR_CODE_KEYS = _keys_from_whole_file(CONDUCTOR_MODEL_FILE) | {
    # validate_config/reconfigure read these through `_DEPENDENCY_ATTRS`
    # (a loop over a tuple of key names) rather than a literal
    # `attrs["world"]`, so the literal-string regex can't see them.
    "world",
    "arm",
    "motion",
}

SORTER_SENSOR_CODE_KEYS = _keys_from_whole_file(SORTER_SENSOR_MODEL_FILE)

CODE_KEYS = {
    "world": WORLD_CODE_KEYS,
    "arm": ARM_CODE_KEYS,
    "gripper": GRIPPER_CODE_KEYS,
    "camera": CAMERA_CODE_KEYS,
    "conductor": CONDUCTOR_CODE_KEYS,
    "sorter-sensor": SORTER_SENSOR_CODE_KEYS,
}

# Keys the code reads that are deliberately absent from the README table:
# mock-only test knobs never exposed as a documented attribute, or a key
# read generically by shared validation but not meaningful for that model.
ALLOWLIST: dict[str, set[str]] = {
    "world": set(),
    "arm": {
        # sim_manager.py MockArmHandle test knobs (mock-only; ARM-13 stall
        # simulation and DOF-count override), never a documented attribute.
        "mock_stall_fraction",
        "mock_dof",
        # validate_sim_component() reads "parent_prim" generically for every
        # model that uses it (arm/camera/base share the function), but
        # _create_arm_isaac never consumes it - only a camera actually
        # mounts onto another prim this way (see README "Frames").
        "parent_prim",
    },
    "gripper": set(),
    "camera": {
        # apply_frame_to_attrs() (models/utils.py) derives this from the
        # standard frame config when parent_prim + frame are both set; it is
        # never a directly user-authored json key (the README documents the
        # frame config itself, not this internal representation).
        "local_orientation_wxyz",
    },
    "conductor": set(),
    "sorter-sensor": set(),
}

SECTION_HEADINGS = {
    "world": "### world attributes",
    "arm": "### arm attributes",
    "gripper": "### gripper attributes",
    "camera": "### camera attributes",
    "conductor": "### conductor attributes",
    "sorter-sensor": "### sorter-sensor attributes",
}


def _table_keys_for_section(readme_lines: list[str], heading: str) -> set[str]:
    start = next(i for i, line in enumerate(readme_lines) if line.strip() == heading)
    table_start = None
    for i in range(start, len(readme_lines)):
        if readme_lines[i].startswith("| attribute"):
            table_start = i
            break
    assert table_start is not None, f"no '| attribute |' table found under {heading!r}"

    keys: set[str] = set()
    # skip the header row and the "|---|---|---|" separator row
    for line in readme_lines[table_start + 2 :]:
        if not line.startswith("|"):
            break
        first_cell = line.split("|")[1]
        keys.update(re.findall(r"`([a-zA-Z0-9_]+)`", first_cell))
    return keys


@pytest.fixture(scope="module")
def readme_lines() -> list[str]:
    return README.read_text().splitlines()


@pytest.mark.parametrize(
    "model", ["world", "arm", "gripper", "camera", "conductor", "sorter-sensor"]
)
def test_readme_attribute_table_matches_code(readme_lines: list[str], model: str) -> None:
    documented = _table_keys_for_section(readme_lines, SECTION_HEADINGS[model])
    expected_code_keys = CODE_KEYS[model] - ALLOWLIST[model]

    missing_from_readme = expected_code_keys - documented
    stale_in_readme = documented - CODE_KEYS[model]

    assert not missing_from_readme, (
        f"{model}: code reads these attrs but the README table doesn't "
        f"document them: {sorted(missing_from_readme)}"
    )
    assert not stale_in_readme, (
        f"{model}: README documents these attrs but no scanned code reads "
        f"them: {sorted(stale_in_readme)}"
    )


def test_readme_models_table_matches_meta_json(readme_lines: list[str]) -> None:
    """The `## Models` table's model names must equal `meta.json`'s, in
    both directions: no model shipped without a README row, no row for a
    model the manifest no longer registers."""
    start = next(i for i, line in enumerate(readme_lines) if line.strip() == "## Models")
    table_start = next(
        i
        for i in range(start, len(readme_lines))
        if readme_lines[i].startswith("| Model") or readme_lines[i].startswith("|Model")
    )
    documented: set[str] = set()
    for line in readme_lines[table_start + 2 :]:
        if not line.startswith("|"):
            break
        first_cell = line.split("|")[1]
        documented.update(re.findall(r"`([a-zA-Z0-9_:.\-]+)`", first_cell))

    manifest_models = {entry["model"] for entry in json.loads(META_JSON.read_text())["models"]}
    assert documented == manifest_models
