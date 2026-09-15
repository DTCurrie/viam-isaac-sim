import json
import re
from pathlib import Path
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.vacuum import IsaacVacuum
from isaac_module.sim_manager import KNOWN_ASSETS
from test_readme_tables import documented_attributes

SIMULATES_PATH = Path(__file__).resolve().parent.parent / "simulates.json"
META_PATH = Path(__file__).resolve().parent.parent / "meta.json"
SIMULATION_DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "SIMULATION.md"

SIM_MODEL_CLASSES = (IsaacArm, IsaacGripper, IsaacVacuum, IsaacBase, IsaacCamera)

REAL_MODEL_PATTERN = re.compile(r"^[a-z0-9-]+:[a-z0-9_-]+:[a-z0-9_-]+$")

KNOWN_APIS = {
    "rdk:component:arm",
    "rdk:component:gripper",
    "rdk:component:base",
    "rdk:component:camera",
}
CATCH_ALL_SIM_MODEL = "rdk:builtin:fake"


def _is_catch_all(row: dict[str, Any]) -> bool:
    return row["real_model"] == "*" and row["api"] == "*"


def _sim_rows() -> list[dict[str, Any]]:
    return [row for row in _rows() if not _is_catch_all(row)]


def _sim_row_ids() -> list[str]:
    return [f"{row['real_model']}|{row['api']}" for row in _sim_rows()]


FRAME_PARENT_PLACEHOLDER = "pick-arm"


def _simulates() -> dict[str, Any]:
    return json.loads(SIMULATES_PATH.read_text())


def _rows() -> list[dict[str, Any]]:
    return _simulates()["rows"]


def _row_ids() -> list[str]:
    return [row["real_model"] for row in _rows()]


def test_schema_version_is_1() -> None:
    assert _simulates()["schema_version"] == 1


def test_module_id_matches_meta_json() -> None:
    meta = json.loads(META_PATH.read_text())
    assert _simulates()["module_id"] == meta["module_id"]


def test_world_attribute_is_world() -> None:
    assert _simulates()["world_attribute"] == "world"


def test_default_world_matches_the_module_default() -> None:
    assert _simulates()["default_world"] == DEFAULT_WORLD_NAME


def test_no_semicolon_separator_in_schema_or_note_strings() -> None:
    simulates = _simulates()
    strings = list(simulates["schema"].values())
    strings += [row["note"] for row in simulates["rows"] if "note" in row]
    for value in strings:
        assert "; " not in value, value


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_row_has_exactly_the_allowed_keys(row: dict[str, Any]) -> None:
    required = {"real_model", "api", "sim_model", "template", "carry", "verified", "source"}
    allowed = required | {"note"}
    assert required <= set(row.keys())
    assert set(row.keys()) <= allowed


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_row_field_types(row: dict[str, Any]) -> None:
    for field in ("real_model", "api", "sim_model", "source"):
        assert isinstance(row[field], str) and row[field], field
    assert isinstance(row["template"], dict)
    assert isinstance(row["carry"], dict)
    assert isinstance(row["verified"], bool)
    if "note" in row:
        assert isinstance(row["note"], str) and row["note"]


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_carry_keys_and_values_are_non_empty_strings_with_no_collisions(
    row: dict[str, Any],
) -> None:
    carry = row["carry"]
    for real_key, sim_key in carry.items():
        assert isinstance(real_key, str) and real_key, carry
        assert isinstance(sim_key, str) and sim_key, carry
    assert len(set(carry.values())) == len(carry.values()), (
        f"row {row['real_model']!r} carries two real-side keys onto the same sim attribute: {carry}"
    )


def _sim_model_short_name(row: dict[str, Any]) -> str:
    return row["sim_model"].rsplit(":", maxsplit=1)[-1]


@pytest.mark.parametrize("row", _sim_rows(), ids=_sim_row_ids())
def test_carry_sim_side_names_are_documented_attributes(row: dict[str, Any]) -> None:
    short_name = _sim_model_short_name(row)
    if short_name == "base":
        pytest.skip("the base model has no README attribute table today")
    documented = documented_attributes(short_name)
    for sim_key in row["carry"].values():
        assert sim_key in documented, (
            f"row {row['real_model']!r} carries into {sim_key!r}, which is not in "
            f"the README's {short_name} attribute table"
        )


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_carry_never_overwrites_a_templated_attribute(row: dict[str, Any]) -> None:
    overwritten = set(row["carry"].values()) & set(row["template"].keys())
    assert not overwritten, (
        f"row {row['real_model']!r} carries into {overwritten}, which the template already sets"
    )


def test_at_least_one_row_carries_something() -> None:
    assert any(row["carry"] for row in _rows())


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_real_model_is_wildcard_or_a_model_triple(row: dict[str, Any]) -> None:
    real_model = row["real_model"]
    assert real_model == "*" or REAL_MODEL_PATTERN.match(real_model), real_model


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_api_is_one_of_the_known_component_apis(row: dict[str, Any]) -> None:
    if _is_catch_all(row):
        return
    assert row["api"] in KNOWN_APIS


def test_exactly_one_catch_all_row_and_it_is_a_placeholder_generic() -> None:
    catch_alls = [row for row in _rows() if _is_catch_all(row)]
    assert len(catch_alls) == 1
    assert catch_alls[0]["sim_model"] == CATCH_ALL_SIM_MODEL
    assert catch_alls[0]["template"] == {} and catch_alls[0]["carry"] == {}
    assert not any(row["real_model"] == "*" and row["api"] != "*" for row in _rows())


def test_real_model_api_pairs_are_unique() -> None:
    pairs = [(row["real_model"], row["api"]) for row in _rows()]
    assert len(pairs) == len(set(pairs))


def _model_class_by_str_model() -> dict[str, type]:
    return {str(cls.MODEL): cls for cls in SIM_MODEL_CLASSES}


@pytest.mark.parametrize("row", _sim_rows(), ids=_sim_row_ids())
def test_sim_model_is_a_registered_model_serving_the_rows_api(row: dict[str, Any]) -> None:
    by_model = _model_class_by_str_model()
    assert row["sim_model"] in by_model, row["sim_model"]
    matched = by_model[row["sim_model"]]
    assert str(matched.API) == row["api"]


@pytest.mark.parametrize("row", _sim_rows(), ids=_sim_row_ids())
def test_sim_model_is_in_this_module_family(row: dict[str, Any]) -> None:
    assert row["sim_model"].startswith(f"{NAMESPACE}:{FAMILY}:")


@pytest.mark.parametrize("row", _rows(), ids=_row_ids())
def test_template_assets_are_known(row: dict[str, Any]) -> None:
    asset = row["template"].get("asset")
    if asset is not None:
        assert asset in KNOWN_ASSETS, asset


def test_every_known_asset_is_used_by_some_row() -> None:
    used = {row["template"].get("asset") for row in _rows()}
    assert set(KNOWN_ASSETS.keys()) <= used


def _verified_arm_rows() -> list[dict[str, Any]]:
    return [row for row in _rows() if row["api"] == "rdk:component:arm" and row["verified"]]


@pytest.mark.parametrize(
    "row", _verified_arm_rows(), ids=[row["real_model"] for row in _verified_arm_rows()]
)
def test_every_verified_arm_rows_asset_carries_kinematics_and_ee_prim(
    row: dict[str, Any],
) -> None:
    asset = row["template"]["asset"]
    meta = KNOWN_ASSETS[asset]
    assert meta.get("kinematics"), (
        f"row {row['real_model']!r} is verified but its asset {asset!r} carries no "
        "kinematics, so GetKinematics/MoveToPosition would fail on a resolved arm"
    )
    assert meta.get("ee_prim"), (
        f"row {row['real_model']!r} is verified but its asset {asset!r} carries no "
        "ee_prim, so GetEndPosition would fail on a resolved arm"
    )


def _resolve_template(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    resolved: dict[str, Any] = {}
    frame_parent_referenced = False
    for key, value in row["template"].items():
        if value == "$frame.parent":
            resolved[key] = FRAME_PARENT_PLACEHOLDER
            frame_parent_referenced = True
        elif isinstance(value, str) and value.startswith("$"):
            raise AssertionError(
                f"row {row['real_model']!r} template references {value!r}, which "
                "test_simulates.py does not know how to resolve; extend "
                "_resolve_template (see simulates.json's schema.rows[].template)"
            )
        else:
            resolved[key] = value
    return resolved, frame_parent_referenced


SUBSTITUTION_TABLE_HEADER = "| Real model | API | Sim model | Template | Carry | Verified |"
CATCH_ALL_TABLE_SUFFIX = " (catch-all)"
CARRY_ARROW = "→"


def _unquote(cell: str) -> str:
    return cell.removesuffix(CATCH_ALL_TABLE_SUFFIX).strip().strip("`")


def _carry_from_cell(cell: str) -> dict[str, str]:
    if _unquote(cell) == "{}":
        return {}
    carry: dict[str, str] = {}
    for pair in cell.split(","):
        real_key, sim_key = pair.split(CARRY_ARROW)
        carry[_unquote(real_key)] = _unquote(sim_key)
    return carry


def _documented_rows() -> list[dict[str, Any]]:
    lines = SIMULATION_DOC_PATH.read_text().splitlines()
    start = lines.index(SUBSTITUTION_TABLE_HEADER)
    rows: list[dict[str, Any]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        real_model, api, sim_model, template, carry, verified = cells
        rows.append(
            {
                "real_model": _unquote(real_model),
                "api": _unquote(api),
                "sim_model": _unquote(sim_model),
                "template": json.loads(_unquote(template)),
                "carry": _carry_from_cell(carry),
                "verified": {"yes": True, "no": False}[verified],
            }
        )
    return rows


def test_simulation_doc_table_mirrors_simulates_json_row_for_row() -> None:
    documented = _documented_rows()
    shipped = [
        {field: row[field] for field in ("real_model", "api", "sim_model", "template", "carry")}
        | {"verified": row["verified"]}
        for row in _rows()
    ]
    assert documented == shipped, (
        f"docs/SIMULATION.md's substitution table and {SIMULATES_PATH.name} disagree. "
        "The table mirrors the file row for row, in the file's order."
    )


@pytest.mark.parametrize("row", _sim_rows(), ids=_sim_row_ids())
def test_template_validates_against_its_sim_model_with_the_default_world(
    row: dict[str, Any],
) -> None:
    by_model = _model_class_by_str_model()
    matched = by_model[row["sim_model"]]

    resolved, frame_parent_referenced = _resolve_template(row)
    config = ComponentConfig(name="swapped", attributes=dict_to_struct(resolved))
    if frame_parent_referenced:
        config.frame.parent = FRAME_PARENT_PLACEHOLDER

    deps, _ = matched.validate_config(config)
    deps = list(deps)
    assert deps[0] == DEFAULT_WORLD_NAME
    if frame_parent_referenced:
        assert FRAME_PARENT_PLACEHOLDER in deps
