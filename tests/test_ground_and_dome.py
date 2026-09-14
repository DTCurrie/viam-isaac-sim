from isaac_module.sim_manager import (
    SimConfig,
    SimManager,
    dome_light_settings,
    ground_is_matte,
    ground_plan,
    missing_texture_warning,
)


def _fresh_manager() -> SimManager:
    return SimManager()


# ----------------------------------------------------------------------
# dome_light_settings - pure helper
# ----------------------------------------------------------------------


def test_dome_light_settings_with_texture_resolves_path_and_defaults_rotation():
    resolved: list[str] = []

    def fake_resolve(value: str) -> str:
        resolved.append(value)
        return f"/resolved/{value}"

    settings = dome_light_settings({"texture": "module://hdri/foo.hdr"}, fake_resolve)

    assert resolved == ["module://hdri/foo.hdr"]
    assert settings["texture"] == "/resolved/module://hdri/foo.hdr"
    assert settings["texture_format"] == "latlong"
    assert settings["rotate_xyz"] is None


def test_dome_light_settings_rotation_without_texture():
    settings = dome_light_settings({"rotation_deg": 90}, lambda v: v)

    assert settings["texture"] is None
    assert settings["rotate_xyz"] == (0.0, 0.0, 90.0)


def test_dome_light_settings_with_neither_texture_nor_rotation():
    settings = dome_light_settings({}, lambda v: v)

    assert settings["texture"] is None
    assert settings["rotate_xyz"] is None
    assert settings["texture_format"] == "latlong"


def test_dome_light_settings_honors_explicit_texture_format():
    settings = dome_light_settings(
        {"texture": "module://hdri/foo.hdr", "texture_format": "angular"}, lambda v: v
    )

    assert settings["texture_format"] == "angular"


# ----------------------------------------------------------------------
# ground_plan - pure helper
# ----------------------------------------------------------------------


def test_ground_plan_defaults_to_grid_when_ground_is_none():
    kind, kwargs = ground_plan(None, None)
    assert (kind, kwargs) == ("grid", {})


def test_ground_plan_grid_kind():
    kind, kwargs = ground_plan({"kind": "grid"}, None)
    assert (kind, kwargs) == ("grid", {})


def test_ground_plan_none_kind():
    kind, kwargs = ground_plan({"kind": "none"}, None)
    assert (kind, kwargs) == ("none", {})


def test_ground_plan_plane_kind_fills_defaults():
    kind, kwargs = ground_plan({"kind": "plane"}, None)
    assert kind == "plane"
    assert kwargs == {
        "size": 100.0,
        "color": [0.5, 0.5, 0.5],
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }


def test_ground_plan_plane_kind_honors_overrides():
    kind, kwargs = ground_plan(
        {"kind": "plane", "size": 5.0, "color": [1, 0, 0], "friction": 0.2, "restitution": 0.9},
        None,
    )
    assert kind == "plane"
    assert kwargs == {
        "size": 5.0,
        "color": [1.0, 0.0, 0.0],
        "static_friction": 0.2,
        "dynamic_friction": 0.2,
        "restitution": 0.9,
    }


def test_ground_plan_skips_with_reason_when_usd_stage_and_ground_both_set():
    kind, kwargs = ground_plan({"kind": "plane"}, "/some/stage.usd")
    assert kind == "skip"
    assert "reason" in kwargs


def test_ground_plan_plane_kwargs_never_carry_matte():
    kind, kwargs = ground_plan({"kind": "plane", "matte": True}, None)
    assert kind == "plane"
    assert "matte" not in kwargs


# ----------------------------------------------------------------------
# ground_is_matte - pure helper
# ----------------------------------------------------------------------


def test_ground_is_matte_none_is_false():
    assert ground_is_matte(None) is False


def test_ground_is_matte_empty_dict_is_false():
    assert ground_is_matte({}) is False


def test_ground_is_matte_true_when_set():
    assert ground_is_matte({"kind": "plane", "matte": True}) is True


# ----------------------------------------------------------------------
# mock boot -> ground visible via status()
# ----------------------------------------------------------------------


def test_mock_boot_stores_ground_config_visible_via_status():
    mgr = _fresh_manager()
    mgr.cfg = SimConfig(mock=True, ground={"kind": "plane"})

    mgr._boot()

    assert mgr.ground == {"kind": "plane"}
    assert mgr.status()["ground"] == {"kind": "plane"}


def test_mock_boot_with_no_ground_config_reports_none():
    mgr = _fresh_manager()
    mgr.cfg = SimConfig(mock=True)

    mgr._boot()

    assert mgr.ground is None
    assert mgr.status()["ground"] is None


def test_mock_boot_stores_matte_ground_config_visible_via_status():
    mgr = _fresh_manager()
    mgr.cfg = SimConfig(mock=True, ground={"kind": "plane", "matte": True})

    mgr._boot()

    assert mgr.ground == {"kind": "plane", "matte": True}
    assert mgr.status()["ground"] == {"kind": "plane", "matte": True}


def test_missing_texture_warning_names_a_local_path_that_is_not_on_disk():
    warning = missing_texture_warning("/opt/x/assets/hdri/nope.hdr", exists=lambda _p: False)

    assert warning is not None
    assert "nope.hdr" in warning


def test_missing_texture_warning_is_silent_for_remote_urls_and_existing_files():
    assert missing_texture_warning("https://example.com/a.hdr", exists=lambda _p: False) is None
    assert missing_texture_warning("omniverse://srv/a.hdr", exists=lambda _p: False) is None
    assert missing_texture_warning("/opt/x/a.hdr", exists=lambda _p: True) is None
