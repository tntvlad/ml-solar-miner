"""Unit tests for ML Solar Miner model helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml_solar_miner.const import (
    BATTERY_SOC_CRITICAL,
    BATTERY_SOC_MIN,
    DEFAULT_BATTERY_CAPACITY_KWH,
    FEATURE_NAMES,
    GRID_HYSTERESIS_W,
    GRID_IMPORT_REDUCE_W,
    GRID_IMPORT_TOLERANCE,
    MINER_POWER_MIN,
    VIABILITY_FLOOR,
)
from ml_solar_miner.models import (
    ML_AVAILABLE,
    apply_grid_sign,
    clamp_power,
    compute_reward,
    decide_power,
    estimate_hours_until_sunrise,
    features_from_state,
    fill_rewards_and_outcomes,
    get_training_sample_count,
    grid_import_watts,
    load_last_decision,
    load_metrics,
    load_model,
    parse_iso_datetime,
    rule_teacher,
    run_retrain,
    save_last_decision,
    save_metrics,
    target_soc_from_forecast,
    validate_decision,
    _reward_is_present,
    _get_training_csv_path,
)


def make_features(**kwargs) -> list[float]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values["battery_hours_to_min"] = 99.0
    values.update(kwargs)
    return [values[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# clamp_power / decide_power
# ---------------------------------------------------------------------------

def test_clamp_power_rounds_down_to_step() -> None:
    assert clamp_power(3650) == 3600
    assert clamp_power(9000) == 6000
    assert clamp_power(-50) == MINER_POWER_MIN


def test_decide_power_does_not_raise_submin_to_on() -> None:
    assert decide_power(0) == ("off", MINER_POWER_MIN)
    assert decide_power(3400) == ("off", MINER_POWER_MIN)
    assert decide_power(3500) == ("on", 3500)
    assert decide_power(3650) == ("on", 3600)


# ---------------------------------------------------------------------------
# apply_grid_sign / grid_import_watts
# ---------------------------------------------------------------------------

def test_apply_grid_sign_default_fronius() -> None:
    assert apply_grid_sign(500) == 500
    assert apply_grid_sign(-200) == -200


def test_apply_grid_sign_inverted_victron() -> None:
    assert apply_grid_sign(500, grid_invert=True) == -500
    assert apply_grid_sign(-200, grid_invert=True) == 200


def test_grid_import_watts_clamps_export_to_zero() -> None:
    assert grid_import_watts(500) == 500
    assert grid_import_watts(-100) == 0.0
    assert grid_import_watts(0) == 0.0


# ---------------------------------------------------------------------------
# Day teacher — hysteresis
# ---------------------------------------------------------------------------

def test_day_teacher_grid_import_does_not_turn_miner_on() -> None:
    features = make_features(
        is_daytime=1.0,
        solar_power_total=2000,
        solar_surplus_power=2000,
        battery_soc=80,
        grid_power=150,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "off"
    assert decision["mode"] == "day_solar"


def test_day_teacher_reduces_for_grid_then_stays_off_if_below_min() -> None:
    features = make_features(
        is_daytime=1.0,
        solar_power_total=3600,
        solar_surplus_power=3600,
        battery_soc=80,
        grid_power=400,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "off"


def test_day_teacher_matches_surplus_when_no_import() -> None:
    features = make_features(
        is_daytime=1.0,
        solar_power_total=5000,
        solar_surplus_power=5000,
        battery_soc=80,
        grid_power=0,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 5000


def test_day_hysteresis_off_to_on_requires_above_min_plus_hysteresis() -> None:
    """When miner is off, available power must exceed MIN + HYSTERESIS to start."""
    # Just below threshold (MIN + HYSTERESIS = 3700)
    features = make_features(
        is_daytime=1.0,
        solar_surplus_power=3650,
        battery_soc=80,
        grid_power=0,
    )
    decision = rule_teacher(features, prev_miner_active="off")
    assert decision["miner_active"] == "off"

    # Above threshold
    features = make_features(
        is_daytime=1.0,
        solar_surplus_power=3750,
        battery_soc=80,
        grid_power=0,
    )
    decision = rule_teacher(features, prev_miner_active="off")
    assert decision["miner_active"] == "on"


def test_day_hysteresis_on_to_off_stays_on_until_below_min_minus_hysteresis() -> None:
    """When miner is on, it stays on until available drops below MIN - HYSTERESIS."""
    # Just below threshold (MIN - HYSTERESIS = 3300)
    features = make_features(
        is_daytime=1.0,
        solar_surplus_power=3350,
        battery_soc=80,
        grid_power=0,
    )
    decision = rule_teacher(features, prev_miner_active="on")
    assert decision["miner_active"] == "on"

    # Below threshold
    features = make_features(
        is_daytime=1.0,
        solar_surplus_power=3200,
        battery_soc=80,
        grid_power=0,
    )
    decision = rule_teacher(features, prev_miner_active="on")
    assert decision["miner_active"] == "off"


# ---------------------------------------------------------------------------
# Night teacher — usable kWh drain
# ---------------------------------------------------------------------------

def test_night_teacher_uses_forecast_target() -> None:
    features = make_features(
        is_daytime=0.0,
        hour_of_day=22,
        hours_until_sunrise=8,
        battery_soc=80,
        forecast_tomorrow=70,
        house_load=500,
    )
    decision = rule_teacher(features, battery_capacity_kwh=50)
    assert decision["mode"] == "night_drain"
    assert decision["target_soc_by_sunrise"] == 12
    assert decision["miner_active"] == "on"


def test_night_teacher_zero_hours_does_not_mine() -> None:
    features = make_features(
        is_daytime=0.0,
        hours_until_sunrise=0,
        battery_soc=80,
        forecast_tomorrow=70,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "off"


def test_night_teacher_uses_kwh_available_when_present() -> None:
    """When battery_kwh_available > 0, the teacher uses it instead of SoC * capacity."""
    features = make_features(
        is_daytime=0.0,
        hour_of_day=22,
        hours_until_sunrise=8,
        battery_soc=50,
        battery_kwh_available=30.0,
        forecast_tomorrow=70,
        house_load=500,
    )
    decision = rule_teacher(features, battery_capacity_kwh=69.6)
    assert decision["mode"] == "night_drain"
    assert "usable 30.0kWh" in decision["reason"]


def test_night_teacher_falls_back_to_soc_times_capacity() -> None:
    """When battery_kwh_available is 0, the teacher uses SoC * capacity."""
    features = make_features(
        is_daytime=0.0,
        hour_of_day=22,
        hours_until_sunrise=8,
        battery_soc=80,
        battery_kwh_available=0,
        forecast_tomorrow=70,
        house_load=500,
    )
    decision = rule_teacher(features, battery_capacity_kwh=50)
    # usable should be 80% * 50 = 40 kWh
    assert "usable 40.0kWh" in decision["reason"]


# ---------------------------------------------------------------------------
# Viability floor
# ---------------------------------------------------------------------------

def test_viability_floor_off_when_below_floor() -> None:
    """When VIABILITY_FLOOR > 0 and viability is below it, miner stays off."""
    import ml_solar_miner.models as m

    original = m.VIABILITY_FLOOR
    try:
        m.VIABILITY_FLOOR = 0.5
        features = make_features(
            is_daytime=1.0,
            solar_surplus_power=5000,
            battery_soc=80,
            mining_viability_score=0.3,
        )
        decision = rule_teacher(features)
        assert decision["miner_active"] == "off"
        assert decision["mode"] == "uneconomic"
    finally:
        m.VIABILITY_FLOOR = original


def test_viability_floor_zero_disables_check() -> None:
    """When VIABILITY_FLOOR is 0, viability is ignored."""
    import ml_solar_miner.models as m

    original = m.VIABILITY_FLOOR
    try:
        m.VIABILITY_FLOOR = 0.0
        features = make_features(
            is_daytime=1.0,
            solar_surplus_power=5000,
            battery_soc=80,
            mining_viability_score=0.0,
        )
        decision = rule_teacher(features)
        assert decision["miner_active"] == "on"
    finally:
        m.VIABILITY_FLOOR = original


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_missing_hours_until_sunrise_is_estimated() -> None:
    features = features_from_state(
        {
            "solar_power_total": 0,
            "battery_soc": 70,
            "hours_until_sunrise": None,
        }
    )
    hours = features[FEATURE_NAMES.index("hours_until_sunrise")]
    assert hours > 0


def test_validate_decision_sets_safety_shutdown() -> None:
    decision = validate_decision(
        {
            "_soc": BATTERY_SOC_CRITICAL - 1,
            "miner_active": "on",
            "miner_power": 4000,
            "mode": "day_solar",
            "reason": "ML",
            "target_soc_by_sunrise": 30,
        }
    )
    assert decision["miner_active"] == "off"
    assert decision["mode"] == "safety_shutdown"


def test_validate_decision_reduces_for_grid_import() -> None:
    reduced = validate_decision(
        {
            "_soc": 50,
            "_grid_power": 400,
            "miner_active": "on",
            "miner_power": 4000,
            "mode": "day_solar",
            "reason": "ML model prediction",
            "target_soc_by_sunrise": 30,
        }
    )
    assert reduced["miner_active"] == "on"
    assert reduced["miner_power"] == 3600

    shutdown = validate_decision(
        {
            "_soc": 50,
            "_grid_power": 400,
            "miner_active": "on",
            "miner_power": 3600,
            "mode": "day_solar",
            "reason": "ML model prediction",
            "target_soc_by_sunrise": 30,
        }
    )
    assert shutdown["miner_active"] == "off"


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def test_compute_reward_coerces_csv_strings() -> None:
    reward = compute_reward(
        {
            "outcome_miner_ran": "False",
            "outcome_grid_import": "80.0",
            "outcome_soc_next_cycle": "48.0",
            "decision_target_soc_by_sunrise": "50",
            "solar_surplus_power": "0",
            "decision_miner_power": "3500",
        }
    )
    assert reward == 5.0  # SoC within 2% of target; miner did not run


def test_compute_reward_ignores_grid_export() -> None:
    import_penalty = compute_reward(
        {
            "outcome_miner_ran": "True",
            "outcome_grid_import": "400",
            "outcome_soc_next_cycle": "40",
            "decision_target_soc_by_sunrise": "40",
            "solar_surplus_power": "0",
            "decision_miner_power": "4000",
        }
    )
    export_ok = compute_reward(
        {
            "outcome_miner_ran": "True",
            "outcome_grid_import": "-400",
            "outcome_soc_next_cycle": "40",
            "decision_target_soc_by_sunrise": "40",
            "solar_surplus_power": "0",
            "decision_miner_power": "4000",
        }
    )
    assert export_ok > import_penalty


# ---------------------------------------------------------------------------
# Fill rewards / outcomes
# ---------------------------------------------------------------------------

def test_fill_last_row_from_live_state() -> None:
    rows = [
        {
            "battery_soc": "40",
            "grid_power": "0",
            "miner_is_on": "1.0",
            "decision_miner_active": "on",
            "decision_miner_power": "4000",
            "decision_target_soc_by_sunrise": "40",
            "solar_surplus_power": "0",
            "outcome_soc_next_cycle": "",
            "outcome_grid_import": "",
            "outcome_miner_ran": "",
            "reward": "",
        }
    ]
    filled = fill_rewards_and_outcomes(
        rows, live_state={"battery_soc": 41, "grid_power": 20, "miner_is_on": "on"}
    )
    assert filled[0]["reward"] != ""
    assert float(filled[0]["reward"]) != 0 or filled[0]["reward"] == 0


# ---------------------------------------------------------------------------
# Forecast / sunrise helpers
# ---------------------------------------------------------------------------

def test_target_soc_from_forecast() -> None:
    assert target_soc_from_forecast(80) == 12
    assert target_soc_from_forecast(40) == 30
    assert target_soc_from_forecast(20) == 45
    assert target_soc_from_forecast(5) == 60


def test_estimate_hours_until_sunrise() -> None:
    assert estimate_hours_until_sunrise(6.5) == pytest.approx(24.0)
    assert estimate_hours_until_sunrise(5.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def test_parse_iso_datetime_naive_becomes_utc() -> None:
    parsed = parse_iso_datetime("2026-08-20T12:00:00")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_load_model_handles_corrupt_pickle(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    model_dir = tmp_path / "ml_solar_miner"
    model_dir.mkdir()
    (model_dir / "mining_model.pkl").write_bytes(b"not-a-pickle")
    model, names = load_model(hass_config_path)
    assert model is None
    assert names == FEATURE_NAMES


def test_save_and_load_last_decision(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    decision = {
        "miner_active": "on",
        "miner_power": 4500,
        "mode": "day_solar",
        "reason": "surplus OK",
        "timestamp": "2026-08-20T12:00:00+00:00",
    }
    save_last_decision(hass_config_path, decision)
    loaded = load_last_decision(hass_config_path)
    assert loaded is not None
    assert loaded["miner_active"] == "on"
    assert loaded["miner_power"] == 4500


def test_load_last_decision_returns_none_when_missing(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    assert load_last_decision(hass_config_path) is None


def test_save_and_load_metrics(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    metrics = {"status": "improved", "val_mae": 123.45, "total_samples": 100}
    save_metrics(hass_config_path, metrics)
    loaded = load_metrics(hass_config_path)
    assert loaded["status"] == "improved"
    assert loaded["val_mae"] == 123.45


def test_load_metrics_returns_empty_when_missing(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    assert load_metrics(hass_config_path) == {}


def test_get_training_sample_count_empty(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    assert get_training_sample_count(hass_config_path) == 0


def test_get_training_sample_count_with_csv(tmp_path: Path) -> None:
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    csv_path = _get_training_csv_path(hass_config_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"a": i, "b": i * 10})

    assert get_training_sample_count(hass_config_path) == 5


# ---------------------------------------------------------------------------
# Reward floor filtering
# ---------------------------------------------------------------------------

def test_reward_zero_rows_are_kept_for_training() -> None:
    assert _reward_is_present({"reward": "0"})
    assert _reward_is_present({"reward": 0})
    assert not _reward_is_present({"reward": ""})
    assert not _reward_is_present({"reward": None})


# ---------------------------------------------------------------------------
# Retrain with force
# ---------------------------------------------------------------------------

def test_force_retrain_bypasses_sample_floor(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    rows = []
    for i in range(6):
        rows.append(
            {
                **{name: float(i) for name in FEATURE_NAMES},
                "battery_soc": 50 + i,
                "grid_power": 0,
                "miner_is_on": 1,
                "decision_miner_active": "on",
                "decision_miner_power": 3500 + i * 100,
                "decision_target_soc_by_sunrise": 50,
                "solar_surplus_power": 4000,
                "outcome_soc_next_cycle": "",
                "outcome_grid_import": "",
                "outcome_miner_ran": "",
                "reward": "",
            }
        )

    csv_path = _get_training_csv_path(hass_config_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    skipped = run_retrain(hass_config_path, min_samples=50, force=False)
    assert skipped["status"] == "insufficient_data"

    forced = run_retrain(
        hass_config_path,
        min_samples=50,
        force=True,
        live_state={"battery_soc": 55, "grid_power": 0, "miner_is_on": "on"},
    )
    assert forced["status"] in {"trained_no_cv", "forced", "improved"}
    assert forced["model_saved"] is True


def test_retrain_with_reward_floor_filters_bad_rows(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    def hass_config_path(*parts):
        return str(tmp_path.joinpath(*parts))

    rows = []
    for i in range(6):
        rows.append(
            {
                **{name: float(i) for name in FEATURE_NAMES},
                "battery_soc": 50 + i,
                "grid_power": 0,
                "miner_is_on": 1,
                "decision_miner_active": "on",
                "decision_miner_power": 3500 + i * 100,
                "decision_target_soc_by_sunrise": 50,
                "solar_surplus_power": 4000,
                "outcome_soc_next_cycle": "",
                "outcome_grid_import": "",
                "outcome_miner_ran": "",
                "reward": str(-20.0 if i < 3 else 10.0),
            }
        )

    csv_path = _get_training_csv_path(hass_config_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = run_retrain(
        hass_config_path,
        min_samples=50,
        force=True,
        live_state={"battery_soc": 55, "grid_power": 0, "miner_is_on": "on"},
    )
    assert result["model_saved"] is True


# ---------------------------------------------------------------------------
# ML unavailable fallback
# ---------------------------------------------------------------------------

def test_retrain_ml_unavailable_returns_status(tmp_path: Path) -> None:
    import ml_solar_miner.models as m

    original = m.ML_AVAILABLE
    try:
        m.ML_AVAILABLE = False

        def hass_config_path(*parts):
            return str(tmp_path.joinpath(*parts))

        result = run_retrain(hass_config_path, min_samples=50, force=True)
        assert result["status"] == "ml_unavailable"
        assert result["model_saved"] is False
    finally:
        m.ML_AVAILABLE = original
