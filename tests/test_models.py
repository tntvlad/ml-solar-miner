"""Unit tests for ML Solar Miner model helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml_solar_miner.const import (
    BATTERY_SOC_CRITICAL,
    FEATURE_NAMES,
    MINER_POWER_MIN,
)
from ml_solar_miner.models import (  # noqa: E402
    clamp_power,
    compute_reward,
    decide_power,
    estimate_hours_until_sunrise,
    features_from_state,
    fill_rewards_and_outcomes,
    load_model,
    parse_iso_datetime,
    rule_teacher,
    run_retrain,
    target_soc_from_forecast,
    validate_decision,
)


def make_features(**kwargs) -> list[float]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values["battery_hours_to_min"] = 99.0
    values.update(kwargs)
    return [values[name] for name in FEATURE_NAMES]


def test_clamp_power_rounds_down_to_step() -> None:
    assert clamp_power(3650) == 3600
    assert clamp_power(9000) == 6000
    assert clamp_power(-50) == MINER_POWER_MIN


def test_decide_power_does_not_raise_submin_to_on() -> None:
    assert decide_power(0) == ("off", MINER_POWER_MIN)
    assert decide_power(3400) == ("off", MINER_POWER_MIN)
    assert decide_power(3500) == ("on", 3500)
    assert decide_power(3650) == ("on", 3600)


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


def test_target_soc_from_forecast() -> None:
    assert target_soc_from_forecast(80) == 12
    assert target_soc_from_forecast(40) == 30
    assert target_soc_from_forecast(20) == 45
    assert target_soc_from_forecast(5) == 60


def test_estimate_hours_until_sunrise() -> None:
    assert estimate_hours_until_sunrise(6.5) == pytest.approx(24.0)
    assert estimate_hours_until_sunrise(5.5) == pytest.approx(1.0)


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

    from ml_solar_miner.models import _get_training_csv_path
    import csv

    csv_path = _get_training_csv_path(hass_config_path)
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


def test_reward_zero_rows_are_kept_for_training() -> None:
    from ml_solar_miner.models import _reward_is_present

    assert _reward_is_present({"reward": "0"})
    assert _reward_is_present({"reward": 0})
    assert not _reward_is_present({"reward": ""})
    assert not _reward_is_present({"reward": None})
