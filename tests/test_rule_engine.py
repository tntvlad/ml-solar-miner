"""Unit tests for the rule engine (port of the standalone script)."""
from __future__ import annotations

from ml_solar_miner.rule_engine import (
    FEATURE_NAMES,
    clamp_power,
    clamp_target_soc,
    decide,
    miner_needed_watts,
    rule_teacher,
)


def make_features(**kwargs) -> list[float]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values["battery_hours_to_min"] = 99.0
    values.update(kwargs)
    return [values[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# clamp helpers
# ---------------------------------------------------------------------------

def test_clamp_power_rounds_down_to_step() -> None:
    assert clamp_power(3650) == 3600
    assert clamp_power(4060) == 4000
    assert clamp_power(9000) == 6000
    assert clamp_power(-50) == 3500


def test_clamp_target_soc_range() -> None:
    assert clamp_target_soc(5) == 12
    assert clamp_target_soc(90) == 65
    assert clamp_target_soc(30) == 30


def test_miner_needed_watts() -> None:
    assert miner_needed_watts(6.96, 8, 400) == 470.0
    assert miner_needed_watts(1.0, 0, 400) == -400.0


# ---------------------------------------------------------------------------
# night mode
# ---------------------------------------------------------------------------

def test_night_below_minimum_stays_off() -> None:
    features = make_features(
        battery_soc=40,
        hours_until_sunrise=8,
        house_load=400,
        solar_power_total=0,
    )
    decision = rule_teacher(features, capacity_kwh=69.6, target_soc_morning=30)
    assert decision["miner_active"] == "off"
    assert decision["miner_power"] == 3500
    assert decision["target_soc_by_sunrise"] == 30
    assert decision["mode"] == "night_drain"
    assert "NIGHT: Need 470W" in decision["reason"]


def test_night_mid_range_power() -> None:
    features = make_features(
        battery_soc=65,
        hours_until_sunrise=6,
        house_load=0,
    )
    decision = rule_teacher(features, capacity_kwh=69.6, target_soc_morning=30)
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 4000


def test_night_clamps_to_max_power() -> None:
    features = make_features(battery_soc=90, hours_until_sunrise=4)
    decision = rule_teacher(features, capacity_kwh=69.6, target_soc_morning=30)
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 6000


def test_night_target_is_clamped() -> None:
    features = make_features(battery_soc=90, hours_until_sunrise=4)
    high = rule_teacher(features, capacity_kwh=69.6, target_soc_morning=90)
    low = rule_teacher(features, capacity_kwh=69.6, target_soc_morning=5)
    assert high["target_soc_by_sunrise"] == 65
    assert low["target_soc_by_sunrise"] == 12


# ---------------------------------------------------------------------------
# day mode
# ---------------------------------------------------------------------------

def test_day_follows_surplus() -> None:
    features = make_features(
        solar_power_total=4200,
        solar_surplus_power=3800,
        grid_power=0,
        battery_soc=50,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 3800
    assert decision["mode"] == "day_solar"


def test_day_reduces_for_grid_import() -> None:
    features = make_features(
        solar_power_total=5200,
        solar_surplus_power=5000,
        grid_power=800,
        battery_soc=50,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 4200


def test_day_below_minimum_stays_off() -> None:
    features = make_features(
        solar_power_total=2100,
        solar_surplus_power=2000,
        battery_soc=50,
    )
    decision = rule_teacher(features)
    assert decision["miner_active"] == "off"
    assert decision["miner_power"] == 3500


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------

def test_safety_shutdown_below_critical() -> None:
    features = make_features(battery_soc=9, solar_power_total=5000)
    decision = rule_teacher(features)
    assert decision["miner_active"] == "off"
    assert decision["mode"] == "safety_shutdown"


# ---------------------------------------------------------------------------
# decide() wrapper (integration entry point)
# ---------------------------------------------------------------------------

def _state(**kwargs) -> dict:
    base = {
        "solar_power_total": 0,
        "solar_surplus_power": 0,
        "battery_soc": 0,
        "battery_current": 0,
        "battery_drain_rate": 0,
        "battery_hours_to_min": 99,
        "hours_until_sunrise": 0,
        "total_load_power": 0,
        "miner_consumption": 0,
        "forecast_tomorrow": 0,
        "grid_power": 0,
        "current_miner_power": 0,
        "miner_is_on": "off",
    }
    base.update(kwargs)
    return base


def test_decide_returns_stable_keys() -> None:
    decision = decide(
        _state(battery_soc=40, hours_until_sunrise=8, total_load_power=400),
        capacity_kwh=69.6,
        target_soc_morning=30,
    )
    assert set(decision) == {
        "miner_active",
        "miner_power",
        "target_soc_by_sunrise",
        "mode",
        "reason",
    }
    assert decision["miner_active"] == "off"


def test_decide_grid_invert() -> None:
    # Export-positive grid: -800W raw becomes +800W import after inversion.
    decision = decide(
        _state(
            solar_power_total=5200,
            solar_surplus_power=5000,
            grid_power=-800,
            battery_soc=50,
        ),
        grid_invert=True,
    )
    assert decision["miner_active"] == "on"
    assert decision["miner_power"] == 4200
