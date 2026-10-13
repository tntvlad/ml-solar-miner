"""Rule-based mining decision engine.

This module is a faithful port of the standalone engine that used to live in
``/config/scripts/mining_ml_utils.py`` (with its ``mining_ml_decision.py``
entry point). Users who select the **Rule-based** engine get exactly the same
on/off + power decisions the shell_command automation produced.

Keep this file in lock-step with the legacy script:

* the feature vector order (``FEATURE_NAMES``) must not change;
* day mode follows ``solar_power_total > 100`` and the solar surplus;
* night mode drains from the current SoC to the *user-configured* morning
  target using ``(soc - target) / 100 * capacity_kwh``;
* the clamping behaviour of ``clamp_power`` / ``clamp_target_soc`` matches.

The only adaptation for the integration is that missing/``None`` state values
are coerced to safe defaults instead of raising ``TypeError``.
"""
from __future__ import annotations

import math
from datetime import datetime

from .const import (
    BATTERY_SOC_CRITICAL,
    BATTERY_SOC_MIN,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_TARGET_SOC_MORNING,
    GRID_IMPORT_TOLERANCE,
    MINER_POWER_MAX,
    MINER_POWER_MIN,
    MINER_POWER_STEP,
    TARGET_SOC_MAX,
)

# Same ordering as the feature vector returned by features_from_state().
FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "hours_until_sunrise",
    "solar_power_total",
    "solar_surplus_power",
    "battery_soc",
    "battery_current",
    "battery_drain_rate",
    "battery_hours_to_min",
    "house_load",
    "forecast_tomorrow",
    "grid_power",
    "current_miner_power",
    "miner_is_on",
]


def _num(value, default: float = 0.0) -> float:
    """Coerce an HA state value to float, tolerating None/''/bad values."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_power(raw_power) -> int:
    """Clamp miner power to the valid range, rounded to the 100W step."""
    clamped = max(MINER_POWER_MIN, min(MINER_POWER_MAX, int(raw_power)))
    return int((clamped // MINER_POWER_STEP) * MINER_POWER_STEP)


def clamp_target_soc(raw_target: float) -> float:
    """Clamp the morning target SoC to the safe range."""
    return max(float(BATTERY_SOC_MIN), min(float(TARGET_SOC_MAX), float(raw_target)))


def features_from_state(state: dict) -> list[float]:
    """Extract the ordered feature vector from the HA state dictionary."""
    now = datetime.now()
    hour = now.hour + now.minute / 60.0
    solar = _num(state.get("solar_power_total"), 0)
    soc = _num(state.get("battery_soc"), 0)
    house_load_raw = _num(state.get("total_load_power"), 0)
    miner_actual = _num(state.get("miner_consumption"), 0)
    house_load = max(0, abs(house_load_raw) - miner_actual)

    return [
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        _num(state.get("hours_until_sunrise"), 0),
        solar,
        _num(state.get("solar_surplus_power"), 0),
        soc,
        _num(state.get("battery_current"), 0),
        _num(state.get("battery_drain_rate"), 0),
        _num(state.get("battery_hours_to_min"), 99),
        house_load,
        _num(state.get("forecast_tomorrow"), 0),
        _num(state.get("grid_power"), 0),
        _num(state.get("current_miner_power"), 0),
        1.0 if state.get("miner_is_on", "off") == "on" else 0.0,
    ]


def miner_needed_watts(
    energy_to_drain_kwh: float, hours_to_sunrise: float, house_load: float
) -> float:
    """Required miner wattage to drain a given amount of energy before sunrise."""
    if hours_to_sunrise > 0:
        required_watts = (energy_to_drain_kwh * 1000) / hours_to_sunrise
    else:
        required_watts = 0
    return required_watts - house_load


def night_drain_power(features: list[float], target_soc: float, capacity_kwh: float) -> tuple:
    """Night-mode power logic: drain from current SoC to target by sunrise.

    Returns (miner_active, miner_power).
    """
    f = dict(zip(FEATURE_NAMES, features))
    soc = float(f["battery_soc"])
    target_soc = clamp_target_soc(target_soc)
    house_load = float(f["house_load"])
    hours_to_sunrise = float(f["hours_until_sunrise"])

    energy_to_drain = ((soc - target_soc) / 100.0) * float(capacity_kwh)

    if hours_to_sunrise > 0:
        required_watts = (energy_to_drain * 1000) / hours_to_sunrise
    else:
        required_watts = 0
    miner_needed = required_watts - house_load

    if miner_needed < MINER_POWER_MIN:
        return "off", MINER_POWER_MIN
    elif miner_needed > MINER_POWER_MAX:
        return "on", MINER_POWER_MAX
    else:
        return "on", clamp_power(miner_needed)


def validate_decision(decision: dict) -> dict:
    """Enforce safety constraints on a decision dict."""
    soc = float(decision.get("_soc", 50))
    miner_active = decision.get("miner_active", "off")
    miner_power = int(decision.get("miner_power", MINER_POWER_MIN))

    if soc < BATTERY_SOC_CRITICAL:
        miner_active = "off"
        miner_power = MINER_POWER_MIN
    elif soc < BATTERY_SOC_MIN and miner_active == "on":
        miner_active = "off"
        miner_power = MINER_POWER_MIN

    miner_power = clamp_power(miner_power)
    target_soc = max(
        BATTERY_SOC_CRITICAL,
        min(TARGET_SOC_MAX, int(round(decision.get("target_soc_by_sunrise", 30)))),
    )

    return {
        "miner_active": miner_active,
        "miner_power": miner_power,
        "target_soc_by_sunrise": target_soc,
        "mode": decision.get("mode", "unknown"),
        "reason": decision.get("reason", ""),
    }


def rule_teacher(
    features: list[float],
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    target_soc_morning: float = DEFAULT_TARGET_SOC_MORNING,
) -> dict:
    """Rule-based decision.

    - Day (solar > 100W): miner power follows solar surplus (physics).
    - Night: drain the battery to the user-configured morning target SoC.
    - Battery capacity comes from the dashboard setting (energy math).

    Returns a decision dict with miner_active / miner_power /
    target_soc_by_sunrise / mode / reason.
    """
    f = dict(zip(FEATURE_NAMES, features))
    soc = f["battery_soc"]
    surplus = f["solar_surplus_power"]
    grid_power = f["grid_power"]
    is_daytime = f["solar_power_total"] > 100
    target_soc = clamp_target_soc(target_soc_morning)

    if soc < BATTERY_SOC_CRITICAL:
        return {
            "miner_active": "off",
            "miner_power": MINER_POWER_MIN,
            "target_soc_by_sunrise": target_soc,
            "mode": "safety_shutdown",
            "reason": f"Safety: SoC {soc:.0f}% below critical {BATTERY_SOC_CRITICAL}%",
        }

    if is_daytime:
        available_for_miner = surplus
        if available_for_miner < MINER_POWER_MIN:
            decision_power = 0
        elif available_for_miner > MINER_POWER_MAX:
            decision_power = MINER_POWER_MAX
        else:
            decision_power = clamp_power(available_for_miner)

        if grid_power > GRID_IMPORT_TOLERANCE:
            decision_power = clamp_power(decision_power - grid_power)

        miner_active = "on" if decision_power >= MINER_POWER_MIN else "off"
        return {
            "miner_active": miner_active,
            "miner_power": (
                max(MINER_POWER_MIN, decision_power)
                if miner_active == "on"
                else MINER_POWER_MIN
            ),
            "target_soc_by_sunrise": target_soc,
            "mode": "day_solar",
            "reason": f"DAY: Solar available {available_for_miner:.0f}W, miner set to {decision_power}W",
        }

    # Night mode: drain to the user-configured morning target.
    capacity = float(capacity_kwh)
    energy_to_drain = ((soc - target_soc) / 100.0) * capacity
    miner_active, decision_power = night_drain_power(features, target_soc, capacity)
    needed = miner_needed_watts(
        energy_to_drain, f["hours_until_sunrise"], f["house_load"]
    )
    return {
        "miner_active": miner_active,
        "miner_power": decision_power,
        "target_soc_by_sunrise": target_soc,
        "mode": "night_drain",
        "reason": (
            f"NIGHT: Need {needed:.0f}W to drain {energy_to_drain:.1f}kWh by sunrise "
            f"(target SoC {target_soc:.0f}%, capacity {capacity:.1f}kWh)"
        ),
    }


def decide(
    state: dict,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    target_soc_morning: float = DEFAULT_TARGET_SOC_MORNING,
    grid_invert: bool = False,
) -> dict:
    """Build features from an HA state dict and return a validated decision."""
    work = dict(state)
    if grid_invert:
        work["grid_power"] = -_num(state.get("grid_power"), 0.0)

    features = features_from_state(work)
    decision = rule_teacher(
        features,
        capacity_kwh=capacity_kwh,
        target_soc_morning=target_soc_morning,
    )
    decision["_soc"] = _num(state.get("battery_soc"), 50.0)
    return validate_decision(decision)
