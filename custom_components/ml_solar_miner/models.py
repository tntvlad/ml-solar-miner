"""ML model management for the Solar Miner integration.

Consolidates logic from mining_ml_utils.py, mining_ml_decision.py,
and mining_ml_retrain.py into HA-independent modules.
"""
import csv
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

from .const import (
    BATTERY_SOC_CRITICAL,
    BATTERY_SOC_MIN,
    CROSS_VAL_FOLDS,
    FEATURE_NAMES,
    GRID_IMPORT_TOLERANCE,
    METRICS_FILENAME,
    MIN_SAMPLES_FOR_TEACHER_ONLY,
    MINER_POWER_MAX,
    MINER_POWER_MIN,
    MINER_POWER_STEP,
    ML_DATA_DIR,
    MODEL_FILENAME,
    TRAINING_CSV_FILENAME,
)

_LOGGER = logging.getLogger(__name__)


def _get_data_dir(hass_config_path) -> Path:
    """Get the data directory path, creating if needed."""
    data_dir = Path(hass_config_path(ML_DATA_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_model_path(hass_config_path) -> Path:
    return _get_data_dir(hass_config_path) / MODEL_FILENAME


def _get_training_csv_path(hass_config_path) -> Path:
    return _get_data_dir(hass_config_path) / TRAINING_CSV_FILENAME


def _get_metrics_path(hass_config_path) -> Path:
    return _get_data_dir(hass_config_path) / METRICS_FILENAME


def clamp_power(raw_power: int) -> int:
    """Clamp miner power to valid range and round to nearest 100W step."""
    clamped = max(MINER_POWER_MIN, min(MINER_POWER_MAX, int(raw_power)))
    return int((clamped // MINER_POWER_STEP) * MINER_POWER_STEP)


def features_from_state(state: dict) -> list[float]:
    """Extract ordered feature vector from HA state dictionary."""
    from datetime import datetime as dt

    hour = dt.now().hour
    solar = float(state.get("solar_power_total", 0))
    soc = float(state.get("battery_soc", 0))
    house_load_raw = float(state.get("total_load_power", 0))
    miner_actual = float(state.get("miner_consumption", 0))
    house_load = max(0, abs(house_load_raw) - miner_actual)

    return [
        float(hour),
        1.0 if solar > 100 else 0.0,
        float(state.get("hours_until_sunrise", 0)),
        solar,
        float(state.get("solar_surplus_power", 0)),
        soc,
        float(state.get("battery_voltage", 0)),
        float(state.get("battery_current", 0)),
        float(state.get("battery_power", 0)),
        float(state.get("battery_kwh_available", 0)),
        float(state.get("battery_drain_rate", 0)),
        float(state.get("battery_hours_to_min", 99)),
        house_load,
        float(state.get("forecast_tomorrow", 0)),
        float(state.get("forecast_day3", 0)),
        float(state.get("grid_power", 0)),
        float(state.get("mining_viability_score", 0)),
        float(state.get("current_miner_power", 0)),
        1.0 if state.get("miner_is_on", "off") == "on" else 0.0,
    ]


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
    target_soc = max(10, min(65, int(decision.get("target_soc_by_sunrise", 30))))

    return {
        "miner_active": miner_active,
        "miner_power": miner_power,
        "target_soc_by_sunrise": target_soc,
        "mode": decision.get("mode", "unknown"),
        "reason": decision.get("reason", ""),
    }


def compute_reward(row: dict) -> float:
    """Compute reward score for a historical decision+outcome row."""
    reward = 0.0

    miner_ran = row.get("outcome_miner_ran", False)
    grid_import = abs(row.get("outcome_grid_import", 0))
    soc_next = row.get("outcome_soc_next_cycle", 50)
    soc_at_decision = row.get("battery_soc", 50)
    target_soc = row.get("decision_target_soc_by_sunrise", 30)
    solar_surplus = row.get("solar_surplus_power", 0)
    miner_power = row.get("decision_miner_power", 0)

    if miner_ran:
        reward += 10.0

    if miner_ran and grid_import < GRID_IMPORT_TOLERANCE:
        reward += 15.0

    if grid_import > 300:
        reward -= 20.0
    elif grid_import > GRID_IMPORT_TOLERANCE:
        reward -= 5.0

    if soc_next < BATTERY_SOC_CRITICAL:
        reward -= 30.0
    elif soc_next < BATTERY_SOC_MIN:
        reward -= 15.0

    soc_error = abs(soc_next - target_soc)
    if soc_error <= 2:
        reward += 5.0
    elif soc_error <= 5:
        reward += 2.0

    if miner_ran and solar_surplus > 0:
        wasted = max(0, solar_surplus - miner_power)
        if wasted < 500:
            reward += 3.0
        elif wasted < 1500:
            reward += 1.0

    return round(reward, 2)


def rule_teacher(features: list[float], battery_capacity_kwh: float = 69.6) -> dict:
    """Rule-based teacher that replicates Ollama logic exactly.

    Used for bootstrapping before ML model has enough data, and as fallback.
    """
    f = dict(zip(FEATURE_NAMES, features))

    hour = f["hour_of_day"]
    is_daytime = f["is_daytime"]
    hours_to_sunrise = f["hours_until_sunrise"]
    solar = f["solar_power_total"]
    surplus = f["solar_surplus_power"]
    soc = f["battery_soc"]
    drain_rate = f["battery_drain_rate"]
    hours_to_min = f["battery_hours_to_min"]
    house_load = f["house_load"]
    forecast_tmrw = f["forecast_tomorrow"]
    grid_power = f["grid_power"]
    miner_power = f["current_miner_power"]
    miner_on = f["miner_is_on"]

    if soc < BATTERY_SOC_CRITICAL:
        return {
            "miner_active": "off",
            "miner_power": MINER_POWER_MIN,
            "target_soc_by_sunrise": 30,
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
            "miner_power": max(MINER_POWER_MIN, decision_power) if miner_active == "on" else MINER_POWER_MIN,
            "target_soc_by_sunrise": 30,
            "mode": "day_solar",
            "reason": f"DAY: Solar available {available_for_miner:.0f}W, miner set to {decision_power}W",
        }

    if forecast_tmrw > 60:
        target_soc = 12
    elif forecast_tmrw > 30:
        target_soc = 30
    elif forecast_tmrw > 15:
        target_soc = 45
    else:
        target_soc = 60

    energy_to_drain = ((soc - target_soc) / 100) * battery_capacity_kwh
    if hours_to_sunrise > 0:
        required_watts = (energy_to_drain * 1000) / hours_to_sunrise
    else:
        required_watts = 0
    miner_needed = required_watts - house_load

    if miner_needed < MINER_POWER_MIN:
        miner_active = "off"
        decision_power = MINER_POWER_MIN
    elif miner_needed > MINER_POWER_MAX:
        miner_active = "on"
        decision_power = MINER_POWER_MAX
    else:
        miner_active = "on"
        decision_power = clamp_power(miner_needed)

    return {
        "miner_active": miner_active,
        "miner_power": decision_power if miner_active == "on" else MINER_POWER_MIN,
        "target_soc_by_sunrise": target_soc,
        "mode": "night_drain",
        "reason": f"NIGHT: Need {miner_needed:.0f}W to drain {energy_to_drain:.1f}kWh by sunrise (target SoC {target_soc}%)",
    }


def load_model(hass_config_path) -> tuple:
    """Load trained model from disk. Returns (model, feature_names)."""
    model_path = _get_model_path(hass_config_path)
    if not model_path.exists():
        return None, FEATURE_NAMES
    with open(model_path, "rb") as f:
        return pickle.load(f), FEATURE_NAMES


def save_model(hass_config_path, model) -> None:
    """Save trained model to disk."""
    model_path = _get_model_path(hass_config_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(hass_config_path, metrics: dict) -> None:
    """Save training metrics to JSON."""
    metrics_path = _get_metrics_path(hass_config_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(hass_config_path) -> dict:
    """Load training metrics from JSON."""
    metrics_path = _get_metrics_path(hass_config_path)
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r") as f:
        return json.load(f)


def log_decision_to_csv(hass_config_path, state: dict, decision: dict) -> None:
    """Append this decision + state to training CSV for future retraining."""
    csv_path = _get_training_csv_path(hass_config_path)

    file_exists = csv_path.exists()
    features = features_from_state(state)

    row = {name: val for name, val in zip(FEATURE_NAMES, features)}
    row["timestamp"] = datetime.now().isoformat()
    row["decision_miner_active"] = decision["miner_active"]
    row["decision_miner_power"] = decision["miner_power"]
    row["decision_target_soc_by_sunrise"] = decision["target_soc_by_sunrise"]
    row["decision_mode"] = decision["mode"]
    row["decision_reason"] = decision["reason"]
    row["outcome_soc_next_cycle"] = ""
    row["outcome_grid_import"] = ""
    row["outcome_miner_ran"] = ""
    row["reward"] = ""

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_training_sample_count(hass_config_path) -> int:
    """Get the number of training samples from CSV."""
    csv_path = _get_training_csv_path(hass_config_path)
    if not csv_path.exists():
        return 0
    with open(csv_path, "r") as f:
        return max(0, sum(1 for _ in f) - 1)


def fill_rewards_and_outcomes(rows: list[dict]) -> list[dict]:
    """Fill reward and outcome fields by looking at consecutive rows."""
    for i in range(len(rows) - 1):
        current = rows[i]
        next_row = rows[i + 1]

        if not current.get("outcome_soc_next_cycle"):
            current["outcome_soc_next_cycle"] = next_row.get("battery_soc", "")

        if not current.get("outcome_grid_import"):
            current["outcome_grid_import"] = next_row.get("grid_power", "")

        if not current.get("outcome_miner_ran"):
            current["outcome_miner_ran"] = current.get("decision_miner_active", "off") == "on"

        if not current.get("reward"):
            current["reward"] = compute_reward(current)

    return rows


def read_training_data(hass_config_path) -> list[dict]:
    """Read and process all training data from CSV."""
    csv_path = _get_training_csv_path(hass_config_path)
    if not csv_path.exists():
        return []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = fill_rewards_and_outcomes(rows)

    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def train_model(rows: list[dict], min_samples: int = 50) -> tuple:
    """Train gradient boosted regressor on historical data.

    Returns (model, val_score, top_features, importances).
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    valid_rows = [r for r in rows if r.get("reward") and float(r.get("reward", 0)) != 0]

    X = []
    y = []
    weights = []
    for row in valid_rows:
        features = [float(row.get(name, 0)) for name in FEATURE_NAMES]
        power = float(row.get("decision_miner_power", 3500))
        reward = float(row.get("reward", 0))

        X.append(features)
        y.append(power)
        weights.append(max(0.1, 1.0 + reward / 30.0))

    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )

    if len(X) >= min_samples and len(X) >= CROSS_VAL_FOLDS:
        n_folds = min(CROSS_VAL_FOLDS, len(X))
        scores = cross_val_score(model, X, y, cv=n_folds, scoring="neg_mean_absolute_error")
        val_score = float(-scores.mean())
    else:
        val_score = None

    model.fit(X, y, sample_weight=weights)

    importances = dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_]))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    return model, val_score, top_features, importances


def run_retrain(
    hass_config_path,
    min_samples: int = 50,
    force: bool = False,
) -> dict:
    """Run a full retrain cycle. Returns metrics dict."""
    rows = read_training_data(hass_config_path)
    total = len(rows)

    if total < MIN_SAMPLES_FOR_TEACHER_ONLY:
        metrics = {
            "last_retrain": datetime.now().isoformat(),
            "total_samples": total,
            "status": "insufficient_data",
            "message": f"Need {MIN_SAMPLES_FOR_TEACHER_ONLY}+ samples, have {total}. Using rule teacher.",
        }
        save_metrics(hass_config_path, metrics)
        return metrics

    model, val_score, top_features, importances = train_model(rows, min_samples)

    current_metrics = load_metrics(hass_config_path)
    current_val = current_metrics.get("best_val_score")

    should_save = False
    if val_score is None:
        should_save = True
        status = "trained_no_cv"
    elif current_val is None or val_score < current_val:
        should_save = True
        status = "improved"
    elif force:
        should_save = True
        status = "forced"
    else:
        status = "not_improved"

    if should_save:
        save_model(hass_config_path, model)

    avg_reward = sum(float(r.get("reward", 0)) for r in rows if r.get("reward")) / max(1, total)
    rewards = [float(r.get("reward", 0)) for r in rows if r.get("reward")]

    metrics = {
        "last_retrain": datetime.now().isoformat(),
        "total_samples": total,
        "status": status,
        "val_mae": round(val_score, 2) if val_score else None,
        "best_val_score": (
            round(min(val_score, current_val), 2)
            if val_score and current_val
            else round(val_score, 2)
            if val_score
            else current_val
        ),
        "avg_reward": round(avg_reward, 2),
        "min_reward": round(min(rewards), 2) if rewards else 0,
        "max_reward": round(max(rewards), 2) if rewards else 0,
        "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
        "model_saved": should_save,
    }
    save_metrics(hass_config_path, metrics)
    return metrics


def migrate_legacy_data(hass_config_path) -> bool:
    """Migrate data from legacy /config/ml_models/ to new location.

    Returns True if migration was performed.
    """
    legacy_dir = Path(hass_config_path(ML_DATA_DIR)).parent / "ml_models"
    if not legacy_dir.exists():
        return False

    new_dir = _get_data_dir(hass_config_path)
    migrated = False

    for legacy_name, new_name in [
        ("mining_model.pkl", MODEL_FILENAME),
        ("training_data.csv", TRAINING_CSV_FILENAME),
        ("training_metrics.json", METRICS_FILENAME),
    ]:
        legacy_file = legacy_dir / legacy_name
        new_file = new_dir / new_name
        if legacy_file.exists() and not new_file.exists():
            import shutil

            shutil.copy2(legacy_file, new_file)
            _LOGGER.info("Migrated legacy file: %s -> %s", legacy_file, new_file)
            migrated = True

    return migrated
