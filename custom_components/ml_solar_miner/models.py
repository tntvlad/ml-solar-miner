"""ML model management for the Solar Miner integration.

Consolidates logic from mining_ml_utils.py, mining_ml_decision.py,
and mining_ml_retrain.py into HA-independent modules.
"""
import csv
import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .const import (
    BATTERY_SOC_CRITICAL,
    BATTERY_SOC_MIN,
    CROSS_VAL_FOLDS,
    FEATURE_NAMES,
    GRID_IMPORT_REDUCE_W,
    GRID_IMPORT_TOLERANCE,
    LEGACY_METRICS_FILENAME,
    LEGACY_ML_MODELS_DIR,
    LEGACY_MODEL_FILENAME,
    LEGACY_TRAINING_CSV_FILENAME,
    METRICS_FILENAME,
    MIN_SAMPLES_FOR_FORCE,
    MIN_SAMPLES_FOR_TEACHER_ONLY,
    MINER_POWER_MAX,
    MINER_POWER_MIN,
    MINER_POWER_STEP,
    ML_DATA_DIR,
    MODEL_FILENAME,
    SUNRISE_HOUR,
    TRAINING_CSV_FILENAME,
)

_LOGGER = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return utc_now().isoformat()


def parse_iso_datetime(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _as_float(value, default: float = 0.0) -> float:
    """Coerce CSV/HA values to float."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    """Coerce CSV/HA values to bool. The string 'False' is False."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_blank(value) -> bool:
    return value is None or value == ""


def _reward_is_present(row: dict) -> bool:
    reward = row.get("reward")
    if _is_blank(reward):
        return False
    try:
        float(reward)
        return True
    except (TypeError, ValueError):
        return False


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


def clamp_power(raw_power: float) -> int:
    """Clamp miner power to the valid on-range and round down to 100W."""
    clamped = max(MINER_POWER_MIN, min(MINER_POWER_MAX, int(raw_power)))
    return int((clamped // MINER_POWER_STEP) * MINER_POWER_STEP)


def decide_power(raw_power: float) -> tuple[str, int]:
    """Return (miner_active, power). Never clamps a sub-min value up to min."""
    if raw_power < MINER_POWER_MIN:
        return "off", MINER_POWER_MIN
    return "on", clamp_power(raw_power)


def estimate_hours_until_sunrise(hour: float) -> float:
    """Estimate hours until ~06:30 when the sensor is missing."""
    hour = hour % 24
    if hour < SUNRISE_HOUR:
        return SUNRISE_HOUR - hour
    return (24.0 - hour) + SUNRISE_HOUR


def target_soc_from_forecast(forecast_tmrw: float) -> int:
    """Night target SoC from tomorrow's solar forecast (kWh-ish score)."""
    if forecast_tmrw > 60:
        return 12
    if forecast_tmrw > 30:
        return 30
    if forecast_tmrw > 15:
        return 45
    return 60


def grid_import_watts(grid_power: float) -> float:
    """Positive watts imported from the grid. Export (negative) is 0."""
    return max(0.0, grid_power)


def features_from_state(state: dict) -> list[float]:
    """Extract ordered feature vector from HA state dictionary."""
    hour = float(utc_now().astimezone().hour)
    solar = _as_float(state.get("solar_power_total"), 0)
    soc = _as_float(state.get("battery_soc"), 0)
    house_load_raw = _as_float(state.get("total_load_power"), 0)
    miner_actual = _as_float(state.get("miner_consumption"), 0)
    house_load = max(0.0, abs(house_load_raw) - miner_actual)

    raw_hours = state.get("hours_until_sunrise")
    if raw_hours is None or raw_hours == "":
        hours_until_sunrise = estimate_hours_until_sunrise(hour)
    else:
        hours_until_sunrise = _as_float(raw_hours, 0)

    return [
        hour,
        1.0 if solar > 100 else 0.0,
        hours_until_sunrise,
        solar,
        _as_float(state.get("solar_surplus_power"), 0),
        soc,
        _as_float(state.get("battery_voltage"), 0),
        _as_float(state.get("battery_current"), 0),
        _as_float(state.get("battery_power"), 0),
        _as_float(state.get("battery_kwh_available"), 0),
        _as_float(state.get("battery_drain_rate"), 0),
        _as_float(state.get("battery_hours_to_min"), 99),
        house_load,
        _as_float(state.get("forecast_tomorrow"), 0),
        _as_float(state.get("forecast_day3"), 0),
        _as_float(state.get("grid_power"), 0),
        _as_float(state.get("mining_viability_score"), 0),
        _as_float(state.get("current_miner_power"), 0),
        1.0 if _as_bool(state.get("miner_is_on"), False) else 0.0,
    ]


def validate_decision(decision: dict) -> dict:
    """Enforce SoC and grid-import safety constraints on a decision dict."""
    soc = _as_float(decision.get("_soc"), 50)
    grid_import = grid_import_watts(_as_float(decision.get("_grid_power"), 0))
    miner_active = "on" if _as_bool(decision.get("miner_active"), False) else "off"
    miner_power = int(_as_float(decision.get("miner_power"), MINER_POWER_MIN))
    mode = decision.get("mode", "unknown")
    reason = decision.get("reason", "")

    if soc < BATTERY_SOC_CRITICAL:
        miner_active = "off"
        miner_power = MINER_POWER_MIN
        mode = "safety_shutdown"
        reason = f"Safety: SoC {soc:.0f}% below critical {BATTERY_SOC_CRITICAL}%"
    elif soc < BATTERY_SOC_MIN and miner_active == "on":
        miner_active = "off"
        miner_power = MINER_POWER_MIN
        mode = "safety_shutdown"
        reason = f"Safety: SoC {soc:.0f}% below minimum {BATTERY_SOC_MIN}%"
    elif grid_import > GRID_IMPORT_REDUCE_W and miner_active == "on":
        reduced = miner_power - grid_import
        miner_active, miner_power = decide_power(reduced)
        if miner_active == "off":
            reason = f"Safety: grid import {grid_import:.0f}W, miner off"
        else:
            reason = f"{reason} (reduced for {grid_import:.0f}W grid import)".strip()

    if miner_active == "on":
        miner_power = clamp_power(miner_power)
    else:
        miner_power = MINER_POWER_MIN

    target_soc = max(10, min(65, int(_as_float(decision.get("target_soc_by_sunrise"), 30))))

    return {
        "miner_active": miner_active,
        "miner_power": miner_power,
        "target_soc_by_sunrise": target_soc,
        "mode": mode,
        "reason": reason,
    }


def compute_reward(row: dict) -> float:
    """Compute reward score for a historical decision+outcome row."""
    reward = 0.0

    miner_ran = _as_bool(row.get("outcome_miner_ran"), False)
    grid_import = grid_import_watts(_as_float(row.get("outcome_grid_import"), 0))
    soc_next = _as_float(row.get("outcome_soc_next_cycle"), 50)
    target_soc = _as_float(row.get("decision_target_soc_by_sunrise"), 30)
    solar_surplus = _as_float(row.get("solar_surplus_power"), 0)
    miner_power = _as_float(row.get("decision_miner_power"), 0)

    if miner_ran:
        reward += 10.0

    if miner_ran and grid_import < GRID_IMPORT_TOLERANCE:
        reward += 15.0

    if grid_import > GRID_IMPORT_REDUCE_W:
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
        wasted = max(0.0, solar_surplus - miner_power)
        if wasted < 500:
            reward += 3.0
        elif wasted < 1500:
            reward += 1.0

    return round(reward, 2)


def rule_teacher(features: list[float], battery_capacity_kwh: float = 69.6) -> dict:
    """Rule-based teacher used to bootstrap and as a fallback."""
    f = dict(zip(FEATURE_NAMES, features))

    is_daytime = f["is_daytime"]
    hours_to_sunrise = f["hours_until_sunrise"]
    surplus = f["solar_surplus_power"]
    soc = f["battery_soc"]
    house_load = f["house_load"]
    forecast_tmrw = f["forecast_tomorrow"]
    grid_import = grid_import_watts(f["grid_power"])

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
        if grid_import > GRID_IMPORT_TOLERANCE:
            available_for_miner -= grid_import

        miner_active, decision_power = decide_power(available_for_miner)
        return {
            "miner_active": miner_active,
            "miner_power": decision_power,
            "target_soc_by_sunrise": 30,
            "mode": "day_solar",
            "reason": (
                f"DAY: Solar available {available_for_miner:.0f}W, "
                f"miner set to {decision_power}W"
            ),
        }

    target_soc = target_soc_from_forecast(forecast_tmrw)
    energy_to_drain = ((soc - target_soc) / 100) * battery_capacity_kwh
    if hours_to_sunrise > 0:
        required_watts = (energy_to_drain * 1000) / hours_to_sunrise
    else:
        required_watts = 0
    miner_needed = required_watts - house_load

    miner_active, decision_power = decide_power(miner_needed)
    return {
        "miner_active": miner_active,
        "miner_power": decision_power,
        "target_soc_by_sunrise": target_soc,
        "mode": "night_drain",
        "reason": (
            f"NIGHT: Need {miner_needed:.0f}W to drain {energy_to_drain:.1f}kWh "
            f"by sunrise (target SoC {target_soc}%)"
        ),
    }


def load_model(hass_config_path) -> tuple:
    """Load trained model from disk. Returns (model, feature_names)."""
    model_path = _get_model_path(hass_config_path)
    if not model_path.exists():
        return None, FEATURE_NAMES
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f), FEATURE_NAMES
    except Exception as err:  # noqa: BLE001 — pickle/sklearn version mismatches
        _LOGGER.warning("Failed to load model %s (%s); using rule teacher", model_path, err)
        return None, FEATURE_NAMES


def save_model(hass_config_path, model) -> None:
    """Save trained model to disk."""
    model_path = _get_model_path(hass_config_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        _LOGGER.warning("Failed to load metrics: %s", err)
        return {}


def log_decision_to_csv(hass_config_path, state: dict, decision: dict) -> None:
    """Append this decision + state to training CSV for future retraining."""
    csv_path = _get_training_csv_path(hass_config_path)

    file_exists = csv_path.exists()
    features = features_from_state(state)

    row = {name: val for name, val in zip(FEATURE_NAMES, features)}
    row["timestamp"] = utc_now_iso()
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


def _fill_row_outcome(current: dict, next_state: dict) -> None:
    """Fill outcome/reward on current from the following cycle's state."""
    if _is_blank(current.get("outcome_soc_next_cycle")):
        current["outcome_soc_next_cycle"] = next_state.get("battery_soc", "")

    if _is_blank(current.get("outcome_grid_import")):
        current["outcome_grid_import"] = next_state.get("grid_power", "")

    if _is_blank(current.get("outcome_miner_ran")):
        current["outcome_miner_ran"] = _as_bool(next_state.get("miner_is_on"), False)

    if _is_blank(current.get("reward")):
        current["reward"] = compute_reward(current)


def fill_rewards_and_outcomes(
    rows: list[dict], live_state: dict | None = None
) -> list[dict]:
    """Fill reward and outcome fields from the next row, or live state for the last."""
    for i in range(len(rows) - 1):
        _fill_row_outcome(rows[i], rows[i + 1])

    if live_state and rows:
        _fill_row_outcome(rows[-1], live_state)

    return rows


def read_training_data(hass_config_path, live_state: dict | None = None) -> list[dict]:
    """Read and process all training data from CSV."""
    csv_path = _get_training_csv_path(hass_config_path)
    if not csv_path.exists():
        return []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = fill_rewards_and_outcomes(rows, live_state=live_state)

    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def _make_regressor(n_samples: int):
    from sklearn.ensemble import GradientBoostingRegressor

    min_leaf = min(5, max(1, n_samples // 4))
    return GradientBoostingRegressor(
        n_estimators=100 if n_samples >= 20 else min(50, max(10, n_samples * 5)),
        max_depth=min(4, max(1, n_samples // 3)),
        learning_rate=0.1,
        min_samples_leaf=min_leaf,
        subsample=0.8 if n_samples >= 10 else 1.0,
        random_state=42,
    )


def train_model(rows: list[dict], min_samples: int = 50) -> tuple:
    """Train gradient boosted regressor on historical data.

    Returns (model, val_score, top_features, importances).
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold

    valid_rows = [r for r in rows if _reward_is_present(r)]
    if len(valid_rows) < MIN_SAMPLES_FOR_FORCE:
        raise ValueError(f"Need at least {MIN_SAMPLES_FOR_FORCE} rewarded rows, have {len(valid_rows)}")

    X = []
    y = []
    weights = []
    for row in valid_rows:
        features = [_as_float(row.get(name), 0) for name in FEATURE_NAMES]
        power = _as_float(row.get("decision_miner_power"), 3500)
        reward = _as_float(row.get("reward"), 0)

        X.append(features)
        y.append(power)
        weights.append(max(0.1, 1.0 + reward / 30.0))

    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    n_samples = len(X)

    model = _make_regressor(n_samples)

    if n_samples >= min_samples and n_samples >= CROSS_VAL_FOLDS:
        n_folds = min(CROSS_VAL_FOLDS, n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        maes: list[float] = []
        for train_idx, test_idx in kf.split(X):
            fold = _make_regressor(len(train_idx))
            fold.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
            pred = fold.predict(X[test_idx])
            maes.append(float(mean_absolute_error(y[test_idx], pred)))
        val_score = float(np.mean(maes))
    else:
        val_score = None

    model.fit(X, y, sample_weight=weights)

    importances = dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_]))
    top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]

    return model, val_score, top_features, importances


def run_retrain(
    hass_config_path,
    min_samples: int = 50,
    force: bool = False,
    live_state: dict | None = None,
) -> dict:
    """Run a full retrain cycle. Returns metrics dict."""
    rows = read_training_data(hass_config_path, live_state=live_state)
    total = len(rows)
    floor = MIN_SAMPLES_FOR_FORCE if force else MIN_SAMPLES_FOR_TEACHER_ONLY

    if total < floor:
        metrics = {
            "last_retrain": utc_now_iso(),
            "total_samples": total,
            "status": "insufficient_data",
            "message": f"Need {floor}+ samples, have {total}. Using rule teacher.",
            "model_saved": False,
        }
        save_metrics(hass_config_path, metrics)
        return metrics

    try:
        model, val_score, top_features, importances = train_model(rows, min_samples)
    except ValueError as err:
        metrics = {
            "last_retrain": utc_now_iso(),
            "total_samples": total,
            "status": "insufficient_data",
            "message": str(err),
            "model_saved": False,
        }
        save_metrics(hass_config_path, metrics)
        return metrics

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

    rewarded = [r for r in rows if _reward_is_present(r)]
    rewards = [_as_float(r.get("reward"), 0) for r in rewarded]
    avg_reward = sum(rewards) / max(1, len(rewards))

    if should_save:
        # Track the score of the model actually on disk.
        best_val_score = round(val_score, 2) if val_score is not None else current_val
    else:
        best_val_score = current_val

    metrics = {
        "last_retrain": utc_now_iso(),
        "total_samples": total,
        "status": status,
        "val_mae": round(val_score, 2) if val_score is not None else None,
        "best_val_score": best_val_score,
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
    legacy_dir = Path(hass_config_path(LEGACY_ML_MODELS_DIR))
    if not legacy_dir.exists():
        return False

    new_dir = _get_data_dir(hass_config_path)
    migrated = False

    for legacy_name, new_name in [
        (LEGACY_MODEL_FILENAME, MODEL_FILENAME),
        (LEGACY_TRAINING_CSV_FILENAME, TRAINING_CSV_FILENAME),
        (LEGACY_METRICS_FILENAME, METRICS_FILENAME),
    ]:
        legacy_file = legacy_dir / legacy_name
        new_file = new_dir / new_name
        if legacy_file.exists() and not new_file.exists():
            shutil.copy2(legacy_file, new_file)
            _LOGGER.info("Migrated legacy file: %s -> %s", legacy_file, new_file)
            migrated = True

    return migrated
