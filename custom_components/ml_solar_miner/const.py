"""Constants for the ML Solar Miner integration."""

DOMAIN = "ml_solar_miner"
DEFAULT_NAME = "ML Solar Miner"

# Coordinator
DEFAULT_SCAN_INTERVAL = 1200  # 20 minutes

# Grid sign convention: +1 = positive means import (default / Fronius)
#                       -1 = positive means export (Victron-style)
GRID_SIGN_DEFAULT = 1

# Feature names for the ML model (17 — excludes current_miner_power and miner_is_on
# so the model cannot just "keep doing what we did")
ML_FEATURE_NAMES = [
    "hour_of_day",
    "is_daytime",
    "hours_until_sunrise",
    "solar_power_total",
    "solar_surplus_power",
    "battery_soc",
    "battery_voltage",
    "battery_current",
    "battery_power",
    "battery_kwh_available",
    "battery_drain_rate",
    "battery_hours_to_min",
    "house_load",
    "forecast_tomorrow",
    "forecast_day3",
    "grid_power",
    "mining_viability_score",
]

# Back-compat alias used by sensor/coordinator for feature extraction
FEATURE_NAMES = ML_FEATURE_NAMES

# Miner power constraints
MINER_POWER_MIN = 3500
MINER_POWER_MAX = 6000
MINER_POWER_STEP = 100

# Battery safety thresholds
BATTERY_SOC_MIN = 12
BATTERY_SOC_CRITICAL = 10
TARGET_SOC_MAX = 65
DEFAULT_BATTERY_CAPACITY_KWH = 69.6
DEFAULT_TARGET_SOC_MORNING = 30

# Grid import (positive watts = import from grid after sign convention applied)
GRID_IMPORT_TOLERANCE = 100
GRID_IMPORT_REDUCE_W = 300
GRID_IMPORT_RETRAIN_W = 500
GRID_IMPORT_RETRAIN_MINUTES = 10

# Grid hysteresis — prevents on/off chattering around the 3500W boundary
# When miner was OFF: require surplus > MINER_POWER_MIN + HYSTERESIS to start
# When miner was ON:  keep running until surplus < MINER_POWER_MIN - HYSTERESIS
GRID_HYSTERESIS_W = 200

# Missed surplus
MISSED_SURPLUS_W = 4000
MISSED_SURPLUS_MINUTES = 15

# Mining viability — hashprice floor below which the miner should stay off
# even with surplus available.  0.0 disables the check.
VIABILITY_FLOOR = 0.0

# Auto-retrain
RETRAIN_EVENT_COOLDOWN_SECONDS = 3600
WEEKLY_RETRAIN_WEEKDAY = 6  # Sunday
WEEKLY_RETRAIN_HOUR = 3
SUNRISE_HOUR = 6.5

# ML training
DEFAULT_MIN_SAMPLES_FOR_MODEL = 50
MIN_SAMPLES_FOR_TEACHER_ONLY = 20
MIN_SAMPLES_FOR_FORCE = 2
DEFAULT_RETRAIN_INTERVAL = 604800  # 7 days in seconds
CROSS_VAL_FOLDS = 5

# Minimum reward score for a row to be included in training
REWARD_FLOOR_FOR_TRAINING = -10.0

# Time-split validation: hold out the last N days as test set
MODEL_OBSERVATION_DAYS = 7

# Shadow period: teacher runs alone for N days before the model is allowed to act
MODEL_SHADOW_DAYS = 7

# Config flow keys
CONF_MINER_SWITCH = "miner_switch"
CONF_MINER_POWER_NUMBER = "miner_power_number"
CONF_SOLAR_POWER_TOTAL = "solar_power_total"
CONF_SOLAR_SURPLUS_POWER = "solar_surplus_power"
CONF_BATTERY_SOC = "battery_soc"
CONF_BATTERY_VOLTAGE = "battery_voltage"
CONF_BATTERY_CURRENT = "battery_current"
CONF_BATTERY_POWER = "battery_power"
CONF_BATTERY_KWH_AVAILABLE = "battery_kwh_available"
CONF_BATTERY_DRAIN_RATE = "battery_drain_rate"
CONF_BATTERY_HOURS_TO_MIN = "battery_hours_to_min"
CONF_HOURS_UNTIL_SUNRISE = "hours_until_sunrise"
CONF_TOTAL_LOAD_POWER = "total_load_power"
CONF_MINER_CONSUMPTION = "miner_consumption"
CONF_FORECAST_TOMORROW = "forecast_tomorrow"
CONF_FORECAST_DAY3 = "forecast_day3"
CONF_GRID_POWER = "grid_power"
CONF_MINING_VIABILITY_SCORE = "mining_viability_score"
CONF_SCAN_INTERVAL_OPTION = "scan_interval"
CONF_AUTO_CONTROL = "auto_control"
CONF_BATTERY_CAPACITY_KWH = "battery_capacity_kwh"
CONF_MIN_SAMPLES_FOR_MODEL = "min_samples_for_model"
CONF_RETRAIN_INTERVAL = "retrain_interval"
CONF_GRID_INVERT = "grid_invert"

# Decision engine selection -------------------------------------------------
# rule  -> always use the standalone script's rule engine (rule_engine.py)
# ml    -> use the trained ML model, falling back to the rule engine
# auto  -> use ML once it is ready, otherwise the rule engine
CONF_MODEL_TYPE = "model_type"
ENGINE_RULE = "rule"
ENGINE_ML = "ml"
ENGINE_AUTO = "auto"
ENGINE_OPTIONS = (ENGINE_RULE, ENGINE_ML, ENGINE_AUTO)
DEFAULT_MODEL_TYPE = ENGINE_AUTO

# Rule engine settings (mirrors the legacy /config/scripts/mining_ml_utils.py)
CONF_TARGET_SOC_MORNING = "target_soc_morning"
CONF_TARGET_SOC_ENTITY = "target_soc_entity"
CONF_BATTERY_CAPACITY_ENTITY = "battery_capacity_entity"
# While this input_boolean is "on" the component holds the miner off (the legacy
# decision engine had the same condition). Defaults to the helper the legacy
# automations used, so a cutover keeps the lock working out of the box.
CONF_EMERGENCY_LOCK_ENTITY = "emergency_lock_entity"
DEFAULT_EMERGENCY_LOCK_ENTITY = "input_boolean.emergency_battery_lock"

# Battery capacity auto-tune -------------------------------------------------
# Measures the real pack capacity from the system power balance
# (solar + grid import - load) against the SoC change reported by the BMS.
# Needed here because the measured battery current only sees part of a bank
# when extra batteries are wired in parallel.
CONF_CAPACITY_AUTOTUNE = "capacity_autotune"
DEFAULT_CAPACITY_AUTOTUNE = False
CONF_BATTERY_EFFICIENCY = "battery_efficiency"
DEFAULT_BATTERY_EFFICIENCY = 95  # percent, one-way (LiFePO4)
BATTERY_EFFICIENCY_MIN = 80
BATTERY_EFFICIENCY_MAX = 100

# Accepted-window rules for the capacity estimator
CAPACITY_SOC_LOW = 20.0  # avoid the flat bottom of the SoC curve
CAPACITY_SOC_HIGH = 90.0  # avoid the flat top / tapering
CAPACITY_MIN_SOC_DELTA = 15.0  # percentage points per window
CAPACITY_MIN_WINDOW_MINUTES = 30.0
CAPACITY_MAX_WINDOW_HOURS = 24.0
CAPACITY_MIN_WINDOW_SAMPLES = 3
CAPACITY_MIN_POWER_W = 100.0  # flows below this do not set a direction
CAPACITY_MIN_WINDOWS = 3  # accepted windows before the value is applied
CAPACITY_HISTORY_LIMIT = 20
CAPACITY_MAX_CHANGE_PER_UPDATE = 0.20
CAPACITY_SANITY_RATIO = 3.0
CAPACITY_MIN_KWH = 10.0
CAPACITY_MAX_KWH = 200.0

# Entity attribute keys
ATTR_LAST_RETRAIN = "last_retrain"
ATTR_TOTAL_SAMPLES = "total_samples"
ATTR_STATUS = "status"
ATTR_VAL_MAE = "val_mae"
ATTR_AVG_REWARD = "avg_reward"
ATTR_MIN_REWARD = "min_reward"
ATTR_MAX_REWARD = "max_reward"
ATTR_MODEL_SAVED = "model_saved"
ATTR_TOP_FEATURES = "top_features"
ATTR_MODE = "mode"
ATTR_REASON = "reason"
ATTR_SOURCE = "source"
ATTR_MINER_ACTIVE = "miner_active"
ATTR_MINER_POWER = "miner_power"
ATTR_TARGET_SOC = "target_soc_by_sunrise"

# File paths (relative to HA config dir)
ML_DATA_DIR = "ml_solar_miner"
MODEL_FILENAME = "mining_model.pkl"
TRAINING_CSV_FILENAME = "training_data.csv"
METRICS_FILENAME = "training_metrics.json"
LAST_DECISION_FILENAME = "last_decision.json"
LEARNED_CAPACITY_FILENAME = "learned_capacity.json"

# Canonical column order for the training CSV. Kept fixed (instead of trusting
# the first row's keys) so the writer and the retrain reader always agree, even
# when rows are rewritten or a column is added.
TRAINING_CSV_FIELDS = tuple(ML_FEATURE_NAMES) + (
    "timestamp",
    "decision_miner_active",
    "decision_miner_power",
    "decision_target_soc_by_sunrise",
    "decision_mode",
    "decision_reason",
    "outcome_soc_next_cycle",
    "outcome_grid_import",
    "outcome_miner_ran",
    "reward",
)

# Legacy paths for migration
LEGACY_ML_MODELS_DIR = "ml_models"
LEGACY_MODEL_FILENAME = "mining_model.pkl"
LEGACY_TRAINING_CSV_FILENAME = "training_data.csv"
LEGACY_METRICS_FILENAME = "training_metrics.json"

# Sensor keys stored on the config entry
SENSOR_STATE_KEYS = (
    "solar_power_total",
    "solar_surplus_power",
    "battery_soc",
    "battery_voltage",
    "battery_current",
    "battery_power",
    "battery_kwh_available",
    "battery_drain_rate",
    "battery_hours_to_min",
    "hours_until_sunrise",
    "total_load_power",
    "miner_consumption",
    "forecast_tomorrow",
    "forecast_day3",
    "grid_power",
    "mining_viability_score",
)


def get_entry_value(entry, key, default=None):
    """Read a value from options, falling back to data for older entries."""
    if key in entry.options:
        return entry.options[key]
    return entry.data.get(key, default)
