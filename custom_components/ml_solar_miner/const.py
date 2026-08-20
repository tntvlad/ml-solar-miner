"""Constants for the ML Solar Miner integration."""

DOMAIN = "ml_solar_miner"
DEFAULT_NAME = "ML Solar Miner"

# Coordinator
DEFAULT_SCAN_INTERVAL = 1200  # 20 minutes

# Feature names (19 features for ML model)
FEATURE_NAMES = [
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
    "current_miner_power",
    "miner_is_on",
]

# Miner power constraints
MINER_POWER_MIN = 3500
MINER_POWER_MAX = 6000
MINER_POWER_STEP = 100

# Battery safety thresholds
BATTERY_SOC_MIN = 12
BATTERY_SOC_CRITICAL = 10
DEFAULT_BATTERY_CAPACITY_KWH = 69.6

# Grid import (positive watts = import from grid)
GRID_IMPORT_TOLERANCE = 100
GRID_IMPORT_REDUCE_W = 300
GRID_IMPORT_RETRAIN_W = 500
GRID_IMPORT_RETRAIN_MINUTES = 10
MISSED_SURPLUS_W = 4000
MISSED_SURPLUS_MINUTES = 15

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
