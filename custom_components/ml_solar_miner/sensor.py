"""Sensor platform for ML Solar Miner."""
import logging
from datetime import datetime

from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_AVG_REWARD,
    ATTR_LAST_RETRAIN,
    ATTR_MAX_REWARD,
    ATTR_MIN_REWARD,
    ATTR_MINER_ACTIVE,
    ATTR_MINER_POWER,
    ATTR_MODE,
    ATTR_MODEL_SAVED,
    ATTR_REASON,
    ATTR_SOURCE,
    ATTR_STATUS,
    ATTR_TARGET_SOC,
    ATTR_TOP_FEATURES,
    ATTR_TOTAL_SAMPLES,
    ATTR_VAL_MAE,
    DOMAIN,
)
from .coordinator import MLSolarMinerCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ML Solar Miner sensors."""
    coordinator: MLSolarMinerCoordinator = hass.data[DOMAIN][config_entry.entry_id]

    entities = [
        MLSolarMinerDecisionModeSensor(coordinator, config_entry),
        MLSolarMinerDecisionReasonSensor(coordinator, config_entry),
        MLSolarMinerDecisionPowerSensor(coordinator, config_entry),
        MLSolarMinerTargetSocSensor(coordinator, config_entry),
        MLSolarMinerModelSourceSensor(coordinator, config_entry),
        MLSolarMinerTrainingSamplesSensor(coordinator, config_entry),
        MLSolarMinerTrainingStatusSensor(coordinator, config_entry),
        MLSolarMinerLastRetrainSensor(coordinator, config_entry),
        MLSolarMinerLastDecisionSensor(coordinator, config_entry),
    ]

    async_add_entities(entities)


class MLSolarMinerSensorBase(CoordinatorEntity):
    """Base class for ML Solar Miner sensors."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: MLSolarMinerCoordinator, config_entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._config_entry = config_entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name="ML Solar Miner",
            manufacturer="ML Solar Miner",
            model="ML Decision Engine",
            sw_version="1.0.0",
        )


class MLSolarMinerDecisionModeSensor(MLSolarMinerSensorBase):
    """Sensor for the current decision mode."""

    _attr_unique_id = "ml_solar_miner_decision_mode"
    _attr_translation_key = "ml_solar_miner_decision_mode"

    @property
    def native_value(self) -> str | None:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("mode")
        return None


class MLSolarMinerDecisionReasonSensor(MLSolarMinerSensorBase):
    """Sensor for the current decision reason."""

    _attr_unique_id = "ml_solar_miner_decision_reason"
    _attr_translation_key = "ml_solar_miner_decision_reason"

    @property
    def native_value(self) -> str | None:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("reason")
        return None


class MLSolarMinerDecisionPowerSensor(MLSolarMinerSensorBase):
    """Sensor for the target miner power."""

    _attr_unique_id = "ml_solar_miner_miner_power"
    _attr_translation_key = "ml_solar_miner_miner_power"
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> int | None:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("miner_power")
        return None


class MLSolarMinerTargetSocSensor(MLSolarMinerSensorBase):
    """Sensor for the target SoC by sunrise."""

    _attr_unique_id = "ml_solar_miner_target_soc"
    _attr_translation_key = "ml_solar_miner_target_soc"

    @property
    def native_value(self) -> int | None:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("target_soc_by_sunrise")
        return None


class MLSolarMinerModelSourceSensor(MLSolarMinerSensorBase):
    """Sensor indicating whether ML model or rule teacher is active."""

    _attr_unique_id = "ml_solar_miner_model_source"
    _attr_translation_key = "ml_solar_miner_model_source"

    @property
    def native_value(self) -> str | None:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("source")
        return None


class MLSolarMinerTrainingSamplesSensor(MLSolarMinerSensorBase):
    """Sensor for total training samples."""

    _attr_unique_id = "ml_solar_miner_training_samples"
    _attr_translation_key = "ml_solar_miner_training_samples"
    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    @property
    def native_value(self) -> int:
        if self.coordinator.last_decision:
            return self.coordinator.last_decision.get("training_samples", 0)
        return 0


class MLSolarMinerTrainingStatusSensor(MLSolarMinerSensorBase):
    """Sensor for training status with metrics as attributes."""

    _attr_unique_id = "ml_solar_miner_training_status"
    _attr_translation_key = "ml_solar_miner_training_status"

    @property
    def native_value(self) -> str | None:
        from .models import load_metrics

        metrics = load_metrics(self.coordinator.hass_config_path)
        return metrics.get("status", "no_data")

    @property
    def extra_state_attributes(self) -> dict:
        from .models import load_metrics

        metrics = load_metrics(self.coordinator.hass_config_path)
        return {
            ATTR_LAST_RETRAIN: metrics.get("last_retrain"),
            ATTR_TOTAL_SAMPLES: metrics.get("total_samples", 0),
            ATTR_STATUS: metrics.get("status", "no_data"),
            ATTR_VAL_MAE: metrics.get("val_mae"),
            ATTR_AVG_REWARD: metrics.get("avg_reward"),
            ATTR_MIN_REWARD: metrics.get("min_reward"),
            ATTR_MAX_REWARD: metrics.get("max_reward"),
            ATTR_MODEL_SAVED: metrics.get("model_saved"),
            ATTR_TOP_FEATURES: metrics.get("top_features", []),
        }


class MLSolarMinerLastRetrainSensor(MLSolarMinerSensorBase):
    """Sensor for the last retrain timestamp."""

    _attr_unique_id = "ml_solar_miner_last_retrain"
    _attr_translation_key = "ml_solar_miner_last_retrain"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> str | None:
        from .models import load_metrics

        metrics = load_metrics(self.coordinator.hass_config_path)
        ts = metrics.get("last_retrain")
        if ts:
            try:
                return datetime.fromisoformat(ts).isoformat()
            except ValueError:
                return ts
        return None


class MLSolarMinerLastDecisionSensor(MLSolarMinerSensorBase):
    """Sensor for the last decision timestamp."""

    _attr_unique_id = "ml_solar_miner_last_decision"
    _attr_translation_key = "ml_solar_miner_last_decision"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> str | None:
        if self.coordinator.last_decision:
            ts = self.coordinator.last_decision.get("timestamp")
            if ts:
                try:
                    return datetime.fromisoformat(ts).isoformat()
                except ValueError:
                    return ts
        return None
