"""DataUpdateCoordinator for ML Solar Miner."""
import asyncio
import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_AUTO_CONTROL,
    CONF_BATTERY_CAPACITY_KWH,
    CONF_MINER_POWER_NUMBER,
    CONF_MINER_SWITCH,
    CONF_MIN_SAMPLES_FOR_MODEL,
    CONF_SCAN_INTERVAL_OPTION,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_MIN_SAMPLES_FOR_MODEL,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    MINER_POWER_MIN,
)
from .models import (
    features_from_state,
    get_training_sample_count,
    load_metrics,
    load_model,
    log_decision_to_csv,
    rule_teacher,
    validate_decision,
)

_LOGGER = logging.getLogger(__name__)


class MLSolarMinerCoordinator(DataUpdateCoordinator):
    """Coordinator that runs ML decision cycles."""

    def __init__(self, hass: HomeAssistant, config_entry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(
                seconds=config_entry.data.get(CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL)
            ),
        )
        self.config_entry = config_entry
        self.hass_config_path = hass.config.path

        # Configured entity IDs
        self.miner_switch = config_entry.data[CONF_MINER_SWITCH]
        self.miner_power_number = config_entry.data[CONF_MINER_POWER_NUMBER]
        self.auto_control = config_entry.data.get(CONF_AUTO_CONTROL, True)
        self.battery_capacity_kwh = config_entry.data.get(
            CONF_BATTERY_CAPACITY_KWH, DEFAULT_BATTERY_CAPACITY_KWH
        )
        self.min_samples_for_model = config_entry.data.get(
            CONF_MIN_SAMPLES_FOR_MODEL, DEFAULT_MIN_SAMPLES_FOR_MODEL
        )

        # Entity ID mapping (sensor config keys -> HA entity IDs)
        self.entity_map = {
            "solar_power_total": config_entry.data.get("solar_power_total"),
            "solar_surplus_power": config_entry.data.get("solar_surplus_power"),
            "battery_soc": config_entry.data.get("battery_soc"),
            "battery_voltage": config_entry.data.get("battery_voltage"),
            "battery_current": config_entry.data.get("battery_current"),
            "battery_power": config_entry.data.get("battery_power"),
            "battery_kwh_available": config_entry.data.get("battery_kwh_available"),
            "battery_drain_rate": config_entry.data.get("battery_drain_rate"),
            "battery_hours_to_min": config_entry.data.get("battery_hours_to_min"),
            "hours_until_sunrise": config_entry.data.get("hours_until_sunrise"),
            "total_load_power": config_entry.data.get("total_load_power"),
            "miner_consumption": config_entry.data.get("miner_consumption"),
            "forecast_tomorrow": config_entry.data.get("forecast_tomorrow"),
            "forecast_day3": config_entry.data.get("forecast_day3"),
            "grid_power": config_entry.data.get("grid_power"),
            "mining_viability_score": config_entry.data.get("mining_viability_score"),
            "current_miner_power": config_entry.data.get("miner_power_number"),
            "miner_is_on": config_entry.data.get("miner_switch"),
        }

        # State
        self.last_decision = None
        self._model = None
        self._model_loaded = False
        self._csv_lock = asyncio.Lock()

    async def _async_update_data(self) -> dict:
        """Run ML decision cycle. Called every scan_interval."""
        try:
            # 1. Read entity states
            state = await self._read_entity_states()

            # 2. Run ML inference in executor thread
            decision = await self.hass.async_add_executor_job(
                self._run_decision, state
            )

            # 3. Log decision to CSV
            async with self._csv_lock:
                await self.hass.async_add_executor_job(
                    log_decision_to_csv, self.hass_config_path, state, decision
                )

            # 4. Apply decision if auto-control enabled
            if self.auto_control:
                await self._apply_decision(decision)

            # 5. Store for sensors
            self.last_decision = decision
            return decision

        except Exception as err:
            _LOGGER.error("ML decision cycle failed: %s", err)
            raise UpdateFailed(f"ML decision cycle failed: {err}") from err

    async def _read_entity_states(self) -> dict:
        """Build state dict from configured HA entity IDs."""
        state = {}
        for key, entity_id in self.entity_map.items():
            if not entity_id:
                state[key] = 0.0 if key != "miner_is_on" else "off"
                continue

            ent = self.hass.states.get(entity_id)
            if ent and ent.state not in ("unavailable", "unknown"):
                try:
                    state[key] = (
                        float(ent.state) if key != "miner_is_on" else ent.state
                    )
                except (ValueError, TypeError):
                    state[key] = 0.0 if key != "miner_is_on" else "off"
            else:
                state[key] = 0.0 if key != "miner_is_on" else "off"
                if ent:
                    _LOGGER.warning(
                        "Entity %s is %s, using default", entity_id, ent.state
                    )
                else:
                    _LOGGER.warning("Entity %s not found, using default", entity_id)

        return state

    def _run_decision(self, state: dict) -> dict:
        """Synchronous ML inference — runs in executor thread."""
        features = features_from_state(state)

        # Load model (cached after first load)
        if not self._model_loaded:
            self._model, _ = load_model(self.hass_config_path)
            self._model_loaded = True

        if self._model is not None:
            import numpy as np

            X = np.array([features])
            predicted_power = int(self._model.predict(X)[0])
            mode = "day_solar" if state.get("solar_power_total", 0) > 100 else "night_drain"
            decision = {
                "miner_active": "on" if predicted_power >= MINER_POWER_MIN else "off",
                "miner_power": predicted_power,
                "target_soc_by_sunrise": 30,
                "mode": mode,
                "reason": "ML model prediction",
            }
            source = "ml_model"
        else:
            decision = rule_teacher(features, self.battery_capacity_kwh)
            source = "rule_teacher"

        decision["_soc"] = state.get("battery_soc", 50)
        decision = validate_decision(decision)
        decision["source"] = source
        decision["training_samples"] = get_training_sample_count(self.hass_config_path)
        decision["timestamp"] = datetime.now().isoformat()

        return decision

    async def _apply_decision(self, decision: dict) -> None:
        """Apply miner switch and power level via HA services."""
        try:
            # Set switch
            service = "turn_on" if decision["miner_active"] == "on" else "turn_off"
            await self.hass.services.async_call(
                "switch", service, {"entity_id": self.miner_switch}, blocking=True
            )

            # Set power level if miner is on
            if decision["miner_active"] == "on":
                await self.hass.services.async_call(
                    "number",
                    "set_value",
                    {
                        "entity_id": self.miner_power_number,
                        "value": decision["miner_power"],
                    },
                    blocking=True,
                )

            _LOGGER.debug(
                "Applied decision: miner=%s, power=%dW, mode=%s",
                decision["miner_active"],
                decision["miner_power"],
                decision["mode"],
            )

        except Exception as err:
            _LOGGER.error("Failed to apply decision: %s", err)

    def invalidate_model_cache(self) -> None:
        """Force model reload on next decision cycle."""
        self._model = None
        self._model_loaded = False

    async def async_retrain(self, force: bool = False) -> dict:
        """Trigger model retraining."""
        from .models import run_retrain

        metrics = await self.hass.async_add_executor_job(
            run_retrain,
            self.hass_config_path,
            self.min_samples_for_model,
            force,
        )

        # Invalidate model cache so next cycle loads the new model
        self.invalidate_model_cache()

        return metrics

    async def async_request_decision(self) -> dict:
        """Trigger an immediate decision cycle."""
        await self.async_request_refresh()
        return self.last_decision

    def get_status(self) -> dict:
        """Return current status for service response."""
        metrics = load_metrics(self.hass_config_path)
        return {
            "auto_control": self.auto_control,
            "last_decision": self.last_decision,
            "training_samples": get_training_sample_count(self.hass_config_path),
            "model_loaded": self._model is not None,
            "metrics": metrics,
        }
