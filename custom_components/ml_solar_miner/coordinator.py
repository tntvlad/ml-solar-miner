"""DataUpdateCoordinator for ML Solar Miner."""
import asyncio
import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change, async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    BATTERY_SOC_MIN,
    CONF_AUTO_CONTROL,
    CONF_BATTERY_CAPACITY_KWH,
    CONF_GRID_INVERT,
    CONF_MINER_POWER_NUMBER,
    CONF_MINER_SWITCH,
    CONF_MIN_SAMPLES_FOR_MODEL,
    CONF_RETRAIN_INTERVAL,
    CONF_SCAN_INTERVAL_OPTION,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_MIN_SAMPLES_FOR_MODEL,
    DEFAULT_RETRAIN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    GRID_IMPORT_RETRAIN_MINUTES,
    GRID_IMPORT_RETRAIN_W,
    MINER_POWER_MIN,
    MISSED_SURPLUS_MINUTES,
    MISSED_SURPLUS_W,
    MODEL_SHADOW_DAYS,
    RETRAIN_EVENT_COOLDOWN_SECONDS,
    SENSOR_STATE_KEYS,
    WEEKLY_RETRAIN_HOUR,
    WEEKLY_RETRAIN_WEEKDAY,
    get_entry_value,
)
from .models import (
    ML_AVAILABLE,
    apply_grid_sign,
    features_from_state,
    get_training_sample_count,
    grid_import_watts,
    load_last_decision,
    load_metrics,
    load_model,
    log_decision_to_csv,
    parse_iso_datetime,
    rule_teacher,
    run_retrain,
    save_last_decision,
    target_soc_from_forecast,
    utc_now_iso,
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
                seconds=get_entry_value(
                    config_entry, CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL
                )
            ),
        )
        self.config_entry = config_entry
        self.hass_config_path = hass.config.path
        self.last_decision = None
        self.metrics: dict = {}
        self.training_samples = 0
        self._model = None
        self._model_loaded = False
        self._csv_lock = asyncio.Lock()
        self._retraining = False
        self._unsubs: list = []
        self._grid_high_since: datetime | None = None
        self._surplus_missed_since: datetime | None = None
        self._last_auto_retrain: datetime | None = None
        self.apply_config(config_entry)

    def apply_config(self, config_entry) -> None:
        """Apply config entry data/options to the running coordinator."""
        self.config_entry = config_entry
        self.miner_switch = config_entry.data[CONF_MINER_SWITCH]
        self.miner_power_number = config_entry.data[CONF_MINER_POWER_NUMBER]
        self.auto_control = get_entry_value(config_entry, CONF_AUTO_CONTROL, True)
        self.battery_capacity_kwh = get_entry_value(
            config_entry, CONF_BATTERY_CAPACITY_KWH, DEFAULT_BATTERY_CAPACITY_KWH
        )
        self.min_samples_for_model = get_entry_value(
            config_entry, CONF_MIN_SAMPLES_FOR_MODEL, DEFAULT_MIN_SAMPLES_FOR_MODEL
        )
        self.retrain_interval = get_entry_value(
            config_entry, CONF_RETRAIN_INTERVAL, DEFAULT_RETRAIN_INTERVAL
        )
        scan_seconds = get_entry_value(
            config_entry, CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL
        )
        self.update_interval = timedelta(seconds=scan_seconds)
        self.grid_invert = get_entry_value(config_entry, CONF_GRID_INVERT, False)

        self.entity_map = {
            key: config_entry.data.get(key) for key in SENSOR_STATE_KEYS
        }
        self.entity_map["current_miner_power"] = config_entry.data.get(
            CONF_MINER_POWER_NUMBER
        )
        self.entity_map["miner_is_on"] = config_entry.data.get(CONF_MINER_SWITCH)

    async def async_setup_listeners(self) -> None:
        """Start scheduled and event-driven retrain watchers."""
        # Load persisted state so sensors are not empty until first tick
        self.metrics = await self.hass.async_add_executor_job(
            load_metrics, self.hass_config_path
        )
        self.training_samples = await self.hass.async_add_executor_job(
            get_training_sample_count, self.hass_config_path
        )
        persisted_decision = await self.hass.async_add_executor_job(
            load_last_decision, self.hass_config_path
        )
        if persisted_decision:
            self.last_decision = persisted_decision

        last = parse_iso_datetime(self.metrics.get("last_retrain"))
        if last:
            self._last_auto_retrain = last

        self._unsubs.append(
            async_track_time_change(
                self.hass,
                self._handle_weekly_retrain,
                hour=WEEKLY_RETRAIN_HOUR,
                minute=0,
                second=0,
            )
        )
        self._unsubs.append(
            async_track_time_interval(
                self.hass, self._check_retrain_triggers, timedelta(minutes=1)
            )
        )
        # Watchdog: check SoC / grid safety every minute, independent of
        # the 20-minute decision cycle.
        self._unsubs.append(
            async_track_time_interval(
                self.hass, self._watchdog_check, timedelta(minutes=1)
            )
        )

    def async_unload_listeners(self) -> None:
        """Cancel retrain watchers."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()

    async def _handle_weekly_retrain(self, now: datetime) -> None:
        """Sunday 03:00 retrain."""
        if now.weekday() != WEEKLY_RETRAIN_WEEKDAY:
            return
        _LOGGER.info("Weekly retrain triggered")
        await self.async_retrain(force=False)

    async def _check_retrain_triggers(self, now: datetime) -> None:
        """Event-driven and interval-based retraining."""
        state = await self._read_entity_states()
        grid_power_raw = state.get("grid_power") or 0
        grid_import = grid_import_watts(apply_grid_sign(grid_power_raw, self.grid_invert))
        soc = float(state.get("battery_soc") or 0)
        surplus = float(state.get("solar_surplus_power") or 0)
        miner_on = state.get("miner_is_on") == "on"

        reason = None
        if grid_import > GRID_IMPORT_RETRAIN_W:
            if self._grid_high_since is None:
                self._grid_high_since = now
            elif now - self._grid_high_since >= timedelta(
                minutes=GRID_IMPORT_RETRAIN_MINUTES
            ):
                reason = "grid_import"
        else:
            self._grid_high_since = None

        if soc < BATTERY_SOC_MIN and miner_on:
            reason = "soc_safety"

        if surplus > MISSED_SURPLUS_W and not miner_on:
            if self._surplus_missed_since is None:
                self._surplus_missed_since = now
            elif now - self._surplus_missed_since >= timedelta(
                minutes=MISSED_SURPLUS_MINUTES
            ):
                reason = "missed_surplus"
        else:
            self._surplus_missed_since = None

        if reason is None:
            last = parse_iso_datetime(self.metrics.get("last_retrain"))
            if last is None or (now - last).total_seconds() >= self.retrain_interval:
                if self.training_samples:
                    reason = "interval"

        if reason is None:
            return

        if (
            reason != "interval"
            and self._last_auto_retrain
            and (now - self._last_auto_retrain).total_seconds()
            < RETRAIN_EVENT_COOLDOWN_SECONDS
        ):
            return

        _LOGGER.info("Auto-retrain triggered: %s", reason)
        await self.async_retrain(force=reason != "interval")
        self._last_auto_retrain = now
        self._grid_high_since = None
        self._surplus_missed_since = None

    async def _watchdog_check(self, now: datetime) -> None:
        """Separate watchdog: kill miner if SoC drops or grid import spikes,
        independent of the 20-minute decision cycle."""
        if not self.auto_control:
            return

        state = await self._read_entity_states()
        soc = float(state.get("battery_soc") or 0)
        grid_power_raw = state.get("grid_power") or 0
        grid_import = grid_import_watts(apply_grid_sign(grid_power_raw, self.grid_invert))
        miner_on = state.get("miner_is_on") == "on"

        if not miner_on:
            return

        should_kill = False
        reason = ""

        if soc < BATTERY_SOC_MIN:
            should_kill = True
            reason = f"Watchdog: SoC {soc:.0f}% below minimum {BATTERY_SOC_MIN}%"
        elif grid_import > GRID_IMPORT_RETRAIN_W:
            should_kill = True
            reason = f"Watchdog: grid import {grid_import:.0f}W exceeds {GRID_IMPORT_RETRAIN_W}W"

        if should_kill:
            _LOGGER.warning("Watchdog killing miner: %s", reason)
            try:
                await self.hass.services.async_call(
                    "switch", "turn_off",
                    {"entity_id": self.miner_switch},
                    blocking=True,
                )
                # Record the watchdog event
                if self.last_decision:
                    self.last_decision["mode"] = "safety_shutdown"
                    self.last_decision["reason"] = reason
                else:
                    self.last_decision = {
                        "miner_active": "off",
                        "miner_power": MINER_POWER_MIN,
                        "mode": "safety_shutdown",
                        "reason": reason,
                        "source": "watchdog",
                        "timestamp": utc_now_iso(),
                    }
                self.async_update_listeners()
            except Exception as err:
                _LOGGER.error("Watchdog failed to turn off miner: %s", err)

    async def _async_update_data(self) -> dict:
        """Run ML decision cycle. Called every scan_interval."""
        try:
            state = await self._read_entity_states()

            decision = await self.hass.async_add_executor_job(
                self._run_decision, state
            )

            async with self._csv_lock:
                await self.hass.async_add_executor_job(
                    log_decision_to_csv, self.hass_config_path, state, decision
                )

            if self.auto_control:
                await self._apply_decision(decision)

            self.last_decision = decision
            self.training_samples = decision.get("training_samples", self.training_samples)

            # Persist so sensors are not empty until next tick
            await self.hass.async_add_executor_job(
                save_last_decision, self.hass_config_path, decision
            )

            return decision

        except Exception as err:
            _LOGGER.error("ML decision cycle failed: %s", err)
            raise UpdateFailed(f"ML decision cycle failed: {err}") from err

    async def _read_entity_states(self) -> dict:
        """Build state dict from configured HA entity IDs."""
        state = {}
        for key, entity_id in self.entity_map.items():
            if not entity_id:
                if key == "miner_is_on":
                    state[key] = "off"
                elif key == "hours_until_sunrise":
                    state[key] = None
                else:
                    state[key] = 0.0
                continue

            ent = self.hass.states.get(entity_id)
            if ent and ent.state not in ("unavailable", "unknown"):
                try:
                    if key == "miner_is_on":
                        state[key] = ent.state
                    else:
                        state[key] = float(ent.state)
                except (ValueError, TypeError):
                    state[key] = "off" if key == "miner_is_on" else 0.0
            else:
                if key == "miner_is_on":
                    state[key] = "off"
                elif key == "hours_until_sunrise":
                    state[key] = None
                else:
                    state[key] = 0.0
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
        samples = get_training_sample_count(self.hass_config_path)
        forecast = float(state.get("forecast_tomorrow") or 0)

        if not self._model_loaded:
            self._model, _ = load_model(self.hass_config_path)
            self._model_loaded = True

        # Shadow period: only allow model after MODEL_SHADOW_DAYS of data
        # and the model must have been trained at least once.
        now = utc_now()
        retrain_time = parse_iso_datetime(self.metrics.get("last_retrain"))
        shadow_ok = (
            retrain_time is not None
            and (now - retrain_time).total_seconds() >= MODEL_SHADOW_DAYS * 86400
        )

        use_model = (
            ML_AVAILABLE
            and self._model is not None
            and samples >= self.min_samples_for_model
            and shadow_ok
        )

        # Determine previous miner state for teacher hysteresis
        prev_miner_active = state.get("miner_is_on", "off")
        if prev_miner_active not in ("on", "off"):
            prev_miner_active = "off"

        if use_model:
            import numpy as np

            predicted_power = int(self._model.predict(np.array([features]))[0])
            mode = (
                "day_solar"
                if float(state.get("solar_power_total") or 0) > 100
                else "night_drain"
            )
            decision = {
                "miner_active": "on" if predicted_power >= MINER_POWER_MIN else "off",
                "miner_power": predicted_power,
                "target_soc_by_sunrise": 30,
                "mode": mode,
                "reason": "ML residual prediction",
            }
            source = "ml_model"
        else:
            decision = rule_teacher(
                features,
                self.battery_capacity_kwh,
                prev_miner_active=prev_miner_active,
            )
            source = "rule_teacher"

        decision["_soc"] = state.get("battery_soc", 50)
        decision["_grid_power"] = apply_grid_sign(
            _as_float(state.get("grid_power"), 0), self.grid_invert
        )
        decision = validate_decision(decision)
        decision["source"] = source
        decision["training_samples"] = samples
        decision["timestamp"] = utc_now_iso()

        return decision

    async def _apply_decision(self, decision: dict) -> None:
        """Apply miner switch and power level via HA services."""
        try:
            service = "turn_on" if decision["miner_active"] == "on" else "turn_off"
            await self.hass.services.async_call(
                "switch", service, {"entity_id": self.miner_switch}, blocking=True
            )

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
        if self._retraining:
            _LOGGER.info("Retrain already in progress")
            return self.metrics

        self._retraining = True
        try:
            live_state = await self._read_entity_states()
            async with self._csv_lock:
                metrics = await self.hass.async_add_executor_job(
                    run_retrain,
                    self.hass_config_path,
                    self.min_samples_for_model,
                    force,
                    live_state,
                )
            self.metrics = metrics
            self.training_samples = metrics.get("total_samples", self.training_samples)
            self.invalidate_model_cache()
            self.async_update_listeners()
            return metrics
        finally:
            self._retraining = False

    async def async_request_decision(self) -> dict:
        """Trigger an immediate decision cycle."""
        await self.async_request_refresh()
        return self.last_decision

    def get_status(self) -> dict:
        """Return current status for service response."""
        return {
            "auto_control": self.auto_control,
            "last_decision": self.last_decision,
            "training_samples": self.training_samples,
            "model_loaded": self._model is not None,
            "metrics": self.metrics,
        }


def _as_float(value, default: float = 0.0) -> float:
    """Coerce value to float (local copy to avoid circular import)."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
