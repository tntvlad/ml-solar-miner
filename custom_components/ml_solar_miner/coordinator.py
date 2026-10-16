"""DataUpdateCoordinator for ML Solar Miner."""
import asyncio
import logging
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change, async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    BATTERY_SOC_MIN,
    BATTERY_EFFICIENCY_MAX,
    BATTERY_EFFICIENCY_MIN,
    CONF_AUTO_CONTROL,
    CONF_BATTERY_CAPACITY_ENTITY,
    CONF_BATTERY_CAPACITY_KWH,
    CONF_BATTERY_EFFICIENCY,
    CONF_CAPACITY_AUTOTUNE,
    CONF_EMERGENCY_LOCK_ENTITY,
    CONF_GRID_INVERT,
    CONF_MINER_POWER_NUMBER,
    CONF_MINER_SWITCH,
    CONF_MIN_SAMPLES_FOR_MODEL,
    CONF_MODEL_TYPE,
    CONF_RETRAIN_INTERVAL,
    CONF_SCAN_INTERVAL_OPTION,
    CONF_TARGET_SOC_ENTITY,
    CONF_TARGET_SOC_MORNING,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_BATTERY_EFFICIENCY,
    DEFAULT_CAPACITY_AUTOTUNE,
    DEFAULT_EMERGENCY_LOCK_ENTITY,
    DEFAULT_MIN_SAMPLES_FOR_MODEL,
    DEFAULT_MODEL_TYPE,
    DEFAULT_RETRAIN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_TARGET_SOC_MORNING,
    DOMAIN,
    ENGINE_AUTO,
    ENGINE_ML,
    ENGINE_RULE,
    GRID_IMPORT_RETRAIN_MINUTES,
    GRID_IMPORT_RETRAIN_W,
    LEARNED_CAPACITY_FILENAME,
    MINER_POWER_MIN,
    MISSED_SURPLUS_MINUTES,
    MISSED_SURPLUS_W,
    ML_DATA_DIR,
    ML_FEATURE_NAMES,
    MODEL_SHADOW_DAYS,
    RETRAIN_EVENT_COOLDOWN_SECONDS,
    SENSOR_STATE_KEYS,
    WEEKLY_RETRAIN_HOUR,
    WEEKLY_RETRAIN_WEEKDAY,
    get_entry_value,
)
from .capacity_estimator import (
    CapacityEstimator,
    load_capacity_state,
    save_capacity_state,
)
from . import rule_engine
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
    utc_now,
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
        self._last_target_soc: float | None = None
        self._last_capacity_kwh: float | None = None
        self._missing_logged: set[str] = set()
        # Battery capacity auto-tune (see capacity_estimator.py)
        self.capacity_autotune = DEFAULT_CAPACITY_AUTOTUNE
        self.battery_efficiency = DEFAULT_BATTERY_EFFICIENCY / 100.0
        self.learned_capacity_kwh: float | None = None
        self.capacity_estimator: CapacityEstimator | None = None
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
        self.model_type = get_entry_value(
            config_entry, CONF_MODEL_TYPE, DEFAULT_MODEL_TYPE
        )
        self.target_soc_morning = get_entry_value(
            config_entry, CONF_TARGET_SOC_MORNING, DEFAULT_TARGET_SOC_MORNING
        )
        self.target_soc_entity = get_entry_value(
            config_entry, CONF_TARGET_SOC_ENTITY, None
        )
        self.battery_capacity_entity = get_entry_value(
            config_entry, CONF_BATTERY_CAPACITY_ENTITY, None
        )
        self.emergency_lock_entity = get_entry_value(
            config_entry,
            CONF_EMERGENCY_LOCK_ENTITY,
            DEFAULT_EMERGENCY_LOCK_ENTITY,
        )

        # Capacity auto-tune: the estimator is (re)created when the feature is
        # switched on so a stale in-progress window is not carried over.
        self.capacity_autotune = bool(
            get_entry_value(
                config_entry, CONF_CAPACITY_AUTOTUNE, DEFAULT_CAPACITY_AUTOTUNE
            )
        )
        self.battery_efficiency = _clamp_efficiency(
            get_entry_value(
                config_entry, CONF_BATTERY_EFFICIENCY, DEFAULT_BATTERY_EFFICIENCY
            )
        )
        if self.capacity_autotune:
            if self.capacity_estimator is None:
                self.capacity_estimator = CapacityEstimator(
                    efficiency=self.battery_efficiency
                )
            else:
                self.capacity_estimator.efficiency = self.battery_efficiency
        else:
            self.capacity_estimator = None

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

        persisted_capacity = await self.hass.async_add_executor_job(
            load_capacity_state, self._learned_capacity_path()
        )
        if persisted_capacity:
            value = persisted_capacity.get("capacity_kwh")
            if value is not None:
                self.learned_capacity_kwh = float(value)
            if self.capacity_estimator is not None:
                self.capacity_estimator.restore_windows(persisted_capacity)

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
        missing = state.get("_missing", set())
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

        if soc < BATTERY_SOC_MIN and miner_on and "battery_soc" not in missing:
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
        missing = state.get("_missing", set())
        soc = float(state.get("battery_soc") or 0)
        grid_power_raw = state.get("grid_power") or 0
        grid_import = grid_import_watts(apply_grid_sign(grid_power_raw, self.grid_invert))
        miner_on = state.get("miner_is_on") == "on"

        if not miner_on:
            return

        should_kill = False
        reason = ""

        if "battery_soc" in missing:
            should_kill = True
            reason = "Watchdog: battery SoC sensor unavailable"
        elif soc < BATTERY_SOC_MIN:
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
                # Record the watchdog event, refreshing timestamp/source so the
                # last-decision sensor reflects *this* safety trip.
                if self.last_decision:
                    self.last_decision.update(
                        {
                            "miner_active": "off",
                            "miner_power": MINER_POWER_MIN,
                            "mode": "safety_shutdown",
                            "reason": reason,
                            "source": "watchdog",
                            "timestamp": utc_now_iso(),
                        }
                    )
                else:
                    self.last_decision = {
                        "miner_active": "off",
                        "miner_power": MINER_POWER_MIN,
                        "mode": "safety_shutdown",
                        "reason": reason,
                        "source": "watchdog",
                        "timestamp": utc_now_iso(),
                    }
                await self.hass.async_add_executor_job(
                    save_last_decision, self.hass_config_path, self.last_decision
                )
                self.async_update_listeners()
            except Exception as err:
                _LOGGER.error("Watchdog failed to turn off miner: %s", err)

    async def _async_update_data(self) -> dict:
        """Run ML decision cycle. Called every scan_interval."""
        try:
            state = await self._read_entity_states()

            # Feed the power balance into the capacity estimator before the
            # early-exit checks: the estimator only needs SoC + balance, and it
            # decides for itself whether the sample is usable.
            await self._update_capacity_estimate(state)

            missing = state.get("_missing", set())
            if "battery_soc" in missing:
                # Without a trustworthy SoC we cannot make a safe decision, and
                # a fake "SoC 0%" row would poison the training data. Fail safe
                # (hold the miner off), record why, and skip this cycle.
                await self._handle_missing_critical_sensor(state)
                return self.last_decision or {}

            if self._emergency_lock_active():
                # User (or the emergency automation) locked mining: hold off and
                # skip this cycle, exactly like the legacy engine's condition.
                await self._handle_emergency_lock()
                return self.last_decision or {}

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
        """Build state dict from configured HA entity IDs.

        Unavailable inputs fall back to a safe default, but are also collected
        in ``state["_missing"]`` so the decision cycle can refuse to act on
        incomplete data instead of fabricating a decision.
        """
        state: dict = {}
        missing: set[str] = set()

        for key, entity_id in self.entity_map.items():
            if not entity_id:
                state[key] = self._default_state_value(key)
                continue

            ent = self.hass.states.get(entity_id)
            if ent and ent.state not in ("unavailable", "unknown"):
                try:
                    if key == "miner_is_on":
                        state[key] = ent.state
                    else:
                        state[key] = float(ent.state)
                except (ValueError, TypeError):
                    state[key] = self._default_state_value(key)
                    missing.add(key)
            else:
                state[key] = self._default_state_value(key)
                missing.add(key)

        self._log_missing_transitions(missing)

        # Optional dashboard helpers used by the rule engine: the morning target
        # SoC and the battery capacity can be read live from input_number
        # entities instead of the static options.
        state["_target_soc_entity"] = self._read_entity_value(self.target_soc_entity)
        state["_capacity_entity"] = self._read_entity_value(self.battery_capacity_entity)
        state["_missing"] = missing

        return state

    @staticmethod
    def _default_state_value(key: str):
        """Safe default for an input whose entity is missing or unavailable."""
        if key == "miner_is_on":
            return "off"
        if key == "hours_until_sunrise":
            return None
        return 0.0

    def _log_missing_transitions(self, missing: set[str]) -> None:
        """Log unavailable inputs only when the set changes (no per-tick spam)."""
        appeared = missing - self._missing_logged
        recovered = self._missing_logged - missing
        for key in sorted(appeared):
            _LOGGER.warning(
                "Input '%s' (%s) is unavailable — using default until it recovers",
                key,
                self.entity_map.get(key) or "not configured",
            )
        for key in sorted(recovered):
            _LOGGER.info("Input '%s' recovered", key)
        self._missing_logged = set(missing)

    def _emergency_lock_active(self) -> bool:
        """Return True while the configured emergency lock helper is on."""
        entity_id = self.emergency_lock_entity
        return bool(entity_id) and self.hass.states.is_state(entity_id, "on")

    async def _hold_miner_off(self, mode: str, reason: str) -> None:
        """Turn the miner off (if controlled) and record why.

        Deliberately does not append a training row: a held-off cycle is not a
        decision the model should learn from.
        """
        if self.auto_control and self.hass.states.is_state(self.miner_switch, "on"):
            try:
                await self.hass.services.async_call(
                    "switch",
                    "turn_off",
                    {"entity_id": self.miner_switch},
                    blocking=True,
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.error("Failed to hold miner off: %s", err)

        update = {
            "miner_active": "off",
            "miner_power": MINER_POWER_MIN,
            "mode": mode,
            "reason": reason,
            "source": "safety",
            "timestamp": utc_now_iso(),
        }
        if self.last_decision:
            self.last_decision.update(update)
        else:
            self.last_decision = {
                "target_soc_by_sunrise": int(self.target_soc_morning),
                **update,
            }
        await self.hass.async_add_executor_job(
            save_last_decision, self.hass_config_path, self.last_decision
        )
        self.async_update_listeners()

    async def _handle_missing_critical_sensor(self, state: dict) -> None:
        """Fail safe when a critical sensor is unavailable.

        The engines have no trustworthy SoC here, so rather than decide from
        zeros (which would also write a bogus training row) we hold the miner
        off and record the reason. Mirrors the legacy safety watchdog.
        """
        await self._hold_miner_off(
            "sensor_unavailable",
            "Battery SoC sensor unavailable — miner held off",
        )

    async def _handle_emergency_lock(self) -> None:
        """Hold the miner off while the emergency lock helper is on.

        Mirrors the legacy decision-engine condition on
        ``input_boolean.emergency_battery_lock``.
        """
        await self._hold_miner_off(
            "emergency_lock",
            f"Emergency battery lock ({self.emergency_lock_entity}) is on — miner held off",
        )

    def _read_entity_value(self, entity_id: str | None) -> float | None:
        """Read a numeric value from an optional helper entity."""
        if not entity_id:
            return None
        ent = self.hass.states.get(entity_id)
        if ent is None or ent.state in ("unavailable", "unknown"):
            return None
        try:
            return float(ent.state)
        except (TypeError, ValueError):
            return None

    def _resolve_capacity_kwh(self, state: dict) -> float:
        """Battery capacity: learned value, live helper, last known, else option."""
        if self.capacity_autotune and self.learned_capacity_kwh:
            return float(self.learned_capacity_kwh)
        value = state.get("_capacity_entity")
        if value is not None and value > 0:
            self._last_capacity_kwh = float(value)
            return float(value)
        if self._last_capacity_kwh is not None:
            return float(self._last_capacity_kwh)
        return float(self.battery_capacity_kwh)

    def _learned_capacity_path(self) -> str:
        """Absolute path of the persisted learned-capacity state file."""
        return self.hass_config_path(ML_DATA_DIR, LEARNED_CAPACITY_FILENAME)

    @property
    def capacity_status(self) -> str:
        """Human-readable state of the capacity auto-tune feature."""
        if not self.capacity_autotune or self.capacity_estimator is None:
            return "disabled"
        if self.learned_capacity_kwh is None or not self.capacity_estimator.ready:
            return "learning"
        return "applied"

    def _balance_power_w(self, state: dict) -> float:
        """Whole-system battery power estimate (positive = charging).

        Uses solar + grid import - load rather than the measured battery
        current, because the parallel DIY banks are not individually metered.
        """
        solar = _as_float(state.get("solar_power_total"), 0.0)
        load = abs(_as_float(state.get("total_load_power"), 0.0))
        grid = apply_grid_sign(
            _as_float(state.get("grid_power"), 0.0), self.grid_invert
        )
        return solar + grid - load

    async def _update_capacity_estimate(self, state: dict) -> None:
        """Feed one sample into the capacity estimator and apply any result."""
        if not self.capacity_autotune or self.capacity_estimator is None:
            return

        missing = state.get("_missing", set())
        required = (
            "battery_soc",
            "solar_power_total",
            "total_load_power",
            "grid_power",
        )
        if any(not self.entity_map.get(key) for key in required):
            return
        if any(key in missing for key in required):
            return
        soc = state.get("battery_soc")
        if soc is None:
            return

        window = self.capacity_estimator.add_sample(
            utc_now(), float(soc), self._balance_power_w(state)
        )
        if window is None:
            return

        baseline = self._resolve_capacity_kwh(state)
        candidate = self.capacity_estimator.accept(window, baseline)

        # Persist even a rejected-window append so a restart keeps the count.
        await self._persist_capacity()

        if candidate is None:
            return

        self.learned_capacity_kwh = candidate
        _LOGGER.info(
            "Battery capacity auto-tuned to %.1f kWh from %d windows "
            "(latest %s %.1f kWh %.0f->%.0f%%, %.0f min)",
            candidate,
            self.capacity_estimator.window_count,
            window.direction,
            window.estimate_kwh,
            window.soc_start,
            window.soc_end,
            window.duration_min,
        )
        await self._mirror_capacity_to_helper(candidate)

    async def _persist_capacity(self) -> None:
        """Persist the learned capacity and window history."""
        if self.capacity_estimator is None:
            return
        payload = {
            **self.capacity_estimator.to_dict(),
            "capacity_kwh": self.learned_capacity_kwh,
            "updated": utc_now_iso(),
        }
        await self.hass.async_add_executor_job(
            save_capacity_state, self._learned_capacity_path(), payload
        )

    async def _mirror_capacity_to_helper(self, value: float) -> None:
        """Write the learned capacity back to the configured helper entity."""
        entity_id = self.battery_capacity_entity
        if not entity_id:
            return
        if entity_id.startswith("input_number."):
            domain = "input_number"
        elif entity_id.startswith("number."):
            domain = "number"
        else:
            return
        try:
            await self.hass.services.async_call(
                domain,
                "set_value",
                {"entity_id": entity_id, "value": round(float(value), 1)},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Could not write learned capacity to %s: %s", entity_id, err
            )

    def _resolve_target_soc(self, state: dict) -> float:
        """Morning target SoC: live helper, else last known, else the option.

        Mirrors the legacy script's ``last_target.json`` behaviour: an
        unavailable helper must not silently reset the target to the static
        default while the user's dashboard value is momentarily unreadable.
        """
        value = state.get("_target_soc_entity")
        if value is not None:
            self._last_target_soc = float(value)
            return float(value)
        if self._last_target_soc is not None:
            return float(self._last_target_soc)
        return float(self.target_soc_morning)

    def _run_decision(self, state: dict) -> dict:
        """Synchronous decision — runs in executor thread."""
        samples = get_training_sample_count(self.hass_config_path)

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

        model_ready = (
            ML_AVAILABLE
            and self._model is not None
            and samples >= self.min_samples_for_model
            and shadow_ok
        )

        target_soc = self._resolve_target_soc(state)

        if self._use_rule_engine(model_ready):
            decision = rule_engine.decide(
                state,
                capacity_kwh=self._resolve_capacity_kwh(state),
                target_soc_morning=target_soc,
                grid_invert=self.grid_invert,
            )
            source = "rule_teacher"
        else:
            features = features_from_state(state)

            # Determine previous miner state for teacher hysteresis
            prev_miner_active = state.get("miner_is_on", "off")
            if prev_miner_active not in ("on", "off"):
                prev_miner_active = "off"

            if model_ready:
                residual = self._model.predict_one(
                    dict(zip(ML_FEATURE_NAMES, features))
                )
                # The model is trained on the residual versus the rule teacher,
                # so the teacher power must be added back for the final power.
                teacher = rule_teacher(
                    features,
                    self._resolve_capacity_kwh(state),
                    prev_miner_active=prev_miner_active,
                )
                predicted_power = int(
                    float(teacher["miner_power"]) + float(residual or 0)
                )
                mode = (
                    "day_solar"
                    if _as_float(state.get("solar_power_total"), 0) > 100
                    else "night_drain"
                )
                decision = {
                    "miner_active": (
                        "on" if predicted_power >= MINER_POWER_MIN else "off"
                    ),
                    "miner_power": predicted_power,
                    "target_soc_by_sunrise": int(target_soc),
                    "mode": mode,
                    "reason": "ML residual prediction",
                }
                source = "ml_model"
            else:
                decision = rule_teacher(
                    features,
                    self._resolve_capacity_kwh(state),
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

    def _use_rule_engine(self, model_ready: bool) -> bool:
        """Return True when the script-identical rule engine should decide.

        - rule: always use the rule engine.
        - ml / auto: use the rule engine until the ML model is ready.
        """
        if self.model_type == ENGINE_RULE:
            return True
        return not model_ready

    def active_engine(self, model_ready: bool | None = None) -> str:
        """Return the engine that would be used for the next decision."""
        if model_ready is None:
            model_ready = ML_AVAILABLE and self._model is not None
        return ENGINE_RULE if self._use_rule_engine(model_ready) else ENGINE_ML

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
            "model_type": self.model_type,
            "active_engine": self.active_engine(),
            "last_decision": self.last_decision,
            "training_samples": self.training_samples,
            "model_loaded": self._model is not None,
            "metrics": self.metrics,
            "capacity_autotune": self.capacity_autotune,
            "capacity_status": self.capacity_status,
            "battery_efficiency": self.battery_efficiency,
            "learned_capacity_kwh": self.learned_capacity_kwh,
            "capacity_windows": (
                self.capacity_estimator.window_count
                if self.capacity_estimator is not None
                else 0
            ),
        }


def _clamp_efficiency(value) -> float:
    """Return a one-way efficiency fraction from a percent option."""
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return DEFAULT_BATTERY_EFFICIENCY / 100.0
    percent = min(max(percent, BATTERY_EFFICIENCY_MIN), BATTERY_EFFICIENCY_MAX)
    return percent / 100.0


def _as_float(value, default: float = 0.0) -> float:
    """Coerce value to float (local copy to avoid circular import)."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
