"""Config flow for ML Solar Miner."""
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import PERCENTAGE
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    BATTERY_SOC_MIN,
    CONF_AUTO_CONTROL,
    CONF_BATTERY_CAPACITY_ENTITY,
    CONF_BATTERY_CAPACITY_KWH,
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
    DEFAULT_MIN_SAMPLES_FOR_MODEL,
    DEFAULT_MODEL_TYPE,
    DEFAULT_RETRAIN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_TARGET_SOC_MORNING,
    DOMAIN,
    ENGINE_AUTO,
    ENGINE_ML,
    ENGINE_RULE,
    SENSOR_STATE_KEYS,
    TARGET_SOC_MAX,
    get_entry_value,
)


def _entity_selector(domain):
    """Create an entity selector filtered by one or more domains."""
    return selector.EntitySelector(
        selector.EntityFilterSelectorConfig(domain=domain)
    )


def _engine_selector() -> selector.SelectSelector:
    """Selector for the decision engine."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=[
                selector.SelectOptionDict(
                    value=ENGINE_RULE,
                    label="Rule-based (identical to the standalone script)",
                ),
                selector.SelectOptionDict(value=ENGINE_ML, label="Machine Learning"),
                selector.SelectOptionDict(
                    value=ENGINE_AUTO,
                    label="Auto - ML when ready, otherwise rules",
                ),
            ],
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _optional_entity(key: str, value: Any) -> vol.Optional:
    """An optional entity field that pre-fills when a value is present."""
    if value:
        return vol.Optional(key, description={"suggested_value": value})
    return vol.Optional(key)


STEP_1_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MINER_SWITCH): _entity_selector("switch"),
        vol.Required(CONF_MINER_POWER_NUMBER): _entity_selector("number"),
    }
)

STEP_2_SCHEMA = vol.Schema(
    {vol.Optional(key): _entity_selector("sensor") for key in SENSOR_STATE_KEYS}
)


def _options_schema(current: dict[str, Any]) -> vol.Schema:
    """Build the settings schema, pre-filled with ``current`` values."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_MODEL_TYPE,
                default=current.get(CONF_MODEL_TYPE, DEFAULT_MODEL_TYPE),
            ): _engine_selector(),
            vol.Optional(
                CONF_TARGET_SOC_MORNING,
                default=current.get(
                    CONF_TARGET_SOC_MORNING, DEFAULT_TARGET_SOC_MORNING
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=BATTERY_SOC_MIN,
                    max=TARGET_SOC_MAX,
                    step=1,
                    mode=selector.NumberSelectorMode.SLIDER,
                    unit_of_measurement=PERCENTAGE,
                )
            ),
            _optional_entity(CONF_TARGET_SOC_ENTITY, current.get(CONF_TARGET_SOC_ENTITY)): _entity_selector(
                ["input_number", "number"]
            ),
            _optional_entity(
                CONF_BATTERY_CAPACITY_ENTITY,
                current.get(CONF_BATTERY_CAPACITY_ENTITY),
            ): _entity_selector(["input_number", "number"]),
            vol.Optional(
                CONF_SCAN_INTERVAL_OPTION,
                default=current.get(
                    CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL // 60
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=5, max=1440)),
            vol.Optional(
                CONF_AUTO_CONTROL, default=current.get(CONF_AUTO_CONTROL, True)
            ): bool,
            vol.Optional(
                CONF_BATTERY_CAPACITY_KWH,
                default=current.get(
                    CONF_BATTERY_CAPACITY_KWH, DEFAULT_BATTERY_CAPACITY_KWH
                ),
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=1000.0)),
            vol.Optional(
                CONF_MIN_SAMPLES_FOR_MODEL,
                default=current.get(
                    CONF_MIN_SAMPLES_FOR_MODEL, DEFAULT_MIN_SAMPLES_FOR_MODEL
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
            vol.Optional(
                CONF_RETRAIN_INTERVAL,
                default=current.get(
                    CONF_RETRAIN_INTERVAL, DEFAULT_RETRAIN_INTERVAL // 3600
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=720)),
            vol.Optional(
                CONF_GRID_INVERT, default=current.get(CONF_GRID_INVERT, False)
            ): bool,
        }
    )


STEP_3_SCHEMA = _options_schema({})


def _split_options(user_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert minutes/hours form values into data + options dicts."""
    scan_seconds = user_input.pop(CONF_SCAN_INTERVAL_OPTION) * 60
    retrain_seconds = user_input.pop(CONF_RETRAIN_INTERVAL) * 3600
    options = {
        CONF_SCAN_INTERVAL_OPTION: scan_seconds,
        CONF_RETRAIN_INTERVAL: retrain_seconds,
        CONF_AUTO_CONTROL: user_input.pop(CONF_AUTO_CONTROL),
        CONF_BATTERY_CAPACITY_KWH: user_input.pop(CONF_BATTERY_CAPACITY_KWH),
        CONF_MIN_SAMPLES_FOR_MODEL: user_input.pop(CONF_MIN_SAMPLES_FOR_MODEL),
        CONF_GRID_INVERT: user_input.pop(CONF_GRID_INVERT, False),
        CONF_MODEL_TYPE: user_input.pop(CONF_MODEL_TYPE, DEFAULT_MODEL_TYPE),
        CONF_TARGET_SOC_MORNING: int(
            user_input.pop(CONF_TARGET_SOC_MORNING, DEFAULT_TARGET_SOC_MORNING)
        ),
        CONF_TARGET_SOC_ENTITY: user_input.pop(CONF_TARGET_SOC_ENTITY, None) or None,
        CONF_BATTERY_CAPACITY_ENTITY: user_input.pop(
            CONF_BATTERY_CAPACITY_ENTITY, None
        )
        or None,
    }
    return user_input, options


class MLSolarMinerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ML Solar Miner."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self.data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step — miner control entities."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        errors: dict[str, str] = {}

        if user_input is not None:
            if not self.hass.states.get(user_input[CONF_MINER_SWITCH]):
                errors[CONF_MINER_SWITCH] = "entity_not_found"
            elif not self.hass.states.get(user_input[CONF_MINER_POWER_NUMBER]):
                errors[CONF_MINER_POWER_NUMBER] = "entity_not_found"
            else:
                self.data.update(user_input)
                return await self.async_step_sensors()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_1_SCHEMA,
            errors=errors,
            description_placeholders={"name": "ML Solar Miner"},
        )

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the sensor mapping step."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_options()

        return self.async_show_form(
            step_id="sensors",
            data_schema=STEP_2_SCHEMA,
        )

    async def async_step_options(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the options step."""
        if user_input is not None:
            data, options = _split_options(user_input)
            self.data.update(data)
            return self.async_create_entry(
                title="ML Solar Miner", data=self.data, options=options
            )

        return self.async_show_form(
            step_id="options",
            data_schema=STEP_3_SCHEMA,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "MLSolarMinerOptionsFlow":
        """Get the options flow for this handler."""
        return MLSolarMinerOptionsFlow()


class MLSolarMinerOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for ML Solar Miner."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            _, options = _split_options(user_input)
            return self.async_create_entry(title="", data=options)

        entry = self.config_entry
        current = {
            CONF_MODEL_TYPE: get_entry_value(
                entry, CONF_MODEL_TYPE, DEFAULT_MODEL_TYPE
            ),
            CONF_TARGET_SOC_MORNING: get_entry_value(
                entry, CONF_TARGET_SOC_MORNING, DEFAULT_TARGET_SOC_MORNING
            ),
            CONF_TARGET_SOC_ENTITY: get_entry_value(
                entry, CONF_TARGET_SOC_ENTITY, None
            ),
            CONF_BATTERY_CAPACITY_ENTITY: get_entry_value(
                entry, CONF_BATTERY_CAPACITY_ENTITY, None
            ),
            CONF_SCAN_INTERVAL_OPTION: get_entry_value(
                entry, CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL
            )
            // 60,
            CONF_AUTO_CONTROL: get_entry_value(entry, CONF_AUTO_CONTROL, True),
            CONF_BATTERY_CAPACITY_KWH: get_entry_value(
                entry, CONF_BATTERY_CAPACITY_KWH, DEFAULT_BATTERY_CAPACITY_KWH
            ),
            CONF_MIN_SAMPLES_FOR_MODEL: get_entry_value(
                entry, CONF_MIN_SAMPLES_FOR_MODEL, DEFAULT_MIN_SAMPLES_FOR_MODEL
            ),
            CONF_RETRAIN_INTERVAL: get_entry_value(
                entry, CONF_RETRAIN_INTERVAL, DEFAULT_RETRAIN_INTERVAL
            )
            // 3600,
            CONF_GRID_INVERT: get_entry_value(entry, CONF_GRID_INVERT, False),
        }

        return self.async_show_form(
            step_id="init",
            data_schema=_options_schema(current),
        )
