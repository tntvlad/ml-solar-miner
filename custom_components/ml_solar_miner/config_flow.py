"""Config flow for ML Solar Miner."""
import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_AUTO_CONTROL,
    CONF_BATTERY_CAPACITY_KWH,
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
)

_LOGGER = logging.getLogger(__name__)


def _entity_selector(domain: str) -> selector.EntitySelector:
    """Create an entity selector filtered by domain."""
    return selector.EntitySelector(
        selector.EntityFilterSelectorConfig(domain=domain)
    )


# Step 1 schema
STEP_1_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MINER_SWITCH): _entity_selector("switch"),
        vol.Required(CONF_MINER_POWER_NUMBER): _entity_selector("number"),
    }
)

# Step 2 schema — sensor entity mapping
STEP_2_SCHEMA = vol.Schema(
    {
        vol.Optional("solar_power_total"): _entity_selector("sensor"),
        vol.Optional("solar_surplus_power"): _entity_selector("sensor"),
        vol.Optional("battery_soc"): _entity_selector("sensor"),
        vol.Optional("battery_voltage"): _entity_selector("sensor"),
        vol.Optional("battery_current"): _entity_selector("sensor"),
        vol.Optional("battery_power"): _entity_selector("sensor"),
        vol.Optional("battery_kwh_available"): _entity_selector("sensor"),
        vol.Optional("battery_drain_rate"): _entity_selector("sensor"),
        vol.Optional("battery_hours_to_min"): _entity_selector("sensor"),
        vol.Optional("hours_until_sunrise"): _entity_selector("sensor"),
        vol.Optional("total_load_power"): _entity_selector("sensor"),
        vol.Optional("miner_consumption"): _entity_selector("sensor"),
        vol.Optional("forecast_tomorrow"): _entity_selector("sensor"),
        vol.Optional("forecast_day3"): _entity_selector("sensor"),
        vol.Optional("grid_power"): _entity_selector("sensor"),
        vol.Optional("mining_viability_score"): _entity_selector("sensor"),
    }
)

# Step 3 schema — options
STEP_3_SCHEMA = vol.Schema(
    {
        vol.Optional(
            CONF_SCAN_INTERVAL_OPTION, default=DEFAULT_SCAN_INTERVAL // 60
        ): vol.All(vol.Coerce(int), vol.Range(min=5, max=1440)),
        vol.Optional(CONF_AUTO_CONTROL, default=True): bool,
        vol.Optional(
            CONF_BATTERY_CAPACITY_KWH, default=DEFAULT_BATTERY_CAPACITY_KWH
        ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=1000.0)),
        vol.Optional(
            CONF_MIN_SAMPLES_FOR_MODEL, default=DEFAULT_MIN_SAMPLES_FOR_MODEL
        ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
        vol.Optional(
            CONF_RETRAIN_INTERVAL, default=DEFAULT_RETRAIN_INTERVAL // 3600
        ): vol.All(vol.Coerce(int), vol.Range(min=1, max=720)),
    }
)


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
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate entities exist
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
            description_placeholders={
                "name": "ML Solar Miner",
            },
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
            # Convert minutes/hours to seconds
            scan_seconds = user_input.pop(CONF_SCAN_INTERVAL_OPTION) * 60
            retrain_hours = user_input.pop(CONF_RETRAIN_INTERVAL) * 3600

            self.data[CONF_SCAN_INTERVAL_OPTION] = scan_seconds
            self.data[CONF_RETRAIN_INTERVAL] = retrain_hours
            self.data.update(user_input)

            # Set unique ID and create entry
            await self.async_set_unique_id("ml_solar_miner")
            self._abort_if_unique_id_configured()

            return self.async_create_entry(title="ML Solar Miner", data=self.data)

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
        return MLSolarMinerOptionsFlow(config_entry)


class MLSolarMinerOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for ML Solar Miner."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.options = dict(config_entry.data)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            scan_seconds = user_input.pop(CONF_SCAN_INTERVAL_OPTION) * 60
            retrain_hours = user_input.pop(CONF_RETRAIN_INTERVAL) * 3600

            self.options[CONF_SCAN_INTERVAL_OPTION] = scan_seconds
            self.options[CONF_RETRAIN_INTERVAL] = retrain_hours
            self.options.update(user_input)

            return self.async_create_entry(title="", data=self.options)

        # Pre-fill current values
        current = {
            CONF_SCAN_INTERVAL_OPTION: self.options.get(
                CONF_SCAN_INTERVAL_OPTION, DEFAULT_SCAN_INTERVAL
            )
            // 60,
            CONF_AUTO_CONTROL: self.options.get(CONF_AUTO_CONTROL, True),
            CONF_BATTERY_CAPACITY_KWH: self.options.get(
                CONF_BATTERY_CAPACITY_KWH, DEFAULT_BATTERY_CAPACITY_KWH
            ),
            CONF_MIN_SAMPLES_FOR_MODEL: self.options.get(
                CONF_MIN_SAMPLES_FOR_MODEL, DEFAULT_MIN_SAMPLES_FOR_MODEL
            ),
            CONF_RETRAIN_INTERVAL: self.options.get(
                CONF_RETRAIN_INTERVAL, DEFAULT_RETRAIN_INTERVAL
            )
            // 3600,
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SCAN_INTERVAL_OPTION,
                        default=current[CONF_SCAN_INTERVAL_OPTION],
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=1440)),
                    vol.Optional(
                        CONF_AUTO_CONTROL,
                        default=current[CONF_AUTO_CONTROL],
                    ): bool,
                    vol.Optional(
                        CONF_BATTERY_CAPACITY_KWH,
                        default=current[CONF_BATTERY_CAPACITY_KWH],
                    ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=1000.0)),
                    vol.Optional(
                        CONF_MIN_SAMPLES_FOR_MODEL,
                        default=current[CONF_MIN_SAMPLES_FOR_MODEL],
                    ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
                    vol.Optional(
                        CONF_RETRAIN_INTERVAL,
                        default=current[CONF_RETRAIN_INTERVAL],
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=720)),
                }
            ),
        )
