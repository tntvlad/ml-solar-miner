"""ML Solar Miner integration."""
import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .coordinator import MLSolarMinerCoordinator
from .models import migrate_legacy_data

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.SWITCH]


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up ML Solar Miner from a config entry."""
    await hass.async_add_executor_job(migrate_legacy_data, hass.config.path)

    coordinator = MLSolarMinerCoordinator(hass, config_entry)
    await coordinator.async_setup_listeners()
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[config_entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    async def handle_retrain(call: ServiceCall) -> None:
        """Handle retrain service call."""
        force = call.data.get("force", False)
        metrics = await coordinator.async_retrain(force=force)
        _LOGGER.info("Retrain completed: %s", metrics.get("status"))

    async def handle_decision(call: ServiceCall) -> None:
        """Handle decision service call."""
        decision = await coordinator.async_request_decision()
        _LOGGER.info("Decision triggered: %s", decision)

    hass.services.async_register(
        DOMAIN,
        "retrain",
        handle_retrain,
        schema=vol.Schema({vol.Optional("force", default=False): cv.boolean}),
    )
    hass.services.async_register(DOMAIN, "decision", handle_decision)

    config_entry.async_on_unload(config_entry.add_update_listener(_async_update_listener))
    config_entry.async_on_unload(coordinator.async_unload_listeners)

    return True


async def _async_update_listener(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Apply options changes to the running coordinator."""
    coordinator: MLSolarMinerCoordinator | None = hass.data.get(DOMAIN, {}).get(
        config_entry.entry_id
    )
    if coordinator is None:
        return
    coordinator.apply_config(config_entry)
    _LOGGER.info(
        "Updated ML Solar Miner options: interval=%s auto_control=%s",
        coordinator.update_interval,
        coordinator.auto_control,
    )


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id, None)
        hass.services.async_remove(DOMAIN, "retrain")
        hass.services.async_remove(DOMAIN, "decision")
    return unload_ok
