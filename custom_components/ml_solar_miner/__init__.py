"""ML Solar Miner integration."""
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from .const import DOMAIN
from .coordinator import MLSolarMinerCoordinator
from .models import migrate_legacy_data

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor", "switch"]


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up ML Solar Miner from a config entry."""
    # Migrate legacy data on first setup
    await hass.async_add_executor_job(migrate_legacy_data, hass.config.path)

    coordinator = MLSolarMinerCoordinator(hass, config_entry)
    await coordinator.async_config_entry_first_setup()

    hass.data.setdefault(DOMAIN, {})[config_entry.entry_id] = coordinator

    # Forward to platforms
    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    # Register services
    async def handle_retrain(call: ServiceCall) -> None:
        """Handle retrain service call."""
        force = call.data.get("force", False)
        metrics = await coordinator.async_retrain(force=force)
        _LOGGER.info("Retrain completed: %s", metrics.get("status"))

    async def handle_decision(call: ServiceCall) -> None:
        """Handle decision service call."""
        decision = await coordinator.async_request_decision()
        _LOGGER.info("Decision triggered: %s", decision)

    hass.services.async_register(DOMAIN, "retrain", handle_retrain)
    hass.services.async_register(DOMAIN, "decision", handle_decision)

    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id)
        hass.services.async_remove(DOMAIN, "retrain")
        hass.services.async_remove(DOMAIN, "decision")
    return unload_ok
