"""Switch platform for ML Solar Miner."""
import logging

from homeassistant.components.switch import SwitchDeviceClass, SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_AUTO_CONTROL, DOMAIN
from .coordinator import MLSolarMinerCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ML Solar Miner switch."""
    coordinator: MLSolarMinerCoordinator = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities([MLSolarMinerControlSwitch(coordinator, config_entry)])


class MLSolarMinerControlSwitch(SwitchEntity):
    """Switch to enable/disable ML auto-control of the miner."""

    _attr_device_class = SwitchDeviceClass.SWITCH
    _attr_has_entity_name = True
    _attr_unique_id = "ml_solar_miner_control"
    _attr_translation_key = "ml_solar_miner_control"

    def __init__(
        self, coordinator: MLSolarMinerCoordinator, config_entry: ConfigEntry
    ) -> None:
        """Initialize the switch."""
        self.coordinator = coordinator
        self._config_entry = config_entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name="ML Solar Miner",
            manufacturer="ML Solar Miner",
            model="ML Decision Engine",
            sw_version="1.0.1",
        )

    @property
    def is_on(self) -> bool:
        """Return true if auto-control is enabled."""
        return self.coordinator.auto_control

    async def _async_set_auto_control(self, enabled: bool) -> None:
        self.coordinator.auto_control = enabled
        new_options = dict(self._config_entry.options)
        new_options[CONF_AUTO_CONTROL] = enabled
        new_data = dict(self._config_entry.data)
        new_data[CONF_AUTO_CONTROL] = enabled
        self.hass.config_entries.async_update_entry(
            self._config_entry, data=new_data, options=new_options
        )
        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs) -> None:
        """Enable auto-control."""
        await self._async_set_auto_control(True)
        _LOGGER.info("ML auto-control enabled")

    async def async_turn_off(self, **kwargs) -> None:
        """Disable auto-control."""
        await self._async_set_auto_control(False)
        _LOGGER.info("ML auto-control disabled")

    @property
    def extra_state_attributes(self) -> dict:
        """Return extra state attributes."""
        attrs = {}
        if self.coordinator.last_decision:
            attrs["last_mode"] = self.coordinator.last_decision.get("mode", "unknown")
            attrs["last_reason"] = self.coordinator.last_decision.get("reason", "")
            attrs["miner_active"] = self.coordinator.last_decision.get(
                "miner_active", "off"
            )
        return attrs
