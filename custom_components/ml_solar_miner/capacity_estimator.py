"""Adaptive battery-capacity estimator for the ML Solar Miner integration.

The measured battery current in this installation only covers part of the
battery bank (the Pylontech stack). The parallel DIY banks are not measured, so
integrating the battery current undercounts the real energy flow.

Instead the energy that moves into or out of the whole bank is taken from the
system power balance::

    P_battery = solar_power_total + grid_import - total_load

with ``P_battery > 0`` meaning charging.  Between two anchor points this energy
is integrated and compared with the SoC change reported by the (trustworthy)
BMS to derive the full-pack capacity::

    charging   : capacity = efficiency * energy_in  / (d_soc / 100)
    discharging: capacity = (energy_out / efficiency) / (|d_soc| / 100)

Only well-behaved windows are accepted: the SoC must stay away from the flat
ends of the curve, the swing must be meaningful, the direction must not reverse,
and the sample count/duration must be sane.  The value that is finally applied
is a trimmed mean of recent accepted windows.

The module deliberately has no Home Assistant imports so it can be reasoned
about and unit tested on its own.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .const import (
    CAPACITY_HISTORY_LIMIT,
    CAPACITY_MAX_CHANGE_PER_UPDATE,
    CAPACITY_MAX_KWH,
    CAPACITY_MAX_WINDOW_HOURS,
    CAPACITY_MIN_KWH,
    CAPACITY_MIN_POWER_W,
    CAPACITY_MIN_SOC_DELTA,
    CAPACITY_MIN_WINDOW_MINUTES,
    CAPACITY_MIN_WINDOW_SAMPLES,
    CAPACITY_MIN_WINDOWS,
    CAPACITY_SANITY_RATIO,
    CAPACITY_SOC_HIGH,
    CAPACITY_SOC_LOW,
)

_LOGGER = logging.getLogger(__name__)

STATE_VERSION = 1

# A window is rejected when the "minor" direction carries more than this share
# of the total energy: applying the efficiency twice would skew the result.
_MIXED_DIRECTION_SHARE = 0.2


@dataclass
class CapacityWindow:
    """One accepted measurement window."""

    started: str
    ended: str
    direction: str  # "charge" | "discharge"
    soc_start: float
    soc_end: float
    energy_kwh: float
    estimate_kwh: float
    efficiency: float
    samples: int
    duration_min: float


class CapacityEstimator:
    """Detect single-direction SoC windows and derive the real capacity."""

    def __init__(
        self,
        efficiency: float = 0.95,
        *,
        soc_low: float = CAPACITY_SOC_LOW,
        soc_high: float = CAPACITY_SOC_HIGH,
        min_soc_delta: float = CAPACITY_MIN_SOC_DELTA,
        min_window_minutes: float = CAPACITY_MIN_WINDOW_MINUTES,
        max_window_hours: float = CAPACITY_MAX_WINDOW_HOURS,
        min_window_samples: int = CAPACITY_MIN_WINDOW_SAMPLES,
        min_power_w: float = CAPACITY_MIN_POWER_W,
        min_windows: int = CAPACITY_MIN_WINDOWS,
        sanity_ratio: float = CAPACITY_SANITY_RATIO,
        max_change: float = CAPACITY_MAX_CHANGE_PER_UPDATE,
        min_kwh: float = CAPACITY_MIN_KWH,
        max_kwh: float = CAPACITY_MAX_KWH,
        history_limit: int = CAPACITY_HISTORY_LIMIT,
    ) -> None:
        self.efficiency = float(efficiency)
        self.soc_low = float(soc_low)
        self.soc_high = float(soc_high)
        self.min_soc_delta = float(min_soc_delta)
        self.min_window_minutes = float(min_window_minutes)
        self.max_window_hours = float(max_window_hours)
        self.min_window_samples = int(min_window_samples)
        self.min_power_w = float(min_power_w)
        self.min_windows = int(min_windows)
        self.sanity_ratio = float(sanity_ratio)
        self.max_change = float(max_change)
        self.min_kwh = float(min_kwh)
        self.max_kwh = float(max_kwh)
        self.history_limit = int(history_limit)
        self.windows: list[CapacityWindow] = []
        self._reset_window()

    # -- status helpers ----------------------------------------------------
    @property
    def window_count(self) -> int:
        """Number of accepted windows kept so far."""
        return len(self.windows)

    @property
    def ready(self) -> bool:
        """True once enough windows exist to apply a learned value."""
        return len(self.windows) >= self.min_windows

    def latest_estimate(self) -> float | None:
        """Estimate from the most recent accepted window, if any."""
        if not self.windows:
            return None
        return self.windows[-1].estimate_kwh

    def recent_windows(self, limit: int = 5) -> list[dict]:
        """The most recent accepted windows as plain dicts (for attributes)."""
        return [asdict(window) for window in self.windows[-limit:]]

    # -- window bookkeeping ------------------------------------------------
    def _reset_window(self) -> None:
        self._anchor_ts: datetime | None = None
        self._anchor_soc: float | None = None
        self._last_ts: datetime | None = None
        self._energy_wh = 0.0
        self._charge_wh = 0.0
        self._discharge_wh = 0.0
        self._samples = 0

    def _reanchor(self, ts: datetime, soc: float) -> None:
        self._reset_window()
        self._anchor_ts = ts
        self._anchor_soc = float(soc)
        self._last_ts = ts
        self._samples = 1

    def add_sample(
        self,
        ts: datetime,
        soc: float | None,
        power_w: float | None,
    ) -> CapacityWindow | None:
        """Feed one (timestamp, SoC, balance power) sample.

        Returns an accepted window when one completes, otherwise ``None``.
        """
        if soc is None or power_w is None:
            return None

        # Only accumulate inside the usable SoC band; outside it the BMS curve
        # is non-linear and the arithmetic would be meaningless.
        if not (self.soc_low <= soc <= self.soc_high):
            self._reset_window()
            return None

        if self._anchor_soc is None or self._anchor_ts is None:
            self._reanchor(ts, soc)
            return None

        dt_h = (ts - self._last_ts).total_seconds() / 3600.0
        if dt_h > 0:
            if power_w >= self.min_power_w:
                self._energy_wh += power_w * dt_h
                self._charge_wh += power_w * dt_h
                self._samples += 1
            elif power_w <= -self.min_power_w:
                self._energy_wh += power_w * dt_h
                self._discharge_wh += -power_w * dt_h
                self._samples += 1
        self._last_ts = ts

        duration_min = (ts - self._anchor_ts).total_seconds() / 60.0
        if duration_min > self.max_window_hours * 60.0:
            # Too long without a decisive swing (long floats, partial days):
            # start over so stale energy is not attributed to a future change.
            self._reanchor(ts, soc)
            return None

        delta = soc - self._anchor_soc
        if abs(delta) < self.min_soc_delta:
            return None

        window = self._close_window(ts, soc, duration_min)
        self._reanchor(ts, soc)
        return window

    def _close_window(
        self, ts: datetime, soc: float, duration_min: float
    ) -> CapacityWindow | None:
        if self._anchor_soc is None or self._anchor_ts is None:
            return None
        d_soc = soc - self._anchor_soc
        if abs(d_soc) < self.min_soc_delta:
            return None
        if self._samples < self.min_window_samples:
            return None
        if duration_min < self.min_window_minutes:
            return None

        total = self._charge_wh + self._discharge_wh
        if total <= 0:
            return None
        if min(self._charge_wh, self._discharge_wh) > _MIXED_DIRECTION_SHARE * total:
            # The battery changed direction materially during the window.
            return None

        efficiency = self.efficiency
        if d_soc > 0:
            direction = "charge"
            energy_kwh = self._charge_wh / 1000.0
            estimate = efficiency * energy_kwh / (d_soc / 100.0)
        else:
            direction = "discharge"
            energy_kwh = self._discharge_wh / 1000.0
            estimate = (energy_kwh / efficiency) / (abs(d_soc) / 100.0)

        if estimate <= 0:
            return None

        window = CapacityWindow(
            started=self._anchor_ts.isoformat(),
            ended=ts.isoformat(),
            direction=direction,
            soc_start=round(self._anchor_soc, 2),
            soc_end=round(soc, 2),
            energy_kwh=round(energy_kwh, 3),
            estimate_kwh=round(estimate, 2),
            efficiency=efficiency,
            samples=self._samples,
            duration_min=round(duration_min, 1),
        )
        _LOGGER.debug("Capacity window accepted: %s", window)
        return window

    # -- aggregation -------------------------------------------------------
    def accept(self, window: CapacityWindow, baseline_kwh: float | None) -> float | None:
        """Record a window and return a new candidate capacity when ready.

        ``baseline_kwh`` is the currently applied capacity (0/None when
        unknown); it is used both to reject obvious outliers and to rate-limit
        the change so a single bad day cannot move the value far.
        """
        baseline = baseline_kwh if baseline_kwh and baseline_kwh > 0 else None

        if baseline is not None:
            low = baseline / self.sanity_ratio
            high = baseline * self.sanity_ratio
            if not (low <= window.estimate_kwh <= high):
                _LOGGER.debug(
                    "Capacity estimate %.1f kWh rejected (outside %.1f-%.1f)",
                    window.estimate_kwh,
                    low,
                    high,
                )
                return None

        self.windows.append(window)
        if len(self.windows) > self.history_limit:
            self.windows = self.windows[-self.history_limit:]

        if len(self.windows) < self.min_windows:
            return None

        values = sorted(item.estimate_kwh for item in self.windows)
        if len(values) >= 5:
            values = values[1:-1]  # drop the single lowest and highest
        target = sum(values) / len(values)

        target = min(max(target, self.min_kwh), self.max_kwh)
        if baseline is not None:
            low = baseline * (1 - self.max_change)
            high = baseline * (1 + self.max_change)
            target = min(max(target, low), high)
        return round(target, 2)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the estimator state."""
        return {
            "version": STATE_VERSION,
            "efficiency": self.efficiency,
            "windows": [asdict(window) for window in self.windows],
        }

    def restore_windows(self, data: dict[str, Any] | None) -> None:
        """Restore the accepted-window history from persisted state."""
        if not data:
            return
        windows: list[CapacityWindow] = []
        for raw in data.get("windows", []):
            try:
                windows.append(CapacityWindow(**raw))
            except TypeError:
                continue
        self.windows = windows[-self.history_limit:]


def load_capacity_state(path: Any) -> dict[str, Any]:
    """Read the persisted learned-capacity state (best effort)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        _LOGGER.warning("Could not read learned capacity state: %s", err)
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_capacity_state(path: Any, state: dict[str, Any]) -> None:
    """Write the learned-capacity state atomically."""
    from pathlib import Path

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        tmp.replace(target)
    except OSError as err:
        _LOGGER.warning("Could not write learned capacity state: %s", err)
