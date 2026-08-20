# ML Solar Miner

[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1+-41BDF5?logo=home-assistant&logoColor=white)](https://www.home-assistant.io/)
[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/tntvlad/ml-solar-miner)](https://github.com/tntvlad/ml-solar-miner/releases)
[![GitHub Stars](https://img.shields.io/github/stars/tntvlad/ml-solar-miner?style=flat&logo=github)](https://github.com/tntvlad/ml-solar-miner/stargazers)

ML-powered Bitcoin miner power management for Home Assistant. Uses machine learning to optimize miner power allocation based on solar production, battery state, and energy forecasts on off-grid installations.

---

## Features

### ML Decision Engine

- **GradientBoostingRegressor** trained on historical decisions with reward-weighted learning
- **Rule-based teacher fallback** replicates proven Ollama LLM logic during bootstrap phase
- **Self-improving**: model retrains weekly and after performance degradation events
- **19 input features**: solar, battery, grid, forecasts, miner state, time-of-day

### Decision Modes

- **DAY (solar)**: Matches miner power to real-time solar surplus (3500-6000W)
- **NIGHT (drain)**: Calculates optimal drain rate to reach target SoC by sunrise based on next-day forecast
- **SAFETY**: Hard limits — SoC < 12% = miner OFF (`safety_shutdown`), grid import > 300W = reduce power (or off if below 3500W). Positive `grid_power` is treated as import.

### Auto-Retraining Triggers

- Weekly schedule (Sunday 03:00)
- Grid import > 500W for 10 min (bad decision)
- SoC drops < 12% with miner running (safety violation)
- Solar surplus > 4000W with miner off for 15 min (missed opportunity)

### Integration Features

- **Config flow UI** — 3-step setup wizard, no YAML editing needed
- **DataUpdateCoordinator** — runs in-process, no shell_command subprocesses
- **Auto-control switch** — toggle AI control from the dashboard
- **Services** — `ml_solar_miner.retrain` and `ml_solar_miner.decision`
- **Automatic dependency install** — scikit-learn + numpy installed by HA via manifest.json
- **Legacy data migration** — auto-migrates training data from old shell_command setup

---

## Technical Architecture

### Decision Cycle (every 20 min)

```
HA Entity States (19 features)
        │
        ▼
DataUpdateCoordinator
        │
        ├── Read entity states via hass.states.get()
        ├── Build feature vector (features_from_state)
        ├── Load model from cache or disk (executor thread)
        │
        ├── If model exists + sample count ≥ min_samples (default 50):
        │   └── GradientBoostingRegressor.predict()
        └── Else:
            └── Rule-based teacher (day_solar / night_drain)
        │
        ├── Validate decision (safety clamp)
        ├── Log to training_data.csv
        └── Apply: switch.turn_on/off + number.set_value
```

### Retraining Cycle

```
Trigger (weekly / event-driven / service call)
        │
        ├── Read training_data.csv
        ├── Fill rewards from consecutive rows
        ├── Weight samples by reward score
        ├── Fit GradientBoostingRegressor (100 trees, depth 4)
        ├── 5-fold cross-validation (if ≥50 samples)
        ├── Save only if validation MAE improves
        └── Update coordinator sensors
```

### Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | GradientBoostingRegressor |
| Estimators | 100 |
| Max depth | 4 |
| Learning rate | 0.1 |
| Min samples leaf | 5 |
| Subsample | 0.8 |
| Target | Optimal miner power (W) |
| Weighting | Reward-weighted (high reward = more influence) |

---

## Requirements

- **Home Assistant** version 2024.1 or newer
- **HACS** installed (recommended)
- Solar energy system with battery storage
- Bitcoin miner with adjustable power (switch + number entities)
- Sensor entities for: solar production, battery SoC, grid power, forecasts

### Required HA Entities

| Entity Type | Purpose | Example |
|-------------|---------|---------|
| `switch` | Miner on/off control | `switch.antminer_active_2` |
| `number` | Miner power setting (3500-6000W) | `number.antminer_power_limit_2` |
| `sensor` | Total solar production | `sensor.solar_power_total` |
| `sensor` | Solar surplus (solar - house load) | `sensor.solar_surplus_power` |
| `sensor` | Battery state of charge | `sensor.battery_soc` |
| `sensor` | Battery voltage | `sensor.battery_voltage` |
| `sensor` | Battery current | `sensor.battery_current` |
| `sensor` | Battery power | `sensor.battery_power` |
| `sensor` | Battery kWh available | `sensor.battery_kwh_available` |
| `sensor` | Battery drain rate | `sensor.battery_drain_rate` |
| `sensor` | Hours to battery minimum | `sensor.battery_hours_to_minimum` |
| `sensor` | Hours until sunrise | `sensor.hours_until_sunrise` |
| `sensor` | Total house load | `sensor.total_load_power` |
| `sensor` | Miner actual consumption | `sensor.miner_consumption` |
| `sensor` | Solar forecast tomorrow | `sensor.forecast_tomorrow` |
| `sensor` | Solar forecast day 3 | `sensor.forecast_day3` |
| `sensor` | Grid import power | `sensor.grid_power` |
| `sensor` | Mining viability score | `sensor.mining_viability_score` |

---

## Installation

### HACS (Recommended)

1. Add this repository as a custom repository in HACS:

   [![Open your Home Assistant instance and show the add-on repository in the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=tntvlad&repository=ml-solar-miner&category=integration)

2. Search for **ML Solar Miner** and install it.
3. Restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration → ML Solar Miner**.
5. Follow the 3-step config flow.

### Manual

1. Download or clone this repository.
2. Copy `custom_components/ml_solar_miner/` to your HA `custom_components/` directory.
3. Restart Home Assistant.
4. Add the integration via **Settings → Devices & Services**.

---

## Configuration

The integration uses a 3-step config flow:

### Step 1: Miner Control

Select your miner switch and power number entities.

### Step 2: Sensor Mapping

Map your HA sensor entities to the 19 ML engine inputs. Entity selectors make this easy.

### Step 3: Options

| Option | Default | Description |
|--------|---------|-------------|
| Decision Interval | 20 min | How often the ML engine runs |
| Auto-Apply | Yes | Automatically apply decisions to the miner |
| Battery Capacity | 69.6 kWh | Total battery bank capacity |
| Min Samples for ML | 50 | Training samples before ML takes over |
| Auto-Retrain Interval | 168 hours | How often to retrain the model |

---

## Entities Created

| Entity | Type | Description |
|--------|------|-------------|
| `sensor.ml_solar_miner_decision_mode` | Sensor | `day_solar`, `night_drain`, or `safety_shutdown` |
| `sensor.ml_solar_miner_decision_reason` | Sensor | Human-readable reason for the decision |
| `sensor.ml_solar_miner_miner_power` | Sensor | Target miner power in watts |
| `sensor.ml_solar_miner_target_soc` | Sensor | Target battery SoC by sunrise (%) |
| `sensor.ml_solar_miner_model_source` | Sensor | `ml_model` or `rule_teacher` |
| `sensor.ml_solar_miner_training_samples` | Sensor | Total training rows in CSV |
| `sensor.ml_solar_miner_training_status` | Sensor | Last retrain status + metrics |
| `sensor.ml_solar_miner_last_retrain` | Sensor | Timestamp of last retrain |
| `sensor.ml_solar_miner_last_decision` | Sensor | Timestamp of last decision |
| `switch.ml_solar_miner_control` | Switch | Enable/disable AI auto-control |

---

## Services

### `ml_solar_miner.retrain`

Trigger immediate model retraining.

```yaml
action: ml_solar_miner.retrain
data:
  force: false  # Retrain even below minimum samples
```

### `ml_solar_miner.decision`

Trigger an immediate ML decision cycle (bypasses schedule).

```yaml
action: ml_solar_miner.decision
data: {}
```

---

## Usage Examples

### Automation: Notify on Safety Shutdown

```yaml
alias: ML Miner Safety Shutdown
trigger:
  - platform: state
    entity_id: sensor.ml_solar_miner_decision_mode
    to: "safety_shutdown"
action:
  - service: notify.notify
    data:
      title: "Miner Safety Shutdown"
      message: >-
        ML engine shut down the miner.
        Battery SoC is critically low.
mode: single
```

### Automation: Retrain on Grid Spike

```yaml
alias: ML Retrain on Grid Spike
trigger:
  - platform: numeric_state
    entity_id: sensor.victron_vebus_activein_l1_power_227
    above: 500
    for:
      minutes: 10
action:
  - service: ml_solar_miner.retrain
    data:
      force: true
mode: single
```

### Dashboard Card

```yaml
type: vertical-stack
cards:
  - type: horizontal-stack
    cards:
      - type: tile
        entity: sensor.ml_solar_miner_model_source
        name: ML Model
        icon: mdi:brain
      - type: tile
        entity: sensor.ml_solar_miner_training_samples
        name: Samples
        icon: mdi:database
      - type: tile
        entity: switch.ml_solar_miner_control
        name: AI Control
        icon: mdi:robot

  - type: entities
    title: "ML Training Metrics"
    show_header_toggle: false
    entities:
      - entity: sensor.ml_solar_miner_training_status
        name: Status
      - entity: sensor.ml_solar_miner_last_retrain
        name: Last Retrain

  - type: glance
    title: "Latest Decision"
    entities:
      - entity: sensor.ml_solar_miner_decision_mode
        name: Mode
      - entity: sensor.ml_solar_miner_miner_power
        name: Power (W)
      - entity: sensor.ml_solar_miner_decision_reason
        name: Reason
```

---

## File Structure

```
custom_components/ml_solar_miner/
├── __init__.py          # Entry point, service registration
├── manifest.json        # HA integration metadata (scikit-learn in requirements)
├── const.py             # Constants, feature names, config keys
├── config_flow.py       # 3-step UI config wizard
├── coordinator.py       # DataUpdateCoordinator — core ML logic
├── models.py            # ML model management (load/save/train/reward)
├── sensor.py            # 9 sensor entities
├── switch.py            # Auto-control toggle switch
├── services.yaml        # Service definitions
├── strings.json         # UI strings
├── translations/
│   └── en.json          # English translations
└── brand/
    └── icon.png         # Integration icon

tests/
└── test_models.py       # Unit tests for teacher, safety, reward, retrain
```

---

## Migration from YAML Setup

If you're migrating from the shell_command-based setup:

1. Install the integration (it auto-migrates training data from `/config/ml_models/`)
2. Run with `auto_control: false` for 1-2 weeks alongside the old automations
3. Compare decisions via the integration sensors vs old template sensors
4. Enable `auto_control: true` and disable the old `mining_decision_engine_ml` automation
5. Remove old shell_command entries and Python scripts from `configuration.yaml`

> The `mining_safety_watchdog` automation should remain as an independent safety layer.

---

## Contributing

Contributions are welcome! Submit a pull request or [report issues](https://github.com/tntvlad/ml-solar-miner/issues).

## Support

If you like this integration, give it a ⭐ on [GitHub](https://github.com/tntvlad/ml-solar-miner)!

## License

MIT
