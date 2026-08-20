# ML Solar Miner

Home Assistant-based Bitcoin miner power management on an off-grid solar+battery installation. Uses machine learning to optimize miner power allocation based on solar production, battery state, and energy forecasts.

## Features

- **ML Decision Engine**: GradientBoostingRegressor trained on historical decisions
- **Rule-based fallback**: Replicates original Ollama LLM logic
- **Auto-retraining**: Weekly + event-driven triggers
- **Solar surplus tracking**: Real-time solar available for mining
- **Battery drain planning**: Night mode optimizes battery usage until sunrise
- **Safety constraints**: SoC minimums, grid import avoidance, power clamping

## Installation

### HACS (Recommended)

1. Add this repository as a custom repository in HACS:
   - Go to HACS → Integrations → Click the three dots (top right) → Custom repositories
   - Enter the repository URL and select "Integration" as the category
2. Install "ML Solar Miner" from HACS
3. Restart Home Assistant
4. Go to Settings → Devices & Services → Add Integration → ML Solar Miner
5. Follow the config flow to map your entities

### Manual

1. Copy `custom_components/ml_solar_miner/` to your HA `custom_components/` directory
2. Restart Home Assistant
3. Add the integration via Settings → Devices & Services

## Configuration

The integration uses a 3-step config flow:

1. **Miner Control**: Select your miner switch and power number entities
2. **Sensor Mapping**: Map your HA sensor entities to the ML engine inputs (19 features)
3. **Options**: Set decision interval, auto-control, battery capacity, retrain interval

### Required HA Integrations

Your existing HA setup must provide these entities:

| Entity | Purpose |
|--------|---------|
| `switch.antminer_active_2` | Miner on/off |
| `number.antminer_power_limit_2` | Power setting (3500-6000W) |
| `sensor.solar_power_total` | Total solar production |
| `sensor.solar_surplus_power` | Solar surplus (solar - house load) |
| `sensor.infinisolar_v_multiphase_x_3_battery_state_of_charge` | Battery SoC |
| `sensor.victron_vebus_activein_l1_power_227` | Grid import power |
| `sensor.solcast_pv_forecast_atelier_forecast_tomorrow` | Tomorrow's solar forecast |
| `sensor.antminer_miner_consumption_2` | Actual miner power draw |

Plus battery voltage, current, power, drain rate, hours to minimum, total load, and day-3 forecast.

## How It Works

```
HA Entity States (19 features)
        │
        ▼
DataUpdateCoordinator (every 20 min)
        │
        ├── ML model (if trained + ≥50 samples)
        └── Rule teacher (fallback)
        │
        ▼
Switch/Number entities → Miner ON/OFF + Power setting
```

## Services

| Service | Description |
|---------|-------------|
| `ml_solar_miner.retrain` | Trigger immediate model retraining |
| `ml_solar_miner.decision` | Trigger immediate ML decision cycle |

## Hardware

- 3x Infinisolar V Multiphase X-3 inverters
- 2x Victron solar chargers
- ~69.6 kWh battery bank (Victron + Pylontech)
- Antminer (adjustable 3500-6000W)
- Oil immersion cooling with PID temperature control
- SNMP-controlled PDU for power management

## Migration from YAML Setup

If you're migrating from the shell_command-based setup:

1. Install the integration (it auto-migrates training data from `/config/ml_models/`)
2. Run with `auto_control: false` for 1-2 weeks alongside the old automations
3. Compare decisions via the integration sensors vs old template sensors
4. Enable `auto_control: true` and disable the old `mining_decision_engine_ml` automation
5. Remove old shell_command entries and Python scripts from `configuration.yaml`

The `mining_safety_watchdog` automation should remain as an independent safety layer.

## License

MIT
