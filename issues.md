# Issues

Review of `ml-solar-miner` as of 2026-08-20. All code issues below were fixed in 1.0.1.

Publishing a GitHub release (item 22) is a repo-admin step and is not done here.

## Critical

### 1. Setup calls a method that does not exist — fixed

Setup now calls `async_config_entry_first_refresh()`.

### 2. Options flow crashes on Home Assistant 2025.12+ — fixed

Options flow no longer assigns `self.config_entry`. It uses the HA-provided property.

## High

### 3. Advertised auto-retrain is not implemented — fixed

Weekly Sunday 03:00, configured interval, and event triggers (grid import, low SoC, missed surplus) run from the coordinator.

### 4. Options changes do not apply — fixed

Options are stored on `config_entry.options`. The coordinator reads options (with data fallback) and an update listener applies changes live.

### 5. ML takeover does not check sample count — fixed

The model is used only when it loads **and** `training_samples >= min_samples_for_model`.

### 6. Day-mode grid import can turn the miner on — fixed

`decide_power()` never raises a sub-minimum value to 3500W/`on`. Import is subtracted before the on/off decision.

### 7. Retrain can crash on CSV string types — fixed

Reward/outcome helpers coerce floats and treat `"False"` as false.

### 8. `force` retrain does not bypass the sample floor — fixed

`force=True` trains from 2 rewarded rows. Service description matches.

### 9. Safety clamp does not set `safety_shutdown` — fixed

Low SoC now sets `mode` to `safety_shutdown`.

### 10. Grid import > 300W reduce-power rule is missing — fixed

`validate_decision()` reduces (or turns off) on import above 300W. Positive `grid_power` is import.

## Medium

### 11. Missing `hours_until_sunrise` disables night mining — fixed

A missing sensor is estimated from time of day. An explicit `0` (sunrise) still means no night drain.

### 12. ML path hardcodes target SoC to 30 — fixed

Target SoC comes from `target_soc_from_forecast()`.

### 13. Last CSV row is never trained — fixed

Retrain fills the last row from live entity state.

### 14. Reward `0` samples are dropped — fixed

Any present numeric reward, including `0`, is used.

### 15. Timestamp sensors use naive ISO strings — fixed

Timestamps are UTC ISO and sensors return timezone-aware `datetime` objects.

### 16. Blocking disk I/O on the event loop — fixed

Sensors read cached `coordinator.metrics`.

### 17. Missing entity translations — fixed

`strings.json` and `translations/en.json` include entity names.

### 18. Training-samples sensor uses `TOTAL_INCREASING` — fixed

It now uses `MEASUREMENT`.

### 19. Target SoC sensor has no unit — fixed

Unit is `%`.

### 20. Retrain vs decision CSV race — fixed

Retrain takes the same `_csv_lock` as decision logging.

### 21. Documented `brand/icon.png` is missing — fixed

Icon is at `custom_components/ml_solar_miner/brand/icon.png`.

### 22. No GitHub releases — remaining

Create and publish a GitHub release (e.g. `v1.0.1`) when you are ready to ship. The README badge will then resolve.

## Lower

### 23. No tests — fixed

`tests/test_models.py` plus `.github/workflows/pytest.yaml`.

### 24. `.gitignore` typo — fixed

`__pycache__/` (and `.pytest_cache/`).

### 25. Pickle model is brittle — fixed

Load failures fall back to the rule teacher instead of crashing.

### 26. Cross-validation does not use sample weights — fixed

K-fold CV fits with the same reward weights as the final model.

### 27. Forced worse model vs `best_val_score` — fixed

`best_val_score` tracks the model actually on disk.

### 28. Grid sign / `abs()` treats export as import — fixed

Only positive grid watts count as import. Export is not penalized.

### 29. Manifest metadata is a poor fit — fixed

`integration_type` is `service`.

### 30. Unused / leftover constants — fixed

Legacy path constants are used in migration. Teacher no longer binds unused locals. Sensor attribute imports are trimmed.
