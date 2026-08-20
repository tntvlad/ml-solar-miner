"""Load the integration modules without importing Home Assistant."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "custom_components" / "ml_solar_miner"

if "ml_solar_miner" not in sys.modules:
    pkg = types.ModuleType("ml_solar_miner")
    pkg.__path__ = [str(ROOT)]
    pkg.__file__ = str(ROOT / "__init__.py")
    sys.modules["ml_solar_miner"] = pkg

    for name in ("const", "models"):
        spec = importlib.util.spec_from_file_location(
            f"ml_solar_miner.{name}", ROOT / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ml_solar_miner.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
