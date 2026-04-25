"""Project-level metrics entrypoint for deterministic and quantile forecasts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.evaluation.metrics import (
    compute_forecast_metrics,
    compute_gate_closure_metrics,
    gate_hour_for_target,
)

__all__ = ["compute_forecast_metrics", "compute_gate_closure_metrics", "gate_hour_for_target"]
