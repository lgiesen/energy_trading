"""Compatibility wrapper for export-oriented XGBoost training CLI.

Use `src/energy_trading/models/train_xgboost_export.py` for the explicit
entrypoint used by run orchestration scripts.
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("train_xgboost_export.py")
    runpy.run_path(str(target), run_name="__main__")
