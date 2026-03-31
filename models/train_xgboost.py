"""Compatibility wrapper for challenger XGBoost training CLI.

Use `models/train_xgboost_challenger.py` for the explicit entrypoint.
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("train_xgboost_challenger.py")
    runpy.run_path(str(target), run_name="__main__")
