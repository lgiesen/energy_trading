"""Backward-compatible wrapper for Energy Charts day-ahead prices.

Use:
    python -m ingestion.fetch_energy_charts_prices ...
"""
from __future__ import annotations

from energy_trading.ingestion.fetch_energy_charts import main


if __name__ == "__main__":
    main()
