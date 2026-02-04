"""
Sanity Check Script for Energy Trading Data.
Validates 'data/processed/all_data.parquet' against physical and market reality.

Usage: 
    ./.venv/bin/python -m energy_trading.utils.raw_data_sanity_check
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import polars as pl

# --- Configuration ---
DATA_PATH = Path("data/processed/all_data.parquet")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
LOGGER = logging.getLogger("SanityCheck")

@dataclass
class Constraints:
    """Defines what 'Realistic' means for a column."""
    hard_min: float = -float("inf")
    hard_max: float = float("inf")
    soft_min: float | None = None
    soft_max: float | None = None
    allow_nulls_pct: float = 1.0  # 1.0% tolerance
    allow_zeros_pct: float = 100.0 # Default allow all (e.g. solar at night)
    strict_positivity: bool = False
    warn_if_all_zeros: bool = False # Warn if column is dead (all 0)

# --- Define Rules Per Column Group ---

# 1. Load (Consumption) - Critical
# Germany load roughly 30GW - 100GW. Never 0.
LOAD_RULES = Constraints(
    hard_min=10_000.0, 
    hard_max=120_000.0, 
    allow_nulls_pct=0.1, 
    allow_zeros_pct=0.0,  # Load is never 0
    strict_positivity=True
)

# Residual Load (can go negative)
RESIDUAL_LOAD_RULES = Constraints(
    hard_min=-50_000.0,
    hard_max=100_000.0,
    allow_nulls_pct=1.0,
)

# 2. Generation (Actuals)
# Can be 0 (Solar at night, outages). Max ~60GW (Wind) to ~30GW (Coal).
GEN_RULES = Constraints(
    hard_min=0.0,
    hard_max=80_000.0, # Generous upper bound for Wind
    allow_nulls_pct=1.0,
    strict_positivity=True,
    warn_if_all_zeros=True # Warning if a generation type is completely missing
)

# 3. Forecasts (MW)
# Similar to generation, but distinct.
FORECAST_RULES = Constraints(
    hard_min=0.0,
    hard_max=80_000.0,
    allow_nulls_pct=1.0,
    strict_positivity=True
)

# 4. Forecast Errors (MW)
# Can be positive or negative. Should be within +/- 20GW usually.
ERROR_RULES = Constraints(
    hard_min=-25_000.0,
    hard_max=25_000.0,
    allow_nulls_pct=2.0
)

# 5. Prices (EUR/MWh) - Power
# Can be negative. Historic max ~4000 (crisis). Historic min ~-500.
PRICE_RULES = Constraints(
    hard_min=-9_999.0,
    hard_max=9_999.0,
    soft_min=-4_000.0,
    soft_max=4_000.0,
    allow_nulls_pct=0.5,
    allow_zeros_pct=5.0 # Rare to hit exactly 0.00
)

# 6. Commodities (Gas, Coal, CO2) - EUR
# Always positive.
COMMODITY_RULES = Constraints(
    hard_min=0.1,
    hard_max=1_000.0,
    allow_nulls_pct=35.0, # High tolerance: Trading holidays / yfinance gaps
    strict_positivity=True
)

# 7. Balancing Volumes (MW/MWh)
# aFRR/mFRR. Can be 0 (often). Max < 10GW.
BALANCING_VOL_RULES = Constraints(
    hard_min=-6_000.0,
    hard_max=6_000.0,
    allow_nulls_pct=5.0,
    warn_if_all_zeros=True # Warn if aFRR is dead (mFRR might be legitimately dead/sparse)
)

# Netztransparenz balance series can have publication lag.
NETZTRANS_BALANCE_RULES = Constraints(
    hard_min=BALANCING_VOL_RULES.hard_min,
    hard_max=BALANCING_VOL_RULES.hard_max,
    allow_nulls_pct=6.0,
    allow_zeros_pct=BALANCING_VOL_RULES.allow_zeros_pct,
    strict_positivity=BALANCING_VOL_RULES.strict_positivity,
    warn_if_all_zeros=BALANCING_VOL_RULES.warn_if_all_zeros,
)

# 8. Balancing Prices
# Can be extreme.
BALANCING_PRICE_RULES = Constraints(
    hard_min=-10_000.0,
    hard_max=20_000.0,
    allow_nulls_pct=10.0
)

# 9. Outages (MW)
# >= 0.
OUTAGE_RULES = Constraints(
    hard_min=0.0,
    hard_max=50_000.0,
    allow_nulls_pct=1.0,
    strict_positivity=True
)

# Outage-specific overrides (MW)
OUTAGE_BASELOAD_RULES = Constraints(
    hard_min=0.0,  # 0 MW outage is the ideal state (all plants operational). A minimum of 0 is valid and not an error.
    hard_max=50_000.0,
    allow_nulls_pct=1.0,
    strict_positivity=True,
)
OUTAGE_GAS_RULES = Constraints(
    hard_min=0.0,
    hard_max=50_000.0,  # Higher limit to account for reserve plants and overlapping outage events.
    allow_nulls_pct=1.0,
    strict_positivity=True,
)
OUTAGE_HARD_COAL_RULES = Constraints(
    hard_min=0.0,
    hard_max=40_000.0,  # Includes plants in grid reserve; values > installed active capacity are possible in this dataset.
    allow_nulls_pct=1.0,
    strict_positivity=True,
)

def get_constraints(col_name: str) -> Constraints:
    """Maps column names to physical constraints."""
    c = col_name.lower()
    
    # OUTAGES (must be checked early to avoid collisions with "load"/"gas"/"coal" keywords)
    if "outage_baseload_mw" in c:
        return OUTAGE_BASELOAD_RULES
    if "outage_gas_mw" in c:
        return OUTAGE_GAS_RULES
    if "outage_hard_coal_mw" in c:
        return OUTAGE_HARD_COAL_RULES
    if "outage" in c:
        return OUTAGE_RULES

    # LOAD
    if "residual_load" in c or "residual load" in c:
        return RESIDUAL_LOAD_RULES

    if "load" in c and "residual" not in c and "forecast" not in c:
        return LOAD_RULES
    
    # GENERATION
    if "generation" in c:
        return GEN_RULES
    
    # FORECASTS (Physical)
    if "forecast" in c and "error" not in c:
        return FORECAST_RULES
    
    # ERRORS
    if "error" in c:
        return ERROR_RULES
    
    # POWER PRICES (Day Ahead, Intraday, Neighbors)
    if "price_eur" in c or "da_price" in c:
        return PRICE_RULES
    
    # COMMODITIES
    if any(x in c for x in ["co2", "coal", "gas"]):
        return COMMODITY_RULES
    
    # BALANCING VOLUMES
    if "nrv_balance" in c or "rz_saldo" in c:
        return NETZTRANS_BALANCE_RULES
    if any(x in c for x in ["activated", "volume", "balance", "saldo", "import_export"]):
        # Special case for absolute volumes (pos/neg split) -> usually positive numbers
        if "neg" in c or "pos" in c:
             return Constraints(hard_min=-500.0, hard_max=10_000.0, allow_nulls_pct=5.0, strict_positivity=False) # Allow small neg noise
        return BALANCING_VOL_RULES

    # BALANCING PRICES
    if "activation_price" in c or "capacity_price" in c or "rebap" in c:
        return BALANCING_PRICE_RULES
    
        
    # Default fallback (loose)
    return Constraints()

def check_column(df: pl.DataFrame, col: str) -> List[str]:
    """Runs checks on a single column. Returns list of failure messages."""
    rules = get_constraints(col)
    failures = []
    
    # 1. Null Check
    null_count = df[col].null_count()
    null_pct = (null_count / df.height) * 100
    if null_pct > rules.allow_nulls_pct:
        failures.append(f"Nulls: {null_pct:.2f}% (Limit: {rules.allow_nulls_pct}%)")

    # Filter out nulls for numeric checks
    valid_data = df.select(pl.col(col).drop_nulls())
    if valid_data.height == 0:
        failures.append("Column is empty (all nulls)")
        return failures

    min_val = valid_data[col].min()
    max_val = valid_data[col].max()
    
    # 2. Range Check
    if min_val < rules.hard_min:
        failures.append(f"Min Value: {min_val:.2f} < Hard Limit {rules.hard_min}")
    if max_val > rules.hard_max:
        failures.append(f"Max Value: {max_val:.2f} > Hard Limit {rules.hard_max}")

    # Soft limits (warnings)
    if rules.soft_min is not None and min_val < rules.soft_min:
        failures.append(f"Min Value: {min_val:.2f} < Soft Limit {rules.soft_min}")
    if rules.soft_max is not None and max_val > rules.soft_max:
        failures.append(f"Max Value: {max_val:.2f} > Soft Limit {rules.soft_max}")
        
    # 3. Strict Positivity
    if rules.strict_positivity and min_val < 0:
        failures.append(f"Negative values found ({min_val:.2f}) but strict positivity required")

    # 4. Zero Check
    zero_count = valid_data.filter(pl.col(col) == 0).height
    zero_pct = (zero_count / df.height) * 100
    
    if zero_pct > rules.allow_zeros_pct:
        failures.append(f"Zeros: {zero_pct:.2f}% (Limit: {rules.allow_zeros_pct}%)")
        
    if rules.warn_if_all_zeros and zero_count == valid_data.height:
        failures.append("Column contains ONLY zeros")

    return failures

def check_mathematical_consistency(df: pl.DataFrame):
    """
    Verifies that Derived Columns = Col A - Col B.
    """
    LOGGER.info("--- Checking Mathematical Consistency ---")
    
    # Check 1: Solar Error
    # Formula: Error = Actual - Forecast (or Forecast - Actual, depending on definition)
    # We check correlation first to see definition, then exact match.
    if {"solar_actual", "solar_forecast", "solar_error"}.issubset(df.columns):
        # Calculate expected error (assuming Actual - Forecast)
        calc_error = df["solar_actual"] - df["solar_forecast"]
        diff = (calc_error - df["solar_error"]).abs().mean()
        
        if diff < 0.1:
             LOGGER.info("[OK] Solar Error matches (Actual - Forecast).")
        else:
             # Try other direction
             diff_reverse = (df["solar_forecast"] - df["solar_actual"] - df["solar_error"]).abs().mean()
             if diff_reverse < 0.1:
                 LOGGER.info("[OK] Solar Error matches (Forecast - Actual).")
             else:
                 LOGGER.warning(f"[FAIL] Solar Error inconsistent! Mean Diff: {diff:.2f}")

    # Check 2: Wind Onshore Error
    if {"wind_onshore_actual", "wind_onshore_forecast", "wind_onshore_error"}.issubset(df.columns):
        calc_error = df["wind_onshore_actual"] - df["wind_onshore_forecast"]
        diff = (calc_error - df["wind_onshore_error"]).abs().mean()
        if diff < 0.1:
             LOGGER.info("[OK] Wind Onshore Error matches formula.")
        else:
             LOGGER.warning(f"[FAIL] Wind Onshore Error inconsistent! Mean Diff: {diff:.2f}")

def check_temporal_integrity(df: pl.DataFrame):
    """
    Checks for duplicates and time gaps.
    """
    LOGGER.info("--- Checking Temporal Integrity ---")
    
    # 1. Duplicates
    if "timestamp_utc" not in df.columns:
        LOGGER.warning("[WARN] timestamp_utc not found; skipping temporal integrity checks.")
        return
    tz = df["timestamp_utc"].dtype.time_zone
    if tz != "UTC":
        LOGGER.error(f"[FAIL] timestamp_utc timezone is {tz}, expected UTC.")
    else:
        LOGGER.info("[OK] timestamp_utc timezone is UTC.")
    nulls = df["timestamp_utc"].null_count()
    if nulls:
        LOGGER.warning(f"[WARN] timestamp_utc has {nulls} nulls (these will be excluded from dup/gap checks).")
    valid = df.filter(pl.col("timestamp_utc").is_not_null())
    dupes = valid["timestamp_utc"].is_duplicated().sum()
    if dupes > 0:
        LOGGER.error(f"[FAIL] Found {dupes} duplicate timestamps!")
    else:
        LOGGER.info("[OK] No duplicate timestamps.")

    # 2. Gaps
    # Sort and check diff
    df = valid.unique(subset=["timestamp_utc"], keep="last").sort("timestamp_utc")
    time_diff = df["timestamp_utc"].diff().dt.total_hours()
    
    # Filter out the first row (null diff)
    gaps = time_diff.filter(time_diff > 1.0)
    if gaps.len() > 0:
        LOGGER.warning(f"[FAIL] Found {gaps.len()} time gaps > 1 hour.")
        LOGGER.warning(f"      First gap at: {df.filter(time_diff > 1.0)['timestamp_utc'].head(1)}")
    else:
        LOGGER.info("[OK] Time series is continuous (hourly).")

def check_neighbor_correlations(df: pl.DataFrame):
    """
    Checks if German prices correlate with neighbors. 
    Low correlation implies timezone mismatches or data corruption.
    """
    LOGGER.info("--- Checking Neighbor Correlations (Timezone Proxy) ---")
    
    neighbors = ["da_price_AT", "da_price_FR", "da_price_NL", "da_price_PL"]
    target = "da_price_eur"
    
    if target not in df.columns:
        return

    for n in neighbors:
        if n in df.columns:
            # Drop nulls for correlation
            valid = df.select([target, n]).drop_nulls()
            if valid.height > 100:
                corr = valid.select(pl.corr(target, n)).item()
                
                if corr > 0.8:
                    LOGGER.info(f"[OK] {n} correlation: {corr:.4f}")
                elif corr > 0.5:
                    LOGGER.warning(f"[WARN] {n} correlation weak: {corr:.4f} (Market decoupling?)")
                else:
                    LOGGER.error(f"[FAIL] {n} correlation extremely low: {corr:.4f}. CHECK TIMEZONES!")

def check_balancing_logic(df: pl.DataFrame):
    """
    Checks if aggregated volumes match components (if present).
    """
    LOGGER.info("--- Checking Balancing Logic ---")
    
    # Check if aFRR + mFRR roughly equals Net Balance (if columns exist)
    # Note: 'activated_volume_pos_mw' is the sum.
    
    cols_needed = ["activated_volume_pos_mw", "afrr_activated_mw_pos", "mfrr_activated_mw_pos"]
    if set(cols_needed).issubset(df.columns):
        # Allow small floating point diff
        diff = (df["afrr_activated_mw_pos"].fill_null(0) + df["mfrr_activated_mw_pos"].fill_null(0) - df["activated_volume_pos_mw"].fill_null(0)).abs().mean()
        
        if diff < 1.0: # 1 MW tolerance
            LOGGER.info("[OK] Positive Activated Volume matches (aFRR + mFRR).")
        else:
            LOGGER.warning(f"[FAIL] Positive Activated Volume mismatch! Mean Diff: {diff:.2f} MW")

def main():
    if not DATA_PATH.exists():
        LOGGER.error(f"File not found: {DATA_PATH}")
        return

    LOGGER.info(f"Reading {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    LOGGER.info(f"Loaded {df.height} rows, {len(df.columns)} columns.")

    # Check Timestamps first
    ts_cols = [c for c in df.columns if "timestamp" in c]
    for ts in ts_cols:
        nulls = df[ts].null_count()
        if nulls > 0:
            LOGGER.warning(f"TIMESTAMP ISSUE: {ts} has {nulls} nulls!")
        else:
            LOGGER.info(f"Timestamp {ts}: OK")

    # Check Data Columns
    cols_to_check = [c for c in df.columns if c not in ts_cols]

    warnings_count = 0

    for col in sorted(cols_to_check):
        failures = check_column(df, col)

        if not failures:
            LOGGER.info(f"[OK] {col}")
        else:
            warnings_count += 1
            LOGGER.warning(f"[FAIL] {col}")
            for f in failures:
                LOGGER.warning(f"    -> {f}")

    print("-" * 50)
    if warnings_count == 0:
        LOGGER.info("SUCCESS: All columns passed univariate sanity checks.")
    else:
        LOGGER.warning(f"FINISHED: {warnings_count} columns triggered univariate warnings. Review above.")

    # --- Advanced relational checks ---
    check_temporal_integrity(df)
    check_mathematical_consistency(df)
    check_neighbor_correlations(df)
    check_balancing_logic(df)

if __name__ == "__main__":
    main()
