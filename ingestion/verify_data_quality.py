"""Verify data quality for regelleistung and smard time series.

Usage:
    python -m ingestion.verify_data_quality --data-dir energy_trading/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _print_result(name: str, ok: bool, details: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {details}")


def _load_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)


def _ensure_ts(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col in df.columns:
        return df.with_columns(pl.col(col).cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias(col))
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias(col)
        )
        return df
    raise ValueError(f"Missing timestamp column: {col}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify data quality for regelleistung and smard.")
    parser.add_argument("--data-dir", default="energy_trading/data", help="Directory containing parquet files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reg_path = data_dir / "regelleistung.parquet"
    smard_path = data_dir / "smard.parquet"

    reg = _ensure_ts(_load_parquet(reg_path), "timestamp_utc")
    smard = _ensure_ts(_load_parquet(smard_path), "timestamp_utc")

    print("=== Data Quality Verification ===")

    # Blockiness test (Jan 2022)
    jan = reg.filter(
        (pl.col("timestamp_utc") >= pl.datetime(2022, 1, 1, time_zone="UTC"))
        & (pl.col("timestamp_utc") < pl.datetime(2022, 2, 1, time_zone="UTC"))
    ).sort("timestamp_utc")
    block_col = "afrr_activation_avg_price_pos"
    if block_col in jan.columns and jan.height > 1:
        diffs = jan.select(pl.col(block_col).diff()).to_series()
        zero_pct = (diffs == 0).sum() / max(len(diffs), 1) * 100
        ok = zero_pct >= 50
        _print_result(
            "Blockiness Test",
            ok,
            f"{block_col} zero-change hours: {zero_pct:.2f}% (expected high due to 4h blocks)",
        )
    else:
        _print_result("Blockiness Test", False, f"Column {block_col} missing or insufficient rows.")

    # Noon test (Jan 2022)
    noon_rows = jan.filter(pl.col("timestamp_utc").dt.hour() == 12).height
    _print_result(
        "Noon Test",
        noon_rows > 0,
        f"Rows at 12:00 UTC in Jan 2022: {noon_rows}",
    )

    # Spread test (join regelleistung with smard)
    join_cols = ["timestamp_utc"]
    if "da_price_eur" in smard.columns:
        merged = reg.join(smard.select(["timestamp_utc", "da_price_eur"]), on="timestamp_utc", how="inner")
        if merged.height > 0 and "afrr_activation_avg_price_pos" in merged.columns and "afrr_activation_avg_price_neg" in merged.columns:
            spread_pos = (merged["afrr_activation_avg_price_pos"] - merged["da_price_eur"]).mean()
            spread_neg = (merged["afrr_activation_avg_price_neg"] - merged["da_price_eur"]).mean()
            _print_result(
                "Spread Test (Pos)",
                spread_pos > 0,
                f"Mean spread pos: {spread_pos:.2f}",
            )
            _print_result(
                "Spread Test (Neg)",
                spread_neg < 0,
                f"Mean spread neg: {spread_neg:.2f}",
            )
        else:
            _print_result("Spread Test", False, "Missing activation price columns or no overlap.")
    else:
        _print_result("Spread Test", False, "Missing da_price_eur in smard.")

    # Global sanity (min/max for price columns)
    price_cols = [c for c in reg.columns if "price" in c]
    if price_cols:
        print("\n=== Price Column Min/Max (regelleistung) ===")
        for c in price_cols:
            col = reg[c]
            print(f"{c}: min={col.min()}, max={col.max()}")

    smard_price_cols = [c for c in smard.columns if "price" in c]
    if smard_price_cols:
        print("\n=== Price Column Min/Max (smard) ===")
        for c in smard_price_cols:
            col = smard[c]
            print(f"{c}: min={col.min()}, max={col.max()}")


if __name__ == "__main__":
    main()
