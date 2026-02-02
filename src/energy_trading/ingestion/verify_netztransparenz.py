"""Audit netztransparenz.parquet for aggregation and physics consistency.

Usage:
    python -m energy_trading.ingestion.verify_netztransparenz \
        --path data/raw/netztransparenz.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify netztransparenz parquet integrity.")
    parser.add_argument(
        "--path",
        default="data/raw/netztransparenz.parquet",
        help="Path to netztransparenz.parquet (default: data/raw/netztransparenz.parquet).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pl.read_parquet(path).sort("timestamp_utc")

    report = []
    report.append("NETZTRANSPARENZ DATA QUALITY REPORT")
    report.append(f"rows: {df.height}")
    report.append(f"path: {path}")

    # Check 1: Magnitude check
    afrr_pos_mean = df.select(pl.col("afrr_activated_mw_pos").mean()).item()
    afrr_neg_mean = df.select(pl.col("afrr_activated_mw_neg").mean()).item()
    report.append("")
    report.append("Check 1: Magnitude (aFRR mean)")
    report.append(f"  afrr_activated_mw_pos mean: {afrr_pos_mean:.3f}")
    report.append(f"  afrr_activated_mw_neg mean: {afrr_neg_mean:.3f}")
    if afrr_pos_mean < 50 or afrr_neg_mean < 50:
        report.append("  RESULT: FAIL (likely averaged across TSOs)")
    elif afrr_pos_mean > 100 and afrr_neg_mean > 100:
        report.append("  RESULT: PASS (plausible sum across TSOs)")
    else:
        report.append("  RESULT: WARN (borderline magnitude)")

    # Check 2: Summation consistency
    report.append("")
    report.append("Check 2: Summation Consistency (activated_volume = aFRR + mFRR)")
    tol = 1e-3
    diff_pos = df.select(
        (pl.col("activated_volume_pos_mw") - (pl.col("afrr_activated_mw_pos") + pl.col("mfrr_activated_mw_pos")))
        .abs()
        .max()
    ).item()
    diff_neg = df.select(
        (pl.col("activated_volume_neg_mw") - (pl.col("afrr_activated_mw_neg") + pl.col("mfrr_activated_mw_neg")))
        .abs()
        .max()
    ).item()
    report.append(f"  max abs diff pos: {diff_pos:.6f}")
    report.append(f"  max abs diff neg: {diff_neg:.6f}")
    if diff_pos <= tol and diff_neg <= tol:
        report.append("  RESULT: PASS")
    else:
        report.append("  RESULT: WARN (mismatch above tolerance)")

    # Check 3: Sign consistency
    report.append("")
    report.append("Check 3: Sign Consistency (NEG columns)")
    for col in ["afrr_activated_mw_neg", "mfrr_activated_mw_neg"]:
        s = df.select(pl.col(col).drop_nulls())
        if s.height == 0:
            report.append(f"  {col}: NO DATA")
            continue
        neg = s.select((pl.col(col) < 0).sum()).item()
        pos = s.select((pl.col(col) > 0).sum()).item()
        if neg > 0 and pos > 0:
            report.append(f"  {col}: CRITICAL ERROR (mixed signs)")
        elif neg > 0:
            report.append(f"  {col}: NEGATIVE SIGN")
        else:
            report.append(f"  {col}: POSITIVE SIGN")

    # Check 4: Zero-inflation
    report.append("")
    report.append("Check 4: Zero-Inflation (mFRR)")
    for col in ["mfrr_activated_mw_pos", "mfrr_activated_mw_neg"]:
        zeros = df.select((pl.col(col) == 0).sum()).item()
        total = df.select(pl.col(col).count()).item()
        report.append(f"  {col}: { _pct(zeros, total):.2f}% zeros")
        if zeros == total:
            report.append(f"  {col}: WARN (100% zeros, possible missing data)")

    # Check 5: Physics check (rebap vs net balance)
    report.append("")
    report.append("Check 5: Physics (Price/Volume correlation)")
    if "reBAP_shortage_surplus" not in df.columns:
        report.append("  reBAP_shortage_surplus column missing")
    else:
        net_balance = (pl.col("activated_volume_pos_mw") - pl.col("activated_volume_neg_mw")).alias("net_balance_mw")
        corr = (
            df.with_columns(net_balance)
            .select(pl.corr("net_balance_mw", "reBAP_shortage_surplus"))
            .item()
        )
        report.append(f"  Pearson corr(net_balance_mw, reBAP): {corr:.4f}")
        if corr is None or corr < 0.2:
            report.append("  WARN: Price/Volume decoupling detected (Data Mismatch or Timezone Shift)")
        else:
            report.append("  PASS: correlation plausible")

    print("\n".join(report))


if __name__ == "__main__":
    main()
