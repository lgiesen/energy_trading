#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from energy_trading.evaluation.forecast_postprocessing import canonicalize_prediction_frame


def _run_self_test() -> int:
    raw = pd.DataFrame(
        {
            "p10": [-100.0],
            "p30": [-80.0],
            "p50": [-50.0],
            "p70": [-20.0],
            "p90": [-10.0],
            "predicted_value": [-50.0],
        }
    )
    out, _ = canonicalize_prediction_frame(
        raw,
        target_name="pred_afrr_activation_price_neg",
        quantile_cols=["p10", "p30", "p50", "p70", "p90"],
        predicted_value_col="predicted_value",
    )
    exp = {"p10": 10.0, "p30": 20.0, "p50": 50.0, "p70": 80.0, "p90": 100.0, "predicted_value": 50.0}
    for k, v in exp.items():
        got = float(out.iloc[0][k])
        if abs(got - v) > 1e-9:
            raise SystemExit(f"Self-test failed for {k}: got={got}, expected={v}")
    print("[OK] postprocessing self-test passed")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check forecast postprocessing for one prediction parquet")
    p.add_argument("--prediction-file", default="")
    p.add_argument("--target", default="pred_afrr_activation_price_neg")
    p.add_argument("--target-value-mode", default="raw_signed_legacy")
    p.add_argument("--out-dir", default="artifacts/diagnostics/forecast_postprocessing_check")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return _run_self_test()

    pred_path = Path(args.prediction_file)
    if not pred_path.exists():
        raise SystemExit(f"prediction file not found: {pred_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pred_path)
    q_cols = [c for c in ["p01", "p05", "p10", "p30", "p50", "p70", "p90", "p95", "p99"] if c in df.columns]
    before_cols = [c for c in ["predicted_value", "p10", "p30", "p50", "p70", "p90"] if c in df.columns]
    before = df[before_cols].copy() if before_cols else pd.DataFrame()

    out, report = canonicalize_prediction_frame(
        df,
        target_name=str(args.target),
        quantile_cols=q_cols,
        predicted_value_col="predicted_value",
        target_value_mode=str(args.target_value_mode),
    )

    after_cols = [c for c in ["predicted_value", "p10", "p30", "p50", "p70", "p90"] if c in out.columns]
    after = out[after_cols].copy() if after_cols else pd.DataFrame()

    rep = pd.DataFrame(
        [
            {
                **report,
                "target": args.target,
                "rows": len(df),
            }
        ]
    )
    rep.to_csv(out_dir / "postprocessing_check_report.csv", index=False)

    sample = pd.concat(
        [before.head(20).add_prefix("before_"), after.head(20).add_prefix("after_")],
        axis=1,
    )
    sample.to_csv(out_dir / "postprocessing_before_after_sample.csv", index=False)

    # Basic invariant check for flipped targets.
    if str(args.target) in {"pred_afrr_capacity_price_neg", "pred_afrr_activation_price_neg"} and {"p10", "p90"}.issubset(df.columns):
        x = pd.to_numeric(df["p90"], errors="coerce")
        y = pd.to_numeric(out["p10"], errors="coerce")
        if not ((y + x).abs() < 1e-6).all():
            raise SystemExit("postprocessing check failed: expected after_p10 == -before_p90 for flipped target")

    print(f"[OK] wrote: {out_dir / 'postprocessing_check_report.csv'}")
    print(f"[OK] wrote: {out_dir / 'postprocessing_before_after_sample.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
