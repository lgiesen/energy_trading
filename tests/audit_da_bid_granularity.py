from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/simulation_runs")
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
MIN_SIZE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
TOL = 1e-8

# Market-facing DA quantities should obey min size and granularity.
# Candidate/pre-filter columns are useful diagnostics but may be allowed to violate before rounding.
FINAL_DA_COLS = [
    "submitted_da_buy_mw",
    "submitted_da_sell_mw",
    "submitted_da_buy_mwh",
    "submitted_da_sell_mwh",
    "da_precommit_selected_lockable_buy_mw",
    "da_precommit_selected_lockable_sell_mw",
    "da_precommit_selected_lockable_buy_mwh",
    "da_precommit_selected_lockable_sell_mwh",
    "da_selected_lockable_buy_mwh",
    "da_selected_lockable_sell_mwh",
    "locked_da_buy_mwh",
    "locked_da_sell_mwh",
    "real_da_buy_mwh",
    "real_da_sell_mwh",
]

# Pre-round / diagnostic columns. Violations here are not necessarily bugs,
# but they show whether rounding happens late.
CANDIDATE_DA_COLS = [
    "da_precommit_candidate_buy_mw",
    "da_precommit_candidate_sell_mw",
    "da_precommit_candidate_buy_mwh",
    "da_precommit_candidate_sell_mwh",
    "raw_optimizer_plan_charge_mw",
    "raw_optimizer_plan_discharge_mw",
    "raw_optimizer_da_buy_mw",
    "raw_optimizer_da_sell_mw",
]

def is_multiple(x, step):
    if abs(x) <= TOL:
        return True
    r = abs(x) / step
    return abs(r - round(r)) <= 1e-6

def audit_col(df, col, *, final=True):
    if col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    nz = s.abs() > TOL

    below_min = nz & (s.abs() + TOL < MIN_SIZE)
    bad_step = nz & ~s.apply(lambda x: is_multiple(float(x), STEP))

    bad = below_min | bad_step

    return {
        "column": col,
        "kind": "final_market_facing" if final else "candidate_or_raw",
        "nonzero_rows": int(nz.sum()),
        "below_min_rows": int(below_min.sum()),
        "bad_step_rows": int(bad_step.sum()),
        "bad_rows": int(bad.sum()),
        "min_nonzero": float(s[nz].abs().min()) if nz.any() else None,
        "max_abs": float(s.abs().max()) if len(s) else None,
        "examples": df.loc[bad, ["timestamp_utc", col]].head(20).to_dict("records") if bad.any() and "timestamp_utc" in df.columns else [],
    }

def audit_file(path):
    df = pd.read_parquet(path)
    print(f"\n=== {path} ===")
    print(f"rows={len(df)}, cols={len(df.columns)}")
    print(f"DA min size={MIN_SIZE}, DA step={STEP}")

    results = []
    for c in FINAL_DA_COLS:
        r = audit_col(df, c, final=True)
        if r:
            results.append(r)

    for c in CANDIDATE_DA_COLS:
        r = audit_col(df, c, final=False)
        if r:
            results.append(r)

    if not results:
        print("No DA quantity columns found.")
        return

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "examples"}
        for r in results
    ])

    print("\nSummary:")
    print(summary.to_string(index=False))

    final_bad = [r for r in results if r["kind"] == "final_market_facing" and r["bad_rows"] > 0]
    candidate_bad = [r for r in results if r["kind"] == "candidate_or_raw" and r["bad_rows"] > 0]

    if final_bad:
        print("\nBUG: final market-facing DA quantities violate min size or step.")
        for r in final_bad:
            print(f"\n{r['column']} examples:")
            for ex in r["examples"]:
                print(ex)
    else:
        print("\nOK: all final market-facing DA quantities obey min size and step.")

    if candidate_bad:
        print("\nNote: candidate/raw DA quantities violate min size or step. This is only a bug if they are submitted or locked before rounding.")
        for r in candidate_bad[:10]:
            print(f"- {r['column']}: bad_rows={r['bad_rows']}")

for p in sorted(ROOT.rglob("backtest_hourly.parquet")):
    audit_file(p)

for p in sorted(ROOT.rglob("backtest_plan_history.parquet")):
    audit_file(p)
