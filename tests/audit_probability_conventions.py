from pathlib import Path
import sys
import pandas as pd
import re
import math

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/simulation_runs")

EPS = 1e-9

def q_to_float(q):
    if q is None:
        return None
    s = str(q).strip().lower()
    m = re.search(r"p(\d+)", s)
    if not m:
        return None
    return int(m.group(1)) / 100.0

def check_probability(df, prob_col, q_col, expected_fn, label):
    if prob_col not in df.columns:
        print(f"{label}: missing {prob_col}")
        return
    if q_col not in df.columns:
        print(f"{label}: missing {q_col}")
        return

    rows = []
    for i, row in df[[prob_col, q_col]].dropna().iterrows():
        q = q_to_float(row[q_col])
        if q is None:
            continue
        observed = pd.to_numeric(pd.Series([row[prob_col]]), errors="coerce").iloc[0]
        if pd.isna(observed):
            continue
        expected = expected_fn(q)
        if abs(float(observed) - float(expected)) > 1e-9:
            rows.append((i, row[q_col], float(observed), float(expected)))

    print(f"{label}: checked={len(df[[prob_col, q_col]].dropna())}, mismatches={len(rows)}")
    if rows:
        print("First mismatches:")
        for r in rows[:20]:
            print("  index=%s q=%s observed=%s expected=%s" % r)

def inspect_file(path):
    df = pd.read_parquet(path)
    print(f"\n=== {path} ===")

    checks = [
        (
            "da_buy_execution_probability",
            "da_buy_execution_probability_source_quantile",
            lambda q: q,
            "DA buy p_exec = q",
        ),
        (
            "da_sell_execution_probability",
            "da_sell_execution_probability_source_quantile",
            lambda q: 1.0 - q,
            "DA sell p_exec = 1-q",
        ),
        (
            "bcm_award_probability",
            "bcm_award_probability_source_quantile",
            lambda q: 1.0 - q,
            "BCM p_award = 1-q",
        ),
        (
            "bem_execution_probability",
            "bem_execution_probability_source_quantile",
            lambda q: 1.0 - q,
            "BEM p_exec = 1-q",
        ),
    ]

    any_found = False
    for prob_col, q_col, fn, label in checks:
        if prob_col in df.columns or q_col in df.columns:
            any_found = True
            check_probability(df, prob_col, q_col, fn, label)

    if not any_found:
        print("No probability diagnostic columns found.")

    # Also print available probability-related columns for debugging.
    prob_cols = [
        c for c in df.columns
        if "probability" in c.lower()
        or "p_award" in c.lower()
        or "p_exec" in c.lower()
        or "source_quantile" in c.lower()
    ]
    print("Probability-related columns:")
    for c in prob_cols[:100]:
        print(" ", c)

for p in sorted(ROOT.rglob("backtest_hourly.parquet")):
    inspect_file(p)

for p in sorted(ROOT.rglob("backtest_plan_history.parquet")):
    inspect_file(p)
