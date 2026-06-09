from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/simulation_runs")

EPS = 1e-9

def to_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def to_local(x):
    return to_utc(x).dt.tz_convert("Europe/Berlin")

def local_date(x):
    return to_local(x).dt.date

def nonzero_mask(df, cols):
    mask = pd.Series(False, index=df.index)
    used = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0).abs()
            mask |= s > EPS
            used.append(c)
    return mask, used

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def audit_da(df, path):
    delivery_col = "timestamp_utc"
    if delivery_col not in df.columns:
        return

    da_candidate_cols = [
        "da_precommit_candidate_buy_mw",
        "da_precommit_candidate_sell_mw",
        "da_precommit_candidate_buy_mwh",
        "da_precommit_candidate_sell_mwh",
        "da_precommit_selected_lockable_buy_mw",
        "da_precommit_selected_lockable_sell_mw",
        "da_precommit_selected_lockable_buy_mwh",
        "da_precommit_selected_lockable_sell_mwh",
        "da_selected_lockable_buy_mwh",
        "da_selected_lockable_sell_mwh",
    ]

    da_submitted_cols = [
        "submitted_da_buy_mw",
        "submitted_da_sell_mw",
        "submitted_da_buy_mwh",
        "submitted_da_sell_mwh",
    ]

    da_raw_optimizer_cols = [
        "raw_optimizer_plan_charge_mw",
        "raw_optimizer_plan_discharge_mw",
        "raw_optimizer_da_buy_mw",
        "raw_optimizer_da_sell_mw",
    ]

    candidate_mask, candidate_used = nonzero_mask(df, da_candidate_cols)
    submitted_mask, submitted_used = nonzero_mask(df, da_submitted_cols)
    raw_mask, raw_used = nonzero_mask(df, da_raw_optimizer_cols)

    snapshot_cols = [
        "da_originating_source_snapshot_utc",
        "da_precommit_source_snapshot_utc",
        "da_precommit_da_bid_source_snapshot_utc",
        "da_bid_source_snapshot_utc",
    ]
    snapshot_used = [c for c in snapshot_cols if c in df.columns]

    print(f"\n--- DA gate audit: {path} ---")
    print("candidate cols:", candidate_used)
    print("submitted cols:", submitted_used)
    print("raw optimizer cols:", raw_used)
    print("DA snapshot cols:", snapshot_used)

    if not snapshot_used:
        print("NO DA snapshot column found. Cannot prove DA gate causality from this file.")
        return

    delivery_local_date = local_date(df[delivery_col])

    for snap_col in snapshot_used:
        snap_local = to_local(df[snap_col])
        snap_local_date = snap_local.dt.date
        snap_hour = snap_local.dt.hour

        # DA next-day rule:
        # Any new DA candidate/submission created from snapshot D may only target delivery dates after D.
        same_or_past_delivery = delivery_local_date <= snap_local_date

        bad_candidate = candidate_mask & same_or_past_delivery
        bad_submitted = submitted_mask & same_or_past_delivery
        bad_raw = raw_mask & same_or_past_delivery

        print(f"\nDA using snapshot {snap_col}:")
        print("illegal same-day/past candidate rows:", int(bad_candidate.sum()))
        print("illegal same-day/past submitted rows:", int(bad_submitted.sum()))
        print("illegal same-day/past raw optimizer rows:", int(bad_raw.sum()))

        show_cols = [
            "timestamp_utc",
            snap_col,
            "da_precommit_selection_reason",
            "da_precommit_selected_incumbent",
            "da_precommit_candidate_feasible",
            "da_precommit_candidate_selection_pnl_eur",
            "da_precommit_selected_lockable_pnl_eur",
        ] + candidate_used + submitted_used + raw_used

        if bad_candidate.any():
            print("\nDA illegal candidate examples:")
            print(df.loc[bad_candidate, [c for c in show_cols if c in df.columns]].head(30).to_string(index=False))

        if bad_submitted.any():
            print("\nDA illegal submitted examples:")
            print(df.loc[bad_submitted, [c for c in show_cols if c in df.columns]].head(30).to_string(index=False))

        if bad_raw.any():
            print("\nDA illegal raw optimizer examples:")
            print(df.loc[bad_raw, [c for c in show_cols if c in df.columns]].head(30).to_string(index=False))

def audit_bcm(df, path):
    delivery_col = "timestamp_utc"
    if delivery_col not in df.columns:
        return

    product_col = first_existing(df, [
        "bcm_capacity_block_start_utc",
        "bcm_product_start_utc",
        "bcm_capacity_product_start_utc",
    ])

    if product_col is None:
        print(f"\n--- BCM gate audit: {path} ---")
        print("NO BCM product start column found.")
        return

    bcm_candidate_cols = [
        "bcm_candidate_pos_mw",
        "bcm_candidate_neg_mw",
        "bcm_precommit_candidate_pos_mw",
        "bcm_precommit_candidate_neg_mw",
    ]

    bcm_locked_cols = [
        "bcm_precommit_locked_pos_mw",
        "bcm_precommit_locked_neg_mw",
        "locked_bcm_capacity_pos_mw",
        "locked_bcm_capacity_neg_mw",
    ]

    bcm_submitted_cols = [
        "submitted_bcm_capacity_pos_mw",
        "submitted_bcm_capacity_neg_mw",
    ]

    # Do not include generic submitted_afrr_* here, because it may include BEM-only bids.
    candidate_mask, candidate_used = nonzero_mask(df, bcm_candidate_cols)
    locked_mask, locked_used = nonzero_mask(df, bcm_locked_cols)
    submitted_mask, submitted_used = nonzero_mask(df, bcm_submitted_cols)

    snapshot_cols = [
        "bcm_bid_source_snapshot_utc",
        "reserve_commitment_source_snapshot_utc",
    ]
    snapshot_used = [c for c in snapshot_cols if c in df.columns]

    print(f"\n--- BCM gate audit: {path} ---")
    print("product_col:", product_col)
    print("candidate cols:", candidate_used)
    print("locked cols:", locked_used)
    print("submitted cols:", submitted_used)
    print("BCM snapshot cols:", snapshot_used)

    if not snapshot_used:
        print("NO BCM snapshot column found. Cannot prove BCM gate causality from this file.")
        return

    product_local_date = local_date(df[product_col])

    for snap_col in snapshot_used:
        snap_local = to_local(df[snap_col])
        snap_local_date = snap_local.dt.date

        # BCM rule:
        # New BCM bids at day D gate may only target next-day products.
        # Therefore product local date must be > snapshot local date.
        same_or_past_product = product_local_date <= snap_local_date

        bad_candidate = candidate_mask & same_or_past_product
        bad_locked = locked_mask & same_or_past_product
        bad_submitted = submitted_mask & same_or_past_product

        print(f"\nBCM using snapshot {snap_col}:")
        print("illegal same-day/past candidate rows:", int(bad_candidate.sum()))
        print("illegal same-day/past locked rows:", int(bad_locked.sum()))
        print("illegal same-day/past submitted rows:", int(bad_submitted.sum()))

        show_cols = [
            "timestamp_utc",
            snap_col,
            product_col,
            "bcm_allowed",
            "bcm_zero_reason",
            "bcm_precommit_zero_reason",
            "reserve_retry_factor",
            "bcm_precommit_feasibility_pass",
        ] + candidate_used + locked_used + submitted_used

        if bad_candidate.any():
            print("\nBCM illegal candidate examples:")
            print(df.loc[bad_candidate, [c for c in show_cols if c in df.columns]].head(40).to_string(index=False))

        if bad_locked.any():
            print("\nBCM illegal locked examples:")
            print(df.loc[bad_locked, [c for c in show_cols if c in df.columns]].head(40).to_string(index=False))

        if bad_submitted.any():
            print("\nBCM illegal submitted examples:")
            print(df.loc[bad_submitted, [c for c in show_cols if c in df.columns]].head(40).to_string(index=False))

for path in sorted(ROOT.rglob("backtest_hourly.parquet")):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        continue

    audit_da(df, path)
    audit_bcm(df, path)

for path in sorted(ROOT.rglob("backtest_plan_history.parquet")):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        continue

    print(f"\n=== PLAN HISTORY: {path} ===")
    audit_da(df, path)
    audit_bcm(df, path)
