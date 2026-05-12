"""Run LP-based battery backtest from ML predictions + ground truth parquet files.

Usage (manifest-autoload, recommended):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --run-id 2026-04-20T15-36-58Z \
      --split test \
      --horizon-hours 48 \
      --reopt-step-hours 1 \
      --da-gate-hour-cet 11 \
      --soc-feedback-mode realized

Usage (explicit manifest path override):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --run-manifest artifacts/model_runs/latest.json \
      --split test

Usage (manual files):
    ./.venv/bin/python scripts/run_battery_backtest.py \
      --predictions artifacts/simulation_runs/manual/backtest_table_test.parquet \
      --ground-truth data/features/all_data_features.parquet \
      --timestamp-col timestamp_utc \
      --pred-da-col pred_da_price \
      --true-da-col da_price
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.simulation.battery_backtest import (
    BacktestColumnMap,
    BatteryBacktester,
    PhaseTimeoutError,
    load_and_align_market_data,
    load_prediction_warehouse_long,
)


def _phase_timeout_seconds(phase: str) -> float:
    key = "BACKTEST_PHASE_TIMEOUT_" + "".join(ch if ch.isalnum() else "_" for ch in phase.upper()) + "_S"
    if key in os.environ:
        return float(os.environ[key])
    return float(os.environ.get("BACKTEST_PHASE_TIMEOUT_S", "0"))


@contextmanager
def _phase_watchdog(phase: str):
    timeout_s = _phase_timeout_seconds(phase)
    t0 = time.monotonic()
    print(f"[PHASE] START {phase}")

    use_signal = (
        timeout_s > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    old_handler = None
    old_timer = None
    if use_signal:
        def _alarm_handler(_signum: int, _frame: object) -> None:
            raise PhaseTimeoutError(f"Phase timeout in '{phase}' after {timeout_s:.1f}s.")

        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
        signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        yield
    finally:
        if use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_timer is not None:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        dt = time.monotonic() - t0
        print(f"[PHASE] END {phase} | elapsed={dt:.2f}s")


def _plot_cumulative_pnl(hourly: pd.DataFrame, ts_col: str, out_path: Path) -> None:
    if hourly.empty:
        return
    d = hourly.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col)
    required = ["real_pnl_eur", "naive_pnl_eur", "oracle_pnl_eur"]
    if not set(required).issubset(d.columns):
        return
    d["model_cum_pnl_eur"] = pd.to_numeric(d["real_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["naive_cum_pnl_eur"] = pd.to_numeric(d["naive_pnl_eur"], errors="coerce").fillna(0.0).cumsum()
    d["oracle_cum_pnl_eur"] = pd.to_numeric(d["oracle_pnl_eur"], errors="coerce").fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d[ts_col], d["model_cum_pnl_eur"], label="Model", linewidth=2)
    ax.plot(d[ts_col], d["naive_cum_pnl_eur"], label="Naive 24h", linewidth=2)
    ax.plot(d[ts_col], d["oracle_cum_pnl_eur"], label="Oracle", linewidth=2)
    ax.set_title("Cumulative PnL Contribution")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative PnL [EUR]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_backtest_diagnostics(hourly: pd.DataFrame, summary: dict[str, object]) -> dict[str, object]:
    d = hourly.copy()
    numeric = d.select_dtypes(include=["number"])
    nonfinite_total = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    nan_total = int(numeric.isna().sum().sum()) if not numeric.empty else 0

    key_cols = [
        "real_pnl_eur",
        "pred_pnl_eur",
        "naive_pnl_eur",
        "oracle_pnl_eur",
        "soc_mwh",
        "charge_mw",
        "discharge_mw",
        "reserve_pos_mw",
        "reserve_neg_mw",
    ]
    key_col_nan_counts = {
        c: int(pd.to_numeric(d[c], errors="coerce").isna().sum())
        for c in key_cols
        if c in d.columns
    }

    infeasibility_flags = {
        "final_soc_constraint_satisfied": bool(summary.get("final_soc_constraint_satisfied", False)),
        "numeric_nonfinite_total": nonfinite_total,
    }
    return {
        "rows_hourly": int(len(d)),
        "numeric_nan_total": nan_total,
        "numeric_nonfinite_total": nonfinite_total,
        "key_column_nan_counts": key_col_nan_counts,
        "infeasibility_flags": infeasibility_flags,
    }


def _build_state_machine_audit(hourly: pd.DataFrame) -> dict[str, object]:
    d = hourly.copy()
    if d.empty:
        return {
            "rows": 0,
            "gate_09_reopt_triggers": 0,
            "gate_12_da_locked_rows": 0,
            "t25_energy_bid_rows": 0,
            "da_submitted_buy_mw_total": 0.0,
            "da_submitted_sell_mw_total": 0.0,
            "da_executed_buy_mw_total": 0.0,
            "da_executed_sell_mw_total": 0.0,
            "afrr_capacity_awarded_pos_mw_total": 0.0,
            "afrr_capacity_awarded_neg_mw_total": 0.0,
            "afrr_activation_accept_pos_count": 0,
            "afrr_activation_accept_neg_count": 0,
            "obligation_fulfilled_rate": float("nan"),
        }

    def _sum_col(name: str) -> float:
        return float(pd.to_numeric(d[name], errors="coerce").fillna(0.0).sum()) if name in d.columns else 0.0

    def _count_true(name: str) -> int:
        if name not in d.columns:
            return 0
        return int((pd.to_numeric(d[name], errors="coerce").fillna(0.0) > 0.5).sum())

    out: dict[str, object] = {
        "rows": int(len(d)),
        "gate_09_reopt_triggers": _count_true("event_reopt_triggered"),
        "gate_12_da_locked_rows": _count_true("da_bid_locked"),
        "t25_energy_bid_rows": _count_true("real_aFRR_Energy_Gate_Closure_Min"),
        "da_submitted_buy_mw_total": _sum_col("real_submitted_da_buy_mw"),
        "da_submitted_sell_mw_total": _sum_col("real_submitted_da_sell_mw"),
        "da_executed_buy_mw_total": _sum_col("real_executed_charge_mw"),
        "da_executed_sell_mw_total": _sum_col("real_executed_discharge_mw"),
        "afrr_capacity_awarded_pos_mw_total": _sum_col("real_executed_reserve_pos_mw"),
        "afrr_capacity_awarded_neg_mw_total": _sum_col("real_executed_reserve_neg_mw"),
        "afrr_activation_accept_pos_count": _count_true("real_afrr_act_pos_accepted"),
        "afrr_activation_accept_neg_count": _count_true("real_afrr_act_neg_accepted"),
    }
    if "real_Obligation_Fulfilled" in d.columns:
        out["obligation_fulfilled_rate"] = float(pd.to_numeric(d["real_Obligation_Fulfilled"], errors="coerce").fillna(0.0).mean())
    else:
        out["obligation_fulfilled_rate"] = float("nan")
    return out


def _resolve_out_dir(
    out_dir_arg: str,
    *,
    run_id: str | None,
    split: str,
) -> Path:
    if out_dir_arg.strip():
        out = Path(out_dir_arg)
        out.mkdir(parents=True, exist_ok=True)
        return out
    rid = run_id or "manual"
    out = Path("artifacts/simulation_runs") / rid / split
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_quantile_token(token: str) -> str:
    t = token.strip().lower()
    if not t:
        raise ValueError("Empty quantile token.")
    if t.startswith("p") and len(t) == 3 and t[1:].isdigit():
        q = int(t[1:])
        if q <= 0 or q >= 100:
            raise ValueError(f"Quantile out of range: {token}")
        return f"p{q:02d}"
    val = float(t)
    if val <= 0.0 or val >= 1.0:
        raise ValueError(f"Quantile out of range: {token}")
    q = int(round(val * 100.0))
    q = max(1, min(99, q))
    return f"p{q:02d}"


def _parse_quantile_pairs(raw: str) -> list[tuple[str, str]]:
    s = raw.strip()
    if not s:
        return []
    out: list[tuple[str, str]] = []
    for part in s.split(","):
        pair = part.strip()
        if not pair:
            continue
        if "-" not in pair:
            raise ValueError(f"Invalid quantile pair '{pair}'. Use p10-p90 or 0.1-0.9.")
        lo_raw, hi_raw = pair.split("-", 1)
        q_lo = _parse_quantile_token(lo_raw)
        q_hi = _parse_quantile_token(hi_raw)
        if int(q_lo[1:]) > int(q_hi[1:]):
            raise ValueError(f"Invalid pair '{pair}': low quantile must be <= high quantile.")
        out.append((q_lo, q_hi))
    return out


def _apply_quantile_pair_to_warehouse(
    warehouse: dict[str, pd.DataFrame],
    *,
    q_low: str,
    q_high: str,
    da_role: str,
) -> dict[str, pd.DataFrame]:
    da_q = {"low": q_low, "high": q_high, "mid": "p50"}[da_role]
    quantile_by_target = {
        "pred_da_price": da_q,
        "pred_afrr_capacity_price_pos": q_high,
        "pred_afrr_activation_price_pos": q_high,
        "pred_afrr_activation_rate_pos": q_high,
        "pred_afrr_capacity_price_neg": q_low,
        "pred_afrr_activation_price_neg": q_low,
        "pred_afrr_activation_rate_neg": q_low,
    }
    out: dict[str, pd.DataFrame] = {}
    for pred_col, df in warehouse.items():
        q_col = quantile_by_target.get(pred_col, "p50")
        cur = df.copy()
        if q_col not in cur.columns:
            available = [c for c in cur.columns if re.fullmatch(r"p\d{2}", str(c))]
            raise KeyError(
                f"Requested quantile '{q_col}' missing for {pred_col}. Available: {available}"
            )
        cur["predicted_value"] = pd.to_numeric(cur[q_col], errors="coerce")
        out[pred_col] = cur
    return out


def _scenario_suffix(q_low: str, q_high: str) -> str:
    return f"{q_low}_{q_high}"


def _matches_model_key(path: Path, model_key: str) -> bool:
    if not model_key:
        return True
    mk = model_key.strip().lower()
    name = path.name.lower()
    if mk in {"xgb", "xgboost"}:
        return "xgboost" in name
    if mk == "tft":
        return "xgboost" not in name
    return mk in name


def _score_long_candidate(path: Path, *, split: str) -> tuple[int, int]:
    """Higher score = better split match for long prediction files."""
    name = path.name.lower()
    has_test = "test" in name
    # Prefer explicit split markers first.
    if split == "test":
        return (2 if has_test else 1, len(name))
    # val: prefer files without test marker; many val files have no explicit "val" token
    return (2 if not has_test else 1, len(name))


def _resolve_existing_file(path_like: str | Path, *, manifest_dir: Path) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    # Common case for downloaded artifacts: manifest keeps server-absolute path.
    cands = [
        manifest_dir / p.name,
        manifest_dir / "predictions" / p.name,
        Path.cwd() / p.name,
        Path.cwd() / "data" / "features" / p.name,
        repo_root / "data" / "features" / p.name,
        repo_root / "data" / "features" / "all_data_features.parquet",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"File not found from manifest path '{p}'. Tried local fallbacks near {manifest_dir}.")


def _resolve_long_prediction_path(
    *,
    pred_col: str,
    configured_path: str | Path,
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> Path:
    p = Path(configured_path)
    if p.exists() and _matches_model_key(p, model_key):
        return p

    pred_dir = manifest_dir / "predictions"
    candidates: list[Path] = []
    if pred_dir.exists():
        patterns = [
            f"*{split}*{pred_col}*long*.parquet",
            f"*{pred_col}*long*{split}*.parquet",
            f"*{pred_col}*{split}*long*.parquet",
            f"*{pred_col}*long*.parquet",
        ]
        for pat in patterns:
            candidates.extend(sorted(pred_dir.glob(pat)))
        # Deduplicate while preserving order
        if candidates:
            seen: set[str] = set()
            deduped: list[Path] = []
            for c in candidates:
                key = str(c.resolve())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(c)
            candidates = deduped
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if candidates:
        candidates = sorted(candidates, key=lambda c: _score_long_candidate(c, split=split), reverse=True)
        return candidates[0]

    # Final explicit fallbacks by filename
    for c in [manifest_dir / p.name, manifest_dir / "predictions" / p.name]:
        if c.exists() and _matches_model_key(c, model_key):
            return c
    raise FileNotFoundError(
        f"Could not resolve long prediction file for pred_col='{pred_col}', split='{split}', model_key='{model_key}'. "
        f"Configured path: {p}"
    )


def _resolve_long_map(
    *,
    long_map: dict[str, str],
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for pred_col, p in long_map.items():
        rp = _resolve_long_prediction_path(
            pred_col=pred_col,
            configured_path=p,
            manifest_dir=manifest_dir,
            split=split,
            model_key=model_key,
        )
        resolved[pred_col] = str(rp)
    return resolved


def _read_parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        # Fallback if pyarrow schema access is unavailable.
        return list(pd.read_parquet(path).columns)


def _preflight_manifest_and_quantiles(
    *,
    manifest_path: Path,
    manifest_payload: dict[str, object],
    split: str,
    model_key: str,
    manifest_dir: Path,
    expected_quantiles: set[str],
) -> None:
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    bundles = manifest_payload.get("bundles", {}) if isinstance(manifest_payload, dict) else {}
    da_long = bundles.get("da", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    afrr_long = bundles.get("afrr", {}).get("predictions_long", {}).get(split, {}) if isinstance(bundles, dict) else {}
    long_map = {**da_long, **afrr_long}
    if not long_map:
        raise RuntimeError(
            "Preflight failed: no long-format prediction files in manifest. "
            "Quantile simulation requires predictions_long entries."
        )

    resolved = _resolve_long_map(
        long_map=long_map,
        manifest_dir=manifest_dir,
        split=split,
        model_key=model_key,
    )
    expected = {str(q).lower() for q in expected_quantiles}
    if not expected:
        expected = {"p50"}
    failures: list[str] = []
    for pred_col, file_path in sorted(resolved.items()):
        cols = set(_read_parquet_columns(Path(file_path)))
        missing = sorted(expected - cols)
        if missing:
            failures.append(
                f"{pred_col}: missing {len(missing)} required quantile columns "
                f"(first 12: {missing[:12]}) in {file_path}"
            )
    if failures:
        msg = (
            "Preflight failed: required quantiles derived from --quantile-pairs are not fully available.\n"
            f"Expected quantiles: {sorted(expected)}\n"
            + "\n".join(failures)
        )
        raise RuntimeError(msg)


def _resolve_bundle_prediction_path(
    *,
    configured_path: str | Path,
    manifest_dir: Path,
    split: str,
    model_key: str,
) -> Path:
    """Resolve wide prediction parquet for bundle fallback mode."""
    p = Path(configured_path)
    if p.exists() and _matches_model_key(p, model_key):
        return p

    pred_dir = manifest_dir / "predictions"
    candidates: list[Path] = []
    if pred_dir.exists():
        patterns = [
            f"*{split}*.parquet",
            "*.parquet",
        ]
        for pat in patterns:
            candidates.extend(sorted(pred_dir.glob(pat)))
        candidates = [c for c in candidates if "long" not in c.name.lower()]
    if model_key:
        candidates = [c for c in candidates if _matches_model_key(c, model_key)]
    if candidates:
        candidates = sorted(candidates, key=lambda c: _score_long_candidate(c, split=split), reverse=True)
        # Prefer same bundle family where possible.
        name_hint = p.name.lower()
        if "da" in name_hint:
            da_cands = [c for c in candidates if "da" in c.name.lower()]
            if da_cands:
                return da_cands[0]
        if "afrr" in name_hint:
            afrr_cands = [c for c in candidates if "afrr" in c.name.lower()]
            if afrr_cands:
                return afrr_cands[0]
        return candidates[0]

    for c in [manifest_dir / p.name, manifest_dir / "predictions" / p.name]:
        if c.exists() and _matches_model_key(c, model_key):
            return c
    raise FileNotFoundError(
        f"Could not resolve prediction parquet for split='{split}', model_key='{model_key}'. "
        f"Configured path: {p}"
    )


def _apply_fallback_column_map(pred: pd.DataFrame, truth: pd.DataFrame, colmap: BacktestColumnMap) -> BacktestColumnMap:
    """Use project-aware fallback candidates to reduce manual mapping overhead."""

    def pick(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        for c in [primary, *candidates]:
            if c in frame.columns:
                return c
        raise KeyError(f"Missing required column. Tried: {[primary, *candidates]}")

    def pick_pred(frame: pd.DataFrame, primary: str, candidates: list[str]) -> str:
        # In long-warehouse mode pred preview can be empty. Keep configured names.
        if frame.empty:
            return primary
        return pick(frame, primary, candidates)

    mapped = BacktestColumnMap(
        timestamp=pick(pred if colmap.timestamp in pred.columns else truth, colmap.timestamp, ["timestamp", "datetime", "date"]),

        pred_da_price=pick_pred(
            pred,
            colmap.pred_da_price,
            ["da_price_pred", "y_pred_da_price", "prediction_da_price", "pred_target_da_price"],
        ),
        pred_afrr_capacity_price_pos=pick_pred(pred, colmap.pred_afrr_capacity_price_pos, ["afrr_capacity_price_pos_pred", "pred_target_afrr_capacity_price_pos"]),
        pred_afrr_capacity_price_neg=pick_pred(pred, colmap.pred_afrr_capacity_price_neg, ["afrr_capacity_price_neg_pred", "pred_target_afrr_capacity_price_neg"]),
        pred_afrr_activation_price_pos=pick_pred(pred, colmap.pred_afrr_activation_price_pos, ["afrr_activation_price_vwap_pos_pred", "pred_target_afrr_activation_price_vwap_pos"]),
        pred_afrr_activation_price_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_price_neg,
            ["afrr_activation_price_vwap_neg_pred", "pred_target_afrr_activation_price_vwap_neg"],
        ),
        pred_afrr_activation_rate_pos=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_pos,
            ["pred_target_afrr_activation_rate_pos", "afrr_activation_rate_pred"],
        ),
        pred_afrr_activation_rate_neg=pick_pred(
            pred,
            colmap.pred_afrr_activation_rate_neg,
            ["pred_target_afrr_activation_rate_neg", "afrr_activation_rate_pred"],
        ),

        true_da_price=pick(truth, colmap.true_da_price, ["da_price_actual", "target_da_price"]),
        true_afrr_capacity_price_pos=pick(truth, colmap.true_afrr_capacity_price_pos, ["target_afrr_capacity_price_pos"]),
        true_afrr_capacity_price_neg=pick(truth, colmap.true_afrr_capacity_price_neg, ["target_afrr_capacity_price_neg"]),
        true_afrr_activation_price_pos=pick(
            truth,
            colmap.true_afrr_activation_price_pos,
            [
                "target_afrr_activation_price_vwap_pos_raw",
                "target_afrr_activation_price_vwap_pos",
                "afrr_activation_price_vwap",
            ],
        ),
        true_afrr_activation_price_neg=pick(
            truth,
            colmap.true_afrr_activation_price_neg,
            [
                "target_afrr_activation_price_vwap_neg_raw",
                "target_afrr_activation_price_vwap_neg",
                "afrr_activation_price_vwap",
                "afrr_activation_price_vwap_pos",
                "target_afrr_activation_price_vwap_pos_raw",
                "target_afrr_activation_price_vwap_pos",
            ],
        ),
        true_afrr_activation_rate_pos=pick(
            truth,
            colmap.true_afrr_activation_rate_pos,
            [
                "activation_rate_phys_pos",
                "afrr_activation_rate_pos",
                "target_afrr_activation_rate_pos",
                "afrr_activation_rate",
            ],
        ),
        true_afrr_activation_rate_neg=pick(
            truth,
            colmap.true_afrr_activation_rate_neg,
            [
                "activation_rate_phys_neg",
                "afrr_activation_rate_neg",
                "target_afrr_activation_rate_neg",
                "afrr_activation_rate",
            ],
        ),
    )
    if mapped.true_afrr_activation_price_neg == mapped.true_afrr_activation_price_pos:
        print(
            "[WARN] Missing dedicated negative activation-price truth column. "
            "Using positive activation-price column as fallback for *_neg settlement."
        )
    return mapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run battery LP backtest with predicted-vs-realized settlement.")
    p.add_argument("--predictions", default="", help="Path to predictions parquet file.")
    p.add_argument("--ground-truth", default="", help="Path to ground-truth parquet file.")
    p.add_argument(
        "--run-id",
        default="2026-04-20T15-36-58Z",
        help=(
            "Run id used to resolve manifest path when --run-manifest is not set. "
            "Default: 2026-04-20T15-36-58Z"
        ),
    )
    p.add_argument(
        "--run-manifest",
        "--manifest",
        default="",
        help=(
            "Optional manifest path or latest-pointer json for simulation autoload. "
            "If omitted, resolves to artifacts/model_runs/<run-id>/manifest.json."
        ),
    )
    p.add_argument("--split", choices=["val", "test"], default="test", help="Prediction split for manifest mode.")
    p.add_argument(
        "--model-key",
        default="",
        help="Optional model selector when one run dir contains multiple models (e.g. 'xgboost' or 'tft').",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for hourly/aggregated results. If empty, uses artifacts/simulation_runs/<run_id>/<split>/",
    )
    p.add_argument(
        "--quantile-pairs",
        default="",
        help=(
            "Optional comma-separated sweep list, e.g. "
            "'p50-p50,p30-p70,p10-p90' or '0.5-0.5,0.3-0.7'. "
            "Requires long-format prediction warehouse."
        ),
    )
    p.add_argument(
        "--da-quantile-role",
        choices=["low", "mid", "high"],
        default="mid",
        help="How DA uses quantile pair in sweep mode: low/high/mid (mid -> p50).",
    )

    p.add_argument("--timestamp-col", default="timestamp_utc")
    p.add_argument("--pred-da-col", default="pred_da_price")
    p.add_argument("--pred-cap-pos-col", default="pred_afrr_capacity_price_pos")
    p.add_argument("--pred-cap-neg-col", default="pred_afrr_capacity_price_neg")
    p.add_argument("--pred-act-pos-col", default="pred_afrr_activation_price_pos")
    p.add_argument("--pred-act-neg-col", default="pred_afrr_activation_price_neg")
    p.add_argument("--pred-rate-pos-col", default="pred_afrr_activation_rate_pos")
    p.add_argument("--pred-rate-neg-col", default="pred_afrr_activation_rate_neg")

    p.add_argument("--true-da-col", default="da_price")
    p.add_argument("--true-cap-pos-col", default="afrr_capacity_price_pos")
    p.add_argument("--true-cap-neg-col", default="afrr_capacity_price_neg")
    p.add_argument("--true-act-pos-col", default="afrr_activation_price_vwap_pos")
    p.add_argument("--true-act-neg-col", default="afrr_activation_price_vwap_neg")
    p.add_argument("--true-rate-pos-col", default="activation_rate_phys_pos")
    p.add_argument("--true-rate-neg-col", default="activation_rate_phys_neg")

    p.add_argument("--start", default=None, help="Optional UTC start filter.")
    p.add_argument("--end", default=None, help="Optional UTC end filter.")
    p.add_argument("--horizon-hours", type=int, default=48, help="Rolling-horizon window length in hours.")
    p.add_argument("--reopt-step-hours", type=int, default=1, help="Re-optimization step in hours.")
    p.add_argument(
        "--da-gate-hour-cet",
        type=int,
        default=12,
        help="Day-Ahead gate-closure hour in CET/CEST used for locking next-day DA bids (default: 12).",
    )
    p.add_argument(
        "--da-gate-hour-utc",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--soc-feedback-mode",
        choices=["realized", "predicted"],
        default="realized",
        help="State carryover mode between re-optimizations.",
    )
    p.add_argument(
        "--enforce-final-soc-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce final SoC >= soc_target_end (default: enabled).",
    )
    p.add_argument(
        "--disable-rolling-horizon",
        action="store_true",
        help="Use a single full-horizon optimization instead of rolling horizon.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id: str | None = args.run_id.strip() or None
    quantile_pairs = _parse_quantile_pairs(args.quantile_pairs)
    required_quantiles: set[str] = {"p50"}
    for q_lo, q_hi in quantile_pairs:
        required_quantiles.add(q_lo.lower())
        required_quantiles.add(q_hi.lower())

    predictions_path = args.predictions.strip()
    ground_truth_path = args.ground_truth.strip()
    payload: dict[str, object] = {}
    manifest_path: Path | None = None

    if not predictions_path:
        if args.run_manifest.strip():
            manifest_path = Path(args.run_manifest.strip())
        else:
            if not run_id:
                raise ValueError("Missing run id. Provide --run-id or --run-manifest.")
            manifest_path = Path("artifacts/model_runs") / run_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Run manifest/latest pointer not found: {manifest_path}. "
                "Pass --run-id <RUN_ID> or --run-manifest <PATH>."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "manifest_path" in payload:
            run_id = payload.get("run_id")
            manifest_path = Path(payload["manifest_path"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = run_id or payload.get("run_id") or (args.run_id.strip() or None)
        manifest_dir = manifest_path.parent

        # Strict fail-fast preflight for thesis reproducibility:
        # verify manifest and complete P01..P99 quantile grid before simulation allocation.
        _preflight_manifest_and_quantiles(
            manifest_path=manifest_path,
            manifest_payload=payload,
            split=args.split,
            model_key=args.model_key.strip(),
            manifest_dir=manifest_dir,
            expected_quantiles=required_quantiles,
        )
    else:
        manifest_dir = Path.cwd()

    out_dir = _resolve_out_dir(args.out_dir, run_id=run_id, split=args.split)
    forecast_warehouse: dict[str, pd.DataFrame] | None = None
    coverage_min: pd.Timestamp | None = None
    coverage_max: pd.Timestamp | None = None

    if not predictions_path:
        da_long = payload.get("bundles", {}).get("da", {}).get("predictions_long", {}).get(args.split, {})
        afrr_long = payload.get("bundles", {}).get("afrr", {}).get("predictions_long", {}).get(args.split, {})
        long_map = {**da_long, **afrr_long}
        if long_map:
            resolved_long_map = _resolve_long_map(
                long_map=long_map,
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            forecast_warehouse = load_prediction_warehouse_long(resolved_long_map)
            print(f"[INFO] Long-format forecast warehouse loaded for split='{args.split}' with {len(long_map)} files.")
            cov_min_list: list[pd.Timestamp] = []
            cov_max_list: list[pd.Timestamp] = []
            for wdf in forecast_warehouse.values():
                t = pd.to_datetime(wdf["target_time_utc"], utc=True, errors="coerce").dropna()
                if not t.empty:
                    cov_min_list.append(t.min())
                    cov_max_list.append(t.max())
            if cov_min_list and cov_max_list:
                coverage_min = min(cov_min_list)
                coverage_max = max(cov_max_list)
        else:
            da_pred = _resolve_bundle_prediction_path(
                configured_path=payload["bundles"]["da"]["predictions"][args.split],
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            afrr_pred = _resolve_bundle_prediction_path(
                configured_path=payload["bundles"]["afrr"]["predictions"][args.split],
                manifest_dir=manifest_dir,
                split=args.split,
                model_key=args.model_key.strip(),
            )
            predictions_path = str((out_dir / f"backtest_table_{args.split}.parquet").resolve())
            da_df = pd.read_parquet(da_pred)
            afrr_df = pd.read_parquet(afrr_pred)
            backtest_table = da_df.merge(afrr_df, on="timestamp_utc", how="inner")
            backtest_table.to_parquet(predictions_path, index=False)
            print(f"[INFO] Backtest table created: {predictions_path}")

        if not ground_truth_path:
            ground_truth_path = str(_resolve_existing_file(payload["ground_truth"]["default_path"], manifest_dir=manifest_dir))

    if not ground_truth_path:
        raise ValueError("Either provide --predictions and --ground-truth, or use --run-manifest.")
    truth_preview = pd.read_parquet(ground_truth_path)
    pred_preview = pd.read_parquet(predictions_path) if predictions_path else pd.DataFrame()

    colmap_in = BacktestColumnMap(
        timestamp=args.timestamp_col,
        pred_da_price=args.pred_da_col,
        pred_afrr_capacity_price_pos=args.pred_cap_pos_col,
        pred_afrr_capacity_price_neg=args.pred_cap_neg_col,
        pred_afrr_activation_price_pos=args.pred_act_pos_col,
        pred_afrr_activation_price_neg=args.pred_act_neg_col,
        pred_afrr_activation_rate_pos=args.pred_rate_pos_col,
        pred_afrr_activation_rate_neg=args.pred_rate_neg_col,
        true_da_price=args.true_da_col,
        true_afrr_capacity_price_pos=args.true_cap_pos_col,
        true_afrr_capacity_price_neg=args.true_cap_neg_col,
        true_afrr_activation_price_pos=args.true_act_pos_col,
        true_afrr_activation_price_neg=args.true_act_neg_col,
        true_afrr_activation_rate_pos=args.true_rate_pos_col,
        true_afrr_activation_rate_neg=args.true_rate_neg_col,
    )
    colmap = _apply_fallback_column_map(pred_preview, truth_preview, colmap_in)

    if predictions_path:
        df = load_and_align_market_data(predictions_path, ground_truth_path, colmap)
    else:
        df = truth_preview.copy()
        if colmap.timestamp not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df[colmap.timestamp] = df.index
        df[colmap.timestamp] = pd.to_datetime(df[colmap.timestamp], utc=True, errors="coerce")
        df = df.dropna(subset=[colmap.timestamp]).sort_values(colmap.timestamp).reset_index(drop=True)
        if forecast_warehouse and coverage_min is not None and coverage_max is not None:
            df = df[(df[colmap.timestamp] >= coverage_min) & (df[colmap.timestamp] <= coverage_max)].copy()
    if args.start:
        df = df[df[colmap.timestamp] >= pd.to_datetime(args.start, utc=True)].copy()
    if args.end:
        df = df[df[colmap.timestamp] <= pd.to_datetime(args.end, utc=True)].copy()
    if df.empty:
        raise ValueError("No rows after timestamp filtering.")

    scenarios: list[tuple[str, dict[str, pd.DataFrame] | None]] = [("default", forecast_warehouse)]
    if quantile_pairs:
        if forecast_warehouse is None:
            raise ValueError("--quantile-pairs requires long-format predictions from --run-manifest.")
        scenarios = []
        for q_low, q_high in quantile_pairs:
            name = _scenario_suffix(q_low, q_high)
            wh = _apply_quantile_pair_to_warehouse(
                forecast_warehouse,
                q_low=q_low,
                q_high=q_high,
                da_role=args.da_quantile_role,
            )
            scenarios.append((name, wh))

    backtester = BatteryBacktester()
    sweep_rows: list[dict[str, object]] = []
    for scenario_name, scenario_warehouse in scenarios:
        scenario_out_dir = out_dir if scenario_name == "default" and not quantile_pairs else out_dir / scenario_name
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        with _phase_watchdog("backtester_run"):
            outputs = backtester.run(
                df,
                colmap,
                use_rolling_horizon=not args.disable_rolling_horizon,
                horizon_hours=args.horizon_hours,
                reopt_step_hours=args.reopt_step_hours,
                forecast_warehouse=scenario_warehouse,
                da_gate_hour_cet=args.da_gate_hour_cet if args.da_gate_hour_utc is None else args.da_gate_hour_utc,
                soc_feedback_mode=args.soc_feedback_mode,
                enforce_final_soc_min=args.enforce_final_soc_min,
            )

        hourly_path = scenario_out_dir / "backtest_hourly.parquet"
        planned_ledger_path = scenario_out_dir / "planned_ledger.parquet"
        executed_ledger_path = scenario_out_dir / "executed_ledger.parquet"
        realized_ledger_path = scenario_out_dir / "realized_ledger.parquet"
        plan_history_path = scenario_out_dir / "backtest_plan_history.parquet"
        milp_event_log_path = scenario_out_dir / "backtest_milp_event_log.parquet"
        milp_event_summary_path = scenario_out_dir / "backtest_milp_event_summary.csv"
        volatility_path = scenario_out_dir / "backtest_decision_volatility.csv"
        monthly_path = scenario_out_dir / "backtest_monthly.csv"
        yearly_path = scenario_out_dir / "backtest_yearly.csv"
        summary_path = scenario_out_dir / "backtest_summary.json"
        state_machine_audit_path = scenario_out_dir / "state_machine_audit.json"
        diagnostics_path = scenario_out_dir / "backtest_diagnostics.json"
        diagnostics_txt_path = scenario_out_dir / "backtest_diagnostics.txt"
        oracle_paradox_path = scenario_out_dir / "oracle_paradox_hours.csv"
        pnl_plot_path = scenario_out_dir / "backtest_cumulative_pnl.png"

        with _phase_watchdog("write_hourly"):
            outputs.hourly.to_parquet(hourly_path, index=False)
        planned_cols = [c for c in [
            colmap.timestamp, "charge_mw", "discharge_mw", "reserve_pos_mw", "reserve_neg_mw", "soc_lp_mwh",
            "planned_soc_mwh", "aFRR_Capacity_Won_Pos_MW", "aFRR_Capacity_Won_Neg_MW", "aFRR_Capacity_Won_MW",
            "aFRR_Energy_Price_EUR_MWh_Pos", "aFRR_Energy_Price_EUR_MWh_Neg", "event_reopt_triggered",
            "event_reopt_rejected_mw_total", "predicted_objective_eur"
        ] if c in outputs.hourly.columns]
        with _phase_watchdog("write_planned_ledger"):
            outputs.hourly[planned_cols].to_parquet(planned_ledger_path, index=False)

        executed_cols = [colmap.timestamp]
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_planned_")])
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_submitted_")])
        executed_cols.extend([c for c in outputs.hourly.columns if c.startswith("real_executed_")])
        executed_cols.extend([c for c in [
            "real_da_buy_accepted", "real_da_sell_accepted", "real_afrr_cap_pos_awarded", "real_afrr_cap_neg_awarded",
            "real_afrr_act_pos_accepted", "real_afrr_act_neg_accepted", "real_aFRR_Capacity_Won_MW",
            "real_DA_Energy_Sold_MW", "real_aFRR_Energy_Price_EUR_MWh", "real_Obligation_Fulfilled",
            "real_aFRR_Energy_Gate_Closure_Min", "shock_source", "soc_before_mwh", "soc_after_planned_mwh",
            "soc_after_executed_mwh", "soc_shock_mwh"
        ] if c in outputs.hourly.columns])
        with _phase_watchdog("write_executed_ledger"):
            outputs.hourly[[*dict.fromkeys(executed_cols)]].to_parquet(executed_ledger_path, index=False)

        realized_cols = [colmap.timestamp]
        realized_cols.extend([
            c for c in outputs.hourly.columns if c.startswith("real_") and c in {
                "real_pnl_eur", "real_da_buy_mwh", "real_da_sell_mwh", "real_act_pos_mwh", "real_act_neg_mwh",
                "real_revenue_da_eur", "real_cost_da_eur", "real_revenue_capacity_eur", "real_revenue_activation_eur",
                "real_transaction_cost_eur", "real_degradation_cost_eur", "real_penalty_eur",
                "real_missed_activation_mwh", "real_missed_capacity_mw", "real_missed_capacity_pos_mw",
                "real_missed_capacity_neg_mw", "real_requested_activation_revenue_eur",
                "real_delivered_activation_revenue_eur", "real_missed_activation_revenue_eur", "real_soc_mwh",
            }
        ])
        with _phase_watchdog("write_realized_ledger"):
            outputs.hourly[[*dict.fromkeys(realized_cols)]].to_parquet(realized_ledger_path, index=False)

        with _phase_watchdog("write_plan_history"):
            outputs.plan_history.to_parquet(plan_history_path, index=False)
        with _phase_watchdog("write_milp_event_log"):
            if "milp_event_type" in outputs.plan_history.columns:
                ev_cols = [c for c in outputs.plan_history.columns if c.startswith("ev_")]
                reserve_bin_cols = [c for c in outputs.plan_history.columns if c.startswith("reserve_pos_bin_") or c.startswith("reserve_neg_bin_")]
                keep_cols = [
                    "snapshot_time_utc",
                    "target_time_utc",
                    "lead_time_h",
                    "milp_event_type",
                    "charge_mw",
                    "discharge_mw",
                    "reserve_pos_mw",
                    "reserve_neg_mw",
                    "slack_pos_mw",
                    "slack_neg_mw",
                    "predicted_objective_eur",
                    "ev_objective_rebuild_eur",
                    *reserve_bin_cols,
                    *ev_cols,
                ]
                keep_cols = list(dict.fromkeys(c for c in keep_cols if c in outputs.plan_history.columns))
                event_log = outputs.plan_history.loc[
                    outputs.plan_history["milp_event_type"].astype(str).ne("none"),
                    keep_cols,
                ].copy()
                if not event_log.empty:
                    event_log["snapshot_time_utc"] = pd.to_datetime(event_log["snapshot_time_utc"], utc=True, errors="coerce")
                    event_log["snapshot_date_utc"] = event_log["snapshot_time_utc"].dt.date.astype(str)
                    event_log.to_parquet(milp_event_log_path, index=False)
                    summary = (
                        event_log.groupby(["snapshot_time_utc", "snapshot_date_utc", "milp_event_type"], dropna=False)
                        .agg(
                            hours_covered=("target_time_utc", "count"),
                            ev_da_charge_eur=("ev_da_charge_eur", "sum"),
                            ev_da_discharge_eur=("ev_da_discharge_eur", "sum"),
                            ev_afrr_pos_eur=("ev_afrr_pos_eur", "sum"),
                            ev_afrr_neg_eur=("ev_afrr_neg_eur", "sum"),
                            ev_slack_penalty_pos_eur=("ev_slack_penalty_pos_eur", "sum"),
                            ev_slack_penalty_neg_eur=("ev_slack_penalty_neg_eur", "sum"),
                            ev_terminal_soc_credit_eur=("ev_terminal_soc_credit_eur", "sum"),
                            ev_objective_rebuild_eur=("ev_objective_rebuild_eur", "sum"),
                            predicted_objective_eur=("predicted_objective_eur", "mean"),
                            avg_pred_da_price_eur_mwh=("ev_pred_da_price_eur_mwh", "mean"),
                            avg_pred_act_rate_pos=("ev_pred_act_rate_pos", "mean"),
                            avg_pred_act_rate_neg=("ev_pred_act_rate_neg", "mean"),
                            avg_pred_cap_pos_eur_mw=("ev_pred_cap_pos_eur_mw", "mean"),
                            avg_pred_cap_neg_eur_mw=("ev_pred_cap_neg_eur_mw", "mean"),
                        )
                        .reset_index()
                        .sort_values(["snapshot_time_utc", "milp_event_type"])
                    )
                    summary.to_csv(milp_event_summary_path, index=False)
        with _phase_watchdog("write_volatility"):
            outputs.volatility.to_csv(volatility_path, index=False)
        with _phase_watchdog("write_monthly"):
            outputs.monthly.to_csv(monthly_path, index=False)
        with _phase_watchdog("write_yearly"):
            outputs.yearly.to_csv(yearly_path, index=False)
        with _phase_watchdog("write_summary_json"):
            summary_path.write_text(json.dumps(outputs.summary, indent=2), encoding="utf-8")
        with _phase_watchdog("write_state_machine_audit"):
            state_machine_audit_path.write_text(json.dumps(_build_state_machine_audit(outputs.hourly), indent=2), encoding="utf-8")
        with _phase_watchdog("build_and_write_diagnostics"):
            diagnostics = _build_backtest_diagnostics(outputs.hourly, outputs.summary)
            def _fmt_summary_val(key: str) -> str:
                v = outputs.summary.get(key, float("nan"))
                try:
                    fv = float(v)
                except Exception:
                    return str(v)
                return f"{fv:.2f}" if pd.notna(fv) else "nan"
            ts_col = colmap.timestamp if colmap.timestamp in outputs.hourly.columns else None
            if ts_col is not None and not outputs.hourly.empty:
                _ts = pd.to_datetime(outputs.hourly[ts_col], utc=True, errors="coerce").dropna()
            else:
                _ts = pd.Series(dtype="datetime64[ns, UTC]")
            if len(_ts) > 0:
                timeframe_start = _ts.min()
                timeframe_end = _ts.max()
                num_days_total = float((timeframe_end - timeframe_start).total_seconds() / 86400.0) + (1.0 / 24.0)
                timeframe_start_txt = timeframe_start.isoformat()
                timeframe_end_txt = timeframe_end.isoformat()
            else:
                num_days_total = float("nan")
                timeframe_start_txt = "n/a"
                timeframe_end_txt = "n/a"
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            diagnostics_txt_path.write_text(
                "\n".join([
                    "Backtest Diagnostics",
                    f"timeframe_start_utc={timeframe_start_txt}",
                    f"timeframe_end_utc={timeframe_end_txt}",
                    f"timeframe_total_days={num_days_total:.4f}" if pd.notna(num_days_total) else "timeframe_total_days=nan",
                    f"rows_hourly={diagnostics['rows_hourly']}",
                    f"numeric_nan_total={diagnostics['numeric_nan_total']}",
                    f"numeric_nonfinite_total={diagnostics['numeric_nonfinite_total']}",
                    f"final_soc_constraint_satisfied={diagnostics['infeasibility_flags']['final_soc_constraint_satisfied']}",
                    "",
                    "PnL Summary",
                    f"realized_total_pnl_eur={_fmt_summary_val('realized_total_pnl_eur')}",
                    f"oracle_total_pnl_eur={_fmt_summary_val('oracle_total_pnl_eur')}",
                    f"realized_da_only_total_pnl_eur={_fmt_summary_val('realized_da_only_total_pnl_eur')}",
                    f"oracle_da_only_total_pnl_eur={_fmt_summary_val('oracle_da_only_total_pnl_eur')}",
                    f"realized_afrr_only_total_pnl_eur={_fmt_summary_val('realized_afrr_only_total_pnl_eur')}",
                    f"oracle_afrr_only_total_pnl_eur={_fmt_summary_val('oracle_afrr_only_total_pnl_eur')}",
                ]) + "\n",
                encoding="utf-8",
            )
        with _phase_watchdog("write_oracle_paradox_report"):
            hp = outputs.hourly.copy()
            if {"oracle_pnl_eur", "real_pnl_eur"}.issubset(hp.columns):
                hp = hp[hp["oracle_pnl_eur"] < hp["real_pnl_eur"]].copy()
            else:
                hp = hp.iloc[0:0].copy()
            pos_share_cols = sorted([c for c in hp.columns if c.startswith("ev_expected_act_share_pos_bin_")])
            neg_share_cols = sorted([c for c in hp.columns if c.startswith("ev_expected_act_share_neg_bin_")])
            pos_res_cols = sorted([c for c in hp.columns if c.startswith("reserve_pos_bin_") and c.endswith("_mw")])
            neg_res_cols = sorted([c for c in hp.columns if c.startswith("reserve_neg_bin_") and c.endswith("_mw")])
            n_pos = min(len(pos_share_cols), len(pos_res_cols))
            n_neg = min(len(neg_share_cols), len(neg_res_cols))
            if not hp.empty and n_pos > 0:
                exp_pos = np.zeros(len(hp), dtype=float)
                for i in range(n_pos):
                    exp_pos += pd.to_numeric(hp[pos_share_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float) * pd.to_numeric(hp[pos_res_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                hp["expected_act_pos_mwh_from_objective"] = exp_pos
            else:
                hp["expected_act_pos_mwh_from_objective"] = 0.0
            if not hp.empty and n_neg > 0:
                exp_neg = np.zeros(len(hp), dtype=float)
                for i in range(n_neg):
                    exp_neg += pd.to_numeric(hp[neg_share_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float) * pd.to_numeric(hp[neg_res_cols[i]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                hp["expected_act_neg_mwh_from_objective"] = exp_neg
            else:
                hp["expected_act_neg_mwh_from_objective"] = 0.0
            hp["expected_act_total_mwh_from_objective"] = hp["expected_act_pos_mwh_from_objective"] + hp["expected_act_neg_mwh_from_objective"]
            hp["realized_act_total_mwh"] = pd.to_numeric(hp.get("real_act_pos_mwh", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(hp.get("real_act_neg_mwh", 0.0), errors="coerce").fillna(0.0)
            hp["oracle_act_total_mwh"] = pd.to_numeric(hp.get("oracle_act_pos_mwh", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(hp.get("oracle_act_neg_mwh", 0.0), errors="coerce").fillna(0.0)
            keep = [
                colmap.timestamp,
                "real_pnl_eur",
                "oracle_pnl_eur",
                "real_penalty_eur",
                "oracle_penalty_eur",
                "reserve_pos_mw",
                "reserve_neg_mw",
                "expected_act_pos_mwh_from_objective",
                "expected_act_neg_mwh_from_objective",
                "expected_act_total_mwh_from_objective",
                "realized_act_total_mwh",
                "oracle_act_total_mwh",
                "real_act_pos_mwh",
                "real_act_neg_mwh",
                "oracle_act_pos_mwh",
                "oracle_act_neg_mwh",
            ]
            keep = [c for c in keep if c in hp.columns]
            hp[keep].to_csv(oracle_paradox_path, index=False)
        with _phase_watchdog("plot_cumulative_pnl"):
            _plot_cumulative_pnl(outputs.hourly, colmap.timestamp, pnl_plot_path)

        print(f"[OK] Battery backtest completed for scenario={scenario_name}.")
        ts_col = colmap.timestamp if colmap.timestamp in outputs.hourly.columns else None
        if ts_col is not None and not outputs.hourly.empty:
            _ts = pd.to_datetime(outputs.hourly[ts_col], utc=True, errors="coerce").dropna()
        else:
            _ts = pd.Series(dtype="datetime64[ns, UTC]")
        if len(_ts) > 0:
            timeframe_start = _ts.min()
            timeframe_end = _ts.max()
            num_days_total = float((timeframe_end - timeframe_start).total_seconds() / 86400.0) + (1.0 / 24.0)
            print(f"- timeframe_utc: {timeframe_start.isoformat()} -> {timeframe_end.isoformat()}")
            print(f"- timeframe_total_days: {num_days_total:.4f}")
        print(f"- realized_total_pnl_eur: {outputs.summary.get('realized_total_pnl_eur', float('nan')):.2f}")
        print(f"- oracle_total_pnl_eur: {outputs.summary.get('oracle_total_pnl_eur', float('nan')):.2f}")
        print(
            "- DA-only pnl (realized/oracle): "
            f"{outputs.summary.get('realized_da_only_total_pnl_eur', float('nan')):.2f} / "
            f"{outputs.summary.get('oracle_da_only_total_pnl_eur', float('nan')):.2f}"
        )
        print(
            "- aFRR-only pnl (realized/oracle): "
            f"{outputs.summary.get('realized_afrr_only_total_pnl_eur', float('nan')):.2f} / "
            f"{outputs.summary.get('oracle_afrr_only_total_pnl_eur', float('nan')):.2f}"
        )
        print(
            "- Multi-market pnl (realized/oracle): "
            f"{outputs.summary.get('realized_total_pnl_eur', float('nan')):.2f} / "
            f"{outputs.summary.get('oracle_total_pnl_eur', float('nan')):.2f}"
        )
        r_da = outputs.summary.get("realized_vs_oracle_ratio_da_only", float("nan"))
        r_afrr = outputs.summary.get("realized_vs_oracle_ratio_afrr_only", float("nan"))
        r_multi = outputs.summary.get("realized_vs_oracle_ratio_multi_market", float("nan"))
        print(
            "- realized/oracle ratio % (DA-only | aFRR-only | Multi): "
            f"{(100.0 * float(r_da)) if pd.notna(r_da) else float('nan'):.2f}% | "
            f"{(100.0 * float(r_afrr)) if pd.notna(r_afrr) else float('nan'):.2f}% | "
            f"{(100.0 * float(r_multi)) if pd.notna(r_multi) else float('nan'):.2f}%"
        )
        print(f"- output_dir: {scenario_out_dir}")

        row: dict[str, object] = {
            "scenario": scenario_name,
            "da_quantile_role": args.da_quantile_role,
            "realized_total_pnl_eur": outputs.summary.get("realized_total_pnl_eur"),
            "predicted_total_pnl_eur": outputs.summary.get("predicted_total_pnl_eur"),
            "naive_total_pnl_eur": outputs.summary.get("naive_total_pnl_eur"),
            "oracle_total_pnl_eur": outputs.summary.get("oracle_total_pnl_eur"),
            "cost_of_forecast_error_total_eur": outputs.summary.get("cost_of_forecast_error_total_eur"),
            "pnl_gap_total_eur": outputs.summary.get("pnl_gap_total_eur"),
            "economic_opportunity_gap_ratio": outputs.summary.get("economic_opportunity_gap_ratio"),
            "roi_on_max_capital": outputs.summary.get("roi_on_max_capital"),
            "output_dir": str(scenario_out_dir),
        }
        if scenario_name != "default":
            q_lo, q_hi = scenario_name.split("_", 1)
            row["quantile_low"] = q_lo
            row["quantile_high"] = q_hi
        sweep_rows.append(row)

    global_plan_history_path = Path("artifacts/backtest_plan_history.parquet")
    global_plan_history_path.parent.mkdir(parents=True, exist_ok=True)
    if scenarios:
        # Keep previous behavior for downstream consumers: export last scenario plan history globally.
        outputs.plan_history.to_parquet(global_plan_history_path, index=False)

    if quantile_pairs and sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_csv = out_dir / "quantile_sweep_summary.csv"
        sweep_json = out_dir / "quantile_sweep_summary.json"
        sweep_df.to_csv(sweep_csv, index=False)
        sweep_json.write_text(sweep_df.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"[OK] Quantile sweep summary: {sweep_csv}")


if __name__ == "__main__":
    main()
