"""Model audit toolkit: ablation + SHAP interpretability + leakage safety checks.

Usage:
    ./.venv/bin/python evaluation/model_audit.py \
      --base-dir data/model_input \
      --bundle afrr \
      --target-col target_afrr_activation_price_vwap_pos \
      --out-dir data/reports/model_audit

Notes:
    - If --target-col is omitted, all targets of the selected bundle are audited.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Allow script usage from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.prepare_ml_bundles import BundleName, load_processed_data


def _pick_scalar_tag(tags: list[str], patterns: list[str]) -> str | None:
    """Pick the first scalar tag matching any regex pattern (case-insensitive)."""
    if not tags:
        return None
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for t in tags:
            if rx.search(t):
                return t
    return None


def _series_from_events(events: list) -> tuple[list[int], list[float]]:
    """Convert EventAccumulator scalar events to plain step/value arrays."""
    steps = [int(e.step) for e in events]
    vals = [float(e.value) for e in events]
    return steps, vals


def _training_status(
    train_vals: list[float],
    val_vals: list[float],
    *,
    min_points: int = 8,
) -> str:
    """
    Heuristic training-status classification:
    - Converged
    - Overfitting Detected
    - Underfitting Suspected
    - Unstable
    """
    if len(val_vals) < min_points or len(train_vals) < min_points:
        return "Insufficient Data"

    train_final = train_vals[-1]
    val_final = val_vals[-1]
    val_best = min(val_vals)
    best_idx = int(np.argmin(val_vals))
    tail = max(3, int(0.2 * len(val_vals)))
    val_tail = val_vals[-tail:]

    # Overfitting: validation clearly degrades after best epoch while train still improves.
    if best_idx < len(val_vals) - 3 and val_final > (val_best * 1.05):
        if train_final <= min(train_vals[max(0, best_idx - 1) :]):
            return "Overfitting Detected"

    # Underfitting: both losses stay high and almost flat in tail.
    val_range = float(np.max(val_tail) - np.min(val_tail))
    if val_range <= max(1e-8, 0.01 * max(abs(val_final), 1.0)):
        if val_final > train_final * 0.95:
            return "Underfitting Suspected"

    # Unstable: large oscillations in validation tail.
    if len(val_tail) >= 4:
        diffs = np.diff(np.asarray(val_tail, dtype=float))
        if float(np.std(diffs)) > max(1e-8, 0.05 * max(abs(val_final), 1.0)):
            return "Unstable"

    return "Converged"


def audit_training_logs(log_dir: str | Path) -> dict[str, dict[str, object]]:
    """
    Audit TensorBoard scalar logs for each run directory.

    Returns JSON-ready dict keyed by run-name with fields:
        final_train_loss, final_val_loss, best_epoch, total_epochs, status
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError as exc:
        return {
            "_meta": {
                "log_dir": str(Path(log_dir).resolve()),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "Dependency Missing",
                "message": f"tensorboard not installed: {exc}",
            }
        }

    root = Path(log_dir)
    if not root.exists():
        return {
            "_meta": {
                "log_dir": str(root.resolve()),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "Missing Directory",
                "message": f"log_dir not found: {root}",
            }
        }

    out: dict[str, dict[str, object]] = {}
    size_guidance = {"scalars": 0}

    for run_root in sorted([p for p in root.iterdir() if p.is_dir()]):
        run_key = run_root.name
        target_rows: list[dict[str, object]] = []

        # Typical structure: run_root/<target_dir>/events.out.tfevents...
        target_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])
        if not target_dirs:
            # Fallback: run_root itself might contain event files.
            target_dirs = [run_root]

        for tdir in target_dirs:
            event_files = sorted(tdir.glob("events.out.tfevents.*"))
            if not event_files:
                continue

            ea = event_accumulator.EventAccumulator(str(tdir), size_guidance=size_guidance)
            try:
                ea.Reload()
            except Exception as exc:  # pragma: no cover - defensive parser guard
                target_rows.append(
                    {
                        "target": tdir.name,
                        "status": "Parse Error",
                        "message": str(exc),
                    }
                )
                continue

            scalar_tags = ea.Tags().get("scalars", [])
            train_tag = _pick_scalar_tag(
                scalar_tags,
                patterns=[
                    r"train.*loss.*epoch",
                    r"train.*loss",
                    r"loss/train",
                    r"train_loss",
                ],
            )
            val_tag = _pick_scalar_tag(
                scalar_tags,
                patterns=[
                    r"val.*loss.*epoch",
                    r"valid.*loss",
                    r"loss/val",
                    r"val_loss",
                ],
            )

            if not train_tag or not val_tag:
                target_rows.append(
                    {
                        "target": tdir.name,
                        "status": "Missing Loss Scalars",
                        "train_tag": train_tag,
                        "val_tag": val_tag,
                        "available_tags": scalar_tags,
                    }
                )
                continue

            train_steps, train_vals = _series_from_events(ea.Scalars(train_tag))
            val_steps, val_vals = _series_from_events(ea.Scalars(val_tag))
            if not train_vals or not val_vals:
                target_rows.append(
                    {
                        "target": tdir.name,
                        "status": "Missing Loss Values",
                        "train_tag": train_tag,
                        "val_tag": val_tag,
                    }
                )
                continue

            best_idx = int(np.argmin(np.asarray(val_vals, dtype=float)))
            best_epoch = int(val_steps[best_idx]) if best_idx < len(val_steps) else best_idx
            total_epochs = int(max(train_steps[-1] if train_steps else 0, val_steps[-1] if val_steps else 0) + 1)
            status = _training_status(train_vals, val_vals)

            target_rows.append(
                {
                    "target": tdir.name,
                    "train_loss_tag": train_tag,
                    "val_loss_tag": val_tag,
                    "final_train_loss": float(train_vals[-1]),
                    "final_val_loss": float(val_vals[-1]),
                    "best_epoch": best_epoch,
                    "total_epochs": total_epochs,
                    "status": status,
                }
            )

        # Aggregate run-level status from targets.
        statuses = [str(r.get("status", "")) for r in target_rows if "status" in r]
        if any(s == "Overfitting Detected" for s in statuses):
            run_status = "Overfitting Detected"
        elif any(s == "Unstable" for s in statuses):
            run_status = "Unstable"
        elif any(s == "Converged" for s in statuses):
            run_status = "Converged"
        elif target_rows:
            run_status = statuses[0]
        else:
            run_status = "No Event Files Found"

        final_train = [float(r["final_train_loss"]) for r in target_rows if "final_train_loss" in r]
        final_val = [float(r["final_val_loss"]) for r in target_rows if "final_val_loss" in r]
        best_epochs = [int(r["best_epoch"]) for r in target_rows if "best_epoch" in r]
        total_epochs = [int(r["total_epochs"]) for r in target_rows if "total_epochs" in r]

        out[run_key] = {
            "final_train_loss": float(np.mean(final_train)) if final_train else None,
            "final_val_loss": float(np.mean(final_val)) if final_val else None,
            "best_epoch": int(np.median(best_epochs)) if best_epochs else None,
            "total_epochs": int(np.max(total_epochs)) if total_epochs else 0,
            "status": run_status,
            "targets": target_rows,
        }

    out["_meta"] = {
        "log_dir": str(root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": int(sum(1 for k in out if k != "_meta")),
    }
    return out


def _resolve_target(bundle: BundleName, requested: str | None, y_cols: list[str]) -> str:
    if requested:
        if requested not in y_cols:
            raise KeyError(f"Requested target '{requested}' not available in y columns.")
        return requested
    default = "target_da_price" if bundle == "da" else "target_afrr_activation_price_vwap_pos"
    if default not in y_cols:
        raise KeyError(f"Default target '{default}' not found in y columns.")
    return default


def _resolve_targets(bundle: BundleName, requested: str | None, y_cols: list[str]) -> list[str]:
    if requested:
        return [_resolve_target(bundle, requested, y_cols)]
    if bundle == "da":
        return [_resolve_target(bundle, None, y_cols)]
    ordered = [
        "target_afrr_activation_price_vwap_pos",
        "target_afrr_activation_price_vwap_neg",
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
    ]
    targets = [t for t in ordered if t in y_cols]
    if not targets:
        raise KeyError("No aFRR targets found in y columns.")
    return targets


def _feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Create cumulative feature groups for ablation path."""
    cols = list(feature_cols)

    core_patterns = [
        r"_lag_(1|2|3|6|12|24|48|168)h$",
        r"^(hour|dayofweek|weekday|month)_(sin|cos)$",
        r"^is_(weekend|morning|afternoon|evening|night|bridge_day|christmas_break|payday_period)$",
        r"^da_price_(pit|lag_|diff|mean_|std_|ewma|slog1p)",
        r"^(da_price_pit|market_regime_picasso|is_picasso_active)$",
    ]
    weather_patterns = [
        r"(wind|solar|load_forecast|residual_load_forecast|renewable_share_forecast)",
        r"(temperature|temp|weather)",
    ]
    outages_patterns = [
        r"(planned_outages|unplanned_outages|outage)",
    ]
    neighbors_patterns = [
        r"(neighbor_spread|da_spread_de_|da_price_(AT|FR|NL)|cross_border|interconnector)",
    ]

    def pick(patterns: list[str]) -> list[str]:
        rx = re.compile("|".join(patterns))
        return sorted([c for c in cols if rx.search(c)])

    core = pick(core_patterns)
    weather = pick(weather_patterns)
    outages = pick(outages_patterns)
    neighbors = pick(neighbors_patterns)

    # Ensure cumulative design.
    g1 = sorted(set(core))
    g2 = sorted(set(g1 + weather))
    g3 = sorted(set(g2 + outages))
    g4 = sorted(set(g3 + neighbors))

    # Fallback guard: if core becomes too narrow, keep all lag + time cols.
    if not g1:
        g1 = sorted([c for c in cols if ("_lag_" in c) or c.endswith("_sin") or c.endswith("_cos")])
    if not g2:
        g2 = g1
    if not g3:
        g3 = g2
    if not g4:
        g4 = g3

    return {
        "core_only": g1,
        "plus_weather": g2,
        "plus_outages": g3,
        "plus_neighbors": g4,
    }


def _fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        device=device,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _spread_pnl_proxy(y_true: pd.Series, y_pred: np.ndarray, X: pd.DataFrame) -> float:
    """PnL proxy in EUR/MWh units using DA spread direction capture."""
    da_ref_col = "da_price_pit" if "da_price_pit" in X.columns else None
    if da_ref_col is None:
        # fallback to neutral baseline to keep script robust on DA bundle variants
        da_ref = pd.Series(0.0, index=X.index)
    else:
        da_ref = pd.to_numeric(X[da_ref_col], errors="coerce")
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(pd.Series(y_pred, index=y.index), errors="coerce")
    valid = y.notna() & p.notna() & da_ref.notna()
    if not bool(valid.any()):
        return float("nan")
    signal = np.sign(p.loc[valid].to_numpy() - da_ref.loc[valid].to_numpy())
    realized_spread = y.loc[valid].to_numpy() - da_ref.loc[valid].to_numpy()
    return float(np.sum(signal * realized_spread))


def _evaluate_variant(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
) -> tuple[XGBRegressor, float, float]:
    model = _fit_xgb(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
    )
    pred = model.predict(X_eval)
    rmse = float(np.sqrt(mean_squared_error(y_eval, pred)))
    pnl = _spread_pnl_proxy(y_eval, pred, X_eval)
    return model, rmse, pnl


def _safety_leak_check(top_features: list[str]) -> pd.DataFrame:
    risky = []
    patterns = [
        ("contains_id", re.compile(r"\bid\b|_id_|^id_", re.IGNORECASE)),
        ("target_like", re.compile(r"^target_", re.IGNORECASE)),
        ("future_hint", re.compile(r"(lead|future|next|t\+|lookahead)", re.IGNORECASE)),
        ("raw_unlagged_actual_like", re.compile(r"(actual|vwap|activated_mw)", re.IGNORECASE)),
    ]
    for f in top_features:
        hits = [name for name, rx in patterns if rx.search(f)]
        # allow explicit lagged features as non-leak by design
        if "_lag_" in f:
            hits = [h for h in hits if h != "raw_unlagged_actual_like"]
        risky.append({"feature": f, "risk_flags": ", ".join(hits) if hits else "none"})
    return pd.DataFrame(risky)


def _build_local_explainer_fn(
    *,
    shap_values,
    X_eval: pd.DataFrame,
    timestamps: pd.Series,
    out_dir: Path,
) -> Callable[[str], Path]:
    import shap  # local import to keep startup cheap

    def explain_prediction(timestamp) -> Path:
        ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
        idx = X_eval.index[pd.to_datetime(timestamps, utc=True, errors="coerce") == ts]
        if len(idx) == 0:
            raise KeyError(f"Timestamp not found in evaluation frame: {timestamp}")
        i = int(np.where(X_eval.index == idx[0])[0][0])
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i], max_display=20, show=False)
        out = out_dir / f"shap_waterfall_{ts.strftime('%Y%m%dT%H%M%SZ')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()
        return out

    return explain_prediction


def _load_split_timestamps(base_dir: Path, bundle: BundleName, split: str) -> pd.Series:
    cfg = json.loads((base_dir / "feature_config.json").read_text(encoding="utf-8"))
    split_path = Path(cfg["bundles"][bundle]["files"][split])
    df = pd.read_parquet(split_path, columns=["timestamp_utc"])
    return pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")


def main() -> None:
    p = argparse.ArgumentParser(description="Feature ablation and SHAP audit for production/thesis sign-off.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr")
    p.add_argument("--target-col", default=None)
    p.add_argument("--out-dir", default="data/reports/model_audit")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--tb-log-dir", default=None, help="Optional TensorBoard log root (e.g. artifacts/tensorboard_logs).")
    p.add_argument("--tb-audit-json", default=None, help="Optional output path for TensorBoard audit JSON.")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_raw, y_train_df = load_processed_data(bundle=args.bundle, split="train", base_dir=base_dir)
    X_val_raw, y_val_df = load_processed_data(bundle=args.bundle, split="val", base_dir=base_dir)
    X_test_raw, y_test_df = load_processed_data(bundle=args.bundle, split="test", base_dir=base_dir)
    ts_train = _load_split_timestamps(base_dir, args.bundle, "train")
    ts_val = _load_split_timestamps(base_dir, args.bundle, "val")
    ts_test = _load_split_timestamps(base_dir, args.bundle, "test")
    targets = _resolve_targets(args.bundle, args.target_col, list(y_train_df.columns))
    aggregate_rows: list[dict[str, object]] = []

    for idx, target in enumerate(targets, start=1):
        print(f"[audit] ({idx}/{len(targets)}) target={target}")
        target_out_dir = out_dir if len(targets) == 1 else (out_dir / target)
        target_out_dir.mkdir(parents=True, exist_ok=True)

        y_train = pd.to_numeric(y_train_df[target], errors="coerce")
        y_val = pd.to_numeric(y_val_df[target], errors="coerce")
        y_test = pd.to_numeric(y_test_df[target], errors="coerce")
        m_train = y_train.notna()
        m_val = y_val.notna()
        m_test = y_test.notna()
        X_train_t = X_train_raw.loc[m_train].copy()
        X_val_t = X_val_raw.loc[m_val].copy()
        X_test_t = X_test_raw.loc[m_test].copy()
        ts_test_t = ts_test.loc[m_test].copy()
        y_train_t = y_train.loc[m_train].copy()
        y_test_t = y_test.loc[m_test].copy()

        groups = _feature_groups(list(X_train_t.columns))
        results = []
        trained_models: dict[str, XGBRegressor] = {}
        prev_rmse = None
        prev_pnl = None

        for name, cols in groups.items():
            use_cols = [c for c in cols if c in X_train_t.columns]
            Xtr = X_train_t[use_cols].copy()
            Xte = X_test_t[use_cols].copy()

            model, rmse, pnl = _evaluate_variant(
                Xtr,
                y_train_t,
                Xte,
                y_test_t,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                device=args.device,
            )
            trained_models[name] = model
            results.append(
                {
                    "variant": name,
                    "n_features": len(use_cols),
                    "rmse": rmse,
                    "pnl_proxy_eur": pnl,
                    "delta_rmse_vs_prev": (rmse - prev_rmse) if prev_rmse is not None else np.nan,
                    "delta_pnl_vs_prev_eur": (pnl - prev_pnl) if prev_pnl is not None else np.nan,
                }
            )
            prev_rmse = rmse
            prev_pnl = pnl
            print(f"[ablation] {target}::{name}: n_features={len(use_cols)} rmse={rmse:.4f} pnl_proxy_eur={pnl:.2f}")

        manifesto = pd.DataFrame(results)
        manifesto.to_csv(target_out_dir / "feature_value_manifesto.csv", index=False)

        final_variant = "plus_neighbors"
        final_cols = [c for c in groups[final_variant] if c in X_train_t.columns]
        final_model = trained_models[final_variant]
        X_test_final = X_test_t[final_cols].copy()
        pred_test_final = final_model.predict(X_test_final)

        X_eval_local = X_test_final.copy()
        da_ref = pd.to_numeric(X_test_final["da_price_pit"], errors="coerce") if "da_price_pit" in X_test_final.columns else pd.Series(0.0, index=X_test_final.index)
        y_eval = pd.to_numeric(y_test_t, errors="coerce")
        pnl_hour = np.sign(pred_test_final - da_ref.to_numpy()) * (y_eval.to_numpy() - da_ref.to_numpy())
        spike_threshold = float(np.nanquantile(pred_test_final, 0.95)) if len(pred_test_final) else np.nan
        local_df = pd.DataFrame(
            {
                "timestamp_utc": ts_test_t.to_numpy(),
                "pnl_proxy_hour_eur": pnl_hour,
                "y_true": y_eval.to_numpy(),
                "y_pred": pred_test_final,
            }
        ).dropna(subset=["timestamp_utc", "pnl_proxy_hour_eur"])
        spike_df = local_df[local_df["y_pred"] >= spike_threshold].copy() if np.isfinite(spike_threshold) else local_df.copy()
        top3 = spike_df.sort_values("pnl_proxy_hour_eur", ascending=False).head(3)
        if len(top3) < 3:
            fallback = local_df.sort_values("pnl_proxy_hour_eur", ascending=False).head(3)
            top3 = pd.concat([top3, fallback], axis=0).drop_duplicates(subset=["timestamp_utc"]).head(3)
        top3.to_csv(target_out_dir / "top3_profitable_spikes.csv", index=False)

        beeswarm_path = target_out_dir / "shap_summary_beeswarm_top20.png"
        local_plot_paths: list[str] = []
        shap_top20_path = target_out_dir / "shap_top20_features.csv"
        shap_leak_path = target_out_dir / "shap_leak_safety_audit.csv"
        try:
            import shap  # type: ignore

            sample_n = min(5000, len(X_test_final))
            X_shap = X_test_final.sample(n=sample_n, random_state=42) if sample_n < len(X_test_final) else X_test_final.copy()
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer(X_shap)

            plt.figure(figsize=(11, 7))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=220)
            plt.close()

            mean_abs = np.abs(shap_values.values).mean(axis=0)
            shap_rank = (
                pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            top20 = shap_rank.head(20).copy()
            top20.to_csv(shap_top20_path, index=False)

            leak_audit = _safety_leak_check(top20["feature"].tolist())
            leak_audit.to_csv(shap_leak_path, index=False)

            shap_values_full = explainer(X_test_final)
            explain_prediction = _build_local_explainer_fn(
                shap_values=shap_values_full,
                X_eval=X_eval_local,
                timestamps=ts_test_t,
                out_dir=target_out_dir,
            )
            for ts in top3["timestamp_utc"].tolist():
                try:
                    local_plot_paths.append(str(explain_prediction(ts)))
                except Exception as exc:
                    local_plot_paths.append(f"ERROR:{ts}:{exc}")
        except ModuleNotFoundError:
            pd.DataFrame(
                [{"feature": "N/A", "mean_abs_shap": np.nan, "note": "Install shap to enable SHAP audit."}]
            ).to_csv(shap_top20_path, index=False)
            pd.DataFrame(
                [{"feature": "N/A", "risk_flags": "N/A", "note": "Install shap to enable leak safety SHAP audit."}]
            ).to_csv(shap_leak_path, index=False)
            print("[WARN] SHAP is not installed in this environment. Global/local SHAP outputs were skipped.")

        mae_points = pd.DataFrame(
            [
                {"lead_time_h": 1, "mae": float(np.mean(np.abs(y_eval.to_numpy() - pred_test_final)))},
                {"lead_time_h": 24, "mae": np.nan},
                {"lead_time_h": 48, "mae": np.nan},
            ]
        )
        mae_points.to_csv(target_out_dir / "mae_leadtime_1_24_48.csv", index=False)
        plt.figure(figsize=(7, 4))
        pp = mae_points.dropna(subset=["mae"])
        plt.plot(pp["lead_time_h"], pp["mae"], marker="o", linewidth=2, color="#2C7FB8")
        plt.xticks([1, 24, 48])
        plt.xlabel("Lead Time (h)")
        plt.ylabel("MAE")
        plt.title("MAE by Lead Time (1h / 24h / 48h)")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(target_out_dir / "mae_leadtime_1_24_48.png", dpi=220)
        plt.close()

        summary = {
            "bundle": args.bundle,
            "target_col": target,
            "final_variant": final_variant,
            "final_n_features": len(final_cols),
            "outputs": {
                "feature_value_manifesto": str((target_out_dir / "feature_value_manifesto.csv").resolve()),
                "shap_summary_beeswarm": str(beeswarm_path.resolve()),
                "shap_top20": str((target_out_dir / "shap_top20_features.csv").resolve()),
                "shap_leak_safety_audit": str((target_out_dir / "shap_leak_safety_audit.csv").resolve()),
                "top3_profitable_spikes": str((target_out_dir / "top3_profitable_spikes.csv").resolve()),
                "mae_leadtime_points": str((target_out_dir / "mae_leadtime_1_24_48.csv").resolve()),
                "mae_leadtime_plot": str((target_out_dir / "mae_leadtime_1_24_48.png").resolve()),
                "local_waterfalls": local_plot_paths,
            },
        }
        (target_out_dir / "model_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        aggregate_rows.append(
            {
                "target_col": target,
                "out_dir": str(target_out_dir.resolve()),
                "final_variant": final_variant,
                "final_n_features": len(final_cols),
                "mae_h1": float(mae_points.loc[mae_points["lead_time_h"] == 1, "mae"].iloc[0]),
                "final_rmse": float(manifesto.loc[manifesto["variant"] == final_variant, "rmse"].iloc[0]),
                "final_pnl_proxy_eur": float(manifesto.loc[manifesto["variant"] == final_variant, "pnl_proxy_eur"].iloc[0]),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("target_col").reset_index(drop=True)
    aggregate_df.to_csv(out_dir / "model_audit_targets_overview.csv", index=False)
    (out_dir / "model_audit_targets_overview.json").write_text(
        json.dumps(aggregate_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    # Optional training-curve audit from TensorBoard logs.
    if args.tb_log_dir:
        tb_audit = audit_training_logs(args.tb_log_dir)
        tb_out = Path(args.tb_audit_json) if args.tb_audit_json else (out_dir / "training_log_audit.json")
        tb_out.parent.mkdir(parents=True, exist_ok=True)
        tb_out.write_text(json.dumps(tb_audit, indent=2), encoding="utf-8")
        print(f"[OK] Training-log audit written to: {tb_out}")

    print(f"[OK] Model audit completed for {len(targets)} target(s). Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
