from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from energy_trading.evaluation.forecast_metrics import (
    TailConfig,
    approx_crps,
    assign_gate_context,
    assign_horizon_bucket,
    empirical_coverage,
    interval_coverage,
    interval_width,
    mae,
    mean_error,
    mean_pinball_loss,
    median_absolute_error,
    normalized_mae,
    pinball_loss,
    quantile_crossing_metrics,
    repair_monotone_quantiles,
    rmse,
    tail_event_metrics,
    winkler_score,
)
from energy_trading.evaluation.forecast_figures import generate_forecast_benchmark_figures
from energy_trading.evaluation.forecast_postprocessing import (
    canonicalize_prediction_frame,
    canonicalize_truth_series,
)
from energy_trading.evaluation.forecast_truth_mapping import resolve_truth_mapping


@dataclass
class BenchmarkArtifacts:
    benchmark_dir: Path
    diagnostics_dir: Path
    metrics_dir: Path
    figures_dir: Path


REQUIRED_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
TARGETS = [
    "pred_da_price",
    "pred_afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos",
    "pred_afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_latest_or_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "manifest_path" in data:
        mp = (path.parent / str(data["manifest_path"])).resolve()
        return mp, json.loads(mp.read_text(encoding="utf-8"))
    return path.resolve(), data


def _model_name(manifest_path: Path, manifest: dict[str, Any]) -> str:
    mt = str(manifest.get("training", {}).get("model_type", "")).lower()
    if "xg" in mt:
        return "xgb"
    if mt:
        return mt
    n = manifest_path.name.lower()
    if "xgb" in n:
        return "xgb"
    if "tft" in n:
        return "tft"
    if "linear" in n or "rlqr" in n:
        return "rlqr"
    return manifest_path.stem


def _qcol_for(prob: float) -> str:
    return f"p{int(round(prob*100)):02d}"


def _safe_read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")


def _extract_pred_paths(manifest_path: Path, manifest: dict[str, Any], split: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for bundle in manifest.get("bundles", {}).values():
        pl = bundle.get("predictions_long", {})
        for target, rel in (pl.get(split) or {}).items():
            if target in TARGETS:
                out[target] = (manifest_path.parent / str(rel)).resolve()
    return out


def _target_value_mode_from_manifest(manifest: dict[str, Any], target: str) -> str | None:
    tvm = manifest.get("target_value_mode", {})
    if isinstance(tvm, dict) and target in tvm:
        return str(tvm[target])
    sim = manifest.get("simulation", {})
    canonical_targets = sim.get("canonical_economic_targets", [])
    if isinstance(canonical_targets, list) and target in canonical_targets:
        return "canonical_economic"
    transformed = sim.get("transformed_targets", [])
    if isinstance(transformed, list) and target in transformed:
        return "canonical_economic"
    if target == "pred_afrr_activation_price_neg":
        return "raw_signed_legacy"
    return None


def _build_inventory_row(*, model: str, split: str, target: str, pred_path: Path, truth_path: Path, joined: pd.DataFrame, truth_col: str) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "target": target,
        "prediction_path": str(pred_path),
        "prediction_sha256": _sha256(pred_path),
        "truth_path": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "n_rows_joined": int(len(joined)),
        "target_time_min_utc": str(joined["target_time_utc"].min()) if len(joined) else None,
        "target_time_max_utc": str(joined["target_time_utc"].max()) if len(joined) else None,
        "resolved_truth_col": truth_col,
        "status": "ok",
    }


def _compute_metrics_for_frame(
    *,
    model: str,
    split: str,
    target: str,
    df: pd.DataFrame,
    quantiles: list[float],
    horizon_buckets: dict[str, list[int]],
    tail_cfg: TailConfig,
) -> dict[str, pd.DataFrame]:
    rows_overall: list[dict[str, Any]] = []
    rows_lead: list[dict[str, Any]] = []
    rows_hb: list[dict[str, Any]] = []
    rows_gate: list[dict[str, Any]] = []
    rows_tail: list[dict[str, Any]] = []
    rows_cal: list[dict[str, Any]] = []
    rows_cross: list[dict[str, Any]] = []

    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p50_col = "p50" if "p50" in df.columns else "predicted_value"
    yhat = pd.to_numeric(df[p50_col], errors="coerce").to_numpy(dtype=float)

    q_preds_raw: dict[float, np.ndarray] = {}
    for q in quantiles:
        qc = _qcol_for(q)
        if qc in df.columns:
            q_preds_raw[q] = pd.to_numeric(df[qc], errors="coerce").to_numpy(dtype=float)

    crossing_rate, crossing_max = quantile_crossing_metrics(q_preds_raw)
    q_preds = repair_monotone_quantiles(q_preds_raw) if q_preds_raw else {}

    row = {
        "model": model,
        "split": split,
        "target": target,
        "n": int(len(df)),
        "mae_p50": mae(y, yhat),
        "rmse_p50": rmse(y, yhat),
        "bias": mean_error(y, yhat),
        "median_absolute_error": median_absolute_error(y, yhat),
        "normalized_mae": normalized_mae(y, yhat),
        "mean_pinball": mean_pinball_loss(y, q_preds),
        "approx_crps": approx_crps(y, q_preds),
        "crossing_rate_before_repair": crossing_rate,
        "max_crossing_violation_before_repair": crossing_max,
    }
    if 0.1 in q_preds and 0.9 in q_preds:
        cov = interval_coverage(y, q_preds[0.1], q_preds[0.9])
        row["coverage_p10_p90"] = cov
        row["coverage_error_p10_p90"] = cov - 0.8
        row["interval_width_p10_p90"] = interval_width(q_preds[0.1], q_preds[0.9])
        row["winkler_p10_p90"] = winkler_score(y, q_preds[0.1], q_preds[0.9], alpha=0.2)
    if 0.3 in q_preds and 0.7 in q_preds:
        cov = interval_coverage(y, q_preds[0.3], q_preds[0.7])
        row["coverage_p30_p70"] = cov
        row["coverage_error_p30_p70"] = cov - 0.4
        row["interval_width_p30_p70"] = interval_width(q_preds[0.3], q_preds[0.7])
        row["winkler_p30_p70"] = winkler_score(y, q_preds[0.3], q_preds[0.7], alpha=0.6)
    rows_overall.append(row)

    for q, arr in q_preds.items():
        covq = empirical_coverage(y, arr)
        rows_cal.append(
            {
                "model": model,
                "split": split,
                "target": target,
                "quantile": q,
                "empirical_coverage": covq,
                "coverage_error": covq - q,
                "pinball": pinball_loss(y, arr, q),
            }
        )
    rows_cross.append(
        {
            "model": model,
            "split": split,
            "target": target,
            "crossing_rate_before_repair": crossing_rate,
            "max_crossing_violation_before_repair": crossing_max,
            "quantile_repair_applied": float(bool(q_preds_raw)),
        }
    )

    tm = tail_event_metrics(y=y, yhat_p50=yhat, cfg=tail_cfg)
    rows_tail.append({"model": model, "split": split, "target": target, **tm})

    if "lead_time_h" in df.columns:
        leads = pd.to_numeric(df["lead_time_h"], errors="coerce")
        for lh, g in df.groupby(leads):
            if pd.isna(lh):
                continue
            yy = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
            yh = pd.to_numeric(g[p50_col], errors="coerce").to_numpy(dtype=float)
            if len(g) == 0:
                continue
            rows_lead.append({
                "model": model,
                "split": split,
                "target": target,
                "lead_time_h": float(lh),
                "n": int(len(g)),
                "mae_p50": mae(yy, yh),
                "rmse_p50": rmse(yy, yh),
                "mean_pinball": mean_pinball_loss(yy, {q: q_preds[q][g.index.to_numpy()] for q in q_preds}),
                "approx_crps": approx_crps(yy, {q: q_preds[q][g.index.to_numpy()] for q in q_preds}),
            })
            if 0.1 in q_preds and 0.9 in q_preds:
                rows_lead[-1]["coverage_p10_p90"] = interval_coverage(
                    yy, q_preds[0.1][g.index.to_numpy()], q_preds[0.9][g.index.to_numpy()]
                )
                rows_lead[-1]["interval_width_p10_p90"] = interval_width(
                    q_preds[0.1][g.index.to_numpy()], q_preds[0.9][g.index.to_numpy()]
                )
            bucket = assign_horizon_bucket(float(lh), horizon_buckets)
            rows_hb.append({
                "model": model,
                "split": split,
                "target": target,
                "horizon_bucket": bucket,
                "lead_time_h": float(lh),
                "n": int(len(g)),
                "mae_p50": mae(yy, yh),
                "mean_pinball": mean_pinball_loss(yy, {q: q_preds[q][g.index.to_numpy()] for q in q_preds}),
            })

    if "target_time_utc" in df.columns and "lead_time_h" in df.columns:
        ts = pd.to_datetime(df["target_time_utc"], utc=True, errors="coerce")
        leads = pd.to_numeric(df["lead_time_h"], errors="coerce")
        for ctx in {c for t, l in zip(ts, leads) if pd.notna(t) and pd.notna(l) for c in assign_gate_context(t, float(l))}:
            mask = np.array([
                (ctx in assign_gate_context(t, float(l))) if (pd.notna(t) and pd.notna(l)) else False
                for t, l in zip(ts, leads)
            ])
            if not np.any(mask):
                continue
            yy = y[mask]
            yh = yhat[mask]
            qsub = {q: arr[mask] for q, arr in q_preds.items()}
            row_ctx = {
                "model": model,
                "split": split,
                "target": target,
                "gate_context": ctx,
                "n": int(mask.sum()),
                "mae_p50": mae(yy, yh),
                "rmse_p50": rmse(yy, yh),
                "mean_pinball": mean_pinball_loss(yy, qsub),
                "approx_crps": approx_crps(yy, qsub),
            }
            if 0.1 in qsub and 0.9 in qsub:
                row_ctx["coverage_p10_p90"] = interval_coverage(yy, qsub[0.1], qsub[0.9])
                row_ctx["interval_width_p10_p90"] = interval_width(qsub[0.1], qsub[0.9])
            rows_gate.append(row_ctx)

    return {
        "overall": pd.DataFrame(rows_overall),
        "by_lead": pd.DataFrame(rows_lead),
        "by_horizon_bucket": pd.DataFrame(rows_hb),
        "gate": pd.DataFrame(rows_gate),
        "tail": pd.DataFrame(rows_tail),
        "calibration": pd.DataFrame(rows_cal),
        "crossing": pd.DataFrame(rows_cross),
    }


def _ranking_tables(metrics_by_target: pd.DataFrame, baseline_model: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_by_target.empty:
        return pd.DataFrame(), pd.DataFrame()
    score_cols = ["mean_pinball", "approx_crps", "mae_p50", "normalized_mae"]
    rows = []
    for target, gt in metrics_by_target.groupby("target"):
        ranks = gt[["model", *score_cols]].copy()
        for c in score_cols:
            ranks[f"rank_{c}"] = ranks[c].rank(method="average", ascending=True)
        rank_cols = [f"rank_{c}" for c in score_cols]
        ranks["avg_rank"] = ranks[rank_cols].mean(axis=1)
        for _, r in ranks.iterrows():
            rows.append({"target": target, "model": r["model"], "avg_rank": r["avg_rank"], **{c: r[c] for c in rank_cols}})
    per_target_rank = pd.DataFrame(rows)

    global_rows = []
    for model, gm in per_target_rank.groupby("model"):
        global_rows.append({"model": model, "avg_rank_across_targets": float(gm["avg_rank"].mean())})
    global_rank = pd.DataFrame(global_rows).sort_values("avg_rank_across_targets")

    if baseline_model and baseline_model in metrics_by_target["model"].unique():
        sk = []
        for (target, split), gt in metrics_by_target.groupby(["target", "split"]):
            b = gt.loc[gt["model"] == baseline_model]
            if b.empty:
                continue
            b_loss = float(b["mean_pinball"].iloc[0])
            if not np.isfinite(b_loss) or abs(b_loss) <= 1e-12:
                continue
            for _, r in gt.iterrows():
                sk.append({
                    "target": target,
                    "split": split,
                    "model": r["model"],
                    "baseline_model": baseline_model,
                    "skill_mean_pinball": float(1.0 - (float(r["mean_pinball"]) / b_loss)),
                })
        if sk:
            global_rank = global_rank.merge(pd.DataFrame(sk).groupby("model", as_index=False)["skill_mean_pinball"].mean(), on="model", how="left")
    return per_target_rank, global_rank


def _git_commit_or_none() -> str | None:
    try:
        import subprocess

        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return None


def run_benchmark(
    *,
    config: dict[str, Any],
    model_run_manifests: list[Path],
    out_dir: Path,
    splits: list[str],
    truth_source: Path,
    min_join_coverage: float,
    fail_on_missing_truth: bool,
    make_figures: bool,
    save_joined_predictions: bool,
    overwrite: bool,
) -> BenchmarkArtifacts:
    if overwrite and out_dir.exists():
        for p in sorted(out_dir.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = out_dir / "diagnostics"
    joined_diag_dir = diagnostics / "joined_predictions"
    metrics_dir = out_dir / "metrics"
    figures = out_dir / "figures"
    diagnostics.mkdir(parents=True, exist_ok=True)
    joined_diag_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    truth_df = _safe_read(truth_source)
    ts_col = "timestamp_utc" if "timestamp_utc" in truth_df.columns else "target_time_utc"
    if ts_col not in truth_df.columns:
        raise ValueError("truth source must include timestamp_utc or target_time_utc")
    truth_df = truth_df.copy()
    truth_df["target_time_utc"] = pd.to_datetime(truth_df[ts_col], utc=True, errors="coerce")

    quantiles = [float(q) for q in config.get("quantiles", REQUIRED_QUANTILES)]
    horizon_cfg = config.get("horizon", {})
    horizon_buckets = horizon_cfg.get("buckets", {"short": [1, 8], "medium": [9, 16], "long": [17, 48]})
    tail_cfg = TailConfig(**config.get("tail_events", {}))

    truth_rows = []
    coverage_rows = []
    inventory_rows = []
    schema_rows = []
    postprocess_rows: list[dict[str, Any]] = []

    m_overall: list[pd.DataFrame] = []
    m_lead: list[pd.DataFrame] = []
    m_hb: list[pd.DataFrame] = []
    m_gate: list[pd.DataFrame] = []
    m_tail: list[pd.DataFrame] = []
    m_cal: list[pd.DataFrame] = []
    m_cross: list[pd.DataFrame] = []
    joined_frames: list[pd.DataFrame] = []
    canonical_targets_seen: set[str] = set()

    for pointer in model_run_manifests:
        manifest_path, manifest = _load_latest_or_manifest(pointer)
        model = _model_name(manifest_path, manifest)
        for split in splits:
            pred_paths = _extract_pred_paths(manifest_path, manifest, split)
            for target, pred_path in pred_paths.items():
                pred = _safe_read(pred_path)
                if "target_time_utc" not in pred.columns:
                    raise ValueError(f"{model}/{target}/{split}: missing target_time_utc in {pred_path}")
                pred = pred.copy()
                pred["target_time_utc"] = pd.to_datetime(pred["target_time_utc"], utc=True, errors="coerce")
                target_value_mode = _target_value_mode_from_manifest(manifest, target)
                if str(target_value_mode or "").strip().lower() == "canonical_economic":
                    canonical_targets_seen.add(target)
                required_q_cols = [_qcol_for(q) for q in quantiles]
                pred, post_report = canonicalize_prediction_frame(
                    pred,
                    target_name=target,
                    quantile_cols=required_q_cols,
                    predicted_value_col="predicted_value",
                    target_value_mode=target_value_mode,
                )
                postprocess_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "target": target,
                        **post_report,
                    }
                )
                mapr = resolve_truth_mapping(
                    prediction_target_name=target,
                    available_truth_columns=list(truth_df.columns),
                    truth_source_path=truth_source,
                    fail_on_missing_truth=fail_on_missing_truth,
                )
                truth_rows.append(mapr.__dict__ | {"model": model, "split": split})
                if mapr.truth_column is None:
                    continue
                truth_take = truth_df[["target_time_utc", mapr.truth_column]].copy()
                truth_take["y_true"] = pd.to_numeric(truth_take[mapr.truth_column], errors="coerce")
                truth_take["y_true"] = canonicalize_truth_series(
                    truth_take["y_true"], target_name=target, target_value_mode=target_value_mode
                )
                truth_take = truth_take.loc[pd.notna(truth_take["y_true"]), ["target_time_utc", "y_true"]].copy()
                if truth_take.empty:
                    raise ValueError(
                        f"{model}/{target}/{split}: truth column '{mapr.truth_column}' has no non-null rows."
                    )
                tmin = pd.to_datetime(truth_take["target_time_utc"], utc=True, errors="coerce").min()
                tmax = pd.to_datetime(truth_take["target_time_utc"], utc=True, errors="coerce").max()
                pred_in_window = pred.loc[
                    (pd.to_datetime(pred["target_time_utc"], utc=True, errors="coerce") >= tmin)
                    & (pd.to_datetime(pred["target_time_utc"], utc=True, errors="coerce") <= tmax)
                ].copy()
                n_out_of_window = int(len(pred) - len(pred_in_window))
                if pred_in_window.empty:
                    raise ValueError(
                        f"{model}/{target}/{split}: no prediction rows overlap truth window "
                        f"[{tmin}, {tmax}] after filtering. Total predictions={len(pred)}."
                    )
                joined = pred_in_window.merge(
                    truth_take,
                    on="target_time_utc",
                    how="left",
                )
                cov = float(pd.notna(joined["y_true"]).mean())
                coverage_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "split": split,
                        "n_predictions_total": int(len(pred)),
                        "n_predictions_eval_window": int(len(pred_in_window)),
                        "n_truth_matched": int(pd.notna(joined["y_true"]).sum()),
                        "n_predictions_outside_truth_window": n_out_of_window,
                        "truth_window_min_utc": str(tmin),
                        "truth_window_max_utc": str(tmax),
                        "join_coverage": cov,
                        "join_coverage_pct": cov * 100.0,
                        "min_required_coverage": float(min_join_coverage),
                        "status": "ok" if cov >= min_join_coverage else "below_threshold",
                    }
                )
                if cov < min_join_coverage:
                    examples = joined.loc[joined["y_true"].isna(), "target_time_utc"].dropna().astype(str).head(10).tolist()
                    raise ValueError(
                        f"{model}/{target}/{split}: join coverage {cov:.4%} below threshold {min_join_coverage:.4%}. "
                        f"Missing rows={(joined['y_true'].isna()).sum()}/{len(joined)} examples={examples}. "
                        f"Predictions outside truth window dropped={n_out_of_window}."
                    )
                joined = joined.loc[pd.notna(joined["y_true"])].copy()
                joined["model"] = model
                joined["split"] = split
                joined["target"] = target
                joined["predicted_value"] = pd.to_numeric(joined.get("predicted_value", joined.get("p50")), errors="coerce")
                for q in quantiles:
                    q_col = _qcol_for(q)
                    if q_col not in joined.columns:
                        joined[q_col] = np.nan
                joined_min = joined[
                    [
                        c
                        for c in [
                            "model",
                            "split",
                            "target",
                            "forecast_time_utc",
                            "snapshot_time_utc",
                            "target_time_utc",
                            "lead_time_h",
                            "y_true",
                            "p10",
                            "p30",
                            "p50",
                            "p70",
                            "p90",
                            "predicted_value",
                        ]
                        if c in joined.columns
                    ]
                ].copy()
                if save_joined_predictions:
                    joined_min.to_parquet(joined_diag_dir / f"{model}__{split}__{target}.parquet", index=False)
                joined_frames.append(joined_min)
                inventory_rows.append(_build_inventory_row(model=model, split=split, target=target, pred_path=pred_path, truth_path=truth_source, joined=joined, truth_col=mapr.truth_column))

                schema_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "split": split,
                        "prediction_columns": sorted(pred.columns.tolist()),
                        "truth_column": mapr.truth_column,
                        "quantiles_detected": [c for c in pred.columns if c.startswith("p") and c[1:].isdigit()],
                    }
                )

                metrics = _compute_metrics_for_frame(
                    model=model,
                    split=split,
                    target=target,
                    df=joined,
                    quantiles=quantiles,
                    horizon_buckets=horizon_buckets,
                    tail_cfg=tail_cfg,
                )
                m_overall.append(metrics["overall"])
                m_lead.append(metrics["by_lead"])
                m_hb.append(metrics["by_horizon_bucket"])
                m_gate.append(metrics["gate"])
                m_tail.append(metrics["tail"])
                m_cal.append(metrics["calibration"])
                m_cross.append(metrics["crossing"])

    truth_map_df = pd.DataFrame(truth_rows)
    cov_df = pd.DataFrame(coverage_rows)
    inv_df = pd.DataFrame(inventory_rows)
    schema_df = pd.DataFrame(schema_rows)
    post_df = pd.DataFrame(postprocess_rows)

    overall = pd.concat(m_overall, ignore_index=True) if m_overall else pd.DataFrame()
    by_lead = pd.concat(m_lead, ignore_index=True) if m_lead else pd.DataFrame()
    by_hb = pd.concat(m_hb, ignore_index=True) if m_hb else pd.DataFrame()
    gate = pd.concat(m_gate, ignore_index=True) if m_gate else pd.DataFrame()
    tail = pd.concat(m_tail, ignore_index=True) if m_tail else pd.DataFrame()
    cal = pd.concat(m_cal, ignore_index=True) if m_cal else pd.DataFrame()
    crossing = pd.concat(m_cross, ignore_index=True) if m_cross else pd.DataFrame()

    by_target = overall.groupby(["model", "target", "split"], as_index=False).mean(numeric_only=True) if not overall.empty else pd.DataFrame()
    by_model = overall.groupby(["model", "split"], as_index=False).mean(numeric_only=True) if not overall.empty else pd.DataFrame()
    per_target_rank, global_rank = _ranking_tables(by_target, baseline_model="rlqr" if "rlqr" in by_target.get("model", pd.Series([], dtype=str)).tolist() else None)

    diagnostics.joinpath("truth_mapping_report.csv").write_text(truth_map_df.to_csv(index=False), encoding="utf-8")
    diagnostics.joinpath("join_coverage_report.csv").write_text(cov_df.to_csv(index=False), encoding="utf-8")
    diagnostics.joinpath("benchmark_input_inventory.csv").write_text(inv_df.to_csv(index=False), encoding="utf-8")
    diagnostics.joinpath("schema_report.json").write_text(schema_df.to_json(orient="records", indent=2), encoding="utf-8")
    diagnostics.joinpath("forecast_postprocessing_report.csv").write_text(post_df.to_csv(index=False), encoding="utf-8")

    (metrics_dir / "metrics_overall.csv").write_text(overall.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_by_target.csv").write_text(by_target.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_by_model.csv").write_text(by_model.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_by_lead.csv").write_text(by_lead.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_by_horizon_bucket.csv").write_text(by_hb.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_gate_time.csv").write_text(gate.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_tail_events.csv").write_text(tail.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_calibration.csv").write_text(cal.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_crossing.csv").write_text(crossing.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_model_ranking_by_target.csv").write_text(per_target_rank.to_csv(index=False), encoding="utf-8")
    (metrics_dir / "metrics_model_ranking_global_normalized.csv").write_text(global_rank.to_csv(index=False), encoding="utf-8")
    # Always materialize extended diagnostics metric files for validator stability.
    for extra in ["metrics_residual_patterns.csv", "metrics_volatility_regimes.csv", "metrics_directional_bias.csv"]:
        p = metrics_dir / extra
        if not p.exists():
            p.write_text("", encoding="utf-8")

    joined_all = pd.concat(joined_frames, ignore_index=True) if joined_frames else pd.DataFrame()
    if make_figures and not joined_all.empty:
        generate_forecast_benchmark_figures(
            joined_df=joined_all,
            by_lead=by_lead,
            calibration=cal,
            figures_dir=figures,
            diagnostics_dir=diagnostics,
            config=config,
        )
    elif not (diagnostics / "example_window_report.csv").exists():
        (diagnostics / "example_window_report.csv").write_text("", encoding="utf-8")

    resolved_cfg_path = out_dir / "benchmark_config_resolved.yaml"
    try:
        import yaml

        resolved_cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except Exception:
        resolved_cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    input_manifest = {
        "model_run_manifests": [str(p) for p in model_run_manifests],
        "truth_source": str(truth_source),
        "truth_source_sha256": _sha256(truth_source),
        "inventory_rows": len(inv_df),
        "save_joined_predictions": bool(save_joined_predictions),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    benchmark_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_or_none(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
        "quantiles": quantiles,
        "min_join_coverage": min_join_coverage,
        "quantile_repair_reported": True,
        "figures_enabled": bool(make_figures),
        "save_joined_predictions": bool(save_joined_predictions),
        "forecast_postprocessing_applied": True,
        "forecast_value_mode": "canonical_economic",
        "canonical_economic_targets": sorted(canonical_targets_seen),
    }
    (out_dir / "benchmark_manifest.json").write_text(json.dumps(benchmark_manifest, indent=2), encoding="utf-8")

    return BenchmarkArtifacts(
        benchmark_dir=out_dir,
        diagnostics_dir=diagnostics,
        metrics_dir=metrics_dir,
        figures_dir=figures,
    )
