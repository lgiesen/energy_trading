#!/usr/bin/env python3
"""SHAP audit + feature pruning utility for XGBoost targets.

Implements:
1) Environment audit for SHAP artifacts and SHAP-using scripts.
2) Decision logic:
   - If SHAP value files exist -> load and plot top-10 summary per target.
   - Else if SHAP scripts exist -> compute SHAP via TreeExplainer on held-out test.
   - Else -> fallback still computes SHAP (fresh path) if model/data are provided.
3) Feature pruning output (>1% impact share threshold).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Script usage from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_trading.models.prepare_ml_bundles import load_processed_data


SHAP_ARTIFACT_EXTS = {".pkl", ".npy", ".csv", ".png", ".json", ".parquet"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_environment(root: Path) -> dict[str, Any]:
    """Scan repo for SHAP artifacts and SHAP imports."""
    skip_dirs = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}

    def _should_skip(path: Path) -> bool:
        return any(part in skip_dirs for part in path.parts)

    shap_artifacts: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip(p):
            continue
        if "shap" not in p.name.lower():
            continue
        if p.suffix.lower() in SHAP_ARTIFACT_EXTS:
            shap_artifacts.append(str(p.relative_to(root)))

    shap_import_scripts: list[str] = []
    py_files = list(root.rglob("*.py"))
    ipynb_files = list(root.rglob("*.ipynb"))
    import_regex = re.compile(r"\b(import\s+shap|from\s+shap\b)")
    for p in [*py_files, *ipynb_files]:
        if _should_skip(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if import_regex.search(text):
            shap_import_scripts.append(str(p.relative_to(root)))

    shap_value_files = [
        f
        for f in shap_artifacts
        if Path(f).suffix.lower() in {".pkl", ".npy", ".parquet", ".json"}
        and "shap_values" in Path(f).name.lower()
    ]

    return {
        "generated_at_utc": _utc_now_iso(),
        "scan_root": str(root.resolve()),
        "n_shap_artifacts": len(shap_artifacts),
        "n_shap_import_scripts": len(shap_import_scripts),
        "n_shap_value_files": len(shap_value_files),
        "shap_artifacts": sorted(shap_artifacts),
        "shap_import_scripts": sorted(shap_import_scripts),
        "shap_value_files": sorted(shap_value_files),
    }


def _unwrap_single_xgb(model_obj: Any) -> XGBRegressor:
    """Return a single XGBRegressor for SHAP from wrappers/payloads."""
    if isinstance(model_obj, XGBRegressor):
        return model_obj
    if hasattr(model_obj, "estimators_"):
        ests = getattr(model_obj, "estimators_", None)
        if isinstance(ests, (list, tuple, np.ndarray)) and len(ests) > 0:
            first = ests[0]
            if isinstance(first, XGBRegressor):
                return first
    raise TypeError(f"Unsupported model type for SHAP: {type(model_obj)}")


def _extract_target_p50_lead1_model(model_payload: Any, target: str) -> XGBRegressor:
    """Extract target-specific h+1 p50 model from train_xgboost_export payload."""
    payload = model_payload
    if isinstance(payload, dict) and target in payload:
        payload = payload[target]

    if isinstance(payload, dict):
        # Multi-horizon dict {1: {"p50": model, ...}, ...}
        if 1 in payload and isinstance(payload[1], dict):
            lead1 = payload[1]
            if "p50" in lead1:
                return _unwrap_single_xgb(lead1["p50"])
        # Direct quantile dict {"p50": model, ...}
        if "p50" in payload:
            return _unwrap_single_xgb(payload["p50"])

    # Single model fallback
    return _unwrap_single_xgb(payload)


def _select_model_path(run_dir: Path) -> Path:
    model_dir = run_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found under run dir: {model_dir}")
    candidates = sorted(model_dir.glob("*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No .joblib model files in: {model_dir}")
    # Prefer xgboost export naming.
    xgb_first = [p for p in candidates if "xgboost" in p.name.lower()]
    return xgb_first[0] if xgb_first else candidates[0]


def _resolve_targets(bundle: str, base_dir: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    cfg = json.loads((base_dir / "feature_config.json").read_text(encoding="utf-8"))
    if bundle not in cfg["bundles"]:
        raise KeyError(f"Unknown bundle '{bundle}' in feature_config.")
    return list(cfg["bundles"][bundle]["targets"])


def _coerce_shap_array(obj: Any) -> np.ndarray:
    if hasattr(obj, "values"):  # shap.Explanation
        arr = np.asarray(obj.values)
    else:
        arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def pruned_feature_list_from_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    min_impact_share: float = 0.01,
) -> list[str]:
    """Return features contributing more than min_impact_share of total impact."""
    arr = _coerce_shap_array(shap_values)
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature mismatch: shap has {arr.shape[1]} columns, "
            f"feature_names has {len(feature_names)}."
        )
    mean_abs = np.mean(np.abs(arr), axis=0)
    total = float(np.sum(mean_abs))
    if total <= 0:
        return []
    shares = mean_abs / total
    keep = [feature_names[i] for i, s in enumerate(shares) if float(s) > float(min_impact_share)]
    return keep


def _save_top10_bar_plot(
    mean_abs_series: pd.Series,
    target: str,
    out_png: Path,
) -> None:
    top = mean_abs_series.sort_values(ascending=False).head(10).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"Top 10 SHAP Features - {target}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _compute_shap_for_target(
    *,
    model_payload: Any,
    target: str,
    bundle: str,
    base_dir: Path,
    out_dir: Path,
    background_max_rows: int,
    test_max_rows: int,
    min_impact_share: float,
    seed: int,
) -> dict[str, Any]:
    import shap  # type: ignore

    X_train, _ = load_processed_data(
        bundle=bundle, split="train", base_dir=base_dir, target_col_for_feature_routing=target
    )
    X_test, _ = load_processed_data(
        bundle=bundle, split="test", base_dir=base_dir, target_col_for_feature_routing=target
    )
    if len(X_test) == 0:
        raise ValueError(f"Empty test split for target '{target}'.")

    feature_names = list(X_test.columns)
    model = _extract_target_p50_lead1_model(model_payload, target=target)

    if len(X_train) > background_max_rows:
        bg = shap.sample(X_train, background_max_rows, random_state=seed)
    else:
        bg = X_train

    if len(X_test) > test_max_rows:
        X_eval = X_test.sample(n=test_max_rows, random_state=seed).copy()
    else:
        X_eval = X_test.copy()

    explainer = shap.TreeExplainer(model, data=bg)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_arr = _coerce_shap_array(shap_values)

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"shap_values_{target}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "target": target,
                "feature_names": feature_names,
                "shap_values": shap_arr,
                "n_rows": int(shap_arr.shape[0]),
                "n_features": int(shap_arr.shape[1]),
                "generated_at_utc": _utc_now_iso(),
            },
            f,
        )

    # SHAP summary plot (top10).
    summary_png = out_dir / f"shap_summary_{target}.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_arr, X_eval, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=200)
    plt.close()

    mean_abs = pd.Series(np.abs(shap_arr).mean(axis=0), index=feature_names, dtype=float)
    top10_csv = out_dir / f"shap_top10_{target}.csv"
    mean_abs.sort_values(ascending=False).head(10).rename("mean_abs_shap").to_csv(top10_csv, index=True)

    pruned = pruned_feature_list_from_shap(
        shap_values=shap_arr, feature_names=feature_names, min_impact_share=min_impact_share
    )
    pruned_json = out_dir / f"pruned_feature_list_{target}.json"
    pruned_json.write_text(
        json.dumps(
            {
                "target": target,
                "min_impact_share": float(min_impact_share),
                "n_total_features": int(len(feature_names)),
                "n_pruned_features": int(len(pruned)),
                "pruned_feature_list": pruned,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Additional deterministic top10 bar.
    top10_bar_png = out_dir / f"shap_top10_bar_{target}.png"
    _save_top10_bar_plot(mean_abs, target=target, out_png=top10_bar_png)

    return {
        "target": target,
        "status": "computed_from_model",
        "shap_values_path": str(pkl_path),
        "summary_plot_path": str(summary_png),
        "top10_csv_path": str(top10_csv),
        "top10_bar_plot_path": str(top10_bar_png),
        "pruned_feature_list_path": str(pruned_json),
        "n_eval_rows": int(len(X_eval)),
        "n_features": int(len(feature_names)),
    }


def _load_existing_shap_and_plot(
    shap_value_files: list[Path],
    out_dir: Path,
    min_impact_share: float,
) -> list[dict[str, Any]]:
    """Load existing SHAP value files and generate top-10 summary outputs."""
    results: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in shap_value_files:
        target = p.stem.replace("shap_values_", "")
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)
            shap_values = payload["shap_values"] if isinstance(payload, dict) else payload
            feature_names = payload.get("feature_names") if isinstance(payload, dict) else None
            arr = _coerce_shap_array(shap_values)
            if not feature_names:
                feature_names = [f"f{i}" for i in range(arr.shape[1])]

            mean_abs = pd.Series(np.abs(arr).mean(axis=0), index=feature_names, dtype=float)
            bar_out = out_dir / f"shap_top10_bar_{target}.png"
            _save_top10_bar_plot(mean_abs, target=target, out_png=bar_out)

            top10_csv = out_dir / f"shap_top10_{target}.csv"
            mean_abs.sort_values(ascending=False).head(10).rename("mean_abs_shap").to_csv(top10_csv, index=True)

            pruned = pruned_feature_list_from_shap(arr, list(feature_names), min_impact_share=min_impact_share)
            pruned_json = out_dir / f"pruned_feature_list_{target}.json"
            pruned_json.write_text(
                json.dumps(
                    {
                        "target": target,
                        "min_impact_share": float(min_impact_share),
                        "n_total_features": int(len(feature_names)),
                        "n_pruned_features": int(len(pruned)),
                        "pruned_feature_list": pruned,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            results.append(
                {
                    "target": target,
                    "status": "loaded_existing_shap_values",
                    "source_shap_values_path": str(p),
                    "top10_csv_path": str(top10_csv),
                    "top10_bar_plot_path": str(bar_out),
                    "pruned_feature_list_path": str(pruned_json),
                    "n_features": int(len(feature_names)),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "target": target,
                    "status": "failed_loading_existing_shap_values",
                    "source_shap_values_path": str(p),
                    "error": str(exc),
                }
            )
    return results


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SHAP audit + target-wise pruning utility.")
    p.add_argument("--root", default=".", help="Project root to scan for SHAP artifacts/imports.")
    p.add_argument("--base-dir", default="data/model_input", help="Bundle base directory.")
    p.add_argument("--bundle", choices=["da", "afrr"], default="afrr", help="Bundle to analyze.")
    p.add_argument("--run-dir", default="", help="Model run dir for loading trained XGBoost payload.")
    p.add_argument("--model-path", default="", help="Optional explicit model joblib path.")
    p.add_argument("--targets", default="", help="Comma-separated target list. Defaults to bundle targets.")
    p.add_argument("--out-dir", default="data/reports/shap_pruning", help="Output directory.")
    p.add_argument("--background-max-rows", type=int, default=1000, help="Max rows for SHAP background set.")
    p.add_argument("--test-max-rows", type=int, default=5000, help="Max rows to explain from test set.")
    p.add_argument("--impact-threshold", type=float, default=0.01, help="Min impact share for pruning.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-json-out", default="", help="Optional explicit audit summary json path.")
    p.add_argument(
        "--mode",
        choices=["auto", "audit_only", "compute_only", "load_only"],
        default="auto",
        help="Execution mode. 'auto' follows requested decision logic.",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()

    root = Path(args.root).resolve()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    audit = audit_environment(root=root)
    audit_out = Path(args.audit_json_out) if args.audit_json_out else out_dir / "shap_environment_audit.json"
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"[audit] shap artifacts: {audit['n_shap_artifacts']}")
    print(f"[audit] shap import scripts: {audit['n_shap_import_scripts']}")
    print(f"[audit] shap value files: {audit['n_shap_value_files']}")
    print(f"[audit] summary json: {audit_out}")

    if args.mode == "audit_only":
        return

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] or None
    targets = _resolve_targets(bundle=args.bundle, base_dir=base_dir, explicit=target_list)

    shap_value_files = [root / rel for rel in audit["shap_value_files"]]
    shap_scripts_exist = audit["n_shap_import_scripts"] > 0

    results: list[dict[str, Any]] = []
    mode_used = ""

    if args.mode == "load_only" or (args.mode == "auto" and len(shap_value_files) > 0):
        mode_used = "load_existing_shap_values"
        results = _load_existing_shap_and_plot(
            shap_value_files=shap_value_files,
            out_dir=out_dir,
            min_impact_share=float(args.impact_threshold),
        )
    else:
        mode_used = "compute_treeexplainer"
        if args.model_path.strip():
            model_path = Path(args.model_path)
        elif args.run_dir.strip():
            model_path = _select_model_path(Path(args.run_dir))
        else:
            # In auto mode with script existing but no SHAP values, we still need model path.
            # Try latest pointer as convenience.
            latest = Path("artifacts/model_runs/latest.json")
            if latest.exists():
                latest_payload = json.loads(latest.read_text(encoding="utf-8"))
                rid = str(latest_payload.get("run_id", "")).strip()
                if rid:
                    model_path = _select_model_path(Path("artifacts/model_runs") / rid)
                else:
                    raise ValueError("Could not resolve run_id from artifacts/model_runs/latest.json.")
            elif args.mode == "compute_only" or shap_scripts_exist:
                raise ValueError("Model path not resolvable. Provide --run-dir or --model-path.")
            else:
                raise ValueError("No SHAP values and no model path provided.")

        model_payload = joblib.load(model_path)
        print(f"[shap] model path: {model_path}")
        for tgt in targets:
            print(f"[shap] target={tgt} ...")
            rec = _compute_shap_for_target(
                model_payload=model_payload,
                target=tgt,
                bundle=args.bundle,
                base_dir=base_dir,
                out_dir=out_dir,
                background_max_rows=int(args.background_max_rows),
                test_max_rows=int(args.test_max_rows),
                min_impact_share=float(args.impact_threshold),
                seed=int(args.seed),
            )
            results.append(rec)

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "mode_requested": args.mode,
        "mode_used": mode_used,
        "decision_inputs": {
            "n_shap_value_files": audit["n_shap_value_files"],
            "shap_scripts_exist": shap_scripts_exist,
        },
        "bundle": args.bundle,
        "targets": targets,
        "results": results,
    }
    summary_path = out_dir / "shap_feature_pruning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] SHAP feature pruning summary: {summary_path}")


if __name__ == "__main__":
    main()
