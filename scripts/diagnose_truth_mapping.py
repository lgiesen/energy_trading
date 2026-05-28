#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from difflib import get_close_matches

import pandas as pd

PRED_KEY_TO_TARGET = {
    "pred_da_price": "da_price",
    "pred_afrr_capacity_price_pos": "afrr_capacity_price_pos",
    "pred_afrr_capacity_price_neg": "afrr_capacity_price_neg",
    "pred_afrr_activation_price_pos": "afrr_activation_price_pos",
    "pred_afrr_activation_price_neg": "afrr_activation_price_neg",
    "pred_afrr_activation_rate_pos": "afrr_activation_rate_pos",
    "pred_afrr_activation_rate_neg": "afrr_activation_rate_neg",
}

TRUTH_COLUMN_CANDIDATES = {
    "pred_da_price": ["da_price", "target_da_price", "true_da_price"],
    "pred_afrr_capacity_price_pos": [
        "afrr_capacity_price_pos",
        "target_afrr_capacity_price_pos",
        "true_afrr_capacity_price_pos",
    ],
    "pred_afrr_capacity_price_neg": [
        "afrr_capacity_price_neg",
        "target_afrr_capacity_price_neg",
        "true_afrr_capacity_price_neg",
    ],
    "pred_afrr_activation_price_pos": [
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_pos",
        "target_afrr_activation_price_pos",
        "true_afrr_activation_price_pos",
    ],
    "pred_afrr_activation_price_neg": [
        "afrr_activation_price_vwap_neg",
        "afrr_activation_price_neg",
        "target_afrr_activation_price_neg",
        "true_afrr_activation_price_neg",
    ],
    "pred_afrr_activation_rate_pos": [
        "activation_rate_phys_pos",
        "afrr_activation_rate_pos",
        "target_afrr_activation_rate_pos",
        "true_afrr_activation_rate_pos",
    ],
    "pred_afrr_activation_rate_neg": [
        "activation_rate_phys_neg",
        "afrr_activation_rate_neg",
        "target_afrr_activation_rate_neg",
        "true_afrr_activation_rate_neg",
    ],
}

QUANTILES_REQUIRED = ["p10", "p30", "p50", "p70", "p90"]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _resolve_latest_manifest_pointer(latest_path: Path) -> Path:
    if not latest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {latest_path}")
    data = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Latest manifest must be dict: {latest_path}")

    run_manifest_rel = data.get("manifest_path") or data.get("manifest_path_abs")
    if isinstance(run_manifest_rel, str) and run_manifest_rel.strip():
        cand = (latest_path.parent / run_manifest_rel).resolve()
        if cand.exists():
            return cand

    run_id = data.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        cand = (latest_path.parent / run_id / "manifest.json").resolve()
        if cand.exists():
            return cand

    if any(k in data for k in ("predictions_path", "prediction_path", "outputs")):
        return latest_path

    raise KeyError(
        f"Could not resolve run manifest from latest manifest pointer: {latest_path}. "
        f"Available keys: {list(data.keys())}"
    )


def _resolve_bundle_split_truth_path(repo_root: Path, bundle_name: str, split: str) -> Path | None:
    cand = (repo_root / "data" / "model_input" / str(bundle_name) / f"{split}.parquet").resolve()
    return cand if cand.exists() else None


def _resolve_truth_column_for_pred_key(truth_df: pd.DataFrame, pred_key: str) -> str:
    candidates = TRUTH_COLUMN_CANDIDATES[pred_key]
    matches = [c for c in candidates if c in truth_df.columns]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        nearby = get_close_matches(pred_key, list(truth_df.columns), n=10, cutoff=0.35)
        raise KeyError(
            f"No truth column match for pred_key={pred_key}. Tried={candidates}. "
            f"Nearby columns={nearby}"
        )
    raise ValueError(
        f"Ambiguous truth mapping for pred_key={pred_key}. Matches={matches}. Candidates={candidates}"
    )


def _iter_bundle_long_tables(run_manifest_path: Path, bundles: dict, split_filter: str | None) -> list[tuple[str, str, str, Path]]:
    out: list[tuple[str, str, str, Path]] = []
    for bundle_name, bundle_obj in bundles.items():
        if not isinstance(bundle_obj, dict):
            continue
        pl = bundle_obj.get("predictions_long")
        if not isinstance(pl, dict):
            continue
        for split, split_obj in pl.items():
            if split_filter and split != split_filter:
                continue
            if not isinstance(split_obj, dict):
                continue
            for pred_key, rel_path in split_obj.items():
                if pred_key not in PRED_KEY_TO_TARGET:
                    continue
                if not isinstance(rel_path, str) or not rel_path.strip():
                    continue
                out.append((str(bundle_name), str(split), str(pred_key), (run_manifest_path.parent / rel_path).resolve()))
    return out


def diagnose(manifest: Path, truth_override: Path | None, split: str | None, coverage_threshold: float) -> int:
    repo_root = Path.cwd().resolve()
    run_manifest_path = _resolve_latest_manifest_pointer(manifest)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    bundles = run_manifest.get("bundles")
    if not isinstance(bundles, dict):
        raise KeyError(f"Missing bundles in run manifest: {run_manifest_path}")

    rows = []
    failures = []

    for bundle_name, split_name, pred_key, pred_path in _iter_bundle_long_tables(run_manifest_path, bundles, split):
        pred = _read_table(pred_path)
        miss_q = [q for q in QUANTILES_REQUIRED if q not in pred.columns]
        if miss_q:
            failures.append(f"{pred_path.name}: missing quantiles {miss_q}")
            continue

        if "target_time_utc" not in pred.columns:
            failures.append(f"{pred_path.name}: missing target_time_utc")
            continue

        if truth_override is not None:
            truth_path = truth_override
        else:
            bundle_truth = _resolve_bundle_split_truth_path(repo_root, bundle_name, split_name)
            if bundle_truth is None:
                gt = run_manifest.get("ground_truth", {})
                p = gt.get("default_path") if isinstance(gt, dict) else None
                if not isinstance(p, str) or not p.strip():
                    failures.append(f"{pred_path.name}: no truth path available")
                    continue
                truth_path = (run_manifest_path.parent / p).resolve()
            else:
                truth_path = bundle_truth

        truth = _read_table(truth_path)
        ts_col = "timestamp_utc" if "timestamp_utc" in truth.columns else ("target_time_utc" if "target_time_utc" in truth.columns else None)
        if ts_col is None:
            failures.append(f"{pred_path.name}: truth missing timestamp_utc/target_time_utc in {truth_path}")
            continue

        try:
            truth_col = _resolve_truth_column_for_pred_key(truth, pred_key)
        except Exception as e:
            failures.append(f"{pred_path.name}: {e}")
            continue

        truth_sub = truth[[ts_col, truth_col]].copy()
        truth_sub["timestamp_utc"] = pd.to_datetime(truth_sub[ts_col], utc=True, errors="coerce")
        truth_sub = truth_sub.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"], keep="last")

        pred_sub = pred[["target_time_utc"]].copy()
        pred_sub["target_time_utc"] = pd.to_datetime(pred_sub["target_time_utc"], utc=True, errors="coerce")

        merged = pred_sub.merge(truth_sub[["timestamp_utc", truth_col]], left_on="target_time_utc", right_on="timestamp_utc", how="left", validate="m:1")
        pred_rows = len(merged)
        missing = int(merged[truth_col].isna().sum())
        coverage = 0.0 if pred_rows == 0 else (pred_rows - missing) / pred_rows
        status = "ok" if coverage >= coverage_threshold else "coverage_fail"

        if status != "ok":
            failures.append(
                f"{pred_path.name}: coverage {coverage:.4%} below threshold {coverage_threshold:.4%} "
                f"(missing {missing}/{pred_rows})"
            )

        rows.append(
            {
                "model": run_manifest.get("training", {}).get("model_name", "unknown"),
                "target": PRED_KEY_TO_TARGET[pred_key],
                "split": split_name,
                "pred_key": pred_key,
                "pred_rows": pred_rows,
                "truth_col": truth_col,
                "coverage_pct": coverage * 100.0,
                "missing": missing,
                "status": status,
                "prediction_file": str(pred_path),
                "truth_file": str(truth_path),
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["target", "split", "pred_key"])
        print(df.to_csv(index=False))
    else:
        print("No prediction tables found to diagnose.")

    if failures:
        print("Failures:")
        for f in failures:
            print("-", f)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose truth mapping coverage for benchmark notebook predictions.")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to latest_*.json or run manifest JSON.")
    ap.add_argument("--truth", type=Path, default=None, help="Optional explicit truth table path.")
    ap.add_argument("--split", type=str, default=None, help="Optional split filter (e.g. test).")
    ap.add_argument("--coverage-threshold", type=float, default=0.999, help="Minimum acceptable truth join coverage.")
    ap.add_argument("--self-check", action="store_true", help="Run lightweight resolver self-checks and exit.")
    args = ap.parse_args()
    if args.self_check:
        truth = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"], utc=True),
                "da_price": [10.0, 11.0],
                "afrr_activation_price_vwap_pos": [100.0, 120.0],
            }
        )
        assert _resolve_truth_column_for_pred_key(truth, "pred_da_price") == "da_price"
        assert _resolve_truth_column_for_pred_key(truth, "pred_afrr_activation_price_pos") == "afrr_activation_price_vwap_pos"
        try:
            _resolve_truth_column_for_pred_key(pd.DataFrame({"timestamp_utc": []}), "pred_da_price")
        except KeyError:
            pass
        else:
            raise AssertionError("Expected KeyError for missing da_price mapping")
        try:
            _resolve_truth_column_for_pred_key(
                pd.DataFrame({"timestamp_utc": [], "da_price": [], "target_da_price": []}),
                "pred_da_price",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for ambiguous da_price mapping")
        print("self-check passed")
        return 0
    return diagnose(args.manifest.resolve(), args.truth.resolve() if args.truth else None, args.split, args.coverage_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
