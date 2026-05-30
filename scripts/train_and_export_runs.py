"""Train DA + aFRR models and export a versioned run artifact."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AFRR_TARGETS = [
    "target_afrr_activation_price_vwap_pos",
    "target_afrr_activation_price_vwap_neg",
    "target_afrr_activation_rate_pos",
    "target_afrr_activation_rate_neg",
    "target_afrr_capacity_price_pos",
    "target_afrr_capacity_price_neg",
]
DA_TARGET = "target_da_price"


def filter_afrr_targets(available_targets: list[str], requested_csv: str) -> list[str]:
    requested = {t.strip() for t in str(requested_csv).split(",") if t.strip()}
    if not requested:
        return list(available_targets)
    return [t for t in available_targets if t in requested]


def load_hpo_artifact_map(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid HPO artifact map in {path}: expected JSON object.")
    out: dict[str, str] = {}
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RuntimeError(f"Invalid HPO artifact map entry in {path}: keys/values must be strings.")
        out[k.strip()] = v.strip()
    return out


def load_hpo_best_params(path: str | Path) -> dict[str, object]:
    hpo_path = Path(path)
    if not hpo_path.exists():
        raise FileNotFoundError(f"HPO artifact not found: {hpo_path}")
    payload = json.loads(hpo_path.read_text(encoding="utf-8"))
    best_params = payload.get("best_params", {})
    if not isinstance(best_params, dict):
        raise RuntimeError(f"Invalid HPO artifact format in {hpo_path}: missing dict 'best_params'")
    return best_params


def load_hpo_artifact_payload(path: str | Path) -> dict[str, object]:
    hpo_path = Path(path)
    if not hpo_path.exists():
        raise FileNotFoundError(f"HPO artifact not found: {hpo_path}")
    payload = json.loads(hpo_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid HPO artifact format in {hpo_path}: expected object")
    return payload


def hpo_override_cli_args(model_type: str, best_params: dict[str, object]) -> list[str]:
    def _v(key: str) -> object | None:
        return best_params.get(key)

    args: list[str] = []
    if model_type == "xgboost":
        mapping = [
            ("max_depth", "--max-depth"),
            ("learning_rate", "--learning-rate"),
            ("subsample", "--subsample"),
            ("colsample_bytree", "--colsample-bytree"),
            ("min_child_weight", "--min-child-weight"),
            ("reg_alpha", "--reg-alpha"),
            ("reg_lambda", "--reg-lambda"),
        ]
    elif model_type == "linear":
        mapping = [
            ("alpha", "--alpha"),
            ("l1_ratio", "--l1-ratio"),
            ("learning_rate", "--learning-rate"),
            ("eta0", "--eta0"),
        ]
    elif model_type == "tft":
        mapping = [
            ("hidden_size", "--hidden-size"),
            ("attention_head_size", "--attention-head-size"),
            ("dropout", "--dropout"),
            ("learning_rate", "--learning-rate"),
            ("gradient_clip_val", "--gradient-clip-val"),
            ("max_encoder_length", "--max-encoder-length"),
            ("max_epochs", "--max-epochs"),
            ("early_stopping_patience", "--early-stopping-patience"),
        ]
    else:
        mapping = []
    for key, flag in mapping:
        val = _v(key)
        if val is not None:
            args.extend([flag, str(val)])
    return args


def validate_hpo_cli_choice(hpo_artifact: str | None, hpo_artifact_map: str | None) -> None:
    if hpo_artifact and hpo_artifact_map:
        raise ValueError("Use either --hpo-artifact or --hpo-artifact-map, not both.")


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _normalize_json_paths_inplace(json_path: Path, *, repo_root: Path) -> bool:
    """Rewrite absolute repo/server paths in one JSON file to relative paths."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    changed = False
    base_dir = json_path.parent.resolve()
    repo_root = repo_root.resolve()

    def _to_rel(value: object) -> object:
        nonlocal changed
        if isinstance(value, dict):
            return {k: _to_rel(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_rel(v) for v in value]
        if not isinstance(value, str):
            return value

        s = value.strip()
        if not s:
            return value

        local_abs: Path | None = None
        p = Path(s)
        if p.is_absolute():
            local_abs = p
        elif "/energy_trading/" in s:
            suffix = s.split("/energy_trading/", 1)[1]
            local_abs = repo_root / suffix
        elif "/artifacts/model_runs/" in s:
            suffix = s.split("/artifacts/model_runs/", 1)[1]
            local_abs = repo_root / "artifacts" / "model_runs" / suffix

        if local_abs is None:
            return value
        try:
            rel = os.path.relpath(str(local_abs.resolve()), start=str(base_dir))
            changed = True
            return str(rel)
        except Exception:
            return value

    normalized = _to_rel(payload)
    if changed:
        json_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return changed


def _cmd_to_shell_string(cmd: list[str]) -> str:
    try:
        return shlex.join(cmd)
    except Exception:
        return " ".join(cmd)


def _safe_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        out = proc.stdout.strip()
        return out or None
    except Exception:
        return None


def _slugify_token(value: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in str(value))
    out = out.strip("._")
    return out or "cmd"


def _isoformat_or_none(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _run_train_cmd(cmd: list[str], *, run_dir: Path, log_stem: str) -> dict[str, object]:
    shell_cmd = _cmd_to_shell_string(cmd)
    print("[CMD]", shell_cmd)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_slugify_token(log_stem)}.log"
    t0 = time.time()
    return_code = 0
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
        return_code = int(proc.returncode or 0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    t1 = time.time()
    return {
        "command": cmd,
        "command_shell": shell_cmd,
        "return_code": return_code,
        "log_path": str(log_path.resolve()),
        "started_at_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.fromtimestamp(t1, tz=timezone.utc).isoformat(),
        "duration_seconds": float(t1 - t0),
    }


def _load_fragment(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest fragment: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_prediction_files(paths: list[Path], out_path: Path) -> Path:
    frames: list[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "timestamp_utc" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No prediction files to merge.")

    stacked = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    stacked["timestamp_utc"] = pd.to_datetime(stacked["timestamp_utc"], utc=True, errors="coerce")
    stacked = stacked.dropna(subset=["timestamp_utc"]).copy()

    def _first_non_null(s: pd.Series):
        nn = s.dropna()
        return nn.iloc[0] if not nn.empty else pd.NA

    merged = (
        stacked.sort_values("timestamp_utc")
        .groupby("timestamp_utc", as_index=False)
        .agg(_first_non_null)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return out_path


def _resolve_available_afrr_targets(base_dir: str | Path) -> list[str]:
    """Return only aFRR targets that are available in prepared bundle config."""
    cfg_path = Path(base_dir) / "feature_config.json"
    if not cfg_path.exists():
        return list(AFRR_TARGETS)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    available = list(cfg.get("bundles", {}).get("afrr", {}).get("targets", []))
    preferred = [t for t in AFRR_TARGETS if t in available]
    return preferred if preferred else available


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bundle_input_checksums(base_dir: str | Path) -> dict[str, object]:
    """Compute checksums for feature_config and referenced split files."""
    root = Path(base_dir)
    out: dict[str, object] = {
        "base_dir": str(root.resolve()),
        "feature_config_path": str((root / "feature_config.json").resolve()),
        "feature_config_sha256": None,
        "bundles": {},
        "missing_files": [],
    }
    cfg_path = root / "feature_config.json"
    if not cfg_path.exists():
        out["missing_files"] = [str(cfg_path.resolve())]
        return out

    out["feature_config_sha256"] = _sha256_file(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bundles = cfg.get("bundles", {})

    for bundle_name, bcfg in bundles.items():
        files = bcfg.get("files", {}) or {}
        bundle_rec: dict[str, object] = {"files": {}}
        for split_name, split_path_raw in files.items():
            split_path = Path(split_path_raw)
            rec = {
                "path": str(split_path.resolve()),
                "exists": split_path.exists(),
                "sha256": None,
                "size_bytes": None,
            }
            if split_path.exists():
                rec["sha256"] = _sha256_file(split_path)
                rec["size_bytes"] = int(split_path.stat().st_size)
            else:
                out["missing_files"].append(str(split_path.resolve()))
            bundle_rec["files"][split_name] = rec
        out["bundles"][bundle_name] = bundle_rec
    return out


def _bundle_data_integrity_report(base_dir: str | Path) -> dict[str, object]:
    """Build hard-proof checks for split alignment and leakage assertions."""
    root = Path(base_dir)
    cfg_path = root / "feature_config.json"
    out: dict[str, object] = {
        "base_dir": str(root.resolve()),
        "config_path": str(cfg_path.resolve()),
        "config_exists": cfg_path.exists(),
        "splits": {},
        "bundles": {},
        "passed": True,
        "failures": [],
    }
    if not cfg_path.exists():
        out["passed"] = False
        out["failures"].append("feature_config.json missing")
        return out

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    split_cfg = cfg.get("splits", {}) or {}
    train_end = pd.to_datetime(split_cfg.get("train_end_exclusive"), utc=True, errors="coerce")
    val_end = pd.to_datetime(split_cfg.get("val_end_exclusive"), utc=True, errors="coerce")
    test_end = pd.to_datetime(split_cfg.get("test_end_inclusive"), utc=True, errors="coerce")
    out["splits"] = {
        "train_end_exclusive": _isoformat_or_none(train_end),
        "val_end_exclusive": _isoformat_or_none(val_end),
        "test_end_inclusive": _isoformat_or_none(test_end),
        "purge_gap_rows": int(split_cfg.get("purge_gap_rows", 0)),
    }

    bundles = cfg.get("bundles", {}) or {}
    for bundle_name, bcfg in bundles.items():
        features = [str(c) for c in (bcfg.get("features", []) or [])]
        targets = [str(c) for c in (bcfg.get("targets", []) or [])]
        files = bcfg.get("files", {}) or {}
        bundle_rec: dict[str, object] = {
            "n_features": len(features),
            "n_targets": len(targets),
            "files": {},
            "leakage_assertions": {},
            "split_alignment": {},
            "passed": True,
            "failures": [],
        }

        feature_target_overlap = sorted(set(features).intersection(targets))
        feature_target_prefixed = sorted([c for c in features if c.startswith("target_")])
        bundle_rec["leakage_assertions"] = {
            "feature_target_overlap_count": len(feature_target_overlap),
            "feature_target_overlap_sample": feature_target_overlap[:20],
            "target_prefixed_features_count": len(feature_target_prefixed),
            "target_prefixed_features_sample": feature_target_prefixed[:20],
        }
        if feature_target_overlap:
            bundle_rec["passed"] = False
            bundle_rec["failures"].append("features overlap with targets")

        split_dfs: dict[str, pd.DataFrame] = {}
        split_ts_sets: dict[str, set[pd.Timestamp]] = {}
        for split_name in ("train", "val", "test"):
            p_raw = files.get(split_name, "")
            p = Path(p_raw) if p_raw else Path("")
            split_info = {
                "path": str(p.resolve()) if p_raw else "",
                "exists": bool(p_raw) and p.exists(),
                "rows": 0,
                "timestamp_min": None,
                "timestamp_max": None,
                "timestamp_monotonic_non_decreasing": None,
                "timestamp_duplicate_count": None,
                "timestamp_null_count": None,
                "missing_feature_columns_count": None,
                "missing_feature_columns_sample": [],
                "missing_target_columns_count": None,
                "missing_target_columns_sample": [],
            }
            if split_info["exists"]:
                df = pd.read_parquet(p)
                if "timestamp_utc" not in df.columns:
                    bundle_rec["passed"] = False
                    bundle_rec["failures"].append(f"{split_name}: timestamp_utc missing")
                else:
                    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
                    split_info["rows"] = int(len(df))
                    split_info["timestamp_null_count"] = int(ts.isna().sum())
                    ts_clean = ts.dropna()
                    split_info["timestamp_min"] = _isoformat_or_none(ts_clean.min() if not ts_clean.empty else None)
                    split_info["timestamp_max"] = _isoformat_or_none(ts_clean.max() if not ts_clean.empty else None)
                    split_info["timestamp_monotonic_non_decreasing"] = bool(ts_clean.is_monotonic_increasing)
                    split_info["timestamp_duplicate_count"] = int(ts_clean.duplicated().sum())
                    split_ts_sets[split_name] = set(pd.DatetimeIndex(ts_clean))
                    split_dfs[split_name] = pd.DataFrame({"timestamp_utc": ts_clean})
                    if split_info["timestamp_null_count"] > 0:
                        bundle_rec["passed"] = False
                        bundle_rec["failures"].append(f"{split_name}: null timestamps > 0")
                    if split_info["timestamp_duplicate_count"] > 0:
                        bundle_rec["passed"] = False
                        bundle_rec["failures"].append(f"{split_name}: duplicate timestamps > 0")

                cols = set(df.columns)
                missing_features = [c for c in features if c not in cols]
                missing_targets = [c for c in targets if c not in cols]
                split_info["missing_feature_columns_count"] = len(missing_features)
                split_info["missing_feature_columns_sample"] = missing_features[:20]
                split_info["missing_target_columns_count"] = len(missing_targets)
                split_info["missing_target_columns_sample"] = missing_targets[:20]
                if missing_features:
                    bundle_rec["passed"] = False
                    bundle_rec["failures"].append(f"{split_name}: missing configured feature columns")
                if missing_targets:
                    bundle_rec["passed"] = False
                    bundle_rec["failures"].append(f"{split_name}: missing configured target columns")
            else:
                bundle_rec["passed"] = False
                bundle_rec["failures"].append(f"{split_name}: split file missing")
            bundle_rec["files"][split_name] = split_info

        # Split alignment proofs (no overlap + chronological order).
        train_ts = split_ts_sets.get("train", set())
        val_ts = split_ts_sets.get("val", set())
        test_ts = split_ts_sets.get("test", set())
        overlap_train_val = int(len(train_ts.intersection(val_ts)))
        overlap_train_test = int(len(train_ts.intersection(test_ts)))
        overlap_val_test = int(len(val_ts.intersection(test_ts)))
        train_max = pd.Timestamp(max(train_ts)) if train_ts else None
        val_min = pd.Timestamp(min(val_ts)) if val_ts else None
        val_max = pd.Timestamp(max(val_ts)) if val_ts else None
        test_min = pd.Timestamp(min(test_ts)) if test_ts else None

        align = {
            "timestamp_overlap_train_val": overlap_train_val,
            "timestamp_overlap_train_test": overlap_train_test,
            "timestamp_overlap_val_test": overlap_val_test,
            "train_max_lt_val_min": bool(train_max < val_min) if train_max is not None and val_min is not None else None,
            "val_max_lt_test_min": bool(val_max < test_min) if val_max is not None and test_min is not None else None,
            "train_max": _isoformat_or_none(train_max),
            "val_min": _isoformat_or_none(val_min),
            "val_max": _isoformat_or_none(val_max),
            "test_min": _isoformat_or_none(test_min),
        }
        if train_max is not None and pd.notna(train_end):
            align["train_max_lt_train_end_exclusive"] = bool(train_max < train_end)
        if val_min is not None and pd.notna(train_end):
            align["val_min_ge_train_end_exclusive"] = bool(val_min >= train_end)
        if val_max is not None and pd.notna(val_end):
            align["val_max_lt_val_end_exclusive"] = bool(val_max < val_end)
        if test_min is not None and pd.notna(val_end):
            align["test_min_ge_val_end_exclusive"] = bool(test_min >= val_end)
        bundle_rec["split_alignment"] = align
        if overlap_train_val or overlap_train_test or overlap_val_test:
            bundle_rec["passed"] = False
            bundle_rec["failures"].append("timestamp overlap between splits")
        if align.get("train_max_lt_val_min") is False:
            bundle_rec["passed"] = False
            bundle_rec["failures"].append("train max timestamp is not < val min timestamp")
        if align.get("val_max_lt_test_min") is False:
            bundle_rec["passed"] = False
            bundle_rec["failures"].append("val max timestamp is not < test min timestamp")

        out["bundles"][bundle_name] = bundle_rec
        if not bundle_rec["passed"]:
            out["passed"] = False
            for msg in bundle_rec["failures"]:
                out["failures"].append(f"{bundle_name}: {msg}")
    return out


def _infer_prediction_family(pred_col: str) -> str:
    p = str(pred_col).lower()
    if "activation_rate" in p:
        return "activation_rate"
    if "activation_price" in p:
        return "activation_price"
    if "capacity_price" in p:
        return "capacity_price"
    if "da_price" in p:
        return "da_price"
    return "other"


def _prediction_output_quality_report(
    *,
    da_meta: dict[str, object] | None,
    afrr_meta_list: list[dict[str, object]],
    afrr_pred_val: Path,
    afrr_pred_test: Path,
) -> dict[str, object]:
    """Centralized quality checks for prediction artifacts."""
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_checked": [],
        "summary": {
            "n_files": 0,
            "n_failures": 0,
            "n_warnings": 0,
        },
        "passed": True,
    }

    checks: list[tuple[str, Path, str | None, str]] = []
    # Wide DA files
    for split, p in ((da_meta or {}).get("predictions", {}) or {}).items():
        checks.append((f"da_wide_{split}", Path(p), None, "wide"))
    # Wide merged aFRR files
    checks.append(("afrr_wide_val", afrr_pred_val, None, "wide"))
    checks.append(("afrr_wide_test", afrr_pred_test, None, "wide"))
    # Long DA files
    for split, pred_map in ((da_meta or {}).get("predictions_long", {}) or {}).items():
        for pred_col, p in (pred_map or {}).items():
            checks.append((f"da_long_{split}_{pred_col}", Path(p), pred_col, "long"))
    # Long aFRR files
    for meta in afrr_meta_list:
        for split, pred_map in (meta.get("predictions_long", {}) or {}).items():
            for pred_col, p in (pred_map or {}).items():
                checks.append((f"afrr_long_{split}_{pred_col}", Path(p), pred_col, "long"))

    dedup: dict[str, tuple[str, Path, str | None, str]] = {}
    for rec in checks:
        _, path, _, _ = rec
        dedup[str(path.resolve())] = rec
    checks = list(dedup.values())

    for file_id, path, pred_col_hint, file_kind in checks:
        rec: dict[str, object] = {
            "file_id": file_id,
            "path": str(path.resolve()),
            "kind": file_kind,
            "exists": path.exists(),
            "rows": 0,
            "pred_columns_checked": [],
            "failures": [],
            "warnings": [],
        }
        if not path.exists():
            rec["failures"].append("missing file")
            report["files_checked"].append(rec)
            continue

        df = pd.read_parquet(path)
        rec["rows"] = int(len(df))
        if len(df) == 0:
            rec["failures"].append("empty file")
            report["files_checked"].append(rec)
            continue

        pred_cols: list[str] = []
        if file_kind == "long":
            if "predicted_value" in df.columns:
                pred_cols.append("predicted_value")
            pred_cols += [c for c in df.columns if c.lower() in {"p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"}]
            if not pred_cols:
                rec["failures"].append("no prediction columns found in long file")
        else:
            pred_cols = [c for c in df.columns if str(c).startswith("pred_")]
            if not pred_cols:
                rec["warnings"].append("no pred_* columns found in wide file")

        for c in sorted(set(pred_cols)):
            s = pd.to_numeric(df[c], errors="coerce")
            nan_count = int(s.isna().sum())
            inf_count = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(s) else 0
            min_v = float(s.min()) if s.notna().any() else None
            max_v = float(s.max()) if s.notna().any() else None
            csum = {
                "column": c,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "min": min_v,
                "max": max_v,
            }
            family_src = pred_col_hint or c
            fam = _infer_prediction_family(family_src)
            if fam == "activation_rate" and s.notna().any():
                oor = int(((s < 0.0) | (s > 1.0)).sum())
                csum["out_of_range_0_1_count"] = oor
                if oor > 0:
                    rec["failures"].append(f"{c}: activation rate out of [0,1] in {oor} rows")
            if fam in {"da_price", "activation_price", "capacity_price"} and s.notna().any():
                extreme = int((s.abs() > 10000.0).sum())
                csum["extreme_abs_gt_10000_count"] = extreme
                if extreme > 0:
                    rec["warnings"].append(f"{c}: extreme |value| > 10000 in {extreme} rows")
            if inf_count > 0:
                rec["failures"].append(f"{c}: contains inf values ({inf_count})")
            rec["pred_columns_checked"].append(csum)

        report["files_checked"].append(rec)

    n_fail = 0
    n_warn = 0
    for r in report["files_checked"]:
        n_fail += len(r.get("failures", []))
        n_warn += len(r.get("warnings", []))
    report["summary"]["n_files"] = len(report["files_checked"])
    report["summary"]["n_failures"] = int(n_fail)
    report["summary"]["n_warnings"] = int(n_warn)
    report["passed"] = bool(n_fail == 0)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and export DA+aFRR model run artifacts.")
    p.add_argument("--base-dir", default="data/model_input")
    p.add_argument("--run-root", default="artifacts/model_runs")
    p.add_argument("--run-id", default="")
    p.add_argument("--model-type", choices=["xgboost", "tft", "linear"], default="xgboost")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default="mps")
    p.add_argument("--model-name", default="")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--forecast-horizon-hours", type=int, default=48)
    p.add_argument("--lead-weight-start", type=int, default=16)
    p.add_argument("--lead-weight-end", type=int, default=48)
    p.add_argument("--lead-weight-max", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--afrr-parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel aFRR target training jobs (useful for linear model runs).",
    )
    p.add_argument(
        "--afrr-targets",
        default="",
        help="Optional comma-separated aFRR target filter, e.g. target_afrr_activation_price_vwap_neg",
    )
    p.add_argument(
        "--skip-da",
        action="store_true",
        help="Skip day-ahead model training and train only selected aFRR targets.",
    )
    p.add_argument(
        "--lead-parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel lead-time jobs inside one linear target training run.",
    )
    p.add_argument("--linear-alpha", type=float, default=1.0)
    p.add_argument("--linear-l1-ratio", type=float, default=0.15)
    p.add_argument("--linear-learning-rate", default="invscaling")
    p.add_argument("--linear-eta0", type=float, default=0.01)
    p.add_argument("--tft-hidden-size", type=int, default=None)
    p.add_argument("--tft-attention-head-size", type=int, default=None)
    p.add_argument("--tft-dropout", type=float, default=None)
    p.add_argument("--tft-learning-rate", type=float, default=None)
    p.add_argument("--tft-gradient-clip-val", type=float, default=None)
    p.add_argument("--tft-max-encoder-length", type=int, default=None)
    p.add_argument("--tft-max-epochs", type=int, default=None)
    p.add_argument("--tft-early-stopping-patience", type=int, default=None)
    p.add_argument(
        "--tft-precision",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        default="auto",
    )
    p.add_argument(
        "--hpo-artifact",
        type=str,
        default=None,
        help="Path to Optuna/tuning JSON artifact",
    )
    p.add_argument(
        "--hpo-artifact-map",
        type=str,
        default=None,
        help="Path to JSON map: target_col -> tuning JSON artifact",
    )
    p.add_argument(
        "--cleanup-lightning-checkpoints",
        action="store_true",
        help="For TFT runs: delete intermediate Lightning checkpoint files under ./checkpoints after training.",
    )
    p.add_argument("--ground-truth-path", default="data/features/all_data_features.parquet")
    p.add_argument(
        "--enable-da-stacking-afrr",
        action="store_true",
        help="Build leakage-safe DA stacking feature and train aFRR on stacked bundle dir.",
    )
    p.add_argument(
        "--stacked-base-dir",
        default="data/model_input_stacked",
        help="Output dir for stacked bundles when --enable-da-stacking-afrr is used.",
    )
    p.add_argument("--stack-cv-n-splits", type=int, default=8)
    p.add_argument("--stack-cv-test-size", type=int, default=24 * 7)
    p.add_argument("--stack-cv-gap-hours", type=int, default=72)
    p.add_argument("--skip-latest-pointer", action="store_true")
    p.add_argument(
        "--strict-data-integrity",
        action="store_true",
        help="Fail the run if generated data-integrity proof report contains failing assertions.",
    )
    p.add_argument(
        "--strict-output-correctness",
        action="store_true",
        help="Fail the run if centralized prediction output quality checks detect failures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    IS_SMOKE_TEST = os.environ.get("IS_SMOKE_TEST", "0") == "1"
    if IS_SMOKE_TEST:
        print("⚠️ SMOKE TEST MODE AKTIVIERT - Reduziere Rechenlast!")
        args.n_estimators = 5
        if int(args.forecast_horizon_hours) > 24:
            args.forecast_horizon_hours = 24
        # Cross-script compatibility with smoke-test contract.
        args.n_trials = 2
        args.epochs = 1
    if args.model_type == "tft" and args.enable_da_stacking_afrr:
        raise ValueError("--enable-da-stacking-afrr is only supported for model-type=xgboost.")
    if args.model_type == "linear" and args.enable_da_stacking_afrr:
        raise ValueError("--enable-da-stacking-afrr is only supported for model-type=xgboost.")

    validate_hpo_cli_choice(args.hpo_artifact, args.hpo_artifact_map)

    global_hpo_best_params: dict[str, object] | None = None
    hpo_artifact_map: dict[str, str] = {}
    if args.hpo_artifact:
        global_hpo_best_params = load_hpo_best_params(args.hpo_artifact)
    elif args.hpo_artifact_map:
        hpo_artifact_map = load_hpo_artifact_map(args.hpo_artifact_map)

    run_id = args.run_id.strip() or _run_id_now()
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started_utc = datetime.now(timezone.utc)
    cmd_records: list[dict[str, object]] = []

    da_fragment = run_dir / "da_manifest_fragment.json"
    afrr_fragment_paths: list[Path] = []
    afrr_base_dir = args.base_dir
    hpo_used_by_target: dict[str, str] = {}
    hpo_best_params_by_target: dict[str, dict[str, object]] = {}
    hpo_selection_metric_by_target: dict[str, str] = {}

    if args.enable_da_stacking_afrr:
        stack_cmd = [
            sys.executable,
            "scripts/build_afrr_da_stacking_feature.py",
            "--base-dir",
            args.base_dir,
            "--out-dir",
            args.stacked_base_dir,
            "--n-estimators",
            str(args.n_estimators),
            "--max-depth",
            str(args.max_depth),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            args.device,
            "--cv-n-splits",
            str(args.stack_cv_n_splits),
            "--cv-test-size",
            str(args.stack_cv_test_size),
            "--cv-gap-hours",
            str(args.stack_cv_gap_hours),
        ]
        if args.allow_cpu:
            stack_cmd.append("--allow-cpu")
        cmd_records.append(_run_train_cmd(stack_cmd, run_dir=run_dir, log_stem="00_build_afrr_da_stacking_feature"))
        afrr_base_dir = args.stacked_base_dir

    if args.model_type == "xgboost":
        model_name = args.model_name.strip() or "xgboost_v1"
        base_cmd = [
            sys.executable,
            "-m",
            "src.energy_trading.models.train_xgboost_export",
            "--device",
            args.device,
            "--model-name",
            model_name,
            "--n-estimators",
            str(args.n_estimators),
            "--max-depth",
            str(args.max_depth),
            "--learning-rate",
            str(args.learning_rate),
            "--subsample",
            str(args.subsample),
            "--colsample-bytree",
            str(args.colsample_bytree),
            "--min-child-weight",
            str(args.min_child_weight),
            "--reg-alpha",
            str(args.reg_alpha),
            "--reg-lambda",
            str(args.reg_lambda),
            "--early-stopping-rounds",
            str(args.early_stopping_rounds),
            "--run-dir",
            str(run_dir),
            "--prediction-splits",
            "val,test",
            "--forecast-horizon-hours",
            str(args.forecast_horizon_hours),
            "--lead-weight-start",
            str(args.lead_weight_start),
            "--lead-weight-end",
            str(args.lead_weight_end),
            "--lead-weight-max",
            str(args.lead_weight_max),
            "--seed",
            str(args.seed),
        ]
        if args.allow_cpu:
            base_cmd.append("--allow-cpu")
    elif args.model_type == "tft":
        model_name = args.model_name.strip() or "tft_v1"
        tft_max_encoder_length = int(args.tft_max_encoder_length) if args.tft_max_encoder_length is not None else 168
        base_cmd = [
            sys.executable,
            "-m",
            "src.energy_trading.models.train_tft_export",
            "--device",
            args.device,
            "--model-name",
            model_name,
            "--run-dir",
            str(run_dir),
            "--max-encoder-length",
            str(tft_max_encoder_length),
            "--max-prediction-length",
            str(args.forecast_horizon_hours),
            "--lead-weight-start",
            str(args.lead_weight_start),
            "--lead-weight-end",
            str(args.lead_weight_end),
            "--lead-weight-max",
            str(args.lead_weight_max),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--precision",
            str(args.tft_precision),
        ]
        if args.tft_learning_rate is not None:
            base_cmd += ["--learning-rate", str(args.tft_learning_rate)]
        if args.tft_gradient_clip_val is not None:
            base_cmd += ["--gradient-clip-val", str(args.tft_gradient_clip_val)]
        if args.tft_hidden_size is not None:
            base_cmd += ["--hidden-size", str(args.tft_hidden_size)]
        if args.tft_attention_head_size is not None:
            base_cmd += ["--attention-head-size", str(args.tft_attention_head_size)]
        if args.tft_dropout is not None:
            base_cmd += ["--dropout", str(args.tft_dropout)]
        if args.tft_max_epochs is not None:
            base_cmd += ["--max-epochs", str(args.tft_max_epochs)]
        if args.tft_early_stopping_patience is not None:
            base_cmd += ["--early-stopping-patience", str(args.tft_early_stopping_patience)]
        if args.cleanup_lightning_checkpoints:
            base_cmd.append("--cleanup-lightning-checkpoints")
    else:
        model_name = args.model_name.strip() or "linear_ridge_v1"
        base_cmd = [
            sys.executable,
            "-m",
            "src.energy_trading.models.train_linear_export",
            "--model-name",
            model_name,
            "--run-dir",
            str(run_dir),
            "--alpha",
            str(args.linear_alpha),
            "--l1-ratio",
            str(args.linear_l1_ratio),
            "--learning-rate",
            str(args.linear_learning_rate),
            "--eta0",
            str(args.linear_eta0),
            "--forecast-horizon-hours",
            str(args.forecast_horizon_hours),
            "--lead-weight-start",
            str(args.lead_weight_start),
            "--lead-weight-end",
            str(args.lead_weight_end),
            "--lead-weight-max",
            str(args.lead_weight_max),
            "--seed",
            str(args.seed),
        ]

    # Train DA model unless explicitly skipped.
    if not args.skip_da:
        cmd_da = [*base_cmd, "--bundle", "da", "--manifest-fragment-out", str(da_fragment)]
        cmd_da = [*cmd_da, "--base-dir", args.base_dir]
        if hpo_artifact_map:
            hpo_path = hpo_artifact_map.get(DA_TARGET, "").strip()
            if not hpo_path:
                raise RuntimeError(f"Missing HPO artifact map entry for {DA_TARGET}")
            hpo_payload = load_hpo_artifact_payload(hpo_path)
            da_best_params = hpo_payload.get("best_params", {})
            if not isinstance(da_best_params, dict):
                raise RuntimeError(f"Invalid best_params in HPO artifact: {hpo_path}")
            cmd_da.extend(hpo_override_cli_args(args.model_type, da_best_params))
            cmd_da.extend(
                [
                    "--hpo-artifact-path",
                    str(Path(hpo_path).resolve()),
                    "--hpo-selection-metric",
                    str(hpo_payload.get("selection_metric", "")),
                    "--hpo-best-params-json",
                    json.dumps(da_best_params),
                ]
            )
            hpo_used_by_target[DA_TARGET] = str(Path(hpo_path).resolve())
            hpo_best_params_by_target[DA_TARGET] = {str(k): v for k, v in da_best_params.items()}
            hpo_selection_metric_by_target[DA_TARGET] = str(hpo_payload.get("selection_metric", ""))
        elif global_hpo_best_params is not None:
            cmd_da.extend(hpo_override_cli_args(args.model_type, global_hpo_best_params))
            hpo_used_by_target[DA_TARGET] = str(Path(args.hpo_artifact).resolve())
            hpo_best_params_by_target[DA_TARGET] = {str(k): v for k, v in global_hpo_best_params.items()}
        cmd_records.append(_run_train_cmd(cmd_da, run_dir=run_dir, log_stem=f"01_train_da_{args.model_type}"))

    # Train aFRR models target-wise to produce full canonical prediction columns.
    afrr_targets = filter_afrr_targets(_resolve_available_afrr_targets(afrr_base_dir), args.afrr_targets)
    if not afrr_targets:
        raise RuntimeError(f"No aFRR targets found in bundle config under '{afrr_base_dir}'.")
    skipped = [t for t in AFRR_TARGETS if t not in afrr_targets]
    if skipped:
        print(f"[WARN] Skipping unavailable aFRR targets: {', '.join(skipped)}")
    afrr_jobs: list[tuple[str, Path, list[str], str]] = []
    for tgt in afrr_targets:
        frag = run_dir / f"afrr_{tgt}_manifest_fragment.json"
        afrr_fragment_paths.append(frag)
        cmd_afrr = [
            *base_cmd,
            "--base-dir",
            afrr_base_dir,
            "--bundle",
            "afrr",
            "--target-col",
            tgt,
            "--manifest-fragment-out",
            str(frag),
        ]
        if hpo_artifact_map:
            hpo_path = hpo_artifact_map.get(tgt, "").strip()
            if not hpo_path:
                raise RuntimeError(f"Missing HPO artifact map entry for {tgt}")
            hpo_payload = load_hpo_artifact_payload(hpo_path)
            tgt_best_params = hpo_payload.get("best_params", {})
            if not isinstance(tgt_best_params, dict):
                raise RuntimeError(f"Invalid best_params in HPO artifact: {hpo_path}")
            cmd_afrr.extend(hpo_override_cli_args(args.model_type, tgt_best_params))
            cmd_afrr.extend(
                [
                    "--hpo-artifact-path",
                    str(Path(hpo_path).resolve()),
                    "--hpo-selection-metric",
                    str(hpo_payload.get("selection_metric", "")),
                    "--hpo-best-params-json",
                    json.dumps(tgt_best_params),
                ]
            )
            hpo_used_by_target[tgt] = str(Path(hpo_path).resolve())
            hpo_best_params_by_target[tgt] = {str(k): v for k, v in tgt_best_params.items()}
            hpo_selection_metric_by_target[tgt] = str(hpo_payload.get("selection_metric", ""))
        elif global_hpo_best_params is not None:
            cmd_afrr.extend(hpo_override_cli_args(args.model_type, global_hpo_best_params))
            hpo_used_by_target[tgt] = str(Path(args.hpo_artifact).resolve())
            hpo_best_params_by_target[tgt] = {str(k): v for k, v in global_hpo_best_params.items()}
        afrr_jobs.append((tgt, frag, cmd_afrr, f"02_train_afrr_{args.model_type}_{tgt}"))

    afrr_parallel_jobs = max(1, int(args.afrr_parallel_jobs))
    # Keep conservative default behavior (sequential) unless explicitly requested.
    if afrr_parallel_jobs == 1 or len(afrr_jobs) <= 1:
        for _tgt, _frag, cmd_afrr, log_stem in afrr_jobs:
            cmd_records.append(_run_train_cmd(cmd_afrr, run_dir=run_dir, log_stem=log_stem))
    else:
        print(f"[INFO] Running aFRR target training in parallel with jobs={afrr_parallel_jobs}.")
        with ThreadPoolExecutor(max_workers=afrr_parallel_jobs) as pool:
            fut_to_target = {
                pool.submit(_run_train_cmd, cmd_afrr, run_dir=run_dir, log_stem=log_stem): tgt
                for tgt, _frag, cmd_afrr, log_stem in afrr_jobs
            }
            for fut in as_completed(fut_to_target):
                tgt = fut_to_target[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"aFRR training failed for target '{tgt}': {exc}") from exc
                cmd_records.append(rec)

    da_meta = _load_fragment(da_fragment) if not args.skip_da else None
    afrr_meta_list = [_load_fragment(p) for p in afrr_fragment_paths if p.exists()]
    if not afrr_meta_list:
        raise RuntimeError("No aFRR manifest fragments found.")

    # Lead-time MAE summary logging for both model families.
    def _print_leadtime_summary(metrics_path: str, label: str) -> None:
        def _fmt(x: object) -> str:
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "nan"

        try:
            payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        except Exception:
            return
        # TFT metrics
        if "leadtime_mae_val_h1" in payload:
            h_last = int(payload.get("leadtime_last_h", 48))
            val_last = payload.get(
                f"leadtime_mae_val_h{h_last}",
                payload.get("leadtime_mae_val_h_last", payload.get("leadtime_mae_val_h48")),
            )
            test_last = payload.get(
                f"leadtime_mae_test_h{h_last}",
                payload.get("leadtime_mae_test_h_last", payload.get("leadtime_mae_test_h48")),
            )
            print(
                f"[MAE] {label}: val h1={_fmt(payload.get('leadtime_mae_val_h1'))}, "
                f"val h{h_last}={_fmt(val_last)}, "
                f"test h1={_fmt(payload.get('leadtime_mae_test_h1'))}, "
                f"test h{h_last}={_fmt(test_last)}"
            )
            return
        # XGBoost metrics
        if "per_target_metrics" in payload:
            ptm = payload["per_target_metrics"]
            tgt = payload.get("target_col")
            if tgt in ptm:
                m = ptm[tgt]
                print(
                    f"[MAE] {label}: val h1={m.get('mae', float('nan')):.4f}, "
                    f"val h48={m.get('mae_h48', float('nan')):.4f}"
                )

    if da_meta:
        _print_leadtime_summary(da_meta["metrics_path"], "da")
    for i, meta in enumerate(afrr_meta_list, start=1):
        _print_leadtime_summary(meta["metrics_path"], f"afrr_target_{i}")

    # Merge aFRR split prediction files into one file per split.
    afrr_pred_val = _merge_prediction_files(
        [Path(m["predictions"]["val"]) for m in afrr_meta_list if "val" in m.get("predictions", {})],
        run_dir / "predictions" / "afrr_val.parquet",
    )
    afrr_pred_test = _merge_prediction_files(
        [Path(m["predictions"]["test"]) for m in afrr_meta_list if "test" in m.get("predictions", {})],
        run_dir / "predictions" / "afrr_test.parquet",
    )

    afrr_long_by_split: dict[str, dict[str, str]] = {"val": {}, "test": {}}
    for meta in afrr_meta_list:
        for split, pred_map in meta.get("predictions_long", {}).items():
            for pred_col, path in pred_map.items():
                afrr_long_by_split.setdefault(split, {})
                afrr_long_by_split[split][pred_col] = path

    prediction_quality_report = _prediction_output_quality_report(
        da_meta=da_meta,
        afrr_meta_list=afrr_meta_list,
        afrr_pred_val=afrr_pred_val,
        afrr_pred_test=afrr_pred_test,
    )
    prediction_quality_report_path = run_dir / "prediction_output_quality_report.json"
    prediction_quality_report_path.write_text(json.dumps(prediction_quality_report, indent=2), encoding="utf-8")

    training_context = {
        "run_id": run_id,
        "started_at_utc": run_started_utc.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": float(sum(float(r.get("duration_seconds", 0.0)) for r in cmd_records)),
        "launcher_script": str(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _safe_git_commit(),
        "cli_args": vars(args),
        "resolved": {
            "run_root": str(Path(args.run_root).resolve()),
            "run_dir": str(run_dir.resolve()),
            "base_dir": str(Path(args.base_dir).resolve()),
            "afrr_base_dir": str(Path(afrr_base_dir).resolve()),
            "model_type": args.model_type,
            "model_name": model_name,
            "available_afrr_targets": afrr_targets,
            "skipped_afrr_targets": skipped,
            "target_policy_source": "energy_trading.models.training_policy",
            "hpo_mode": "artifact_map" if hpo_artifact_map else ("single_artifact" if global_hpo_best_params is not None else "none"),
        },
        "input_checksums": {
            "da_base_dir": _bundle_input_checksums(args.base_dir),
            "afrr_base_dir": _bundle_input_checksums(afrr_base_dir),
        },
        "executed_commands": cmd_records,
        "hpo_artifacts_by_target": hpo_used_by_target,
        "hpo_best_params_by_target": hpo_best_params_by_target,
        "hpo_selection_metric_by_target": hpo_selection_metric_by_target,
    }
    training_context["prediction_output_quality_report_path"] = str(prediction_quality_report_path.resolve())
    training_context["prediction_output_quality_passed"] = bool(prediction_quality_report.get("passed", False))
    data_integrity_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "da_base_dir_report": _bundle_data_integrity_report(args.base_dir),
        "afrr_base_dir_report": _bundle_data_integrity_report(afrr_base_dir),
    }
    data_integrity_report["passed"] = bool(
        data_integrity_report["da_base_dir_report"].get("passed", False)
        and data_integrity_report["afrr_base_dir_report"].get("passed", False)
    )
    data_integrity_report_path = run_dir / "data_integrity_report.json"
    data_integrity_report_path.write_text(json.dumps(data_integrity_report, indent=2), encoding="utf-8")
    training_context["data_integrity_report_path"] = str(data_integrity_report_path.resolve())
    training_context["data_integrity_passed"] = bool(data_integrity_report["passed"])
    training_context_path = run_dir / "training_run_context.json"
    training_context_path.write_text(json.dumps(training_context, indent=2), encoding="utf-8")
    if args.strict_data_integrity and not bool(data_integrity_report["passed"]):
        raise RuntimeError(
            "Data integrity report contains failing assertions. "
            f"See: {data_integrity_report_path}"
        )
    if args.strict_output_correctness and not bool(prediction_quality_report.get("passed", False)):
        raise RuntimeError(
            "Prediction output quality report contains failing assertions. "
            f"See: {prediction_quality_report_path}"
        )

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": {
            "context_path": str(training_context_path.resolve()),
            "data_integrity_report_path": str(data_integrity_report_path.resolve()),
            "prediction_output_quality_report_path": str(prediction_quality_report_path.resolve()),
            "model_type": args.model_type,
            "model_name": model_name,
            "git_commit": training_context.get("git_commit"),
            "hpo_mode": training_context["resolved"]["hpo_mode"],
            "hpo_artifacts_by_target": hpo_used_by_target,
            "hpo_best_params_by_target": hpo_best_params_by_target,
            "hpo_selection_metric_by_target": hpo_selection_metric_by_target,
        },
        "bundles": {
            "afrr": {
                "model_paths": [m["model_path"] for m in afrr_meta_list],
                "metrics_paths": [m["metrics_path"] for m in afrr_meta_list],
                "predictions": {
                    "val": str(afrr_pred_val.resolve()),
                    "test": str(afrr_pred_test.resolve()),
                },
                "predictions_long": afrr_long_by_split,
                "prediction_columns": [
                    "pred_afrr_capacity_price_pos",
                    "pred_afrr_capacity_price_neg",
                    "pred_afrr_activation_price_pos",
                    "pred_afrr_activation_price_neg",
                    "pred_afrr_activation_rate_pos",
                    "pred_afrr_activation_rate_neg",
                ],
                "target_columns": afrr_targets,
            },
        },
        "ground_truth": {
            "default_path": str(Path(args.ground_truth_path).resolve()),
        },
        "simulation": {
            "default_split": "test",
            "canonical_economic_targets": ["pred_afrr_activation_price_neg"],
            "transformed_targets": ["pred_afrr_activation_price_neg"],
        },
        "target_value_mode": {
            "pred_afrr_activation_price_neg": "canonical_economic",
        },
    }
    if da_meta:
        manifest["bundles"]["da"] = {
            "model_path": da_meta["model_path"],
            "metrics_path": da_meta["metrics_path"],
            "predictions": da_meta["predictions"],
            "predictions_long": da_meta.get("predictions_long", {}),
            "prediction_columns": da_meta["prediction_columns"],
            "target_columns": da_meta["target_columns"],
        }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_latest_pointer:
        now_iso = datetime.now(timezone.utc).isoformat()
        latest_model_path = Path(args.run_root) / f"latest_{args.model_type}.json"
        try:
            # Prefer a portable relative pointer for multi-machine workflows.
            manifest_pointer = str(manifest_path.resolve().relative_to(latest_model_path.parent.resolve()))
        except ValueError:
            # Fallback if paths are on different roots.
            manifest_pointer = str(manifest_path.resolve())
        latest_payload = {
            "run_id": run_id,
            "manifest_path": manifest_pointer,
            "manifest_path_abs": str(manifest_path.resolve()),
            "updated_at_utc": now_iso,
            "model_type": args.model_type,
        }
        latest_model_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    # Normalize generated JSON artifacts to portable relative paths for
    # cross-machine reproducibility.
    repo_root = Path(__file__).resolve().parents[1]
    for p in run_dir.rglob("*.json"):
        _normalize_json_paths_inplace(p, repo_root=repo_root)
    if not args.skip_latest_pointer:
        _normalize_json_paths_inplace(latest_model_path, repo_root=repo_root)

    print("[OK] Run export complete.")
    print(f"- run_id: {run_id}")
    print(f"- run_dir: {run_dir}")
    print(f"- training_context: {training_context_path}")
    print(f"- manifest: {manifest_path}")
    if not args.skip_latest_pointer:
        print(f"- latest_model: {Path(args.run_root) / f'latest_{args.model_type}.json'}")


if __name__ == "__main__":
    main()
