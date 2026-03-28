"""Build reusable ML data bundles for DA and aFRR model tracks.

Design goals:
- strict chronological split (no shuffle),
- explicit leakage prevention via target/metadata exclusion from X,
- reusable feature/target config for future model scripts,
- train-only scaler fitting for causal preprocessing.

Usage:
    ./.venv/bin/python -m src.energy_trading.models.prepare_ml_bundles \
        --input data/features/all_data_features.parquet \
        --output-dir data/model_input \
        --doc-path docs/features_documentation.md \
        --scaler-out models/preprocessing/scaler.joblib
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BundleName = Literal["da", "afrr"]
PICASSO_START_UTC = "2022-06-22 22:00:00+00:00"


@dataclass(frozen=True)
class SplitBounds:
    train_end_exclusive: pd.Timestamp
    val_end_exclusive: pd.Timestamp
    test_end_inclusive: pd.Timestamp


class MLDataFactory:
    """Factory for creating scalable train/val/test bundles from feature artifact.

    Leakage guard:
    - all selected target columns and known excluded columns are removed from X
      before any scaler fit.
    - scaler pipeline is fit on train split only.
    """

    DA_TARGET_CANDIDATES = ["da_price", "target_da_price_h1"]

    AFRR_TARGET_SPECS = {
        "afrr_capacity_price_pos": ["afrr_capacity_price_pos"],
        "afrr_capacity_price_neg": ["afrr_capacity_price_neg"],
        "afrr_activation_price_vwap_pos": [
            "afrr_activation_price_vwap_pos",
            "target_afrr_activation_price_vwap_pos_h1",
        ],
        "afrr_activation_price_vwap_neg": [
            "afrr_activation_price_vwap_neg",
            "target_afrr_activation_price_vwap_neg_h1",
        ],
        "afrr_activation_rate": ["afrr_activation_rate", "target_afrr_rate_h1"],
    }

    HARD_META_EXCLUDE = {"timestamp_utc"}

    def __init__(
        self,
        input_path: str | Path = "data/features/all_data_features.parquet",
        output_dir: str | Path = "data/model_input",
        doc_path: str | Path = "docs/features_documentation.md",
        scaler_path: str | Path = "models/preprocessing/scaler.joblib",
        null_report_path: str | Path = "data/reports/null_report_features.csv",
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.doc_path = Path(doc_path)
        self.scaler_path = Path(scaler_path)
        self.null_report_path = Path(null_report_path)
        self.config_path = self.output_dir / "feature_config.json"
        # Law #1: strict causality purge gap between adjacent splits for 1h rows.
        self.gap_rows = 72
        # Small-gap repair for sparse API holes on X only.
        self.feature_ffill_limit = 12

    @staticmethod
    def _default_bounds() -> SplitBounds:
        return SplitBounds(
            train_end_exclusive=pd.Timestamp("2024-01-01T00:00:00Z"),
            val_end_exclusive=pd.Timestamp("2024-07-01T00:00:00Z"),
            test_end_inclusive=pd.Timestamp(datetime.now(timezone.utc)),
        )

    @staticmethod
    def _extract_backticked_tokens(line: str) -> list[str]:
        out: list[str] = []
        cur = ""
        open_tick = False
        for ch in line:
            if ch == "`":
                if open_tick:
                    if cur.strip():
                        out.append(cur.strip())
                    cur = ""
                open_tick = not open_tick
                continue
            if open_tick:
                cur += ch
        return out

    def _excluded_from_docs(self) -> set[str]:
        """Parse docs row(s) marked as excluded from X."""
        if not self.doc_path.exists():
            return set()
        lines = self.doc_path.read_text(encoding="utf-8").splitlines()
        excluded: set[str] = set()
        for line in lines:
            if "aus `X` ausgeschlossen" not in line:
                continue
            for token in self._extract_backticked_tokens(line):
                # Drop non-column tokens like "X".
                if token != "X":
                    excluded.add(token)
        return excluded

    def _load_df(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input artifact not found: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        if "timestamp_utc" not in df.columns:
            raise KeyError("Required column `timestamp_utc` is missing from artifact")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
        return df

    @staticmethod
    def _apply_post_picasso_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Apply in-memory regime filter while keeping raw artifact unchanged."""
        cut = pd.Timestamp(PICASSO_START_UTC, tz="UTC")
        rows_before = len(df)
        # Remove pre-PICASSO rows due to structural market changes
        # (e.g., pay-as-bid vs pay-as-cleared activation-price dynamics).
        out = df.loc[df["timestamp_utc"] >= cut].copy()
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        dropped = rows_before - len(out)
        return out, dropped

    def _resolve_da_targets(self, df: pd.DataFrame) -> list[str]:
        targets = [c for c in self.DA_TARGET_CANDIDATES if c in df.columns]
        if not targets:
            raise KeyError(f"No DA target found. Tried: {self.DA_TARGET_CANDIDATES}")
        return [targets[0]]

    def _resolve_afrr_targets(self, df: pd.DataFrame) -> list[str]:
        resolved: list[str] = []
        for _semantic_name, candidates in self.AFRR_TARGET_SPECS.items():
            chosen = next((c for c in candidates if c in df.columns), None)
            if chosen is not None:
                resolved.append(chosen)
        if not resolved:
            raise KeyError("No aFRR targets found in artifact for configured target set")
        # Primary aFRR target must be activation VWAP POS if available.
        primary_candidates = self.AFRR_TARGET_SPECS["afrr_activation_price_vwap_pos"]
        primary = next((c for c in primary_candidates if c in resolved), None)
        uniq = []
        seen = set()
        for c in resolved:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        if primary is None:
            return uniq
        ordered = [primary] + [c for c in uniq if c != primary]
        return ordered

    @staticmethod
    def _canonical_primary_target(bundle: BundleName) -> str:
        if bundle == "da":
            return "da_price"
        if bundle == "afrr":
            return "afrr_activation_price_vwap_pos"
        raise ValueError(f"Unsupported bundle: {bundle}")

    def _bundle_targets(self, df: pd.DataFrame, bundle: BundleName) -> list[str]:
        if bundle == "da":
            return self._resolve_da_targets(df)
        if bundle == "afrr":
            return self._resolve_afrr_targets(df)
        raise ValueError(f"Unsupported bundle: {bundle}")

    def _feature_columns(self, df: pd.DataFrame, targets: list[str]) -> list[str]:
        # Law 2: explicit target drop before model fitting/preprocessing.
        excluded = set(targets) | self.HARD_META_EXCLUDE | self._excluded_from_docs()
        X = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    @staticmethod
    def _drop_target_nans(part: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
        """Remove rows where any target value is missing."""
        existing = [c for c in target_cols if c in part.columns]
        if not existing:
            return part
        return part.dropna(subset=existing).copy()

    def _impute_sparse_feature_gaps(self, part: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Apply conservative ffill on X only (no backward fill, no target fill)."""
        existing = [c for c in feature_cols if c in part.columns]
        if not existing:
            return part
        part = part.sort_values("timestamp_utc").copy()
        part.loc[:, existing] = part.loc[:, existing].ffill(limit=self.feature_ffill_limit)
        return part

    @staticmethod
    def _reason_category(col: str) -> str:
        c = col.lower()
        if c.startswith("target_"):
            return "target_shift_boundary"
        if "co2_price" in c:
            return "source_history_gap"
        if "_lag_" in c:
            return "lag_warmup_or_sparse_source"
        if "forecast" in c:
            return "forecast_source_gap"
        return "other_or_unknown"

    def _write_null_report(self, df: pd.DataFrame, out_path: Path) -> None:
        if "timestamp_utc" in df.columns:
            ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        else:
            ts = pd.Series(pd.NaT, index=df.index)

        rows: list[dict] = []
        for col in df.columns:
            mask = df[col].isna()
            n = int(mask.sum())
            if n == 0:
                continue
            idx = mask[mask].index
            rows.append(
                {
                    "col": col,
                    "null_count": n,
                    "first_null_ts": ts.loc[idx[0]] if len(idx) else pd.NaT,
                    "last_null_ts": ts.loc[idx[-1]] if len(idx) else pd.NaT,
                    "reason_category": self._reason_category(col),
                }
            )
        rep = pd.DataFrame(rows).sort_values(["null_count", "col"], ascending=[False, True]) if rows else pd.DataFrame(
            columns=["col", "null_count", "first_null_ts", "last_null_ts", "reason_category"]
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out_path, index=False)

    def _split_masks(self, df: pd.DataFrame, bounds: SplitBounds) -> dict[str, pd.Series]:
        """Create chronological split masks with strict 72-row purge gaps.

        Purge logic:
        - Between train and val, drop first `gap_rows` rows of validation period.
        - Between val and test, drop first `gap_rows` rows of test period.
        """
        ts = df["timestamp_utc"]
        idx = pd.RangeIndex(len(df))

        train_idx = idx[ts < bounds.train_end_exclusive]
        val_idx_raw = idx[(ts >= bounds.train_end_exclusive) & (ts < bounds.val_end_exclusive)]
        test_idx_raw = idx[(ts >= bounds.val_end_exclusive) & (ts <= bounds.test_end_inclusive)]

        val_idx = val_idx_raw[self.gap_rows :] if len(val_idx_raw) > self.gap_rows else val_idx_raw[:0]
        test_idx = test_idx_raw[self.gap_rows :] if len(test_idx_raw) > self.gap_rows else test_idx_raw[:0]

        return {
            "train": idx.isin(train_idx),
            "val": idx.isin(val_idx),
            "test": idx.isin(test_idx),
        }

    def _fit_scaler_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X_train)
        return pipe

    def build(self) -> dict:
        df = self._load_df()
        # Lawful regime selection for training bundles (in-memory only).
        # We do not modify the source artifact on disk.
        df, dropped_picasso = self._apply_post_picasso_filter(df)
        print(
            "Regime filter applied: Training on Post-PICASSO data only "
            f"(Starting {PICASSO_START_UTC}). Rows available: {len(df)}"
        )
        print(f"Rows dropped due to Picasso-Cut: {dropped_picasso}")
        bounds = self._default_bounds()
        masks = self._split_masks(df, bounds)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

        config: dict = {
            "artifact_path": str(self.input_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "splits": {
                "train_end_exclusive": bounds.train_end_exclusive.isoformat(),
                "val_end_exclusive": bounds.val_end_exclusive.isoformat(),
                "test_end_inclusive": bounds.test_end_inclusive.isoformat(),
                "purge_gap_rows": self.gap_rows,
            },
            "bundles": {},
        }

        scaler_store: dict[str, Pipeline] = {}
        remaining_nans_x_total = 0

        for bundle in ("da", "afrr"):
            targets = self._bundle_targets(df, bundle=bundle)
            primary_canonical = self._canonical_primary_target(bundle)
            primary_source = targets[0]
            features = self._feature_columns(df, targets=targets)

            bundle_dir = self.output_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)

            for split_name, mask in masks.items():
                part = df.loc[mask, ["timestamp_utc", *features, *targets]].copy()
                # Mandatory target cleaning: drop boundary rows with missing y.
                part = self._drop_target_nans(part, targets)
                # Sparse-gap repair on X only; keep remaining NaNs (model can handle).
                part = self._impute_sparse_feature_gaps(part, features)
                remaining_nans_x_total += int(part[features].isna().sum().sum())
                if primary_source != primary_canonical:
                    part = part.rename(columns={primary_source: primary_canonical})
                part.to_parquet(bundle_dir / f"{split_name}.parquet", index=False)

            targets_out = [primary_canonical] + [t for t in targets[1:] if t != primary_canonical]

            train_df = df.loc[masks["train"], features]
            scaler_store[bundle] = self._fit_scaler_pipeline(train_df)

            config["bundles"][bundle] = {
                "primary_target": primary_canonical,
                "target_source_columns": targets,
                "features": features,
                "targets": targets_out,
                "n_features": len(features),
                "n_targets": len(targets_out),
                "files": {
                    "train": str(bundle_dir / "train.parquet"),
                    "val": str(bundle_dir / "val.parquet"),
                    "test": str(bundle_dir / "test.parquet"),
                },
            }

        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        joblib.dump(scaler_store, self.scaler_path)
        # Thesis transparency report after post-PICASSO filtering and target cleaning.
        report_frames: list[pd.DataFrame] = []
        for split in ("train", "val", "test"):
            p = self.output_dir / "afrr" / f"{split}.parquet"
            if p.exists():
                report_frames.append(pd.read_parquet(p))
        if report_frames:
            report_df = pd.concat(report_frames, ignore_index=True, sort=False)
            self._write_null_report(report_df, self.null_report_path)
        print(f"Remaining NaNs in X : {remaining_nans_x_total}")
        return config


def load_processed_data(
    bundle: BundleName,
    split: Literal["train", "val", "test"] = "train",
    base_dir: str | Path = "data/model_input",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared X/y matrices for a given bundle and split."""
    base = Path(base_dir)
    cfg = json.loads((base / "feature_config.json").read_text(encoding="utf-8"))
    if bundle not in cfg["bundles"]:
        raise KeyError(f"Unknown bundle '{bundle}'. Available: {list(cfg['bundles'].keys())}")
    bcfg = cfg["bundles"][bundle]
    file_path = Path(bcfg["files"][split])
    df = pd.read_parquet(file_path)
    X = df[bcfg["features"]].copy()
    y = df[bcfg["targets"]].copy()
    return X, y


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build DA/aFRR ML bundles from final feature artifact.")
    p.add_argument("--input", default="data/features/all_data_features.parquet", help="Input feature artifact path.")
    p.add_argument("--output-dir", default="data/model_input", help="Output directory for split bundles.")
    p.add_argument("--doc-path", default="docs/features_documentation.md", help="Path to features documentation.")
    p.add_argument(
        "--scaler-out",
        default="models/preprocessing/scaler.joblib",
        help="Output path for train-fitted scaler pipeline(s).",
    )
    p.add_argument(
        "--null-report-out",
        default="data/reports/null_report_features.csv",
        help="Output CSV for post-cleaning null report.",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    factory = MLDataFactory(
        input_path=args.input,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
        scaler_path=args.scaler_out,
        null_report_path=args.null_report_out,
    )
    cfg = factory.build()
    print("[OK] ML bundles erstellt.")
    for bundle, bcfg in cfg["bundles"].items():
        print(f"- {bundle}: n_features={bcfg['n_features']} n_targets={bcfg['n_targets']}")
    print(f"- config: {factory.config_path}")
    print(f"- scaler: {factory.scaler_path}")


if __name__ == "__main__":
    main()
