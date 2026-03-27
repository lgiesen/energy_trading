"""Build reusable ML data bundles for DA and aFRR model tracks.

Design goals:
- strict chronological split (no shuffle),
- explicit leakage prevention via target/metadata exclusion from X,
- reusable feature/target config for future model scripts,
- train-only scaler fitting for causal preprocessing.
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

    DA_TARGET_CANDIDATES = ["target_da_price_h1", "da_price"]

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
        output_dir: str | Path = "data/processed_ml",
        doc_path: str | Path = "docs/features_documentation.md",
        scaler_path: str | Path = "models/preprocessing/scaler.joblib",
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.doc_path = Path(doc_path)
        self.scaler_path = Path(scaler_path)
        self.config_path = self.output_dir / "feature_config.json"

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
        return sorted(set(resolved))

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
    def _split_masks(df: pd.DataFrame, bounds: SplitBounds) -> dict[str, pd.Series]:
        ts = df["timestamp_utc"]
        return {
            "train": ts < bounds.train_end_exclusive,
            "val": (ts >= bounds.train_end_exclusive) & (ts < bounds.val_end_exclusive),
            "test": (ts >= bounds.val_end_exclusive) & (ts <= bounds.test_end_inclusive),
        }

    def _fit_scaler_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X_train)
        return pipe

    def build(self) -> dict:
        df = self._load_df()
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
            },
            "bundles": {},
        }

        scaler_store: dict[str, Pipeline] = {}

        for bundle in ("da", "afrr"):
            targets = self._bundle_targets(df, bundle=bundle)
            features = self._feature_columns(df, targets=targets)

            bundle_dir = self.output_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)

            for split_name, mask in masks.items():
                part = df.loc[mask, ["timestamp_utc", *features, *targets]].copy()
                part.to_parquet(bundle_dir / f"{split_name}.parquet", index=False)

            train_df = df.loc[masks["train"], features]
            scaler_store[bundle] = self._fit_scaler_pipeline(train_df)

            config["bundles"][bundle] = {
                "features": features,
                "targets": targets,
                "n_features": len(features),
                "n_targets": len(targets),
                "files": {
                    "train": str(bundle_dir / "train.parquet"),
                    "val": str(bundle_dir / "val.parquet"),
                    "test": str(bundle_dir / "test.parquet"),
                },
            }

        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        joblib.dump(scaler_store, self.scaler_path)
        return config


def load_processed_data(
    bundle: BundleName,
    split: Literal["train", "val", "test"] = "train",
    base_dir: str | Path = "data/processed_ml",
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
    p.add_argument("--output-dir", default="data/processed_ml", help="Output directory for split bundles.")
    p.add_argument("--doc-path", default="docs/features_documentation.md", help="Path to features documentation.")
    p.add_argument(
        "--scaler-out",
        default="models/preprocessing/scaler.joblib",
        help="Output path for train-fitted scaler pipeline(s).",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    factory = MLDataFactory(
        input_path=args.input,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
        scaler_path=args.scaler_out,
    )
    cfg = factory.build()
    print("[OK] ML bundles erstellt.")
    for bundle, bcfg in cfg["bundles"].items():
        print(f"- {bundle}: n_features={bcfg['n_features']} n_targets={bcfg['n_targets']}")
    print(f"- config: {factory.config_path}")
    print(f"- scaler: {factory.scaler_path}")


if __name__ == "__main__":
    main()
