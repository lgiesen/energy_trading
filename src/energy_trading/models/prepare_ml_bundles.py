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
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_trading.constants import PICASSO_RELEASE_UTC


BundleName = Literal["da", "afrr"]
LOGGER = logging.getLogger(__name__)


def get_da_optimized_features(df_165: pd.DataFrame) -> pd.DataFrame:
    """Reduce a full feature matrix to DA-auction-causal features only.

    D-1 auction causality constraint:
    At DA gate closure, short-horizon balancing and activation signals are not
    yet physically available for delivery day D and are therefore removed.
    """
    df_out = df_165.copy()
    cols = df_out.columns.tolist()

    keyword_drop_pattern = re.compile(
        r"^(afrr_|mfrr_|nrv_|rz_saldo_|picasso_|mari_|is_activated_|"
        r"system_stress_|grid_stress_|scarcity_|nrv_zscore_|nrv_quantile_)"
    )
    short_lag_pattern = re.compile(r"_lag_(1|2|3|6|12)h$")
    da_spread_lag_pattern = re.compile(r"^da_spread_[a-z0-9_]+_lag_(\d+)h$")

    cols_to_drop: list[str] = []
    for col in cols:
        if col == "da_price_pit":
            # Explicit keep-exception: latest DA value available at gate closure.
            continue
        # Intraday forecast snapshots are not available at DA D-1 gate closure.
        # Keep only causal day-ahead forecast variants.
        if "_forecast_id_" in col:
            cols_to_drop.append(col)
            continue
        if keyword_drop_pattern.search(col):
            cols_to_drop.append(col)
            continue
        if short_lag_pattern.search(col):
            cols_to_drop.append(col)
            continue
        if col.startswith("total_wind_solar_id_error"):
            cols_to_drop.append(col)
            continue
        # For DA model, bilateral spread features are only valid as day-seasonal
        # memory (>=24h lag). Same-hour spreads are not available at D-1 auction.
        if col.startswith("da_spread_") and "_lag_" not in col:
            cols_to_drop.append(col)
            continue
        m_spread = da_spread_lag_pattern.match(col)
        if m_spread and int(m_spread.group(1)) < 24:
            cols_to_drop.append(col)
            continue

    cols_to_drop = sorted(set(cols_to_drop))
    df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    LOGGER.info(
        "[da-opt] removed %s columns (from %s to %s) under D-1 auction causality constraint.",
        len(cols_to_drop),
        len(cols),
        df_out.shape[1],
    )
    if cols_to_drop:
        LOGGER.info("[da-opt] sample dropped cols: %s", ", ".join(cols_to_drop[:15]))
    return df_out


def get_afrr_optimized_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """Reduce aFRR feature set by dropping stale long-term forecast lags.

    Rule:
    - Drop columns that contain `_forecast_` and end with `_lag_48h` or `_lag_168h`.
    - Keep `_lag_24h` forecast terms and all long lags for realized/market signals.
    """
    df_out = df_full.copy()
    cols = df_out.columns.tolist()
    long_forecast_lag_pattern = re.compile(r".*forecast.*_lag_(48|168)h$")

    cols_to_drop = [c for c in cols if long_forecast_lag_pattern.match(c)]
    cols_to_drop = sorted(set(cols_to_drop))
    df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    LOGGER.info(
        "[afrr-opt] removed %s long-term forecast lag columns (from %s to %s).",
        len(cols_to_drop),
        len(cols),
        df_out.shape[1],
    )
    if cols_to_drop:
        LOGGER.info("[afrr-opt] sample dropped cols: %s", ", ".join(cols_to_drop[:15]))
    return df_out


@dataclass(frozen=True)
class SplitBounds:
    train_end_exclusive: pd.Timestamp
    val_end_exclusive: pd.Timestamp
    test_end_inclusive: pd.Timestamp


@dataclass
class ForecastFamilyPCA:
    family: str
    columns: list[str]
    scaler: StandardScaler
    pca: PCA
    pc_names: list[str]
    explained_variance_ratio: list[float]
    max_abs_corr: float
    max_vif: float


class MLDataFactory:
    """Factory for creating scalable train/val/test bundles from feature artifact.

    Leakage guard:
    - all selected target columns and known excluded columns are removed from X
      before any scaler fit.
    - scaler pipeline is fit on train split only.
    """

    # Strict forecast labels for training (h+1 only).
    DA_TRAIN_TARGET = "target_da_price"
    AFRR_PRIMARY_TRAIN_TARGET = "target_afrr_activation_price_vwap_pos"
    AFRR_OPTIONAL_TRAIN_TARGETS = [
        "target_afrr_activation_price_vwap_neg",
        "target_afrr_activation_rate_pos",
        "target_afrr_activation_rate_neg",
        "target_afrr_capacity_price_pos",
        "target_afrr_capacity_price_neg",
    ]

    # Optional unshifted audit labels (y_true), never used as training targets.
    DA_AUDIT_CANDIDATES = ["da_price"]
    AFRR_AUDIT_CANDIDATES = [
        "afrr_activation_price_vwap_pos",
        "afrr_activation_price_vwap_neg",
        "afrr_activation_rate_pos",
        "afrr_activation_rate_neg",
    ]

    HARD_META_EXCLUDE = {
        "timestamp_utc",
        # Analysis-only regime marker; excluded from model training features.
        "is_picasso_active",
    }
    FORECAST_FAMILY_PATTERNS: dict[str, str] = {
        "load_forecast": r"^load_forecast",
        "residual_load_forecast": r"^residual_load_forecast",
        "wind_onshore_forecast": r"^wind_onshore_forecast",
        "wind_offshore_forecast": r"^wind_offshore_forecast",
        "solar_forecast": r"^solar_forecast",
    }

    def __init__(
        self,
        input_path: str | Path = "data/features/all_data_features.parquet",
        output_dir: str | Path = "data/model_input",
        doc_path: str | Path = "docs/features_documentation.md",
        scaler_path: str | Path = "models/preprocessing/scaler.joblib",
        null_report_path: str | Path = "data/reports/null_report_features.csv",
        quality_report_all_path: str | Path = "data/reports/feature_quality_report_all.csv",
        use_forecast_pca: bool = False,
        forecast_pca_var_threshold: float = 0.95,
        forecast_corr_threshold: float = 0.9,
        forecast_vif_threshold: float = 10.0,
        forecast_pca_drop_raw: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.doc_path = Path(doc_path)
        self.scaler_path = Path(scaler_path)
        self.null_report_path = Path(null_report_path)
        self.quality_report_all_path = Path(quality_report_all_path)
        self.config_path = self.output_dir / "feature_config.json"
        # Law #1: strict causality purge gap between adjacent splits for 1h rows.
        self.gap_rows = 72
        # Small-gap repair for sparse API holes on X only.
        self.feature_ffill_limit = 12
        self.nan_warn_threshold_pct = 20.0
        self.use_forecast_pca = bool(use_forecast_pca)
        self.forecast_pca_var_threshold = float(forecast_pca_var_threshold)
        self.forecast_corr_threshold = float(forecast_corr_threshold)
        self.forecast_vif_threshold = float(forecast_vif_threshold)
        self.forecast_pca_drop_raw = bool(forecast_pca_drop_raw)

    @staticmethod
    def _default_bounds() -> SplitBounds:
        return SplitBounds(
            train_end_exclusive=pd.Timestamp("2024-07-01T00:00:00Z"),
            val_end_exclusive=pd.Timestamp("2025-01-01T00:00:00Z"),
            test_end_inclusive=pd.Timestamp("2026-03-01T01:00:00Z"),
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
        cut = pd.Timestamp(PICASSO_RELEASE_UTC, tz="UTC")
        rows_before = len(df)
        # Remove pre-PICASSO rows due to structural market changes
        # (e.g., pay-as-bid vs pay-as-cleared activation-price dynamics).
        out = df.loc[df["timestamp_utc"] >= cut].copy()
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        dropped = rows_before - len(out)
        return out, dropped

    def _resolve_da_targets(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Return (train_targets, audit_targets) for DA bundle."""
        if self.DA_TRAIN_TARGET not in df.columns:
            raise KeyError(
                "Strict target validation failed for DA bundle: "
                f"required `{self.DA_TRAIN_TARGET}` is missing."
            )
        audit = [c for c in self.DA_AUDIT_CANDIDATES if c in df.columns]
        return [self.DA_TRAIN_TARGET], audit

    def _resolve_afrr_targets(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Return (train_targets, audit_targets) for aFRR bundle."""
        if self.AFRR_PRIMARY_TRAIN_TARGET not in df.columns:
            raise KeyError(
                "Strict target validation failed for aFRR bundle: "
                f"required `{self.AFRR_PRIMARY_TRAIN_TARGET}` is missing."
            )
        train_targets = [self.AFRR_PRIMARY_TRAIN_TARGET]
        train_targets.extend([c for c in self.AFRR_OPTIONAL_TRAIN_TARGETS if c in df.columns])
        audit = [c for c in self.AFRR_AUDIT_CANDIDATES if c in df.columns]
        return train_targets, audit

    @staticmethod
    def _canonical_primary_target(bundle: BundleName) -> str:
        if bundle == "da":
            return "target_da_price"
        if bundle == "afrr":
            return "target_afrr_activation_price_vwap_pos"
        raise ValueError(f"Unsupported bundle: {bundle}")

    def _bundle_targets(self, df: pd.DataFrame, bundle: BundleName) -> tuple[list[str], list[str]]:
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

    def _feature_quality_from_train(
        self,
        train_part: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], pd.DataFrame]:
        """Build feature-quality report on train split and decide keep/drop actions."""
        existing = [c for c in feature_cols if c in train_part.columns]
        if not existing:
            return [], pd.DataFrame(columns=["feature", "train_nan_pct", "action"])

        nan_pct = train_part[existing].isna().mean().mul(100.0)
        rows: list[dict[str, object]] = []
        kept: list[str] = []
        for c in existing:
            pct = float(nan_pct[c])
            if pct >= 100.0:
                action = "drop_all_nan_train"
            elif pct > self.nan_warn_threshold_pct:
                action = "warn_keep_high_nan"
                kept.append(c)
            else:
                action = "keep"
                kept.append(c)
            rows.append({"feature": c, "train_nan_pct": round(pct, 6), "action": action})
        report = (
            pd.DataFrame(rows)
            .sort_values(["train_nan_pct", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        return kept, report

    @staticmethod
    def _print_feature_quality_report(bundle: BundleName, report: pd.DataFrame) -> None:
        """Print compact feature quality report."""
        if report.empty:
            print(f"[quality][{bundle}] No features available for quality report.")
            return
        print(f"\n[quality][{bundle}] feature | train_nan_pct | action")
        print("| feature | train_nan_pct | action |")
        print("|---|---:|---|")
        for _, row in report.iterrows():
            print(f"| {row['feature']} | {row['train_nan_pct']:.4f} | {row['action']} |")

    def _impute_features_with_train_fit(
        self,
        part: pd.DataFrame,
        feature_cols: list[str],
        train_medians: pd.Series,
        *,
        bundle: BundleName,
        split_name: str,
    ) -> pd.DataFrame:
        """Impute X using split-local ffill + train-fitted median fallback."""
        existing = [c for c in feature_cols if c in part.columns]
        if not existing:
            return part
        part = part.sort_values("timestamp_utc").copy()

        before = part[existing].isna().sum()
        # Structural capacity series change slowly and can be carried forward across
        # longer sparse source gaps without violating causality.
        structural_cols = [c for c in existing if "capacity" in c.lower()]
        regular_cols = [c for c in existing if c not in structural_cols]

        if structural_cols:
            part.loc[:, structural_cols] = part.loc[:, structural_cols].ffill()
        if regular_cols:
            part.loc[:, regular_cols] = part.loc[:, regular_cols].ffill(limit=self.feature_ffill_limit)
        after_ffill = part[existing].isna().sum()

        median_cols = [c for c in existing if c in train_medians.index and pd.notna(train_medians[c])]
        if median_cols:
            part.loc[:, median_cols] = part.loc[:, median_cols].fillna(train_medians[median_cols])
        after_final = part[existing].isna().sum()

        for c in existing:
            ffilled = int(before[c] - after_ffill[c])
            medianed = int(after_ffill[c] - after_final[c])
            remaining = int(after_final[c])
            if ffilled > 0 or medianed > 0:
                median_value = train_medians.get(c, np.nan)
                LOGGER.info(
                    "[impute][%s][%s] %s: ffill_count=%s median_imputed_count=%s "
                    "train_median_value=%s remaining=%s",
                    bundle,
                    split_name,
                    c,
                    ffilled,
                    medianed,
                    median_value,
                    remaining,
                )
                # Surface potentially weak columns where constant fallback dominates.
                if len(part) > 0:
                    median_ratio = medianed / len(part)
                    if median_ratio >= 0.05:
                        LOGGER.warning(
                            "[impute][%s][%s] %s: high median fallback share %.2f%%",
                            bundle,
                            split_name,
                            c,
                            median_ratio * 100.0,
                        )
        return part

    @staticmethod
    def _max_abs_corr(x: pd.DataFrame) -> float:
        if x.shape[1] < 2:
            return 0.0
        corr = x.corr().abs()
        np.fill_diagonal(corr.values, np.nan)
        v = np.nanmax(corr.values)
        return float(v) if np.isfinite(v) else 0.0

    @staticmethod
    def _compute_vif_max(x: pd.DataFrame) -> float:
        """Compute maximum VIF in a block using least squares (no statsmodels)."""
        if x.shape[1] < 2:
            return 1.0
        arr = np.asarray(x, dtype=float)
        max_vif = 1.0
        for i in range(arr.shape[1]):
            y = arr[:, i]
            others = np.delete(arr, i, axis=1)
            if others.shape[1] == 0:
                continue
            # Add intercept.
            X = np.column_stack([np.ones(len(y)), others])
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                y_hat = X @ coef
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                if ss_tot <= 0:
                    continue
                r2 = 1.0 - (ss_res / ss_tot)
                r2 = min(max(r2, 0.0), 0.999999)
                vif = 1.0 / (1.0 - r2)
                if np.isfinite(vif):
                    max_vif = max(max_vif, float(vif))
            except Exception:
                continue
        return float(max_vif)

    def _forecast_family_columns(self, feature_cols: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for fam, pattern in self.FORECAST_FAMILY_PATTERNS.items():
            rx = re.compile(pattern)
            cols = [c for c in feature_cols if rx.match(c)]
            # Restrict to family signals and their lagged variants.
            cols = [c for c in cols if "_lag_" in c or "forecast" in c]
            if cols:
                out[fam] = cols
        return out

    def _fit_forecast_family_pca(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> list[ForecastFamilyPCA]:
        """Fit PCA modules on forecast families using train data only."""
        families = self._forecast_family_columns(feature_cols)
        models: list[ForecastFamilyPCA] = []
        for fam, cols in families.items():
            if len(cols) < 2:
                continue
            block = train_df[cols].copy()
            # Train-fitted fallback for PCA fit robustness.
            med = block.median(numeric_only=True)
            block = block.fillna(med).fillna(0.0)

            max_corr = self._max_abs_corr(block)
            max_vif = self._compute_vif_max(block)
            if max_corr < self.forecast_corr_threshold and max_vif < self.forecast_vif_threshold:
                LOGGER.info(
                    "[forecast-pca][skip] %s: low redundancy (max_corr=%.3f, max_vif=%.3f)",
                    fam,
                    max_corr,
                    max_vif,
                )
                continue

            scaler = StandardScaler()
            scaled = scaler.fit_transform(block)
            pca_full = PCA().fit(scaled)
            cum = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cum, self.forecast_pca_var_threshold) + 1)
            n_comp = max(1, min(n_comp, len(cols)))
            pca = PCA(n_components=n_comp).fit(scaled)
            pc_names = [f"{fam}_pc{i+1}" for i in range(n_comp)]

            models.append(
                ForecastFamilyPCA(
                    family=fam,
                    columns=cols,
                    scaler=scaler,
                    pca=pca,
                    pc_names=pc_names,
                    explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
                    max_abs_corr=float(max_corr),
                    max_vif=float(max_vif),
                )
            )
            LOGGER.info(
                "[forecast-pca][fit] %s: cols=%s n_comp=%s explained=%.4f max_corr=%.3f max_vif=%.3f",
                fam,
                len(cols),
                n_comp,
                float(np.sum(pca.explained_variance_ratio_)),
                max_corr,
                max_vif,
            )
        return models

    def _apply_forecast_family_pca(
        self,
        part: pd.DataFrame,
        models: list[ForecastFamilyPCA],
    ) -> pd.DataFrame:
        if not models:
            return part
        out = part.copy()
        for m in models:
            cols = [c for c in m.columns if c in out.columns]
            if len(cols) != len(m.columns):
                LOGGER.warning(
                    "[forecast-pca][skip] %s: missing columns at transform time.",
                    m.family,
                )
                continue
            block = out[cols].copy()
            med = block.median(numeric_only=True)
            block = block.fillna(med).fillna(0.0)
            scaled = m.scaler.transform(block)
            pcs = m.pca.transform(scaled)
            for i, name in enumerate(m.pc_names):
                out[name] = pcs[:, i]
            if self.forecast_pca_drop_raw:
                out = out.drop(columns=cols, errors="ignore")
        return out

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
            f"(Starting {PICASSO_RELEASE_UTC}). Rows available: {len(df)}"
        )
        print(f"Rows dropped due to Picasso-Cut: {dropped_picasso}")
        if self.use_forecast_pca and self.forecast_pca_drop_raw:
            LOGGER.warning(
                "forecast_pca_drop_raw=True enabled. Use this only after CV MAE/PnL "
                "A/B validation confirms no degradation."
            )
        bounds = self._default_bounds()
        masks = self._split_masks(df, bounds)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

        config: dict = {
            "artifact_path": str(self.input_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "options": {
                "use_forecast_pca": self.use_forecast_pca,
                "forecast_pca_var_threshold": self.forecast_pca_var_threshold,
                "forecast_corr_threshold": self.forecast_corr_threshold,
                "forecast_vif_threshold": self.forecast_vif_threshold,
                "forecast_pca_drop_raw": self.forecast_pca_drop_raw,
            },
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
        quality_reports: list[pd.DataFrame] = []

        for bundle in ("da", "afrr"):
            targets, audit_targets = self._bundle_targets(df, bundle=bundle)
            primary_canonical = self._canonical_primary_target(bundle)
            primary_source = targets[0]
            base_features = self._feature_columns(df, targets=targets)
            if bundle == "da" and base_features:
                # DA model uses a stricter, auction-causal subset of X.
                base_features = get_da_optimized_features(df[base_features]).columns.tolist()
            elif bundle == "afrr" and base_features:
                # aFRR model drops stale long-term forecast lags to reduce feature overkill.
                base_features = get_afrr_optimized_features(df[base_features]).columns.tolist()

            # Train slice for quality diagnostics + train-fitted imputation stats.
            train_cols = ["timestamp_utc", *base_features, *targets]
            train_cols = list(dict.fromkeys([c for c in train_cols if c in df.columns]))
            train_part = df.loc[masks["train"], train_cols].copy()
            train_part = self._drop_target_nans(train_part, targets)
            features, quality_report = self._feature_quality_from_train(train_part, base_features)
            self._print_feature_quality_report(bundle, quality_report)
            quality_report = quality_report.copy()
            quality_report.insert(0, "bundle", bundle)
            quality_reports.append(quality_report)
            train_medians = train_part[features].median(numeric_only=True) if features else pd.Series(dtype=float)

            bundle_dir = self.output_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)
            forecast_pca_models = (
                self._fit_forecast_family_pca(train_part, features)
                if self.use_forecast_pca
                else []
            )

            for split_name, mask in masks.items():
                cols = ["timestamp_utc", *features, *targets, *audit_targets]
                cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
                part = df.loc[mask, cols].copy()
                # Mandatory target cleaning: drop boundary rows with missing y.
                part = self._drop_target_nans(part, targets)
                # Split-local ffill + train-fitted median fallback.
                part = self._impute_features_with_train_fit(
                    part,
                    features,
                    train_medians,
                    bundle=bundle,
                    split_name=split_name,
                )
                part = self._apply_forecast_family_pca(part, forecast_pca_models)
                active_features = [c for c in features if c in part.columns]
                remaining_nans_x_total += int(part[active_features].isna().sum().sum())
                if primary_source != primary_canonical:
                    part = part.rename(columns={primary_source: primary_canonical})
                part.to_parquet(bundle_dir / f"{split_name}.parquet", index=False)

            targets_out = [primary_canonical] + [t for t in targets[1:] if t != primary_canonical]
            pca_feature_names = [pc for m in forecast_pca_models for pc in m.pc_names]
            if self.forecast_pca_drop_raw:
                dropped_cols = {c for m in forecast_pca_models for c in m.columns}
                features = [c for c in features if c not in dropped_cols]
            features = list(dict.fromkeys(features + pca_feature_names))

            train_for_scaler = self._apply_forecast_family_pca(train_part, forecast_pca_models)
            train_df = train_for_scaler[features].copy()
            scaler_store[bundle] = self._fit_scaler_pipeline(train_df)

            config["bundles"][bundle] = {
                "primary_target": primary_canonical,
                "target_source_columns": targets,
                "features": features,
                "targets": targets_out,
                "audit_targets": audit_targets,
                "feature_quality_report": str(bundle_dir / "feature_quality_report.csv"),
                "forecast_pca": [
                    {
                        "family": m.family,
                        "columns": m.columns,
                        "pc_names": m.pc_names,
                        "explained_variance_ratio": m.explained_variance_ratio,
                        "explained_variance_sum": float(sum(m.explained_variance_ratio)),
                        "max_abs_corr": m.max_abs_corr,
                        "max_vif": m.max_vif,
                    }
                    for m in forecast_pca_models
                ],
                "n_features": len(features),
                "n_targets": len(targets_out),
                "files": {
                    "train": str(bundle_dir / "train.parquet"),
                    "val": str(bundle_dir / "val.parquet"),
                    "test": str(bundle_dir / "test.parquet"),
                },
            }
            quality_report[["feature", "train_nan_pct", "action"]].to_csv(
                bundle_dir / "feature_quality_report.csv",
                index=False,
            )

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
        if quality_reports:
            quality_all = pd.concat(quality_reports, ignore_index=True, sort=False)
            self.quality_report_all_path.parent.mkdir(parents=True, exist_ok=True)
            quality_all.to_csv(self.quality_report_all_path, index=False)
        print(f"Remaining NaNs in X : {remaining_nans_x_total}")
        return config


def load_processed_data(
    bundle: BundleName,
    split: Literal["train", "val", "test"] = "train",
    base_dir: str | Path = "data/model_input",
    *,
    target_col_for_feature_routing: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared X/y matrices for a given bundle and split.

    If `target_col_for_feature_routing` is provided, X is reduced to the
    policy-selected feature variant for that target.
    """
    base = Path(base_dir)
    cfg = json.loads((base / "feature_config.json").read_text(encoding="utf-8"))
    if bundle not in cfg["bundles"]:
        raise KeyError(f"Unknown bundle '{bundle}'. Available: {list(cfg['bundles'].keys())}")
    bcfg = cfg["bundles"][bundle]
    file_path = Path(bcfg["files"][split])
    df = pd.read_parquet(file_path)
    X = df[bcfg["features"]].copy()
    y = df[bcfg["targets"]].copy()
    if target_col_for_feature_routing:
        try:
            from energy_trading.models.training_policy import resolve_feature_columns_for_target

            _, routed_cols = resolve_feature_columns_for_target(list(X.columns), target_col_for_feature_routing)
            X = X[routed_cols].copy()
        except Exception:
            # Keep default full-feature behavior if policy routing is unavailable.
            pass
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
    p.add_argument(
        "--quality-report-all-out",
        default="data/reports/feature_quality_report_all.csv",
        help="Output CSV for combined cross-bundle feature quality report.",
    )
    p.add_argument(
        "--use-forecast-pca",
        action="store_true",
        help="Enable train-fit PCA modules for forecast feature families.",
    )
    p.add_argument(
        "--forecast-pca-var-threshold",
        type=float,
        default=0.95,
        help="Minimum cumulative explained variance per forecast family PCA.",
    )
    p.add_argument(
        "--forecast-corr-threshold",
        type=float,
        default=0.9,
        help="Minimum max absolute correlation to trigger PCA for a family.",
    )
    p.add_argument(
        "--forecast-vif-threshold",
        type=float,
        default=10.0,
        help="Minimum max VIF to trigger PCA for a family.",
    )
    p.add_argument(
        "--forecast-pca-drop-raw",
        action="store_true",
        help="Drop raw family forecast columns after PCA (recommended only after A/B validation).",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli().parse_args()
    factory = MLDataFactory(
        input_path=args.input,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
        scaler_path=args.scaler_out,
        null_report_path=args.null_report_out,
        quality_report_all_path=args.quality_report_all_out,
        use_forecast_pca=args.use_forecast_pca,
        forecast_pca_var_threshold=args.forecast_pca_var_threshold,
        forecast_corr_threshold=args.forecast_corr_threshold,
        forecast_vif_threshold=args.forecast_vif_threshold,
        forecast_pca_drop_raw=args.forecast_pca_drop_raw,
    )
    cfg = factory.build()
    print("[OK] ML bundles erstellt.")
    for bundle, bcfg in cfg["bundles"].items():
        print(f"- {bundle}: n_features={bcfg['n_features']} n_targets={bcfg['n_targets']}")
    print(f"- config: {factory.config_path}")
    print(f"- scaler: {factory.scaler_path}")


if __name__ == "__main__":
    main()
