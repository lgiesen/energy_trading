"""Central model-data preparation utilities for time-series training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler


LEAKAGE_TARGET_COLS = [
    "da_price",
    "afrr_capacity_price_pos",
    "afrr_capacity_price_neg",
    "afrr_activation_price_vwap_pos",
    "afrr_activation_price_vwap_neg",
    "afrr_activated_mw_pos",
    "afrr_activated_mw_neg",
    "afrr_da_price_spread",
    "afrr_neg_da_price_spread",
]

NON_FEATURE_META_COLS = [
    "is_local_reconstruction_only",
    "data_is_lagged",
    "pit_lagged_column_count",
    # Must be train-fit only; drop any precomputed cluster to avoid leakage.
    "market_state_cluster",
]


@dataclass(frozen=True)
class SplitFrameSet:
    """Chronological train/val/test split containers."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class DualPathPreparedData:
    """Prepared raw + scaled data paths for XGBoost and TFT."""

    timestamps: SplitFrameSet
    X_raw: SplitFrameSet
    y_raw: SplitFrameSet
    X_scaled: SplitFrameSet
    y_scaled: SplitFrameSet
    feature_columns: list[str]
    target_col: str


class DualPathPreprocessor:
    """Leakage-safe dual path preprocessor.

    Path A:
    - XGBoost path keeps unscaled raw numeric features.

    Path B:
    - TFT path applies RobustScaler to X and y.
    - Scalers are fit strictly on train split and reused for val/test/inference.
    """

    def __init__(
        self,
        *,
        timestamp_col: str = "timestamp_utc",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError("train_ratio, val_ratio, test_ratio must be > 0.")
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
        self.timestamp_col = timestamp_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.feature_columns: list[str] = []
        self.target_col: str | None = None
        self._is_fitted = False

    def _sort_and_filter(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not found in dataframe")

        data = df.copy()
        if self.timestamp_col not in data.columns:
            raise KeyError(f"timestamp column '{self.timestamp_col}' not found in dataframe")
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col], utc=True, errors="coerce")
        data = data.dropna(subset=[self.timestamp_col]).sort_values(self.timestamp_col).reset_index(drop=True)

        y = pd.to_numeric(data[target_col], errors="coerce")
        keep = y.notna()
        data = data.loc[keep].reset_index(drop=True)
        if len(data) < 20:
            raise ValueError("Not enough rows after chronological filtering and non-null target selection.")
        return data

    def _build_feature_frame(self, data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        drop_from_x = [c for c in LEAKAGE_TARGET_COLS if c in data.columns]
        drop_from_x.extend([c for c in NON_FEATURE_META_COLS if c in data.columns])
        drop_from_x.extend([c for c in data.columns if c.startswith("target_")])
        X = data.drop(columns=sorted(set(drop_from_x)), errors="ignore")
        X = X.select_dtypes(include=[np.number]).copy()
        y = pd.to_numeric(data[target_col], errors="coerce")
        ts = pd.to_datetime(data[self.timestamp_col], utc=True, errors="coerce")
        return X, y, ts

    @staticmethod
    def _split_indices(n_rows: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
        tr_end = int(n_rows * train_ratio)
        va_end = tr_end + int(n_rows * val_ratio)
        if tr_end <= 0 or va_end <= tr_end or va_end >= n_rows:
            raise ValueError("Invalid chronological split boundaries. Check ratios and dataset size.")
        return slice(0, tr_end), slice(tr_end, va_end), slice(va_end, n_rows)

    @staticmethod
    def _to_frame(values: np.ndarray, index: pd.Index, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=cols)

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        scaler_out: str | Path | None = None,
    ) -> DualPathPreparedData:
        data = self._sort_and_filter(df, target_col=target_col)
        X, y, ts = self._build_feature_frame(data, target_col=target_col)

        tr_sl, va_sl, te_sl = self._split_indices(len(X), self.train_ratio, self.val_ratio)

        X_train = X.iloc[tr_sl].copy()
        X_val = X.iloc[va_sl].copy()
        X_test = X.iloc[te_sl].copy()
        y_train = y.iloc[tr_sl].copy()
        y_val = y.iloc[va_sl].copy()
        y_test = y.iloc[te_sl].copy()

        # Train-fit imputation only.
        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        self.feature_columns = list(X_train.columns)
        self.target_col = target_col

        Xtr_np = X_train.to_numpy(dtype=float)
        Xva_np = X_val.to_numpy(dtype=float)
        Xte_np = X_test.to_numpy(dtype=float)
        ytr_np = y_train.to_numpy(dtype=float).reshape(-1, 1)
        yva_np = y_val.to_numpy(dtype=float).reshape(-1, 1)
        yte_np = y_test.to_numpy(dtype=float).reshape(-1, 1)

        self.feature_scaler.fit(Xtr_np)
        self.target_scaler.fit(ytr_np)

        Xtr_sc = self.feature_scaler.transform(Xtr_np)
        Xva_sc = self.feature_scaler.transform(Xva_np)
        Xte_sc = self.feature_scaler.transform(Xte_np)
        ytr_sc = self.target_scaler.transform(ytr_np).reshape(-1)
        yva_sc = self.target_scaler.transform(yva_np).reshape(-1)
        yte_sc = self.target_scaler.transform(yte_np).reshape(-1)

        self._is_fitted = True
        if scaler_out is not None:
            self.save_scalers(scaler_out)

        idx_train = X_train.index
        idx_val = X_val.index
        idx_test = X_test.index

        return DualPathPreparedData(
            timestamps=SplitFrameSet(
                train=ts.iloc[tr_sl].to_frame(name=self.timestamp_col),
                val=ts.iloc[va_sl].to_frame(name=self.timestamp_col),
                test=ts.iloc[te_sl].to_frame(name=self.timestamp_col),
            ),
            X_raw=SplitFrameSet(train=X_train, val=X_val, test=X_test),
            y_raw=SplitFrameSet(
                train=y_train.to_frame(name=target_col),
                val=y_val.to_frame(name=target_col),
                test=y_test.to_frame(name=target_col),
            ),
            X_scaled=SplitFrameSet(
                train=self._to_frame(Xtr_sc, idx_train, self.feature_columns),
                val=self._to_frame(Xva_sc, idx_val, self.feature_columns),
                test=self._to_frame(Xte_sc, idx_test, self.feature_columns),
            ),
            y_scaled=SplitFrameSet(
                train=pd.DataFrame({target_col: ytr_sc}, index=idx_train),
                val=pd.DataFrame({target_col: yva_sc}, index=idx_val),
                test=pd.DataFrame({target_col: yte_sc}, index=idx_test),
            ),
            feature_columns=self.feature_columns,
            target_col=target_col,
        )

    def transform_features(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        if not self.feature_columns:
            raise RuntimeError("No feature column metadata present.")
        X = X_df.reindex(columns=self.feature_columns).copy()
        X = X.fillna(X.median(numeric_only=True))
        arr = self.feature_scaler.transform(X.to_numpy(dtype=float))
        return pd.DataFrame(arr, index=X.index, columns=self.feature_columns)

    def transform_target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        arr = np.asarray(y, dtype=float).reshape(-1, 1)
        return self.target_scaler.transform(arr).reshape(-1)

    def inverse_transform_target(self, y_scaled: pd.Series | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        arr = np.asarray(y_scaled, dtype=float).reshape(-1, 1)
        return self.target_scaler.inverse_transform(arr).reshape(-1)

    def save_scalers(self, out_path: str | Path) -> Path:
        if not self._is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_col": self.target_col,
            "timestamp_col": self.timestamp_col,
        }
        joblib.dump(payload, out)
        return out

    def load_scalers(self, in_path: str | Path) -> None:
        payload = joblib.load(in_path)
        self.feature_scaler = payload["feature_scaler"]
        self.target_scaler = payload["target_scaler"]
        self.feature_columns = list(payload["feature_columns"])
        self.target_col = payload["target_col"]
        self._is_fitted = True


def inverse_transform_scaled_target(
    y_scaled: pd.Series | np.ndarray,
    *,
    scaler_pkl_path: str | Path,
) -> np.ndarray:
    """Inverse-transform TFT target predictions from scaled space."""
    payload = joblib.load(scaler_pkl_path)
    scaler: RobustScaler = payload["target_scaler"]
    arr = np.asarray(y_scaled, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def _add_market_state_cluster_feature(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit scaler + KMeans on train and append cluster ids to train/test.

    Uses only features available in both train and test. If insufficient inputs
    are present, returns the original frames unchanged.
    """
    candidates = [
        "nrv_zscore_24h",
        "grid_stress_index",
        "picasso_flow_rate_lag_1h",
        "picasso_flow_rate_lag_24h",
        "mfrr_active_lag",
    ]
    use_cols = [c for c in candidates if c in X_train_df.columns and c in X_test_df.columns]
    if len(use_cols) < 2:
        return X_train_df, X_test_df

    train_block = X_train_df[use_cols].copy().fillna(0.0)
    test_block = X_test_df[use_cols].copy().fillna(0.0)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_block)
    test_scaled = scaler.transform(test_block)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster = km.fit_predict(train_scaled).astype(float)
    test_cluster = km.predict(test_scaled).astype(float)

    out_train = X_train_df.copy()
    out_test = X_test_df.copy()
    out_train["market_state_cluster"] = train_cluster
    out_test["market_state_cluster"] = test_cluster
    return out_train, out_test


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str,
    model_type: Literal["xgboost", "rf", "linear", "nn"] = "xgboost",
    test_size: float = 0.2,
    add_train_fit_cluster: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare leakage-safe train/test tensors for model training.

    This helper is the single entrypoint for model-ready dataset creation across
    tree-based, linear, and neural pipelines.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing features, ground-truth targets, and optionally
        `timestamp_utc`.
    target_col : str
        Name of the target column to predict.
    model_type : {"xgboost", "rf", "linear", "nn"}, default "xgboost"
        Controls formatting/scaling:
        - `"xgboost"` / `"rf"`: no feature scaling
        - `"linear"` / `"nn"`: apply `StandardScaler` fit on train only
    test_size : float, default 0.2
        Fraction of rows reserved for test split (chronological tail).
    add_train_fit_cluster : bool, default True
        If True, appends `market_state_cluster` generated by train-only fit of
        `StandardScaler` + `KMeans`, then predicts cluster ids on test.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        `(X_train, X_test, y_train, y_test)` prepared for direct model fitting.

    Raises
    ------
    KeyError
        If `target_col` is not present in `df`.
    ValueError
        If `test_size` is invalid, or too few rows remain after filtering.

    Notes
    -----
    - **Target leakage prevention:** unlagged primary market targets
      (e.g., `afrr_vwap_*`, `afrr_activated_*`, spreads) are removed from `X`.
      Additionally, all columns prefixed with `target_` are excluded from `X`
      except the selected `target_col`.
    - **Strict metadata exclusion:** technical QA/meta columns
      (`is_local_reconstruction_only`, `data_is_lagged`,
      `pit_lagged_column_count`) are always removed from `X`.
    - **Chronological split only:** data is sorted by `timestamp_utc` (if
      available) and split sequentially (no shuffling), preventing look-ahead
      bias in power-market time series.
    - **Scaling discipline:** scaler is fit on `X_train` only and then applied to
      `X_test`, avoiding information bleed from test to train.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = prepare_model_data(
    ...     df, target_col="target_afrr_activation_price_vwap_pos_h1", model_type="xgboost"
    ... )
    >>> X_train_lin, X_test_lin, y_train_lin, y_test_lin = prepare_model_data(
    ...     df, target_col="target_afrr_activation_price_vwap_pos_h1", model_type="linear"
    ... )
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")

    data = df.copy()

    if "timestamp_utc" in data.columns:
        ts = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
        data = data.assign(timestamp_utc=ts).sort_values("timestamp_utc").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    y = pd.to_numeric(data[target_col], errors="coerce")
    keep_mask = y.notna()
    data = data.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)

    if len(data) < 10:
        raise ValueError("Not enough rows after target filtering to perform train/test split.")

    # Remove leakage-prone unlagged market targets from feature matrix.
    drop_from_x = [c for c in LEAKAGE_TARGET_COLS if c in data.columns]
    # Remove strict non-feature metadata columns.
    drop_from_x.extend([c for c in NON_FEATURE_META_COLS if c in data.columns])
    # Targets are supervised labels and must never appear in X.
    drop_from_x.extend([c for c in data.columns if c.startswith("target_")])
    drop_from_x = sorted(set(drop_from_x))
    X = data.drop(columns=drop_from_x, errors="ignore")

    # Keep only numeric features for model readiness.
    X = X.select_dtypes(include=[np.number])

    # Basic, leakage-safe imputation for model compatibility.
    X = X.fillna(X.median(numeric_only=True))

    split_idx = int(len(X) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Invalid split point computed. Check test_size and dataset length.")

    X_train_df = X.iloc[:split_idx].copy()
    X_test_df = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].to_numpy(dtype=float)
    y_test = y.iloc[split_idx:].to_numpy(dtype=float)

    if add_train_fit_cluster:
        X_train_df, X_test_df = _add_market_state_cluster_feature(X_train_df, X_test_df)

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)

    mt = model_type.lower()
    if mt in {"xgboost", "rf"}:
        return X_train, X_test, y_train, y_test
    if mt in {"linear", "nn"}:
        # Scale only explicit lag features to preserve interpretability of
        # calendar/flag/meta-derived magnitudes.
        lag_cols = [i for i, c in enumerate(X_train_df.columns) if "_lag_" in c]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if lag_cols:
            scaler = StandardScaler()
            X_train_scaled[:, lag_cols] = scaler.fit_transform(X_train[:, lag_cols])
            X_test_scaled[:, lag_cols] = scaler.transform(X_test[:, lag_cols])
        return X_train_scaled, X_test_scaled, y_train, y_test

    raise ValueError(f"Unsupported model_type '{model_type}'. Use xgboost, rf, linear, or nn.")
