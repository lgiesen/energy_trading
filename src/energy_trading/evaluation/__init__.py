"""Evaluation utilities for forecast and trading model diagnostics."""

from .metrics import (
    compute_forecast_metrics,
    compute_gate_closure_metrics,
    gate_hour_for_target,
    pinball_loss_by_quantile,
    prediction_interval_coverage_probability,
    prediction_interval_normalized_average_width,
    summarize_probabilistic_interval_metrics,
    winkler_score,
)

__all__ = [
    "compute_forecast_metrics",
    "compute_gate_closure_metrics",
    "gate_hour_for_target",
    "pinball_loss_by_quantile",
    "winkler_score",
    "prediction_interval_coverage_probability",
    "prediction_interval_normalized_average_width",
    "summarize_probabilistic_interval_metrics",
    "compute_shared_metrics",
    "EvalMetadata",
    "metrics_to_canonical_rows",
    "append_canonical_metrics_parquet",
    "predictions_to_canonical_df",
    "append_canonical_predictions_parquet",
    "log_metrics_tensorboard_canonical",
]

_SHARED_EVALUATOR_EXPORTS = {
    "EvalMetadata",
    "append_canonical_metrics_parquet",
    "append_canonical_predictions_parquet",
    "compute_shared_metrics",
    "log_metrics_tensorboard_canonical",
    "metrics_to_canonical_rows",
    "predictions_to_canonical_df",
}


def __getattr__(name: str):
    if name in _SHARED_EVALUATOR_EXPORTS:
        from . import shared_evaluator

        return getattr(shared_evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
