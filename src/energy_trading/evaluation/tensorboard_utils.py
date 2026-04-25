"""Shared TensorBoard helpers for model training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())


def tensorboard_log_root() -> Path:
    return REPO_ROOT / "artifacts" / "tensorboard_logs"


def tensorboard_target_version(model_family: str, bundle: str, target_col: str) -> str:
    return f"{_safe_token(model_family)}_{_safe_token(bundle)}_{_safe_token(target_col)}"


def tensorboard_target_log_dir(
    *,
    run_dir: Path,
    model_family: str,
    bundle: str,
    target_col: str,
) -> Path:
    return tensorboard_log_root() / _safe_token(run_dir.name) / tensorboard_target_version(
        model_family=model_family,
        bundle=bundle,
        target_col=target_col,
    )


def create_summary_writer(log_dir: Path):
    """Create torch SummaryWriter if available; else return None."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None


def log_numeric_scalars(
    writer: Any,
    payload: dict[str, object],
    *,
    prefix: str = "",
    step: int = 0,
) -> int:
    """Recursively log numeric values from nested dict payloads."""
    if writer is None:
        return 0
    logged = 0
    for key, value in payload.items():
        tag = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            logged += log_numeric_scalars(writer, value, prefix=tag, step=step)
            continue
        if isinstance(value, bool):
            writer.add_scalar(tag, float(value), step)
            logged += 1
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(tag, float(value), step)
            logged += 1
            continue
    return logged

