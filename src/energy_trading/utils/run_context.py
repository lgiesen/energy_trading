"""Shared run-id resolution for notebooks and scripts.

This module centralizes run-id discovery via model-specific latest pointers
and robust fallbacks to local artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "xgboost": ("xgboost", "xgb"),
    "tft": ("tft",),
    "linear": ("linear",),
}

def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "src").exists() and (cand / "scripts").exists():
            return cand
    raise FileNotFoundError("Could not detect repo root (expected folders: src/, scripts/).")


def _iter_candidate_run_dirs(model_runs_root: Path, *, aliases: Iterable[str]) -> list[Path]:
    if not model_runs_root.exists():
        return []
    alias_l = tuple(a.lower() for a in aliases)
    cands = []
    for p in sorted(model_runs_root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if any(a in name for a in alias_l):
            cands.append(p)
    return cands


def resolve_model_run_dirs(
    *,
    repo_root: Path | None = None,
    models: Iterable[str] = ("xgboost", "tft", "linear"),
) -> dict[str, Path]:
    """Resolve one run directory per model using latest_<model>.json first, then fallback."""
    root = (repo_root or find_repo_root()).resolve()
    model_runs_root = root / "artifacts" / "model_runs"

    out: dict[str, Path] = {}
    for model in models:
        m = model.lower()
        aliases = MODEL_ALIASES.get(m, (m,))

        # 1) model-specific latest pointer
        latest_ptr = model_runs_root / f"latest_{m}.json"
        if latest_ptr.exists():
            try:
                payload = json.loads(latest_ptr.read_text(encoding="utf-8"))
                rid = str(payload.get("run_id", "")).strip()
                if rid:
                    rd = model_runs_root / rid
                    if rd.exists() and (rd / "manifest.json").exists():
                        out[m] = rd
                        continue
            except Exception:
                pass

        # 2) fallback: latest folder containing model alias and manifest
        for rd in _iter_candidate_run_dirs(model_runs_root, aliases=aliases):
            if (rd / "manifest.json").exists():
                out[m] = rd
                break

    return out
