"""Shared run-id resolution for notebooks and scripts.

This module centralizes run-id discovery with `.env` support and robust
fallbacks to local artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "xgboost": ("xgboost", "xgb"),
    "tft": ("tft",),
    "linear": ("linear",),
}

ENV_RUN_ID_KEYS: dict[str, tuple[str, ...]] = {
    "xgboost": ("RUN_ID_XGBOOST", "MODEL_RUN_ID_XGBOOST", "XGB_RUN_ID"),
    "tft": ("RUN_ID_TFT", "MODEL_RUN_ID_TFT", "TFT_RUN_ID"),
    "linear": ("RUN_ID_LINEAR", "MODEL_RUN_ID_LINEAR", "LINEAR_RUN_ID"),
}


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "src").exists() and (cand / "scripts").exists():
            return cand
    raise FileNotFoundError("Could not detect repo root (expected folders: src/, scripts/).")


def load_dotenv_values(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not path.exists():
        return vals
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("'").strip('"')
        if key:
            vals[key] = val
    return vals


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
    """Resolve one run directory per model using `.env` first, then fallback."""
    root = (repo_root or find_repo_root()).resolve()
    model_runs_root = root / "artifacts" / "model_runs"
    env = load_dotenv_values(root / ".env")

    out: dict[str, Path] = {}
    for model in models:
        m = model.lower()
        aliases = MODEL_ALIASES.get(m, (m,))

        # 1) explicit model-specific env keys
        rid = None
        for key in ENV_RUN_ID_KEYS.get(m, ()):
            val = env.get(key, "").strip()
            if val:
                rid = val
                break
        if rid:
            rd = model_runs_root / rid
            if rd.exists():
                out[m] = rd
                continue

        # 2) generic RUN_ID_PREFIX + model suffix
        prefix = env.get("RUN_ID_PREFIX", "").strip()
        if prefix:
            for a in aliases:
                rd = model_runs_root / f"{prefix}_{a}"
                if rd.exists():
                    out[m] = rd
                    break
            if m in out:
                continue

        # 3) fallback: latest folder containing model alias and manifest
        for rd in _iter_candidate_run_dirs(model_runs_root, aliases=aliases):
            if (rd / "manifest.json").exists():
                out[m] = rd
                break

    return out

