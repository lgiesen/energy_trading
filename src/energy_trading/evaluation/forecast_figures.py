from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_trading.visualization.style import THESIS_PALETTE, apply_geo_style


@dataclass
class FigureGenerationResult:
    generated_files: list[Path]
    example_window_report: pd.DataFrame


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def select_typical_week(df: pd.DataFrame, window_days: int = 7, min_coverage: float = 0.8) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    return _select_window(df, mode="typical", window_days=window_days, min_coverage=min_coverage)


def select_high_volatility_week(df: pd.DataFrame, window_days: int = 7, min_coverage: float = 0.8) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    return _select_window(df, mode="high_volatility", window_days=window_days, min_coverage=min_coverage)


def select_spike_week(df: pd.DataFrame, window_days: int = 7, min_coverage: float = 0.8) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    return _select_window(df, mode="spike", window_days=window_days, min_coverage=min_coverage)


def _select_window(df: pd.DataFrame, mode: str, window_days: int, min_coverage: float) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    frame = df.copy()
    frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["target_time_utc", "y_true"]).sort_values("target_time_utc")
    if frame.empty:
        ts = pd.Timestamp("1970-01-01", tz="UTC")
        return ts, ts, {"n": 0, "volatility_score": np.nan, "tail_event_count": 0, "example_window_short": 1}

    frame = frame.set_index("target_time_utc")
    freq = pd.Timedelta(hours=1)
    expected = int(window_days * 24)
    starts = pd.date_range(frame.index.min().floor("D"), frame.index.max().ceil("D"), freq="D", tz="UTC")
    candidates: list[dict[str, Any]] = []
    abs_series = frame["y_true"].astype(float)
    q90 = float(np.nanquantile(abs_series.to_numpy(dtype=float), 0.9))
    for st in starts:
        en = st + pd.Timedelta(days=window_days) - freq
        cut = frame.loc[(frame.index >= st) & (frame.index <= en)]
        if cut.empty:
            continue
        n = int(len(cut))
        coverage = n / max(expected, 1)
        if coverage < min_coverage:
            continue
        vol = float(cut["y_true"].diff().abs().mean())
        tails = int((cut["y_true"].abs() >= q90).sum())
        candidates.append({"start": st, "end": en, "n": n, "coverage": coverage, "vol": vol, "tails": tails})

    if not candidates:
        st = frame.index.min()
        en = frame.index.max()
        cut = frame.loc[(frame.index >= st) & (frame.index <= en)]
        vol = float(cut["y_true"].diff().abs().mean()) if not cut.empty else np.nan
        tails = int((cut["y_true"].abs() >= q90).sum()) if not cut.empty else 0
        return st, en, {"n": int(len(cut)), "volatility_score": vol, "tail_event_count": tails, "example_window_short": 1}

    cand_df = pd.DataFrame(candidates)
    if mode == "typical":
        target_vol = float(cand_df["vol"].median())
        idx = (cand_df["vol"] - target_vol).abs().idxmin()
    elif mode == "high_volatility":
        thresh = float(cand_df["vol"].quantile(0.9))
        high = cand_df[cand_df["vol"] >= thresh]
        idx = high["vol"].idxmax() if not high.empty else cand_df["vol"].idxmax()
    else:
        top = cand_df.sort_values(["tails", "vol"], ascending=[False, False]).iloc[0]
        return top["start"], top["end"], {"n": int(top["n"]), "volatility_score": float(top["vol"]), "tail_event_count": int(top["tails"]), "example_window_short": 0}
    row = cand_df.loc[idx]
    return row["start"], row["end"], {"n": int(row["n"]), "volatility_score": float(row["vol"]), "tail_event_count": int(row["tails"]), "example_window_short": 0}


def plot_leadtime_metric_comparison(by_lead: pd.DataFrame, figures_dir: Path, dpi: int) -> list[Path]:
    out: list[Path] = []
    if by_lead.empty:
        return out
    for (split, target), tg in by_lead.groupby(["split", "target"]):
        tg = tg.sort_values("lead_time_h")
        for metric in ["mae_p50", "mean_pinball", "approx_crps"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for model, mg in tg.groupby("model"):
                ax.plot(mg["lead_time_h"], mg[metric], marker="o", label=model)
            ax.set_title(f"{metric} by lead | {split} | {target}")
            ax.set_xlabel("lead_time_h")
            ax.set_ylabel(metric)
            ax.legend()
            p = figures_dir / split / target / f"leadtime_{metric}.png"
            _save_fig(fig, p, dpi)
            out.append(p)
    return out


def plot_calibration_curve(calibration: pd.DataFrame, figures_dir: Path, dpi: int) -> list[Path]:
    out: list[Path] = []
    if calibration.empty:
        return out
    for (split, target), tg in calibration.groupby(["split", "target"]):
        fig, ax = plt.subplots(figsize=(7, 6))
        for model, mg in tg.groupby("model"):
            mg = mg.sort_values("quantile")
            ax.plot(mg["quantile"], mg["empirical_coverage"], marker="o", label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["neutral_dark"], label="perfect")
        ax.set_title(f"Calibration | {split} | {target}")
        ax.set_xlabel("nominal quantile")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        p = figures_dir / split / target / "calibration_curve.png"
        _save_fig(fig, p, dpi)
        out.append(p)
    return out


def plot_coverage_and_width_by_lead(by_lead: pd.DataFrame, figures_dir: Path, dpi: int) -> list[Path]:
    out: list[Path] = []
    if by_lead.empty or "coverage_p10_p90" not in by_lead.columns:
        return out
    for (split, target), tg in by_lead.groupby(["split", "target"]):
        tg = tg.sort_values("lead_time_h")
        for metric in ["coverage_p10_p90", "interval_width_p10_p90"]:
            if metric not in tg.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            for model, mg in tg.groupby("model"):
                ax.plot(mg["lead_time_h"], mg[metric], marker="o", label=model)
            ax.set_title(f"{metric} by lead | {split} | {target}")
            ax.set_xlabel("lead_time_h")
            ax.set_ylabel(metric)
            ax.legend()
            name = "coverage_p10_p90_by_lead.png" if metric == "coverage_p10_p90" else "interval_width_p10_p90_by_lead.png"
            p = figures_dir / split / target / name
            _save_fig(fig, p, dpi)
            out.append(p)
    return out


def plot_forecast_band_example(df: pd.DataFrame, figures_dir: Path, dpi: int, example_type: str, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
    window = df.loc[(df["target_time_utc"] >= start) & (df["target_time_utc"] <= end)].copy()
    if window.empty:
        return None
    window = window.sort_values("target_time_utc")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window["target_time_utc"], window["y_true"], label="truth", color=THESIS_PALETTE["neutral_dark"])
    ax.plot(window["target_time_utc"], window["p50"], label="p50", color=THESIS_PALETTE["primary"])
    if {"p10", "p90"}.issubset(window.columns):
        ax.fill_between(window["target_time_utc"], window["p10"], window["p90"], alpha=0.2, color=THESIS_PALETTE["secondary"], label="p10-p90")
    if {"p30", "p70"}.issubset(window.columns):
        ax.fill_between(window["target_time_utc"], window["p30"], window["p70"], alpha=0.25, color=THESIS_PALETTE["tertiary"], label="p30-p70")
    model = str(window["model"].iloc[0])
    target = str(window["target"].iloc[0])
    split = str(window["split"].iloc[0])
    ax.set_title(f"{example_type} | {split} | {target} | {model}")
    ax.legend()
    p = figures_dir / split / target / model / f"{example_type}_forecast_band.png"
    _save_fig(fig, p, dpi)
    return p


def plot_tail_event_scatter(df: pd.DataFrame, figures_dir: Path, dpi: int) -> list[Path]:
    out: list[Path] = []
    for (split, target, model), g in df.groupby(["split", "target", "model"]):
        if g.empty:
            continue
        q90 = float(np.nanquantile(g["y_true"].to_numpy(dtype=float), 0.9))
        mask = g["y_true"] >= q90
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(g["y_true"], g["p50"], s=10, alpha=0.5, color=THESIS_PALETTE["neutral_dark"])
        ax.scatter(g.loc[mask, "y_true"], g.loc[mask, "p50"], s=16, alpha=0.8, color=THESIS_PALETTE["tertiary"])
        ax.set_xlabel("true value")
        ax.set_ylabel("p50 forecast")
        ax.set_title(f"Tail scatter | {split} | {target} | {model}")
        p = figures_dir / split / target / model / "tail_event_scatter.png"
        _save_fig(fig, p, dpi)
        out.append(p)
    return out


def _residual_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = df.copy()
    w["residual"] = pd.to_numeric(w["p50"], errors="coerce") - pd.to_numeric(w["y_true"], errors="coerce")
    w["target_time_utc"] = pd.to_datetime(w["target_time_utc"], utc=True, errors="coerce")
    w["hour_of_day"] = w["target_time_utc"].dt.hour
    if "lead_time_h" in w.columns:
        w["lead_time_h"] = pd.to_numeric(w["lead_time_h"], errors="coerce")
    w["true_value_bin"] = pd.qcut(pd.to_numeric(w["y_true"], errors="coerce"), q=10, duplicates="drop").astype(str)
    return w, w.dropna(subset=["residual"])


def plot_residual_diagnostics(df: pd.DataFrame, figures_dir: Path, dpi: int) -> tuple[list[Path], pd.DataFrame]:
    out: list[Path] = []
    rows: list[dict[str, Any]] = []
    for (split, target, model), g in df.groupby(["split", "target", "model"]):
        full, clean = _residual_frames(g)
        for col, fn in [("hour_of_day", "residual_by_hour_of_day.png"), ("lead_time_h", "residual_by_lead_time.png"), ("true_value_bin", "residual_by_true_value_bin.png")]:
            if col not in clean.columns:
                continue
            gr = clean.groupby(col, dropna=True)
            agg = gr["residual"].agg(["count", "mean"]).reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(agg[col].astype(str), agg["mean"], marker="o")
            ax.set_title(f"Residual mean by {col} | {split} | {target} | {model}")
            ax.set_xlabel(col)
            ax.set_ylabel("mean residual (p50 - y_true)")
            p = figures_dir / split / target / model / fn
            _save_fig(fig, p, dpi)
            out.append(p)
            for _, r in gr:
                yy = pd.to_numeric(r["y_true"], errors="coerce").to_numpy(dtype=float)
                ph = pd.to_numeric(r["p50"], errors="coerce").to_numpy(dtype=float)
                resid = ph - yy
                gv = str(r[col].iloc[0])
                rows.append({
                    "model": model, "split": split, "target": target, "group_type": col, "group_value": gv, "n": int(len(r)),
                    "mean_residual": float(np.nanmean(resid)),
                    "mae_p50": float(np.nanmean(np.abs(resid))),
                    "rmse_p50": float(np.sqrt(np.nanmean(resid**2))),
                    "overprediction_rate": float(np.nanmean(resid > 0)),
                    "underprediction_rate": float(np.nanmean(resid < 0)),
                })
    return out, pd.DataFrame(rows)


def plot_volatility_diagnostics(df: pd.DataFrame, figures_dir: Path, dpi: int) -> tuple[list[Path], pd.DataFrame]:
    out: list[Path] = []
    rows: list[dict[str, Any]] = []
    for (split, target), g in df.groupby(["split", "target"]):
        w = g.copy()
        w["abs_err"] = (pd.to_numeric(w["p50"], errors="coerce") - pd.to_numeric(w["y_true"], errors="coerce")).abs()
        w["realized_vol"] = pd.to_numeric(w["y_true"], errors="coerce").diff().abs().fillna(0.0)
        if w["realized_vol"].nunique() <= 1:
            continue
        w["volatility_bucket"] = pd.qcut(w["realized_vol"], q=5, duplicates="drop").astype(str)
        agg = w.groupby(["model", "volatility_bucket"], as_index=False).agg(
            n=("abs_err", "size"),
            mae_p50=("abs_err", "mean"),
            rmse_p50=("abs_err", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mean_pinball=("abs_err", "mean"),
        )
        for _, r in agg.iterrows():
            rows.append({"model": r["model"], "split": split, "target": target, "volatility_bucket": r["volatility_bucket"], "n": int(r["n"]), "mae_p50": float(r["mae_p50"]), "rmse_p50": float(r["rmse_p50"]), "mean_pinball": float(r["mean_pinball"]), "coverage_p10_p90": np.nan, "interval_width_p10_p90": np.nan})
        fig, ax = plt.subplots(figsize=(10, 4))
        for model, mg in agg.groupby("model"):
            ax.plot(mg["volatility_bucket"], mg["mae_p50"], marker="o", label=model)
        ax.set_title(f"Error by volatility bucket | {split} | {target}")
        ax.set_xlabel("volatility bucket")
        ax.set_ylabel("mae_p50")
        ax.legend()
        p1 = figures_dir / split / target / "error_by_volatility_bucket.png"
        _save_fig(fig, p1, dpi)
        out.append(p1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(w["realized_vol"], w["abs_err"], s=8, alpha=0.4)
        ax.set_xlabel("realized volatility")
        ax.set_ylabel("absolute error")
        ax.set_title(f"Interval width vs realized abs error | {split} | {target}")
        p2 = figures_dir / split / target / "interval_width_vs_realized_abs_error.png"
        _save_fig(fig, p2, dpi)
        out.append(p2)
    return out, pd.DataFrame(rows)


def generate_forecast_benchmark_figures(
    *,
    joined_df: pd.DataFrame,
    by_lead: pd.DataFrame,
    calibration: pd.DataFrame,
    figures_dir: Path,
    diagnostics_dir: Path,
    config: dict[str, Any],
) -> FigureGenerationResult:
    apply_geo_style()
    fig_cfg = config.get("figures", {})
    make_cfg = fig_cfg.get("make", {})
    dpi = int(fig_cfg.get("dpi", 150))
    window_days = int(fig_cfg.get("example_windows", {}).get("window_days", 7))
    min_coverage = float(fig_cfg.get("example_windows", {}).get("min_coverage", 0.8))

    generated: list[Path] = []
    if bool(make_cfg.get("leadtime_metrics", True)):
        generated.extend(plot_leadtime_metric_comparison(by_lead, figures_dir, dpi))
    if bool(make_cfg.get("calibration", True)):
        generated.extend(plot_calibration_curve(calibration, figures_dir, dpi))
    if bool(make_cfg.get("coverage_width", True)):
        generated.extend(plot_coverage_and_width_by_lead(by_lead, figures_dir, dpi))
    if bool(make_cfg.get("tail_scatter", True)):
        generated.extend(plot_tail_event_scatter(joined_df, figures_dir, dpi))

    residual_df = pd.DataFrame()
    if bool(make_cfg.get("residual_diagnostics", True)):
        g, residual_df = plot_residual_diagnostics(joined_df, figures_dir, dpi)
        generated.extend(g)
    vol_df = pd.DataFrame()
    if bool(make_cfg.get("volatility_diagnostics", True)):
        g, vol_df = plot_volatility_diagnostics(joined_df, figures_dir, dpi)
        generated.extend(g)

    ex_rows: list[dict[str, Any]] = []
    if bool(make_cfg.get("forecast_bands", True)):
        for (model, split, target), g in joined_df.groupby(["model", "split", "target"]):
            for ex_type, selector in [
                ("typical_week", select_typical_week),
                ("high_volatility_week", select_high_volatility_week),
                ("spike_week", select_spike_week),
            ]:
                st, en, meta = selector(g, window_days=window_days, min_coverage=min_coverage)
                out = plot_forecast_band_example(g, figures_dir, dpi, ex_type, st, en)
                if out is not None:
                    generated.append(out)
                ex_rows.append({"model": model, "split": split, "target": target, "example_type": ex_type, "start_utc": str(st), "end_utc": str(en), **meta})

    example_report = pd.DataFrame(ex_rows)
    if not example_report.empty:
        example_report.to_csv(diagnostics_dir / "example_window_report.csv", index=False)
    else:
        (diagnostics_dir / "example_window_report.csv").write_text("", encoding="utf-8")

    if not residual_df.empty:
        residual_df.to_csv(figures_dir.parent / "metrics" / "metrics_residual_patterns.csv", index=False)
    else:
        (figures_dir.parent / "metrics" / "metrics_residual_patterns.csv").write_text("", encoding="utf-8")
    if not vol_df.empty:
        vol_df.to_csv(figures_dir.parent / "metrics" / "metrics_volatility_regimes.csv", index=False)
    else:
        (figures_dir.parent / "metrics" / "metrics_volatility_regimes.csv").write_text("", encoding="utf-8")

    # Directional bias summary.
    bias_rows: list[dict[str, Any]] = []
    for (model, split, target), g in joined_df.groupby(["model", "split", "target"]):
        resid = pd.to_numeric(g["p50"], errors="coerce") - pd.to_numeric(g["y_true"], errors="coerce")
        q90 = float(np.nanquantile(pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float), 0.9))
        tail = pd.to_numeric(g["y_true"], errors="coerce") >= q90
        bias_rows.append({
            "model": model,
            "split": split,
            "target": target,
            "n": int(len(g)),
            "bias": float(np.nanmean(resid)),
            "overprediction_rate": float(np.nanmean(resid > 0)),
            "underprediction_rate": float(np.nanmean(resid < 0)),
            "tail_underprediction_rate": float(np.nanmean((resid < 0) & tail)),
            "tail_overprediction_rate": float(np.nanmean((resid > 0) & tail)),
            "directional_accuracy_delta": float(np.nanmean(np.sign(pd.to_numeric(g["p50"], errors="coerce").diff().fillna(0)) == np.sign(pd.to_numeric(g["y_true"], errors="coerce").diff().fillna(0)))),
        })
    pd.DataFrame(bias_rows).to_csv(figures_dir.parent / "metrics" / "metrics_directional_bias.csv", index=False)

    return FigureGenerationResult(generated_files=generated, example_window_report=example_report)
