#!/usr/bin/env python3
"""Create thesis-ready training diagnostics from TensorBoard scalar logs.

The figures are intentionally limited to diagnostics that are defensible in a
thesis: final TFT convergence by target and HPO trial behavior. They are not a
replacement for out-of-sample forecast benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_trading.evaluation.style import THESIS_PALETTE, apply_geo_style, thesis_titlecase


TARGET_LABELS = {
    "tft_da_target_da_price": "DA price",
    "tft_afrr_target_afrr_capacity_price_pos": "aFRR capacity price pos",
    "tft_afrr_target_afrr_capacity_price_neg": "aFRR capacity price neg",
    "tft_afrr_target_afrr_activation_price_vwap_pos": "aFRR activation price pos",
    "tft_afrr_target_afrr_activation_price_vwap_neg": "aFRR activation price neg",
    "tft_afrr_target_afrr_activation_rate_pos": "aFRR activation rate pos",
    "tft_afrr_target_afrr_activation_rate_neg": "aFRR activation rate neg",
}

TRAIN_TAG_RE = re.compile(r"(^|[/_])train(_|/)?loss($|[_/])|^train_loss", re.IGNORECASE)
VAL_TAG_RE = re.compile(r"(^|[/_])val(idation)?(_|/)?loss($|[_/])|^val_loss", re.IGNORECASE)
EVENT_GLOB = "events.out.tfevents*"


@dataclass(frozen=True)
class Output:
    artifact_type: str
    path: Path
    thesis_use: str
    description: str


def _require_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: tensorboard. Install it in the active environment, e.g. "
            "`./.venv/bin/python -m pip install tensorboard`, then rerun this script."
        ) from exc
    return EventAccumulator


def _latex_escape(value: Any) -> str:
    s = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(old, new)
    return s


def _latex_color_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", role).lower()


def _latex_color_defs() -> list[str]:
    return [
        f"\\definecolor{{{_latex_color_name(role)}}}{{HTML}}{{{hex_color.lstrip('#').upper()}}}"
        for role, hex_color in THESIS_PALETTE.items()
    ]


def _tex_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    return f"{x:.6g}" if np.isfinite(x) else "nan"


def _tex_symbol(value: Any) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    if not raw:
        raw = "x"
    if raw[0].isdigit():
        raw = "x_" + raw
    return raw


def _write_native_training_summary(path: Path, summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None
    d = summary.sort_values("best_val_loss", ascending=True)
    coords = " ".join(f"({_tex_num(v)},{_tex_symbol(t)})" for t, v in zip(d["target_label"], d["best_val_loss"]))
    labels = d["target_label"].astype(str).tolist()
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[",
        r"                xbar,",
        r"                bar width=12pt,",
        r"                width=0.98\textwidth,",
        r"                height=7cm,",
        r"                xlabel={Best validation loss},",
        "                symbolic y coords={" + ",".join(_tex_symbol(x) for x in labels) + "},",
        "                ytick={" + ",".join(_tex_symbol(x) for x in labels) + "},",
        "                yticklabels={" + ",".join(_latex_escape(x) for x in labels) + "},",
        r"                yticklabel style={font=\scriptsize},",
        r"                axis lines*=left,",
        r"                xmin=0,",
        r"            ]",
        rf"                \addplot[xbar, fill=tertiary, draw=tertiary] coordinates {{{coords}}};",
        r"            \end{axis}",
        r"        \end{tikzpicture}}",
        rf"    \caption{{{_latex_escape(thesis_titlecase('Best validation loss and selected epoch for the final TFT model of each target.'))}}}",
        r"    \label{fig:tft-final-best-validation-loss-by-target}",
        r"\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_native_hpo_objective(path: Path, trial_summary: pd.DataFrame) -> Path | None:
    if trial_summary.empty:
        return None
    d = trial_summary.sort_values("trial_number")
    coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(d["trial_number"], d["best_val_loss"]))
    best = d["best_val_loss"].cummin()
    best_coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(d["trial_number"], best))
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        r"\begin{figure}[p]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[width=0.96\textwidth,height=7cm,xlabel={Trial number},ylabel={Best validation loss},grid=major,axis lines*=left,legend style={at={(0.5,1.08)}, anchor=south, legend columns=-1, draw=none, fill=none, text=black},legend cell align={left}]",
        rf"                \addplot[color=naive, mark=*, mark options={{fill=naive, draw=naive}}, line width=0.8pt] coordinates {{{coords}}};",
        rf"                \addplot[color=primary, mark=none, line width=1.2pt] coordinates {{{best_coords}}};",
        r"                \legend{Trial best, Best observed so far}",
        r"            \end{axis}",
        r"        \end{tikzpicture}}",
        rf"    \caption{{{_latex_escape(thesis_titlecase('TFT HPO objective trace showing the best validation loss observed after each trial.'))}}}",
        r"    \label{fig:tft-hpo-objective-trace}",
        r"\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_native_final_convergence(path: Path, selected: pd.DataFrame, *, final_run: str) -> Path | None:
    df = selected[(selected["run"].eq(final_run)) & (selected["metric"].eq("val"))].copy()
    if df.empty:
        return None
    keep = (
        df.groupby("target_label", as_index=False)["value"]
        .min()
        .sort_values("value", ascending=True)
        .head(7)["target_label"]
        .tolist()
    )
    colors = ["primary", "tertiary", "secondary", "naive", "perfect_foresight", "benchmark", "neutral_dark"]
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        r"\begin{figure}[p]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[width=0.96\textwidth,height=7cm,xlabel={Epoch},ylabel={Validation loss},grid=major,axis lines*=left,legend style={at={(0.5,1.08)}, anchor=south, legend columns=2, font=\scriptsize, draw=none, fill=none, text=black},legend cell align={left}]",
    ]
    legend: list[str] = []
    for i, target in enumerate(keep):
        g = df[df["target_label"].eq(target)].sort_values("step")
        if g.empty:
            continue
        coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(g["step"], g["value"]))
        color = _latex_color_name(colors[i % len(colors)])
        lines.append(rf"                \addplot[color={color}, mark=none, line width=1.1pt] coordinates {{{coords}}};")
        legend.append(_latex_escape(target))
    if not legend:
        return None
    lines.extend(
        [
            "                \\legend{" + ",".join(legend) + "}",
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            rf"    \caption{{{_latex_escape(thesis_titlecase('Validation-loss convergence for the final TFT models by target. The native pgfplots figure is generated from the extracted TensorBoard scalar data.'))}}}",
            r"    \label{fig:tft-final-convergence-by-target}",
            r"\end{figure}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_native_hpo_trajectories(path: Path, selected: pd.DataFrame, trial_summary: pd.DataFrame) -> Path | None:
    if selected.empty or trial_summary.empty:
        return None
    val = selected[(selected["is_trial"]) & (selected["metric"].eq("val"))].copy()
    if val.empty:
        return None
    best_trial = str(trial_summary.loc[trial_summary["best_val_loss"].idxmin(), "trial"])
    trial_order = trial_summary.sort_values("best_val_loss")["trial"].astype(str).tolist()
    selected_trials = [best_trial] + [t for t in trial_order if t != best_trial][:9]
    lines = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Requires: \usepackage{xcolor}",
        r"% Recommended in preamble: \pgfplotsset{compat=1.18}",
        *_latex_color_defs(),
        r"\begin{figure}[p]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \begin{tikzpicture}",
        r"            \begin{axis}[width=0.96\textwidth,height=7cm,xlabel={Epoch},ylabel={Validation loss},grid=major,axis lines*=left,legend style={at={(0.5,1.08)}, anchor=south, legend columns=2, font=\scriptsize, draw=none, fill=none, text=black},legend cell align={left}]",
    ]
    legend: list[str] = []
    for trial in selected_trials:
        g = val[val["run"].eq(trial)].sort_values("step")
        if g.empty:
            continue
        coords = " ".join(f"({_tex_num(x)},{_tex_num(y)})" for x, y in zip(g["step"], g["value"]))
        if trial == best_trial:
            lines.append(rf"                \addplot[color=primary, mark=none, line width=1.5pt] coordinates {{{coords}}};")
            legend.append(_latex_escape(f"{trial} best"))
        else:
            lines.append(rf"                \addplot[color=naive, mark=none, opacity=0.35, line width=0.8pt] coordinates {{{coords}}};")
            legend.append(_latex_escape(trial))
    if not legend:
        return None
    lines.extend(
        [
            "                \\legend{" + ",".join(legend) + "}",
            r"            \end{axis}",
            r"        \end{tikzpicture}}",
            rf"    \caption{{{_latex_escape(thesis_titlecase('Validation-loss trajectories for selected TFT HPO trials. The best trial is highlighted; the remaining lines show the strongest competing trials by best validation loss.'))}}}",
            r"    \label{fig:tft-hpo-validation-loss-trajectories}",
            r"\end{figure}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _target_label(name: str) -> str:
    return TARGET_LABELS.get(name, name.removeprefix("tft_").replace("_target_", " ").replace("_", " "))


def _read_scalars(log_dir: Path) -> pd.DataFrame:
    EventAccumulator = _require_event_accumulator()
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    rows: list[dict[str, Any]] = []
    for tag in ea.Tags().get("scalars", []):
        for event in ea.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["tag", "step", "value", "wall_time"])
    df = pd.DataFrame(rows)
    return (
        df.sort_values(["tag", "step", "wall_time"])
        .drop_duplicates(["tag", "step"], keep="last")
        .reset_index(drop=True)
    )


def _select_tag(tags: list[str], kind: str) -> str | None:
    regex = TRAIN_TAG_RE if kind == "train" else VAL_TAG_RE
    matches = [tag for tag in tags if regex.search(tag)]
    if not matches:
        return None
    preferred = [
        f"{kind}_loss_epoch",
        f"{kind}_loss",
        "epoch_train_loss" if kind == "train" else "epoch_val_loss",
        "train/loss_epoch" if kind == "train" else "val/loss",
    ]
    lower = {tag.lower(): tag for tag in matches}
    for candidate in preferred:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return sorted(matches, key=lambda t: ("/" in t, len(t), t))[0]


def collect_tensorboard_scalars(log_root: Path, *, final_run: str, include_trials: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    run_dirs = [log_root / final_run]
    if include_trials:
        run_dirs.extend(sorted(p for p in log_root.glob("trial_[0-9][0-9][0-9][0-9]") if p.is_dir()))
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        for target_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if not list(target_dir.glob(EVENT_GLOB)):
                continue
            scalar_df = _read_scalars(target_dir)
            if scalar_df.empty:
                continue
            scalar_df["run"] = run_dir.name
            scalar_df["target"] = target_dir.name
            scalar_df["target_label"] = _target_label(target_dir.name)
            scalar_df["is_trial"] = bool(re.fullmatch(r"trial_\d{4}", run_dir.name))
            rows.append(scalar_df)
    if not rows:
        empty = pd.DataFrame(columns=["run", "target", "target_label", "tag", "step", "value", "wall_time", "is_trial"])
        return empty, empty
    scalars = pd.concat(rows, ignore_index=True)

    selected_rows: list[pd.DataFrame] = []
    for (run, target), group in scalars.groupby(["run", "target"], sort=False):
        tags = sorted(group["tag"].dropna().unique().tolist())
        for kind in ["train", "val"]:
            tag = _select_tag(tags, kind)
            if tag is None:
                continue
            part = group[group["tag"].eq(tag)].copy()
            part["metric"] = kind
            selected_rows.append(part)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return scalars, selected


def _savefig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_final_convergence(selected: pd.DataFrame, *, final_run: str, out_dir: Path) -> Path | None:
    df = selected[(selected["run"].eq(final_run)) & (selected["metric"].isin(["train", "val"]))].copy()
    if df.empty:
        return None
    import matplotlib.pyplot as plt

    targets = sorted(df["target_label"].unique())
    ncols = 2
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(4, 2.7 * nrows)), sharex=False)
    axes_arr = np.atleast_1d(axes).ravel()
    colors = {"train": THESIS_PALETTE["naive"], "val": THESIS_PALETTE["tertiary"]}
    labels = {"train": "Training loss", "val": "Validation loss"}
    for ax, target in zip(axes_arr, targets):
        target_df = df[df["target_label"].eq(target)]
        for metric in ["train", "val"]:
            line = target_df[target_df["metric"].eq(metric)].sort_values("step")
            if line.empty:
                continue
            ax.plot(line["step"], line["value"], color=colors[metric], label=labels[metric], linewidth=1.9)
            if metric == "val":
                best_idx = line["value"].idxmin()
                ax.scatter(
                    [line.loc[best_idx, "step"]],
                    [line.loc[best_idx, "value"]],
                    color=THESIS_PALETTE["primary"],
                    s=28,
                    zorder=3,
                    label="Best validation epoch",
                )
        ax.set_title(thesis_titlecase(target))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    for ax in axes_arr[len(targets) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=True)
    fig.suptitle(thesis_titlecase("TFT training convergence by target"), y=1.02)
    fig.tight_layout()
    path = _savefig(fig, out_dir / "figures" / "tft_final_convergence_by_target.png")
    plt.close(fig)
    return path


def _final_summary(selected: pd.DataFrame, *, final_run: str) -> pd.DataFrame:
    df = selected[selected["run"].eq(final_run)].copy()
    rows: list[dict[str, Any]] = []
    for target, group in df.groupby("target_label", sort=True):
        train = group[group["metric"].eq("train")].sort_values("step")
        val = group[group["metric"].eq("val")].sort_values("step")
        if val.empty:
            continue
        best = val.loc[val["value"].idxmin()]
        train_same = train[train["step"].eq(int(best["step"]))]
        train_at_best = float(train_same["value"].iloc[-1]) if not train_same.empty else np.nan
        rows.append(
            {
                "target_label": target,
                "best_epoch": int(best["step"]),
                "best_val_loss": float(best["value"]),
                "train_loss_at_best_epoch": train_at_best,
                "generalization_gap_at_best_epoch": float(best["value"]) - train_at_best if np.isfinite(train_at_best) else np.nan,
                "last_val_loss": float(val["value"].iloc[-1]),
                "n_val_points": int(len(val)),
            }
        )
    return pd.DataFrame(rows)


def plot_final_summary(summary: pd.DataFrame, *, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    import matplotlib.pyplot as plt

    df = summary.sort_values("best_val_loss", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    y = np.arange(len(df))
    ax.barh(y, df["best_val_loss"], color=THESIS_PALETTE["tertiary"], edgecolor=THESIS_PALETTE["neutral_dark"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["target_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Best validation loss")
    ax.set_title(thesis_titlecase("Best TFT validation loss by target"))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["best_val_loss"], i, f"  epoch {int(row['best_epoch'])}", va="center", fontsize=9, color=THESIS_PALETTE["neutral_dark"])
    fig.tight_layout()
    path = _savefig(fig, out_dir / "figures" / "tft_final_best_validation_loss_by_target.png")
    plt.close(fig)
    return path


def _trial_summary(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected[(selected["is_trial"]) & (selected["metric"].eq("val"))].copy()
    rows: list[dict[str, Any]] = []
    for run, group in df.groupby("run", sort=True):
        if group.empty:
            continue
        best = group.loc[group["value"].idxmin()]
        rows.append(
            {
                "trial": run,
                "trial_number": int(run.split("_")[-1]),
                "target_label": str(best["target_label"]),
                "best_epoch": int(best["step"]),
                "best_val_loss": float(best["value"]),
                "last_val_loss": float(group.sort_values("step")["value"].iloc[-1]),
                "n_val_points": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("trial_number").reset_index(drop=True) if rows else pd.DataFrame()


def plot_hpo_trials(selected: pd.DataFrame, trial_summary: pd.DataFrame, *, out_dir: Path) -> list[Path]:
    if selected.empty or trial_summary.empty:
        return []
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    best_trial = str(trial_summary.loc[trial_summary["best_val_loss"].idxmin(), "trial"])
    val = selected[(selected["is_trial"]) & (selected["metric"].eq("val"))].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    for trial, group in val.groupby("run", sort=True):
        g = group.sort_values("step")
        is_best = trial == best_trial
        ax.plot(
            g["step"],
            g["value"],
            color=THESIS_PALETTE["primary"] if is_best else THESIS_PALETTE["naive"],
            alpha=1.0 if is_best else 0.22,
            linewidth=2.4 if is_best else 1.1,
            label=f"{trial} (best)" if is_best else None,
        )
    ax.set_title(thesis_titlecase("TFT HPO validation-loss trajectories"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend(loc="best")
    fig.tight_layout()
    outputs.append(_savefig(fig, out_dir / "figures" / "tft_hpo_validation_loss_trajectories.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.4))
    df = trial_summary.copy()
    ax.plot(df["trial_number"], df["best_val_loss"], color=THESIS_PALETTE["naive"], linewidth=1.4, marker="o", markersize=4)
    running = df["best_val_loss"].cummin()
    ax.step(df["trial_number"], running, where="post", color=THESIS_PALETTE["primary"], linewidth=2.2, label="Best observed so far")
    best = df.loc[df["best_val_loss"].idxmin()]
    ax.scatter([best["trial_number"]], [best["best_val_loss"]], color=THESIS_PALETTE["tertiary"], s=52, zorder=3, label="Selected trial")
    ax.set_title(thesis_titlecase("TFT HPO objective trace"))
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best validation loss")
    ax.legend(loc="best")
    fig.tight_layout()
    outputs.append(_savefig(fig, out_dir / "figures" / "tft_hpo_objective_trace.png"))
    plt.close(fig)
    return outputs


def build_outputs(*, log_root: Path, out_dir: Path, final_run: str, include_trials: bool = True) -> dict[str, Any]:
    apply_geo_style()
    for rel in ["figures", "latex_figures", "csv", "diagnostics"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
    for tex in (out_dir / "latex_figures").glob("*.tex"):
        if r"\includegraphics" in tex.read_text(encoding="utf-8", errors="ignore"):
            tex.unlink()

    raw, selected = collect_tensorboard_scalars(log_root, final_run=final_run, include_trials=include_trials)
    outputs: list[Output] = []
    if raw.empty:
        raise SystemExit(f"No TensorBoard scalar logs found below {log_root}")

    raw_path = out_dir / "csv" / "tensorboard_scalars_long.csv"
    selected_path = out_dir / "csv" / "tensorboard_selected_loss_curves.csv"
    raw.to_csv(raw_path, index=False)
    selected.to_csv(selected_path, index=False)
    outputs.extend(
        [
            Output("csv", raw_path, "backup data", "All scalar tags extracted from TensorBoard event files."),
            Output("csv", selected_path, "backup data", "Selected train/validation loss curves used for figures."),
        ]
    )

    summary = _final_summary(selected, final_run=final_run)
    summary_path = out_dir / "csv" / "tft_final_training_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(Output("csv", summary_path, "backup data", "Best validation epoch and loss by final TFT target."))

    fig = plot_final_convergence(selected, final_run=final_run, out_dir=out_dir)
    if fig is not None:
        outputs.append(Output("figure", fig, "methodology or appendix figure", "Final TFT training and validation loss curves by target."))
        tex = _write_native_final_convergence(
            out_dir / "latex_figures" / "tft_final_convergence_by_target.tex",
            selected,
            final_run=final_run,
        )
        if tex is not None:
            outputs.append(Output("latex_figure", tex, "copy-paste native pgfplots figure code", "Native LaTeX/pgfplots code for final TFT validation convergence curves."))

    fig = plot_final_summary(summary, out_dir=out_dir)
    if fig is not None:
        outputs.append(Output("figure", fig, "appendix figure", "Best final TFT validation loss by target."))
        tex = _write_native_training_summary(out_dir / "latex_figures" / "tft_final_best_validation_loss_by_target.tex", summary)
        if tex is not None:
            outputs.append(Output("latex_figure", tex, "copy-paste native pgfplots figure code", "Native LaTeX/pgfplots code for final validation summary."))

    if include_trials:
        trial_summary = _trial_summary(selected)
        trial_summary_path = out_dir / "csv" / "tft_hpo_trial_summary.csv"
        trial_summary.to_csv(trial_summary_path, index=False)
        outputs.append(Output("csv", trial_summary_path, "backup data", "Best validation loss by TFT HPO trial."))
        for fig in plot_hpo_trials(selected, trial_summary, out_dir=out_dir):
            outputs.append(Output("figure", fig, "methodology or appendix figure", fig.stem.replace("_", " ").capitalize()))
            if fig.stem == "tft_hpo_validation_loss_trajectories":
                tex = _write_native_hpo_trajectories(out_dir / "latex_figures" / f"{fig.stem}.tex", selected, trial_summary)
            elif fig.stem == "tft_hpo_objective_trace":
                tex = _write_native_hpo_objective(out_dir / "latex_figures" / f"{fig.stem}.tex", trial_summary)
            else:
                tex = None
            if tex is not None:
                outputs.append(Output("latex_figure", tex, "copy-paste native pgfplots figure code", f"Native LaTeX/pgfplots code for {fig.name}."))

    tag_inventory = (
        raw.groupby(["run", "target", "tag"], as_index=False)
        .agg(n_points=("value", "size"), min_step=("step", "min"), max_step=("step", "max"))
        .sort_values(["run", "target", "tag"])
    )
    inventory_path = out_dir / "diagnostics" / "tensorboard_tag_inventory.csv"
    tag_inventory.to_csv(inventory_path, index=False)
    outputs.append(Output("diagnostics", inventory_path, "diagnostics", "Available TensorBoard scalar tags by run and target."))

    manifest = {
        "description": "Thesis-ready TensorBoard training diagnostics.",
        "log_root": str(log_root),
        "final_run": final_run,
        "outputs": [
            {
                "artifact_type": out.artifact_type,
                "path": str(out.path),
                "thesis_use": out.thesis_use,
                "description": out.description,
            }
            for out in outputs
        ],
    }
    manifest_path = out_dir / "tensorboard_training_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build thesis-ready figures from TensorBoard logs.")
    p.add_argument("--log-root", default="artifacts/tensorboard_logs", help="Root containing TensorBoard run folders.")
    p.add_argument("--out-dir", default="artifacts/rq1_ml_model_benchmark/tensorboard_training_diagnostics", help="Output directory for PNG, LaTeX, CSV and manifest files.")
    p.add_argument("--final-run", default="tft_20260526_123034", help="Final TFT run folder to visualize by target.")
    p.add_argument("--no-trials", action="store_true", help="Skip trial_XXXX HPO diagnostics.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_outputs(
        log_root=Path(args.log_root),
        out_dir=Path(args.out_dir),
        final_run=str(args.final_run),
        include_trials=not bool(args.no_trials),
    )
    print(f"[OK] TensorBoard diagnostics written: {Path(args.out_dir) / 'tensorboard_training_diagnostics_manifest.json'}")
    print(f"[OK] outputs={len(manifest['outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
