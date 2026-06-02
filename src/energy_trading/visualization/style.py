"""Central plotting style for thesis figures (functional minimalist palette).

Plotting best practices for thesis figures:
- Use "primary" (#226E9C) for the main focus of the plot
  (e.g., actual profit, proposed model).
- Use "secondary" (#d9b98d) primarily for bar charts, areas, or thick lines
  to show context/comparison.
- Use "tertiary" (#7C1D6F) for a third distinct category or theoretical
  maximums (Oracle).
- Use "neutral_dark" (#333333) for naive benchmarks, borders, and thin
  context lines.

Reuse this module across Python scripts and notebooks:

    from energy_trading.visualization.style import apply_geo_style, THESIS_PALETTE
    apply_geo_style()
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

# GeoDataViz multi-hue diverging palette.
GEO_DIVERGING: Dict[str, str] = {
    "div_1": "#045275",
    "div_2": "#089099",
    "div_3": "#7CCBA2",
    "div_4": "#FCDE9C",
    "div_5": "#F0746E",
    "div_6": "#DC3977",
    "div_7": "#7C1D6F",
}

# GeoDataViz single-hue sequential blue palette.
GEO_SEQUENTIAL_BLUE: Dict[str, str] = {
    "seq_1": "#E4F1F7",
    "seq_2": "#C5E1EF",
    "seq_3": "#9EC9E2",
    "seq_4": "#6CB0D6",
    "seq_5": "#3C93C2",
    "seq_6": "#226E9C",
    "seq_7": "#0D4A70",
}

# Functional thesis palette (minimalist, high-contrast roles).
THESIS_PALETTE: Dict[str, str] = {
    "primary": "#226E9C",  # 34, 110, 156
    "secondary": "#d9b98d",  # 217, 185, 141
    "tertiary": "#7C1D6F",  # 124, 29, 111
    "neutral_dark": "#333333",  # 51, 51, 51
}

# Fixed semantic mapping for thesis figures.
MODEL_COLOR_MAP: Dict[str, str] = {
    "truth": THESIS_PALETTE["neutral_dark"],
    "linear": THESIS_PALETTE["secondary"],
    "xgb": THESIS_PALETTE["primary"],
    "tft": THESIS_PALETTE["tertiary"],
}


def get_color(role: str, default: str = "#333333") -> str:
    """Return a semantic color for the given role, fallback to default."""
    return THESIS_PALETTE.get(role, default)


def get_model_color(model_name: str, default: str | None = None) -> str:
    """Return the fixed thesis color for a model/truth series."""
    key = str(model_name).strip().lower()
    fallback = default if default is not None else THESIS_PALETTE["neutral_dark"]
    return MODEL_COLOR_MAP.get(key, fallback)


def apply_geo_style() -> None:
    """Apply global plotting style for scientific thesis plots."""
    # Keep seaborn optional to avoid hard dependency in scripts.
    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid", context="paper")
    except Exception:
        pass

    # Deterministic default color cycle aligned to semantic roles.
    color_cycle = [
        THESIS_PALETTE["primary"],
        THESIS_PALETTE["secondary"],
        THESIS_PALETTE["tertiary"],
        THESIS_PALETTE["neutral_dark"],
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "figure.figsize": (12, 6),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "axes.prop_cycle": plt.cycler(color=color_cycle),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#EEEEEE",
            "grid.linestyle": "-",
            "grid.alpha": 0.35,
            "patch.edgecolor": "#333333",
            "patch.linewidth": 0.8,
            "patch.force_edgecolor": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
