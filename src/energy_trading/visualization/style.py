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
    "primary": "#226E9C",
    "secondary": "#d9b98d",
    "tertiary": "#7C1D6F",
    "neutral_dark": "#333333",
}


def get_color(role: str, default: str = "#333333") -> str:
    """Return a semantic color for the given role, fallback to default."""
    return THESIS_PALETTE.get(role, default)


def apply_geo_style() -> None:
    """Apply global matplotlib style for scientific thesis plots."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.figsize": (12, 6),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#EEEEEE",
            "grid.linestyle": "--",
            "patch.edgecolor": "#333333",
            "patch.linewidth": 0.8,
            "patch.force_edgecolor": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
