"""Central plotting style for thesis figures (GeoDataViz-based).

Plotting best practices for thesis figures:
- Use alpha=0.6 or 0.7 for "solar" and "wind" when using plt.fill_between()
  to keep the background light.
- Use linewidth=2.0 or 2.5 for "actual_profit" and "oracle_profit" to make
  them stand out.
- Use linestyle="--" for "oracle_profit" and linestyle=":" for
  "benchmark_profit".

Reuse this module across Python scripts and notebooks:

    from energy_trading.visualization.style import apply_geo_style, ROLE_COLORS
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

# Functional color identities for consistent semantics across plots.
ROLE_COLORS: Dict[str, str] = {
    "wind": "#089099",
    "solar": "#FCDE9C",
    "load": "#333333",
    "actual_profit": "#045275",
    "oracle_profit": "#7C1D6F",
    "benchmark_profit": "#999999",
}


def get_color(role: str, default: str = "#999999") -> str:
    """Return a semantic color for the given role, fallback to default."""
    return ROLE_COLORS.get(role, default)


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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
