"""Compatibility import for thesis evaluation plotting style.

The central style implementation lives in ``energy_trading.visualization.style``.
Evaluation scripts import through this module so thesis benchmark code has a
stable evaluation-local style path without duplicating palette definitions.
"""

from energy_trading.visualization.style import (  # noqa: F401
    BACKTEST_LINE_STYLES,
    BIAS_DIVERGING,
    GEO_DIVERGING,
    GEO_SEQUENTIAL_BLUE,
    MARKET_COLOR_MAP,
    MODEL_COLOR_MAP,
    THESIS_PALETTE,
    apply_geo_style,
    get_backtest_line_style,
    get_bias_diverging_cmap,
    get_color,
    get_geo_sequential_blue_cmap,
    get_model_color,
    thesis_titlecase,
)
