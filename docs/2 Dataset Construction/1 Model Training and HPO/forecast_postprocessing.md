# Forecast Postprocessing

## Why this exists
Only legacy negative aFRR activation-price artifacts may be raw market-signed values, while optimizer/settlement should use canonical economic provider values.

## Conventions
- Raw signed: market sign can be negative on negative-side targets.
- Canonical economic: provider-value sign (non-negative where defined as value).

For monotone transform `y = -x`:
`Q_y(q) = -Q_x(1-q)`.

## Affected targets
- Mode-aware: `pred_afrr_activation_price_neg`
  - `raw_signed_legacy`: sign flip + quantile swap
  - `canonical_economic`: no sign flip, no quantile swap
- `pred_afrr_capacity_price_neg`: **not** sign-flipped or quantile-swapped

## Quantile transform
For `raw_signed_legacy` mode, symmetric quantiles are swapped and sign-flipped:
- `p10 <- -p90`
- `p30 <- -p70`
- `p50 <- -p50`
- `p70 <- -p30`
- `p90 <- -p10`
(and optional `p01/p99`, `p05/p95` pairs)

## Clipping rules
- `pred_afrr_activation_rate_pos`, `pred_afrr_activation_rate_neg`: clipped to `[0, 1]`
- Capacity values (pos/neg): lower clipped at `0` (no sign inversion)
- Canonical neg-side provider values: lower clipped at `0`
- DA price: no clipping

## Where it is applied
- ML benchmark loader (predictions and truth before scoring)
- Simulation/backtest forecast loader (merged table and long warehouse) before quantile selection, bidding, clearing, and settlement

## Verification
- Import wiring:
  - `rg -n "forecast_postprocessing|canonicalize_prediction_frame|canonicalize_truth_series" src scripts`
- Unit tests:
  - `./.venv/bin/python -m pytest -q tests/test_forecast_postprocessing.py`
- Self-test:
  - `./.venv/bin/python scripts/check_forecast_postprocessing.py --self-test`

## Thesis wording
Forecasts for negative-side aFRR price targets were transformed into the economic provider-value convention used by the optimizer. Since the transformation `y=-x` reverses quantile order, symmetric quantiles were swapped before benchmark evaluation and simulation.
