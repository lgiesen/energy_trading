# Negative aFRR Activation Price Canonicalization

## Problem
`afrr_activation_price_vwap_neg` is raw market-signed and often negative, while in provider economics this corresponds to positive provider value when activated.

## New convention
For the negative activation-price target only:

`canonical = -raw`

Implemented as exact sign inversion (not `abs`).

## Pipeline location
Implemented in:
- `src/energy_trading/processing/refine_market_data.py`

Behavior:
- preserve raw as `afrr_activation_price_vwap_neg_raw`
- set canonical training column `afrr_activation_price_vwap_neg = -afrr_activation_price_vwap_neg_raw`

## Why before lag creation
Transformation occurs before feature engineering so all lag/rolling features for this target are created on canonical economic scale.

## Scope
Only affected target retraining:
- `pred_afrr_activation_price_neg`

No transformation changes for:
- negative capacity price target
- DA price
- positive activation price

## Legacy vs canonical artifacts
Postprocessing is mode-aware for `pred_afrr_activation_price_neg`:
- `target_value_mode=canonical_economic`: no sign flip, no quantile swap
- `target_value_mode=raw_signed_legacy`: sign flip + symmetric quantile swap

## HPO/log policy
- No new HPO required by default.
- Reuse existing best hyperparameters if desired.
- Existing HPO artifacts and TensorBoard logs remain unchanged.
- New training must write to new run directories.

## Regeneration and verification
- Feature generation writes `feature_generation_report.csv` in the feature output directory.
- Includes raw/canonical ranges and lag range diagnostics for negative activation target.

## Thesis wording
The negative aFRR activation price target was transformed into the economic provider-value convention before feature engineering. Consequently, the corresponding models were retrained for this target only. Forecast quantiles for this target are interpreted directly on the economic value scale and are not sign-flipped or quantile-swapped in the final benchmark and simulation.
