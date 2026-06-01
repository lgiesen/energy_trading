# ID Recourse Methodology

This thesis implementation is **not** a full DA-ID-aFRR co-optimizer.

- DA, BCM and BEM decisions come from optimization / market-clearing logic.
- ID is implemented as a **rule-based, price-taking recourse layer** for short-term SoC/headroom correction.
- ID prices are synthetic exogenous proxies:
  - `id_buy_price = min(id_price_cap, da_price + id_spread)`
  - `id_sell_price = max(id_price_floor, da_price - id_spread)`

## Policy Modes

- `id_recourse_mode=common`
  - Technical ID recourse allowed for all strategies (`multi`, `da_only`, `afrr_only`, `bcm_only`, `bem_only`).
- `id_recourse_mode=disabled`
  - ID recourse fully disabled (no pending ID and no settled ID energy/PnL).
- `id_recourse_mode=afrr_obligation_only`
  - ID recourse disabled for `da_only`, enabled for aFRR-obligation strategies.

## Why no ID-only baseline

No standalone `ID_only` strategy is included in this thesis version. ID is treated as operational recourse, not as an independently optimized trading strategy.

## Reporting

ID is reported explicitly in hourly and summary outputs (`id_recourse_mode`, `id_allowed`, ID MWh, ID prices, ID PnL, reason codes). Total PnL includes ID when enabled and also reports `realized_total_pnl_excluding_id_eur` for diagnostics.
