# Simulation Methodology

## A. Timing and Causality
- Optimization uses forecast columns (`pred_*`).
- Settlement uses realized columns (`true_*`).
- DA role `mid` uses `p50` for DA central forecast; DA limit quantiles stay independent.
- aFRR quantile ranges define allowed bid bins for BCM and BEM.

## B. Battery State
For hour `t` with duration `Δt`:

- `SoC_{t+1} = SoC_t + η_in * E_in - E_out/η_out - E_aux`
- `E_in = (P_ch + P_id,ch) * Δt + E_act,neg,internal`
- `E_out = (P_dis + P_id,dis) * Δt + E_act,pos,internal`

Constraints:
- `0 <= P_ch, P_dis <= P_max`
- `SoC_min <= SoC_t <= SoC_max` (softened by explicit slack penalties)
- Reserve/BEM deliverability headroom constraints use configured activation headroom hours.

## C. aFRR Quantile Bins
Canonical ordered bins:
`[p01,p05,p10,p30,p50,p70,p90,p95,p99]`.

Range expansion examples:
- `p30-p70 -> {p30,p50,p70}`
- `p10-p90 -> {p10,p30,p50,p70,p90}`

Input policy:
- Active bin columns, including `p50`, are mandatory in strict thesis mode.
- `p50` can only be materialized from `predicted_value` via explicit compatibility option (`--allow-p50-from-predicted-value`), and is flagged as `materialized_p50_from_predicted_value=1`.
- No silent zero-fill and no synthetic non-`p50` quantile creation is allowed.

## D. Optimization Variables
Per hour `t` and allowed bin `q`:
- BCM reserve: `r_pos[t,q]`, `r_neg[t,q]`
- BEM-only: `bem_pos[t,q]`, `bem_neg[t,q]`

Aggregates:
- `R_pos[t] = Σ_q r_pos[t,q]`
- `R_neg[t] = Σ_q r_neg[t,q]`
- `BEM_pos[t] = Σ_q bem_pos[t,q]`
- `BEM_neg[t] = Σ_q bem_neg[t,q]`

## E. Objective (per-bin EV)
Maximization is represented as MILP minimization over negative coefficients.

For BCM positive bin `q`:
- `EV_BCM_pos[t,q] = p_acc[q] * CapPrice_pos[t,q]`
- `+ p_acc[q]*ActRate_pos[t,q]* (ActPrice_pos[t,q] - c_tx - c_deg_dis)`
- `- OfferCost`
- `- AuxCost_BCM_pos[t,q]`

For BCM negative bin `q`:
- `EV_BCM_neg[t,q] = p_acc[q] * CapPrice_neg[t,q]`
- `+ p_acc[q]*ActRate_neg[t,q]* (-ActPrice_neg[t,q] - c_tx - c_deg_ch)`
- `- OfferCost`
- `- AuxCost_BCM_neg[t,q]`

For BEM positive bin `q`:
- `EV_BEM_pos[t,q] = p_exec[q]*ActRate_pos[t,q]*(ActPrice_pos[t,q] - c_tx - c_deg_dis)`
- `- AuxCost_BEM_pos[t,q]`

For BEM negative bin `q`:
- `EV_BEM_neg[t,q] = p_exec[q]*ActRate_neg[t,q]*(-ActPrice_neg[t,q] - c_tx - c_deg_ch)`
- `- AuxCost_BEM_neg[t,q]`

Notes:
- Current proxy uses `p_acc[q] = 1 - q_level` (modeling proxy, not empirically calibrated acceptance probability).
- BEM uses side-specific per-bin execution proxies (`p_exec_pos[q]`, `p_exec_neg[q]`), currently mapped to `p_acc_pos[q]` and `p_acc_neg[q]`.

## F. Auxiliary Cost Model
State-dependent duties:
- `aux_standby_mw = aux_peak_mw * duty_standby`
- `aux_trading_mw = aux_peak_mw * duty_trading`
- `aux_active_mw = aux_peak_mw * duty_afrr_active`

BCM expected auxiliary cost per MW bin:
- `AuxCost_BCM = price_aux * h_eq * (p_acc*act_rate*aux_active_mw + (p_acc-p_acc*act_rate)*aux_standby_mw)`

BEM expected auxiliary cost per MW bin:
- `AuxCost_BEM = price_aux * h_eq * (p_exec*act_rate*aux_active_mw)`

where `h_eq = duration / P_max`.

## G. Settlement and PnL
- Realized settlement always uses realized market values (`true_*`).
- Forecast quantiles affect dispatch decisions only.
- Per-bin selected MW are aggregated to settlement-facing totals:
  - `submitted_reserve_pos_mw = Σ_q r_pos[q]`
  - `submitted_reserve_neg_mw = Σ_q r_neg[q]`
  - `submitted_bem_pos_mw = Σ_q bem_pos[q]`
  - `submitted_bem_neg_mw = Σ_q bem_neg[q]`

## H. Assumptions and Limits
- `p_acc = 1-q` is a quantile-based proxy.
- Quantile-bin optimization approximates distribution-aware bidding, not a full stochastic program.
- Missing critical per-bin quantile inputs fail fast in strict mode.

## I. BCM SoC Drift (Bin-Consistent)
The BCM contribution to expected SoC drift is computed per bin with the same `q`
activation rates used in EV:

- `exp_pos[t,q] = p_acc_pos[t,q] * ActRate_pos[t,q]`
- `exp_neg[t,q] = p_acc_neg[t,q] * ActRate_neg[t,q]`

and enters SoC dynamics with efficiency-adjusted energy terms.

## J. Strategy Modes
Supported strategy modes:

- `multi`: DA + BCM + BEM enabled, `id_mode=economic` (technical repair also possible).
- `da_only`: DA enabled, BCM/BEM disabled, default `id_mode=none`.
- `afrr_only`: DA disabled, BCM + BEM enabled, default `id_mode=technical_repair`.
- `bcm_only`: DA disabled, BCM enabled (including activation settlement from awarded reserve), BEM-only disabled, default `id_mode=technical_repair`.
- `bem_only`: DA disabled, BCM disabled, standalone BEM enabled, default `id_mode=technical_repair`.

Notes:
- `bcm_only` is not capacity-only accounting; awarded capacity can still trigger activation settlement.
- `bem_only` excludes BCM capacity awards and therefore excludes capacity revenue.
- Baseline contamination guard: economic ID is blocked for non-`multi` strategies unless explicitly overridden for robustness runs.
- Disabled markets are hard-gated in optimizer bounds and validated in settlement outputs.
## BCM vs BEM Product Granularity

- BCM/aFRR capacity is modeled as a fixed 4-hour product in German market local time (`Europe/Berlin`): `00-04`, `04-08`, `08-12`, `12-16`, `16-20`, `20-24`.
- BEM/aFRR balancing energy remains hourly.
- `afrr_only` includes both:
  - block-based BCM capacity commitments
  - hourly BEM bids/activation
- Partial first/last observed blocks are flagged via:
  - `partial_bcm_block_at_start`
  - `partial_bcm_block_at_end`
- Strict validity enforces BCM block consistency; non-constant BCM submitted/awarded MW inside a block triggers `bcm_block_consistency`.
