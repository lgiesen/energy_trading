# Simulation Methodology and Rigor

## 1. System Objective & Scope

This simulation framework evaluates stacked battery trading across Day-Ahead (DA)
and aFRR markets under sequential market gates and rolling-horizon decision
updates. The setup models a single battery asset with explicit physical
constraints and ex-post financial settlement.

Scope:

- Market stack: DA energy, aFRR capacity, aFRR activation.
- Decision mode: rolling-horizon optimization with repeated replanning.
- Asset model: single battery with power, energy, efficiency, and SoC limits.
- Evaluation design: four baseline paths (Predicted, Realized, Naive, Oracle).

---

## 2. Chronology & Market Dynamics (The State Machine)

The simulation enforces chronological market sequencing:

1. **D-1 09:00 CET**: aFRR Capacity Gate
2. **D-1 12:00 CET**: Day-Ahead Gate
3. **Rolling T-25 min**: aFRR Activation Gate (energy bid update before delivery)

This chronological structure is implemented as a state machine that carries the
executed physical state of charge ($SoC_t$) into subsequent decisions.

At the capacity gate, awards are propagated as lockbook obligations for next-day
delivery intervals (implemented in 4-hour auction blocks and then mapped to
hourly dispatch/settlement steps).

### Event-Driven Replanning

If a gate outcome materially invalidates the active plan (for example reserve
capacity rejection at 09:00 am, or downstream execution mismatch causing a strong
SoC shock), the system triggers immediate re-optimization from the current
snapshot state. This restart logic prevents stranded schedules and restores
causal consistency before the next gate decisions are finalized.

---

## 3. Optimization & Physical Constraints (The MILP Core)

The dispatch planner is formulated as a mixed-integer linear program (MILP).

### Objective Function

The MILP objective is **total expected profit over the horizon**, not
activation rate or any single remuneration component in isolation.

For each interval $t$, and bid-price bin $j$, the optimizer solves:

$\max \sum_{t=1}^{T} \hat{\Pi}_t$

with

\[
\hat{\Pi}_t =
\underbrace{P^{DA}_t\!\left(q^{sell}_{t}-q^{buy}_{t}\right)}_{\text{DA gross margin}}
+\underbrace{\sum_j p^{cap,+}_{acc,j,t}\,P^{cap,+}_t\,r^{+}_{j,t}
           +\sum_j p^{cap,-}_{acc,j,t}\,P^{cap,-}_t\,r^{-}_{j,t}}_{\text{expected capacity remuneration}}
+\underbrace{\sum_j p^{cap,+}_{acc,j,t}\,\hat{\alpha}^{+}_t\,\tilde{P}^{act,+}_{j,t}\,r^{+}_{j,t}
           -\sum_j p^{cap,-}_{acc,j,t}\,\hat{\alpha}^{-}_t\,\tilde{P}^{act,-}_{j,t}\,r^{-}_{j,t}}_{\text{expected activation value}}
-\underbrace{C^{tx}_t(q^{buy}_{t}+q^{sell}_{t})}_{\text{transaction cost}}
-\underbrace{C^{deg}_t}_{\text{degradation cost}}
\]

Interpretation:

- $p^{cap,\pm}_{acc,j,t}$ are **capacity acceptance probabilities by bid bin**.
- $\hat{\alpha}^{+}_t,\hat{\alpha}^{-}_t$ are **predicted activation rates**
  used for expected activated volume conditional on capacity award.
- $\tilde{P}^{act,\pm}_{j,t}$ are bid/settlement-side activation-price terms
  used in expected activation remuneration.
- The optimizer compares all feasible actions per hour (DA, aFRR, or no trade)
  and picks the decision vector that maximizes the **sum of expected net profit**
  under battery and market constraints.

### Asymmetric Bidding

Positive and negative aFRR reserves are modeled as independent decision
variables:
$r_t^{+} \;\text{and}\; r_t^{-} \quad \text{(strictly decoupled)}$

This allows economically and physically asymmetric reserve positioning.

### Capacity Acceptance vs Activation (Decoupled)

Capacity acceptance and activation are modeled as distinct processes:

- Capacity acceptance uses a CDF bridge from **capacity-price quantiles**
  (when available), yielding $p^{cap}_{acc,j,t}$ per bid bin.
- Activation rate forecasts ($\hat{\alpha}^{\pm}_t$ and quantiles) are used for
  expected activation volume/value and SoC drift, not to determine capacity
  award probabilities.

Fallback policy when quantile-derived capacity CDF is unavailable:

- scalar fallback acceptance is applied **only** to the most competitive
  bid bin (price-taker bin),
- all higher bins receive probability 0.

This prevents the optimizer from exploiting high-price bins with unsupported
uniform acceptance assumptions.

### p90 SoC Safety Buffers (Chance Constraints)

To ensure physical deliverability under stressed activation, the MILP keeps
expected-value SoC dynamics but adds p90 safety constraints:

Upward reserve safety:
\[
SoC_t - \sum_j\left(p^{cap,+}_{acc,j,t}\,\alpha^{+}_{p90,t}\,r^{+}_{j,t}\,\frac{\Delta t}{\eta_{out}}\right)\ge SoC_{min}
\]

Downward reserve safety:
\[
SoC_t + \sum_j\left(p^{cap,-}_{acc,j,t}\,\alpha^{-}_{p90,t}\,r^{-}_{j,t}\,\Delta t\,\eta_{in}\right)\le SoC_{max}
\]

SoC trajectories are therefore optimized on expected activation rates while
remaining feasible under high (p90) activation stress scenarios.

---

## 4. Bidding Strategy & Market Clearing

### DA Bidding Strategy: Price Taker vs Limit

DA bids are derived from MILP schedule volumes and classified by intent:

- **Price Taker** for hedge-critical volume (execution certainty preferred).
- **Limit** for discretionary DA arbitrage.

### State-Aware Defensive Bidding (T-25)

At the activation gate, the system evaluates current physical feasibility before
finalizing aFRR energy bid prices. If physical delivery risk is critical, it
defensively prices itself out of merit order:

For upward activation infeasibility:
$p^{act,+}_{bid} = +9999$

For downward activation infeasibility:
$p^{act,-}_{bid} = -9999$

This mechanism reduces non-delivery exposure and protects against imbalance
penalties when SoC/headroom becomes critical.

---

## 5. Evaluation Baselines (The 4 Paths)

### Predicted

Ex-ante economic view using forecast-side settlement inputs (without strict
market-clearing rejection filters in the path definition).

### Realized

Ex-post economic view: ML-driven plan routed through strict market clearing and
settled on realized prices/rates with executed quantities.

### Naive

Benchmark strategy based on 24-hour lag forecasts:
$\hat{y}^{naive}_t = y_{t-24}$

### Oracle

Perfect-foresight benchmark using realized values as optimization inputs. Oracle
uses **Oracle-Aware bidding** (extreme guaranteed-clearing bid levels, e.g.
$-9999$ where required) to ensure the perfect-foresight schedule is executable
through the same clearing pipeline. Settlement remains pay-as-cleared on true
market prices. This defines a strict mathematical upper bound.

---

## 6. Financial Settlement & KPIs

### Full PnL Identity

Per interval $t$:
$\Pi_t = R^{DA}_t - C^{DA}_t + R^{cap}_t + R^{act}_t - C^{deg}_t - C^{tx}_t - C^{pen}_t$

Horizon total:
$\Pi^{raw} = \sum_{t=1}^{T} \Pi_t$

Terminal mark-to-market inventory value (liquidatable grid value):
$V_{term} = \max\!\left(0, (SoC_T - SoC_{min})\right)\cdot \eta_{out}\cdot P_{DA,T}$

Reported total:
$\Pi_{total} = \Pi^{raw} + V_{term}$

### Capacity Revenue Time-Scaling

If capacity prices are interpreted in €/MW/h, capacity remuneration is scaled by
interval length $\Delta t$:
$R^{cap}_t = \left(Q^{cap,+}_{t,del}\,P^{cap,+}_t + Q^{cap,-}_{t,del}\,P^{cap,-}_t\right)\Delta t$

Capacity non-delivery penalties include an explicit 2.0 multiplier and a
penalty-price floor to avoid zero-price loopholes:
$C^{cap,pen}_t =
2.0\cdot\left(
M^{cap,+}_t\cdot\max(P^{cap,+}_t, 10.0) +
M^{cap,-}_t\cdot\max(P^{cap,-}_t, 10.0)
\right)\Delta t$
where $M^{cap,+}_t$ and $M^{cap,-}_t$ are missed positive/negative capacity MW.

### No-Double-Counting Rule

Activation revenue is booked on delivered activation energy, while replenishment
economics are represented by subsequent DA trading. This avoids synthetic
double-counting of activation-related energy replacement.

### Cashflow for Liquidity Metrics (Non-cash Degradation Excluded)

For working-capital analysis, the simulation uses net cashflow excluding
non-cash degradation accounting:
$CF_{net,t} = R^{DA}_t - C^{DA}_t + R^{cap}_t + R^{act}_t - C^{tx}_t - C^{pen}_t$
The cumulative cash trajectory is:
$Cash_t = Cash_0 + \sum_{\tau=1}^{t} CF_{net,\tau}$
and peak capital requirement is:
$\text{max\_cap\_req} = \max\left(0,\,-\min_t\left(Cash_0 + \sum_{\tau=1}^{t} CF_{net,\tau}\right)\right)$

### Operational KPIs

Equivalent Full Cycles (EFC):
$\text{EFC} = \frac{\left(\sum_t E_{sent,t} + \sum_t E_{act,t}^{+}\right)/\eta_{out}}{E_{cap}}$

Return on Capital Requirement:
$ROI_{cap} = \dfrac{\Pi_{total}}{\max(1.0,\text{max\_cap\_req})}$

### 6.5 Nomenclature Alignment & Code Mapping

To ensure traceability between the simulation results and the thesis text, the
following mapping is used:

- $\Pi_{total}$ : `realized_total_pnl_eur`
- $\text{max\_cap\_req}$ : `max_capital_required_eur`
- EFC : `total_equivalent_full_cycles`
- $SoC_t$ : `executed_soc_mwh`

Note: While code variables use descriptive underscores for JSON serialization,
this document and the main thesis text strictly adhere to the LaTeX notation
defined in the Appendix.

---

## 7. Reproducibility & Output Artifacts

### Run Logging Protocol

For each reported run, store:

- run command
- run identifier
- configuration snapshot
- git commit hash
- input data references

### Required Artifacts

- hourly consolidated output
- planned ledger
- executed ledger
- realized ledger
- summary JSON
- diagnostics and state-machine audit outputs

### Consistency Checks

- realized component balance check
- oracle component balance check
- oracle upper-bound consistency flag
- final SoC feasibility checks
