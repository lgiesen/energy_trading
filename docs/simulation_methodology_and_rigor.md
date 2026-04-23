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
3. **Rolling T-25**: aFRR Activation Gate (energy bid update before delivery)

This chronological structure is implemented as a state machine that carries the
executed physical state of charge ($SoC^{\text{exec}}$) into subsequent decisions.

### Event-Driven Replanning

If a gate outcome materially invalidates the active plan (for example reserve
capacity rejection at 09:00, or downstream execution mismatch causing a strong
SoC shock), the system triggers immediate re-optimization from the current
snapshot state. This hard restart logic prevents stranded schedules and restores
causal consistency before the next gate decisions are finalized.

---

## 3. Optimization & Physical Constraints (The MILP Core)

The dispatch planner is formulated as a mixed-integer linear program (MILP).

### Objective Function

At high level, the optimizer maximizes expected net economic value:

- DA margin
- aFRR capacity remuneration
- aFRR activation value
- minus transaction and degradation costs

### Asymmetric Bidding

Positive and negative aFRR reserves are modeled as independent decision
variables:
$r_t^{+} \;\text{and}\; r_t^{-} \quad \text{(strictly decoupled)}$

This allows economically and physically asymmetric reserve positioning.

### Worst-Case Feasibility Buffers

To ensure physical deliverability under severe activation, the MILP enforces
reserve feasibility buffers with efficiency-adjusted SoC constraints:

Upward reserve buffer:
$SoC_t - \frac{r_t^{+}\Delta t}{\eta_{out}} \ge SoC_{min}$

Downward reserve buffer:
$SoC_t + r_t^{-}\Delta t\,\eta_{in} \le SoC_{max}$

These constraints guarantee that reserve awards remain physically supportable
even under continuous 100% activation assumptions.

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
\(-9999\) where required) to ensure the perfect-foresight schedule is executable
through the same clearing pipeline. Settlement remains pay-as-cleared on true
market prices. This defines a strict mathematical upper bound.

---

## 6. Financial Settlement & KPIs

### Full PnL Identity

Per interval \(t\):
$\Pi_t = R^{DA}_t - C^{DA}_t + R^{cap}_t + R^{act}_t - C^{deg}_t - C^{tx}_t - C^{pen}_t$

Horizon total:
$\Pi^{raw} = \sum_{t=1}^{T} \Pi_t$

Terminal mark-to-market inventory value (liquidatable grid value):
$V^{term} = \max\!\left(0, (SoC_T - SoC_{min})\right)\cdot \eta_{out}\cdot P_{DA,T}$

Reported total:
$\Pi^{total} = \Pi^{raw} + V^{term}$

### Capacity Revenue Time-Scaling

If capacity prices are interpreted in €/MW/h, capacity remuneration is scaled by
interval length \(\Delta t\):
$R^{cap}_t = \left(Q^{cap,+}_{t,del}\,P^{cap,+}_t + Q^{cap,-}_{t,del}\,P^{cap,-}_t\right)\Delta t$

### No-Double-Counting Rule

Activation revenue is booked on delivered activation energy, while replenishment
economics are represented by subsequent DA trading. This avoids synthetic
double-counting of activation-related energy replacement.

### Cashflow for Liquidity Metrics (Non-cash Degradation Excluded)

For working-capital analysis, the simulation uses net cashflow excluding
non-cash degradation accounting:
$CF^{net}_t = R^{DA}_t - C^{DA}_t + R^{cap}_t + R^{act}_t - C^{tx}_t - C^{pen}_t$
The cumulative cash trajectory is:
$Cash_t = Cash_0 + \sum_{\tau=1}^{t} CF^{net}_\tau$
and peak capital requirement is:
$\text{max\_capital\_required} = \max\left(0,\,-\min_t\left(Cash_0 + \sum_{\tau=1}^{t} CF^{net}_\tau\right)\right)$

### Operational KPIs

Equivalent Full Cycles (EFC):
$\text{EFC} = \frac{\left(\sum_t E_{sent,t} + \sum_t E^{+}_{act,t}\right)/\eta_{out}}{E_{cap}}$

Return on Capital Requirement:
$ROI_{cap} = \begin{cases} \dfrac{\Pi^{total}_{realized}}{\text{max\_capital\_required}}, & \text{if } \text{max\_capital\_required} > 0 \\ \text{undefined}, & \text{otherwise} \end{cases}$

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
