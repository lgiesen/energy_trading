# Model Architecture Comparison for German aFRR Forecasting (72h Horizon)

## Scope and Data Context

This comparison targets the German aFRR setting with three prediction tasks:
- activation rate (quantity-like, strongly zero-inflated),
- capacity price (Leistungspreis),
- activation/work price (Arbeitspreis).

Data characteristics:
- 338 engineered features (post-redundancy check),
- hourly time series with seasonality and autocorrelation,
- structural regime shifts and market design effects,
- zero-inflation in activation-related targets.

## Quick Comparison Table

| Model | Methodology | Pros for Power Markets | Cons / Risks | Suitability for aFRR Targets |
|---|---|---|---|---|
| XGBoost / LightGBM | Gradient-boosted decision trees minimizing residual error in sequential boosting rounds. | Strong on nonlinearity, interactions, mixed feature scales, robust to moderate collinearity, strong tabular performance. | Hyperparameter-sensitive, can overfit in small samples, less transparent than linear baselines, direct multi-horizon setup can be heavy. | **Work/Capacity price:** very strong. **Activation rate:** strong if framed with appropriate objective/transform. |
| Random Forest | Bagged ensemble of decorrelated trees with averaging. | Stable baseline, low tuning burden, captures nonlinearity, robust against noisy predictors. | Typically weaker extrapolation and peak sensitivity than boosting, less efficient on subtle tail events, larger model size. | **Capacity/work price:** moderate. **Activation rate:** moderate, good robustness baseline. |
| Deep Neural Networks (MLP) | Multi-layer nonlinear function approximation trained via gradient descent. | Flexible representation learning, can model complex cross-feature structure. | Data-hungry, sensitive to scaling/regularization, harder reproducibility/interpretability, prone to unstable training on small-to-medium tabular datasets. | **All targets:** potentially good, but high implementation/training risk in this dataset regime. |
| Lasso-AR (Econometric Baseline) | Linear autoregressive model with L1 regularization for sparse coefficient selection. | High interpretability, strong baseline discipline, explicit shrinkage under high-dimensional feature sets, easy diagnostics. | Linear structure limits nonlinear effects and interaction capture; can underfit spikes/regime transitions. | **Activation rate/price:** useful baseline, especially for benchmarking and coefficient-level interpretation. |
| Hurdle Models | Two-stage modeling: (1) event/non-event probability, (2) positive outcome magnitude conditional on event. | Natural fit for zero-inflated targets, separates activation occurrence from magnitude dynamics, economically interpretable decomposition. | More pipeline complexity, calibration burden across both stages, error propagation between stages. | **Activation rate:** excellent fit. **Capacity/work price:** useful when conditional-on-activation behavior matters. |

## Model-Specific Assessment

## 1) XGBoost / LightGBM
- Methodik: additive tree boosting with stage-wise residual correction.
- Pros:
  - captures nonlinear threshold effects common in balancing markets,
  - handles heterogeneous feature families (prices, outages, weather errors, regime flags),
  - usually top performer on structured tabular market data.
- Contras:
  - needs strict temporal CV (purged walk-forward) to avoid optimistic bias,
  - sensitive to leakage in lag/feature construction,
  - marginal interpretability (requires SHAP/partial dependence tooling).
- aFRR suitability:
  - excellent for `Arbeitspreis` and `Leistungspreis`,
  - strong for activation-rate regression/classification with proper loss design.

## 2) Random Forest
- Methodik: bootstrap aggregation of randomized trees.
- Pros:
  - robust and easy to tune,
  - good benchmark for nonlinear relationships without heavy optimization.
- Contras:
  - can smooth away rare extreme price events,
  - often inferior to boosting for sharp market transitions.
- aFRR suitability:
  - solid fallback benchmark for all three targets,
  - less suited as final best-performance model.

## 3) Deep Neural Networks (MLP)
- Methodik: dense feed-forward nonlinear network.
- Pros:
  - flexible universal approximator for high-order interactions.
- Contras:
  - higher variance and tuning complexity,
  - less attractive when sample size is modest relative to feature complexity,
  - weaker interpretability for thesis defense compared with sparse/trees.
- aFRR suitability:
  - secondary candidate; useful if regularized and benchmarked rigorously against GBDT.

## 4) Lasso-AR
- Methodik: linear ARX-style setup with L1 penalty.
- Pros:
  - transparent coefficient-level interpretation,
  - natural baseline for publication-grade comparison,
  - useful to test whether complex models deliver real incremental value.
- Contras:
  - limited nonlinear capture,
  - may miss asymmetric regime effects and tail behavior.
- aFRR suitability:
  - excellent baseline for all targets, especially as interpretability anchor.

## 5) Hurdle Models
- Methodik:
  - Stage 1: classify whether activation occurs (`y > 0` vs `y = 0` or signed variants),
  - Stage 2: regress positive magnitude conditional on activation.
- Pros:
  - directly addresses zero-inflation in activation processes,
  - clearer economic interpretation (event probability vs event size).
- Contras:
  - requires careful probability calibration and recombination logic,
  - more engineering and validation overhead.
- aFRR suitability:
  - primary method for activation-rate/activation-volume style targets,
  - can be extended to conditional price modeling during activation hours.

## Data-Structure Fit Summary

| Data Property | Implication | Best-Fit Models |
|---|---|---|
| 338 tabular engineered features | high-dimensional nonlinear tabular problem | XGBoost/LightGBM, Lasso-AR baseline |
| Time dependence (seasonality, autocorrelation) | strict temporal split and lag discipline required | all models with purged CV; GBDT benefits strongly |
| Zero-inflation (activation) | event vs magnitude separation is beneficial | Hurdle models (possibly with GBDT in each stage) |
| Regime shifts / structural breaks | robust validation and adaptive nonlinear models needed | GBDT + regime features; Lasso-AR as stability baseline |

## Recommended Final Lineup (Master Thesis)

1. **LightGBM or XGBoost (primary performance model)**  
   Main model for price and quantity-related forecasting on tabular features.

2. **Hurdle framework (primary for zero-inflated activation targets)**  
   Preferably with tree-based learners in both stages for consistency.

3. **Lasso-AR (econometric benchmark baseline)**  
   Ensures interpretability and a defensible reference model.

## Why These Complement Each Other

- **Performance frontier:** GBDT captures nonlinear interactions and regime effects.
- **Distributional realism:** Hurdle structure handles sparse/zero-heavy activation behavior.
- **Interpretability baseline:** Lasso-AR provides sparse linear diagnostics and a robust benchmark.

This lineup balances predictive quality, methodological rigor, and thesis defensibility.
