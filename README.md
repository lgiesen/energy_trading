# Energy Trading: Probabilistic Forecasting and BESS Trading Simulation

This repository implements an end-to-end pipeline for probabilistic forecasting and battery energy storage system (BESS) trading simulation in German electricity markets. The project links forecast evaluation with realized trading performance to analyze how model choice, quantile policy and market participation design affect BESS profitability.

The study focuses on the Day-Ahead (DA), Intraday (ID), Balancing Capacity Market (BCM) and Balancing Energy Market (BEM). The aFRR market is modeled through its two linked components. BCM represents capacity procurement, while BEM represents activation energy.

## Project Scope

The repository supports the empirical analysis for a master thesis on multi-market BESS trading under forecast uncertainty. It combines three elements:

1. **Probabilistic forecasting** of DA prices, aFRR capacity prices, aFRR activation prices and aFRR activation rates.
2. **Rolling BESS trading simulation** across DA, ID, BCM and BEM under physical battery constraints and market rules.
3. **Economic evaluation** of model, quantile and market-strategy performance based on realized net profit.

The core objective is not only to minimize forecast error, but to assess whether improvements in probabilistic forecast performance translate into higher realized trading value.

## Pipeline Overview

The workflow consists of four stages.

### 1. Data Preparation

The data layer collects, cleans and aligns market, system and exogenous inputs on a common hourly timeline. The resulting feature dataset is used for multi-horizon probabilistic forecasting.

Main data categories include:

- DA market prices
- aFRR capacity prices by direction
- aFRR activation prices by direction
- aFRR activation rates by direction
- load, renewable generation and residual-load related features
- calendar and time features
- additional market and system variables where available

### 2. Forecasting Models

The forecasting benchmark compares three model classes with different assumptions about the data-generating process:

- **TFT**: Temporal Fusion Transformer for sequential DL forecasting.
- **XGB**: Gradient-boosted decision trees for nonlinear tabular feature interactions.
- **RLQR**: Regularized Linear Quantile Regression as a transparent linear probabilistic baseline.

At each forecast origin, the models generate conditional quantile forecasts for lead hours \(h = 1,\dots,48\). The exported quantile levels are:

`p01`, `p05`, `p10`, `p30`, `p50`, `p70`, `p90`, `p95`, `p99`

Before export, raw quantile predictions are sorted row-wise to enforce monotonicity and prevent quantile crossing. This creates internally consistent prediction intervals but does not improve the underlying calibration of the forecast distribution.

### 3. Trading Simulation Backtest

The simulation converts probabilistic forecasts into sequential market decisions for a BESS. The battery participates in DA, ID, BCM and BEM depending on the selected strategy.

Market treatment:

- **DA and ID** are modeled as price-taking energy transactions. Submitted volumes are assumed to clear and are settled at realized market prices.
- **BCM and BEM** are modeled as limit-price decisions. The selected forecast quantile determines the bid price and realized market outcomes determine clearing.
- **ID** is used as a short-term recourse mechanism for SoC adjustments where feasible.

The simulation enforces battery constraints, including SoC limits, charge and discharge limits, reserve headroom, efficiency losses, auxiliary consumption and terminal SoC requirements.

### 4. Evaluation and Analysis

Forecast performance and trading performance are evaluated jointly.

Forecast metrics include:

- Mean Pinball Loss (MPL)
- MAE of the median forecast
- MBE of the median forecast
- Winkler score
- empirical quantile calibration
- prediction interval coverage
- lead-time and decision-window diagnostics
- tail and spike regime diagnostics

Trading metrics include:

- realized net profit
- annualized net profit
- revenue by market
- cost components
- submitted and cleared bid volumes
- bid-clearing ratios
- throughput and equivalent cycles
- SoC behavior
- fallback and feasibility diagnostics

## Market Strategies

The simulation supports multi-market and single-market strategy variants.

Main strategy groups:

- **multi**: DA, BCM, BEM and ID recourse
- **da_only**: DA trading with ID recourse where applicable
- **bcm_only**: BCM participation
- **bem_only**: BEM participation
- **afrr_only**: BCM and BEM without DA participation

This structure enables comparison between revenue stacking and isolated market participation.

## Important Reproducibility Note

The reported simulation results were generated with two code states. All simulation runs except the DA-only strategy were generated with commit:

```text
6a4ac6637b5ff5c9af2386b5728c6d1dc54519ba
```

Afterwards, DA lockbook handling was corrected and ID recourse logic was refined to improve the DA-only simulation. The DA-only results are therefore based on the revised implementation, while the remaining strategy results remain reproducible from the documented commit. Comparisons involving DA-only results need to be interpreted with this version difference in mind.

## Repository Structure

Typical project components include:

```text
data/                         # input and processed data, not necessarily tracked in Git
artifacts/                    # model outputs, forecasts and simulation runs
src/energy_trading/           # core package
scripts/                      # training, export, simulation and validation scripts
tests/                        # unit and market-semantics tests
docs/                         # additional documentation
```

Large data and model artifacts may be stored outside Git depending on the execution environment.

## Key Outputs

The pipeline produces:

- standardized forecast exports for all models and targets
- forecast benchmark tables and figures
- rolling BESS simulation outputs
- strategy-level profitability tables
- market-specific revenue and cost decompositions
- validity and fallback diagnostics

These outputs are used to answer the thesis research questions on probabilistic forecasting performance, realized economic value and multi-market BESS strategy design.

## Methodological Caveats

The simulation is a backtest and does not represent guaranteed real-world trading profits. Several assumptions simplify real market participation, including price-taking treatment for DA and ID, deterministic settlement based on realized prices and simplified project-cost assumptions. In addition, some simulation runs contain fallback events or feasibility diagnostics. The results are therefore most informative for comparing model-policy behavior within the implemented backtest, while absolute profit levels need to be interpreted cautiously.

## Additional Documentation

- Pipeline runbook: `docs/pipeline_runbook.md`
- Data collection guide: `docs/data_collection_guide.md`
