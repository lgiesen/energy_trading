# Energy Trading: Risk-Aware Forecasting and BESS Optimization

This repository implements an end-to-end energy trading pipeline designed for **risk-aware decision making** in volatile power markets.  
The system links probabilistic forecasting to physical battery operation and financial outcomes, enabling apples-to-apples benchmarking across model families and trading strategies.

## Project Overview & Pipeline

The project is organized into a four-stage production workflow:

1. **Data Collection & Preparation**  
   Ingest multi-source market, system, and exogenous data; clean and align it on a unified hourly timeline; build modeling-ready datasets.

2. **Model Training & Calibration**  
   Train probabilistic forecasters with tail-aware objectives, enforce quantile consistency, and calibrate prediction intervals for reliable uncertainty.

3. **BESS Simulation**  
   Convert forecasts into actionable charging/discharging and reserve bids under physical battery constraints and market rules.

4. **Evaluation & Analysis**  
   Benchmark forecast skill and simulated trading PnL in a canonical metrics framework for robust model comparison.

---

## Stage 1: Data Collection & Preparation

### Sources
The data layer integrates multiple external and local providers, including:

- **ENTSO-E Transparency Platform** for market/system fundamentals
- **Weather providers** for wind/solar and related meteorological forecasts
- **Local market operators** (e.g., Day-Ahead and balancing/aFRR related data)

### Feature Engineering
Feature construction is designed to capture both market dynamics and physical drivers:

- **Historical Market Data**
  - Past Day-Ahead prices
  - aFRR activation prices and rates
- **System Fundamentals**
  - Load forecasts
  - Scheduled generation
  - Cross-border flow signals
- **Exogenous Variables**
  - Weather forecast signals (e.g., wind/solar proxies)
  - Calendar/time encodings (hour, weekday, seasonality patterns)

The output of Stage 1 is a clean, timestamp-aligned model input layer suitable for deterministic and probabilistic model training.

---

## Stage 2: Training, Validation & Testing

### Probabilistic Model Ensemble
The forecasting stack uses a three-pillar ensemble:

- **TFT (Temporal Fusion Transformer)**
  - Joint-quantile learning for nonlinear temporal dependencies and regime shifts
- **Tail-Weighted XGBoost**
  - Gradient boosting with asymmetric/tail emphasis to improve extreme event sensitivity
- **Linear SGD**
  - Regularized quantile baseline for stable, interpretable benchmarking

### Integrity Layers
To make probabilistic outputs production-safe and comparable:

- **Monotonic Sorting**
  - Enforces ordered quantiles (prevents quantile crossing)
- **Split-Conformal Calibration**
  - Adjusts predicted quantiles to improve empirical coverage on the P01–P99 grid

This stage produces calibrated probabilistic forecasts with consistent schema across models.

---

## Stage 3: BESS Simulation (Trading Engine)

The simulation translates forecast distributions into market actions for a battery energy storage system (BESS).

### Central Rules

- **Physical Constraints**
  - Charge/discharge efficiency losses
  - Power-to-energy ratio limits
  - State-of-Charge (SoC) dynamics and operational bounds
- **Market Logic**
  - Multi-market participation (Day-Ahead arbitrage + aFRR reserve products)
- **Bidding Strategy**
  - Quantile-based limit order logic
  - Example: conservative charging from lower quantiles (e.g., P10), aggressive discharge thresholds from upper quantiles (e.g., P90)

This stage outputs time-resolved dispatch decisions and simulated economic outcomes under realistic constraints.

---

## Stage 4: Evaluation & Analysis

### Forecasting Metrics
Model quality is assessed with both global and tail-sensitive diagnostics, including:

- Tail-MAE
- Pinball Loss
- Spike detection metrics (Recall / F1)

### Simulation Metrics
Trading performance is evaluated with asset-level business KPIs:

- Total Profit (PnL)
- ROI
- Max Drawdown
- Cycles per year

### Reporting Standard
All model families are compared through canonical exports, including:

- `canonical_metrics.parquet` for unified benchmarking
- standardized prediction artifacts for downstream simulation and diagnostics

---

## Why This Architecture

This project is intentionally built as a **forecast-to-action** system:  
probabilistic market views are calibrated, transformed into physically feasible battery decisions, and evaluated on realized financial performance.  
The core design objective is not only predictive accuracy, but **robust risk-aware trading performance under uncertainty**.

## Additional Documentation

- Pipeline runbook: `docs/pipeline_runbook.md`
- Data collection guide: `docs/data_collection_guide.md`
