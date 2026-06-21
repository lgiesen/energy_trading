# Regret Driver Diagnostics: thesis_final_multi_2m_20260620T091938Z

These diagnostics are descriptive and not causal. A driver label means the regret occurred during or was associated with that condition; it does not prove structural causality without counterfactual reruns.

## Discovery
- Scenarios discovered: 17
- Models: RLQR, TFT, XGB
- Quantiles: p10, p30, p50, p70, p90
- Benchmark requested: rhpf

## Regret Bridge
- TFT p10: total regret -52230.07 EUR; realized 52230.07 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- RLQR p70: total regret -56098.65 EUR; realized 56098.65 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- RLQR p90: total regret -66655.01 EUR; realized 66655.01 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- RLQR p10: total regret -71581.68 EUR; realized 71581.68 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- RLQR p30: total regret -79192.81 EUR; realized 79192.81 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- RLQR p50: total regret -93509.54 EUR; realized 93509.54 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- TFT p50: total regret -98789.38 EUR; realized 98789.38 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- TFT p70: total regret -100385.77 EUR; realized 100385.77 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- TFT p30: total regret -104696.80 EUR; realized 104696.80 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.
- XGB p10: total regret -107984.83 EUR; realized 107984.83 EUR; benchmark 0.00 EUR; benchmark source `benchmark_scenario:artifacts/simulation_runs/thesis_final_multi_2m_20260620T091938Z/benchmarks_rhpf/multi/p50_p50:realized_net_revenue_eur`.

## Missing Data Limitations
- Missing column mappings recorded: 240
- Warnings recorded: 0

Use `debug/missing_columns.csv` and `debug/column_mapping.json` before interpreting unavailable components.
