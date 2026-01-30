"""Backtest a simple value stacking strategy (DA arbitrage vs aFRR)."""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np


BATTERY_CONFIG = {
    # Physical
    "capacity_mwh": 2.0,
    "power_mw": 1.0,
    "efficiency": 0.90,
    "soc_min": 0.05,
    "soc_max": 0.95,
    # Losses & Costs
    "degradation_cost_eur_mwh": 20.0,
    "trading_fee_eur_mwh": 0.50,
    "auxiliary_power_mw": 0.002,
    # Financials
    "capex_total_eur": 600000,
    "opex_annual_eur": 10000,
    # Heuristic thresholds
    "buy_threshold_eur_mwh": 50.0,
    "sell_threshold_eur_mwh": 100.0,
}


REQUIRED_ACTUALS = [
    "da_price_actual",
    "afrr_cap_price_actual",
    "afrr_act_price_actual",
    "afrr_activation_rate_actual",
]

REQUIRED_FORECASTS = [
    "da_price_forecast",
    "afrr_cap_price_forecast",
    "afrr_act_price_forecast",
    "afrr_activation_rate_forecast",
]


def _make_dummy_data(path: Path) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", "2022-12-31 23:00:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    df["da_price_actual"] = rng.normal(100, 30, size=len(idx)).clip(-50, 400)
    df["afrr_cap_price_actual"] = rng.normal(10, 3, size=len(idx)).clip(0, 50)
    df["afrr_act_price_actual"] = rng.normal(120, 40, size=len(idx)).clip(-50, 600)
    df["afrr_activation_rate_actual"] = rng.uniform(0.0, 0.4, size=len(idx))
    # Forecasts with slight noise
    df["da_price_forecast"] = df["da_price_actual"] + rng.normal(0, 10, size=len(idx))
    df["afrr_cap_price_forecast"] = df["afrr_cap_price_actual"] + rng.normal(0, 1, size=len(idx))
    df["afrr_act_price_forecast"] = df["afrr_act_price_actual"] + rng.normal(0, 15, size=len(idx))
    df["afrr_activation_rate_forecast"] = df["afrr_activation_rate_actual"] + rng.normal(0, 0.05, size=len(idx))
    df["afrr_activation_rate_forecast"] = df["afrr_activation_rate_forecast"].clip(0, 1)
    return df


def _ensure_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    for act, fc in zip(REQUIRED_ACTUALS, REQUIRED_FORECASTS):
        if fc not in df.columns and act in df.columns:
            df[fc] = df[act]
    return df


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    cfg = BATTERY_CONFIG
    cap = cfg["capacity_mwh"]
    power = cfg["power_mw"]
    eff = cfg["efficiency"]
    soc_min = cfg["soc_min"] * cap
    soc_max = cfg["soc_max"] * cap
    aux_loss = cfg["auxiliary_power_mw"]
    deg_cost = cfg["degradation_cost_eur_mwh"]
    fee = cfg["trading_fee_eur_mwh"]

    soc = soc_min
    results = []

    for ts, row in df.iterrows():
        # Step A: standby loss
        soc -= aux_loss
        emergency_cost = 0.0
        if soc < 0:
            # emergency charge from grid
            deficit = soc_min - soc
            emergency_cost = deficit * row["da_price_actual"]
            soc = soc_min

        # Step B: decision (avoid comparing revenue vs cost directly)
        da_buy = row["da_price_forecast"] < cfg["buy_threshold_eur_mwh"]
        da_sell = row["da_price_forecast"] > cfg["sell_threshold_eur_mwh"]

        profit_afrr = (
            row["afrr_cap_price_forecast"]
            + row["afrr_act_price_forecast"] * row["afrr_activation_rate_forecast"]
            - (row["afrr_activation_rate_forecast"] * deg_cost)
        ) * power

        action = "idle"
        if da_buy and soc < soc_max:
            if profit_afrr > 150:
                action = "afrr"
            else:
                action = "charge"
        elif da_sell and soc > soc_min:
            profit_da = (row["da_price_forecast"] - fee - deg_cost) * power
            if profit_afrr > profit_da:
                action = "afrr"
            else:
                action = "discharge"
        elif profit_afrr > 0 and soc > soc_min:
            action = "afrr"

        # Step C: accounting
        cashflow = 0.0
        revenue = 0.0
        cost = 0.0
        throughput = 0.0

        if action == "charge":
            energy = min(power, soc_max - soc)
            soc += energy * eff
            cashflow = -(row["da_price_actual"] + fee) * energy
            cost = -cashflow
            throughput = energy
        elif action == "discharge":
            energy = max(0.0, min(power, soc - soc_min))
            soc -= energy / eff
            cashflow = (row["da_price_actual"] - fee) * energy
            revenue = cashflow
            throughput = energy
        elif action == "afrr":
            # Simplified: treat activation as discharge energy
            act_energy = row["afrr_activation_rate_actual"] * power
            act_energy = min(act_energy, soc - soc_min)
            soc -= act_energy / eff
            cap_rev = row["afrr_cap_price_actual"] * power
            act_rev = row["afrr_act_price_actual"] * act_energy
            cashflow = cap_rev + act_rev - fee * act_energy
            revenue = cashflow
            throughput = act_energy

        cashflow -= emergency_cost
        if emergency_cost > 0:
            cost += emergency_cost

        results.append(
            {
                "timestamp": ts,
                "soc_mwh": soc,
                "action": action,
                "cashflow_eur": cashflow,
                "revenue_eur": revenue,
                "cost_eur": cost,
                "degradation_cost_eur": throughput * deg_cost,
            }
        )

    return pd.DataFrame(results).set_index("timestamp")


def summarize(results: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    revenue = results.loc[results["cashflow_eur"] > 0, "cashflow_eur"].sum()
    energy_cost = -results.loc[results["cashflow_eur"] < 0, "cashflow_eur"].sum()
    gross_profit = revenue - energy_cost
    degradation = results["degradation_cost_eur"].sum()
    years = max((end - start).days / 365.25, 1e-9)
    net_profit = gross_profit - degradation - BATTERY_CONFIG["opex_annual_eur"] * years
    payback = BATTERY_CONFIG["capex_total_eur"] / max(net_profit / years, 1e-9)

    print("=== Backtest Summary ===")
    print(f"Total Revenue: {revenue:,.2f} EUR")
    print(f"Total Energy Cost: {energy_cost:,.2f} EUR")
    print(f"Gross Profit (EBITDA): {gross_profit:,.2f} EUR")
    print(f"Degradation Cost: {degradation:,.2f} EUR")
    print(f"Net Profit: {net_profit:,.2f} EUR")
    print(f"Payback Period (years): {payback:,.2f}")


def main() -> None:
    input_path = Path("data/simulation_results.parquet")
    if input_path.exists():
        df = pd.read_parquet(input_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
    else:
        df = _make_dummy_data(input_path)

    df = _ensure_forecasts(df)
    start = df.index.min()
    end = df.index.max()

    results = backtest(df)
    summarize(results, start, end)

    out_path = Path("data/backtest_results.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_path, index=True)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
