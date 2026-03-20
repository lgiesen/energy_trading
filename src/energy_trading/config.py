"""Shared configuration for battery specs, economics, and market constraints."""

# --- MODEL / TIME CONVENTIONS ---
MODEL_SPECS = {
    "time_step_hours": 1.0,  # Optimization time step (Delta t): 1h
    "reserve_product_duration_h": 1.0,  # Duration used for activation-energy bounds
    "market_scope": "DE_LU",  # Sign/settlement convention scope
}
MODEL_SPECS["optimization_step_min"] = int(MODEL_SPECS["time_step_hours"] * 60)

# --- PHYSICAL BATTERY SPECS ---
BATTERY_SPECS = {
    "capacity_mwh": 20.0,  # Total energy capacity (E_cap)
    "power_mw": 10.0,  # Maximum converter power (P_max) -> 2-hour duration

    # One-way efficiencies (primary inputs); round-trip efficiency is derived below.
    "efficiency_in": 0.9487,
    "efficiency_out": 0.9487,

    # Operation limits as fractions of capacity_mwh
    "soc_min": 0.1,  # 10% minimum SoC to protect battery health
    "soc_max": 0.9,  # 90% maximum SoC
    "initial_soc": 0.5,  # Start at 50% SoC
    "soc_target_end": 0.5,  # Cyclic neutrality target (SoC_0 ~= SoC_T)

    # Costs
    "degradation_cost": 25.0,  # EUR/MWh internal throughput
    "aux_power_mw": 0.2,  # AC-side house load (cooling/BMS/etc.)
}
BATTERY_SPECS["efficiency_rt"] = BATTERY_SPECS["efficiency_in"] * BATTERY_SPECS["efficiency_out"]

# --- ECONOMIC PARAMETERS ---
FINANCIAL_PARAMS = {
    "initial_cash": 0.0,  # Starting at 0 to track maximum drawdown
    "currency": "EUR",

    # Investment assumptions for ROI calculation at the end of the Thesis
    "capex_per_kwh": 350.0,
    "annual_opex_fixed": 5000.0,
    "transaction_cost_eur_per_mwh": 1.0,  # C_trans in thesis
    "risk_margin_eur_per_mwh": 0.0,  # Optional conservative margin for thresholds
}

# --- MARKET CONSTRAINTS ---
MARKET_SPECS = {
    "afrr_min_bid_size": 1.0,  # Minimum bid size in MW (Germany)
    "afrr_bid_granularity": 1.0,  # Bid steps (often 1 MW units)
    # Keep separate from optimization step on purpose:
    # optimization uses 60 min, settlement/input granularity is 15 min.
    "settlement_period_min": 15,  # Settlement/input granularity for imbalance/SRL
}
MARKET_SPECS["bid_power_max_mw"] = BATTERY_SPECS["power_mw"]


def _validate_config() -> None:
    dt_h = MODEL_SPECS["time_step_hours"]
    dt_min = MODEL_SPECS["optimization_step_min"]
    if dt_h <= 0:
        raise ValueError("MODEL_SPECS['time_step_hours'] must be > 0.")
    if dt_min != int(dt_h * 60):
        raise ValueError("optimization_step_min must equal time_step_hours * 60.")

    eta_in = BATTERY_SPECS["efficiency_in"]
    eta_out = BATTERY_SPECS["efficiency_out"]
    if not (0 < eta_in <= 1) or not (0 < eta_out <= 1):
        raise ValueError("efficiency_in and efficiency_out must be in (0, 1].")

    soc_min = BATTERY_SPECS["soc_min"]
    soc_max = BATTERY_SPECS["soc_max"]
    soc_initial = BATTERY_SPECS["initial_soc"]
    soc_target_end = BATTERY_SPECS["soc_target_end"]
    if not (0 <= soc_min < soc_max <= 1):
        raise ValueError("Require 0 <= soc_min < soc_max <= 1.")
    if not (soc_min <= soc_initial <= soc_max):
        raise ValueError("initial_soc must be within [soc_min, soc_max].")
    if not (soc_min <= soc_target_end <= soc_max):
        raise ValueError("soc_target_end must be within [soc_min, soc_max].")

    if BATTERY_SPECS["capacity_mwh"] <= 0:
        raise ValueError("capacity_mwh must be > 0.")
    if BATTERY_SPECS["power_mw"] <= 0:
        raise ValueError("power_mw must be > 0.")
    if BATTERY_SPECS["aux_power_mw"] < 0:
        raise ValueError("aux_power_mw must be >= 0.")
    if BATTERY_SPECS["degradation_cost"] < 0:
        raise ValueError("degradation_cost must be >= 0.")

    if FINANCIAL_PARAMS["transaction_cost_eur_per_mwh"] < 0:
        raise ValueError("transaction_cost_eur_per_mwh must be >= 0.")
    if FINANCIAL_PARAMS["risk_margin_eur_per_mwh"] < 0:
        raise ValueError("risk_margin_eur_per_mwh must be >= 0.")

    if MARKET_SPECS["afrr_bid_granularity"] <= 0:
        raise ValueError("afrr_bid_granularity must be > 0.")
    if MARKET_SPECS["afrr_min_bid_size"] < 0:
        raise ValueError("afrr_min_bid_size must be >= 0.")
    if MARKET_SPECS["settlement_period_min"] <= 0:
        raise ValueError("settlement_period_min must be > 0.")
    if MARKET_SPECS["bid_power_max_mw"] <= 0:
        raise ValueError("bid_power_max_mw must be > 0.")
    if MARKET_SPECS["bid_power_max_mw"] > BATTERY_SPECS["power_mw"]:
        raise ValueError("bid_power_max_mw cannot exceed battery power_mw.")


_validate_config()
