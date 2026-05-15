"""Shared configuration for battery specs, economics, and market constraints."""

# --- MODEL / TIME CONVENTIONS ---
MODEL_SPECS = {
    "time_step_hours": 1.0,  # Optimization time step (Delta t): 1h
    "reserve_product_duration_h": 4.0,  # Duration used for activation-energy bounds
    "market_scope": "DE_LU",  # Sign/settlement convention scope
    "terminal_soc_value_discount": 0.8,  # Discount for terminal SoC value in objective
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
    # State-dependent auxiliary duty-cycle model (recommended):
    # OFF -> STANDBY -> TRADING -> aFRR_ACTIVE
    # Duty values are multipliers of aux_power_peak_mw.
    # Example: duty=0.95 means 95% of aux_power_peak_mw (not 0.95 MW absolute).
    "aux_power_mode": "state_dependent",  # "state_dependent" | "constant"
    "aux_power_peak_mw": 0.20,  # Full-load auxiliary draw (cooling+BMS+PCS)
    "aux_power_standby_duty": 0.20,  # 20% of peak aux when reserve-ready and idle
    "aux_power_trading_duty": 0.70,  # 70% of peak aux during active DA/ID charging/discharging
    "aux_power_afrr_active_duty": 0.95,  # 95% of peak aux during real aFRR activation
    "aux_power_off_duty": 0.00,  # 0% of peak aux when fully off
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
    "afrr_offer_cost_eur_mw_h": 0.0,  # Fixed reserve availability cost per offered MW/h (independent of acceptance)
    "risk_margin_eur_per_mwh": 0.0,  # Optional conservative margin for thresholds
    "imbalance_penalty_eur_mwh": 500.0,  # Non-delivery/imbalance penalty proxy
    # aFRR penalty calibration (regelleistung-style proxy inputs/fallbacks)
    "afrr_penalty_aufschlag_eur_mwh": 30.0,
    "afrr_penalty_aufschlag_eur_mw_h": 3.0,
    "afrr_penalty_default_marginal_energy_price_eur_mwh": 150.0,
    "afrr_penalty_default_avg_capacity_price_product_eur_mw_h": 12.5,
    "afrr_penalty_default_idaep_eur_mwh": 100.0,
}

# --- MARKET CONSTRAINTS ---
MARKET_SPECS = {
    "da_min_bid_size": 0.1,  # Minimum bid size in MW (Germany)
    "da_bid_granularity": 0.1,  # Bid steps
    "afrr_min_bid_size": 1.0,  # Minimum bid size in MW (Germany)
    "afrr_bid_granularity": 1.0,  # Bid steps
    "da_execution_mode": "limit",  # "price_taker" or "limit"
    "da_arbitrage_mode": "limit",  # mode for non-hedging DA volumes
    "da_link_to_awarded_afrr": True,  # cancel hedges if aFRR capacity was not awarded
    "afrr_capacity_bid_risk_lambda": 0.2,
    "afrr_activation_bid_risk_lambda": 0.2,
    "afrr_energy_bid_strategy": "forecast",  # "forecast" | "marginal_cost" | "hybrid"
    "da_buy_limit_price_eur_mwh": 3000.0,
    "da_sell_limit_price_eur_mwh": -500.0,
    # Limit-order aggressiveness around expected DA price (EUR/MWh)
    # buy_limit = pred + buy_offset, sell_limit = pred - sell_offset
    "da_buy_limit_offset_eur_mwh": 0.0,
    "da_sell_limit_offset_eur_mwh": 2.0,
    # Quantile-backed DA limit thresholds (absolute prices from forecast tails):
    # buy uses conservative upper tail (e.g. p90/p95), sell lower tail (e.g. p10/p05).
    "da_buy_limit_quantile": "p90",
    "da_sell_limit_quantile": "p10",
    # Debug guard for DA limit bids:
    # if enabled, fail hard when configured DA quantile inputs are missing/invalid
    # instead of silently falling back to weaker pricing logic.
    "da_bid_fail_fast_debug": False,
    # Synthetic ID rescue pricing around DA with market caps/floors
    "id_rescue_spread_eur_mwh": 30.0,
    "id_buy_price_cap_eur_mwh": 3000.0,
    "id_sell_price_floor_eur_mwh": -500.0,
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
    aux_mode = str(BATTERY_SPECS.get("aux_power_mode", "state_dependent")).strip().lower()
    if aux_mode not in {"state_dependent", "constant"}:
        raise ValueError("aux_power_mode must be one of {'state_dependent','constant'}.")
    # Legacy constant-aux key is only required/validated in constant mode.
    if aux_mode == "constant":
        aux_const = float(BATTERY_SPECS.get("aux_power_mw", 0.0))
        if aux_const < 0:
            raise ValueError("aux_power_mw must be >= 0.")
    for k in (
        "aux_power_peak_mw",
        "aux_power_standby_duty",
        "aux_power_trading_duty",
        "aux_power_afrr_active_duty",
        "aux_power_off_duty",
    ):
        v = float(BATTERY_SPECS.get(k, 0.0))
        if v < 0:
            raise ValueError(f"{k} must be >= 0.")
    if BATTERY_SPECS["degradation_cost"] < 0:
        raise ValueError("degradation_cost must be >= 0.")

    if FINANCIAL_PARAMS["transaction_cost_eur_per_mwh"] < 0:
        raise ValueError("transaction_cost_eur_per_mwh must be >= 0.")
    if FINANCIAL_PARAMS["risk_margin_eur_per_mwh"] < 0:
        raise ValueError("risk_margin_eur_per_mwh must be >= 0.")

    if MARKET_SPECS["afrr_bid_granularity"] <= 0:
        raise ValueError("afrr_bid_granularity must be > 0.")
    if MARKET_SPECS["da_execution_mode"] not in {"price_taker", "limit"}:
        raise ValueError("da_execution_mode must be one of {'price_taker', 'limit'}.")
    if MARKET_SPECS["da_arbitrage_mode"] not in {"price_taker", "limit"}:
        raise ValueError("da_arbitrage_mode must be one of {'price_taker', 'limit'}.")
    if str(MARKET_SPECS.get("da_buy_limit_quantile", "p90")).lower() not in {"p05", "p10", "p90", "p95"}:
        raise ValueError("da_buy_limit_quantile must be one of {'p05','p10','p90','p95'}.")
    if str(MARKET_SPECS.get("da_sell_limit_quantile", "p10")).lower() not in {"p05", "p10", "p90", "p95"}:
        raise ValueError("da_sell_limit_quantile must be one of {'p05','p10','p90','p95'}.")
    if not isinstance(MARKET_SPECS.get("da_bid_fail_fast_debug", False), bool):
        raise ValueError("da_bid_fail_fast_debug must be boolean.")
    if MARKET_SPECS["afrr_energy_bid_strategy"] not in {"forecast", "marginal_cost", "hybrid"}:
        raise ValueError("afrr_energy_bid_strategy must be one of {'forecast', 'marginal_cost', 'hybrid'}.")
    if MARKET_SPECS["afrr_capacity_bid_risk_lambda"] < 0:
        raise ValueError("afrr_capacity_bid_risk_lambda must be >= 0.")
    if MARKET_SPECS["afrr_activation_bid_risk_lambda"] < 0:
        raise ValueError("afrr_activation_bid_risk_lambda must be >= 0.")
    if MARKET_SPECS["afrr_min_bid_size"] < 0:
        raise ValueError("afrr_min_bid_size must be >= 0.")
    if MARKET_SPECS["settlement_period_min"] <= 0:
        raise ValueError("settlement_period_min must be > 0.")
    if MARKET_SPECS["bid_power_max_mw"] <= 0:
        raise ValueError("bid_power_max_mw must be > 0.")
    if MARKET_SPECS["bid_power_max_mw"] > BATTERY_SPECS["power_mw"]:
        raise ValueError("bid_power_max_mw cannot exceed battery power_mw.")


_validate_config()
