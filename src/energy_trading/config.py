"""Shared configuration for battery specs, economics, and market constraints."""

# --- PHYSICAL BATTERY SPECS ---
BATTERY_SPECS = {
    "capacity_mwh": 2.0,       # Total energy capacity (E_max)
    "power_mw": 1.0,           # Maximum power (P_max) -> 2-hour duration
    
    # Efficiency: 90% Round-trip AC-to-AC
    # We use the square root for one-way efficiency (sqrt(0.9) ≈ 0.9487)
    "efficiency_rt": 0.9,      
    "efficiency_in": 0.9487,   
    "efficiency_out": 0.9487,  

    # Operation Limits
    "soc_min": 0.1,            # 10% min SOC to protect battery health
    "soc_max": 0.9,            # 90% max SOC
    "initial_soc": 0.1,        # Start at soc_min (realistic empty state)
    
    # Costs
    "degradation_cost": 25.0,  # €/MWh throughput (slightly more conservative value)
}

# --- ECONOMIC PARAMETERS ---
# This section follows the "Cumulative Cashflow" approach
FINANCIAL_PARAMS = {
    "initial_cash": 0.0,        # Starting at 0 to track maximum drawdown
    "currency": "EUR",
    
    # Investment assumptions for ROI calculation at the end of the Thesis
    "capex_per_kwh": 350.0,     # Estimated investment costs (e.g., 350k € per MWh)
    "annual_opex_fixed": 5000.0 # Fixed maintenance/insurance per year
}

# --- MARKET CONSTRAINTS ---
MARKET_SPECS = {
    "afrr_min_bid_size": 1.0,    # Minimum bid size in MW (Germany)
    "afrr_bid_granularity": 1.0, # Bid steps (often 1 MW units)
    "settlement_period_min": 15, # 15-minute intervals for imbalance/SRL
}