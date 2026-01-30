"""Shared configuration for pricing, dispatch, and profit calculation."""

BATTERY_SPECS = {
    "capacity_mwh": 2.0,      # battery size/installed capacity E_{max} = 2 MWh.
    # A 2-hour duration is the current industry "sweet spot" for combining Arbitrage + Ancillary Services.
    "power_mw": 1.0,          # Maximum charge/discharge rate. Installed Power P_{max} = 1 MW. 
    # Matches the minimum bid size for many markets (1 MW is standard for aFRR/SRL).
    "efficiency": 0.9,       # Round-trip efficiency (AC-to-AC)
    "soc_min": 0.1,          # 10% minimum state of charge
    "soc_max": 0.9,           # 90% maximum state of charge
    "initial_soc": 0.5,           # Start simulation at 50%
    "degradation_cost": 20.0, # Marginal cost per MWh throughput
    # Regulatorik für Regelleistung (aFRR)
    "afrr_min_bid_size": 1.0, # Mindestangebot (in DE oft 1 MW)
}
