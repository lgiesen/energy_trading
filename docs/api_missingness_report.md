# API Missingness Report

- Generated at (UTC): `2026-03-29T16:41:52.446968+00:00`
- Scope: `data/raw/*.parquet`

## Forensischer Befund April 2025 (ENTSO-E)

- Geprüftes Fenster: `2025-04-01 00:00:00+00:00` bis
  `2025-04-02 23:00:00+00:00`.
- Re-Fetch-Vergleich: `data/reports/april_refetch_comparison.csv`
- Lag-Traceback: `data/reports/april_hard_gap_propagation.csv`
- Befund: Für `wind_onshore_forecast_id_entsoe` blieb die Anzahl fehlender
  Werte unverändert (`old_null_count=22`, `new_null_count=22`).
- Schlussfolgerung: **Hard Source Gap** auf Primärquellenebene
  (ENTSO-E-Transparenzplattform), kein Ingestionsfehler der Pipeline.

## Überblick je Quelle

| API | Datei | Spalten mit NaNs | Null-Zellen gesamt | Max Null-% in einer Spalte |
|---|---|---:|---:|---:|
| Netztransparenz | `data/raw/netztransparenz.parquet` | 13 | 346078 | 29.74 |
| SMARD | `data/raw/smard.parquet` | 14 | 106826 | 60.25 |
| Yahoo Finance | `data/raw/yfinance.parquet` | 1 | 7704 | 16.76 |
| Regelleistung.net | `data/raw/regelleistung.parquet` | 20 | 142 | 0.11 |
| ENTSO-E | `data/raw/entsoe.parquet` | 6 | 112 | 0.11 |
| Energy-Charts | `data/raw/energy_charts.parquet` | 1 | 24 | 0.05 |

## Top 50 problematische Spalten

| API | Spalte | Null-% | Null Count | First Null TS | Last Null TS |
|---|---|---:|---:|---|---|
| SMARD | `lignite_capacity` | 60.25 | 27724 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `wind_offshore_capacity` | 60.25 | 27724 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `hard_coal_capacity` | 40.90 | 18820 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `generation_nuclear_mw` | 39.39 | 18124 | 2024-02-04 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `afrr_picasso_mw_neg` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `afrr_picasso_mw_pos` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `afrr_picasso_net_mw` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `mfrr_mari_mw_neg` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `mfrr_mari_mw_pos` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `mfrr_mari_net_mw` | 29.74 | 54725 | 2020-11-30 00:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `pumped_storage_capacity` | 21.92 | 10084 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Yahoo Finance | `co2_price` | 16.76 | 7704 | 2020-12-01 00:00:00+00:00 | 2021-10-17 23:00:00+00:00 |
| Netztransparenz | `rz_saldo_mw_qs` | 4.80 | 8841 | 2022-02-28 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `NRV_balance_qs` | 4.80 | 8828 | 2022-02-28 23:00:00+00:00 | 2022-05-31 21:45:00+00:00 |
| SMARD | `gas_capacity` | 3.14 | 1444 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `solar_capacity` | 3.14 | 1444 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| SMARD | `wind_onshore_capacity` | 3.14 | 1444 | 2020-11-29 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| ENTSO-E | `load_forecast_da_entsoe` | 0.11 | 50 | 2022-02-21 23:00:00+00:00 | 2024-10-26 22:00:00+00:00 |
| Regelleistung.net | `afrr_activation_offered_mw_neg` | 0.11 | 49 | 2021-06-23 18:00:00+00:00 | 2021-11-29 22:00:00+00:00 |
| Regelleistung.net | `afrr_activation_offered_mw_pos` | 0.11 | 49 | 2021-06-23 18:00:00+00:00 | 2021-11-29 22:00:00+00:00 |
| ENTSO-E | `solar_forecast_id_entsoe` | 0.05 | 24 | 2021-08-12 22:00:00+00:00 | 2021-08-13 21:00:00+00:00 |
| ENTSO-E | `wind_onshore_forecast_id_entsoe` | 0.05 | 24 | 2025-03-31 22:00:00+00:00 | 2025-04-01 21:00:00+00:00 |
| Energy-Charts | `da_price_BE` | 0.05 | 24 | 2025-09-29 22:00:00+00:00 | 2025-09-30 21:00:00+00:00 |
| ENTSO-E | `hydro_reservoir_actual_entsoe` | 0.02 | 8 | 2024-09-26 13:00:00+00:00 | 2026-02-12 13:00:00+00:00 |
| Regelleistung.net | `afrr_capacity_offered_mw_neg` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Regelleistung.net | `afrr_capacity_offered_mw_pos` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Regelleistung.net | `afrr_capacity_price_neg` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Regelleistung.net | `afrr_capacity_price_pos` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Regelleistung.net | `capacity_import_export_mw_neg` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Regelleistung.net | `capacity_import_export_mw_pos` | 0.01 | 5 | 2021-10-31 22:00:00+00:00 | 2025-10-26 22:00:00+00:00 |
| Netztransparenz | `afrr_activated_mw_neg` | 0.01 | 13 | 2026-02-28 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `afrr_activated_mw_pos` | 0.01 | 13 | 2026-02-28 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `mfrr_activated_mw_neg` | 0.01 | 13 | 2026-02-28 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| Netztransparenz | `mfrr_activated_mw_pos` | 0.01 | 13 | 2026-02-28 23:00:00+00:00 | 2026-03-01 02:00:00+00:00 |
| ENTSO-E | `solar_forecast_da_entsoe` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| ENTSO-E | `wind_onshore_forecast_da_entsoe` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `solar_error` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `solar_forecast` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `system_stress_signal` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `wind_forecast_de` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `wind_onshore_error` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| SMARD | `wind_onshore_forecast` | 0.01 | 3 | 2023-10-28 22:00:00+00:00 | 2025-10-25 22:00:00+00:00 |
| Regelleistung.net | `afrr_activation_price_vwap_neg` | 0.00 | 2 | 2020-11-29 23:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Regelleistung.net | `afrr_activation_price_vwap_pos` | 0.00 | 2 | 2020-11-29 23:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Netztransparenz | `rz_saldo_mw_op` | 0.00 | 7 | 2021-01-15 23:00:00+00:00 | 2025-05-09 09:30:00+00:00 |
| Regelleistung.net | `afrr_avg_activation_price_neg` | 0.00 | 1 | 2021-10-31 22:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Regelleistung.net | `afrr_avg_activation_price_pos` | 0.00 | 1 | 2021-10-31 22:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Regelleistung.net | `afrr_bid_avg_activation_price_neg` | 0.00 | 1 | 2021-10-31 22:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Regelleistung.net | `afrr_bid_avg_activation_price_pos` | 0.00 | 1 | 2021-10-31 22:00:00+00:00 | 2021-10-31 22:00:00+00:00 |
| Regelleistung.net | `afrr_bid_vwap_activation_price_neg` | 0.00 | 1 | 2021-10-31 22:00:00+00:00 | 2021-10-31 22:00:00+00:00 |

Hinweis: Diese Datei dokumentiert Rohquellen-Lücken. Ob diese im finalen ML-Bundle noch sichtbar sind, hängt von der späteren Imputation/Feature-Logik ab.
