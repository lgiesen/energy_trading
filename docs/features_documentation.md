# Feature-Dokumentation und Datenwörterbuch

## Einleitung

Dieses Dokument beschreibt den finalen, kausal abgesicherten Merkmalsraum für die aFRR-Prognose.

- Finale Artefaktdatei: `data/features/all_data_features.parquet`
- Aktueller Stand nach PiT-Latenzkorrektur: **173 Spalten**
- Trainingsmerkmale `X` nach `prepare_model_data(...)` (numerisch): **165**
- `timestamp_utc` ist ein **Metadatum/Zeitindex** und wird nicht als Trainingsmerkmal gezählt.

## Regeln zur Publikationslatenz (PiT)

| Datengruppe                                                           | Kausalregel                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------- |
| ENTSO-E-Actuals (Erzeugung/Last/Outages/NRV)                          | mindestens `lag_2h`                                             |
| aFRR/mFRR-Aktivierung und Aktivierungspreise                          | mindestens `lag_1h`                                             |
| Netz-Statistiken Stress (`system_stress_signal`, `grid_stress_index`) | explizit nur `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h` |
| Day-Ahead-Preis                                                       | `da_price_pit` mit D-1 13:00-UTC-Freigabelogik                  |
| Forecast-Signale                                                      | ex ante nutzbar, keine Zukunftsauffüllung                       |

## Daten-Imputation (methodische Begründung)

- Zur Vermeidung künstlicher Datenknappheit werden installierte Kapazitäten
  (`*_capacity`) im Feature-Bau rückwärts aufgefüllt (`backfill`).
- Der erste verfügbare Meldewert (typisch ab Ende 2023) wird rückwirkend für
  frühere Zeitstempel bis mindestens zum PICASSO-Start
  (`2022-06-22 22:00:00+00:00`) als konstante Strukturgröße verwendet.
- Diese Imputation ist wissenschaftlich vertretbar, da installierte Kapazitäten
  sich nur langsam ändern und die stündliche Marktdynamik über Ist- und
  Aktivierungsdaten modelliert wird.

## Metadaten und ausgeschlossene Spalten

| Typ                                                    | Spalten                                                                                                                                       |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Metadatum/Index                                        | `timestamp_utc`                                                                                                                               |
| Technische Metadaten (aus `X` ausgeschlossen)          | `data_is_lagged`, `is_local_reconstruction_only`, `pit_lagged_column_count`, `market_state_cluster`                                           |
| Zielvariablen/Outcome-Spalten (aus `X` ausgeschlossen) | `target_afrr_activation_price_vwap_pos_h1`, `target_da_price_h1`, `target_afrr_rate_h1`, `afrr_capacity_price_pos`, `afrr_capacity_price_neg` |
| Hart entfernt (>90% Missing, nicht modellrelevant)     | `bid_provider_to_grid_share_pos`, `bid_provider_to_grid_share_neg`, `afrr_reconstructed_marginal_price_pos`, `afrr_reconstructed_marginal_price_neg` |

Hinweis: Legacy-Aliase `afrr_activation_price_vwap_pos`/`afrr_activation_price_vwap_neg` wurden auf die kanonischen Namen
`afrr_activation_price_vwap_pos`/`afrr_activation_price_vwap_neg` vereinheitlicht.

## Komprimiertes Datenwörterbuch (Merkmalsfamilien)

| Merkmal (verfügbare Lags)                                                                                                                                           | Einheit      | Beschreibung                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `afrr_activation_price_vwap_pos` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                              | EUR/MWh      | Positiver aFRR-Aktivierungspreis (VWAP) als zentrales Signal für kurzfristige Preisregime und Wochenmuster.            |
| `afrr_activation_price_vwap_neg` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                              | EUR/MWh      | Negativer aFRR-Aktivierungspreis (VWAP); bildet asymmetrische Balancing-Kosten gegenüber POS-Preisen ab.               |
| `afrr_da_price_spread` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                        | EUR/MWh      | Differenz zwischen aFRR-Aktivierungspreis und Day-Ahead-Preis als direkter Opportunitätskosten-Indikator.              |
| `da_price_pit` (ohne Lag, 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                      | EUR/MWh      | PiT-gegatterter Day-Ahead-Preis, der nur nach Veröffentlichungszeitpunkt modellseitig verfügbar ist.                   |
| `da_price` (24h, 48h, 168h)                                                                                                                                         | EUR/MWh      | Historische Day-Ahead-Preisniveaus zur Erfassung stabiler Tages- und Wochenzyklen.                                     |
| `gas_price`, `coal_price`, `co2_price` (ohne Lag)                                                                                                                   | EUR/MWh      | Brennstoff- und Emissionskosten als exogene Kostentreiber der Merit-Order und Strompreisdynamik.                       |
| `da_price_slog1p`, `da_price_diff1`, `da_price_diff24`, `da_price_ewma24` (ohne Lag)                                                                                | EUR/MWh      | Transformierte Preisniveaus und Preisänderungen zur robusteren Abbildung von Sprüngen und Kurzfristtrends.             |
| `da_price_mean_24h`, `da_price_std_24h`, `da_price_mean_168h`, `da_price_std_168h`, `da_price_volatility_30d` (ohne Lag)                                            | EUR/MWh      | Rollierende Mittel-/Volatilitätsmaße zur Quantifizierung von Marktregimewechseln und Unsicherheit.                     |
| `system_stress_signal` (2h, 3h, 6h, 12h, 24h)                                                                                                                       | Index        | Verdichtetes Stresssignal aus Systemungleichgewichten; nur mit 2h+ Latenz gemäß PiT-Regel nutzbar.                     |
| `grid_stress_index` (2h, 3h, 6h, 12h, 24h)                                                                                                                          | Index        | Kompositindex für Netzanspannung, der mehrere Belastungskomponenten in einen robusten Steuerindikator bündelt.         |
| `nrv_zscore_24h` (2h, 3h, 4h)                                                                                                                                       | Index        | Standardisierte Abweichung des NRV gegenüber dem 24h-Verlauf zur Identifikation ungewöhnlicher Balance-Zustände.       |
| `nrv_quantile_5` (2h, 3h, 4h)                                                                                                                                       | Index        | Quantilisierte NRV-Lage (5 Klassen) zur robusten Regimekodierung auch bei Ausreißern.                                  |
| `NRV_balance` (2h)                                                                                                                                                  | MW           | Netto-Regelverbundsaldo als physikalisches Kernsignal für Systemüberschuss oder -defizit.                              |
| `neighbor_spread_avg`, `relative_price_competitiveness`, `price_volatility_short_term`, `scarcity_price_premium` (je 2h)                                            | EUR/MWh      | Abgeleitete Wettbewerbs-, Knappheits- und Volatilitätsindikatoren zur Erklärung kurzfristiger Preisaufschläge.         |
| `afrr_activated_mw_pos/neg` (1h, 24h)                                                                                                                               | MW           | Tatsächlich aktivierte aFRR-Leistung als direktes Maß für den realen Regelenergiebedarf im System.                     |
| `afrr_capacity_awarded_mw_pos/neg` (1h)                                                                                                                             | MW           | Bezuschlagte aFRR-Vorhaltemengen als Information über erwartete Balancing-Anforderungen.                               |
| `afrr_activation_offered_mw_pos/neg` (1h)                                                                                                                           | MW           | Angebotsseitige Aktivierungsmengen als Liquiditäts- und Spannungsindikator der Balancing-Märkte.                       |
| `afrr_activation_rate` (1h, 24h)                                                                                                                                    | Anteil (0-1) | Verhältnis aktivierter zu vorgehaltener aFRR-Leistung als Intensitätsmaß der Systembeanspruchung.                      |
| `is_activated` (1h)                                                                                                                                                 | Flag (0/1)   | Binärer Aktivierungsstatus zur Trennung von Aktivierungs- und Nicht-Aktivierungsstunden.                               |
| `mfrr_activated_mw_pos/neg` (1h), `mfrr_mari_net_mw` (1h), `mfrr_active_lag` (ohne Lag)                                                                             | MW / Index   | mFRR-Aktivierung und MARI-Flüsse als ergänzende Balancing-Signale für systemweite Reserveanspannung.                   |
| `residual_load_actual`, `residual_load_calc` (je 2h)                                                                                                                | MW           | Reallast abzüglich erneuerbarer Einspeisung als zentraler Treiber konventioneller Fahrweise.                           |
| `generation_fossil_total_mw`, `generation_hydro_pumped_storage_mw`, `generation_hydro_actual_total`, `generation_baseload_total` (je 2h)                            | MW / Anteil  | Aggregierte Erzeugungsblöcke zur Abbildung der aktuellen Angebotsstruktur im deutschen Stromsystem.                    |
| `wind_onshore_actual_entsoe` (2h, 24h, 48h, 168h)                                                                                                                   | MW           | Tatsächliche Onshore-Winderzeugung mit Kurz- bis Wochenhistorie zur Modellierung wettergetriebener Regime.             |
| `wind_offshore_actual_entsoe`, `solar_actual_entsoe` (je 2h)                                                                                                        | MW           | Tatsächliche Offshore-Wind- und Solarleistung als unmittelbare Determinanten der Residuallast.                         |
| `wind_onshore_error_da/id`, `wind_offshore_error_da/id`, `solar_error_da/id`, `wind_total_error_da`, `total_wind_solar_id_error` (je 2h)                            | MW           | Forecast-Fehler gegenüber Istwerten als Proxy für Prognoseunsicherheit und spätere Regelenergiebedarfe.                |
| `wind_onshore_actual_entsoe_mean_24h/std_24h/mean_168h/std_168h` (je 2h)                                                                                            | MW           | Rollierende Lage- und Streuungsmaße der Onshore-Einspeisung zur Stabilisierung der Windregime-Erkennung.               |
| `unplanned_outages_mw` (2h), `planned_outages_mw` (ohne Lag)                                                                                                        | MW           | Ungeplante und geplante Kraftwerksausfälle als Angebotsrestriktionssignal im kurzfristigen Dispatch.                   |
| `wind_onshore_forecast_id_entsoe`, `wind_offshore_forecast_id_entsoe`, `solar_forecast_id_entsoe` (ohne Lag, 24h, 48h, 168h)                                        | MW           | Ex-ante Einspeiseprognosen für erneuerbare Energien zur frühzeitigen Abbildung erwarteter Volatilität.                 |
| `renewable_share_forecast`, `residual_load_forecast` (ohne Lag, 24h, 48h, 168h)                                                                                     | Anteil / MW  | Prognostizierter EE-Anteil und erwartete Residuallast als Schlüsselgrößen für Day-Ahead- und Balancing-Lage.           |
| `wind_forecast_update`, `wind_onshore_forecast_update`, `solar_forecast_update` (ohne Lag)                                                                          | Index        | Änderungsmaße der Forecasts als Frühindikator für neue Wetterinformationen und Repricing-Risiken.                      |
| `wind_onshore_capacity`, `wind_offshore_capacity`, `solar_capacity`, `gas_capacity`, `hard_coal_capacity`, `lignite_capacity`, `pumped_storage_capacity` (ohne Lag) | MW           | Verfügbare Kapazitäten als strukturelle Obergrenzen des Erzeugungs- und Flexibilitätspotenzials.                       |
| `picasso_flow_rate` (1h, 24h)                                                                                                                                       | Anteil (0-1) | Anteil grenzüberschreitender PICASSO-Aktivierung als Indikator für europäische Kopplungseffekte.                       |
| `TE_hour_regime_activation` (1h)                                                                                                                                    | Index        | Zeitregime-basierte Aktivierungskodierung zur expliziten Trennung typischer Stundenmuster.                             |
| `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `weekday_sin`, `weekday_cos`, `month_sin`, `month_cos` (ohne Lag)                                         | Index        | Zyklische Zeitkodierungen für Intraday-, Wochen- und Saisonrhythmen ohne künstliche Sprungstellen.                     |
| `is_weekend`, `is_afternoon`, `is_evening`, `is_morning`, `is_night`, `is_bridge_day`, `is_payday_period`, `is_christmas_break`, `is_picasso_regime` (ohne Lag)     | Flag (0/1)   | Binäre Regimeindikatoren für kalender- und marktstrukturbedingte Nachfragemuster und Aktivierungswahrscheinlichkeiten. |
| `holiday_severity`, `market_regime_picasso` (ohne Lag)                                                                                                              | Index        | Verdichtete Kalender- und Marktregime-Indizes zur robusten Trennung außergewöhnlicher Tage und Betriebsphasen.         |

## Kausalitäts-Check zur Lag-Benennung

- Alle Lag-Spalten folgen dem Schema `*_lag_Xh`.
- `X` beschreibt die absolute Verzögerung zur Echtzeit.
- Für `system_stress_signal` und `grid_stress_index` existiert **kein** `lag_1h`.
