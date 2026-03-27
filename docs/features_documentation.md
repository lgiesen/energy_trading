# Feature Documentation & Data Dictionary (ML-Training X)

## Einleitung
Diese Dokumentation enthält **nur Spalten, die tatsächlich als ML-Input (`X`)** in `prepare_model_data(...)` verbleiben.
- Basisartefakt: `data/features/all_data_features.parquet` (**172** Spalten)
- Dokumentierte ML-Features (`X`): **166**
- Automatisch aus `X` ausgeschlossene Spalten (fachlich relevant): **6**

## Momentum-Lags (Final)
| Gruppe | Spaltenfamilien | Zusätzliche Lags |
|---|---|---|
| Momentum | aFRR-Preise (`afrr_activation_price_vwap_*`), `afrr_da_price_spread`, `system_stress_signal`, `nrv_zscore_24h` | `[1, 2, 3]` |
| Trend-Historie (Schlüsselvariablen) | `afrr_da_price_spread`, `afrr_activation_price_vwap_pos/neg`, `da_price_pit`, `system_stress_signal` | `[1, 2, 3, 6, 12, 24, 48, 168]` |
| Saison | `da_price_pit`, Wetter-/Forecast-Familie (`*_forecast_id_entsoe`, `renewable_share_forecast`), Last/Residuallast | `[24, 48, 168]` |

Hinweis zur Benennung: Lags sind als **absolute Latenz** codiert. Beispiel: `*_lag_1h` + zusätzlicher Shift `2h` ergibt `*_lag_3h`.
Die breite Lag-Tiefe bis 168h erlaubt dem Modell, sowohl kurzfristige Markttrends (Momentum) als auch Wochenmuster robust zu erfassen.

## Publication Latency Rules (PiT)
| Datengruppe | Regel |
|---|---|
| ENTSO-E/Actuals (Erzeugung/Last/Outages/NRV) | mindestens `lag_2h` |
| aFRR/mFRR Aktivierung & Aktivierungspreise | mindestens `lag_1h` |
| Day-Ahead Preis | `da_price_pit` mit D-1 13:00 UTC Gate; vor Freigabe Fallback `shift(24)` |
| Forecast-basierte Größen | publication-gated je Horizont, kein Future-Backfill |

### Aus `X` entfernte Spalten (nicht als Trainingsfeature)
- `afrr_activated_mw_pos`
- `afrr_activated_mw_neg`
- `afrr_activation_price_vwap_pos`
- `afrr_activation_price_vwap_neg`
- `afrr_da_price_spread`
- `target_afrr_activation_price_vwap_pos_h1`
- `target_da_price_h1`
- `target_afrr_rate_h1`

## Autoregressive Lags (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `mfrr_active_lag` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Rohwert |
| `picasso_flow_rate_lag_1h` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `picasso_flow_rate_lag_24h` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag24` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag48` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag168` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag_24h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag_48h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `da_price_eur_lag_168h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag24` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag48` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag168` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag_24h` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag_48h` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `wind_onshore_actual_entsoe_lag_168h` | Numerisch | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_pos_lag1` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_pos_lag24` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_pos_lag_1h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_pos_lag_24h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_neg_lag1` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_neg_lag24` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_neg_lag_1h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_price_vwap_neg_lag_24h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_rate_lag1` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_rate_lag24` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_rate_lag_1h` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activation_rate_lag_24h` | Anteil (0–1) | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_pos_lag1` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_pos_lag24` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_pos_lag_1h` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_pos_lag_24h` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_neg_lag1` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_neg_lag24` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_neg_lag_1h` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_activated_mw_neg_lag_24h` | MW | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_da_price_spread_lag1` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_da_price_spread_lag24` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_da_price_spread_lag_1h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |
| `afrr_da_price_spread_lag_24h` | EUR/MWh | Historischer Wert derselben Größe; autoregressives Input-Feature (X). | Lag-Feature (historisch, X) |

## Marktpreise, Spreads & Kosten (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `afrr_capacity_price_neg` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `afrr_capacity_price_pos` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `da_price_eur` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `gas_price` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `coal_price` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `co2_price` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `neighbor_spread_avg` | EUR/MWh | Preisabstand als Knappheits-/Wettbewerbssignal zwischen Marktsegmenten. | Abgeleiteter Rohwert |
| `da_price_eur_slog1p` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Signed-log Transform |
| `relative_price_competitiveness` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |
| `scarcity_price_premium` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Rohwert |

## Systemzustand, Volumen & Kapazitäten (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `mfrr_activated_mw_pos` | MW | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Rohwert |
| `mfrr_activated_mw_neg` | MW | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Rohwert |
| `mfrr_mari_net_mw` | MW | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Rohwert |
| `afrr_capacity_offered_mw_neg` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `afrr_capacity_offered_mw_pos` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `afrr_activation_offered_mw_neg` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `afrr_activation_offered_mw_pos` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `wind_onshore_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `wind_offshore_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `solar_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `gas_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `hard_coal_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `lignite_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `pumped_storage_capacity` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `planned_outages_mw` | MW | Verfügbarkeitsstörung (geplant/ungeplant) als Treiber von Knappheit und Preisstress. | Rohwert |
| `unplanned_outages_mw` | MW | Verfügbarkeitsstörung (geplant/ungeplant) als Treiber von Knappheit und Preisstress. | Rohwert |
| `afrr_capacity_awarded_mw_pos` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |
| `afrr_capacity_awarded_mw_neg` | MW | Volumen-/Kapazitätsinformation zur Angebots- und Systemverfügbarkeitslage. | Rohwert |

## Fundamentaldaten & abgeleitete Signale (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `wind_onshore_actual_entsoe` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `wind_offshore_actual_entsoe` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `solar_actual_entsoe` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `wind_onshore_forecast_id_entsoe` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `wind_offshore_forecast_id_entsoe` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `solar_forecast_id_entsoe` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `NRV_balance` | Numerisch | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Rohwert |
| `generation_hydro_pumped_storage_mw` | MW | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `system_stress_signal` | Index | Modellrelevantes numerisches Signal der finalen Feature-Matrix. | Abgeleiteter Rohwert |
| `wind_total_error_da` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `solar_error_da` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `wind_forecast_update` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `residual_load_calc` | Numerisch | Residuallastsignal als zentrale Knappheitsgröße im Stromsystem. | Abgeleiteter Rohwert |
| `residual_load_forecast` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `renewable_share_forecast` | Anteil (0–1) | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `generation_fossil_total_mw` | MW | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `generation_baseload_total` | Anteil (0–1) | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `generation_hydro_actual_total` | Anteil (0–1) | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Rohwert |
| `wind_onshore_error_da` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `wind_offshore_error_da` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `wind_onshore_forecast_update` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `solar_forecast_update` | Numerisch | Prognose- oder Prognoseupdate-Signal für ex-ante Erwartungsbildung. | Abgeleiteter Rohwert |
| `afrr_activation_rate` | Anteil (0–1) | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Rohwert |
| `wind_onshore_error_id` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `wind_offshore_error_id` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `solar_error_id` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `total_wind_solar_id_error` | Numerisch | Prognosefehler als Unsicherheits- und Stressindikator. | Abgeleiteter Rohwert |
| `grid_stress_index` | Index | Modellrelevantes numerisches Signal der finalen Feature-Matrix. | Abgeleiteter Rohwert |
| `market_regime_picasso` | Index | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Abgeleiteter Rohwert |
| `TE_hour_regime_activation` | Index | Modellrelevantes numerisches Signal der finalen Feature-Matrix. | Abgeleiteter Rohwert |

## Statistik- & Regimefeatures (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `da_price_volatility_30d` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_mean_24h` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_std_24h` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_mean_168h` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_std_168h` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `wind_onshore_actual_entsoe_mean_24h` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Statistisch transformiert |
| `wind_onshore_actual_entsoe_std_24h` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Statistisch transformiert |
| `wind_onshore_actual_entsoe_mean_168h` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Statistisch transformiert |
| `wind_onshore_actual_entsoe_std_168h` | Numerisch | Fundamentalgröße der Erzeugungsseite mit Einfluss auf Regelenergiebedarf. | Statistisch transformiert |
| `nrv_zscore_24h` | Numerisch | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Statistisch transformiert |
| `price_volatility_short_term` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `nrv_quantile_5` | Numerisch | Balancing-/Regelenergiesignal zur Abbildung von Systemungleichgewichten. | Statistisch transformiert |
| `da_price_eur_diff1` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_diff24` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |
| `da_price_eur_ewma24` | EUR/MWh | Preisbezogene Marktinformation mit direkter Relevanz für Merit-Order und Knappheitssignale. | Statistisch transformiert |

## Zeitliche & Kategorische Features (X)
| Feature Name | Einheit | Beschreibung & Rationale | Transformation |
|---|---|---|---|
| `holiday_severity` | Index | Modellrelevantes numerisches Signal der finalen Feature-Matrix. | Rohwert |
| `is_bridge_day` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_christmas_break` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_picasso_regime` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Abgeleiteter Rohwert |
| `is_activated` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `hour_sin` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `hour_cos` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `dayofweek_sin` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `dayofweek_cos` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `month_sin` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `month_cos` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `weekday_sin` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `weekday_cos` | Index [-1,1] | Zyklische Kodierung zeitlicher Muster zur Vermeidung künstlicher Sprünge. | Rohwert |
| `is_weekend` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_payday_period` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_morning` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_afternoon` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_evening` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
| `is_night` | Flag (0/1) | Binärer Regime-/Kalenderindikator für nichtlineare Effekte. | Rohwert |
