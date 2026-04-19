# Feature-Dokumentation und Datenwörterbuch

## Einleitung

Dieses Dokument beschreibt den finalen, kausal abgesicherten Merkmalsraum für die aFRR-Prognose.

- Finale Artefaktdatei: `data/features/all_data_features.parquet`
- Aktueller Stand (Snapshot 2026-03-31): **356 Spalten** im Feature-Artefakt.
- **Primär-Target:** `target_afrr_activation_price_vwap_pos`.  
  **Rationale:** Beschreibt den volumengewichteten durchschnittlichen
  Aktivierungspreis der Stunde _t+1_. Durch den 1h-Versatz wird sichergestellt,
  dass das Modell zum Prognosezeitpunkt _t_ keine Information über die
  tatsächliche Preisbildung der Zielstunde verwendet.
- Trainingsmerkmale `X` werden **modellspezifisch** gebildet:
  - **DA-Bundle:** reduzierte, auktionskausale Feature-Menge (fundamental getrieben).
  - **aFRR-Bundle:** erweiterte Feature-Menge mit Stress-/Spread-Signalen (kurzfristig getrieben).
- `timestamp_utc` ist ein **Metadatum/Zeitindex** und wird nicht als Trainingsmerkmal gezählt.

## Reproduzierbarkeits-Block

| Element                           | Wert                                      |
| --------------------------------- | ----------------------------------------- |
| **Snapshot-Datum**                | 2026-03-31                                |
| **Feature-Artefakt**              | `data/features/all_data_features.parquet` |
| **Artefaktgröße (Snapshot)**      | `45,985` Zeilen, `356` Spalten            |
| **Bundle-Konfiguration**          | `data/model_input/feature_config.json`    |
| **DA-Featureanzahl (Snapshot)**   | `136`                                     |
| **aFRR-Featureanzahl (Snapshot)** | `347`                                     |
| **Regime-Cut**                    | `2022-06-22 22:00:00+00:00`               |
| **DA-Gate**                       | `D-1 13:00 UTC` (`da_price_pit`)          |

**Rebuild-Kommandos (kanonischer Ablauf):**

```bash
./.venv/bin/python -m energy_trading.ingestion.merge_data \
  --data-dir data/raw \
  --out data/processed/all_data.parquet \
  --clip-start 2020-11-30T23:00:00Z \
  --clip-end 2026-03-25T23:00:00Z

./.venv/bin/python scripts/post_collection_pipeline.py \
  --input data/processed/all_data.parquet

./.venv/bin/python -m src.energy_trading.models.prepare_ml_bundles \
  --input data/features/all_data_features.parquet \
  --output-dir data/model_input
```

Hinweis: Die Snapshot-Zahlen sind **run-abhängig** und können sich bei neuem
Datenstand oder geänderter Feature-Logik ändern.

Shape-Lineage-Hinweis:
- Die vollständige Entwicklung von `rows x cols` über alle Pipeline-Stufen
  (raw -> processed -> features -> bundles) wird operativ im Runbook geführt:
  `docs/pipeline_runbook.md` (Abschnitt "Shape lineage across all major artifacts").
- Empfohlenes Laufartefakt: `data/reports/pipeline_shape_lineage.csv`.

## Strategische Design-Entscheidungen

### Runtime-Dynamics-Features im Training (Direct Multi-Output)

Zusätzlich zu den statischen Bundle-Features aus
`data/model_input/feature_config.json` werden im Training
(`src/energy_trading/models/train_xgboost_export.py`) dynamische
Kurzfristfeatures on-the-fly erzeugt:

| Feature | Formel / Aufbau | Einheit | Zweck |
| --- | --- | --- | --- |
| `nrv_velocity_1h` | `NRV_balance_lag_2h - NRV_balance_lag_3h` | MW | Kurzfristige Imbalance-Momentum-Approximation (Steigen/Fallen des NRV). |
| `load_ramp_signed_1h` | bevorzugt `load_forecast_da_entsoe_h1 - load_forecast_da_entsoe` (Fallback: `h2-h1`) | MW | Erfasst Richtung und Stärke der Last-Rampe zwischen benachbarten Forecast-Punkten. |
| `load_ramp_abs_1h` | `abs(load_ramp_signed_1h)` | MW | Rampenintensität unabhängig von Vorzeichen zur Stressmodellierung. |
| `res_load_ramp_signed_1h` | bevorzugt `residual_load_forecast_h1 - residual_load_forecast` (Fallback: `h2-h1`) | MW | Kurzfristige Änderung der prognostizierten Residuallast als Flexibilitätsdruck-Signal. |
| `res_load_ramp_x_wind_total_error_da_lag_2h` | `res_load_ramp_signed_1h * wind_total_error_da_lag_2h` | MW² | Interaktion aus Rampenstress und jüngstem Windfehler als Fragilitätsmerkmal. |

Hinweise:
- Diese Spalten sind **modellspezifische Runtime-Features** und erscheinen
  deshalb nicht notwendigerweise als physische Spalten in den statischen
  Bundle-Parquet-Dateien.
- Wenn Dynamics-Features aktiv sind, werden redundante Mid-Term-Lags
  (`*_lag_4h`, `*_lag_6h`) im Training optional entfernt, um Dimensionalität
  zu reduzieren.

### Erwogene und entfernte Features (aktueller Stand)

Die folgende Übersicht dokumentiert explizit, welche Featurefamilien im
Projektverlauf geprüft und anschließend (je nach Modellpfad) entfernt bzw.
beibehalten wurden.

| Status | Scope | Feature/Pattern | Regelquelle | Begründung |
| --- | --- | --- | --- | --- |
| **Entfernt** | DA-Bundle | `afrr_*`, `mfrr_*`, `nrv_*`, `rz_saldo_*`, `picasso_*`, `mari_*`, `is_activated_*`, `system_stress_*`, `grid_stress_*`, `scarcity_*`, `nrv_zscore_*`, `nrv_quantile_*` | `prepare_ml_bundles.py:get_da_optimized_features` | D-1-Auktionskausalität: kurzfristige Balancing-/Stresssignale sind ex ante für DA nicht zulässig. |
| **Entfernt** | DA-Bundle | Kurzfrist-Lags `*_lag_(1|2|3|6|12)h` | `prepare_ml_bundles.py:get_da_optimized_features` | Vermeidung nicht kausaler Nahzeit-Information im DA-Pfad. |
| **Entfernt** | DA-Bundle | `da_spread_*` ohne Lag sowie `da_spread_*_lag_<24h` | `prepare_ml_bundles.py:get_da_optimized_features` | Bilaterale Spreads im DA-Pfad nur als day-seasonal Memory (>=24h) zugelassen. |
| **Entfernt** | DA-Bundle | `total_wind_solar_id_error*` | `prepare_ml_bundles.py:get_da_optimized_features` | Intraday-Fehlerindikatoren werden aus dem DA-Feature-Set ausgeschlossen. |
| **Entfernt (runtime)** | Training (DA+aFRR Export-Training) | `*_lag_4h`, `*_lag_6h` (wenn Dynamics aktiv) | `train_xgboost_export.py:_add_dynamics_features` | Reduktion redundanter Mid-Term-Lags bei vorhandenen 1h-Momentum/Rampenmerkmalen. |
| **Entfernt (Legacy)** | Transformed-Feature-Layer | Spalten mit `reconstructed` oder `grid_share` | `scripts/check_removed_features.py` | Bereinigung veralteter Featurefamilien; aktueller Audit zeigt `legacy_bad_count=0`. |
| **Erwogen (optional, standardmäßig aus)** | Bundle-Build | PCA auf Forecast-Familien inkl. optionalem Drop der Rohspalten | `prepare_ml_bundles.py` (`use_forecast_pca`, `forecast_pca_drop_raw`) | Redundanzreduktion wurde implementiert, aber standardmäßig deaktiviert, bis robuste CV/PnL-Verbesserung vorliegt. |
| **Erwogen (Ablation), aktuell nicht global entfernt** | aFRR-Modellierung | `cross_border`, `hydro_pumped`, `load_error`, `picasso_flow`, `orderbook_depth` | `scripts/run_feature_ablation.py`, `data/reports/feature_ablation_report.csv` | Gruppenweise Removal wird evidenzbasiert geprüft; Entscheidungen werden nicht pauschal in den Bundle-Bau erzwungen. |

Interpretationshinweis:
- `data/model_input/feature_config.json` beschreibt die **statische**
  Bundle-Selektion.
- Zusätzliche Runtime-Transformationen (z. B. Dynamics-Features und Mid-Term-
  Lag-Pruning) passieren erst im Training und erscheinen daher nicht zwingend
  als physische Spalten im Bundle-Parquet.

### Verzicht auf PCA trotz Multikollinearität

- Referenz: `notebooks/13_forecast_collinearity_pca_audit.ipynb`.
- Ergebnis des Audits: innerhalb der Forecast-Familien (insb. Solar) liegen hohe
  Kollinearitäten vor (u. a. **VIF > 16**).
- **Design-Entscheidung:** bewusster Verzicht auf PCA im finalen Standard-Training.
- **Begründung:**
  1. **XGBoost** ist gegenüber multikollinearen Eingängen robust.
  2. Die **physikalische Interpretierbarkeit** der Rohfeatures bleibt erhalten
     (zentrale Anforderung für die wissenschaftliche Nachvollziehbarkeit der
     Masterarbeit).

### Verzicht auf Skalierung (modellseitig)

- **Design-Entscheidung:** Im finalen XGBoost-Training wird bewusst auf eine
  globale Feature-Skalierung verzichtet.
- **Rationale:** XGBoost ist gegenüber monotonen Skalentransformationen
  weitgehend **invariant**; eine explizite Standardisierung ist für die
  Baum-Split-Logik nicht erforderlich.
- **Methodischer Nutzen:** Der Erhalt physischer Einheiten (z. B. **MW**,
  **EUR/MWh**) verbessert die ökonomische Interpretierbarkeit von
  Feature-Importances und SHAP-Effekten.

### Zyklische Enkodierung zeitlicher Merkmale

- **Features:** `hour_sin/cos`, `dayofweek_sin/cos`, `month_sin/cos`.
- **Rationale:** Abbildung periodischer Zeitachsen auf den Einheitskreis zur
  Wahrung **zirkulärer Kontinuität**.
- **Methodischer Nutzen:** Numerische Nachbarschaft zwischen Periodenenden
  (z. B. Stunde 23 und 00) wird korrekt modellierbar, ohne künstliche
  Diskontinuitäten.

### Modell-Trennung (DA vs. aFRR)

- **DA-Set:** Strikt auf den D-1-Auktionszeitpunkt limitiert; enthält nur
  fundamentale, ex ante verfügbare Merit-Order-Treiber (z. B.
  Commodity-Preise, Forecasts, Kalender- und saisonale Signale).
  **Intraday-/Balancing-Informationen werden ausgeschlossen**, um
  Information-Leakage in die DA-Prognose zu verhindern.
- **aFRR-Set:** Als **Add-on-Modell** auf der fundamentalen DA-Basis
  konzipiert und um transiente Stress-Signale erweitert (z. B.
  `load_error_da_lag_2h`, `NRV_balance_lag_2h`, Spread-/Flow-Signale).
- Zweck der Trennung: höhere **kausale Konsistenz** je Entscheidungszeitpunkt
  und bessere **ökonomische Modelladäquanz**.

| Feature-Set                    | Inhalt                                                                                   | Ausschlüsse / Zusatz                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **DA-Modell-Set**              | Fundamentale Forecasts, Commodity-Preise, Kalender-/Saisonalitätsmerkmale                | Ausschluss von Balancing- und Stress-Signalen gemäß D-1-Kausalität |
| **aFRR-Modell-Set (Full Set)** | DA-Set plus hochfrequente Stress-/Flow-/Spread-Signale (z. B. NRV, Lags, Spreads, Flows) | Ziel: Modellierung der kurzfristigen Abweichung vom DA-Preis       |

## Regeln zur Publikationslatenz (PiT)

| Datengruppe                                                           | Kausalregel                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------- |
| ENTSO-E-Actuals (Erzeugung/Last/Outages/NRV)                          | mindestens `lag_2h`                                             |
| aFRR/mFRR-Aktivierung und Aktivierungspreise                          | mindestens `lag_1h`                                             |
| Netz-Statistiken Stress (`system_stress_signal`, `grid_stress_index`) | explizit nur `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h` |
| Day-Ahead-Preis                                                       | `da_price_pit` mit D-1 13:00-UTC-Freigabelogik                  |
| DA-Forecast-Horizonte (`*_h1..h24`)                                  | publikationsgegated (D-1 13:00 UTC), sonst kausaler Fallback `shift(24)` |
| Forecast-Signale (allgemein)                                          | ex ante nutzbar; keine nicht-kausale Zukunftsauffüllung          |

## Daten-Imputation (methodische Begründung)

- Zur Vermeidung künstlicher Datenknappheit werden installierte Kapazitäten
  (`*_capacity`) im Feature-Bau rückwärts aufgefüllt (`backfill`).
- Der erste verfügbare Meldewert (typisch ab Ende 2023) wird rückwirkend für
  frühere Zeitstempel bis mindestens zum PICASSO-Start
  (`2022-06-22 22:00:00+00:00`) als konstante Strukturgröße verwendet.
- Diese Imputation ist wissenschaftlich vertretbar, da installierte Kapazitäten
  sich nur langsam ändern und die stündliche Marktdynamik über Ist- und
  Aktivierungsdaten modelliert wird.
- Für `generation_baseload_total` gilt:
  `generation_baseload_total = biomass_actual_entsoe + generation_nuclear_mw`
  (fehlende Werte in `generation_nuclear_mw` werden als `0` gesetzt).
- Interpretation: Seit dem deutschen Atomausstieg (letzte Abschaltungen am 15. April 2023; im Stundenraster ab den Folgestunden) ist der Nuclear-Anteil
  faktisch `0`. Das Merkmal wirkt in der jüngeren Periode daher als
  biomass-dominierter Baseload-Proxie.

### Beobachtete Imputationen (letzter Bundle-Run)

Die operative Imputation in `prepare_ml_bundles.py` erfolgt auf `X` je Split
mit `ffill(limit=12)` und anschließendem train-fitted Median-Fallback. Bereits
in `handle_missing_values.py` gelten folgende Spezialfälle:
- Commodity-Preise (`co2_price`, `gas_price`, `coal_price`): `ffill()` ohne Limit
  plus `bfill()` nur für verbleibende führende Startlücken
- Strukturkapazitäten (`*_capacity`): `ffill()` ohne Limit
- `da_price_BE`: Fallback auf exakt gleichen UTC-Stundenwert des Vortags (`t-24h`)
Die folgenden Spalten wurden im letzten Lauf tatsächlich betroffen geloggt:

| Bundle / Split | Betroffene Spalten (Auszug)                                                                                                                                                                                                                                                                                                                                                                                                                                          | Methode                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| DA / test      | `da_price_BE`, `gas_price`, `co2_price`                                                                                                                                                                                                                                                                                                                                                                                                                              | `ffill(limit=12)`; Rest per Train-Median                           |
| DA / test      | `wind_onshore_capacity`, `wind_offshore_capacity`, `solar_capacity`, `gas_capacity`, `hard_coal_capacity`, `lignite_capacity`, `pumped_storage_capacity`                                                                                                                                                                                                                                                                                                             | primär `ffill(limit=12)` (lange Blöcke), kein/kaum Median-Fallback |
| aFRR / train   | `wind_total_error_da_lag_2h`                                                                                                                                                                                                                                                                                                                                                                                                                                         | geringes `ffill`, kein Median-Fallback                             |
| aFRR / test    | wie DA-Commodity/Capacity plus `afrr_activated_mw_pos_lag_1h`, `afrr_activated_mw_neg_lag_1h`, `mfrr_activated_mw_pos_lag_1h`, `mfrr_activated_mw_neg_lag_1h`, `mfrr_mari_net_mw_lag_1h`, `afrr_capacity_offered_mw_pos_lag_1h`, `afrr_capacity_offered_mw_neg_lag_1h`, `afrr_capacity_awarded_mw_pos_lag_1h`, `afrr_capacity_awarded_mw_neg_lag_1h`, `afrr_activation_offered_mw_pos_lag_1h`, `afrr_activation_offered_mw_neg_lag_1h`, `wind_total_error_da_lag_2h` | überwiegend kleines `ffill`, i. d. R. ohne Median-Fallback         |

Hinweis: Die exakten Imputations-Counts sind run-abhängig und werden pro Lauf
in den Reports dokumentiert:

- `data/model_input/da/feature_quality_report.csv`
- `data/model_input/afrr/feature_quality_report.csv`
- `data/reports/feature_quality_report_all.csv`

## Metadaten und ausgeschlossene Spalten

| Typ                                                    | Spalten                                                                                                                                                           |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Metadatum/Index                                        | `timestamp_utc`                                                                                                                                                   |
| Technische Metadaten (aus `X` ausgeschlossen)          | `data_is_lagged`, `is_local_reconstruction_only`, `pit_lagged_column_count`                                                                                       |
| Zielvariablen/Outcome-Spalten (aus `X` ausgeschlossen) | `target_afrr_activation_price_vwap_pos`, `target_afrr_activation_price_vwap_neg`, `target_afrr_activation_rate_pos`, `target_afrr_activation_rate_neg`, `target_afrr_capacity_price_pos`, `target_afrr_capacity_price_neg`, `target_da_price` |

Hinweis: `target_afrr_capacity_price_pos` und
`target_afrr_capacity_price_neg` sind reine **Label-Spalten** (`y`) und
werden strikt aus dem Feature-Set (`X`) ausgeschlossen.

## Spezifische Feature-Logik & Rationale

| **Feature-Klasse**                                                                           | **Rationale**                                                                                                                              | **Zweck**                                                                                                                  |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **Symmetric Log Transformation** (`da_price_slog1p`)                                         | Kompression extremer Preisspikes bei Erhalt des Vorzeichens.                                                                               | Stabilisierung der Gradienten-Berechnung in XGBoost und Reduktion von Outlier-Rauschen.                                    |
| **Pumped-Hydro-Isolierung** (`generation_hydro_pumped_storage_mw_lag_2h`)                    | Trennung preisreaktiver Flexibilität (Pumpspeicher) von unflexiblen Laufwasser-/Aggregatanteilen, um Signal-Verwässerung zu vermeiden.     | Erkennung taktischer Flex-Aktivierung; hohe Einspeisung signalisiert Grenz-Kapazitätsnutzung und erhöhte aFRR-Volatilität. |
| **Renewable-Share-Forecast** (`renewable_share_forecast`)                                    | Maß der relativen Verdrängung konventioneller Flexibilität in der Merit-Order; informativer als absolute Einzel-MW-Werte.                  | Robuster Indikator für Merit-Order-Verschiebungen und Flexibilitätsbedarf.                                                 |
| **EWMA-Derivate** (`*_ewma24`)                                                               | Stärkere Gewichtung rezenter Preisinformationen gegenüber älteren Beobachtungen.                                                           | Schnellere Reaktion auf Marktmomentum und Regimewechsel im Intraday-Umfeld.                                                |
| **DA-Forecast-Kurvenfeatures** (`*_h1,_h2,_h3,_h6,_h12,_h24`)                               | Explizite Abbildung der erwarteten Forecast-Trajektorie für den nächsten Tag statt rein punktueller Ex-ante-Werte.                         | Verbessert Multi-Horizon-Prognosen durch Sichtbarkeit von Verlauf, Rampen und Tagesprofilen im Prognosefenster.             |
| **Komprimierte Forecast-Kurven** (`*_next24_*`)                                              | Verdichtung der 24h-Trajektorie in Lage-/Streuungs- und Rampenmaße (`mean/min/max/std`, `ramp`) zur Reduktion von Dimensionalität.        | Robuste Erfassung von Trend- und Volatilitätsmustern bei kontrollierter Modellkomplexität (insb. für Tree-Modelle).         |
| **Load-Error-Feature** (`load_error_da_lag_2h`)                                              | Unvorhergesehene Lastschwankungen sind primäre Treiber kurzfristiger System-Ungleichgewichte.                                              | Direkter Prädiktor für die Aktivierung von Regelenergie (aFRR).                                                            |
| **Cross-Border-Spreads** (`da_spread_de_at/de_fr/de_nl`, inkl. Lags)                         | Abbildung von Import-/Exportdruck und Kopplungsgrad benachbarter Day-Ahead-Märkte.                                                         | Zusätzliche Erklärungskraft für Preis- und Spread-Regime durch grenzüberschreitende Arbitrage-/Engpasssignale.             |
| **PICASSO-Regime-Flag** (`is_picasso_active`) | Abbildung der aktiven PICASSO-Marktphase (`ab Juli 2024`) als struktureller Regimeanker. | Ermöglicht dem Modell die Trennung zwischen Vor-PICASSO- und PICASSO-gekoppelter europäischer Preissetzungslogik.          |
| **PiT-Latenzregel für ENTSO-E-Actuals** (`*_lag_2h`)                                          | Physische Istwerte sind zum Entscheidungszeitpunkt nicht sofort stabil publiziert.                                                          | Vermeidung von Information-Leakage durch konsistente kausale Verzögerung.                                                   |
| **DA-Gate-Logik** (`da_price_pit`)                                                            | Day-Ahead-Auktionsergebnisse sind erst nach dem D-1-Gate (`13:00 UTC`) verfügbar.                                                          | Kausal korrekte Abbildung der Informationsverfügbarkeit für DA- und aFRR-nahe Features.                                    |
| **Stress-Signale ohne `lag_1h`** (`system_stress_signal`, `grid_stress_index`)               | Diese Kennzahlen basieren auf Actual-/NRV-nahen Quellen mit zusätzlicher Publikationslatenz.                                               | Robuste PiT-Integrität durch Start bei `lag_2h` (statt kurzfristig leakage-anfälligem `lag_1h`).                           |
| **Strict Target Policy** (`target_*`)                                                         | Unverschobene Outcome-Reihen würden Nowcasting statt Forecasting erzwingen.                                                                  | Trennung von `X` und `y` entlang der Zeitachse; Prognoseziel bleibt strikt `t+1`.                                          |

### Gezielte zusätzliche Intraday-Lags (Update)

| Familie / Spalten                                                                                          | Neue bzw. erweiterte Lags          | Begründung (kurz)                                                                                 |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `NRV_balance`, `nrv_zscore_24h`, `nrv_quantile_5`                                                          | `2h, 3h, 4h, 6h, 12h, 24h`         | Kurzfristige Stressdynamik (Rampen/Abklingen) wird besser erfasst; kein `1h` wegen PiT-Latenz. |
| `afrr_activation_rate_pos`, `afrr_activation_rate_neg`, `is_activated`, `mfrr_active_lag`                | `1h, 2h, 3h, 6h, 12h, 24h`         | Aktivierungsintensität zeigt ausgeprägte Intraday-Momentum-Cluster, richtungsgetrennt für POS/NEG. |
| `afrr_activated_mw_pos/neg`, `mfrr_activated_mw_pos/neg`, `mfrr_mari_net_mw`, `afrr_activation_offered_*` | `1h, 2h, 3h, 6h, 12h, 24h`         | Mengen- und Flussverläufe sind kurzfristig persistent und für aFRR-Preis-/Rate-Prognosen relevant. |
| `afrr_capacity_awarded_*`, `afrr_capacity_offered_*`, `afrr_capacity_price_*`                              | `1h, 2h, 3h, 6h, 12h, 24h`         | Auktionsergebnisse und Capacity-Marktspannung wirken typischerweise über mehrere Stunden nach.   |
| `wind_forecast_update`, `wind_onshore_forecast_update`, `solar_forecast_update`, `*_error_da`             | `1h, 2h, 3h, 6h, 12h, 24h`         | Forecast-„News“-Signale entfalten ihren Effekt nicht nur instantan, sondern über den Folgetag.   |

## Komprimiertes Datenwörterbuch (Merkmalsfamilien)

Hinweis zur Vollständigkeit: Die nachfolgenden Merkmalsfamilien decken den
finalen Artefaktumfang von **356 Spalten** vollständig ab (inklusive
Target-Spalten, Metadaten-/Governance-Variablen und modellrelevanter
Feature-Gruppen). Die Darstellung ist bewusst gruppiert, nicht zeilenweise je
Einzelspalte.

| Merkmal (verfügbare Lags)                                                                                                                                                            | Einheit      | Beschreibung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `afrr_activation_price_vwap_pos` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                               | EUR/MWh      | Positiver aFRR-Aktivierungspreis (VWAP) als zentrales Signal für kurzfristige Preisregime und Wochenmuster.                                                                                                                                                                                                                                                                                                                                                                                            |
| `afrr_activation_price_vwap_neg` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                               | EUR/MWh      | Negativer aFRR-Aktivierungspreis (VWAP); bildet asymmetrische Balancing-Kosten gegenüber POS-Preisen ab.                                                                                                                                                                                                                                                                                                                                                                                               |
| `afrr_da_price_spread` (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                         | EUR/MWh      | Differenz zwischen aFRR-Aktivierungspreis und Day-Ahead-Preis als direkter Opportunitätskosten-Indikator.                                                                                                                                                                                                                                                                                                                                                                                              |
| `da_price_pit` (ohne Lag, 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)                                                                                                                       | EUR/MWh      | PiT-gegatterter Day-Ahead-Preis, der nur nach Veröffentlichungszeitpunkt modellseitig verfügbar ist.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `da_price` (24h, 48h, 168h)                                                                                                                                                          | EUR/MWh      | Historische Day-Ahead-Preisniveaus zur Erfassung stabiler Tages- und Wochenzyklen.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `gas_price`, `coal_price`, `co2_price` (ohne Lag)                                                                                                                                    | EUR/MWh      | Brennstoff- und Emissionskosten als exogene Kostentreiber der Merit-Order und Strompreisdynamik.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `da_price_slog1p`, `da_price_diff1`, `da_price_diff24`, `da_price_ewma24` (ohne Lag)                                                                                                 | EUR/MWh      | Transformierte Preisniveaus und Preisänderungen zur robusteren Abbildung von Sprüngen und Kurzfristtrends.                                                                                                                                                                                                                                                                                                                                                                                             |
| `da_price_mean_24h`, `da_price_std_24h`, `da_price_mean_168h`, `da_price_std_168h`, `da_price_volatility_30d` (ohne Lag)                                                             | EUR/MWh      | Rollierende Mittel-/Volatilitätsmaße zur Quantifizierung von Marktregimewechseln und Unsicherheit.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `system_stress_signal` (2h, 3h, 6h, 12h, 24h)                                                                                                                                        | Index        | Verdichtetes Stresssignal aus Systemungleichgewichten; nur mit 2h+ Latenz gemäß PiT-Regel nutzbar.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `grid_stress_index` (2h, 3h, 6h, 12h, 24h)                                                                                                                                           | Index        | Kompositindex für Netzanspannung, der mehrere Belastungskomponenten in einen robusten Steuerindikator bündelt.                                                                                                                                                                                                                                                                                                                                                                                         |
| `nrv_zscore_24h` (2h, 3h, 4h, 6h, 12h, 24h)                                                                                                                                          | Index        | Standardisierte Abweichung des NRV gegenüber dem 24h-Verlauf zur Identifikation ungewöhnlicher Balance-Zustände.                                                                                                                                                                                                                                                                                                                                                                                       |
| `nrv_zscore_24h_lag_2h`                                                                                                                                                              | Index        | PiT-konformer Kurzfristindikator für akute Netzanspannung und Balancing-Wahrscheinlichkeit in der Folgestunde.                                                                                                                                                                                                                                                                                                                                                                                         |
| `nrv_quantile_5` (2h, 3h, 4h, 6h, 12h, 24h)                                                                                                                                          | Index        | Quantilisierte NRV-Lage (5 Klassen) zur robusten Regimekodierung auch bei Ausreißern.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `NRV_balance` (2h, 3h, 6h, 12h, 24h)                                                                                                                                                 | MW           | Netto-Regelverbundsaldo als physikalisches Kernsignal für Systemüberschuss oder -defizit.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `neighbor_spread_avg`, `relative_price_competitiveness`, `price_volatility_short_term`, `scarcity_price_premium` (je 2h)                                                             | EUR/MWh      | Abgeleitete Wettbewerbs-, Knappheits- und Volatilitätsindikatoren zur Erklärung kurzfristiger Preisaufschläge.                                                                                                                                                                                                                                                                                                                                                                                         |
| `load_error_da_lag_2h`                                                                                                                                                               | MW           | Lastfehler (Ist minus Prognose) mit PiT-konformer Verzögerung als direkter Treiber kurzfristiger Systemungleichgewichte.                                                                                                                                                                                                                                                                                                                                                                               |
| `da_spread_de_at_lag_2h/24h/48h/168h`, `da_spread_de_fr_lag_2h/24h/48h/168h`, `da_spread_de_nl_lag_2h/24h/48h/168h`                                                                  | EUR/MWh      | Bilaterale DE-Nachbarland-Spreads als Proxy für grenzüberschreitenden Import-/Exportdruck und Arbitragespannungen; **kein `lag_1h`** aufgrund der D-1-Auktionskausalität.                                                                                                                                                                                                                                                                                                                              |
| `afrr_activated_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                               | MW           | Tatsächlich aktivierte aFRR-Leistung als direktes Maß für den realen Regelenergiebedarf im System.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `afrr_activated_mw_pos_lag_1h`                                                                                                                                                       | MW           | Kurzfristiger Aktivierungsimpuls als Momentum-Signal für positive Regelleistungsnachfrage.                                                                                                                                                                                                                                                                                                                                                                                                             |
| `afrr_capacity_awarded_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                        | MW           | Bezuschlagte aFRR-Vorhaltemengen als Information über erwartete Balancing-Anforderungen.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `afrr_activation_offered_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h)                                                                                                                      | MW           | Angebotsseitige Aktivierungsmengen als Liquiditäts- und Spannungsindikator der Balancing-Märkte.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `afrr_activation_rate_pos`, `afrr_activation_rate_neg` (je 1h, 2h, 3h, 6h, 12h, 24h)                                                                                               | Anteil (0-1) | Richtungsgetrennte Aktivierungsraten (POS/NEG) als Verhältnis aktivierter zu vorgehaltener aFRR-Leistung; zentrale Intensitätssignale für kurzfristige Regelenergiebeanspruchung.                                                                                                                                                                                                                                                                                                                     |
| `is_activated` (1h)                                                                                                                                                                  | Flag (0/1)   | Binärer Aktivierungsstatus zur Trennung von Aktivierungs- und Nicht-Aktivierungsstunden.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `mfrr_activated_mw_pos/neg` (1h, 2h, 3h, 6h, 12h, 24h), `mfrr_mari_net_mw` (1h, 2h, 3h, 6h, 12h, 24h), `mfrr_active_lag` (1h, 2h, 3h, 6h, 12h, 24h) | MW / Index   | mFRR-Aktivierung und MARI-Flüsse als ergänzende Balancing-Signale für systemweite Reserveanspannung.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `residual_load_actual`, `residual_load_calc` (je 2h)                                                                                                                                 | MW           | Reallast abzüglich erneuerbarer Einspeisung als zentraler Treiber konventioneller Fahrweise.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `generation_fossil_total_mw_lag_2h`, `generation_hydro_pumped_storage_mw_lag_2h`, `generation_hydro_actual_total_lag_2h`, `generation_baseload_total_lag_2h`                         | MW / Anteil  | Aggregierte Erzeugungsblöcke zur Abbildung der Angebotsstruktur; `generation_hydro_pumped_storage_mw_lag_2h` wird **explizit separat** geführt, da Pumpspeicher kurzfristig steuerbare Flexibilität (Laden/Entladen) abbilden und damit für aFRR-Regime deutlich informativer sind als reine Laufwasser-/Gesamthydro-Signale. `generation_baseload_total_lag_2h` ist definiert als Biomasse + Kernkraft (`nuclear`-Missing -> `0`) und seit dem Atomausstieg im April 2023 faktisch biomassedominiert. |
| `wind_onshore_actual_entsoe` (2h, 24h, 48h, 168h)                                                                                                                                    | MW           | Tatsächliche Onshore-Winderzeugung mit Kurz- bis Wochenhistorie zur Modellierung wettergetriebener Regime.                                                                                                                                                                                                                                                                                                                                                                                             |
| `wind_offshore_actual_entsoe`, `solar_actual_entsoe` (je 2h)                                                                                                                         | MW           | Tatsächliche Offshore-Wind- und Solarleistung als unmittelbare Determinanten der Residuallast.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `wind_onshore_error_da/id`, `wind_offshore_error_da/id`, `solar_error_da/id`, `wind_total_error_da`, `total_wind_solar_id_error` (je 2h)                                             | MW           | Forecast-Fehler gegenüber Istwerten als Proxy für Prognoseunsicherheit und spätere Regelenergiebedarfe.                                                                                                                                                                                                                                                                                                                                                                                                |
| `wind_onshore_actual_entsoe_mean_24h/std_24h/mean_168h/std_168h` (je 2h)                                                                                                             | MW           | Rollierende Lage- und Streuungsmaße der Onshore-Einspeisung zur Stabilisierung der Windregime-Erkennung.                                                                                                                                                                                                                                                                                                                                                                                               |
| `unplanned_outages_mw` (2h), `planned_outages_mw` (ohne Lag)                                                                                                                         | MW           | Ungeplante und geplante Kraftwerksausfälle als Angebotsrestriktionssignal im kurzfristigen Dispatch.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `wind_onshore_forecast_id_entsoe`, `wind_offshore_forecast_id_entsoe`, `solar_forecast_id_entsoe` (ohne Lag, 24h, 48h, 168h)                                                         | MW           | Ex-ante Einspeiseprognosen für erneuerbare Energien zur frühzeitigen Abbildung erwarteter Volatilität.                                                                                                                                                                                                                                                                                                                                                                                                 |
| `renewable_share_forecast`, `residual_load_forecast` (ohne Lag, 24h, 48h, 168h)                                                                                                      | Anteil / MW  | Prognostizierter EE-Anteil und erwartete Residuallast als Schlüsselgrößen für Day-Ahead- und Balancing-Lage.                                                                                                                                                                                                                                                                                                                                                                                           |
| `load_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `wind_onshore_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `wind_offshore_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `solar_forecast_da_entsoe_h1/h2/h3/h6/h12/h24`, `residual_load_forecast_da_h1/.../h24`, `renewable_share_forecast_h1/.../h24` | MW / Anteil  | Sparse DA-Forecast-Horizonte (1..24h) als kausal gegatete Trajektorienpunkte zur expliziten Multi-Horizon-Signalgebung.                                                                                                                                                                                                                                                                                                                                                                                |
| `*_next24_mean/min/max/std`, `*_next24_ramp` (für DA-forecastbasierte Kernreihen)                                                                                | MW / Anteil  | Komprimierte Kurvenbeschreibung der erwarteten 24h-Entwicklung; reduziert Feature-Flut bei Erhalt der Forminformation (Level, Streuung, Rampen).                                                                                                                                                                                                                                                                                                                                                       |
| `wind_forecast_update`, `wind_onshore_forecast_update`, `solar_forecast_update` (1h, 2h, 3h, 6h, 12h, 24h)                                                                          | Index        | Änderungsmaße der Forecasts als Frühindikator für neue Wetterinformationen und Repricing-Risiken.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `wind_onshore_capacity`, `wind_offshore_capacity`, `solar_capacity`, `gas_capacity`, `hard_coal_capacity`, `lignite_capacity`, `pumped_storage_capacity` (ohne Lag)                  | MW           | Verfügbare Kapazitäten als strukturelle Obergrenzen des Erzeugungs- und Flexibilitätspotenzials.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `picasso_flow_rate` (lag_1h, lag_24h)                                                                                                                                                | Anteil (0-1) | Anteil grenzüberschreitender PICASSO-Aktivierung als Indikator für europäische Kopplungseffekte.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `TE_hour_regime_activation` (1h)                                                                                                                                                     | Index        | Zeitregime-basierte Aktivierungskodierung zur expliziten Trennung typischer Stundenmuster.                                                                                                                                                                                                                                                                                                                                                                                                             |
| `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `weekday_sin`, `weekday_cos`, `month_sin`, `month_cos` (ohne Lag)                                                          | Index        | Zyklische Zeitkodierungen mit zirkulärer Kontinuität (insb. 23:00 -> 00:00) zur robusten Modellierung periodischer Muster.                                                                                                                                                                                                                                                                                                                                                                             |
| `is_weekend`, `is_afternoon`, `is_evening`, `is_morning`, `is_night`, `is_bridge_day`, `is_payday_period`, `is_christmas_break`, `is_picasso_active` (ohne Lag) | Flag (0/1)   | Binäre Regimeindikatoren für kalender- und marktstrukturbedingte Nachfragemuster und Aktivierungswahrscheinlichkeiten.                                                                                                                                                                                                                                                                                                                                                                                 |
| `holiday_severity` (ohne Lag)                                                                                                                                                        | Index        | Verdichteter Kalenderindex zur robusten Trennung außergewöhnlicher Tage und Betriebsphasen.                                                                                                                                                                                                                                                                                                                                                                                                             |

## Feature-Taxonomie (Informationsquellen)

| Kategorie           | Typische Merkmale                                       | Informationsfunktion                                                      |
| ------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Fundamental**     | Last, Erneuerbare, Outages, Commodity-Preise            | Physikalische und kostengetriebene Basiskräfte der Merit-Order.           |
| **Markt-Sentiment** | DA-Preise (historisch/PiT), Spreads, Volatilitäten      | Preisregime, relative Bewertung und Erwartungsdynamik im Markt.           |
| **Echtzeit-Stress** | NRV, Aktivierungsraten, Netz-Statistiken, PICASSO-Flows | Kurzfristige Systemanspannung und Balancing-Bedarf.                       |
| **Kalendarium**     | Feiertage, Brückentage, zyklische Zeitgeber             | Strukturierte saisonale und verhaltensbedingte Nachfrage-/Angebotsmuster. |

## Feature-Taxonomie der Modell-Bundles

| Bundle          | Taxonomie                 | Inhaltlicher Fokus                                                                                                                      |
| --------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **DA-Bundle**   | **Fundamental & Ex-Ante** | Last- und EE-Forecasts, Commodity-Preise, Kalender-/Saisonalitätsmerkmale; keine Balancing-/Stress-Signale gemäß D-1-Kausalität.        |
| **aFRR-Bundle** | **Stress & Momentum**     | DA-Basis plus kurzfristige Stress-/Momentum-Signale (NRV, Aktivierungs-Lags, Spreads, Flows) zur Prognose der Abweichung vom DA-Niveau. |

## Kausalitäts-Check zur Lag-Benennung

- Alle Lag-Spalten folgen dem Schema `*_lag_Xh`.
- `X` beschreibt die absolute Verzögerung zur Echtzeit.
- Für `system_stress_signal` und `grid_stress_index` existiert **kein** `lag_1h`.

## Validierungsmethodik

| Methode                        | Konfiguration                                                       | Zweck                                                                                                                      |
| ------------------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Purged Cross-Validation**    | `72h` Gap zwischen Train- und Validierungsfenster                   | Ausschluss zeitlicher Leakage-Pfade durch Autokorrelation in Wetter-, Last- und Preissignalen.                             |
| **Ablation-Tests**             | Gruppenweise Feature-Entfernung/Beibehaltung auf identischen Splits | Quantifizierung des inkrementellen Nutzens einzelner Feature-Klassen.                                                      |
| **PnL-Proxy (Spread-Capture)** | Fold-weise Logging im CV-Loop                                       | Prüfung, ob ein Feature nicht nur Fehlermaße verbessert, sondern auch ökonomisch verwertbare Richtungsinformation liefert. |

## Methodische Governance und Evidenz

| Thema                             | Umgesetzte Regel                                                                                                                 | Nutzen für Nachvollziehbarkeit                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Regime-Cut (Post-PICASSO)**     | Trainings-Bundles werden ab `2022-06-22 22:00:00+00:00` gefiltert.                                                               | Verhindert Regime-Mischung zwischen historischer und aktueller Marktdynamik.    |
| **Strict Target Policy**          | Für Training sind ausschließlich `target_*` als Zielvariablen zulässig; unshifted Ziel-nahe Reihen dienen nur Audit/`y_true`. | Stellt sicher, dass Forecasting (h+1) nicht in Nowcasting kippt.                |
| **DA-vs-aFRR Feature Governance** | DA-Bundle wird um Balancing-/Stress-Signale bereinigt; aFRR-Bundle behält diese Signale als zentrale Treiber.                    | Klare kausale Trennung nach Entscheidungszeitpunkt und Modellzweck.             |
| **Cross-Border-DA-Spreads (PiT)** | `da_spread_de_at/de_fr/de_nl` basieren auf PIT-gegatterten DA-Preisen (`D-1 13:00 UTC`), nicht auf Rohpreisen.                   | Vermeidet versteckte Leakage-Pfade in grenzüberschreitenden Preisrelationen.    |
| **Imputation Governance**         | Bundle-seitig: `ffill(limit=12)` auf `X` plus train-fitted Median-Fallback; Logging je Spalte mit Imputation-Counts.             | Transparente, reproduzierbare Behandlung kleiner Quelllücken ohne Ziel-Leakage. |
| **Ablation-Prozess**              | Feature-Gruppen werden nur übernommen, wenn sie in Purged-CV und Holdout robusten Mehrwert liefern.                              | Datengetriebene Feature-Auswahl statt heuristischer Überfrachtung.              |

## Primärquellen-Lücken und April-2025-Forensik

- Zentrale Missingness-Dokumentation: `docs/api_missingness_report.md` und
  `data/reports/api_missingness_audit.csv`.
- Für **2025-04-01 00:00 bis 2025-04-02 23:00 UTC** wurde ein gezielter
  Re-Fetch durchgeführt (`data/reports/april_refetch_comparison.csv`).
- Befund: `wind_onshore_forecast_id_entsoe` blieb mit **22 fehlenden Werten**
  unverändert; Klassifikation als **Hard Source Gap** (Primärquellenlimit),
  nicht als Ingestionsfehler.
- Nachweis der Lag-Propagation: `data/reports/april_hard_gap_propagation.csv`.

## Empirische Validität

Zur empirischen Begründung der finalen Trainingsmerkmale werden nach dem
XGBoost-Training zwei komplementäre Wichtigkeitsmaße exportiert:

- **XGBoost Gain**: Informationsgewinn je Feature im Baumwachstum.
- **SHAP (mean absolute values)**: durchschnittlicher marginaler Beitrag je
  Feature zur Vorhersage.

Erzeugte Artefakte:

- `data/reports/model_training/importance_report.csv`
- `data/reports/model_training/shap_summary.png`

Diese Nachweise dienen als prüfbare Evidenz im Methodik-/Ergebnis-Kapitel,
dass die verwendeten `X`-Merkmale (inkl. Lag-Struktur) nicht nur formal
kausal definiert, sondern auch modellseitig substanziell wirksam sind.

## Parameter-Rationale

| Parameter / Feature-Logik          | Konfiguration                                                    | Methodische Begründung                                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tages-/Wochen-Saisonalität         | Fenster `24h` und `168h` (z. B. Mittelwert/Std/Lags)             | Strom- und Regelenergiemärkte zeigen stabile Intraday- und Wochenrhythmen (Lastprofile, Wochenend-Effekt). Die Fenster bilden diese wiederkehrenden Muster explizit ab. |
| PiT-Latenz für Actuals             | `lag_2h` für ENTSO-E-Istwerte/abgeleitete Actual-Features        | Vermeidung von Look-ahead-Bias durch Publikationslatenz: Istwerte gelten erst mit Verzögerung als beobachtbar.                                                          |
| PiT-Latenz für aFRR/Marktreaktion  | `lag_1h` für aFRR-Aktivierungs-/preisnahe und Kapazitäts-Signale | Marktsignale werden erst nach Ablauf/Publikation der Stunde als sicher verfügbar behandelt.                                                                             |
| DA-Informationsgate                | `da_price_pit` mit D-1 `13:00 UTC` Freigabelogik                 | Strikte Abbildung der realen Informationsverfügbarkeit vor Lieferstunde; verhindert Vorwissen aus noch nicht publizierten DA-Auktionswerten.                            |
| DA-Forecast-Trajektorien           | Sparse Horizonte `h1,h2,h3,h6,h12,h24` plus `next24`-Kompression | Vereint Kurvenform (Rampen/Volatilität) und Dimensionalitätskontrolle für Multi-Horizon-Modelle ohne Bruch der PiT-Kausalität.                                           |
| DA-Derivate (Diff/EWMA/Stats/Slog) | Berechnung auf `da_price_pit` statt Roh-`da_price`               | Konsistente PiT-Kausalität für alle aus DA abgeleiteten Merkmale; verhindert indirekte Leakage-Pfade über Derivate.                                                     |
| Load Error                         | `load_error_da_lag_2h`                                           | Direkter Ungleichgewichtsindikator aus Last-Ist vs. Last-Prognose; kausal verzögert zur Vermeidung von Leakage.                                                         |
| Cross-Border-Spreads               | `da_spread_de_at/de_fr/de_nl` (inkl. Lags)                       | Erfasst Kopplungsdruck zwischen DE und Nachbarmärkten (Import/Export, Arbitrage, Engpässe).                                                                             |
| Pumped-Hydro-Isolierung            | `generation_hydro_pumped_storage_mw_lag_2h`                      | Separates Flexibilitäts-Signal für taktische Speicherfahrweise; verhindert Signal-Verwässerung in aggregierten Hydro-Blöcken.                                           |
| 30-Tage-Volatilität                | `da_price_volatility_30d` (rollierend, kausal verschoben)        | Erfasst mittelfristige Regimewechsel und Stressphasen, die kurzfristige Preis- und Aktivierungsdynamik beeinflussen.                                                    |
| Signed-Log-Transformation          | `slog1p(x) = sign(x) * log1p(abs(x))`                            | Komprimiert Preisspitzen (positive und negative) ohne Vorzeichenverlust; stabilisiert numerische Gradienten und reduziert Dominanz extremer Ausreißer.                  |

Hinweis zur Modellierung:

- Die in `notebooks/13_forecast_collinearity_pca_audit.ipynb` identifizierte
  Kollinearität (insb. Solar-Forecast-Familie) wird für XGBoost bewusst
  beibehalten, da Baumverfahren robust gegenüber kollinearen Eingängen sind
  und Rohfeatures für die ökonomische Interpretation erhalten bleiben.
