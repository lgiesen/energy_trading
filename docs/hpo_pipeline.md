# HPO Pipeline (Target-Specific Default)

## Default behavior
The Makefile default now tunes **all targets** per model family (DA + all aFRR targets).

- `make tune-xgb` -> `HPO_SCOPE=all` by default
- `make tune-linear` -> `HPO_SCOPE=all`
- `make tune-tft` -> `HPO_SCOPE=all`

Targets:
- `target_da_price`
- `target_afrr_activation_price_vwap_pos`
- `target_afrr_activation_price_vwap_neg`
- `target_afrr_activation_rate_pos`
- `target_afrr_activation_rate_neg`
- `target_afrr_capacity_price_pos`
- `target_afrr_capacity_price_neg`

## Target-specific HPO artifacts
Each model/target writes a dedicated JSON/CSV under `artifacts/hpo`.

Examples:
- XGB: `xgb_optuna_afrr_target_afrr_activation_price_vwap_neg.json`
- Linear: `linear_sgd_tuning_afrr_target_afrr_activation_price_vwap_neg.json`
- TFT: `tft_optuna_afrr_target_afrr_activation_price_vwap_neg.json`

TFT trial folders are separated per target via `--run-root artifacts/hpo/tft_trials/<bundle>_<target>`.

## Training uses HPO maps (not one DA file)
Training now supports `--hpo-artifact-map`:

- map format: `{ "target_col": "path/to/hpo.json", ... }`
- DA and each aFRR target uses its own HPO artifact
- missing target in map fails loudly
- `--hpo-artifact` is still supported for single-artifact workflows
- passing both `--hpo-artifact` and `--hpo-artifact-map` fails

## Build/validate HPO map
- Build map:
  - `python3 scripts/build_hpo_artifact_map.py --model-type xgboost --out artifacts/hpo/xgb_hpo_artifact_map.json`
- Validate inventory:
  - `python3 scripts/hpo_inventory.py --validate`

## Fast mode
`HPO_SCOPE` controls tuning scope:
- `all` (default)
- `da`
- `afrr`

Examples:
- `make HPO_SCOPE=da tune-xgb`
- `make tune-xgb-da-only`

## Thesis wording
Hyperparameters were tuned separately per model family and target using the validation split. The negative aFRR activation price target was tuned on the canonical economic provider-value scale.
