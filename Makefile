SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Global reproducibility / stability settings
export PYTHONUNBUFFERED := 1
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1

SEED ?= 42
FORECAST_HOURS ?= 48
SIM_HORIZON_HOURS ?= auto
export IS_SMOKE_TEST ?= 0
SIM_SMOKE_DAYS ?= 7
DEVICE ?= cuda
TFT_PRECISION ?= bf16-mixed
LEAD_WEIGHT_START ?= 16
LEAD_WEIGHT_END ?= 48
LEAD_WEIGHT_MAX ?= 2.0
LINEAR_AFRR_PARALLEL_JOBS ?= 1
LINEAR_LEAD_PARALLEL_JOBS ?= 4
SIM_QUANTILE_SWEEP_DEFAULT ?= p50-p50,p30-p70,p10-p90,p10-p30,p30-p50,p50-p70,p70-p90
SIM_QUANTILE_PAIRS ?= $(SIM_QUANTILE_SWEEP_DEFAULT)
SIM_DA_ROLES ?= low mid high
DA_QUANTILE_ROLE ?= mid
GRID_DA_ROLES ?= low mid high
GRID_STRATEGIES ?= multi da_only afrr_only
GRID_QUANTILE_PAIRS ?= $(SIM_QUANTILE_SWEEP_DEFAULT)
GRID_SMOKE_HOURS ?= 24
SIM_GRID_STAMP ?= $(shell date +%Y%m%d_%H%M%S)

# Run IDs for training outputs (evaluated once per make invocation)
RUN_ID_XGB := xgb_$(shell date +%Y%m%d_%H%M%S)
RUN_ID_LINEAR := linear_$(shell date +%Y%m%d_%H%M%S)
RUN_ID_TFT := tft_$(shell date +%Y%m%d_%H%M%S)

# Common artifacts
DATA_HASH_FILE := artifacts/hpo/data_model_input.md5

# HPO outputs
XGB_TUNE_JSON := artifacts/hpo/xgb_optuna_da_target_da_price.json
XGB_TUNE_CSV := artifacts/hpo/xgb_optuna_da_target_da_price_trials.csv
LINEAR_TUNE_JSON := artifacts/hpo/linear_sgd_tuning_da_target_da_price.json
LINEAR_TUNE_CSV := artifacts/hpo/linear_sgd_tuning_da_target_da_price_trials.csv
TFT_TUNE_JSON := artifacts/hpo/tft_optuna_da_target_da_price.json
TFT_TUNE_CSV := artifacts/hpo/tft_optuna_da_target_da_price_trials.csv

# Train outputs
XGB_MANIFEST := artifacts/model_runs/$(RUN_ID_XGB)/manifest.json
LINEAR_MANIFEST := artifacts/model_runs/$(RUN_ID_LINEAR)/manifest.json
TFT_MANIFEST := artifacts/model_runs/$(RUN_ID_TFT)/manifest.json

# Simulation markers
XGB_SIM_DONE := artifacts/model_runs/$(RUN_ID_XGB)/.sim.done
LINEAR_SIM_DONE := artifacts/model_runs/$(RUN_ID_LINEAR)/.sim.done
TFT_SIM_DONE := artifacts/model_runs/$(RUN_ID_TFT)/.sim.done

# Audit outputs
XGB_AUDIT_ZIP := artifacts/model_runs/$(RUN_ID_XGB)_deliverable.zip
LINEAR_AUDIT_ZIP := artifacts/model_runs/$(RUN_ID_LINEAR)_deliverable.zip
TFT_AUDIT_ZIP := artifacts/model_runs/$(RUN_ID_TFT)_deliverable.zip

.PHONY: help \
	doctor \
	data_hash \
	tune-xgb tune-linear tune-tft \
	train-xgb train-linear train-tft \
	sim-xgb sim-linear sim-tft \
	sim-latest-xgb sim-latest-linear sim-latest-tft \
	sim-all-quantiles \
	sim-grid-full sim-grid-smoke \
	build-hybrid sim-hybrid \
	residual-report \
	strategy-diagnostics \
	audit-xgb audit-linear audit-tft \
	thesis-report \
	all-xgb all-linear all-tft \
	smoke-test smoke-xgb smoke-linear smoke-tft clean-markers

help: ## Show available commands
	@echo "Model pipelines:"
	@echo "  make all-xgb      # tune -> train -> sim -> audit (XGBoost)"
	@echo "  make all-linear   # tune -> train -> sim -> audit (Linear)"
	@echo "  make all-tft      # tune -> train -> sim -> audit (TFT)"
	@echo ""
	@echo "Actions per model:"
	@echo "  tune-xgb | tune-linear | tune-tft"
	@echo "  train-xgb | train-linear | train-tft"
	@echo "  sim-xgb | sim-linear | sim-tft"
	@echo "  sim-latest-xgb | sim-latest-linear | sim-latest-tft   # standalone simulation (no retrain)"
	@echo "  audit-xgb | audit-linear | audit-tft"
	@echo ""
	@echo "Global:"
	@echo "  make smoke-test   # runs all-xgb, all-linear, all-tft with IS_SMOKE_TEST=1"
	@echo "  make smoke-xgb    # smoke DAG only for XGBoost (24h train + 7d sim, full quantile/DA-role sweep)"
	@echo "  make smoke-linear # smoke DAG only for Linear (24h train + 7d sim, full quantile/DA-role sweep)"
	@echo "  make smoke-tft    # smoke DAG only for TFT (24h train + 7d sim, full quantile/DA-role sweep)"
	@echo "  make sim-all-quantiles  # run xgb/linear/tft simulation with standard thesis quantile sweep"
	@echo "  make sim-grid-full      # all models x all strategies x DA-role(low/mid/high) x quantile pairs (test horizon)"
	@echo "  make sim-grid-smoke     # same grid as sim-grid-full, but only GRID_SMOKE_HOURS"
	@echo "  make build-hybrid       # build champion-by-target (hybrid/ensemble) prediction table"
	@echo "  make sim-hybrid         # run backtest on hybrid table (requires HYBRID_GROUND_TRUTH)"
	@echo "  make thesis-report      # aggregate quantile sweep outputs into one benchmark report"
	@echo "  make residual-report RUN_DIR=artifacts/model_runs/<run_id> SPLIT=test"
	@echo "  make strategy-diagnostics [SIM_ROOT=artifacts/simulation_runs] [OUT_DIR=artifacts/analysis/strategy_diagnostics]"
	@echo ""
	@echo "Optional overrides:"
	@echo "  SEED=42 FORECAST_HOURS=48 DEVICE=cuda|cpu"
	@echo "  LINEAR_AFRR_PARALLEL_JOBS=1   # parallel aFRR target jobs for linear training"
	@echo "  LINEAR_LEAD_PARALLEL_JOBS=4   # parallel lead-time jobs inside each linear target"
	@echo "  SIM_QUANTILE_PAIRS='p50-p50,p30-p70,p10-p90' SIM_DA_ROLES='low mid high'"
	@echo "  LEAD_WEIGHT_START=16 LEAD_WEIGHT_END=48 LEAD_WEIGHT_MAX=2.0"
	@echo "  HYBRID_RECO_CSV=... HYBRID_OUT=... HYBRID_GROUND_TRUTH=... HYBRID_SPLIT=test"

doctor: ## Preflight checks for data and python dependencies
	@test -d data/model_input || (echo "Missing data/model_input" && exit 1)
	@python3 -c "import importlib.util; mods=['pandas','numpy','xgboost','optuna','lightning','torch']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; (_ for _ in ()).throw(RuntimeError(f'Missing python deps: {missing}')) if missing else print('[OK] Python deps available.')"

all-xgb: audit-xgb ## Full XGBoost DAG
all-linear: audit-linear ## Full Linear DAG
all-tft: audit-tft ## Full TFT DAG

smoke-test: ## Run all model pipelines in smoke mode (IS_SMOKE_TEST=1)
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 all-xgb
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 all-linear
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 DEVICE=$(DEVICE) all-tft

smoke-xgb: ## Smoke pipeline for XGBoost only (24h train + 7d sim)
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 all-xgb

smoke-linear: ## Smoke pipeline for Linear only (24h train + 7d sim)
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 all-linear

smoke-tft: ## Smoke pipeline for TFT only (24h train + 7d sim)
	$(MAKE) IS_SMOKE_TEST=1 FORECAST_HOURS=24 SIM_HORIZON_HOURS=24 DEVICE=$(DEVICE) all-tft

$(DATA_HASH_FILE): doctor ## Generate MD5 provenance hash for data/model_input parquet files
	@mkdir -p $(dir $(DATA_HASH_FILE))
	@python3 scripts/make_data_hash.py

data_hash: $(DATA_HASH_FILE) ## Alias for provenance hash generation

$(XGB_TUNE_JSON): $(DATA_HASH_FILE)
	@mkdir -p artifacts/hpo
	python3 scripts/tune_xgboost.py \
	  --bundle da \
	  --target-col target_da_price \
	  --n-trials 60 \
	  --n-estimators 400 \
	  --selection-metric tail_upper_mae \
	  --allow-cpu

tune-xgb: $(XGB_TUNE_JSON) ## Tune XGBoost (depends on data_hash)

$(LINEAR_TUNE_JSON): $(DATA_HASH_FILE)
	@mkdir -p artifacts/hpo
	python3 scripts/tune_linear.py \
	  --bundle da \
	  --target-col target_da_price \
	  --selection-metric tail_upper_mae \
	  --alpha-grid 1e-4,5e-4,1e-3 \
	  --l1-ratio-grid 0.1,0.3,0.5 \
	  --learning-rate-grid optimal,adaptive \
	  --eta0-grid 0.001,0.01

tune-linear: $(LINEAR_TUNE_JSON) ## Tune Linear model (depends on data_hash)

$(TFT_TUNE_JSON): $(DATA_HASH_FILE)
	@mkdir -p artifacts/hpo
	python3 scripts/tune_tft.py \
	  --bundle da \
	  --target-col target_da_price \
	  --selection-metric leadtime_pinball_p90_val_weighted \
	  --fallback-metric leadtime_mae_val_weighted \
	  --n-trials 24 \
	  --device $(DEVICE) \
	  --precision $(TFT_PRECISION) \
	  --seed $(SEED)

tune-tft: $(TFT_TUNE_JSON) ## Tune TFT model (depends on data_hash)

$(XGB_MANIFEST): $(XGB_TUNE_JSON)
		python3 scripts/train_and_export_runs.py \
	  --model-type xgboost \
	  --run-id "$(RUN_ID_XGB)" \
	  --device $(DEVICE) \
	  --forecast-horizon-hours $(FORECAST_HOURS) \
	  --lead-weight-start $(LEAD_WEIGHT_START) \
	  --lead-weight-end $(LEAD_WEIGHT_END) \
	  --lead-weight-max $(LEAD_WEIGHT_MAX) \
	  --seed $(SEED) \
	  --hpo-artifact "$(XGB_TUNE_JSON)" \
	  --n-estimators 400 \
	  --early-stopping-rounds 50 \
	  --allow-cpu

train-xgb: $(XGB_MANIFEST) ## Train+evaluate XGBoost (depends on tune-xgb output)

$(LINEAR_MANIFEST): $(LINEAR_TUNE_JSON)
	python3 scripts/train_and_export_runs.py \
	  --model-type linear \
	  --run-id "$(RUN_ID_LINEAR)" \
	  --forecast-horizon-hours $(FORECAST_HOURS) \
	  --lead-weight-start $(LEAD_WEIGHT_START) \
	  --lead-weight-end $(LEAD_WEIGHT_END) \
	  --lead-weight-max $(LEAD_WEIGHT_MAX) \
	  --seed $(SEED) \
	  --afrr-parallel-jobs $(LINEAR_AFRR_PARALLEL_JOBS) \
	  --lead-parallel-jobs $(LINEAR_LEAD_PARALLEL_JOBS) \
	  --hpo-artifact "$(LINEAR_TUNE_JSON)"

train-linear: $(LINEAR_MANIFEST) ## Train+evaluate Linear (depends on tune-linear output)

$(TFT_MANIFEST): $(TFT_TUNE_JSON)
	@DEVICE_USE="$(DEVICE)"; \
	if [ "$(IS_SMOKE_TEST)" != "1" ] && [ "$$DEVICE_USE" != "cuda" ]; then \
	  echo "TFT final runs must use CUDA. Got DEVICE=$$DEVICE_USE"; \
	  exit 1; \
	fi; \
	if [ "$(IS_SMOKE_TEST)" != "1" ]; then \
	  command -v nvidia-smi >/dev/null 2>&1 || (echo "CUDA requested but nvidia-smi not found." && exit 1); \
	  nvidia-smi >/dev/null 2>&1 || (echo "CUDA requested but GPU is unavailable." && exit 1); \
	fi; \
	python3 scripts/train_and_export_runs.py \
	  --model-type tft \
	  --run-id "$(RUN_ID_TFT)" \
	  --forecast-horizon-hours $(FORECAST_HOURS) \
	  --lead-weight-start $(LEAD_WEIGHT_START) \
	  --lead-weight-end $(LEAD_WEIGHT_END) \
	  --lead-weight-max $(LEAD_WEIGHT_MAX) \
	  --seed $(SEED) \
		  --hpo-artifact "$(TFT_TUNE_JSON)" \
		  --device $$DEVICE_USE \
		  --tft-precision $(TFT_PRECISION) \
		  --num-workers 0

train-tft: $(TFT_MANIFEST) ## Train+evaluate TFT (depends on tune-tft output)

define SIM_RULE
$($(1)_SIM_DONE): $$($(1)_MANIFEST)
	if [ "$(SIM_HORIZON_HOURS)" != "auto" ]; then \
	  case "$(SIM_HORIZON_HOURS)" in ''|*[!0-9]*) echo "SIM_HORIZON_HOURS must be an integer or 'auto', got '$(SIM_HORIZON_HOURS)'"; exit 1;; esac; \
	fi
	SIM_HOURS=$(SIM_HORIZON_HOURS); \
	if [ "$$$$SIM_HOURS" = "auto" ]; then \
	  SIM_HOURS=$$$$(python3 -c "import json,sys;from pathlib import Path;m=Path('$$($(1)_MANIFEST)');d=json.loads(m.read_text(encoding='utf-8'));ctx=Path(d.get('training',{}).get('context_path',''));h=''; \
if ctx.exists(): \
  cd=json.loads(ctx.read_text(encoding='utf-8')); \
  h=str(cd.get('cli_args',{}).get('forecast_horizon_hours','')).strip(); \
print(h if h else '48')"); \
	fi; \
	case "$$$$SIM_HOURS" in ''|*[!0-9]*) echo "Resolved simulation horizon is not an integer: '$$$$SIM_HOURS' (manifest=$$($(1)_MANIFEST))"; exit 1;; esac; \
	read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);test_end=pd.to_datetime(s['test_end_inclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);sim_end=(sim_start+pd.Timedelta(days=int('$(SIM_SMOKE_DAYS)'))) if '$(IS_SMOKE_TEST)'=='1' else test_end;print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),sim_end.strftime('%Y-%m-%dT%H:%M:%SZ'))")
	for DA_ROLE in $(SIM_DA_ROLES); do \
	  OUT_ROOT="artifacts/simulation_runs/default_$(2)_$(3)/$$$$DA_ROLE"; \
	  echo "[SIM] model=$(2) run_id=$(3) manifest=$$($(1)_MANIFEST) horizon_hours=$$$$SIM_HOURS da_role=$$$$DA_ROLE start=$$$$SIM_START end=$$$$SIM_END out=$$$$OUT_ROOT"; \
	  python3 scripts/run_battery_backtest.py \
	    --run-manifest "$$($(1)_MANIFEST)" \
	    --split test \
	    --model-key $(2) \
	    --horizon-hours "$$$$SIM_HOURS" \
	    --quantile-pairs "$(SIM_QUANTILE_PAIRS)" \
	    --da-quantile-role "$$$$DA_ROLE" \
	    --start "$$$$SIM_START" \
	    --end "$$$$SIM_END" \
	    --out-dir "$$$$OUT_ROOT"; \
	done
	@touch $$($(1)_SIM_DONE)
endef

$(eval $(call SIM_RULE,XGB,xgboost,$(RUN_ID_XGB)))
$(eval $(call SIM_RULE,LINEAR,linear,$(RUN_ID_LINEAR)))
$(eval $(call SIM_RULE,TFT,tft,$(RUN_ID_TFT)))

sim-xgb: $(XGB_SIM_DONE) ## Run XGBoost simulation
sim-linear: $(LINEAR_SIM_DONE) ## Run Linear simulation
sim-tft: $(TFT_SIM_DONE) ## Run TFT simulation
sim-latest-xgb: ## Standalone simulation from artifacts/model_runs/latest_xgboost.json
	@LATEST_JSON="artifacts/model_runs/latest_xgboost.json"; \
	test -f "$$LATEST_JSON" || (echo "Missing $$LATEST_JSON" && exit 1); \
	if [ "$(SIM_HORIZON_HOURS)" != "auto" ]; then \
	  case "$(SIM_HORIZON_HOURS)" in ''|*[!0-9]*) echo "SIM_HORIZON_HOURS must be an integer or 'auto', got '$(SIM_HORIZON_HOURS)'"; exit 1;; esac; \
	fi; \
	MANIFEST_PATH=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print(d.get('manifest_path','').strip())"); \
	test -n "$$MANIFEST_PATH" || (echo "manifest_path missing in $$LATEST_JSON" && exit 1); \
	RUN_ID=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print((d.get('run_id') or '').strip())"); \
	SIM_HOURS="$(SIM_HORIZON_HOURS)"; \
	if [ "$$SIM_HOURS" = "auto" ]; then \
	  SIM_HOURS=$$(python3 -c "import json;from pathlib import Path;m=Path('$$MANIFEST_PATH');d=json.loads(m.read_text(encoding='utf-8'));ctx=Path(d.get('training',{}).get('context_path',''));h=''; \
if ctx.exists(): \
  cd=json.loads(ctx.read_text(encoding='utf-8')); \
  h=str(cd.get('cli_args',{}).get('forecast_horizon_hours','')).strip(); \
print(h if h else '48')"); \
	fi; \
	case "$$SIM_HOURS" in ''|*[!0-9]*) echo "Resolved simulation horizon is not an integer: '$$SIM_HOURS' (manifest=$$MANIFEST_PATH)"; exit 1;; esac; \
	read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);test_end=pd.to_datetime(s['test_end_inclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),test_end.strftime('%Y-%m-%dT%H:%M:%SZ'))"); \
	echo "[INFO] sim-latest-xgb model=xgboost run_id=$$RUN_ID manifest=$$MANIFEST_PATH horizon_hours=$$SIM_HOURS using $$LATEST_JSON"; \
	for DA_ROLE in $(SIM_DA_ROLES); do \
	  python3 scripts/run_battery_backtest.py \
	    --run-manifest "$$MANIFEST_PATH" \
	    --split test \
	    --model-key xgboost \
	    --horizon-hours "$$SIM_HOURS" \
	    --quantile-pairs "$(SIM_QUANTILE_PAIRS)" \
	    --da-quantile-role "$$DA_ROLE" \
	    --start "$$SIM_START" \
	    --end "$$SIM_END" \
	    --out-dir "artifacts/simulation_runs/latest_xgboost/$$DA_ROLE"; \
	done
sim-latest-linear: ## Standalone simulation from artifacts/model_runs/latest_linear.json
	@LATEST_JSON="artifacts/model_runs/latest_linear.json"; \
	test -f "$$LATEST_JSON" || (echo "Missing $$LATEST_JSON" && exit 1); \
	if [ "$(SIM_HORIZON_HOURS)" != "auto" ]; then \
	  case "$(SIM_HORIZON_HOURS)" in ''|*[!0-9]*) echo "SIM_HORIZON_HOURS must be an integer or 'auto', got '$(SIM_HORIZON_HOURS)'"; exit 1;; esac; \
	fi; \
	MANIFEST_PATH=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print(d.get('manifest_path','').strip())"); \
	test -n "$$MANIFEST_PATH" || (echo "manifest_path missing in $$LATEST_JSON" && exit 1); \
	RUN_ID=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print((d.get('run_id') or '').strip())"); \
	SIM_HOURS="$(SIM_HORIZON_HOURS)"; \
	if [ "$$SIM_HOURS" = "auto" ]; then \
	  SIM_HOURS=$$(python3 -c "import json;from pathlib import Path;m=Path('$$MANIFEST_PATH');d=json.loads(m.read_text(encoding='utf-8'));ctx=Path(d.get('training',{}).get('context_path',''));h=''; \
if ctx.exists(): \
  cd=json.loads(ctx.read_text(encoding='utf-8')); \
  h=str(cd.get('cli_args',{}).get('forecast_horizon_hours','')).strip(); \
print(h if h else '48')"); \
	fi; \
	case "$$SIM_HOURS" in ''|*[!0-9]*) echo "Resolved simulation horizon is not an integer: '$$SIM_HOURS' (manifest=$$MANIFEST_PATH)"; exit 1;; esac; \
	read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);test_end=pd.to_datetime(s['test_end_inclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),test_end.strftime('%Y-%m-%dT%H:%M:%SZ'))"); \
	echo "[INFO] sim-latest-linear model=linear run_id=$$RUN_ID manifest=$$MANIFEST_PATH horizon_hours=$$SIM_HOURS using $$LATEST_JSON"; \
	for DA_ROLE in $(SIM_DA_ROLES); do \
	  python3 scripts/run_battery_backtest.py \
	    --run-manifest "$$MANIFEST_PATH" \
	    --split test \
	    --model-key linear \
	    --horizon-hours "$$SIM_HOURS" \
	    --quantile-pairs "$(SIM_QUANTILE_PAIRS)" \
	    --da-quantile-role "$$DA_ROLE" \
	    --start "$$SIM_START" \
	    --end "$$SIM_END" \
	    --out-dir "artifacts/simulation_runs/latest_linear/$$DA_ROLE"; \
	done
sim-latest-tft: ## Standalone simulation from artifacts/model_runs/latest_tft.json
	@LATEST_JSON="artifacts/model_runs/latest_tft.json"; \
	test -f "$$LATEST_JSON" || (echo "Missing $$LATEST_JSON" && exit 1); \
	if [ "$(SIM_HORIZON_HOURS)" != "auto" ]; then \
	  case "$(SIM_HORIZON_HOURS)" in ''|*[!0-9]*) echo "SIM_HORIZON_HOURS must be an integer or 'auto', got '$(SIM_HORIZON_HOURS)'"; exit 1;; esac; \
	fi; \
	MANIFEST_PATH=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print(d.get('manifest_path','').strip())"); \
	test -n "$$MANIFEST_PATH" || (echo "manifest_path missing in $$LATEST_JSON" && exit 1); \
	RUN_ID=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print((d.get('run_id') or '').strip())"); \
	SIM_HOURS="$(SIM_HORIZON_HOURS)"; \
	if [ "$$SIM_HOURS" = "auto" ]; then \
	  SIM_HOURS=$$(python3 -c "import json;from pathlib import Path;m=Path('$$MANIFEST_PATH');d=json.loads(m.read_text(encoding='utf-8'));ctx=Path(d.get('training',{}).get('context_path',''));h=''; \
if ctx.exists(): \
  cd=json.loads(ctx.read_text(encoding='utf-8')); \
  h=str(cd.get('cli_args',{}).get('forecast_horizon_hours','')).strip(); \
print(h if h else '48')"); \
	fi; \
	case "$$SIM_HOURS" in ''|*[!0-9]*) echo "Resolved simulation horizon is not an integer: '$$SIM_HOURS' (manifest=$$MANIFEST_PATH)"; exit 1;; esac; \
	read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);test_end=pd.to_datetime(s['test_end_inclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),test_end.strftime('%Y-%m-%dT%H:%M:%SZ'))"); \
	echo "[INFO] sim-latest-tft model=tft run_id=$$RUN_ID manifest=$$MANIFEST_PATH horizon_hours=$$SIM_HOURS using $$LATEST_JSON"; \
	for DA_ROLE in $(SIM_DA_ROLES); do \
	  python3 scripts/run_battery_backtest.py \
	    --run-manifest "$$MANIFEST_PATH" \
	    --split test \
	    --model-key tft \
	    --horizon-hours "$$SIM_HOURS" \
	    --quantile-pairs "$(SIM_QUANTILE_PAIRS)" \
	    --da-quantile-role "$$DA_ROLE" \
	    --start "$$SIM_START" \
	    --end "$$SIM_END" \
	    --out-dir "artifacts/simulation_runs/latest_tft/$$DA_ROLE"; \
	done
sim-all-quantiles: clean-markers ## Run quantile sweep simulation for xgb, linear, tft
	$(MAKE) sim-xgb SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" SIM_DA_ROLES="low mid high"
	$(MAKE) sim-linear SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" SIM_DA_ROLES="low mid high"
	$(MAKE) sim-tft SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" SIM_DA_ROLES="low mid high"

sim-grid-full: ## Full grid: all models x strategies x DA roles x quantile pairs on full test horizon
	@read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);test_end=pd.to_datetime(s['test_end_inclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),test_end.strftime('%Y-%m-%dT%H:%M:%SZ'))"); \
	for MODEL in xgboost linear tft; do \
	  LATEST_JSON="artifacts/model_runs/latest_$${MODEL}.json"; \
	  test -f "$$LATEST_JSON" || (echo "Missing $$LATEST_JSON" && exit 1); \
	  RUN_ID=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print((d.get('run_id') or '').strip())" 2>/dev/null || true); \
	  test -n "$$RUN_ID" || (echo "Missing run_id in $$LATEST_JSON" && exit 1); \
	  MANIFEST="artifacts/model_runs/$${RUN_ID}/manifest.json"; \
	  test -f "$$MANIFEST" || (echo "Missing manifest: $$MANIFEST" && exit 1); \
	  for DA_ROLE in $(GRID_DA_ROLES); do \
	    OUT_ROOT="artifacts/simulation_runs/quantile_grid_$${DA_ROLE}_$(SIM_GRID_STAMP)/$${MODEL}"; \
	    for STRAT in $(GRID_STRATEGIES); do \
	      echo "[GRID] model=$$MODEL strategy=$$STRAT da_role=$$DA_ROLE out=$$OUT_ROOT"; \
	      BACKTEST_MILP_TIME_LIMIT_S=300 BACKTEST_MILP_REL_GAP=1e-4 \
	      ./.venv/bin/python -u scripts/run_battery_backtest.py \
	        --run-manifest "$$MANIFEST" \
	        --split test \
	        --model-key "$$MODEL" \
	        --trading-strategy "$$STRAT" \
	        --quantile-pairs "$(GRID_QUANTILE_PAIRS)" \
	        --da-quantile-role "$$DA_ROLE" \
	        --start "$$SIM_START" \
	        --end "$$SIM_END" \
	        --out-dir "$$OUT_ROOT"; \
	    done; \
	  done; \
	done

sim-grid-smoke: ## Smoke grid: all models x strategies x DA roles x quantile pairs on short window
	@read SIM_START SIM_END < <(python3 -c "import json,pandas as pd;from pathlib import Path;cfg=json.loads(Path('data/model_input/feature_config.json').read_text(encoding='utf-8'));s=cfg.get('splits',{});val_end=pd.to_datetime(s['val_end_exclusive'],utc=True);gap=int(s.get('purge_gap_rows',72));sim_start=val_end+pd.Timedelta(hours=gap);sim_end=sim_start+pd.Timedelta(hours=int('$(GRID_SMOKE_HOURS)'));print(sim_start.strftime('%Y-%m-%dT%H:%M:%SZ'),sim_end.strftime('%Y-%m-%dT%H:%M:%SZ'))"); \
	for MODEL in xgboost linear tft; do \
	  LATEST_JSON="artifacts/model_runs/latest_$${MODEL}.json"; \
	  test -f "$$LATEST_JSON" || (echo "Missing $$LATEST_JSON" && exit 1); \
	  RUN_ID=$$(python3 -c "import json;from pathlib import Path;p=Path('$$LATEST_JSON');d=json.loads(p.read_text(encoding='utf-8'));print((d.get('run_id') or '').strip())" 2>/dev/null || true); \
	  test -n "$$RUN_ID" || (echo "Missing run_id in $$LATEST_JSON" && exit 1); \
	  MANIFEST="artifacts/model_runs/$${RUN_ID}/manifest.json"; \
	  test -f "$$MANIFEST" || (echo "Missing manifest: $$MANIFEST" && exit 1); \
	  for DA_ROLE in $(GRID_DA_ROLES); do \
	    OUT_ROOT="artifacts/simulation_runs/quantile_grid_smoke_$${DA_ROLE}_$(SIM_GRID_STAMP)/$${MODEL}"; \
	    for STRAT in $(GRID_STRATEGIES); do \
	      echo "[GRID-SMOKE] model=$$MODEL strategy=$$STRAT da_role=$$DA_ROLE out=$$OUT_ROOT"; \
	      BACKTEST_MILP_TIME_LIMIT_S=120 BACKTEST_MILP_REL_GAP=1e-4 \
	      ./.venv/bin/python -u scripts/run_battery_backtest.py \
	        --run-manifest "$$MANIFEST" \
	        --split test \
	        --model-key "$$MODEL" \
	        --trading-strategy "$$STRAT" \
	        --quantile-pairs "$(GRID_QUANTILE_PAIRS)" \
	        --da-quantile-role "$$DA_ROLE" \
	        --start "$$SIM_START" \
	        --end "$$SIM_END" \
	        --out-dir "$$OUT_ROOT"; \
	    done; \
	  done; \
	done

define AUDIT_RULE
$($(1)_AUDIT_ZIP): $$($(1)_SIM_DONE)
	@test -f "$$($(1)_MANIFEST)"
	@test -d "artifacts/model_runs/$(3)/metrics"
	@METRIC_COUNT=$$$$(find artifacts/model_runs/$(3)/metrics -type f | wc -l); test "$$$$METRIC_COUNT" -gt 0
	@SIM_COUNT=$$$$(find artifacts/model_runs/$(3) -type f | grep -E 'backtest|simulation|summary|ledger|pnl' | wc -l); test "$$$$SIM_COUNT" -gt 0
	@mkdir -p "artifacts/model_runs/$(3)/metadata"
	@python3 --version > "artifacts/model_runs/$(3)/metadata/python_version.txt"
	@pip freeze > "artifacts/model_runs/$(3)/metadata/pip_freeze.txt"
	@git rev-parse HEAD > "artifacts/model_runs/$(3)/metadata/git_commit.txt" || true
	@date -u +"%Y-%m-%dT%H:%M:%SZ" > "artifacts/model_runs/$(3)/metadata/timestamp_utc.txt"
	@printf "SEED=%s\nFORECAST_HOURS=%s\nDEVICE=%s\nLEAD_WEIGHT_START=%s\nLEAD_WEIGHT_END=%s\nLEAD_WEIGHT_MAX=%s\nSIM_QUANTILE_PAIRS=%s\nSIM_DA_ROLES=%s\nDA_QUANTILE_ROLE=%s\n" "$(SEED)" "$(FORECAST_HOURS)" "$(DEVICE)" "$(LEAD_WEIGHT_START)" "$(LEAD_WEIGHT_END)" "$(LEAD_WEIGHT_MAX)" "$(SIM_QUANTILE_PAIRS)" "$(SIM_DA_ROLES)" "$(DA_QUANTILE_ROLE)" > "artifacts/model_runs/$(3)/metadata/run_parameters.txt"
	@python3 scripts/package_audit.py "$(3)"
endef

$(eval $(call AUDIT_RULE,XGB,xgboost,$(RUN_ID_XGB)))
$(eval $(call AUDIT_RULE,LINEAR,linear,$(RUN_ID_LINEAR)))
$(eval $(call AUDIT_RULE,TFT,tft,$(RUN_ID_TFT)))

audit-xgb: $(XGB_AUDIT_ZIP) ## Audit+package XGBoost run
audit-linear: $(LINEAR_AUDIT_ZIP) ## Audit+package Linear run
audit-tft: $(TFT_AUDIT_ZIP) ## Audit+package TFT run
thesis-report: ## Merge quantile sweep summaries into artifacts/thesis_benchmark_report.csv
	python3 scripts/generate_thesis_report.py

residual-report: ## Export reproducible residual artifacts for one run (set RUN_DIR and optional SPLIT)
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR, e.g. make residual-report RUN_DIR=artifacts/model_runs/xgb_YYYYmmdd_HHMMSS" && exit 1)
	@SPLIT_USE="$(or $(SPLIT),test)"; \
	OUT_DIR_USE="$(or $(OUT_DIR),artifacts/residual_reports/$$(basename "$(RUN_DIR)")_$${SPLIT_USE})"; \
	python3 scripts/export_residual_analysis.py \
	  --run-dir "$(RUN_DIR)" \
	  --split "$${SPLIT_USE}" \
	  --out-dir "$${OUT_DIR_USE}"

strategy-diagnostics: ## Analyze mapping/failure-modes/constraints/calibration-link/comparative diagnosis
	@SIM_ROOT_USE="$(or $(SIM_ROOT),artifacts/simulation_runs)"; \
	OUT_DIR_USE="$(or $(OUT_DIR),artifacts/analysis/strategy_diagnostics)"; \
	python3 scripts/generate_strategy_diagnostics.py \
	  --simulation-root "$${SIM_ROOT_USE}" \
	  --out-dir "$${OUT_DIR_USE}"

build-hybrid: ## Build champion-by-target hybrid (ensemble) prediction table
	@RECO_USE="$(or $(HYBRID_RECO_CSV),artifacts/benchmarks/final_report/recommendation_per_target_test.csv)"; \
	OUT_USE="$(or $(HYBRID_OUT),artifacts/simulation_runs/hybrid/test/backtest_table_test.parquet)"; \
	SPLIT_USE="$(or $(HYBRID_SPLIT),test)"; \
	python3 scripts/build_hybrid_predictions.py \
	  --recommendation-csv "$${RECO_USE}" \
	  --split "$${SPLIT_USE}" \
	  --out "$${OUT_USE}"

sim-hybrid: build-hybrid ## Simulate champion-by-target hybrid table
	@test -n "$(HYBRID_GROUND_TRUTH)" || (echo "Set HYBRID_GROUND_TRUTH (required), e.g. make sim-hybrid HYBRID_GROUND_TRUTH=data/model_input/backtest_truth_test.parquet" && exit 1)
	@OUT_USE="$(or $(HYBRID_OUT),artifacts/simulation_runs/hybrid/test/backtest_table_test.parquet)"; \
	SPLIT_USE="$(or $(HYBRID_SPLIT),test)"; \
	python3 scripts/run_battery_backtest.py \
	  --predictions "$${OUT_USE}" \
	  --ground-truth "$(HYBRID_GROUND_TRUTH)" \
	  --split "$${SPLIT_USE}" \
	  --model-key hybrid \
	  --out-dir "artifacts/simulation_runs/hybrid/$${SPLIT_USE}"

clean-markers: ## Remove simulation markers only (forces re-sim)
	rm -f $(XGB_SIM_DONE) $(LINEAR_SIM_DONE) $(TFT_SIM_DONE)
