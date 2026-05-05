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
export IS_SMOKE_TEST ?= 0
DEVICE ?= cuda
SIM_QUANTILE_PAIRS ?=
DA_QUANTILE_ROLE ?= mid
SIM_QUANTILE_SWEEP_DEFAULT ?= p50-p50,p30-p70,p10-p90,p10-p30,p30-p50,p50-p70,p70-p90

# Run IDs (evaluated once per make invocation; command-line/env overrides still win)
RUN_ID_XGB := $(or $(RUN_ID_XGB),xgb_$(shell date +%Y%m%d_%H%M%S))
RUN_ID_LINEAR := $(or $(RUN_ID_LINEAR),linear_$(shell date +%Y%m%d_%H%M%S))
RUN_ID_TFT := $(or $(RUN_ID_TFT),tft_$(shell date +%Y%m%d_%H%M%S))

# Common artifacts
DATA_HASH_FILE := artifacts/hpo/data_model_input.md5

# HPO outputs
XGB_TUNE_JSON := artifacts/hpo/xgb_optuna_da_target_da_price.json
XGB_TUNE_CSV := artifacts/hpo/xgb_optuna_da_target_da_price_trials.csv
LINEAR_TUNE_JSON := artifacts/hpo/linear_sgd_tuning_da_target_da_price.json
LINEAR_TUNE_CSV := artifacts/hpo/linear_sgd_tuning_da_target_da_price_trials.csv

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
	tune-xgb tune-linear \
	train-xgb train-linear train-tft \
	sim-xgb sim-linear sim-tft \
	sim-all-quantiles \
	audit-xgb audit-linear audit-tft \
	thesis-report \
	all-xgb all-linear all-tft \
	smoke-test clean-markers

help: ## Show available commands
	@echo "Model pipelines:"
	@echo "  make all-xgb      # tune -> train -> sim -> audit (XGBoost)"
	@echo "  make all-linear   # tune -> train -> sim -> audit (Linear)"
	@echo "  make all-tft      # train -> sim -> audit (TFT, no tune)"
	@echo ""
	@echo "Actions per model:"
	@echo "  tune-xgb | tune-linear"
	@echo "  train-xgb | train-linear | train-tft"
	@echo "  sim-xgb | sim-linear | sim-tft"
	@echo "  audit-xgb | audit-linear | audit-tft"
	@echo ""
	@echo "Global:"
	@echo "  make smoke-test   # runs all-xgb, all-linear, all-tft with IS_SMOKE_TEST=1"
	@echo "  make sim-all-quantiles  # run xgb/linear/tft simulation with standard thesis quantile sweep"
	@echo "  make thesis-report      # aggregate quantile sweep outputs into one benchmark report"
	@echo ""
	@echo "Optional overrides:"
	@echo "  RUN_ID_XGB=... RUN_ID_LINEAR=... RUN_ID_TFT=... SEED=42 FORECAST_HOURS=48 DEVICE=cuda|cpu"
	@echo "  SIM_QUANTILE_PAIRS='p50-p50,p30-p70,p10-p90' DA_QUANTILE_ROLE=mid|low|high"

doctor: ## Preflight checks for data and python dependencies
	@test -d data/model_input || (echo "Missing data/model_input" && exit 1)
	@python3 -c "import importlib.util; mods=['pandas','numpy','xgboost','optuna','lightning','torch']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; (_ for _ in ()).throw(RuntimeError(f'Missing python deps: {missing}')) if missing else print('[OK] Python deps available.')"

all-xgb: audit-xgb ## Full XGBoost DAG
all-linear: audit-linear ## Full Linear DAG
all-tft: audit-tft ## Full TFT DAG

smoke-test: ## Run all model pipelines in smoke mode (IS_SMOKE_TEST=1)
	$(MAKE) IS_SMOKE_TEST=1 all-xgb
	$(MAKE) IS_SMOKE_TEST=1 all-linear
	$(MAKE) IS_SMOKE_TEST=1 DEVICE=cpu all-tft

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

$(XGB_MANIFEST): $(XGB_TUNE_JSON)
	python3 scripts/train_and_export_runs.py \
	  --model-type xgboost \
	  --run-id "$(RUN_ID_XGB)" \
	  --forecast-horizon-hours $(FORECAST_HOURS) \
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
	  --seed $(SEED) \
	  --hpo-artifact "$(LINEAR_TUNE_JSON)"

train-linear: $(LINEAR_MANIFEST) ## Train+evaluate Linear (depends on tune-linear output)

$(TFT_MANIFEST): $(DATA_HASH_FILE)
	@DEVICE_USE="$(DEVICE)"; \
	if [ "$(IS_SMOKE_TEST)" = "1" ]; then DEVICE_USE="cpu"; fi; \
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
	  --seed $(SEED) \
	  --device $$DEVICE_USE \
	  --num-workers 0

train-tft: $(TFT_MANIFEST) ## Train+evaluate TFT (depends only on data_hash; no tune)

define SIM_RULE
$($(1)_SIM_DONE): $$($(1)_MANIFEST)
	python3 scripts/run_battery_backtest.py \
	  --run-manifest "$$($(1)_MANIFEST)" \
	  --split test \
	  --model-key $(2) \
	  --quantile-pairs "$(SIM_QUANTILE_PAIRS)" \
	  --da-quantile-role "$(DA_QUANTILE_ROLE)" \
	  --start 2024-01-01T00:00:00Z \
	  --end 2024-03-31T23:00:00Z
	@touch $$($(1)_SIM_DONE)
endef

$(eval $(call SIM_RULE,XGB,xgboost))
$(eval $(call SIM_RULE,LINEAR,linear))
$(eval $(call SIM_RULE,TFT,tft))

sim-xgb: $(XGB_SIM_DONE) ## Run XGBoost simulation
sim-linear: $(LINEAR_SIM_DONE) ## Run Linear simulation
sim-tft: $(TFT_SIM_DONE) ## Run TFT simulation
sim-all-quantiles: clean-markers ## Run quantile sweep simulation for xgb, linear, tft
	$(MAKE) sim-xgb SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" DA_QUANTILE_ROLE=mid
	$(MAKE) sim-linear SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" DA_QUANTILE_ROLE=mid
	$(MAKE) sim-tft SIM_QUANTILE_PAIRS="$(SIM_QUANTILE_SWEEP_DEFAULT)" DA_QUANTILE_ROLE=mid

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
	@printf "SEED=%s\nFORECAST_HOURS=%s\nDEVICE=%s\nSIM_QUANTILE_PAIRS=%s\nDA_QUANTILE_ROLE=%s\n" "$(SEED)" "$(FORECAST_HOURS)" "$(DEVICE)" "$(SIM_QUANTILE_PAIRS)" "$(DA_QUANTILE_ROLE)" > "artifacts/model_runs/$(3)/metadata/run_parameters.txt"
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

clean-markers: ## Remove simulation markers only (forces re-sim)
	rm -f $(XGB_SIM_DONE) $(LINEAR_SIM_DONE) $(TFT_SIM_DONE)
