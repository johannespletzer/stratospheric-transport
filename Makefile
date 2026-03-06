PYTHON ?= python
RESUME_OPT_FLAG := $(if $(filter true,$(RESUME_LOAD_OPT)),--resume-load-optimizer,$(if $(filter false,$(RESUME_LOAD_OPT)),--no-resume-load-optimizer,))
RESUME_SCALER_FLAG := $(if $(filter true,$(RESUME_LOAD_SCALER)),--resume-load-scaler,$(if $(filter false,$(RESUME_LOAD_SCALER)),--no-resume-load-scaler,))
TRAIN_ARGS = $(strip \
	$(if $(RESUME_FROM),--resume-from "$(RESUME_FROM)") \
	$(if $(RESUME_CONFIG),--resume-config "$(RESUME_CONFIG)") \
	$(RESUME_OPT_FLAG) \
	$(RESUME_SCALER_FLAG) \
)
PLOT_FIELD_ARGS = $(strip \
	$(if $(OUTPUT_PATH),--output-path "$(OUTPUT_PATH)") \
	$(if $(FIELD),--field "$(FIELD)") \
	$(if $(and $(GRID_LAT),$(GRID_ALT)),--grid-res "$(GRID_LAT)" "$(GRID_ALT)") \
	$(if $(TIME_VALUE),--time-value "$(TIME_VALUE)") \
	$(if $(SOURCE_VALUE),--source-value "$(SOURCE_VALUE)") \
	$(if $(GAMMA_VALUE),--gamma-value "$(GAMMA_VALUE)") \
	$(if $(DEVICE),--device "$(DEVICE)") \
)

.PHONY: setup train plot plot-field compare

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	@test -n "$(CONFIG)" || (echo "Usage: make train CONFIG=configs/run.example.yaml" && exit 1)
	$(PYTHON) scripts/workflow.py train --config "$(CONFIG)" $(TRAIN_ARGS)

plot:
	@test -n "$(RUN_DIR)" || (echo "Usage: make plot RUN_DIR=runs/<run_id>" && exit 1)
	$(PYTHON) scripts/workflow.py plot --run-dir "$(RUN_DIR)"

plot-field:
	@test -n "$(RUN_DIR)" || (echo "Usage: make plot-field RUN_DIR=runs/<run_id> [FIELD=tau_R] [GRID_LAT=64 GRID_ALT=40]" && exit 1)
	@test -z "$(GRID_LAT)$(GRID_ALT)" || (test -n "$(GRID_LAT)" -a -n "$(GRID_ALT)" || (echo "Set both GRID_LAT and GRID_ALT together." && exit 1))
	$(PYTHON) scripts/workflow.py plot-field --run-dir "$(RUN_DIR)" $(PLOT_FIELD_ARGS)

compare:
	@test -n "$(RUN_DIRS)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	$(PYTHON) scripts/workflow.py compare --run-dirs $(RUN_DIRS) --output-dir "$(OUTPUT_DIR)"
