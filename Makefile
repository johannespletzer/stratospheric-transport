PYTHON ?= python
RESUME_OPT_FLAG := $(if $(filter true,$(RESUME_LOAD_OPT)),--resume-load-optimizer,$(if $(filter false,$(RESUME_LOAD_OPT)),--no-resume-load-optimizer,))
RESUME_SCALER_FLAG := $(if $(filter true,$(RESUME_LOAD_SCALER)),--resume-load-scaler,$(if $(filter false,$(RESUME_LOAD_SCALER)),--no-resume-load-scaler,))

.PHONY: setup train plot compare

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	@test -n "$(CONFIG)" || (echo "Usage: make train CONFIG=configs/run.example.yaml" && exit 1)
	$(PYTHON) scripts/workflow.py train --config "$(CONFIG)" \
		$(if $(RESUME_FROM),--resume-from "$(RESUME_FROM)",) \
		$(if $(RESUME_CONFIG),--resume-config "$(RESUME_CONFIG)",) \
		$(RESUME_OPT_FLAG) \
		$(RESUME_SCALER_FLAG)

plot:
	@test -n "$(RUN_DIR)" || (echo "Usage: make plot RUN_DIR=runs/<run_id>" && exit 1)
	$(PYTHON) scripts/workflow.py plot --run-dir "$(RUN_DIR)"

compare:
	@test -n "$(RUN_DIRS)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	$(PYTHON) scripts/workflow.py compare --run-dirs $(RUN_DIRS) --output-dir "$(OUTPUT_DIR)"
