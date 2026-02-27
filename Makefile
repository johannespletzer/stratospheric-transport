PYTHON ?= python

.PHONY: setup train plot compare

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	@test -n "$(CONFIG)" || (echo "Usage: make train CONFIG=configs/run.example.yaml" && exit 1)
	$(PYTHON) scripts/workflow.py train --config "$(CONFIG)"

plot:
	@test -n "$(RUN_DIR)" || (echo "Usage: make plot RUN_DIR=runs/<run_id>" && exit 1)
	$(PYTHON) scripts/workflow.py plot --run-dir "$(RUN_DIR)"

compare:
	@test -n "$(RUN_DIRS)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "Usage: make compare RUN_DIRS=\"runs/a runs/b\" OUTPUT_DIR=reports/compare/latest" && exit 1)
	$(PYTHON) scripts/workflow.py compare --run-dirs $(RUN_DIRS) --output-dir "$(OUTPUT_DIR)"
