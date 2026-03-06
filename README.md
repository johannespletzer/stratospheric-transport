# Modeling of Stratospheric Transport

This repository provides a Physics-Informed Neural Network (PINN) workflow for
stratospheric transport studies. The model is designed to predict residence time
(`tau_R`) from age-of-air observations (`Gamma`) with an optional supervised
target (`tau`) and optional tropopause-derived features.

## Quickstart

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the workflow:

```bash
python scripts/workflow.py train --config configs/run.example.yaml
python scripts/workflow.py plot --run-dir runs/<run_id>
python scripts/workflow.py compare --run-dirs runs/<run_a> runs/<run_b> --output-dir reports/compare/latest
```

Optional Makefile wrappers:

```bash
make setup
make train CONFIG=configs/run.example.yaml
make plot RUN_DIR=runs/<run_id>
make compare RUN_DIRS="runs/<run_a> runs/<run_b>" OUTPUT_DIR=reports/compare/latest
```

## Repository Structure

- `scripts/`: workflow and data preparation entry points
- `src/residence_time/`: model, data loading, training, and config logic
- `configs/`: example YAML run configurations
- `tests/`: unit and workflow integration tests
- `runs/`: generated training artifacts (created during workflow runs)

## Documentation

- [Run Configuration](docs/run-configuration.md)
- [Data and Outputs](docs/data-and-outputs.md)

## Acknowledgements

This work builds on age-of-air and tropopause datasets from:

- Garny et al. 2024: ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)
- Hoffmann and Spang 2021: ["Reanalysis Tropopause Data Repository"](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)
