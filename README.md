# Modeling of Stratospheric Transport

This repository provides a Physics-Informed Neural Network (PINN) for stratospheric transport, focused on predicting residence time (`tau_R`) from age-of-air observations (`Gamma`) and optional tropopause features.

The default physics constraint is:

`tau_R = 2 * D^2 / Gamma`

where `D` is an effective diffusivity predicted by an optional second network branch.

## Quick Start

1. Install:

```bash
python -m pip install -e .
```

2. Download age-of-air data:

```bash
python scripts/download_age_of_air_data.py --output data/age_of_air
```

3. Train (example):

```bash
python scripts/train_model.py \
  --sat-paths data/age_of_air/path_to_satellite.nc \
  --insitu-paths data/age_of_air/path_to_insitu.nc \
  --tau-path data/path_to_tau_file.nc \
  --epochs 300
```

## Requirements

- Python `>=3.8`
- Core dependencies are declared in [pyproject.toml](pyproject.toml)
- GPU is optional; training can run on CPU with `--device cpu`

## Data Preparation

Download age-of-air data:

```bash
python scripts/download_age_of_air_data.py --output data/age_of_air
```

Optional: download and concatenate ERA5 tropopause data:

```bash
python scripts/download_and_process_era5.py --start-year 1980 --end-year 2018
```

Optional: extract monthly tropopause features:

```bash
python scripts/process_tropopause_features.py --infile data/tropopause/era5_tropopause_combined.nc
```

Default output CSV:

`data/tropopause/tropopause_features_monthly.csv`

## Data Layout

Typical layout:

```text
data/
  age_of_air/
    ... AoA .nc files ...
  tropopause/
    era5_tropopause_combined.nc
    tropopause_features_monthly.csv
  ... optional tau_R files ...
```

## Training

Main entrypoint:

`python scripts/train_model.py`

Common options:

- `--sat-paths`, `--insitu-paths`, `--model-paths`: one or more input NetCDF files
- `--tau-path`: optional supervised residence-time target file
- `--use-tropopause-features`: append tropopause feature inputs
- `--enable-time-encoding`, `--use-cyclical`, `--use-rbf-seasonal`, `--use-rbf-absolute`
- `--hidden-dim`, `--hidden-layers`, `--batch-size`, `--epochs`
- `--device cpu|cuda`

Physics-only training example (no `--tau-path`):

```bash
python scripts/train_model.py \
  --sat-paths data/age_of_air/path_to_satellite.nc \
  --epochs 300
```

## Configuration

Global defaults are in [src/residence_time/config.py](src/residence_time/config.py):

- `PHYSICS_CONSTRAINT`: `"diffusivity"` or `"harmonic"`
- `USE_TROPOPAUSE_FEATURES`
- `DEFAULT_TIME_ENCODING_CONFIG`
- `INSITU_REFERENCE_YEAR`
- `LEARNING_RATE`

Note: time encoding flags are intended for diffusivity mode. Harmonic mode expects the base time column only.
Base time is represented as integer year; seasonal encodings are generated from a separate seasonal phase.

## Outputs

After training, checkpoints and config snapshots are saved under:

- `models/checkpoints/`
- `models/configs/`

## Quality Checks

Run linter and tests:

```bash
ruff check src/ tests/ scripts/
pytest -v
```

CI currently runs:

```bash
pytest -v
```

## Evaluation and Visualization

Plotting utilities are in [src/residence_time/plot.py](src/residence_time/plot.py), including:

- training loss curves
- prediction-vs-target scatter
- physics residual histograms
- latitude-altitude field plots for `tau_R` (and `D` when available)

## Project Structure

```text
src/residence_time/   # model, training, data, plotting, utils
scripts/              # data download/processing and training entrypoint
tests/                # unit tests
.github/workflows/    # CI
```

## Troubleshooting

- If downloads fail, verify network access and remote server availability.
- If CUDA is unavailable, set `--device cpu`.
- If you see zero/empty training samples, verify data paths and `--tau-path` coverage.
- ERA5 processing can be large; use a restricted year range while iterating.

## Acknowledgements

This work builds on age-of-air and tropopause datasets from:

- Garny et al. 2024: ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)
- Hoffmann and Spang 2021: ["Reanalysis Tropopause Data Repository"](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)
