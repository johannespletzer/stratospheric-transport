# Modeling of Stratospheric Transport

This repository provides a Physics-Informed Neural Network (PINN) for stratospheric transport,
focused on predicting residence time (`tau_R`) from age-of-air observations (`Gamma`) and
optional tropopause features.

Default physics constraint:

`tau_R = 2 * D^2 / Gamma`

## Canonical Workflow

The standard end-to-end workflow is now:

1. clone + setup
2. train from YAML config
3. generate run plots
4. compare runs

Cross-platform source of truth:

```bash
python scripts/workflow.py train --config configs/run.example.yaml
python scripts/workflow.py plot --run-dir runs/<run_id>
python scripts/workflow.py compare --run-dirs runs/<run_a> runs/<run_b> --output-dir reports/compare/latest
```

`Makefile` convenience wrappers are also available:

```bash
make setup
make train CONFIG=configs/run.example.yaml
make train CONFIG=configs/run.resume.example.yaml
make plot RUN_DIR=runs/<run_id>
make compare RUN_DIRS="runs/<run_a> runs/<run_b>" OUTPUT_DIR=reports/compare/latest
```

## Setup

Install package + dev tooling:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run Config (Hybrid Path Resolution)

See `configs/run.example.yaml` and `configs/run.resume.example.yaml`.

Path precedence for training data is:

1. explicit CLI paths
2. explicit YAML paths
3. auto-discovery (if enabled)

Auto-discovery defaults:

- `enabled: true`
- `search_roots: ["data"]`
- `discover_model_paths: false`
- `discover_tau_path: false`

Auto-discovery is schema-based:

- Satellite AoA: `AoA`, `AoA_STD`, and `time/lat/alt`
- In-situ AoA: `Mean_Age_SF6_corr` or `Mean_Age_CO2` and `lat/Altitude`
- Model AoA: `AOA` and `time/lat/lev`
- Tau target: `tau` and `lat/lev`

If multiple tau candidates are discovered, training fails and requires explicit `tau_path`.

Tropopause CSV resolution:

1. explicit CLI/YAML `tropopause_csv`
2. exact default `data/tropopause/tropopause_features_monthly.csv`
3. first discovered CSV with required tropopause feature columns

Resume options:

- `resume.enabled: true` enables checkpoint loading before training starts
- `resume.checkpoint_path` points to a previous `.pth` checkpoint
- `resume.source_config_path` optionally records the original run config path
- `resume.load_optimizer` and `resume.load_scaler` control which states are restored

CLI overrides for `workflow train`:

- `--resume-from <checkpoint_path>`
- `--resume-config <resolved_config.yaml>`
- `--resume-load-optimizer` / `--no-resume-load-optimizer`
- `--resume-load-scaler` / `--no-resume-load-scaler`

Makefile passthrough variables:

- `RESUME_FROM`
- `RESUME_CONFIG`
- `RESUME_LOAD_OPT=true|false`
- `RESUME_LOAD_SCALER=true|false`

Examples:

```bash
python scripts/workflow.py train --config configs/run.resume.example.yaml
python scripts/workflow.py train --config configs/run.resume.example.yaml --resume-from runs/baseline_example/model_checkpoint.pth

make train CONFIG=configs/run.resume.example.yaml
make train CONFIG=configs/run.resume.example.yaml RESUME_FROM=runs/baseline_example/model_checkpoint.pth
```

## Data Preparation

Download age-of-air data:

```bash
python scripts/download_age_of_air_data.py --output data/age_of_air
```

Optional ERA5 tropopause preprocessing:

```bash
python scripts/download_and_process_era5.py --start-year 1980 --end-year 2018
python scripts/process_tropopause_features.py --infile data/tropopause/era5_tropopause_combined.nc
```

## Training Outputs

Each workflow run writes:

```text
runs/<run_id>/
  history.csv
  summary.json
  resolved_config.yaml
  model_checkpoint.pth
  plots/  # after running workflow plot
```

`summary.json` includes final/best losses, seed, device, git SHA, and resolved data paths.
For resumed runs it also records resume metadata (`resumed`, source checkpoint/config paths, and restored state flags).

## Comparison Outputs

`workflow compare` writes:

- `leaderboard.csv` (ranked by `best_val_loss`)
- `val_loss_overlay.png`

## Legacy Entrypoint

The original trainer remains available:

```bash
python scripts/train_model.py --sat-paths <...> --epochs 300
```

It now also accepts:

```bash
--tropopause-csv <path_to_tropopause_features_monthly.csv>
```

## Quality Checks

```bash
ruff check src/ tests/ scripts/
python -m pytest -v
```

CI runs lint/tests and a mini workflow smoke test.
