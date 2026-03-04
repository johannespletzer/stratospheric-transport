# Data and Outputs

This page summarizes input data preparation and artifacts produced by the
workflow commands.

## Input Data Preparation

### Age-of-air observations

Download and extract the Zenodo dataset:

```bash
python scripts/download_age_of_air_data.py --output data/age_of_air
```

### ERA5 tropopause features (optional)

Download and concatenate ERA5 tropopause NetCDF files:

```bash
python scripts/download_and_process_era5.py --start-year 1980 --end-year 2018 --output data/tropopause
```

Convert the combined NetCDF file to monthly tropopause features:

```bash
python scripts/process_tropopause_features.py --infile data/tropopause/era5_tropopause_combined.nc --outfile data/tropopause/tropopause_features_monthly.csv
```

## Expected Data Layout

Typical layout under `data/`:

```text
data/
  age_of_air/...
  tropopause/
    era5_tropopause_combined.nc
    tropopause_features_monthly.csv
```

Notes:

- Input NetCDF files do not need fixed file names; they are matched by schema.
- Tropopause CSV has one preferred exact default path:
  `data/tropopause/tropopause_features_monthly.csv`.
- Search scope for autodiscovery is controlled by `data.autodiscover.search_roots`.

## Training Artifacts (`workflow train`)

Each run creates:

```text
runs/<run_id>/
  history.csv
  summary.json
  resolved_config.yaml
  model_checkpoint.pth
```

By default, if no `run_id` is provided, the workflow generates one like
`run_YYYYMMDD_HHMMSS` (UTC).

### `history.csv`

Per-epoch losses with columns:

- `epoch`
- `train_loss`
- `val_loss`
- `physics_loss`
- `supervised_loss`

### `summary.json`

Key fields include:

- Run metadata: `run_id`, `run_dir`, `created_at_utc`, `git_sha`
- Runtime/data metadata: `physics_constraint`, `seed`, `device`, `n_samples`, `data_paths`
- Final and best losses: `final_*`, `best_*`, `best_epoch_by_val_loss`
- Resume metadata:
  - `resumed`
  - `resume_checkpoint_path`
  - `resume_source_config_path`
  - `resume_loaded_optimizer`
  - `resume_loaded_scaler`

### `resolved_config.yaml`

Merged run configuration written at launch, including resolved absolute paths,
path source labels, and autodiscovered candidates.

## Plot Artifacts (`workflow plot`)

Default output directory:

```text
runs/<run_id>/plots/
```

Produced files:

- `training_losses.png`
- `loss_components.png`

You can override output location with `--output-dir`.

## Comparison Artifacts (`workflow compare`)

Default output directory:

```text
reports/compare/latest/
```

Produced files:

- `leaderboard.csv` (ranked by `best_val_loss`)
- `val_loss_overlay.png` (validation-loss curves for compared runs)
