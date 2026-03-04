# Run Configuration

This page documents how `scripts/workflow.py train` resolves settings from YAML
and CLI flags.

## Configuration Model

Training configuration is loaded from a YAML file (for example
`configs/run.example.yaml`) and merged with defaults in
`src/residence_time/workflow_data.py`.

Most settings can then be overridden from the command line. For path settings,
the precedence is:

1. Explicit CLI values
2. Explicit YAML values
3. Autodiscovery (if enabled)

## Autodiscovery Defaults

Default values:

- `enabled: true`
- `search_roots: ["data"]`
- `discover_model_paths: false`
- `discover_tau_path: false`

What these do:

- `enabled`: turns discovery on/off for path candidates under `search_roots`
- `search_roots`: directories recursively scanned for `*.nc` and `*.csv`
- `discover_model_paths`: allows autodiscovery to populate `model_paths`
- `discover_tau_path`: allows autodiscovery to populate `tau_path`

Note: autodiscovery still requires at least one resolved input dataset from
`sat_paths`, `insitu_paths`, or `model_paths`.

## Dataset Schema Expectations

Autodiscovery classifies NetCDF files by schema:

- Satellite AoA:
  - variables: `AoA`, `AoA_STD`
  - coordinates/dimensions include: `time`, `lat`, `alt`
- In-situ AoA:
  - variables: `Mean_Age_SF6_corr`, `Mean_Age_CO2`,
    `Mean_Age_SF6_corr_STD`, `Mean_Age_CO2_STD`
  - dimensions include: `season`, `lat`, `Altitude`
- Model AoA:
  - variable: `AOA`
  - coordinates/dimensions include: `time`, `lat`, `lev`
- Tau target:
  - variable: `tau` or `Residence time [yrs]`
  - coordinates/dimensions include: `lat`, `lev`

If `discover_tau_path` is enabled and multiple tau candidates are found,
training fails with an ambiguity error and you must set an explicit `tau_path`.

## Tropopause CSV Resolution

`tropopause_csv` resolution order:

1. Explicit CLI `--tropopause-csv` or YAML `data.tropopause_csv`
2. Exact path `data/tropopause/tropopause_features_monthly.csv`
3. First discovered CSV under `search_roots` that contains all required columns

Required columns:

- `time`
- `tp_WMO_tro`
- `tp_WMO_sh_pol`
- `tp_WMO_nh_pol`
- `tp_WMO_tro_std`
- `tp_WMO_sh_pol_std`
- `tp_WMO_nh_pol_std`

If `use_tropopause_features: true` and no CSV is resolved, training exits with
an error.

## Resume Training

YAML `resume` section:

- `resume.enabled`: enable checkpoint loading
- `resume.checkpoint_path`: checkpoint (`.pth`) to load
- `resume.source_config_path`: optional reference to source run config
- `resume.load_optimizer`: restore optimizer state
- `resume.load_scaler`: restore feature scaler state

CLI resume overrides:

- `--resume-from <checkpoint_path>`
- `--resume-config <resolved_config.yaml>`
- `--resume-load-optimizer` / `--no-resume-load-optimizer`
- `--resume-load-scaler` / `--no-resume-load-scaler`

Makefile passthrough variables:

- `RESUME_FROM`
- `RESUME_CONFIG`
- `RESUME_LOAD_OPT=true|false`
- `RESUME_LOAD_SCALER=true|false`

## Practical Examples

Baseline training:

```bash
python scripts/workflow.py train --config configs/run.example.yaml
```

Resume from a previous checkpoint:

```bash
python scripts/workflow.py train --config configs/run.resume.example.yaml
python scripts/workflow.py train --config configs/run.resume.example.yaml --resume-from runs/baseline_example/model_checkpoint.pth
```

Explicitly choose tau when autodiscovery would be ambiguous:

```bash
python scripts/workflow.py train --config configs/run.example.yaml --tau-path data/targets/tau_selected.nc
```
