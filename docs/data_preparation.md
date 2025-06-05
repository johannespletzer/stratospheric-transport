# Data Preparation

To train the network you need measurements of the mean age of air and optional tropopause parameters. The repository provides scripts for downloading and processing these data.

Download the satellite and in‑situ age of air observations:

```bash
python scripts/download_age_of_air_data.py --output data/age_of_air
```

Retrieve and process ERA5 tropopause fields. The `--file-limit` flag is useful when testing the pipeline with a small subset of files:

```bash
python scripts/download_and_process_era5.py --start-year 1980 --end-year 2018 --file-limit 2
```

Extract the final training features from the combined file:

```bash
python scripts/process_tropopause_features.py --infile data/processed/era5_tropopause_combined.nc
```
