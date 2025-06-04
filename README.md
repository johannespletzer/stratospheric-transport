# Modeling of Stratospheric Transport

Today, age of air and residence time are known concepts of atmospheric transport. However, the connection is not established analytically for all cases. This repository includes a Physics-Informed Neural Network to model **residence time (τ_R)** in the stratosphere using estimates of the **mean age of air (Γ)** from satellite, in-situ, and model datasets. Tropopause reanalysis data can be used as additional features and multiple feature extensions are possible to track changes over time on short and long scales.

The network is based on:
- Supervised learning with residence time data 
- Partly-supervised learning with two optional physics-based constraints
	- $τ_R = 2·D^2 / \Gamma$
	- $\text{residual} = \frac{d^2 \tau_R}{dt^2} + \Omega^2 \left( \tau_R - \tau_0 \right)$

---

## Project Highlights

- Model learns spatial and temporal variation of residence time
- Optional prediction of physical parameter D (diffusivity)
- Input features lat, alt, time, source and Γ (Age of air)
- Tropopause height in tropics, southern or northern Hemisphere are optional features

---

## Installation and code changes

Install the package

```bash
pip install -e .
```

---

## Data Preparation

Download age of air data for training from satellite and in-situ measurements.

```python
python scripts/download_age_of_air_data.py --output data/age_of_air
```

Download tropopause parameters for training. This is optional. Consider using the --output parameter to define a download directory due to the data size. The ``--file-limit`` option can be used to restrict the number of files downloaded per year, which is useful for testing.

```python
python scripts/download_and_process_era5.py --start-year 1980 --end-year 2018 --file-limit 2
```

Extract tropopause parameters. Input file and output directory can be declared freely.

```python
python scripts/process_tropopause_features.py --infile data/processed/era5_tropopause_combined.nc
```

---

## Acknowledgements

This project builds on atmospheric age of air datasets from satellite and in-situ measurements and tropopause parameters from reanalysis. These originate from the following sources:

- Garny et al. 2024: ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)

- Hoffmann and Spang 2021: ["Reanalysis Tropopause Data Repository"](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)
