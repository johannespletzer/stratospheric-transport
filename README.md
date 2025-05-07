# Modeling of Stratospheric Transport

Today, age of air and residence time are known concepts of atmospheric transport. However, the connection is not established analytically for all cases. This repository includes a Physics-Informed Neural Network (PINN) to model **residence time (τ_R)** in the stratosphere using estimates of the **mean age of air (Γ)** from satellite, in-situ, and model datasets.

The network is based on:
- Supervised learning with τ_R observations
- Partly-supervised learning with physics-based constraints: τ_R = 2·D² / Γ

---

## Project Highlights

- Learns spatial and temporal variation of τ_R
- Uses magnitude of Γ (mean age of air) as an input feature
- Optional prediction of diffusivity (D)
- Input features lat, alt, time, source and Γ

---

## Repository Structure

```text
stratospheric-transport/
│
├── residence_time/         # Core Python package
│   ├── data.py             # Load satellite, in-situ, model datasets
│   ├── model.py            # PINN model definition
│   ├── train.py            # Training loop with tracking
│   ├── utils.py
│
├── scripts/
├── tests/            
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
```

---

## Data Preparation

Download tropopause parameters for training. Consider using the --output parameter to define a download directory due to the data size.

```python
python script/download_and_process_era5.py --start-year 1980 --end-year 2018
```
---

## Acknowledgements

This project builds on atmospheric age of air datasets from satellite and in-situ measurements and tropopause parameters from reanalysis. These originate from the following sources:

- Garny et al. 2024: ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)

- Hoffmann and Spang 2021: ["Reanalysis Tropopause Data Repository"](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)
