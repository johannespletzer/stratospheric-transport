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

## Acknowledgements

This project builds on atmospheric age of air datasets of satellite and in-situ measurements provided by:

Garny et al. 2024 ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)
