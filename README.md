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

## Adaptive Loss Weighting

During training the balance between supervised and physics losses can be adjusted automatically. Set

```python
ADAPTIVE_WEIGHTING = True
```

in `residence_time/config.py` or pass ``adaptive_weighting=True`` to ``train_model`` to activate automatic updates of ``lambda_phys`` and ``lambda_sup``.

Set ``ADAPTIVE_METHOD`` to either ``"gradnorm"`` or ``"relobralo"`` to choose the weighting algorithm.

## Training utilities

See [docs/training_options.md](docs/training_options.md) for details on adaptive loss weighting and pseudo-labelling.

---

## Pseudo-Label Training

Unlabeled samples for $\tau_R$ can be leveraged by the `train_with_pseudo_labels`
function. After a configurable number of epochs, the network predicts $\tau_R$ for
all records and adds those with a small physics residual as additional training targets.

```python
from residence_time.model import PINNModel
from residence_time.train import train_with_pseudo_labels

model = PINNModel(time_encoding_config={"enabled": False})
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

labels = train_with_pseudo_labels(
    model,
    X,
    Gamma,
    W,
    tau_R,
    optimizer,
    iterations=3,
    epochs_per_iteration=20,
    residual_threshold=0.05,
    update_every=1,
)
```

Pseudo-label training utilizes additional $\Gamma$ and tropopause information to
improve predictions when direct $\tau_R$ measurements are scarce.

---

## Acknowledgements

This project builds on atmospheric age of air datasets from satellite and in-situ measurements and tropopause parameters from reanalysis. These originate from the following sources:

- Garny et al. 2024: ["Age of stratospheric air: observational data sets (v2)"](https://zenodo.org/records/13906743)

- Hoffmann and Spang 2021: ["Reanalysis Tropopause Data Repository"](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)

---

## Running the tests

The repository provides small NetCDF samples in `tests/sample_data`. Use `pytest` to run the suite locally without downloading additional files.

