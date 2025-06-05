# Advanced Training Options

This project allows two strategies to stabilize training when the amount of labeled residence-time data is limited: **adaptive loss weighting** and **pseudo-labelling**.

## Adaptive loss weighting

Physics-informed models combine losses from physical constraints and from supervised observations. The weight applied to the physics term can be adapted over time. A typical schedule starts with a high value and decays every few epochs:

```python
train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=300,
    lambda_phys_start=5.0,   # initial weight for the physics term
    lambda_sup=1.0,          # supervised weight stays constant
    decay_rate=0.9,          # apply every ~30 epochs
)
```

A slower decay keeps the physical relationship strong; faster decay favours supervised data. Monitor validation losses to adjust the start value and decay rate.

## Pseudo-labelling

Pseudo-labelling uses predictions from a previous model run as additional targets when τ₍R₎ observations are missing. Supply a path to pseudo-label files in the dataloader step and enable the option:

```python
X, Gamma, W, tau = load_training_data(...)
# tau contains NaNs where no measurements are available
pseudo_path = "data/pseudo_labels.nc"
train_loader, val_loader = create_train_val_loaders(
    X, Gamma, W, tau, batch_size=256,
    time_encoding_config={"enabled": True},
    pseudo_labels=pseudo_path,  # new option
)
```

The training routine will merge these pseudo targets with the real measurements. Confidence in pseudo-labels can be tuned through their associated weights.

### Iterative pseudo-label training

The `train_with_pseudo_labels` helper automates multiple rounds of training and
label refinement:

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

This process leverages additional Γ and tropopause information to improve
predictions when direct τ₍R₎ observations are scarce.

## Balancing physics and supervision

- Start with `lambda_phys_start` around 1–10 depending on the uncertainty of Γ.
- Use a `decay_rate` between 0.9 and 0.98 so that the physics term slowly decreases.
- Keep `lambda_sup` at 1.0 for most cases. Increase it if supervised data is abundant.
- When pseudo-labelling, assign lower weights to the pseudo-targets than to real observations.

## Running the tests

Sample NetCDF files for the unit tests are included under `tests/sample_data`. To run the test suite locally, simply execute:

```bash
pytest
```

Pytest will automatically find the local sample data and run without downloading from the internet.
