import numpy as np
import torch

from residence_time import train as train_module
from residence_time.feature_config import N_BASE_FEATURES
from residence_time.model import PINNModel
from residence_time.train import train_with_pseudo_labels


def test_train_with_pseudo_labels_runs() -> None:
    """Ensure pseudo-label loop executes and updates some labels."""
    N = 40
    X = np.random.rand(N, N_BASE_FEATURES)
    Gamma = np.random.rand(N)
    W = np.ones(N)
    tau_R = np.random.rand(N)
    tau_R[::2] = np.nan

    model = PINNModel(
        input_dim=N_BASE_FEATURES,
        include_D=True,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_module.USE_TROPOPAUSE_FEATURES = False

    updated = train_with_pseudo_labels(
        model,
        X,
        Gamma,
        W,
        tau_R,
        optimizer,
        iterations=1,
        epochs_per_iteration=1,
        residual_threshold=1e6,
        update_every=1,
        batch_size=8,
        val_split=0.2,
        device="cpu",
    )

    assert np.isnan(updated).sum() < np.isnan(tau_R).sum()
