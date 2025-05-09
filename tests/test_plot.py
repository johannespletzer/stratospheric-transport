from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from residence_time.model import PINNModel
from residence_time.plot import (
    plot_field_from_data,
    plot_physics_residual,
    plot_tau_R_prediction_vs_target,
    plot_training_progress,
)


@pytest.fixture(autouse=True)
def mock_show() -> None:
    """Automatically mock plt.show() in all tests to avoid GUI blocking."""
    with patch("matplotlib.pyplot.show"):
        yield


def make_dummy_dataloader(N: int = 100) -> DataLoader:
    """Create a dummy DataLoader with random values for model input/output."""
    X = torch.rand(N, 5)
    Gamma = torch.rand(N, 1)
    W = torch.ones(N, 1)
    tau = torch.rand(N, 1)
    dataset = TensorDataset(X, Gamma, W, tau)
    return DataLoader(dataset, batch_size=32)


def test_plot_field_from_data_runs() -> None:
    """Test that plot_field_from_data runs without error using dummy data."""
    model = PINNModel()
    X = np.random.rand(200, 5)
    scaler = MinMaxScaler().fit(X)

    plot_field_from_data(
        model=model,
        X=X,
        scaler_X=scaler,
        field="tau_R",
        time_value=2010.0,
        source_value=2,
        gamma_value=3.0,
        device="cpu",
    )


def test_plot_physics_residual_runs() -> None:
    """Test that plot_physics_residual executes with dummy data."""
    model = PINNModel()
    dataloader = make_dummy_dataloader()
    plot_physics_residual(model, dataloader, device="cpu")


def test_plot_tau_R_prediction_vs_target_runs() -> None:
    """Test that the τ_R prediction vs target plot function runs successfully."""
    model = PINNModel()
    dataloader = make_dummy_dataloader()
    plot_tau_R_prediction_vs_target(model, dataloader, device="cpu")


def test_plot_training_progress_runs() -> None:
    """Test the training progress plotting function using synthetic loss data."""
    n_epochs = 20
    train_losses = np.random.rand(n_epochs).tolist()
    val_losses = np.random.rand(n_epochs).tolist()
    physics_losses = np.random.rand(n_epochs).tolist()
    supervised_losses = np.random.rand(n_epochs).tolist()

    plot_training_progress(
        train_losses,
        val_losses,
        physics_losses,
        supervised_losses,
        log_scale=False,
    )
