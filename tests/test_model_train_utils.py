import numpy as np
import pytest
import torch

from residence_time.feature_config import N_BASE_FEATURES
from residence_time.model import PINNModel
from residence_time.train import (
    _compute_supervised_loss,
    create_train_val_loaders,
    scale_variables,
    train_model,
)
from residence_time.utils import datetime64_to_year_fraction


def test_model_forward_pass() -> None:
    """Check that forward pass returns tau and D with correct shapes."""
    model = PINNModel(
        input_dim=N_BASE_FEATURES,
        include_D=True,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False}
    )
    x = torch.rand(16, N_BASE_FEATURES)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is not None and D.shape == (16, 1)


def test_model_forward_without_D() -> None:
    """Ensure D is omitted from the output when include_D is False."""
    model = PINNModel(
        input_dim=N_BASE_FEATURES,
        include_D=False,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False}
    )
    x = torch.rand(16, N_BASE_FEATURES)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is None


def test_train_val_loader_shapes() -> None:
    """Ensure that dataloaders return batches with correct shapes."""
    N = 100
    X = np.random.rand(N, N_BASE_FEATURES)
    Gamma = np.random.rand(N)
    W = np.ones(N)
    tau_R = np.random.rand(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R, batch_size=16, val_split=0.25, device='cpu'
    )

    xb, gb, wb, tb = next(iter(train_loader))
    assert xb.shape[1] == N_BASE_FEATURES
    assert gb.shape == (16, 1)
    assert wb.shape == (16, 1)
    assert tb.shape == (16, 1)


def test_scale_variables_shape() -> None:
    """Verify that scaled output has correct shape and type."""
    X = np.random.rand(100, N_BASE_FEATURES)
    X_scaled, scaler = scale_variables(X)
    assert X_scaled.shape == X.shape
    assert hasattr(scaler, "transform")


def test_datetime64_to_year_fraction_output() -> None:
    """Check conversion from datetime64 to fractional year."""
    dates = np.array(["2000-01-01", "2001-07-01"], dtype="datetime64")
    result = datetime64_to_year_fraction(dates)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[1] > result[0]


def test_adaptive_weights_evolve() -> None:
    """Verify that adaptive weighting updates lambda values over epochs."""
    N = 64
    X = np.random.rand(N, N_BASE_FEATURES)
    Gamma = np.random.rand(N)
    W = np.ones(N)
    tau_R = np.random.rand(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R, batch_size=16, val_split=0.2, device="cpu"
    )
    model = PINNModel(
        input_dim=N_BASE_FEATURES,
        include_D=True,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    import residence_time.train as train_mod
    original_tp = train_mod.USE_TROPOPAUSE_FEATURES
    train_mod.USE_TROPOPAUSE_FEATURES = False

    *_ , lambda_phys_hist, lambda_sup_hist = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        n_epochs=3,
        lambda_phys_start=1.0,
        lambda_sup=1.0,
        lambda_tp=0.0,
        adaptive_weighting=True,
        adaptive_method="gradnorm",
    )

    *_ , lambda_phys_hist2, lambda_sup_hist2 = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        n_epochs=3,
        lambda_phys_start=1.0,
        lambda_sup=1.0,
        lambda_tp=0.0,
        adaptive_weighting=True,
        adaptive_method="relobralo",
    )

    train_mod.USE_TROPOPAUSE_FEATURES = original_tp

    changed1 = lambda_phys_hist[0] != lambda_phys_hist[-1] or lambda_sup_hist[0] != lambda_sup_hist[-1]
    changed2 = lambda_phys_hist2[0] != lambda_phys_hist2[-1] or lambda_sup_hist2[0] != lambda_sup_hist2[-1]
    assert changed1 and changed2

    
def test_compute_supervised_loss_with_nan_targets() -> None:
    """Loss should ignore NaNs and run on the specified device."""
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.0], [float("nan")], [2.0]])
    loss = _compute_supervised_loss(pred, target, device="cpu")
    assert torch.isclose(loss, torch.tensor(0.5))
    assert loss.device.type == "cpu"


def test_compute_supervised_loss_all_nan() -> None:
    """All-NaN targets should yield zero loss."""
    pred = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[float("nan")], [float("nan")]])
    loss = _compute_supervised_loss(pred, target, device="cpu")
    assert loss.item() == pytest.approx(0.0)