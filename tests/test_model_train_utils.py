import numpy as np
import torch

from residence_time.model import PINNModel
from residence_time.train import create_train_val_loaders, scale_variables
from residence_time.utils import datetime64_to_year_fraction


def test_model_forward_pass() -> None:
    """Check that forward pass returns tau and D with correct shapes."""
    model = PINNModel(
        input_dim=6,
        include_D=True,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False}
    )
    x = torch.rand(16, 6)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is not None and D.shape == (16, 1)


def test_model_forward_without_D() -> None:
    """Ensure D is omitted from the output when include_D is False."""
    model = PINNModel(
        input_dim=6,
        include_D=False,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False}
    )
    x = torch.rand(16, 6)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is None


def test_train_val_loader_shapes() -> None:
    """Ensure that dataloaders return batches with correct shapes."""
    N = 100
    X = np.random.rand(N, 6)
    Gamma = np.random.rand(N)
    W = np.ones(N)
    tau_R = np.random.rand(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R, batch_size=16, val_split=0.25, device='cpu'
    )

    xb, gb, wb, tb = next(iter(train_loader))
    assert xb.shape[1] == 6
    assert gb.shape == (16, 1)
    assert wb.shape == (16, 1)
    assert tb.shape == (16, 1)


def test_scale_variables_shape() -> None:
    """Verify that scaled output has correct shape and type."""
    X = np.random.rand(100, 6)
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
