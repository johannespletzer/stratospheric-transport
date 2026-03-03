import numpy as np
import pytest
import torch

import residence_time.train as train_module
from residence_time.model import PINNModel
from residence_time.train import (
    add_rbf_time_features,
    apply_time_encoding,
    create_train_val_loaders,
    extract_phase_and_transform_with_scaler,
    scale_variables,
    scale_variables_columnwise,
    train_model,
)
from residence_time.utils import datetime64_to_year_fraction


def test_model_forward_pass() -> None:
    """Test that the PINN model forward method returns correct shapes."""
    model = PINNModel(input_dim=5, include_D=True)
    x = torch.rand(16, 5)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is not None and D.shape == (16, 1)


def test_model_forward_without_D() -> None:
    """Test PINN model forward when include_D=False."""
    model = PINNModel(input_dim=5, include_D=False)
    x = torch.rand(16, 5)
    tau, D = model(x)
    assert tau.shape == (16, 1)
    assert D is None


def test_train_val_loader_shapes() -> None:
    """Ensure that dataloaders return batches with correct shapes."""
    N = 100
    X = np.random.rand(N, 5)
    Gamma = np.random.rand(N)
    W = np.ones(N)
    tau_R = np.random.rand(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R, batch_size=16, val_split=0.25, device='cpu'
    )

    xb, gb, wb, tb = next(iter(train_loader))
    assert xb.shape[1] == 5
    assert gb.shape == (16, 1)
    assert wb.shape == (16, 1)
    assert tb.shape == (16, 1)
    assert train_loader.sampler.__class__.__name__ == "RandomSampler"
    assert val_loader.sampler.__class__.__name__ == "SequentialSampler"


def test_scale_variables_shape() -> None:
    """Verify that scaled output has correct shape and type."""
    X = np.random.rand(100, 5)
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


def test_train_model_without_supervised_targets_stays_finite() -> None:
    """Train for one epoch without tau targets and verify finite losses."""
    N = 64
    X = np.random.rand(N, 5)
    Gamma = np.random.rand(N) + 0.1
    W = np.ones(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R=None, batch_size=16, val_split=0.25, device='cpu'
    )

    model = PINNModel(input_dim=5, include_D=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses, physics_losses, supervised_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=1,
    )

    assert np.isfinite(train_losses[0])
    assert np.isfinite(val_losses[0])
    assert np.isfinite(physics_losses[0])
    assert np.isfinite(supervised_losses[0])
    assert supervised_losses[0] == 0.0


def test_scale_variables_columnwise_scales_time_column() -> None:
    """Time feature must be standardized after scaling (no longer passthrough)."""
    X = np.array(
        [
            [10.0, 18.0, 1985.0, 0.0, 2.0],
            [-5.0, 22.0, 2010.0, 1.0, 3.0],
            [30.0, 25.0, 2020.0, 2.0, 2.5],
        ]
    )
    X_scaled, scaler = scale_variables_columnwise(X, scaler="StandardScaler")
    # Time should now be standardized, not equal to original year values.
    assert not np.allclose(X_scaled[:, 2], X[:, 2])
    # Standardized time should have approximately zero mean.
    np.testing.assert_allclose(X_scaled[:, 2].mean(), 0.0, atol=1e-10)


def test_scale_variables_columnwise_keeps_additional_features() -> None:
    """Scaling must preserve extra feature columns (e.g., tropopause features)."""
    X = np.array(
        [
            [10.0, 18.0, 1985.0, 0.0, 2.0, 100.0, 200.0, 300.0],
            [-5.0, 22.0, 2010.0, 1.0, 3.0, 110.0, 210.0, 310.0],
            [30.0, 25.0, 2020.0, 2.0, 2.5, 120.0, 220.0, 320.0],
        ]
    )
    X_scaled, _ = scale_variables_columnwise(X, scaler="StandardScaler")
    assert X_scaled.shape == X.shape
    # Time is now standardized, not preserved.
    assert not np.allclose(X_scaled[:, 2], X[:, 2])
    # Source ID is centered: {0, 1, 2} -> {-1, 0, 1}.
    np.testing.assert_allclose(X_scaled[:, 3], X[:, 3] - 1)
    assert not np.allclose(X_scaled[:, 5:], X[:, 5:])


def test_add_rbf_time_features_clips_absolute_time_range() -> None:
    """Absolute RBF encoding should clip years outside [t_min, t_max]."""
    X = np.array(
        [
            [0.0, 0.0, 1900.0, 0.0, 1.0],  # below t_min
            [0.0, 0.0, 2100.0, 0.0, 1.0],  # above t_max
        ]
    )
    X_ext = add_rbf_time_features(
        X,
        time_col=2,
        kind="absolute",
        n_rbf=3,
        gamma=1.0,
        t_min=1980.0,
        t_max=2025.0,
    )
    rbf = X_ext[:, 5:]
    centers = np.linspace(0, 1, 3)
    expected_low = np.exp(-1.0 * (0.0 - centers) ** 2)
    expected_high = np.exp(-1.0 * (1.0 - centers) ** 2)
    np.testing.assert_allclose(rbf[0], expected_low)
    np.testing.assert_allclose(rbf[1], expected_high)


def test_harmonic_mode_rejects_time_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harmonic physics mode should reject precomputed time encodings."""
    monkeypatch.setattr(train_module, "PHYSICS_CONSTRAINT", "harmonic")

    X = np.random.rand(40, 5)
    X[:, 2] = np.linspace(1990.0, 2020.0, 40)
    Gamma = np.random.rand(40) + 0.1
    W = np.ones(40)
    tau_R = np.random.rand(40)

    with pytest.raises(ValueError, match="Time encoding is not supported"):
        create_train_val_loaders(
            X,
            Gamma,
            W,
            tau_R=tau_R,
            batch_size=16,
            val_split=0.2,
            device="cpu",
            time_encoding_config={"enabled": True, "use_cyclical": True},
        )


def test_train_model_requires_d_branch_for_diffusivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diffusivity physics mode requires a model that predicts D."""
    monkeypatch.setattr(train_module, "PHYSICS_CONSTRAINT", "diffusivity")

    N = 32
    X = np.random.rand(N, 5)
    Gamma = np.random.rand(N) + 0.1
    W = np.ones(N)

    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R=None, batch_size=8, val_split=0.25, device="cpu"
    )

    model = PINNModel(input_dim=5, include_D=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="requires a model with diffusivity output"):
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=1,
        )


def test_create_train_val_loaders_fallback_floors_time() -> None:
    """Fallback path (no seasonal_phase) should floor the time column."""
    X = np.array(
        [
            [0.0, 20.0, 2000.10, 0.0, 2.0],
            [1.0, 21.0, 2001.90, 0.0, 2.1],
            [2.0, 22.0, 2002.25, 0.0, 2.2],
            [3.0, 23.0, 2003.75, 0.0, 2.3],
        ]
    )
    Gamma = np.ones(4)
    W = np.ones(4)
    tau = np.ones(4)

    # Without seasonal_phase, the function extracts phase internally and
    # replaces the time column with floor(time).
    train_loader, val_loader = create_train_val_loaders(
        X, Gamma, W, tau_R=tau, batch_size=2, val_split=0.5, device="cpu"
    )

    x_train, _, _, _ = next(iter(train_loader))
    x_val, _, _, _ = next(iter(val_loader))
    all_years = np.concatenate([x_train[:, 2].numpy(), x_val[:, 2].numpy()])
    np.testing.assert_allclose(all_years, np.floor(all_years))


def test_apply_time_encoding_season_is_independent_from_base_year() -> None:
    """Seasonal encodings should depend on provided phase, not integer year value."""
    X = np.array(
        [
            [0.0, 20.0, 2000.0, 0.0, 2.0],
            [1.0, 21.0, 2015.0, 0.0, 2.0],
        ]
    )
    phase = np.array([0.1, 0.1])
    cfg = {"enabled": True, "use_cyclical": True}

    X_enc = apply_time_encoding(X, cfg, seasonal_phase=phase)
    # last two columns are sin/cos cyclical features
    np.testing.assert_allclose(X_enc[0, -2:], X_enc[1, -2:])


def test_extract_phase_and_transform_with_prefit_scaler() -> None:
    """Resume helper should transform with an existing fitted scaler."""
    X_fit = np.array(
        [
            [0.0, 20.0, 2000.0, 0.0, 2.0],
            [1.0, 21.0, 2001.0, 1.0, 2.1],
            [2.0, 22.0, 2002.0, 2.0, 2.2],
        ]
    )
    fitted_input = X_fit.copy()
    fitted_input[:, 2] = np.floor(fitted_input[:, 2])
    _, scaler = scale_variables_columnwise(fitted_input, scaler="StandardScaler")

    X_new = np.array(
        [
            [3.0, 23.0, 2003.25, 0.0, 2.3],
            [4.0, 24.0, 2004.75, 1.0, 2.4],
        ]
    )
    X_expected = X_new.copy()
    expected_phase = np.mod(X_expected[:, 2], 1.0)
    X_expected[:, 2] = np.floor(X_expected[:, 2])

    X_scaled, seasonal_phase, time_std = extract_phase_and_transform_with_scaler(
        X_new,
        scaler_X=scaler,
    )

    np.testing.assert_allclose(seasonal_phase, expected_phase)
    np.testing.assert_allclose(X_scaled, scaler.transform(X_expected))
    assert np.isclose(time_std, float(scaler.named_transformers_["time"].scale_[0]))
