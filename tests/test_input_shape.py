import torch

from residence_time.model import PINNModel

BASE_INPUT_DIM = 5


def test_pinnmodel_base_input() -> None:
    """Test with base features only (no tropopause, no time features)."""
    model = PINNModel(input_dim=BASE_INPUT_DIM, use_tropopause_features=False, time_encoding_config={"enabled": False})
    x = torch.randn(10, BASE_INPUT_DIM)
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)


def test_pinnmodel_with_tropopause() -> None:
    """Test with base + tropopause features (no time)."""
    model = PINNModel(input_dim=BASE_INPUT_DIM, use_tropopause_features=True, time_encoding_config={"enabled": False})
    x = torch.randn(10, BASE_INPUT_DIM + 3)
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)


def test_pinnmodel_with_cyclical_time() -> None:
    """Test with cyclical time encoding."""
    time_cfg = {"enabled": True, "use_cyclical": True}
    model = PINNModel(input_dim=BASE_INPUT_DIM, use_tropopause_features=False, time_encoding_config=time_cfg)
    x = torch.randn(10, BASE_INPUT_DIM + 2)  # sin/cos
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)


def test_pinnmodel_with_absolute_rbf() -> None:
    """Test with absolute RBF time encoding."""
    time_cfg = {
        "enabled": True,
        "use_cyclical": False,
        "use_rbf_absolute": True,
        "n_rbf_absolute": 10
    }
    model = PINNModel(input_dim=BASE_INPUT_DIM, use_tropopause_features=False, time_encoding_config=time_cfg)
    x = torch.randn(10, BASE_INPUT_DIM + 10)
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)


def test_pinnmodel_with_all_time_features_and_tropopause() -> None:
    """Test full feature combination: base + tropopause + cyclical + RBF."""
    time_cfg = {
        "enabled": True,
        "use_cyclical": True,
        "use_rbf_seasonal": True,
        "n_rbf_seasonal": 12,
        "use_rbf_absolute": True,
        "n_rbf_absolute": 8
    }
    model = PINNModel(input_dim=BASE_INPUT_DIM, use_tropopause_features=True, time_encoding_config=time_cfg)

    extra = 3 + 2 + 12 + 8  # tropopause + cyclical + seasonal RBF + absolute RBF
    x = torch.randn(10, BASE_INPUT_DIM + extra)
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)
