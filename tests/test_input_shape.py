import torch

from residence_time.model import PINNModel


def test_pinnmodel_input_shape_without_tropopause():
    """Test PINNModel with default input_dim=5"""
    model = PINNModel(use_tropopause_features=False)
    x = torch.randn(10, 5)  # batch of 10 samples with 5 features
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)

def test_pinnmodel_input_shape_with_tropopause():
    """Test PINNModel with 3 extra tropopause features"""
    model = PINNModel(use_tropopause_features=True)
    x = torch.randn(10, 8)  # batch of 10 samples with 8 features
    tau_R, D = model(x)
    assert tau_R.shape == (10, 1)
    if model.include_D:
        assert D is not None and D.shape == (10, 1)
