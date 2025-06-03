from collections import ChainMap
from typing import Optional, Tuple

import torch
import torch.nn as nn

from residence_time.config import (
    DEFAULT_TIME_ENCODING_CONFIG,
    DROPOUT_RATE,
    MEAN_TAU,
    PHYSICS_CONSTRAINT,
    USE_TROPOPAUSE_FEATURES,
    INPUT_DIM,
)


class PINNModel(nn.Module):
    """Physics-Informed Neural Network (PINN) model for predicting τ_R (and optionally D).

    This model optionally uses tropopause-based input features and supports a dual-branch
    architecture for τ_R and effective diffusivity D.

    Parameters
    ----------
    input_dim : int, default=5
        Number of base input features: [lat, alt, time, source, G_EI].

    hidden_dim : int, default=64
        Number of hidden units per hidden layer.

    hidden_layers : int, default=3
        Number of hidden layers in each branch.

    include_D : bool, default=True
        Whether to include a separate output branch for diffusivity D.

    use_tropopause_features : bool, default=USE_TROPOPAUSE_FEATURES
        If True, the model expects 3 additional input features:
        [tp_WMO_tro, tp_WMO_sh_pol, tp_WMO_nh_pol].

    time_encoding_config : dict, default=DEFAULT_TIME_ENCODING_CONFIG
        Option to activate seasonal and annual features via config

    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        include_D: bool = True,
        use_tropopause_features: bool = USE_TROPOPAUSE_FEATURES,
        time_encoding_config: dict = DEFAULT_TIME_ENCODING_CONFIG
    ):
        super().__init__()
        self.include_D = include_D if PHYSICS_CONSTRAINT == 'diffusivity' else False

        self.tauR_intercept = nn.Parameter(torch.tensor(MEAN_TAU)) if PHYSICS_CONSTRAINT == 'harmonic' else None

        time_encoding_config = dict(ChainMap(time_encoding_config, DEFAULT_TIME_ENCODING_CONFIG))

        # Adjust input dimension if other features are included
        effective_input_dim = input_dim + (4 if use_tropopause_features else 0)

        if time_encoding_config.get("enabled", False):
            effective_input_dim += 2 if time_encoding_config["use_cyclical"] else 0
            effective_input_dim += time_encoding_config["n_rbf_seasonal"] if time_encoding_config["use_rbf_seasonal"] else 0
            effective_input_dim += time_encoding_config["n_rbf_absolute"] if time_encoding_config["use_rbf_absolute"] else 0

        # t_R branch
        layers_tau = []
        in_dim = effective_input_dim
        for _ in range(hidden_layers):
            layers_tau.append(nn.Linear(in_dim, hidden_dim))
            layers_tau.append(nn.BatchNorm1d(hidden_dim))
            layers_tau.append(nn.Tanh())
            layers_tau.append(nn.Dropout(DROPOUT_RATE))
            in_dim = hidden_dim
        layers_tau.append(nn.Linear(in_dim, 1))
        self.tauR_branch = nn.Sequential(*layers_tau)

        # D branch
        if include_D:
            layers_D = []
            in_dim = effective_input_dim
            for _ in range(hidden_layers):
                layers_D.append(nn.Linear(in_dim, hidden_dim))
                layers_D.append(nn.BatchNorm1d(hidden_dim))
                layers_D.append(nn.Tanh())
                layers_D.append(nn.Dropout(DROPOUT_RATE))
                in_dim = hidden_dim
            layers_D.append(nn.Linear(in_dim, 1))
            self.D_branch = nn.Sequential(*layers_D)
        else:
            self.D_branch = None

        self.softplus = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the t_R and optionally D output branches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, input_dim + 3 if tropopause features are used].

        Returns
        -------
        tau_R_pred : torch.Tensor
            Predicted residence time t_R, shape [N, 1].

        D_pred : torch.Tensor or None
            Predicted diffusivity D, shape [N, 1], or None if include_D is False.

        """
        tau_R_pred = self.softplus(self.tauR_branch(x))
        D_pred = self.softplus(self.D_branch(x)) if self.include_D else None

        return tau_R_pred, D_pred
