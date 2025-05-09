from typing import Optional, Tuple

import torch
import torch.nn as nn


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

    use_tropopause_features : bool, default=False
        If True, the model expects 3 additional input features:
        [tp_WMO_tro, tp_WMO_sh_pol, tp_WMO_nh_pol].

    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        include_D: bool = True,
        use_tropopause_features: bool = False,
    ):
        super().__init__()
        self.include_D = include_D
        self.use_tropopause_features = use_tropopause_features

        # Adjust input dimension if tropopause features are included
        effective_input_dim = input_dim + (3 if use_tropopause_features else 0)

        # t_R branch
        layers_tau = []
        in_dim = effective_input_dim
        for _ in range(hidden_layers):
            layers_tau.append(nn.Linear(in_dim, hidden_dim))
            layers_tau.append(nn.Tanh())
            in_dim = hidden_dim
        layers_tau.append(nn.Linear(in_dim, 1))
        self.tauR_branch = nn.Sequential(*layers_tau)

        # D branch
        if include_D:
            layers_D = []
            in_dim = effective_input_dim
            for _ in range(hidden_layers):
                layers_D.append(nn.Linear(in_dim, hidden_dim))
                layers_D.append(nn.Tanh())
                in_dim = hidden_dim
            layers_D.append(nn.Linear(in_dim, 1))
            self.D_branch = nn.Sequential(*layers_D)
        else:
            self.D_branch = None

        self.softplus = nn.Softplus()

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
