import torch.nn as nn


class PINNModel(nn.Module):
    """Physics-Informed Neural Network (PINN) model for predicting τ_R (and optionally D),
    with optional tropopause-based input features.

    Parameters
    ----------
    input_dim : int, default=5
        Number of base input features: [lat, alt, time, source, Γ_EI].

    hidden_dim : int, default=64
        Hidden layer width.

    hidden_layers : int, default=3
        Number of hidden layers.

    include_D : bool, default=True
        Whether to include the D output branch.

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

        # τ_R branch
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

    def forward(self, x):
        """Forward pass through τ_R and optionally D branch.

        Parameters
        ----------
        x : torch.Tensor [N, input_dim + 3 if use_tropopause_features]

        Returns
        -------
        tau_R_pred : torch.Tensor [N, 1]
        D_pred : torch.Tensor [N, 1] or None

        """
        tau_R_pred = self.softplus(self.tauR_branch(x))
        D_pred = self.softplus(self.D_branch(x)) if self.include_D else None
        return tau_R_pred, D_pred
