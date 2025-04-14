import torch
import torch.nn as nn


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model for predicting residence time (τ_R)
    and optionally effective diffusivity (D), based on input features such as latitude,
    altitude, time, source type, and mean age of air (Γ_EI).

    The model has two separate branches:
    - τ_R branch: always active
    - D branch: only active if include_D=True

    Output predictions are passed through a softplus activation to ensure positivity.

    Parameters
    ----------
    input_dim : int, default=5
        Number of input features. Typically: [lat, alt, time, source, Γ_EI].

    hidden_dim : int, default=64
        Number of hidden units per layer.

    hidden_layers : int, default=3
        Number of hidden layers in each branch.

    include_D : bool, default=True
        Whether to include the branch that predicts effective diffusivity D.
        If False, the model only predicts τ_R.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, hidden_layers: int = 3, include_D: bool = True):
        super().__init__()
        self.include_D = include_D

        # Define τ_R branch (shared architecture: MLP with Tanh)
        self.tauR_branch = self._build_mlp(input_dim, hidden_dim, hidden_layers, output_dim=1)

        # Define D branch if enabled
        self.D_branch = self._build_mlp(input_dim, hidden_dim, hidden_layers, output_dim=1) if include_D else None

        # Ensure positive outputs
        self.softplus = nn.Softplus()

    def _build_mlp(self, input_dim, hidden_dim, hidden_layers, output_dim):
        """Utility to build a feedforward MLP."""
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor [N, input_dim]
            Input tensor with features [lat, alt, time, source, Γ_EI].

        Returns
        -------
        tau_R_pred : torch.Tensor [N, 1]
            Predicted residence time (τ_R), always returned.

        D_pred : torch.Tensor [N, 1] or None
            Predicted diffusivity (D), or None if include_D=False.
        """
        tau_R_pred = self.softplus(self.tauR_branch(x))
        D_pred = self.softplus(self.D_branch(x)) if self.include_D else None
        return tau_R_pred, D_pred
