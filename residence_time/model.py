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
        layers_tau = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers_tau.append(nn.Linear(in_dim, hidden_dim))
            layers_tau.append(nn.Tanh())  # non-linear activation for hidden layers
            in_dim = hidden_dim
        layers_tau.append(nn.Linear(in_dim, 1))  # output layer for tau_R
        self.tauR_branch = nn.Sequential(*layers_tau)
        
        # Define the D branch (MLP), if enabled
        if include_D:
            layers_D = []
            in_dim = input_dim
            for _ in range(hidden_layers):
                layers_D.append(nn.Linear(in_dim, hidden_dim))
                layers_D.append(nn.Tanh())
                in_dim = hidden_dim
            layers_D.append(nn.Linear(in_dim, 1))  # output layer for D
            self.D_branch = nn.Sequential(*layers_D)
        else:
            self.D_branch = None
        
        # Softplus activation for outputs to ensure positivity
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Forward pass: returns (tau_R_pred, D_pred).
        If include_D=False, D_pred will be None.

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
        # Compute tau_R prediction
        tau_R_pred = self.softplus(self.tauR_branch(x))
        # Compute D prediction if applicable
        D_pred = self.softplus(self.D_branch(x)) if self.include_D else None
            
        return tau_R_pred, D_pred
