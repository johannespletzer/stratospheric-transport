import torch.nn as nn

class PINNModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, hidden_layers=3, include_D=True):
        """
        Physics-Informed NN model with separate branches for tau_R and D.
        :param input_dim: Number of input features (default 5: lat, alt, time, source, Gamma).
        :param hidden_dim: Number of neurons in each hidden layer.
        :param hidden_layers: Number of hidden layers in each branch.
        :param include_D: If True, include the D prediction branch; if False, only tau_R is predicted.
        """

        super(PINNModel, self).__init__()
        self.include_D = include_D
        
        # Define the tau_R branch (MLP)
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
        """
        # Compute tau_R prediction
        tau_R_pred = self.softplus(self.tauR_branch(x))
        # Compute D prediction if applicable
        D_pred = None
        if self.include_D:
            D_pred = self.softplus(self.D_branch(x))
            
        return tau_R_pred, D_pred
