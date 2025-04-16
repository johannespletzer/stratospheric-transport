import torch


def create_train_val_loaders(
    X,
    Gamma,
    W,
    tau_R=None,
    batch_size=256,
    val_split=0.2,
    device='cuda'
    ):
    """
    Splits data into training and validation sets and returns DataLoaders
    with (X, Gamma, W, tau_R) tuples for PINN training.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape [N, D] (e.g., lat, alt, time, source, Gamma_EI).

    Gamma : np.ndarray
        Effective irreversibility (Γ_EI) values, shape [N,].

    W : np.ndarray
        Weights for the physics loss (e.g., 1 / std²), shape [N,].

    tau_R : np.ndarray or None, optional
        Supervised residence time target values τ_R, shape [N,]. If None,
        NaNs will be filled for compatibility with the model interface.

    batch_size : int, default=256
        Batch size for the DataLoaders.

    val_split : float, default=0.2
        Fraction of the dataset to use for validation.

    device : str, default='cuda'
        Device to which the tensors are moved (e.g., 'cuda' or 'cpu').

    Returns
    -------
    train_loader : DataLoader
        PyTorch DataLoader for training data.

    val_loader : DataLoader
        PyTorch DataLoader for validation data.
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    if tau_R is not None:
        X_train, X_val, Gamma_train, Gamma_val, W_train, W_val, tau_train, tau_val = train_test_split(
            X, Gamma, W, tau_R, test_size=val_split, random_state=42
        )
    else:
        X_train, X_val, Gamma_train, Gamma_val, W_train, W_val = train_test_split(
            X, Gamma, W, test_size=val_split, random_state=42
        )
        tau_train = np.full(len(X_train), np.nan, dtype=np.float32)
        tau_val = np.full(len(X_val), np.nan, dtype=np.float32)

    def make_loader(X, Gamma, W, tau):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Gamma_tensor = torch.tensor(Gamma, dtype=torch.float32).unsqueeze(1).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).unsqueeze(1).to(device)
        tau_tensor = torch.tensor(tau, dtype=torch.float32).unsqueeze(1).to(device)
        dataset = TensorDataset(X_tensor, Gamma_tensor, W_tensor, tau_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return make_loader(X_train, Gamma_train, W_train, tau_train), \
           make_loader(X_val, Gamma_val, W_val, tau_val)


def scale_variables(X_obs):

    from sklearn.preprocessing import MinMaxScaler

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_obs)

    return X_scaled, scaler_X


def scale_variables_columnwise(X_obs, scaler='MinMaxScaler'):
    """
    Scales selected columns of X_obs using StandardScaler, and leaves others untouched.

    Columns:
    [0] Latitude
    [1] Altitude
    [2] Time
    [3] Source ID (no scaling)
    [4] Gamma_EI (no scaling)

    Returns
    -------
    X_scaled : np.ndarray
        Scaled input array.
    scaler_X : sklearn.compose.ColumnTransformer
        Fitted transformer for later use on validation/test data.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer

    scaler_X = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler() if scaler == 'StandardScaler' else MinMaxScaler(), [0, 1, 2]),
            ("identity", FunctionTransformer(validate=False), [3, 4])  # identity transform
        ]
    )

    X_scaled = scaler_X.fit_transform(X_obs)
    return X_scaled, scaler_X


def train_model(model, train_loader, val_loader, optimizer, n_epochs=500,
                              lambda_phys_start=1.0, lambda_sup_start=1.0, decay_rate=0.95):
    """
    Trains a PINN model using both physics-based and supervised losses.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model to train.

    train_loader : DataLoader
        DataLoader for training data.

    val_loader : DataLoader
        DataLoader for validation data.

    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.

    n_epochs : int
        Number of training epochs.

    lambda_phys_start : float
        Initial weight for physics-based loss.

    lambda_sup_start : float
        Initial weight for supervised loss.

    decay_rate : float
        Exponential decay rate for lambda_phys over epochs.

    Returns
    -------
    train_losses : list of float
    val_losses : list of float
    physics_losses : list of float
    supervised_losses : list of float
    """
    train_losses, val_losses, physics_losses, supervised_losses = [], [], [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        phys_loss_total = 0
        sup_loss_total = 0

        # Exponentially decay physics constraint weight
        lambda_phys = lambda_phys_start * (decay_rate ** (epoch // 50))
        lambda_sup = lambda_sup_start

        for Xb, Gb, Wb, Tb in train_loader:
            optimizer.zero_grad()  # always do this before backward()
        
            Gamma = Gb.clamp(min=1e-2)
            tau_R_pred, D_pred = model(Xb)
        
#            print("D_pred stats:",
#                  torch.isnan(D_pred).any().item(),
#                  D_pred.min().item() if not torch.isnan(D_pred).any() else 'nan',
#                  D_pred.max().item() if not torch.isnan(D_pred).any() else 'nan')
        
            if D_pred is not None:
                D_clamped = D_pred.clamp(min=1e-2, max=10.0)
                tau_R_phys = 2 * D_clamped**2 / Gamma
                loss_phys = torch.mean(Wb * (tau_R_pred - tau_R_phys) ** 2)
            else:
                loss_phys = torch.tensor(0.0, device=Xb.device)
        
            loss_sup = torch.mean((tau_R_pred - Tb)**2) if Tb is not None else torch.tensor(0.0, device=Xb.device)
        
            loss = lambda_phys * loss_phys + lambda_sup * loss_sup
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔒 clip here
            optimizer.step()

            train_loss += loss.item()
            phys_loss_total += loss_phys.item()
            sup_loss_total += loss_sup.item()

#        if epoch==1:
#            print("tau_R_pred:", tau_R_pred[:5].flatten())
#            print("D_pred:", D_pred[:5].flatten())
#            print("Gamma:", Gamma[:5])
#            print("tau_R_phys:", tau_R_phys[:5].flatten())
#            print("loss_phys:", loss_phys.item())

        train_losses.append(train_loss)
        physics_losses.append(phys_loss_total)
        supervised_losses.append(sup_loss_total)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, Gb, Wb, Tb in val_loader:
                Gamma = Gb.clamp(min=1e-2)
                tau_R_pred, D_pred = model(Xb)

                if D_pred is not None:
                    D_clamped = D_pred.clamp(min=0.0, max=100.0)
                    tau_R_phys = 2 * D_clamped**2 / Gamma
                    loss_phys = torch.mean(Wb * (tau_R_pred - tau_R_phys) ** 2)
                else:
                    loss_phys = torch.tensor(0.0, device=Xb.device)
                loss_sup = torch.mean((tau_R_pred - Tb)**2) if Tb is not None else torch.tensor(0.0, device=Xb.device)

                loss = lambda_phys * loss_phys + lambda_sup * loss_sup
                val_loss += loss.item()

        val_losses.append(val_loss)

        if (epoch % 5 == 0) or (epoch==1):
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                  f"Phys Loss: {loss_phys:.2e} | Sup Loss: {loss_sup:.2e}")

    return train_losses, val_losses, physics_losses, supervised_losses
