from collections import ChainMap
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from residence_time.config import (
    DEFAULT_TIME_ENCODING_CONFIG,
    OMEGA,
    PHYSICS_CONSTRAINT,
)


def make_loader(
    X: np.ndarray,
    Gamma: np.ndarray,
    W: np.ndarray,
    tau: np.ndarray,
    batch_size: int,
    device: str
) -> DataLoader:
    """Convert input arrays into a PyTorch DataLoader for training or validation.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape [N, D].

    Gamma : np.ndarray
        Mean age of air (Γ_EI), shape [N,].

    W : np.ndarray
        Uncertainty weights for physics loss, shape [N,].

    tau : np.ndarray
        Target residence time values (τ_R), shape [N,].

    batch_size : int
        Number of samples per batch in the DataLoader.

    device : str
        Device to move tensors to ('cuda' or 'cpu').

    Returns
    -------
    DataLoader
        PyTorch DataLoader containing (X, Γ, W, τ_R) batches.

    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Gamma_tensor = torch.tensor(Gamma, dtype=torch.float32).unsqueeze(1).to(device)
    W_tensor = torch.tensor(W, dtype=torch.float32).unsqueeze(1).to(device)
    tau_tensor = torch.tensor(tau, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_tensor, Gamma_tensor, W_tensor, tau_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def scale_variables(X_obs: np.ndarray, default: bool = True) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale features for model training."""
    if default:
        scaler_X = StandardScaler()
    else:
        scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_obs)

    return X_scaled, scaler_X


def add_cyclical_time_features(X: np.ndarray, time_col: int = 2) -> np.ndarray:
    """Add sin/cos of fractional year (e.g. 0.25 = spring) to feature array."""
    time_frac = X[:, time_col] % 1  # isolate seasonal phase
    sin_time = np.sin(2 * np.pi * time_frac)
    cos_time = np.cos(2 * np.pi * time_frac)
    return np.concatenate([X, sin_time[:, None], cos_time[:, None]], axis=1)


def rbf_transformer(time: np.ndarray, centers: np.ndarray, gamma: float=100.0) -> np.ndarray:
    """Return radial basis functions for input values.
    
    time: (N,) input values
    centers: (M,) RBF centers
    gamma: float, controls width of each basis function
    
    Returns: (N, M) RBF matrix
    """
    time = time[:, None]  # (N, 1)
    centers = centers[None, :]  # (1, M)
    return np.exp(-gamma * (time - centers) ** 2)


def add_rbf_time_features(
    X: np.ndarray,
    time_col: int = 2,
    kind: str = "seasonal",  # "seasonal" or "absolute"
    n_rbf: int = 12,
    gamma: float = 100.0,
    t_min: float = 1980.0,
    t_max: float = 2025.0
) -> np.ndarray:
    """Add RBF time features to input array.

    Parameters
    ----------
    X : np.ndarray
        Input features [N, D]
    time_col : int
        Index of time column (default: 2)
    kind : str
        "seasonal" → use time % 1 (cyclical)
        "absolute" → use full time range
    n_rbf : int
        Number of basis functions
    gamma : float
        Width control for RBFs
    t_min : float
        Minimum time value for absolute encoding
    t_max : float
        Maximum time value for absolute encoding

    Returns
    -------
    X_ext : np.ndarray
        Extended input with RBF time features

    """
    time = X[:, time_col]

    if kind == "seasonal":
        time_transformed = time % 1
        centers = np.linspace(0, 1, n_rbf, endpoint=False)
    elif kind == "absolute":
        # Normalize time to [0, 1]
        time_transformed = (time - t_min) / (t_max - t_min)
        centers = np.linspace(0, 1, n_rbf)
    else:
        raise ValueError("kind must be 'seasonal' or 'absolute'")

    rbf_feats = rbf_transformer(time_transformed, centers, gamma)
    return np.concatenate([X, rbf_feats], axis=1)


def apply_time_encoding(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Use config to apply time encoding options."""
    cfg = dict(ChainMap(cfg, DEFAULT_TIME_ENCODING_CONFIG))

    if not cfg.get("enabled", False):
        return X

    X_out = X.copy()
    t_col = cfg["time_col"]
    time = X[:, t_col]

    features_to_add = []

    if cfg["use_cyclical"]:
        X_out = add_cyclical_time_features(X_out, time_col=t_col)

    if cfg["use_rbf_seasonal"]:
        time_frac = time % 1
        centers = np.linspace(0, 1, cfg["n_rbf_seasonal"], endpoint=False)
        rbf_seasonal = rbf_transformer(time_frac, centers, cfg["gamma_seasonal"])
        features_to_add.append(rbf_seasonal)

    if cfg["use_rbf_absolute"]:
        time_norm = (time - cfg["t_min"]) / (cfg["t_max"] - cfg["t_min"])
        centers = np.linspace(0, 1, cfg["n_rbf_absolute"])
        rbf_abs = rbf_transformer(time_norm, centers, cfg["gamma_absolute"])
        features_to_add.append(rbf_abs)

    if features_to_add:
        X_out = np.concatenate([X_out] + features_to_add, axis=1)

    return X_out


def scale_variables_columnwise(
    X_obs: np.ndarray,
    scaler: str = 'StandardScaler'
) -> Tuple[np.ndarray, ColumnTransformer]:
    """Scale selected columns of X_obs using StandardScaler or MinMaxScaler.

    Column order: [0] Latitude, [1] Altitude, [2] Time, [3] Source ID, [4] Gamma_EI

    Returns
    -------
    X_scaled : np.ndarray
        Scaled input array.
    scaler_X : sklearn.compose.ColumnTransformer
        Fitted transformer with column order preserved.

    """
    scaler_cls = StandardScaler if scaler == 'StandardScaler' else MinMaxScaler

    # Identity transform applied to column 2 (time)
    # Other columns scaled
    time_transf = FunctionTransformer(validate=False) if PHYSICS_CONSTRAINT == 'harmonic' else scaler_cls()

    scaler_X = ColumnTransformer(
        transformers=[
            ("lat",     scaler_cls(), [0]),
            ("alt",     scaler_cls(), [1]),
            ("time",    time_transf, [2]),
            ("source",  FunctionTransformer(validate=False), [3]),  # unscaled
            ("gamma",   scaler_cls(), [4]),
        ]
    )

    X_scaled = scaler_X.fit_transform(X_obs)
    scaler_X.scaler_name = scaler

    return X_scaled, scaler_X


def create_train_val_loaders(
    X: np.ndarray,
    Gamma: np.ndarray,
    W: np.ndarray,
    tau_R: Optional[np.ndarray] = None,
    batch_size: int = 256,
    val_split: float = 0.2,
    device: str = 'cuda',
    time_encoding_config: dict = DEFAULT_TIME_ENCODING_CONFIG
) -> Tuple[DataLoader, DataLoader]:
    """Split data into training and validation sets and return DataLoaders.

    This function constructs PyTorch DataLoaders with (X, Gamma, W, tau_R) tuples 
    for use in training a PINN model. If no tau_R targets are provided, the 
    function fills them with NaNs for compatibility.

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

    time_encoding_config : bool, default=DEFAULT_TIME_ENCODING_CONFIG
        Option to add features for seasonal and long-term trends

    Returns
    -------
    train_loader : DataLoader
        PyTorch DataLoader for training data.

    val_loader : DataLoader
        PyTorch DataLoader for validation data.

    """
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

    time_encoding_config = dict(ChainMap(time_encoding_config, DEFAULT_TIME_ENCODING_CONFIG))

    if time_encoding_config.get("enabled", False):
        X_train = apply_time_encoding(X_train, time_encoding_config)
        X_val = apply_time_encoding(X_val, time_encoding_config)

    return make_loader(X_train, Gamma_train, W_train, tau_train, batch_size, device), \
           make_loader(X_val, Gamma_val, W_val, tau_val, batch_size, device)


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    n_epochs: int = 300,
    lambda_phys_start: float = 1.0,
    lambda_sup_start: float = 1.0,
    decay_rate: float = 0.95
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a PINN model using both physics-based and supervised losses.

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
        lambda_phys = lambda_phys_start * (decay_rate ** (epoch // 30))
        lambda_sup = lambda_sup_start

        # Deactivate D_pred if harmonic oscillator constrains neural network
        model.include_D = False if PHYSICS_CONSTRAINT == "harmonic" else True

        for Xb, Gb, Wb, Tb in train_loader:
            optimizer.zero_grad()  # always do this before backward()
        
            Gamma = Gb.clamp(min=1e-2)
            tau_R_pred, D_pred = model(Xb)
        
            if PHYSICS_CONSTRAINT == "diffusivity":
                assert D_pred is not None, "D prediction required for diffusivity-based constraint."

                D_clamped = D_pred.clamp(min=1e-2, max=10.0)
                tau_R_phys = 2 * D_clamped**2 / Gamma
                loss_phys = torch.mean(Wb * (tau_R_pred - tau_R_phys) ** 2)
            
            elif PHYSICS_CONSTRAINT == "harmonic":
                time_col = 2  # index of time in input
                t = Xb[:, time_col].unsqueeze(1).clone()
                t.requires_grad_(True)
            
                # Rebuild Xb_phys with t linked to autograd
                Xb_phys = torch.cat([Xb[:, :time_col], t, Xb[:, time_col+1:]], dim=1)
            
                tau_R_pred_phys, _ = model(Xb_phys)
            
                dtau_dt = torch.autograd.grad(
                    tau_R_pred_phys, t,
                    grad_outputs=torch.ones_like(tau_R_pred_phys),
                    create_graph=True
                )[0]
            
                d2tau_dt2 = torch.autograd.grad(
                    dtau_dt, t,
                    grad_outputs=torch.ones_like(dtau_dt),
                    create_graph=True
                )[0]
            
                #residual = d2tau_dt2 + (OMEGA ** 2) * tau_R_pred_phys
                residual = d2tau_dt2 + (OMEGA ** 2) * (tau_R_pred_phys - model.tauR_intercept)
                loss_phys = torch.mean(Wb * residual**2)
            
            else:
                raise ValueError(f"Unknown physics constraint: {PHYSICS_CONSTRAINT}")
                loss_phys = torch.tensor(0.0, device=Xb.device)
        
            loss_sup = torch.mean((tau_R_pred - Tb)**2) if Tb is not None else torch.tensor(0.0, device=Xb.device)
        
            loss = lambda_phys * loss_phys + lambda_sup * loss_sup
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔒 clip here
            optimizer.step()

            train_loss += loss.item()
            phys_loss_total += loss_phys.item()
            sup_loss_total += loss_sup.item()

        train_losses.append(train_loss)
        physics_losses.append(phys_loss_total)
        supervised_losses.append(sup_loss_total)

        # Validation
        model.eval()
        val_loss = 0
        for Xb, Gb, Wb, Tb in val_loader:
            Gamma = Gb.clamp(min=1e-2)
            tau_R_pred, D_pred = model(Xb)

            if PHYSICS_CONSTRAINT == 'diffusivity': 
                with torch.no_grad():
                    assert D_pred is not None, "D prediction required for diffusivity-based constraint."

                    D_clamped = D_pred.clamp(min=1e-2, max=10.0)
                    tau_R_phys = 2 * D_clamped**2 / Gamma
                    loss_phys = torch.mean(Wb * (tau_R_pred - tau_R_phys) ** 2)

            elif PHYSICS_CONSTRAINT == "harmonic":
                time_col = 2  # index of time in input
                t = Xb[:, time_col].unsqueeze(1).clone()
                t.requires_grad_(True)
            
                # Rebuild Xb_phys with t linked to autograd
                Xb_phys = torch.cat([Xb[:, :time_col], t, Xb[:, time_col+1:]], dim=1)
            
                tau_R_pred_phys, _ = model(Xb_phys)
            
                dtau_dt = torch.autograd.grad(
                    tau_R_pred_phys, t,
                    grad_outputs=torch.ones_like(tau_R_pred_phys),
                    create_graph=True
                )[0]
            
                d2tau_dt2 = torch.autograd.grad(
                    dtau_dt, t,
                    grad_outputs=torch.ones_like(dtau_dt),
                    create_graph=True
                )[0]
            
                #residual = d2tau_dt2 + (OMEGA ** 2) * tau_R_pred_phys
                residual = d2tau_dt2 + (OMEGA ** 2) * (tau_R_pred_phys - model.tauR_intercept)
                loss_phys = torch.mean(Wb * residual**2)
                    
            else:
                raise ValueError(f"Unknown physics constraint: {PHYSICS_CONSTRAINT}")
                loss_phys = torch.tensor(0.0, device=Xb.device)

            loss_sup = torch.mean((tau_R_pred - Tb)**2) if Tb is not None else torch.tensor(0.0, device=Xb.device)

            loss = lambda_phys * loss_phys + lambda_sup * loss_sup
            val_loss += loss.item()

        val_losses.append(val_loss)

        if (epoch % 10 == 0) or (epoch==1):
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                  f"Phys Loss: {loss_phys:.2e} | Sup Loss: {loss_sup:.2e}")

    return train_losses, val_losses, physics_losses, supervised_losses
