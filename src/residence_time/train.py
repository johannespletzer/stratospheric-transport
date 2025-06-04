from collections import ChainMap
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from residence_time.config import (
    DEFAULT_TIME_ENCODING_CONFIG,
    DEVICE,
    EARLY_STOP_PATIENCE,
    OMEGA,
    PHYSICS_CONSTRAINT,
    USE_TROPOPAUSE_FEATURES,
)
from residence_time.feature_config import FeatureIndex


def make_loader(
    X: np.ndarray,
    Gamma: np.ndarray,
    W: np.ndarray,
    tau: np.ndarray,
    batch_size: int,
    device: str=DEVICE
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


def add_cyclical_time_features(
    X: np.ndarray, time_col: int = FeatureIndex.TIME
) -> np.ndarray:
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
    time_col: int = FeatureIndex.TIME,
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
        Index of time column (default: ``FeatureIndex.TIME``)
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
    t_col = cfg.get("time_col", FeatureIndex.TIME)
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
                  [5] tau_R nan

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
    #time_transf = FunctionTransformer(validate=False) if PHYSICS_CONSTRAINT == 'harmonic' else scaler_cls()
    time_transf = FunctionTransformer(validate=False)

    if USE_TROPOPAUSE_FEATURES:
        scaler_X = ColumnTransformer(
            transformers=[
                ("lat", scaler_cls(), [FeatureIndex.LAT]),
                ("alt", scaler_cls(), [FeatureIndex.ALT]),
                ("time", time_transf, [FeatureIndex.TIME]),
                (
                    "source",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    [FeatureIndex.SOURCE],
                ),
                ("gamma", scaler_cls(), [FeatureIndex.GAMMA]),
                (
                    "nan_bool",
                    FunctionTransformer(validate=False),
                    [FeatureIndex.TAU_FLAG],
                ),
                ("tp_trop", scaler_cls(), [FeatureIndex.TP_TROPO]),
                ("tp_sh", scaler_cls(), [FeatureIndex.TP_SH]),
                ("tp_nh", scaler_cls(), [FeatureIndex.TP_NH]),
                (
                    "tp_bool",
                    FunctionTransformer(validate=False),
                    [FeatureIndex.TP_FLAG],
                ),
            ]
        )
    else:
        scaler_X = ColumnTransformer(
            transformers=[
                ("lat", scaler_cls(), [FeatureIndex.LAT]),
                ("alt", scaler_cls(), [FeatureIndex.ALT]),
                ("time", time_transf, [FeatureIndex.TIME]),
                (
                    "source",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    [FeatureIndex.SOURCE],
                ),
                ("gamma", scaler_cls(), [FeatureIndex.GAMMA]),
                (
                    "nan_bool",
                    FunctionTransformer(validate=False),
                    [FeatureIndex.TAU_FLAG],
                ),
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
    device: str = DEVICE,
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


def _compute_physics_loss(model: Module, Xb: Tensor, D_pred: Tensor, Gamma: Tensor, Wb: Tensor) -> Tensor:
    """Compute physics-based loss from model predictions.

    Returns
    -------
    loss_phys : torch.Tensor
        The physics residual loss term.

    """
    if D_pred is None:
        return torch.tensor(0.0, device=Xb.device)

    if PHYSICS_CONSTRAINT == "diffusivity":
        D_clamped = D_pred.clamp(min=1e-2, max=10.0)
        tau_R_phys = 2 * D_clamped**2 / Gamma
        tau_R_pred, _ = model(Xb)
        valid = ~torch.isnan(tau_R_pred) & ~torch.isnan(tau_R_phys) & ~torch.isnan(Wb)
        if valid.any():
            return torch.mean(Wb[valid] * (tau_R_pred[valid] - tau_R_phys[valid]) ** 2)
        return torch.tensor(0.0, device=Xb.device)

    elif PHYSICS_CONSTRAINT == "harmonic":
        time_col = FeatureIndex.TIME
        t = Xb[:, time_col].unsqueeze(1).clone().requires_grad_(True)
        Xb_phys = torch.cat([Xb[:, :time_col], t, Xb[:, time_col+1:]], dim=1)
        tau_R_pred_phys, _ = model(Xb_phys)
        dtau_dt = torch.autograd.grad(tau_R_pred_phys, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
        d2tau_dt2 = torch.autograd.grad(dtau_dt, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
        residual = d2tau_dt2 + (OMEGA ** 2) * (tau_R_pred_phys - model.tauR_intercept)
        return torch.mean(Wb * residual**2)

    raise ValueError(f"Unknown physics constraint: {PHYSICS_CONSTRAINT}")


def _compute_supervised_loss(tau_R_pred: Tensor, Tb: Tensor, device: str=DEVICE) -> Tensor:
    """Compute supervised MSE loss where target values are available.

    Returns
    -------
    loss_sup : torch.Tensor
        Supervised loss over non-NaN targets.

    """
    mask = ~torch.isnan(Tb)
    if mask.any():
        return torch.mean((tau_R_pred[mask] - Tb[mask])**2)
    return torch.tensor(0.0, device=device)


def _compute_tropopause_constraint_loss(
    model: Module,
    Xb: Tensor,
    tp_cols: Tuple[int, int] = (FeatureIndex.TP_SH, FeatureIndex.TP_NH),
    tp_flag_col: int = FeatureIndex.TP_FLAG
) -> Tensor:
    """Penalize model when τ_R decreases with increasing tropopause pressure.

    Assumes tropopause pressures (hPa) are at columns `tp_cols`.
    We expect d(τ_R)/dP ≥ 0 → penalize if derivative is negative.
    """
    # Clone input and ensure gradient tracking
    Xb_mod = Xb.detach().clone()
    Xb_mod.requires_grad = True

    tau_R_out, _ = model(Xb_mod)
    grad = torch.autograd.grad(
        outputs=tau_R_out.sum(),
        inputs=Xb_mod,
        create_graph=True,
        retain_graph=True
    )[0]

    mask = Xb_mod[:, tp_flag_col] > 0  # only apply penalty where data is valid
    total_penalty = 0.0

    for col in tp_cols:
        if mask.any():
            grad_tp = grad[:, col]
            violation = torch.clamp(-grad_tp[mask], min=0.0)
            total_penalty += torch.mean(violation**2)

    if total_penalty == 0.0:
        return torch.tensor(0.0, device=Xb.device)
    else:
        return total_penalty / len(tp_cols)


def _run_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optional[Optimizer],
    lambda_phys: float,
    lambda_sup: float,
    lambda_tp: float,
    train: bool
) -> Tuple[float, float, float]:
    """Run one training or evaluation epoch.

    Returns
    -------
    Tuple of total_loss, physics_loss, supervised_loss.

    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss, total_phys, total_sup, total_tp = 0.0, 0.0, 0.0, 0.0

    for Xb, Gb, Wb, Tb in dataloader:
        Gamma = Gb.clamp(min=1e-2)
        tau_R_pred, D_pred = model(Xb)

        loss_phys = _compute_physics_loss(model, Xb, D_pred, Gamma, Wb)
        loss_sup = _compute_supervised_loss(tau_R_pred, Tb, device=Xb.device)
        loss = lambda_phys * loss_phys + lambda_sup * loss_sup

        if USE_TROPOPAUSE_FEATURES:
            loss_tp = _compute_tropopause_constraint_loss(model, Xb)
            loss += lambda_tp * loss_tp
        else:
            loss_tp = 0.

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_phys += loss_phys.item()
        total_sup += loss_sup.item()
        total_tp += loss_tp.item() if isinstance(loss_tp, torch.Tensor) else loss_tp

    return total_loss, total_phys, total_sup, total_tp


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    n_epochs: int = 300,
    lambda_phys_start: float = 1.0,
    lambda_sup: float = 1.0,
    lambda_tp: float = 1.0,
    decay_rate: float = 0.95
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a physics-informed model with both supervised and physics losses.

    Returns
    -------
    Tuple of lists: train_losses, val_losses, physics_losses, supervised_losses

    """
    train_losses, val_losses, physics_losses, supervised_losses, tropopause_losses = [], [], [], [], []
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, n_epochs + 1):
        lambda_phys = lambda_phys_start * (decay_rate ** (epoch // 30))
        model.include_D = PHYSICS_CONSTRAINT != "harmonic"

        train_loss, phys_loss, sup_loss, tp_loss = _run_epoch(
            model, train_loader, optimizer, lambda_phys, lambda_sup, lambda_tp, train=True
        )

        val_loss, val_phys, val_sup, tp_loss = _run_epoch(
            model, val_loader, optimizer=None, lambda_phys=lambda_phys, lambda_sup=lambda_sup, lambda_tp=lambda_tp, train=False
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        physics_losses.append(phys_loss)
        supervised_losses.append(sup_loss)
        tropopause_losses.append(tp_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                  f"Phys Loss: {phys_loss:.2e} | Sup Loss: {sup_loss:.2e} | TP Loss: {tp_loss:.2e}")

    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, physics_losses, supervised_losses, tropopause_losses
