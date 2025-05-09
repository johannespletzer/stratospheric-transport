import os
from datetime import datetime
from glob import glob
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from sklearn.base import BaseEstimator


def datetime64_to_year_fraction(
    t: Union[np.ndarray, xr.DataArray]
) -> np.ndarray:
    """Convert datetime64[ns] array to fractional years.

    Parameters
    ----------
    t : np.ndarray or xarray.DataArray
        Datetime64 array.

    Returns
    -------
    np.ndarray
        Array of fractional years (e.g. 2004.04).

    """
    t = pd.to_datetime(t)
    year = t.year
    start_of_year = pd.to_datetime(year.astype(str))
    start_of_next_year = pd.to_datetime((year + 1).astype(str))
    fraction = (t - start_of_year) / (start_of_next_year - start_of_year)
    return year + fraction.values


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    optimizer: Optional[optim.Optimizer] = None,
    return_scaler: bool = False
) -> Optional[object]:
    """Load model and optimizer state from checkpoint. Optionally return stored scaler.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to load state into.
    checkpoint_path : str
        Path to the .pth checkpoint file.
    device : str, default='cuda'
        Device to map the loaded tensors onto.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into (if present).
    return_scaler : bool, default=False
        If True, returns the stored scaler object.

    Returns
    -------
    Optional[object]
        Scaler object if return_scaler=True and scaler is in checkpoint; otherwise None.

    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")

    if return_scaler:
        return checkpoint.get("scaler", None)

    return None

def load_latest_checkpoint(
    model: nn.Module,
    device: str = 'cuda',
    optimizer: Optional[optim.Optimizer] = None,
    checkpoint_dir: Optional[str] = None,
    return_scaler: bool = False
) -> Optional[object]:
    """Load the latest checkpoint from the specified directory into the model and optionally the optimizer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to load state into.

    device : str, default='cuda'
        Device to load model onto ('cuda' or 'cpu').

    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into if checkpoint contains optimizer state.

    checkpoint_dir : str, optional
        Path to checkpoint directory. If None, uses 'models/checkpoints' in project root.

    return_scaler : bool, default=False
        If True, returns the scaler saved in the checkpoint (if present).

    Returns
    -------
    Optional[object]
        The scaler object if `return_scaler` is True and scaler is present in the checkpoint,
        otherwise None.

    """
    if checkpoint_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "../../"))
        checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "pinn_checkpoint_*.pth")))
    if not checkpoint_files:
        print("No checkpoint found in", checkpoint_dir)
        return None

    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {latest_checkpoint}")

    if return_scaler:
        return checkpoint.get("scaler", None)

    return None


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    name: Optional[str] = None,
    scaler: Optional[object] = None,
    checkpoint_dir: Optional[str] = None
) -> None:
    """Save model, optimizer, and optionally a scaler to a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to save.

    optimizer : torch.optim.Optimizer
        Optimizer to save.

    name : str, optional
        Filename to use for the checkpoint (e.g., 'pinn_best.pth').
        If not provided, a timestamped filename will be generated.

    scaler : object, optional
        A scikit-learn scaler or other serializable object to include.

    checkpoint_dir : str, optional
        Directory to save to. Defaults to 'models/checkpoints' in the project root.

    Returns
    -------
    None

    """
    if checkpoint_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "../../"))
        checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"pinn_checkpoint_{timestamp}.pth"

    checkpoint_path = os.path.join(checkpoint_dir, name)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def save_scaler(
    scaler: BaseEstimator,
    scaler_dir: str = "checkpoints",
    prefix: str = "scaler"
) -> str:
    """Save a fitted scikit-learn scaler to disk with a timestamped filename.

    Parameters
    ----------
    scaler : BaseEstimator
        The fitted scaler (e.g., MinMaxScaler or StandardScaler).

    scaler_dir : str, default="checkpoints"
        Directory to save the scaler file.

    prefix : str, default="scaler"
        Prefix for the saved file name.

    Returns
    -------
    str
        Full path to the saved scaler file.

    """
    os.makedirs(scaler_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scaler_path = os.path.join(scaler_dir, f"{prefix}_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)

    return scaler_path
