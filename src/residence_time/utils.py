import os
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr


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
    return (year + fraction.values).to_numpy()

def year_fraction_to_datetime64(
    year_frac: Union[np.ndarray, list, float]
) -> np.ndarray:
    """Convert fractional years to datetime64[ns].

    Parameters
    ----------
    year_frac : float, list, or np.ndarray
        Fractional years (e.g. 2004.04).

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] corresponding to the input fractional years.

    """
    year_frac = np.atleast_1d(year_frac).astype(float)
    years = np.floor(year_frac).astype(int)
    frac = year_frac - years

    start_of_year = pd.to_datetime(years.astype(str))
    start_of_next_year = pd.to_datetime((years + 1).astype(str))
    delta = (start_of_next_year - start_of_year) * frac

    result = start_of_year + delta
    return result.to_numpy()

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

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")

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


