import glob
import os
from datetime import datetime

import joblib
import pandas as pd
import torch


def datetime64_to_year_fraction(t):
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

def load_checkpoint(model, checkpoint_path, device='cuda', optimizer=None, return_scaler=False):
    """Load model and optimizer state from checkpoint. Optionally return stored scaler.

    Args:
        model: PyTorch model to load.
        checkpoint_path: Path to .pth file.
        device: 'cuda' or 'cpu'.
        optimizer: (Optional) optimizer to load state into.
        return_scaler: If True, return the stored scaler (if present).

    Returns:
        scaler if return_scaler=True and scaler is in checkpoint; otherwise None.

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

def load_latest_checkpoint(model, device='cuda', optimizer=None, checkpoint_dir=None, return_scaler=False):
    """Load the latest checkpoint from the specified directory.

    Args:
        model: PyTorch model to load state into.
        device: Device to load model onto ('cuda' or 'cpu').
        optimizer: (Optional) Optimizer to load state into.
        checkpoint_dir: Path to checkpoint directory. Defaults to project-root/models/checkpoints.
        return_scaler: If True, return the scaler saved with the checkpoint (if available).

    Returns:
        scaler if return_scaler=True and present in checkpoint, otherwise None.

    """
    if checkpoint_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "../../"))
        checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "pinn_checkpoint_*.pth")))
    if not checkpoint_files:
        print("No checkpoint found in", checkpoint_dir)
        return None if return_scaler else None

    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {latest_checkpoint}")

    if return_scaler:
        return checkpoint.get("scaler", None)

def save_checkpoint(model, optimizer, name=None, scaler=None, checkpoint_dir=None):
    """Save model, optimizer, and optionally scaler to a checkpoint file.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer to save.
        name: (Optional) Filename to use for the checkpoint (e.g., 'pinn_best.pth').
              If not provided, a timestamped filename will be used.
        scaler: (Optional) Scikit-learn scaler or other serializable object to include.
        checkpoint_dir: (Optional) Directory to save to. Defaults to project-root/models/checkpoints.

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

def save_scaler(scaler, scaler_dir="checkpoints", prefix="scaler"):
    """Saves a fitted sklearn scaler (e.g., MinMaxScaler) to disk with a timestamped filename.

    Parameters
    ----------
    scaler : sklearn scaler
        The fitted scaler to save (e.g., from MinMaxScaler or StandardScaler).

    scaler_dir : str
        Directory to save the scaler file (default: "checkpoints").

    prefix : str
        Filename prefix (default: "scaler").

    Returns
    -------
    scaler_path : str
        Path to the saved scaler file.

    """
    os.makedirs(scaler_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scaler_path = os.path.join(scaler_dir, f"{prefix}_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)

    return scaler_path
