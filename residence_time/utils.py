import os
import torch
import joblib
from datetime import datetime

def datetime64_to_year_fraction(t):
    """
    Convert datetime64[ns] array to fractional years.

    Parameters
    ----------
    t : np.ndarray or xarray.DataArray
        Datetime64 array.

    Returns
    -------
    np.ndarray
        Array of fractional years (e.g. 2004.04).
    """
    import pandas as pd

    t = pd.to_datetime(t)
    year = t.year
    start_of_year = pd.to_datetime(year.astype(str))
    start_of_next_year = pd.to_datetime((year + 1).astype(str))
    fraction = (t - start_of_year) / (start_of_next_year - start_of_year)

    return year + fraction.values

def load_latest_checkpoint(model, optimizer=None, checkpoint_dir="checkpoints"):

    import glob

    checkpoint_files = sorted(glob.glob(f"{checkpoint_dir}/pinn_checkpoint_*.pth"))
    if not checkpoint_files:
        print("No checkpoint found.")
        return

    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"Loaded checkpoint from {latest_checkpoint}")

def save_checkpoint(model, optimizer, checkpoint_dir="checkpoints"):

    from datetime import datetime

    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f"{checkpoint_dir}/pinn_checkpoint_{timestamp}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": timestamp,
    }, checkpoint_path)
        
    print(f"Saved latest checkpoint in {checkpoint_dir}")

def save_scaler(scaler, scaler_dir="checkpoints", prefix="scaler"):
    """
    Saves a fitted sklearn scaler (e.g., MinMaxScaler) to disk with a timestamped filename.

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
