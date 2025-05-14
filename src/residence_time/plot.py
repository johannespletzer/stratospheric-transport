from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader

from residence_time.data import extend_with_tropopause_features
from residence_time.train import add_cyclical_time_features


def plot_physics_residual(
    model: nn.Module,
    dataloader: DataLoader,
    xlim: Optional[Tuple[float, float]] = None,
    device: str = 'cuda',
    return_residuals: bool = False
) -> Optional[np.ndarray]:
    """Plot a histogram of physics residuals: t_R^pred - (2·D² / G_EI).

    Parameters
    ----------
    model : nn.Module
        Trained PINN model returning (t_R, D).

    dataloader : DataLoader
        DataLoader yielding batches of (X, G_EI, W, t_R).
        Only X and G_EI are used here.

    xlim : tuple, optional
        x-axis limits for the histogram.

    device : str, default='cuda'
        Device to evaluate the model on.

    return_residuals : bool, default=False
        If True, return the residual values.

    Returns
    -------
    np.ndarray or None
        Physics residuals if `return_residuals` is True, otherwise None.

    """
    model.eval()
    residuals = []

    with torch.no_grad():
        for Xb, Gb, _, _ in dataloader:
            Xb = Xb.to(device)
            Gb = Gb.clamp(min=1e-4).to(device)

            tau_R_pred, D_pred = model(Xb)
            tau_R_phys = 2 * D_pred**2 / Gb
            residuals.append((tau_R_pred - tau_R_phys).cpu().numpy())

    residuals = np.concatenate(residuals)

    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=50, color='orange', edgecolor='black', alpha=0.9)
    plt.xlabel(r"Physics Residual: $\tau_R^{pred} - \frac{2D^2}{\Gamma_{EI}}$")
    plt.ylabel("Count")
    plt.title("Histogram of Physics Residuals")
    plt.grid(True)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()

    return residuals if return_residuals else None


def float_to_year_month(float_time_ns: float) -> str:
    """Convert a float64 datetime64 in nanoseconds to a 'YYYY-MM' string.

    Parameters
    ----------
    float_time_ns : float
        Datetime64 in nanoseconds as float64.

    Returns
    -------
    str
        Date string formatted as 'YYYY-MM'.

    """
    dt = np.datetime64(int(float_time_ns), 'ns').astype('datetime64[M]')
    return str(dt)


def plot_field_from_data(
    model: nn.Module,
    X: np.ndarray,
    scaler_X: TransformerMixin,
    field: str = 'tau_R',
    grid_res: Tuple[int, int] = (64, 40),
    time_value: Optional[float] = None,
    source_value: Optional[float] = None,
    gamma_value: Optional[float] = None,
    device: str = 'cuda',
    return_data: bool = False,
    use_tropopause_features: bool = False,
    use_seasonal_features: bool = False,
    tp_csv_path: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Plot a spatial field (t_R or D) from a PINN model, optionally using tropopause features.

    Parameters
    ----------
    model : nn.Module
        Trained PINN model returning (t_R, D).

    X : np.ndarray
        Input dataset used to determine bounds for lat/alt.

    scaler_X : sklearn transformer
        Scaler used to normalize input features.

    field : str, default='tau_R'
        Field to visualize ('tau_R' or 'D').

    grid_res : tuple, default=(64, 40)
        Grid resolution in (lat, alt).

    time_value : float, optional
        Fixed time value to use across the grid.

    source_value : float, optional
        Fixed source ID.

    gamma_value : float, optional
        Fixed G_EI value.

    device : str, default='cuda'
        Device to run inference on.

    return_data : bool, default=False
        If True, return the field grid and axis values.

    use_tropopause_features : bool, default=False
        Whether to include tropopause-based model inputs.

    use_seasonal_features : bool, default=False
        Whether to include cyclical features to address seasonal patterns 

    tp_csv_path : str, optional
        Path to the CSV file with tropopause features.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        (lat_vals, alt_vals, field_grid) if return_data is True, else None.

    """
    model.eval()

    lat_min, lat_max = np.min(X[:, 0]), np.max(X[:, 0])
    alt_min, alt_max = np.min(X[:, 1]), np.max(X[:, 1])

    lat_vals = np.linspace(lat_min, lat_max, grid_res[0])
    alt_vals = np.linspace(alt_min, alt_max, grid_res[1])
    lat_grid, alt_grid = np.meshgrid(lat_vals, alt_vals, indexing='ij')

    time_fixed = time_value if time_value is not None else np.median(X[:, 2])
    source_fixed = source_value if source_value is not None else np.median(X[:, 3])
    gamma_fixed = gamma_value if gamma_value is not None else np.median(X[:, 4])

    time_array = np.full_like(lat_grid.flatten(), time_fixed)
    source_array = np.full_like(lat_grid.flatten(), source_fixed)
    gamma_array = np.full_like(lat_grid.flatten(), gamma_fixed)

    X_base = np.stack([lat_grid.flatten(), alt_grid.flatten(), time_array, source_array, gamma_array], axis=1)

    if use_tropopause_features:
        X_full, _ = extend_with_tropopause_features(X_base, csv_path=tp_csv_path)
    else:
        X_full = X_base

    X_scaled = scaler_X.transform(X_full)

    if use_seasonal_features:
        X_scaled = add_cyclical_time_features(X_scaled)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        tau_R_pred, D_pred = model(X_tensor)

    field_pred = tau_R_pred if field == 'tau_R' else D_pred
    field_grid = field_pred.cpu().numpy().reshape(grid_res)

    plt.figure(figsize=(7, 5))
    contour = plt.contourf(lat_vals, alt_vals, field_grid.T, levels=20, cmap='viridis')
    plt.xlabel("Latitude [°]")
    plt.ylabel("Altitude [km]")

    time_label = np.round(float_to_year_month(time_fixed) if time_fixed > 1e4 else time_fixed,2)
    plt.title(
        rf"Predicted {field} | time={time_label}, source={int(source_fixed)}, $\Gamma_{{EI}}$={gamma_fixed:.2f}"
    )
    plt.colorbar(contour, label=field)
    plt.tight_layout()
    plt.show()

    if return_data:
        return lat_vals, alt_vals, field_grid
    return None


def plot_training_progress(
    train_losses: List[float],
    val_losses: List[float],
    physics_losses: List[float],
    supervised_losses: List[float],
    log_scale: bool = True,
    save_path: Optional[str] = None
) -> None:
    """Plot the evolution of loss values during training.

    Parameters
    ----------
    train_losses : list of float
        Total training loss per epoch.

    val_losses : list of float
        Total validation loss per epoch.

    physics_losses : list of float
        Physics-based loss values (training).

    supervised_losses : list of float
        Supervised t_R loss values (training).

    log_scale : bool, default=True
        Use logarithmic scale on the y-axis.

    save_path : str, optional
        If set, save the figure to this path.

    Returns
    -------
    None

    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Total", lw=2)
    plt.plot(epochs, val_losses, label="Val Total", lw=2, linestyle='--')
    plt.plot(epochs, physics_losses, label="Physics Loss (Train)", lw=1.5, linestyle='-.')
    plt.plot(epochs, supervised_losses, label="Supervised t_R Loss (Train)", lw=1.5, linestyle=':')

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Progress")
    plt.legend(loc="best")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"?? Saved training progress plot to: {save_path}")

    plt.show()


def plot_tau_R_prediction_vs_target(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> None:
    """Plot predicted t_R vs. target t_R on a validation batch.

    Parameters
    ----------
    model : nn.Module
        Trained PINN model returning (t_R, D).

    dataloader : DataLoader
        Validation data loader (X, G, W, t_R).

    device : str, default='cuda'
        Device to run inference on.

    Returns
    -------
    None

    """
    model.eval()
    Xb, _, _, Tb = next(iter(dataloader))
    Xb, Tb = Xb.to(device), Tb.to(device)

    with torch.no_grad():
        tau_R_pred, _ = model(Xb)

    true = Tb.cpu().numpy()
    pred = tau_R_pred.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.3, edgecolor='k', linewidth=0.1)
    plt.plot([0, 7], [0, 7], 'r--', label="Ideal (y = x)")
    plt.xlabel("True t_R")
    plt.ylabel("Predicted t_R")
    plt.title("t_R Prediction vs. Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
