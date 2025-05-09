import matplotlib.pyplot as plt
import numpy as np
import torch

from residence_time.data import extend_with_tropopause_features

def plot_physics_residual(
    model,
    dataloader,
    xlim: tuple = None,
    device: str = 'cuda',
    return_residuals: bool = False
):
    """Plots a histogram of physics residuals: τ_R predicted - (2·D² / Γ),
    showing how well the model satisfies the physical relationship.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model that returns (τ_R, D) predictions.

    dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of (X, Gamma_EI, W, tau_I_to_X).
        Only X and Gamma_EI are used here.

    xlim : tuple, optional
        Tuple (xmin, xmax) to set x-axis limits of the histogram.

    device : str, default='cuda'
        Device on which the model runs.

    return_residuals : bool, default=False
        If True, return the raw residual array instead of just plotting.

    Returns
    -------
    residuals : np.ndarray, optional
        Array of physics residuals if `return_residuals=True`.

    """
    model.eval()
    residuals = []

    with torch.no_grad():
        for Xb, Gb, _, _ in dataloader:
            Xb = Xb.to(device)
            Gb = Gb.clamp(min=1e-4).to(device)

            tau_R_pred, D_pred = model(Xb)
            tau_R_phys = 2 * D_pred**2 / Gb

            residual = (tau_R_pred - tau_R_phys).cpu().numpy()
            residuals.append(residual)

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

    if return_residuals:
        return residuals


def float_to_year_month(float_time_ns: float) -> str:
    """Converts a float64 datetime64 in nanoseconds to a YYYY-MM string.
    
    Assumes the float was created via: np.datetime64(..., 'ns').astype('float64')
    """
    dt = np.datetime64(int(float_time_ns), 'ns').astype('datetime64[M]')
    return str(dt)


def plot_field_from_data(
    model,
    X: np.ndarray,
    scaler_X,
    field: str = 'tau_R',
    grid_res: tuple = (64, 40),
    time_value: float = None,
    source_value: float = None,
    gamma_value: float = None,
    device: str = 'cuda',
    return_data: bool = False,
    use_tropopause_features: bool = False,
    tp_csv_path: str = None,
):
    """
    Plot a spatial field (τ_R or D) from a PINN model, optionally using tropopause features.
    """

    model.eval()

    # Extract lat/alt bounds
    lat_min, lat_max = np.min(X[:, 0]), np.max(X[:, 0])
    alt_min, alt_max = np.min(X[:, 1]), np.max(X[:, 1])

    lat_vals = np.linspace(lat_min, lat_max, grid_res[0])
    alt_vals = np.linspace(alt_min, alt_max, grid_res[1])
    lat_grid, alt_grid = np.meshgrid(lat_vals, alt_vals, indexing='ij')

    lat_flat = lat_grid.flatten()
    alt_flat = alt_grid.flatten()

    # Default values if not provided
    time_fixed = time_value if time_value is not None else np.median(X[:, 2])
    source_fixed = source_value if source_value is not None else np.median(X[:, 3])
    gamma_fixed = gamma_value if gamma_value is not None else np.median(X[:, 4])

    # Construct base input
    time_array = np.full_like(lat_flat, time_fixed)
    source_array = np.full_like(lat_flat, source_fixed)
    gamma_array = np.full_like(lat_flat, gamma_fixed)

    X_base = np.stack([lat_flat, alt_flat, time_array, source_array, gamma_array], axis=1)

    # Optionally extend with tropopause features
    if use_tropopause_features:
        X_full, _ = extend_with_tropopause_features(X_base, csv_path=tp_csv_path)
    else:
        X_full = X_base

    # Scale and run model
    X_scaled = scaler_X.transform(X_full)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        tau_R_pred, D_pred = model(X_tensor)

    field_pred = tau_R_pred if field == 'tau_R' else D_pred
    field_grid = field_pred.cpu().numpy().reshape(grid_res)

    # Plotting
    plt.figure(figsize=(7, 5))
    contour = plt.contourf(lat_vals, alt_vals, field_grid.T, levels=20, cmap='viridis')
    plt.xlabel("Latitude [°]")
    plt.ylabel("Altitude [km]")

    time_label = float_to_year_month(time_fixed) if time_fixed > 1e4 else time_fixed
    plt.title(
        rf"Predicted {field} | time={time_label}, source={int(source_fixed)}, $\Gamma_{{EI}}$={gamma_fixed:.2f}"
    )
    plt.colorbar(contour, label=field)
    plt.tight_layout()
    plt.show()

    if return_data:
        return lat_vals, alt_vals, field_grid

def plot_training_progress(
    train_losses: list[float],
    val_losses: list[float],
    physics_losses: list[float],
    supervised_losses: list[float],
    log_scale: bool = True,
    save_path: str = None
):
    """Plots the evolution of training and validation losses over epochs,
    including physics-based and supervised loss components.

    Parameters
    ----------
    train_losses : list of float
        Total training loss values per epoch.

    val_losses : list of float
        Total validation loss values per epoch.

    physics_losses : list of float
        Physics-informed loss values (train) per epoch.

    supervised_losses : list of float
        Supervised τ_R loss values (train) per epoch.

    log_scale : bool, default=True
        Whether to use a logarithmic y-axis scale.

    save_path : str, optional
        If provided, saves the plot to the specified path.

    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))

    # Plot loss curves
    plt.plot(epochs, train_losses, label="Train Total", lw=2)
    plt.plot(epochs, val_losses, label="Val Total", lw=2, linestyle='--')
    plt.plot(epochs, physics_losses, label="Physics Loss (Train)", lw=1.5, linestyle='-.')
    plt.plot(epochs, supervised_losses, label="Supervised τ_R Loss (Train)", lw=1.5, linestyle=':')

    # Log scale (optional)
    if log_scale:
        plt.yscale("log")

    # Labels and layout
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Progress")
    plt.legend(loc="best")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 Saved training progress plot to: {save_path}")

    plt.show()


def plot_tau_R_prediction_vs_target(model, dataloader, device: str = 'cuda'):
    """Plots a scatter plot comparing predicted τ_R vs. target τ_R from a single validation batch.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model returning (tau_R_pred, D_pred).

    dataloader : torch.utils.data.DataLoader
        Dataloader containing (X, Γ_EI, W, τ_R_target).

    device : str, default='cuda'
        Device to run the model on.

    Returns
    -------
    None
        Displays a matplotlib scatter plot.

    """
    model.eval()

    # Get a single batch
    Xb, _, _, Tb = next(iter(dataloader))
    Xb = Xb.to(device)
    Tb = Tb.to(device)

    with torch.no_grad():
        tau_R_pred, _ = model(Xb)

    # Detach for plotting
    true = Tb.cpu().numpy()
    pred = tau_R_pred.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.3, edgecolor='k', linewidth=0.1)
    plt.plot([0, 7], [0, 7], 'r--', label="Ideal (y = x)")
    plt.xlabel("True τ_R")
    plt.ylabel("Predicted τ_R")
    plt.title("τ_R Prediction vs. Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
