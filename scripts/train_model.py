"""Train the residence-time PINN with a reproducible command-line workflow."""

import argparse
from typing import Dict, Optional, Tuple

import torch

from residence_time.config import (
    DEFAULT_TIME_ENCODING_CONFIG,
    LEARNING_RATE,
    PHYSICS_CONSTRAINT,
    USE_TROPOPAUSE_FEATURES,
)
from residence_time.data import load_all_data_combined, load_tau_R
from residence_time.model import PINNModel
from residence_time.train import (
    create_train_val_loaders,
    scale_variables_columnwise,
    train_model,
)
from residence_time.utils import save_checkpoint, save_config


def parse_time_range(time_start: Optional[str], time_end: Optional[str]) -> Optional[Tuple[str, str]]:
    """Build optional time-range tuple from CLI args."""
    if time_start is None and time_end is None:
        return None
    if time_start is None or time_end is None:
        raise ValueError("Both --time-start and --time-end must be provided together.")
    return (time_start, time_end)


def build_time_encoding_config(args: argparse.Namespace) -> Dict[str, object]:
    """Create a validated time-encoding config dictionary from CLI args."""
    config = dict(DEFAULT_TIME_ENCODING_CONFIG)
    config["enabled"] = bool(
        args.enable_time_encoding
        or args.use_cyclical
        or args.use_rbf_seasonal
        or args.use_rbf_absolute
    )
    config["use_cyclical"] = args.use_cyclical
    config["use_rbf_seasonal"] = args.use_rbf_seasonal
    config["use_rbf_absolute"] = args.use_rbf_absolute
    config["n_rbf_seasonal"] = args.n_rbf_seasonal
    config["n_rbf_absolute"] = args.n_rbf_absolute
    config["gamma_seasonal"] = args.gamma_seasonal
    config["gamma_absolute"] = args.gamma_absolute
    config["t_min"] = args.t_min
    config["t_max"] = args.t_max
    config["time_col"] = args.time_col
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train the stratospheric transport PINN.")

    parser.add_argument("--sat-paths", nargs="*", default=None, help="Satellite AoA NetCDF paths.")
    parser.add_argument("--insitu-paths", nargs="*", default=None, help="In-situ AoA NetCDF paths.")
    parser.add_argument("--model-paths", nargs="*", default=None, help="Model AoA NetCDF paths.")
    parser.add_argument("--tau-path", type=str, default=None, help="Optional tau_R NetCDF file path.")

    parser.add_argument("--time-start", type=str, default=None, help="Optional data filter start (e.g., 2005).")
    parser.add_argument("--time-end", type=str, default=None, help="Optional data filter end (e.g., 2020).")

    parser.add_argument(
        "--use-tropopause-features",
        action="store_true",
        default=USE_TROPOPAUSE_FEATURES,
        help="Augment model inputs with tropopause features.",
    )
    parser.add_argument(
        "--scaler",
        choices=["StandardScaler", "MinMaxScaler"],
        default="StandardScaler",
        help="Column-wise scaler used for base features.",
    )

    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--hidden-layers", type=int, default=3, help="Number of hidden layers.")
    parser.add_argument(
        "--disable-d-output",
        action="store_true",
        help="Disable diffusivity output branch (only valid when PHYSICS_CONSTRAINT is not 'diffusivity').",
    )

    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument("--lambda-phys-start", type=float, default=1.0, help="Initial physics-loss weight.")
    parser.add_argument("--lambda-sup-start", type=float, default=1.0, help="Initial supervised-loss weight.")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="Physics-weight decay rate.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device.")

    parser.add_argument("--enable-time-encoding", action="store_true", help="Enable time encoding pipeline.")
    parser.add_argument("--use-cyclical", action="store_true", help="Add cyclical sin/cos time features.")
    parser.add_argument("--use-rbf-seasonal", action="store_true", help="Add seasonal RBF time features.")
    parser.add_argument("--use-rbf-absolute", action="store_true", help="Add absolute-time RBF features.")
    parser.add_argument("--n-rbf-seasonal", type=int, default=12, help="Seasonal RBF feature count.")
    parser.add_argument("--n-rbf-absolute", type=int, default=24, help="Absolute-time RBF feature count.")
    parser.add_argument("--gamma-seasonal", type=float, default=100.0, help="Seasonal RBF gamma.")
    parser.add_argument("--gamma-absolute", type=float, default=5.0, help="Absolute-time RBF gamma.")
    parser.add_argument("--t-min", type=float, default=1980.0, help="Lower bound for absolute time normalization.")
    parser.add_argument("--t-max", type=float, default=2025.0, help="Upper bound for absolute time normalization.")
    parser.add_argument("--time-col", type=int, default=2, help="Base input index for time.")

    parser.add_argument("--checkpoint-name", type=str, default=None, help="Optional checkpoint filename.")

    return parser.parse_args()


def main() -> None:
    """Train model end-to-end from configured data sources."""
    args = parse_args()
    time_range = parse_time_range(args.time_start, args.time_end)
    time_config = build_time_encoding_config(args)

    if PHYSICS_CONSTRAINT == "harmonic" and time_config.get("enabled", False):
        raise ValueError(
            "Time encoding is not supported when PHYSICS_CONSTRAINT='harmonic'. "
            "Disable time encoding flags or switch physics mode."
        )
    if PHYSICS_CONSTRAINT == "diffusivity" and args.disable_d_output:
        raise ValueError(
            "--disable-d-output is incompatible with PHYSICS_CONSTRAINT='diffusivity'. "
            "Enable the D branch or switch the physics constraint."
        )

    if not any([args.sat_paths, args.insitu_paths, args.model_paths]):
        raise ValueError("Provide at least one dataset via --sat-paths, --insitu-paths, or --model-paths.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, Gamma, W = load_all_data_combined(
        sat_paths=args.sat_paths,
        insitu_paths=args.insitu_paths,
        model_paths=args.model_paths,
        time_range=time_range,
        trop_features=args.use_tropopause_features,
    )

    tau_R = None
    if args.tau_path:
        tau_interp, valid_mask = load_tau_R(args.tau_path, X)
        X = X[valid_mask]
        Gamma = Gamma[valid_mask]
        W = W[valid_mask]
        tau_R = tau_interp[valid_mask]

    if len(X) == 0:
        raise ValueError("No training samples remain after filtering. Check data paths and tau interpolation coverage.")

    X_scaled, scaler_X = scale_variables_columnwise(X, scaler=args.scaler)

    train_loader, val_loader = create_train_val_loaders(
        X_scaled,
        Gamma,
        W,
        tau_R=tau_R,
        batch_size=args.batch_size,
        val_split=args.val_split,
        device=device,
        time_encoding_config=time_config,
    )

    model = PINNModel(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        include_D=not args.disable_d_output,
        use_tropopause_features=args.use_tropopause_features,
        time_encoding_config=time_config,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses, val_losses, _, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        lambda_phys_start=args.lambda_phys_start,
        lambda_sup_start=args.lambda_sup_start,
        decay_rate=args.decay_rate,
    )

    save_checkpoint(model, optimizer, name=args.checkpoint_name, scaler=scaler_X)
    save_config(time_config)

    print(
        "Training completed. "
        f"Final train loss: {train_losses[-1]:.4e}, final validation loss: {val_losses[-1]:.4e}"
    )


if __name__ == "__main__":
    main()
