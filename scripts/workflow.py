"""Workflow CLI: train, plot, and compare runs with reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from yaml import safe_dump

from residence_time.config import LEARNING_RATE, PHYSICS_CONSTRAINT
from residence_time.data import load_all_data_combined, load_tau_R
from residence_time.model import PINNModel
from residence_time.train import (
    create_train_val_loaders,
    extract_phase_and_scale,
    extract_phase_and_transform_with_scaler,
    train_model,
)
from residence_time.utils import load_checkpoint, save_checkpoint
from residence_time.workflow_data import (
    PROJECT_ROOT,
    ResolvedDataPaths,
    load_workflow_config,
    parse_time_range,
    resolve_data_paths,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt



def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_resume_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate resume settings from merged config."""
    resume_cfg = dict(config.get("resume", {}))
    resume_cfg.setdefault("enabled", False)
    resume_cfg.setdefault("checkpoint_path", None)
    resume_cfg.setdefault("source_config_path", None)
    resume_cfg.setdefault("load_optimizer", True)
    resume_cfg.setdefault("load_scaler", True)

    resume_cfg["enabled"] = bool(resume_cfg["enabled"])
    resume_cfg["load_optimizer"] = bool(resume_cfg["load_optimizer"])
    resume_cfg["load_scaler"] = bool(resume_cfg["load_scaler"])

    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path is not None and str(checkpoint_path).strip():
        resume_cfg["checkpoint_path"] = str(_resolve_path(str(checkpoint_path)))
    else:
        resume_cfg["checkpoint_path"] = None

    source_config_path = resume_cfg.get("source_config_path")
    if source_config_path is not None and str(source_config_path).strip():
        resume_cfg["source_config_path"] = str(_resolve_path(str(source_config_path)))
    else:
        resume_cfg["source_config_path"] = None

    if resume_cfg["enabled"]:
        if resume_cfg["checkpoint_path"] is None:
            raise ValueError("resume.enabled=true requires resume.checkpoint_path to be set.")
        if not Path(resume_cfg["checkpoint_path"]).is_file():
            raise FileNotFoundError(
                f"Resume checkpoint file not found: {resume_cfg['checkpoint_path']}"
            )

    if (
        resume_cfg["enabled"]
        and resume_cfg["source_config_path"]
        and not Path(resume_cfg["source_config_path"]).is_file()
    ):
        raise FileNotFoundError(
            f"Resume source config file not found: {resume_cfg['source_config_path']}"
        )

    return resume_cfg


def _get_git_sha() -> str | None:
    """Return current short git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _build_time_encoding_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize time encoding config from merged workflow config."""
    time_cfg = dict(config.get("time_encoding", {}))
    # Keep backward-compatible behavior: enabling any feature toggles full encoding mode.
    time_cfg["enabled"] = bool(
        time_cfg.get("enabled", False)
        or time_cfg.get("use_cyclical", False)
        or time_cfg.get("use_rbf_seasonal", False)
        or time_cfg.get("use_rbf_absolute", False)
    )
    return time_cfg


def _default_run_id() -> str:
    """Build a default run identifier."""
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def _create_run_dir(config: dict[str, Any]) -> Path:
    """Create and return run output directory."""
    run_cfg = config["run"]
    output_root = _resolve_path(str(run_cfg.get("output_root", "runs")))
    run_id = run_cfg.get("run_id") or _default_run_id()
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _apply_train_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply optional CLI overrides to merged config."""
    if args.run_id:
        config["run"]["run_id"] = args.run_id
    if args.output_root:
        config["run"]["output_root"] = args.output_root

    override_map: list[tuple[str, str, str, type[Any]]] = [
        ("runtime", "device", "device", str),
        ("runtime", "seed", "seed", int),
        ("training", "epochs", "epochs", int),
        ("training", "batch_size", "batch_size", int),
        ("training", "val_split", "val_split", float),
        ("training", "learning_rate", "learning_rate", float),
        ("training", "lambda_phys_start", "lambda_phys_start", float),
        ("training", "lambda_sup_start", "lambda_sup_start", float),
        ("training", "decay_rate", "decay_rate", float),
        ("model", "hidden_dim", "hidden_dim", int),
        ("model", "hidden_layers", "hidden_layers", int),
        ("model", "disable_d_output", "disable_d_output", bool),
        ("model", "checkpoint_name", "checkpoint_name", str),
        ("data", "use_tropopause_features", "use_tropopause_features", bool),
        ("data", "scaler", "scaler", str),
        ("data", "time_start", "time_start", str),
        ("data", "time_end", "time_end", str),
        ("time_encoding", "enabled", "time_encoding_enabled", bool),
        ("time_encoding", "use_cyclical", "use_cyclical", bool),
        ("time_encoding", "use_rbf_seasonal", "use_rbf_seasonal", bool),
        ("time_encoding", "use_rbf_absolute", "use_rbf_absolute", bool),
        ("resume", "load_optimizer", "resume_load_optimizer", bool),
        ("resume", "load_scaler", "resume_load_scaler", bool),
    ]

    for section, key, arg_name, caster in override_map:
        value = getattr(args, arg_name, None)
        if value is not None:
            config[section][key] = caster(value)

    if args.resume_from is not None:
        config["resume"]["checkpoint_path"] = str(args.resume_from)
        config["resume"]["enabled"] = True
    if args.resume_config is not None:
        config["resume"]["source_config_path"] = str(args.resume_config)


def _cli_data_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build data-resolution overrides from CLI args."""
    return {
        "sat_paths": args.sat_paths,
        "insitu_paths": args.insitu_paths,
        "model_paths": args.model_paths,
        "tau_path": args.tau_path,
        "tropopause_csv": args.tropopause_csv,
        "search_roots": args.search_roots,
        "autodiscover_enabled": args.autodiscover_enabled,
        "discover_model_paths": args.discover_model_paths,
        "discover_tau_path": args.discover_tau_path,
    }


def _print_resolved_paths(resolved: ResolvedDataPaths) -> None:
    """Print resolved paths and source labels."""
    print("Resolved data paths:")
    print(f"  sat_paths ({resolved.sources['sat_paths']}): {resolved.sat_paths}")
    print(f"  insitu_paths ({resolved.sources['insitu_paths']}): {resolved.insitu_paths}")
    print(f"  model_paths ({resolved.sources['model_paths']}): {resolved.model_paths}")
    print(f"  tau_path ({resolved.sources['tau_path']}): {resolved.tau_path}")
    print(f"  tropopause_csv ({resolved.sources['tropopause_csv']}): {resolved.tropopause_csv}")


def _write_resolved_config(
    run_dir: Path,
    config: dict[str, Any],
    resolved: ResolvedDataPaths,
) -> None:
    """Write resolved workflow config to run directory."""
    resolved_config = deepcopy(config)
    resolved_config["data"]["sat_paths"] = resolved.sat_paths
    resolved_config["data"]["insitu_paths"] = resolved.insitu_paths
    resolved_config["data"]["model_paths"] = resolved.model_paths
    resolved_config["data"]["tau_path"] = resolved.tau_path
    resolved_config["data"]["tropopause_csv"] = resolved.tropopause_csv
    resolved_config["data"]["resolution_sources"] = resolved.sources
    resolved_config["data"]["autodiscovered_candidates"] = resolved.discovered

    target = run_dir / "resolved_config.yaml"
    with target.open("w", encoding="utf-8") as handle:
        safe_dump(resolved_config, handle, sort_keys=False)


def _write_history(
    run_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    physics_losses: list[float],
    supervised_losses: list[float],
) -> Path:
    """Write per-epoch loss history to CSV."""
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_losses) + 1, dtype=int),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "physics_loss": physics_losses,
            "supervised_loss": supervised_losses,
        }
    )
    history_path = run_dir / "history.csv"
    history.to_csv(history_path, index=False)
    return history_path


def _write_summary(
    run_dir: Path,
    config: dict[str, Any],
    resolved: ResolvedDataPaths,
    device: str,
    train_losses: list[float],
    val_losses: list[float],
    physics_losses: list[float],
    supervised_losses: list[float],
    n_samples: int,
    *,
    resumed: bool,
    resume_checkpoint_path: str | None,
    resume_source_config_path: str | None,
    resume_loaded_optimizer: bool,
    resume_loaded_scaler: bool,
) -> Path:
    """Write run summary JSON."""
    best_val_idx = int(np.argmin(np.asarray(val_losses)))
    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "git_sha": _get_git_sha(),
        "physics_constraint": PHYSICS_CONSTRAINT,
        "seed": int(config["runtime"]["seed"]),
        "device": device,
        "n_samples": int(n_samples),
        "data_paths": {
            "sat_paths": resolved.sat_paths,
            "insitu_paths": resolved.insitu_paths,
            "model_paths": resolved.model_paths,
            "tau_path": resolved.tau_path,
            "tropopause_csv": resolved.tropopause_csv,
            "sources": resolved.sources,
        },
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "final_physics_loss": float(physics_losses[-1]),
        "final_supervised_loss": float(supervised_losses[-1]),
        "best_epoch_by_val_loss": best_val_idx + 1,
        "best_val_loss": float(val_losses[best_val_idx]),
        "best_train_loss": float(train_losses[int(np.argmin(np.asarray(train_losses)))]),
        "best_physics_loss": float(physics_losses[int(np.argmin(np.asarray(physics_losses)))]),
        "best_supervised_loss": float(supervised_losses[int(np.argmin(np.asarray(supervised_losses)))]),
        "resumed": bool(resumed),
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_source_config_path": resume_source_config_path,
        "resume_loaded_optimizer": bool(resume_loaded_optimizer),
        "resume_loaded_scaler": bool(resume_loaded_scaler),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_train(args: argparse.Namespace) -> None:
    """Execute workflow training command."""
    config = load_workflow_config(args.config)
    _apply_train_overrides(config, args)
    resume_cfg = _resolve_resume_settings(config)
    config["resume"] = resume_cfg

    resolved = resolve_data_paths(
        config["data"],
        cli_overrides=_cli_data_overrides(args),
        project_root=PROJECT_ROOT,
    )
    _print_resolved_paths(resolved)

    run_dir = _create_run_dir(config)
    print(f"Run directory: {run_dir}")

    _write_resolved_config(run_dir, config, resolved)

    seed = int(config["runtime"]["seed"])
    set_seed(seed)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    time_cfg = _build_time_encoding_config(config)

    if PHYSICS_CONSTRAINT == "harmonic" and time_cfg.get("enabled", False):
        raise ValueError(
            "Time encoding is not supported when PHYSICS_CONSTRAINT='harmonic'. "
            "Disable time encoding options."
        )
    if PHYSICS_CONSTRAINT == "diffusivity" and model_cfg.get("disable_d_output", False):
        raise ValueError(
            "disable_d_output is incompatible with PHYSICS_CONSTRAINT='diffusivity'."
        )

    time_range = parse_time_range(data_cfg.get("time_start"), data_cfg.get("time_end"))
    configured_device = config["runtime"].get("device")
    device = configured_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, gamma, weights = load_all_data_combined(
        sat_paths=resolved.sat_paths or None,
        insitu_paths=resolved.insitu_paths or None,
        model_paths=resolved.model_paths or None,
        time_range=time_range,
        trop_features=bool(data_cfg.get("use_tropopause_features", False)),
        tropopause_csv_path=resolved.tropopause_csv,
    )

    tau_r = None
    if resolved.tau_path:
        tau_interp, valid_mask = load_tau_R(resolved.tau_path, X)
        X = X[valid_mask]
        gamma = gamma[valid_mask]
        weights = weights[valid_mask]
        tau_r = tau_interp[valid_mask]

    if len(X) == 0:
        raise ValueError("No training samples remain after filtering/interpolation.")

    model = PINNModel(
        input_dim=5,
        hidden_dim=int(model_cfg["hidden_dim"]),
        hidden_layers=int(model_cfg["hidden_layers"]),
        include_D=not bool(model_cfg.get("disable_d_output", False)),
        use_tropopause_features=bool(data_cfg.get("use_tropopause_features", False)),
        time_encoding_config=time_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", LEARNING_RATE)))

    resume_loaded_optimizer = False
    resume_loaded_scaler = False
    scaler_from_checkpoint = None
    if resume_cfg["enabled"]:
        optimizer_for_load = optimizer if resume_cfg["load_optimizer"] else None
        scaler_from_checkpoint = load_checkpoint(
            model,
            checkpoint_path=str(resume_cfg["checkpoint_path"]),
            device=device,
            optimizer=optimizer_for_load,
            return_scaler=bool(resume_cfg["load_scaler"]),
        )
        resume_loaded_optimizer = bool(resume_cfg["load_optimizer"])

        if resume_cfg["load_scaler"]:
            if scaler_from_checkpoint is None:
                raise ValueError(
                    "Resume checkpoint does not contain a scaler, but resume.load_scaler=true."
                )
            x_scaled, seasonal_phase, time_std = extract_phase_and_transform_with_scaler(
                X,
                scaler_X=scaler_from_checkpoint,
            )
            scaler_x = scaler_from_checkpoint
            resume_loaded_scaler = True

    if not resume_loaded_scaler:
        x_scaled, scaler_x, seasonal_phase, time_std = extract_phase_and_scale(
            X,
            scaler=str(data_cfg.get("scaler", "StandardScaler")),
        )

    train_loader, val_loader = create_train_val_loaders(
        x_scaled,
        gamma,
        weights,
        tau_R=tau_r,
        batch_size=int(train_cfg["batch_size"]),
        val_split=float(train_cfg["val_split"]),
        device=device,
        time_encoding_config=time_cfg,
        seasonal_phase=seasonal_phase,
    )

    train_losses, val_losses, physics_losses, supervised_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=int(train_cfg["epochs"]),
        lambda_phys_start=float(train_cfg["lambda_phys_start"]),
        lambda_sup_start=float(train_cfg["lambda_sup_start"]),
        decay_rate=float(train_cfg["decay_rate"]),
        time_std=time_std,
    )

    checkpoint_name = str(model_cfg.get("checkpoint_name", "model_checkpoint.pth"))
    save_checkpoint(
        model,
        optimizer,
        name=checkpoint_name,
        scaler=scaler_x,
        checkpoint_dir=str(run_dir),
    )
    history_path = _write_history(run_dir, train_losses, val_losses, physics_losses, supervised_losses)
    summary_path = _write_summary(
        run_dir,
        config,
        resolved,
        device,
        train_losses,
        val_losses,
        physics_losses,
        supervised_losses,
        n_samples=len(x_scaled),
        resumed=bool(resume_cfg["enabled"]),
        resume_checkpoint_path=resume_cfg["checkpoint_path"],
        resume_source_config_path=resume_cfg["source_config_path"],
        resume_loaded_optimizer=resume_loaded_optimizer,
        resume_loaded_scaler=resume_loaded_scaler,
    )

    print(f"Wrote: {history_path}")
    print(f"Wrote: {summary_path}")
    print(
        "Training completed. "
        f"Final train loss: {train_losses[-1]:.4e}, final validation loss: {val_losses[-1]:.4e}"
    )


def run_plot(args: argparse.Namespace) -> None:
    """Generate per-run plots from history artifacts."""
    run_dir = _resolve_path(args.run_dir)
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"history.csv not found in run directory: {run_dir}")

    output_dir = _resolve_path(args.output_dir) if args.output_dir else (run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path)
    required_cols = {"epoch", "train_loss", "val_loss", "physics_loss", "supervised_loss"}
    if not required_cols.issubset(set(history.columns)):
        raise ValueError(f"history.csv is missing required columns: {required_cols}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss - {run_dir.name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    total_loss_path = output_dir / "training_losses.png"
    plt.savefig(total_loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["physics_loss"], label="Physics", linewidth=2)
    plt.plot(history["epoch"], history["supervised_loss"], label="Supervised", linewidth=2, linestyle=":")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Component")
    plt.title(f"Loss Components - {run_dir.name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    component_path = output_dir / "loss_components.png"
    plt.savefig(component_path, dpi=200)
    plt.close()

    print(f"Wrote: {total_loss_path}")
    print(f"Wrote: {component_path}")


def run_compare(args: argparse.Namespace) -> None:
    """Compare multiple runs and write leaderboard + overlay plot."""
    run_dirs = [_resolve_path(path_str) for path_str in args.run_dirs]
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary.json not found: {summary_path}")
        if not history_path.is_file():
            raise FileNotFoundError(f"history.csv not found: {history_path}")

        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "run_id": summary.get("run_id", run_dir.name),
                "run_dir": str(run_dir),
                "best_epoch_by_val_loss": summary["best_epoch_by_val_loss"],
                "best_val_loss": summary["best_val_loss"],
                "final_train_loss": summary["final_train_loss"],
                "final_val_loss": summary["final_val_loss"],
                "final_physics_loss": summary["final_physics_loss"],
                "final_supervised_loss": summary["final_supervised_loss"],
                "device": summary.get("device"),
                "git_sha": summary.get("git_sha"),
            }
        )
        histories[summary.get("run_id", run_dir.name)] = pd.read_csv(history_path)

    leaderboard = pd.DataFrame(rows).sort_values("best_val_loss", ascending=True).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    plt.figure(figsize=(9, 6))
    for _, row in leaderboard.iterrows():
        run_id = row["run_id"]
        history = histories[run_id]
        if not {"epoch", "val_loss"}.issubset(set(history.columns)):
            raise ValueError(f"history.csv for run '{run_id}' is missing epoch/val_loss columns.")
        plt.plot(history["epoch"], history["val_loss"], label=run_id, linewidth=2)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    overlay_path = output_dir / "val_loss_overlay.png"
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    print(f"Wrote: {leaderboard_path}")
    print(f"Wrote: {overlay_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build top-level workflow CLI parser."""
    parser = argparse.ArgumentParser(description="Workflow commands for train/plot/compare.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training from YAML config.")
    train_parser.add_argument("--config", required=True, help="Path to run config YAML.")
    train_parser.add_argument("--run-id", default=None, help="Optional run identifier override.")
    train_parser.add_argument("--output-root", default=None, help="Optional output root override.")

    train_parser.add_argument("--sat-paths", nargs="*", default=None, help="Explicit satellite NetCDF paths.")
    train_parser.add_argument("--insitu-paths", nargs="*", default=None, help="Explicit in-situ NetCDF paths.")
    train_parser.add_argument("--model-paths", nargs="*", default=None, help="Explicit model NetCDF paths.")
    train_parser.add_argument("--tau-path", default=None, help="Explicit tau NetCDF path.")
    train_parser.add_argument("--tropopause-csv", default=None, help="Explicit tropopause CSV path.")
    train_parser.add_argument("--search-roots", nargs="*", default=None, help="Autodiscovery search roots.")

    train_parser.add_argument("--autodiscover", dest="autodiscover_enabled", action="store_true", default=None)
    train_parser.add_argument("--no-autodiscover", dest="autodiscover_enabled", action="store_false")
    train_parser.add_argument("--discover-model-paths", dest="discover_model_paths", action="store_true", default=None)
    train_parser.add_argument("--no-discover-model-paths", dest="discover_model_paths", action="store_false")
    train_parser.add_argument("--discover-tau-path", dest="discover_tau_path", action="store_true", default=None)
    train_parser.add_argument("--no-discover-tau-path", dest="discover_tau_path", action="store_false")

    train_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    train_parser.add_argument("--seed", type=int, default=None)

    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--val-split", type=float, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--lambda-phys-start", type=float, default=None)
    train_parser.add_argument("--lambda-sup-start", type=float, default=None)
    train_parser.add_argument("--decay-rate", type=float, default=None)

    train_parser.add_argument("--hidden-dim", type=int, default=None)
    train_parser.add_argument("--hidden-layers", type=int, default=None)
    train_parser.add_argument("--disable-d-output", dest="disable_d_output", action="store_true", default=None)
    train_parser.add_argument("--enable-d-output", dest="disable_d_output", action="store_false")
    train_parser.add_argument("--checkpoint-name", default=None)
    train_parser.add_argument("--resume-from", dest="resume_from", default=None)
    train_parser.add_argument("--resume-config", dest="resume_config", default=None)
    train_parser.add_argument(
        "--resume-load-optimizer",
        dest="resume_load_optimizer",
        action="store_true",
        default=None,
    )
    train_parser.add_argument(
        "--no-resume-load-optimizer",
        dest="resume_load_optimizer",
        action="store_false",
    )
    train_parser.add_argument(
        "--resume-load-scaler",
        dest="resume_load_scaler",
        action="store_true",
        default=None,
    )
    train_parser.add_argument(
        "--no-resume-load-scaler",
        dest="resume_load_scaler",
        action="store_false",
    )

    train_parser.add_argument("--use-tropopause-features", dest="use_tropopause_features", action="store_true", default=None)
    train_parser.add_argument("--no-use-tropopause-features", dest="use_tropopause_features", action="store_false")
    train_parser.add_argument("--scaler", choices=["StandardScaler", "MinMaxScaler"], default=None)
    train_parser.add_argument("--time-start", default=None)
    train_parser.add_argument("--time-end", default=None)

    train_parser.add_argument("--enable-time-encoding", dest="time_encoding_enabled", action="store_true", default=None)
    train_parser.add_argument("--disable-time-encoding", dest="time_encoding_enabled", action="store_false")
    train_parser.add_argument("--use-cyclical", dest="use_cyclical", action="store_true", default=None)
    train_parser.add_argument("--no-use-cyclical", dest="use_cyclical", action="store_false")
    train_parser.add_argument("--use-rbf-seasonal", dest="use_rbf_seasonal", action="store_true", default=None)
    train_parser.add_argument("--no-use-rbf-seasonal", dest="use_rbf_seasonal", action="store_false")
    train_parser.add_argument("--use-rbf-absolute", dest="use_rbf_absolute", action="store_true", default=None)
    train_parser.add_argument("--no-use-rbf-absolute", dest="use_rbf_absolute", action="store_false")
    train_parser.set_defaults(func=run_train)

    plot_parser = subparsers.add_parser("plot", help="Generate plots from run history.")
    plot_parser.add_argument("--run-dir", required=True, help="Run directory containing history.csv.")
    plot_parser.add_argument("--output-dir", default=None, help="Optional output directory for plots.")
    plot_parser.set_defaults(func=run_plot)

    compare_parser = subparsers.add_parser("compare", help="Compare multiple run directories.")
    compare_parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare.")
    compare_parser.add_argument(
        "--output-dir",
        default="reports/compare/latest",
        help="Output directory for leaderboard and comparison plots.",
    )
    compare_parser.set_defaults(func=run_compare)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
