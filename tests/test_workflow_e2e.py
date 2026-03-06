import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from residence_time.model import PINNModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_TIMEOUT_SECONDS = 300


def _run_workflow_raw(args: List[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "workflow.py"), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        env=env,
    )


def _run_workflow(args: List[str]) -> None:
    result = _run_workflow_raw(args)
    assert result.returncode == 0, f"workflow command failed: {result.stdout}\n{result.stderr}"


def _write_satellite_nc(path: Path) -> None:
    time = pd.date_range("2001-01-01", periods=6, freq="MS")
    lat = np.array([-30.0, 0.0, 30.0])
    alt = np.array([18.0, 22.0])

    base = np.linspace(1.5, 3.0, time.size * lat.size * alt.size)
    aoa = base.reshape(time.size, lat.size, alt.size)
    aoa_std = np.full_like(aoa, 0.2)

    ds = xr.Dataset(
        {
            "AoA": (("time", "lat", "alt"), aoa),
            "AoA_STD": (("time", "lat", "alt"), aoa_std),
        },
        coords={"time": time, "lat": lat, "alt": alt},
    )
    ds.to_netcdf(path)


def _write_config(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _build_config(run_id: str, output_root: Path, sat_path: Path, seed: int, hidden_dim: int) -> Dict[str, Any]:
    return {
        "run": {"run_id": run_id, "output_root": str(output_root)},
        "data": {
            "sat_paths": [str(sat_path)],
            "insitu_paths": [],
            "model_paths": [],
            "tau_path": None,
            "tropopause_csv": None,
            "use_tropopause_features": False,
            "autodiscover": {
                "enabled": True,
                "search_roots": [str(sat_path.parent)],
                "discover_model_paths": False,
                "discover_tau_path": False,
            },
        },
        "model": {
            "hidden_dim": hidden_dim,
            "hidden_layers": 2,
            "disable_d_output": False,
            "checkpoint_name": "model_checkpoint.pth",
        },
        "training": {
            "batch_size": 8,
            "val_split": 0.25,
            "epochs": 2,
            "learning_rate": 0.001,
            "lambda_phys_start": 1.0,
            "lambda_sup_start": 1.0,
            "decay_rate": 0.95,
        },
        "runtime": {"device": "cpu", "seed": seed},
    }


def _write_checkpoint_without_scaler(path: Path, hidden_dim: int, hidden_layers: int) -> None:
    model = PINNModel(
        input_dim=5,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        include_D=True,
        use_tropopause_features=False,
        time_encoding_config={"enabled": False},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": "20260101_000000",
        },
        path,
    )


def test_workflow_train_plot_compare_e2e(tmp_path: Path) -> None:
    """Workflow CLI should produce train, plot, and compare artifacts end-to-end."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config_a = tmp_path / "run_a.yaml"
    config_b = tmp_path / "run_b.yaml"
    _write_config(config_a, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _write_config(config_b, _build_config("run_b", output_root, sat_path, seed=11, hidden_dim=16))

    _run_workflow(["train", "--config", str(config_a)])
    _run_workflow(["train", "--config", str(config_b)])

    run_a_dir = output_root / "run_a"
    run_b_dir = output_root / "run_b"
    assert (run_a_dir / "history.csv").is_file()
    assert (run_a_dir / "summary.json").is_file()
    assert (run_a_dir / "resolved_config.yaml").is_file()
    assert (run_a_dir / "model_checkpoint.pth").is_file()
    assert (run_b_dir / "history.csv").is_file()

    _run_workflow(["plot", "--run-dir", str(run_a_dir)])
    assert (run_a_dir / "plots" / "training_losses.png").is_file()
    assert (run_a_dir / "plots" / "loss_components.png").is_file()
    _run_workflow(["plot-field", "--run-dir", str(run_a_dir)])
    assert (run_a_dir / "plots" / "tau_R_field.png").is_file()

    compare_out = tmp_path / "compare"
    _run_workflow(
        [
            "compare",
            "--run-dirs",
            str(run_a_dir),
            str(run_b_dir),
            "--output-dir",
            str(compare_out),
        ]
    )
    assert (compare_out / "leaderboard.csv").is_file()
    assert (compare_out / "val_loss_overlay.png").is_file()


def test_workflow_resume_from_previous_checkpoint_e2e(tmp_path: Path) -> None:
    """Resume training should create a new run with resume metadata in summary."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config_a = tmp_path / "run_a.yaml"
    _write_config(config_a, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config_a)])

    run_a_dir = output_root / "run_a"
    run_a_checkpoint = (run_a_dir / "model_checkpoint.pth").resolve()
    run_a_resolved_cfg = (run_a_dir / "resolved_config.yaml").resolve()

    config_resume = _build_config("run_resume", output_root, sat_path, seed=13, hidden_dim=12)
    config_resume["training"]["epochs"] = 1
    config_resume["resume"] = {
        "enabled": True,
        "checkpoint_path": str(run_a_checkpoint),
        "source_config_path": str(run_a_resolved_cfg),
        "load_optimizer": True,
        "load_scaler": True,
    }
    config_resume_path = tmp_path / "run_resume.yaml"
    _write_config(config_resume_path, config_resume)
    _run_workflow(["train", "--config", str(config_resume_path)])

    run_resume_dir = output_root / "run_resume"
    assert (run_resume_dir / "history.csv").is_file()
    assert (run_resume_dir / "summary.json").is_file()
    assert (run_resume_dir / "resolved_config.yaml").is_file()
    assert (run_resume_dir / "model_checkpoint.pth").is_file()

    with (run_resume_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["resumed"] is True
    assert summary["resume_checkpoint_path"] == str(run_a_checkpoint)
    assert summary["resume_source_config_path"] == str(run_a_resolved_cfg)
    assert summary["resume_loaded_optimizer"] is True
    assert summary["resume_loaded_scaler"] is True


def test_workflow_resume_cli_checkpoint_overrides_yaml(tmp_path: Path) -> None:
    """CLI --resume-from should override YAML resume.checkpoint_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config_a = tmp_path / "run_a.yaml"
    _write_config(config_a, _build_config("run_a", output_root, sat_path, seed=3, hidden_dim=12))
    _run_workflow(["train", "--config", str(config_a)])

    run_a_dir = output_root / "run_a"
    run_a_checkpoint = (run_a_dir / "model_checkpoint.pth").resolve()
    run_a_resolved_cfg = (run_a_dir / "resolved_config.yaml").resolve()

    config_override = _build_config("run_cli_override", output_root, sat_path, seed=5, hidden_dim=12)
    config_override["training"]["epochs"] = 1
    config_override["resume"] = {
        "enabled": True,
        "checkpoint_path": "missing/invalid/path.pth",
        "source_config_path": None,
        "load_optimizer": True,
        "load_scaler": True,
    }
    config_override_path = tmp_path / "run_cli_override.yaml"
    _write_config(config_override_path, config_override)

    _run_workflow(
        [
            "train",
            "--config",
            str(config_override_path),
            "--resume-from",
            str(run_a_checkpoint),
            "--resume-config",
            str(run_a_resolved_cfg),
        ]
    )

    with (output_root / "run_cli_override" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["resumed"] is True
    assert summary["resume_checkpoint_path"] == str(run_a_checkpoint)
    assert summary["resume_source_config_path"] == str(run_a_resolved_cfg)


def test_workflow_resume_requires_checkpoint_path(tmp_path: Path) -> None:
    """resume.enabled=true should fail when checkpoint_path is unset."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config = _build_config("run_missing_checkpoint", output_root, sat_path, seed=7, hidden_dim=12)
    config["training"]["epochs"] = 1
    config["resume"] = {
        "enabled": True,
        "checkpoint_path": None,
        "source_config_path": None,
        "load_optimizer": True,
        "load_scaler": True,
    }
    config_path = tmp_path / "run_missing_checkpoint.yaml"
    _write_config(config_path, config)

    result = _run_workflow_raw(["train", "--config", str(config_path)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "resume.checkpoint_path" in output


def test_workflow_resume_requires_scaler_when_enabled(tmp_path: Path) -> None:
    """Resume should fail if scaler loading is requested but scaler is absent."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    checkpoint_path = tmp_path / "checkpoint_no_scaler.pth"
    _write_checkpoint_without_scaler(checkpoint_path, hidden_dim=12, hidden_layers=2)

    output_root = tmp_path / "runs"
    config = _build_config("run_missing_scaler", output_root, sat_path, seed=9, hidden_dim=12)
    config["training"]["epochs"] = 1
    config["resume"] = {
        "enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "source_config_path": None,
        "load_optimizer": True,
        "load_scaler": True,
    }
    config_path = tmp_path / "run_missing_scaler.yaml"
    _write_config(config_path, config)

    result = _run_workflow_raw(["train", "--config", str(config_path)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "does not contain a scaler" in output


def test_workflow_resume_surfaces_incompatible_architecture_error(tmp_path: Path) -> None:
    """Resume should surface a clear load_state_dict error for incompatible models."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config_a = tmp_path / "run_a.yaml"
    _write_config(config_a, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config_a)])

    run_a_checkpoint = (output_root / "run_a" / "model_checkpoint.pth").resolve()

    config_bad = _build_config("run_bad_resume", output_root, sat_path, seed=10, hidden_dim=16)
    config_bad["training"]["epochs"] = 1
    config_bad["resume"] = {
        "enabled": True,
        "checkpoint_path": str(run_a_checkpoint),
        "source_config_path": None,
        "load_optimizer": True,
        "load_scaler": True,
    }
    config_bad_path = tmp_path / "run_bad_resume.yaml"
    _write_config(config_bad_path, config_bad)

    result = _run_workflow_raw(["train", "--config", str(config_bad_path)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "loading state_dict" in output.lower() or "size mismatch" in output.lower()


def test_workflow_plot_field_requires_checkpoint(tmp_path: Path) -> None:
    """plot-field should fail with a clear message when checkpoint is missing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config = tmp_path / "run_a.yaml"
    _write_config(config, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config)])

    run_dir = output_root / "run_a"
    (run_dir / "model_checkpoint.pth").unlink()

    result = _run_workflow_raw(["plot-field", "--run-dir", str(run_dir)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "Checkpoint file not found in run directory" in output


def test_workflow_plot_field_requires_resolved_config(tmp_path: Path) -> None:
    """plot-field should fail with a clear message when resolved config is missing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config = tmp_path / "run_a.yaml"
    _write_config(config, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config)])

    run_dir = output_root / "run_a"
    (run_dir / "resolved_config.yaml").unlink()

    result = _run_workflow_raw(["plot-field", "--run-dir", str(run_dir)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "resolved_config.yaml not found in run directory" in output


def test_workflow_plot_field_rejects_invalid_resolved_config(tmp_path: Path) -> None:
    """plot-field should fail with a clear message for malformed resolved config."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config = tmp_path / "run_a.yaml"
    _write_config(config, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config)])

    run_dir = output_root / "run_a"
    (run_dir / "resolved_config.yaml").write_text("{invalid_yaml: [", encoding="utf-8")

    result = _run_workflow_raw(["plot-field", "--run-dir", str(run_dir)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "Failed to parse resolved workflow config" in output


def test_workflow_plot_field_requires_scaler_in_checkpoint(tmp_path: Path) -> None:
    """plot-field should fail when checkpoint does not contain a stored scaler."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    output_root = tmp_path / "runs"
    config = tmp_path / "run_a.yaml"
    _write_config(config, _build_config("run_a", output_root, sat_path, seed=7, hidden_dim=12))
    _run_workflow(["train", "--config", str(config)])

    run_dir = output_root / "run_a"
    checkpoint_path = run_dir / "model_checkpoint.pth"
    _write_checkpoint_without_scaler(checkpoint_path, hidden_dim=12, hidden_layers=2)

    result = _run_workflow_raw(["plot-field", "--run-dir", str(run_dir)])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "requires a checkpoint with a stored scaler" in output
