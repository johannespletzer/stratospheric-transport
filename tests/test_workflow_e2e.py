import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_TIMEOUT_SECONDS = 300


def _run_workflow(args: List[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "workflow.py"), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        env=env,
    )
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
