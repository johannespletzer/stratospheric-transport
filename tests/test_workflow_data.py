from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from residence_time.workflow_data import (
    discover_data_files,
    get_default_workflow_config,
    load_workflow_config,
    resolve_data_paths,
)


def _write_satellite_nc(path: Path) -> None:
    time = pd.date_range("2000-01-01", periods=2, freq="MS")
    lat = np.array([-10.0, 10.0])
    alt = np.array([18.0, 20.0])
    aoa = np.full((time.size, lat.size, alt.size), 2.0)
    aoa_std = np.full((time.size, lat.size, alt.size), 0.2)
    ds = xr.Dataset(
        {
            "AoA": (("time", "lat", "alt"), aoa),
            "AoA_STD": (("time", "lat", "alt"), aoa_std),
        },
        coords={"time": time, "lat": lat, "alt": alt},
    )
    ds.to_netcdf(path)


def _write_insitu_nc(path: Path) -> None:
    ds = xr.Dataset(
        {
            "Mean_Age_SF6_corr": (("season", "lat", "Altitude"), np.full((1, 1, 1), 2.4)),
            "Mean_Age_CO2": (("season", "lat", "Altitude"), np.full((1, 1, 1), 2.5)),
            "Mean_Age_SF6_corr_STD": (("season", "lat", "Altitude"), np.full((1, 1, 1), 0.2)),
            "Mean_Age_CO2_STD": (("season", "lat", "Altitude"), np.full((1, 1, 1), 0.3)),
        },
        coords={"season": [0], "lat": [5.0], "Altitude": [20.0]},
    )
    ds.to_netcdf(path)


def _write_insitu_co2_only_nc(path: Path) -> None:
    ds = xr.Dataset(
        {
            "Mean_Age_CO2": (("season", "lat", "Altitude"), np.full((1, 1, 1), 2.5)),
            "Mean_Age_CO2_STD": (("season", "lat", "Altitude"), np.full((1, 1, 1), 0.3)),
        },
        coords={"season": [0], "lat": [5.0], "Altitude": [20.0]},
    )
    ds.to_netcdf(path)


def _write_model_nc(path: Path) -> None:
    ds = xr.Dataset(
        {
            "AOA": (("time", "lev", "lat"), np.full((1, 2, 2), 2.0)),
        },
        coords={"time": pd.date_range("2000-01-01", periods=1), "lev": [50e2, 70e2], "lat": [-10.0, 10.0]},
    )
    ds.to_netcdf(path)


def _write_tau_nc(path: Path) -> None:
    ds = xr.Dataset(
        {
            "tau": (("lev", "lat"), np.full((2, 2), 1.2)),
        },
        coords={"lev": [18.0, 22.0], "lat": [-10.0, 10.0]},
    )
    ds.to_netcdf(path)


def _base_config() -> Dict[str, object]:
    return {
        "sat_paths": [],
        "insitu_paths": [],
        "model_paths": [],
        "tau_path": None,
        "tropopause_csv": None,
        "use_tropopause_features": False,
        "autodiscover": {
            "enabled": True,
            "search_roots": ["data"],
            "discover_model_paths": False,
            "discover_tau_path": False,
        },
    }


def test_discovery_classifies_netcdf_schemas(tmp_path: Path) -> None:
    """Discovery should classify all supported NetCDF schema types."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    sat_path = data_dir / "satellite.nc"
    insitu_path = data_dir / "insitu.nc"
    model_path = data_dir / "model.nc"
    tau_path = data_dir / "tau.nc"
    _write_satellite_nc(sat_path)
    _write_insitu_nc(insitu_path)
    _write_model_nc(model_path)
    _write_tau_nc(tau_path)

    discovered = discover_data_files(search_roots=["data"], project_root=tmp_path)
    assert str(sat_path.resolve()) in discovered["sat_paths"]
    assert str(insitu_path.resolve()) in discovered["insitu_paths"]
    assert str(model_path.resolve()) in discovered["model_paths"]
    assert str(tau_path.resolve()) in discovered["tau_paths"]


def test_discovery_skips_incomplete_insitu_schema(tmp_path: Path) -> None:
    """Discovery should not classify in-situ files that loader cannot parse."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    insitu_path = data_dir / "insitu_co2_only.nc"
    _write_insitu_co2_only_nc(insitu_path)

    discovered = discover_data_files(search_roots=["data"], project_root=tmp_path)
    assert str(insitu_path.resolve()) not in discovered["insitu_paths"]


def test_resolve_paths_prefers_yaml_explicit_over_discovery(tmp_path: Path) -> None:
    """Explicit YAML paths should win over discovered candidates."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir(parents=True)

    discovered_sat = data_dir / "satellite_discovered.nc"
    explicit_sat = explicit_dir / "satellite_explicit.nc"
    _write_satellite_nc(discovered_sat)
    _write_satellite_nc(explicit_sat)

    config = _base_config()
    config["sat_paths"] = [str(explicit_sat)]

    resolved = resolve_data_paths(config, project_root=tmp_path)
    assert resolved.sat_paths == [str(explicit_sat.resolve())]
    assert resolved.sources["sat_paths"] == "yaml_explicit"


def test_resolve_paths_prefers_cli_over_yaml(tmp_path: Path) -> None:
    """CLI data paths should override YAML-defined paths."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    yaml_sat = data_dir / "satellite_yaml.nc"
    cli_sat = data_dir / "satellite_cli.nc"
    _write_satellite_nc(yaml_sat)
    _write_satellite_nc(cli_sat)

    config = _base_config()
    config["sat_paths"] = [str(yaml_sat)]

    resolved = resolve_data_paths(
        config,
        project_root=tmp_path,
        cli_overrides={"sat_paths": [str(cli_sat)]},
    )
    assert resolved.sat_paths == [str(cli_sat.resolve())]
    assert resolved.sources["sat_paths"] == "cli_explicit"


def test_tau_discovery_raises_on_ambiguous_candidates(tmp_path: Path) -> None:
    """Tau discovery should fail when multiple candidate files exist."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    sat_path = data_dir / "satellite.nc"
    tau_a = data_dir / "tau_a.nc"
    tau_b = data_dir / "tau_b.nc"
    _write_satellite_nc(sat_path)
    _write_tau_nc(tau_a)
    _write_tau_nc(tau_b)

    config = _base_config()
    config["autodiscover"]["discover_tau_path"] = True

    with pytest.raises(ValueError, match="Ambiguous tau_path"):
        resolve_data_paths(config, project_root=tmp_path)


def test_tropopause_csv_fallback_schema_detection(tmp_path: Path) -> None:
    """Fallback discovery should pick the first schema-compatible tropopause CSV."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir(parents=True)

    sat_path = data_dir / "satellite.nc"
    _write_satellite_nc(sat_path)

    tropopause_csv = custom_dir / "fallback_tp.csv"
    df = pd.DataFrame(
        {
            "time": ["2000-01-01"],
            "tp_WMO_tro": [1.0],
            "tp_WMO_sh_pol": [2.0],
            "tp_WMO_nh_pol": [3.0],
            "tp_WMO_tro_std": [0.1],
            "tp_WMO_sh_pol_std": [0.1],
            "tp_WMO_nh_pol_std": [0.1],
        }
    )
    df.to_csv(tropopause_csv, index=False)

    config = _base_config()
    config["use_tropopause_features"] = True
    config["autodiscover"]["search_roots"] = ["data", "custom"]

    resolved = resolve_data_paths(config, project_root=tmp_path)
    assert resolved.tropopause_csv == str(tropopause_csv.resolve())
    assert resolved.sources["tropopause_csv"] == "autodiscovered_schema"


def test_default_workflow_config_includes_resume_defaults() -> None:
    """Workflow defaults should include resume settings."""
    cfg = get_default_workflow_config()
    assert cfg["resume"]["enabled"] is False
    assert cfg["resume"]["checkpoint_path"] is None
    assert cfg["resume"]["source_config_path"] is None
    assert cfg["resume"]["load_optimizer"] is True
    assert cfg["resume"]["load_scaler"] is True


def test_load_workflow_config_merges_resume_defaults(tmp_path: Path) -> None:
    """Loaded configs should receive resume defaults when section is omitted."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  run_id: sample\n", encoding="utf-8")

    loaded = load_workflow_config(str(config_path))
    assert loaded["run"]["run_id"] == "sample"
    assert loaded["resume"]["enabled"] is False
    assert loaded["resume"]["checkpoint_path"] is None
    assert loaded["resume"]["source_config_path"] is None
    assert loaded["resume"]["load_optimizer"] is True
    assert loaded["resume"]["load_scaler"] is True
