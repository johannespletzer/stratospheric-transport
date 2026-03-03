"""Config loading and hybrid data-path resolution for workflow commands."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import xarray as xr
import yaml

from residence_time.config import (
    DEFAULT_TIME_ENCODING_CONFIG,
    LEARNING_RATE,
    USE_TROPOPAUSE_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TROPOPAUSE_COLUMNS = {
    "time",
    "tp_WMO_tro",
    "tp_WMO_sh_pol",
    "tp_WMO_nh_pol",
    "tp_WMO_tro_std",
    "tp_WMO_sh_pol_std",
    "tp_WMO_nh_pol_std",
}


@dataclass
class ResolvedDataPaths:
    """Resolved data paths and their provenance."""

    sat_paths: list[str]
    insitu_paths: list[str]
    model_paths: list[str]
    tau_path: str | None
    tropopause_csv: str | None
    sources: dict[str, str]
    discovered: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "sat_paths": self.sat_paths,
            "insitu_paths": self.insitu_paths,
            "model_paths": self.model_paths,
            "tau_path": self.tau_path,
            "tropopause_csv": self.tropopause_csv,
            "sources": self.sources,
            "discovered": self.discovered,
        }


def get_default_workflow_config() -> dict[str, Any]:
    """Return workflow defaults."""
    return {
        "run": {
            "run_id": None,
            "output_root": "runs",
        },
        "data": {
            "sat_paths": [],
            "insitu_paths": [],
            "model_paths": [],
            "tau_path": None,
            "tropopause_csv": None,
            "use_tropopause_features": USE_TROPOPAUSE_FEATURES,
            "time_start": None,
            "time_end": None,
            "scaler": "StandardScaler",
            "autodiscover": {
                "enabled": True,
                "search_roots": ["data"],
                "discover_model_paths": False,
                "discover_tau_path": False,
            },
        },
        "model": {
            "hidden_dim": 64,
            "hidden_layers": 3,
            "disable_d_output": False,
            "checkpoint_name": "model_checkpoint.pth",
        },
        "training": {
            "batch_size": 256,
            "val_split": 0.2,
            "epochs": 300,
            "learning_rate": LEARNING_RATE,
            "lambda_phys_start": 1.0,
            "lambda_sup_start": 1.0,
            "decay_rate": 0.95,
        },
        "time_encoding": dict(DEFAULT_TIME_ENCODING_CONFIG),
        "runtime": {
            "device": None,
            "seed": 42,
        },
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into base."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_workflow_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and merge with defaults."""
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Workflow config file not found: {config_file}")

    with config_file.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        raise ValueError("Workflow config must be a mapping at the top level.")

    merged = _deep_merge(get_default_workflow_config(), loaded)
    merged["__config_path__"] = str(config_file)
    return merged


def write_yaml(data: Mapping[str, Any], output_path: str) -> None:
    """Write mapping as YAML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _as_path_list(value: object) -> list[str]:
    """Normalize scalar/list values to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    raise ValueError(f"Expected path list or string, got: {type(value)}")


def _normalize_paths(paths: list[str], project_root: Path) -> list[str]:
    """Resolve path list against project root."""
    normalized: list[str] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        normalized.append(str(candidate.resolve()))
    return normalized


def _normalize_optional_path(path: str | None, project_root: Path) -> str | None:
    """Resolve an optional path against project root."""
    if path is None or not str(path).strip():
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return str(candidate.resolve())


def _classify_netcdf(path: Path) -> str | None:
    """Classify a NetCDF file by schema."""
    try:
        dataset = xr.open_dataset(path)
    except Exception:
        return None

    try:
        var_names = set(dataset.data_vars)
        coord_names = set(dataset.coords) | set(dataset.dims)

        tau_var_candidates = {"Residence time [yrs]", "tau"}
        if tau_var_candidates.intersection(var_names) and {"lat", "lev"}.issubset(coord_names):
            return "tau_paths"
        if {"AoA", "AoA_STD"}.issubset(var_names) and {"time", "lat", "alt"}.issubset(coord_names):
            return "sat_paths"
        insitu_required_vars = {
            "Mean_Age_SF6_corr",
            "Mean_Age_CO2",
            "Mean_Age_SF6_corr_STD",
            "Mean_Age_CO2_STD",
        }
        if insitu_required_vars.issubset(var_names) and {"season", "lat", "Altitude"}.issubset(set(dataset.dims)):
            return "insitu_paths"
        if "AOA" in var_names and {"time", "lat", "lev"}.issubset(coord_names):
            return "model_paths"
    finally:
        dataset.close()

    return None


def _has_tropopause_columns(path: Path) -> bool:
    """Check if a CSV file contains all required tropopause columns."""
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return REQUIRED_TROPOPAUSE_COLUMNS.issubset(set(header.columns))


def _discover_candidates(
    search_roots: list[str],
    project_root: Path,
) -> dict[str, list[str]]:
    """Discover and classify NetCDF/CSV files under search roots."""
    nc_files: list[Path] = []
    csv_files: list[Path] = []

    for root in search_roots:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        root_path = root_path.resolve()

        if not root_path.exists():
            continue

        nc_files.extend(sorted(root_path.rglob("*.nc")))
        csv_files.extend(sorted(root_path.rglob("*.csv")))

    nc_files = sorted({str(path.resolve()) for path in nc_files})
    csv_files = sorted({str(path.resolve()) for path in csv_files})

    discovered = {
        "sat_paths": [],
        "insitu_paths": [],
        "model_paths": [],
        "tau_paths": [],
        "csv_candidates": csv_files,
    }

    for file_path in nc_files:
        classification = _classify_netcdf(Path(file_path))
        if classification:
            discovered[classification].append(file_path)

    return discovered


def discover_data_files(
    search_roots: list[str],
    project_root: Path | None = None,
) -> dict[str, list[str]]:
    """Public discovery helper used by tests and workflow commands."""
    root = project_root or PROJECT_ROOT
    return _discover_candidates(search_roots, root)


def _resolve_list_paths(
    *,
    cli_value: object,
    yaml_value: object,
    discovered_values: list[str],
    allow_discovery: bool,
) -> tuple[list[str], str]:
    """Resolve list-type paths with precedence rules."""
    if cli_value is not None:
        return _as_path_list(cli_value), "cli_explicit"

    yaml_paths = _as_path_list(yaml_value)
    if yaml_paths:
        return yaml_paths, "yaml_explicit"

    if allow_discovery and discovered_values:
        return list(discovered_values), "autodiscovered"

    return [], "none"


def _resolve_single_path(
    *,
    cli_value: str | None,
    yaml_value: str | None,
    discovered_values: list[str],
    allow_discovery: bool,
    label: str,
) -> tuple[str | None, str]:
    """Resolve single-value paths with precedence rules."""
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value), "cli_explicit"
    if yaml_value is not None and str(yaml_value).strip():
        return str(yaml_value), "yaml_explicit"

    if allow_discovery:
        if len(discovered_values) == 1:
            return discovered_values[0], "autodiscovered"
        if len(discovered_values) > 1:
            joined = ", ".join(discovered_values)
            raise ValueError(
                f"Ambiguous {label} candidates discovered. "
                f"Set an explicit path. Candidates: {joined}"
            )

    return None, "none"


def _select_tropopause_csv(
    *,
    cli_value: str | None,
    yaml_value: str | None,
    discovered_csv_candidates: list[str],
    use_tropopause_features: bool,
    allow_discovery: bool,
    project_root: Path,
) -> tuple[str | None, str]:
    """Resolve tropopause CSV path with exact-path preference and fallback schema check."""
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value), "cli_explicit"
    if yaml_value is not None and str(yaml_value).strip():
        return str(yaml_value), "yaml_explicit"

    if not use_tropopause_features:
        return None, "not_required"

    if not allow_discovery:
        return None, "none"

    exact = (project_root / "data" / "tropopause" / "tropopause_features_monthly.csv").resolve()
    if exact.is_file():
        return str(exact), "autodiscovered_exact"

    for candidate in discovered_csv_candidates:
        path = Path(candidate)
        if _has_tropopause_columns(path):
            return str(path.resolve()), "autodiscovered_schema"

    return None, "none"


def _validate_existing_files(paths: list[str], label: str) -> None:
    """Validate list paths exist on disk."""
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files: {', '.join(missing)}")


def resolve_data_paths(
    config_data: Mapping[str, Any],
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> ResolvedDataPaths:
    """Resolve data paths using CLI > YAML > autodiscovery precedence."""
    root = (project_root or PROJECT_ROOT).resolve()
    overrides = dict(cli_overrides or {})

    autodiscover_cfg = dict(config_data.get("autodiscover", {}))
    autodiscover_cfg.setdefault("enabled", True)
    autodiscover_cfg.setdefault("search_roots", ["data"])
    autodiscover_cfg.setdefault("discover_model_paths", False)
    autodiscover_cfg.setdefault("discover_tau_path", False)

    if overrides.get("autodiscover_enabled") is not None:
        autodiscover_cfg["enabled"] = bool(overrides["autodiscover_enabled"])
    if overrides.get("search_roots") is not None:
        autodiscover_cfg["search_roots"] = list(overrides["search_roots"])
    if overrides.get("discover_model_paths") is not None:
        autodiscover_cfg["discover_model_paths"] = bool(overrides["discover_model_paths"])
    if overrides.get("discover_tau_path") is not None:
        autodiscover_cfg["discover_tau_path"] = bool(overrides["discover_tau_path"])

    search_roots = [str(path) for path in autodiscover_cfg.get("search_roots", ["data"])]
    autodiscover_enabled = bool(autodiscover_cfg.get("enabled", True))

    discovered = {
        "sat_paths": [],
        "insitu_paths": [],
        "model_paths": [],
        "tau_paths": [],
        "csv_candidates": [],
    }
    if autodiscover_enabled:
        discovered = _discover_candidates(search_roots, root)

    sat_paths_raw, sat_source = _resolve_list_paths(
        cli_value=overrides.get("sat_paths"),
        yaml_value=config_data.get("sat_paths"),
        discovered_values=discovered["sat_paths"],
        allow_discovery=autodiscover_enabled,
    )
    insitu_paths_raw, insitu_source = _resolve_list_paths(
        cli_value=overrides.get("insitu_paths"),
        yaml_value=config_data.get("insitu_paths"),
        discovered_values=discovered["insitu_paths"],
        allow_discovery=autodiscover_enabled,
    )
    model_paths_raw, model_source = _resolve_list_paths(
        cli_value=overrides.get("model_paths"),
        yaml_value=config_data.get("model_paths"),
        discovered_values=discovered["model_paths"],
        allow_discovery=autodiscover_enabled and bool(autodiscover_cfg.get("discover_model_paths", False)),
    )

    tau_path_raw, tau_source = _resolve_single_path(
        cli_value=overrides.get("tau_path"),
        yaml_value=config_data.get("tau_path"),
        discovered_values=discovered["tau_paths"],
        allow_discovery=autodiscover_enabled and bool(autodiscover_cfg.get("discover_tau_path", False)),
        label="tau_path",
    )

    use_tropopause_features = bool(config_data.get("use_tropopause_features", USE_TROPOPAUSE_FEATURES))
    tropopause_raw, tropopause_source = _select_tropopause_csv(
        cli_value=overrides.get("tropopause_csv"),
        yaml_value=config_data.get("tropopause_csv"),
        discovered_csv_candidates=discovered["csv_candidates"],
        use_tropopause_features=use_tropopause_features,
        allow_discovery=autodiscover_enabled,
        project_root=root,
    )

    sat_paths = _normalize_paths(sat_paths_raw, root)
    insitu_paths = _normalize_paths(insitu_paths_raw, root)
    model_paths = _normalize_paths(model_paths_raw, root)
    tau_path = _normalize_optional_path(tau_path_raw, root)
    tropopause_csv = _normalize_optional_path(tropopause_raw, root)

    _validate_existing_files(sat_paths, "satellite")
    _validate_existing_files(insitu_paths, "insitu")
    _validate_existing_files(model_paths, "model")
    if tau_path and not Path(tau_path).is_file():
        raise FileNotFoundError(f"Missing tau file: {tau_path}")
    if tropopause_csv and not Path(tropopause_csv).is_file():
        raise FileNotFoundError(f"Missing tropopause CSV file: {tropopause_csv}")

    if use_tropopause_features and tropopause_csv is None:
        raise ValueError(
            "use_tropopause_features is enabled but no tropopause CSV was resolved. "
            "Set data.tropopause_csv explicitly or place a valid CSV under search_roots."
        )

    if not (sat_paths or insitu_paths or model_paths):
        raise ValueError(
            "No input datasets resolved. Provide explicit paths or enable autodiscovery with valid files."
        )

    return ResolvedDataPaths(
        sat_paths=sat_paths,
        insitu_paths=insitu_paths,
        model_paths=model_paths,
        tau_path=tau_path,
        tropopause_csv=tropopause_csv,
        sources={
            "sat_paths": sat_source,
            "insitu_paths": insitu_source,
            "model_paths": model_source,
            "tau_path": tau_source,
            "tropopause_csv": tropopause_source,
        },
        discovered={
            "sat_paths": discovered["sat_paths"],
            "insitu_paths": discovered["insitu_paths"],
            "model_paths": discovered["model_paths"],
            "tau_paths": discovered["tau_paths"],
            "csv_candidates": discovered["csv_candidates"],
        },
    )
