import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from aerocalc3 import std_atm
from scipy.interpolate import RegularGridInterpolator

from residence_time.config import INSITU_REFERENCE_YEAR, USE_TROPOPAUSE_FEATURES
from residence_time.utils import datetime64_to_year_fraction


def load_insitu_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load balloon/aircraft (in-situ) age-of-air observations from NetCDF.

    Prefers SF6-based estimates, falling back to CO2 where needed.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file with in-situ measurements.

    Returns
    -------
    X : np.ndarray
        Input array [N, 5] with columns [lat, alt, time=reference_year, source_id=1, G].

    Gamma : np.ndarray
        Mean age values (G), shape [N,].

    W : np.ndarray
        Inverse variance weights, shape [N,].

    """
    ds = xr.open_dataset(filepath).mean('season')

    lat = ds['lat'].values
    alt = ds['Altitude'].values

    sf6 = ds['Mean_Age_SF6_corr'].values
    co2 = ds['Mean_Age_CO2'].values
    std_sf6 = ds['Mean_Age_SF6_corr_STD'].values
    std_co2 = ds['Mean_Age_CO2_STD'].values

    lat_grid, alt_grid = np.meshgrid(lat, alt, indexing='ij')
    lat_flat, alt_flat = lat_grid.flatten(), alt_grid.flatten()
    sf6_flat, co2_flat = sf6.flatten(), co2.flatten()
    std_sf6_flat, std_co2_flat = std_sf6.flatten(), std_co2.flatten()

    use_sf6 = ~np.isnan(sf6_flat)
    use_co2 = ~use_sf6 & ~np.isnan(co2_flat)

    age = np.full_like(sf6_flat, np.nan)
    std = np.full_like(sf6_flat, np.nan)

    age[use_sf6] = sf6_flat[use_sf6]
    std[use_sf6] = std_sf6_flat[use_sf6]
    age[use_co2] = co2_flat[use_co2]
    std[use_co2] = std_co2_flat[use_co2]

    time = np.full_like(age, INSITU_REFERENCE_YEAR, dtype=float)
    source = np.ones_like(age)

    valid = ~np.isnan(age) & ~np.isnan(std)
    X = np.stack([lat_flat, alt_flat, time, source, age], axis=1)[valid]
    Gamma = age[valid]
    W = 1.0 / (std[valid]**2 + 1e-8)

    return X, Gamma, W


def load_satellite_dataset(filepath: str, time_range: Optional[Tuple[str, str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load satellite-based age-of-air observations from NetCDF.

    Parameters
    ----------
    filepath : str
        Path to satellite NetCDF file with AoA and AoA_STD variables.

    time_range : tuple of str, optional
        Optional start and end time range (e.g. ("2005", "2020")).

    Returns
    -------
    X : np.ndarray
        Input array [N, 5] with [lat, alt, time, source_id=0, AoA].

    Gamma : np.ndarray
        AoA values, shape [N,].

    W : np.ndarray
        Inverse variance weights, shape [N,].

    """
    ds = xr.open_dataset(filepath)
    ds = ds.where(ds.AoA >= 1e-5)

    if time_range:
        start, end = time_range
        ds = ds.sel(time=slice(start, end))

    lat = ds["lat"].values
    alt = ds["alt"].values
    time = datetime64_to_year_fraction(ds["time"].values)
    age = ds["AoA"].values
    std = ds["AoA_STD"].values
    sou = np.full_like(age, 0)

    time_grid, lat_grid, alt_grid = np.meshgrid(time, lat, alt, indexing='ij')
    lat_flat, alt_flat, time_flat = lat_grid.flatten(), alt_grid.flatten(), time_grid.flatten()
    age_flat, std_flat = age.flatten(), std.flatten()

    valid = (age_flat > 0) & (std_flat > 0) & ~np.isnan(age_flat) & ~np.isnan(std_flat)
    X = np.stack([lat_flat, alt_flat, time_flat, sou.flatten(), age_flat], axis=1)[valid]
    Gamma = age_flat[valid]
    W = 1.0 / (std_flat[valid]**2 + 1e-8)

    return X, Gamma, W


def load_model_dataset(filepath: str, time_range: Optional[Tuple[str, str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model-simulated mean age of air from NetCDF and convert pressure levels to altitude.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file with AOA(time, lat, lev) in pressure coordinates.

    time_range : tuple of str, optional
        Optional time range for filtering.

    Returns
    -------
    X : np.ndarray
        Input array [N, 5] with [lat, alt_km, time, source_id=2, AOA].

    Gamma : np.ndarray
        AOA values (target), shape [N,].

    W : np.ndarray
        Synthetic weights, shape [N,].

    """
    ds = xr.open_dataset(filepath)
    if time_range:
        ds = ds.sel(time=slice(*time_range))

    ds = ds.where(ds.AOA >= 1e-5).isel(lon=0).ffill('lev')
    alt_km = [std_atm.press2alt(p, press_units='pa', alt_units='km') for p in ds.lev.values]
    ds = ds.assign_coords(lev=alt_km)

    lat = ds["lat"].values
    alt = ds["lev"].values
    time = datetime64_to_year_fraction(ds["time"].values)
    age = np.transpose(ds["AOA"].values, (0, 2, 1))
    std = np.ones_like(age) * 0.5
    sou = np.full_like(age, 2)

    time_grid, lat_grid, alt_grid = np.meshgrid(time, lat, alt, indexing='ij')
    lat_flat, alt_flat, time_flat = lat_grid.flatten(), alt_grid.flatten(), time_grid.flatten()
    age_flat, std_flat = age.flatten(), std.flatten()

    valid = ~np.isnan(age_flat) & ~np.isnan(std_flat)
    X = np.stack([lat_flat, alt_flat, time_flat, sou.flatten(), age_flat], axis=1)[valid]
    Gamma = np.clip(age_flat[valid], 0., None)
    W = np.clip(1.0 / (std_flat[valid]**2 + 1e-8), 0, 1e3)

    return X, Gamma, W


def load_all_data_combined(
    sat_paths: Optional[List[str]] = None,
    insitu_paths: Optional[List[str]] = None,
    model_paths: Optional[List[str]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    trop_features: bool = USE_TROPOPAUSE_FEATURES,
    tropopause_csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and combine satellite, in-situ, and model datasets into unified arrays.

    Parameters
    ----------
    sat_paths : list of str, optional
        Paths to satellite NetCDF files.

    insitu_paths : list of str, optional
        Paths to in-situ NetCDF files.

    model_paths : list of str, optional
        Paths to model NetCDF files.

    time_range : tuple of str, optional
        Optional time range to apply to satellite/model data.

    trop_features : bool, optional
        Switch to extend satellite data with tropopause features.

    tropopause_csv_path : str, optional
        Optional explicit path for tropopause feature CSV.

    Returns
    -------
    X : np.ndarray
        Combined input features [N, 5].

    Gamma : np.ndarray
        Combined age values (targets), shape [N,].

    W : np.ndarray
        Combined weight values, shape [N,].

    """
    X_all, Gamma_all, W_all = [], [], []

    if sat_paths:
        for path in sat_paths:
            X, Gamma, W = load_satellite_dataset(path, time_range)
            X_all.append(X)
            Gamma_all.append(Gamma)
            W_all.append(W)

    if insitu_paths:
        for path in insitu_paths:
            X, Gamma, W = load_insitu_dataset(path)
            X_all.append(X)
            Gamma_all.append(Gamma)
            W_all.append(W)

    if model_paths:
        for path in model_paths:
            X, Gamma, W = load_model_dataset(path, time_range)
            X_all.append(X)
            Gamma_all.append(Gamma)
            W_all.append(W)

    if not X_all:
        raise ValueError("No input data provided. Set at least one of sat_paths, insitu_paths, or model_paths.")

    X_out = np.vstack(X_all)
    if trop_features:
        X_out, W_tp = extend_with_tropopause_features(X_out, csv_path=tropopause_csv_path)
        W_out = np.concatenate(W_all) * W_tp
    else:
        W_out = np.concatenate(W_all)

    return (
        X_out,
        np.concatenate(Gamma_all),
        W_out
    )


def load_tau_R(filename: str, X_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate t_R (residence time) from model grid to observation coordinates.

    Parameters
    ----------
    filename : str
        Path to NetCDF file with t(lev, lat).

    X_obs : np.ndarray
        Array [N, 5] containing [lat, alt, time, source_id, G].

    Returns
    -------
    tau_R_interp : np.ndarray
        Interpolated t_R values at X_obs locations.

    mask_valid : np.ndarray
        Boolean mask where interpolation was successful.

    """
    def _coerce_to_lev_lat(var: xr.DataArray) -> Optional[xr.DataArray]:
        if not {"lev", "lat"}.issubset(set(var.dims)):
            return None

        extra_dims = [dim for dim in var.dims if dim not in {"lev", "lat"}]
        for dim in extra_dims:
            if var.sizes.get(dim, 0) != 1:
                return None

        if extra_dims:
            var = var.isel({dim: 0 for dim in extra_dims})

        return var.transpose("lev", "lat")

    def _resolve_tau_variable(ds: xr.Dataset) -> xr.DataArray:
        preferred_names = (
            "tau",
            "tau_R",
            "t_R",
            "residence_time",
            "residence_time_mean",
        )

        for name in preferred_names:
            if name not in ds.data_vars:
                continue
            candidate = _coerce_to_lev_lat(ds[name])
            if candidate is not None:
                return candidate

        for _, var in ds.data_vars.items():
            candidate = _coerce_to_lev_lat(var)
            if candidate is not None:
                return candidate

        available = ", ".join(str(name) for name in ds.data_vars) or "<none>"
        raise KeyError(
            f"Could not find a tau-like variable in {filename}. "
            f"Expected a data variable with lev/lat dimensions; found: {available}"
        )

    with xr.open_dataset(filename) as ds:
        if "lat" in ds.dims:
            try:
                ds = ds.ffill("lat")
            except (ModuleNotFoundError, ImportError, RuntimeError):
                # xarray forward-fill can depend on optional bottleneck/numbagg.
                pass

        tau_var = _resolve_tau_variable(ds)
        interp = RegularGridInterpolator(
            (ds["lev"].values, ds["lat"].values),
            tau_var.values,
            bounds_error=False,
            fill_value=np.nan,
        )

    coords = np.stack([X_obs[:, 1], X_obs[:, 0]], axis=1)
    tau_R_interp = interp(coords)
    mask_valid = ~np.isnan(tau_R_interp)
    return tau_R_interp, mask_valid


def extend_with_tropopause_features(
    X: np.ndarray,
    csv_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend a [N, 5] array with tropopause features, an indicator, and weights.

    Satellite rows (source_id == 0) receive real tropopause values looked up
    by nearest time. Non-satellite rows receive the global mean of the
    tropopause lookup table as imputation so that after StandardScaler the
    imputed values map to approximately zero. A binary indicator column
    (1 = real, 0 = imputed) lets the network distinguish the two cases.

    Parameters
    ----------
    X : np.ndarray
        Array [N, 5] with columns [lat, alt, time, source_id, G].

    csv_path : str, optional
        Path to the CSV with monthly tropopause features.

    Returns
    -------
    X_ext : np.ndarray
        Extended array [N, 9] with 4 new columns:
        [tp_WMO_tro, tp_WMO_sh_pol, tp_WMO_nh_pol, tropopause_present].

    W : np.ndarray
        Weight vector based on inverse variance of tropopause features.

    """
    if csv_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "../../"))
        csv_path = os.path.join(project_root, "data", "tropopause", "tropopause_features_monthly.csv")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["year_frac"] = df["time"].dt.year + (df["time"].dt.month - 1) / 12

    features = ["tp_WMO_tro", "tp_WMO_sh_pol", "tp_WMO_nh_pol"]
    stds = ["tp_WMO_tro_std", "tp_WMO_sh_pol_std", "tp_WMO_nh_pol_std"]

    tp_df = (
        df[["year_frac", *features, *stds]]
        .sort_values("year_frac")
        .drop_duplicates(subset="year_frac", keep="last")
    )
    if tp_df.empty:
        raise ValueError("No tropopause features found in CSV.")

    tp_times = tp_df["year_frac"].to_numpy(dtype=float)
    tp_values = tp_df[features].to_numpy(dtype=float)
    tp_stds = tp_df[stds].to_numpy(dtype=float)

    times = X[:, 2]
    sources = X[:, 3]

    # Global mean of tropopause values for imputing non-satellite rows.
    global_tp_mean = tp_values.mean(axis=0)

    X_ext = np.zeros((X.shape[0], 9))
    W = np.ones(X.shape[0])
    X_ext[:, :5] = X

    # Default: mean-imputed tropopause values and indicator = 0.
    X_ext[:, 5:8] = global_tp_mean
    # X_ext[:, 8] already 0.0 (tropopause_present indicator)

    sat_mask = sources == 0
    if np.any(sat_mask):
        sat_times = times[sat_mask]
        right_idx = np.searchsorted(tp_times, sat_times, side="left")
        left_idx = np.clip(right_idx - 1, 0, len(tp_times) - 1)
        right_idx = np.clip(right_idx, 0, len(tp_times) - 1)

        # Preserve argmin tie behavior: equal distances choose the lower-time index.
        use_right = np.abs(tp_times[right_idx] - sat_times) < np.abs(sat_times - tp_times[left_idx])
        nearest_idx = np.where(use_right, right_idx, left_idx)

        X_ext[sat_mask, 5:8] = tp_values[nearest_idx]
        X_ext[sat_mask, 8] = 1.0  # indicator: real tropopause data
        sat_stds = tp_stds[nearest_idx]
        W[sat_mask] = 1.0 / (np.sum(sat_stds**2, axis=1) + 1e-8)

    return X_ext, W
