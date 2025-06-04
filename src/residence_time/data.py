import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from aerocalc3 import std_atm
from scipy.interpolate import RegularGridInterpolator

from residence_time.config import USE_TROPOPAUSE_FEATURES
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
        Input array [N, 5] with columns [lat, alt, time=0, source_id=1, G].

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

    time = np.zeros_like(age)
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
    trop_features: bool = USE_TROPOPAUSE_FEATURES
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

    X_out = np.vstack(X_all)
    W_out = np.concatenate(W_all)

    if trop_features:
        X_out, W_tp = extend_with_tropopause_features(X_out)
        W_out = W_out * W_tp

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
    ds = xr.open_dataset(filename).ffill('lat')
    interp = RegularGridInterpolator(
        (ds['lev'].values, ds['lat'].values),
        ds['tau'].values,
        bounds_error=False,
        fill_value=np.nan
    )
    coords = np.stack([X_obs[:, 1], X_obs[:, 0]], axis=1)
    tau_R_interp = interp(coords)
    mask_valid = ~np.isnan(tau_R_interp)
    return tau_R_interp, mask_valid


def extend_with_tropopause_features(
    X: np.ndarray,
    csv_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend a [N, 5] array with tropopause features and compute weights.

    Parameters
    ----------
    X : np.ndarray
        Array [N, 5] with columns [lat, alt, time, source_id, G].

    csv_path : str, optional
        Path to the CSV with monthly tropopause features.

    Returns
    -------
    X_ext : np.ndarray
        Extended array [N, 8] with 3 new columns:
        [tp_WMO_tro, tp_WMO_sh_pol, tp_WMO_nh_pol].

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

    tp_values = df.set_index("year_frac")[features]
    tp_stds = df.set_index("year_frac")[stds]

    times = X[:, 2]
    sources = X[:, 3]

    X_ext = np.zeros((X.shape[0], 8))
    W = np.ones(X.shape[0])
    X_ext[:, :5] = X

    for i, (t, sid) in enumerate(zip(times, sources)):
        if sid != 0:
            X_ext[i, 5:] = 0.0
            W[i] = 1.0
        else:
            nearest_time = tp_values.index[np.abs(tp_values.index - t).argmin()]
            X_ext[i, 5:] = tp_values.loc[nearest_time].values
            std_vals = tp_stds.loc[nearest_time].values
            W[i] = 1.0 / (np.sum(std_vals**2) + 1e-8)

    return X_ext, W
