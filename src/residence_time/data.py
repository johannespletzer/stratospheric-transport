import os

import numpy as np
import pandas as pd
import xarray as xr
from aerocalc3 import std_atm
from scipy.interpolate import RegularGridInterpolator

from residence_time.utils import datetime64_to_year_fraction


def load_insitu_dataset(filepath):
    """Loads balloon/aircraft (in-situ) age-of-air observations from NetCDF.

    Prefers SF6 data where available, falls back to CO2 otherwise.

    Parameters
    ----------
    - filepath: path to NetCDF file with in-situ measurements

    Returns
    -------
    - X: [N, 5] array with columns [lat, alt, time, source_id=1, mean_age]
    - Gamma: [N,] mean age (target variable)
    - W: [N,] inverse variance weights

    """
    ds = xr.open_dataset(filepath)
    ds = ds.mean('season')

    lat = ds['lat'].values
    alt = ds['Altitude'].values

    sf6 = ds['Mean_Age_SF6_corr'].values
    co2 = ds['Mean_Age_CO2'].values
    std_sf6 = ds['Mean_Age_SF6_corr_STD'].values
    std_co2 = ds['Mean_Age_CO2_STD'].values

    lat_grid, alt_grid = np.meshgrid(lat, alt, indexing='ij')
    lat_flat = lat_grid.flatten()
    alt_flat = alt_grid.flatten()

    sf6_flat = sf6.flatten()
    co2_flat = co2.flatten()
    std_sf6_flat = std_sf6.flatten()
    std_co2_flat = std_co2.flatten()

    use_sf6 = ~np.isnan(sf6_flat)
    use_co2 = ~use_sf6 & ~np.isnan(co2_flat)

    age = np.full_like(sf6_flat, np.nan)
    std = np.full_like(std_sf6_flat, np.nan)

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


def load_satellite_dataset(filepath, time_range=None):
    """Loads satellite-based age-of-air observations from NetCDF (e.g. ACE-FTS or MIPAS).

    Parameters
    ----------
    - filepath: path to satellite NetCDF file with AoA and AoA_STD variables

    Returns
    -------
    - X: [N, 5] array with [lat, alt, time, source_id=0, AoA]
    - Gamma: [N,] AoA values (target variable)
    - W: [N,] inverse variance weights

    """
    ds = xr.open_dataset(filepath)
    ds = ds.where(ds.AoA>=1e-5)

    # Apply time slicing if requested
    if time_range is not None:
        start, end = time_range
        ds = ds.sel(time=slice(start, end))

    lat = ds["lat"].values
    alt = ds["alt"].values
    time = ds["time"].values
    time = datetime64_to_year_fraction(time)
    age = ds["AoA"].values
    std = ds["AoA_STD"].values
    sou = np.full_like(age,0)      # source: 0 satellite, 1 in-situ, 2 model

    # Create meshgrid
    time_grid, lat_grid, alt_grid = np.meshgrid(time, lat, alt, indexing='ij')
    lat_flat = lat_grid.flatten()
    alt_flat = alt_grid.flatten()
    time_flat = time_grid.flatten()

    # Flatten age and std
    age_flat = age.flatten()
    std_flat = std.flatten()
    sou_flat = sou.flatten()

    fillval = -999
    valid = (age_flat != fillval) & (std_flat != fillval) & ~np.isnan(age_flat) & ~np.isnan(std_flat)

    X = np.stack([lat_flat, alt_flat, time_flat, sou_flat, age_flat], axis=1)[valid]
    Gamma = age_flat[valid]
    W = 1.0 / (std_flat[valid]**2 + 1e-8)

    return X, Gamma, W


def load_model_dataset(filepath, time_range=None):
    """Loads model-simulated mean age of air from NetCDF and converts pressure levels to km.

    Parameters
    ----------
    - filepath: path to NetCDF file with AOA(time, lat, lev), lev in pressure units

    Returns
    -------
    - X: [N, 5] array with [lat, alt, time, source_id=2, AOA]
    - Gamma: [N,] AOA values (target variable)
    - W: [N,] synthetic uncertainty weights

    """
    ds = xr.open_dataset(filepath)
    if time_range is not None:
        ds = ds.sel(time=slice(*time_range))

    ds = ds.where(ds.AOA>=1e-5).isel(lon=0).ffill('lev')
    km = [std_atm.press2alt(x,press_units='pa',alt_units='km') for x in ds.lev.values]
    ds = ds.assign_coords(lev=km)

    lat = ds["lat"].values
    alt = ds["lev"].values
    time = ds["time"].values
    time = datetime64_to_year_fraction(time)
    age = ds["AOA"].values  
    age = np.transpose(age, (0, 2, 1))
    std = np.ones_like(age) * 0.5 
    sou = np.full_like(age,2)      # source: 0 satellite, 1 in-situ, 2 model
    
    # Create meshgrid
    time_grid, lat_grid, alt_grid = np.meshgrid(time, lat, alt, indexing='ij')
    lat_flat = lat_grid.flatten()
    alt_flat = alt_grid.flatten()
    time_flat = time_grid.flatten()

    # Flatten age and std
    age_flat = age.flatten()
    std_flat = std.flatten()
    sou_flat = sou.flatten()
    
    valid = ~np.isnan(age_flat) & ~np.isnan(std_flat)

    X = np.stack([lat_flat, alt_flat, time_flat, sou_flat, age_flat], axis=1)[valid]
    
    Gamma = np.clip(age_flat[valid], 0., None)
    
    W = 1.0 / (std_flat**2 + 1e-8)
    W = np.clip(W, 0, 1e3)

    return X, Gamma, W


def load_all_data_combined(
    sat_paths=None,
    insitu_paths=None,
    model_paths=None,
    time_range: tuple[str, str] = None
    ):
    """Loads and concatenates multiple datasets into one training-ready array.

    Parameters
    ----------
    - sat_paths: list of paths to satellite NetCDF files (optional)
    - insitu_paths: list of paths to in-situ NetCDF files (optional)
    - model_paths: list of paths to model output NetCDF files (optional)

    Returns
    -------
    - X: [N, 5] stacked array with [lat, alt, time, source, Gamma]
    - Gamma: [N,] target values
    - W: [N,] weights (1 / std²)

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

    return (
        np.vstack(X_all),
        np.concatenate(Gamma_all),
        np.concatenate(W_all)
    )


def load_tau_R(filename, X_obs):
    """Interpolates τ_R (residence time) from model grid to observation coordinates.

    Parameters
    ----------
    - filename: path to NetCDF with tau(lev, lat)
    - X_obs: [N, 5] array of observation inputs, must include [lat, alt] in columns 0, 1

    Returns
    -------
    - tau_R_interp: [N,] interpolated τ_R values
    - mask_valid: [N,] boolean mask where interpolation is valid (not NaN)

    """
    ds = xr.open_dataset(filename).ffill('lat')

    alts = ds['lev'].values
    lats = ds['lat'].values
    tau = ds['tau'].values

    interp = RegularGridInterpolator((alts, lats), tau, bounds_error=False, fill_value=np.nan)

    coords = np.stack([X_obs[:, 1], X_obs[:, 0]], axis=1)  # [alt, lat]
    tau_R_interp = interp(coords)
    mask_valid = ~np.isnan(tau_R_interp)

    return tau_R_interp, mask_valid

def extend_with_tropopause_features(X: np.ndarray, csv_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Extends a [N, 5] input array X with 3 tropopause features and returns an uncertainty weight vector W.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape [N, 5]: [lat, alt, time, source_id, Γ]
    csv_path : str or None
        Path to tropopause features CSV. If None, uses default in data/tropopause/.

    Returns
    -------
    X_ext : np.ndarray
        Extended input array of shape [N, 8] with appended features:
        [lat, alt, time, source_id, Γ, tp_WMO_tro, tp_WMO_sh_pol, tp_WMO_nh_pol]

    W : np.ndarray
        Uncertainty weights derived from 1 / (std² + ε) for the 3 added features.
        Shape: [N,]

    """
    if csv_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "../../"))
        csv_path = os.path.join(project_root, "data", "tropopause", "tropopause_features_monthly.csv")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["year_frac"] = df["time"].dt.year + (df["time"].dt.month - 1) / 12

    # Tropopause values and stds
    features = ["tp_WMO_tro", "tp_WMO_sh_pol", "tp_WMO_nh_pol"]
    stds = ["tp_WMO_tro_std", "tp_WMO_sh_pol_std", "tp_WMO_nh_pol_std"]

    tp_values = df.set_index("year_frac")[features]
    tp_stds = df.set_index("year_frac")[stds]

    times = X[:, 2]
    sources = X[:, 3]

    X_ext = np.zeros((X.shape[0], 8))
    W = np.ones(X.shape[0])  # default weights

    X_ext[:, :5] = X

    for i, (t, sid) in enumerate(zip(times, sources)):
        if sid != 0:
            X_ext[i, 5:] = 0.0
            W[i] = 1.0  # uniform default weight
        else:
            # Find nearest time match
            nearest_time = tp_values.index[np.abs(tp_values.index - t).argmin()]
            X_ext[i, 5:] = tp_values.loc[nearest_time].values

            std_vals = tp_stds.loc[nearest_time].values
            W[i] = 1.0 / (np.sum(std_vals**2) + 1e-8)  # combined inverse-variance weight

    return X_ext, W
