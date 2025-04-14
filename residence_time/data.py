import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from aerocalc3 import std_atm


def _flatten_3d_fields(lat, alt, time, field, std, source_id):
    """
    Flattens 3D gridded data fields (lat, alt, time) into a 2D array of samples for modeling.

    Parameters:
    - lat, alt, time: 1D coordinate arrays
    - field: 3D array of the main variable (e.g. age of air)
    - std: 3D array of standard deviations for field
    - source_id: int, source label (0=satellite, 1=in-situ, 2=model)

    Returns:
    - X: [N, 5] array with columns [lat, alt, time, source_id, field]
    - Gamma: [N,] array with field values (target)
    - W: [N,] array of weights (1 / std²)
    """
    time_grid, lat_grid, alt_grid = np.meshgrid(time, lat, alt, indexing='ij')

    flat_X = np.stack([
        lat_grid.flatten(),
        alt_grid.flatten(),
        time_grid.flatten(),
        np.full_like(field.flatten(), source_id),
        field.flatten()
    ], axis=1)

    std_flat = std.flatten()
    valid = ~np.isnan(flat_X[:, 4]) & ~np.isnan(std_flat)

    Gamma = np.clip(flat_X[valid, 4], 0., None)
    std_safe = np.clip(std_flat[valid], 1e-2, None)
    W = 1.0 / (std_safe**2 + 1e-8)
    W = np.clip(W, 0, 1e3)

    return flat_X[valid], Gamma, W


def load_insitu_dataset(filepath):
    """
    Loads balloon/aircraft (in-situ) age-of-air observations from NetCDF.

    Prefers SF6 data where available, falls back to CO2 otherwise.

    Parameters:
    - filepath: path to NetCDF file with in-situ measurements

    Returns:
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


def load_satellite_dataset(filepath):
    """
    Loads satellite-based age-of-air observations from NetCDF (e.g. ACE-FTS or MIPAS).

    Parameters:
    - filepath: path to satellite NetCDF file with AoA and AoA_STD variables

    Returns:
    - X: [N, 5] array with [lat, alt, time, source_id=0, AoA]
    - Gamma: [N,] AoA values (target variable)
    - W: [N,] inverse variance weights
    """
    ds = xr.open_dataset(filepath)
    ds = ds.where((ds.AoA > 0e-2) & (ds.AoA < 1e3), drop=True)

    lat = ds["lat"].values
    alt = ds["alt"].values
    time = ds["time"].values.astype("float64")

    age = ds["AoA"].values
    std = ds["AoA_STD"].values

    return _flatten_3d_fields(lat, alt, time, age, std, source_id=0)


def load_model_dataset(filepath):
    """
    Loads model-simulated mean age of air from NetCDF and converts pressure levels to km.

    Parameters:
    - filepath: path to NetCDF file with AOA(time, lat, lev), lev in pressure units

    Returns:
    - X: [N, 5] array with [lat, alt, time, source_id=2, AOA]
    - Gamma: [N,] AOA values (target variable)
    - W: [N,] synthetic uncertainty weights
    """
    ds = xr.open_dataset(filepath)
    ds = ds.isel(lon=0).ffill('lev')

    km = [std_atm.press2alt(p, press_units='pa', alt_units='km') for p in ds.lev.values]
    ds = ds.assign_coords(lev=km)

    lat = ds["lat"].values
    alt = ds["lev"].values
    time = ds["time"].values.astype("float64")

    age = ds["AOA"].values
    std = np.ones_like(age) * 0.5

    return _flatten_3d_fields(lat, alt, time, age, std, source_id=2)


def load_all_data_combined(sat_paths=None, insitu_paths=None, model_paths=None):
    """
    Loads and concatenates multiple datasets into one training-ready array.

    Parameters:
    - sat_paths: list of paths to satellite NetCDF files (optional)
    - insitu_paths: list of paths to in-situ NetCDF files (optional)
    - model_paths: list of paths to model output NetCDF files (optional)

    Returns:
    - X: [N, 5] stacked array with [lat, alt, time, source, Gamma]
    - Gamma: [N,] target values
    - W: [N,] weights (1 / std²)
    """
    X_all, Gamma_all, W_all = [], [], []

    if sat_paths:
        for path in sat_paths:
            X, Gamma, W = load_satellite_dataset(path)
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
            X, Gamma, W = load_model_dataset(path)
            X_all.append(X)
            Gamma_all.append(Gamma)
            W_all.append(W)

    return (
        np.vstack(X_all),
        np.concatenate(Gamma_all),
        np.concatenate(W_all)
    )


def load_tau_R(filename, X_obs):
    """
    Interpolates τ_R (residence time) from model grid to observation coordinates.

    Parameters:
    - filename: path to NetCDF with tau(lev, lat)
    - X_obs: [N, 5] array of observation inputs, must include [lat, alt] in columns 0, 1

    Returns:
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
