import xarray as xr
import numpy as np

def load_insitu_dataset(filepath):
    ds = xr.open_dataset(filepath)
    ds = ds.mean('season')
    lat = ds['lat'].values
    alt = ds['Altitude'].values

    age_sf6 = ds['Mean_Age_SF6_corr'].values
    age_co2 = ds['Mean_Age_CO2'].values
    std_sf6 = ds['Mean_Age_SF6_corr_STD'].values
    std_co2 = ds['Mean_Age_CO2_STD'].values

    lat_grid, alt_grid = np.meshgrid(lat, alt, indexing='ij')
    lat_flat = lat_grid.flatten()
    alt_flat = alt_grid.flatten()

    age_sf6_flat = age_sf6.flatten()
    age_co2_flat = age_co2.flatten()
    std_sf6_flat = std_sf6.flatten()
    std_co2_flat = std_co2.flatten()

    use_sf6 = ~np.isnan(age_sf6_flat)
    use_co2 = ~use_sf6 & ~np.isnan(age_co2_flat)

    age_final = np.full_like(age_sf6_flat, np.nan)
    std_final = np.full_like(std_sf6_flat, np.nan)

    age_final[use_sf6] = age_sf6_flat[use_sf6]
    std_final[use_sf6] = std_sf6_flat[use_sf6]
    age_final[use_co2] = age_co2_flat[use_co2]
    std_final[use_co2] = std_co2_flat[use_co2]
    sou_final = np.full_like(age_final,1)      # source: 0 satellite, 1 in-situ, 2 model
    time_final = np.full_like(age_final,0)

    valid = ~np.isnan(age_final) & ~np.isnan(std_final)
    X = np.stack([lat_flat, alt_flat, time_final, sou_final, age_final], axis=1)[valid]
    Gamma = age_final[valid]
    W = 1.0 / (std_final[valid]**2 + 1e-8)

    return X, Gamma, W

def load_satellite_dataset(filepath):
    ds = xr.open_dataset(filepath)
    ds.where((ds.AoA>0e-2) & (ds.AoA < 1e3),drop=True)

    lat = ds["lat"].values
    alt = ds["alt"].values
    time = ds["time"].values.astype("float64") 
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

def load_model_dataset(filepath):

    from aerocalc3 import std_atm
    
    ds = xr.open_dataset(filepath)
    ds = ds.isel(lon=0).ffill('lev')
    km = [std_atm.press2alt(x,press_units='pa',alt_units='km') for x in ds.lev.values]
    ds = ds.assign_coords(lev=km)

    lat = ds["lat"].values
    alt = ds["lev"].values
    time = ds["time"].values.astype("float64") 
    age = ds["AOA"].values         # shape: (time, lat, alt)
    std = np.ones_like(age) * 0.5  # reliable 10-100, moderately reliable 1, uncertain 0.01-0.1
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
    
    std_safe = np.clip(std_flat[valid], 1e-2, None)
    W = 1.0 / (std_safe**2 + 1e-8)
    W = np.clip(W, 0, 1e3)

    return X, Gamma, W

def load_all_data_combined(sat_paths=[], insitu_paths=[], model_paths=[]):
    
    X_all, Gamma_all, W_all = [], [], []

    if sat_paths:
        for path in sat_paths:
            X, Gamma, W = load_satellite_dataset(path)
            X_all.append(X)
            Gamma_all.append(Gamma)
            W_all.append(W)


    if insitu_paths:
        for path in insitu_paths:
            X, Gamma, W = load_model_dataset(path)
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
    Load and interpolate τ_R (residence time) to observation (lat, alt) coordinates.

    Parameters:
    - filename: NetCDF file with tau(lev, lat)
    - X_obs: array with columns [lat, alt, ...]

    Returns:
    - tau_R_interp: interpolated τ_R values [N,]
    """

    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(filename).ffill('lat')

    alts = ds['lev'].values  # shape (24,)
    lats = ds['lat'].values  # shape (20,)
    tau = ds['tau'].values   # shape (24, 20)

    # Create interpolator: grid order must match dimensions in tau (lev, lat)
    interp = RegularGridInterpolator((alts, lats), tau, bounds_error=False, fill_value=np.nan)

    # Interpolate to (alt, lat) coordinates from X_obs
    coords = np.stack([X_obs[:, 1], X_obs[:, 0]], axis=1)  # [alt, lat]
    tau_R_interp = interp(coords)
    mask_valid = ~np.isnan(tau_R_interp)
    
    return tau_R_interp, mask_valid
