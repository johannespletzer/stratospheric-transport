"""Download and export ERA5 tropopause data to monthly CSV features.
The NetCDF file is downloaded into a temporary 'resources/' directory,
which is deleted after processing. The final CSV is saved into 'data/'.
"""

import os
import shutil

import numpy as np
import requests
import xarray as xr
from dask.distributed import Client

# ─────────────────────────────────────────────────────────────────────────────
# Dask Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_dask(ncpu: int = 10, jup_server_mem: int = 20, nworker: int = 1) -> Client:
    """Set up a Dask client for parallel processing.

    Args:
        ncpu (int): Total number of CPUs available.
        jup_server_mem (int): Memory in GB available for the server.
        nworker (int): Number of Dask workers.

    Returns:
        Client: Dask distributed client.

    """
    threads: int = ncpu // nworker
    mem_limit: float = jup_server_mem / nworker
    print(f"Setting up Dask: {nworker} worker(s), {threads} threads each, {mem_limit:.1f} GB per worker")

    return Client(
        processes=False,
        threads_per_worker=threads,
        n_workers=nworker,
        memory_limit=f"{mem_limit}GB"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def coefficient_of_variation(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(x, axis=axis) / np.mean(x, axis=axis)


def extract_features(ds: xr.Dataset) -> xr.Dataset:
    """Compute monthly-averaged tropopause features from ERA5 data.

    Args:
        ds (xr.Dataset): Original dataset with time, lat, lon dimensions.

    Returns:
        xr.Dataset: Feature dataset with monthly statistics.

    """
    ds_tp: xr.Dataset = xr.Dataset(coords=ds.coords)
    ds_tp["tp_WMO"] = ds["wmo_1st_p"]

    ds_tp_sel = ds_tp.mean('lon').resample(time='M').mean('time')
    ds_tp_sel["tp_WMO_std"] = ds_tp["tp_WMO"].mean('lon').resample(time='M').std('time')

    ds_tp_sel_nh = ds_tp_sel.where(ds_tp_sel.lat >= 0)
    ds_tp_sel_sh = ds_tp_sel.where(ds_tp_sel.lat <= 0)

    ds_tp_sel["tp_WMO_cv"] = ds_tp_sel.tp_WMO.reduce(coefficient_of_variation, dim='lat')
    ds_tp_sel["tp_WMO_minmax"] = ds_tp_sel.tp_WMO.max(dim='lat') - ds_tp_sel.tp_WMO.min(dim='lat')
    ds_tp_sel["tp_WMO_tro"] = ds_tp_sel.tp_WMO.min(dim='lat')
    ds_tp_sel["tp_WMO_tro_std"] = ds_tp_sel.tp_WMO_std.where(
        ds_tp_sel.tp_WMO <= ds_tp_sel.tp_WMO.min(dim='lat') * 1.05
    ).std(dim='lat')

    ds_tp_sel["tp_WMO_sh_pol"] = ds_tp_sel_sh.tp_WMO.max(dim='lat')
    ds_tp_sel["tp_WMO_sh_pol_std"] = ds_tp_sel_sh.tp_WMO_std.where(
        ds_tp_sel_sh.tp_WMO >= ds_tp_sel_sh.tp_WMO.max(dim='lat') * 0.9
    ).std(dim='lat')

    ds_tp_sel["tp_WMO_nh_pol"] = ds_tp_sel_nh.tp_WMO.max(dim='lat')
    ds_tp_sel["tp_WMO_nh_pol_std"] = ds_tp_sel_nh.tp_WMO_std.where(
        ds_tp_sel_nh.tp_WMO >= ds_tp_sel_nh.tp_WMO.max(dim='lat') * 0.9
    ).std(dim='lat')

    return ds_tp_sel


# ─────────────────────────────────────────────────────────────────────────────
# Download ERA5 NetCDF File
# ─────────────────────────────────────────────────────────────────────────────

def download_netcdf(output_dir: str) -> str:
    """Download the ERA5 tropopause NetCDF file.

    Args:
        output_dir (str): Directory to save the file.

    Returns:
        str: Path to the downloaded file.

    """
    url: str = "https://datapub.fz-juelich.de/slcs/tropopause/era5_tropopause_combined.nc"
    output_path: str = os.path.join(output_dir, "era5_tropopause_combined.nc")
    print(f"Downloading {url} → {output_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Main Routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Determine script directory
    script_dir: str = os.path.dirname(__file__)

    # Define resource and data directories
    resource_dir: str = os.path.join(script_dir, "resources")
    data_dir: str = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(resource_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Step 1: Download NetCDF to temporary resources directory
        nc_path: str = download_netcdf(resource_dir)

        # Step 2: Load and process dataset
        setup_dask()
        ds: xr.Dataset = xr.open_mfdataset(nc_path, chunks={"time": 10})
        ds_features: xr.Dataset = extract_features(ds)

        # Step 3: Export CSV with selected variables
        vars_to_export: list[str] = list(ds_features.data_vars)[2:]
        ds_out: xr.Dataset = ds_features[vars_to_export]
        out_path: str = os.path.join(data_dir, "tropopause_features_monthly.csv")
        ds_out.to_dataframe().to_csv(out_path)
        print(f"Saved features to: {out_path}")

    finally:
        # Step 4: Clean up downloaded NetCDF file
        print(f"Cleaning up temporary resources in {resource_dir}")
        shutil.rmtree(resource_dir)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
