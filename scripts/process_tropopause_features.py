"""
Post-process ERA5 tropopause data to extract monthly features for model training.

References:
- Hoffmann & Spang (2022), ACP, https://doi.org/10.5194/acp-22-4019-2022
- Zou et al. (2023), Front. Earth Sci, https://doi.org/10.3389/feart.2023.1177502
- Hoffmann & Spang (2021), Data Repository, https://doi.org/10.26165/JUELICH-DATA/UBNGI2

This script:
- Loads the concatenated ERA5 tropopause dataset
- Computes monthly means, standard deviations, and derived features
- Exports results as a CSV
"""

import os
import shutil
import argparse
import numpy as np
import xarray as xr
from dask.distributed import Client
from typing import Union

def coefficient_of_variation(x: Union[np.ndarray, xr.DataArray], axis: int = 0) -> Union[float, np.ndarray]:
    """Compute coefficient of variation along a given axis."""
    return np.std(x, axis=axis) / np.mean(x, axis=axis)

def main(infile: str, outfile: str):
    print(f"Loading data from {infile}")

    # Setup Dask
    ncpu = 10
    nworker = 1
    threads = ncpu // nworker
    mem_limit = 20 / nworker
    client = Client(
        processes=False,
        threads_per_worker=threads,
        n_workers=nworker,
        memory_limit=f"{mem_limit}GB"
    )
    print(f"Dask client set up with {threads} threads per worker.")

    ds = xr.open_mfdataset(infile, chunks={"time": 5})
    ds_tp = xr.Dataset(coords=ds.coords)
    ds_tp['tp_WMO'] = ds['wmo_1st_p']

    # Monthly resampling
    ds_tp_sel = ds_tp.mean('lon').resample(time='M').mean('time')
    ds_tp_sel['tp_WMO_std'] = ds_tp['tp_WMO'].mean('lon').resample(time='M').std('time')

    # Hemisphere splits
    ds_tp_sel_nh = ds_tp_sel.where(ds_tp_sel.lat >= 0.)
    ds_tp_sel_sh = ds_tp_sel.where(ds_tp_sel.lat <= 0.)

    # Derived features
    ds_tp_sel['tp_WMO_cv'] = ds_tp_sel.tp_WMO.reduce(coefficient_of_variation, dim='lat')
    ds_tp_sel['tp_WMO_minmax'] = ds_tp_sel.tp_WMO.max(dim='lat') - ds_tp_sel.tp_WMO.min(dim='lat')
    ds_tp_sel['tp_WMO_tro'] = ds_tp_sel.tp_WMO.min(dim='lat')
    ds_tp_sel['tp_WMO_tro_std'] = ds_tp_sel.tp_WMO_std.where(
        ds_tp_sel.tp_WMO <= ds_tp_sel.tp_WMO.min(dim='lat') * 1.05).std(dim='lat')

    ds_tp_sel['tp_WMO_sh_pol'] = ds_tp_sel_sh.tp_WMO.max(dim='lat')
    ds_tp_sel['tp_WMO_sh_pol_std'] = ds_tp_sel_sh.tp_WMO_std.where(
        ds_tp_sel_sh.tp_WMO >= ds_tp_sel_sh.tp_WMO.max(dim='lat') * 0.9).std(dim='lat')

    ds_tp_sel['tp_WMO_nh_pol'] = ds_tp_sel_nh.tp_WMO.max(dim='lat')
    ds_tp_sel['tp_WMO_nh_pol_std'] = ds_tp_sel_nh.tp_WMO_std.where(
        ds_tp_sel_nh.tp_WMO >= ds_tp_sel_nh.tp_WMO.max(dim='lat') * 0.9).std(dim='lat')

    # Output only final features
    features = [v for v in list(ds_tp_sel.data_vars)[2:]]
    ds_out = ds_tp_sel[features]

    print(f"Exporting to {outfile}")
    ds_out.to_dataframe().to_csv(outfile)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract monthly tropopause features from ERA5 data.")
    parser.add_argument("--infile", type=str, required=True, help="Path to concatenated ERA5 NetCDF file.")
    parser.add_argument("--outfile", type=str, default=None, help="Path to output CSV file.")

    args = parser.parse_args()
    if args.outfile is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../"))
        data_dir = os.path.join(project_root, "data", "tropopause")
        os.makedirs(data_dir, exist_ok=True)
        args.outfile = os.path.join(data_dir, "tropopause_features_monthly.csv")

    main(args.infile, args.outfile)
