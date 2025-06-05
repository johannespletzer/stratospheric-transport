import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def create_satellite_nc(path: Path) -> None:
    """Write a minimal satellite NetCDF dataset."""
    time = [np.datetime64("2000-01-01")]
    lat = [0.0]
    alt = [10.0]
    ds = xr.Dataset(
        {
            "AoA": (("time", "lat", "alt"), np.array([[[1.0]]], dtype=np.float32)),
            "AoA_STD": (
                ("time", "lat", "alt"),
                np.array([[[0.1]]], dtype=np.float32),
            ),
        },
        coords={"time": time, "lat": lat, "alt": alt},
    )
    ds.to_netcdf(path)


def create_insitu_nc(path: Path) -> None:
    """Write a minimal in-situ NetCDF dataset."""
    ds = xr.Dataset(
        {
            "Mean_Age_SF6_corr": (
                ("season", "lat", "Altitude"),
                np.array([[[1.0]]], dtype=np.float32),
            ),
            "Mean_Age_CO2": (
                ("season", "lat", "Altitude"),
                np.array([[[1.5]]], dtype=np.float32),
            ),
            "Mean_Age_SF6_corr_STD": (
                ("season", "lat", "Altitude"),
                np.array([[[0.1]]], dtype=np.float32),
            ),
            "Mean_Age_CO2_STD": (
                ("season", "lat", "Altitude"),
                np.array([[[0.1]]], dtype=np.float32),
            ),
        },
        coords={"season": [0], "lat": [0.0], "Altitude": [10.0]},
    )
    ds.to_netcdf(path)


def create_era5_nc(path: Path) -> None:
    """Write a minimal ERA5 tropopause NetCDF dataset."""
    ds = xr.Dataset(
        {
            "wmo_1st_p": (
                ("time", "lat", "lon"),
                np.ones((1, 1, 1), dtype=np.float32),
            ),
        },
        coords={"time": [np.datetime64("2000-01-01")], "lat": [0], "lon": [0]},
    )
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def sample_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory containing generated satellite and in-situ datasets."""
    d = tmp_path_factory.mktemp("sample_data")
    create_satellite_nc(d / "ACE-FTS_sinkcorr_test.nc")
    create_insitu_nc(d / "Binned_seasonal_Balloon.nc")
    return d


@pytest.fixture(scope="module")
def sample_zip_bytes(sample_data_dir: Path) -> bytes:
    """Return the contents of a ZIP archive with the generated datasets."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname in ["ACE-FTS_sinkcorr_test.nc", "Binned_seasonal_Balloon.nc"]:
            zf.write(sample_data_dir / fname, arcname=fname)
    return buf.getvalue()
