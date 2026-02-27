import glob
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import requests
import pandas as pd
import xarray as xr

import residence_time.data as data_module
from residence_time.config import INSITU_REFERENCE_YEAR
from residence_time.data import load_insitu_dataset, load_satellite_dataset

ZENODO_URL = "https://zenodo.org/records/13906743/files/Age_Data_v2.zip?download=1"

@pytest.fixture(scope="module")
def test_data_dir() -> str:
    """Create temporary directory, download and extract data and return the directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "Age_Data_v2.zip")

        print("Downloading satellite and in-situ test data...")
        response = requests.get(ZENODO_URL)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        yield tmpdir  # top-level temp dir (root of the extracted structure)

def test_load_satellite_dataset(test_data_dir: Union[str, Path]) -> None:
    """Test loading a satellite dataset from a NetCDF file and validate its structure.

    Parameters
    ----------
    test_data_dir : Union[str, Path]
        Path to the directory containing the extracted test data.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If expected file is missing, or if the output arrays are invalid.

    """
    # Dynamically find the ACE satellite .nc file
    ace_files = glob.glob(os.path.join(test_data_dir, "**", "ACE-FTS*sinkcorr*.nc"), recursive=True)
    assert ace_files, "ACE file not found in extracted data"
    ace_path = ace_files[0]

    X, G, W = load_satellite_dataset(ace_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert not np.isnan(X).all()
    assert not (G == 0).all(), "Gamma should not be zero"

def test_load_insitu_dataset(test_data_dir: Union[str, Path]) -> None:
    """Test loading the in-situ balloon dataset from a NetCDF file and validate output shapes and weights.

    Parameters
    ----------
    test_data_dir : Union[str, Path]
        Path to the directory containing the extracted test data.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the expected file is missing, or if the returned data is invalid.

    """
    # Dynamically find the in-situ balloon .nc file
    insitu_files = glob.glob(os.path.join(test_data_dir, "**", "Binned_seasonal_Balloon.nc"), recursive=True)
    assert insitu_files, "In-situ balloon file not found in extracted data"
    insitu_path = insitu_files[0]

    X, G, W = load_insitu_dataset(insitu_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert (W >= 0).all(), "Weights must be non-negative"


def test_load_all_data_combined_applies_tropopause_weighting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify tropopause extension returns expanded X and merged weights."""
    X_mock = np.array(
        [
            [10.0, 20.0, 2005.0, 0.0, 2.0],
            [15.0, 21.0, 2005.5, 0.0, 2.5],
        ]
    )
    Gamma_mock = np.array([2.0, 2.5])
    W_mock = np.array([4.0, 5.0])
    W_trop_mock = np.array([0.5, 2.0])

    def fake_load_satellite_dataset(filepath: str, time_range: Optional[Tuple[str, str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _ = (filepath, time_range)
        return X_mock, Gamma_mock, W_mock

    def fake_extend_with_tropopause_features(X: np.ndarray, csv_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        _ = csv_path
        X_ext = np.hstack([X, np.ones((X.shape[0], 3))])
        return X_ext, W_trop_mock

    monkeypatch.setattr(data_module, "load_satellite_dataset", fake_load_satellite_dataset)
    monkeypatch.setattr(data_module, "extend_with_tropopause_features", fake_extend_with_tropopause_features)

    X_out, Gamma_out, W_out = data_module.load_all_data_combined(
        sat_paths=["dummy.nc"],
        trop_features=True
    )

    assert X_out.shape == (2, 8)
    np.testing.assert_allclose(Gamma_out, Gamma_mock)
    np.testing.assert_allclose(W_out, W_mock * W_trop_mock)


def test_load_insitu_dataset_uses_reference_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-situ records should use configured reference year for the time column."""
    ds = xr.Dataset(
        data_vars={
            "Mean_Age_SF6_corr": (("season", "lat", "Altitude"), np.array([[[2.0]], [[2.2]]])),
            "Mean_Age_CO2": (("season", "lat", "Altitude"), np.array([[[2.1]], [[2.3]]])),
            "Mean_Age_SF6_corr_STD": (("season", "lat", "Altitude"), np.array([[[0.2]], [[0.2]]])),
            "Mean_Age_CO2_STD": (("season", "lat", "Altitude"), np.array([[[0.3]], [[0.3]]])),
        },
        coords={"season": [0, 1], "lat": [10.0], "Altitude": [20.0]},
    )

    def fake_open_dataset(filepath: str) -> xr.Dataset:
        _ = filepath
        return ds

    monkeypatch.setattr(data_module.xr, "open_dataset", fake_open_dataset)
    X, _, _ = data_module.load_insitu_dataset("dummy.nc")

    assert X.shape[1] == 5
    np.testing.assert_allclose(X[:, 2], INSITU_REFERENCE_YEAR)


def test_extend_with_tropopause_features_matches_reference_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectorized nearest-time lookup must match the original per-row behavior."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]),
            "tp_WMO_tro": [1.0, 10.0, 100.0],
            "tp_WMO_sh_pol": [2.0, 20.0, 200.0],
            "tp_WMO_nh_pol": [3.0, 30.0, 300.0],
            "tp_WMO_tro_std": [0.1, 0.2, 0.3],
            "tp_WMO_sh_pol_std": [0.1, 0.2, 0.3],
            "tp_WMO_nh_pol_std": [0.1, 0.2, 0.3],
        }
    )

    def fake_read_csv(csv_path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
        _ = (csv_path, parse_dates)
        return df.copy()

    monkeypatch.setattr(data_module.pd, "read_csv", fake_read_csv)

    X = np.array(
        [
            [0.0, 20.0, 2000.5, 0.0, 2.0],   # tie -> choose lower year (2000)
            [0.0, 20.0, 2001.2, 0.0, 2.0],   # nearest 2001
            [0.0, 20.0, 2001.8, 0.0, 2.0],   # nearest 2002
            [0.0, 20.0, 2001.0, 1.0, 2.0],   # non-satellite source
        ]
    )
    X_ext, W = data_module.extend_with_tropopause_features(X, csv_path="dummy.csv")

    df_ref = df.copy()
    df_ref["year_frac"] = df_ref["time"].dt.year + (df_ref["time"].dt.month - 1) / 12
    features = ["tp_WMO_tro", "tp_WMO_sh_pol", "tp_WMO_nh_pol"]
    stds = ["tp_WMO_tro_std", "tp_WMO_sh_pol_std", "tp_WMO_nh_pol_std"]
    tp_values = df_ref.set_index("year_frac")[features]
    tp_stds = df_ref.set_index("year_frac")[stds]

    expected_X = np.zeros((X.shape[0], 8))
    expected_W = np.ones(X.shape[0])
    expected_X[:, :5] = X
    for i, (t, sid) in enumerate(zip(X[:, 2], X[:, 3])):
        if sid == 0:
            nearest_time = tp_values.index[np.abs(tp_values.index - t).argmin()]
            expected_X[i, 5:] = tp_values.loc[nearest_time].values
            std_vals = tp_stds.loc[nearest_time].values
            expected_W[i] = 1.0 / (np.sum(std_vals**2) + 1e-8)

    np.testing.assert_allclose(X_ext, expected_X)
    np.testing.assert_allclose(W, expected_W)
