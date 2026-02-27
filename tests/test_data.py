import glob
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import requests

import residence_time.data as data_module
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
