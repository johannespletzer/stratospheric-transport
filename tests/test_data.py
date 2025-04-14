import os
import tempfile
import zipfile
import glob

import numpy as np
import pytest
import requests

from residence_time.data import load_satellite_dataset, load_insitu_dataset

ZENODO_URL = "https://zenodo.org/records/13906743/files/Age_Data_v2.zip?download=1"

@pytest.fixture(scope="module")
def test_data_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "Age_Data_v2.zip")

        # Download the zip file
        print("Downloading satellite and in-situ test data...")
        response = requests.get(ZENODO_URL)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Yield the directory containing extracted files
        yield tmpdir  # top-level temp dir (root of the extracted structure)

def test_load_satellite_dataset(test_data_dir):
    # Dynamically find the ACE satellite .nc file
    ace_files = glob.glob(os.path.join(test_data_dir, "**", "ACE-FTS*sinkcorr*.nc"), recursive=True)
    assert ace_files, "ACE file not found in extracted data"
    ace_path = ace_files[0]

    X, G, W = load_satellite_dataset(ace_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert not np.isnan(X).all()
    assert not (G == 0).all(), "Gamma should not be zero"

def test_load_insitu_dataset(test_data_dir):
    # Dynamically find the in-situ balloon .nc file
    insitu_files = glob.glob(os.path.join(test_data_dir, "**", "Binned_seasonal_Balloon.nc"), recursive=True)
    assert insitu_files, "In-situ balloon file not found in extracted data"
    insitu_path = insitu_files[0]

    X, G, W = load_insitu_dataset(insitu_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert (W >= 0).all(), "Weights must be non-negative"
