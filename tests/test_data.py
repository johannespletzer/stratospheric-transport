import glob
import os
from pathlib import Path
from typing import Union

import numpy as np

from residence_time.data import load_insitu_dataset, load_satellite_dataset


def test_load_satellite_dataset(sample_data_dir: Union[str, Path]) -> None:
    """Test loading a satellite dataset from a NetCDF file and validate its structure.

    Parameters
    ----------
    sample_data_dir : Union[str, Path]
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
    ace_files = glob.glob(os.path.join(sample_data_dir, "**", "ACE-FTS*sinkcorr*.nc"), recursive=True)
    assert ace_files, "ACE file not found in extracted data"
    ace_path = ace_files[0]

    X, G, W = load_satellite_dataset(ace_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert not np.isnan(X).all()
    assert not (G == 0).all(), "Gamma should not be zero"

def test_load_insitu_dataset(sample_data_dir: Union[str, Path]) -> None:
    """Test loading the in-situ balloon dataset from a NetCDF file and validate output shapes and weights.

    Parameters
    ----------
    sample_data_dir : Union[str, Path]
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
    insitu_files = glob.glob(os.path.join(sample_data_dir, "**", "Binned_seasonal_Balloon.nc"), recursive=True)
    assert insitu_files, "In-situ balloon file not found in extracted data"
    insitu_path = insitu_files[0]

    X, G, W = load_insitu_dataset(insitu_path)

    assert X.shape[1] == 5
    assert X.shape[0] == G.shape[0] == W.shape[0]
    assert (W >= 0).all(), "Weights must be non-negative"
