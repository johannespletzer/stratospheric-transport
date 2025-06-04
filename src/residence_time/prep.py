import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import netCDF4
import requests
import xarray as xr
from bs4 import BeautifulSoup


def list_year_folders(base_url: str) -> List[str]:
    """List subfolders at a URL that correspond to 4-digit year folders (e.g., '1980/').

    Parameters
    ----------
    base_url : str
        Base HTTP directory URL containing year-named folders.

    Returns
    -------
    List[str]
        List of year strings (e.g., ['1979', '1980']).

    """
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    folders = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('/') and href[:-1].isdigit():
            folders.append(href.strip('/'))
    return folders


def list_files_in_folder(url: str) -> List[str]:
    """List all .nc files in a given web directory.

    Parameters
    ----------
    url : str
        URL of the folder to list.

    Returns
    -------
    List[str]
        List of filenames ending in '.nc'.

    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.nc'):
            files.append(href)
    return files


def download_single_file(file_url: str, local_path: str) -> None:
    """Download a file from a URL to a local path, skipping if it already exists.

    Parameters
    ----------
    file_url : str
        URL of the file to download.

    local_path : str
        Local file path to save the download.

    Returns
    -------
    None

    """
    if not os.path.exists(local_path):
        try:
            with requests.get(file_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded: {os.path.basename(local_path)}")
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
    else:
        print(f"Already exists: {os.path.basename(local_path)}")


def download_files_multithreaded(
    file_urls: List[str],
    local_paths: List[str],
    max_workers: int = 8
) -> None:
    """Download multiple files concurrently using a thread pool.

    Parameters
    ----------
    file_urls : List[str]
        List of file URLs to download.

    local_paths : List[str]
        Corresponding local file paths to save each download.

    max_workers : int, default=8
        Number of concurrent download threads.

    Returns
    -------
    None

    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_single_file, file_url, local_path)
            for file_url, local_path in zip(file_urls, local_paths)
        ]
        for future in as_completed(futures):
            future.result()


def concatenate_files(download_dir: str, allowed_years: Optional[List[str]] = None) -> xr.Dataset:
    """Concatenate valid NetCDF files found in a directory into a single xarray.Dataset.

    Parameters
    ----------
    download_dir : str
        Base directory to search for .nc files.
    allowed_years : List[str], optional
        If provided, only include files from folders named with these years.

    Returns
    -------
    xr.Dataset
        Combined dataset containing the 'wmo_1st_p' variable from all files.

    Raises
    ------
    ValueError
        If no matching and valid .nc files are found.

    """
    dataset_files = []
    for root, _, files in os.walk(download_dir):
        if allowed_years is not None:
            year_dirname = os.path.basename(root)
            if year_dirname not in allowed_years:
                continue
        for file in files:
            if file.endswith(".nc"):
                dataset_files.append(os.path.join(root, file))

    if not dataset_files:
        raise ValueError(f"No NetCDF files found in {download_dir} with allowed_years={allowed_years}")

    print(f"Checking {len(dataset_files)} NetCDF files...")

    valid_files = []
    for path in dataset_files:
        try:
            with netCDF4.Dataset(path, "r") as ds:
                _ = ds.variables["wmo_1st_p"]  # test key variable is present
            valid_files.append(path)
        except Exception as e:
            print(f"Skipping invalid NetCDF: {path} ({e})")

    if not valid_files:
        raise ValueError("No valid NetCDF files with 'wmo_1st_p' found after filtering.")

    print(f"Opening {len(valid_files)} valid NetCDF files...")

    ds = xr.open_mfdataset(
        valid_files,
        combine="by_coords",
        parallel=True,
        chunks={},
        preprocess=lambda ds: ds[["wmo_1st_p"]],
    )

    return ds
