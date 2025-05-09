import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import xarray as xr
from bs4 import BeautifulSoup


def list_year_folders(base_url):
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    folders = []
    for link in soup.find_all('a'):
        href = link.get('href')
        # Only year folders like 1979/, 1980/
        if href and href.endswith('/') and href[:-1].isdigit():
            folders.append(href.strip('/'))
    return folders

def list_files_in_folder(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.nc'):
            files.append(href)
    return files

def download_single_file(file_url, local_path):
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

def download_files_multithreaded(file_urls, local_paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_url, local_path in zip(file_urls, local_paths):
            futures.append(executor.submit(download_single_file, file_url, local_path))
        for future in as_completed(futures):
            future.result()

def concatenate_files(download_dir, allowed_years=None):
    dataset_files = []
    for root, _, files in os.walk(download_dir):
        # Check if current folder matches one of the allowed years (if provided)
        if allowed_years is not None:
            year_dirname = os.path.basename(root)
            if year_dirname not in allowed_years:
                continue

        for file in files:
            if file.endswith(".nc"):
                dataset_files.append(os.path.join(root, file))
    print(f"Opening {len(dataset_files)} NetCDF files...")

    ds = xr.open_mfdataset(
        dataset_files,
        combine="by_coords",
        parallel=True,
        chunks={},
        preprocess=lambda ds: ds[['wmo_1st_p']]
    )

    return ds
