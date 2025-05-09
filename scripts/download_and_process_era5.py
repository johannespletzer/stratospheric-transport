import argparse
import os

from residence_time.prep import (
    concatenate_files,
    download_files_multithreaded,
    list_files_in_folder,
    list_year_folders,
)

BASE_URL = "https://datapub.fz-juelich.de/slcs/tropopause/data/v1/era5low/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "../", "data", "tropopause")

def main(start_year: int, end_year: int, download_dir: str) -> None:
    """Download and concatenate ERA5 tropopause data (low resolution) from the Jülich SLCS server.
    
    This script:
    - Downloads NetCDF files by year (based on --start-year and --end-year),
    - Stores them in a specified directory (--output),
    - Concatenates them into a single dataset,
    - Saves the result as 'era5_tropopause_combined.nc' in that same directory.
    """
    os.makedirs(download_dir, exist_ok=True)
    output_file = os.path.join(download_dir, "era5_tropopause_combined.nc")

    # Get list of available years on server and filter by year range
    year_folders = list_year_folders(BASE_URL)
    selected_years = [y for y in year_folders if y.isdigit() and start_year <= int(y) <= end_year]

    total_years = len(selected_years)
    for i, year in enumerate(selected_years, start=1):
        print(f"\n[{i}/{total_years}] Processing year: {year} ({int((i / total_years) * 100)}% complete)")
        year_url = f"{BASE_URL}{year}/"
        year_dir = os.path.join(download_dir, year)
        os.makedirs(year_dir, exist_ok=True)
    
        nc_files = list_files_in_folder(year_url)
        file_urls = [f"{year_url}{file}" for file in nc_files]
        local_paths = [os.path.join(year_dir, file) for file in nc_files]
    
        # Filter out already-downloaded files
        filtered_file_urls = []
        filtered_local_paths = []
        for url, path in zip(file_urls, local_paths):
            if not os.path.exists(path):
                filtered_file_urls.append(url)
                filtered_local_paths.append(path)
            else:
                print(f"  Skipping existing file: {os.path.basename(path)}")
    
        if filtered_file_urls:
            print(f"  Downloading {len(filtered_file_urls)} new files...")
            download_files_multithreaded(filtered_file_urls, filtered_local_paths, max_workers=8)
        else:
            print("  All files already downloaded.")

    print("Concatenating files...")
    dataset = concatenate_files(download_dir, allowed_years=[str(y) for y in selected_years])

    print(f"Saving dataset to {output_file}")
    dataset.to_netcdf(output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process ERA5 tropopause data.")
    parser.add_argument("--start-year", type=int, required=True, help="Start year of data to download (e.g., 2000)")
    parser.add_argument("--end-year", type=int, required=True, help="End year of data to download (e.g., 2020)")
    parser.add_argument("--output", type=str, default=DEFAULT_DOWNLOAD_DIR, help="Directory to store downloaded files and output file")

    args = parser.parse_args()
    main(args.start_year, args.end_year, args.output)
