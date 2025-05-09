"""Download and extract Age of Air observational data from Zenodo.

Data Source:
https://zenodo.org/records/13906743

This script downloads the ZIP archive containing satellite and in-situ age-of-air measurements
and extracts its contents to a specified directory.
"""

import argparse
import os
import zipfile

import requests

ZENODO_URL = "https://zenodo.org/records/13906743/files/Age_Data_v2.zip?download=1"

def download_and_extract_age_data(output_dir: str) -> None:
    """Download and extract the Age of Air dataset from Zenodo.

    The function downloads a ZIP archive containing satellite and in-situ 
    observational age-of-air data, extracts its contents into the specified 
    output directory, and removes the ZIP file afterward.

    Parameters
    ----------
    output_dir : str
        Path to the directory where the ZIP file should be extracted.
        If the directory does not exist, it will be created.

    Returns
    -------
    None

    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "Age_Data_v2.zip")

    print(f"Downloading Age of Air data from Zenodo to {zip_path} ...")
    response = requests.get(ZENODO_URL)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)

    print("Extracting ZIP archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)  # clean up ZIP after extraction
    print(f"Done. Data extracted to {output_dir}")

def main() -> None:
    """Command-line interface for downloading and extracting Age of Air data.

    Parses the --output argument, determines the output directory (defaulting to 
    project-root/data/age_of_air), and downloads + extracts the Zenodo dataset 
    into that location using `download_and_extract_age_data`.

    Returns
    -------
    None

    """
    parser = argparse.ArgumentParser(description="Download Age of Air observational data from Zenodo.")
    
    # Default to project-root/data/aoa/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_output = os.path.join(project_root, "data", "age_of_air")

    parser.add_argument("--output", type=str, default=default_output,
                        help="Directory to extract the data to (default: data/aoa/)")

    args = parser.parse_args()
    download_and_extract_age_data(args.output)

if __name__ == "__main__":
    main()
