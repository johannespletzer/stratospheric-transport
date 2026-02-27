import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ERA5_TEST_FILE_LIMIT = "1"
SCRIPT_TIMEOUT_SECONDS = 300


def run_script(script_path: str, args: List[str]) -> None:
    """Run a Python script with the given arguments and assert success.

    Parameters
    ----------
    script_path : str
        Path to the Python script to execute.

    args : List[str]
        List of command-line arguments to pass to the script.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the script returns a non-zero exit code.

    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script_path), *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"{script_path} timed out after {SCRIPT_TIMEOUT_SECONDS} seconds"
        ) from exc

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"{script_path} failed"

def test_download_age_of_air_data() -> None:
    """Download file and assert nc files were found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script = "scripts/download_age_of_air_data.py"
        run_script(script, ["--output", tmpdir])

        # Recursively check for .nc files
        found_nc = False
        for _, _, files in os.walk(tmpdir):
            if any(fname.endswith(".nc") for fname in files):
                found_nc = True
                break

        assert found_nc, "No .nc files found after extraction"

def test_download_and_process_era5() -> None:
    """Download era5 tropopause features and assert files were combined to a single nc file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script = "scripts/download_and_process_era5.py"
        run_script(
            script,
            [
                "--start-year",
                "2000",
                "--end-year",
                "2000",
                "--output",
                tmpdir,
                "--file-limit",
                ERA5_TEST_FILE_LIMIT,
            ],
        )
        combined_path = os.path.join(tmpdir, "era5_tropopause_combined.nc")
        assert os.path.exists(combined_path)

def test_process_tropopause_features() -> None:
    """Extract specific tropopause features on monthly basis and assert write to file works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First download ERA5 to get a valid input file
        era5_script = "scripts/download_and_process_era5.py"
        run_script(
            era5_script,
            [
                "--start-year",
                "2000",
                "--end-year",
                "2000",
                "--output",
                tmpdir,
                "--file-limit",
                ERA5_TEST_FILE_LIMIT,
            ],
        )
        combined_file = os.path.join(tmpdir, "era5_tropopause_combined.nc")

        # Then run the processing script
        process_script = "scripts/process_tropopause_features.py"
        output_csv = os.path.join(tmpdir, "features.csv")
        run_script(process_script, ["--infile", combined_file, "--outfile", output_csv])
        assert os.path.exists(output_csv)
