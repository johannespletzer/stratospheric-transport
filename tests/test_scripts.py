import os
import subprocess
import tempfile


def run_script(script_path, args):
    """Helper to run a script and capture errors."""
    result = subprocess.run(
        ["python", script_path] + args,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"{script_path} failed"

def test_download_age_of_air_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = "scripts/download_age_of_air_data.py"  # or download_age_of_air.py if renamed
        run_script(script, ["--output", tmpdir])

        # Recursively check for .nc files
        found_nc = False
        for root, _, files in os.walk(tmpdir):
            if any(fname.endswith(".nc") for fname in files):
                found_nc = True
                break

        assert found_nc, "No .nc files found after extraction"

def test_download_and_process_era5():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = "scripts/download_and_process_era5.py"
        run_script(script, ["--start-year", "2000", "--end-year", "2000", "--output", tmpdir])
        combined_path = os.path.join(tmpdir, "era5_tropopause_combined.nc")
        assert os.path.exists(combined_path)

def test_process_tropopause_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        # First download ERA5 to get a valid input file
        era5_script = "scripts/download_and_process_era5.py"
        run_script(era5_script, ["--start-year", "2000", "--end-year", "2000", "--output", tmpdir])
        combined_file = os.path.join(tmpdir, "era5_tropopause_combined.nc")

        # Then run the processing script
        process_script = "scripts/process_tropopause_features.py"
        output_csv = os.path.join(tmpdir, "features.csv")
        run_script(process_script, ["--infile", combined_file, "--outfile", output_csv])
        assert os.path.exists(output_csv)
