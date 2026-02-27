import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

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


def run_script_raw(script_path: str, args: List[str]) -> subprocess.CompletedProcess:
    """Run a script and return the CompletedProcess without return-code assertions."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script_path), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        env=env,
    )


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


def test_month_end_frequency_alias_falls_back_to_m(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use legacy month-end alias when pandas runtime does not support 'ME'."""
    module_path = PROJECT_ROOT / "scripts" / "process_tropopause_features.py"
    spec = importlib.util.spec_from_file_location("process_tropopause_features", module_path)
    assert spec is not None and spec.loader is not None
    process_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_script)

    frequencies_checked = []

    def fake_to_offset(freq: str):
        frequencies_checked.append(freq)
        if freq == "ME":
            raise ValueError("Invalid frequency: ME")
        return object()

    monkeypatch.setattr(process_script.pd.tseries.frequencies, "to_offset", fake_to_offset)

    assert process_script._month_end_frequency_alias() == "M"
    assert frequencies_checked == ["ME", "M"]


def test_train_model_rejects_disable_d_output_in_diffusivity_mode() -> None:
    """CLI must fail fast when D output is disabled in diffusivity mode."""
    result = run_script_raw("scripts/train_model.py", ["--disable-d-output"])
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "--disable-d-output is incompatible" in output


def test_download_and_process_era5_passes_strict_file_allow_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only budgeted files should be passed to concatenation."""
    module_path = PROJECT_ROOT / "scripts" / "download_and_process_era5.py"
    spec = importlib.util.spec_from_file_location("download_and_process_era5", module_path)
    assert spec is not None and spec.loader is not None
    era5_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(era5_script)

    captured = {}

    def fake_list_year_folders(base_url: str) -> List[str]:
        _ = base_url
        return ["2000", "2001"]

    def fake_list_files_in_folder(url: str) -> List[str]:
        if url.endswith("2000/"):
            return ["b.nc", "a.nc"]
        if url.endswith("2001/"):
            return ["d.nc", "c.nc"]
        return []

    def fake_download_files_multithreaded(
        file_urls: List[str], local_paths: List[str], max_workers: int = 8
    ) -> None:
        _ = (file_urls, local_paths, max_workers)

    class DummyDataset:
        def to_netcdf(self, output_path: str) -> None:
            captured["output_path"] = output_path

    def fake_concatenate_files(
        download_dir: str, allowed_years: List[str] = None, allowed_files: List[str] = None
    ) -> DummyDataset:
        _ = (download_dir, allowed_years)
        captured["allowed_files"] = list(allowed_files or [])
        return DummyDataset()

    monkeypatch.setattr(era5_script, "list_year_folders", fake_list_year_folders)
    monkeypatch.setattr(era5_script, "list_files_in_folder", fake_list_files_in_folder)
    monkeypatch.setattr(era5_script, "download_files_multithreaded", fake_download_files_multithreaded)
    monkeypatch.setattr(era5_script, "concatenate_files", fake_concatenate_files)

    era5_script.main(2000, 2001, str(tmp_path), file_limit=2)

    expected_paths = [
        os.path.join(str(tmp_path), "2000", "a.nc"),
        os.path.join(str(tmp_path), "2000", "b.nc"),
    ]
    assert captured["allowed_files"] == expected_paths
