import os
import sys
import tempfile
import types
from pathlib import Path
from typing import List
from unittest import mock

from conftest import create_era5_nc  # helper to build NetCDF

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Provide dummy dask.distributed if dependency is absent
if "dask.distributed" not in sys.modules:
    dummy = types.ModuleType("dask.distributed")
    dummy.Client = lambda *_, **__: None
    sys.modules["dask.distributed"] = dummy

from download_age_of_air_data import download_and_extract_age_data  # noqa: E402
from download_and_process_era5 import main as era5_main  # noqa: E402
from process_tropopause_features import main as process_main  # noqa: E402


def test_download_age_of_air_data(sample_zip_bytes: bytes) -> None:
    """Extract zipped archive from fixture and ensure .nc files are present."""

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:  # pragma: no cover - no failure
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(
            "download_age_of_air_data.requests.get",
            return_value=DummyResponse(sample_zip_bytes),
        ):
            download_and_extract_age_data(tmpdir)

        found_nc = any(p.suffix == ".nc" for p in Path(tmpdir).rglob("*.nc"))
        assert found_nc, "No .nc files found after extraction"


def test_download_and_process_era5() -> None:
    """Combine generated ERA5 files into a single NetCDF."""

    def dummy_download(urls: List[str], paths: List[str], max_workers: int = 8) -> None:
        for path in paths:
            create_era5_nc(Path(path))

    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(
            "download_and_process_era5.list_year_folders",
            return_value=["2000"],
        ), mock.patch(
            "download_and_process_era5.list_files_in_folder",
            return_value=["era5_sample.nc"],
        ), mock.patch(
            "download_and_process_era5.download_files_multithreaded",
            side_effect=dummy_download,
        ):
            era5_main(2000, 2000, tmpdir, file_limit=1)

        combined_path = os.path.join(tmpdir, "era5_tropopause_combined.nc")
        assert os.path.exists(combined_path)


def test_process_tropopause_features() -> None:
    """Process ERA5 sample file and ensure CSV is written."""
    with tempfile.TemporaryDirectory() as tmpdir:
        combined_file = os.path.join(tmpdir, "era5_tropopause_combined.nc")
        create_era5_nc(Path(combined_file))

        output_csv = os.path.join(tmpdir, "features.csv")
        process_main(combined_file, output_csv)
        assert os.path.exists(output_csv)
