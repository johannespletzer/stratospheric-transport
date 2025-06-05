# Running the test suite

The project uses `pytest` for automated testing. No external data downloads are
required because small NetCDF files are generated during setup.

Install the package and its dependencies first:

```bash
pip install -e .
```

Then simply run:

```bash
pytest -q
```

This will exercise data loading utilities, training helpers and plotting
functions using the temporary data created in `tests/conftest.py`.
