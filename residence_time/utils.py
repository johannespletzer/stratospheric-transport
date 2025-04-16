def datetime64_to_year_fraction(t):
    """
    Convert datetime64[ns] array to fractional years.

    Parameters
    ----------
    t : np.ndarray or xarray.DataArray
        Datetime64 array.

    Returns
    -------
    np.ndarray
        Array of fractional years (e.g. 2004.04).
    """
    import pandas as pd

    t = pd.to_datetime(t)
    year = t.year
    start_of_year = pd.to_datetime(year.astype(str))
    start_of_next_year = pd.to_datetime((year + 1).astype(str))
    fraction = (t - start_of_year) / (start_of_next_year - start_of_year)

    return year + fraction.values
