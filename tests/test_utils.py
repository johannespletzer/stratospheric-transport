import numpy as np

from residence_time.utils import (
    datetime64_to_year_fraction,
    year_fraction_to_datetime64,
)


def test_year_fraction_roundtrip() -> None:
    """Convert years to datetime and back and ensure values are preserved."""
    fractions = np.array([2000.0, 2000.5, 2001.0])
    dates = year_fraction_to_datetime64(fractions)
    result = datetime64_to_year_fraction(dates)
    assert np.allclose(result, fractions)
