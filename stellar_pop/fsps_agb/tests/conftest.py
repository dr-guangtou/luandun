import numpy as np
import pytest


@pytest.fixture
def linear_spectrum():
    """A spectrum linear in wavelength: every pseudo-continuum index must be zero on it."""
    wave_a = np.linspace(3300.0, 22500.0, 40000)
    flux_lambda = 3.0 - 1e-4 * wave_a
    return wave_a, flux_lambda
