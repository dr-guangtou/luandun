import numpy as np

from csp_integrate import (
    agb_two_spectra,
    age_weights,
    csp_spectra,
    csp_track,
    epoch_weight_matrix,
    interpolate_log_z,
    surviving_mass_per_epoch,
)
from sfh_model import cumulative_mass, time_bin_edges
from ssp_grid import SspGrid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def test_age_weights_are_linear_in_log_age():
    lookback_gyr = np.array([10 ** (5.025 - 9), 1.0, 1e-6, 30.0])
    weights = age_weights(LOG_AGE, lookback_gyr)
    assert weights.shape == (4, LOG_AGE.size)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.isclose(weights[0, 0], 0.5) and np.isclose(weights[0, 1], 0.5)
    assert np.isclose(weights[1, np.argmin(np.abs(LOG_AGE - 9.0))], 1.0)
    assert np.isclose(weights[2, 0], 1.0)  # below 1e5 yr uses the youngest SSP
    assert np.isclose(weights[3, -1], 1.0)  # beyond the oldest SSP uses the oldest


def test_epoch_weight_matrix_uses_only_bins_before_each_epoch():
    edges = time_bin_edges()
    matrix = epoch_weight_matrix(edges, 3.0, 0.3, LOG_AGE)
    assert matrix.shape == (260, LOG_AGE.size)
    expected_mass = cumulative_mass(edges[1:], 3.0, 0.3)
    assert np.allclose(matrix.sum(axis=1), expected_mass, rtol=1e-10)
    assert np.all(matrix >= 0)


def test_interpolate_log_z_is_linear_between_grid_points():
    grid = np.array([-0.5, -0.25, 0.0, 0.25])
    flux = np.arange(4.0)[:, None, None] * np.ones((4, 3, 5))
    assert np.allclose(interpolate_log_z(flux, grid, 0.0), 2.0)
    assert np.allclose(interpolate_log_z(flux, grid, 0.125), 2.5)
    assert np.allclose(interpolate_log_z(flux, grid, -0.4), 0.4)


def test_csp_spectra_is_normalized_per_mass_formed():
    weights = np.array([[1.0, 3.0], [2.0, 2.0]])
    ssp = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    out = csp_spectra(weights, ssp)
    assert np.allclose(out[0], (1 * 1 + 3 * 2) / 4.0)
    assert np.allclose(out[1], 1.5)


def _constant_grid():
    wave = np.linspace(3400.0, 22000.0, 50)
    flux = np.ones((4, 2, LOG_AGE.size, wave.size))
    flux[:, 1] = 3.0
    return SspGrid(
        wave_a=wave,
        log_age_yr=LOG_AGE,
        log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=flux,
        native_sigma_km_s=np.full(wave.size, 42.4),
        product="native",
    )


def test_csp_track_of_constant_ssps_is_constant():
    grid = _constant_grid()
    epochs, flux, mass = csp_track(grid, log_z=0.1, agb_index=1, t_q_gyr=3.0, tau_q_gyr=0.3)
    assert epochs.shape == (260,) and np.isclose(epochs[-1], 13.0)
    assert flux.shape == (260, 50)
    assert np.allclose(flux, 3.0)
    assert np.all(np.diff(mass) >= 0)
    assert mass[-1] > mass[0]


def test_agb_two_spectra_is_linear_extrapolation():
    assert np.allclose(agb_two_spectra(np.array([1.0]), np.array([3.0])), 5.0)


def test_surviving_mass_per_epoch_weights_fraction_by_mass_at_each_age():
    weights = np.array([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0]])
    fraction = np.array([1.0, 0.5, 0.25])
    assert np.allclose(surviving_mass_per_epoch(weights, fraction), [2.5, 4.0])
