"""Composite stellar population spectra from cached SSP grids.

For an observation epoch t_obs, every SFH bin with center t_i < t_obs
contributes its formed mass at lookback age t_obs - t_i. The SSP at that age
is interpolated linearly in log age between the bracketing grid ages, the
same kernel FSPS uses (sfh_weight.f90, interpolation_type = 0). Metallicity
is interpolated linearly in log Z between grid SSPs, as zcontinuous = 1 does.
"""

import numpy as np

from sfh_model import bin_masses, time_bin_edges

MIN_LOG_AGE_YR = 5.0


def age_weights(log_age_grid_yr, lookback_gyr):
    log_lookback = np.log10(np.maximum(np.asarray(lookback_gyr, dtype=float) * 1e9, 1.0))
    log_lookback = np.clip(log_lookback, log_age_grid_yr[0], log_age_grid_yr[-1])
    upper = np.clip(
        np.searchsorted(log_age_grid_yr, log_lookback, side="right"), 1, log_age_grid_yr.size - 1
    )
    lower = upper - 1
    fraction = (log_lookback - log_age_grid_yr[lower]) / (
        log_age_grid_yr[upper] - log_age_grid_yr[lower]
    )
    weights = np.zeros((log_lookback.size, log_age_grid_yr.size))
    rows = np.arange(log_lookback.size)
    weights[rows, lower] = 1.0 - fraction
    weights[rows, upper] += fraction
    return weights


def epoch_weight_matrix(edges_gyr, masses, log_age_grid_yr):
    centers = 0.5 * (edges_gyr[:-1] + edges_gyr[1:])
    n_epochs = edges_gyr.size - 1
    matrix = np.zeros((n_epochs, log_age_grid_yr.size))
    for k in range(n_epochs):
        t_obs = edges_gyr[k + 1]
        active = centers < t_obs
        lookback = t_obs - centers[active]
        matrix[k] = masses[active] @ age_weights(log_age_grid_yr, lookback)
    return matrix


def interpolate_log_z(flux_by_z, log_z_grid, log_z):
    if not log_z_grid[0] <= log_z <= log_z_grid[-1]:
        raise ValueError(f"log_z {log_z} outside grid [{log_z_grid[0]}, {log_z_grid[-1]}]")
    upper = int(np.clip(np.searchsorted(log_z_grid, log_z, side="right"), 1, log_z_grid.size - 1))
    lower = upper - 1
    fraction = (log_z - log_z_grid[lower]) / (log_z_grid[upper] - log_z_grid[lower])
    return (1.0 - fraction) * flux_by_z[lower] + fraction * flux_by_z[upper]


def csp_spectra(weight_matrix, ssp_flux):
    mass_formed = weight_matrix.sum(axis=1)
    return (weight_matrix @ ssp_flux) / mass_formed[:, None]


def agb_two_spectra(flux_agb0, flux_agb1):
    return 2.0 * flux_agb1 - flux_agb0


def csp_track(grid, log_z, agb_index, t_q_gyr, tau_q_gyr, edges_gyr=None):
    edges_gyr = time_bin_edges() if edges_gyr is None else edges_gyr
    masses = bin_masses(edges_gyr, t_q_gyr, tau_q_gyr)
    weights = epoch_weight_matrix(edges_gyr, masses, grid.log_age_yr)
    ssp_flux = interpolate_log_z(grid.flux_nu[:, agb_index], grid.log_z_grid, log_z)
    return edges_gyr[1:], csp_spectra(weights, ssp_flux), weights.sum(axis=1)
