"""Composite stellar population spectra from cached SSP grids.

For an observation epoch t_obs, the SFH between t_obs and the start of star
formation is integrated on a log-spaced lookback sub-grid rather than on the
coarse 0.05 Gyr SFH bins: a 0.05 Gyr bin is too wide for stars younger than
about 100 Myr, whose SSP flux changes steeply with age. Each sub-bin's exact
mass (from the analytic cumulative SFH) is placed at its geometric-mean
lookback age, interpolated linearly in log age between the bracketing SSP
grid ages, the same kernel FSPS uses (sfh_weight.f90, interpolation_type =
0). Metallicity is interpolated linearly in log Z between grid SSPs, as
zcontinuous = 1 does.
"""

from functools import partial

import numpy as np

from sfh_model import cumulative_mass, time_bin_edges

LOOKBACK_STEP_DEX = 0.01
MIN_LOOKBACK_GYR = 1e-4


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


def _lookback_subgrid_edges(t_obs):
    """Lookback edges (Gyr) from 0 to t_obs: [0, MIN_LOOKBACK_GYR], then
    log-spaced at LOOKBACK_STEP_DEX up to (but excluding) t_obs, with t_obs
    appended as the final edge."""
    log_edges = np.log10(MIN_LOOKBACK_GYR) + LOOKBACK_STEP_DEX * np.arange(
        int(np.floor((np.log10(t_obs) - np.log10(MIN_LOOKBACK_GYR)) / LOOKBACK_STEP_DEX)) + 1
    )
    edges = 10.0**log_edges
    edges = edges[edges < t_obs]
    return np.concatenate([[0.0], edges, [t_obs]])


def epoch_weight_matrix_from_cumulative(edges_gyr, cumulative_mass_fn, log_age_grid_yr):
    """Epoch-by-SSP-age mass weights for any SFH given as its cumulative mass formed,
    `cumulative_mass_fn(time_gyr)`, vectorized over time."""
    n_epochs = edges_gyr.size - 1
    matrix = np.zeros((n_epochs, log_age_grid_yr.size))
    for k in range(n_epochs):
        t_obs = edges_gyr[k + 1]
        a_edges = _lookback_subgrid_edges(t_obs)
        a_lo, a_hi = a_edges[:-1], a_edges[1:]
        sub_masses = cumulative_mass_fn(t_obs - a_lo) - cumulative_mass_fn(t_obs - a_hi)
        a_rep = np.sqrt(a_lo * a_hi)
        a_rep[0] = 0.5 * MIN_LOOKBACK_GYR
        matrix[k] = sub_masses @ age_weights(log_age_grid_yr, a_rep)
    return matrix


def epoch_weight_matrix(edges_gyr, t_q_gyr, tau_q_gyr, log_age_grid_yr):
    return epoch_weight_matrix_from_cumulative(
        edges_gyr, partial(cumulative_mass, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr), log_age_grid_yr
    )


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


def surviving_mass_per_epoch(weight_matrix, fraction_by_age):
    return weight_matrix @ fraction_by_age


def agb_two_spectra(flux_agb0, flux_agb1):
    return 2.0 * flux_agb1 - flux_agb0


def csp_track(grid, log_z, agb_index, t_q_gyr, tau_q_gyr, edges_gyr=None):
    edges_gyr = time_bin_edges() if edges_gyr is None else edges_gyr
    weights = epoch_weight_matrix(edges_gyr, t_q_gyr, tau_q_gyr, grid.log_age_yr)
    ssp_flux = interpolate_log_z(grid.flux_nu[:, agb_index], grid.log_z_grid, log_z)
    return edges_gyr[1:], csp_spectra(weights, ssp_flux), weights.sum(axis=1)
