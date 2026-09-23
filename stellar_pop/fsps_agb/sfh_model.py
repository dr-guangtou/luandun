"""Delayed-tau star formation followed by exponential quenching, with tau = t_q.

Times are in Gyr since the start of star formation. The SFR is dimensionless
up to a constant; only relative masses matter for spectral indices.
"""

import numpy as np
from scipy.stats import truncnorm

TIME_STEP_GYR = 0.05
TIME_END_GYR = 13.0

T_Q_RANGE_GYR = (1.0, 5.9)
TAU_Q_RANGE_GYR = (0.1, 3.0)
LOG_Z_RANGE = (-0.5, 0.2)
LOG_Z_MEAN = 0.0
LOG_Z_SIGMA = 0.2


def time_bin_edges(step_gyr=TIME_STEP_GYR, end_gyr=TIME_END_GYR):
    n_bins = int(round(end_gyr / step_gyr))
    return np.linspace(0.0, n_bins * step_gyr, n_bins + 1)


def star_formation_rate(time_gyr, t_q_gyr, tau_q_gyr):
    time_gyr = np.asarray(time_gyr, dtype=float)
    forming = (time_gyr / t_q_gyr) * np.exp(-time_gyr / t_q_gyr)
    quenching = np.exp(-1.0) * np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr)
    return np.where(time_gyr < t_q_gyr, forming, quenching)


def _forming_cumulative_mass(time_gyr, t_q_gyr):
    """Integral of (t / t_q) exp(-t / t_q) from 0 to time_gyr."""
    return t_q_gyr - np.exp(-time_gyr / t_q_gyr) * (time_gyr + t_q_gyr)


def _quenching_cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr):
    """Integral of exp(-1) exp(-(t - t_q) / tau_q) from t_q to time_gyr."""
    return np.exp(-1.0) * tau_q_gyr * (1.0 - np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr))


def cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr):
    time_gyr = np.asarray(time_gyr, dtype=float)
    before = _forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), t_q_gyr)
    after = np.where(
        time_gyr > t_q_gyr,
        _quenching_cumulative_mass(np.maximum(time_gyr, t_q_gyr), t_q_gyr, tau_q_gyr),
        0.0,
    )
    return before + after


def bin_masses(edges_gyr, t_q_gyr, tau_q_gyr):
    cumulative = cumulative_mass(edges_gyr, t_q_gyr, tau_q_gyr)
    return np.diff(cumulative)


def draw_population(n_draws, seed):
    rng = np.random.default_rng(seed)
    t_q_gyr = rng.uniform(*T_Q_RANGE_GYR, size=n_draws)
    log_tau_q = rng.uniform(
        np.log10(TAU_Q_RANGE_GYR[0]), np.log10(TAU_Q_RANGE_GYR[1]), size=n_draws
    )
    lower = (LOG_Z_RANGE[0] - LOG_Z_MEAN) / LOG_Z_SIGMA
    upper = (LOG_Z_RANGE[1] - LOG_Z_MEAN) / LOG_Z_SIGMA
    log_z = truncnorm.rvs(
        lower, upper, loc=LOG_Z_MEAN, scale=LOG_Z_SIGMA, size=n_draws, random_state=rng
    )
    return {"t_q_gyr": t_q_gyr, "tau_q_gyr": 10.0**log_tau_q, "log_z": log_z}
