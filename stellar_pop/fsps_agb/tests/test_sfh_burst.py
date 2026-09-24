from functools import partial

import numpy as np

from csp_integrate import epoch_weight_matrix, epoch_weight_matrix_from_cumulative
from sfh_model import TIME_END_GYR, burst_cumulative_mass, cumulative_mass, time_bin_edges

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)
NO_QUENCH_TAU_Q_GYR = 1e6


def _base(time_gyr):
    return cumulative_mass(time_gyr, 3.0, NO_QUENCH_TAU_Q_GYR)


def test_burst_adds_mass_fraction_of_base_final_mass():
    total = burst_cumulative_mass(TIME_END_GYR, 4.0, 0.1, 0.1, _base)
    added = total - _base(TIME_END_GYR)
    assert abs(added - 0.1 * _base(TIME_END_GYR)) < 1e-9


def test_burst_rate_at_burst_centre_equals_gaussian_peak():
    step = 1e-5
    t_burst, width, fraction = 4.0, 0.1, 0.1

    def added(time_gyr):
        return burst_cumulative_mass(time_gyr, t_burst, width, fraction, _base) - _base(time_gyr)

    derivative = (added(t_burst + step) - added(t_burst - step)) / (2.0 * step)
    peak = fraction * _base(TIME_END_GYR) / (np.sqrt(2.0 * np.pi) * width)
    assert abs(derivative - peak) < 1e-6


def test_burst_is_negligible_well_before_the_burst():
    time_gyr = np.array([0.5, 1.0, 3.0])
    total = burst_cumulative_mass(time_gyr, 4.0, 0.1, 0.1, _base)
    assert np.allclose(total, _base(time_gyr), rtol=0.0, atol=1e-12)


def test_epoch_weight_matrix_from_cumulative_matches_delayed_tau():
    edges = time_bin_edges()
    expected = epoch_weight_matrix(edges, 3.0, 0.3, LOG_AGE)
    delayed_tau = partial(cumulative_mass, t_q_gyr=3.0, tau_q_gyr=0.3)
    result = epoch_weight_matrix_from_cumulative(edges, delayed_tau, LOG_AGE)
    assert np.allclose(result, expected, rtol=0.0, atol=0.0)
