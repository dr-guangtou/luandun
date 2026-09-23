import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

from sfh_model import bin_masses, draw_population, star_formation_rate, time_bin_edges


def test_time_bin_edges_default_grid():
    edges = time_bin_edges()
    assert edges.shape == (261,)
    assert edges[0] == 0.0
    assert np.isclose(edges[-1], 13.0)
    assert np.allclose(np.diff(edges), 0.05)


def test_star_formation_rate_is_continuous_at_t_q():
    t_q, tau_q = 3.0, 0.3
    before = star_formation_rate(np.array([t_q - 1e-9]), t_q, tau_q)
    after = star_formation_rate(np.array([t_q + 1e-9]), t_q, tau_q)
    assert np.isclose(before, after, rtol=1e-6)
    assert np.isclose(after, np.exp(-1.0), rtol=1e-6)


def test_bin_masses_match_numerical_integration():
    t_q, tau_q = 3.0, 0.3
    edges = time_bin_edges()
    masses = bin_masses(edges, t_q, tau_q)
    assert masses.shape == (260,)
    for i in (0, 10, 59, 60, 61, 100, 259):
        expected, _ = quad(
            lambda t: star_formation_rate(np.array([t]), t_q, tau_q)[0],
            edges[i],
            edges[i + 1],
            points=[t_q],
        )
        assert np.isclose(masses[i], expected, rtol=1e-8), i


def test_bin_masses_handle_bin_straddling_t_q():
    t_q, tau_q = 3.02, 0.5
    edges = time_bin_edges()
    masses = bin_masses(edges, t_q, tau_q)
    expected, _ = quad(
        lambda t: star_formation_rate(np.array([t]), t_q, tau_q)[0], 3.0, 3.05, points=[t_q]
    )
    assert np.isclose(masses[60], expected, rtol=1e-8)


def test_draw_population_respects_priors():
    draws = draw_population(20000, seed=1)
    assert set(draws) == {"t_q_gyr", "tau_q_gyr", "log_z"}
    assert draws["t_q_gyr"].min() >= 1.0 and draws["t_q_gyr"].max() <= 5.9
    assert draws["tau_q_gyr"].min() >= 0.1 and draws["tau_q_gyr"].max() <= 3.0
    assert draws["log_z"].min() >= -0.5 and draws["log_z"].max() <= 0.2
    log_tau = np.log10(draws["tau_q_gyr"])
    counts, _ = np.histogram(log_tau, bins=5, range=(-1.0, np.log10(3.0)))
    assert counts.max() / counts.min() < 1.2
    expected_mean = truncnorm.mean((-0.5 - 0.0) / 0.2, (0.2 - 0.0) / 0.2, loc=0.0, scale=0.2)
    assert abs(np.mean(draws["log_z"]) - expected_mean) < 0.01


def test_draw_population_is_reproducible():
    a = draw_population(10, seed=5)
    b = draw_population(10, seed=5)
    assert np.array_equal(a["t_q_gyr"], b["t_q_gyr"])
