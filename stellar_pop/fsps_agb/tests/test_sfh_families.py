from functools import partial

import numpy as np
from scipy.integrate import quad

from sfh_model import (
    FAMILIES,
    TAU_GYR_RANGE,
    cumulative_mass,
    draw_decoupled_tau,
    linear_quench_cumulative_mass,
    sfh_family_cumulative,
    star_formation_rate,
    star_formation_rate_family,
    truncated_cumulative_mass,
)


def _old_forming_cumulative_mass(time_gyr, t_q_gyr):
    """The pre-generalization formula, kept independent of the implementation under
    test, to prove tau_gyr=None reproduces the old values exactly."""
    return t_q_gyr - np.exp(-time_gyr / t_q_gyr) * (time_gyr + t_q_gyr)


def _old_cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr):
    time_gyr = np.asarray(time_gyr, dtype=float)
    before = _old_forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), t_q_gyr)
    after = np.where(
        time_gyr > t_q_gyr,
        np.exp(-1.0)
        * tau_q_gyr
        * (1.0 - np.exp(-(np.maximum(time_gyr, t_q_gyr) - t_q_gyr) / tau_q_gyr)),
        0.0,
    )
    return before + after


def test_cumulative_mass_with_default_tau_reproduces_old_values():
    time_gyr = np.array([0.0, 0.5, 1.0, 2.9999, 3.0, 3.0001, 5.0, 13.0])
    for t_q_gyr, tau_q_gyr in ((3.0, 0.3), (1.5, 1.2), (5.9, 0.1)):
        expected = _old_cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr)
        result = cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr)
        assert np.array_equal(result, expected)
        result_explicit_none = cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr, tau_gyr=None)
        assert np.array_equal(result_explicit_none, expected)


def test_cumulative_mass_matches_quad_of_star_formation_rate_general_tau():
    t_q_gyr, tau_q_gyr, tau_gyr = 3.0, 0.4, 1.7
    edges = np.array([0.0, 1.0, 2.5, 2.9, 3.0, 3.1, 3.5, 5.0, 8.0, 13.0])
    cumulative = cumulative_mass(edges, t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    for i in range(edges.size - 1):
        expected, _ = quad(
            lambda t: star_formation_rate(np.array([t]), t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)[0],
            edges[i],
            edges[i + 1],
            points=[t_q_gyr] if edges[i] < t_q_gyr < edges[i + 1] else None,
        )
        assert np.isclose(cumulative[i + 1] - cumulative[i], expected, rtol=1e-7, atol=1e-10), i


def test_star_formation_rate_general_tau_is_continuous_at_t_q():
    t_q_gyr, tau_q_gyr, tau_gyr = 3.0, 0.4, 1.7
    before = star_formation_rate(np.array([t_q_gyr - 1e-9]), t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    after = star_formation_rate(np.array([t_q_gyr + 1e-9]), t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    assert np.isclose(before, after, rtol=1e-6)


def test_linear_quench_cumulative_mass_matches_quad_including_ramp_end():
    t_q_gyr, tau_q_gyr = 3.0, 0.3
    delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr

    def sfr(t):
        if t < t_q_gyr:
            return (t / t_q_gyr) * np.exp(-t / t_q_gyr)
        return np.exp(-1.0) * max(0.0, 1.0 - (t - t_q_gyr) / delta_q_gyr)

    edges = np.array(
        [
            0.0,
            1.0,
            2.9,
            3.0,
            3.1,
            t_q_gyr + delta_q_gyr - 0.02,
            t_q_gyr + delta_q_gyr + 0.05,
            8.0,
            13.0,
        ]
    )
    cumulative = linear_quench_cumulative_mass(edges, t_q_gyr, delta_q_gyr)
    for i in range(edges.size - 1):
        points = [p for p in (t_q_gyr, t_q_gyr + delta_q_gyr) if edges[i] < p < edges[i + 1]]
        expected, _ = quad(sfr, edges[i], edges[i + 1], points=points or None)
        assert np.isclose(cumulative[i + 1] - cumulative[i], expected, rtol=1e-7, atol=1e-10), i


def test_linear_quench_cumulative_mass_constant_after_ramp_end():
    t_q_gyr, tau_q_gyr = 3.0, 0.3
    delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
    end_value = linear_quench_cumulative_mass(
        np.array([t_q_gyr + delta_q_gyr]), t_q_gyr, delta_q_gyr
    )[0]
    later = linear_quench_cumulative_mass(
        np.array([t_q_gyr + delta_q_gyr + 1.0, 13.0]), t_q_gyr, delta_q_gyr
    )
    assert np.allclose(later, end_value)


def test_truncated_cumulative_mass_matches_quad():
    t_q_gyr = 3.0

    def sfr(t):
        return (t / t_q_gyr) * np.exp(-t / t_q_gyr) if t < t_q_gyr else 0.0

    edges = np.array([0.0, 1.0, 2.9, 3.0, 3.1, 8.0, 13.0])
    cumulative = truncated_cumulative_mass(edges, t_q_gyr)
    for i in range(edges.size - 1):
        points = [t_q_gyr] if edges[i] < t_q_gyr < edges[i + 1] else None
        expected, _ = quad(sfr, edges[i], edges[i + 1], points=points)
        assert np.isclose(cumulative[i + 1] - cumulative[i], expected, rtol=1e-7, atol=1e-10), i


def test_truncated_cumulative_mass_is_constant_after_t_q():
    t_q_gyr = 3.0
    at_t_q = truncated_cumulative_mass(np.array([t_q_gyr]), t_q_gyr)[0]
    later = truncated_cumulative_mass(np.array([t_q_gyr + 1.0, 13.0]), t_q_gyr)
    assert np.allclose(later, at_t_q)


def test_decoupled_family_is_continuous_at_t_q():
    t_q_gyr, tau_q_gyr, tau_gyr = 3.0, 0.4, 1.7
    cumulative_fn = sfh_family_cumulative("decoupled", t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    step = 1e-6
    derivative_before = (
        cumulative_fn(np.array([t_q_gyr])) - cumulative_fn(np.array([t_q_gyr - step]))
    ) / step
    derivative_after = (
        cumulative_fn(np.array([t_q_gyr + step])) - cumulative_fn(np.array([t_q_gyr]))
    ) / step
    assert np.isclose(derivative_before, derivative_after, rtol=1e-3)


def test_sfh_family_cumulative_dispatch_matches_dedicated_functions():
    t_q_gyr, tau_q_gyr, tau_gyr = 3.0, 0.3, 1.7
    time_gyr = np.array([0.5, 1.0, 2.9, 3.1, 5.0, 13.0])

    exponential = sfh_family_cumulative("exponential", t_q_gyr, tau_q_gyr)
    assert np.array_equal(exponential(time_gyr), cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr))

    delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
    linear = sfh_family_cumulative("linear", t_q_gyr, tau_q_gyr)
    assert np.array_equal(
        linear(time_gyr), linear_quench_cumulative_mass(time_gyr, t_q_gyr, delta_q_gyr)
    )

    truncation = sfh_family_cumulative("truncation", t_q_gyr, tau_q_gyr)
    assert np.array_equal(truncation(time_gyr), truncated_cumulative_mass(time_gyr, t_q_gyr))

    decoupled = sfh_family_cumulative("decoupled", t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    assert np.array_equal(
        decoupled(time_gyr), cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr, tau_gyr=tau_gyr)
    )


def test_sfh_family_cumulative_rejects_unknown_family():
    import pytest

    with pytest.raises(ValueError, match="family"):
        sfh_family_cumulative("bogus", 3.0, 0.3)


def test_sfh_family_cumulative_requires_tau_for_decoupled():
    import pytest

    with pytest.raises(ValueError, match="tau_gyr"):
        sfh_family_cumulative("decoupled", 3.0, 0.3)


def test_star_formation_rate_family_integrates_to_the_matching_cumulative_mass():
    t_q_gyr, tau_q_gyr, tau_gyr = 3.0, 0.3, 1.7
    for family in FAMILIES:
        kwargs = {"tau_gyr": tau_gyr} if family == "decoupled" else {}
        rate_fn = star_formation_rate_family(family, t_q_gyr, tau_q_gyr, **kwargs)
        cumulative_fn = sfh_family_cumulative(family, t_q_gyr, tau_q_gyr, **kwargs)
        lo, hi = 2.0, 4.0
        expected, _ = quad(
            lambda t, rate_fn=rate_fn: rate_fn(np.array([t]))[0], lo, hi, points=[t_q_gyr]
        )
        result = cumulative_fn(np.array([hi]))[0] - cumulative_fn(np.array([lo]))[0]
        assert np.isclose(result, expected, rtol=1e-6, atol=1e-9), family


def test_draw_decoupled_tau_respects_range_and_is_reproducible():
    tau = draw_decoupled_tau(20000, seed=20260926)
    assert tau.min() >= TAU_GYR_RANGE[0]
    assert tau.max() <= TAU_GYR_RANGE[1]
    log_tau = np.log10(tau)
    counts, _ = np.histogram(
        log_tau, bins=5, range=(np.log10(TAU_GYR_RANGE[0]), np.log10(TAU_GYR_RANGE[1]))
    )
    assert counts.max() / counts.min() < 1.2
    a = draw_decoupled_tau(10, seed=7)
    b = draw_decoupled_tau(10, seed=7)
    assert np.array_equal(a, b)


def test_epoch_weight_matrix_from_cumulative_accepts_family_callables():
    from csp_integrate import epoch_weight_matrix_from_cumulative
    from sfh_model import time_bin_edges

    log_age = np.round(np.arange(5.0, 10.3001, 0.05), 3)
    edges = time_bin_edges()
    for family in ("exponential", "linear", "truncation"):
        cumulative_fn = sfh_family_cumulative(family, 3.0, 0.3)
        matrix = epoch_weight_matrix_from_cumulative(edges, cumulative_fn, log_age)
        assert matrix.shape == (edges.size - 1, log_age.size)
        assert np.all(np.isfinite(matrix))


def test_partial_wrapping_still_works_for_backwards_compatible_calls():
    delayed_tau = partial(cumulative_mass, t_q_gyr=3.0, tau_q_gyr=0.3)
    assert np.isclose(
        delayed_tau(np.array([5.0]))[0], cumulative_mass(np.array([5.0]), 3.0, 0.3)[0]
    )
