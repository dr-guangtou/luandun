"""Delayed-tau star formation followed by quenching, in four families.

Times are in Gyr since the start of star formation. The SFR is dimensionless
up to a constant; only relative masses matter for spectral indices.

All four families share the same delayed-tau rise, `SFR = (t / tau) exp(-t /
tau)` for `t < t_q`, with `tau = t_q` by default (`tau_gyr=None`); only the
`decoupled` family draws an independent rise timescale. They differ only in
the SFR after `t_q`:

- `exponential` (the original model): exponential decline with e-folding
  `tau_q`, continuous with the rise at `t_q`.
- `linear` (FSPS `sfh = 5` form): linear ramp to zero over `delta_q_gyr =
  2 ln(2) tau_q`, i.e. the same SFR half-life as the exponential family.
- `truncation` (FSPS `sfh = 4` with `sf_trunc`): hard cut to zero at `t_q`.
- `decoupled`: exponential decline as in `exponential`, but the rise
  timescale `tau` is independent of `t_q` (drawn separately).
"""

from functools import partial

import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm

TIME_STEP_GYR = 0.05
TIME_END_GYR = 13.0

T_Q_RANGE_GYR = (1.0, 5.9)
TAU_Q_RANGE_GYR = (0.1, 3.0)
LOG_Z_RANGE = (-0.5, 0.2)
LOG_Z_MEAN = 0.0
LOG_Z_SIGMA = 0.2

TAU_GYR_RANGE = (0.5, 5.0)

FAMILIES = ("exponential", "linear", "truncation", "decoupled")


def time_bin_edges(step_gyr=TIME_STEP_GYR, end_gyr=TIME_END_GYR):
    n_bins = int(round(end_gyr / step_gyr))
    return np.linspace(0.0, n_bins * step_gyr, n_bins + 1)


def _peak_rate(t_q_gyr, tau_gyr):
    """SFR of the delayed-tau rise, (t / tau) exp(-t / tau), evaluated at t = t_q."""
    return (t_q_gyr / tau_gyr) * np.exp(-t_q_gyr / tau_gyr)


def star_formation_rate(time_gyr, t_q_gyr, tau_q_gyr, tau_gyr=None):
    """The `exponential` (tau_gyr=None, i.e. tau=t_q) or `decoupled` (tau_gyr given)
    family: delayed-tau rise with e-folding `tau_gyr`, then exponential quenching
    with e-folding `tau_q_gyr`, continuous at `t_q_gyr` by construction."""
    tau_gyr = t_q_gyr if tau_gyr is None else tau_gyr
    time_gyr = np.asarray(time_gyr, dtype=float)
    forming = (time_gyr / tau_gyr) * np.exp(-time_gyr / tau_gyr)
    quenching = _peak_rate(t_q_gyr, tau_gyr) * np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr)
    return np.where(time_gyr < t_q_gyr, forming, quenching)


def linear_quench_star_formation_rate(time_gyr, t_q_gyr, delta_q_gyr):
    """The `linear` family: delayed-tau rise (tau = t_q), then a linear ramp to
    zero over `delta_q_gyr`."""
    time_gyr = np.asarray(time_gyr, dtype=float)
    forming = (time_gyr / t_q_gyr) * np.exp(-time_gyr / t_q_gyr)
    ramp = np.exp(-1.0) * np.clip(1.0 - (time_gyr - t_q_gyr) / delta_q_gyr, 0.0, None)
    return np.where(time_gyr < t_q_gyr, forming, ramp)


def truncated_star_formation_rate(time_gyr, t_q_gyr):
    """The `truncation` family: delayed-tau rise (tau = t_q), then a hard cut to
    zero at `t_q_gyr`."""
    time_gyr = np.asarray(time_gyr, dtype=float)
    forming = (time_gyr / t_q_gyr) * np.exp(-time_gyr / t_q_gyr)
    return np.where(time_gyr < t_q_gyr, forming, 0.0)


def _forming_cumulative_mass(time_gyr, tau_gyr):
    """Integral of (t / tau) exp(-t / tau) from 0 to time_gyr."""
    return tau_gyr - np.exp(-time_gyr / tau_gyr) * (time_gyr + tau_gyr)


def _quenching_cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr, rate_at_t_q):
    """Integral of rate_at_t_q * exp(-(t - t_q) / tau_q) from t_q to time_gyr."""
    return rate_at_t_q * tau_q_gyr * (1.0 - np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr))


def cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr, tau_gyr=None):
    """The `exponential` (tau_gyr=None) or `decoupled` (tau_gyr given) family's
    cumulative mass formed. `tau_gyr=None` reproduces the original tau = t_q
    behaviour exactly."""
    tau_gyr = t_q_gyr if tau_gyr is None else tau_gyr
    time_gyr = np.asarray(time_gyr, dtype=float)
    before = _forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), tau_gyr)
    rate_at_t_q = _peak_rate(t_q_gyr, tau_gyr)
    after = np.where(
        time_gyr > t_q_gyr,
        _quenching_cumulative_mass(np.maximum(time_gyr, t_q_gyr), t_q_gyr, tau_q_gyr, rate_at_t_q),
        0.0,
    )
    return before + after


def linear_quench_cumulative_mass(time_gyr, t_q_gyr, delta_q_gyr):
    """The `linear` family's cumulative mass formed: the delayed-tau rise mass
    formed by t_q, plus the integral of the linear ramp, `e^-1 * [(t - t_q) -
    (t - t_q)^2 / (2 delta_q)]`, clipped to the ramp's domain `[t_q, t_q +
    delta_q]` and held constant after it ends."""
    time_gyr = np.asarray(time_gyr, dtype=float)
    before = _forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), t_q_gyr)
    dt = np.clip(time_gyr - t_q_gyr, 0.0, delta_q_gyr)
    ramp_mass = np.exp(-1.0) * (dt - dt**2 / (2.0 * delta_q_gyr))
    return before + ramp_mass


def truncated_cumulative_mass(time_gyr, t_q_gyr):
    """The `truncation` family's cumulative mass formed: the delayed-tau rise mass
    formed by t_q, held constant after it (SFR = 0 for t >= t_q)."""
    time_gyr = np.asarray(time_gyr, dtype=float)
    return _forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), t_q_gyr)


def sfh_family_cumulative(family, t_q_gyr, tau_q_gyr, tau_gyr=None):
    """Return `cumulative_mass_fn(time_gyr)` for one of `FAMILIES`."""
    if family == "exponential":
        return partial(cumulative_mass, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr)
    if family == "linear":
        delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
        return partial(linear_quench_cumulative_mass, t_q_gyr=t_q_gyr, delta_q_gyr=delta_q_gyr)
    if family == "truncation":
        return partial(truncated_cumulative_mass, t_q_gyr=t_q_gyr)
    if family == "decoupled":
        if tau_gyr is None:
            raise ValueError("the decoupled family requires tau_gyr")
        return partial(cumulative_mass, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr, tau_gyr=tau_gyr)
    raise ValueError(f"unknown SFH family: {family!r}, expected one of {FAMILIES}")


def star_formation_rate_family(family, t_q_gyr, tau_q_gyr, tau_gyr=None):
    """Return `star_formation_rate_fn(time_gyr)` for one of `FAMILIES`, the SFR
    counterpart of `sfh_family_cumulative`."""
    if family == "exponential":
        return partial(star_formation_rate, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr)
    if family == "linear":
        delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
        return partial(linear_quench_star_formation_rate, t_q_gyr=t_q_gyr, delta_q_gyr=delta_q_gyr)
    if family == "truncation":
        return partial(truncated_star_formation_rate, t_q_gyr=t_q_gyr)
    if family == "decoupled":
        if tau_gyr is None:
            raise ValueError("the decoupled family requires tau_gyr")
        return partial(star_formation_rate, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr, tau_gyr=tau_gyr)
    raise ValueError(f"unknown SFH family: {family!r}, expected one of {FAMILIES}")


def burst_cumulative_mass(time_gyr, t_burst_gyr, width_gyr, mass_fraction, base_cumulative):
    """Cumulative mass of `base_cumulative` plus a Gaussian burst centred at `t_burst_gyr`
    with standard deviation `width_gyr` that adds `mass_fraction` of the base mass formed
    by TIME_END_GYR."""
    time_gyr = np.asarray(time_gyr, dtype=float)
    burst_mass = mass_fraction * base_cumulative(TIME_END_GYR)
    burst = 0.5 * burst_mass * (1.0 + erf((time_gyr - t_burst_gyr) / (np.sqrt(2.0) * width_gyr)))
    return base_cumulative(time_gyr) + burst


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


def draw_decoupled_tau(n_draws, seed=20260926):
    """The `decoupled` family's independent rise timescale, log-uniform on
    TAU_GYR_RANGE, drawn with its own seed (independent of `draw_population`)."""
    rng = np.random.default_rng(seed)
    log_tau = rng.uniform(np.log10(TAU_GYR_RANGE[0]), np.log10(TAU_GYR_RANGE[1]), size=n_draws)
    return 10.0**log_tau
