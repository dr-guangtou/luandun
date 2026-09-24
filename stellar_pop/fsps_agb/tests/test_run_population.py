from functools import partial

import numpy as np
import pytest

from run_population import (
    OUTPUT_DIR,
    _default_out_dir,
    _history_indices,
    population_indices,
    specific_sfr_windows,
)
from sfh_model import FAMILIES, cumulative_mass, draw_decoupled_tau, draw_population, time_bin_edges
from ssp_grid import SspGrid, load_ssp_grid, load_surviving_mass, save_ssp_grid, save_surviving_mass

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def _write_toy_surviving_mass(grid_dir, fraction_value=1.0):
    fraction = np.full((4, LOG_AGE.size), fraction_value)
    save_surviving_mass(grid_dir, LOG_AGE, np.array([-0.5, -0.25, 0.0, 0.25]), fraction)
    return load_surviving_mass(grid_dir)[2]


def _toy_grids(grid_dir, fraction_value=1.0):
    fraction = _write_toy_surviving_mass(grid_dir, fraction_value)
    wave = np.linspace(3400.0, 22000.0, 6000)
    base = np.ones((4, 2, LOG_AGE.size, wave.size)) * (1.0 + 1e-5 * wave) * wave**2
    grids = {}
    for product, lo, hi in (("sigma300", 3400.0, 22000.0), ("r100", 12500.0, 21000.0)):
        keep = (wave >= lo) & (wave <= hi)
        grids[product] = SspGrid(
            wave_a=wave[keep],
            log_age_yr=LOG_AGE,
            log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
            agb_weights=np.array([0.0, 1.0]),
            flux_nu=base[..., keep],
            native_sigma_km_s=np.full(keep.sum(), 42.4),
            product=product,
            surviving_mass_fraction=fraction,
        )
    return grids


def test_specific_sfr_windows_for_constant_sfr():
    edges = time_bin_edges()
    masses = np.full(260, 0.05)  # SFR = 1 per Gyr
    recent, previous = specific_sfr_windows(edges, masses, 199, surviving_mass=10.0)
    assert np.isclose(recent, 1.0 / 10.0, rtol=1e-6)
    assert np.isclose(previous, 1.0 / 10.0, rtol=1e-6)


def test_specific_sfr_windows_divide_by_surviving_mass():
    edges = time_bin_edges()
    masses = np.full(260, 0.05)
    recent, previous = specific_sfr_windows(edges, masses, 199, surviving_mass=6.0)
    assert np.isclose(recent, 1.0 / 6.0, rtol=1e-6)
    assert np.isclose(previous, 1.0 / 6.0, rtol=1e-6)


def _formed_mass_ssfr(edges, masses, epoch_index):
    t_obs = edges[epoch_index + 1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    lookback = t_obs - centers
    formed = masses[centers < t_obs].sum()
    recent = masses[(lookback > 0) & (lookback <= 0.1)].sum() / 0.1
    previous = masses[(lookback > 0.1) & (lookback <= 1.0)].sum() / 0.9
    return recent / formed, previous / formed


def test_history_ssfr_with_unit_fraction_reproduces_formed_mass_normalization(tmp_path):
    edges = time_bin_edges()[:81]
    out = _history_indices(_toy_grids(tmp_path), 2.5, 0.3, 0.1, edges)
    masses = np.diff(cumulative_mass(edges, 2.5, 0.3))
    expected = np.array([_formed_mass_ssfr(edges, masses, k) for k in range(80)])
    assert np.allclose(out["ssfr_0_100_myr"], expected[:, 0], rtol=1e-9)
    assert np.allclose(out["ssfr_100_1000_myr"], expected[:, 1], rtol=1e-9)
    assert np.allclose(out["surviving_mass_fraction"], 1.0, rtol=1e-12)


def test_history_ssfr_doubles_with_half_surviving_fraction(tmp_path):
    edges = time_bin_edges()[:81]
    unit = _history_indices(_toy_grids(tmp_path / "unit"), 2.5, 0.3, 0.1, edges)
    half = _history_indices(_toy_grids(tmp_path / "half", 0.5), 2.5, 0.3, 0.1, edges)
    for key in ("ssfr_0_100_myr", "ssfr_100_1000_myr"):
        assert np.allclose(half[key], 2.0 * unit[key], rtol=1e-12)
    assert np.allclose(half["surviving_mass_fraction"], 0.5, rtol=1e-12)
    assert np.array_equal(half["d4000_agb2"], unit["d4000_agb2"])


def test_history_indices_require_surviving_mass(tmp_path):
    grids = _toy_grids(tmp_path)
    for grid in grids.values():
        grid.surviving_mass_fraction = None
    with pytest.raises(ValueError, match="surviving_mass"):
        _history_indices(grids, 2.5, 0.3, 0.1, time_bin_edges()[:5])


def test_load_ssp_grid_attaches_surviving_mass(tmp_path):
    grid = _toy_grids(tmp_path)["r100"]
    grid.surviving_mass_fraction = None
    save_ssp_grid(grid, tmp_path)
    assert load_ssp_grid(tmp_path, "r100").surviving_mass_fraction.shape == (4, LOG_AGE.size)
    (tmp_path / "surviving_mass.npz").unlink()
    assert load_ssp_grid(tmp_path, "r100").surviving_mass_fraction is None


def test_population_indices_layout(tmp_path):
    draws = draw_population(3, seed=2)
    table = population_indices(_toy_grids(tmp_path), draws, time_bin_edges()[:21])
    assert table["history_id"].shape == (3 * 20,)
    assert set(table) >= {
        "t_q_gyr",
        "tau_q_gyr",
        "log_z",
        "epoch_gyr",
        "time_since_quenching_gyr",
        "sfr",
        "ssfr_0_100_myr",
        "ssfr_100_1000_myr",
        "d4000_agb0",
        "hdelta_a_agb2",
        "h_minus_bump_sigma300_agb0",
        "h_minus_bump_r100_agb2",
        "d4000_agb1",
        "hdelta_a_agb1",
        "h_minus_bump_sigma300_agb1",
        "h_minus_bump_r100_agb1",
        "surviving_mass_fraction",
    }
    assert np.allclose(table["hdelta_a_agb0"], 0.0, atol=1e-9)
    assert np.allclose(table["time_since_quenching_gyr"], table["epoch_gyr"] - table["t_q_gyr"])


def test_history_indices_cumulative_mass_fn_matches_default_path(tmp_path):
    grids = _toy_grids(tmp_path)
    edges = time_bin_edges()[:41]
    default = _history_indices(grids, 1.5, 0.3, 0.1, edges)
    delayed_tau = partial(cumulative_mass, t_q_gyr=1.5, tau_q_gyr=0.3)
    generic = _history_indices(grids, None, None, 0.1, edges, cumulative_mass_fn=delayed_tau)
    assert set(generic) == set(default)
    for key in default:
        if key != "sfr":
            assert np.array_equal(generic[key], default[key]), key
    bin_mean_sfr = np.diff(cumulative_mass(edges, 1.5, 0.3)) / np.diff(edges)
    assert np.allclose(generic["sfr"], bin_mean_sfr, rtol=1e-12)


def test_population_indices_exponential_family_is_bit_identical_to_default(tmp_path):
    draws = draw_population(3, seed=2)
    edges = time_bin_edges()[:21]
    grids = _toy_grids(tmp_path)
    default = population_indices(grids, draws, edges)
    explicit = population_indices(grids, draws, edges, family="exponential")
    assert set(explicit) == set(default)
    for key in default:
        assert np.array_equal(explicit[key], default[key]), key


def test_population_indices_every_family_runs_and_matches_the_dedicated_cumulative(tmp_path):
    draws = draw_population(3, seed=2)
    edges = time_bin_edges()[:41]
    for family in FAMILIES:
        grids = _toy_grids(tmp_path / family)
        if family == "decoupled":
            draws = dict(draws, tau_gyr=draw_decoupled_tau(3))
        table = population_indices(grids, draws, edges, family=family)
        assert table["history_id"].shape == (3 * 40,)
        assert np.all(np.isfinite(table["sfr"]))
        assert np.all(table["sfr"] >= -1e-12)


def test_population_indices_truncation_family_has_zero_mass_growth_after_t_q(tmp_path):
    draws = draw_population(3, seed=2)
    edges = time_bin_edges()[:101]
    grids = _toy_grids(tmp_path)
    table = population_indices(grids, draws, edges, family="truncation")
    # surviving_mass_fraction * formed mass up to t_obs stays flat once t_obs > t_q, so the
    # per-epoch sSFR windows must be exactly zero there (no new mass forms after t_q).
    after_t_q = table["epoch_gyr"] > table["t_q_gyr"] + 1.0
    assert np.any(after_t_q)
    assert np.allclose(table["ssfr_0_100_myr"][after_t_q], 0.0, atol=1e-12)
    assert np.allclose(table["ssfr_100_1000_myr"][after_t_q], 0.0, atol=1e-12)


def test_default_out_dir_derives_from_family():
    assert _default_out_dir("exponential") == OUTPUT_DIR
    assert _default_out_dir("linear") == OUTPUT_DIR.parent / "population_linear"
    assert _default_out_dir("truncation") == OUTPUT_DIR.parent / "population_truncation"
    assert _default_out_dir("decoupled") == OUTPUT_DIR.parent / "population_decoupled"
