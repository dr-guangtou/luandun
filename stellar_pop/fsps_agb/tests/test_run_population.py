from functools import partial

import numpy as np

from run_population import _history_indices, population_indices, specific_sfr_windows
from sfh_model import cumulative_mass, draw_population, time_bin_edges
from ssp_grid import SspGrid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def _toy_grids():
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
        )
    return grids


def test_specific_sfr_windows_for_constant_sfr():
    edges = time_bin_edges()
    masses = np.full(260, 0.05)  # SFR = 1 per Gyr
    recent, previous = specific_sfr_windows(edges, masses, epoch_index=199)  # t_obs = 10 Gyr
    assert np.isclose(recent, 1.0 / 10.0, rtol=1e-6)
    assert np.isclose(previous, 1.0 / 10.0, rtol=1e-6)


def test_population_indices_layout():
    draws = draw_population(3, seed=2)
    table = population_indices(_toy_grids(), draws, time_bin_edges()[:21])
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
    }
    assert np.allclose(table["hdelta_a_agb0"], 0.0, atol=1e-9)
    assert np.allclose(table["time_since_quenching_gyr"], table["epoch_gyr"] - table["t_q_gyr"])


def test_history_indices_cumulative_mass_fn_matches_default_path():
    grids = _toy_grids()
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
