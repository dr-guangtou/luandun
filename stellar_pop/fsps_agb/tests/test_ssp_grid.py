import json

import numpy as np
import pytest

from ssp_grid import (
    LOG_Z_GRID,
    PARAM_KEYS,
    ZMET_BY_LOG_Z,
    SspGrid,
    _provenance_dict,
    build_ssp,
    load_ssp_grid,
    save_ssp_grid,
)


def test_zmet_lookup_matches_mist_zlegend():
    assert ZMET_BY_LOG_Z == {-0.5: 9, -0.25: 10, 0.0: 11, 0.25: 12}
    assert LOG_Z_GRID == (-0.5, -0.25, 0.0, 0.25)


@pytest.mark.slow
def test_build_ssp_returns_window_and_provenance():
    wave_a, log_age_yr, flux_nu, provenance = build_ssp(log_z=0.0, agb=1.0)
    assert log_age_yr.shape == (107,)
    assert np.isclose(log_age_yr[0], 5.0) and np.isclose(log_age_yr[-1], 10.3)
    assert wave_a[0] >= 3400.0 and wave_a[-1] <= 22000.0
    assert flux_nu.shape == (107, wave_a.size)
    assert np.all(flux_nu >= 0)
    assert provenance["libraries"] == ["mist", "c3k_hr", "DL07"]
    assert provenance["params"]["imf_type"] == 1
    assert provenance["params"]["agb"] == 1.0
    assert provenance["params"]["zmet"] == 11


@pytest.mark.slow
@pytest.mark.parametrize("use_lw_tpagb", [0, 1])
def test_agb_weight_is_exactly_linear(use_lw_tpagb):
    extra_params = {"use_lw_tpagb": use_lw_tpagb}
    wave_a, _, flux_0, _ = build_ssp(log_z=0.0, agb=0.0, extra_params=extra_params)
    _, _, flux_1, _ = build_ssp(log_z=0.0, agb=1.0, extra_params=extra_params)
    _, _, flux_2, _ = build_ssp(log_z=0.0, agb=2.0, extra_params=extra_params)
    predicted = 2.0 * flux_1 - flux_0
    scale = np.max(flux_2)
    assert np.max(np.abs(flux_2 - predicted)) / scale < 1e-10


def test_build_ssp_extra_params_recorded():
    fake_params = dict.fromkeys(PARAM_KEYS, 0) | {"agb": 1.0, "use_lw_tpagb": 1}
    provenance = _provenance_dict(
        "1.0.0", ["mist", "c3k_hr", "DL07"], fake_params, None, {"use_lw_tpagb": 1}
    )
    assert provenance["extra_params"] == {"use_lw_tpagb": 1}
    assert provenance["params"]["use_lw_tpagb"] == 1
    assert provenance["params"]["agb"] == 1.0
    assert provenance["sps_home"] == "unset"


def test_save_and_load_round_trip(tmp_path):
    wave_a = np.linspace(3400.0, 22000.0, 100)
    grid = SspGrid(
        wave_a=wave_a,
        log_age_yr=np.linspace(5.0, 10.3, 107),
        log_z_grid=np.array(LOG_Z_GRID),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=np.ones((4, 2, 107, 100)),
        native_sigma_km_s=np.full(100, 42.4),
        product="native",
        provenance={"note": "test"},
    )
    save_ssp_grid(grid, tmp_path)
    loaded = load_ssp_grid(tmp_path, "native")
    assert loaded.flux_nu.shape == (4, 2, 107, 100)
    assert loaded.product == "native"
    assert json.loads((tmp_path / "provenance.json").read_text())["note"] == "test"


def test_save_and_load_product_provenance(tmp_path):
    wave_a = np.linspace(3400.0, 22000.0, 100)
    grid = SspGrid(
        wave_a=wave_a,
        log_age_yr=np.linspace(5.0, 10.3, 107),
        log_z_grid=np.array(LOG_Z_GRID),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=np.ones((4, 2, 107, 100)),
        native_sigma_km_s=np.full(100, 42.4),
        product="sigma300",
        provenance={"total_sigma_km_s": 300.0},
    )
    save_ssp_grid(grid, tmp_path)
    assert (tmp_path / "provenance_sigma300.json").exists()
    loaded = load_ssp_grid(tmp_path, "sigma300")
    assert loaded.provenance["total_sigma_km_s"] == 300.0
