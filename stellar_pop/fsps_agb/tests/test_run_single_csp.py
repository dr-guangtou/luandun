import numpy as np

from run_single_csp import compute_track_indices
from ssp_grid import SspGrid, save_ssp_grid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def _write_toy_grids(directory):
    wave = np.linspace(3400.0, 22000.0, 6000)
    base = np.ones((4, 2, LOG_AGE.size, wave.size)) * (1.0 + 1e-5 * wave) * wave**2
    for product, lo, hi in (("sigma300", 3400.0, 22000.0), ("r100", 12500.0, 21000.0)):
        keep = (wave >= lo) & (wave <= hi)
        grid = SspGrid(
            wave_a=wave[keep],
            log_age_yr=LOG_AGE,
            log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
            agb_weights=np.array([0.0, 1.0]),
            flux_nu=base[..., keep],
            native_sigma_km_s=np.full(keep.sum(), 42.4),
            product=product,
        )
        save_ssp_grid(grid, directory)


def test_compute_track_indices_shapes(tmp_path):
    _write_toy_grids(tmp_path)
    result = compute_track_indices(tmp_path, t_q_gyr=3.0, tau_q_gyr=0.3, log_z=0.0)
    assert result["epoch_gyr"].shape == (260,)
    assert result["sfr"].shape == (260,)
    for agb in ("agb0", "agb2"):
        assert result[agb]["sigma300"]["d4000"].shape == (260,)
        assert result[agb]["sigma300"]["hdelta_a"].shape == (260,)
        assert result[agb]["sigma300"]["h_minus_bump"].shape == (260,)
        assert result[agb]["r100"]["h_minus_bump"].shape == (260,)
        assert np.allclose(result[agb]["sigma300"]["hdelta_a"], 0.0, atol=1e-9)
