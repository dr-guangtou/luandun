"""D4000 measured on the R = 50 optical product, for the JWST comparison figure only.

The NIRSpec PRISM resolution near the observed-frame 4000 A break of the z ~ 1 sample
is R ~ 50, far below the sigma = 300 km/s product used for the model D4000 everywhere
else. This module recomputes D4000 on the `r50` product (`broadening.py`, R = 50 FWHM
plus the 300 km/s dispersion) for the exponential-family population of both TP-AGB
configurations and for the fiducial-history metallicity tracks, and caches the result
in `<population dir>/d4000_r50.npz` with the row order of `indices.npz`.
"""

import numpy as np

from csp_integrate import agb_two_spectra, csp_track
from sfh_model import time_bin_edges
from spectral_indices import d4000
from ssp_grid import load_ssp_grid

CACHE_NAME = "d4000_r50.npz"


def _history_d4000(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr):
    """D4000 on the r50 grid at every epoch, for agb = 0, 1 and 2, of one history."""
    _, flux_agb0, _ = csp_track(grid, log_z, 0, t_q_gyr, tau_q_gyr, edges_gyr)
    _, flux_agb1, _ = csp_track(grid, log_z, 1, t_q_gyr, tau_q_gyr, edges_gyr)
    flux_agb2 = agb_two_spectra(flux_agb0, flux_agb1)
    return {
        "agb0": d4000(grid.wave_a, flux_agb0),
        "agb1": d4000(grid.wave_a, flux_agb1),
        "agb2": d4000(grid.wave_a, flux_agb2),
    }


def population_d4000_r50(grid_dir, population_dir, force=False):
    """D4000 on the r50 product for every history and epoch of one population, cached.
    Returns a dict with `history_id`, `epoch_gyr` and `d4000_r50_agb{0,1,2}` in the
    history-major, epoch-minor order that `run_population.py` writes."""
    cache = population_dir / CACHE_NAME
    if cache.exists() and not force:
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    grid = load_ssp_grid(grid_dir, "r50")
    edges = time_bin_edges()
    with np.load(population_dir / "draws.npz") as data:
        draws = {key: data[key] for key in data.files}
    n_draws = draws["t_q_gyr"].size
    n_epochs = edges.size - 1
    columns = {key: np.empty(n_draws * n_epochs) for key in ("agb0", "agb1", "agb2")}
    for i in range(n_draws):
        values = _history_d4000(
            grid, draws["log_z"][i], draws["t_q_gyr"][i], draws["tau_q_gyr"][i], edges
        )
        for key, column in columns.items():
            column[i * n_epochs : (i + 1) * n_epochs] = values[key]
    out = {
        "history_id": np.repeat(np.arange(n_draws), n_epochs),
        "epoch_gyr": np.tile(edges[1:], n_draws),
        "product_sigma_km_s": np.array(grid.provenance.get("total_sigma_km_s", np.nan)),
    }
    out.update({f"d4000_r50_{key}": column for key, column in columns.items()})
    np.savez(cache, **out)
    return out


def substitute_population_d4000(table, r50):
    """Return a copy of a loaded population table (possibly epoch-masked) whose
    `d4000_agb*` columns are the r50 values, aligned by history id and epoch."""
    key_full = r50["history_id"] * 1000 + np.round(r50["epoch_gyr"] * 20).astype(int)
    key_table = table["history_id"] * 1000 + np.round(table["epoch_gyr"] * 20).astype(int)
    position = {key: i for i, key in enumerate(key_full)}
    index = np.array([position[key] for key in key_table])
    out = dict(table)
    for agb_key in ("agb0", "agb1", "agb2"):
        out[f"d4000_{agb_key}"] = r50[f"d4000_r50_{agb_key}"][index]
    return out


def substitute_track_d4000(result, grid_dir, log_z, t_q_gyr, tau_q_gyr, edges_gyr):
    """Return a copy of a `compute_track_indices` result whose sigma300 D4000 entries are
    replaced by the r50 values, so that the JWST figure's metallicity tracks match."""
    grid = load_ssp_grid(grid_dir, "r50")
    values = _history_d4000(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr)
    out = dict(result)
    for agb_key in ("agb0", "agb2"):
        out[agb_key] = dict(result[agb_key])
        out[agb_key]["sigma300"] = dict(result[agb_key]["sigma300"]) | {"d4000": values[agb_key]}
    return out
