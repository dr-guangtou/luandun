"""Compare the numpy CSP integrator with FSPS's own tabular-SFH CSP.

FSPS interpolates the SFR linearly between table nodes and integrates
against the same log-age kernel; the integrator uses exact bin masses. The
comparison is made on the native FSPS wavelength grid, inside the three
index windows, with the fiducial history at solar metallicity and agb = 1.

Building a `fsps.StellarPopulation` costs about 11 s (SSP construction),
while re-evaluating `get_spectrum(tage=...)` on the same object costs
milliseconds. `fsps_tabular_spectrum` therefore caches the population
object keyed on every parameter except `tage_gyr`, so a caller that sweeps
several epochs at fixed `(t_q_gyr, tau_q_gyr, zmet, agb, edges_gyr)` -
as `run_cross_check` does - only pays the 11 s cost once. The public
signature and return value are unchanged from a naive per-call
implementation.
"""

import json
from pathlib import Path

import numpy as np

from csp_integrate import csp_track
from sfh_model import star_formation_rate, time_bin_edges
from spectral_indices import D4000_BANDS_A, H_MINUS_BANDS_A, HDELTA_A_BANDS_A, measure_all
from ssp_grid import DEFAULT_GRID_DIR, IMF_TYPE_CHABRIER, WAVE_MAX_A, WAVE_MIN_A, load_ssp_grid

FIDUCIAL = {"t_q_gyr": 3.0, "tau_q_gyr": 0.3, "log_z": 0.0}
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "single_csp" / "fsps_cross_check.json"
WINDOWS_A = {
    "d4000": (D4000_BANDS_A["blue"][0], D4000_BANDS_A["red"][1]),
    "hdelta_a": (HDELTA_A_BANDS_A["blue"][0], HDELTA_A_BANDS_A["red"][1]),
    "h_minus_bump": (H_MINUS_BANDS_A["blue"][0], H_MINUS_BANDS_A["red"][1]),
}

_population_cache = {"key": None, "population": None}


def _cached_tabular_population(t_q_gyr, tau_q_gyr, zmet, agb, edges_gyr):
    """Return a StellarPopulation with the tabular SFH already set, rebuilding it
    only when the parameters (other than tage) differ from the cached one."""
    import fsps

    key = (t_q_gyr, tau_q_gyr, zmet, agb, tuple(np.asarray(edges_gyr, dtype=float)))
    if _population_cache["key"] != key:
        population = fsps.StellarPopulation(
            zcontinuous=0, zmet=zmet, imf_type=IMF_TYPE_CHABRIER, sfh=3
        )
        population.params["agb"] = agb
        nodes = edges_gyr[1:]
        population.set_tabular_sfh(nodes, star_formation_rate(nodes, t_q_gyr, tau_q_gyr))
        _population_cache["key"] = key
        _population_cache["population"] = population
    return _population_cache["population"]


def fsps_tabular_spectrum(t_q_gyr, tau_q_gyr, zmet, agb, tage_gyr, edges_gyr):
    population = _cached_tabular_population(t_q_gyr, tau_q_gyr, zmet, agb, edges_gyr)
    wave_a, flux_nu = population.get_spectrum(tage=tage_gyr, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    return wave_a[window], flux_nu[window] / population.formed_mass


def run_cross_check(epochs_gyr=(1.0, 3.0, 3.5, 5.0, 8.0, 13.0), grid_dir=DEFAULT_GRID_DIR):
    grid = load_ssp_grid(grid_dir, "native")
    edges = time_bin_edges()
    epoch_grid, own_flux, _ = csp_track(
        grid, FIDUCIAL["log_z"], 1, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], edges
    )
    report = {"fiducial": FIDUCIAL, "epochs": {}}
    for epoch in epochs_gyr:
        k = int(np.argmin(np.abs(epoch_grid - epoch)))
        wave_a, fsps_flux = fsps_tabular_spectrum(
            FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], 11, 1.0, float(epoch_grid[k]), edges
        )
        assert np.allclose(wave_a, grid.wave_a)
        mine = own_flux[k]
        differences = {}
        for name, (lo, hi) in WINDOWS_A.items():
            inside = (wave_a >= lo) & (wave_a <= hi)
            differences[name] = float(np.max(np.abs(mine[inside] / fsps_flux[inside] - 1.0)))
        own_indices = measure_all(wave_a, mine)
        fsps_indices = measure_all(wave_a, fsps_flux)
        index_difference = {key: float(own_indices[key] - fsps_indices[key]) for key in own_indices}
        report["epochs"][f"{epoch_grid[k]:.2f}"] = {
            "max_relative_flux_difference": differences,
            "index_difference": index_difference,
            "own_indices": {key: float(value) for key, value in own_indices.items()},
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run_cross_check(), indent=2))
