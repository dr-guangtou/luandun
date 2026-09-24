"""Compare the numpy CSP integrator's FSPS-native SFH families with FSPS's own
`sfh = 5` (linear) and `sfh = 4` with `sf_trunc` (truncation) CSPs.

Same pattern as `cross_check_fsps_tabular.py`: one `fsps.StellarPopulation` per
family, cached and reused across the swept epochs (`tage` is the only thing
that changes after the first ~11 s build), compared on the native FSPS
wavelength grid inside the three index windows at the fiducial history
(`t_q = 3 Gyr`, `tau_q = 0.3 Gyr`, solar Z, `agb = 1`).

FSPS's `sfh = 5` ramps `SFR = SFR(sf_trunc) * [1 + sf_slope * (t -
sf_trunc)]` (clipped at zero) after `sf_trunc`, with `sf_slope` in Gyr^-1 of
forward time. `verify_sf_slope_sign` confirms empirically, by reading
`population.sfr`, that `sf_slope = -1 / delta_q_gyr` (not `+1 / delta_q_gyr`)
gives the declining ramp the `linear` family expects.
"""

import json
from pathlib import Path

import numpy as np

from cross_check_fsps_tabular import FIDUCIAL, WINDOWS_A
from csp_integrate import csp_spectra, epoch_weight_matrix_from_cumulative, interpolate_log_z
from sfh_model import sfh_family_cumulative, time_bin_edges
from spectral_indices import measure_all
from ssp_grid import (
    DEFAULT_GRID_DIR,
    IMF_TYPE_CHABRIER,
    WAVE_MAX_A,
    WAVE_MIN_A,
    ZMET_BY_LOG_Z,
    load_ssp_grid,
)

EPOCHS_GYR = (1.0, 3.0, 3.5, 5.0, 8.0, 13.0)
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "output" / "single_csp" / "fsps_family_cross_check.json"
)
FAMILY_FSPS_SFH = {"linear": 5, "truncation": 4}

_population_cache = {"key": None, "population": None}


def _fsps_family_params(family, t_q_gyr, tau_q_gyr):
    """FSPS `StellarPopulation` parameters that reproduce one SFH family."""
    if family == "linear":
        delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
        return {
            "sfh": 5,
            "tau": t_q_gyr,
            "sf_start": 0.0,
            "sf_trunc": t_q_gyr,
            "sf_slope": -1.0 / delta_q_gyr,
        }
    if family == "truncation":
        return {"sfh": 4, "tau": t_q_gyr, "sf_start": 0.0, "sf_trunc": t_q_gyr}
    raise ValueError(f"no FSPS-native mapping for family {family!r}")


def _cached_family_population(family, t_q_gyr, tau_q_gyr, zmet, agb):
    """Return a StellarPopulation for `family` already set up, rebuilding it only
    when the parameters (other than tage) differ from the cached one."""
    import fsps

    key = (family, t_q_gyr, tau_q_gyr, zmet, agb)
    if _population_cache["key"] != key:
        params = _fsps_family_params(family, t_q_gyr, tau_q_gyr)
        population = fsps.StellarPopulation(
            zcontinuous=0, zmet=zmet, imf_type=IMF_TYPE_CHABRIER, **params
        )
        population.params["agb"] = agb
        _population_cache["key"] = key
        _population_cache["population"] = population
    return _population_cache["population"]


def fsps_family_spectrum(family, t_q_gyr, tau_q_gyr, zmet, agb, tage_gyr):
    population = _cached_family_population(family, t_q_gyr, tau_q_gyr, zmet, agb)
    wave_a, flux_nu = population.get_spectrum(tage=tage_gyr, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    return wave_a[window], flux_nu[window] / population.formed_mass


def verify_sf_slope_sign(t_q_gyr, tau_q_gyr, zmet=11, agb=1.0):
    """Empirically confirm the `sf_slope` sign convention: with `sf_slope = -1 /
    delta_q_gyr`, FSPS's `sfr` attribute at `t_q + delta_q / 2` must be below its
    value at `t_q` (a declining ramp, about half, modulo FSPS's internal SFH time
    discretization)."""
    delta_q_gyr = 2.0 * np.log(2.0) * tau_q_gyr
    population = _cached_family_population("linear", t_q_gyr, tau_q_gyr, zmet, agb)
    population.get_spectrum(tage=t_q_gyr, peraa=False)
    sfr_at_t_q = float(population.sfr)
    population.get_spectrum(tage=t_q_gyr + delta_q_gyr / 2.0, peraa=False)
    sfr_at_half_delta_q = float(population.sfr)
    return {
        "sf_slope_per_gyr": -1.0 / delta_q_gyr,
        "sfr_at_t_q": sfr_at_t_q,
        "sfr_at_t_q_plus_half_delta_q": sfr_at_half_delta_q,
        "ratio": sfr_at_half_delta_q / sfr_at_t_q,
        "declining": sfr_at_half_delta_q < sfr_at_t_q,
    }


def run_cross_check(epochs_gyr=EPOCHS_GYR, grid_dir=DEFAULT_GRID_DIR, output_path=OUTPUT_PATH):
    grid = load_ssp_grid(grid_dir, "native")
    edges = time_bin_edges()
    epoch_grid = edges[1:]
    zmet = ZMET_BY_LOG_Z[FIDUCIAL["log_z"]]
    ssp_flux = interpolate_log_z(grid.flux_nu[:, 1], grid.log_z_grid, FIDUCIAL["log_z"])
    sign_check = verify_sf_slope_sign(FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], zmet)
    report = {"fiducial": FIDUCIAL, "sf_slope_sign_check": sign_check, "families": {}}
    for family in FAMILY_FSPS_SFH:
        cumulative_fn = sfh_family_cumulative(family, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"])
        weights = epoch_weight_matrix_from_cumulative(edges, cumulative_fn, grid.log_age_yr)
        own_flux = csp_spectra(weights, ssp_flux)
        family_report = {"epochs": {}}
        for epoch in epochs_gyr:
            k = int(np.argmin(np.abs(epoch_grid - epoch)))
            wave_a, fsps_flux = fsps_family_spectrum(
                family,
                FIDUCIAL["t_q_gyr"],
                FIDUCIAL["tau_q_gyr"],
                zmet,
                1.0,
                float(epoch_grid[k]),
            )
            assert np.allclose(wave_a, grid.wave_a)
            mine = own_flux[k]
            differences = {}
            for name, (lo, hi) in WINDOWS_A.items():
                inside = (wave_a >= lo) & (wave_a <= hi)
                differences[name] = float(np.max(np.abs(mine[inside] / fsps_flux[inside] - 1.0)))
            own_indices = measure_all(wave_a, mine)
            fsps_indices = measure_all(wave_a, fsps_flux)
            index_difference = {
                key: float(own_indices[key] - fsps_indices[key]) for key in own_indices
            }
            family_report["epochs"][f"{epoch_grid[k]:.2f}"] = {
                "max_relative_flux_difference": differences,
                "index_difference": index_difference,
                "own_indices": {key: float(value) for key, value in own_indices.items()},
            }
        report["families"][family] = family_report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run_cross_check(), indent=2))
