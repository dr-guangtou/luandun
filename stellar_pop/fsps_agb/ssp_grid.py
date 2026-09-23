"""Build, cache and load the FSPS SSP grids used by the CSP integrator.

Fixed configuration: MIST isochrones, C3K high-resolution spectra, Chabrier
IMF, all other python-fsps parameters at their defaults. Only the TP-AGB
weight `agb` and the metallicity vary. Spectra are F_nu in Lsun/Hz per
solar mass formed, on the FSPS vacuum wavelength grid restricted to the
3400-22000 A window.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

WAVE_MIN_A = 3400.0
WAVE_MAX_A = 22000.0
LOG_Z_GRID = (-0.5, -0.25, 0.0, 0.25)
ZMET_BY_LOG_Z = {-0.5: 9, -0.25: 10, 0.0: 11, 0.25: 12}
AGB_WEIGHTS = (0.0, 1.0)
IMF_TYPE_CHABRIER = 1
PRODUCTS = ("native", "sigma300", "r100")
DEFAULT_GRID_DIR = Path(__file__).resolve().parent / "output" / "ssp_grid"


@dataclass
class SspGrid:
    wave_a: np.ndarray
    log_age_yr: np.ndarray
    log_z_grid: np.ndarray
    agb_weights: np.ndarray
    flux_nu: np.ndarray
    native_sigma_km_s: np.ndarray
    product: str
    provenance: dict = field(default_factory=dict)


def _git_hash(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_ssp(log_z, agb):
    import fsps

    population = fsps.StellarPopulation(
        zcontinuous=0, zmet=ZMET_BY_LOG_Z[log_z], imf_type=IMF_TYPE_CHABRIER, sfh=0
    )
    population.params["agb"] = agb
    wave_a, flux_nu = population.get_spectrum(tage=0.0, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    resolutions = np.asarray(population.resolutions)
    provenance = {
        "fsps_version": fsps.__version__,
        "libraries": [item.decode() for item in population.libraries],
        "sps_home": os.environ.get("SPS_HOME", "unset"),
        "sps_home_git_hash": _git_hash(os.environ.get("SPS_HOME", ".")),
        "params": {
            key: population.params[key]
            for key in (
                "imf_type",
                "zmet",
                "agb",
                "pagb",
                "add_agb_dust_model",
                "agb_dust",
                "use_lw_tpagb",
                "add_neb_emission",
                "dust1",
                "dust2",
                "sfh",
            )
        },
        "native_sigma_km_s_note": "from StellarPopulation.resolutions; negative means approximate",
    }
    return (
        wave_a[window],
        np.asarray(population.ssp_ages),
        flux_nu[:, window],
        provenance | {"native_sigma_km_s": np.abs(resolutions[window]).tolist()},
    )


def build_and_cache_grid(out_dir=DEFAULT_GRID_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flux = None
    provenance = {"per_ssp": {}}
    for i_z, log_z in enumerate(LOG_Z_GRID):
        for i_agb, agb in enumerate(AGB_WEIGHTS):
            wave_a, log_age_yr, flux_nu, ssp_provenance = build_ssp(log_z, agb)
            native_sigma = np.array(ssp_provenance.pop("native_sigma_km_s"))
            if flux is None:
                flux = np.empty((len(LOG_Z_GRID), len(AGB_WEIGHTS)) + flux_nu.shape)
            flux[i_z, i_agb] = flux_nu
            provenance["per_ssp"][f"log_z={log_z:+.2f},agb={agb:g}"] = ssp_provenance
    grid = SspGrid(
        wave_a=wave_a,
        log_age_yr=log_age_yr,
        log_z_grid=np.array(LOG_Z_GRID),
        agb_weights=np.array(AGB_WEIGHTS),
        flux_nu=flux,
        native_sigma_km_s=native_sigma,
        product="native",
        provenance=provenance,
    )
    return save_ssp_grid(grid, out_dir)


def save_ssp_grid(grid, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{grid.product}.npz"
    arrays = {
        key: value for key, value in asdict(grid).items() if key not in ("product", "provenance")
    }
    np.savez(path, product=grid.product, **arrays)
    if grid.product == "native" or not (out_dir / "provenance.json").exists():
        (out_dir / "provenance.json").write_text(
            json.dumps(grid.provenance, indent=2, default=str) + "\n"
        )
    return path


def load_ssp_grid(out_dir, product):
    out_dir = Path(out_dir)
    with np.load(out_dir / f"{product}.npz") as data:
        arrays = {key: data[key] for key in data.files if key != "product"}
    provenance_path = out_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    return SspGrid(product=product, provenance=provenance, **arrays)


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    written = build_and_cache_grid()
    print(f"wrote {written} in {time.perf_counter() - start:.1f} s")
