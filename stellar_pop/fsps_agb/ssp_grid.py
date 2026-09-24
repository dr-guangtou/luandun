"""Build, cache and load the FSPS SSP grids used by the CSP integrator.

Fixed configuration: MIST isochrones, C3K high-resolution spectra, Chabrier
IMF, all other python-fsps parameters at their defaults. Only the TP-AGB
weight `agb` and the metallicity vary. Spectra are F_nu in Lsun/Hz per
solar mass formed, on the FSPS vacuum wavelength grid restricted to the
3400-22000 A window.
"""

import hashlib
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
SURVIVING_MASS_AGB = 1.0
SURVIVING_MASS_FILE = "surviving_mass.npz"
PARAM_KEYS = (
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
DEFAULT_GRID_DIR = Path(__file__).resolve().parent / "output" / "ssp_grid"
PYTHON_FSPS_DIR = Path("/Users/shuang/code/python-fsps")
WHEEL_DIR = Path(__file__).resolve().parent / "wheels"


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
    surviving_mass_fraction: np.ndarray | None = None


def _git_command(path, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return f"unknown (git failed: {error})"


def _git_hash(path):
    return _git_command(path, "rev-parse", "HEAD").strip()


def _git_describe(path):
    return _git_command(path, "describe", "--always", "--dirty").strip()


def _git_diff_sha256(path, *pathspec):
    diff = _git_command(path, "diff", "--", *pathspec)
    if diff.startswith("unknown (git failed:"):
        return diff
    return hashlib.sha256(diff.encode()).hexdigest()


def record_build_environment(sps_home=None, python_fsps_dir=PYTHON_FSPS_DIR, wheel_dir=WHEEL_DIR):
    """Reproducibility record for the FSPS build environment: the SPS_HOME and
    python-fsps checkouts and the CMake patch that enables C3K_HR, plus the
    locally built wheel filename. Cheap (a handful of git subprocess calls);
    does not touch the SSP grid itself."""
    sps_home = Path(sps_home or os.environ.get("SPS_HOME", "."))
    python_fsps_dir = Path(python_fsps_dir)
    wheel_dir = Path(wheel_dir)
    wheel_names = sorted(p.name for p in wheel_dir.glob("*.whl")) if wheel_dir.exists() else []
    return {
        "sps_home_git_describe": _git_describe(sps_home),
        "sps_home_diff_sha256": _git_diff_sha256(sps_home),
        "python_fsps_git_describe": _git_describe(python_fsps_dir),
        "python_fsps_cmake_diff_sha256": _git_diff_sha256(
            python_fsps_dir, "src/fsps/CMakeLists.txt"
        ),
        "python_fsps_libfsps_submodule_commit": _git_hash(
            python_fsps_dir / "src" / "fsps" / "libfsps"
        ),
        "wheel_filename": wheel_names[0] if wheel_names else "unknown",
    }


def _provenance_dict(fsps_version, libraries, params, sps_home_env, extra_params):
    sps_home_for_git = sps_home_env or "."
    build_environment = record_build_environment(sps_home=sps_home_for_git)
    return {
        "fsps_version": fsps_version,
        "libraries": libraries,
        "sps_home": sps_home_env or "unset",
        "sps_home_git_hash": _git_hash(sps_home_for_git),
        "sps_home_git_describe": build_environment["sps_home_git_describe"],
        "sps_home_diff_sha256": build_environment["sps_home_diff_sha256"],
        "params": {key: params[key] for key in PARAM_KEYS},
        "extra_params": dict(extra_params or {}),
        "native_sigma_km_s_note": "from StellarPopulation.resolutions; stored as absolute values",
    }


def build_ssp(log_z, agb, extra_params=None):
    import fsps

    population = fsps.StellarPopulation(
        zcontinuous=0, zmet=ZMET_BY_LOG_Z[log_z], imf_type=IMF_TYPE_CHABRIER, sfh=0
    )
    population.params["agb"] = agb
    for key, value in (extra_params or {}).items():
        population.params[key] = value
    wave_a, flux_nu = population.get_spectrum(tage=0.0, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    resolutions = np.asarray(population.resolutions)
    provenance = _provenance_dict(
        fsps.__version__,
        [item.decode() for item in population.libraries],
        population.params,
        os.environ.get("SPS_HOME"),
        extra_params,
    )
    return (
        wave_a[window],
        np.asarray(population.ssp_ages),
        flux_nu[:, window],
        provenance | {"native_sigma_km_s": np.abs(resolutions[window]).tolist()},
    )


def surviving_mass_fraction(log_z, extra_params=None):
    """Surviving stellar mass (living stars plus remnants, the FSPS `add_stellar_remnants`
    default) per solar mass formed at every SSP age, from `StellarPopulation.stellar_mass`
    with `tage = 0`. Built at agb = 1: FSPS rescales the TP-AGB IMF weights by `agb`
    before summing the mass, so agb = 1 is the unmodified isochrone."""
    import fsps

    population = fsps.StellarPopulation(
        zcontinuous=0, zmet=ZMET_BY_LOG_Z[log_z], imf_type=IMF_TYPE_CHABRIER, sfh=0
    )
    population.params["agb"] = SURVIVING_MASS_AGB
    for key, value in (extra_params or {}).items():
        population.params[key] = value
    population.get_spectrum(tage=0.0, peraa=False)
    provenance = _provenance_dict(
        fsps.__version__,
        [item.decode() for item in population.libraries],
        population.params,
        os.environ.get("SPS_HOME"),
        extra_params,
    )
    provenance["add_stellar_remnants"] = bool(population.params["add_stellar_remnants"])
    return (
        np.asarray(population.ssp_ages),
        np.array(population.stellar_mass, dtype=float),
        provenance,
    )


def build_surviving_mass_fractions(out_dir=DEFAULT_GRID_DIR, extra_params=None):
    provenance = {
        "definition": "StellarPopulation.stellar_mass at tage = 0 and agb = 1: surviving "
        "mass (living stars plus remnants) per solar mass formed, per SSP age",
        "per_log_z": {},
        "build_environment": record_build_environment(),
        "extra_params": dict(extra_params or {}),
    }
    rows = []
    for log_z in LOG_Z_GRID:
        log_age_yr, row, log_z_provenance = surviving_mass_fraction(log_z, extra_params)
        rows.append(row)
        provenance["per_log_z"][f"log_z={log_z:+.2f}"] = log_z_provenance
    path = save_surviving_mass(out_dir, log_age_yr, np.array(LOG_Z_GRID), np.vstack(rows))
    (Path(out_dir) / "provenance_surviving_mass.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    return path


def save_surviving_mass(out_dir, log_age_yr, log_z_grid, fraction):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SURVIVING_MASS_FILE
    np.savez(path, log_age_yr=log_age_yr, log_z_grid=log_z_grid, fraction=fraction)
    return path


def load_surviving_mass(out_dir):
    with np.load(Path(out_dir) / SURVIVING_MASS_FILE) as data:
        return data["log_age_yr"], data["log_z_grid"], data["fraction"]


def build_and_cache_grid(out_dir=DEFAULT_GRID_DIR, extra_params=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flux = None
    provenance = {
        "per_ssp": {},
        "build_environment": record_build_environment(),
        "extra_params": dict(extra_params or {}),
    }
    for i_z, log_z in enumerate(LOG_Z_GRID):
        for i_agb, agb in enumerate(AGB_WEIGHTS):
            wave_a, log_age_yr, flux_nu, ssp_provenance = build_ssp(
                log_z, agb, extra_params=extra_params
            )
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
        key: value
        for key, value in asdict(grid).items()
        if key not in ("product", "provenance", "surviving_mass_fraction")
    }
    np.savez(path, product=grid.product, **arrays)
    provenance_text = json.dumps(grid.provenance, indent=2, default=str) + "\n"
    (out_dir / f"provenance_{grid.product}.json").write_text(provenance_text)
    if grid.product == "native" or not (out_dir / "provenance.json").exists():
        (out_dir / "provenance.json").write_text(provenance_text)
    return path


def load_ssp_grid(out_dir, product):
    out_dir = Path(out_dir)
    with np.load(out_dir / f"{product}.npz") as data:
        arrays = {key: data[key] for key in data.files if key != "product"}
    provenance_path = out_dir / f"provenance_{product}.json"
    if not provenance_path.exists():
        provenance_path = out_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    grid = SspGrid(product=product, provenance=provenance, **arrays)
    if (out_dir / SURVIVING_MASS_FILE).exists():
        log_age_yr, log_z_grid, fraction = load_surviving_mass(out_dir)
        if not (
            np.allclose(log_age_yr, grid.log_age_yr) and np.allclose(log_z_grid, grid.log_z_grid)
        ):
            raise ValueError(f"{out_dir / SURVIVING_MASS_FILE} does not match the {product} grid")
        grid.surviving_mass_fraction = fraction
    return grid


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Build and cache the native FSPS SSP grid.")
    parser.add_argument("--out-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument(
        "--use-lw-tpagb",
        action="store_true",
        help="use the Lancon & Mouhcine (2002) empirical O-rich TP-AGB spectra",
    )
    parser.add_argument(
        "--surviving-mass",
        action="store_true",
        help="build only the surviving stellar mass fractions (surviving_mass.npz)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    extra_params = {"use_lw_tpagb": 1} if args.use_lw_tpagb else None
    if args.surviving_mass:
        written = build_surviving_mass_fractions(args.out_dir, extra_params=extra_params)
    else:
        written = build_and_cache_grid(args.out_dir, extra_params=extra_params)
    print(f"wrote {written} in {time.perf_counter() - start:.1f} s")
