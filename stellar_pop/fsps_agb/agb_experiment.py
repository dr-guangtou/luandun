"""Generate and compare SSP spectra under different AGB/TP-AGB assumptions.

Fixed: MIST isochrones, C3K (low-res) spectra, solar metallicity, 1 Gyr age.
Varied: AGB/TP-AGB parameters (see docs/SPEC.md) and IMF (Kroupa, Chabrier).
"""

from pathlib import Path

import numpy as np

import fsps

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

AGE_GYR = 1.0
SOLAR_ZMET = 11  # 1-based index of Z = zsol = 0.0185 in the MIST grid

# Each scenario overrides a subset of the AGB/TP-AGB SSP parameters.
SCENARIOS = {
    "fiducial": {},
    "no_tpagb": {"agb": 0.0},
    "no_pagb": {"pagb": 0.0},
    "no_agb_dust": {"add_agb_dust_model": False},
    "lw02_o_rich": {"use_lw_tpagb": 1},
    "double_tpagb": {"agb": 2.0},
    "no_agb_all": {
        "agb": 0.0,
        "pagb": 0.0,
        "add_agb_dust_model": False,
    },
    "norm_type_0": {"tpagb_norm_type": 0},
    "norm_type_1": {"tpagb_norm_type": 1},
}

IMF_LABELS = {1: "chabrier", 2: "kroupa"}


def build_spectrum(imf_type, overrides):
    """Return (wavelength_A, spectrum_Lsun_per_Hz) for one scenario."""
    sp = fsps.StellarPopulation(imf_type=imf_type, zmet=SOLAR_ZMET)
    for key, value in overrides.items():
        sp.params[key] = value
    wave, spec = sp.get_spectrum(tage=AGE_GYR, peraa=False)
    return wave, spec, sp


def main(out_path, spec_label):
    wave = None
    results = {}
    meta = {}

    for imf_type in (2, 1):  # Kroupa, Chabrier
        imf_name = IMF_LABELS[imf_type]
        for scenario, overrides in SCENARIOS.items():
            w, spec, sp = build_spectrum(imf_type, overrides)
            if wave is None:
                wave = w
            key = f"{imf_name}__{scenario}"
            results[key] = spec
            meta[key] = {
                "imf_type": imf_type,
                "scenario": scenario,
                "log_lbol": float(sp.log_lbol),
                "stellar_mass": float(sp.stellar_mass),
            }

    wave = np.asarray(wave)
    np.savez(
        out_path,
        wavelength_A=wave,
        age_gyr=AGE_GYR,
        solar_zmet=SOLAR_ZMET,
        spec_library=spec_label,
        **{k: v for k, v in results.items()},
    )

    print(f"Wavelength grid: {len(wave)} points, {wave[0]:.1f}-{wave[-1]:.1f} A")
    print(f"{'scenario':<28} {'log Lbol':>9} {'mass (Msun)':>12}")
    for key in meta:
        m = meta[key]
        print(f"{key:<28} {m['log_lbol']:9.3f} {m['stellar_mass']:12.4f}")

    return results, meta, wave


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SSP spectra across AGB scenarios.")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "ssp_spectra.npz"))
    parser.add_argument("--spec-label", default="C3K-LR")
    args = parser.parse_args()
    main(args.out, args.spec_label)
