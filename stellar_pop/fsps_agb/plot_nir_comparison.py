"""SSP spectra under different AGB assumptions: broad SED + two NIR zooms.

One figure per IMF, each with three stacked panels:
1. broad SED (270-1900 nm, linear wavelength, log flux);
2. NIR zoom normalized by the median of the 1495-1535 nm window (norm_1);
3. NIR zoom normalized by the blue-red straight-line continuum (norm_2).

Only the zoom-range points are plotted on panels 2 and 3 so that the
continuum normalization (which extrapolates wildly outside the NIR) cannot
pollute the y-axis autoscale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import norm_utils as nu

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SCENARIOS = [
    ("fiducial", "Fiducial (agb=1, dust on)", "#1f77b4"),
    ("no_tpagb", "No TP-AGB (agb=0)", "#d62728"),
    ("double_tpagb", "2x TP-AGB (agb=2)", "#2ca02c"),
    ("no_pagb", "No post-AGB (pagb=0)", "#9467bd"),
    ("no_agb_dust", "No AGB dust", "#8c564b"),
    ("lw02_o_rich", "LW02 O-rich TP-AGB", "#e377c2"),
    ("no_agb_all", "No AGB (all off)", "#7f7f7f"),
]

IMF_INFO = {"kroupa": "Kroupa", "chabrier": "Chabrier"}


def main(in_path, spec_label):
    nu.set_style()
    data = np.load(in_path)
    wave_nm = data["wavelength_A"] / 10.0
    broad_mask = (wave_nm >= nu.BROAD_RANGE[0]) & (wave_nm <= nu.BROAD_RANGE[1])
    nir_mask = (wave_nm >= nu.NIR_RANGE[0]) & (wave_nm <= nu.NIR_RANGE[1])

    for imf_key, imf_label in IMF_INFO.items():
        curves = [
            (label, color, data[f"{imf_key}__{key}"])
            for key, label, color in SCENARIOS
        ]

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.5))
        ax_broad, ax_n1, ax_n2 = axes

        n1_list, n2_list = [], []
        for label, color, flux in curves:
            n1 = nu.norm_1(flux, wave_nm)[nir_mask]
            n2 = nu.norm_2(flux, wave_nm)[nir_mask]
            n1_list.append(n1)
            n2_list.append(n2)

            ax_broad.plot(
                wave_nm[broad_mask], flux[broad_mask], color=color, lw=1.0, label=label
            )
            ax_n1.plot(wave_nm[nir_mask], n1, color=color, lw=1.0)
            ax_n2.plot(wave_nm[nir_mask], n2, color=color, lw=1.0)

        spec = f"MIST | {imf_label} | {spec_label} | 1 Gyr | $Z_\\odot$"
        fig.suptitle(spec, fontsize=14, x=0.5, ha="center", y=0.94)

        # Panel 1: broad SED, linear wavelength, tight log-flux limits.
        ax_broad.set_yscale("log")
        ax_broad.set_xlim(*nu.BROAD_RANGE)
        broad_flat = np.concatenate([f[broad_mask] for _, _, f in curves])
        ax_broad.set_ylim(broad_flat.min() * 0.8, broad_flat.max() * 1.3)
        ax_broad.set_ylabel(r"$F_\nu$  [$L_\odot\,{\rm Hz}^{-1}$]")
        ax_broad.set_title("Broad SED (270\u20131900 nm)")

        # Panels 2 & 3: NIR zooms, shared tight y-limits.
        lo = min(min(n.min() for n in n1_list), min(n.min() for n in n2_list))
        hi = max(max(n.max() for n in n1_list), max(n.max() for n in n2_list))
        pad = 0.06 * (hi - lo)
        for ax in (ax_n1, ax_n2):
            nu.shade_windows(ax)
            ax.set_xlim(*nu.NIR_RANGE)
            ax.set_ylim(lo - pad, hi + pad)
            ax.axhline(1.0, color="0.55", lw=0.7, ls="--")

        ax_n1.set_ylabel(r"$F_\nu$ / median(1495\u20131535 nm)")
        ax_n1.set_title("NIR zoom \u2014 norm$_1$ (blue-window median)")

        ax_n2.set_xlabel("Rest-frame wavelength [nm]")
        ax_n2.set_ylabel(r"$F_\nu$ / continuum")
        ax_n2.set_title("NIR zoom \u2014 norm$_2$ (blue\u2013red continuum)")

        handles, labels = ax_broad.get_legend_handles_labels()
        ax_n2.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=4, frameon=False, handlelength=1.6, columnspacing=1.2,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.955))
        tag = spec_label.lower().replace("c3k-", "")
        out = OUTPUT_DIR / f"nir_comparison_{tag}_{imf_key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot SSP NIR comparison figures.")
    parser.add_argument("--in", dest="in_path", default=str(OUTPUT_DIR / "ssp_spectra.npz"))
    parser.add_argument("--spec-label", default="C3K-LR")
    args = parser.parse_args()
    main(args.in_path, args.spec_label)
