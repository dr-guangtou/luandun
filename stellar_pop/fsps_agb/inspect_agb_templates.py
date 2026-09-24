"""Inspect the AGB spectral templates directly (native resolution).

Reads the O-rich (Lancon & Mouhcine 2002) and C-rich (Aringer 2009) templates
from $SPS_HOME/SPECTRA/AGB_spectra and plots, per chemical type, three stacked
panels: broad SED (270-1900 nm, linear wavelength), NIR zoom with norm_1, and
NIR zoom with norm_2. Line colors follow the Teff sequence (blue = cold,
red = hot). The broad SED x-axis starts at the templates' blue-end cut-off.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import norm_utils as nu

SPS_HOME = os.environ["SPS_HOME"]
AGB_DIR = Path(SPS_HOME) / "SPECTRA" / "AGB_spectra"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_template(spec_name, n_cols):
    """Return (wavelength_A, spectra[n_cols x n_wave])."""
    data = np.loadtxt(AGB_DIR / spec_name)
    return data[:, 0], data[:, 1 : n_cols + 1].T


def teff_o_rich_solar():
    """Solar-metallicity Teff for the 9 O-rich spectra (logZ/Zsol = 0.0 column)."""
    rows = np.loadtxt(AGB_DIR / "Orich.teff")
    zs = rows[0, 1:]  # row 0 is the logZ/Zsol grid; col 0 is 'color'
    solar_col = np.argmin(np.abs(zs - 0.0)) + 1
    return rows[1:, solar_col]


def plot_templates(wave_a, spectra, teff, title, outname):
    wave_nm = wave_a / 10.0
    colors = nu.teff_colors(teff)
    order = np.argsort(teff)

    broad_mask = (wave_nm >= nu.BROAD_RANGE[0]) & (wave_nm <= nu.BROAD_RANGE[1])
    nir_mask = (wave_nm >= nu.NIR_RANGE[0]) & (wave_nm <= nu.NIR_RANGE[1])

    # Blue-end cut-off of the templates, then peak-normalize for display.
    cutoff = max(nu.blue_cutoff(wave_nm[broad_mask], [s[broad_mask] for s in spectra]),
                 nu.BROAD_RANGE[0])
    plot_mask = broad_mask & (wave_nm >= cutoff)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.5))
    ax_broad, ax_n1, ax_n2 = axes

    n1_list, n2_list = [], []
    broad_flat = []
    for i in order:
        flux = spectra[i]
        peak = flux[broad_mask].max()
        color = colors[i]

        ax_broad.plot(wave_nm[plot_mask], flux[plot_mask] / peak,
                      color=color, lw=1.0, label=f"Teff = {teff[i]:.0f} K")
        broad_flat.append(flux[plot_mask] / peak)

        n1 = nu.norm_1(flux, wave_nm)[nir_mask]
        n2 = nu.norm_2(flux, wave_nm)[nir_mask]
        n1_list.append(n1)
        n2_list.append(n2)
        ax_n1.plot(wave_nm[nir_mask], n1, color=color, lw=1.0)
        ax_n2.plot(wave_nm[nir_mask], n2, color=color, lw=1.0)

    fig.suptitle(f"AGB templates \u2014 {title}", fontsize=14, x=0.5, ha="center", y=0.94)

    # Panel 1: broad SED with tight limits.
    ax_broad.set_yscale("log")
    ax_broad.set_xlim(cutoff, nu.BROAD_RANGE[1])
    bf = np.concatenate(broad_flat)
    ax_broad.set_ylim(bf[bf > 0].min() * 0.8, bf.max() * 1.3)
    ax_broad.set_ylabel("relative flux (peak = 1)")
    ax_broad.set_title(f"Broad SED ({cutoff:.0f}\u20131900 nm)")

    # Panels 2 & 3: NIR zooms, shared tight y-limits.
    lo = min(min(n.min() for n in n1_list), min(n.min() for n in n2_list))
    hi = max(max(n.max() for n in n1_list), max(n.max() for n in n2_list))
    pad = 0.06 * (hi - lo)
    for ax in (ax_n1, ax_n2):
        nu.shade_windows(ax)
        ax.set_xlim(*nu.NIR_RANGE)
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(1.0, color="0.55", lw=0.7, ls="--")

    ax_n1.set_ylabel("flux / median(1495\u20131535 nm)")
    ax_n1.set_title("NIR zoom \u2014 norm$_1$ (blue-window median)")

    ax_n2.set_xlabel("Wavelength [nm]")
    ax_n2.set_ylabel("flux / continuum")
    ax_n2.set_title("NIR zoom \u2014 norm$_2$ (blue\u2013red continuum)")

    handles, labels = ax_broad.get_legend_handles_labels()
    ax_n2.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=5, frameon=False, handlelength=1.6, columnspacing=1.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(outname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}")


def main():
    nu.set_style()

    wave_o, spec_o = load_template("Orich.spec", 9)
    teff_o = teff_o_rich_solar()
    plot_templates(
        wave_o, spec_o, teff_o,
        "O-rich (Lancon & Mouhcine 2002)",
        OUTPUT_DIR / "agb_templates_orich_nir.png",
    )

    wave_c, spec_c = load_template("Crich_Aringer.spec", 9)
    teff_c = np.loadtxt(AGB_DIR / "Crich_Aringer.teff")
    plot_templates(
        wave_c, spec_c, teff_c,
        "C-rich (Aringer et al. 2009)",
        OUTPUT_DIR / "agb_templates_aringer_nir.png",
    )


if __name__ == "__main__":
    main()
