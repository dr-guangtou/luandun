"""Response of the SSP spectrum to the `agb` and `pagb` weight knobs.

`agb` (mod_gb.f90) multiplies the IMF weight of TP-AGB stars (phase=5);
`pagb` multiplies the weight of post-AGB stars (phase=6). Both are pure
multiplicative weights, so the spectrum is linear in each knob.

Outputs:
  response_agb.png      -- broad SED + NIR (absolute & norm_2) vs agb
  response_pagb.png     -- same vs pagb
  phase_contribution.png-- linearity curve + pure-phase difference spectra
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fsps
import norm_utils as nu

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SOLAR_ZMET = 11
AGE_GYR = 1.0
IMF_TYPE = 2  # Kroupa

AGB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PAGB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def build(agb=1.0, pagb=1.0):
    sp = fsps.StellarPopulation(imf_type=IMF_TYPE, zmet=SOLAR_ZMET)
    sp.params["agb"] = agb
    sp.params["pagb"] = pagb
    return sp.get_spectrum(tage=AGE_GYR, peraa=False)


def spec_label():
    return fsps.StellarPopulation().libraries[1].decode().upper()


def knob_colors(values):
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=2.0)
    return [cmap(norm(v)) for v in values]


def response_figure(wave_nm, spectra, knob_values, knob_name, outname):
    nu.set_style()
    lib = spec_label()
    broad_mask = (wave_nm >= nu.BROAD_RANGE[0]) & (wave_nm <= nu.BROAD_RANGE[1])
    nir_mask = (wave_nm >= nu.NIR_RANGE[0]) & (wave_nm <= nu.NIR_RANGE[1])
    colors = knob_colors(knob_values)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.5))
    ax_broad, ax_abs, ax_norm = axes

    for v, s, c in zip(knob_values, spectra, colors):
        ax_broad.plot(wave_nm[broad_mask], s[broad_mask], color=c, lw=1.0,
                      label=f"{knob_name} = {v:g}")
        ax_abs.plot(wave_nm[nir_mask], s[nir_mask], color=c, lw=1.0)
        ax_norm.plot(wave_nm[nir_mask], nu.norm_2(s, wave_nm)[nir_mask], color=c, lw=1.0)

    fig.suptitle(f"MIST | Kroupa | {lib} | 1 Gyr | $Z_\\odot$   \u2014   {knob_name} response",
                 fontsize=14, x=0.5, ha="center", y=0.94)

    ax_broad.set_xscale("log")
    ax_broad.set_yscale("log")
    ax_broad.set_xlim(*nu.BROAD_RANGE)
    bf = np.concatenate([s[broad_mask] for s in spectra])
    ax_broad.set_ylim(bf.min() * 0.8, bf.max() * 1.3)
    ax_broad.set_ylabel(r"$F_\nu$  [$L_\odot\,{\rm Hz}^{-1}$]")
    ax_broad.set_title("Broad SED (270\u20131900 nm)")

    ax_abs.set_xlim(*nu.NIR_RANGE)
    af = np.concatenate([s[nir_mask] for s in spectra])
    ax_abs.set_ylim(af.min() * 0.9, af.max() * 1.05)
    ax_abs.set_ylabel(r"$F_\nu$  [$L_\odot\,{\rm Hz}^{-1}$]")
    ax_abs.set_title("NIR zoom \u2014 absolute")

    for ax in (ax_abs, ax_norm):
        nu.shade_windows(ax)
    ax_norm.set_xlim(*nu.NIR_RANGE)
    nf = np.concatenate([nu.norm_2(s, wave_nm)[nir_mask] for s in spectra])
    ax_norm.set_ylim(nf.min() - 0.02, nf.max() + 0.02)
    ax_norm.axhline(1.0, color="0.55", lw=0.7, ls="--")
    ax_norm.set_xlabel("Rest-frame wavelength [nm]")
    ax_norm.set_ylabel(r"$F_\nu$ / continuum")
    ax_norm.set_title("NIR zoom \u2014 norm$_2$ (1.6 \u00b5m bump shape)")

    handles, labels = ax_broad.get_legend_handles_labels()
    ax_norm.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                   ncol=4, frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(outname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}")


def main():
    nu.set_style()
    w, _ = build(1.0, 1.0)
    wave_nm = w / 10.0

    agb_spectra = [build(agb=v, pagb=1.0)[1] for v in AGB_GRID]
    pagb_spectra = [build(agb=1.0, pagb=v)[1] for v in PAGB_GRID]

    response_figure(wave_nm, agb_spectra, AGB_GRID, "agb",
                    OUTPUT_DIR / "response_agb.png")
    response_figure(wave_nm, pagb_spectra, PAGB_GRID, "pagb",
                    OUTPUT_DIR / "response_pagb.png")

    # ---- Figure: linearity curve + pure-phase difference spectra ----
    blue = (wave_nm >= nu.BLUE_WINDOW[0]) & (wave_nm <= nu.BLUE_WINDOW[1])
    nir_flux_agb = [np.median(s[blue]) for s in agb_spectra]
    nir_flux_pagb = [np.median(s[blue]) for s in pagb_spectra]

    s_tpagb = agb_spectra[AGB_GRID.index(1.0)] - agb_spectra[AGB_GRID.index(0.0)]
    s_postagb = pagb_spectra[PAGB_GRID.index(1.0)] - pagb_spectra[PAGB_GRID.index(0.0)]
    # reference flux at 1.6 um to normalize the difference spectra
    norm_tp = np.interp(16000.0, w, s_tpagb)
    norm_pa = np.interp(16000.0, w, s_postagb)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.5))
    ax_lin, ax_sed = axes

    # linearity panel
    ax_lin.plot(AGB_GRID, nir_flux_agb, "o-", color="#d62728", label="agb (TP-AGB)")
    ax_lin.plot(PAGB_GRID, nir_flux_pagb, "s-", color="#1f77b4", label="pagb (post-AGB)")
    # reference: line through agb=0 and agb=1
    ref = lambda v: nir_flux_agb[0] + v * (nir_flux_agb[AGB_GRID.index(1.0)] - nir_flux_agb[0])
    vv = np.linspace(0, 2, 50)
    ax_lin.plot(vv, ref(vv), "--", color="0.5", lw=1.0, label="linear reference")
    ax_lin.set_xlabel("knob value")
    ax_lin.set_ylabel(r"median $F_\nu$ (1495\u20131535 nm)  [$L_\odot\,{\rm Hz}^{-1}$]")
    ax_lin.set_title("NIR flux response is linear in agb, flat in pagb")
    ax_lin.legend(frameon=False, loc="upper left")

    # pure-phase difference spectra
    wide = (wave_nm >= 100.0) & (wave_nm <= 3e4)
    ax_sed.plot(wave_nm[wide] / 1e3, s_tpagb[wide] / norm_tp, color="#d62728",
                lw=1.2, label="TP-AGB  [S(agb=1) \u2212 S(agb=0)]")
    ax_sed.plot(wave_nm[wide] / 1e3, s_postagb[wide] / norm_pa, color="#1f77b4",
                lw=1.2, label="post-AGB  [S(pagb=1) \u2212 S(pagb=0)]")
    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")
    ax_sed.set_xlabel("Wavelength [\u00b5m]")
    ax_sed.set_ylabel("relative contribution (norm. at 1.6 \u00b5m)")
    ax_sed.set_title("Pure-phase contribution to the SSP spectrum")
    ax_sed.axvline(1.6, color="0.5", lw=0.7, ls=":")
    ax_sed.legend(frameon=False, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / "phase_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'phase_contribution.png'}")

    print(f"\nNIR (1495-1535 nm) median F_nu vs agb: {['%.3e' % f for f in nir_flux_agb]}")
    print(f"NIR (1495-1535 nm) median F_nu vs pagb: {['%.3e' % f for f in nir_flux_pagb]}")


if __name__ == "__main__":
    main()
