"""Shared helpers for spectrum comparison figures.

Wavelengths are expressed in nm throughout.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Lick/IDS-style sidebands (nm).
BLUE_WINDOW = (1495.0, 1535.0)
RED_WINDOW = (1750.0, 1795.0)
BLUE_CENTER = 0.5 * (BLUE_WINDOW[0] + BLUE_WINDOW[1])
RED_CENTER = 0.5 * (RED_WINDOW[0] + RED_WINDOW[1])

BROAD_RANGE = (270.0, 1900.0)  # nm
NIR_RANGE = (1400.0, 1800.0)  # nm


def norm_1(flux, wave_nm):
    """Normalize by the median flux in the blue window (1495-1535 nm)."""
    mask = (wave_nm >= BLUE_WINDOW[0]) & (wave_nm <= BLUE_WINDOW[1])
    return flux / np.median(flux[mask])


def norm_2(flux, wave_nm):
    """Continuum-normalize with a straight line through the blue & red medians.

    The line passes through the median fluxes of the blue (1495-1535 nm) and
    red (1750-1795 nm) windows, evaluated at each window's central wavelength.
    """
    blue_mask = (wave_nm >= BLUE_WINDOW[0]) & (wave_nm <= BLUE_WINDOW[1])
    red_mask = (wave_nm >= RED_WINDOW[0]) & (wave_nm <= RED_WINDOW[1])
    blue_flux = np.median(flux[blue_mask])
    red_flux = np.median(flux[red_mask])
    continuum = blue_flux + (red_flux - blue_flux) * (
        wave_nm - BLUE_CENTER
    ) / (RED_CENTER - BLUE_CENTER)
    return flux / continuum


def shade_windows(ax):
    """Shade the blue and red normalization windows."""
    for lo, hi, color in (
        (BLUE_WINDOW[0], BLUE_WINDOW[1], "tab:blue"),
        (RED_WINDOW[0], RED_WINDOW[1], "tab:red"),
    ):
        ax.axvspan(lo, hi, color=color, alpha=0.08, lw=0)


def blue_cutoff(wave_nm, spectra, thresh=1e-4):
    """Smallest wavelength (nm) where any spectrum is non-negligible."""
    stack = np.vstack(spectra)
    rel = stack / stack.max()
    return float(wave_nm[np.any(rel > thresh, axis=0)].min())


def set_style():
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )


def teff_colors(teff):
    """Return a color per Teff value: blue (cold) -> red (hot)."""
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=teff.min(), vmax=teff.max())
    return [cmap(norm(t)) for t in teff]
