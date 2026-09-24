"""D4000, HdeltaA and the 1.6 micron H-minus bump on vacuum-wavelength model spectra.

Inputs are F_nu (any units) on a vacuum wavelength grid in Angstrom. Bands
defined in air (Lick/IDS, Bruzual D4000) are converted to vacuum with the
Morton (1991) relation used by FSPS. The pseudo-continuum is the straight
line through the band-mean F_lambda of the blue and red windows, anchored
at the band midpoints, following the Lick/IDS convention.
"""

import numpy as np

SPEED_OF_LIGHT_A_PER_S = 2.99792458e18


def air_to_vacuum(wave_a):
    wave_a = np.asarray(wave_a, dtype=float)
    sigma2 = (1e4 / wave_a) ** 2
    factor = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2)
    return np.where(wave_a < 2000.0, wave_a, wave_a * factor)


def _vacuum_band(lower_air_a, upper_air_a):
    return tuple(air_to_vacuum(np.array([lower_air_a, upper_air_a])))


D4000_BANDS_A = {"blue": _vacuum_band(3750.0, 3950.0), "red": _vacuum_band(4050.0, 4250.0)}
HDELTA_A_BANDS_A = {
    "blue": _vacuum_band(4041.60, 4079.75),
    "feature": _vacuum_band(4083.50, 4122.25),
    "red": _vacuum_band(4128.50, 4161.00),
}
H_MINUS_BANDS_A = {
    "blue": (14940.0, 15390.0),
    "feature": (15700.0, 17340.0),
    "red": (17460.0, 17910.0),
}


def flux_nu_to_flux_lambda(wave_a, flux_nu):
    return flux_nu * SPEED_OF_LIGHT_A_PER_S / np.asarray(wave_a) ** 2


def _interpolate(wave_a, flux, target_a):
    """Linear interpolation along the last axis for flux of shape (..., n_pix)."""
    flux = np.asarray(flux)
    flat = flux.reshape(-1, flux.shape[-1])
    values = np.stack([np.interp(target_a, wave_a, row) for row in flat])
    return values.reshape(flux.shape[:-1] + (len(target_a),))


def band_mean(wave_a, flux, lower_a, upper_a):
    """Mean of flux over [lower_a, upper_a] by trapezoid integration on the pixel
    grid augmented with the exact band edges."""
    inside = (wave_a > lower_a) & (wave_a < upper_a)
    nodes = np.concatenate([[lower_a], wave_a[inside], [upper_a]])
    values = _interpolate(wave_a, flux, nodes)
    return np.trapezoid(values, nodes, axis=-1) / (upper_a - lower_a)


def _feature_ratio_mean(wave_a, flux_lambda, bands):
    """Mean over the feature band of F_lambda / F_c, with F_c the line through the
    blue and red band means at their midpoints. Uses 4-point Gauss-Legendre
    quadrature inside every pixel interval, as in the ProGeny resolution study."""
    blue_mean = band_mean(wave_a, flux_lambda, *bands["blue"])
    red_mean = band_mean(wave_a, flux_lambda, *bands["red"])
    blue_mid = 0.5 * sum(bands["blue"])
    red_mid = 0.5 * sum(bands["red"])
    lower, upper = bands["feature"]
    inside = (wave_a > lower) & (wave_a < upper)
    edges = np.concatenate([[lower], wave_a[inside], [upper]])
    nodes, weights = np.polynomial.legendre.leggauss(4)
    half_width = 0.5 * np.diff(edges)
    centers = edges[:-1] + half_width
    target = (centers[:, None] + half_width[:, None] * nodes).ravel()
    quad_weights = (half_width[:, None] * weights / (upper - lower)).ravel()
    fraction = (target - blue_mid) / (red_mid - blue_mid)
    continuum = blue_mean[..., None] * (1.0 - fraction) + red_mean[..., None] * fraction
    return (_interpolate(wave_a, flux_lambda, target) / continuum) @ quad_weights


def d4000(wave_a, flux_nu):
    flux_nu = np.asarray(flux_nu, dtype=float)
    return band_mean(wave_a, flux_nu, *D4000_BANDS_A["red"]) / band_mean(
        wave_a, flux_nu, *D4000_BANDS_A["blue"]
    )


def hdelta_a(wave_a, flux_nu):
    flux_lambda = flux_nu_to_flux_lambda(wave_a, np.asarray(flux_nu, dtype=float))
    lower, upper = HDELTA_A_BANDS_A["feature"]
    return (upper - lower) * (1.0 - _feature_ratio_mean(wave_a, flux_lambda, HDELTA_A_BANDS_A))


def h_minus_bump(wave_a, flux_nu):
    flux_lambda = flux_nu_to_flux_lambda(wave_a, np.asarray(flux_nu, dtype=float))
    return -2.5 * np.log10(_feature_ratio_mean(wave_a, flux_lambda, H_MINUS_BANDS_A))


def measure_all(wave_a, flux_nu):
    return {
        "d4000": d4000(wave_a, flux_nu),
        "hdelta_a": hdelta_a(wave_a, flux_nu),
        "h_minus_bump": h_minus_bump(wave_a, flux_nu),
    }
