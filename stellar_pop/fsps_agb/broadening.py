"""Gaussian velocity broadening of SSP grids on a uniform log-wavelength grid.

The convolution acts on flux per unit log wavelength (lambda F_lambda, equal
to nu F_nu up to a constant), following the ProGeny resolution study. The
native library resolution is subtracted in quadrature before smoothing so
that each product has a known total resolution.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

SPEED_OF_LIGHT_KM_S = 299792.458
VELOCITY_STEP_KM_S = 30.0
SIGMA_GALAXY_KM_S = 300.0
R100_FWHM = 100.0
R100_RANGE_A = (12500.0, 21000.0)
SPLICE_A = 10000.0
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def log_wavelength_grid(wave_min_a, wave_max_a, velocity_step_km_s=VELOCITY_STEP_KM_S):
    step = velocity_step_km_s / SPEED_OF_LIGHT_KM_S
    log_wave = np.arange(np.log(wave_min_a), np.log(wave_max_a) + 0.5 * step, step)
    return np.exp(log_wave[np.exp(log_wave) <= wave_max_a])


def resample_flux(wave_a, flux, target_wave_a):
    flux = np.asarray(flux)
    flat = flux.reshape(-1, flux.shape[-1])
    out = np.stack([np.interp(target_wave_a, wave_a, row) for row in flat])
    return out.reshape(flux.shape[:-1] + (target_wave_a.size,))


def gaussian_broaden(log_wave_a, flux_nu, sigma_km_s):
    step_km_s = np.log(log_wave_a[1] / log_wave_a[0]) * SPEED_OF_LIGHT_KM_S
    per_log_wavelength = flux_nu / log_wave_a
    smoothed = gaussian_filter1d(
        per_log_wavelength, sigma_km_s / step_km_s, axis=-1, mode="nearest", truncate=6.0
    )
    return smoothed * log_wave_a


def added_sigma_km_s(target_sigma_km_s, native_sigma_km_s):
    if target_sigma_km_s <= native_sigma_km_s:
        raise ValueError(
            f"target sigma {target_sigma_km_s} km/s is not above native {native_sigma_km_s} km/s"
        )
    return float(np.sqrt(target_sigma_km_s**2 - native_sigma_km_s**2))


def target_sigma_km_s(product):
    if product == "sigma300":
        return SIGMA_GALAXY_KM_S
    if product == "r100":
        instrument = SPEED_OF_LIGHT_KM_S / (FWHM_PER_SIGMA * R100_FWHM)
        return float(np.hypot(instrument, SIGMA_GALAXY_KM_S))
    raise ValueError(f"unknown product {product!r}")


def _broaden_segment(grid, wave_lo, wave_hi, sigma_target, padding_km_s=6000.0):
    """Broaden one segment of constant native resolution with padding on both sides."""
    pad = np.exp(padding_km_s / SPEED_OF_LIGHT_KM_S)
    lo = max(grid.wave_a[0], wave_lo / pad)
    hi = min(grid.wave_a[-1], wave_hi * pad)
    log_wave = log_wavelength_grid(lo, hi)
    native = np.median(grid.native_sigma_km_s[(grid.wave_a >= wave_lo) & (grid.wave_a <= wave_hi)])
    flux = resample_flux(grid.wave_a, grid.flux_nu, log_wave)
    flux = gaussian_broaden(log_wave, flux, added_sigma_km_s(sigma_target, native))
    keep = (log_wave >= wave_lo) & (log_wave <= wave_hi)
    return log_wave[keep], flux[..., keep], float(native)


def make_resolution_product(grid, product):
    sigma_target = target_sigma_km_s(product)
    if product == "sigma300":
        segments = [(grid.wave_a[0], SPLICE_A), (SPLICE_A, grid.wave_a[-1])]
    else:
        segments = [R100_RANGE_A]
    waves, fluxes, natives = [], [], []
    for wave_lo, wave_hi in segments:
        wave, flux, native = _broaden_segment(grid, wave_lo, wave_hi, sigma_target)
        waves.append(wave)
        fluxes.append(flux)
        natives.append(np.full(wave.size, native))
    wave = np.concatenate(waves)
    order = np.argsort(wave)
    provenance = dict(grid.provenance) | {
        "product": product,
        "total_sigma_km_s": sigma_target,
        "velocity_step_km_s": VELOCITY_STEP_KM_S,
        "segments_a": [list(map(float, segment)) for segment in segments],
    }
    return replace(
        grid,
        wave_a=wave[order],
        flux_nu=np.concatenate(fluxes, axis=-1)[..., order],
        native_sigma_km_s=np.concatenate(natives)[order],
        product=product,
        provenance=provenance,
    )


if __name__ == "__main__":
    import time

    from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid, save_ssp_grid

    native_grid = load_ssp_grid(DEFAULT_GRID_DIR, "native")
    for name in ("sigma300", "r100"):
        start = time.perf_counter()
        path = save_ssp_grid(make_resolution_product(native_grid, name), DEFAULT_GRID_DIR)
        print(f"wrote {path} in {time.perf_counter() - start:.1f} s")
