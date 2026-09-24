"""Q1: can models with and without TP-AGB contribution (agb=0 vs agb=2) be told apart?

SSP-level diagnostics (Steps 1-5) use the cached `native`, `sigma300` and `r100`
grids plus two extra FSPS builds with the empirical TP-AGB spectra
(`use_lw_tpagb=1`). Population-level diagnostics (Step 6) use the cached
520,000-row population table, restricted to `epoch_gyr >= 1.0` as required by
the Phase 3 measurement yardstick.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from broadening import make_resolution_product
from csp_integrate import agb_two_spectra
from spectral_indices import (
    H_MINUS_BANDS_A,
    band_mean,
    d4000,
    flux_nu_to_flux_lambda,
    h_minus_bump,
    hdelta_a,
)
from ssp_grid import DEFAULT_GRID_DIR, SspGrid, build_ssp, load_ssp_grid

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "analysis"
POPULATION_PATH = Path(__file__).resolve().parent / "output" / "population" / "indices.npz"

AGE_WINDOW_GYR = (0.1, 13.0)
YARDSTICK_D4000 = 0.05
YARDSTICK_HDELTA_A_ANGSTROM = 0.5
YARDSTICK_BUMP_MAG = (0.005, 0.010, 0.020)
CO_BANDHEAD_UM = (1.578, 1.598, 1.619, 1.640, 1.661, 1.684, 1.707)
NIR_ZOOM_RANGE_A = (1.3e4, 2.0e4)
STEP2_AGES_GYR = (0.3, 1.0, 3.0)
STEP3_AGES_GYR = (0.3, 1.0, 2.0, 5.0)
STEP4_RECORD_AGES_GYR = (0.3, 1.0, 2.0, 5.0)
STEP4_BLUE_WAVE_A = 4200.0
STEP4_RED_WAVE_A = 16000.0
PILOT_AGES_GYR = (0.3, 1.0, 5.0)
Z_COLORS = plt.cm.viridis(np.linspace(0.0, 0.9, 4))


def _nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def _age_gyr(grid):
    return 10.0**grid.log_age_yr / 1e9


def _solar_index(grid):
    return _nearest_index(grid.log_z_grid, 0.0)


def _pilot_grid(grid):
    """Solar metallicity only, 3 ages: keeps every downstream computation cheap."""
    z_idx = _solar_index(grid)
    age_idx = np.array([_nearest_index(_age_gyr(grid), target) for target in PILOT_AGES_GYR])
    return replace(
        grid,
        log_z_grid=grid.log_z_grid[z_idx : z_idx + 1],
        log_age_yr=grid.log_age_yr[age_idx],
        flux_nu=grid.flux_nu[z_idx : z_idx + 1][:, :, age_idx, :],
    )


def continuum_normalized_flux_lambda(wave_a, flux_lambda, bands):
    """F_lambda / F_c at every native pixel; F_c is the line through the blue and red
    band means anchored at the band midpoints (the `_feature_ratio_mean` pseudo-
    continuum construction in spectral_indices.py)."""
    blue_mean = band_mean(wave_a, flux_lambda, *bands["blue"])
    red_mean = band_mean(wave_a, flux_lambda, *bands["red"])
    blue_mid = 0.5 * sum(bands["blue"])
    red_mid = 0.5 * sum(bands["red"])
    fraction = (wave_a - blue_mid) / (red_mid - blue_mid)
    continuum = blue_mean[..., None] * (1.0 - fraction) + red_mean[..., None] * fraction
    return flux_lambda / continuum


def _out_path(out_dir, prefix, name):
    return out_dir / f"{prefix}{name}"


def _grid_uses_lw_tpagb(grids):
    return grids["native"].provenance.get("extra_params", {}).get("use_lw_tpagb") == 1


def _shade_bump_bands(axis):
    for name, color in (("blue", "tab:blue"), ("feature", "tab:green"), ("red", "tab:red")):
        axis.axvspan(
            H_MINUS_BANDS_A[name][0] / 1e4, H_MINUS_BANDS_A[name][1] / 1e4, color=color, alpha=0.08
        )


# ---------------------------------------------------------------------------
# Step 1: SSP-level deltas
# ---------------------------------------------------------------------------


def step1_ssp_delta_vs_age(grids, out_dir, make_figure, prefix=""):
    grid_s300 = grids["sigma300"]
    grid_r100 = grids["r100"]
    age_gyr = _age_gyr(grid_s300)
    window = (age_gyr >= AGE_WINDOW_GYR[0]) & (age_gyr <= AGE_WINDOW_GYR[1])

    flux0_s300 = grid_s300.flux_nu[:, 0]
    flux2_s300 = agb_two_spectra(grid_s300.flux_nu[:, 0], grid_s300.flux_nu[:, 1])
    flux0_r100 = grid_r100.flux_nu[:, 0]
    flux2_r100 = agb_two_spectra(grid_r100.flux_nu[:, 0], grid_r100.flux_nu[:, 1])

    agb0_values = {
        "d4000": d4000(grid_s300.wave_a, flux0_s300),
        "hdelta_a": hdelta_a(grid_s300.wave_a, flux0_s300),
        "h_minus_bump_sigma300": h_minus_bump(grid_s300.wave_a, flux0_s300),
        "h_minus_bump_r100": h_minus_bump(grid_r100.wave_a, flux0_r100),
    }
    deltas = {
        "d4000": d4000(grid_s300.wave_a, flux2_s300) - agb0_values["d4000"],
        "hdelta_a": hdelta_a(grid_s300.wave_a, flux2_s300) - agb0_values["hdelta_a"],
        "h_minus_bump_sigma300": (
            h_minus_bump(grid_s300.wave_a, flux2_s300) - agb0_values["h_minus_bump_sigma300"]
        ),
        "h_minus_bump_r100": (
            h_minus_bump(grid_r100.wave_a, flux2_r100) - agb0_values["h_minus_bump_r100"]
        ),
    }
    units = {
        "d4000": "",
        "hdelta_a": "_angstrom",
        "h_minus_bump_sigma300": "_mag",
        "h_minus_bump_r100": "_mag",
    }

    summary = {}
    for name, delta in deltas.items():
        unit = units[name]
        abs_delta = np.abs(delta)
        windowed = np.where(window[None, :], abs_delta, -np.inf)
        overall_idx = np.unravel_index(np.argmax(windowed), windowed.shape)
        by_z_max = np.max(windowed, axis=1)
        by_z_age_idx = np.argmax(windowed, axis=1)
        spread = np.max(agb0_values[name], axis=0) - np.min(agb0_values[name], axis=0)
        spread_windowed = np.where(window, spread, -np.inf)
        spread_idx = int(np.argmax(spread_windowed))

        summary[f"max_abs_delta_{name}{unit}"] = float(windowed[overall_idx])
        summary[f"max_abs_delta_{name}_age_gyr"] = float(age_gyr[overall_idx[1]])
        summary[f"max_abs_delta_{name}{unit}_by_log_z"] = {
            f"{z:+.2f}": float(v) for z, v in zip(grid_s300.log_z_grid, by_z_max, strict=True)
        }
        summary[f"max_abs_delta_{name}_age_gyr_by_log_z"] = {
            f"{z:+.2f}": float(age_gyr[idx])
            for z, idx in zip(grid_s300.log_z_grid, by_z_age_idx, strict=True)
        }
        summary[f"max_agb0_metallicity_spread_{name}{unit}"] = float(spread_windowed[spread_idx])
        summary[f"max_agb0_metallicity_spread_{name}_age_gyr"] = float(age_gyr[spread_idx])

    if make_figure:
        _figure_step1(grid_s300, age_gyr, deltas, agb0_values, out_dir, prefix)
    return summary


def _figure_step1(grid, age_gyr, deltas, agb0_values, out_dir, prefix=""):
    figure, axes = plt.subplots(4, 1, figsize=(8, 15), sharex=True, layout="constrained")
    figure.get_layout_engine().set(h_pad=0.08, hspace=0.04)
    row_names = ("d4000", "hdelta_a", "h_minus_bump_sigma300", "h_minus_bump_r100")
    row_labels = (
        r"$\Delta$ D4000",
        r"$\Delta$ H$\delta_A$ [$\AA$]",
        r"$\Delta$ bump, $\sigma$300 [mag]",
        r"$\Delta$ bump, R=100 [mag]",
    )
    yardsticks = {
        "d4000": (YARDSTICK_D4000,),
        "hdelta_a": (YARDSTICK_HDELTA_A_ANGSTROM,),
        "h_minus_bump_sigma300": YARDSTICK_BUMP_MAG,
        "h_minus_bump_r100": YARDSTICK_BUMP_MAG,
    }
    for axis, name, label in zip(axes, row_names, row_labels, strict=True):
        spread = np.max(agb0_values[name], axis=0) - np.min(agb0_values[name], axis=0)
        axis.fill_between(
            age_gyr, -spread / 2, spread / 2, color="0.85", label="agb0 metallicity spread / 2"
        )
        for i_z, log_z in enumerate(grid.log_z_grid):
            axis.plot(
                age_gyr,
                deltas[name][i_z],
                color=Z_COLORS[i_z % len(Z_COLORS)],
                label=f"log Z = {log_z:+.2f}",
            )
        for level in yardsticks[name]:
            axis.axhline(level, color="0.4", ls="--", lw=0.8)
            axis.axhline(-level, color="0.4", ls="--", lw=0.8)
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_ylabel(label, fontsize=10)
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.01, 20.0)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    axes[-1].set_xlabel("age [Gyr]")
    figure.savefig(_out_path(out_dir, prefix, "q1_ssp_delta_vs_age.png"), dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Step 2: component spectrum diagnostic
# ---------------------------------------------------------------------------


def step2_component_spectrum(grids, out_dir, make_figure, prefix=""):
    grid = grids["sigma300"]
    z_idx = _solar_index(grid)
    age_gyr = _age_gyr(grid)
    flux_agb0 = grid.flux_nu[z_idx, 0]
    flux_agb1 = grid.flux_nu[z_idx, 1]
    component = flux_agb1 - flux_agb0

    flux_lambda_component = flux_nu_to_flux_lambda(grid.wave_a, component)
    flux_lambda_agb1 = flux_nu_to_flux_lambda(grid.wave_a, flux_agb1)
    component_band = band_mean(grid.wave_a, flux_lambda_component, *H_MINUS_BANDS_A["feature"])
    agb1_band = band_mean(grid.wave_a, flux_lambda_agb1, *H_MINUS_BANDS_A["feature"])
    fraction = np.divide(
        component_band, agb1_band, out=np.full_like(component_band, np.nan), where=agb1_band != 0
    )

    positive = component_band > 0
    bump_component = np.full(age_gyr.shape, np.nan)
    bump_component[positive] = h_minus_bump(grid.wave_a, component[positive])

    window = (age_gyr >= AGE_WINDOW_GYR[0]) & (age_gyr <= AGE_WINDOW_GYR[1])
    valid = window & positive
    peak_idx = int(np.nanargmax(np.where(window, fraction, np.nan)))
    summary = {
        "component_bump_min_mag": float(np.nanmin(bump_component[valid])) if valid.any() else None,
        "component_bump_max_mag": float(np.nanmax(bump_component[valid])) if valid.any() else None,
        "peak_tpagb_light_fraction": float(fraction[peak_idx]),
        "peak_tpagb_light_fraction_age_gyr": float(age_gyr[peak_idx]),
    }

    if make_figure:
        _figure_step2(grid, age_gyr, bump_component, fraction, component, out_dir, prefix)
    return summary


def _figure_step2(grid, age_gyr, bump_component, fraction, component, out_dir, prefix=""):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    left = axes[0]
    left.plot(age_gyr, bump_component, color="tab:red")
    left.set_xscale("log")
    left.set_xlim(0.01, 20.0)
    left.set_xlabel("age [Gyr]")
    left.set_ylabel("component bump [mag]", color="tab:red", fontsize=10)
    left.tick_params(axis="y", colors="tab:red")
    left.invert_yaxis()
    twin = left.twinx()
    twin.plot(age_gyr, fraction, color="tab:blue")
    twin.set_ylabel("TP-AGB fraction of feature light", color="tab:blue", fontsize=10)
    twin.tick_params(axis="y", colors="tab:blue")

    right = axes[1]
    zoom = (grid.wave_a >= NIR_ZOOM_RANGE_A[0]) & (grid.wave_a <= NIR_ZOOM_RANGE_A[1])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(STEP2_AGES_GYR)))
    for target_age, color in zip(STEP2_AGES_GYR, colors, strict=True):
        idx = _nearest_index(age_gyr, target_age)
        flux_lambda = flux_nu_to_flux_lambda(grid.wave_a, component[idx])
        normalized = continuum_normalized_flux_lambda(grid.wave_a, flux_lambda, H_MINUS_BANDS_A)
        right.plot(
            grid.wave_a[zoom] / 1e4, normalized[zoom], color=color, label=f"{age_gyr[idx]:.2f} Gyr"
        )
    _shade_bump_bands(right)
    right.set_xlabel(r"rest wavelength [$\mu$m]")
    right.set_ylabel(r"norm. $F_\lambda$, S(1)$-$S(0)", fontsize=10)
    right.legend(frameon=False, fontsize=8)
    figure.savefig(_out_path(out_dir, prefix, "q1_component_spectrum.png"), dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Step 3: NIR spectra QA
# ---------------------------------------------------------------------------


def step3_nir_spectra_qa(grids, out_dir, prefix=""):
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, layout="constrained")
    for row, product in enumerate(("sigma300", "r100")):
        grid = grids[product]
        z_idx = _solar_index(grid)
        age_gyr = _age_gyr(grid)
        zoom = (grid.wave_a >= NIR_ZOOM_RANGE_A[0]) & (grid.wave_a <= NIR_ZOOM_RANGE_A[1])
        for col, target_age in enumerate(STEP3_AGES_GYR):
            idx = _nearest_index(age_gyr, target_age)
            flux_agb0 = grid.flux_nu[z_idx, 0, idx]
            flux_agb1 = grid.flux_nu[z_idx, 1, idx]
            flux_agb2 = agb_two_spectra(flux_agb0, flux_agb1)
            ratio = flux_agb2 / flux_agb0
            axis = axes[row, col]
            twin = axis.twinx()
            twin.plot(grid.wave_a[zoom] / 1e4, ratio[zoom], color="0.55", lw=1.0)
            twin.set_ylabel("S(agb2)/S(agb0)", color="0.4", fontsize=8)
            twin.tick_params(axis="y", colors="0.4", labelsize=7)
            for agb_key, flux, color in (
                ("agb0", flux_agb0, "#1f77b4"),
                ("agb2", flux_agb2, "#d62728"),
            ):
                flux_lambda = flux_nu_to_flux_lambda(grid.wave_a, flux)
                normalized = continuum_normalized_flux_lambda(
                    grid.wave_a, flux_lambda, H_MINUS_BANDS_A
                )
                axis.plot(
                    grid.wave_a[zoom] / 1e4, normalized[zoom], color=color, lw=1.2, label=agb_key
                )
            _shade_bump_bands(axis)
            y_lo, y_hi = axis.get_ylim()
            tick_top = y_hi
            tick_bottom = y_hi - 0.05 * (y_hi - y_lo)
            for bandhead in CO_BANDHEAD_UM:
                axis.plot([bandhead, bandhead], [tick_bottom, tick_top], color="black", lw=0.8)
            axis.set_ylim(y_lo, y_hi)
            axis.set_title(f"{product}, {age_gyr[idx]:.2f} Gyr", fontsize=9)
            if row == 1:
                axis.set_xlabel(r"rest wavelength [$\mu$m]")
            if col == 0:
                axis.set_ylabel(r"norm. $F_\lambda$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    figure.savefig(_out_path(out_dir, prefix, "q1_nir_spectra.png"), dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Step 4: broadband contrast
# ---------------------------------------------------------------------------


def step4_broadband_contrast(grids, out_dir, make_figure, prefix=""):
    grid = grids["native"]
    age_gyr = _age_gyr(grid)
    idx_blue = _nearest_index(grid.wave_a, STEP4_BLUE_WAVE_A)
    idx_red = _nearest_index(grid.wave_a, STEP4_RED_WAVE_A)

    flux0 = grid.flux_nu[:, 0]
    flux2 = agb_two_spectra(grid.flux_nu[:, 0], grid.flux_nu[:, 1])
    ratio0 = flux0[..., idx_red] / flux0[..., idx_blue]
    ratio2 = flux2[..., idx_red] / flux2[..., idx_blue]
    relative_difference = (ratio2 - ratio0) / ratio0

    z_idx = _solar_index(grid)
    summary = {}
    for target_age in STEP4_RECORD_AGES_GYR:
        idx = _nearest_index(age_gyr, target_age)
        summary[f"relative_difference_nir_optical_ratio_{target_age:g}gyr_solar_z"] = float(
            relative_difference[z_idx, idx]
        )

    if make_figure:
        _figure_step4(grid, age_gyr, ratio0, ratio2, relative_difference, out_dir, prefix)
    return summary


def _figure_step4(grid, age_gyr, ratio0, ratio2, relative_difference, out_dir, prefix=""):
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for i_z, log_z in enumerate(grid.log_z_grid):
        color = Z_COLORS[i_z % len(Z_COLORS)]
        left.plot(age_gyr, ratio0[i_z], color=color, ls="-", label=f"agb0, log Z={log_z:+.2f}")
        left.plot(age_gyr, ratio2[i_z], color=color, ls="--", label=f"agb2, log Z={log_z:+.2f}")
        right.plot(age_gyr, relative_difference[i_z], color=color, label=f"log Z={log_z:+.2f}")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(0.01, 20.0)
    left.set_xlabel("age [Gyr]")
    left.set_ylabel(r"$F_\nu$(1.6 $\mu$m) / $F_\nu$(4200 $\AA$)")
    left.legend(frameon=False, fontsize=6, ncol=2)
    right.set_xscale("log")
    right.set_xlim(0.01, 20.0)
    right.set_xlabel("age [Gyr]")
    right.set_ylabel("(agb2 - agb0) / agb0")
    right.set_title("relative difference in NIR/optical ratio", fontsize=9)
    right.axhline(0.0, color="black", lw=0.6)
    right.legend(frameon=False, fontsize=7)
    figure.savefig(_out_path(out_dir, prefix, "q1_broadband_ratio.png"), dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Step 5: empirical TP-AGB spectra variant (use_lw_tpagb)
# ---------------------------------------------------------------------------


def _build_or_load_lw02_variant(grid_dir):
    cache_path = Path(grid_dir) / "lw02_solar_native.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            return (
                data["wave_a"],
                data["log_age_yr"],
                data["flux_nu"],
                data["native_sigma_km_s"],
            )
    wave_a, log_age_yr, flux0, provenance0 = build_ssp(0.0, 0.0, extra_params={"use_lw_tpagb": 1})
    _, _, flux1, _ = build_ssp(0.0, 1.0, extra_params={"use_lw_tpagb": 1})
    flux_nu = np.stack([flux0, flux1], axis=0)
    native_sigma_km_s = np.array(provenance0["native_sigma_km_s"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        wave_a=wave_a,
        log_age_yr=log_age_yr,
        flux_nu=flux_nu,
        native_sigma_km_s=native_sigma_km_s,
    )
    return wave_a, log_age_yr, flux_nu, native_sigma_km_s


def step5_lw02_variant(grid_dir, grids, out_dir, prefix=""):
    grid_s300 = grids["sigma300"]
    z_idx = _solar_index(grid_s300)
    c3k_age_gyr = _age_gyr(grid_s300)
    c3k_flux0 = grid_s300.flux_nu[z_idx, 0]
    c3k_flux2 = agb_two_spectra(grid_s300.flux_nu[z_idx, 0], grid_s300.flux_nu[z_idx, 1])
    c3k_delta = h_minus_bump(grid_s300.wave_a, c3k_flux2) - h_minus_bump(
        grid_s300.wave_a, c3k_flux0
    )

    wave_a, log_age_yr, flux_nu, native_sigma_km_s = _build_or_load_lw02_variant(grid_dir)
    grid = SspGrid(
        wave_a=wave_a,
        log_age_yr=log_age_yr,
        log_z_grid=np.array([0.0]),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=flux_nu[None, ...],
        native_sigma_km_s=native_sigma_km_s,
        product="native",
        provenance={},
    )
    smoothed = make_resolution_product(grid, "sigma300")
    lw02_age_gyr = _age_gyr(smoothed)
    lw02_flux0 = smoothed.flux_nu[0, 0]
    lw02_flux2 = agb_two_spectra(smoothed.flux_nu[0, 0], smoothed.flux_nu[0, 1])
    lw02_delta = h_minus_bump(smoothed.wave_a, lw02_flux2) - h_minus_bump(
        smoothed.wave_a, lw02_flux0
    )

    c3k_window = (c3k_age_gyr >= AGE_WINDOW_GYR[0]) & (c3k_age_gyr <= AGE_WINDOW_GYR[1])
    lw02_window = (lw02_age_gyr >= AGE_WINDOW_GYR[0]) & (lw02_age_gyr <= AGE_WINDOW_GYR[1])
    idx_c3k = int(np.argmax(np.abs(c3k_delta[c3k_window])))
    idx_lw02 = int(np.argmax(np.abs(lw02_delta[lw02_window])))
    summary = {
        "max_abs_delta_bump_lw02_sigma300_mag": float(np.abs(lw02_delta[lw02_window])[idx_lw02]),
        "max_abs_delta_bump_lw02_sigma300_age_gyr": float(lw02_age_gyr[lw02_window][idx_lw02]),
        "max_abs_delta_bump_c3k_solar_sigma300_mag": float(np.abs(c3k_delta[c3k_window])[idx_c3k]),
        "max_abs_delta_bump_c3k_solar_sigma300_age_gyr": float(c3k_age_gyr[c3k_window][idx_c3k]),
    }

    figure, axis = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    axis.plot(lw02_age_gyr, lw02_delta, color="tab:red", label="use_lw_tpagb variant")
    axis.plot(c3k_age_gyr, c3k_delta, color="tab:blue", label="default C3K grid")
    for level in YARDSTICK_BUMP_MAG:
        axis.axhline(level, color="0.4", ls="--", lw=0.8)
        axis.axhline(-level, color="0.4", ls="--", lw=0.8)
    axis.axhline(0.0, color="black", lw=0.6)
    axis.set_xscale("log")
    axis.set_xlim(0.01, 20.0)
    axis.set_xlabel("age [Gyr]")
    axis.set_ylabel(r"$\Delta$ bump, $\sigma$300 [mag]")
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(_out_path(out_dir, prefix, "q1_lw02_variant.png"), dpi=150)
    plt.close(figure)
    return summary


# ---------------------------------------------------------------------------
# Step 6: population-level offsets
# ---------------------------------------------------------------------------


SCATTER_DEFINITION = (
    "per-model root-mean-square 16-84 half-width in the bin: "
    "sqrt((hw_agb0^2 + hw_agb2^2) / 2), hw = (p84 - p16) / 2 of each model's own bump values"
)
HEADLINE_BINNING = (
    "paired: both models binned on the agb0 D4000 / HdeltaA of the same epoch; the offset is "
    "the median of the per-epoch difference agb2 - agb0"
)
OWN_BINNING = (
    "unpaired: agb0 epochs binned on d4000_agb0 / hdelta_a_agb0 and agb2 epochs on "
    "d4000_agb2 / hdelta_a_agb2 (same bin edges); the offset is median(agb2) - median(agb0)"
)


def _half_width(values):
    return (np.percentile(values, 84) - np.percentile(values, 16)) / 2.0


def _rms_half_width(agb0_values, agb2_values):
    return float(np.sqrt((_half_width(agb0_values) ** 2 + _half_width(agb2_values) ** 2) / 2.0))


def _bin_edges(bin_values, n_bins):
    lo, hi = np.percentile(bin_values, [1.0, 99.0])
    return np.linspace(lo, hi, n_bins + 1)


def _bin_index(bin_values, edges):
    return np.clip(np.digitize(bin_values, edges[1:-1]), 0, edges.size - 2)


def _bin_offsets(bin_values, agb0, agb2, n_bins=12):
    edges = _bin_edges(bin_values, n_bins)
    bin_index = _bin_index(bin_values, edges)
    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        if not np.any(mask):
            continue
        a0 = agb0[mask]
        a2 = agb2[mask]
        half_width = _rms_half_width(a0, a2)
        diff_value = float(np.median(a2 - a0))
        rows.append(
            {
                "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                "count": int(mask.sum()),
                "median_agb0": float(np.median(a0)),
                "median_agb2": float(np.median(a2)),
                "p16_agb0": float(np.percentile(a0, 16)),
                "p84_agb0": float(np.percentile(a0, 84)),
                "p16_agb2": float(np.percentile(a2, 16)),
                "p84_agb2": float(np.percentile(a2, 84)),
                "median_diff": diff_value,
                "rms_half_width": half_width,
                "offset_over_scatter": diff_value / half_width if half_width > 0 else float("nan"),
            }
        )
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def _own_bin_offsets(bin_values_agb0, bin_values_agb2, agb0, agb2, n_bins=12):
    """Offsets when each model is binned on its own optical index, on the agb0 bin edges."""
    edges = _bin_edges(bin_values_agb0, n_bins)
    index_agb0 = _bin_index(bin_values_agb0, edges)
    index_agb2 = _bin_index(bin_values_agb2, edges)
    rows = []
    for b in range(n_bins):
        a0 = agb0[index_agb0 == b]
        a2 = agb2[index_agb2 == b]
        if a0.size == 0 or a2.size == 0:
            continue
        half_width = _rms_half_width(a0, a2)
        diff_value = float(np.median(a2) - np.median(a0))
        rows.append(
            {
                "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                "count_agb0": int(a0.size),
                "count_agb2": int(a2.size),
                "median_diff": diff_value,
                "offset_over_scatter": diff_value / half_width if half_width > 0 else float("nan"),
            }
        )
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def step6_population_offsets(population_path, out_dir, make_figure, pilot=False, prefix=""):
    with np.load(population_path) as data:
        keep = data["epoch_gyr"] >= 1.0
        table = {key: data[key][keep] for key in data.files}
    if pilot:
        rng = np.random.default_rng(0)
        n_keep = min(20000, table["epoch_gyr"].size)
        subset = rng.choice(table["epoch_gyr"].size, size=n_keep, replace=False)
        table = {key: value[subset] for key, value in table.items()}

    results = {}
    summary = {
        "scatter_definition": SCATTER_DEFINITION,
        "headline_binning": HEADLINE_BINNING,
        "own_bins_binning": OWN_BINNING,
    }
    max_offset, max_offset_source = 0.0, None
    min_offset, min_offset_source = np.inf, None
    for index_name in ("d4000", "hdelta_a"):
        bin_key = f"{index_name}_agb0"
        for product in ("sigma300", "r100"):
            agb0 = table[f"h_minus_bump_{product}_agb0"]
            agb2 = table[f"h_minus_bump_{product}_agb2"]
            binned = _bin_offsets(table[bin_key], agb0, agb2)
            results[(bin_key, product)] = binned
            key_prefix = f"bump_{product}_by_{bin_key}"
            summary[f"{key_prefix}_bin_center"] = binned["bin_center"].tolist()
            summary[f"{key_prefix}_median_agb0_mag"] = binned["median_agb0"].tolist()
            summary[f"{key_prefix}_median_agb2_mag"] = binned["median_agb2"].tolist()
            summary[f"{key_prefix}_median_diff_mag"] = binned["median_diff"].tolist()
            summary[f"{key_prefix}_rms_half_width_mag"] = binned["rms_half_width"].tolist()
            summary[f"{key_prefix}_offset_over_scatter"] = binned["offset_over_scatter"].tolist()
            finite = np.abs(
                binned["offset_over_scatter"][np.isfinite(binned["offset_over_scatter"])]
            )
            if finite.size and np.max(finite) > max_offset:
                max_offset = float(np.max(finite))
                max_offset_source = key_prefix
            if finite.size and np.min(finite) < min_offset:
                min_offset = float(np.min(finite))
                min_offset_source = key_prefix

            own = _own_bin_offsets(table[bin_key], table[f"{index_name}_agb2"], agb0, agb2)
            own_prefix = f"bump_{product}_by_{index_name}"
            summary[f"{own_prefix}_own_bins_bin_center"] = own["bin_center"].tolist()
            summary[f"{own_prefix}_own_bins_median_diff_mag"] = own["median_diff"].tolist()
            summary[f"{own_prefix}_own_bins_offset_over_scatter"] = own[
                "offset_over_scatter"
            ].tolist()
    summary["max_offset_over_scatter"] = max_offset
    summary["max_offset_over_scatter_source"] = max_offset_source
    summary["min_offset_over_scatter"] = min_offset
    summary["min_offset_over_scatter_source"] = min_offset_source

    if make_figure:
        _figure_step6(results, out_dir, prefix)
    return summary


def _figure_step6(results, out_dir, prefix=""):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    row_keys = ("d4000_agb0", "hdelta_a_agb0")
    row_labels = ("D4000 (agb0) bin", r"H$\delta_A$ (agb0) bin [$\AA$]")
    first_twin = None
    for row, (bin_key, bin_label) in enumerate(zip(row_keys, row_labels, strict=True)):
        for col, product in enumerate(("sigma300", "r100")):
            binned = results[(bin_key, product)]
            axis = axes[row, col]
            axis.fill_between(
                binned["bin_center"],
                binned["p16_agb0"],
                binned["p84_agb0"],
                color="#1f77b4",
                alpha=0.3,
            )
            axis.plot(binned["bin_center"], binned["median_agb0"], color="#1f77b4", label="agb0")
            axis.fill_between(
                binned["bin_center"],
                binned["p16_agb2"],
                binned["p84_agb2"],
                color="#d62728",
                alpha=0.3,
            )
            axis.plot(binned["bin_center"], binned["median_agb2"], color="#d62728", label="agb2")
            axis.invert_yaxis()
            axis.set_xlabel(bin_label)
            axis.set_ylabel(f"H$^-$ bump, {product} [mag]")
            axis.set_title(product, fontsize=9)

            twin = axis.twinx()
            twin.plot(
                binned["bin_center"],
                binned["offset_over_scatter"],
                color="black",
                ls="--",
                lw=1.0,
                label="offset/scatter",
            )
            yardstick_over_scatter = np.divide(
                0.01,
                binned["rms_half_width"],
                out=np.full_like(binned["rms_half_width"], np.nan),
                where=binned["rms_half_width"] > 0,
            )
            twin.plot(
                binned["bin_center"],
                yardstick_over_scatter,
                color="0.5",
                ls=":",
                lw=1.0,
                label="0.01 mag / scatter",
            )
            twin.axhline(0.0, color="0.7", lw=0.5)
            twin.set_ylabel("offset / RMS scatter")
            if first_twin is None:
                first_twin = twin
    handles, labels = axes[0, 0].get_legend_handles_labels()
    twin_handles, twin_labels = first_twin.get_legend_handles_labels()
    figure.legend(
        handles + twin_handles,
        labels + twin_labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    figure.savefig(_out_path(out_dir, prefix, "q1_population_offsets.png"), dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Q1: TP-AGB separability analysis.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--population-dir", default=str(POPULATION_PATH.parent))
    parser.add_argument("--out-prefix", default="")
    parser.add_argument(
        "--pilot", action="store_true", help="solar Z, 3 ages, skip step 5 and figures"
    )
    args = parser.parse_args()

    start = time.perf_counter()
    grid_dir = Path(args.grid_dir)
    population_path = Path(args.population_dir) / "indices.npz"
    prefix = args.out_prefix
    grids = {
        product: load_ssp_grid(grid_dir, product) for product in ("native", "sigma300", "r100")
    }
    if args.pilot:
        grids = {product: _pilot_grid(grid) for product, grid in grids.items()}

    out_dir = OUTPUT_DIR
    make_figure = not args.pilot
    if make_figure:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "step1": step1_ssp_delta_vs_age(grids, out_dir, make_figure, prefix),
        "step2": step2_component_spectrum(grids, out_dir, make_figure, prefix),
        "step4": step4_broadband_contrast(grids, out_dir, make_figure, prefix),
    }
    if make_figure:
        step3_nir_spectra_qa(grids, out_dir, prefix)
        if _grid_uses_lw_tpagb(grids):
            summary["step5"] = {
                "skipped": True,
                "note": "grid provenance already has use_lw_tpagb = 1 (this is the LW02 grid "
                "itself); the LW02-variant comparison is meaningless here",
            }
        else:
            summary["step5"] = step5_lw02_variant(grid_dir, grids, out_dir, prefix)
    summary["step6"] = step6_population_offsets(
        population_path, out_dir, make_figure, pilot=args.pilot, prefix=prefix
    )

    elapsed = time.perf_counter() - start
    print(f"{'pilot ' if args.pilot else ''}run finished in {elapsed:.2f} s")
    if make_figure:
        summary_path = _out_path(out_dir, prefix, "q1_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
