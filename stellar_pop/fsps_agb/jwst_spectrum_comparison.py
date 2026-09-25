"""The observed JWST stack against FSPS mock spectra in the H-minus bump region.

Observed side: the 19 NIRSpec PRISM quiescent galaxies of Lu+2026 (rest-frame,
`output/publication/final/qg_spec/`), each divided by a straight line fitted to all
pixels in the blue and red pseudo-continuum windows of the H-minus bump, then combined
on a common rest-frame grid as an S/N-weighted mean.

Model side: composite spectra from the cached FSPS SSP grids on the R = 100 product,
solar metallicity, for a grid of quenching histories (delayed-tau rise with tau = t_q,
exponential quench), normalised the same way. Two configurations: AGB off (C3K, agb = 0)
and AGB on (LW02, agb = 2). The figure shows, per configuration, the envelope of all
post-quench mock spectra, a few representative epochs after quenching, and the observed
stack, with the observed-to-model ratio below.

Outputs go to `output/publication/final/`: the figure (`observed_stack_vs_fsps_mocks`),
`observed_stack_vs_fsps_mocks.json`, and an exploration figure of the individual observed
spectra (`../jwst_spectra_bump_region`).
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from csp_integrate import agb_two_spectra, csp_track
from publication_figures import (
    AGB_CONFIGS,
    CONFIG_COLORS,
    CONFIG_ORDER,
    DOUBLE_COLUMN_IN,
    FINAL_DIR,
    GRID_DIRS,
    OUTPUT_DIR,
    PUBLICATION_RC,
    _to_native,
    panel_label,
    save_figure,
)
from sfh_model import time_bin_edges
from spectral_indices import H_MINUS_BANDS_A, flux_nu_to_flux_lambda
from ssp_grid import load_ssp_grid

JWST_TABLE = FINAL_DIR / "JWST_QG_indices.npz"
JWST_SPECTRA_DIR = FINAL_DIR / "qg_spec"
STEM = "observed_stack_vs_fsps_mocks"

PLOT_RANGE_A = (13500.0, 19500.0)
STACK_STEP_A = 30.0
LOG_Z = 0.0
MOCK_T_Q_GYR = (1.5, 3.0, 4.5)
MOCK_TAU_Q_GYR = (0.1, 0.3, 1.0, 3.0)
MOCK_POST_QUENCH_WINDOW_GYR = (0.0, 6.0)
MOCK_MIN_EPOCH_GYR = 1.0
REPRESENTATIVE_T_Q_GYR = 3.0
REPRESENTATIVE_TAU_Q_GYR = 0.3
REPRESENTATIVE_DELAYS_GYR = (0.5, 1.0, 2.0, 5.0)
REPRESENTATIVE_COLORS = tuple(plt.get_cmap("viridis")(level) for level in (0.0, 0.33, 0.66, 0.95))
BAND_COLORS = {"blue": "#0072B2", "feature": "#E69F00", "red": "#D55E00"}


# ---------------------------------------------------------------------------
# Observed spectra
# ---------------------------------------------------------------------------


def pseudo_continuum_line(wave_a, flux_lambda):
    """Straight line fitted to every pixel inside the blue and red side windows (the
    pyphot degree-1 convention adopted by the paper), evaluated at every pixel."""
    blue, red = H_MINUS_BANDS_A["blue"], H_MINUS_BANDS_A["red"]
    inside = ((wave_a > blue[0]) & (wave_a < blue[1])) | ((wave_a > red[0]) & (wave_a < red[1]))
    coefficients = np.polyfit(wave_a[inside], flux_lambda[inside], 1)
    return np.polyval(coefficients, wave_a)


def bump_index_from_normalised(wave_a, normalised):
    """-2.5 log10 of the mean normalised flux inside the feature band (pixel mean)."""
    low, high = H_MINUS_BANDS_A["feature"]
    inside = (wave_a > low) & (wave_a < high)
    return float(-2.5 * np.log10(np.mean(normalised[inside])))


def load_observed_spectra(table_path=JWST_TABLE, spectra_dir=JWST_SPECTRA_DIR):
    """Rest-frame, pseudo-continuum-normalised spectra of the galaxies in the index table.

    Files are observed-frame (wavelength in A, F_lambda, error); the redshift comes from
    the table. Non-finite pixels and non-positive errors are dropped."""
    with np.load(table_path) as table:
        redshift = {int(i): float(z) for i, z in zip(table["ID"], table["z"], strict=True)}
        bump = {int(i): float(b) for i, b in zip(table["ID"], table["Hbump"], strict=True)}
    spectra = {}
    for path in sorted(glob.glob(str(spectra_dir / "*.txt"))):
        galaxy_id = int(re.search(r"(\d{4,5})", os.path.basename(path)).group(1))
        if galaxy_id not in redshift:
            continue
        data = np.loadtxt(path)
        data = data[np.isfinite(data).all(axis=1) & (data[:, 2] > 0)]
        z = redshift[galaxy_id]
        wave_a = data[:, 0] / (1.0 + z)
        flux = data[:, 1] * (1.0 + z)
        error = data[:, 2] * (1.0 + z)
        inside = (wave_a > PLOT_RANGE_A[0] - 500.0) & (wave_a < PLOT_RANGE_A[1] + 500.0)
        wave_a, flux, error = wave_a[inside], flux[inside], error[inside]
        continuum = pseudo_continuum_line(wave_a, flux)
        normalised = flux / continuum
        spectra[galaxy_id] = {
            "redshift": z,
            "wave_a": wave_a,
            "normalised": normalised,
            "normalised_error": error / continuum,
            "table_bump_mag": bump[galaxy_id],
            "own_bump_mag": bump_index_from_normalised(wave_a, normalised),
            "file": os.path.basename(path),
        }
    return spectra


def stack_observed(spectra, grid_a):
    """S/N-weighted mean of the normalised spectra on `grid_a` (weights (F/sigma)^2 per
    galaxy and pixel) and the weighted scatter between galaxies."""
    values = np.array([np.interp(grid_a, s["wave_a"], s["normalised"]) for s in spectra.values()])
    errors = np.array(
        [np.interp(grid_a, s["wave_a"], s["normalised_error"]) for s in spectra.values()]
    )
    weights = (values / errors) ** 2
    stack = np.sum(values * weights, axis=0) / np.sum(weights, axis=0)
    scatter = np.sqrt(np.sum(weights * (values - stack) ** 2, axis=0) / np.sum(weights, axis=0))
    return stack, scatter


# ---------------------------------------------------------------------------
# Mock spectra
# ---------------------------------------------------------------------------


def mock_spectra(config_key, edges_gyr, t_q_values=MOCK_T_Q_GYR, tau_q_values=MOCK_TAU_Q_GYR):
    """Normalised post-quench composite spectra on the R = 100 grid for one configuration,
    for every (t_q, tau_q) and every epoch with t >= MOCK_MIN_EPOCH_GYR and
    0 <= t - t_q <= 6 Gyr. Returns the wavelength grid, an array (n_spectra, n_pix) of
    normalised spectra and a record of (t_q, tau_q, delay) per row."""
    config = AGB_CONFIGS[config_key]
    grid = load_ssp_grid(GRID_DIRS[config["template"]], "r100")
    wave_a = grid.wave_a
    rows, records = [], []
    for t_q in t_q_values:
        for tau_q in tau_q_values:
            epochs, flux_agb0, _ = csp_track(grid, LOG_Z, 0, t_q, tau_q, edges_gyr)
            if config["agb"] == "agb0":
                flux_nu = flux_agb0
            else:
                _, flux_agb1, _ = csp_track(grid, LOG_Z, 1, t_q, tau_q, edges_gyr)
                flux_nu = agb_two_spectra(flux_agb0, flux_agb1)
            delay = epochs - t_q
            keep = (
                (epochs >= MOCK_MIN_EPOCH_GYR)
                & (delay >= MOCK_POST_QUENCH_WINDOW_GYR[0] - 1e-9)
                & (delay <= MOCK_POST_QUENCH_WINDOW_GYR[1] + 1e-9)
            )
            for index in np.flatnonzero(keep):
                flux_lambda = flux_nu_to_flux_lambda(wave_a, flux_nu[index])
                rows.append(flux_lambda / pseudo_continuum_line(wave_a, flux_lambda))
                records.append((t_q, tau_q, float(delay[index])))
    return wave_a, np.array(rows), records


def representative_rows(records, t_q=REPRESENTATIVE_T_Q_GYR, tau_q=REPRESENTATIVE_TAU_Q_GYR):
    rows = {}
    for delay in REPRESENTATIVE_DELAYS_GYR:
        candidates = [
            (abs(record[2] - delay), i)
            for i, record in enumerate(records)
            if record[0] == t_q and record[1] == tau_q
        ]
        rows[delay] = min(candidates)[1]
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _shade_bands(axis):
    for name, color in BAND_COLORS.items():
        low, high = H_MINUS_BANDS_A[name]
        axis.axvspan(low / 1e4, high / 1e4, color=color, alpha=0.12, lw=0)


def figure_observed_spectra(spectra, grid_a, stack, scatter, out_dir):
    """Exploration figure: every normalised observed spectrum coloured by redshift, and
    the stack with its galaxy-to-galaxy scatter."""
    figure, axes = plt.subplots(
        2, 1, figsize=(DOUBLE_COLUMN_IN, 5.2), sharex=True, layout="constrained"
    )
    redshifts = np.array([s["redshift"] for s in spectra.values()])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(redshifts.min(), redshifts.max())
    for s in spectra.values():
        axes[0].plot(
            s["wave_a"] / 1e4, s["normalised"], color=cmap(norm(s["redshift"])), lw=0.7, alpha=0.85
        )
    for axis in axes:
        _shade_bands(axis)
        axis.axhline(1.0, color="0.5", lw=0.6, ls="--")
        axis.set_xlim(PLOT_RANGE_A[0] / 1e4, PLOT_RANGE_A[1] / 1e4)
    figure.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0], label="redshift", pad=0.01
    )
    axes[0].set_ylim(0.72, 1.28)
    axes[0].set_ylabel(r"$F_\lambda$ / straight-line pseudo-continuum")
    axes[0].set_title(
        f"{len(spectra)} JWST PRISM quiescent galaxies (Lu+2026), rest frame", fontsize=9
    )
    axes[1].fill_between(
        grid_a / 1e4,
        stack - scatter,
        stack + scatter,
        color="0.75",
        alpha=0.6,
        lw=0,
        label="S/N-weighted scatter between galaxies",
    )
    axes[1].plot(
        grid_a / 1e4,
        stack,
        color="black",
        lw=1.3,
        label="S/N-weighted mean of the normalised spectra",
    )
    axes[1].set_ylim(0.86, 1.14)
    axes[1].set_xlabel(r"rest-frame wavelength [$\mu$m]")
    axes[1].set_ylabel("normalised flux")
    axes[1].legend(loc="lower left", fontsize=7)
    save_figure(figure, out_dir, "jwst_spectra_bump_region")


def figure_stack_versus_mocks(grid_a, stack, scatter, mocks, out_dir, stem=STEM):
    """Two columns (AGB off, AGB on): top, the observed stack against the post-quench mock
    envelope and four representative epochs after quenching; bottom, observed over model."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_IN, 4.9),
        sharex=True,
        height_ratios=(2.6, 1.0),
        layout="constrained",
    )
    summary = {}
    x_stack = grid_a / 1e4
    for col, config_key in enumerate(CONFIG_ORDER):
        config = AGB_CONFIGS[config_key]
        wave_a, rows, records = mocks[config_key]
        x_mock = wave_a / 1e4
        top, bottom = axes[0, col], axes[1, col]
        for axis in (top, bottom):
            _shade_bands(axis)
        envelope_low, envelope_high = rows.min(axis=0), rows.max(axis=0)
        top.fill_between(
            x_mock,
            envelope_low,
            envelope_high,
            color=CONFIG_COLORS[config_key],
            alpha=0.22,
            lw=0,
            zorder=2,
        )
        chosen = representative_rows(records)
        for row_index, color in zip(chosen.values(), REPRESENTATIVE_COLORS, strict=True):
            top.plot(x_mock, rows[row_index], color=color, lw=1.0, zorder=3)
            ratio = stack / np.interp(grid_a, wave_a, rows[row_index])
            bottom.plot(x_stack, ratio, color=color, lw=1.0, zorder=3)
        top.fill_between(
            x_stack, stack - scatter, stack + scatter, color="0.55", alpha=0.35, lw=0, zorder=4
        )
        top.plot(x_stack, stack, color="black", lw=1.3, zorder=5)
        bottom.fill_between(
            x_stack,
            1.0 - scatter / stack,
            1.0 + scatter / stack,
            color="0.55",
            alpha=0.35,
            lw=0,
            zorder=1,
        )
        bottom.axhline(1.0, color="black", lw=0.7, ls="--", zorder=2)
        top.text(
            0.97,
            0.95,
            config["short_label"],
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            zorder=20,
        )
        top.set_ylim(0.84, 1.16)
        bottom.set_ylim(0.9, 1.16)
        bottom.set_xlabel(r"rest-frame wavelength [$\mu$m]")
        if col == 0:
            top.set_ylabel(r"$F_\lambda$ / straight-line pseudo-continuum")
            bottom.set_ylabel("observed / model")
        top.set_xlim(PLOT_RANGE_A[0] / 1e4, PLOT_RANGE_A[1] / 1e4)
        panel_label(top, f"({'ab'[col]})", x=0.03, y=0.95)
        panel_label(bottom, f"({'cd'[col]})", x=0.03, y=0.93)
        mock_bumps = np.array([bump_index_from_normalised(wave_a, row) for row in rows])
        feature = (wave_a > H_MINUS_BANDS_A["feature"][0]) & (
            wave_a < H_MINUS_BANDS_A["feature"][1]
        )
        summary[config_key] = {
            "n_mock_spectra": int(rows.shape[0]),
            "t_q_gyr": list(MOCK_T_Q_GYR),
            "tau_q_gyr": list(MOCK_TAU_Q_GYR),
            "post_quench_window_gyr": list(MOCK_POST_QUENCH_WINDOW_GYR),
            "mock_bump_index_range_mag": [float(mock_bumps.min()), float(mock_bumps.max())],
            "mock_envelope_peak_normalised_flux_range": [
                float(envelope_low[feature].max()),
                float(envelope_high[feature].max()),
            ],
            "representative": {
                f"delay_{delay:g}_gyr": {
                    "t_q_gyr": records[row_index][0],
                    "tau_q_gyr": records[row_index][1],
                    "delay_gyr": records[row_index][2],
                    "bump_index_mag": float(mock_bumps[row_index]),
                    "median_observed_over_model_in_feature": float(
                        np.median(
                            (stack / np.interp(grid_a, wave_a, rows[row_index]))[
                                (grid_a > H_MINUS_BANDS_A["feature"][0])
                                & (grid_a < H_MINUS_BANDS_A["feature"][1])
                            ]
                        )
                    ),
                }
                for delay, row_index in chosen.items()
            },
        }
    handles = [
        Line2D([], [], color="black", lw=1.3, label="observed stack (Lu+2026, 19 galaxies)"),
        Patch(facecolor="0.55", alpha=0.35, label="galaxy-to-galaxy scatter of the stack"),
    ]
    handles += [
        Patch(
            facecolor=CONFIG_COLORS[key],
            alpha=0.22,
            label=f"{AGB_CONFIGS[key]['short_label']}: all post-quench mock spectra",
        )
        for key in CONFIG_ORDER
    ]
    handles += [
        Line2D([], [], color=color, lw=1.0, label=rf"$t - t_q = {delay:g}$ Gyr")
        for delay, color in zip(REPRESENTATIVE_DELAYS_GYR, REPRESENTATIVE_COLORS, strict=True)
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=7.5,
        columnspacing=1.4,
    )
    save_figure(figure, out_dir, stem)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Observed JWST stack against FSPS mock spectra.")
    parser.add_argument("--out-dir", default=str(FINAL_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PUBLICATION_RC)

    spectra = load_observed_spectra()
    grid_a = np.arange(PLOT_RANGE_A[0], PLOT_RANGE_A[1] + STACK_STEP_A, STACK_STEP_A)
    stack, scatter = stack_observed(spectra, grid_a)
    figure_observed_spectra(spectra, grid_a, stack, scatter, OUTPUT_DIR)

    edges = time_bin_edges()
    mocks = {key: mock_spectra(key, edges) for key in CONFIG_ORDER}
    summary = {
        "observed": {
            "n_galaxies": len(spectra),
            "redshift_range": [
                min(s["redshift"] for s in spectra.values()),
                max(s["redshift"] for s in spectra.values()),
            ],
            "stack_grid_step_a": STACK_STEP_A,
            "stack_bump_index_mag": bump_index_from_normalised(grid_a, stack),
            "stack_peak_normalised_flux": float(np.max(stack[(grid_a > 15700) & (grid_a < 17340)])),
            "per_galaxy": {
                str(galaxy_id): {
                    "file": s["file"],
                    "redshift": s["redshift"],
                    "table_bump_mag": s["table_bump_mag"],
                    "own_bump_mag": s["own_bump_mag"],
                }
                for galaxy_id, s in spectra.items()
            },
        },
        "mocks": figure_stack_versus_mocks(grid_a, stack, scatter, mocks, out_dir),
        "normalisation": "straight line fitted to all pixels in the blue (1.494-1.539 um) "
        "and red (1.746-1.791 um) windows, observed and model alike; bump index = "
        "-2.5 log10(mean normalised flux over 1.570-1.734 um)",
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(_to_native(summary), indent=2) + "\n")
    print(f"wrote {out_dir / STEM}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
