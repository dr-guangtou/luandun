"""The observed JWST stack against FSPS mock spectra in the H-minus bump region.

Observed side: the 19 NIRSpec PRISM quiescent galaxies of Lu+2026 (rest-frame,
`output/publication/final/qg_spec/`), each divided by a straight line fitted to all
pixels in the blue and red pseudo-continuum windows of the H-minus bump, then combined
on a common rest-frame grid as an S/N-weighted mean.

Model side: composite spectra from the cached FSPS SSP grids on the R = 100 product at
a single epoch, the cosmic age at the sample's median redshift (flat LCDM, H0 = 70,
Omega_m = 0.3, star formation starting at t = 0). The panel curves keep t - t_q fixed
and vary the quenching timescale tau_q at solar metallicity; a separate search over
metallicity, t_q and tau_q at the same epoch finds the model closest to the observed
stack for each configuration (TP-AGB off = C3K agb 0, TP-AGB on = LW02 agb 2). All spectra
are normalised the same way and drawn as steps.

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
from scipy.integrate import quad

from csp_integrate import (
    agb_two_spectra,
    csp_spectra,
    epoch_weight_matrix,
    interpolate_log_z,
)
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
COMPARISON_RANGE_A = (14600.0, 18300.0)
SEARCH_CONFIGS = ("agb_on",)
FIT_RANGE_A = (H_MINUS_BANDS_A["blue"][0], H_MINUS_BANDS_A["red"][1])
STACK_STEP_A = 45.0
HUBBLE_CONSTANT = 70.0
OMEGA_MATTER = 0.3
PANEL_LOG_Z = 0.0
PANEL_DELAY_GYR = 1.0
PANEL_TAU_Q_GYR = (0.1, 0.3, 1.0, 3.0)
PANEL_COLORS = tuple(plt.get_cmap("viridis")(level) for level in (0.0, 0.33, 0.66, 0.95))
SEARCH_LOG_Z = tuple(np.round(np.arange(-0.5, 0.2501, 0.05), 2))
SEARCH_T_Q_STEP_GYR = 0.1
SEARCH_TAU_Q_GYR = tuple(np.round(np.logspace(np.log10(0.1), np.log10(3.0), 13), 3))
MIN_T_Q_GYR = 1.0
STEP_STYLE = {"drawstyle": "steps-mid"}
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
# Mock spectra at one epoch
# ---------------------------------------------------------------------------


def cosmic_age_gyr(redshift, hubble_constant=HUBBLE_CONSTANT, omega_matter=OMEGA_MATTER):
    """Age of a flat LCDM universe at `redshift`, in Gyr."""

    def integrand(x):
        return 1.0 / ((1.0 + x) * np.sqrt(omega_matter * (1.0 + x) ** 3 + 1.0 - omega_matter))

    return (977.8 / hubble_constant) * quad(integrand, redshift, np.inf)[0]


def mock_epoch_gyr(redshift, edges_gyr):
    """The time-grid edge closest to the cosmic age at `redshift` (star formation is
    assumed to start at t = 0)."""
    age = cosmic_age_gyr(redshift)
    return float(edges_gyr[1:][np.argmin(np.abs(edges_gyr[1:] - age))])


class MockLibrary:
    """Normalised R = 100 composite spectra at one epoch for one AGB configuration."""

    def __init__(self, config_key, epoch_gyr, edges_gyr):
        self.config = AGB_CONFIGS[config_key]
        self.epoch_gyr = epoch_gyr
        self.edges_gyr = edges_gyr[: int(np.argmin(np.abs(edges_gyr - epoch_gyr))) + 1]
        self.grid = load_ssp_grid(GRID_DIRS[self.config["template"]], "r100")
        self.wave_a = self.grid.wave_a
        self._ssp_cache = {}

    def _ssp_flux(self, log_z, agb_index):
        key = (round(float(log_z), 4), agb_index)
        if key not in self._ssp_cache:
            self._ssp_cache[key] = interpolate_log_z(
                self.grid.flux_nu[:, agb_index], self.grid.log_z_grid, log_z
            )
        return self._ssp_cache[key]

    def spectrum(self, log_z, t_q_gyr, tau_q_gyr):
        """Normalised composite spectrum at the library epoch."""
        weights = epoch_weight_matrix(self.edges_gyr, t_q_gyr, tau_q_gyr, self.grid.log_age_yr)[-1:]
        flux_nu = csp_spectra(weights, self._ssp_flux(log_z, 0))[0]
        if self.config["agb"] != "agb0":
            flux_nu = agb_two_spectra(flux_nu, csp_spectra(weights, self._ssp_flux(log_z, 1))[0])
        flux_lambda = flux_nu_to_flux_lambda(self.wave_a, flux_nu)
        return flux_lambda / pseudo_continuum_line(self.wave_a, flux_lambda)


def mismatch(wave_a, model, grid_a, stack, scatter):
    """RMS of (model - stack) / scatter over the H-minus window span, model interpolated
    onto the stack grid."""
    inside = (grid_a >= FIT_RANGE_A[0]) & (grid_a <= FIT_RANGE_A[1])
    residual = (np.interp(grid_a[inside], wave_a, model) - stack[inside]) / scatter[inside]
    return float(np.sqrt(np.mean(residual**2)))


def bin_to_grid(wave_a, model, grid_a):
    """Mean of the model pixels inside each cell of the (uniform) stack grid, so that a
    model on the fine R = 100 grid is drawn at the observed pixel scale."""
    edges = np.concatenate([[grid_a[0] - STACK_STEP_A / 2], grid_a + STACK_STEP_A / 2])
    counts, _ = np.histogram(wave_a, bins=edges)
    sums, _ = np.histogram(wave_a, bins=edges, weights=model)
    binned = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    empty = counts == 0
    if empty.any():
        binned[empty] = np.interp(grid_a[empty], wave_a, model)
    return binned


def search_best_match(library, grid_a, stack, scatter):
    """Grid search over metallicity, t_q and tau_q at the library epoch; returns the
    best model and the full table of mismatches."""
    t_q_values = np.round(
        np.arange(MIN_T_Q_GYR, library.epoch_gyr - SEARCH_T_Q_STEP_GYR / 2, SEARCH_T_Q_STEP_GYR), 3
    )
    table = []
    best = None
    for log_z in SEARCH_LOG_Z:
        for t_q in t_q_values:
            for tau_q in SEARCH_TAU_Q_GYR:
                model = library.spectrum(log_z, t_q, tau_q)
                value = mismatch(library.wave_a, model, grid_a, stack, scatter)
                table.append((float(log_z), float(t_q), float(tau_q), value))
                if best is None or value < best["mismatch"]:
                    best = {
                        "log_z": float(log_z),
                        "t_q_gyr": float(t_q),
                        "tau_q_gyr": float(tau_q),
                        "delay_gyr": float(library.epoch_gyr - t_q),
                        "mismatch": value,
                        "bump_index_mag": bump_index_from_normalised(library.wave_a, model),
                        "spectrum": model,
                    }
    return best, np.array(table)


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
            s["wave_a"] / 1e4,
            s["normalised"],
            color=cmap(norm(s["redshift"])),
            lw=0.7,
            alpha=0.85,
            **STEP_STYLE,
        )
    for axis in axes:
        _shade_bands(axis)
        axis.axhline(1.0, color="0.5", lw=0.6, ls="--")
        axis.set_xlim(PLOT_RANGE_A[0] / 1e4, PLOT_RANGE_A[1] / 1e4)
    figure.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0], label="redshift", pad=0.01
    )
    axes[0].set_ylim(0.72, 1.28)
    axes[0].set_ylabel(r"Normalised $F_\lambda$")
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
    axes[1].set_xlabel(r"Rest-frame wavelength [$\mu$m]")
    axes[1].set_ylabel(r"Normalised $F_\lambda$")
    axes[1].legend(loc="lower left", fontsize=7)
    save_figure(figure, out_dir, "jwst_spectra_bump_region")


def figure_stack_versus_mocks(grid_a, stack, scatter, libraries, matches, out_dir, stem=STEM):
    """Two columns (TP-AGB off, TP-AGB on): top, the observed stack against mock spectra at the
    sample epoch with tau_q varied at fixed t - t_q and solar metallicity, plus the
    best-matching model of the search; bottom, observed over model."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_IN, 3.9),
        sharex=True,
        height_ratios=(2.4, 1.0),
        layout="constrained",
    )
    summary = {}
    x_stack = grid_a / 1e4
    feature = (grid_a > H_MINUS_BANDS_A["feature"][0]) & (grid_a < H_MINUS_BANDS_A["feature"][1])
    for col, config_key in enumerate(CONFIG_ORDER):
        library = libraries[config_key]
        best = matches.get(config_key)
        top, bottom = axes[0, col], axes[1, col]
        for axis in (top, bottom):
            _shade_bands(axis)
        t_q_panel = library.epoch_gyr - PANEL_DELAY_GYR
        panel_summary = {}
        for tau_q, color in zip(PANEL_TAU_Q_GYR, PANEL_COLORS, strict=True):
            model = library.spectrum(PANEL_LOG_Z, t_q_panel, tau_q)
            binned = bin_to_grid(library.wave_a, model, grid_a)
            top.plot(x_stack, binned, color=color, lw=1.0, zorder=3, **STEP_STYLE)
            ratio = stack / binned
            bottom.plot(x_stack, ratio, color=color, lw=1.0, zorder=3, **STEP_STYLE)
            panel_summary[f"tau_q_{tau_q:g}"] = {
                "bump_index_mag": bump_index_from_normalised(library.wave_a, model),
                "mismatch": mismatch(library.wave_a, model, grid_a, stack, scatter),
                "median_observed_over_model_in_feature": float(np.median(ratio[feature])),
            }
        best_ratio = None
        if best is not None:
            best_binned = bin_to_grid(library.wave_a, best["spectrum"], grid_a)
            top.plot(
                x_stack,
                best_binned,
                color=CONFIG_COLORS[config_key],
                lw=1.1,
                ls="--",
                zorder=4,
                **STEP_STYLE,
            )
            best_ratio = stack / best_binned
            bottom.plot(
                x_stack,
                best_ratio,
                color=CONFIG_COLORS[config_key],
                lw=1.1,
                ls="--",
                zorder=4,
                **STEP_STYLE,
            )
            top.text(
                0.97,
                0.05,
                (
                    rf"Closest model: $\log Z/Z_\odot = {best['log_z']:+.2f}$, "
                    rf"$t_q = {best['t_q_gyr']:.1f}$, $\tau_q = {best['tau_q_gyr']:.2f}$ Gyr"
                ),
                transform=top.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=CONFIG_COLORS[config_key],
                zorder=20,
            )
        top.fill_between(
            x_stack,
            stack - scatter,
            stack + scatter,
            color="0.55",
            alpha=0.35,
            lw=0,
            step="mid",
            zorder=5,
        )
        top.plot(x_stack, stack, color="black", lw=1.3, zorder=6, **STEP_STYLE)
        bottom.fill_between(
            x_stack,
            1.0 - scatter / stack,
            1.0 + scatter / stack,
            color="0.55",
            alpha=0.35,
            lw=0,
            step="mid",
            zorder=1,
        )
        bottom.axhline(1.0, color="black", lw=0.7, ls="--", zorder=2)
        top.text(
            0.97,
            0.95,
            library.config["short_label"],
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            zorder=20,
        )
        top.set_ylim(0.93, 1.12)
        bottom.set_ylim(0.95, 1.1)
        bottom.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
        if col == 0:
            top.set_ylabel(r"Normalised $F_\lambda$")
            bottom.set_ylabel("Observed / model")
        top.set_xlim(COMPARISON_RANGE_A[0] / 1e4, COMPARISON_RANGE_A[1] / 1e4)
        panel_label(top, f"({'ab'[col]})", x=0.03, y=0.95)
        panel_label(bottom, f"({'cd'[col]})", x=0.03, y=0.93)
        summary[config_key] = {
            "epoch_gyr": library.epoch_gyr,
            "panel": {
                "log_z": PANEL_LOG_Z,
                "t_q_gyr": t_q_panel,
                "delay_gyr": PANEL_DELAY_GYR,
                "by_tau_q": panel_summary,
            },
            "best_match": (
                {key: value for key, value in best.items() if key != "spectrum"}
                if best is not None
                else None
            ),
            "best_match_median_observed_over_model_in_feature": (
                float(np.median(best_ratio[feature])) if best_ratio is not None else None
            ),
        }
    handles = [
        Line2D([], [], color="black", lw=1.3, label="observed stack (Lu+2026, 19 galaxies)"),
        Patch(facecolor="0.55", alpha=0.35, label="galaxy-to-galaxy scatter of the stack"),
    ]
    handles += [
        Line2D([], [], color=color, lw=1.0, label=rf"$\tau_q = {tau_q:g}$ Gyr")
        for tau_q, color in zip(PANEL_TAU_Q_GYR, PANEL_COLORS, strict=True)
    ]
    handles += [
        Line2D(
            [],
            [],
            color=CONFIG_COLORS[key],
            lw=1.1,
            ls="--",
            label=f"{AGB_CONFIGS[key]['short_label']}: closest model in the search",
        )
        for key in matches
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
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
    median_redshift = float(np.median([s["redshift"] for s in spectra.values()]))
    epoch = mock_epoch_gyr(median_redshift, edges)
    libraries = {key: MockLibrary(key, epoch, edges) for key in CONFIG_ORDER}
    matches, tables = {}, {}
    for key in SEARCH_CONFIGS:
        matches[key], tables[key] = search_best_match(libraries[key], grid_a, stack, scatter)
        print(
            f"{key}: best match log Z {matches[key]['log_z']:+.2f}, "
            f"t_q {matches[key]['t_q_gyr']:.1f}, tau_q {matches[key]['tau_q_gyr']:.2f} Gyr, "
            f"mismatch {matches[key]['mismatch']:.2f}"
        )
    summary = {
        "observed": {
            "n_galaxies": len(spectra),
            "redshift_range": [
                min(s["redshift"] for s in spectra.values()),
                max(s["redshift"] for s in spectra.values()),
            ],
            "median_redshift": median_redshift,
            "cosmic_age_at_median_redshift_gyr": cosmic_age_gyr(median_redshift),
            "mock_epoch_gyr": epoch,
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
        "search": {
            "log_z_grid": list(SEARCH_LOG_Z),
            "t_q_step_gyr": SEARCH_T_Q_STEP_GYR,
            "tau_q_grid_gyr": list(SEARCH_TAU_Q_GYR),
            "mismatch_definition": "RMS of (model - stack) / scatter over 1.494-1.791 um on "
            "the 30 A stack grid",
            "configurations_searched": list(SEARCH_CONFIGS),
            "n_models_per_configuration": int(tables[SEARCH_CONFIGS[0]].shape[0]),
            "mismatch_range": {
                key: [float(tables[key][:, 3].min()), float(tables[key][:, 3].max())]
                for key in SEARCH_CONFIGS
            },
        },
        "mocks": figure_stack_versus_mocks(grid_a, stack, scatter, libraries, matches, out_dir),
        "normalisation": "straight line fitted to all pixels in the blue (1.494-1.539 um) "
        "and red (1.746-1.791 um) windows, observed and model alike; bump index = "
        "-2.5 log10(mean normalised flux over 1.570-1.734 um)",
    }
    for key in SEARCH_CONFIGS:
        np.save(out_dir / f"{STEM}_search_{key}.npy", tables[key])
    (out_dir / f"{STEM}.json").write_text(json.dumps(_to_native(summary), indent=2) + "\n")
    print(f"wrote {out_dir / STEM}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
