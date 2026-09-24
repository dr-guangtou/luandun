"""Phase 5: does the choice of SFH family change the Phase 3/4 conclusions?

Five figures compare the delayed-tau `exponential` family (Phase 3/4) against three
alternatives (`linear`, `truncation`, `decoupled`; `sfh_model.FAMILIES`), for both
TP-AGB template sets (C3K, LW02), all traced from the same 2000 draws of
`(t_q, tau_q, log_z)`:

- S1: the fiducial history's SFR and the three indices versus time, one family per
  line style/color, one track per family computed directly with
  `csp_integrate.epoch_weight_matrix_from_cumulative` (the `csp_track`-equivalent
  call for non-exponential families; `csp_track` itself only supports `exponential`).
- S2: the population's 16-84 band of the bump versus D4000 per family, and the
  rapid-quenching class density contour.
- S3: class fractions per family (stacked bars).
- S4: the rapid-quenching kNN classifier's completeness/purity gain from adding the
  bump at 0.010 mag, agb = 2, unbalanced, 3 noise seeds, per family.
- S5: the SFH-recovery RMS gain from the bump at 0.005 and 0.010 mag, per family.
- S6: the paired TP-AGB offset bump(agb2) - bump(agb0) versus D4000, per family and
  template (the population tables already carry the agb0/agb1/agb2 columns, so this
  needs no rerun), plus the C3K-versus-LW02 band-median separation per family.

Every classifier/regression helper is imported from `population_classes.py`,
`analysis_fast_quenching.py` and `analysis_conclusion_figures.py`, not copied.

`--only` (comma-separated figure ids, e.g. `--only s6`) regenerates a subset of
figures/summary entries without recomputing the others: it loads the existing
`sfh_sensitivity_summary.json` (if present) and merges the requested entries into it,
leaving every other figure's entry untouched. S6 needs the population tables (always
loaded when any of s2/s3/s4/s5/s6 is requested) and S2's per-family band medians (always
recomputed together with S6, cheaply, since `compute_s2` and the band percentile fit
take a fraction of a second; only the S2 PNG/JSON entry itself is skipped unless `s2` is
also requested).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_conclusion_figures import (
    C1_D4000_SLICES,
    C1_N_BINS,
    C1_YARDSTICK_MAG,
    GRID_DIRS,
    TEMPLATE_LABELS,
    TEMPLATE_ORDER,
    _c1_slice_mask,
    load_post_quench_table,
    percentile_band,
    run_regression_template,
    summarize_c2c,
)
from analysis_fast_quenching import (
    NOISE_SEEDS,
    across_seed_statistics,
    class_summary,
    is_post_starburst,
    load_population,
    run_classifier,
)
from csp_integrate import agb_two_spectra, csp_spectra, epoch_weight_matrix_from_cumulative
from csp_integrate import interpolate_log_z as interp_log_z
from population_classes import RAPID_QUENCHING, assign_classes
from run_single_csp import FIDUCIAL
from sfh_model import (
    FAMILIES,
    TAU_GYR_RANGE,
    sfh_family_cumulative,
    star_formation_rate_family,
    time_bin_edges,
)
from spectral_indices import d4000, h_minus_bump, hdelta_a
from ssp_grid import load_ssp_grid

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output" / "analysis"

POPULATION_DIRS = {
    "exponential": {"c3k": PROJECT_DIR / "output" / "population"},
    "linear": {"c3k": PROJECT_DIR / "output" / "population_linear"},
    "truncation": {"c3k": PROJECT_DIR / "output" / "population_truncation"},
    "decoupled": {"c3k": PROJECT_DIR / "output" / "population_decoupled"},
}
for _family in FAMILIES:
    _suffix = "" if _family == "exponential" else f"_{_family}"
    POPULATION_DIRS[_family]["lw02"] = PROJECT_DIR / "output" / f"population{_suffix}_lw02"

DECOUPLED_FIDUCIAL_TAU_GYR = float(np.sqrt(TAU_GYR_RANGE[0] * TAU_GYR_RANGE[1]))
BUMP_PRODUCT = "sigma300"
AGB_KEY = "agb2"
CLASSIFIER_BUMP_KEY = f"bump_{BUMP_PRODUCT}_0.010"
REGRESSION_PRECISIONS_SHOWN = (0.005, 0.010)
N_FOLDS = 5

FAMILY_STYLES = {
    "exponential": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
    "linear": {"color": "tab:orange", "linestyle": "--", "marker": "s"},
    "truncation": {"color": "tab:green", "linestyle": ":", "marker": "^"},
    "decoupled": {"color": "tab:red", "linestyle": "-.", "marker": "D"},
}
TEMPLATE_HATCH = {"c3k": "", "lw02": "//"}

FIGURE_IDS = ("s1", "s2", "s3", "s4", "s5", "s6")


def _to_native(obj):
    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# S1: fiducial track per family, both templates
# ---------------------------------------------------------------------------


def family_fiducial_track(grid_dir, family, edges_gyr, agb_key=AGB_KEY):
    """The `csp_track`-equivalent call for any SFH family: build the epoch weight
    matrix directly from `sfh_family_cumulative` via
    `epoch_weight_matrix_from_cumulative` (the generic hook `csp_track` itself does
    not expose), then measure the three indices, at the fiducial (t_q, tau_q, log_z).
    """
    tau_gyr = DECOUPLED_FIDUCIAL_TAU_GYR if family == "decoupled" else None
    cumulative_fn = sfh_family_cumulative(
        family, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], tau_gyr=tau_gyr
    )
    sfr_fn = star_formation_rate_family(
        family, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], tau_gyr=tau_gyr
    )
    grid = load_ssp_grid(grid_dir, "sigma300")
    weights = epoch_weight_matrix_from_cumulative(edges_gyr, cumulative_fn, grid.log_age_yr)
    flux_agb0 = csp_spectra(
        weights, interp_log_z(grid.flux_nu[:, 0], grid.log_z_grid, FIDUCIAL["log_z"])
    )
    flux_agb1 = csp_spectra(
        weights, interp_log_z(grid.flux_nu[:, 1], grid.log_z_grid, FIDUCIAL["log_z"])
    )
    flux = flux_agb0 if agb_key == "agb0" else agb_two_spectra(flux_agb0, flux_agb1)
    return {
        "epoch_gyr": edges_gyr[1:],
        "time_since_quenching_gyr": edges_gyr[1:] - FIDUCIAL["t_q_gyr"],
        "sfr": sfr_fn(edges_gyr[1:]),
        "d4000": d4000(grid.wave_a, flux),
        "hdelta_a": hdelta_a(grid.wave_a, flux),
        "h_minus_bump": h_minus_bump(grid.wave_a, flux),
        "tau_gyr": tau_gyr,
    }


def compute_s1_tracks(edges_gyr):
    return {
        template: {
            family: family_fiducial_track(GRID_DIRS[template], family, edges_gyr)
            for family in FAMILIES
        }
        for template in TEMPLATE_ORDER
    }


def figure_s1(tracks_by_template, out_path):
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), layout="constrained", sharex="col")
    columns = ("sfr", "d4000", "hdelta_a", "h_minus_bump")
    column_titles = ("SFR [arbitrary]", "D4000", r"H$\delta_A$ [$\AA$]", r"H$^-$ bump [mag]")
    for row, template in enumerate(TEMPLATE_ORDER):
        for col, (key, title) in enumerate(zip(columns, column_titles, strict=True)):
            axis = axes[row, col]
            for family in FAMILIES:
                track = tracks_by_template[template][family]
                style = FAMILY_STYLES[family]
                x = track["epoch_gyr"] if key == "sfr" else track["time_since_quenching_gyr"]
                axis.plot(x, track[key], label=family, lw=1.6, markevery=15, ms=4.5, **style)
            axis.set_ylabel(title)
            if key == "h_minus_bump":
                axis.invert_yaxis()
            if col == 0:
                axis.axvline(FIDUCIAL["t_q_gyr"], color="0.6", ls=":", lw=1.0, zorder=0)
            else:
                axis.axvline(0.0, color="0.6", ls=":", lw=1.0, zorder=0)
            if row == 1:
                axis.set_xlabel(
                    "time since SF start [Gyr]" if col == 0 else "time since quenching [Gyr]",
                    fontsize=9,
                )
        axes[row, 0].annotate(
            TEMPLATE_LABELS[template],
            xy=(0.04, 0.92),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=10)
    figure.suptitle(
        f"fiducial history (t_q={FIDUCIAL['t_q_gyr']:g}, tau_q={FIDUCIAL['tau_q_gyr']:g} Gyr, "
        f"log Z={FIDUCIAL['log_z']:g}), agb=2; decoupled tau={DECOUPLED_FIDUCIAL_TAU_GYR:.3f} Gyr",
        fontsize=12,
    )
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def summarize_s1(tracks_by_template):
    return {
        template: {
            family: {
                key: track[key]
                for key in (
                    "epoch_gyr",
                    "time_since_quenching_gyr",
                    "sfr",
                    "d4000",
                    "hdelta_a",
                    "h_minus_bump",
                    "tau_gyr",
                )
            }
            for family, track in families.items()
        }
        for template, families in tracks_by_template.items()
    }


# ---------------------------------------------------------------------------
# Population loading and class assignment, shared by S2-S5
# ---------------------------------------------------------------------------


def load_family_population(family, template):
    input_path = POPULATION_DIRS[family][template] / "indices.npz"
    table = load_population(input_path)
    codes = assign_classes(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    post_starburst = is_post_starburst(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    return table, codes, post_starburst


# ---------------------------------------------------------------------------
# S2: population band and rapid-quenching locus, per family/template
# ---------------------------------------------------------------------------


def compute_s2(table, codes, family):
    x = table[f"d4000_{AGB_KEY}"]
    y = table[f"h_minus_bump_{BUMP_PRODUCT}_{AGB_KEY}"]
    rq_mask = codes == RAPID_QUENCHING
    slices = {}
    for lo, hi in C1_D4000_SLICES:
        is_last = (lo, hi) == C1_D4000_SLICES[-1]
        mask = _c1_slice_mask(x, lo, hi, is_last)
        values = y[mask]
        slices[f"d4000_{lo:.1f}_{hi:.1f}"] = {
            "n_rows": int(values.size),
            "median_mag": float(np.median(values)) if values.size else float("nan"),
        }
    locus = {
        "n_rapid_quenching": int(rq_mask.sum()),
        "d4000_median": float(np.median(x[rq_mask])) if rq_mask.any() else float("nan"),
        "h_minus_bump_median": float(np.median(y[rq_mask])) if rq_mask.any() else float("nan"),
    }
    return {"x": x, "y": y, "rq_mask": rq_mask, "band_slices": slices, "rq_locus_centroid": locus}


def compute_s2_summary(s2_by_template):
    """The S2 percentile bands (16-84 of the bump vs. D4000, `C1_N_BINS` bins spanning
    each template's own concatenated D4000 range across the 4 families) and the
    per-family D4000-slice/rapid-quenching-locus numbers already in `s2_by_template`.
    No plotting; split out from the original `figure_s2` so S6 can reuse the band
    medians (and the same bin edges) without redrawing the S2 PNG. Returns
    `(summary, edges_x_by_template)`; `edges_x_by_template` is not written to the JSON,
    only used internally (by `draw_s2_figure` and `compute_s6_summary`)."""
    summary = {}
    edges_x_by_template = {}
    for template in TEMPLATE_ORDER:
        entries = s2_by_template[template]
        x_all = np.concatenate([entries[family]["x"] for family in FAMILIES])
        edges_x = np.linspace(x_all.min(), x_all.max(), C1_N_BINS + 1)
        edges_x_by_template[template] = edges_x
        summary[template] = {}
        for family in FAMILIES:
            entry = entries[family]
            centers, lower, median, upper = percentile_band(entry["x"], entry["y"], edges_x)
            summary[template][family] = {
                "band_bin_centers": centers.tolist(),
                "band_p16_mag": lower.tolist(),
                "band_median_mag": median.tolist(),
                "band_p84_mag": upper.tolist(),
                "d4000_slices": entry["band_slices"],
                "rq_locus_centroid": entry["rq_locus_centroid"],
            }
    return summary, edges_x_by_template


def draw_s2_figure(s2_by_template, band_summary, edges_x_by_template, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for col, template in enumerate(TEMPLATE_ORDER):
        axis = axes[col]
        entries = s2_by_template[template]
        edges_x = edges_x_by_template[template]
        y_all = np.concatenate([entries[family]["y"] for family in FAMILIES])
        edges_y = np.linspace(y_all.min(), y_all.max(), 30)
        centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
        for family in FAMILIES:
            entry = entries[family]
            style = FAMILY_STYLES[family]
            band = band_summary[template][family]
            centers = np.array(band["band_bin_centers"])
            lower = np.array(band["band_p16_mag"])
            median = np.array(band["band_median_mag"])
            upper = np.array(band["band_p84_mag"])
            axis.fill_between(centers, lower, upper, color=style["color"], alpha=0.18, linewidth=0)
            axis.plot(
                centers, median, color=style["color"], ls=style["linestyle"], lw=1.8, label=family
            )
            rq_x = entry["x"][entry["rq_mask"]]
            rq_y = entry["y"][entry["rq_mask"]]
            if rq_x.size:
                density, _, _ = np.histogram2d(rq_x, rq_y, bins=[edges_x, edges_y])
                if density.max() > 0:
                    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
                    levels = 0.3 * density.max() * np.array([1.0, 2.0, 3.0])
                    levels = np.unique(levels[levels > 0])
                    if levels.size:
                        axis.contour(
                            centers_x,
                            centers_y,
                            density.T,
                            levels=levels,
                            colors=style["color"],
                            linewidths=0.9,
                            linestyles=":",
                        )
        axis.invert_yaxis()
        axis.set_xlabel("D4000")
        axis.set_ylabel(r"H$^-$ bump [mag]")
        axis.set_title(
            f"{TEMPLATE_LABELS[template]}, agb2 (dotted contours: rapid-quenching density)",
            loc="left",
            fontsize=9,
        )
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# S3: class fractions per family (stacked bars)
# ---------------------------------------------------------------------------


def figure_s3(class_summaries, out_path):
    from population_classes import CLASS_NAMES

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
    class_colors = {
        "star_forming": "gold",
        "transitional": "gray",
        "quiescent": "firebrick",
        "rapid_quenching": "deepskyblue",
    }
    for col, template in enumerate(TEMPLATE_ORDER):
        axis = axes[col]
        bottoms = np.zeros(len(FAMILIES))
        x = np.arange(len(FAMILIES))
        for class_name in CLASS_NAMES:
            heights = np.array(
                [
                    class_summaries[family][template]["fraction_percent"][class_name]
                    for family in FAMILIES
                ]
            )
            axis.bar(
                x,
                heights,
                bottom=bottoms,
                color=class_colors[class_name],
                label=class_name,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += heights
        axis.set_xticks(x)
        axis.set_xticklabels(FAMILIES, rotation=20, ha="right")
        axis.set_title(TEMPLATE_LABELS[template], loc="left", fontsize=12)
        if col == 0:
            axis.set_ylabel(r"class fraction [\%]", fontsize=12)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# S4: rapid-quenching classifier gain per family/template (agb2, 0.010 mag)
# ---------------------------------------------------------------------------


def compute_s4(table, codes, seeds=NOISE_SEEDS, n_folds=N_FOLDS):
    """Rapid-quenching classifier gain from the bump, agb2/unbalanced, at
    `CLASSIFIER_BUMP_KEY`. Returns the raw per-seed paired fold difference
    (`paired_gain_per_seed`, `population_classes`'s `completeness_gain_mean`/
    `completeness_gain_standard_error`/`completeness_gain_over_standard_error` and the
    `purity_` counterparts, one 5-fold cross-validation per seed) alongside the
    across-seed mean/std of every metric (`across_seed_statistics`)."""
    results_by_seed = {
        str(seed): {AGB_KEY: run_classifier(table, codes, AGB_KEY, n_folds=n_folds, seed=seed)}
        for seed in seeds
    }
    across_seeds = across_seed_statistics(results_by_seed)
    paired_gain_per_seed = {
        seed: results[AGB_KEY][CLASSIFIER_BUMP_KEY]["unbalanced"]["paired_difference_vs_no_bump"]
        for seed, results in results_by_seed.items()
    }
    bump_entry = dict(across_seeds[AGB_KEY][CLASSIFIER_BUMP_KEY]["unbalanced"])
    bump_entry["paired_gain_per_seed"] = paired_gain_per_seed
    return {
        "no_bump": across_seeds[AGB_KEY]["no_bump"]["unbalanced"],
        CLASSIFIER_BUMP_KEY: bump_entry,
    }


def figure_s4(s4_by_family_template, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    width = 0.35
    x = np.arange(len(FAMILIES))
    for axis, metric, title in zip(
        axes, ("completeness", "purity"), ("completeness gain", "purity gain"), strict=True
    ):
        for offset, template in zip((-width / 2, width / 2), TEMPLATE_ORDER, strict=True):
            gains = np.array(
                [
                    s4_by_family_template[family][template][CLASSIFIER_BUMP_KEY][
                        f"{metric}_gain_mean_mean_over_seeds"
                    ]
                    for family in FAMILIES
                ]
            )
            errors = np.array(
                [
                    s4_by_family_template[family][template][CLASSIFIER_BUMP_KEY][
                        f"{metric}_gain_mean_std_over_seeds"
                    ]
                    for family in FAMILIES
                ]
            )
            axis.bar(
                x + offset,
                gains,
                width,
                yerr=errors,
                capsize=3,
                label=TEMPLATE_LABELS[template],
                hatch=TEMPLATE_HATCH[template],
                color="tab:blue" if template == "c3k" else "tab:red",
                alpha=0.75,
            )
        axis.axhline(0.0, color="0.3", lw=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(FAMILIES, rotation=20, ha="right")
        axis.set_ylabel(f"{title} [dimensionless]")
        axis.set_title(title, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "rapid-quenching classifier gain (+bump at 0.010 mag) minus optical-only\n"
        "agb2, unbalanced; error bars: std of the per-seed paired-gain mean over 3 noise seeds",
        fontsize=9,
    )
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# S5: SFH-recovery RMS gain per family/template (agb2, 0.005/0.010 mag)
# ---------------------------------------------------------------------------


def compute_s5(post_quench_table, seeds=NOISE_SEEDS, n_folds=N_FOLDS):
    label = "agb2"
    results_by_seed = {
        str(seed): {
            label: run_regression_template(post_quench_table, AGB_KEY, seed, n_folds=n_folds)
        }
        for seed in seeds
    }
    return summarize_c2c(results_by_seed, n_folds=n_folds)[label]


TARGET_NAMES = ("log10_time_since_quenching_gyr", "log10_tau_q_gyr")
TARGET_LABELS = {
    "log10_time_since_quenching_gyr": "log10 time since quenching [Gyr]",
    "log10_tau_q_gyr": "log10 tau_q [Gyr]",
}


def figure_s5(s5_by_family_template, out_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    width = 0.35
    x = np.arange(len(FAMILIES))
    for row, target_name in enumerate(TARGET_NAMES):
        for col, precision in enumerate(REGRESSION_PRECISIONS_SHOWN):
            axis = axes[row, col]
            key = f"bump_{precision:.3f}"
            for offset, template in zip((-width / 2, width / 2), TEMPLATE_ORDER, strict=True):
                gains = np.array(
                    [
                        s5_by_family_template[family][template][key][
                            "paired_gain_mean_over_seeds_dex"
                        ][row]
                        for family in FAMILIES
                    ]
                )
                errors = np.array(
                    [
                        s5_by_family_template[family][template][key][
                            "paired_gain_standard_error_mean_over_seeds_dex"
                        ][row]
                        for family in FAMILIES
                    ]
                )
                axis.bar(
                    x + offset,
                    gains,
                    width,
                    yerr=errors,
                    capsize=3,
                    label=TEMPLATE_LABELS[template],
                    hatch=TEMPLATE_HATCH[template],
                    color="tab:blue" if template == "c3k" else "tab:red",
                    alpha=0.75,
                )
            axis.axhline(0.0, color="0.3", lw=0.8)
            axis.set_xticks(x)
            axis.set_xticklabels(FAMILIES, rotation=20, ha="right")
            axis.set_ylabel("RMS gain [dex]")
            axis.set_title(f"{TARGET_LABELS[target_name]}, {precision:.3f} mag", fontsize=9)
            if row == 0 and col == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "SFH-recovery RMS gain (bump minus no-bump), agb2; "
        "error bars: mean over 3 noise seeds of the per-fold standard error",
        fontsize=9,
    )
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# S6: paired TP-AGB offset bump(agb2) - bump(agb0) per family/template, and the
# C3K-versus-LW02 band-median separation per family (reuses S2's band medians)
# ---------------------------------------------------------------------------


def compute_s6(table, family):
    """The paired per-row TP-AGB offset (sigma300 bump, agb2 minus agb0) and the two
    raw bump columns, binned later by `d4000_agb2` (the same D4000 column S2 bins on),
    at every epoch >= 1 Gyr already in `table` (`load_family_population`'s filter)."""
    return {
        "x": table[f"d4000_{AGB_KEY}"],
        "offset": table[f"h_minus_bump_{BUMP_PRODUCT}_agb2"]
        - table[f"h_minus_bump_{BUMP_PRODUCT}_agb0"],
        "bump_agb0": table[f"h_minus_bump_{BUMP_PRODUCT}_agb0"],
        "bump_agb2": table[f"h_minus_bump_{BUMP_PRODUCT}_agb2"],
    }


def _half_width_16_84(values):
    p16, p84 = np.percentile(values, [16, 84])
    return float(0.5 * (p84 - p16))


def compute_s6_summary(s6_by_family_template, edges_x_by_template):
    """Per family/template: the paired-offset 16-84 band in the same `C1_N_BINS` D4000
    bins S2 uses (`edges_x_by_template`, from `compute_s2_summary`), and the three
    Phase 4 D4000 slices with the median offset, the pooled per-model RMS scatter
    (Phase 3's corrected metric, `sqrt((hw0^2 + hw2^2) / 2)` with `hw` the 16-84
    half-width of each model's own bump within the slice) and offset/scatter."""
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for template in TEMPLATE_ORDER:
            entry = s6_by_family_template[family][template]
            edges_x = edges_x_by_template[template]
            centers, lower, median, upper = percentile_band(entry["x"], entry["offset"], edges_x)
            slices = {}
            for lo, hi in C1_D4000_SLICES:
                is_last = (lo, hi) == C1_D4000_SLICES[-1]
                mask = _c1_slice_mask(entry["x"], lo, hi, is_last)
                offset_slice = entry["offset"][mask]
                median_offset = (
                    float(np.median(offset_slice)) if offset_slice.size else float("nan")
                )
                hw0 = (
                    _half_width_16_84(entry["bump_agb0"][mask])
                    if offset_slice.size
                    else float("nan")
                )
                hw2 = (
                    _half_width_16_84(entry["bump_agb2"][mask])
                    if offset_slice.size
                    else float("nan")
                )
                pooled_scatter = float(np.sqrt((hw0**2 + hw2**2) / 2.0))
                slices[f"d4000_{lo:.1f}_{hi:.1f}"] = {
                    "n_rows": int(offset_slice.size),
                    "median_offset_mag": median_offset,
                    "hw0_mag": hw0,
                    "hw2_mag": hw2,
                    "pooled_scatter_mag": pooled_scatter,
                    "offset_over_scatter": (
                        median_offset / pooled_scatter if pooled_scatter > 0 else float("nan")
                    ),
                }
            summary[family][template] = {
                "band_bin_centers": centers.tolist(),
                "band_p16_offset_mag": lower.tolist(),
                "band_median_offset_mag": median.tolist(),
                "band_p84_offset_mag": upper.tolist(),
                "d4000_slices": slices,
            }
    return summary


def compute_template_separation(s2_band_summary):
    """Per family: the LW02-minus-C3K band-median separation (agb2), interpolated
    onto C3K's D4000 bin centers (the two templates' `compute_s2_summary` bin edges are
    numerically close but not identical -- the agb2 = 2*agb1 - agb0 linear combination
    uses each template's own agb1 spectrum, so D4000 itself differs by up to about
    0.01 mag between templates at fixed history/epoch), and its maximum absolute value."""
    separation = {}
    for family in FAMILIES:
        c3k_band = s2_band_summary["c3k"][family]
        lw02_band = s2_band_summary["lw02"][family]
        centers = np.array(c3k_band["band_bin_centers"])
        median_c3k = np.array(c3k_band["band_median_mag"])
        lw02_centers = np.array(lw02_band["band_bin_centers"])
        median_lw02 = np.array(lw02_band["band_median_mag"])
        valid = np.isfinite(lw02_centers) & np.isfinite(median_lw02)
        median_lw02_interp = np.interp(centers, lw02_centers[valid], median_lw02[valid])
        separation_mag = median_lw02_interp - median_c3k
        finite = np.isfinite(separation_mag)
        separation[family] = {
            "band_bin_centers": centers.tolist(),
            "separation_mag": separation_mag.tolist(),
            "max_abs_separation_mag": (
                float(np.max(np.abs(separation_mag[finite]))) if finite.any() else float("nan")
            ),
        }
    return separation


def _yardstick_reference(offset_summary, template):
    """A representative (x, y) for the 0.01 mag yardstick: the 3rd finite D4000 bin
    center, and the mean over families of the median offset there (families share the
    same bin centers within a template, since `compute_s2_summary` fits every family's
    band on that template's single shared set of edges)."""
    centers = np.array(offset_summary[FAMILIES[0]][template]["band_bin_centers"])
    finite = np.flatnonzero(np.isfinite(centers))
    x_ref = centers[finite[2]] if finite.size > 2 else centers[finite[0]]
    medians_at_ref = [
        np.interp(
            x_ref,
            offset_summary[family][template]["band_bin_centers"],
            offset_summary[family][template]["band_median_offset_mag"],
        )
        for family in FAMILIES
    ]
    return x_ref, float(np.mean(medians_at_ref))


def figure_s6(offset_summary, separation_summary, out_path):
    mosaic = [["c3k_offset", "sep"], ["lw02_offset", "sep"]]
    figure, axd = plt.subplot_mosaic(mosaic, figsize=(15, 11), layout="constrained")
    legend_handles = None
    for template, panel_key in (("c3k", "c3k_offset"), ("lw02", "lw02_offset")):
        axis = axd[panel_key]
        for family in FAMILIES:
            style = FAMILY_STYLES[family]
            band = offset_summary[family][template]
            centers = np.array(band["band_bin_centers"])
            lower = np.array(band["band_p16_offset_mag"])
            median = np.array(band["band_median_offset_mag"])
            upper = np.array(band["band_p84_offset_mag"])
            axis.fill_between(centers, lower, upper, color=style["color"], alpha=0.15, linewidth=0)
            axis.plot(
                centers, median, color=style["color"], ls=style["linestyle"], lw=1.8, label=family
            )
        x_ref, y_ref = _yardstick_reference(offset_summary, template)
        axis.errorbar(
            [x_ref],
            [y_ref],
            yerr=[C1_YARDSTICK_MAG],
            fmt="none",
            ecolor="black",
            capsize=4,
            label=f"{C1_YARDSTICK_MAG:.3f} mag yardstick",
        )
        axis.invert_yaxis()
        axis.set_xlabel("D4000")
        axis.set_ylabel(r"H$^-$ bump offset [mag]", fontsize=13)
        axis.set_title(f"{TEMPLATE_LABELS[template]}: agb2 $-$ agb0", loc="left", fontsize=10)
        if legend_handles is None:
            legend_handles = axis.get_legend_handles_labels()

    axis = axd["sep"]
    for family in FAMILIES:
        style = FAMILY_STYLES[family]
        sep = separation_summary[family]
        axis.plot(
            sep["band_bin_centers"],
            sep["separation_mag"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.8,
        )
    axis.axhline(0.0, color="0.3", lw=0.8)
    axis.invert_yaxis()
    axis.set_xlabel("D4000")
    axis.set_ylabel(r"band-median bump separation [mag]", fontsize=13)
    axis.set_title("LW02 $-$ C3K, agb2", loc="left", fontsize=10)
    figure.legend(*legend_handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    figure.suptitle(
        "per-family TP-AGB offset (left column) and C3K-LW02 template separation "
        f"(right), agb2, sigma300 bump, {C1_N_BINS} D4000 bins",
        fontsize=10,
    )
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_pilot():
    start = time.perf_counter()
    edges = time_bin_edges()
    t0 = time.perf_counter()
    tracks = compute_s1_tracks(edges)
    print(f"s1 tracks: {time.perf_counter() - t0:.2f} s")
    figure_s1(tracks, Path("/tmp/s1_pilot.png"))

    family, template = "linear", "c3k"
    t0 = time.perf_counter()
    table = load_population(POPULATION_DIRS[family][template] / "indices.npz", stride=50)
    codes = assign_classes(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    print(f"load+classes (stride 50): {time.perf_counter() - t0:.2f} s, {codes.size} rows")

    t0 = time.perf_counter()
    compute_s4(table, codes, seeds=(NOISE_SEEDS[0],), n_folds=2)
    elapsed = time.perf_counter() - t0
    projected = elapsed * (N_FOLDS / 2) * len(NOISE_SEEDS) * len(FAMILIES) * len(TEMPLATE_ORDER)
    print(f"s4 pilot (1 family/template, 1 seed, 2 folds): {elapsed:.2f} s")
    print(f"projected full s4 (all families/templates, 3 seeds, 5 folds): {projected / 60:.1f} min")

    print(f"pilot total: {time.perf_counter() - start:.2f} s")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: SFH-family sensitivity.")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--pilot", action="store_true", help="subsampled timing run, nothing saved")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated figure ids to (re)generate, e.g. 's6' or 's2,s6' "
        f"(any of {FIGURE_IDS}); default: all. A partial run merges the requested "
        "entries into the existing summary JSON (if any) without touching the rest.",
    )
    args = parser.parse_args()

    if args.pilot:
        _run_pilot()
        return

    requested = set(args.only.split(",")) if args.only else set(FIGURE_IDS)
    unknown = requested - set(FIGURE_IDS)
    if unknown:
        raise ValueError(
            f"unknown --only figure id(s) {sorted(unknown)}; expected any of {FIGURE_IDS}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "sfh_sensitivity_summary.json"
    existing_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    edges = time_bin_edges()
    start = time.perf_counter()
    timing = dict(existing_summary.get("timing_s", {}))

    s1_summary = existing_summary.get("s1_fiducial_families")
    if "s1" in requested:
        t0 = time.perf_counter()
        tracks = compute_s1_tracks(edges)
        figure_s1(tracks, out_dir / "s1_fiducial_families.png")
        s1_summary = summarize_s1(tracks)
        timing["s1_s"] = time.perf_counter() - t0
        print(f"s1 done in {timing['s1_s']:.1f} s")

    need_bands = bool(requested & {"s2", "s6"})
    need_tables = bool(requested & {"s2", "s3", "s4", "s5", "s6"})

    s2_by_template = {template: {} for template in TEMPLATE_ORDER}
    class_summaries = {family: {} for family in FAMILIES}
    s4_by_family_template = {family: {} for family in FAMILIES}
    s5_by_family_template = {family: {} for family in FAMILIES}
    s6_by_family_template = {family: {} for family in FAMILIES}
    population_provenance = dict(existing_summary.get("population_provenance", {}))

    if need_tables:
        for family in FAMILIES:
            for template in TEMPLATE_ORDER:
                combo_label = f"{family}/{template}"
                t0 = time.perf_counter()
                table, codes, post_starburst = load_family_population(family, template)
                t_load = time.perf_counter() - t0
                print(f"[{combo_label}] loaded {codes.size} rows in {t_load:.1f} s")

                population_provenance.setdefault(family, {})[template] = json.loads(
                    (POPULATION_DIRS[family][template] / "summary.json").read_text()
                )

                if "s3" in requested:
                    class_summaries[family][template] = class_summary(codes, post_starburst)

                if need_bands:
                    t0 = time.perf_counter()
                    s2_by_template[template][family] = compute_s2(table, codes, family)
                    print(f"[{combo_label}] s2 bands: {time.perf_counter() - t0:.1f} s")

                if "s4" in requested:
                    t0 = time.perf_counter()
                    s4_by_family_template[family][template] = compute_s4(table, codes)
                    print(f"[{combo_label}] s4 classifier: {time.perf_counter() - t0:.1f} s")

                if "s5" in requested:
                    t0 = time.perf_counter()
                    post_quench_table = load_post_quench_table(POPULATION_DIRS[family][template])
                    s5_by_family_template[family][template] = compute_s5(post_quench_table)
                    print(f"[{combo_label}] s5 regression: {time.perf_counter() - t0:.1f} s")

                if "s6" in requested:
                    t0 = time.perf_counter()
                    s6_by_family_template[family][template] = compute_s6(table, family)
                    print(f"[{combo_label}] s6 offset: {time.perf_counter() - t0:.1f} s")

    s2_band_summary = existing_summary.get("s2_population_bands")
    edges_x_by_template = None
    if need_bands:
        t0 = time.perf_counter()
        s2_band_summary, edges_x_by_template = compute_s2_summary(s2_by_template)
        timing["s2_bands_compute_s"] = time.perf_counter() - t0

    if "s2" in requested:
        t0 = time.perf_counter()
        draw_s2_figure(
            s2_by_template,
            s2_band_summary,
            edges_x_by_template,
            out_dir / "s2_population_bands.png",
        )
        timing["s2_figure_s"] = time.perf_counter() - t0

    if "s3" in requested:
        t0 = time.perf_counter()
        figure_s3(class_summaries, out_dir / "s3_class_fractions.png")
        timing["s3_figure_s"] = time.perf_counter() - t0

    if "s4" in requested:
        t0 = time.perf_counter()
        figure_s4(s4_by_family_template, out_dir / "s4_classifier_by_family.png")
        timing["s4_figure_s"] = time.perf_counter() - t0

    if "s5" in requested:
        t0 = time.perf_counter()
        figure_s5(s5_by_family_template, out_dir / "s5_recovery_by_family.png")
        timing["s5_figure_s"] = time.perf_counter() - t0

    s6_summary = existing_summary.get("s6_tpagb_offset_by_family")
    if "s6" in requested:
        t0 = time.perf_counter()
        s6_offset_summary = compute_s6_summary(s6_by_family_template, edges_x_by_template)
        s6_separation_summary = compute_template_separation(s2_band_summary)
        s6_summary = {
            "offset_by_family": s6_offset_summary,
            "c3k_lw02_separation_by_family": s6_separation_summary,
        }
        figure_s6(
            s6_offset_summary, s6_separation_summary, out_dir / "s6_tpagb_offset_by_family.png"
        )
        timing["s6_s"] = time.perf_counter() - t0
        print(f"s6 done in {timing['s6_s']:.1f} s")

    timing["total_s"] = time.perf_counter() - start
    timing["last_only"] = sorted(requested)

    definitions = dict(existing_summary.get("definitions", {}))
    definitions.update(
        {
            "classifier_bump_key": CLASSIFIER_BUMP_KEY,
            "classifier_agb": AGB_KEY,
            "classifier_noise_seeds": list(NOISE_SEEDS),
            "regression_agb": AGB_KEY,
            "regression_precisions_shown": list(REGRESSION_PRECISIONS_SHOWN),
            "regression_noise_seeds": list(NOISE_SEEDS),
            "n_folds": N_FOLDS,
            "s6_bump_product": BUMP_PRODUCT,
            "s6_offset_definition": "bump(agb2) - bump(agb0), paired per epoch row, "
            "epoch >= 1 Gyr, binned by d4000_agb2 in the same C1_N_BINS bins as S2",
            "s6_pooled_scatter_definition": "sqrt((hw0^2 + hw2^2) / 2), hw = 16-84 "
            "half-width of each model's own bump within the D4000 slice (Phase 3's "
            "corrected per-model RMS scatter metric)",
        }
    )

    summary = dict(existing_summary)
    summary["families"] = list(FAMILIES)
    summary["templates"] = list(TEMPLATE_ORDER)
    summary["decoupled_fiducial_tau_gyr"] = DECOUPLED_FIDUCIAL_TAU_GYR
    summary["population_provenance"] = population_provenance
    if "s1" in requested:
        summary["s1_fiducial_families"] = s1_summary
    if "s2" in requested:
        summary["s2_population_bands"] = s2_band_summary
    if "s3" in requested:
        summary["s3_class_fractions"] = class_summaries
    if "s4" in requested:
        summary["s4_classifier_by_family"] = s4_by_family_template
    if "s5" in requested:
        summary["s5_recovery_by_family"] = s5_by_family_template
    if "s6" in requested:
        summary["s6_tpagb_offset_by_family"] = s6_summary
    summary["definitions"] = definitions
    summary["timing_s"] = timing

    summary_text = json.dumps(_to_native(summary), indent=2) + "\n"
    summary_path.write_text(summary_text)
    print(f"wrote {out_dir} in {timing['total_s']:.1f} s (only={sorted(requested)})")


if __name__ == "__main__":
    main()
