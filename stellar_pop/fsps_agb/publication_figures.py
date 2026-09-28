"""Publication figure candidates for the H-minus bump model section.

Two take-home messages, each with a qualitative and a quantitative figure, plus
robustness and caveat figures:

- Figure 1 (index planes): the H-minus bump plane separates the TP-AGB-on and
  TP-AGB-off models, and inside the TP-AGB-on model the fast-quenching epochs occupy
  their own locus, ordered by the quenching timescale.
- Figure 2 (prescription offsets): the population bump locus versus the
  optical indices for four TP-AGB prescriptions, with the 0.01 mag yardstick.
- Figure 3 (quenching clocks): HdeltaA and the bump peak at different delays
  after quenching, and the lag between them grows with the quenching timescale.
- Figure 4 (information gain): rapid-quenching completeness and purity and the
  SFH-recovery RMS, optical only versus optical plus bump, TP-AGB on versus off.
- Figures 5 and 6: the same planes and gains for the other three SFH families.
- Figure 7 (metallicity): how much of the bump's metallicity trend comes from
  the non-AGB stars, and whether knowing the metallicity sharpens the bump.

Conventions: TP-AGB on = LW02 empirical O-rich TP-AGB templates at agb = 2; AGB
off = default C3K templates at agb = 0 (the no-TP-AGB control). Optical indices
at sigma = 300 km/s, the bump at R = 100. The default SFH is the tau = t_q
exponential-quench family; the other families appear only in Figures 5 and 6.
Every number drawn is written to `publication_summary.json`.
"""

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

from analysis_conclusion_figures import (
    NOISE_SEEDS,
    SEED,
    cross_validated_regression,
    load_population_table,
    load_post_quench_table,
    percentile_band,
    regression_targets,
)
from analysis_fast_quenching import D4000_SIGMA, HDELTA_A_SIGMA
from index_planes import PLANES
from population_classes import (
    RAPID_QUENCHING,
    add_measurement_noise,
    assign_classes,
    cross_validated_metrics,
)
from run_single_csp import FIDUCIAL, compute_track_indices
from sfh_model import FAMILIES, time_bin_edges

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output" / "publication"
GRID_DIRS = {
    "c3k": PROJECT_DIR / "output" / "ssp_grid",
    "lw02": PROJECT_DIR / "output" / "ssp_grid_lw02",
}


def population_dir(family, template):
    suffix = "" if family == "exponential" else f"_{family}"
    suffix += "" if template == "c3k" else "_lw02"
    return PROJECT_DIR / "output" / f"population{suffix}"


BUMP_PRODUCT = "r100"
OPTICAL_PRODUCT = "sigma300"
AGB_CONFIGS = {
    "agb_off": {
        "template": "c3k",
        "agb": "agb0",
        "label": "TP-AGB off (C3K, agb = 0)",
        "short_label": "TP-AGB off",
    },
    "agb_on": {
        "template": "lw02",
        "agb": "agb2",
        "label": "TP-AGB on (LW02, agb = 2)",
        "short_label": "TP-AGB on",
    },
}
CONFIG_COLORS = {"agb_off": "#0072B2", "agb_on": "#D55E00"}
CONFIG_ORDER = ("agb_off", "agb_on")

PRESCRIPTIONS = (
    ("c3k", "agb0", "C3K, agb = 0", "#0072B2", "-"),
    ("c3k", "agb1", "C3K, agb = 1", "#56B4E9", "--"),
    ("lw02", "agb1", "LW02, agb = 1", "#E69F00", "--"),
    ("lw02", "agb2", "LW02, agb = 2", "#D55E00", "-"),
)
YARDSTICK_MAG = 0.01
D4000_SLICE = (1.3, 1.5)
BAND_BINS = 20
HIST_BINS = 40

RAPID_QUENCHING_COLOR = "#d62728"
POPULATION_LEVELS = (0.68, 0.95, 0.995)
CLASS_LEVELS = (0.68, 0.95)
DENSITY_BINS = 90
DENSITY_SMOOTH_BINS = 1.2

LOG_Z_TRACKS = (-0.5, -0.25, 0.0, 0.25)
LOG_Z_TRACK_COLORS = tuple(plt.get_cmap("Blues")(level) for level in (0.4, 0.55, 0.7, 0.85))
TRACK_MARKER_OFFSETS_GYR = (0.0, 0.5, 1.0, 2.0, 5.0)
TRACK_MARKER_SHAPES = ("o", "s", "^", "D", "P")
TRACK_WINDOW_GYR = (0.0, 5.0)

CLOCK_TAU_Q_GYR = (0.1, 0.3, 1.0, 3.0)
CLOCK_TAU_Q_COLORS = tuple(plt.get_cmap("viridis")(level) for level in (0.0, 0.33, 0.66, 0.95))
CLOCK_T_Q_GYR = (1.5, 3.0, 4.5)
CLOCK_T_Q_STYLES = ("--", "-", ":")
CLOCK_TAU_Q_FINE_GYR = tuple(np.round(np.logspace(np.log10(0.1), np.log10(3.0), 9), 3))
CLOCK_WINDOW_GYR = (-0.5, 6.0)
CLOCK_MARKER_STEP_GYR = 0.5
CLOCK_MARKER_MAX_GYR = 3.0

BUMP_PRECISIONS = (0.005, 0.010, 0.020)
CLASSIFIER_PRECISION = 0.010
RECOVERY_PRECISION = 0.005
LOG_Z_SIGMA_DEX = 0.1
K_NEIGHBORS = 25
N_FOLDS = 5
TARGET_LABELS = (r"$\log_{10}(t - t_q)$", r"$\log_{10}\tau_q$")

FAMILY_LABELS = {
    "exponential": r"exponential ($\tau = t_q$)",
    "linear": "linear ramp",
    "truncation": "truncation",
    "decoupled": r"decoupled $\tau$",
}
FAMILY_SHORT_LABELS = {
    "exponential": "exp.",
    "linear": "linear",
    "truncation": "trunc.",
    "decoupled": "decoupled",
}
ROBUSTNESS_FAMILIES = ("linear", "truncation", "decoupled")

METALLICITY_OFFSETS_GYR = (0.0, 1.0, 2.0, 5.0)
METALLICITY_D4000_SLICE = (1.4, 1.7)

AXIS_LABELS = {
    "d4000": r"D4000",
    "hdelta_a": r"H$\delta_{\rm A}$ [\AA]",
    "h_minus_bump": r"H$^-$ bump [mag]",
}
FIGURE_IDS = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7")
FINAL_DIR = OUTPUT_DIR / "final"
FINAL_STEMS = {
    "fig1": "index_planes_agb_on_off",
    "fig5": "index_planes_sfh_families",
    "combined": "age_sensitivity_and_classifier_gain",
    "jwst": "d4000_hminus_plane_with_jwst_data",
    "jwst_epoch": "d4000_hminus_plane_with_jwst_data_epoch_matched",
    "jwst_epoch_agb1": "d4000_hminus_plane_with_jwst_data_epoch_matched_agb1",
}
JWST_SYMBOL_SIZE = 3.8 * 1.5
JWST_HUBBLE_CONSTANT = 70.0
JWST_OMEGA_MATTER = 0.3
JWST_DATA_PATH = FINAL_DIR / "JWST_QG_indices.npz"
CLOCK_LOG_WINDOW_GYR = (0.05, 6.0)
INDEX_LINESTYLES = {"hdelta_a": "-", "h_minus_bump": "--", "d4000": ":"}
INDEX_COLORS = {"hdelta_a": "#0072B2", "h_minus_bump": "#D55E00", "d4000": "#009E73"}
INDEX_SHORT_LABELS = {
    "hdelta_a": r"H$\delta_{\rm A}$",
    "h_minus_bump": r"H$^-$ bump strength",
    "d4000": "D4000",
}
COMBINED_TAU_Q_GYR = (0.3, 1.0, 3.0)

PUBLICATION_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "pdf.fonttype": 42,
}
DOUBLE_COLUMN_IN = 7.1
SINGLE_COLUMN_IN = 3.5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    if isinstance(obj, dict):
        return {str(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def capitalise_sentence(text):
    """Upper-case the first letter of a legend entry treated as a sentence: skip a leading
    panel reference such as "(d, e) ", skip LaTeX math ($...$) and commands, and leave
    entries that start inside math (for example "$\\tau_q = 0.3$ Gyr") unchanged."""
    match = re.match(r"^\([^)]*\)\s*", text)
    start = match.end() if match else 0
    if text[start:].startswith("$"):
        return text
    for position in range(start, len(text)):
        character = text[position]
        if character == "$":
            return text
        if character.isalpha():
            if position > 0 and text[position - 1] == "\\":
                return text
            return text[:position] + character.upper() + text[position + 1 :]
    return text


def _capitalise_legends(figure):
    legends = list(figure.legends) + [
        axis.get_legend() for axis in figure.axes if axis.get_legend() is not None
    ]
    for legend in legends:
        for entry in legend.get_texts():
            entry.set_text(capitalise_sentence(entry.get_text()))
        title = legend.get_title()
        if title.get_text():
            title.set_text(capitalise_sentence(title.get_text()))


def save_figure(figure, out_dir, stem):
    _capitalise_legends(figure)
    for extension in ("pdf", "png"):
        figure.savefig(out_dir / f"{stem}.{extension}")
    plt.close(figure)


def panel_label(axis, text, x=0.03, y=0.95, ha="left"):
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        ha=ha,
        va="top",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
    )


def plane_columns(table, agb_key):
    return {
        "d4000": table[f"d4000_{agb_key}"],
        "hdelta_a": table[f"hdelta_a_{agb_key}"],
        "h_minus_bump": table[f"h_minus_bump_{BUMP_PRODUCT}_{agb_key}"],
    }


def track_series(result, agb_key):
    return {
        "epoch_gyr": result["epoch_gyr"],
        "d4000": result[agb_key][OPTICAL_PRODUCT]["d4000"],
        "hdelta_a": result[agb_key][OPTICAL_PRODUCT]["hdelta_a"],
        "h_minus_bump": result[agb_key][BUMP_PRODUCT]["h_minus_bump"],
    }


def load_family_table(family, template, stride=1):
    table = load_population_table(population_dir(family, template), stride=stride)
    codes = assign_classes(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    return table, codes


def density_levels(x, y, fractions, x_range, y_range, n_bins=DENSITY_BINS):
    """Smoothed 2-D histogram and the density levels enclosing `fractions` of the samples."""
    counts, x_edges, y_edges = np.histogram2d(
        x, y, bins=n_bins, range=[list(x_range), list(y_range)]
    )
    counts = gaussian_filter(counts, DENSITY_SMOOTH_BINS)
    flat = np.sort(counts.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    levels = [float(flat[np.searchsorted(cumulative, fraction)]) for fraction in fractions]
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, counts.T, sorted(set(levels))


def _plane_ranges(indices_list):
    ranges = {}
    for key in ("d4000", "hdelta_a", "h_minus_bump"):
        values = np.concatenate([indices[key] for indices in indices_list])
        low, high = np.percentile(values, [0.05, 99.95])
        pad = 0.06 * (high - low)
        ranges[key] = (low - pad, high + pad)
    return ranges


def draw_population_contours(axis, x, y, x_range, y_range, color="0.55", fill=True):
    x_centers, y_centers, density, levels = density_levels(
        x, y, POPULATION_LEVELS, x_range, y_range
    )
    if fill:
        axis.contourf(
            x_centers,
            y_centers,
            density,
            levels=[*levels, density.max() * 1.01],
            colors=["0.93", "0.85", "0.75"],
            zorder=1,
        )
    axis.contour(
        x_centers, y_centers, density, levels=levels, colors=color, linewidths=0.5, zorder=2
    )


def draw_class_contours(axis, x, y, x_range, y_range, color, levels=CLASS_LEVELS, zorder=3):
    x_centers, y_centers, density, level_values = density_levels(x, y, levels, x_range, y_range)
    axis.contourf(
        x_centers,
        y_centers,
        density,
        levels=[level_values[0], density.max() * 1.01],
        colors=[color],
        alpha=0.55,
        zorder=zorder,
    )
    axis.contour(
        x_centers,
        y_centers,
        density,
        levels=level_values,
        colors=[color],
        linewidths=(0.9, 0.6),
        zorder=zorder + 1,
    )


def set_plane_axes(axis, x_key, y_key, ranges, xlabel=True, ylabel=True):
    axis.set_xlim(*ranges[x_key])
    low, high = ranges[y_key]
    axis.set_ylim((high, low) if y_key == "h_minus_bump" else (low, high))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if xlabel:
        axis.set_xlabel(AXIS_LABELS[x_key])
    if ylabel:
        axis.set_ylabel(AXIS_LABELS[y_key])


# ---------------------------------------------------------------------------
# Figure 1: index planes, TP-AGB off versus on
# ---------------------------------------------------------------------------


def compute_metallicity_tracks(template, edges_gyr, log_z_grid=LOG_Z_TRACKS):
    return {
        log_z: compute_track_indices(
            GRID_DIRS[template],
            FIDUCIAL["t_q_gyr"],
            FIDUCIAL["tau_q_gyr"],
            log_z,
            edges_gyr=edges_gyr,
        )
        for log_z in log_z_grid
    }


def _draw_metallicity_tracks(axes, tracks, agb_key, summary, planes=PLANES):
    for (log_z, result), color in zip(tracks.items(), LOG_Z_TRACK_COLORS, strict=True):
        series = track_series(result, agb_key)
        delay = series["epoch_gyr"] - FIDUCIAL["t_q_gyr"]
        window = (delay >= TRACK_WINDOW_GYR[0]) & (delay <= TRACK_WINDOW_GYR[1])
        marker_indices = [
            int(np.argmin(np.abs(delay - offset))) for offset in TRACK_MARKER_OFFSETS_GYR
        ]
        summary[f"log_z_{log_z:+.2f}"] = {
            key: [float(series[key][index]) for index in marker_indices]
            for key in ("d4000", "hdelta_a", "h_minus_bump")
        }
        for axis, (x_key, y_key) in zip(axes, planes, strict=True):
            axis.plot(series[x_key][window], series[y_key][window], color=color, lw=0.9, zorder=6)
            for shape, index in zip(TRACK_MARKER_SHAPES, marker_indices, strict=True):
                axis.plot(
                    series[x_key][index],
                    series[y_key][index],
                    marker=shape,
                    ms=3.6,
                    mfc=color,
                    mec="0.25",
                    mew=0.35,
                    ls="none",
                    zorder=7,
                )


def _bump_statistics(values):
    if values.size == 0:
        return {"n_rows": 0, "bump_median_mag": None, "bump_p16_p84_mag": None}
    return {
        "n_rows": int(values.size),
        "bump_median_mag": float(np.median(values)),
        "bump_p16_p84_mag": np.percentile(values, [16, 84]).tolist(),
    }


def draw_plane_row(axes, table, codes, agb_key, ranges):
    """One row of the three planes: the full population as grey filled contours and the
    rapid-quenching class as one filled contour."""
    indices = plane_columns(table, agb_key)
    rapid = codes == RAPID_QUENCHING
    row_summary = {"n_rows": int(rapid.size), "n_rapid_quenching": int(rapid.sum())}
    tau_q_rapid = table["tau_q_gyr"][rapid]
    row_summary["rapid_quenching_tau_q_range_gyr"] = (
        [float(tau_q_rapid.min()), float(tau_q_rapid.max())] if rapid.any() else None
    )
    for axis, (x_key, y_key) in zip(axes, PLANES, strict=True):
        draw_population_contours(axis, indices[x_key], indices[y_key], ranges[x_key], ranges[y_key])
        draw_class_contours(
            axis,
            indices[x_key][rapid],
            indices[y_key][rapid],
            ranges[x_key],
            ranges[y_key],
            RAPID_QUENCHING_COLOR,
        )
        set_plane_axes(axis, x_key, y_key, ranges)
    row_summary["rapid_quenching"] = _bump_statistics(indices["h_minus_bump"][rapid])
    return row_summary


def _plane_legend_handles(with_tracks=True):
    handles = [
        Patch(facecolor="0.85", edgecolor="0.55", lw=0.5, label="all epochs (68, 95, 99.5\\%)"),
        Patch(
            facecolor=RAPID_QUENCHING_COLOR,
            edgecolor=RAPID_QUENCHING_COLOR,
            alpha=0.6,
            label="rapid-quenching class (68, 95\\%)",
        ),
    ]
    if with_tracks:
        for log_z, color in zip(LOG_Z_TRACKS, LOG_Z_TRACK_COLORS, strict=True):
            handles.append(
                Line2D([], [], color=color, lw=0.9, label=rf"$\log Z/Z_\odot = {log_z:+.2f}$")
            )
        for shape, offset in zip(TRACK_MARKER_SHAPES, TRACK_MARKER_OFFSETS_GYR, strict=True):
            label = "$t_q$" if offset == 0 else f"$t_q + {offset:g}$ Gyr"
            handles.append(
                Line2D(
                    [],
                    [],
                    marker=shape,
                    ls="none",
                    mfc="0.6",
                    mec="0.25",
                    mew=0.35,
                    ms=3.6,
                    label=label,
                )
            )
    return handles


FIDUCIAL_SFH_CAPTION = (
    rf"fiducial SFH: delayed-$\tau$ rise with $\tau = t_q = {FIDUCIAL['t_q_gyr']:g}$ Gyr, "
    rf"exponential quench with $\tau_q = {FIDUCIAL['tau_q_gyr']:g}$ Gyr, "
    rf"drawn from $t_q$ to $t_q + {TRACK_WINDOW_GYR[1]:g}$ Gyr"
)


def row_title(axis, text, fontsize=12):
    axis.text(
        0.95,
        0.93,
        text,
        transform=axis.transAxes,
        fontsize=fontsize,
        ha="right",
        va="top",
        zorder=20,
    )


def figure_1(tables, tracks_by_template, out_dir, stem="fig1_index_planes"):
    figure, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 5.4), layout="constrained")
    summary = {}
    for row, config_key in enumerate(CONFIG_ORDER):
        config = AGB_CONFIGS[config_key]
        table, codes = tables[config_key]
        indices = plane_columns(table, config["agb"])
        ranges = _plane_ranges([indices])
        summary[config_key] = draw_plane_row(axes[row], table, codes, config["agb"], ranges)
        summary[config_key]["metallicity_tracks"] = {}
        _draw_metallicity_tracks(
            axes[row],
            tracks_by_template[config["template"]],
            config["agb"],
            summary[config_key]["metallicity_tracks"],
        )
        row_title(axes[row, 0], config["short_label"])
        for col in range(3):
            panel_label(axes[row, col], f"({'abcdef'[3 * row + col]})", x=0.04, y=0.96)
    figure.legend(
        handles=_plane_legend_handles(),
        loc="outside lower center",
        ncol=4,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.4,
        title=FIDUCIAL_SFH_CAPTION,
        title_fontsize=8,
    )
    save_figure(figure, out_dir, stem)
    return summary


# ---------------------------------------------------------------------------
# Figure 2: bump offsets for four prescriptions
# ---------------------------------------------------------------------------


def _prescription_series(tables, template, agb_key, x_key):
    table = tables[template]
    return table[f"{x_key}_{agb_key}"], table[f"h_minus_bump_{BUMP_PRODUCT}_{agb_key}"]


def figure_2(tables_by_template, out_dir):
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), layout="constrained")
    summary = {"bands": {}, "slice": {}}
    for axis, x_key in zip(axes[:2], ("d4000", "hdelta_a"), strict=True):
        x_all = np.concatenate(
            [_prescription_series(tables_by_template, t, a, x_key)[0] for t, a, *_ in PRESCRIPTIONS]
        )
        edges = np.linspace(np.percentile(x_all, 0.1), np.percentile(x_all, 99.9), BAND_BINS + 1)
        summary["bands"][x_key] = {}
        for template, agb_key, label, color, linestyle in PRESCRIPTIONS:
            x, y = _prescription_series(tables_by_template, template, agb_key, x_key)
            centers, lower, median, upper = percentile_band(x, y, edges)
            axis.fill_between(centers, lower, upper, color=color, alpha=0.18, lw=0)
            axis.plot(centers, median, color=color, ls=linestyle, lw=1.3, label=label)
            summary["bands"][x_key][label] = {
                "bin_centers": centers,
                "median_mag": median,
                "p16_mag": lower,
                "p84_mag": upper,
            }
        axis.set_xlabel(AXIS_LABELS[x_key])
        axis.set_ylabel(AXIS_LABELS["h_minus_bump"])
        axis.invert_yaxis()
    x_ref = axes[0].get_xlim()[1] - 0.12 * np.diff(axes[0].get_xlim())[0]
    y_ref = np.mean(axes[0].get_ylim())
    axes[0].errorbar(
        [x_ref], [y_ref], yerr=[YARDSTICK_MAG], fmt="none", ecolor="black", capsize=3, lw=1
    )
    axes[0].text(
        x_ref, y_ref + 1.6 * YARDSTICK_MAG, "0.01 mag", va="top", ha="center", fontsize=6.5
    )
    axes[0].legend(loc="lower left", fontsize=6.5)

    axis = axes[2]
    low, high = D4000_SLICE
    values_by_label = {}
    for template, agb_key, label, *_ in PRESCRIPTIONS:
        x, y = _prescription_series(tables_by_template, template, agb_key, "d4000")
        values_by_label[label] = y[(x >= low) & (x < high)]
    all_values = np.concatenate(list(values_by_label.values()))
    bin_edges = np.linspace(all_values.min(), all_values.max(), HIST_BINS + 1)
    for (_template, _agb_key, label, color, linestyle), values in zip(
        PRESCRIPTIONS, values_by_label.values(), strict=True
    ):
        axis.hist(
            values, bins=bin_edges, histtype="step", density=True, color=color, ls=linestyle, lw=1.2
        )
        p16, median, p84 = np.percentile(values, [16, 50, 84])
        summary["slice"][label] = {
            "n_rows": int(values.size),
            "median_mag": float(median),
            "p16_mag": float(p16),
            "p84_mag": float(p84),
        }
    labels = [entry[2] for entry in PRESCRIPTIONS]
    summary["slice"]["neighbouring_separations"] = []
    for label_a, label_b in zip(labels[:-1], labels[1:], strict=True):
        delta = summary["slice"][label_b]["median_mag"] - summary["slice"][label_a]["median_mag"]
        half_width = 0.5 * max(
            summary["slice"][label]["p84_mag"] - summary["slice"][label]["p16_mag"]
            for label in (label_a, label_b)
        )
        summary["slice"]["neighbouring_separations"].append(
            {
                "pair": f"{label_a} -> {label_b}",
                "delta_median_mag": float(delta),
                "delta_over_wider_half_width": float(delta / half_width),
                "delta_over_yardstick": float(delta / YARDSTICK_MAG),
            }
        )
    axis.invert_xaxis()
    axis.set_xlabel(AXIS_LABELS["h_minus_bump"])
    axis.set_ylabel("density")
    axis.set_title(rf"D4000 in [{low:.1f}, {high:.1f})", fontsize=8)
    for axis, letter in zip(axes, "abc", strict=True):
        panel_label(axis, f"({letter})", x=0.04, y=0.96)
    save_figure(figure, out_dir, "fig2_bump_offsets")
    return summary


# ---------------------------------------------------------------------------
# Figure 3: quenching clocks
# ---------------------------------------------------------------------------


def compute_clock_tracks(edges_gyr, tau_q_values, t_q_values):
    """Tracks for every (t_q, tau_q) pair on both templates, keyed by (t_q, tau_q)."""
    tracks = {}
    for template in ("c3k", "lw02"):
        tracks[template] = {}
        for t_q in t_q_values:
            for tau_q in tau_q_values:
                tracks[template][(t_q, tau_q)] = compute_track_indices(
                    GRID_DIRS[template], t_q, tau_q, FIDUCIAL["log_z"], edges_gyr=edges_gyr
                )
    return tracks


def clock_extrema(result, agb_key, t_q, window=CLOCK_WINDOW_GYR):
    """Delay after t_q of the HdeltaA maximum and of the bump minimum (bump most negative)."""
    series = track_series(result, agb_key)
    delay = series["epoch_gyr"] - t_q
    inside = (delay >= window[0]) & (delay <= window[1])
    hdelta_index = int(np.argmax(series["hdelta_a"][inside]))
    bump_index = int(np.argmin(series["h_minus_bump"][inside]))
    delay_inside = delay[inside]
    n_inside = int(inside.sum())
    return {
        "hdelta_a_peak_delay_gyr": float(delay_inside[hdelta_index]),
        "hdelta_a_peak_at_window_edge": hdelta_index in (0, n_inside - 1),
        "bump_minimum_delay_gyr": float(delay_inside[bump_index]),
        "bump_minimum_at_window_edge": bump_index in (0, n_inside - 1),
        "bump_minimum_mag": float(series["h_minus_bump"][inside][bump_index]),
        "lag_gyr": float(delay_inside[bump_index] - delay_inside[hdelta_index]),
    }


def figure_3(clock_tracks, out_dir):
    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 5.2), layout="constrained")
    on_template, on_agb = AGB_CONFIGS["agb_on"]["template"], AGB_CONFIGS["agb_on"]["agb"]
    off_template, off_agb = AGB_CONFIGS["agb_off"]["template"], AGB_CONFIGS["agb_off"]["agb"]
    t_q_fiducial = FIDUCIAL["t_q_gyr"]
    summary = {"tau_q_family": {}, "lag_grid": {}}

    # (a) HdeltaA and (b) bump versus time since quenching, tau_q family at t_q = 3 Gyr.
    for tau_q, color in zip(CLOCK_TAU_Q_GYR, CLOCK_TAU_Q_COLORS, strict=True):
        key = (t_q_fiducial, tau_q)
        on = track_series(clock_tracks[on_template][key], on_agb)
        off = track_series(clock_tracks[off_template][key], off_agb)
        delay = on["epoch_gyr"] - t_q_fiducial
        window = (delay >= CLOCK_WINDOW_GYR[0]) & (delay <= CLOCK_WINDOW_GYR[1])
        label = rf"$\tau_q$ = {tau_q:g} Gyr"
        axes[0, 0].plot(delay[window], on["hdelta_a"][window], color=color, label=label)
        axes[0, 1].plot(delay[window], on["h_minus_bump"][window], color=color, label=label)
        axes[0, 1].plot(delay[window], off["h_minus_bump"][window], color=color, lw=0.8, ls=":")
        extrema = clock_extrema(clock_tracks[on_template][key], on_agb, t_q_fiducial)
        summary["tau_q_family"][f"tau_q_{tau_q:g}"] = {
            "agb_on": extrema,
            "agb_off": clock_extrema(clock_tracks[off_template][key], off_agb, t_q_fiducial),
        }
        axes[0, 0].plot(
            extrema["hdelta_a_peak_delay_gyr"],
            on["hdelta_a"][window][int(np.argmax(on["hdelta_a"][window]))],
            marker="*",
            ms=7,
            color=color,
            mec="black",
            mew=0.4,
            ls="none",
            zorder=5,
        )
        if not extrema["bump_minimum_at_window_edge"]:
            axes[0, 1].plot(
                extrema["bump_minimum_delay_gyr"],
                extrema["bump_minimum_mag"],
                marker="*",
                ms=7,
                color=color,
                mec="black",
                mew=0.4,
                ls="none",
                zorder=5,
            )
    for axis in axes[0]:
        axis.axvline(0.0, color="0.6", lw=0.6, ls="--", zorder=0)
        axis.set_xlabel(r"$t - t_q$ [Gyr]")
        axis.set_xlim(*CLOCK_WINDOW_GYR)
    axes[0, 0].set_ylabel(AXIS_LABELS["hdelta_a"])
    axes[0, 1].set_ylabel(AXIS_LABELS["h_minus_bump"])
    axes[0, 1].invert_yaxis()
    axes[0, 0].legend(loc="upper right", title=f"$t_q$ = {t_q_fiducial:g} Gyr", title_fontsize=7)
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color="0.3", lw=1.2, label="TP-AGB on"),
            Line2D([], [], color="0.3", lw=0.8, ls=":", label="TP-AGB off"),
            Line2D([], [], marker="*", ms=7, color="0.5", mec="black", ls="none", label="extremum"),
        ],
        loc="center right",
    )

    # (c) HdeltaA-bump plane, TP-AGB on, tau_q family with time markers.
    axis = axes[1, 0]
    marker_delays = np.arange(0.0, CLOCK_MARKER_MAX_GYR + 1e-9, CLOCK_MARKER_STEP_GYR)
    for tau_q, color in zip(CLOCK_TAU_Q_GYR, CLOCK_TAU_Q_COLORS, strict=True):
        on = track_series(clock_tracks[on_template][(t_q_fiducial, tau_q)], on_agb)
        delay = on["epoch_gyr"] - t_q_fiducial
        window = (delay >= -0.5) & (delay <= 6.0)
        axis.plot(on["hdelta_a"][window], on["h_minus_bump"][window], color=color, lw=1.1)
        for k, t_mark in enumerate(marker_delays):
            index = int(np.argmin(np.abs(delay - t_mark)))
            axis.plot(
                on["hdelta_a"][index],
                on["h_minus_bump"][index],
                marker=TRACK_MARKER_SHAPES[k % len(TRACK_MARKER_SHAPES)] if k < 5 else "o",
                ms=4.5 if k < 5 else 3.0,
                mfc=color,
                mec="black",
                mew=0.4,
                ls="none",
                zorder=5,
            )
    axis.set_xlabel(AXIS_LABELS["hdelta_a"])
    axis.set_ylabel(AXIS_LABELS["h_minus_bump"])
    axis.invert_yaxis()
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=shape,
                ls="none",
                mfc="0.6",
                mec="black",
                mew=0.4,
                ms=4.5,
                label=label,
            )
            for shape, label in zip(
                TRACK_MARKER_SHAPES,
                ("$t_q$", "+0.5 Gyr", "+1 Gyr", "+1.5 Gyr", "+2 Gyr"),
                strict=True,
            )
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="0.6",
                mec="black",
                mew=0.4,
                ms=3,
                label="+2.5, +3 Gyr",
            )
        ],
        loc="upper left",
        ncol=2,
        fontsize=6.5,
        title="colours as in (a)",
        title_fontsize=6.5,
    )

    # (d) delay of each extremum versus tau_q, three t_q values, TP-AGB on.
    axis = axes[1, 1]
    for t_q, linestyle in zip(CLOCK_T_Q_GYR, CLOCK_T_Q_STYLES, strict=True):
        hdelta_delays, bump_delays, lags, edge_flags = [], [], [], []
        for tau_q in CLOCK_TAU_Q_FINE_GYR:
            extrema = clock_extrema(clock_tracks[on_template][(t_q, tau_q)], on_agb, t_q)
            hdelta_delays.append(extrema["hdelta_a_peak_delay_gyr"])
            bump_delays.append(extrema["bump_minimum_delay_gyr"])
            lags.append(extrema["lag_gyr"])
            edge_flags.append(extrema["bump_minimum_at_window_edge"])
        summary["lag_grid"][f"t_q_{t_q:g}"] = {
            "tau_q_gyr": list(CLOCK_TAU_Q_FINE_GYR),
            "hdelta_a_peak_delay_gyr": hdelta_delays,
            "bump_minimum_delay_gyr": bump_delays,
            "lag_gyr": lags,
            "bump_minimum_at_window_edge": edge_flags,
        }
        hdelta_edges = np.array(
            [
                extrema_flag
                for extrema_flag in (
                    clock_extrema(clock_tracks[on_template][(t_q, tau_q)], on_agb, t_q)[
                        "hdelta_a_peak_at_window_edge"
                    ]
                    for tau_q in CLOCK_TAU_Q_FINE_GYR
                )
            ]
        )
        summary["lag_grid"][f"t_q_{t_q:g}"]["hdelta_a_peak_at_window_edge"] = hdelta_edges
        tau_grid = np.array(CLOCK_TAU_Q_FINE_GYR)
        axis.plot(
            tau_grid, bump_delays, color=CONFIG_COLORS["agb_on"], ls=linestyle, marker="o", ms=3
        )
        for edge_flag, marker_fill in ((False, "0.25"), (True, "white")):
            mask = hdelta_edges == edge_flag
            axis.plot(
                tau_grid[mask],
                np.array(hdelta_delays)[mask],
                color="0.25",
                ls="none",
                marker="s",
                ms=3,
                mfc=marker_fill,
            )
        interior_delays = np.where(hdelta_edges, np.nan, np.array(hdelta_delays))
        axis.plot(tau_grid, interior_delays, color="0.25", ls=linestyle, lw=0.8)
    axis.set_xscale("log")
    axis.set_xlabel(r"$\tau_q$ [Gyr]")
    axis.set_ylabel("delay of extremum after $t_q$ [Gyr]")
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=CONFIG_COLORS["agb_on"],
                marker="o",
                ms=3,
                label="H$^-$ bump minimum (TP-AGB on)",
            ),
            Line2D([], [], color="0.25", marker="s", ms=3, label=r"H$\delta_{\rm A}$ maximum"),
            Line2D(
                [],
                [],
                color="0.25",
                marker="s",
                ms=3,
                mfc="white",
                ls="none",
                label="no interior maximum (window edge)",
            ),
        ]
        + [
            Line2D([], [], color="0.5", ls=linestyle, label=f"$t_q$ = {t_q:g} Gyr")
            for t_q, linestyle in zip(CLOCK_T_Q_GYR, CLOCK_T_Q_STYLES, strict=True)
        ],
        loc="upper left",
        fontsize=6.5,
    )
    for axis, letter in zip(axes.ravel(), "abcd", strict=True):
        panel_label(axis, f"({letter})", x=0.04, y=0.96 if letter != "a" else 0.10)
    save_figure(figure, out_dir, "fig3_quenching_clocks")
    return summary


# ---------------------------------------------------------------------------
# Figures 4 and 6: information gain (classifier and recovery), at R = 100
# ---------------------------------------------------------------------------


def build_feature_sets(table, agb_key, seed, precisions=BUMP_PRECISIONS, with_log_z=False):
    """(D4000, HdeltaA) noised once, plus the R = 100 bump at each precision reusing the same
    optical draw. With `with_log_z`, log Z noised at LOG_Z_SIGMA_DEX is appended to every set
    (and an optical-plus-Z set is added) to test whether a known metallicity sharpens the bump."""
    streams = np.random.SeedSequence(seed).spawn(2 + len(precisions))
    optical = add_measurement_noise(
        np.column_stack([table[f"d4000_{agb_key}"], table[f"hdelta_a_{agb_key}"]]),
        np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        np.random.default_rng(streams[0]),
    )
    optical_scales = np.array([D4000_SIGMA, HDELTA_A_SIGMA])
    bump = table[f"h_minus_bump_{BUMP_PRODUCT}_{agb_key}"]
    noisy_bumps = {
        precision: add_measurement_noise(
            bump[:, None], np.array([precision]), np.random.default_rng(stream)
        )[:, 0]
        for stream, precision in zip(streams[1:-1], precisions, strict=True)
    }
    feature_sets = {"no_bump": (optical, optical_scales)}
    for precision, noisy_bump in noisy_bumps.items():
        feature_sets[f"bump_{precision:.3f}"] = (
            np.column_stack([optical, noisy_bump]),
            np.array([*optical_scales, precision]),
        )
    if with_log_z:
        noisy_log_z = add_measurement_noise(
            table["log_z"][:, None], np.array([LOG_Z_SIGMA_DEX]), np.random.default_rng(streams[-1])
        )[:, 0]
        feature_sets["no_bump_z"] = (
            np.column_stack([optical, noisy_log_z]),
            np.array([*optical_scales, LOG_Z_SIGMA_DEX]),
        )
        for precision, noisy_bump in noisy_bumps.items():
            feature_sets[f"bump_{precision:.3f}_z"] = (
                np.column_stack([optical, noisy_bump, noisy_log_z]),
                np.array([*optical_scales, precision, LOG_Z_SIGMA_DEX]),
            )
    return feature_sets


def _paired_gain(per_fold, baseline_per_fold):
    diff = np.asarray(per_fold, dtype=float) - np.asarray(baseline_per_fold, dtype=float)
    mean = float(np.nanmean(diff, axis=0)) if diff.ndim == 1 else np.nanmean(diff, axis=0)
    standard_error = np.nanstd(diff, axis=0, ddof=1) / np.sqrt(diff.shape[0])
    return mean, standard_error


def run_classifier(table, codes, agb_key, seeds=NOISE_SEEDS, with_log_z=False):
    """Rapid-quenching kNN classifier, grouped 5-fold, one sweep per noise seed. Per seed the
    paired completeness and purity gains over `no_bump` (same folds) with their fold standard
    errors; across seeds the mean and ddof-1 std of every quantity."""
    per_seed = {}
    for seed in seeds:
        feature_sets = build_feature_sets(table, agb_key, seed, with_log_z=with_log_z)
        results = {}
        for key, (features, scales) in feature_sets.items():
            results[key] = cross_validated_metrics(
                features,
                codes,
                table["history_id"],
                scales,
                k=K_NEIGHBORS,
                n_folds=N_FOLDS,
                seed=seed,
                positive_class=RAPID_QUENCHING,
            )
        baseline_key = {key: "no_bump_z" if key.endswith("_z") else "no_bump" for key in results}
        for key, metrics in results.items():
            for metric in ("completeness", "purity"):
                baseline = results[baseline_key[key]]
                gain, standard_error = _paired_gain(
                    metrics[f"{metric}_per_fold"], baseline[f"{metric}_per_fold"]
                )
                metrics[f"{metric}_gain"] = gain
                metrics[f"{metric}_gain_standard_error"] = float(standard_error)
        per_seed[str(seed)] = results
    return _across_seeds(per_seed)


def run_recovery(post_quench_table, agb_key, seeds=NOISE_SEEDS, with_log_z=False):
    """kNN recovery of log10(t - t_q) and log10(tau_q) on post-quench epochs; RMS per fold,
    paired gain over `no_bump` per seed, mean and std across seeds."""
    targets = regression_targets(post_quench_table)
    per_seed = {}
    for seed in seeds:
        feature_sets = build_feature_sets(post_quench_table, agb_key, seed, with_log_z=with_log_z)
        rms_by_key = {
            key: cross_validated_regression(
                features,
                targets,
                post_quench_table["history_id"],
                scales,
                k=K_NEIGHBORS,
                n_folds=N_FOLDS,
                seed=seed,
            )
            for key, (features, scales) in feature_sets.items()
        }
        results = {}
        for key, rms_per_fold in rms_by_key.items():
            baseline_key = "no_bump_z" if key.endswith("_z") else "no_bump"
            gain, standard_error = _paired_gain(rms_per_fold, rms_by_key[baseline_key])
            results[key] = {
                "rms_mean_dex": rms_per_fold.mean(axis=0),
                "rms_fold_standard_error_dex": rms_per_fold.std(axis=0, ddof=1) / np.sqrt(N_FOLDS),
                "rms_gain_dex": np.asarray(gain),
                "rms_gain_standard_error_dex": np.asarray(standard_error),
            }
        per_seed[str(seed)] = results
    return _across_seeds(per_seed)


def _across_seeds(per_seed):
    seeds = list(per_seed)
    keys = list(per_seed[seeds[0]])
    out = {"per_seed": per_seed, "across_seeds": {}}
    for key in keys:
        entry = {}
        for metric, first_value in per_seed[seeds[0]][key].items():
            if isinstance(first_value, list) or (
                isinstance(first_value, np.ndarray) and first_value.ndim > 1
            ):
                continue
            values = np.array([per_seed[seed][key][metric] for seed in seeds], dtype=float)
            entry[f"{metric}_mean"] = values.mean(axis=0)
            entry[f"{metric}_std"] = (
                values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(values[0])
            )
        out["across_seeds"][key] = entry
    return out


def _significance_text(gain, standard_error):
    if standard_error <= 0 or not np.isfinite(standard_error):
        return ""
    return rf"{abs(gain) / standard_error:.0f}$\sigma$"


def figure_4(gains, out_dir):
    """Gains for the exponential family: classifier (a, b) and recovery (c, d) versus bump
    precision, TP-AGB on and off, optical-only baseline as horizontal bands."""
    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.6), layout="constrained")
    precisions = np.array(BUMP_PRECISIONS)
    x_shift = {"agb_off": 0.985, "agb_on": 1.015}
    for config_key in CONFIG_ORDER:
        color = CONFIG_COLORS[config_key]
        label = AGB_CONFIGS[config_key]["label"]
        classifier = gains[config_key]["classifier"]["across_seeds"]
        for axis, metric in zip(axes[:2], ("completeness", "purity"), strict=True):
            baseline = classifier["no_bump"][f"{metric}_mean_mean"]
            baseline_spread = classifier["no_bump"][f"{metric}_std_mean"] / np.sqrt(N_FOLDS)
            axis.axhline(baseline, color=color, ls="--", lw=0.9, alpha=0.8)
            axis.axhspan(
                baseline - baseline_spread,
                baseline + baseline_spread,
                color=color,
                alpha=0.08,
                lw=0,
            )
            means = [classifier[f"bump_{p:.3f}"][f"{metric}_mean_mean"] for p in precisions]
            errors = [
                classifier[f"bump_{p:.3f}"][f"{metric}_std_mean"] / np.sqrt(N_FOLDS)
                for p in precisions
            ]
            axis.errorbar(
                precisions * x_shift[config_key],
                means,
                yerr=errors,
                marker="o",
                ms=3.5,
                capsize=2,
                color=color,
                label=label,
            )
            for p, mean in zip(precisions, means, strict=True):
                entry = classifier[f"bump_{p:.3f}"]
                text = _significance_text(
                    entry[f"{metric}_gain_mean"], entry[f"{metric}_gain_standard_error_mean"]
                )
                axis.annotate(
                    text,
                    (p * x_shift[config_key], mean),
                    xytext=(5 if config_key == "agb_on" else -5, 0),
                    textcoords="offset points",
                    ha="left" if config_key == "agb_on" else "right",
                    va="center",
                    fontsize=5.5,
                    color=color,
                )
            axis.set_ylabel(f"rapid-quenching {metric}")
        recovery = gains[config_key]["recovery"]["across_seeds"]
        for col, (axis, target_label) in enumerate(zip(axes[2:], TARGET_LABELS, strict=True)):
            baseline = recovery["no_bump"]["rms_mean_dex_mean"][col]
            baseline_spread = recovery["no_bump"]["rms_fold_standard_error_dex_mean"][col]
            axis.axhline(baseline, color=color, ls="--", lw=0.9, alpha=0.8)
            axis.axhspan(
                baseline - baseline_spread,
                baseline + baseline_spread,
                color=color,
                alpha=0.08,
                lw=0,
            )
            means = [recovery[f"bump_{p:.3f}"]["rms_mean_dex_mean"][col] for p in precisions]
            errors = [
                recovery[f"bump_{p:.3f}"]["rms_fold_standard_error_dex_mean"][col]
                for p in precisions
            ]
            axis.errorbar(
                precisions * x_shift[config_key],
                means,
                yerr=errors,
                marker="o",
                ms=3.5,
                capsize=2,
                color=color,
                label=label,
            )
            for p, mean in zip(precisions, means, strict=True):
                entry = recovery[f"bump_{p:.3f}"]
                text = _significance_text(
                    entry["rms_gain_dex_mean"][col], entry["rms_gain_standard_error_dex_mean"][col]
                )
                axis.annotate(
                    text,
                    (p * x_shift[config_key], mean),
                    xytext=(5 if config_key == "agb_on" else -5, 0),
                    textcoords="offset points",
                    ha="left" if config_key == "agb_on" else "right",
                    va="center",
                    fontsize=5.5,
                    color=color,
                )
            axis.set_ylabel(f"RMS error of {target_label} [dex]")
    for axis, letter in zip(axes, "abcd", strict=True):
        axis.set_xscale("log")
        axis.set_xticks(precisions)
        axis.set_xticklabels([f"{p:g}" for p in precisions])
        axis.set_xlim(0.0033, 0.03)
        axis.set_xlabel("bump precision [mag]")
        axis.minorticks_off()
        if letter in "cd":
            panel_label(axis, f"({letter})", x=0.95, y=0.08, ha="right")
        else:
            panel_label(axis, f"({letter})", x=0.05, y=0.96)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=CONFIG_COLORS[key],
                marker="o",
                ms=3.5,
                label=AGB_CONFIGS[key]["label"],
            )
            for key in CONFIG_ORDER
        ]
        + [Line2D([], [], color="0.4", ls="--", label="optical only (band: fold standard error)")],
        loc="outside lower center",
        ncol=3,
        fontsize=6.5,
    )
    save_figure(figure, out_dir, "fig4_information_gain")


def figure_6(gains_by_family, out_dir):
    """Per SFH family: the bump's paired gain in completeness and purity (0.010 mag) and in
    the recovery RMS of both targets (0.005 mag), TP-AGB on versus off. Error bars are the
    fold standard error of the paired gain, averaged over the noise seeds."""
    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.8), layout="constrained")
    width = 0.36
    x = np.arange(len(FAMILIES))
    classifier_key = f"bump_{CLASSIFIER_PRECISION:.3f}"
    recovery_key = f"bump_{RECOVERY_PRECISION:.3f}"
    for offset, config_key in zip((-width / 2, width / 2), CONFIG_ORDER, strict=True):
        color = CONFIG_COLORS[config_key]
        for axis, metric in zip(axes[:2], ("completeness", "purity"), strict=True):
            values = [
                gains_by_family[family][config_key]["classifier"]["across_seeds"][classifier_key][
                    f"{metric}_gain_mean"
                ]
                for family in FAMILIES
            ]
            errors = [
                gains_by_family[family][config_key]["classifier"]["across_seeds"][classifier_key][
                    f"{metric}_gain_standard_error_mean"
                ]
                for family in FAMILIES
            ]
            axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=color,
                alpha=0.85,
                label=AGB_CONFIGS[config_key]["label"],
            )
            axis.set_ylabel(rf"$\Delta$ {metric}")
        for col, axis in enumerate(axes[2:]):
            values, errors = [], []
            for family in FAMILIES:
                entry = gains_by_family[family][config_key]["recovery"]["across_seeds"][
                    recovery_key
                ]
                applicable = not (family == "truncation" and col == 1)
                values.append(-entry["rms_gain_dex_mean"][col] if applicable else np.nan)
                errors.append(entry["rms_gain_standard_error_dex_mean"][col] if applicable else 0.0)
            axis.bar(x + offset, values, width, yerr=errors, capsize=2, color=color, alpha=0.85)
            axis.set_ylabel(rf"RMS gain, {TARGET_LABELS[col]} [dex]")
    for axis, letter in zip(axes, "abcd", strict=True):
        axis.axhline(0.0, color="0.3", lw=0.6)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_SHORT_LABELS[family] for family in FAMILIES],
            fontsize=7,
            rotation=30,
            ha="right",
        )
        axis.minorticks_off()
        panel_label(axis, f"({letter})", x=0.05, y=0.96)
    axes[3].text(x[2], 0.0, "n/a", ha="center", va="bottom", fontsize=6, color="0.4")
    axes[0].set_title(f"bump at {CLASSIFIER_PRECISION:g} mag", fontsize=8)
    axes[1].set_title(f"bump at {CLASSIFIER_PRECISION:g} mag", fontsize=8)
    axes[2].set_title(f"bump at {RECOVERY_PRECISION:g} mag", fontsize=8)
    axes[3].set_title(f"bump at {RECOVERY_PRECISION:g} mag", fontsize=8)
    figure.legend(
        handles=[
            Patch(facecolor=CONFIG_COLORS[key], alpha=0.85, label=AGB_CONFIGS[key]["label"])
            for key in CONFIG_ORDER
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=6.5,
    )
    save_figure(figure, out_dir, "fig6_robustness_gains")


# ---------------------------------------------------------------------------
# Figure 5: planes for the other SFH families, TP-AGB on
# ---------------------------------------------------------------------------


def figure_5(tables_by_family, out_dir, stem="fig5_robustness_planes"):
    figure, axes = plt.subplots(
        len(ROBUSTNESS_FAMILIES),
        3,
        figsize=(DOUBLE_COLUMN_IN, 2.55 * len(ROBUSTNESS_FAMILIES)),
        layout="constrained",
    )
    config = AGB_CONFIGS["agb_on"]
    summary = {}
    all_indices = [plane_columns(table, config["agb"]) for table, _ in tables_by_family.values()]
    ranges = _plane_ranges(all_indices)
    for row, family in enumerate(ROBUSTNESS_FAMILIES):
        table, codes = tables_by_family[family]
        summary[family] = draw_plane_row(axes[row], table, codes, config["agb"], ranges)
        row_title(axes[row, 0], f"{config['short_label']}\n{FAMILY_LABELS[family]}", fontsize=11)
        for col in range(3):
            panel_label(axes[row, col], f"({'abcdefghi'[3 * row + col]})", x=0.04, y=0.96)
    figure.legend(
        handles=_plane_legend_handles(with_tracks=False),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    save_figure(figure, out_dir, stem)
    return summary


# ---------------------------------------------------------------------------
# Figure 7: metallicity decomposition and the value of a known metallicity
# ---------------------------------------------------------------------------


def _bump_at_offset(result, agb_key, offset_gyr, subtract_key=None):
    series = track_series(result, agb_key)
    delay = series["epoch_gyr"] - FIDUCIAL["t_q_gyr"]
    index = int(np.argmin(np.abs(delay - offset_gyr)))
    value = series["h_minus_bump"][index]
    if subtract_key is not None:
        value -= track_series(result, subtract_key)["h_minus_bump"][index]
    return float(value)


def _relative_gain_entries(gains_with_z):
    """The bump's paired gain as a fraction of the optical-only (or optical-plus-Z) baseline,
    for completeness, purity and both recovery RMS values, with and without a known Z."""
    classifier = gains_with_z["classifier"]["across_seeds"]
    recovery = gains_with_z["recovery"]["across_seeds"]
    entries = []
    for metric in ("completeness", "purity"):
        for suffix, key, baseline_key in (
            ("Z unknown", f"bump_{CLASSIFIER_PRECISION:.3f}", "no_bump"),
            ("Z known", f"bump_{CLASSIFIER_PRECISION:.3f}_z", "no_bump_z"),
        ):
            baseline = classifier[baseline_key][f"{metric}_mean_mean"]
            gain = classifier[key][f"{metric}_gain_mean"] / baseline
            error = classifier[key][f"{metric}_gain_standard_error_mean"] / baseline
            short_metric = {"completeness": "compl.", "purity": "purity"}[metric]
            entries.append((f"{short_metric}, {suffix}", gain, error, suffix, baseline))
    for col, target_label in enumerate(TARGET_LABELS):
        for suffix, key, baseline_key in (
            ("Z unknown", f"bump_{RECOVERY_PRECISION:.3f}", "no_bump"),
            ("Z known", f"bump_{RECOVERY_PRECISION:.3f}_z", "no_bump_z"),
        ):
            baseline = recovery[baseline_key]["rms_mean_dex_mean"][col]
            gain = -recovery[key]["rms_gain_dex_mean"][col] / baseline
            error = recovery[key]["rms_gain_standard_error_dex_mean"][col] / baseline
            entries.append((f"{target_label} RMS, {suffix}", gain, error, suffix, baseline))
    return entries


def figure_7(tracks_by_template, tables, gains_with_z, out_dir):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_IN, 2.7),
        layout="constrained",
        width_ratios=(1.0, 1.0, 1.15),
    )
    summary = {"tracks": {}, "population_slice": {}, "known_z": {}}
    log_z = np.array(LOG_Z_TRACKS)

    # (a) bump versus log Z at two post-quench delays: the non-AGB stars alone (C3K, agb = 0),
    # the full TP-AGB-on model (LW02, agb = 2) and the TP-AGB increment of each template.
    axis = axes[0]
    components = {
        "non_agb_stars": ("c3k", "agb0", None, "#0072B2", "non-AGB stars (C3K, agb 0)"),
        "agb_on_total": ("lw02", "agb2", None, "#D55E00", "TP-AGB on (LW02, agb 2)"),
        "lw02_increment": ("lw02", "agb2", "agb0", "#E69F00", "TP-AGB part, LW02"),
        "c3k_increment": ("c3k", "agb2", "agb0", "#56B4E9", "TP-AGB part, C3K"),
    }
    offsets = {1.0: "-"}
    for name, (template, agb_key, subtract_key, color, label) in components.items():
        summary["tracks"][name] = {}
        for offset in METALLICITY_OFFSETS_GYR:
            values = [
                _bump_at_offset(tracks_by_template[template][z], agb_key, offset, subtract_key)
                for z in LOG_Z_TRACKS
            ]
            summary["tracks"][name][f"t_q+{offset:g}_gyr"] = {
                "log_z": log_z.tolist(),
                "bump_mag": values,
                "slope_mag_per_dex": float(np.polyfit(log_z, values, 1)[0]),
            }
            if offset in offsets:
                axis.plot(
                    log_z,
                    values,
                    color=color,
                    ls=offsets[offset],
                    marker="o",
                    ms=2.5,
                    label=label if offset == 1.0 else None,
                )
    axis.axhline(0.0, color="0.6", lw=0.5)
    axis.set_xlabel(r"$\log Z/Z_\odot$")
    axis.set_ylabel(AXIS_LABELS["h_minus_bump"])
    axis.invert_yaxis()
    axis.legend(
        loc="upper left",
        fontsize=5.5,
        title="$t_q + 1$ Gyr",
        title_fontsize=6,
        handlelength=1.4,
        borderaxespad=0.3,
    )

    # (b) population: bump versus log Z inside one D4000 slice, TP-AGB on, rapid quenching vs all.
    axis = axes[1]
    table, codes = tables["agb_on"]
    agb_key = AGB_CONFIGS["agb_on"]["agb"]
    low, high = METALLICITY_D4000_SLICE
    in_slice = (table[f"d4000_{agb_key}"] >= low) & (table[f"d4000_{agb_key}"] < high)
    bump = table[f"h_minus_bump_{BUMP_PRODUCT}_{agb_key}"]
    rapid = codes == RAPID_QUENCHING
    z_range = (-0.55, 0.25)
    bump_values = bump[in_slice]
    b_low, b_high = np.percentile(bump_values, [0.1, 99.9])
    b_range = (b_low - 0.05 * (b_high - b_low), b_high + 0.05 * (b_high - b_low))
    draw_population_contours(axis, table["log_z"][in_slice], bump_values, z_range, b_range)
    if (in_slice & rapid).sum() > 50:
        draw_class_contours(
            axis,
            table["log_z"][in_slice & rapid],
            bump[in_slice & rapid],
            z_range,
            b_range,
            RAPID_QUENCHING_COLOR,
        )
    axis.set_xlim(*z_range)
    axis.set_ylim(b_range[1], b_range[0])
    axis.set_xlabel(r"$\log Z/Z_\odot$")
    axis.set_ylabel(AXIS_LABELS["h_minus_bump"])
    axis.set_title(rf"TP-AGB on, D4000 $\in$ [{low:.1f}, {high:.1f})", fontsize=8)
    for name, mask in (
        ("all", in_slice),
        ("rapid_quenching", in_slice & rapid),
        ("not_rapid_quenching", in_slice & ~rapid),
    ):
        if mask.sum() < 10:
            continue
        z_values, b_values = table["log_z"][mask], bump[mask]
        slope, intercept = np.polyfit(z_values, b_values, 1)
        residual = b_values - (slope * z_values + intercept)
        summary["population_slice"][name] = {
            "n_rows": int(mask.sum()),
            "bump_16_84_half_width_mag": float(0.5 * np.diff(np.percentile(b_values, [16, 84]))[0]),
            "bump_slope_mag_per_dex": float(slope),
            "residual_16_84_half_width_mag": float(
                0.5 * np.diff(np.percentile(residual, [16, 84]))[0]
            ),
        }
    axis.legend(
        handles=[
            Patch(facecolor="0.85", edgecolor="0.55", lw=0.5, label="all epochs in slice"),
            Patch(facecolor=RAPID_QUENCHING_COLOR, alpha=0.6, label="rapid quenching"),
        ],
        loc="lower left",
        fontsize=6,
    )

    # (c) does a known metallicity change the bump's gains? Exponential family, TP-AGB on.
    axis = axes[2]
    entries = _relative_gain_entries(gains_with_z)
    positions = np.arange(len(entries))
    for position, (label, gain, error, suffix, baseline) in zip(positions, entries, strict=True):
        color = CONFIG_COLORS["agb_on"] if suffix == "Z known" else "0.5"
        axis.barh(position, 100.0 * gain, xerr=100.0 * error, color=color, capsize=2, height=0.7)
        summary["known_z"][label] = {
            "relative_gain": float(gain),
            "relative_gain_standard_error": float(error),
            "baseline": float(baseline),
        }
    axis.set_yticks(positions)
    axis.set_yticklabels([entry[0] for entry in entries], fontsize=5.5)
    axis.invert_yaxis()
    axis.axvline(0.0, color="0.3", lw=0.6)
    axis.set_xlabel(r"bump gain [\% of baseline]")
    axis.set_title("TP-AGB on", fontsize=8)
    axis.set_xlim(right=1.3 * max(100.0 * entry[1] for entry in entries))
    axis.minorticks_off()
    figure.legend(
        handles=[
            Patch(facecolor="0.5", label="(c) grey: baseline is optical only"),
            Patch(
                facecolor=CONFIG_COLORS["agb_on"],
                label=rf"(c) orange: baseline is optical + $\log Z$ ({LOG_Z_SIGMA_DEX:g} dex)",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=6.5,
    )
    for axis, letter in zip(axes, "abc", strict=True):
        panel_label(axis, f"({letter})", x=0.04, y=0.96)
    save_figure(figure, out_dir, "fig7_metallicity")
    return summary


# ---------------------------------------------------------------------------
# Combined figure: age sensitivity after quenching and the classifier gain
# ---------------------------------------------------------------------------


def scaled_post_quench_tracks(result, agb_key, t_q, window=CLOCK_LOG_WINDOW_GYR):
    """Each index over `window` after t_q, scaled to [0, 1] over that window; the bump enters
    as its strength (minus the index) so that every curve rises when the feature strengthens."""
    series = track_series(result, agb_key)
    delay = series["epoch_gyr"] - t_q
    inside = (delay >= window[0] - 1e-9) & (delay <= window[1] + 1e-9)
    out = {"log_delay": np.log10(delay[inside])}
    for key in ("hdelta_a", "h_minus_bump", "d4000"):
        values = series[key][inside]
        if key == "h_minus_bump":
            values = -values
        low, high = values.min(), values.max()
        out[key] = (values - low) / (high - low)
        out[f"{key}_range"] = [float(low), float(high)]
    return out


def figure_combined(clock_tracks, gains, out_dir, stem=FINAL_STEMS["combined"]):
    """Left: one panel per tau_q with the three indices against log10(t - t_q), TP-AGB on,
    each index scaled to its own post-quench range, with the HdeltaA and H-minus bump peaks
    marked. Right: rapid-quenching completeness and purity against H-minus bump precision,
    TP-AGB on, with the optical-only baseline."""
    figure = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.2), layout="constrained")
    grid = figure.add_gridspec(len(COMBINED_TAU_Q_GYR), 4, width_ratios=(1, 1, 1, 1))
    axes_clock = [figure.add_subplot(grid[0, :2])]
    axes_clock += [
        figure.add_subplot(grid[row, :2], sharex=axes_clock[0])
        for row in range(1, len(COMBINED_TAU_Q_GYR))
    ]
    axes_gain = [figure.add_subplot(grid[:, 2]), figure.add_subplot(grid[:, 3])]
    on_template, on_agb = AGB_CONFIGS["agb_on"]["template"], AGB_CONFIGS["agb_on"]["agb"]
    t_q = FIDUCIAL["t_q_gyr"]
    summary = {"scaled_tracks": {}, "classifier": {}}

    for axis, tau_q in zip(axes_clock, COMBINED_TAU_Q_GYR, strict=True):
        scaled = scaled_post_quench_tracks(clock_tracks[on_template][(t_q, tau_q)], on_agb, t_q)
        entry = {key: {"range": scaled[f"{key}_range"]} for key in INDEX_COLORS}
        for key, color in INDEX_COLORS.items():
            axis.plot(scaled["log_delay"], scaled[key], color=color, lw=1.3)
            if key == "d4000":
                continue
            peak = int(np.argmax(scaled[key]))
            at_edge = peak in (0, scaled[key].size - 1)
            entry[key]["peak_delay_gyr"] = float(10.0 ** scaled["log_delay"][peak])
            entry[key]["peak_at_window_edge"] = at_edge
            axis.plot(
                scaled["log_delay"][peak],
                scaled[key][peak],
                marker="*",
                ms=9,
                mfc="white" if at_edge else color,
                mec=color,
                mew=0.9,
                ls="none",
                zorder=6,
            )
        summary["scaled_tracks"][f"tau_q_{tau_q:g}"] = entry
        axis.set_ylim(-0.06, 1.12)
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.text(
            0.05,
            0.60,
            rf"$\tau_q = {tau_q:g}$ Gyr",
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            zorder=8,
            bbox={"facecolor": "white", "edgecolor": "0.6", "lw": 0.5, "alpha": 0.9, "pad": 2.5},
        )
    axes_clock[-1].set_xlabel(r"$\log_{10}(t - t_q)$ [Gyr]")
    axes_clock[1].set_ylabel("index scaled to its post-quench range")
    axes_clock[0].set_xlim(
        np.log10(CLOCK_LOG_WINDOW_GYR[0]) - 0.03, np.log10(CLOCK_LOG_WINDOW_GYR[1]) + 0.03
    )
    for axis in axes_clock[:-1]:
        axis.tick_params(labelbottom=False)

    classifier = gains["agb_on"]["classifier"]["across_seeds"]
    precisions = np.array(BUMP_PRECISIONS)
    point_color = "0.15"
    for axis, metric in zip(axes_gain, ("completeness", "purity"), strict=True):
        baseline = classifier["no_bump"][f"{metric}_mean_mean"]
        baseline_error = classifier["no_bump"][f"{metric}_std_mean"] / np.sqrt(N_FOLDS)
        axis.axhline(baseline, color="0.35", ls="--", lw=0.9)
        axis.axhspan(
            baseline - baseline_error, baseline + baseline_error, color="0.5", alpha=0.15, lw=0
        )
        means = [classifier[f"bump_{p:.3f}"][f"{metric}_mean_mean"] for p in precisions]
        errors = [
            classifier[f"bump_{p:.3f}"][f"{metric}_std_mean"] / np.sqrt(N_FOLDS) for p in precisions
        ]
        axis.errorbar(
            precisions, means, yerr=errors, marker="o", ms=4, capsize=2.5, color=point_color, lw=1.2
        )
        summary["classifier"][metric] = {
            "optical_only": float(baseline),
            "optical_only_fold_standard_error": float(baseline_error),
            "bump_precision_mag": precisions.tolist(),
            "with_bump": [float(value) for value in means],
            "with_bump_fold_standard_error": [float(value) for value in errors],
        }
        axis.set_xscale("log")
        axis.set_xticks(precisions)
        axis.set_xticklabels([f"{p:g}" for p in precisions])
        axis.set_xlim(0.0036, 0.028)
        axis.minorticks_off()
        axis.set_xlabel(r"H$^-$ bump precision [mag]")
        axis.set_ylabel(f"rapid-quenching {metric}")
    handles = [
        Line2D([], [], color=color, lw=1.3, label=INDEX_SHORT_LABELS[key])
        for key, color in INDEX_COLORS.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="*",
            ms=9,
            mfc="0.5",
            mec="0.3",
            mew=0.9,
            ls="none",
            label="peak (open: at the window edge)",
        ),
        Line2D(
            [],
            [],
            color=point_color,
            marker="o",
            ms=4,
            lw=1.2,
            label=r"(d, e) D4000, H$\delta_{\rm A}$ and H$^-$ bump",
        ),
        Line2D(
            [],
            [],
            color="0.35",
            ls="--",
            lw=0.9,
            label=r"(d, e) D4000 and H$\delta_{\rm A}$ only (band: fold standard error)",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=7.5,
        title=(
            rf"TP-AGB on; (a--c) tracks at $t_q = {t_q:g}$ Gyr, solar metallicity, "
            r"delayed-$\tau$ rise with $\tau = t_q$"
        ),
        title_fontsize=7.5,
        columnspacing=1.6,
    )
    for axis, letter in zip(axes_clock, "abc", strict=True):
        panel_label(axis, f"({letter})", x=0.07, y=0.95)
    panel_label(axes_gain[0], "(d)", x=0.05, y=0.97)
    panel_label(axes_gain[1], "(e)", x=0.05, y=0.97)
    save_figure(figure, out_dir, stem)
    return summary


# ---------------------------------------------------------------------------
# D4000 versus H-minus bump plane with the JWST quiescent galaxies
# ---------------------------------------------------------------------------


def load_jwst_indices(path=JWST_DATA_PATH):
    """The z ~ 1 JWST quiescent-galaxy sample: D4000 with its 16th and 84th percentile
    bounds, and the H-minus bump with a symmetric error."""
    with np.load(path) as data:
        d4000 = data["D4000"]
        bounds = data["D4000_err"]
        return {
            "id": data["ID"],
            "redshift": data["z"],
            "d4000": d4000,
            "d4000_err_low": d4000 - bounds[:, 0],
            "d4000_err_high": bounds[:, 1] - d4000,
            "h_minus_bump": data["Hbump"],
            "h_minus_bump_err": data["Hbump_err"],
        }


def jwst_inputs_at_r50(tables, tracks_by_template, edges_gyr):
    """The population tables and metallicity tracks with their D4000 replaced by the
    R = 50 measurement (`d4000_r50.py`), for the JWST comparison figure only."""
    from d4000_r50 import (
        population_d4000_r50,
        substitute_population_d4000,
        substitute_track_d4000,
    )

    new_tables = {}
    for config_key, (table, codes) in tables.items():
        template = AGB_CONFIGS[config_key]["template"]
        r50 = population_d4000_r50(GRID_DIRS[template], population_dir("exponential", template))
        new_tables[config_key] = (substitute_population_d4000(table, r50), codes)
    new_tracks = {}
    for template, tracks in tracks_by_template.items():
        new_tracks[template] = {
            log_z: substitute_track_d4000(
                result,
                GRID_DIRS[template],
                log_z,
                FIDUCIAL["t_q_gyr"],
                FIDUCIAL["tau_q_gyr"],
                edges_gyr,
            )
            for log_z, result in tracks.items()
        }
    return new_tables, new_tracks


def figure_jwst_plane(tables, tracks_by_template, jwst, out_dir, stem=FINAL_STEMS["jwst"]):
    """The D4000 versus H-minus bump plane for TP-AGB off (left) and TP-AGB on (right), same
    layers as Figure 1, on one shared bump axis, with the JWST galaxies overplotted."""
    figure, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_IN, 3.4), layout="constrained", sharey=True
    )
    plane = ("d4000", "h_minus_bump")
    indices_by_config = {
        key: plane_columns(tables[key][0], AGB_CONFIGS[key]["agb"]) for key in CONFIG_ORDER
    }
    ranges = _plane_ranges(list(indices_by_config.values()))
    low = min(ranges["h_minus_bump"][0] - 0.006, float(jwst["h_minus_bump"].min()) - 0.006)
    high = max(ranges["h_minus_bump"][1], float(jwst["h_minus_bump"].max()) + 0.006)
    ranges["h_minus_bump"] = (low, high)
    summary = {
        "n_jwst": int(jwst["d4000"].size),
        "redshift_range": [float(jwst["redshift"].min()), float(jwst["redshift"].max())],
    }
    for axis, config_key in zip(axes, CONFIG_ORDER, strict=True):
        config = AGB_CONFIGS[config_key]
        table, codes = tables[config_key]
        indices = indices_by_config[config_key]
        rapid = codes == RAPID_QUENCHING
        draw_population_contours(
            axis, indices["d4000"], indices["h_minus_bump"], ranges["d4000"], ranges["h_minus_bump"]
        )
        draw_class_contours(
            axis,
            indices["d4000"][rapid],
            indices["h_minus_bump"][rapid],
            ranges["d4000"],
            ranges["h_minus_bump"],
            RAPID_QUENCHING_COLOR,
        )
        summary[config_key] = {"metallicity_tracks": {}}
        _draw_metallicity_tracks(
            [axis],
            tracks_by_template[config["template"]],
            config["agb"],
            summary[config_key]["metallicity_tracks"],
            planes=(plane,),
        )
        axis.errorbar(
            jwst["d4000"],
            jwst["h_minus_bump"],
            xerr=[jwst["d4000_err_low"], jwst["d4000_err_high"]],
            yerr=jwst["h_minus_bump_err"],
            fmt="o",
            ms=3.8,
            mfc="black",
            mec="white",
            mew=0.5,
            ecolor="0.2",
            elinewidth=0.6,
            capsize=0,
            zorder=12,
        )
        set_plane_axes(axis, "d4000", "h_minus_bump", ranges, ylabel=config_key == "agb_off")
        axis.set_xlabel(r"D4000 ($R = 50$)")
        row_title(axis, config["short_label"], fontsize=12)
    panel_label(axes[0], "(a)", x=0.04, y=0.96)
    panel_label(axes[1], "(b)", x=0.04, y=0.96)
    z_low, z_high = summary["redshift_range"]
    handles = _plane_legend_handles() + [
        Line2D(
            [],
            [],
            marker="o",
            ms=3.8,
            mfc="black",
            mec="white",
            mew=0.5,
            color="0.2",
            lw=0.6,
            label="Lu+2026",
        )
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.4,
        title=FIDUCIAL_SFH_CAPTION,
        title_fontsize=8,
    )
    save_figure(figure, out_dir, stem)
    return summary


def cosmic_age_gyr(redshift, hubble_constant=JWST_HUBBLE_CONSTANT, omega_matter=JWST_OMEGA_MATTER):
    """Age of a flat LCDM universe at `redshift`, in Gyr."""
    from scipy.integrate import quad

    def integrand(x):
        return 1.0 / ((1.0 + x) * np.sqrt(omega_matter * (1.0 + x) ** 3 + 1.0 - omega_matter))

    return (977.8 / hubble_constant) * quad(integrand, redshift, np.inf)[0]


def _draw_open_contours(axis, x, y, x_range, y_range, fractions, color, linewidth, zorder):
    x_centers, y_centers, density, levels = density_levels(x, y, fractions, x_range, y_range)
    axis.contour(
        x_centers,
        y_centers,
        density,
        levels=levels,
        colors=[color],
        linewidths=linewidth,
        linestyles="--",
        zorder=zorder,
    )


def _fraction_inside(x_points, y_points, x, y, x_range, y_range, fraction):
    """Which points lie inside the smoothed-density contour enclosing `fraction` of (x, y)."""
    from scipy.ndimage import map_coordinates

    x_centers, y_centers, density, levels = density_levels(x, y, (fraction,), x_range, y_range)
    step_x = x_centers[1] - x_centers[0]
    step_y = y_centers[1] - y_centers[0]
    coordinates = [(y_points - y_centers[0]) / step_y, (x_points - x_centers[0]) / step_x]
    return map_coordinates(density, coordinates, order=1, mode="nearest") >= levels[0]


def add_agb1_tracks(tracks_by_template, edges_gyr):
    """Add agb = 1 entries (D4000 at R = 50, HdeltaA at sigma300, bump at R = 100) to the
    fiducial metallicity tracks, from the cached SSP grids."""
    from csp_integrate import csp_track
    from spectral_indices import d4000, h_minus_bump, hdelta_a
    from ssp_grid import load_ssp_grid

    out = {}
    for template, tracks in tracks_by_template.items():
        grids = {
            name: load_ssp_grid(GRID_DIRS[template], name) for name in ("r50", "sigma300", "r100")
        }
        out[template] = {}
        for log_z, result in tracks.items():
            flux = {
                name: csp_track(
                    grid, log_z, 1, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], edges_gyr
                )[1]
                for name, grid in grids.items()
            }
            entry = dict(result)
            entry["agb1"] = {
                "sigma300": {
                    "d4000": d4000(grids["r50"].wave_a, flux["r50"]),
                    "hdelta_a": hdelta_a(grids["sigma300"].wave_a, flux["sigma300"]),
                },
                "r100": {"h_minus_bump": h_minus_bump(grids["r100"].wave_a, flux["r100"])},
            }
            out[template][log_z] = entry
    return out


def figure_jwst_plane_epoch_matched(
    tables,
    tracks_by_template,
    jwst,
    out_dir,
    stem=FINAL_STEMS["jwst_epoch"],
    agb_on_key=None,
):
    """Variant of `figure_jwst_plane` restricted to model epochs whose cosmic age matches the
    redshift range of the JWST sample. The all-epoch population and rapid-quenching class
    are drawn as dashed open contours; the epoch-matched ones are filled. No new models:
    the same tables (with D4000 at R = 50) are masked on `epoch_gyr`."""
    age_low = cosmic_age_gyr(float(jwst["redshift"].max()))
    age_high = cosmic_age_gyr(float(jwst["redshift"].min()))
    half_step = 0.5 * (time_bin_edges()[1] - time_bin_edges()[0])
    figure, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_IN, 3.4), layout="constrained", sharey=True
    )
    agb_by_config = {key: AGB_CONFIGS[key]["agb"] for key in CONFIG_ORDER}
    if agb_on_key is not None:
        agb_by_config["agb_on"] = agb_on_key
    indices_by_config = {
        key: plane_columns(tables[key][0], agb_by_config[key]) for key in CONFIG_ORDER
    }
    ranges = _plane_ranges(list(indices_by_config.values()))
    low = min(ranges["h_minus_bump"][0] - 0.006, float(jwst["h_minus_bump"].min()) - 0.006)
    high = max(ranges["h_minus_bump"][1], float(jwst["h_minus_bump"].max()) + 0.006)
    ranges["h_minus_bump"] = (low, high)
    xr, yr = ranges["d4000"], ranges["h_minus_bump"]
    d_low, d_high = float(jwst["d4000"].min()), float(jwst["d4000"].max())
    summary = {
        "n_jwst": int(jwst["d4000"].size),
        "redshift_range": [float(jwst["redshift"].min()), float(jwst["redshift"].max())],
        "epoch_window_gyr": [age_low, age_high],
        "cosmology": {"H0": JWST_HUBBLE_CONSTANT, "Omega_m": JWST_OMEGA_MATTER},
    }
    for axis, config_key in zip(axes, CONFIG_ORDER, strict=True):
        config = AGB_CONFIGS[config_key]
        table, codes = tables[config_key]
        ind = indices_by_config[config_key]
        x, y = ind["d4000"], ind["h_minus_bump"]
        rapid = codes == RAPID_QUENCHING
        epoch = table["epoch_gyr"]
        window = (epoch >= age_low - half_step) & (epoch <= age_high + half_step)
        rapid_window = rapid & window

        _draw_open_contours(axis, x, y, xr, yr, POPULATION_LEVELS, "0.45", 0.6, 1)
        _draw_open_contours(
            axis, x[rapid], y[rapid], xr, yr, CLASS_LEVELS, RAPID_QUENCHING_COLOR, 0.7, 2
        )
        draw_population_contours(axis, x[window], y[window], xr, yr)
        axis.collections[-1].set_zorder(3)
        if rapid_window.sum() >= 50:
            draw_class_contours(
                axis, x[rapid_window], y[rapid_window], xr, yr, RAPID_QUENCHING_COLOR, zorder=4
            )

        for (_log_z, result), color in zip(
            tracks_by_template[config["template"]].items(), LOG_Z_TRACK_COLORS, strict=True
        ):
            series = track_series(result, agb_by_config[config_key])
            in_window = (series["epoch_gyr"] >= age_low - half_step) & (
                series["epoch_gyr"] <= age_high + half_step
            )
            axis.plot(
                series["d4000"][in_window],
                series["h_minus_bump"][in_window],
                color=color,
                lw=0.9,
                zorder=6,
            )
            delay = series["epoch_gyr"] - FIDUCIAL["t_q_gyr"]
            for shape, offset in zip(TRACK_MARKER_SHAPES, TRACK_MARKER_OFFSETS_GYR, strict=True):
                index = int(np.argmin(np.abs(delay - offset)))
                if not in_window[index]:
                    continue
                axis.plot(
                    series["d4000"][index],
                    series["h_minus_bump"][index],
                    marker=shape,
                    ms=3.6,
                    mfc=color,
                    mec="0.25",
                    mew=0.35,
                    ls="none",
                    zorder=7,
                )

        axis.errorbar(
            jwst["d4000"],
            jwst["h_minus_bump"],
            xerr=[jwst["d4000_err_low"], jwst["d4000_err_high"]],
            yerr=jwst["h_minus_bump_err"],
            fmt="o",
            ms=JWST_SYMBOL_SIZE,
            mfc="black",
            mec="white",
            mew=0.8,
            ecolor="0.2",
            elinewidth=0.6,
            capsize=0,
            zorder=12,
        )
        set_plane_axes(axis, "d4000", "h_minus_bump", ranges, ylabel=config_key == "agb_off")
        axis.set_xlabel(r"D4000 ($R = 50$)")
        title = config["short_label"]
        if config_key == "agb_on" and agb_on_key is not None:
            title += "\n(agb = " + agb_on_key.removeprefix("agb") + ")"
        row_title(axis, title, fontsize=12)

        in_d4000 = window & (x >= d_low) & (x <= d_high)
        entry = {
            "agb": agb_by_config[config_key],
            "n_epochs_in_window": int(window.sum()),
            "n_rapid_quenching_in_window": int(rapid_window.sum()),
            "n_rapid_quenching_all_epochs": int(rapid.sum()),
            "bump_p01_p50_p99_mag_over_observed_d4000": np.percentile(
                y[in_d4000], [1, 50, 99]
            ).tolist(),
            "deepest_bump_mag_over_observed_d4000": float(y[in_d4000].min()),
            "rapid_quenching_bump_median_in_window_mag": (
                float(np.median(y[rapid_window])) if rapid_window.any() else None
            ),
        }
        if rapid_window.sum() >= 50:
            for fraction in CLASS_LEVELS:
                inside = _fraction_inside(
                    jwst["d4000"],
                    jwst["h_minus_bump"],
                    x[rapid_window],
                    y[rapid_window],
                    xr,
                    yr,
                    fraction,
                )
                entry[f"jwst_inside_rapid_quenching_{int(fraction * 100)}"] = [
                    int(i) for i in np.asarray(jwst["id"])[inside]
                ]
        inside_pop = _fraction_inside(
            jwst["d4000"], jwst["h_minus_bump"], x[window], y[window], xr, yr, POPULATION_LEVELS[-1]
        )
        entry["jwst_inside_population_995"] = int(inside_pop.sum())
        entry["jwst_outside_population_995"] = [int(i) for i in np.asarray(jwst["id"])[~inside_pop]]
        summary[config_key] = entry

    panel_label(axes[0], "(a)", x=0.04, y=0.96)
    panel_label(axes[1], "(b)", x=0.04, y=0.96)
    handles = [
        Patch(
            facecolor="0.85",
            edgecolor="0.55",
            lw=0.5,
            label=rf"Epochs at $t = {age_low:.1f}$--${age_high:.1f}$ Gyr (68, 95, 99.5\%)",
        ),
        Patch(
            facecolor=RAPID_QUENCHING_COLOR,
            edgecolor=RAPID_QUENCHING_COLOR,
            alpha=0.6,
            label=r"Rapid-quenching class in that window (68, 95\%)",
        ),
        Line2D([], [], color="0.45", lw=0.6, ls="--", label="All epochs, 1--13 Gyr"),
        Line2D(
            [],
            [],
            color=RAPID_QUENCHING_COLOR,
            lw=0.7,
            ls="--",
            label="Rapid-quenching class, all epochs",
        ),
    ]
    handles += [
        Line2D([], [], color=color, lw=0.9, label=rf"$\log Z/Z_\odot = {log_z:+.2f}$")
        for log_z, color in zip(LOG_Z_TRACKS, LOG_Z_TRACK_COLORS, strict=True)
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=shape,
            ls="none",
            mfc="0.6",
            mec="0.25",
            mew=0.35,
            ms=3.6,
            label=f"$t_q + {offset:g}$ Gyr",
        )
        for shape, offset in zip(TRACK_MARKER_SHAPES, TRACK_MARKER_OFFSETS_GYR, strict=True)
        if age_low - half_step <= FIDUCIAL["t_q_gyr"] + offset <= age_high + half_step
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            ms=JWST_SYMBOL_SIZE,
            mfc="black",
            mec="white",
            mew=0.8,
            color="0.2",
            lw=0.6,
            label="Lu+2026",
        )
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.4,
        title=(
            rf"Model epochs restricted to the cosmic age at "
            rf"$z = {summary['redshift_range'][0]:.2f}$--${summary['redshift_range'][1]:.2f}$; "
            rf"fiducial SFH $t_q = {FIDUCIAL['t_q_gyr']:g}$ Gyr, "
            rf"$\tau_q = {FIDUCIAL['tau_q_gyr']:g}$ Gyr, drawn within the window"
        ),
        title_fontsize=8,
    )
    save_figure(figure, out_dir, stem)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _to_arrays(entry):
    return {
        key: {metric: np.asarray(value) for metric, value in metrics.items()}
        for key, metrics in entry.items()
    }


def cached_gains(summary, family):
    """The across-seed gains of one family as stored by an earlier full run in
    `publication_summary.json` (`fig4_information_gain` for the exponential family,
    `fig6_robustness_gains` otherwise), or None when that run has not happened."""
    source = (
        summary.get("fig4_information_gain")
        if family == "exponential"
        else summary.get("fig6_robustness_gains", {}).get(family)
    )
    if source is None:
        return None
    return {
        config_key: {
            kind: {"across_seeds": _to_arrays(source[config_key][kind])}
            for kind in ("classifier", "recovery")
        }
        for config_key in CONFIG_ORDER
    }


def compute_gains(family, stride=1, seeds=NOISE_SEEDS, with_log_z=False):
    """Classifier and recovery results for both AGB configurations of one SFH family."""
    out = {}
    for config_key in CONFIG_ORDER:
        config = AGB_CONFIGS[config_key]
        table, codes = load_family_table(family, config["template"], stride=stride)
        post_quench = load_post_quench_table(
            population_dir(family, config["template"]), stride=stride
        )
        out[config_key] = {
            "classifier": run_classifier(
                table, codes, config["agb"], seeds=seeds, with_log_z=with_log_z
            ),
            "recovery": run_recovery(
                post_quench, config["agb"], seeds=seeds, with_log_z=with_log_z
            ),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Publication figure candidates.")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--only", default=None, help="comma-separated subset of " + ",".join(FIGURE_IDS)
    )
    parser.add_argument("--pilot", action="store_true", help="stride-50 tables, one seed, 2 folds")
    parser.add_argument(
        "--final",
        action="store_true",
        help="write the selected figures (index planes, SFH families, the combined age "
        "sensitivity and classifier gain figure) under output/publication/final/ with "
        "content-based names; implies --reuse-gains when the stored gains exist",
    )
    parser.add_argument(
        "--reuse-gains",
        action="store_true",
        help="redraw figures 4, 6 and 7 from the gains stored in publication_summary.json "
        "by an earlier full run instead of recomputing them (seconds instead of minutes)",
    )
    args = parser.parse_args()
    figure_ids = FIGURE_IDS if args.only is None else tuple(args.only.split(","))
    if args.final:
        figure_ids = ("final",)
        args.reuse_gains = True
    out_dir = Path(args.out_dir) if not args.pilot else Path(args.out_dir) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = 50 if args.pilot else 1
    seeds = (SEED,) if args.pilot else NOISE_SEEDS
    global N_FOLDS
    if args.pilot:
        N_FOLDS = 2

    plt.rcParams.update(PUBLICATION_RC)
    edges = time_bin_edges()
    summary_path = out_dir / "publication_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["definitions"] = {
        "agb_configs": AGB_CONFIGS,
        "bump_product": BUMP_PRODUCT,
        "optical_product": OPTICAL_PRODUCT,
        "population_contour_levels": POPULATION_LEVELS,
        "class_contour_levels": CLASS_LEVELS,
        "bump_precisions_mag": BUMP_PRECISIONS,
        "classifier_precision_mag": CLASSIFIER_PRECISION,
        "recovery_precision_mag": RECOVERY_PRECISION,
        "log_z_sigma_dex_when_known": LOG_Z_SIGMA_DEX,
        "k_neighbors": K_NEIGHBORS,
        "n_folds": N_FOLDS,
        "noise_seeds": seeds,
        "gain_definition": "per-fold (with bump - without bump) on the same grouped folds; "
        "standard error = ddof-1 std over folds / sqrt(n_folds); reported as the mean over seeds",
        "pilot": args.pilot,
    }
    timing = {}
    start = time.perf_counter()

    tables = {
        key: load_family_table("exponential", AGB_CONFIGS[key]["template"], stride=stride)
        for key in CONFIG_ORDER
    }
    tables_by_template = {
        "c3k": tables["agb_off"][0],
        "lw02": tables["agb_on"][0],
    }

    if "fig1" in figure_ids or "fig7" in figure_ids:
        t0 = time.perf_counter()
        tracks_by_template = {
            template: compute_metallicity_tracks(template, edges) for template in ("c3k", "lw02")
        }
        timing["metallicity_tracks_s"] = time.perf_counter() - t0
    if "fig1" in figure_ids:
        t0 = time.perf_counter()
        summary["fig1_index_planes"] = figure_1(tables, tracks_by_template, out_dir)
        timing["fig1_s"] = time.perf_counter() - t0
        print(f"fig1 done in {timing['fig1_s']:.1f} s")
    if "fig2" in figure_ids:
        t0 = time.perf_counter()
        summary["fig2_bump_offsets"] = figure_2(tables_by_template, out_dir)
        timing["fig2_s"] = time.perf_counter() - t0
        print(f"fig2 done in {timing['fig2_s']:.1f} s")
    if "fig3" in figure_ids:
        t0 = time.perf_counter()
        fine = tuple(sorted(set(CLOCK_TAU_Q_FINE_GYR) | set(CLOCK_TAU_Q_GYR)))
        clock_tracks = compute_clock_tracks(edges, fine, CLOCK_T_Q_GYR)
        summary["fig3_quenching_clocks"] = figure_3(clock_tracks, out_dir)
        timing["fig3_s"] = time.perf_counter() - t0
        print(f"fig3 done in {timing['fig3_s']:.1f} s")
    gains_by_family = {}
    if "fig4" in figure_ids or "fig6" in figure_ids or "fig7" in figure_ids:
        t0 = time.perf_counter()
        cached = cached_gains(summary, "exponential") if args.reuse_gains else None
        gains_by_family["exponential"] = cached or compute_gains(
            "exponential", stride=stride, seeds=seeds, with_log_z=True
        )
        timing["gains_exponential_s"] = time.perf_counter() - t0
        print(f"exponential gains done in {timing['gains_exponential_s']:.1f} s")
    if "fig4" in figure_ids:
        figure_4(gains_by_family["exponential"], out_dir)
        summary["fig4_information_gain"] = {
            key: {
                kind: value["across_seeds"]
                for kind, value in gains_by_family["exponential"][key].items()
            }
            for key in CONFIG_ORDER
        }
        print("fig4 done")
    if "fig5" in figure_ids:
        t0 = time.perf_counter()
        tables_by_family = {
            family: load_family_table(family, AGB_CONFIGS["agb_on"]["template"], stride=stride)
            for family in ROBUSTNESS_FAMILIES
        }
        summary["fig5_robustness_planes"] = figure_5(tables_by_family, out_dir)
        timing["fig5_s"] = time.perf_counter() - t0
        print(f"fig5 done in {timing['fig5_s']:.1f} s")
    if "fig6" in figure_ids:
        t0 = time.perf_counter()
        for family in ROBUSTNESS_FAMILIES:
            cached = cached_gains(summary, family) if args.reuse_gains else None
            gains_by_family[family] = cached or compute_gains(family, stride=stride, seeds=seeds)
            print(f"{family} gains done at {time.perf_counter() - t0:.1f} s")
        figure_6(gains_by_family, out_dir)
        summary["fig6_robustness_gains"] = {
            family: {
                key: {
                    kind: value["across_seeds"]
                    for kind, value in gains_by_family[family][key].items()
                }
                for key in CONFIG_ORDER
            }
            for family in FAMILIES
        }
        timing["fig6_s"] = time.perf_counter() - t0
        print(f"fig6 done in {timing['fig6_s']:.1f} s")
    if "final" in figure_ids:
        t0 = time.perf_counter()
        final_dir = FINAL_DIR if not args.pilot else out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        tracks_by_template = {
            template: compute_metallicity_tracks(template, edges) for template in ("c3k", "lw02")
        }
        final_summary = {}
        final_summary[FINAL_STEMS["fig1"]] = figure_1(
            tables, tracks_by_template, final_dir, stem=FINAL_STEMS["fig1"]
        )
        tables_by_family = {
            family: load_family_table(family, AGB_CONFIGS["agb_on"]["template"], stride=stride)
            for family in ROBUSTNESS_FAMILIES
        }
        final_summary[FINAL_STEMS["fig5"]] = figure_5(
            tables_by_family, final_dir, stem=FINAL_STEMS["fig5"]
        )
        clock_tracks = compute_clock_tracks(edges, COMBINED_TAU_Q_GYR, (FIDUCIAL["t_q_gyr"],))
        cached = cached_gains(summary, "exponential")
        gains = cached or compute_gains("exponential", stride=stride, seeds=seeds, with_log_z=True)
        final_summary[FINAL_STEMS["combined"]] = figure_combined(clock_tracks, gains, final_dir)
        if JWST_DATA_PATH.exists():
            jwst_tables, jwst_tracks = jwst_inputs_at_r50(tables, tracks_by_template, edges)
            final_summary[FINAL_STEMS["jwst"]] = figure_jwst_plane(
                jwst_tables, jwst_tracks, load_jwst_indices(), final_dir
            )
            final_summary[FINAL_STEMS["jwst_epoch"]] = figure_jwst_plane_epoch_matched(
                jwst_tables, jwst_tracks, load_jwst_indices(), final_dir
            )
            final_summary[FINAL_STEMS["jwst_epoch_agb1"]] = figure_jwst_plane_epoch_matched(
                jwst_tables,
                add_agb1_tracks(jwst_tracks, edges),
                load_jwst_indices(),
                final_dir,
                stem=FINAL_STEMS["jwst_epoch_agb1"],
                agb_on_key="agb1",
            )
            final_summary[FINAL_STEMS["jwst"]]["d4000_product"] = (
                "r50: R = 50 (FWHM) instrument plus 300 km/s dispersion, D4000 only; "
                "the bump stays at r100"
            )
        else:
            print(f"no JWST table at {JWST_DATA_PATH}, skipping {FINAL_STEMS['jwst']}")
        (final_dir / "final_summary.json").write_text(
            json.dumps(_to_native(final_summary), indent=2) + "\n"
        )
        timing["final_s"] = time.perf_counter() - t0
        print(f"final figures done in {timing['final_s']:.1f} s")
    if "fig7" in figure_ids:
        t0 = time.perf_counter()
        summary["fig7_metallicity"] = figure_7(
            tracks_by_template, tables, gains_by_family["exponential"]["agb_on"], out_dir
        )
        timing["fig7_s"] = time.perf_counter() - t0
        print(f"fig7 done in {timing['fig7_s']:.1f} s")

    timing["total_s"] = time.perf_counter() - start
    summary.setdefault("timing_s", {}).update(timing)
    summary_path.write_text(json.dumps(_to_native(summary), indent=2) + "\n")
    print(f"wrote {out_dir} in {timing['total_s']:.1f} s")


if __name__ == "__main__":
    main()
