"""Phase 4: conclusion figures and the metallicity figure.

Five figures test two conclusions and isolate metallicity:

- C1: does the TP-AGB prescription move the population H-minus bump locus enough to
  tell prescriptions apart (four prescriptions on the same 2000 histories)?
- C2a-c: do D4000, HdeltaA and the bump behave as different age clocks after
  quenching, so that adding the bump helps recover the quenching timescale?
- C3: how much does metallicity alone move the fiducial track in the three planes,
  compared with the agb 0 to 2 delta at the same epochs?
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from index_planes import AXIS_LABELS, PLANES, new_plane_figure
from population_classes import add_measurement_noise, grouped_folds
from run_single_csp import FIDUCIAL, compute_track_indices
from sfh_model import time_bin_edges

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output" / "analysis"
GRID_DIRS = {
    "c3k": PROJECT_DIR / "output" / "ssp_grid",
    "lw02": PROJECT_DIR / "output" / "ssp_grid_lw02",
}
POPULATION_DIRS = {
    "c3k": PROJECT_DIR / "output" / "population",
    "lw02": PROJECT_DIR / "output" / "population_lw02",
}
TEMPLATE_LABELS = {"c3k": "C3K", "lw02": "LW02"}
TEMPLATE_ORDER = ("c3k", "lw02")

SEED = 20260924
NOISE_SEEDS = (SEED, SEED + 1, SEED + 2)
MIN_EPOCH_GYR = 1.0
BUMP_PRODUCT = "sigma300"
D4000_SIGMA = 0.05
HDELTA_A_SIGMA = 0.5
BUMP_PRECISIONS = (0.005, 0.010, 0.020)
K_NEIGHBORS = 25
N_FOLDS = 5
INDEX_KEYS = ("d4000", "hdelta_a", "h_minus_bump")

C1_PRESCRIPTIONS = (
    ("c3k", "agb0", "C3K agb0"),
    ("c3k", "agb1", "C3K agb1"),
    ("lw02", "agb1", "LW02 agb1"),
    ("lw02", "agb2", "LW02 agb2"),
)
C1_COLORS = {
    "C3K agb0": "tab:blue",
    "C3K agb1": "tab:green",
    "LW02 agb1": "tab:purple",
    "LW02 agb2": "tab:red",
}
C1_N_BINS = 20
C1_YARDSTICK_MAG = 0.01
C1_D4000_SLICES = ((1.3, 1.5), (1.5, 1.7), (1.7, 1.9))
C1_HIST_BINS = 40

C2A_TAU_Q_FAMILY_GYR = (0.1, 0.3, 1.0, 3.0)
C2A_TQ_FAMILY_GYR = (1.5, 3.0, 4.5)
C2A_WINDOW_GYR = (-1.0, 6.0)
C2A_EXTREMUM_KIND = {"d4000": "max", "hdelta_a": "max", "h_minus_bump": "min"}

C2B_MARKER_STEP_GYR = 0.5
C2B_MARKER_MAX_GYR = 6.0
C2B_MARKER_SHAPES = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "p", "<", ">")

C2C_TEMPLATES = (
    ("c3k", "agb2", "C3K agb2"),
    ("lw02", "agb2", "LW02 agb2"),
    ("c3k", "agb0", "C3K agb0 (control)"),
)
C2C_COLORS = {"C3K agb2": "tab:blue", "LW02 agb2": "tab:red", "C3K agb0 (control)": "tab:gray"}
TARGET_NAMES = ("log10_time_since_quenching_gyr", "log10_tau_q_gyr")
POST_QUENCH_WINDOW_GYR = (0.0, 6.0)

C3_LOG_Z_GRID = (-0.5, -0.25, 0.0, 0.25)
C3_MARKER_OFFSETS_GYR = (0.0, 0.5, 1.0, 2.0, 5.0)
C3_MARKER_SHAPES = ("o", "s", "^", "D", "P")
C3_CMAP = "cividis"


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


def load_population_table(pop_dir, epoch_min=MIN_EPOCH_GYR, stride=1):
    with np.load(Path(pop_dir) / "indices.npz") as data:
        table = {key: data[key] for key in data.files}
    mask = table["epoch_gyr"] >= epoch_min
    table = {key: values[mask] for key, values in table.items()}
    if stride > 1:
        table = {key: values[::stride] for key, values in table.items()}
    return table


def load_post_quench_table(pop_dir, stride=1):
    table = load_population_table(pop_dir, stride=stride)
    time_since_quenching = table["time_since_quenching_gyr"]
    window_lo, window_hi = POST_QUENCH_WINDOW_GYR
    mask = (time_since_quenching > window_lo) & (time_since_quenching < window_hi)
    return {key: values[mask] for key, values in table.items()}


# ---------------------------------------------------------------------------
# C1: TP-AGB population test
# ---------------------------------------------------------------------------


def c1_series(tables, pop, agb, x_key):
    x = tables[pop][f"{x_key}_{agb}"]
    y = tables[pop][f"h_minus_bump_{BUMP_PRODUCT}_{agb}"]
    return x, y


def percentile_band(x, y, edges):
    n_bins = edges.size - 1
    bin_index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(n_bins, np.nan)
    lower = np.full(n_bins, np.nan)
    upper = np.full(n_bins, np.nan)
    for bin_k in range(n_bins):
        values = y[bin_index == bin_k]
        if values.size:
            lower[bin_k], median[bin_k], upper[bin_k] = np.percentile(values, [16, 50, 84])
    return centers, lower, median, upper


def _c1_band_panel(axis, tables, x_key):
    x_all = np.concatenate(
        [c1_series(tables, pop, agb, x_key)[0] for pop, agb, _ in C1_PRESCRIPTIONS]
    )
    y_all = np.concatenate(
        [c1_series(tables, pop, agb, x_key)[1] for pop, agb, _ in C1_PRESCRIPTIONS]
    )
    edges = np.linspace(x_all.min(), x_all.max(), C1_N_BINS + 1)
    band_summary = {}
    for pop, agb, label in C1_PRESCRIPTIONS:
        x, y = c1_series(tables, pop, agb, x_key)
        centers, lower, median, upper = percentile_band(x, y, edges)
        color = C1_COLORS[label]
        axis.fill_between(centers, lower, upper, color=color, alpha=0.22, linewidth=0)
        axis.plot(centers, median, color=color, lw=1.8, label=label)
        band_summary[label] = {
            "bin_centers": centers.tolist(),
            "median_mag": median.tolist(),
            "p16_mag": lower.tolist(),
            "p84_mag": upper.tolist(),
        }
    x_ref = edges[0] + 0.08 * (edges[-1] - edges[0])
    y_ref = float(np.nanpercentile(y_all, 90))
    axis.errorbar(
        [x_ref],
        [y_ref],
        yerr=[C1_YARDSTICK_MAG],
        fmt="none",
        ecolor="black",
        capsize=4,
        label="0.01 mag yardstick",
    )
    axis.set_xlabel(AXIS_LABELS[x_key])
    axis.set_ylabel(AXIS_LABELS["h_minus_bump"])
    axis.invert_yaxis()
    return band_summary, {"yardstick_position": {"x": x_ref, "y_mag": y_ref}}


def _c1_slice_mask(x, lo, hi, is_last):
    if is_last:
        return (x >= lo) & (x <= hi)
    return (x >= lo) & (x < hi)


def _c1_hist_panel(axis, tables, lo, hi, is_last):
    values_by_label = {}
    for pop, agb, label in C1_PRESCRIPTIONS:
        x, y = c1_series(tables, pop, agb, "d4000")
        values_by_label[label] = y[_c1_slice_mask(x, lo, hi, is_last)]
    nonempty = [values for values in values_by_label.values() if values.size]
    bin_edges = np.linspace(
        min(values.min() for values in nonempty),
        max(values.max() for values in nonempty),
        C1_HIST_BINS + 1,
    )
    slice_summary = {}
    for label, values in values_by_label.items():
        axis.hist(
            values,
            bins=bin_edges,
            histtype="step",
            density=True,
            color=C1_COLORS[label],
            label=label,
            lw=1.6,
        )
        p16, median, p84 = np.percentile(values, [16, 50, 84])
        slice_summary[label] = {
            "n_rows": int(values.size),
            "median_mag": float(median),
            "p16_mag": float(p16),
            "p84_mag": float(p84),
            "width_16_84_mag": float(p84 - p16),
        }
    separations = []
    for (_, _, label_a), (_, _, label_b) in zip(
        C1_PRESCRIPTIONS[:-1], C1_PRESCRIPTIONS[1:], strict=True
    ):
        half_width = 0.5 * max(
            slice_summary[label_a]["width_16_84_mag"], slice_summary[label_b]["width_16_84_mag"]
        )
        delta = slice_summary[label_b]["median_mag"] - slice_summary[label_a]["median_mag"]
        separations.append(
            {
                "pair": f"{label_a} -> {label_b}",
                "delta_median_mag": float(delta),
                "delta_over_wider_band_half_width": (
                    float(delta / half_width) if half_width > 0 else float("nan")
                ),
                "delta_over_0.01_mag": float(delta / C1_YARDSTICK_MAG),
            }
        )
    axis.invert_xaxis()
    axis.set_xlabel(AXIS_LABELS["h_minus_bump"])
    axis.set_ylabel("density")
    title = f"D4000 in [{lo:.1f}, {hi:.1f}{']' if is_last else ')'}"
    axis.set_title(title, fontsize=9)
    return {"medians_and_widths": slice_summary, "neighbouring_separations": separations}


def figure_c1(tables, out_path):
    mosaic = [["bump_d4000"] * 3, ["bump_hdelta_a"] * 3, ["hist0", "hist1", "hist2"]]
    figure, axd = plt.subplot_mosaic(mosaic, figsize=(12, 12), layout="constrained")
    summary = {"bands": {}, "yardstick_positions": {}, "slices": {}}
    for x_key, panel_key in (("d4000", "bump_d4000"), ("hdelta_a", "bump_hdelta_a")):
        band_summary, position = _c1_band_panel(axd[panel_key], tables, x_key)
        summary["bands"][x_key] = band_summary
        summary["yardstick_positions"][x_key] = position["yardstick_position"]
    axd["bump_d4000"].legend(frameon=False, fontsize=8, loc="best")
    for slice_index, (lo, hi) in enumerate(C1_D4000_SLICES):
        is_last = slice_index == len(C1_D4000_SLICES) - 1
        slice_summary = _c1_hist_panel(axd[f"hist{slice_index}"], tables, lo, hi, is_last)
        summary["slices"][f"d4000_{lo:.1f}_{hi:.1f}"] = slice_summary
    axd["hist0"].legend(frameon=False, fontsize=7, loc="best")
    if out_path is not None:
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return summary


# ---------------------------------------------------------------------------
# C2a/C2b: age clocks and their tracks in the planes
# ---------------------------------------------------------------------------


def _family_params(family_name, value):
    if family_name == "tau_q":
        return FIDUCIAL["t_q_gyr"], value
    return value, FIDUCIAL["tau_q_gyr"]


def compute_family_tracks(grid_dir, family_name, values, edges_gyr):
    tracks = []
    for value in values:
        t_q_gyr, tau_q_gyr = _family_params(family_name, value)
        result = compute_track_indices(
            grid_dir, t_q_gyr, tau_q_gyr, FIDUCIAL["log_z"], edges_gyr=edges_gyr
        )
        tracks.append(
            {
                "family": family_name,
                "value": float(value),
                "t_q_gyr": t_q_gyr,
                "tau_q_gyr": tau_q_gyr,
                "result": result,
            }
        )
    return tracks


def compute_c2_tracks(edges_gyr, tau_q_family=C2A_TAU_Q_FAMILY_GYR, t_q_family=C2A_TQ_FAMILY_GYR):
    tracks_by_template = {}
    for template in TEMPLATE_ORDER:
        tau_tracks = compute_family_tracks(GRID_DIRS[template], "tau_q", tau_q_family, edges_gyr)
        t_q_tracks = compute_family_tracks(GRID_DIRS[template], "t_q", t_q_family, edges_gyr)
        tracks_by_template[template] = tau_tracks + t_q_tracks
    return tracks_by_template


def family_color(family_name, value, tau_q_family, t_q_family):
    if family_name == "tau_q":
        cmap = plt.get_cmap("viridis")
        grid = np.log10(np.asarray(tau_q_family, dtype=float))
        target = np.log10(value)
    else:
        cmap = plt.get_cmap("plasma")
        grid = np.asarray(t_q_family, dtype=float)
        target = value
    norm = plt.Normalize(grid.min(), grid.max())
    return cmap(norm(target))


def track_label(track):
    if track["family"] == "tau_q":
        return f"tau_q={track['value']:g} Gyr"
    return f"t_q={track['value']:g} Gyr"


def track_series(result, agb_key="agb2", bump_product=BUMP_PRODUCT):
    return {
        "epoch_gyr": result["epoch_gyr"],
        "d4000": result[agb_key]["sigma300"]["d4000"],
        "hdelta_a": result[agb_key]["sigma300"]["hdelta_a"],
        "h_minus_bump": result[agb_key][bump_product]["h_minus_bump"],
    }


def track_extremum(delay_gyr, values, kind):
    index = int(np.argmax(values)) if kind == "max" else int(np.argmin(values))
    boundary = index in (0, values.size - 1)
    return {
        "delay_gyr": float(delay_gyr[index]),
        "value": float(values[index]),
        "at_window_boundary": boundary,
    }


def figure_c2a(
    tracks_by_template, out_path, tau_q_family=C2A_TAU_Q_FAMILY_GYR, t_q_family=C2A_TQ_FAMILY_GYR
):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained", sharex=True)
    summary = {}
    legend_handles = None
    for row, template in enumerate(TEMPLATE_ORDER):
        summary[template] = {index_key: {} for index_key in INDEX_KEYS}
        for col, index_key in enumerate(INDEX_KEYS):
            axis = axes[row, col]
            for track in tracks_by_template[template]:
                series = track_series(track["result"])
                delay = series["epoch_gyr"] - track["t_q_gyr"]
                window = (delay >= C2A_WINDOW_GYR[0]) & (delay <= C2A_WINDOW_GYR[1])
                color = family_color(track["family"], track["value"], tau_q_family, t_q_family)
                label = track_label(track)
                axis.plot(
                    delay[window], series[index_key][window], color=color, lw=1.6, label=label
                )
                extremum = track_extremum(
                    delay[window], series[index_key][window], C2A_EXTREMUM_KIND[index_key]
                )
                if extremum["at_window_boundary"]:
                    axis.scatter(
                        [extremum["delay_gyr"]],
                        [extremum["value"]],
                        marker="*",
                        s=110,
                        facecolors="none",
                        edgecolors=color,
                        linewidth=1.6,
                        zorder=5,
                    )
                else:
                    axis.scatter(
                        [extremum["delay_gyr"]],
                        [extremum["value"]],
                        marker="*",
                        s=90,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=5,
                    )
                summary[template][index_key][label] = extremum
            axis.axvline(0.0, color="0.7", ls=":", lw=1.0, zorder=0)
            axis.set_ylabel(AXIS_LABELS[index_key])
            if index_key == "h_minus_bump":
                axis.invert_yaxis()
            if row == 1:
                axis.set_xlabel("time since quenching [Gyr]")
            if row == 0 and col == 0:
                legend_handles = axis.get_legend_handles_labels()
        axes[row, 0].set_title(TEMPLATE_LABELS[template], loc="left")
    boundary_handle = plt.Line2D(
        [],
        [],
        marker="*",
        linestyle="none",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.4,
        label="extremum at window edge (monotonic in window)",
    )
    handles, labels = legend_handles
    handles = [*handles, boundary_handle]
    labels = [*labels, boundary_handle.get_label()]
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    if out_path is not None:
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return summary


def marker_times(step, max_gyr):
    return np.round(np.arange(step, max_gyr + 1e-9, step), 2)


def figure_c2b(
    tracks_by_template, out_path, tau_q_family=C2A_TAU_Q_FAMILY_GYR, t_q_family=C2A_TQ_FAMILY_GYR
):
    figure, axes = new_plane_figure(n_rows=2)
    times = marker_times(C2B_MARKER_STEP_GYR, C2B_MARKER_MAX_GYR)
    summary = {}
    curve_handles = None
    for row, template in enumerate(TEMPLATE_ORDER):
        summary[template] = {}
        for track in tracks_by_template[template]:
            series = track_series(track["result"])
            delay = series["epoch_gyr"] - track["t_q_gyr"]
            post_quench = (delay >= 0.0) & (delay <= C2B_MARKER_MAX_GYR)
            color = family_color(track["family"], track["value"], tau_q_family, t_q_family)
            label = track_label(track)
            for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
                axis.plot(
                    series[x_key][post_quench],
                    series[y_key][post_quench],
                    color=color,
                    lw=1.4,
                    label=label,
                )
            marker_record = {"delay_gyr": [], "d4000": [], "hdelta_a": [], "h_minus_bump": []}
            for shape, t_mark in zip(C2B_MARKER_SHAPES, times, strict=True):
                index = int(np.argmin(np.abs(delay - t_mark)))
                marker_record["delay_gyr"].append(float(delay[index]))
                for key in INDEX_KEYS:
                    marker_record[key].append(float(series[key][index]))
                for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
                    axis.scatter(
                        [series[x_key][index]],
                        [series[y_key][index]],
                        marker=shape,
                        s=45,
                        color=color,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=5,
                    )
            summary[template][label] = marker_record
        if curve_handles is None:
            curve_handles = axes[row, 0].get_legend_handles_labels()
        axes[row, 0].set_title(TEMPLATE_LABELS[template], loc="left")
    figure.legend(*curve_handles, loc="outside upper center", ncol=4, frameon=False, fontsize=7)
    marker_handles = [
        plt.Line2D(
            [],
            [],
            marker=shape,
            linestyle="none",
            color="0.3",
            markeredgecolor="black",
            label=f"+{t:.1f} Gyr",
        )
        for shape, t in zip(C2B_MARKER_SHAPES, times, strict=True)
    ]
    figure.legend(
        handles=marker_handles,
        loc="outside lower center",
        ncol=6,
        frameon=False,
        fontsize=7,
        title="time since quenching",
    )
    if out_path is not None:
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return summary


# ---------------------------------------------------------------------------
# C2c: SFH recovery test
# ---------------------------------------------------------------------------


def build_regression_feature_sets(table, agb, seed):
    streams = np.random.SeedSequence(seed).spawn(1 + len(BUMP_PRECISIONS))
    d4000 = table[f"d4000_{agb}"]
    hdelta_a = table[f"hdelta_a_{agb}"]
    noisy_optical = add_measurement_noise(
        np.column_stack([d4000, hdelta_a]),
        np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        np.random.default_rng(streams[0]),
    )
    feature_sets = {
        "no_bump": {
            "features": noisy_optical,
            "scales": np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        }
    }
    bump = table[f"h_minus_bump_{BUMP_PRODUCT}_{agb}"]
    for stream, precision in zip(streams[1:], BUMP_PRECISIONS, strict=True):
        noisy_bump = add_measurement_noise(
            bump[:, None], np.array([precision]), np.random.default_rng(stream)
        )[:, 0]
        feature_sets[f"bump_{precision:.3f}"] = {
            "features": np.column_stack([noisy_optical, noisy_bump]),
            "scales": np.array([D4000_SIGMA, HDELTA_A_SIGMA, precision]),
        }
    return feature_sets


def knn_regress(features, targets, query, k, feature_scales):
    features = np.asarray(features, dtype=float) / feature_scales
    query = np.asarray(query, dtype=float) / feature_scales
    tree = cKDTree(features)
    _, neighbor_indices = tree.query(query, k=k)
    neighbor_indices = np.asarray(neighbor_indices).reshape(query.shape[0], k)
    return np.asarray(targets)[neighbor_indices].mean(axis=1)


def cross_validated_regression(
    features, targets, groups, feature_scales, k=K_NEIGHBORS, n_folds=N_FOLDS, seed=SEED
):
    folds = grouped_folds(groups, n_folds, np.random.default_rng(seed))
    rms_per_fold = []
    for fold in range(n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        prediction = knn_regress(
            features[train_mask], targets[train_mask], features[test_mask], k, feature_scales
        )
        error = prediction - targets[test_mask]
        rms_per_fold.append(np.sqrt(np.mean(error**2, axis=0)))
    return np.array(rms_per_fold)


def regression_targets(table):
    return np.column_stack(
        [np.log10(table["time_since_quenching_gyr"]), np.log10(table["tau_q_gyr"])]
    )


def run_regression_template(table, agb, seed, n_folds=N_FOLDS):
    targets = regression_targets(table)
    groups = table["history_id"]
    feature_sets = build_regression_feature_sets(table, agb, seed)
    results = {}
    for key, spec in feature_sets.items():
        results[key] = cross_validated_regression(
            spec["features"],
            targets,
            groups,
            spec["scales"],
            k=K_NEIGHBORS,
            n_folds=n_folds,
            seed=seed,
        )
    return results


def compute_c2c_results(post_quench_tables, seeds=NOISE_SEEDS, n_folds=N_FOLDS):
    results_by_seed = {}
    for seed in seeds:
        results_by_seed[str(seed)] = {
            label: run_regression_template(post_quench_tables[pop], agb, seed, n_folds=n_folds)
            for pop, agb, label in C2C_TEMPLATES
        }
    return results_by_seed


TARGET_LABELS = {
    "log10_time_since_quenching_gyr": "log10 time since quenching [Gyr]",
    "log10_tau_q_gyr": "log10 tau_q [Gyr]",
}
C2C_ERROR_BAR_DEFINITION = (
    "mean, over the 3 noise seeds, of the per-fold standard error of the RMS "
    "(ddof 1 std of the 5 per-fold RMS values / sqrt(5))"
)


def fold_standard_error(rms_per_fold, n_folds):
    return rms_per_fold.std(axis=0, ddof=1) / np.sqrt(n_folds)


def paired_regression_difference(rms_per_fold, baseline_rms_per_fold, n_folds):
    diff = rms_per_fold - baseline_rms_per_fold
    mean = diff.mean(axis=0)
    standard_error = diff.std(axis=0, ddof=1) / np.sqrt(n_folds)
    return mean, standard_error


def summarize_c2c(results_by_seed, n_folds=N_FOLDS):
    seeds = list(results_by_seed)
    labels = list(results_by_seed[seeds[0]])
    feature_keys = list(results_by_seed[seeds[0]][labels[0]])
    summary = {}
    for label in labels:
        summary[label] = {}
        for key in feature_keys:
            rms_per_seed = np.array(
                [results_by_seed[seed][label][key].mean(axis=0) for seed in seeds]
            )
            fold_se_per_seed = np.array(
                [fold_standard_error(results_by_seed[seed][label][key], n_folds) for seed in seeds]
            )
            entry = {
                "rms_mean_over_seeds_dex": rms_per_seed.mean(axis=0).tolist(),
                "rms_std_over_seeds_dex": rms_per_seed.std(axis=0, ddof=1).tolist(),
                "rms_fold_standard_error_mean_over_seeds_dex": fold_se_per_seed.mean(
                    axis=0
                ).tolist(),
            }
            if key != "no_bump":
                per_seed = {}
                gains = []
                standard_errors = []
                for seed in seeds:
                    mean_diff, standard_error = paired_regression_difference(
                        results_by_seed[seed][label][key],
                        results_by_seed[seed][label]["no_bump"],
                        n_folds,
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        gain_over_se = np.where(
                            standard_error > 0, mean_diff / standard_error, np.nan
                        )
                    per_seed[seed] = {
                        "paired_gain_mean_dex": mean_diff.tolist(),
                        "paired_gain_standard_error_dex": standard_error.tolist(),
                        "paired_gain_over_standard_error": gain_over_se.tolist(),
                    }
                    gains.append(mean_diff)
                    standard_errors.append(standard_error)
                gains = np.array(gains)
                standard_errors = np.array(standard_errors)
                entry["paired_gain_per_seed"] = per_seed
                entry["paired_gain_mean_over_seeds_dex"] = gains.mean(axis=0).tolist()
                entry["paired_gain_std_over_seeds_dex"] = gains.std(axis=0, ddof=1).tolist()
                entry["paired_gain_standard_error_mean_over_seeds_dex"] = standard_errors.mean(
                    axis=0
                ).tolist()
            summary[label][key] = entry
    return summary


def figure_c2c(summary, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    precisions = np.array(BUMP_PRECISIONS)
    for col, target_name in enumerate(TARGET_NAMES):
        axis = axes[col]
        for label, color in C2C_COLORS.items():
            baseline_mean = summary[label]["no_bump"]["rms_mean_over_seeds_dex"][col]
            baseline_se = summary[label]["no_bump"]["rms_fold_standard_error_mean_over_seeds_dex"][
                col
            ]
            axis.axhline(baseline_mean, color=color, ls="--", lw=1.2, alpha=0.7)
            axis.axhspan(
                baseline_mean - baseline_se, baseline_mean + baseline_se, color=color, alpha=0.08
            )
            means = [
                summary[label][f"bump_{p:.3f}"]["rms_mean_over_seeds_dex"][col] for p in precisions
            ]
            standard_errors = [
                summary[label][f"bump_{p:.3f}"]["rms_fold_standard_error_mean_over_seeds_dex"][col]
                for p in precisions
            ]
            axis.errorbar(
                precisions,
                means,
                yerr=standard_errors,
                marker="o",
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_xlabel("bump precision [mag]")
        axis.set_ylabel("RMS error [dex]")
        axis.set_title(TARGET_LABELS[target_name], fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(f"error bars: {C2C_ERROR_BAR_DEFINITION}", fontsize=8)
    if out_path is not None:
        figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# C3: metallicity only
# ---------------------------------------------------------------------------


def compute_c3_tracks(grid_dir, edges_gyr, log_z_grid=C3_LOG_Z_GRID):
    tracks = []
    for log_z in log_z_grid:
        result = compute_track_indices(
            grid_dir, FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], log_z, edges_gyr=edges_gyr
        )
        tracks.append({"log_z": float(log_z), "result": result})
    return tracks


def compute_c3_tracks_all_templates(edges_gyr, log_z_grid=C3_LOG_Z_GRID):
    return {
        template: compute_c3_tracks(GRID_DIRS[template], edges_gyr, log_z_grid)
        for template in TEMPLATE_ORDER
    }


def figure_c3(tracks_by_template, out_path, marker_offsets=C3_MARKER_OFFSETS_GYR):
    figure, axes = new_plane_figure(n_rows=2)
    cmap = plt.get_cmap(C3_CMAP)
    all_log_z = [track["log_z"] for track in next(iter(tracks_by_template.values()))]
    norm = plt.Normalize(min(all_log_z), max(all_log_z))
    summary = {}
    for row, template in enumerate(TEMPLATE_ORDER):
        offset_agb2 = {key: {offset: [] for offset in marker_offsets} for key in INDEX_KEYS}
        offset_agb0 = {key: {offset: [] for offset in marker_offsets} for key in INDEX_KEYS}
        for track in tracks_by_template[template]:
            series_agb2 = track_series(track["result"], agb_key="agb2")
            series_agb0 = track_series(track["result"], agb_key="agb0")
            delay = series_agb2["epoch_gyr"] - FIDUCIAL["t_q_gyr"]
            post_quench = (delay >= 0.0) & (delay <= max(marker_offsets))
            color = cmap(norm(track["log_z"]))
            for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
                axis.plot(
                    series_agb2[x_key][post_quench],
                    series_agb2[y_key][post_quench],
                    color=color,
                    lw=1.5,
                )
            for shape, offset in zip(C3_MARKER_SHAPES, marker_offsets, strict=True):
                index = int(np.argmin(np.abs(delay - offset)))
                for key in INDEX_KEYS:
                    offset_agb2[key][offset].append(series_agb2[key][index])
                    offset_agb0[key][offset].append(series_agb0[key][index])
                for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
                    axis.scatter(
                        [series_agb2[x_key][index]],
                        [series_agb2[y_key][index]],
                        marker=shape,
                        s=45,
                        color=color,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=5,
                    )
        axes[row, 0].set_title(TEMPLATE_LABELS[template], loc="left")
        summary[template] = {"spread_across_metallicity": {}, "agb0_to_agb2_delta": {}}
        for key in INDEX_KEYS:
            summary[template]["spread_across_metallicity"][key] = {}
            summary[template]["agb0_to_agb2_delta"][key] = {}
            for offset in marker_offsets:
                values_agb2 = np.array(offset_agb2[key][offset])
                values_agb0 = np.array(offset_agb0[key][offset])
                delta = values_agb2 - values_agb0
                offset_key = f"t_q+{offset:.1f}_gyr"
                summary[template]["spread_across_metallicity"][key][offset_key] = {
                    "min": float(values_agb2.min()),
                    "max": float(values_agb2.max()),
                    "spread": float(values_agb2.max() - values_agb2.min()),
                }
                summary[template]["agb0_to_agb2_delta"][key][offset_key] = {
                    "min": float(delta.min()),
                    "max": float(delta.max()),
                    "mean": float(delta.mean()),
                }
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(mappable, ax=list(axes.ravel()), label=r"$\log(Z/Z_\odot)$", pad=0.02)
    marker_handles = [
        plt.Line2D(
            [],
            [],
            marker=shape,
            linestyle="none",
            color="0.3",
            markeredgecolor="black",
            label=(f"$t_q$+{offset:.1f} Gyr" if offset > 0 else "$t_q$"),
        )
        for shape, offset in zip(C3_MARKER_SHAPES, marker_offsets, strict=True)
    ]
    figure.legend(
        handles=marker_handles, loc="outside lower center", ncol=5, frameon=False, fontsize=8
    )
    if out_path is not None:
        figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_pilot():
    start = time.perf_counter()
    edges = time_bin_edges()

    tables = {
        template: load_population_table(POPULATION_DIRS[template], stride=200)
        for template in TEMPLATE_ORDER
    }
    t0 = time.perf_counter()
    figure_c1(tables, None)
    print(f"c1 pilot: {time.perf_counter() - t0:.2f} s, {tables['c3k']['history_id'].size} rows")

    pilot_tau_q = (0.3, 3.0)
    pilot_t_q = (3.0,)
    t0 = time.perf_counter()
    tracks = compute_c2_tracks(edges, tau_q_family=pilot_tau_q, t_q_family=pilot_t_q)
    print(f"c2 tracks pilot: {time.perf_counter() - t0:.2f} s for 3 tracks x 2 templates")
    t0 = time.perf_counter()
    figure_c2a(tracks, None, tau_q_family=pilot_tau_q, t_q_family=pilot_t_q)
    print(f"c2a pilot: {time.perf_counter() - t0:.2f} s")
    t0 = time.perf_counter()
    figure_c2b(tracks, None, tau_q_family=pilot_tau_q, t_q_family=pilot_t_q)
    print(f"c2b pilot: {time.perf_counter() - t0:.2f} s")

    post_quench_tables = {
        template: load_post_quench_table(POPULATION_DIRS[template], stride=200)
        for template in TEMPLATE_ORDER
    }
    n_rows = int(post_quench_tables["c3k"]["history_id"].size)
    t0 = time.perf_counter()
    compute_c2c_results(post_quench_tables, seeds=(SEED,), n_folds=2)
    elapsed = time.perf_counter() - t0
    full_rows_estimate = 240_000
    projected_minutes = (
        elapsed
        * (full_rows_estimate / max(n_rows, 1))
        * (N_FOLDS / 2)
        * (len(NOISE_SEEDS) / 1)
        / 60.0
    )
    print(f"c2c pilot: {elapsed:.2f} s on {n_rows} rows, 2 folds, 1 seed")
    print(f"projected full c2c regression: {projected_minutes:.1f} min")

    pilot_log_z = (-0.25, 0.25)
    t0 = time.perf_counter()
    c3_tracks = compute_c3_tracks_all_templates(edges, log_z_grid=pilot_log_z)
    figure_c3(c3_tracks, None)
    print(f"c3 pilot: {time.perf_counter() - t0:.2f} s")

    print(f"pilot total: {time.perf_counter() - start:.2f} s")


def _serialize_results_by_seed(results_by_seed):
    return {
        seed: {
            label: {key: values.tolist() for key, values in feature_sets.items()}
            for label, feature_sets in labels.items()
        }
        for seed, labels in results_by_seed.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 conclusion and metallicity figures.")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--pilot", action="store_true", help="subsampled run of every figure, nothing saved"
    )
    args = parser.parse_args()

    if args.pilot:
        _run_pilot()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = time_bin_edges()
    start = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    population_tables = {
        template: load_population_table(POPULATION_DIRS[template]) for template in TEMPLATE_ORDER
    }
    timing["load_population_tables_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c1_summary = figure_c1(population_tables, out_dir / "c1_tpagb_population_test.png")
    timing["c1_s"] = time.perf_counter() - t0
    print(f"c1 done in {timing['c1_s']:.1f} s")

    t0 = time.perf_counter()
    c2_tracks = compute_c2_tracks(edges)
    timing["c2_tracks_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c2a_summary = figure_c2a(c2_tracks, out_dir / "c2_age_clocks.png")
    timing["c2a_s"] = time.perf_counter() - t0
    print(f"c2a done in {timing['c2a_s']:.1f} s")

    t0 = time.perf_counter()
    c2b_summary = figure_c2b(c2_tracks, out_dir / "c2_clock_planes.png")
    timing["c2b_s"] = time.perf_counter() - t0
    print(f"c2b done in {timing['c2b_s']:.1f} s")

    t0 = time.perf_counter()
    post_quench_tables = {
        template: load_post_quench_table(POPULATION_DIRS[template]) for template in TEMPLATE_ORDER
    }
    timing["load_post_quench_tables_s"] = time.perf_counter() - t0
    n_rows_by_template = {
        template: int(post_quench_tables[template]["history_id"].size)
        for template in TEMPLATE_ORDER
    }
    print(f"post-quench rows: {n_rows_by_template}")

    t0 = time.perf_counter()
    results_by_seed = compute_c2c_results(post_quench_tables)
    timing["c2c_regression_s"] = time.perf_counter() - t0
    print(f"c2c regression done in {timing['c2c_regression_s']:.1f} s")

    c2c_summary = summarize_c2c(results_by_seed)
    t0 = time.perf_counter()
    figure_c2c(c2c_summary, out_dir / "c2_sfh_recovery.png")
    timing["c2c_figure_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c3_tracks = compute_c3_tracks_all_templates(edges)
    timing["c3_tracks_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c3_summary = figure_c3(c3_tracks, out_dir / "c3_metallicity_planes.png")
    timing["c3_s"] = time.perf_counter() - t0
    print(f"c3 done in {timing['c3_s']:.1f} s")

    timing["total_s"] = time.perf_counter() - start

    summary = {
        "c1_tpagb_population_test": c1_summary,
        "c2_age_clocks": c2a_summary,
        "c2_clock_planes": c2b_summary,
        "c2_sfh_recovery": {
            "definition": {
                "targets": list(TARGET_NAMES),
                "bump_product": BUMP_PRODUCT,
                "k_neighbors": K_NEIGHBORS,
                "n_folds": N_FOLDS,
                "noise_seeds": list(NOISE_SEEDS),
                "post_quench_window_gyr": list(POST_QUENCH_WINDOW_GYR),
                "min_epoch_gyr": MIN_EPOCH_GYR,
                "paired_gain_definition": "per-fold (bump set - no_bump) RMS difference on the "
                "same folds; standard error = ddof 1 std over folds / sqrt(n_folds); reported "
                "per seed as paired_gain_mean_dex/paired_gain_standard_error_dex/"
                "paired_gain_over_standard_error, and across seeds as the mean of that "
                "standard error plus the across-seed mean/std of the gain",
                "figure_error_bar_definition": C2C_ERROR_BAR_DEFINITION,
            },
            "n_rows_by_template": n_rows_by_template,
            "rms_per_fold_dex_by_seed": _serialize_results_by_seed(results_by_seed),
            "summary": c2c_summary,
        },
        "c3_metallicity_planes": c3_summary,
        "timing_s": timing,
    }
    summary_text = json.dumps(_to_native(summary), indent=2) + "\n"
    (out_dir / "conclusion_summary.json").write_text(summary_text)
    print(f"wrote {out_dir} in {timing['total_s']:.1f} s")


if __name__ == "__main__":
    main()
