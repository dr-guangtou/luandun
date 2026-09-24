"""Q2: isolating fast-quenching epochs in the index planes and with a noise-aware classifier.

Population epochs with `epoch_gyr < 1.0` are dropped, classes are assigned with
`population_classes.assign_classes` from the two sSFR windows, and the rapid-quenching
class is tested for separability against the other three classes in the three index
planes (purity maps) and with a grouped, cross-validated k-nearest-neighbour classifier,
with and without the H-minus bump, at three measurement precisions.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from index_planes import PLANES, new_plane_figure
from population_classes import (
    CLASS_NAMES,
    QUIESCENT,
    RAPID_QUENCHING,
    STAR_FORMING,
    TRANSITIONAL,
    add_measurement_noise,
    assign_classes,
    completeness_purity,
    cross_validated_metrics,
    grouped_folds,
    is_post_starburst,
    knn_predict,
)

SEED = 20260924
INPUT_PATH = Path(__file__).resolve().parent / "output" / "population" / "indices.npz"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "analysis"
MIN_EPOCH_GYR = 1.0
BUMP_PRODUCT_FOR_PLANES = "sigma300"
D4000_SIGMA = 0.05
HDELTA_A_SIGMA = 0.5
BUMP_PRECISIONS = (0.005, 0.010, 0.020)
BUMP_PRODUCTS = ("sigma300", "r100")
AGB_SETTINGS = ("agb2", "agb0")
CLASS_COLORS = {
    STAR_FORMING: "gold",
    TRANSITIONAL: "gray",
    QUIESCENT: "firebrick",
    RAPID_QUENCHING: "deepskyblue",
}
PLOT_ORDER = (STAR_FORMING, TRANSITIONAL, QUIESCENT, RAPID_QUENCHING)
NOISE_SEEDS = (SEED, SEED + 1, SEED + 2)
PURITY_GRID_BINS = 40
PURITY_PERCENTILES = (0.0, 100.0)
PURITY_CLIPPED_PERCENTILES = (0.5, 99.5)
PURITY_GRID_DEFINITION = (
    "40 x 40 cells spanning the full range (min to max over all classes) of each axis; "
    "the `_percentile_clipped` keys use cells between the 0.5 and 99.5 percentiles instead. "
    "Noise-free: a noise-free upper bound on isolability."
)
PURITY_MIN_COUNT = 20
PURITY_ISOLABLE_THRESHOLD = 0.5
CLASSIFIER_NON_RQ_SUBSAMPLE = 100_000


def load_population(input_path=INPUT_PATH, epoch_min=MIN_EPOCH_GYR, stride=1):
    with np.load(input_path) as data:
        table = {key: data[key] for key in data.files}
    mask = table["epoch_gyr"] >= epoch_min
    table = {key: values[mask] for key, values in table.items()}
    if stride > 1:
        table = {key: values[::stride] for key, values in table.items()}
    return table


def plane_indices(table, agb, bump_product=BUMP_PRODUCT_FOR_PLANES):
    return {
        "d4000": table[f"d4000_{agb}"],
        "hdelta_a": table[f"hdelta_a_{agb}"],
        "h_minus_bump": table[f"h_minus_bump_{bump_product}_{agb}"],
    }


def class_summary(codes, post_starburst):
    n = int(codes.size)
    counts = {name: int(np.sum(codes == code)) for code, name in enumerate(CLASS_NAMES)}
    return {
        "n_rows": n,
        "counts": counts,
        "fraction_percent": {name: 100.0 * count / n for name, count in counts.items()},
        "post_starburst_count": int(np.sum(post_starburst & (codes == RAPID_QUENCHING))),
    }


def _class_scatter(axis, x, y, codes):
    for code in PLOT_ORDER:
        mask = codes == code
        is_rapid = code == RAPID_QUENCHING
        axis.scatter(
            x[mask],
            y[mask],
            s=10 if is_rapid else 3,
            color=CLASS_COLORS[code],
            linewidths=0,
            alpha=0.7 if is_rapid else 0.4,
            zorder=4 if is_rapid else 3,
            label=f"{CLASS_NAMES[code]} ({int(mask.sum())})",
        )


def figure_class_planes(table, codes, out_path):
    figure, axes = new_plane_figure(n_rows=2)
    for row, agb in enumerate(AGB_SETTINGS):
        indices = plane_indices(table, agb)
        for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
            axis.hexbin(
                indices[x_key],
                indices[y_key],
                gridsize=60,
                cmap="Greys",
                bins="log",
                mincnt=1,
                alpha=0.35,
                linewidths=0.0,
                zorder=1,
            )
            _class_scatter(axis, indices[x_key], indices[y_key], codes)
        axes[row, 0].set_title(f"{agb}, bump at {BUMP_PRODUCT_FOR_PLANES}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def compute_purity_grid(
    x,
    y,
    rq_mask,
    n_bins=PURITY_GRID_BINS,
    percentiles=PURITY_PERCENTILES,
    min_count=PURITY_MIN_COUNT,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_lo, x_hi = np.percentile(x, percentiles)
    y_lo, y_hi = np.percentile(y, percentiles)
    edges_x = np.linspace(x_lo, x_hi, n_bins + 1)
    edges_y = np.linspace(y_lo, y_hi, n_bins + 1)
    total, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    rq_total, _, _ = np.histogram2d(x[rq_mask], y[rq_mask], bins=[edges_x, edges_y])
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(total >= min_count, rq_total / total, np.nan)
    return purity, edges_x, edges_y, rq_total


def isolable_fraction(x, y, rq_mask, purity, edges_x, edges_y, threshold=PURITY_ISOLABLE_THRESHOLD):
    rq_x = np.asarray(x, dtype=float)[rq_mask]
    rq_y = np.asarray(y, dtype=float)[rq_mask]
    n_rq = rq_mask.sum()
    if n_rq == 0:
        return float("nan")
    n_bins_x, n_bins_y = purity.shape
    ix = np.clip(np.searchsorted(edges_x, rq_x, side="right") - 1, 0, n_bins_x - 1)
    iy = np.clip(np.searchsorted(edges_y, rq_y, side="right") - 1, 0, n_bins_y - 1)
    in_range = (
        (rq_x >= edges_x[0]) & (rq_x <= edges_x[-1]) & (rq_y >= edges_y[0]) & (rq_y <= edges_y[-1])
    )
    isolable = np.zeros(rq_x.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        isolable[in_range] = purity[ix[in_range], iy[in_range]] > threshold
    return float(isolable.sum() / n_rq)


def _purity_map_result(x, y, rq_mask, percentiles):
    purity, edges_x, edges_y, rq_density = compute_purity_grid(
        x, y, rq_mask, percentiles=percentiles
    )
    finite_purity = purity[np.isfinite(purity)]
    result = {
        "max_cell_purity": float(finite_purity.max()) if finite_purity.size else float("nan"),
        "isolable_fraction": isolable_fraction(x, y, rq_mask, purity, edges_x, edges_y),
    }
    return result, purity, edges_x, edges_y, rq_density


def figure_purity_maps(table, codes, out_path):
    """Noise-free purity maps on the full-range grid (drawn and headline), with the
    percentile-clipped grid's numbers recorded alongside as `..._percentile_clipped`."""
    figure, axes = new_plane_figure(n_rows=2)
    results = {}
    rq_mask = codes == RAPID_QUENCHING
    mesh = None
    for row, agb in enumerate(AGB_SETTINGS):
        indices = plane_indices(table, agb)
        results[agb] = {}
        for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
            x, y = indices[x_key], indices[y_key]
            result, purity, edges_x, edges_y, rq_density = _purity_map_result(
                x, y, rq_mask, PURITY_PERCENTILES
            )
            clipped, _, _, _, _ = _purity_map_result(x, y, rq_mask, PURITY_CLIPPED_PERCENTILES)
            result["max_cell_purity_percentile_clipped"] = clipped["max_cell_purity"]
            result["isolable_fraction_percentile_clipped"] = clipped["isolable_fraction"]
            results[agb][f"{x_key}_{y_key}"] = result
            mesh = axis.pcolormesh(
                edges_x, edges_y, purity.T, cmap="viridis", vmin=0.0, vmax=1.0, shading="flat"
            )
            centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
            centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
            if rq_density.max() > 0:
                axis.contour(
                    centers_x, centers_y, rq_density.T, levels=4, colors="white", linewidths=0.8
                )
        axes[row, 0].set_title(
            f"{agb}, bump at {BUMP_PRODUCT_FOR_PLANES}: noise-free upper bound", loc="left"
        )
    figure.colorbar(mesh, ax=list(axes.ravel()), label="rapid-quenching purity", pad=0.02)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return results


def balanced_cross_validated_metrics(
    features,
    labels,
    groups,
    feature_scales,
    k=25,
    n_folds=5,
    seed=SEED,
    positive_class=RAPID_QUENCHING,
):
    """Like `population_classes.cross_validated_metrics`, but within each training fold every
    class is subsampled (without replacement, `seed + fold`) to that fold's rapid-quenching
    count before fitting; evaluation still runs on the full held-out fold."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    folds = grouped_folds(groups, n_folds, np.random.default_rng(seed))

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    completeness_per_fold = []
    purity_per_fold = []

    for fold in range(n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        train_features = features[train_mask]
        train_labels = labels[train_mask]
        rq_count = int(np.sum(train_labels == positive_class))
        fold_rng = np.random.default_rng(seed + fold)
        chosen = []
        for class_code in range(len(CLASS_NAMES)):
            class_indices = np.flatnonzero(train_labels == class_code)
            n_take = min(class_indices.size, rq_count)
            if n_take > 0:
                chosen.append(fold_rng.choice(class_indices, size=n_take, replace=False))
        train_indices = np.concatenate(chosen)
        predictions = knn_predict(
            train_features[train_indices],
            train_labels[train_indices],
            features[test_mask],
            k,
            feature_scales,
        )
        true_fold = labels[test_mask]
        np.add.at(confusion, (true_fold, predictions), 1)
        completeness, purity = completeness_purity(true_fold, predictions, positive_class)
        completeness_per_fold.append(completeness)
        purity_per_fold.append(purity)

    return {
        "completeness_mean": float(np.nanmean(completeness_per_fold)),
        "completeness_std": float(np.nanstd(completeness_per_fold, ddof=0)),
        "purity_mean": float(np.nanmean(purity_per_fold)),
        "purity_std": float(np.nanstd(purity_per_fold, ddof=0)),
        "completeness_per_fold": [float(value) for value in completeness_per_fold],
        "purity_per_fold": [float(value) for value in purity_per_fold],
        "confusion": confusion.tolist(),
    }


def build_feature_sets(table, agb, seed=SEED):
    """Build the 7 feature sets for one agb setting: a single (D4000, HdeltaA) baseline noised
    once, plus (D4000, HdeltaA, bump) for each bump product at each of the three precisions,
    each reusing that same noisy (D4000, HdeltaA) draw and adding its own bump noise. The
    optical pair and every bump product/precision draw from independent streams spawned from
    `np.random.SeedSequence(seed)`."""
    streams = np.random.SeedSequence(seed).spawn(1 + len(BUMP_PRODUCTS) * len(BUMP_PRECISIONS))
    d4000 = table[f"d4000_{agb}"]
    hdelta_a = table[f"hdelta_a_{agb}"]
    noisy_d4000_hdelta = add_measurement_noise(
        np.column_stack([d4000, hdelta_a]),
        np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        np.random.default_rng(streams[0]),
    )
    feature_sets = {
        "no_bump": {
            "features": noisy_d4000_hdelta,
            "scales": np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        }
    }
    bump_streams = iter(streams[1:])
    for product in BUMP_PRODUCTS:
        bump = table[f"h_minus_bump_{product}_{agb}"]
        for precision in BUMP_PRECISIONS:
            noisy_bump = add_measurement_noise(
                bump[:, None], np.array([precision]), np.random.default_rng(next(bump_streams))
            )[:, 0]
            key = f"bump_{product}_{precision:.3f}"
            feature_sets[key] = {
                "features": np.column_stack([noisy_d4000_hdelta, noisy_bump]),
                "scales": np.array([D4000_SIGMA, HDELTA_A_SIGMA, precision]),
            }
    return feature_sets


def paired_difference(result, baseline, n_folds):
    """Mean, standard error (ddof = 1 fold-to-fold std / sqrt(n_folds)) and their ratio of the
    per-fold difference `result - baseline`; both were evaluated on the same folds."""
    out = {}
    for metric in ("completeness", "purity"):
        diff = np.asarray(result[f"{metric}_per_fold"]) - np.asarray(baseline[f"{metric}_per_fold"])
        mean = float(np.nanmean(diff))
        standard_error = float(np.nanstd(diff, ddof=1) / np.sqrt(n_folds))
        out[f"{metric}_gain_mean"] = mean
        out[f"{metric}_gain_standard_error"] = standard_error
        out[f"{metric}_gain_over_standard_error"] = (
            mean / standard_error if standard_error > 0 else float("nan")
        )
    return out


def run_classifier(table, codes, agb, n_folds=5, seed=SEED):
    """One classifier sweep: `seed` sets both the measurement noise and the fold assignment,
    so every feature set of one sweep shares the same folds and the gains are paired."""
    groups = table["history_id"]
    feature_sets = build_feature_sets(table, agb, seed=seed)
    results = {}
    for key, spec in feature_sets.items():
        unbalanced = cross_validated_metrics(
            spec["features"],
            codes,
            groups,
            spec["scales"],
            k=25,
            n_folds=n_folds,
            seed=seed,
            positive_class=RAPID_QUENCHING,
        )
        balanced = balanced_cross_validated_metrics(
            spec["features"],
            codes,
            groups,
            spec["scales"],
            k=25,
            n_folds=n_folds,
            seed=seed,
            positive_class=RAPID_QUENCHING,
        )
        results[key] = {"unbalanced": unbalanced, "balanced": balanced}
    for key in results:
        if key == "no_bump":
            continue
        for balance in ("unbalanced", "balanced"):
            results[key][balance]["paired_difference_vs_no_bump"] = paired_difference(
                results[key][balance], results["no_bump"][balance], n_folds
            )
    return results


ACROSS_SEED_METRICS = ("completeness_mean", "completeness_std", "purity_mean", "purity_std")
ACROSS_SEED_GAIN_METRICS = (
    "completeness_gain_mean",
    "completeness_gain_over_standard_error",
    "purity_gain_mean",
    "purity_gain_over_standard_error",
)


def across_seed_statistics(results_by_seed):
    """Mean and standard deviation (ddof = 1) across noise seeds of the headline metrics."""
    seeds = list(results_by_seed)
    first = results_by_seed[seeds[0]]
    out = {}
    for agb, sets in first.items():
        out[agb] = {}
        for key, balances in sets.items():
            out[agb][key] = {}
            for balance, metrics in balances.items():
                entry = {}
                for metric in ACROSS_SEED_METRICS:
                    values = [results_by_seed[seed][agb][key][balance][metric] for seed in seeds]
                    entry[f"{metric}_mean_over_seeds"] = float(np.mean(values))
                    entry[f"{metric}_std_over_seeds"] = float(np.std(values, ddof=1))
                if "paired_difference_vs_no_bump" in metrics:
                    for metric in ACROSS_SEED_GAIN_METRICS:
                        values = [
                            results_by_seed[seed][agb][key][balance][
                                "paired_difference_vs_no_bump"
                            ][metric]
                            for seed in seeds
                        ]
                        entry[f"{metric}_mean_over_seeds"] = float(np.mean(values))
                        entry[f"{metric}_std_over_seeds"] = float(np.std(values, ddof=1))
                out[agb][key][balance] = entry
    return out


def bump_product_difference(table):
    """How much the sigma300 and r100 bump differ epoch by epoch, per agb setting."""
    out = {}
    for agb in AGB_SETTINGS:
        diff = table[f"h_minus_bump_sigma300_{agb}"] - table[f"h_minus_bump_r100_{agb}"]
        out[agb] = {
            "max_abs_difference_mag": float(np.max(np.abs(diff))),
            "p99_abs_difference_mag": float(np.percentile(np.abs(diff), 99)),
            "median_difference_mag": float(np.median(diff)),
            "std_difference_mag": float(np.std(diff)),
        }
    return out


def subsample_for_classifier(table, codes, target_non_rq=CLASSIFIER_NON_RQ_SUBSAMPLE, seed=SEED):
    rq_mask = codes == RAPID_QUENCHING
    non_rq_indices = np.flatnonzero(~rq_mask)
    if non_rq_indices.size <= target_non_rq:
        return table, codes, False
    rng = np.random.default_rng(seed)
    keep_non_rq = rng.choice(non_rq_indices, size=target_non_rq, replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(rq_mask), keep_non_rq]))
    subsampled_table = {key: values[keep] for key, values in table.items()}
    return subsampled_table, codes[keep], True


def figure_classifier(across_seeds, out_path, n_seeds):
    """Across-seed mean of each metric, with bars and band = across-seed mean of the
    fold-to-fold std."""
    figure, axes = plt.subplots(4, 2, figsize=(11, 17), layout="constrained", sharex=True)
    figure.suptitle(
        "rapid-quenching classifier metrics vs bump measurement precision\n"
        f"mean over {n_seeds} noise seeds; bars: fold-to-fold std"
    )
    row_specs = [
        ("agb2", "unbalanced"),
        ("agb2", "balanced"),
        ("agb0", "unbalanced"),
        ("agb0", "balanced"),
    ]
    metrics = [
        ("completeness_mean", "completeness_std", "completeness"),
        ("purity_mean", "purity_std", "purity"),
    ]
    colors = {"sigma300": "tab:blue", "r100": "tab:orange"}
    precisions = np.array(BUMP_PRECISIONS)
    for row, (agb, balance) in enumerate(row_specs):
        agb_results = across_seeds[agb]
        baseline = agb_results["no_bump"][balance]
        for col, (mean_key, std_key, label) in enumerate(metrics):
            axis = axes[row, col]
            base_mean = baseline[f"{mean_key}_mean_over_seeds"]
            base_std = baseline[f"{std_key}_mean_over_seeds"]
            axis.axhline(base_mean, color="black", linestyle="--", label="no bump", zorder=1)
            axis.axhspan(
                base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, zorder=0
            )
            for product in BUMP_PRODUCTS:
                entries = [agb_results[f"bump_{product}_{p:.3f}"][balance] for p in precisions]
                means = np.array([entry[f"{mean_key}_mean_over_seeds"] for entry in entries])
                stds = np.array([entry[f"{std_key}_mean_over_seeds"] for entry in entries])
                axis.errorbar(
                    precisions,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    color=colors[product],
                    label=f"+bump {product}",
                )
            axis.set_ylim(-0.02, 1.02)
            axis.set_ylabel(label)
            axis.set_title(f"{agb}, {balance}", loc="left", pad=8)
            if row == len(row_specs) - 1:
                axis.set_xlabel("bump precision [mag]")
            else:
                axis.tick_params(labelbottom=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def _to_native(obj):
    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Q2: isolating fast quenching.")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument(
        "--population-dir",
        default=None,
        help="directory holding indices.npz; overrides --input when given",
    )
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--pilot", action="store_true", help="1/50 rows, 2 folds, timing only")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    prefix = args.out_prefix
    input_path = (
        Path(args.population_dir) / "indices.npz" if args.population_dir else Path(args.input)
    )

    start = time.perf_counter()
    table = load_population(input_path, stride=50 if args.pilot else 1)
    codes = assign_classes(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    post_starburst = is_post_starburst(table["ssfr_0_100_myr"], table["ssfr_100_1000_myr"])
    classes = class_summary(codes, post_starburst)
    print(f"n_rows={classes['n_rows']} counts={classes['counts']}")

    if args.pilot:
        classifier_table, classifier_codes, subsampled = subsample_for_classifier(table, codes)
        t0 = time.perf_counter()
        for agb in AGB_SETTINGS:
            run_classifier(classifier_table, classifier_codes, agb, n_folds=2)
        classifier_elapsed = time.perf_counter() - t0
        n_folds_full = 5
        n_feature_sets = 1 + len(BUMP_PRODUCTS) * len(BUMP_PRECISIONS)
        n_runs_pilot = len(AGB_SETTINGS) * n_feature_sets * 2 * 2
        n_runs_full = len(NOISE_SEEDS) * len(AGB_SETTINGS) * n_feature_sets * 2 * n_folds_full
        per_run_fold = classifier_elapsed / n_runs_pilot
        projected_minutes = per_run_fold * n_runs_full / 60.0
        print(f"pilot classifier: {classifier_elapsed:.2f} s for {n_runs_pilot} fold-runs")
        print(f"projected full classifier time: {projected_minutes:.1f} min")
        print(f"pilot total: {time.perf_counter() - start:.2f} s")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    figure_class_planes(table, codes, out_dir / f"{prefix}q2_class_planes.png")
    t_class_planes = time.perf_counter() - t0

    t0 = time.perf_counter()
    purity_results = figure_purity_maps(table, codes, out_dir / f"{prefix}q2_purity_maps.png")
    t_purity_maps = time.perf_counter() - t0

    n_feature_sets = 1 + len(BUMP_PRODUCTS) * len(BUMP_PRECISIONS)
    n_runs_full = len(NOISE_SEEDS) * len(AGB_SETTINGS) * n_feature_sets * 2
    probe_spec = build_feature_sets(table, "agb2")["no_bump"]
    t0 = time.perf_counter()
    cross_validated_metrics(
        probe_spec["features"],
        codes,
        table["history_id"],
        probe_spec["scales"],
        k=25,
        n_folds=5,
        seed=SEED,
        positive_class=RAPID_QUENCHING,
    )
    probe_elapsed = time.perf_counter() - t0
    projected_minutes = probe_elapsed * n_runs_full / 60.0
    print(
        f"classifier timing probe: {probe_elapsed:.2f} s/run, projected {projected_minutes:.1f} min"
    )
    subsampled = projected_minutes > 30.0
    if subsampled:
        classifier_table, classifier_codes, subsampled = subsample_for_classifier(table, codes)
        print(
            f"projected time exceeds 30 min: subsampled classifier table to {classifier_codes.size}"
        )
    else:
        classifier_table, classifier_codes = table, codes

    t0 = time.perf_counter()
    results_by_seed = {
        str(seed): {
            agb: run_classifier(classifier_table, classifier_codes, agb, n_folds=5, seed=seed)
            for agb in AGB_SETTINGS
        }
        for seed in NOISE_SEEDS
    }
    t_classifier = time.perf_counter() - t0
    across_seeds = across_seed_statistics(results_by_seed)
    figure_classifier(across_seeds, out_dir / f"{prefix}q2_classifier.png", len(NOISE_SEEDS))

    elapsed_total = time.perf_counter() - start
    summary = {
        "class_summary": classes,
        "purity_grid_definition": PURITY_GRID_DEFINITION,
        "purity_maps": purity_results,
        "bump_product_difference_sigma300_minus_r100": bump_product_difference(table),
        "classifier": {
            "subsampled_non_rapid_quenching": subsampled,
            "n_rows_used": int(classifier_codes.size),
            "timing_probe_s_per_run": probe_elapsed,
            "projected_minutes_without_subsampling": projected_minutes,
            "noise_seeds": list(NOISE_SEEDS),
            "seed_definition": "each seed sets the measurement noise (independent SeedSequence "
            "streams for the optical pair and every bump product/precision) and the grouped "
            "fold assignment",
            "paired_difference_definition": "per-fold (bump set - no_bump) on the same folds; "
            "standard error = ddof 1 std over the 5 folds / sqrt(5)",
            "across_seeds_definition": "mean and ddof 1 std over the noise seeds of each "
            "seed's fold-mean metric or paired gain",
            "across_seeds": across_seeds,
            "results_by_seed": results_by_seed,
        },
        "timing_s": {
            "class_planes_figure": t_class_planes,
            "purity_maps_figure": t_purity_maps,
            "classifier": t_classifier,
            "total": elapsed_total,
        },
    }
    summary_path = out_dir / f"{prefix}q2_summary.json"
    summary_path.write_text(json.dumps(_to_native(summary), indent=2) + "\n")
    print(f"wrote {out_dir} in {elapsed_total:.1f} s")


if __name__ == "__main__":
    main()
