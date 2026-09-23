"""Q2 robustness: bursty star-forming and slowly fading SFH families as contaminants.

Two families of histories outside the delayed-tau-plus-quenching prior are traced over
every epoch with the same code path as `run_population._history_indices`, classified
with the same sSFR rules, and added to the delayed-tau population. The purity maps of
`analysis_fast_quenching.py` are recomputed on the same cell grid with the contaminants
included, and the k-nearest-neighbour classifier trained on the delayed-tau population
is asked to label the contaminant epochs.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from analysis_fast_quenching import (
    AGB_SETTINGS,
    BUMP_PRECISIONS,
    BUMP_PRODUCT_FOR_PLANES,
    D4000_SIGMA,
    HDELTA_A_SIGMA,
    PURITY_ISOLABLE_THRESHOLD,
    PURITY_MIN_COUNT,
    PURITY_PERCENTILES,
    build_feature_sets,
    compute_purity_grid,
    isolable_fraction,
    load_population,
    plane_indices,
)
from index_planes import PLANES, new_plane_figure
from population_classes import (
    CLASS_NAMES,
    RAPID_QUENCHING,
    add_measurement_noise,
    assign_classes,
    knn_predict,
)
from run_population import _history_indices
from sfh_model import (
    LOG_Z_MEAN,
    LOG_Z_RANGE,
    LOG_Z_SIGMA,
    T_Q_RANGE_GYR,
    burst_cumulative_mass,
    cumulative_mass,
    time_bin_edges,
)
from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid

SEED = 20260925
N_HISTORIES_PER_FAMILY = 300
PROJECT_DIR = Path(__file__).resolve().parent
POPULATION_DIR = PROJECT_DIR / "output" / "population"
OUTPUT_DIR = PROJECT_DIR / "output" / "analysis"
CONTAMINANT_TABLE_NAME = "alternative_sfh_indices.npz"
MIN_EPOCH_GYR = 1.0
NO_QUENCH_TAU_Q_GYR = 1e6
BURST_TIME_RANGE_GYR = (2.0, 10.0)
BURST_WIDTH_GYR = 0.1
BURST_MASS_FRACTION = 0.1
SLOW_TAU_Q_RANGE_GYR = (3.0, 6.0)
FAMILY_NAMES = ("bursty", "slow_fading")
BURSTY, SLOW_FADING = range(2)
CLASSIFIER_K = 25
CLASSIFIER_SEED = 20260924
CONTAMINANT_NOISE_SEED = SEED
FULL_RANGE_PERCENTILES = (0.0, 100.0)
TIME_SINCE_BURST_COLOR_RANGE_GYR = (-1.0, 3.0)


def draw_families(n_per_family, seed):
    rng = np.random.default_rng(seed)
    lower = (LOG_Z_RANGE[0] - LOG_Z_MEAN) / LOG_Z_SIGMA
    upper = (LOG_Z_RANGE[1] - LOG_Z_MEAN) / LOG_Z_SIGMA

    def log_z():
        return truncnorm.rvs(
            lower, upper, loc=LOG_Z_MEAN, scale=LOG_Z_SIGMA, size=n_per_family, random_state=rng
        )

    bursty = {
        "t_q_gyr": rng.uniform(*T_Q_RANGE_GYR, size=n_per_family),
        "tau_q_gyr": np.full(n_per_family, NO_QUENCH_TAU_Q_GYR),
        "t_burst_gyr": rng.uniform(*BURST_TIME_RANGE_GYR, size=n_per_family),
        "log_z": log_z(),
    }
    slow_fading = {
        "t_q_gyr": rng.uniform(*T_Q_RANGE_GYR, size=n_per_family),
        "tau_q_gyr": 10.0 ** rng.uniform(*np.log10(SLOW_TAU_Q_RANGE_GYR), size=n_per_family),
        "t_burst_gyr": np.full(n_per_family, np.nan),
        "log_z": log_z(),
    }
    return {"bursty": bursty, "slow_fading": slow_fading}


def history_cumulative_mass(family, t_q_gyr, tau_q_gyr, t_burst_gyr):
    base = partial(cumulative_mass, t_q_gyr=t_q_gyr, tau_q_gyr=tau_q_gyr)
    if family == BURSTY:
        return partial(
            burst_cumulative_mass,
            t_burst_gyr=t_burst_gyr,
            width_gyr=BURST_WIDTH_GYR,
            mass_fraction=BURST_MASS_FRACTION,
            base_cumulative=base,
        )
    return base


def trace_families(grids, draws, edges_gyr):
    n_epochs = edges_gyr.size - 1
    columns = {}
    history_id = 0
    for family, name in enumerate(FAMILY_NAMES):
        family_draws = draws[name]
        for i in range(family_draws["t_q_gyr"].size):
            t_q, tau_q = family_draws["t_q_gyr"][i], family_draws["tau_q_gyr"][i]
            t_burst, log_z = family_draws["t_burst_gyr"][i], family_draws["log_z"][i]
            cumulative = history_cumulative_mass(family, t_q, tau_q, t_burst)
            entry = _history_indices(
                grids, t_q, tau_q, log_z, edges_gyr, cumulative_mass_fn=cumulative
            )
            entry["history_id"] = np.full(n_epochs, history_id)
            entry["family"] = np.full(n_epochs, family)
            entry["t_q_gyr"] = np.full(n_epochs, t_q)
            entry["tau_q_gyr"] = np.full(n_epochs, tau_q)
            entry["t_burst_gyr"] = np.full(n_epochs, t_burst)
            entry["log_z"] = np.full(n_epochs, log_z)
            entry["epoch_gyr"] = edges_gyr[1:]
            entry["time_since_burst_gyr"] = edges_gyr[1:] - t_burst
            for key, values in entry.items():
                columns.setdefault(key, []).append(values)
            history_id += 1
    return {key: np.concatenate(values) for key, values in columns.items()}


def contaminant_class_summary(table, codes):
    summary = {}
    for family, name in enumerate(FAMILY_NAMES):
        mask = table["family"] == family
        summary[name] = {
            "n_epochs": int(mask.sum()),
            "counts": {
                class_name: int(np.sum(codes[mask] == code))
                for code, class_name in enumerate(CLASS_NAMES)
            },
        }
    return summary


def _cell_indices(x, y, edges_x, edges_y):
    n_x, n_y = edges_x.size - 1, edges_y.size - 1
    in_range = (x >= edges_x[0]) & (x <= edges_x[-1]) & (y >= edges_y[0]) & (y <= edges_y[-1])
    ix = np.clip(np.searchsorted(edges_x, x, side="right") - 1, 0, n_x - 1)
    iy = np.clip(np.searchsorted(edges_y, y, side="right") - 1, 0, n_y - 1)
    return ix, iy, in_range


def contaminated_purity(
    base_x,
    base_y,
    base_rq,
    extra_x,
    extra_y,
    extra_rq,
    extra_family,
    percentiles=PURITY_PERCENTILES,
):
    """Purity on the delayed-tau grid of `compute_purity_grid` before and after adding the
    contaminant epochs; statistics for the cells that were above the isolable threshold
    before, and the fraction of delayed-tau rapid-quenching epochs outside the grid."""
    before, edges_x, edges_y, _ = compute_purity_grid(
        base_x, base_y, base_rq, percentiles=percentiles
    )
    total_base, _, _ = np.histogram2d(base_x, base_y, bins=[edges_x, edges_y])
    rq_base, _, _ = np.histogram2d(base_x[base_rq], base_y[base_rq], bins=[edges_x, edges_y])
    total_extra, _, _ = np.histogram2d(extra_x, extra_y, bins=[edges_x, edges_y])
    rq_extra, _, _ = np.histogram2d(extra_x[extra_rq], extra_y[extra_rq], bins=[edges_x, edges_y])
    total_after = total_base + total_extra
    with np.errstate(invalid="ignore", divide="ignore"):
        after = np.where(
            total_after >= PURITY_MIN_COUNT, (rq_base + rq_extra) / total_after, np.nan
        )
    with np.errstate(invalid="ignore"):
        pure_before = before > PURITY_ISOLABLE_THRESHOLD
        pure_after = after > PURITY_ISOLABLE_THRESHOLD
    ix, iy, in_range = _cell_indices(extra_x, extra_y, edges_x, edges_y)
    inside = np.zeros(extra_x.shape, dtype=bool)
    inside[in_range] = pure_before[ix[in_range], iy[in_range]]
    n_rq_before = rq_base[pure_before].sum()
    n_total_before = total_base[pure_before].sum()
    n_rq_after = n_rq_before + rq_extra[pure_before].sum()
    n_total_after = total_after[pure_before].sum()
    _, _, base_in_range = _cell_indices(base_x, base_y, edges_x, edges_y)
    result = {
        "rapid_quenching_fraction_outside_grid": float(np.mean(~base_in_range[base_rq])),
        "n_cells_pure_before": int(pure_before.sum()),
        "n_cells_still_pure_after": int((pure_before & pure_after).sum()),
        "pooled_purity_before": float(n_rq_before / n_total_before)
        if n_total_before
        else float("nan"),
        "pooled_purity_after": float(n_rq_after / n_total_after) if n_total_after else float("nan"),
        "min_cell_purity_after_in_pure_before_cells": float(np.nanmin(after[pure_before]))
        if pure_before.any()
        else float("nan"),
        "n_contaminant_epochs_in_pure_before_cells": int(inside.sum()),
        "n_contaminant_rapid_quenching_epochs_in_pure_before_cells": int((inside & extra_rq).sum()),
        "n_contaminant_epochs_in_pure_before_cells_by_family": {
            name: int((inside & (extra_family == family)).sum())
            for family, name in enumerate(FAMILY_NAMES)
        },
        "isolable_fraction_before": isolable_fraction(
            base_x, base_y, base_rq, before, edges_x, edges_y
        ),
        "isolable_fraction_after": isolable_fraction(
            base_x, base_y, base_rq, after, edges_x, edges_y
        ),
    }
    return result, before, after, edges_x, edges_y


def purity_tests(base_table, base_codes, extra_table, extra_codes, percentiles):
    base_rq = base_codes == RAPID_QUENCHING
    extra_rq = extra_codes == RAPID_QUENCHING
    results, grids = {}, {}
    for agb in AGB_SETTINGS:
        base = plane_indices(base_table, agb)
        extra = plane_indices(extra_table, agb)
        results[agb], grids[agb] = {}, {}
        for x_key, y_key in PLANES:
            result, before, after, edges_x, edges_y = contaminated_purity(
                base[x_key],
                base[y_key],
                base_rq,
                extra[x_key],
                extra[y_key],
                extra_rq,
                extra_table["family"],
                percentiles=percentiles,
            )
            results[agb][f"{x_key}_{y_key}"] = result
            grids[agb][f"{x_key}_{y_key}"] = (before, after, edges_x, edges_y)
    return results, grids


def _contaminant_features(table, agb, bump_precision):
    """Noised features laid out as `analysis_fast_quenching.build_feature_sets`, with
    independent seeds for the optical pair and the bump column."""
    optical = add_measurement_noise(
        np.column_stack([table[f"d4000_{agb}"], table[f"hdelta_a_{agb}"]]),
        np.array([D4000_SIGMA, HDELTA_A_SIGMA]),
        np.random.default_rng(CONTAMINANT_NOISE_SEED),
    )
    if bump_precision is None:
        return optical
    bump = add_measurement_noise(
        table[f"h_minus_bump_{BUMP_PRODUCT_FOR_PLANES}_{agb}"][:, None],
        np.array([bump_precision]),
        np.random.default_rng(CONTAMINANT_NOISE_SEED + 1),
    )
    return np.column_stack([optical, bump])


def _balanced_training_indices(labels, rng):
    rq_count = int(np.sum(labels == RAPID_QUENCHING))
    chosen = []
    for code in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(labels == code)
        n_take = min(indices.size, rq_count)
        if n_take > 0:
            chosen.append(rng.choice(indices, size=n_take, replace=False))
    return np.concatenate(chosen)


def classifier_tests(base_table, base_codes, extra_table, extra_codes, q2_summary):
    """Train on the whole delayed-tau population, noised exactly as in
    `analysis_fast_quenching.build_feature_sets`, and label every contaminant epoch
    (noised independently). The pooled rapid-quenching purity combines the
    cross-validated delayed-tau confusion matrix of `q2_summary` with the contaminant
    predictions."""
    feature_sets = {"no_bump": None} | {
        f"bump_{BUMP_PRODUCT_FOR_PLANES}_{precision:.3f}": precision
        for precision in BUMP_PRECISIONS
    }
    extra_rq = extra_codes == RAPID_QUENCHING
    results = {}
    for agb in AGB_SETTINGS:
        results[agb] = {}
        training_sets = build_feature_sets(base_table, agb)
        for key, precision in feature_sets.items():
            train = training_sets[key]["features"]
            scales = training_sets[key]["scales"]
            test = _contaminant_features(extra_table, agb, precision)
            results[agb][key] = {}
            for balance in ("unbalanced", "balanced"):
                if balance == "balanced":
                    keep = _balanced_training_indices(
                        base_codes, np.random.default_rng(CLASSIFIER_SEED)
                    )
                else:
                    keep = np.arange(base_codes.size)
                predicted_rq = (
                    knn_predict(train[keep], base_codes[keep], test, CLASSIFIER_K, scales)
                    == RAPID_QUENCHING
                )
                confusion = np.array(
                    q2_summary["classifier"]["results"][agb][key][balance]["confusion"]
                )
                true_positive = confusion[RAPID_QUENCHING, RAPID_QUENCHING]
                predicted_positive = confusion[:, RAPID_QUENCHING].sum()
                false_alarms = predicted_rq & ~extra_rq
                results[agb][key][balance] = {
                    "n_contaminant_epochs": int(extra_codes.size),
                    "n_contaminant_predicted_rapid_quenching": int(predicted_rq.sum()),
                    "n_contaminant_false_rapid_quenching": int(false_alarms.sum()),
                    "false_rapid_quenching_by_family": {
                        name: int((false_alarms & (extra_table["family"] == family)).sum())
                        for family, name in enumerate(FAMILY_NAMES)
                    },
                    "false_rapid_quenching_rate": float(false_alarms.sum() / extra_codes.size),
                    "pooled_purity_delayed_tau_only": float(true_positive / predicted_positive),
                    "pooled_purity_with_contaminants": float(
                        (true_positive + (predicted_rq & extra_rq).sum())
                        / (predicted_positive + predicted_rq.sum())
                    ),
                }
    return results


def figure_alternative(base_table, base_codes, extra_table, purity_grids, out_path):
    figure, axes = new_plane_figure(n_rows=1 + len(AGB_SETTINGS))
    base = plane_indices(base_table, "agb2")
    extra = plane_indices(extra_table, "agb2")
    rq = base_codes == RAPID_QUENCHING
    bursty = extra_table["family"] == BURSTY
    slow = extra_table["family"] == SLOW_FADING
    scatter = None
    for axis, (x_key, y_key) in zip(axes[0], PLANES, strict=True):
        axis.hexbin(
            base[x_key],
            base[y_key],
            gridsize=60,
            cmap="Greys",
            bins="log",
            mincnt=1,
            alpha=0.3,
            linewidths=0.0,
            zorder=1,
        )
        axis.scatter(
            extra[x_key][slow],
            extra[y_key][slow],
            s=2,
            color="olivedrab",
            alpha=0.3,
            linewidths=0,
            rasterized=True,
            zorder=2,
            label=f"slow fading ({int(slow.sum())})",
        )
        scatter = axis.scatter(
            extra[x_key][bursty],
            extra[y_key][bursty],
            c=extra_table["time_since_burst_gyr"][bursty],
            cmap="plasma",
            vmin=TIME_SINCE_BURST_COLOR_RANGE_GYR[0],
            vmax=TIME_SINCE_BURST_COLOR_RANGE_GYR[1],
            s=2,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            zorder=3,
            label=f"bursty ({int(bursty.sum())})",
        )
        axis.scatter(
            base[x_key][rq],
            base[y_key][rq],
            s=4,
            color="deepskyblue",
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            zorder=4,
            label=f"delayed-tau rapid-quenching ({int(rq.sum())})",
        )
    axes[0, 0].set_title("agb2: contaminants", loc="left")
    figure.colorbar(
        scatter,
        ax=list(axes[0]),
        label=r"$t_{\rm obs} - t_{\rm burst}$ [Gyr]",
        pad=0.02,
        extend="both",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        markerscale=4,
        frameon=False,
        fontsize=9,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    mesh = None
    for row, agb in enumerate(AGB_SETTINGS, start=1):
        for axis, (x_key, y_key) in zip(axes[row], PLANES, strict=True):
            before, after, edges_x, edges_y = purity_grids[agb][f"{x_key}_{y_key}"]
            mesh = axis.pcolormesh(
                edges_x, edges_y, after.T, cmap="viridis", vmin=0.0, vmax=1.0, shading="flat"
            )
            centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
            centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
            pure_before = np.nan_to_num(before, nan=0.0) > PURITY_ISOLABLE_THRESHOLD
            if pure_before.any():
                axis.contour(
                    centers_x,
                    centers_y,
                    pure_before.T.astype(float),
                    levels=[0.5],
                    colors="red",
                    linewidths=0.9,
                )
        axes[row, 0].set_title(f"{agb}: purity, with contaminants", loc="left")
    figure.colorbar(mesh, ax=list(axes[1:].ravel()), label="rapid-quenching purity", pad=0.02)
    figure.suptitle("red outline: purity $>$ 0.5 before adding contaminants", fontsize=11)
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
    parser = argparse.ArgumentParser(description="Q2 robustness: alternative SFH families.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--population-dir", default=str(POPULATION_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--n-histories", type=int, default=N_HISTORIES_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="load the cached contaminant table from --population-dir instead of tracing",
    )
    parser.add_argument(
        "--pilot", action="store_true", help="3 histories per family, 1/50 rows, no figures"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    population_dir = Path(args.population_dir)
    prefix = args.out_prefix
    cache_path = population_dir / CONTAMINANT_TABLE_NAME
    start = time.perf_counter()

    n_per_family = 3 if args.pilot else args.n_histories
    draws = draw_families(n_per_family, args.seed)
    t0 = time.perf_counter()
    if args.reuse and not args.pilot:
        with np.load(cache_path) as data:
            full_table = {key: data[key] for key in data.files}
    else:
        grids = {product: load_ssp_grid(args.grid_dir, product) for product in ("sigma300", "r100")}
        full_table = trace_families(grids, draws, time_bin_edges())
    t_trace = time.perf_counter() - t0
    n_histories = 2 * n_per_family
    print(f"traced {n_histories} histories x 260 epochs in {t_trace:.1f} s")
    if args.pilot:
        projected = t_trace / n_histories * 2 * args.n_histories
        print(f"projected full tracing time: {projected:.1f} s")
    elif not args.reuse:
        np.savez(cache_path, **full_table)

    keep = full_table["epoch_gyr"] >= MIN_EPOCH_GYR
    extra_table = {key: values[keep] for key, values in full_table.items()}
    extra_codes = assign_classes(extra_table["ssfr_0_100_myr"], extra_table["ssfr_100_1000_myr"])
    classes = contaminant_class_summary(extra_table, extra_codes)
    print(json.dumps(classes))

    base_table = load_population(population_dir / "indices.npz", stride=50 if args.pilot else 1)
    base_codes = assign_classes(base_table["ssfr_0_100_myr"], base_table["ssfr_100_1000_myr"])
    t0 = time.perf_counter()
    purity_results, purity_grids = purity_tests(
        base_table, base_codes, extra_table, extra_codes, PURITY_PERCENTILES
    )
    full_range_results, _ = purity_tests(
        base_table, base_codes, extra_table, extra_codes, FULL_RANGE_PERCENTILES
    )
    t_purity = time.perf_counter() - t0
    if args.pilot:
        print(json.dumps(_to_native(purity_results["agb2"]), indent=1))
        print(f"pilot total: {time.perf_counter() - start:.1f} s")
        return

    q2_summary = json.loads((out_dir / f"{prefix}q2_summary.json").read_text())
    t0 = time.perf_counter()
    classifier_results = classifier_tests(
        base_table, base_codes, extra_table, extra_codes, q2_summary
    )
    t_classifier = time.perf_counter() - t0
    t0 = time.perf_counter()
    figure_alternative(
        base_table,
        base_codes,
        extra_table,
        purity_grids,
        out_dir / f"{prefix}q2_alternative_sfh.png",
    )
    t_figure = time.perf_counter() - t0
    elapsed = time.perf_counter() - start
    summary = {
        "grid_dir": str(args.grid_dir),
        "population_dir": str(population_dir),
        "seed": args.seed,
        "n_histories_per_family": n_per_family,
        "n_delayed_tau_epochs": int(base_codes.size),
        "families": {
            "bursty": {
                "tau_q_gyr": NO_QUENCH_TAU_Q_GYR,
                "t_burst_range_gyr": BURST_TIME_RANGE_GYR,
                "burst_width_gyr": BURST_WIDTH_GYR,
                "burst_mass_fraction_of_base_final_mass": BURST_MASS_FRACTION,
            },
            "slow_fading": {"tau_q_range_gyr_log_uniform": SLOW_TAU_Q_RANGE_GYR},
        },
        "contaminant_classes": classes,
        "purity_maps": purity_results,
        "purity_maps_full_range": full_range_results,
        "classifier": {
            "k": CLASSIFIER_K,
            "bump_product": BUMP_PRODUCT_FOR_PLANES,
            "results": classifier_results,
        },
        "timing_s": {
            "trace_or_load": t_trace,
            "purity": t_purity,
            "classifier": t_classifier,
            "figure": t_figure,
            "total": elapsed,
        },
    }
    (out_dir / f"{prefix}q2_alternative_summary.json").write_text(
        json.dumps(_to_native(summary), indent=2) + "\n"
    )
    print(f"wrote {prefix}q2_alternative_* in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
