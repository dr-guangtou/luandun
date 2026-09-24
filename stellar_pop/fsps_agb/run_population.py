"""Step 2: index distributions for a population of quenching histories.

Each drawn history is traced over every epoch. Only index tables are stored.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csp_integrate import (
    agb_two_spectra,
    csp_spectra,
    epoch_weight_matrix,
    epoch_weight_matrix_from_cumulative,
    interpolate_log_z,
    surviving_mass_per_epoch,
)
from index_planes import new_plane_figure, plot_population
from run_single_csp import FIDUCIAL, compute_track_indices
from sfh_model import bin_masses, draw_population, star_formation_rate, time_bin_edges
from spectral_indices import d4000, h_minus_bump, hdelta_a
from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid

N_DRAWS = 2000
SEED = 20260924
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "population"


def specific_sfr_windows(edges_gyr, masses, epoch_index, surviving_mass):
    """sSFR over the last 100 Myr and over 100 to 1000 Myr before the epoch, per Gyr,
    normalized by the surviving stellar mass (living stars plus remnants) at t_obs."""
    t_obs = edges_gyr[epoch_index + 1]
    centers = 0.5 * (edges_gyr[:-1] + edges_gyr[1:])
    lookback = t_obs - centers
    recent = masses[(lookback > 0) & (lookback <= 0.1)].sum() / 0.1
    previous = masses[(lookback > 0.1) & (lookback <= 1.0)].sum() / 0.9
    return recent / surviving_mass, previous / surviving_mass


def _history_indices(grids, t_q_gyr, tau_q_gyr, log_z, edges_gyr, cumulative_mass_fn=None):
    """Indices and sSFR windows at every epoch of one history. By default the history is
    the delayed-tau-plus-quenching SFH of (t_q_gyr, tau_q_gyr); when `cumulative_mass_fn`
    (cumulative mass formed versus time in Gyr) is given it replaces that SFH, t_q_gyr and
    tau_q_gyr are ignored, and `sfr` is the mean SFR over the 0.05 Gyr bin ending at each
    epoch. sSFRs are normalized by the surviving stellar mass from the grids'
    `surviving_mass_fraction` (see `ssp_grid.load_surviving_mass`)."""
    first_grid = next(iter(grids.values()))
    if first_grid.surviving_mass_fraction is None:
        raise ValueError(
            "grid has no surviving_mass_fraction: build surviving_mass.npz with "
            "`ssp_grid.py --surviving-mass`"
        )
    log_age_yr = first_grid.log_age_yr
    if cumulative_mass_fn is None:
        masses = bin_masses(edges_gyr, t_q_gyr, tau_q_gyr)
        weights = epoch_weight_matrix(edges_gyr, t_q_gyr, tau_q_gyr, log_age_yr)
        sfr = star_formation_rate(edges_gyr[1:], t_q_gyr, tau_q_gyr)
    else:
        masses = np.diff(cumulative_mass_fn(edges_gyr))
        weights = epoch_weight_matrix_from_cumulative(edges_gyr, cumulative_mass_fn, log_age_yr)
        sfr = masses / np.diff(edges_gyr)
    formed_mass = weights.sum(axis=1)
    surviving_mass = surviving_mass_per_epoch(
        weights,
        interpolate_log_z(first_grid.surviving_mass_fraction, first_grid.log_z_grid, log_z),
    )
    out = {}
    for product, grid in grids.items():
        flux_agb0 = csp_spectra(
            weights, interpolate_log_z(grid.flux_nu[:, 0], grid.log_z_grid, log_z)
        )
        flux_agb1 = csp_spectra(
            weights, interpolate_log_z(grid.flux_nu[:, 1], grid.log_z_grid, log_z)
        )
        for agb_key, flux in (
            ("agb0", flux_agb0),
            ("agb1", flux_agb1),
            ("agb2", agb_two_spectra(flux_agb0, flux_agb1)),
        ):
            if product == "sigma300":
                out[f"d4000_{agb_key}"] = d4000(grid.wave_a, flux)
                out[f"hdelta_a_{agb_key}"] = hdelta_a(grid.wave_a, flux)
            out[f"h_minus_bump_{product}_{agb_key}"] = h_minus_bump(grid.wave_a, flux)
    n_epochs = edges_gyr.size - 1
    ssfr = np.array(
        [specific_sfr_windows(edges_gyr, masses, k, surviving_mass[k]) for k in range(n_epochs)]
    )
    out["ssfr_0_100_myr"] = ssfr[:, 0]
    out["ssfr_100_1000_myr"] = ssfr[:, 1]
    out["surviving_mass_fraction"] = surviving_mass / formed_mass
    out["sfr"] = sfr
    return out


def population_indices(grids, draws, edges_gyr):
    n_epochs = edges_gyr.size - 1
    n_draws = draws["t_q_gyr"].size
    columns = {}
    for i in range(n_draws):
        entry = _history_indices(
            grids, draws["t_q_gyr"][i], draws["tau_q_gyr"][i], draws["log_z"][i], edges_gyr
        )
        entry["history_id"] = np.full(n_epochs, i)
        entry["t_q_gyr"] = np.full(n_epochs, draws["t_q_gyr"][i])
        entry["tau_q_gyr"] = np.full(n_epochs, draws["tau_q_gyr"][i])
        entry["log_z"] = np.full(n_epochs, draws["log_z"][i])
        entry["epoch_gyr"] = edges_gyr[1:]
        entry["time_since_quenching_gyr"] = edges_gyr[1:] - draws["t_q_gyr"][i]
        for key, values in entry.items():
            columns.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in columns.items()}


def _write_tables(table, draws, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "draws.npz", **draws)
    np.savez(out_dir / "indices.npz", **table)
    with (out_dir / "indices.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table)
        writer.writerows(zip(*table.values(), strict=True))


def _figure_planes(
    table,
    track,
    out_dir,
    bump_product,
    color_key="time_since_quenching_gyr",
    color_label=r"$t_{\rm obs} - t_q$ [Gyr]",
    suffix="",
):
    figure, axes = new_plane_figure(n_rows=2)
    color = table[color_key]
    for row, agb_key in enumerate(("agb0", "agb2")):
        indices = {
            "d4000": table[f"d4000_{agb_key}"],
            "hdelta_a": table[f"hdelta_a_{agb_key}"],
            "h_minus_bump": table[f"h_minus_bump_{bump_product}_{agb_key}"],
        }
        track_indices = dict(track[agb_key]["sigma300"]) | {
            "h_minus_bump": track[agb_key][bump_product]["h_minus_bump"]
        }
        plot_population(axes[row], indices, color, color_label, track_indices)
        # plot_population's default loc="best" legend collides with the neighboring
        # panel's rotated y-axis label in the narrow gap between subplots; the
        # top-right corner of the D4000-HdeltaA panel is empty of data points.
        axes[row, 0].legend(frameon=False, loc="upper right")
        axes[row, 0].set_title(f"{agb_key}, bump at {bump_product}")
    figure.savefig(out_dir / f"index_planes_{bump_product}{suffix}.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Step 2: population of quenching histories.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pilot", action="store_true", help="10 histories, 10 epochs, timing only")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw figures from the existing indices.npz/draws.npz, no table recomputation",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.figures_only:
        start = time.perf_counter()
        with np.load(out_dir / "indices.npz") as data:
            table = {key: data[key] for key in data.files}
        track = compute_track_indices(args.grid_dir, **FIDUCIAL)
        for bump_product in ("sigma300", "r100"):
            _figure_planes(table, track, out_dir, bump_product)
            _figure_planes(
                table,
                track,
                out_dir,
                bump_product,
                color_key="log_z",
                color_label=r"$\log(Z/Z_\odot)$",
                suffix="_logz",
            )
        print(f"redrew figures in {out_dir} in {time.perf_counter() - start:.1f} s")
        return
    start = time.perf_counter()
    grids = {product: load_ssp_grid(args.grid_dir, product) for product in ("sigma300", "r100")}
    edges = time_bin_edges()
    n_draws = args.n_draws
    if args.pilot:
        edges, n_draws = edges[:11], 10
    draws = draw_population(n_draws, args.seed)
    table = population_indices(grids, draws, edges)
    elapsed = time.perf_counter() - start
    print(f"{n_draws} histories x {edges.size - 1} epochs in {elapsed:.1f} s")
    if args.pilot:
        per_history_epoch = elapsed / (n_draws * (edges.size - 1))
        print(f"extrapolated full run: {per_history_epoch * N_DRAWS * 260 / 60:.1f} min")
        return
    _write_tables(table, draws, out_dir)
    track = compute_track_indices(args.grid_dir, **FIDUCIAL)
    for bump_product in ("sigma300", "r100"):
        _figure_planes(table, track, out_dir, bump_product)
        _figure_planes(
            table,
            track,
            out_dir,
            bump_product,
            color_key="log_z",
            color_label=r"$\log(Z/Z_\odot)$",
            suffix="_logz",
        )
    summary = {
        "n_draws": n_draws,
        "seed": args.seed,
        "n_epochs": int(edges.size - 1),
        "elapsed_s": elapsed,
        "grid_provenance": {product: grid.provenance for product, grid in grids.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {out_dir} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
