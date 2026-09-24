"""Step 1: trace the fiducial CSP in time and measure the three indices.

Products: D4000 and HdeltaA on the sigma300 grid; H-minus bump on sigma300
and r100. Both agb = 0 and agb = 2 (from linearity) are traced.
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

from csp_integrate import agb_two_spectra, csp_track
from index_planes import new_plane_figure, plot_track
from sfh_model import star_formation_rate, time_bin_edges
from spectral_indices import H_MINUS_BANDS_A, d4000, flux_nu_to_flux_lambda, h_minus_bump, hdelta_a
from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid

FIDUCIAL = {"t_q_gyr": 3.0, "tau_q_gyr": 0.3, "log_z": 0.0}
AGB_SETTINGS = {"agb0": 0.0, "agb2": 2.0}
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "single_csp"
ZOOM_EPOCHS_GYR = (1.0, 3.0, 3.5, 4.0, 5.0, 8.0, 13.0)


def _agb_spectra(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr):
    epochs, flux_agb0, mass = csp_track(grid, log_z, 0, t_q_gyr, tau_q_gyr, edges_gyr)
    _, flux_agb1, _ = csp_track(grid, log_z, 1, t_q_gyr, tau_q_gyr, edges_gyr)
    return epochs, {"agb0": flux_agb0, "agb2": agb_two_spectra(flux_agb0, flux_agb1)}, mass


def compute_track_indices(grid_dir, t_q_gyr, tau_q_gyr, log_z, edges_gyr=None, keep_spectra=False):
    edges_gyr = time_bin_edges() if edges_gyr is None else edges_gyr
    grids = {product: load_ssp_grid(grid_dir, product) for product in ("sigma300", "r100")}
    result = {
        "epoch_gyr": edges_gyr[1:],
        "sfr": star_formation_rate(edges_gyr[1:], t_q_gyr, tau_q_gyr),
        "parameters": {"t_q_gyr": t_q_gyr, "tau_q_gyr": tau_q_gyr, "log_z": log_z},
    }
    spectra = {}
    for product, grid in grids.items():
        epochs, flux_by_agb, mass = _agb_spectra(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr)
        result["mass_formed"] = mass
        for agb_key, flux in flux_by_agb.items():
            entry = result.setdefault(agb_key, {}).setdefault(product, {})
            if product == "sigma300":
                entry["d4000"] = d4000(grid.wave_a, flux)
                entry["hdelta_a"] = hdelta_a(grid.wave_a, flux)
            entry["h_minus_bump"] = h_minus_bump(grid.wave_a, flux)
            spectra[(agb_key, product)] = (grid.wave_a, flux)
    if keep_spectra:
        result["spectra"] = spectra
    return result


def _write_tables(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = {
        "epoch_gyr": result["epoch_gyr"],
        "sfr": result["sfr"],
        "mass_formed": result["mass_formed"],
    }
    for agb_key in AGB_SETTINGS:
        for product, indices in result[agb_key].items():
            for name, values in indices.items():
                columns[f"{name}_{product}_{agb_key}"] = values
    np.savez(out_dir / "indices.npz", **columns)
    with (out_dir / "indices.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))
    for (agb_key, product), (wave_a, flux) in result["spectra"].items():
        np.savez(
            out_dir / f"spectra_{product}_{agb_key}.npz",
            wave_a=wave_a,
            flux_nu=flux,
            epoch_gyr=result["epoch_gyr"],
        )
    (out_dir / "parameters.json").write_text(json.dumps(result["parameters"], indent=2) + "\n")


def _figure_time_evolution(result, out_dir):
    figure, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
    epochs = result["epoch_gyr"]
    axes[0].plot(epochs, result["sfr"], color="black")
    axes[0].set_ylabel("SFR [arbitrary]")
    axes[0].axvline(result["parameters"]["t_q_gyr"], color="0.6", ls=":")
    for axis, name in zip(axes[1:], ("d4000", "hdelta_a", "h_minus_bump"), strict=True):
        for agb_key, color in (("agb0", "#1f77b4"), ("agb2", "#d62728")):
            axis.plot(
                epochs, result[agb_key]["sigma300"][name], color=color, label=f"{agb_key}, sigma300"
            )
            if name == "h_minus_bump":
                axis.plot(
                    epochs,
                    result[agb_key]["r100"][name],
                    color=color,
                    ls="--",
                    label=f"{agb_key}, R=100",
                )
        axis.set_ylabel(name)
        axis.axvline(result["parameters"]["t_q_gyr"], color="0.6", ls=":")
    axes[3].invert_yaxis()
    axes[3].set_xlabel("time since start of star formation [Gyr]")
    axes[1].legend(frameon=False, fontsize=8)
    axes[3].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "time_evolution.png", dpi=150)
    plt.close(figure)


def _figure_nir_zoom(result, out_dir):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ZOOM_EPOCHS_GYR)))
    for column, agb_key in enumerate(AGB_SETTINGS):
        for row, product in enumerate(("sigma300", "r100")):
            wave_a, flux = result["spectra"][(agb_key, product)]
            axis = axes[row, column]
            zoom = (wave_a >= 14000.0) & (wave_a <= 18500.0)
            for epoch, color in zip(ZOOM_EPOCHS_GYR, colors, strict=True):
                k = int(np.argmin(np.abs(result["epoch_gyr"] - epoch)))
                flux_lambda = flux_nu_to_flux_lambda(wave_a, flux[k])
                axis.plot(
                    wave_a[zoom] / 1e4,
                    flux_lambda[zoom] / np.median(flux_lambda[zoom]),
                    color=color,
                    lw=1.0,
                    label=f"{result['epoch_gyr'][k]:.2f} Gyr",
                )
            for name, color in (("blue", "tab:blue"), ("feature", "tab:green"), ("red", "tab:red")):
                axis.axvspan(
                    H_MINUS_BANDS_A[name][0] / 1e4,
                    H_MINUS_BANDS_A[name][1] / 1e4,
                    color=color,
                    alpha=0.08,
                )
            axis.set_title(f"{agb_key}, {product}")
            axis.set_ylabel(r"normalized $F_\lambda$")
    axes[1, 0].set_xlabel(r"rest wavelength [$\mu$m]")
    axes[1, 1].set_xlabel(r"rest wavelength [$\mu$m]")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(out_dir / "nir_zoom.png", dpi=150)
    plt.close(figure)


def _figure_index_planes(result, out_dir):
    figure, axes = new_plane_figure(n_rows=2)
    for row, agb_key in enumerate(AGB_SETTINGS):
        indices = dict(result[agb_key]["sigma300"])
        plot_track(axes[row], indices, result["epoch_gyr"], "time [Gyr]")
        axes[row, 0].set_title(f"{agb_key} (sigma300, bump at sigma300)", loc="left")
    figure.savefig(out_dir / "index_planes.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    figure, axes = new_plane_figure(n_rows=2)
    for row, agb_key in enumerate(AGB_SETTINGS):
        indices = dict(result[agb_key]["sigma300"]) | {
            "h_minus_bump": result[agb_key]["r100"]["h_minus_bump"]
        }
        plot_track(axes[row], indices, result["epoch_gyr"], "time [Gyr]")
        axes[row, 0].set_title(f"{agb_key} (bump at R=100)", loc="left")
    figure.savefig(out_dir / "index_planes_r100.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Step 1: fiducial CSP index track.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--pilot", action="store_true", help="only 10 epochs, no figures")
    args = parser.parse_args()
    start = time.perf_counter()
    edges = time_bin_edges()
    if args.pilot:
        edges = edges[:11]
    result = compute_track_indices(
        Path(args.grid_dir), keep_spectra=True, edges_gyr=edges, **FIDUCIAL
    )
    print(f"indices for {result['epoch_gyr'].size} epochs in {time.perf_counter() - start:.2f} s")
    if args.pilot:
        return
    out_dir = Path(args.out_dir)
    _write_tables(result, out_dir)
    _figure_time_evolution(result, out_dir)
    _figure_nir_zoom(result, out_dir)
    _figure_index_planes(result, out_dir)
    print(f"wrote {out_dir} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
