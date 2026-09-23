# FSPS AGB / TP-AGB Experiment

Sandbox to generate single stellar population (SSP) spectra with `python-fsps`
under different **AGB / TP-AGB** assumptions, and compare their rest-frame NIR
spectra around 1.6 micron (1.4–1.8 micron).

## Setup

- `python-fsps` 0.5.1.dev0 (built from source), reading data from
  `$SPS_HOME=/Users/shuang/code/fsps` (FSPS v4.0).
- Compiled configuration: **MIST** isochrones + **C3K** (high-res, `c3k_hr`)
  spectra + Draine & Li 2007 dust (see `StellarPopulation.libraries`).
- The low-res (`c3k_lr`) results are kept in `output/ssp_spectra.npz`; the
  high-res results in `output/ssp_spectra_hr.npz`.

To rebuild with the other C3K resolution, pass `-DC3K_LR=0 -DC3K_HR=1` (or the
inverse) as Fortran compile definitions in `python-fsps`'s `CMakeLists.txt`;
`pip install -U fsps` restores the PyPI `c3k_lr` wheel.

## Phase 2 setup

    uv sync                     # installs numpy/scipy/matplotlib and the local fsps wheel from wheels/
    export SPS_HOME=/Users/shuang/code/fsps
    uv run pytest               # fast tests; add `-m slow` for the FSPS-dependent ones
    uv run pre-commit install

The wheel in `wheels/` is a local build artifact, not published to PyPI. It is built from
python-fsps 7d202b8 with the FSPS submodule (`$SPS_HOME`) at bd187a0, with the two patches in
`docs/patches/` applied: `c3k_hr_nzinit_13.patch` (`$SPS_HOME/src/sps_vars.f90` and the
`python-fsps` `libfsps` submodule) and `python_fsps_cmake_c3k_hr.patch`
(`python-fsps/src/fsps/CMakeLists.txt`, adding the `-cpp -DC3K_LR=0 -DC3K_HR=1` Fortran compile
options). Exact commands (docs/lessons.md, Task 1):

    cd /Users/shuang/code/python-fsps
    git apply /path/to/docs/patches/python_fsps_cmake_c3k_hr.patch
    # apply docs/patches/c3k_hr_nzinit_13.patch to sps_vars.f90 in $SPS_HOME and in
    # src/fsps/libfsps, then reset libfsps to v4.0 first if needed
    FC=/opt/homebrew/bin/gfortran uv build --wheel --python 3.12 --out-dir <repo>/wheels .

`pip install -U fsps` restores the PyPI `c3k_lr` wheel instead.

## Files

| File | Purpose |
| ---- | ------- |
| `agb_experiment.py` | Generate SSP spectra for a grid of AGB scenarios + IMFs (`--out`, `--spec-label`). |
| `plot_nir_comparison.py` | SSP figures: broad SED + two NIR zooms (norm₁, norm₂) (`--in`, `--spec-label`). |
| `inspect_agb_templates.py` | AGB-template figures (`Orich.spec`, `Crich_Aringer.spec`). |
| `agb_pagb_response.py` | Response of the SSP spectrum to the `agb`/`pagb` weights. |
| `norm_utils.py` | Shared normalization helpers and plot style. |
| `docs/SPEC.md` | Configuration, parameters, mechanism. |
| `docs/todo.md` | Task tracking. |
| `docs/lessons.md` | Lessons learned. |
| `output/ssp_spectra.npz` / `ssp_spectra_hr.npz` | Saved spectra (LR / HR). |
| `output/nir_comparison_{lr,hr}_*.png` | SSP comparison figures. |

## The AGB / TP-AGB knobs (for MIST)

From `fsps/src/{sps_vars,mod_gb,getspec}.f90` and the `python-fsps` docstrings:

| Parameter | Default | Effect |
| --------- | ------- | ------ |
| `agb` | 1.0 | Weight of TP-AGB stars (phase=5). **Dominant NIR knob.** |
| `pagb` | 1.0 | Weight of post-AGB stars (phase=6, Rauch 2003 spectra). |
| `add_agb_dust_model` | True | AGB circumstellar dust (Villaume et al. 2014). |
| `agb_dust` | 1.0 | Scales the AGB dust emission. |
| `use_lw_tpagb` | 0 | O-rich TP-AGB template: 0 = C3K grid, 1 = Lancon & Mouhcine (2002). |
| `tpagb_norm_type` | 2 | **Inert for MIST** (guarded by `isoc_type == 'pdva'`). |
| `fcstar` | 1.0 | **Currently inert** (dilution code commented out in `ssp_gen.f90`). |
| `redgb`, `dell`, `delt` | 1.0 / 0 / 0 | RGB weight and TP-AGB logL/logT shifts (Padova-specific). |

C-rich TP-AGB stars always use Aringer et al. (2009) templates
(`Crich_Aringer.spec`, since `cstar_aringer=1` at compile time).

### The "LW02 O-rich TP-AGB" model

The `lw02_o_rich` scenario sets only **`use_lw_tpagb = 1`**; every other AGB
knob stays fiducial:

| Parameter | Value |
| --------- | ----- |
| `use_lw_tpagb` | **1** (use Lancon & Mouhcine 2002 O-rich spectra) |
| `agb` | 1.0 (TP-AGB weight) |
| `pagb` | 1.0 (post-AGB weight) |
| `add_agb_dust_model` | True (AGB circumstellar dust on) |
| `agb_dust` | 1.0 (dust scaling) |
| `tpagb_norm_type` | 2 (inert for MIST) |
| `fcstar` | 1.0 (inert) |
| `redgb` / `dell` / `delt` | 1.0 / 0 / 0 (fiducial) |

Effect: O-rich TP-AGB stars (`phase=5`, `logT < 3.6`, C/O ≤ 1) are assigned the
LW02 empirical templates (`Orich.spec`, 9 solar Teff bins ≈ 2457–3944 K) instead
of the C3K grid. C-rich TP-AGB stars are unaffected (they always use Aringer
2009 via `cstar_aringer=1`).

## Figure conventions

Each figure has three stacked panels:

1. **Broad SED** (linear wavelength, log flux) for context; the x-axis spans
   270–1900 nm (SSP) or starts at the templates' blue-end cut-off (~352 nm for
   the AGB templates);
2. **NIR zoom** normalized by the median flux in the blue window
   **1495–1535 nm** (`norm_1`);
3. **NIR zoom** normalized by a straight line through the median fluxes of the
   blue (**1495–1535 nm**) and red (**1750–1795 nm**) windows (`norm_2`, a
   Lick/IDS-style pseudo-continuum).

The zoom panels only plot points within 1400–1800 nm so the continuum
extrapolation cannot distort the y-axis. The same rules apply to the AGB
templates and the SSP models. The blue and red normalization windows are shaded
on the zoom panels. SSP figure headers state the isochrone, IMF, stellar
library, age and metallicity; the legend states the AGB setup of each curve.
AGB-template line colors follow the Teff sequence (blue = cold, red = hot).

## Results (1 Gyr, solar Z, Kroupa IMF)

Relative NIR (1.4–1.8 µm) flux vs fiducial:

| Scenario | Change in NIR F_nu | log Lbol | Notes |
| -------- | ------------------ | -------- | ----- |
| `no_tpagb` (`agb=0`) | **−31%** | −0.07 dex | TP-AGB dominates NIR light. |
| `double_tpagb` (`agb=2`) | **+31%** | +0.06 dex | | 
| `no_pagb` (`pagb=0`) | <0.1% | 0.00 | post-AGB is hot → negligible NIR. |
| `no_agb_dust` | +0.4% (NIR), −9% (5.7–10 µm) | 0.00 | dust absorbs in NIR, re-emits in mid-IR. |
| `lw02_o_rich` (`use_lw_tpagb=1`) | −3.4% | 0.00 | different O-rich SED shape. |
| `norm_type_0/1` | 0% | 0.00 | confirms `tpagb_norm_type` inert for MIST. |

Chabrier IMF gives the same qualitative behavior (slightly higher Lbol).

The fiducial NIR spectrum shows the expected **1.6 µm bump** (H⁻ opacity
minimum): F_nu rises to a peak near 1.54–1.63 µm before declining.

## Known limitation

The low-res (`c3k_lr`) SSP output has R=100 at 1.6 µm (~80 Å/pixel), smoothing
the CO-band structure present in the native AGB templates. The high-res
(`c3k_hr`) rebuild has R=500 at 1.6 µm (~16 Å/pixel, 251 vs 50 points across
1.4–1.8 µm) and resolves the 1.6 µm bump peak better (norm₂ peak ≈ 1.08 vs
≈ 1.05 for the fiducial).

## Phase 2 — CSP index tracks and populations

Delayed-tau-plus-quench composite stellar population (CSP) spectra, integrated with an in-house
numpy integrator from cached FSPS SSP grids (cross-checked against FSPS's own tabular SFH), and
three spectral indices (D4000, HdeltaA, the 1.6 µm H-minus bump) tracked over cosmic time for a
fiducial quenching history and a population of 2000 randomly drawn ones. See `docs/SPEC.md`
("Phase 2" section) for the full spec and `docs/superpowers/plans/2026-09-24-csp-index-tracks.md`
for the implementation plan.

### Modules

| File | Purpose |
| ---- | ------- |
| `sfh_model.py` | Delayed-tau plus exponential quench SFH, bin edges, analytic bin masses, prior draws. |
| `ssp_grid.py` | Build SSPs with FSPS, cache to `output/ssp_grid/`, load as `SspGrid`. |
| `broadening.py` | Log-wavelength grid, resampling, quadrature-corrected Gaussian smoothing, resolution products. |
| `csp_integrate.py` | Age-interpolation weights (log-spaced lookback sub-grid), log-Z interpolation, CSP matrix product. |
| `spectral_indices.py` | Air-to-vacuum conversion, band means, D4000, HdeltaA, H-minus bump. |
| `cross_check_fsps_tabular.py` | Cross-check the integrator against FSPS's own tabular SFH (`sfh=3`) at a handful of epochs. |
| `index_planes.py` | Shared figure helpers for the three 2-D index planes. |
| `run_single_csp.py` | Step 1 driver: fiducial CSP track, index table, FSPS cross-check, figures. |
| `run_population.py` | Step 2 driver: population of quenching histories, index table (sSFR normalized by mass formed by `t_obs`, not surviving stellar mass — no return fraction), figures. |
| `tests/test_*.py` | Unit tests per module; FSPS-dependent tests marked `slow`. |

### Running

    uv sync
    export SPS_HOME=/Users/shuang/code/fsps
    uv run python ssp_grid.py
    uv run python broadening.py
    uv run python cross_check_fsps_tabular.py
    uv run python run_single_csp.py --pilot
    uv run python run_single_csp.py
    uv run python run_population.py --pilot
    uv run python run_population.py

### Results

FSPS tabular-SFH cross-check, maximum relative flux difference inside any of the three index
windows, fiducial history (`t_q=3.0 Gyr`, `tau_q=0.3 Gyr`, solar Z), from
`output/single_csp/fsps_cross_check.json`:

| Epoch (Gyr) | Max relative flux difference |
| ----------- | ----------------------------- |
| 1.00  | 0.312% (D4000) |
| 3.00  | 0.053% (D4000) |
| 3.50  | 0.197% (D4000) |
| 5.00  | 0.065% (D4000) |
| 8.00  | 0.022% (D4000) |
| 13.00 | 0.006% (D4000) |

All values are well under the 2% validation threshold (docs/lessons.md, Task 8 fix report).

Timings (this laptop, C3K_HR grid, from docs/lessons.md):

| Step | Time |
| ---- | ---- |
| `ssp_grid.py` (8 SSP builds) | 96.9 s |
| `broadening.py` (`sigma300` + `r100`) | 0.7 s + 1.0 s |
| `cross_check_fsps_tabular.py` (5-epoch slow test, one shared FSPS population build) | 19.0 s |
| `run_single_csp.py` (full, 260 epochs, both products, both `agb` settings, tables + 4 figures) | 7.9 s |
| `run_population.py --pilot` (10 histories x 10 epochs) | 0.2 s, extrapolated to 15.4 min for the full run |
| `run_population.py` (full, 2000 histories x 260 epochs, table computation) | 299-301 s (~5.0 min) |
| `run_population.py` (full, including writing output and figures) | 322-334 s (~5.5 min) |

Index ranges over the fiducial CSP track (260 epochs, `output/single_csp/indices.csv`):

| Index | `agb0` range | `agb2` range |
| ----- | ------------ | ------------ |
| D4000 (`sigma300`) | 1.020 – 2.335 | 1.020 – 2.344 |
| HdeltaA (`sigma300`, A) | -4.577 – 6.300 | -4.609 – 6.242 |
| H-minus bump (`sigma300`, mag) | -0.0202 – 0.0112 | -0.0220 – 0.0112 |
| H-minus bump (`r100`, mag) | -0.0209 – 0.0104 | -0.0228 – 0.0104 |

H-minus bump range over the population (2000 histories x 260 epochs, docs/lessons.md, Task 11):

| Product | `agb0` range | `agb2` range |
| ------- | ------------ | ------------ |
| `sigma300` | -0.0239 – +0.0197 mag | -0.0260 – +0.0197 mag |
| `r100` | -0.0245 – +0.0186 mag | -0.0266 – +0.0186 mag |

At 0.5-2 Gyr after quenching, `agb2` is more negative (deeper bump) than `agb0` for 100% of the
population (mean offset -0.0087 mag `sigma300`, -0.0089 mag `r100`), consistent with the fiducial
track's largest `agb2` vs `agb0` bump difference of -0.0111 mag at 3.45 Gyr (0.45 Gyr after the
`t_q=3.0 Gyr` quench).

## Phase 3 — Diagnostic analysis

The written answers are in `docs/ANALYSIS.md`. Scripts (outputs in `output/analysis/`):

| Script | Question | Outputs |
| ------ | -------- | ------- |
| `population_classes.py` | shared: manuscript sSFR classes, noise, grouped folds, kNN | (library) |
| `analysis_agb_separability.py` | Q1: agb 0 vs 2 at SSP, track and population level | `q1_*.png`, `q1_summary.json` |
| `analysis_fast_quenching.py` | Q2: purity maps and noise-aware kNN with and without the bump | `q2_*.png`, `q2_summary.json` |
| `analysis_alternative_sfh.py` | Q2 robustness: bursty and slowly fading SFH families as contaminants | `q2_alternative_sfh.png`, `q2_alternative_summary.json` |

Run order, default C3K TP-AGB templates (after the Phase 2 steps above):

    uv run python analysis_agb_separability.py --pilot
    uv run python analysis_agb_separability.py
    uv run python analysis_fast_quenching.py --pilot
    uv run python analysis_fast_quenching.py
    uv run python analysis_alternative_sfh.py --pilot
    uv run python analysis_alternative_sfh.py

Run order, empirical Lancon & Mouhcine TP-AGB templates (`use_lw_tpagb = 1`), outputs
prefixed `lw02_`:

    uv run python ssp_grid.py --out-dir output/ssp_grid_lw02 --use-lw-tpagb
    uv run python broadening.py --grid-dir output/ssp_grid_lw02
    uv run python run_single_csp.py --grid-dir output/ssp_grid_lw02 --out-dir output/single_csp_lw02
    uv run python run_population.py --grid-dir output/ssp_grid_lw02 --out-dir output/population_lw02
    uv run python analysis_agb_separability.py --grid-dir output/ssp_grid_lw02 \
        --population-dir output/population_lw02 --out-prefix lw02_
    uv run python analysis_fast_quenching.py --population-dir output/population_lw02 --out-prefix lw02_
    uv run python analysis_alternative_sfh.py --grid-dir output/ssp_grid_lw02 \
        --population-dir output/population_lw02 --out-prefix lw02_

`analysis_alternative_sfh.py` caches its traced table as
`<population-dir>/alternative_sfh_indices.npz` (git-ignored); `--reuse` redraws from it in
about 13 s instead of tracing again (about 95 s).

Headline numbers (details and JSON keys in `docs/ANALYSIS.md`):

| Quantity | Default (C3K) | LW02 |
| -------- | ------------- | ---- |
| Max SSP bump delta, agb2 - agb0, solar Z | 0.020 mag at 0.79 Gyr | 0.111 mag at 0.79 Gyr |
| Fiducial-track bump delta (sigma300) | -0.011 mag at 3.45 Gyr | -0.074 mag at 3.65 Gyr |
| Population bump offset at fixed D4000/HdeltaA | 0.003-0.009 mag | 0.032-0.059 mag |
| Offset / pooled 16-84 half-width, max | 1.35 | 1.90 |
| Rapid-quenching isolable fraction, D4000-HdeltaA (agb2) | 0.807 | 0.768 |
| Best bump-plane isolable fraction (agb2, full-range grid) | 0.081 | 0.291 |
| kNN completeness / purity, no bump (agb2) | 0.418 / 0.617 | 0.413 / 0.613 |
| kNN completeness / purity, + bump at 0.01 mag | 0.406 / 0.628 | 0.460 / 0.650 |
| Contaminant epochs in previously pure cells | 0 of 144,600 | 0 of 144,600 |
