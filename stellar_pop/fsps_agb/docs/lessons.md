# Lessons

## 2026-08-31
- The pip-installed `fsps` (0.5.0) is compiled with **MIST** isochrones and the
  **low-resolution C3K** library (`c3k_lr`, 1936 wavelength points, ~80 Å/pix at
  1.6 µm), plus Draine & Li 2007 dust. Confirmed via `StellarPopulation.libraries`.
- Data files (isochrones, spectra) are read at runtime from `$SPS_HOME`
  (`/Users/shuang/code/fsps`); the pip package only ships the compiled Fortran.
- `tpagb_norm_type` only matters for **Padova** isochrones (`mod_gb.f90` guards it
  with `isoc_type == 'pdva'`), so it is inert for MIST.
- `fcstar` dilution is commented out in `ssp_gen.f90` → currently has no effect.
- `agb` (TP-AGB weight), `pagb` (post-AGB weight), `add_agb_dust_model` +
  `agb_dust`, and `use_lw_tpagb` are the meaningful MIST AGB knobs.
- The `python-fsps` `libfsps` submodule was checked out at `e2441ab` (v3.2-41),
  which lacks the `C3K_LR`/`C3K_HR` split; the parent repo records `05b5e550`
  (v4.0), which has it. Rebuilding the wheel therefore required checking out
  v4.0 first.
- gfortran does NOT preprocess lowercase `.f90` by default — the `#ifndef` in
  `sps_vars.f90` needs an explicit `-cpp` flag in the CMake build.
- To build `C3K_HR`: reset `libfsps` to v4.0 and add
  `target_compile_options(_fsps PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp;-DC3K_LR=0;-DC3K_HR=1>)`
  to `src/fsps/CMakeLists.txt`, then `pip wheel .` + install. `pip install -U fsps`
  restores the PyPI `c3k_lr` wheel.
- Integrated quantities (log Lbol, SSP mass) are identical between C3K_LR and
  C3K_HR — only the spectral sampling changes (1936 vs 10992 points; R=100 vs
  R=500 at 1.6 µm).
- `fcstar` is documented as "Currently has no effect" in the FSPS manual
  (`doc/sps.tex`); its only code reference is a commented-out dilution line in
  `ssp_gen.f90`. Verified empirically: `fcstar` = 0.0/0.3/0.5/1.0/1.5 all give
  identical spectra.
- Even if uncommented, `fcstar` only acts on C-rich TP-AGB stars
  (`ffco > 1.0`), and the MIST "Composition" column for `phase=5` is always
  < 1 (C/O = 0.02–0.34, all O-rich) at every metallicity checked. So the C-rich
  (Aringer 2009) TP-AGB branch never activates for MIST — `use_lw_tpagb` only
  swaps the O-rich template (C3K grid ↔ LW02 `Orich.spec`).
- `agb` and `pagb` are pure **multiplicative weights on the IMF weight**
  (`mod_gb.f90`: `wght(i) = wght(i)*agb` for phase=5, `*pagb` for phase=6) —
  they do NOT change stellar logL/logT. Hence the SSP spectrum is **exactly
  linear** in each knob (verified to ~1e-15).
- `agb`: TP-AGB supplies ~30% of the NIR flux at 1.6 µm (0.5% at 400 nm, 35% at
  3 µm). `pagb`: post-AGB contributes <0.5% everywhere (peaks in the UV), so it
  is negligible in practice — as the flat `response_pagb.png` shows.

## 2026-09-24
- C3K_HR native resolution is R=3000 (sigma 42.4 km/s) only over 3000–10000 A;
  from 1 to 2.5 micron it is R=500 (sigma 254.6 km/s) with 16 A pixels, per
  `SPECTRA/C3K/c3k_hr/readme.md` and `c3k_hr.res` (exposed as `sp.resolutions`).
  A "sigma = 300 km/s" NIR product therefore adds only 159 km/s in quadrature.
- The C3K_HR loader reads `nzinit = 11` metallicity files although 13 exist
  (`sps_vars.f90` c3k_hr block, `sps_setup.f90` read loop with `dz` clipped to
  [0, 1]). MIST +0.25 and +0.50 SSPs silently use solar-[Fe/H] spectra.
- No built-in FSPS SFH gives an exponential decline after truncation:
  `sf_trunc` is a hard cut (sfh=4) and `sf_slope` gives a linear ramp (sfh=5).
  Use tabular SFH (`sfh=3`), which interpolates the SFR linearly between nodes
  and is normalized to absolute Msun, not per Msun formed. Set `tage` on a node.
- python-fsps blocks `zcontinuous = 3` (time-varying Z with tabular SFH) with
  an assertion, and `zcontinuous = 2` (MDF) returns zeros because the summation
  loop in `ztinterp.f90` is commented out.
- python-fsps does not expose FSPS's Lick indices (`getindx` only runs for
  `write_compsp = 4`). FSPS's own HdeltaA feature band is 4083.50–4122.25 A and
  Lick bands are converted air -> vacuum (Morton 1991) in `sps_setup.f90`.
- Timings (C3K_HR, this laptop): one SSP metallicity build ~11 s; changing
  `agb`, IMF or `set_lsf` invalidates every cached SSP (~23 s for a
  zcontinuous=1 pair); CSP-only changes (tage, tabular SFH, `sigma_smooth`)
  cost milliseconds. `sigma_smooth` only acts inside
  [`min_wave_smooth`, `max_wave_smooth`] = [1e3, 1e4] A by default.

## 2026-09-24 (Task 1: C3K_HR nzinit fix)
- Fixed the `nzinit=11` bug (see entry above): patched `sps_vars.f90`'s
  `c3k_hr` block (line 323 of both `/Users/shuang/code/fsps` and the
  `python-fsps` `libfsps` submodule checked out at `bd187a0`) to
  `nzinit=13`, leaving the `c3k_lr` block (line 299) untouched. Patch saved
  at `docs/patches/c3k_hr_nzinit_13.patch`. Also had to append
  `target_compile_options(_fsps PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp;-DC3K_LR=0;-DC3K_HR=1>)`
  to `src/fsps/CMakeLists.txt` (after `python_add_library(...)`) so the
  C3K_HR preprocessor branch is actually compiled in.
- Rebuilt the wheel with:
  `FC=/opt/homebrew/bin/gfortran uv build --wheel --python 3.12 --out-dir <repo>/wheels .`
  run from `/Users/shuang/code/python-fsps` (uv 0.10.4). Wheel build took
  ~16 s wall time and produced
  `fsps-0.5.1.dev0+g7d202b8e0.d20260923-cp312-cp312-macosx_15_0_arm64.whl`.
  `uv build` also silently writes a `wheels/.gitignore` containing `*`;
  harmless since the parent `.gitignore` already ignores `wheels/`.
- Verified with `sp.libraries == (b'mist', b'c3k_hr', b'DL07')`. Measured,
  at `tage=1 Gyr`, Chabrier IMF, `zcontinuous=0`, over 3400-22000 A:
  max `|dF/F|` for `zmet=12` old (pre-fix) wheel vs new (post-fix) wheel =
  **0.2151** (supersolar `zmet=12` was silently falling back to a clipped,
  solar-like spectrum before the fix); max `|dF/F|` for `zmet=11` vs
  `zmet=12` on the new wheel = **1.0040** (confirms `zmet=12` now loads its
  own distinct supersolar spectrum rather than reusing `zmet=11`'s).

## 2026-09-24 (Task 5: SSP grid build and cache)
- Full native grid (`uv run python ssp_grid.py`, 8 SSP builds: 4 metallicities
  x 2 AGB weights) took **96.9 s** wall time on this laptop (~12.1 s/build,
  consistent with the ~11 s/build estimate). Cache written to
  `output/ssp_grid/native.npz` (49.9 MB) plus `output/ssp_grid/provenance.json`.
  Loaded shape: `flux_nu.shape == (4, 2, 107, 7263)`, `wave_a.size == 7263`
  (3400-22000 A window on the native C3K_HR grid); `native_sigma_km_s` ranges
  42.44-254.63 km/s, matching the R=3000 (optical) / R=500 (NIR) split
  recorded above.

## 2026-09-24 (Task 6: broadening and resolution products)
- Built both resolution products from the cached native grid
  (`uv run python broadening.py`): `sigma300.npz` in **0.7 s**
  (18655 wavelength points, 3400.0-21988.7 A) and `r100.npz` in **1.0 s**
  (5184 points, 12501.3-20999.3 A), both from `output/ssp_grid/native.npz`.
- Two of the brief's own `tests/test_broadening.py` tests fail against the
  verbatim `broadening.py` code from the brief (diffed byte-identical, not a
  transcription error):
  - `test_log_wavelength_grid_has_constant_velocity_step`: `wave[0]` comes
    back as `3399.999999999999` instead of `>= 3400.0`, a ~3e-13 relative
    `exp(log(x)) != x` float64 roundoff in `log_wavelength_grid`.
  - `test_gaussian_broaden_recovers_quadrature_sum`: `recovered` is `nan`.
    Root cause: `gaussian_broaden` convolves `flux_nu / wave` (a rapidly
    growing function of pixel index) with `mode="nearest"` padding; near the
    domain edges (here 15000 A and 18000 A, ~28500 km/s from the line
    center, far outside the 300 km/s kernel) the constant "nearest" padding
    under-estimates the true declining continuation, producing a tiny
    (~-4e-4) negative "depth" there. The test's second-moment sum weights by
    `velocity**2`, which is huge at the domain edges, so that tiny edge
    artifact flips `sum(depth * velocity**2)` negative and `sqrt` returns
    `nan`. Production use (`_broaden_segment`) pads each segment by 6000
    km/s before convolving and trims the padding afterward, so this edge
    artifact does not reach `make_resolution_product` output (its own test,
    `test_make_resolution_product_shapes_and_ranges`, passes); it only shows
    up in this unit test's unpadded, wide domain. Left both the test and the
    implementation unchanged, per instructions not to alter brief-specified
    test code or logic; reported as a concern for the controller to rule on.
    Follow-up: both tests were later corrected under a controller ruling —
    `test_log_wavelength_grid_has_constant_velocity_step` now allows a
    `1e-6` tolerance on the lower bound and
    `test_gaussian_broaden_recovers_quadrature_sum` restricts its second
    moment to a `|velocity| <= 3000` km/s window — matching the current
    `tests/test_broadening.py`.
- `uv run ruff format` reformatted both new files (long call signatures onto
  multiple lines); reformatting only, no logic or assertion changes.

## 2026-09-24 (Task 10: Step 1 driver — fiducial CSP track)
- Full run of `run_single_csp.py` (fiducial `t_q_gyr=3.0`, `tau_q_gyr=0.3`,
  `log_z=0.0`, 260 epochs, sigma300 + r100, agb0 + agb2, tables + 4 figures)
  took **7.9 s** wall time on this laptop (0.27 s for the index track itself;
  the rest is writing four large spectra `.npz` files and the figures) using
  the cached `output/ssp_grid/{sigma300,r100}.npz` grids.
- Largest `agb2` vs `agb0` H-minus bump difference: **-0.0111 mag** (r100;
  -0.0108 mag on sigma300) at epoch **3.45 Gyr**, i.e. 0.45 Gyr after the
  `t_q=3.0 Gyr` quench — `agb2` is more negative (deeper bump) than `agb0`
  there, as expected from post-quench TP-AGB light domination.
- `new_plane_figure`'s `tight_layout()` runs before `_figure_index_planes`
  adds per-row titles (`axes[row, 0].set_title(...)`); with the default
  `loc="center"`, the long title text on the narrow first-column axis
  overflowed past the left edge of the canvas and was silently clipped by
  `savefig` (e.g. "agb0 (sigma300, bump at sigma300)" rendered as "b0
  (...)"). Re-calling `figure.tight_layout()` after adding the titles makes
  it worse and throws `UserWarning: ... Axes that are not compatible with
  tight_layout`, because `plot_track`'s `figure.colorbar(..., ax=list(axes))`
  already manually shrank the axes grid to make room for the colorbar, and a
  second `tight_layout()` fights that. Fixed by setting the titles with
  `loc="left"` (anchors the text at the axis's left edge instead of
  centering it) and passing `bbox_inches="tight"` to the two
  `index_planes*.png` `savefig` calls; no change to `index_planes.py`.

## 2026-09-24 (Task 11: Step 2 driver — population)
- Pilot (10 histories x 10 epochs, both products, both agb settings, indices on
  the full 18,700-pixel sigma300 and 5,184-pixel r100 spectra) ran in **0.2 s**,
  extrapolating to **15.4 min** for the full 2000-history x 260-epoch run; well
  under the 60-minute threshold, so the Step 4 pixel-slicing optimization was
  not needed.
- Full run (`uv run python run_population.py`, seed 20260924): table
  computation took **299-301 s** (~5.0 min) across two runs; wall time
  including writing `draws.npz`/`indices.npz`/`indices.csv`/`summary.json` and
  both figures was **322-334 s** (~5.5 min). Table size: 2000 histories x 260
  epochs = **520,000 rows**; `indices.npz` is 70.7 MB, `indices.csv` is
  163.0 MB (both git-ignored via `output/population/indices.*`).
- H-minus bump range across the population: sigma300 agb0
  **[-0.0239, +0.0197] mag**, sigma300 agb2 **[-0.0260, +0.0197] mag**, r100
  agb0 **[-0.0245, +0.0186] mag**, r100 agb2 **[-0.0266, +0.0186] mag**. At
  0.5-2 Gyr after quenching, `agb2` is more negative than `agb0` for 100% of
  the population (mean offset -0.0087 mag sigma300, -0.0089 mag r100),
  consistent with the Step 1 fiducial track.
- `plot_population`'s default `loc="best"` legend on the D4000-HdeltaA panel
  placed "fiducial track" at the panel's bottom/right edge, directly
  overlapping the neighboring panel's rotated `H$^-$ bump [mag]` y-axis label
  in the narrow inter-panel gap (both texts became unreadable where they
  crossed). Root cause: the legend is added by `plot_population` after
  `new_plane_figure`'s `tight_layout()` already ran, so the layout does not
  account for it, and the colorbar's manual axes-shrinking (same issue as
  Task 10) rules out re-calling `tight_layout()`. Fixed in `run_population.py`
  only (`index_planes.py` untouched) by forcing
  `axes[row, 0].legend(frameon=False, loc="upper right")` after
  `plot_population`, since the top-right corner of the D4000-HdeltaA panel is
  empty of data points; re-ran the full population after the fix.

## 2026-09-24 (Task 5: LW02 empirical TP-AGB pipeline, use_lw_tpagb = 1)
- Timings: `ssp_grid.py --out-dir output/ssp_grid_lw02 --use-lw-tpagb` (8
  builds) **91.7 s**; `broadening.py --grid-dir output/ssp_grid_lw02`
  **1.6 s** (sigma300 0.7 s + r100 0.9 s); `run_single_csp.py` on the LW02
  grid **1.9 s**; `run_population.py` on the LW02 grid (2000 histories x 260
  epochs, same seed 20260924) **307.0 s** table + **347.0 s** total (~5.8
  min); `analysis_agb_separability.py --grid-dir output/ssp_grid_lw02
  --population-dir output/population_lw02 --out-prefix lw02_` **3.8 s**
  (step 5, the LW02-variant comparison, is correctly skipped since the
  grid provenance's `extra_params.use_lw_tpagb == 1`, with a note written
  into `lw02_q1_summary.json["step5"]` instead); `analysis_fast_quenching.py
  --population-dir output/population_lw02 --out-prefix lw02_` **73.2 s**
  (no subsampling needed, timing probe projected 1.1 min for the 28-run
  sweep). All outputs verified against `uv run pytest -q` (45 passed, 3
  deselected, no regressions) and `ruff check`/`format --check` (clean)
  after adding the new CLI options.
- Largest `agb2` vs `agb0` H-minus bump difference on the LW02 fiducial
  track (`t_q=3.0 Gyr`): **-0.0736 mag** (sigma300; -0.0733 mag r100) at
  epoch **3.65 Gyr**, i.e. 0.65 Gyr after quenching -- about **6.6x** the
  default C3K grid's -0.0111 mag at the same epoch. At the SSP level, solar
  Z, `analysis_agb_separability.py` step 1 measures the peak at
  **-0.1113 mag at 0.79 Gyr** (`lw02_q1_summary.json`
  `step1.max_abs_delta_h_minus_bump_sigma300_mag_by_log_z["+0.00"]`),
  matching the brief's -0.11 mag at 0.8 Gyr expectation.
- LW02 population H-minus bump range (full table, from
  `output/population_lw02/indices.npz`): sigma300 agb0
  **[-0.0239, +0.0197] mag** (same as the default C3K population, since
  `agb=0` has no TP-AGB light and `use_lw_tpagb` only swaps the TP-AGB
  template), sigma300 agb2 **[-0.0990, +0.0197] mag** (vs default
  **[-0.0260, +0.0197] mag** -- about 3.8x deeper on the low end); r100 agb0
  **[-0.0245, +0.0186] mag**, r100 agb2 **[-0.0991, +0.0186] mag** (vs
  default **[-0.0266, +0.0186] mag**).
- Q2 on the LW02 population: class counts are identical to the default
  population (star_forming 102,100; rapid_quenching 4,727; transitional
  110,874; quiescent 264,299 of 482,000 rows at `epoch_gyr >= 1.0`), since
  `assign_classes` depends only on the SFH masses, not on the AGB spectral
  template. Purity maps: agb2 D4000-bump isolable fraction rises to
  **0.134** (vs 0.077 default C3K) though its max cell purity drops to
  0.731 (vs 1.000 default, i.e. a larger but slightly less pure high-purity
  island); agb0 D4000-HdeltaA remains the best plane overall at **0.810**
  isolable (agb2 in the same plane drops slightly to 0.768, vs 0.807
  default, because the LW02 template also perturbs D4000/HdeltaA a little
  relative to the default C3K agb2 spectrum). Classifier (agb2, unbalanced,
  bump_sigma300 @0.010 mag): completeness **0.460 +/- 0.025** vs the
  no-bump baseline **0.413 +/- 0.017** (+2.8 baseline std) and purity
  **0.650 +/- 0.025** vs **0.613 +/- 0.025**; balanced: completeness
  **0.947 +/- 0.010** vs **0.925 +/- 0.007**, purity **0.238 +/- 0.012** vs
  **0.211 +/- 0.012**. Unlike the default C3K grid (Task 3: bump never
  helps beyond fold-to-fold scatter), the LW02 bump does measurably help
  the agb2 classifier at every tested precision; agb0 shows no measurable
  improvement, as expected, since `agb=0` has no TP-AGB light regardless of
  `use_lw_tpagb`.
