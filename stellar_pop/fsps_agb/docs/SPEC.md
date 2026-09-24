# FSPS TP-AGB spectral-index experiment — SPEC

Source of truth for the architecture of this folder. Phase 1 (the finished SSP
sandbox) is kept at the end for reference. Phase 2 is the current work.

## Phase 2 — CSP index tracks and populations (2026-09-24)

### Goal

Quantify how the TP-AGB contribution changes the evolution of three spectral
indices, D4000, HdeltaA and the 1.6 micron H-minus bump, for galaxies that
form stars with a delayed-tau history and then quench exponentially. Two
deliverables:

1. **Step 1, single CSP:** one fiducial history traced in time from the start
   of star formation to 13 Gyr, with the TP-AGB weight `agb` = 0 and 2.
   Output: spectra, an index table, and the track drawn in the three 2-D index
   planes.
2. **Step 2, population:** 2000 histories drawn from prior distributions on
   quenching time, quenching timescale and metallicity, each traced in time
   the same way. Output: index tables and the resulting distributions in the
   three 2-D index planes, one set per `agb` setting.

The manuscript this feeds is `~/Dropbox/Apps/Overleaf/H-bump/main.tex`
(paper I on the H-minus bump as a quenching tracer). Conventions below follow
that paper unless stated.

### Fixed stellar-population configuration

| Setting | Value | Notes |
| ------- | ----- | ----- |
| Isochrones | MIST | compiled in, `zsol = 0.0185`, 107 ages, log(age/yr) 5.00–10.30 in 0.05 dex |
| Spectral library | C3K high-res (`c3k_hr`) | 10992 vacuum wavelengths; R = 3000 (sigma 42.4 km/s) over 3000–10000 A, R = 500 (sigma 254.6 km/s) over 1–2.5 micron |
| IMF | Chabrier (2003), `imf_type = 1` | |
| TP-AGB weight | `agb` = 0 and `agb` = 1 built; `agb` = 2 derived | spectrum is exactly linear in `agb` (Phase 1 lesson), so S(2) = 2 S(1) - S(0) |
| Other AGB knobs | defaults (`pagb` = 1, AGB circumstellar dust on, `use_lw_tpagb` = 0) | |
| Nebular emission, dust attenuation, IGM | off (defaults) | pure stellar continuum |
| Metallicity grid built | log(Z/Zsun) = -0.50, -0.25, 0.00, +0.25 | brackets the prior range [-0.5, +0.2] |

**Required FSPS fix before building.** The C3K_HR loader reads only
`nzinit = 11` metallicity files (`sps_vars.f90`, `c3k_hr` block), so the
MIST +0.25 and +0.50 SSPs currently pair supersolar isochrones with solar
[Fe/H] spectra. Change `nzinit` to 13 in the `c3k_hr` block of
`$SPS_HOME/src/sps_vars.f90`, rebuild the python-fsps wheel with the
`-cpp -DC3K_LR=0 -DC3K_HR=1` definitions (recipe in `docs/lessons.md`), and
verify that the `logzsol = +0.25` spectrum changes relative to the old build.
The patch is kept in `docs/patches/` so the change is reproducible.

### Star formation history

Time `t` is measured from the start of star formation. With `tau = t_q`:

```
SFR(t) = (t / t_q) exp(-t / t_q)                      t <  t_q
SFR(t) = exp(-1) exp(-(t - t_q) / tau_q)              t >= t_q
```

- Time grid: bins of 0.05 Gyr from 0 to 13 Gyr (260 bins). Observation
  epochs `t_obs` are the bin edges 0.05, 0.10, ..., 13.0 Gyr.
- The mass formed in each bin is the analytic integral of SFR over the bin
  (both branches have closed forms), not SFR at the bin center times width.
- Absolute normalization is irrelevant for indices; stored spectra are per
  solar mass formed by `t_obs`.
- No metallicity evolution: each history has a single log(Z/Zsun).

Fiducial history for Step 1: `t_q` = 3 Gyr, `tau_q` = 0.3 Gyr, log(Z/Zsun) = 0.

Priors for Step 2 (seed fixed and recorded in the output):

| Parameter | Distribution | Range |
| --------- | ------------ | ----- |
| `t_q` | uniform | [1, 5.9] Gyr |
| `tau_q` | log-uniform | [0.1, 3] Gyr |
| log(Z/Zsun) | normal, mean 0.0, sigma 0.2 dex, truncated | [-0.5, +0.2] |

Note: the paper's mock table also draws a random `t_obs` per mock; here each
history is traced over all 260 epochs instead, so the two populations are
weighted differently in time.

### CSP assembly (own integrator, FSPS only for SSPs)

FSPS has no built-in SFH with an exponential decline after truncation
(`sf_trunc` is a hard cut, `sf_slope` is linear), and every new metallicity
or `agb` value costs 11–23 s of SSP regeneration. So:

1. `ssp_grid.py` builds the SSPs once with python-fsps (`sfh = 0`,
   `zcontinuous = 0`, `tage = 0` returns all 107 ages) for each of the 4
   metallicities and `agb` in {0, 1}, keeps the 3400 A – 2.2 micron window,
   and caches them with full provenance (library tuple, fsps version, git
   hashes of fsps and python-fsps, every non-default parameter).
2. `broadening.py` smooths the cached SSPs (see next section). Convolution
   and SFH integration are both linear, so smoothing the SSPs once is
   identical to smoothing every CSP.
3. `csp_integrate.py`: for an epoch `t_obs`, the SFH is integrated over
   lookback age on a log-spaced sub-grid (0.01 dex from 10^5 yr to `t_obs`,
   plus the sliver from 0 to 10^5 yr). The mass formed in each sub-bin is the
   analytic integral of the SFR (`sfh_model.cumulative_mass`), and it is
   assigned to the SSP ages by linear interpolation in log age between the
   two bracketing grid ages (ages below 10^5 yr use the youngest SSP),
   matching FSPS's kernel. A fixed 0.05 Gyr bin-center scheme was tried first
   and disagreed with FSPS by up to 45 percent in the D4000 window while
   star formation was ongoing, because stars younger than about 100 Myr
   change their blue flux by large factors within one bin. The metallicity is
   interpolated linearly in log Z between the two bracketing grid SSPs,
   matching `zcontinuous = 1`. The CSP is one matrix product (weights of
   shape epochs x ages, times SSP of shape ages x pixels).
4. `agb` = 2 spectra are formed from the `agb` = 0 and 1 spectra by linearity.

Cross-check (Step 1 only): the same fiducial history is run through FSPS's
tabular SFH (`sfh = 3`, `set_tabular_sfh`, `tage` set to a table node) at a
handful of epochs, and the maximum relative flux difference inside the index
windows plus the index differences are reported in the results.

### Spectral resolution products

All smoothing is done on a uniform log-wavelength grid (30 km/s step) with a
Gaussian in velocity, on lambda F_lambda as in the ProGeny study, after
subtracting the native library resolution in quadrature. The native
resolution is piecewise constant (sigma 42.4 km/s below 10000 A, 254.6 km/s
above), so the two segments are convolved separately with padding; no index
window is within 800 A of the 10000 A splice.

| Product | Range | Added kernel |
| ------- | ----- | ------------ |
| `native` | 3400 A – 2.2 micron | none (reference only) |
| `sigma300` | 3400 A – 2.2 micron | sqrt(300^2 - sigma_lib^2): 297 km/s optical, 159 km/s NIR |
| `r100` | 1.25 – 2.1 micron | sqrt(sigma_R100^2 + 300^2 - 254.6^2), with sigma_R100 = c / (2.3548 x 100) = 1273 km/s (R = 100 defined as FWHM) |

D4000 and HdeltaA are measured on `sigma300` only. The H-minus bump is
measured on `sigma300` and `r100`. Note that in the NIR the `sigma300`
product is close to the native spectrum: the native sigma is already 255
km/s and the pixel is 16 A (300 km/s).

### Index definitions

Wavelengths in the FSPS grid are vacuum. Lick/IDS and D4000 bands are given
in air and are converted with the same Morton (1991) formula FSPS uses
(`vacairconv.f90`). The H-minus bands were defined on model grids and are
used as vacuum values.

| Index | Blue | Feature | Red | Form |
| ----- | ---- | ------- | --- | ---- |
| D4000 | 3750–3950 A (air) | — | 4050–4250 A (air) | ratio of mean F_nu, red over blue (Bruzual 1983 form) |
| HdeltaA | 4041.60–4079.75 A (air) | 4083.50–4122.25 A (air) | 4128.50–4161.00 A (air) | equivalent width in A on F_lambda: integral of (1 - F/F_c) over the feature |
| H-minus bump | 1.494–1.539 micron | 1.570–1.734 micron | 1.746–1.791 micron | magnitude on F_lambda: -2.5 log10 of the mean of F/F_c over the feature; negative for a bump |

`F_c` is the straight line through the band-mean F_lambda of the blue and red
windows, anchored at the band midpoints. Band means use exact band limits
(interpolate at the edges) and Gauss-Legendre quadrature inside pixels, as
in the ProGeny implementation. FSPS output is F_nu (Lsun/Hz); F_lambda is
F_nu c / lambda^2.

### Outputs

- `output/ssp_grid/`: cached SSP grids, one file per (metallicity, agb,
  product), plus a provenance JSON.
- `output/single_csp/`: fiducial spectra at each epoch and product (both
  `agb` settings), the index table, the FSPS cross-check summary, and figures:
  SFH plus the three indices versus time; NIR zooms at selected epochs for
  both products; the three 2-D index planes as time-colored curves.
- `output/population/`: the drawn parameters, the index table (one row per
  history and epoch, wide-format columns for both `agb` settings: `t_q`,
  `tau_q`, `log_z`, `t_obs`, D4000 / HdeltaA / H-minus bump (`sigma300` and
  `r100`) for `agb0`, `agb1` and `agb2`, SFR, sSFR over the last 100 Myr and
  100–1000 Myr for later classification, and `surviving_mass_fraction`;
  git-ignored), and
  figures: the three planes as density plus scatter, fiducial track
  overlaid, one panel per `agb` setting. Spectra are not stored for the
  population.
- sSFR normalization (Phase 4 change): both sSFR windows divide by the
  surviving stellar mass at `t_obs`, living stars plus remnants. `ssp_grid.py
  --surviving-mass` stores FSPS `StellarPopulation.stellar_mass` (`tage = 0`,
  agb = 1, `add_stellar_remnants` at its default of on) per solar mass formed
  for the 4 metallicities x 107 ages in `<grid-dir>/surviving_mass.npz`;
  `load_ssp_grid` attaches it to the grid, and the surviving mass per epoch
  is the epoch weight matrix times this fraction (interpolated in log Z).
  agb = 1 because FSPS rescales the TP-AGB IMF weights by `agb` before
  summing the mass. The ratio R of the two windows is unchanged; the absolute
  thresholds move.

### Validation (all measured, thresholds fixed before running)

- Null tests: a spectrum that is linear in wavelength gives |HdeltaA| and
  |H-minus| below 1e-10 and D4000 equal to its analytic value.
- Broadening: a synthetic Gaussian line of sigma 40 km/s convolved with the
  300 km/s kernel recovers sqrt(40^2 + 300^2) within 1 percent.
- `agb` linearity: S(agb = 2) from FSPS equals 2 S(1) - S(0) to 1e-10
  relative.
- Integrator versus FSPS tabular CSP: relative flux difference inside the
  index windows below 2 percent at every checked epoch; index differences
  reported.
- Scale: a pilot with one metallicity, ten epochs and ten histories is timed
  before the full population; the full run is only launched when the
  extrapolated time is acceptable.

### Code layout

| File | Purpose |
| ---- | ------- |
| `sfh_model.py` | delayed-tau plus exponential quench, bin masses, priors and seeded draws |
| `ssp_grid.py` | build, cache and load FSPS SSP grids with provenance |
| `broadening.py` | log-wavelength resampling and quadrature-corrected Gaussian smoothing |
| `csp_integrate.py` | SSP interpolation in log age and log Z; CSP matrix product |
| `spectral_indices.py` | D4000, HdeltaA, H-minus bump; air-to-vacuum helper |
| `run_single_csp.py` | Step 1 driver and figures |
| `run_population.py` | Step 2 driver and figures |
| `tests/` | unit tests for the validation items above |

Style: `snake_case`, English, no camelCase, uv-managed environment with the
rebuilt python-fsps wheel, Ruff via pre-commit. The Phase 1 scripts are
left untouched.

## Phase 3 — Diagnostic analysis (2026-09-24)

Two questions, each answered with QA figures and measured numbers; a
negative answer is acceptable if the evidence supports it.

### Inputs

The Phase 2 products: the SSP grid (4 Z x 2 agb x 107 ages; `native`,
`sigma300`, `r100`), the fiducial track (Step 1) and the 520,000-row
population table (Step 2) with `agb0` and `agb2` indices, SFR and the two
sSFR windows.

### Measurement yardstick

Three reference precisions for the H-minus index: 0.005, 0.010 and 0.020
mag (the ARP 151 SPHEREx measurement reached 0.018 mag; JWST prism stacks
are at the 0.01 level), with 0.05 in D4000 and 0.5 A in HdeltaA for the
optical indices. Population epochs earlier than 1 Gyr after the start of
star formation are dropped from every population statistic (the discrete
0.05 Gyr epochs make stripes there and no real sample is that young).

### Q1. Can a model with versus without TP-AGB contribution be told apart?

1. SSP level (`analysis_agb_separability.py`): Delta = index(agb = 2) -
   index(agb = 0) versus age for all four metallicities, for the three
   indices at `sigma300` and the bump also at `r100`, drawn next to the
   metallicity spread of the agb = 0 index at fixed age and the yardstick
   lines. Spectral QA: the ratio S(agb = 2) / S(agb = 0) over 1.3-2.0 micron
   at 0.3, 1, 2 and 5 Gyr with the three bump bands shaded, and the
   continuum-normalized NIR spectra themselves, to show that TP-AGB light
   enters mostly as a tilted continuum that the pseudo-continuum removes.
   Broadband contrast: the NIR-to-optical flux ratio F_nu(1.6 micron) /
   F_nu(4200 A) for agb = 0 and 2, to show what a flux-calibrated
   measurement would see instead of the index.
2. CSP level: the fiducial track's Delta versus time since quenching; in
   the population, in bins of D4000 and separately of HdeltaA, the median
   and 16-84 percentiles of the bump for agb0 and agb2 and the offset in
   units of the pooled intrinsic scatter.
   Answer rule: "clearly" means the offset exceeds both the intrinsic
   spread at fixed optical indices and the 0.01 mag precision.

### Q2. With agb = 2, can the fast-quenching population be isolated?

Classes follow the manuscript: R = sSFR(0-100 Myr) / sSFR(100-1000 Myr);
star-forming R > 1; rapid-quenching sSFR(100-1000 Myr) > 1e-10 per yr and
R < 0.1 (post-starburst as its subset with sSFR(0-100 Myr) < 1e-11);
quiescent sSFR(100-1000 Myr) < 1e-10 and sSFR(0-100 Myr) < 1e-11 per yr;
everything else "transitional".

1. Purity maps (`analysis_fast_quenching.py`): 2-D histograms per class
   in the three planes and the fraction of rapid-quenching points per
   cell, for agb2 with agb0 as control.
2. Noise-aware classification: Gaussian noise at the yardstick levels; a
   k-nearest-neighbour classifier (k = 25) with 5-fold cross-validation
   grouped by history; completeness and purity of the rapid-quenching class
   and the confusion matrix for (D4000, HdeltaA) alone versus with the bump
   added, at the three bump precisions. The bump helps only if purity or
   completeness rises beyond the fold-to-fold scatter.
3. Robustness with other SFH families (only if the delayed-tau result is
   marginal, or to test contamination): a bursty star-forming family
   (delayed-tau plus a late Gaussian burst holding 10 percent of the mass,
   100 Myr wide, no quench) and a slowly fading family (tau_q 3-6 Gyr),
   added as contaminants to the purity maps and the classifier.

### Deliverables

Figures and `summary.json` under `output/analysis/`, and `docs/ANALYSIS.md`
with one section per question stating the answer, the numbers and the
figures that support it.

## Phase 4 — Surviving-mass sSFR and conclusion figures (2026-09-24)

### sSFR convention change

Both sSFR windows now normalize by the surviving stellar mass at `t_obs` (living stars
plus remnants) instead of the mass formed. `ssp_grid.py --surviving-mass` stores FSPS
`StellarPopulation.stellar_mass` (`tage = 0`, `agb = 1`, remnants on) per solar mass
formed for the 4 metallicities x 107 ages in `<grid-dir>/surviving_mass.npz`;
`load_ssp_grid` attaches it, and `csp_integrate.surviving_mass_per_epoch` is the epoch
weight matrix times the fraction interpolated in log Z. `agb = 1` is used because FSPS
rescales the TP-AGB IMF weights by `agb` before summing the mass; the fraction is
above 1 below about 2 Myr (youngest MIST isochrones lack low-mass stars) but that age
range carries negligible mass and lies well below the `epoch_gyr >= 1.0` floor used by
every population statistic (docs/ANALYSIS.md, Setup). R, the ratio of the two windows,
is unchanged (a per-epoch normalizer cancels); the absolute thresholds that define the
sSFR classes move by 1.35-1.79x, shifting counts out of quiescent into rapid-quenching
and transitional (docs/lessons.md, 2026-09-24 Phase 4 Task 1 entry).

`run_population.py` also emits `agb = 1` index columns (`d4000_agb1`, `hdelta_a_agb1`,
`h_minus_bump_sigma300_agb1`, `h_minus_bump_r100_agb1`) and a per-epoch
`surviving_mass_fraction` column, both used by the conclusion figures below.

### Conclusion figures

`analysis_conclusion_figures.py` (`--out-dir output/analysis`, `--pilot`) tests two
conclusions proposed for the manuscript against the Phase 3 population and fiducial
track machinery: (1) that the population-level bump locus can test the TP-AGB model,
and (2) that combining the three indices adds SFH/quenching information beyond D4000
and HdeltaA alone. A third figure isolates metallicity by tracing the fiducial SFH at
four metallicities. Five figures and one summary JSON
(`output/analysis/{c1_tpagb_population_test,c2_age_clocks,c2_clock_planes,
c2_sfh_recovery,c3_metallicity_planes}.png`, `conclusion_summary.json`), each entry in
the JSON keyed to the figure and, within it, to the numbers drawn on the page. Answers,
with every number traced to a JSON key, are in docs/ANALYSIS.md, "Supporting figures
for the two conclusions".

## Phase 5 — SFH-family sensitivity (2026-09-24)

### Goal

Test whether the Phase 3 and 4 conclusions depend on the assumed *shape* of the
post-quench star formation rate, by rerunning the population with three alternative
SFH families alongside the existing one, paired history by history with the same 2000
draws, for both TP-AGB template configurations.

### The four families

All four share the delayed-tau rise `SFR(t) = (t/tau) exp(-t/tau)` for `t < t_q` and
the same 2000 draws of `(t_q, tau_q, log_z)` (seed 20260924, unchanged since Phase 2):

- `exponential` (existing, `tau = t_q`): `SFR(t >= t_q) = e^-1 exp(-(t - t_q)/tau_q)`.
- `linear` (FSPS `sfh = 5` form, `tau = t_q`): `SFR(t >= t_q) = e^-1 max(0, 1 -
  (t - t_q)/delta_q)`, `delta_q = 2 ln2 x tau_q` (same SFR half-life as `exponential`).
- `truncation` (FSPS `sfh = 4` with `sf_trunc`, `tau = t_q`): `SFR(t >= t_q) = 0`.
- `decoupled`: exponential quench with `tau_q` as drawn, but the rise `tau` independent
  of `t_q`, log-uniform on [0.5, 5] Gyr, seed 20260926 (new `tau_gyr` column in
  `draws.npz`). `SFR(t >= t_q) = (t_q/tau) exp(-t_q/tau) exp(-(t - t_q)/tau_q)`
  (continuous at `t_q`).

`sfh_model.py` dispatches all four through `sfh_family_cumulative`/
`star_formation_rate_family`; `run_population.py` gained `--sfh-family` (default
`exponential`, unchanged behavior and output path) and `--out-dir` now defaults to
`output/population` for `exponential` and `output/population_<family>` otherwise (both
also get an `_lw02` variant for the LW02 template configuration, same convention as
Phase 3).

### FSPS-native cross-checks

FSPS's own `sfh = 5` (delayed-tau plus a linear ramp, `sf_slope`) and `sfh = 4` with
`sf_trunc` (delayed-tau plus a hard cut) are close FSPS-native analogues of the
`linear` and `truncation` families respectively; `cross_check_fsps_families.py`
compares this project's integrator against a native FSPS population built with those
`sfh` values at the same fiducial history (`t_q = 3.0`, `tau_q = 0.3` Gyr, solar Z), at
6 epochs and the 3 index windows, and asserts a maximum relative flux difference below
2 percent (docs/SPEC.md, "Validation"). Result:
`output/single_csp/fsps_family_cross_check.json`, largest difference 2.69e-04 (0.027
percent), also recording the `sfh = 5` `sf_slope` sign check. `exponential` and
`decoupled` have no matching FSPS-native `sfh` value (a two-branch rise-then-decay, or
a rise `tau` independent of the quench parameters, cannot be built from one native
call); `exponential` was cross-checked in Phase 2 via FSPS's tabular `sfh = 3` instead
(`output/single_csp/fsps_cross_check.json`); `decoupled` has no FSPS-native
cross-check, only the internal `scipy.integrate.quad` and continuity checks in
`tests/test_sfh_families.py`.

### Outputs

- `output/population_{linear,truncation,decoupled}[_lw02]/`: same layout as
  `output/population[_lw02]/` (Phase 2) — `draws.npz`, `indices.npz`/`indices.csv`
  (git-ignored), four `index_planes_*.png`, `summary.json` (now also carrying an
  `sfh_family` provenance key).
- `output/single_csp/fsps_family_cross_check.json`: the `linear`/`truncation`
  cross-check and the `sf_slope` sign check.
- `output/analysis/{s1_fiducial_families,s2_population_bands,s3_class_fractions,
  s4_classifier_by_family,s5_recovery_by_family}.png`, `sfh_sensitivity_summary.json`:
  the family-comparison figures and their numbers (`analysis_sfh_sensitivity.py`).

### Code layout additions

| File | Purpose |
| ---- | ------- |
| `sfh_model.py` | generalized cumulative-mass/SFR (rise `tau` separate from `t_q`), the four-family dispatch, `linear`/`truncation` closed forms, `draw_decoupled_tau`. |
| `cross_check_fsps_families.py` | cross-check `linear`/`truncation` against FSPS's own `sfh = 5`/`sfh = 4`, and the `sf_slope` sign check. |
| `analysis_sfh_sensitivity.py` | S1-S5: fiducial tracks, population bump bands, class fractions, classifier and SFH-recovery gain, per family and template. |

Answers, with every number traced to a JSON key, are in docs/ANALYSIS.md, "Sensitivity
to the star formation history model".

---

## Phase 1 — SSP sandbox (2026-08-31, complete)

### Goal

Use `python-fsps` (which wraps the Fortran `fsps` code) to generate **single
stellar population (SSP)** spectra with different assumptions about the **AGB /
TP-AGB** stellar population, and compare the resulting spectra, especially in
the rest-frame near-infrared (NIR) around 1.6 micron (1.4–1.8 micron).

### Fixed configuration (fiducial)

| Setting            | Value                | Notes                                  |
| ------------------ | -------------------- | -------------------------------------- |
| Isochrones         | MIST                 | compiled in; `zsol = 0.0185`           |
| Spectral library   | C3K (low-res, `c3k_lr`; later `c3k_hr`) | 1936 / 10992 wavelength points |
| Dust emission      | Draine & Li 2007     | compiled in (`DL07`)                   |
| IMF                | Kroupa (2001) / Chabrier (2003) | variable between the two        |
| Metallicity        | Solar (`zmet=11`, Z = 0.0185) |                              |
| Age                | 1 Gyr (`tage=1.0`)   |                                        |
| SFH                | single burst (`sfh=0`) |                                     |

### AGB / TP-AGB parameters (the tunable knobs)

Identified from `fsps/src/sps_vars.f90`, `mod_gb.f90`, `getspec.f90` and the
`python-fsps` docstrings. All are **SSP-level** parameters (regenerate SSPs when
changed).

| Parameter            | Default | Meaning                                                                                     |
| -------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `agb`                | 1.0     | Multiplicative weight of TP-AGB stars (phase=5). `0` removes them.                          |
| `pagb`               | 1.0     | Weight of the post-AGB phase (phase=6, Rauch 2003 spectra). `0` removes them.               |
| `add_agb_dust_model` | True    | Turn on/off AGB circumstellar dust (Villaume et al. 2014).                                   |
| `agb_dust`           | 1.0     | Scales the circumstellar AGB dust emission.                                                  |
| `use_lw_tpagb`       | 0       | If 1, use Lancon & Mouhcine (2002) empirical O-rich TP-AGB spectra; else C3K main grid.      |
| `tpagb_norm_type`    | 2       | TP-AGB normalization scheme. **Only affects Padova isochrones — inert for MIST.**            |
| `fcstar`             | 1.0     | **Inert** — the dilution line is commented out in `ssp_gen.f90` (the FSPS manual says "Currently has no effect"). Historically a Padova-specific C-star fraction knob; MIST has no C-rich TP-AGB stars, so it would not matter anyway. |
| `redgb`              | 1.0     | RGB weight (no effect for Padova; MIST has an explicit RGB phase).                           |
| `dell`, `delt`       | 0.0     | log L / log T shifts of TP-AGB (applied for all isochrone sets in `mod_gb.f90`, contrary to the docstring). |

### Mechanism (how AGB spectra enter)

In `getspec.f90`, stars with `phase=5` (TP-AGB) and `logT < 3.6` are assigned
dedicated templates instead of the C3K grid:

- **O-rich** (`ffco <= 1`, `use_lw_tpagb=1`): Lancon & Mouhcine 2002 (`Orich.spec`).
- **C-rich** (`ffco > 1`): Aringer et al. 2009 (`Crich_Aringer.spec`), since the
  compile-time flag `cstar_aringer=1`; otherwise Lancon & Wood 2002 (`Crich.spec`).
- **post-AGB** (`phase=6`, `logT >= 4.699`): Rauch 2003 non-LTE models.
- **Circumstellar dust** (`add_agb_dust_model=1`): `add_agb_dust.f90` (Villaume 2014).

The templates live in `$SPS_HOME/SPECTRA/AGB_spectra/` and are interpolated onto
the FSPS output wavelength grid.

### Experiment scenarios

For each IMF (Kroupa, Chabrier), run:

1. `fiducial` — all defaults.
2. `no_tpagb` — `agb=0`.
3. `no_pagb` — `pagb=0`.
4. `no_agb_dust` — `add_agb_dust_model=False`.
5. `lw02_o_rich` — `use_lw_tpagb=1`.
6. `double_tpagb` — `agb=2`.
7. `no_agb_all` — `agb=0`, `pagb=0`, `add_agb_dust_model=False`.

### Outputs

- Spectra (F_nu, Lsun/Hz, per solar mass formed) saved to `output/ssp_spectra.npz`
  (low-res) and `output/ssp_spectra_hr.npz` (high-res).
- Comparison plots in `output/`.
