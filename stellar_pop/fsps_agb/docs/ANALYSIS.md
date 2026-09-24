# Phase 3 — Diagnostic analysis: answers

All numbers below are computed by the scripts; each is followed by the file (under
`output/`) and the key it comes from. `a.b.c` denotes nested JSON keys.

## Setup

Models: MIST isochrones with the C3K spectral library in FSPS (Chabrier IMF, no nebular
emission, no dust), SSP grids at four metallicities (log Z = -0.5, -0.25, 0, +0.25) and
107 ages. Each history has one metallicity; CSPs are assembled by our own integrator
(`csp_integrate.py`). Indices: D4000 and HdeltaA at sigma = 300 km/s (`sigma300`), and
the 1.6 micron H-minus bump index at `sigma300` and at R = 100 (`r100`). The bump index
compares a feature band with a pseudo-continuum fitted to two side bands, so a tilted
continuum is removed. Yardstick precisions: 0.005, 0.010 and 0.020 mag for the bump,
0.05 for D4000, 0.5 A for HdeltaA. Population statistics use 2000 delayed-tau-plus-
quenching histories x 260 epochs, keeping epochs with `epoch_gyr >= 1.0` (482,000
rows). "agb" is the FSPS TP-AGB weight: agb = 0 removes TP-AGB stars; agb = 2 doubles
them (built as 2 S(agb=1) - S(agb=0)). "Strong TP-AGB" here means agb = 2, in two
template configurations run side by side:

- **default**: O-rich TP-AGB stars get C3K hydrostatic spectra (FSPS
  `use_lw_tpagb = 0`); outputs `output/ssp_grid`, `output/population`,
  `output/analysis/q*`.
- **LW02**: O-rich TP-AGB stars get the empirical Lancon & Mouhcine (2002) spectra
  (`use_lw_tpagb = 1`); outputs `output/ssp_grid_lw02`, `output/population_lw02`,
  `output/analysis/lw02_q*`.

Both sSFR windows (0-100 Myr and 100-1000 Myr before `t_obs`) are normalized by the
surviving stellar mass at `t_obs` (living stars plus remnants, FSPS
`StellarPopulation.stellar_mass` at agb = 1), not the mass formed; the mass-formed
convention had no return fraction. R, the ratio of the two windows, is unchanged by
this (a per-epoch normalizer cancels), but the absolute sSFR thresholds that define the
classes see values 1.35-1.79 times larger than under the old convention.

## Q1. Can a model with TP-AGB contribution be told apart from one without?

**Answer: conditional on the TP-AGB templates. With the default C3K templates, no: agb 0
versus 2 moves the bump by at most 0.024 mag for SSPs and by 0.003-0.009 mag at fixed
D4000 or HdeltaA in the population. That offset is 0.9-2.2 times the per-model intrinsic
scatter, so the "no" rests on the 0.01 mag precision, not on the scatter: it is below
0.01 mag in every bin. With the empirical LW02 templates, yes: the population offset is
0.032-0.059 mag and 6-16 times the per-model scatter in every bin. For SSPs the optical
indices move by at most 0.034 in D4000 (below the 0.05 precision) and 0.47 A in HdeltaA,
which is at the 0.5 A yardstick; this optical null is an SSP-level statement.**

The answer rule (docs/SPEC.md): "clearly" means the offset is larger than both the
intrinsic spread at fixed optical indices and 0.01 mag.

Evidence, default templates:

- SSP level: the largest |Delta bump| (agb2 - agb0) is 0.0241 mag at 0.71 Gyr, log Z =
  -0.5 (`analysis/q1_summary.json`, `step1.max_abs_delta_h_minus_bump_sigma300_mag`,
  `..._age_gyr`); 0.0204 mag at solar Z
  (`step1.max_abs_delta_h_minus_bump_sigma300_mag_by_log_z.+0.00`); 0.0244 mag at r100
  (`step1.max_abs_delta_h_minus_bump_r100_mag`). The agb0 bump spread across the four
  metallicities at fixed age reaches 0.0186 mag
  (`step1.max_agb0_metallicity_spread_h_minus_bump_sigma300_mag`), comparable to the
  TP-AGB delta.
- Optical indices (SSPs): max |Delta D4000| = 0.034 (`step1.max_abs_delta_d4000`), below
  the 0.05 precision; max |Delta HdeltaA| = 0.47 A
  (`step1.max_abs_delta_hdelta_a_angstrom`), at the 0.5 A yardstick rather than clearly
  below it. Both peak at about 1 Gyr, log Z = -0.5. This is an SSP-level result; the
  population offsets were measured only for the bump.
- Why the bump barely moves: the pure TP-AGB component S(1) - S(0) has a bump index of
  only -0.037 to -0.015 mag (`step2.component_bump_min_mag`, `step2.component_bump_max_mag`),
  although TP-AGB stars supply up to 37.6 percent of the feature-band light at 0.79 Gyr
  (`step2.peak_tpagb_light_fraction`, `..._age_gyr`). TP-AGB light enters mostly as a
  tilted continuum that the pseudo-continuum removes. A flux-calibrated NIR/optical
  ratio F_nu(1.6 micron)/F_nu(4200 A) changes by 84 percent at 1 Gyr and 21 percent at
  5 Gyr, solar Z (`step4.relative_difference_nir_optical_ratio_1gyr_solar_z`,
  `..._5gyr_solar_z`).
- Fiducial track (t_q = 3 Gyr, tau_q = 0.3 Gyr, solar Z): the largest bump delta is
  -0.0108 mag (sigma300) and -0.0111 mag (r100) at 3.45 Gyr, 0.45 Gyr after quenching
  (`single_csp/indices.csv`, minimum of `h_minus_bump_{sigma300,r100}_agb2 - ..._agb0`).
- Population (12 bins of D4000 and, separately, of HdeltaA; both models binned on the
  agb0 index of the same epoch, `step6.headline_binning`): the median paired offset is
  0.0026-0.0092 mag (`analysis/q1_summary.json`,
  `step6.bump_{sigma300,r100}_by_{d4000,hdelta_a}_agb0_median_diff_mag`). The scatter is
  the per-model root-mean-square 16-84 half-width, sqrt((hw0^2 + hw2^2) / 2)
  (`step6.scatter_definition`, `..._rms_half_width_mag`); the offset is 0.90-2.20 times it
  (`step6.min_offset_over_scatter`, `step6.max_offset_over_scatter`, source
  `bump_r100_by_hdelta_a_agb0`) and exceeds it in 44 of 48 bins
  (`..._offset_over_scatter`). Binning each model on its own D4000 or HdeltaA gives the
  same offsets, 0.0026-0.0092 mag (`..._own_bins_median_diff_mag`). The offset never
  reaches 0.01 mag, so the answer rule passes in 0 of 48 bins. With 0.005 mag in place
  of 0.01 mag it would pass in 26 of 48 bins (same keys: |median diff| > 0.005 and
  |offset over scatter| > 1).

Evidence, LW02 templates:

- SSP level: max |Delta bump| = 0.132 mag at 0.79 Gyr, log Z = +0.25; 0.111 mag at solar
  Z, 0.081 mag at log Z = -0.5 (`analysis/lw02_q1_summary.json`,
  `step1.max_abs_delta_h_minus_bump_sigma300_mag`,
  `step1.max_abs_delta_h_minus_bump_sigma300_mag_by_log_z`). That is 5.4 times the
  default solar-Z value (`analysis/q1_summary.json`,
  `step5.max_abs_delta_bump_lw02_sigma300_mag` = 0.111 vs
  `step5.max_abs_delta_bump_c3k_solar_sigma300_mag` = 0.020).
- The pure TP-AGB component has a bump of -0.238 to -0.190 mag (`lw02_q1_summary.json`,
  `step2.component_bump_min_mag`, `step2.component_bump_max_mag`) at the same light
  fraction (0.375, `step2.peak_tpagb_light_fraction`). The template spectrum, not the
  amount of TP-AGB light, sets the answer.
- Mechanism: the LW02 signal comes from H2O absorption in the index side bands of the
  empirical TP-AGB spectra, not from a feature inside the feature band. The component
  spectrum (`lw02_q1_component_spectrum.png`, right panel) drops steeply at 1.33-1.50
  micron, next to the blue side band, and at 1.75-1.80 micron, next to the red side band,
  so the pseudo-continuum is pulled down and the feature band appears as a bump. It has
  straight, interpolated segments at about 1.34-1.42 and 1.81-1.93 micron where the
  empirical spectra have telluric gaps. The component has the same shape at 0.32, 1.00
  and 3.16 Gyr because FSPS uses one fixed set of LW02 templates. The ratio
  S(agb2)/S(agb0) in `lw02_q1_nir_spectra.png` shows the same side-band drops at every
  age.
- Optical indices (SSPs) move less than the yardsticks: max |Delta D4000| = 0.022, max
  |Delta HdeltaA| = 0.31 A (`step1.max_abs_delta_d4000`, `step1.max_abs_delta_hdelta_a_angstrom`).
- Fiducial track: -0.0736 mag (sigma300) and -0.0733 mag (r100) at 3.65 Gyr, 0.65 Gyr
  after quenching (`single_csp_lw02/indices.csv`, same columns).
- Population: the median paired offset is 0.032-0.059 mag in all 48 bins, 6.07-16.31
  times the per-model RMS half-width (`lw02_q1_summary.json`,
  `step6.bump_*_median_diff_mag`, `step6.bump_*_offset_over_scatter`;
  `step6.min_offset_over_scatter` = 6.07, `step6.max_offset_over_scatter` = 16.31).
  Binning each model on its own optical index gives 0.032-0.057 mag
  (`step6.bump_*_own_bins_median_diff_mag`). It passes the answer rule in 48 of 48 bins.

Figures (`output/analysis/`):

- `q1_ssp_delta_vs_age.png`, `lw02_q1_ssp_delta_vs_age.png`: Delta index vs age for the
  four metallicities, with the metallicity-spread band and the yardstick lines.
- `q1_component_spectrum.png`, `lw02_q1_component_spectrum.png`: the pure TP-AGB
  component.
- `q1_nir_spectra.png`, `lw02_q1_nir_spectra.png`: S(agb2)/S(agb0) and the
  continuum-normalized spectra, with the index bands shaded.
- `q1_broadband_ratio.png`: the NIR/optical flux ratio.
- `q1_lw02_variant.png`: default vs LW02 SSP delta at solar Z.
- `q1_population_offsets.png`, `lw02_q1_population_offsets.png`: the population bump
  bands for agb0 and agb2 at fixed D4000 and at fixed HdeltaA, with the offset over the
  per-model RMS scatter (dashed) and 0.01 mag over that scatter (dotted).

The fiducial-track figures are `output/single_csp/time_evolution.png` and
`output/single_csp_lw02/time_evolution.png`.

## Q2. With agb = 2, can the fast-quenching population be isolated?

**Answer: partially, and only in the optical D4000-HdeltaA plane. With the yardstick
noise (D4000 +/- 0.05, HdeltaA +/- 0.5 A) a k-nearest-neighbour classifier recovers
about 40 percent of the rapid-quenching epochs at about 61 percent purity, at a base rate
of about 1.2 percent. Noise-free purity maps, an upper bound, put 79-84 percent of them in
cells more than 50 percent pure. The bump adds nothing TP-AGB-specific with the default
templates: its changes (completeness -0.016 to -0.010, purity +0.007 to +0.011) are the
same size as with agb0, which has no TP-AGB light. With the empirical LW02 templates it
adds a real but modest gain: at 0.005 mag, completeness +0.090 +/- 0.008 and purity
+0.040 +/- 0.003; at 0.010 mag, +0.047 +/- 0.003 and +0.021 +/- 0.008 (mean and
standard deviation over three noise seeds; 5-10 paired fold standard errors). At 0.020
mag the gain is at most +0.012. The bump planes alone isolate at most 32 percent even
without noise.**

Classes follow the manuscript's sSFR rules, now applied to sSFR normalized by the
surviving stellar mass (see Setup) rather than the mass formed. They depend only on the
SFH, so the counts are the same in both configurations: 5,907 rapid-quenching epochs out
of 482,000 (1.23 percent), including 3,159 post-starburst (`analysis/q2_summary.json`,
`class_summary.counts`, `class_summary.post_starburst_count`; identical in
`lw02_q2_summary.json`). Under the previous mass-formed normalization the counts were
4,727 rapid-quenching and 2,845 post-starburst; every move is out of quiescent
(264,299 -> 247,231), split between rapid-quenching (+1,180) and transitional
(110,874 -> 126,762, +15,888); star-forming is unchanged (102,100)
(`docs/lessons.md`, 2026-09-24 Phase 4 Task 1 entry).

Evidence, purity maps (noise-free upper bounds). A 40 x 40 grid spans the full range of
each axis over all classes; a cell needs 20 epochs; "isolable" = fraction of
rapid-quenching epochs in cells with purity > 0.5 (`purity_grid_definition`). The
secondary column uses a grid between the 0.5 and 99.5 percentiles:

| plane | default agb2 full / clipped | LW02 agb2 full / clipped | agb0 (both) full / clipped |
| --- | ---: | ---: | ---: |
| D4000-HdeltaA | 0.839 / 0.822 | 0.843 / 0.803 | 0.816 / 0.788 |
| D4000-bump | 0.075 / 0.077 | 0.319 / 0.174 | 0.222 / 0.197 |
| HdeltaA-bump | 0.084 / 0.076 | 0.144 / 0.057 | 0.039 / 0.025 |

(`q2_summary.json` and `lw02_q2_summary.json`,
`purity_maps.<agb>.<plane>.isolable_fraction` and
`..._isolable_fraction_percentile_clipped`.) The grid choice moves the optical-plane
fraction by up to +/- 0.03, so differences of that size between columns or
configurations are not meaningful. It matters for the LW02 bump planes: the deepest bumps
belong to rapid-quenching epochs, and the clipped grid leaves 25.6 percent of them
outside (`lw02_q2_alternative_summary.json`,
`purity_maps_percentile_clipped.agb2.d4000_h_minus_bump.rapid_quenching_fraction_outside_grid`),
so the clipped 0.174 undercounts.

Evidence, noise-aware k-nearest-neighbour classifier (k = 25, 5 folds grouped by history,
D4000 +/- 0.05 and HdeltaA +/- 0.5 A noise, bump at sigma300). Each of three noise seeds
sets the noise, drawn from independent streams for the optical pair and for every bump
product and precision, and the fold assignment. Table: mean over the seeds of the
fold-mean metric, +/- its standard deviation over the seeds
(`classifier.across_seeds.agb2.<set>.unbalanced.{completeness,purity}_mean_{mean,std}_over_seeds`):

| agb2, unbalanced | default completeness | default purity | LW02 completeness | LW02 purity |
| --- | --- | --- | --- | --- |
| D4000, HdeltaA only | 0.402 +/- 0.006 | 0.612 +/- 0.007 | 0.398 +/- 0.004 | 0.610 +/- 0.005 |
| + bump +/- 0.005 mag | 0.393 +/- 0.009 | 0.622 +/- 0.006 | 0.488 +/- 0.005 | 0.651 +/- 0.007 |
| + bump +/- 0.010 mag | 0.388 +/- 0.005 | 0.623 +/- 0.001 | 0.446 +/- 0.001 | 0.632 +/- 0.008 |
| + bump +/- 0.020 mag | 0.387 +/- 0.009 | 0.619 +/- 0.002 | 0.405 +/- 0.010 | 0.621 +/- 0.006 |

The gain of each bump set over the optical-only set is measured per fold on the same
folds (`classifier.paired_difference_definition`; per seed in
`classifier.results_by_seed.<seed>.<agb>.<set>.<balance>.paired_difference_vs_no_bump`,
per-fold values in `completeness_per_fold`, `purity_per_fold`):

- LW02, agb2, unbalanced: completeness gain +0.090 +/- 0.008 (0.005 mag), +0.047 +/-
  0.003 (0.010 mag), +0.007 +/- 0.006 (0.020 mag); purity gain +0.040 +/- 0.003, +0.021
  +/- 0.008, +0.011 +/- 0.003 (seed mean +/- seed std,
  `across_seeds.agb2.bump_sigma300_<p>.unbalanced.{completeness,purity}_gain_mean_*`).
  In units of the paired fold standard error the 0.005 and 0.010 mag gains are 5.9-11.3
  on average over seeds (`..._gain_over_standard_error_mean_over_seeds`); with only 5
  folds these ratios vary strongly between seeds (standard deviation 0.8-3.4), so the
  seed-to-seed spread of the gain itself is the more robust yardstick, and it is 3-14
  times smaller than the gain. All three seeds share the same population realization
  (the same 2000 histories), so the seed spread excludes population sampling variance. Class-balanced training shows the same pattern: purity
  0.207 rises to 0.242 at 0.010 mag and 0.272 at 0.005 mag
  (`across_seeds.agb2.<set>.balanced.purity_mean_mean_over_seeds`).
- Default, agb2: completeness changes by -0.016 to -0.010 and purity by +0.007 to +0.011
  across the three precisions (sigma300). The agb0 control, which has no TP-AGB light,
  shows changes of the same size (completeness -0.014 to +0.012, purity +0.004 to +0.021
  over both products, `across_seeds.agb0`), so these presumably reflect the metallicity and
  age information that any bump carries, not TP-AGB light.
- The two bump products (sigma300, r100) give nearly identical classifier results
  because their bumps differ by at most 0.0011 mag, almost all of it a constant offset
  (epoch-to-epoch standard deviation 0.0001 mag;
  `bump_product_difference_sigma300_minus_r100.<agb>`), far below the smallest
  precision. They are not independent corroboration.
- In every configuration, rapid-quenching epochs are confused with transitional and
  quiescent epochs, essentially never with star-forming ones (the `confusion` matrices
  in `results_by_seed`).

Evidence, alternative SFH families as contaminants. Two families of 300 histories each
(seed 20260925, epochs at or after 1 Gyr, same integrator and class rules;
`analysis_alternative_sfh.py`):

- bursty star-forming: no quench, plus a 0.1 Gyr Gaussian burst at 2-10 Gyr that adds
  10 percent of the final mass;
- slowly fading: tau_q 3-6 Gyr.

Results:

- Neither family produces a rapid-quenching epoch, and neither can by construction: over
  epochs >= 1 Gyr the sSFR ratio R never drops below 0.404 (bursty) or 0.844 (slowly
  fading), far above the R < 0.1 rule
  (`q2_alternative_summary.json`,
  `contaminant_classes.<family>.min_ssfr_ratio_recent_over_previous`). Counts: bursty
  22,346 star-forming, 49,954 transitional; slowly fading 15,368 star-forming, 56,927
  transitional, 5 quiescent (`contaminant_classes.<family>.counts`; identical for
  LW02). This makes the test a weak one: it only checks whether the contaminants land
  in the rapid-quenching region of the index planes.
- Purity maps (noise-free upper bounds): 0 contaminant epochs fall in any cell that had
  purity > 0.5, in every plane, for agb2 and agb0, in both configurations and on both
  grids. Pooled purity and isolable fractions are therefore unchanged: D4000-HdeltaA
  agb2 pooled purity is 0.843 (default) and 0.840 (LW02) before and after
  (`purity_maps.<agb>.<plane>.n_contaminant_epochs_in_pure_before_cells`,
  `pooled_purity_before`, `pooled_purity_after` in `q2_alternative_summary.json` and
  `lw02_q2_alternative_summary.json`; same result in `purity_maps_percentile_clipped`).
- Classifier trained on the whole delayed-tau population (noise seed 20260924), applied
  to the 144,600 noised contaminant epochs (`classifier.results.<agb>.<set>.<balance>`).
  The pooled purity combines the cross-validated delayed-tau confusion matrix of the same
  seed (true and false positives among delayed-tau epochs) with the contaminant false
  positives of this full-sample model:
  - Unbalanced training labels at most 20 contaminant epochs as rapid-quenching in
    either configuration (`n_contaminant_false_rapid_quenching`; a rate of at most
    0.014 percent).
  - Class-balanced training (optical only, agb2) mislabels 1,912 (default) and 2,003
    (LW02) epochs, about 1.3-1.4 percent, mostly slowly fading. This lowers the pooled
    purity from 0.208 to about 0.193 in both configurations
    (`pooled_purity_delayed_tau_only`, `pooled_purity_with_contaminants`).
  - With the LW02 bump at 0.005 mag, the balanced mislabel count drops to 424 (0.29
    percent), and pooled purity goes from 0.277 to 0.271
    (`lw02_q2_alternative_summary.json`,
    `classifier.results.agb2.bump_sigma300_0.005.balanced`).
  - These counts scale with the arbitrary 600 : 2000 mix of contaminant to delayed-tau
    histories; the per-epoch rates do not.
- Untested: a slowly fading history that crosses into the rapid-quenching region, and
  dust, nebular emission fill-in and internal metallicity spread in any family.

Figures (`output/analysis/`):

- `q2_class_planes.png`, `lw02_q2_class_planes.png`: the four classes in the three
  planes, for agb2 and agb0.
- `q2_purity_maps.png`, `lw02_q2_purity_maps.png`: noise-free rapid-quenching purity per
  cell on the full-range grid, with the rapid-quenching density contours.
- `q2_classifier.png`, `lw02_q2_classifier.png`: completeness and purity vs bump
  precision (mean over the three noise seeds, bars = fold-to-fold std), against the
  no-bump band.
- `q2_alternative_sfh.png`, `lw02_q2_alternative_sfh.png`: the two contaminant families
  on top of the rapid-quenching epochs (agb2; bursty coloured by time since burst), then
  the noise-free purity maps with contaminants added. Red outlines mark the cells that
  had purity > 0.5 before.

## Caveats

- In the default configuration, O-rich TP-AGB stars have hydrostatic C3K model
  spectra. Real TP-AGB stars are pulsating, extended, and dusty, with H2O absorption
  that hydrostatic models underpredict. The LW02 empirical spectra give a bump about 5
  times larger. The Q1 answer and the bump part of Q2 depend on which template is
  right, and this analysis cannot decide that.
- The LW02 signal sits in the H2O bands next to the index side bands. Its strength
  depends on the pulsation phase of the observed stars and on how the telluric H2O was
  corrected in the empirical spectra, which also have gaps at about 1.34-1.42 and
  1.81-1.93 micron that FSPS fills by straight interpolation. FSPS uses one fixed set of
  LW02 templates, so the component has a single shape at every age and metallicity; a
  real population of TP-AGB stars at different phases would not.
- MIST has no C-rich TP-AGB stars (the C-star templates are never used), so the
  carbon-star contribution at 1-2 Gyr and low Z is missing from both configurations.
- No nebular emission and no dust (neither interstellar nor circumstellar). Emission
  fill-in of HdeltaA in the bursty family, and dust reddening, would both move points
  in the optical plane.
- Four metallicities, one Z per history, interpolated linearly in log Z. At fixed
  D4000 the population bump scatter is mostly metallicity (Phase 2 review), so a
  metallicity spread within galaxies would widen the bump planes.
- The population weights every epoch of every history equally, from 1 Gyr to 13 Gyr.
  Class fractions (1.23 percent rapid-quenching) are not those of a real sample, and
  purities scale with the class mix.
- The bump index removes a tilted continuum. Most of the TP-AGB NIR flux excess (84
  percent in F_nu(1.6 micron)/F_nu(4200 A) at 1 Gyr) is invisible to it. A
  flux-calibrated NIR/optical ratio would respond far more strongly, but it is also
  degenerate with dust and metallicity, which were not tested here.
- The contaminant families are two choices of shape. A burst followed by a truncation,
  or a quenching faster than tau_q = 0.1 Gyr, was not tested. The bursty family has
  constant SFR after t_q, so its sSFR ratio sits at the star-forming/transitional
  boundary R = 1.

## What would change the answer

For Q1, the default-template answer is set by the precision: the population offset is
0.003-0.009 mag, 0.9-2.2 times the per-model intrinsic spread, so a bump precision of
0.005 mag would already make it pass the answer rule in 26 of 48 bins. A TP-AGB template
with strong side-band H2O absorption, like LW02 (component bump about -0.2 mag instead of
-0.03 mag), makes the SSP and track differences 5-7 times larger and detectable at
0.01-0.02 mag. Deciding between C3K and LW02 is therefore the key empirical question. It
needs resolved spectroscopy of intermediate-age clusters or post-starburst galaxies at
1.3-1.9 micron, covering the side bands where the LW02 signal sits.

For Q2, the partial isolation of fast quenching rests on D4000 and HdeltaA and would
survive any TP-AGB model. The bump would matter only with LW02-like templates and a
precision of 0.01 mag or better, and even then it would add 5-9 points of completeness
and 2-4 points of purity.
