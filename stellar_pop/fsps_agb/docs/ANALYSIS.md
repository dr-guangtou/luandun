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

The surviving-mass fraction is above 1 (more stars "return" than were formed) for the
youngest SSP ages, crossing 1 between log(age/yr) = 6.30 and 6.35, i.e. about 2.0-2.2
Myr (`output/ssp_grid/surviving_mass.npz`; solar Z, agb = 1 fraction 1.0028 at
10^6.30 yr, 0.9960 at 10^6.35 yr). This is an artifact of the youngest MIST isochrones
lacking low-mass stars (lowest initial mass 2.6 Msun at 10^5 yr), not a physical return
of mass. It carries negligible mass and does not affect any population statistic here,
which is restricted to `epoch_gyr >= 1.0`: the per-epoch surviving fraction there is
0.559-0.743 (docs/lessons.md, 2026-09-24 Phase 4 Task 1 entry).

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

## Supporting figures for the two conclusions

`analysis_conclusion_figures.py` (`--out-dir output/analysis`, `--pilot` for a
subsampled dry run) tests two conclusions proposed for the manuscript, plus a
metallicity-only control, on top of the Q1/Q2 population (2000 histories, 482,000
epochs at `epoch_gyr >= 1.0`, both TP-AGB template configurations). Every number below
is in `output/analysis/conclusion_summary.json`, keyed per figure
(`c1_tpagb_population_test`, `c2_age_clocks`, `c2_clock_planes`, `c2_sfh_recovery`,
`c3_metallicity_planes`).

### Conclusion 1: "the inclusion of AGB makes a huge difference to the H-minus bump, so at the population level we can use it to test the AGB/TP-AGB model"

**The figures support a narrower claim than the one stated: the population bump locus
separates TP-AGB *templates*, not TP-AGB *weight*.** Figure `c1_tpagb_population_test.png`
draws four prescriptions on the same population — C3K agb0, C3K agb1, LW02 agb1, LW02
agb2 — as 16-84 percentile bands of the bump in 20 bins of D4000 (and separately of
HdeltaA), plus histograms in three D4000 slices. In the D4000 in [1.3, 1.5) slice, the
median offset between neighbouring prescriptions is
(`c1_tpagb_population_test.slices.d4000_1.3_1.5.neighbouring_separations`):

| pair | what changes | delta median [mag] | / wider-band half-width | / 0.01 mag yardstick |
| --- | --- | ---: | ---: | ---: |
| C3K agb0 -> agb1 | TP-AGB weight, C3K template | -0.0051 | -1.05 | -0.51 |
| C3K agb1 -> LW02 agb1 | TP-AGB template, fixed weight | -0.0268 | -6.32 | -2.68 |
| LW02 agb1 -> agb2 | TP-AGB weight, LW02 template | -0.0236 | -3.91 | -2.36 |

The same ordering holds in the other two D4000 slices, [1.5, 1.7) and [1.7, 1.9)
(`c1_tpagb_population_test.slices.d4000_{1.5_1.7,1.7_1.9}.neighbouring_separations`):
the weight step within C3K is always smallest (-0.0026 to -0.0051 mag, 0.6-1.1
half-widths, 0.3-0.5 of the yardstick); the template step (C3K agb1 -> LW02 agb1) and
the weight step within LW02 (LW02 agb1 -> agb2) are both several times larger (-0.016
to -0.027 mag, 2.2-6.3 half-widths, 1.6-2.7 yardsticks).

So doubling the TP-AGB weight inside the C3K template configuration moves the
population locus by about half the 0.01 mag yardstick, below what a bump measurement
at that precision could resolve. Swapping to the empirical LW02 template at the fixed
fiducial weight (agb = 1) moves the locus several times further than doubling the
TP-AGB weight does. What the population locus actually tests is which TP-AGB spectral
template is closer to nature (C3K hydrostatic models vs. LW02 empirical spectra) — the
same template dependence already reported for Q1 above — far more than "how much"
TP-AGB light a galaxy has. A population-level bump measurement could plausibly rule the
C3K hydrostatic templates in or out; it cannot, by itself, constrain the TP-AGB mass
fraction, because that quantity barely moves the locus within either template.

### Conclusion 2: "the three indices have different age sensitivities, so by combining them we could gain more understanding about the SFH and the quenching process"

**The qualitative premise (different clocks) holds; the practical payoff (combining
gains information) is template-dependent — real for the empirical LW02 templates,
marginal-to-absent for the default C3K templates.**

Age clocks (Figure `c2_age_clocks.png`: tau_q in {0.1, 0.3, 1, 3} Gyr at fixed t_q = 3
Gyr, and t_q in {1.5, 3, 4.5} Gyr at fixed tau_q = 0.3 Gyr, solar Z, agb = 2,
sigma300). HdeltaA is a fast clock, peaking 0.2-0.3 Gyr after quenching in both
templates with little shift across the tau_q/t_q family
(`c2_age_clocks.{c3k,lw02}.hdelta_a.*.delay_gyr`). D4000 has no interior extremum
within the +6 Gyr window in either template — every track's `at_window_boundary` flag
is true (`c2_age_clocks.{c3k,lw02}.d4000.*.at_window_boundary`) — so alone it only says
"quenched," not "how long ago," beyond the coarse rising trend. The raw H-minus bump
(as opposed to the agb0-to-agb2 delta) has an interior minimum only for LW02, 0.5-1.5
Gyr after quenching depending on tau_q and t_q, and the delay grows with tau_q: 0.5 Gyr
at tau_q = 0.1 Gyr, 0.8 Gyr at tau_q = 0.3 Gyr, 1.25 Gyr at tau_q = 1 Gyr, 1.5 Gyr at
tau_q = 3 Gyr (fixed t_q = 3 Gyr); the t_q family is nearly flat at 0.8-0.9 Gyr (0.9
Gyr at t_q = 1.5 Gyr, 0.8 Gyr at t_q = 3 and 4.5 Gyr, fixed tau_q = 0.3 Gyr)
(`c2_age_clocks.lw02.h_minus_bump.*.delay_gyr`, `at_window_boundary: false` in every
track); **for C3K the raw bump has no interior minimum in the +6 Gyr window at all** —
every C3K `h_minus_bump` track's `at_window_boundary` is true
(`c2_age_clocks.c3k.h_minus_bump.*.at_window_boundary`), so the marker plotted in that
figure sits at the window edge, not at a true extremum: population aging outweighs the
small C3K TP-AGB signal across the whole window. Figure `c2_age_clocks.png` now draws
these window-edge markers as hollow stars (filled stars mark a true interior
extremum), so this distinction is visible at a glance — every C3K D4000 and H-minus
bump track, and the `tau_q` in {1, 3} Gyr HdeltaA tracks, are hollow. Figure
`c2_clock_planes.png` shows the
same picture qualitatively: in the D4000-HdeltaA plane the tau_q and t_q families
collapse onto nearly the same locus for both templates, but in the bump-involving
planes the LW02 tracks fan out into visibly different loops while the C3K tracks stay
compressed.

The quantitative test (Figure `c2_sfh_recovery.png`): a k = 25 nearest-neighbour
regression (cKDTree, 5 folds grouped by history, 3 noise seeds) predicts log10(time
since quenching) and log10(tau_q) for 240,000 post-quench epochs (0 < t - t_q < 6 Gyr,
epoch >= 1 Gyr) from (D4000, HdeltaA) alone versus with the bump added at three
precisions. The paired gain from adding the bump at 0.005 mag, mean over the 3 noise
seeds (`c2_sfh_recovery.summary.<template>.bump_0.005.paired_gain_mean_over_seeds_dex`,
order `[log10(t - t_q), log10(tau_q)]`; `..._std_over_seeds_dex` alongside, the
across-seed spread):

| template | log10(t - t_q) gain [dex] | log10(tau_q) gain [dex] |
| --- | ---: | ---: |
| LW02 agb2 | -0.0183 +/- 0.0001 | -0.0129 +/- 0.0002 |
| C3K agb2 | -0.0039 +/- 0.0003 | -0.0002 +/- 0.0003 |
| C3K agb0 (control, no TP-AGB light) | -0.0063 +/- 0.0004 | -0.0010 +/- 0.0002 |

Significance uses two yardsticks. The primary one is the paired gain's own fold-based
standard error, computed per seed from the 5 grouped folds (ddof-1 std of the 5
per-fold gains / sqrt(5), `c2_sfh_recovery.definition.paired_gain_definition`) — a
distinct, smaller quantity than the RMS error bars drawn on the figure itself, which
are the mean over the 3 seeds of the fold-based standard error of the RMS values, not
of the gain (`c2_sfh_recovery.definition.figure_error_bar_definition`,
`...summary.<template>.bump_0.005.rms_fold_standard_error_mean_over_seeds_dex`). The
per-seed gain, its own fold-based standard error, and their ratio are in
`c2_sfh_recovery.summary.<template>.bump_0.005.paired_gain_per_seed.<seed>.{paired_gain_mean_dex,
paired_gain_standard_error_dex,paired_gain_over_standard_error}` (seeds 20260924,
20260925, 20260926): for **LW02 agb2** at 0.005 mag, `paired_gain_over_standard_error` is
-33.5 to -99.8 for log10(t - t_q) and -19.3 to -27.5 for log10(tau_q) across the three
seeds; at 0.010 mag (`bump_0.010.paired_gain_per_seed`) it is -19.9 to -38.1 and -13.5
to -20.8. Combined, the LW02 gains are 13.5-99.8, i.e. of order **13-100 fold-based
standard errors from zero, in every seed, at both precisions**. For **C3K agb2** at
0.005 mag, `paired_gain_over_standard_error` is -7.5 to -11.7 for log10(t - t_q)
(real but far smaller than LW02) and -0.33, +0.08, -4.4 per seed for log10(tau_q) —
**below 1 standard error from zero in 2 of the 3 seeds** (seeds 20260924 and 20260925)
and only -4.4 in the third (seed 20260926), not a consistent detection. At 0.010 mag it
is -4.5 to -6.4 for log10(t - t_q) (still real, weaker than at 0.005 mag) and +2.0,
+0.21, -8.1 per seed for log10(tau_q) — inconsistent even in sign across seeds, so
still not a consistent detection at this precision either. The second yardstick, the
across-seed standard deviation of
the gain itself (`paired_gain_std_over_seeds_dex`, which excludes population sampling
variance since all three seeds share the same 2000 histories), agrees: for LW02 it is
3-14 times smaller than the gain, so the gain is robust seed to seed; for C3K's
log10(tau_q) target it is comparable to or larger than the gain
(-0.0002 +/- 0.0003 dex), consistent with no robust detection. Critically, the **C3K
agb0 control** — which has no TP-AGB light by construction — shows a gain of the same
size as C3K agb2 on both targets, so the small, marginal C3K gain is not obviously a
TP-AGB effect at all; it more likely reflects age or metallicity information that any
bump-shaped index carries, TP-AGB or not. Conclusion 2 is well supported for the LW02
templates and not clearly supported for the default C3K templates, reported as-is
whichever way it goes.

### Metallicity as a confounder (Figure `c3_metallicity_planes.png`)

The fiducial SFH (t_q = 3, tau_q = 0.3 Gyr, agb = 2) traced at four metallicities (log
Z = -0.5, -0.25, 0, +0.25) isolates the metallicity effect from the TP-AGB effect. The
bump's metallicity spread and the TP-AGB (agb0-to-agb2) delta at the same epochs
(`c3_metallicity_planes.<template>.spread_across_metallicity.h_minus_bump.<epoch>` and
`...agb0_to_agb2_delta.h_minus_bump.<epoch>.mean`):

| epoch (since t_q) | C3K spread [mag] | C3K TP-AGB delta (mean) [mag] | LW02 spread [mag] | LW02 TP-AGB delta (mean) [mag] |
| --- | ---: | ---: | ---: | ---: |
| +0.0 Gyr | 0.0154 | -0.0091 | 0.0144 | -0.0523 |
| +0.5 Gyr | 0.0115 | -0.0111 | 0.0285 | -0.0674 |
| +1.0 Gyr | 0.0105 | -0.0097 | 0.0288 | -0.0650 |
| +2.0 Gyr | 0.0092 | -0.0064 | 0.0242 | -0.0471 |
| +5.0 Gyr | 0.0080 | -0.0039 | 0.0142 | -0.0331 |

For C3K, the metallicity spread (0.008-0.015 mag) is comparable to, and at every epoch
larger in magnitude than, the TP-AGB delta (0.004-0.011 mag): metallicity is a genuine
confounder for a C3K-based TP-AGB test, since an observed bump shift of this size could
equally be a metallicity difference. For LW02, the TP-AGB delta (0.033-0.067 mag) is
2-4 times the metallicity spread (0.014-0.029 mag) at every epoch, so metallicity is a
secondary systematic there.

The direction of the metallicity trend is also opposite between templates. Recomputing
the per-metallicity tracks directly (`analysis_conclusion_figures.compute_c3_tracks`,
consistent with the `min`/`max` values in
`c3_metallicity_planes.<template>.spread_across_metallicity.h_minus_bump`) shows the
C3K bump getting **weaker** (less negative) with increasing metallicity at every one of
the five marker epochs (e.g. at t_q: -0.0121 mag at log Z = -0.5 rising to +0.0033 mag
at log Z = +0.25), while the LW02 bump gets **stronger** (more negative) with
increasing metallicity (e.g. at t_q: -0.0417 mag at log Z = -0.5 deepening to -0.0561
mag at log Z = +0.25). A metallicity correction to the bump derived assuming one
template would move a measurement in the wrong direction if the other template is the
one nature uses.

## Sensitivity to the star formation history model (Phase 5)

**Answer: all four earlier conclusions survive across all four SFH families, with one
remaining gap. Every population run already carries `agb0`, `agb1` and `agb2` bump
columns (`run_population.py` always computes all three), so Phase 3 Q1 and Phase 4
Conclusion 1 (the population TP-AGB offset separates templates, not weight) can be, and
are, retested per family (Figure S6): in the [1.5, 1.7) D4000 slice the median
agb2-agb0 bump offset is -0.0059 to -0.0107 mag for C3K (1-3 times the per-model
scatter) and -0.0460 to -0.0670 mag for LW02 (4-12 times the scatter) in every family,
and the C3K-versus-LW02 template separation is 0.049-0.061 mag, several times any
weight-only step, in every family too. Phase 3 Q2 and Phase 4 Conclusion 2 also
survive: the LW02 bump gain in the classifier and SFH-recovery regression is positive
and multi-sigma, and the C3K gain is small and sign-inconsistent, in every family — but
the base rates and absolute numbers they operate on change a lot with the assumed
post-quench SFH shape (rapid-quenching base rate 0.84-6.38 percent across families).
What remains genuinely untested per family is narrower than "Q1/Conclusion 1": only the
`agb0` control of the classifier (S4) and SFH-recovery (S5) gains was run at `agb2`
only for the three new families, so the specific attribution "the gain is TP-AGB light,
not any third noisy feature" for Q2/Conclusion 2 still rests on the one `agb0` control
run in Phase 3/4, for the `exponential` family alone.**

### FSPS-native SFH forms, and which were used

FSPS provides four native star-formation histories relevant here: `sfh = 1` (a single
exponential decay), `sfh = 4` (the delayed-tau rise `(t/tau) exp(-t/tau)`, with
`sf_trunc` as a hard cut to zero SFR), `sfh = 5` (the same delayed-tau rise plus a
linear ramp after `sf_trunc`, slope set by `sf_slope`), and `sfh = 3` (a tabular SFH,
an arbitrary `SFR(t)` table). None of them can express this project's `exponential`
family (delayed-tau rise, then a *second*, independent exponential decay with its own
`tau_q`) or the `decoupled` family (rise `tau` independent of the quench parameters) in
one native call; that is why the original `exponential` family was cross-checked in
Phase 2 via the tabular route (`sfh = 3`) instead of a native form
(`output/single_csp/fsps_cross_check.json`, docs/SPEC.md "CSP assembly"). Phase 5 uses
`sfh = 5` to cross-check the new `linear` family and `sfh = 4` with `sf_trunc` to
cross-check the new `truncation` family (`cross_check_fsps_families.py`). `decoupled`
has no FSPS-native counterpart at all and was not cross-checked against FSPS's own
Fortran engine; it is validated only internally, against a `scipy.integrate.quad`
numerical integration of its analytic cumulative-mass formula and a continuity check at
`t_q` (`tests/test_sfh_families.py`).

Cross-check numbers (`output/single_csp/fsps_family_cross_check.json`, fiducial
`t_q = 3.0`, `tau_q = 0.3` Gyr, solar Z, `agb = 1`; 6 epochs x 3 index windows, each
family): the largest relative flux difference inside any index window anywhere in the
table is 2.69e-04 (0.027 percent), for `truncation` at 13.0 Gyr, D4000
(`families.truncation.epochs."13.00".max_relative_flux_difference.d4000`); `linear`'s
largest is 2.25e-04 (0.023 percent), also at 13.0 Gyr, D4000
(`families.linear.epochs."13.00".max_relative_flux_difference.d4000`). Both are four
orders of magnitude under the 2 percent validation threshold (docs/SPEC.md,
"Validation"), and tighter than the original `exponential` family's own tabular
cross-check (up to 0.312 percent at 1 Gyr, `output/single_csp/fsps_cross_check.json`).

`sfh = 5`'s `sf_slope` sign was verified directly against FSPS rather than assumed: with
`sf_slope = -1 / delta_q_gyr = -2.40449` (`delta_q_gyr = 2 ln2 x 0.3 = 0.41589`),
`population.sfr` read from FSPS at `t_q = 3.0` and `t_q + delta_q/2 = 3.20795` gives a
ratio of 0.4663 (below 1, declining); the opposite sign gives 1.339 (rising)
(`fsps_family_cross_check.json`, `sf_slope_sign_check.{sf_slope_per_gyr,ratio,
declining}`). This confirms the plan's expected sign with no flip needed; the ratio is
not exactly 0.5 because of FSPS's own internal SFH time-grid discretization, not a bug
in this project's code (task-1-report.md).

### The four-family design

All four families share the same delayed-tau rise, `SFR(t) = (t/tau) exp(-t/tau)` for
`t < t_q`, and the same 2000 paired draws of `(t_q, tau_q, log_z)`, seed 20260924
(unchanged from Phases 2-4), so any difference between families below is purely the
assumed post-quench SFR shape, not a different sampling of the prior:

- `exponential` (existing, rise `tau = t_q`): `SFR(t >= t_q) = e^-1 exp(-(t - t_q) /
  tau_q)`.
- `linear` (FSPS `sfh = 5` form, rise `tau = t_q`): `SFR(t >= t_q) = e^-1 max(0, 1 -
  (t - t_q) / delta_q)`, with `delta_q = 2 ln2 x tau_q` chosen so the ramp has the same
  SFR half-life as the `exponential` family's decay.
- `truncation` (FSPS `sfh = 4` plus `sf_trunc`, rise `tau = t_q`): `SFR(t >= t_q) = 0`, a
  hard cut.
- `decoupled`: the rise e-folding `tau` is drawn independently of `t_q` and `tau_q`,
  log-uniform on [0.5, 5] Gyr, seed 20260926 (a separate RNG stream from the draws'
  seed, stored as the `tau_gyr` column of `draws.npz`); the quench SFR is continuous at
  `t_q`: `SFR(t >= t_q) = (t_q/tau) exp(-t_q/tau) exp(-(t - t_q)/tau_q)`.

For the `decoupled` family's Figure S1 fiducial track only (there is no natural single
`tau` for one history), the implementer fixed `tau = sqrt(0.5 x 5.0) = 1.581` Gyr, the
geometric mean of the population's draw range (`sfh_sensitivity_summary.json`,
`decoupled_fiducial_tau_gyr`) — a documented choice, not a given, and it affects only
the S1 picture: S2-S5 use the full 2000 `tau_gyr` draws.

### Class fractions per family

Counts and fractions over the same 482,000 rows (epochs >= 1 Gyr) used everywhere else
in this document; identical between C3K and LW02 within a family because class
assignment depends only on the SFH, not the TP-AGB template
(`sfh_sensitivity_summary.json`, `s3_class_fractions.<family>.c3k`, `...lw02`):

| family | star_forming | rapid_quenching | transitional | quiescent | rapid_quenching % | post-starburst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoupled | 42,305 | 4,072 | 161,835 | 273,788 | 0.84 | 2,188 |
| exponential | 102,100 | 5,907 | 126,762 | 247,231 | 1.23 | 3,159 |
| linear | 102,394 | 17,262 | 45,296 | 317,048 | 3.58 | 16,000 |
| truncation | 99,922 | 30,730 | 3,466 | 347,882 | 6.38 | 30,394 |

The mechanism is the sharpness of the post-quench SFR drop: a sharper cutoff holds R =
sSFR(0-100 Myr) / sSFR(100 Myr-1 Gyr) below the rapid-quenching threshold (R < 0.1) for
longer, so the rapid-quenching share rises monotonically from 0.84 percent
(`decoupled`, whose SFR after `t_q` still carries the smooth exponential tail of an
independently-drawn, often slower `tau`) through 1.23 percent (`exponential`) and 3.58
percent (`linear`, ramping to zero over `delta_q`) to 6.38 percent (`truncation`, SFR
identically zero the instant SFR stops). `truncation` correspondingly empties
`transitional` (3,466 rows versus `exponential`'s 126,762), since almost nothing sits
in the intermediate-R regime once star formation has truly stopped; `decoupled`
instead moves mass out of `star_forming` into `transitional` (21.18 percent down to
8.78 percent), because many of its 2000 histories, with an independently drawn and
often larger `tau`, are still rising or have only just finished rising at the 1 Gyr
floor.

### The population TP-AGB offset per family (Figure S6)

Phase 3 Q1 and Phase 4 Conclusion 1 both rest on the paired per-epoch offset
bump(agb2) - bump(agb0) at fixed D4000, compared against the per-model 16-84 RMS
scatter. Every population table this project writes already carries `agb0`, `agb1`
and `agb2` bump columns for every SFH family (`run_population.py` computes all three
unconditionally), so this offset can be measured per family with no additional
population run: `compute_s6`/`compute_s6_summary` in `analysis_sfh_sensitivity.py` do
so directly from the existing tables, reusing S2's D4000 binning
(`s6_tpagb_offset_by_family`, Figure `s6_tpagb_offset_by_family.png`).

At the same three D4000 slices used throughout this document
(`s6_tpagb_offset_by_family.offset_by_family.<family>.<template>.d4000_slices`), the
[1.5, 1.7) slice's median offset, pooled per-model scatter and offset-over-scatter are
(`...d4000_1.5_1.7.{median_offset_mag,pooled_scatter_mag,offset_over_scatter}`):

| family | C3K median offset [mag] | C3K offset/scatter | LW02 median offset [mag] | LW02 offset/scatter |
| --- | ---: | ---: | ---: | ---: |
| exponential | -0.0059 | -1.48 | -0.0460 | -6.58 |
| linear | -0.0093 | -2.15 | -0.0595 | -7.01 |
| truncation | -0.0107 | -2.17 | -0.0670 | -7.06 |
| decoupled | -0.0061 | -1.55 | -0.0460 | -7.08 |

Across all three D4000 slices ([1.3, 1.5), [1.5, 1.7), [1.7, 1.9)), the C3K offset is
1-3 times the per-model scatter in every family (`offset_over_scatter` -1.2 to -2.8),
and the LW02 offset is 4-12 times the scatter (`offset_over_scatter` -4.1 to -11.9) —
the same template split Phase 3 Q1 reported for the original `exponential` family
(0.90-2.20 for C3K, 6.07-16.31 for LW02, both measured on the pooled rather than the
per-model scatter there, so not numerically identical to the S6 values above, but the
same qualitative split) now holds for `linear`, `truncation` and `decoupled` too.

The C3K-versus-LW02 band-median separation (agb2, LW02 minus C3K, interpolated onto
C3K's D4000 bin centers since the two templates' bins are not bit-identical —
`s6_tpagb_offset_by_family.c3k_lw02_separation_by_family.<family>.separation_mag`)
peaks at `max_abs_separation_mag` = 0.0485 (exponential), 0.0533 (linear), 0.0613
(truncation), 0.0495 (decoupled) mag in every family — several times the weight-only
step reported in Phase 4 Conclusion 1 (C3K agb0 -> agb1, -0.0026 to -0.0051 mag) —
confirming that the population locus mainly tests which TP-AGB *template* is right,
not how much TP-AGB light there is, regardless of the assumed post-quench SFH shape.

### Does each earlier conclusion survive?

**Phase 3 Q1 (agb0 versus agb2 separability): survives, and is directly retested by
S6.** Q1's central comparison is a paired agb0-versus-agb2 offset in bins of D4000,
against the per-model scatter; the population TP-AGB offset section above (Figure S6)
measures exactly this per family, from the `agb0`/`agb2` columns every population
table already carries. The result reproduces Q1's template-conditional answer in every
family: the C3K offset is 1-3 times the per-model scatter (below Q1's "clearly"
threshold everywhere it was checked) and the LW02 offset is 4-12 times the scatter
(well above it), matching the original `exponential`-family finding qualitatively in
`linear`, `truncation` and `decoupled` too. S1's fiducial track (all four families,
`agb = 2`, both templates) is consistent with this: it converges to nearly the same
D4000/HdeltaA/bump curve beyond about 1 Gyr after quenching regardless of family
(`s1_fiducial_families.png`), so the SFH shape alone was not expected to flip which
template gives a detectable offset, and S6 now confirms it directly rather than by
inference.

**Phase 3 Q2 (isolating rapid quenching): survives, and is directly retested by S4.**
At the 0.010 mag bump precision (`agb2`, unbalanced;
`sfh_sensitivity_summary.json`, `s4_classifier_by_family.<family>.<template>.
bump_sigma300_0.010`): LW02 gives a positive completeness gain that is multi-sigma in
every family and every one of the 3 noise seeds (per-seed
`completeness_gain_over_standard_error` 4.19-12.83 across families: exponential
6.74/5.88/5.23, linear 11.83/9.23/9.78, truncation 9.64/8.09/12.83, decoupled
7.67/6.39/4.19), and a positive multi-sigma purity gain in every family too (per-seed
`purity_gain_over_standard_error` 4.27-16.53). C3K's completeness gain is small and
sign-inconsistent across families: negative and multi-sigma for `exponential`
(-5.57/-4.86/-2.54 sigma, matching the original Q2 finding that the bump actively hurts
C3K completeness) and `decoupled` (-3.12/-0.92/-3.16), but weakly positive for `linear`
(0.77/1.24/2.6) and `truncation` (3.09/2.61/6.86); C3K purity gain straddles zero in
every family (-2.32 to 3.4 sigma). So the qualitative Q2 pattern — a real, multi-sigma
LW02 gain, and a small-to-negative, sign-inconsistent C3K gain — replicates in all four
families, while the absolute numbers move with the base rate: no-bump completeness
alone ranges from 0.319 (`decoupled`) to 0.846 (`truncation`)
(`s4_classifier_by_family.<family>.<template>.no_bump.completeness_mean_mean_over_seeds`),
because a sharper quench (fewer, more separable rapid-quenching epochs) or a slower one
(more confusable ones) changes how well D4000 and HdeltaA alone already separate the
class before the bump is even added.

**Phase 4 Conclusion 1 (population locus tests template, not weight): survives, and is
directly retested by S6.** Like Q1, this needs the `agb0`/`agb1`/`agb2` columns, which
every population run already has. The population TP-AGB offset section above reports
the C3K-versus-LW02 band-median separation per family: 0.049-0.061 mag
(`max_abs_separation_mag`, `linear` largest at 0.0533, `truncation` largest overall at
0.0613), several times the weight-only step Phase 4 originally measured within C3K
(agb0 -> agb1, -0.0026 to -0.0051 mag) for the `exponential` family. The template step
dominates the weight step in every family tested, so Conclusion 1's "the locus mainly
tests which template is right, not how much TP-AGB light there is" is not specific to
the `exponential` family's SFH assumption.

**Phase 4 Conclusion 2 (combining indices adds SFH-recovery information): survives, and
is the most directly retested of the four.** S5 reruns the same k-nearest-neighbour
regression (`log10(time since quenching)`, `log10(tau_q)`, `agb2`, 3 noise seeds, 5
grouped folds) per family. At 0.005 mag
(`sfh_sensitivity_summary.json`, `s5_recovery_by_family.<family>.<template>.
bump_0.005`), the `log10(time since quenching)` gain is negative (an improvement) in
every family and template, and LW02's gain is always several times C3K's (mean +/- std
over 3 seeds, dex): exponential -0.0039 +/- 0.0004 (C3K) vs -0.0183 +/- 0.0004 (LW02);
linear -0.0052 +/- 0.0005 vs -0.0171 +/- 0.0003; truncation -0.0103 +/- 0.0001 vs
-0.0313 +/- 0.0003; decoupled -0.0029 +/- 0.0004 vs -0.0105 +/- 0.0005. Per-seed
significance (`paired_gain_over_standard_error`) is many-sigma in every one of the 8
family/template combinations: 5.9-103.5 sigma for C3K, 17.9-102.7 sigma for LW02 — a
robust detection regardless of family or template.

The `log10(tau_q)` target is where the SFH shape genuinely changes the answer, not just
its size. For `truncation` the gain is consistent with zero in *both* templates
(per-seed gain-over-SE -0.2 to +2.1 sigma; mean gain +0.0002 dex, C3K and LW02 alike) —
a hard cutoff's post-quench SFR is identically zero and carries no `tau_q` information
for the bump, or any other index, to recover, beyond what D4000/HdeltaA's own age clock
already gives. This is a genuinely new, family-specific finding not visible in the
Phase 4 single-family (`exponential`) test. `exponential`, `linear` and `decoupled`
(whose post-quench SFR does depend on `tau_q`) all show a real, LW02-only `tau_q` gain:
-0.0129 +/- 0.0002 (exponential), -0.0043 +/- 0.0003 (linear), -0.0098 +/- 0.0006
(decoupled) dex, versus C3K gains of -0.0002 to -0.0004 dex that are each consistent
with zero (per-seed gain-over-SE -4.4 to +0.1 sigma for exponential, i.e. -0.33, +0.08,
-4.37 across the three seeds; -0.8 to -1.5 for linear, -1.3 to -1.7 for decoupled).

**The caveat that applies to both Q2 and Conclusion 2 above: S4 and S5 carry no `agb0`
control per family.** The Phase 3/4 finding that the small C3K gain is *not*
TP-AGB-specific rests on one explicit `agb0`-control run: for Q2, the C3K agb0 change in
completeness/purity is the same size as the agb2 change (docs/ANALYSIS.md, Q2 evidence
above); for Conclusion 2, the C3K agb0 control gives a gain of the same size as C3K agb2
on *both* SFH-recovery targets (-0.0063 +/- 0.0004 vs -0.0039 +/- 0.0003 dex for
`log10(time since quenching)`, -0.0010 +/- 0.0002 vs -0.0002 +/- 0.0003 dex for
`log10(tau_q)`; `output/analysis/conclusion_summary.json`,
`c2_sfh_recovery.summary."C3K agb0 (control)".bump_0.005.paired_gain_mean_over_seeds_dex`
versus `c2_sfh_recovery.summary."C3K agb2".bump_0.005.paired_gain_mean_over_seeds_dex`,
the "C3K agb0 (control, no TP-AGB light)" row in the Conclusion 2 table above). Both
controls were measured once, for the `exponential` family only; neither was rerun for
`linear`, `truncation` or `decoupled` in this task. What Phase 5 therefore shows is that the
*qualitative pattern* replicates across all four families — LW02 gain positive and
multi-sigma, C3K gain small or sign-inconsistent — while the *attribution* of that
pattern to TP-AGB light specifically, rather than to any third noisy feature, still
rests entirely on the Phase 4 agb0 control measured for the `exponential` family alone.

### What changed and what did not

The population-level bump-vs-D4000 relationship (S2, agb2) is nearly family-invariant
for **C3K** everywhere: the spread across the four families' 20-bin band medians is at
most 0.0028 mag at any D4000 (`sfh_sensitivity_summary.json`,
`s2_population_bands.c3k.<family>.band_median_mag`). For **LW02** the same statement
needs a qualification, not the unqualified "nearly family-invariant" used for C3K: the
four families' band medians are nearly identical only in the single lowest bin (D4000 =
1.185, spread 0.0004 mag); the spread already reaches 0.0017-0.0028 mag in the next two
bins (D4000 = 1.257, 1.329) and keeps growing, peaking at 0.0178 mag at the D4000 =
1.543 bin (exponential -0.0568, linear -0.0672, truncation -0.0746, decoupled -0.0569
mag), before falling back to 0.0002-0.0012 mag from D4000 ~ 1.83 onward
(`s2_population_bands.lw02.<family>.band_median_mag`). In this D4000 ~ 1.5-1.8 range
`truncation` and `linear` sit systematically stronger (more negative) than
`exponential` and `decoupled`. Using the same three D4000 slices as Q1/Conclusion 1
(`s2_population_bands.lw02.<family>.d4000_slices`), the [1.5, 1.7) slice shows the same
pattern: exponential -0.0558, linear -0.0657, truncation -0.0713, decoupled -0.0557 mag
(`...d4000_1.5_1.7.median_mag`) — linear about 0.010 mag and truncation about 0.016 mag
stronger than exponential/decoupled. This tracks the class-fraction shift above: the
rapid-quenching *locus* itself — where the density of rapid-quenching epochs peaks —
sits at essentially the same place for all four families (D4000 1.625-1.646, per-family
`d4000_median` 1.6297/1.6463/1.6247/1.6310 for exponential/linear/truncation/decoupled;
bump -0.0739 to -0.0756 mag, `s2_population_bands.lw02.<family>.rq_locus_centroid`, and
visually the same in `s2_population_bands.png`'s dotted density contours), but
`truncation` and `linear` have 3-5x more rapid-quenching epochs overall (S3, above) to
pull the local median down through that same locus.

What did change with the SFH family: the class fractions (0.84-6.38 percent
rapid-quenching, S3), and the absolute classifier/regression numbers they feed — the
no-bump classifier completeness baseline alone spans 0.319-0.846 across families (S4),
and the LW02 gain's absolute size varies by roughly a factor of 2 between families (S4,
S5). What did not change: which template (C3K or LW02) gives a real, multi-sigma bump
gain in the classifier and the SFH-recovery regression — the answer is LW02 in all four
families — and, except for the specific D4000 ~ 1.5-1.8 / LW02 exception above, the
shape of the population bump-vs-D4000/HdeltaA locus itself.

### Figures (`output/analysis/`)

- `s1_fiducial_families.png`: the fiducial history's SFR and the three indices versus
  time since quenching, 4 families x 2 templates.
- `s2_population_bands.png`: the population's 16-84 bump-vs-D4000 band per family
  (agb2), with the rapid-quenching density contours.
- `s3_class_fractions.png`: class fractions per family, C3K and LW02 panels.
- `s4_classifier_by_family.png`: classifier completeness/purity gain (bump at 0.010 mag
  minus optical-only) per family, C3K versus LW02.
- `s5_recovery_by_family.png`: SFH-recovery RMS gain (bump minus no-bump) per family, at
  0.005 and 0.010 mag, for `log10(time since quenching)` and `log10(tau_q)`.
- `s6_tpagb_offset_by_family.png`: the population's bump(agb2) - bump(agb0) offset
  versus D4000 per family and template, plus the C3K-versus-LW02 band-median
  separation per family.

## Caveats

- In the default configuration, O-rich TP-AGB stars have hydrostatic C3K model
  spectra. Real TP-AGB stars are pulsating, extended, and dusty, with H2O absorption
  that hydrostatic models underpredict. The LW02 empirical spectra give a bump about 5
  times larger. The Q1 answer and the bump part of Q2 depend on which template is
  right, and this analysis cannot decide that. The Phase 4 conclusion figures make the
  same point a second way: the population bump locus moves several times more when the
  TP-AGB *template* is swapped than when the TP-AGB *weight* is doubled within either
  template ("Supporting figures for the two conclusions", Conclusion 1 above), and the
  SFH-recovery gain from adding the bump (Conclusion 2 above) is real for LW02 and not
  clearly distinguishable from a no-TP-AGB control for C3K. This C3K-versus-LW02 split
  is not an artifact of the one delayed-tau-plus-exponential-quench SFH assumed
  everywhere else: the population TP-AGB offset itself (Q1/Conclusion 1) was directly
  retested per family, since every population run already carries `agb0`/`agb1`/`agb2`
  columns, and the split holds in every family ("Sensitivity to the star formation
  history model" above, "The population TP-AGB offset per family"); it also replicates
  in the rapid-quenching classifier and the SFH-recovery regression (Q2/Conclusion 2)
  for three alternative post-quench SFH shapes. The one part not rerun per family is
  the no-TP-AGB (`agb0`) *control* of that classifier/regression gain, which was
  measured once, for the original `exponential` family only.
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
  metallicity spread within galaxies would widen the bump planes. The Phase 4
  metallicity-only figure quantifies this for the post-quench bump specifically
  ("Supporting figures for the two conclusions", "Metallicity as a confounder"
  above): the metallicity spread is comparable to the TP-AGB delta for C3K (a genuine
  confounder) and 2-4 times smaller than it for LW02 (a secondary systematic), and the
  sign of the metallicity trend is opposite between the two templates.
- The population weights every epoch of every history equally, from 1 Gyr to 13 Gyr.
  Class fractions (1.23 percent rapid-quenching) are not those of a real sample, and
  purities scale with the class mix. This is also assumption-dependent: holding the
  weighting scheme fixed and instead changing only the assumed post-quench SFR shape
  moves the rapid-quenching base rate by a factor of 7.6, from 0.84 percent
  (`decoupled`) to 6.38 percent (`truncation`) ("Sensitivity to the star formation
  history model" above).
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
