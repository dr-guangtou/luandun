# Phase 3 — Diagnostic analysis: answers

All numbers below are computed by the scripts; each is followed by the file (under
`output/`) and the key it comes from. `a.b.c` denotes nested JSON keys.

## Setup

Models: MIST isochrones with the C3K spectral library in FSPS (Kroupa IMF, no nebular
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

## Q1. Can a model with TP-AGB contribution be told apart from one without?

**Answer: conditional on the TP-AGB templates. With the default C3K templates, no: agb 0
versus 2 moves the bump by at most 0.024 mag for SSPs and by at most 0.009 mag at fixed
D4000 or HdeltaA in the population, below the 0.01 mag precision in every bin. With the
empirical LW02 templates, yes: the population offset is 0.032-0.059 mag and 1.7-1.9 times
the intrinsic scatter in every bin. The optical indices cannot tell them apart in either
configuration.**

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
- Optical indices: max |Delta D4000| = 0.034 (`step1.max_abs_delta_d4000`), below the
  0.05 precision; max |Delta HdeltaA| = 0.47 A (`step1.max_abs_delta_hdelta_a_angstrom`),
  below 0.5 A. Both peak at about 1 Gyr, log Z = -0.5.
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
- Population (12 percentile bins of D4000 and, separately, of HdeltaA): the median
  paired offset is at most 0.0092 mag (r100, D4000 bin 1.32) and at least 0.0026 mag
  (`analysis/q1_summary.json`,
  `step6.bump_{sigma300,r100}_by_{d4000,hdelta_a}_agb0_median_diff_mag`). In units of
  the pooled 16-84 half-width it reaches 1.35 (`step6.max_offset_over_scatter`, source
  `bump_r100_by_hdelta_a_agb0`). So the offset is about as large as the intrinsic
  (mostly metallicity) scatter, but it never reaches 0.01 mag. It passes the answer
  rule in 0 of 48 bins.

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
- Optical indices remain inseparable: max |Delta D4000| = 0.022, max |Delta HdeltaA| =
  0.31 A (`step1.max_abs_delta_d4000`, `step1.max_abs_delta_hdelta_a_angstrom`).
- Fiducial track: -0.0736 mag (sigma300) and -0.0733 mag (r100) at 3.65 Gyr, 0.65 Gyr
  after quenching (`single_csp_lw02/indices.csv`, same columns).
- Population: the median paired offset is 0.032-0.059 mag in all 48 bins, 1.72-1.90
  times the pooled half-width (`lw02_q1_summary.json`, `step6.bump_*_median_diff_mag`,
  `step6.bump_*_offset_over_scatter`; `step6.max_offset_over_scatter` = 1.90). It passes
  the answer rule in 48 of 48 bins.

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
  bands for agb0 and agb2 at fixed D4000 and at fixed HdeltaA.

The fiducial-track figures are `output/single_csp/time_evolution.png` and
`output/single_csp_lw02/time_evolution.png`.

## Q2. With agb = 2, can the fast-quenching population be isolated?

**Answer: yes, but in the optical D4000-HdeltaA plane, and TP-AGB light gives little or
no help. That plane puts 77-81 percent of rapid-quenching epochs in cells more than
50 percent pure, with or without TP-AGB. None of the 144,600 bursty and slowly fading
contaminant epochs falls in those cells. The bump adds nothing with the default
templates. With the LW02 templates it adds a modest gain: at 0.01 mag precision,
completeness rises from 0.41 to 0.46 and purity from 0.61 to 0.65. With either
template set, the bump planes alone isolate at most 29 percent.**

Classes follow the manuscript's sSFR rules. They depend only on the SFH, so the counts
are the same in both configurations: 4,727 rapid-quenching epochs out of 482,000,
including 2,845 post-starburst (`analysis/q2_summary.json`, `class_summary.counts`,
`class_summary.post_starburst_count`; identical in `lw02_q2_summary.json`).

Evidence, purity maps (40 x 40 cells between the 0.5 and 99.5 percentiles; a cell needs
20 epochs; "isolable" = fraction of rapid-quenching epochs in cells with purity > 0.5):

| plane | default agb2 | default agb0 | LW02 agb2 | LW02 agb0 |
| --- | ---: | ---: | ---: | ---: |
| D4000-HdeltaA | 0.807 | 0.810 | 0.768 | 0.810 |
| D4000-bump | 0.077 | 0.182 | 0.134 | 0.182 |
| HdeltaA-bump | 0.085 | 0.007 | 0.030 | 0.007 |

(`q2_summary.json` and `lw02_q2_summary.json`,
`purity_maps.<agb>.<plane>.isolable_fraction`.)

In LW02 agb2 the deepest bump values fall outside the percentile grid: 29.6 percent of
rapid-quenching epochs lie outside it in the D4000-bump plane
(`analysis/lw02_q2_alternative_summary.json`,
`purity_maps.agb2.d4000_h_minus_bump.rapid_quenching_fraction_outside_grid`), so 0.134
undercounts. On a grid spanning the full range, the isolable fraction is 0.291 for
D4000-bump and 0.127 for HdeltaA-bump, against 0.792 for D4000-HdeltaA
(`lw02_q2_alternative_summary.json`,
`purity_maps_full_range.agb2.<plane>.isolable_fraction_before`). For the default
configuration the full-range values are 0.079, 0.081 and 0.782
(`q2_alternative_summary.json`, same keys). The bump planes stay far behind the
optical plane in both configurations.

Evidence, noise-aware k-nearest-neighbour classifier (k = 25, 5 folds grouped by history,
D4000 +/- 0.05 and HdeltaA +/- 0.5 A noise, bump at sigma300; mean +/- fold-to-fold std;
`classifier.results.agb2.<feature set>.unbalanced.{completeness,purity}_{mean,std}`):

| agb2, unbalanced | default completeness | default purity | LW02 completeness | LW02 purity |
| --- | --- | --- | --- | --- |
| D4000, HdeltaA only | 0.418 +/- 0.016 | 0.617 +/- 0.028 | 0.413 +/- 0.017 | 0.613 +/- 0.025 |
| + bump +/- 0.005 mag | 0.406 +/- 0.012 | 0.630 +/- 0.038 | 0.490 +/- 0.036 | 0.668 +/- 0.028 |
| + bump +/- 0.010 mag | 0.406 +/- 0.014 | 0.628 +/- 0.026 | 0.460 +/- 0.025 | 0.650 +/- 0.025 |
| + bump +/- 0.020 mag | 0.402 +/- 0.017 | 0.622 +/- 0.020 | 0.422 +/- 0.027 | 0.628 +/- 0.021 |

- Default: every with-bump value is within the baseline's fold-to-fold scatter, for
  both bump products, both agb settings and the class-balanced training variant
  (`q2_summary.json`, `classifier.results`). The bump does not help.
- LW02: at 0.005 and 0.010 mag the completeness gain (+0.077, +0.047) exceeds the
  baseline std (0.017). The purity gain (+0.055, +0.037) is 2.2 and 1.5 times the
  baseline std (0.025). At 0.020 mag both gains are within the scatter. Class-balanced
  training shows the same pattern: purity 0.211 +/- 0.012 rises to 0.238 +/- 0.012 at
  0.010 mag (`lw02_q2_summary.json`, `classifier.results.agb2.<set>.balanced`). With
  agb0 there is no gain, as expected.
- In every configuration, rapid-quenching epochs are confused with transitional and
  quiescent epochs, essentially never with star-forming ones (the `confusion` matrices
  in the same files).

Evidence, alternative SFH families as contaminants. Two families of 300 histories each
(seed 20260925, all 260 epochs, same integrator and class rules;
`analysis_alternative_sfh.py`):

- bursty star-forming: no quench, plus a 0.1 Gyr Gaussian burst at 2-10 Gyr that adds
  10 percent of the final mass;
- slowly fading: tau_q 3-6 Gyr.

Results:

- Neither family produces a rapid-quenching epoch. Bursty: 22,346 star-forming, 49,954
  transitional. Slowly fading: 15,368 star-forming, 56,281 transitional, 651 quiescent
  (`q2_alternative_summary.json`, `contaminant_classes`; identical for LW02).
- Purity maps: 0 contaminant epochs fall in any cell that had purity > 0.5, in every
  plane, for agb2 and agb0, in both configurations. Pooled purity and isolable fractions
  are therefore unchanged: D4000-HdeltaA agb2 pooled purity is 0.841 (default) and
  0.852 (LW02) before and after
  (`purity_maps.<agb>.<plane>.n_contaminant_epochs_in_pure_before_cells`,
  `pooled_purity_before`, `pooled_purity_after` in `q2_alternative_summary.json` and
  `lw02_q2_alternative_summary.json`; same result in `purity_maps_full_range`).
- Classifier trained on the delayed-tau population, applied to the 144,600 noised
  contaminant epochs (`classifier.results.<agb>.<set>.<balance>`):
  - Unbalanced training labels at most 13 contaminant epochs as rapid-quenching in
    either configuration (`n_contaminant_false_rapid_quenching`; a rate of at most
    0.01 percent).
  - Class-balanced training (optical only, agb2) mislabels 1,622 (default) and 1,676
    (LW02) epochs, about 1.1 percent, mostly slowly fading. This lowers the pooled
    purity from 0.211 (default) and 0.210 (LW02) to 0.195 in both
    (`pooled_purity_delayed_tau_only`, `pooled_purity_with_contaminants`).
  - With the LW02 bump at 0.005 mag, the balanced mislabel count drops to 387 (0.27
    percent), and pooled purity goes from 0.264 to 0.258
    (`lw02_q2_alternative_summary.json`,
    `classifier.results.agb2.bump_sigma300_0.005.balanced`).
  - These counts scale with the arbitrary 600 : 2000 mix of contaminant to delayed-tau
    histories; the per-epoch rates do not.

Figures (`output/analysis/`):

- `q2_class_planes.png`, `lw02_q2_class_planes.png`: the four classes in the three
  planes, for agb2 and agb0.
- `q2_purity_maps.png`, `lw02_q2_purity_maps.png`: rapid-quenching purity per cell, with
  the rapid-quenching density contours.
- `q2_classifier.png`, `lw02_q2_classifier.png`: completeness and purity vs bump
  precision, against the no-bump band.
- `q2_alternative_sfh.png`, `lw02_q2_alternative_sfh.png`: the two contaminant families
  on top of the rapid-quenching epochs (agb2; bursty coloured by time since burst), then
  the purity maps with contaminants added. Red outlines mark the cells that had purity
  > 0.5 before.

## Caveats

- In the default configuration, O-rich TP-AGB stars have hydrostatic C3K model
  spectra. Real TP-AGB stars are pulsating, extended, and dusty, and their 1.6 micron
  feature is not captured by hydrostatic models. The LW02 empirical spectra give a bump
  about 5 times larger. The Q1 answer and the bump part of Q2 depend on which template
  is right, and this analysis cannot decide that.
- MIST has no C-rich TP-AGB stars (the C-star templates are never used), so the
  carbon-star contribution at 1-2 Gyr and low Z is missing from both configurations.
- No nebular emission and no dust (neither interstellar nor circumstellar). Emission
  fill-in of HdeltaA in the bursty family, and dust reddening, would both move points
  in the optical plane.
- Four metallicities, one Z per history, interpolated linearly in log Z. At fixed
  D4000 the population bump scatter is mostly metallicity (Phase 2 review), so a
  metallicity spread within galaxies would widen the bump planes.
- The population weights every epoch of every history equally, from 1 Gyr to 13 Gyr.
  Class fractions (0.98 percent rapid-quenching) are not those of a real sample, and
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

For Q1, the default-template answer would flip only with a bump precision better than
about 0.005 mag on galaxies whose metallicity is known. The largest population offset
is 0.009 mag at fixed D4000, about equal to the intrinsic spread. A TP-AGB template with
a deep intrinsic 1.6 micron feature, like LW02 (component bump about -0.2 mag instead of
-0.03 mag), makes the SSP and track differences 5-7 times larger and detectable at
0.01-0.02 mag. Deciding between C3K and LW02 is therefore the key empirical question. It
needs resolved spectroscopy of intermediate-age clusters or post-starburst galaxies at
1.5-1.8 micron.

For Q2, the isolation of fast quenching rests on D4000 and HdeltaA and would survive any
TP-AGB model. The bump would matter only with LW02-like templates and a precision of
0.01 mag or better, and even then it would add only about 5 points of completeness and
4 points of purity.
