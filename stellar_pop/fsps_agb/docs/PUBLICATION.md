# Publication figure candidates (Phase 6, 2026-09-24)

`publication_figures.py` writes the figures below, as PDF and PNG, to
`output/publication/`, with every plotted number in
`output/publication/publication_summary.json` (keyed per figure). `--only fig1,fig3`
draws a subset; `--pilot` runs the whole script on stride-50 tables with one noise seed
and two folds into `output/publication/pilot/` (35 s). The full run takes 10 min, almost
all of it in the classifier and recovery sweeps for figures 4 and 6; `--reuse-gains`
redraws those figures (and figure 7) from the sweeps stored in the summary JSON by an
earlier full run.

## Final selection (`output/publication/final/`, 2026-09-25)

`publication_figures.py --final` writes the three figures kept for the paper, named by
content, with their numbers in `final/final_summary.json`:

| File | Content | Origin |
| --- | --- | --- |
| `index_planes_agb_on_off` | Figure 1 below, main text | identical to `fig1_index_planes` |
| `index_planes_sfh_families` | Figure 5 below, appendix | identical to `fig5_robustness_planes` |
| `age_sensitivity_and_classifier_gain` | new two-part figure | Figure 3 (a, b) and Figure 4 (a, b) recombined |
| `d4000_hminus_plane_with_jwst_data` | the D4000 versus H-minus bump plane, TP-AGB off and on, with the JWST z ~ 1 quiescent sample | Figure 1 panels (b) and (e) plus `final/JWST_QG_indices.npz` |
| `observed_stack_vs_fsps_mocks` | the S/N-weighted stack of the 19 JWST spectra against R = 100 mock spectra at the cosmic age of the median redshift (tau_q varied at fixed t - t_q, solar Z) and the closest model in a search over metallicity, t_q and tau_q, TP-AGB off and on, drawn as steps on the observed pixel scale | `jwst_spectrum_comparison.py`, `final/qg_spec/` |

The combined figure: (a-c) one panel per tau_q = 0.3, 1, 3 Gyr at t_q = 3 Gyr, TP-AGB on,
solar metallicity, sharing the log10(t - t_q) axis, with HdeltaA (blue), the H-minus
bump strength (vermilion, minus the bump index) and D4000 (green), each track scaled to
[0, 1] over 0.05 < t - t_q < 6 Gyr so the three indices share one axis (the unscaled
ranges are in `final_summary.json`). Stars mark the HdeltaA and H-minus bump peaks: 0.25
and 0.8 Gyr after t_q at tau_q = 0.3 Gyr, 0.2 and 1.25 Gyr at 1 Gyr, and at 3 Gyr the
HdeltaA maximum sits at the window edge (open star) while the bump still peaks at 1.5
Gyr; D4000 only rises. (d, e) Rapid-quenching completeness and purity against H-minus
bump precision for TP-AGB on, with the D4000-plus-HdeltaA baseline as a dashed line and
its fold standard error as a band. No significance labels: the gain is consistent
across folds but moderate in absolute terms, which is the honest reading for a
three-feature test. Figures 2, 6 and 7 and the recovery panels of Figure 4 are kept as
candidates only.

The JWST figure: the same layers as Figure 1 (population contours, rapid-quenching class,
fiducial-SFH metallicity tracks) in the D4000 versus H-minus bump plane, TP-AGB off on the
left and TP-AGB on on the right, on one shared bump axis, with the 19 quiescent galaxies at
z = 1.0 to 2.0 from `JWST_QG_indices.npz` (Lu+2026 in the legend) overplotted (D4000 bars are the 16th to 84th
percentile bounds stored in the file, bump bars the stored symmetric error, mostly
smaller than the marker). The data occupy D4000 = 1.35 to 1.94 and bump = -0.037 to
-0.073 mag (median -0.059 mag). Over that D4000 range the TP-AGB-on population spans
-0.081 to -0.041 mag (1st to 99th percentile) and the TP-AGB-off population -0.021 to
+0.003 mag, so every galaxy lies inside the TP-AGB-on locus and 0.02 to 0.05 mag outside
the TP-AGB-off one.

Legend entries are treated as sentences: `save_figure` capitalises the first letter of
every legend label and title, skipping panel references such as "(d, e)" and anything
that starts inside LaTeX math.

## Conventions

| item | choice | reason |
| --- | --- | --- |
| TP-AGB on | LW02 empirical O-rich TP-AGB templates (`use_lw_tpagb = 1`), `agb = 2` | the configuration every Phase 3 to 5 test used as the "TP-AGB on" case |
| TP-AGB off | default C3K templates, `agb = 0` | the no-TP-AGB control of the Phase 4 recovery test; C3K at `agb = 1` differs by only 0.005 mag and is shown in Figure 2 |
| optical indices | D4000 and HdeltaA on the sigma = 300 km/s product | manuscript convention |
| H-minus bump | R = 100 product (`h_minus_bump_r100_*`) | manuscript convention; differs from the sigma300 bump by at most 0.001 mag |
| SFH | the paper's tau = t_q exponential-quench family (`output/population`, `_lw02`) | Figures 5 and 6 repeat the tests for the linear, truncation and decoupled families |
| noise | D4000 0.05, HdeltaA 0.5 A, bump 0.005 / 0.010 / 0.020 mag | Phase 3 yardsticks |
| classes | Zhang+2023-style sSFR rules in `population_classes.py` | 5907 rapid-quenching epochs of 482,000 in the exponential family |
| significance | paired per-fold gain on the same grouped folds, standard error = ddof-1 std over 5 folds / sqrt(5), averaged over 3 noise seeds | same definition as Phases 4 and 5 |

The classifier and recovery gains at R = 100 were recomputed here (Phases 4 and 5 quoted
the sigma300 bump). They agree with the Phase 3 R = 100 classifier numbers in
`lw02_q2_summary.json` to within the seed scatter.

## Message 1: the bump diagnoses the TP-AGB contribution

### Figure 1, `fig1_index_planes`

Two rows (TP-AGB off, TP-AGB on) of the three index planes for the exponential family, all
epochs later than 1 Gyr. Grey filled contours enclose 68, 95 and 99.5 per cent of all
epochs; the red filled contour is the rapid-quenching class (68 and 95 per cent); the
light blue curves are the fiducial history (t_q = 3 Gyr, tau_q = 0.3 Gyr) at log Z =
-0.5, -0.25, 0, +0.25 from t_q to t_q + 5 Gyr, with markers at t_q and 0.5, 1, 2, 5 Gyr
later. The row titles sit in the empty top-right corner of the D4000-HdeltaA panels;
all text is Computer Modern (usetex) and the legend carries the fiducial SFH parameters.

What it shows: the D4000-HdeltaA plane is identical in the two rows. In the two bump
planes the TP-AGB-on population sits 0.04 to 0.08 mag deeper than the TP-AGB-off one and its
rapid-quenching class occupies the deepest part of the plane
(`fig1_index_planes.agb_on.rapid_quenching`: median -0.075 mag, 16-84 range -0.083 to
-0.066 mag), while with TP-AGB off the class is confined to a 0.007 mag wide band near zero
(median -0.004 mag). The bump reverses its metallicity dependence between the two rows
(see Figure 7).

One design change from the handover brief: the rapid-quenching class cannot be
coloured by quenching timescale, because the class rule (recent-to-previous sSFR ratio
below 0.1) admits only tau_q < 0.3 Gyr histories in this family
(`fig1_index_planes.agb_on.rapid_quenching_tau_q_range_gyr`). A first version coloured
the epochs within 2 Gyr of t_q by tau_q instead; those contours added no information
beyond Figure 3 and crowded the panels, so they were dropped on review.

### Figure 2, `fig2_bump_offsets`

The population bump versus D4000 (a) and HdeltaA (b) as medians with 16-84 bands for
four prescriptions, plus the bump histogram in the D4000 slice [1.3, 1.5) (c). Median
offsets between neighbouring prescriptions in that slice
(`fig2_bump_offsets.slice.neighbouring_separations`):

| step | what changes | delta median [mag] | over wider 16-84 half width | over 0.01 mag |
| --- | --- | ---: | ---: | ---: |
| C3K agb 0 to agb 1 | TP-AGB weight, C3K template | -0.0052 | 1.1 | 0.5 |
| C3K agb 1 to LW02 agb 1 | template at fixed weight | -0.0266 | 6.4 | 2.7 |
| LW02 agb 1 to agb 2 | TP-AGB weight, LW02 template | -0.0234 | 3.9 | 2.3 |

The population bump locus therefore tests the TP-AGB spectral template far more
sharply than the TP-AGB weight, and doubling the weight inside the hydrostatic C3K
configuration stays below the 0.01 mag yardstick.

## Message 2: with TP-AGB on, the bump adds SFH information and, with HdeltaA, times the quenching

### Figure 3, `fig3_quenching_clocks`

(a) HdeltaA and (b) the bump versus time since quenching for tau_q = 0.1, 0.3, 1, 3 Gyr
at t_q = 3 Gyr, solar metallicity; TP-AGB off drawn dotted in (b). (c) The HdeltaA-bump
plane for the same tracks with markers every 0.5 Gyr. (d) The delay of each extremum
after t_q against tau_q for t_q = 1.5, 3, 4.5 Gyr.

HdeltaA peaks 0.2 to 0.3 Gyr after t_q whenever it peaks at all; for tau_q above about
0.6 Gyr (t_q = 3 or 4.5 Gyr) it has no interior maximum, it simply declines
(`fig3_quenching_clocks.lag_grid.t_q_3.hdelta_a_peak_at_window_edge`). The TP-AGB-on bump
reaches its minimum (deepest) later, and the delay grows monotonically with tau_q: 0.5
Gyr at tau_q = 0.1 Gyr, 0.8 Gyr at 0.3 Gyr, 1.25 Gyr at 1 Gyr, 1.5 Gyr at 3 Gyr for t_q
= 3 Gyr, with t_q shifting the curve by at most 0.3 Gyr
(`fig3_quenching_clocks.lag_grid.*.bump_minimum_delay_gyr`). The TP-AGB-off bump has no
interior extremum for any tau_q (`...tau_q_family.*.agb_off.bump_minimum_at_window_edge`
all true). In (c) the tracks trace loops whose width grows with tau_q, which is the
geometric reason the pair constrains the quenching timescale.

### Figure 4, `fig4_information_gain`

Rapid-quenching completeness (a) and purity (b) of the grouped k = 25 nearest-neighbour
classifier, and the RMS error of the nearest-neighbour recovery of log10(t - t_q) (c)
and log10(tau_q) (d) on 240,000 post-quench epochs, versus bump precision, for both AGB
configurations; dashed lines and bands are the optical-only baselines. Labels give the
paired gain in fold standard errors (`fig4_information_gain.<config>.<kind>.<key>`):

| quantity | optical only | TP-AGB on, +bump 0.005 mag | TP-AGB on, +bump 0.010 mag | TP-AGB off, +bump 0.005 mag |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.398 | 0.488 (+0.090, 11 sigma) | 0.445 (+0.047, 6 sigma) | 0.402 (+0.006, 1 sigma) |
| purity | 0.610 | 0.649 (+0.039, 7 sigma) | 0.632 (+0.021, 8 sigma) | 0.622 (+0.012, 2 sigma) |
| RMS log10(t - t_q) [dex] | 0.240 | 0.222 (-0.018, 49 sigma) | 0.231 (-0.008, 28 sigma) | 0.234 (-0.006, 15 sigma) |
| RMS log10(tau_q) [dex] | 0.311 | 0.298 (-0.013, 21 sigma) | 0.305 (-0.006, 17 sigma) | 0.310 (-0.001, 6 sigma) |

With TP-AGB off the bump adds essentially nothing to the classifier (its completeness gain
is negative at 0.010 and 0.020 mag) and a small amount to the age recovery, which is the
age information any 1.6 micron continuum index carries. The tau_q recovery gain is an
TP-AGB-on effect: 0.013 dex against 0.001 dex.

## Robustness to the SFH family

### Figure 5, `fig5_robustness_planes`

Figure 1's TP-AGB-on row for the linear, truncation and decoupled families, with the
population and rapid-quenching contours only (no metallicity tracks). The
rapid-quenching class lands in the deep-bump region in every family.

### Figure 6, `fig6_robustness_gains`

Per family, the paired gain from the bump at 0.010 mag in completeness (a) and purity
(b), and the RMS reduction at 0.005 mag for log10(t - t_q) (c) and log10(tau_q) (d), AGB
on against off (`fig6_robustness_gains.<family>.<config>`):

| family | TP-AGB-on delta completeness | TP-AGB-on delta purity | TP-AGB-on RMS gain t - t_q [dex] | TP-AGB-on RMS gain tau_q [dex] | TP-AGB-off RMS gain tau_q [dex] |
| --- | ---: | ---: | ---: | ---: | ---: |
| exponential | +0.047 +/- 0.008 | +0.021 +/- 0.003 | 0.018 | 0.013 | 0.001 |
| linear | +0.038 +/- 0.004 | +0.022 +/- 0.003 | 0.017 | 0.004 | 0.001 |
| truncation | +0.026 +/- 0.003 | +0.010 +/- 0.002 | 0.031 | n/a | n/a |
| decoupled | +0.052 +/- 0.010 | +0.036 +/- 0.006 | 0.010 | 0.010 | 0.001 |

The TP-AGB-off completeness gain is within 2 standard errors of zero or negative in every
family. The optical-only baselines differ a lot between families (completeness 0.32 to
0.85) because the class base rate does (1.2 to 6.4 per cent), so the deltas, not the
absolute values, are comparable.

## Metallicity: the FSPS facts, the caveat and the advantage

### What the FSPS code does with TP-AGB stars (read from `getspec.f90`, `sps_setup.f90`, `sps_vars.f90`, `mod_gb.f90`, `add_agb_dust.f90` at commit `bd187a0`)

- With `use_lw_tpagb = 0` (the TP-AGB-off template), a phase-5 star falls through to the
  main library branch (`getspec.f90` lines 150 to 200): bilinear interpolation in
  log Teff and log g of the C3K grid slice at the star's own metallicity index
  `pset%zmet`, clamped at the grid edge (2500 K, log g = -1). The C3K templates are
  therefore fully metallicity dependent.
- With `use_lw_tpagb = 1`, an O-rich phase-5 star with log Teff < 3.6 takes the branch at
  lines 90 to 102 instead: a linear interpolation in Teff only, with no log g and no
  metallicity index, of the nine Lancon & Mouhcine (2002) `Orich.spec` spectra
  (`agb_spec_o(nspec, 9)`), scaled by the star's bolometric luminosity. The one Z-aware
  ingredient is the Teff label table `agb_logt_o(nz, 9)`, built in `sps_setup.f90` lines
  438 to 461 by interpolating `Orich.teff` (22 columns in log Z from -1.98 to +0.20) to
  each isochrone metallicity, without clamping, so log Z = +0.25 is extrapolated. The
  spectral shapes never change with Z; only which template a star of a given Teff blends.
- The C-rich branch (Aringer et al. 2009 templates, Teff only, no Z) is reachable only
  for C/O > 1, which in the MIST tables occurs at [Fe/H] <= -2, so it never activates in
  this grid (log Z >= -0.5).
- For MIST no metallicity-dependent TP-AGB normalisation is applied (`mod_gb.f90` guards
  the Conroy & Gunn 2010 corrections with `isoc_type == 'pdva'`); `agb` is a
  Z-independent multiplicative weight. The circumstellar dust model receives `zz` but
  its only Z term is commented out.
- What does depend on Z is the isochrone: the coolest phase-5 log Teff is 3.412 at
  [Fe/H] = +0.5, 3.456 at 0.0 and 3.533 at -1.0, so the fraction of TP-AGB stars below
  the fixed log Teff = 3.6 switch, hence the fraction assigned the LW02 template, rises
  steeply with Z.

### Figure 7, `fig7_metallicity`

(a) The bump 1 Gyr after t_q for the fiducial history against log Z, decomposed into
the non-AGB stars (C3K, agb = 0), the full TP-AGB-on model, and the TP-AGB increment
(agb = 2 minus agb = 0) of each template (`fig7_metallicity.tracks.*.t_q+1_gyr`):

| component | log Z = -0.5 | 0.0 | +0.25 | slope [mag per dex] |
| --- | ---: | ---: | ---: | ---: |
| non-AGB stars (C3K agb 0) | -0.0075 | -0.0044 | +0.0004 | +0.011 |
| TP-AGB increment, C3K | -0.0105 | -0.0096 | -0.0088 | +0.003 |
| TP-AGB increment, LW02 | -0.0478 | -0.0698 | -0.0844 | -0.049 |
| TP-AGB on total (LW02 agb 2) | -0.0552 | -0.0742 | -0.0840 | -0.038 |

So the TP-AGB-off bump weakens slowly with metallicity, driven by the non-AGB stars (the
C3K increment is flat in Z), whereas the TP-AGB-on bump deepens with metallicity four times
faster and in the opposite direction, and that trend is entirely the LW02 increment.
Since the LW02 spectra carry no metallicity dependence, this trend is an isochrone plus
Teff-cut effect: at higher Z more TP-AGB stars are cool enough to receive the empirical
template, and each receives a slightly cooler template through `agb_logt_o`. The
non-AGB stars contribute a small positive slope that partly cancels it. Any metallicity
correction to the bump is therefore only as good as (i) the MIST TP-AGB Teff
distribution and (ii) the hard log Teff = 3.6 switch, and it changes sign between the
two template configurations, which the paper must state.

(b) In the TP-AGB-on population, the bump against log Z inside D4000 in [1.4, 1.7): all
epochs (grey) and the rapid-quenching class (blue). Within the slice the bump follows
log Z with slope -0.031 mag per dex (all) and -0.047 mag per dex (rapid quenching);
removing the linear trend shrinks the rapid-quenching 16-84 half width from 0.0085 to
0.0049 mag, i.e. 40 per cent of the class's bump scatter at fixed D4000 is metallicity
(`fig7_metallicity.population_slice`). The class stays offset from the rest of the slice
at every metallicity, by 0.013 mag (median difference at log Z in [-0.5, -0.3)) rising to
0.024 mag (log Z in [0.1, 0.25]), measured directly from the tables.

(c) The bump's paired gain, as a fraction of its baseline, with and without a known
metallicity (log Z with 0.1 dex noise added to the optical features and to the bump
set), TP-AGB on, exponential family (`fig7_metallicity.known_z`):

| gain from the bump | Z unknown | Z known (0.1 dex) |
| --- | ---: | ---: |
| completeness | +11.8 +/- 2.1 per cent | +13.0 +/- 1.8 per cent |
| purity | +3.5 +/- 0.5 per cent | +3.8 +/- 0.7 per cent |
| RMS log10(t - t_q) | 7.6 +/- 0.2 per cent | 8.7 +/- 0.2 per cent |
| RMS log10(tau_q) | 4.1 +/- 0.2 per cent | 7.0 +/- 0.2 per cent |

### Caveat or advantage

- Caveat: at fixed optical indices the bump's metallicity spread (16-84 half width
  0.008 mag in the slice) is comparable to the 0.010 mag yardstick and larger than the
  0.005 mag one, so without a metallicity estimate an individual bump measurement mixes
  TP-AGB content, quenching phase and Z. The C3K and LW02 configurations predict opposite
  signs for the trend, so a template-agnostic correction does not exist.
- Advantage: the trend is smooth, nearly linear and, in the TP-AGB-on model, dominated by
  the TP-AGB stars themselves. Metallicity is routinely available from optical absorption
  features or SED fitting at the 0.1 dex level, and supplying it does not dilute the
  bump's information but sharpens it: the tau_q recovery gain rises from 4 to 7 per cent
  and the age gain from 7.6 to 8.7 per cent, with the classifier gains unchanged within
  errors. With TP-AGB off, the same known metallicity makes the bump's completeness gain
  negative (`fig4_information_gain.agb_off.classifier.bump_0.010_z`), because the TP-AGB-off
  bump has little to add once Z is known. The metallicity dependence of the TP-AGB-on bump
  is thus a second handle on the TP-AGB template, not just a nuisance: a sample with
  known metallicities tests the sign and slope of Figure 7(a) directly.

## Open items

- The TP-AGB-off row of Figure 1 and the TP-AGB-off bars of Figure 6 use agb = 0. If the paper
  prefers the FSPS default weight (agb = 1) as its "off" model, rerun with
  `AGB_CONFIGS["agb_off"]["agb"] = "agb1"`; Figure 2 shows the two differ by 0.005 mag.
- Model omissions carried over from Phase 5: no C-rich TP-AGB stars, no nebular
  emission or dust, four metallicities interpolated linearly in log Z, one Z per
  history, uniform time weighting.
