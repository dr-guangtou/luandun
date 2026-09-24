# TODO

## Phase 6 — Publication figure candidates (2026-09-24)

### Plan
- [x] Task 1: check in the FSPS source (`getspec.f90`, `sps_setup.f90`, `sps_vars.f90`,
      `mod_gb.f90`, `add_agb_dust.f90`) how the TP-AGB templates depend on metallicity.
- [x] Task 2: `publication_figures.py` writing PDF and PNG candidates to
      `output/publication/`: Figure 1 (index planes, AGB off versus on, rapid-quenching
      class, tau_q contours, metallicity tracks), Figure 2 (prescription offsets),
      Figure 3 (quenching clocks), Figure 4 (classifier and recovery gain at R = 100),
      Figures 5 and 6 (the other SFH families), Figure 7 (metallicity decomposition and
      the known-Z test), plus `publication_summary.json`.
- [x] Task 3: write-up in `docs/PUBLICATION.md` with the FSPS facts, the metallicity
      caveat and the advantage argument; SPEC, README, lessons, this review.

### Review
- The rapid-quenching class of the exponential family contains only tau_q < 0.3 Gyr
  histories (the ratio rule forces it), so "colour the fast-quenching population by
  quenching timescale" has no dynamic range; Figure 1 colours the epochs within 2 Gyr
  of t_q by tau_q instead and keeps the class as one contour.
- All Phase 4/5 gain numbers were recomputed with the R = 100 bump (they had used the
  sigma300 bump); the changes are within the seed scatter.
- New results: the AGB-on bump minimum lags the HdeltaA peak by 0.3 to 2 Gyr with the
  lag growing monotonically in tau_q (Figure 3d), and a known metallicity (0.1 dex)
  raises the bump's tau_q recovery gain from 4 to 7 per cent of the baseline (Figure 7c).
- The LW02 O-rich templates have no metallicity dependence in their spectra; the AGB-on
  bump's -0.04 mag per dex trend comes from the MIST TP-AGB Teff distribution crossing
  the fixed log Teff = 3.6 template switch and from the Z-dependent Teff labels in
  `Orich.teff`. The AGB-off bump's trend has the opposite sign and comes from the
  non-AGB stars.
- Not done: no agb = 1 "AGB off" variant of Figures 1 and 6 (Figure 2 shows it differs
  from agb = 0 by 0.005 mag); figures are candidates, not final captions.

## Phase 5 — SFH-family sensitivity (2026-09-24)

### Plan
- [x] Task 1: `sfh_model.py` generalized cumulative-mass/SFR and the four-family
      dispatch (`linear`, `truncation`, `decoupled` alongside the existing
      `exponential`); `run_population.py` `--sfh-family`; FSPS-native cross-checks
      (`cross_check_fsps_families.py`) against `sfh = 5` (`linear`) and `sfh = 4` with
      `sf_trunc` (`truncation`), plus the `sf_slope` sign check.
- [x] Task 2: six population runs (`linear`/`truncation`/`decoupled`, C3K and LW02) and
      `analysis_sfh_sensitivity.py` (S1-S5: fiducial tracks, population bump bands,
      class fractions, classifier and SFH-recovery gain, per family and template).
- [x] Task 3: write-up (`docs/ANALYSIS.md` new section, `docs/SPEC.md` Phase 5
      subsection, this Review, `README.md`, `docs/lessons.md`).
- [x] Task 3 fix round 1: Figure S6 (`s6_tpagb_offset_by_family.png`), the per-family
      `agb0`/`agb2` bump offset (every population run already carries all three `agb`
      columns), and three corrected ranges (docs review).

### Review
- All four earlier conclusions survive across all four SFH families. Phase 3 Q1 and
  Phase 4 Conclusion 1 (the population TP-AGB offset separates templates, not weight)
  are directly retested per family by Figure S6, since every population run already
  carries `agb0`/`agb1`/`agb2` columns (no rerun needed): the C3K offset is 1-3 times
  the per-model scatter and the LW02 offset is 4-12 times it in every family, and the
  C3K-versus-LW02 template separation (0.049-0.061 mag) is several times any
  weight-only step, in every family too. Phase 3 Q2 and Phase 4 Conclusion 2 also
  survive: the LW02 bump gain is positive and multi-sigma, and the C3K gain is small
  and sign-inconsistent, in every family (docs/ANALYSIS.md, "Sensitivity to the star
  formation history model"). What remains genuinely untested per family is narrower:
  only the `agb0` *control* of the classifier (S4) and SFH-recovery (S5) gains was run
  at `agb2` only for the three new families.
- The rapid-quenching base rate is highly SFH-shape-dependent even though the
  `(t_q, tau_q, log_z)` prior and the sample size are identical: 0.84 percent
  (`decoupled`) to 6.38 percent (`truncation`), driven by how long a family's post-quench
  SFR keeps R = sSFR(0-100 Myr)/sSFR(100 Myr-1 Gyr) below the 0.1 threshold.
- `linear` and `truncation` were validated against FSPS's own `sfh = 5` and `sfh = 4`
  (with `sf_trunc`) to 0.027 percent maximum relative flux difference, four orders of
  magnitude under the 2 percent threshold
  (`output/single_csp/fsps_family_cross_check.json`); `decoupled` has no FSPS-native
  form and was validated only internally (`quad` integration, continuity at `t_q`).
- S4 and S5 (the classifier and SFH-recovery tests) carry no `agb0` control per family:
  the *pattern* (LW02 real, C3K not) replicates across families, but the *attribution*
  to TP-AGB light specifically still rests on the one Phase 4 `agb0` control, measured
  for the `exponential` family only.
- One genuinely new, family-specific finding: for `truncation`, adding the bump gives
  no `log10(tau_q)` recovery gain in either template (consistent with zero, both C3K
  and LW02) — a hard cutoff's SFR is identically zero after `t_q` and carries no
  `tau_q` information to begin with, unlike the other three families' post-quench SFR.

## Phase 4 — Surviving-mass sSFR and conclusion figures (2026-09-24)

### Plan
- [x] Task 1: surviving-mass sSFR normalization, `agb = 1` population columns, rerun
      both populations and dependent analyses, refresh `docs/ANALYSIS.md`/`README.md`.
- [x] Task 2: `analysis_conclusion_figures.py` — the TP-AGB population test, the age
      clocks and clock planes, the SFH-recovery test, and the metallicity-only figure.
- [x] Task 3: write-up (`docs/ANALYSIS.md` new section, `docs/SPEC.md` Phase 4
      subsection, this Review, `README.md`, `docs/lessons.md`).

### Review
- Conclusion 1 ("population locus tests the TP-AGB model") holds only in a narrower
  form than stated: the locus separates TP-AGB *templates* (C3K vs LW02: -0.017 to
  -0.027 mag at fixed D4000, several times the 0.01 mag yardstick) far more than it
  separates TP-AGB *weight* within a template (C3K agb0 -> agb1: -0.0026 to -0.0051
  mag, about half the yardstick); see docs/ANALYSIS.md, "Supporting figures for the two
  conclusions".
- Conclusion 2 ("combining the indices adds SFH information") is template-dependent:
  the SFH-recovery regression gain from adding the bump is real and multi-sigma for
  LW02 (-0.018 dex on log10(time since quenching) at 0.005 mag precision) but small and
  statistically indistinguishable from a no-TP-AGB control for the default C3K
  templates (-0.004 dex vs. -0.006 dex for the C3K agb0 control).
- The raw C3K H-minus bump (not the agb0-to-agb2 delta) has no interior minimum within
  +6 Gyr of quenching — the marker in `c2_age_clocks.png` sits at the window edge, not
  a true extremum (`conclusion_summary.json`,
  `c2_age_clocks.c3k.h_minus_bump.*.at_window_boundary: true`, every track).
- Metallicity is a genuine confounder for the C3K bump (spread 0.008-0.015 mag,
  comparable to the 0.004-0.011 mag TP-AGB delta at the same epochs) and a secondary
  systematic for LW02 (spread 0.014-0.029 mag, 2-4x smaller than the 0.033-0.067 mag
  TP-AGB delta); the sign of the metallicity trend is opposite between templates (C3K
  bump weakens with increasing Z, LW02 strengthens).
- Surviving-mass sSFR normalization moved rapid-quenching counts from 4,727 to 5,907
  (base rate 0.98% -> 1.23%) with R unchanged; every number depending on it in
  `docs/ANALYSIS.md` and `README.md` was refreshed and cross-checked against the
  regenerated summary JSONs, not estimated.

## Phase 2 — CSP index tracks and populations (2026-09-24)

### Decisions from the interview
- tau_q log-uniform on [0.1, 3] Gyr; log(Z/Zsun) truncated normal (0.0, 0.2) on [-0.5, 0.2].
- Fix the C3K_HR loader (nzinit 11 -> 13), rebuild, interpolate SSPs in log Z.
- R = 100 as Gaussian FWHM, in quadrature with 300 km/s, NIR only.
- Own numpy CSP integrator from cached SSP grids; FSPS tabular SFH as a cross-check.
- 0.05 Gyr step, 2000 SFH draws; fiducial t_q = 3, tau_q = 0.3 Gyr, solar Z.
- HdeltaA red edge 4122.25 A, air-to-vacuum shift, D4000 in F_nu, bump in F_lambda.
- Same folder, uv project with the rebuilt wheel, Ruff pre-commit.

### Plan
- [x] Study Phase 1, FSPS/python-fsps source, and the ProGeny resolution study.
- [x] Interview and design; spec written to docs/SPEC.md.
- [x] Implementation plan: docs/superpowers/plans/2026-09-24-csp-index-tracks.md
- [x] Patch nzinit, rebuild wheel, uv project, pre-commit.
- [x] sfh_model, ssp_grid, broadening, csp_integrate, spectral_indices with tests.
- [x] Step 1 driver, cross-check against FSPS tabular CSP, figures.
- [x] Pilot timing, then Step 2 population and figures.
- [x] Review section, lessons.

### Review
- The bump planes (D4000-vs-bump, HdeltaA-vs-bump) fan out into a visibly
  wider, fan-shaped spread at fixed D4000/HdeltaA than the tight,
  nearly one-dimensional D4000-vs-HdeltaA sequence does, but that spread is
  dominated by metallicity, not by star-formation history: the share of the
  bump's variance at fixed D4000 explained by log Z is 0.95 at D4000
  1.2-1.3, 0.75 at 1.6-1.7 and 0.94 at 2.0-2.1. The SFH contribution is
  small except where age and Z covary (`output/population/index_planes_{sigma300,r100}_logz.png`,
  colored by log Z; `output/population/index_planes_{sigma300,r100}.png`,
  task-11-report.md).
- TP-AGB sensitivity is concentrated in the bump, not in D4000/HdeltaA: over
  the fiducial track (`output/single_csp/indices.csv`, computed directly as
  `(agb2 - agb0) / agb0` for D4000 and `agb2 - agb0` for HdeltaA) the
  largest `agb2` vs `agb0` change is +0.75% relative in D4000 (at 4.2 Gyr)
  and -0.14 A (absolute) in HdeltaA (at 4.0 Gyr), against -0.0111 mag in the
  bump at 3.45 Gyr (0.45 Gyr post-quench) — a small fraction of
  D4000/HdeltaA's own dynamic range (1.02-2.34, -4.6-6.3 A) but roughly a
  third of the bump's own range (~-0.02 to +0.01 mag). At the population
  level, 0.5-2 Gyr after quenching `agb2` gives a more negative bump than
  `agb0` for 100% of the population (mean offset -0.0087 mag `sigma300`,
  -0.0089 mag `r100`).
- Cross-check against FSPS's tabular SFH (after the log-spaced lookback
  sub-grid fix, docs/lessons.md Task 8): maximum relative flux difference
  inside any index window is 0.312% at 1.0 Gyr, decaying to 0.006% by
  13.0 Gyr — comfortably under the 2% threshold at every checked epoch
  (`output/single_csp/fsps_cross_check.json`).
- Validation numbers all met their thresholds: `agb` linearity to 1e-10
  relative (Task 5), broadening recovers sqrt(40^2+300^2) within 1%
  (Task 6), and the population pilot (10 histories x 10 epochs, 0.2 s)
  extrapolated to 15.4 min for the full 2000 x 260 run, well under the
  60-minute threshold, so the Step 4 pixel-slicing optimization was not
  needed.
- No validation threshold was left unmet; the only numeric miss recorded
  during implementation was the original 0.05 Gyr bin-center integrator
  disagreeing with FSPS by up to 45% in D4000 at young epochs, root-caused
  to coarse age discretization of the youngest stars and fixed by the
  log-spaced lookback sub-grid (docs/SPEC.md, "CSP assembly" section).

## Phase 1 — SSP sandbox (complete)

### Done
- [x] Investigate `python-fsps` API and `fsps` Fortran source.
- [x] Identify AGB / TP-AGB tunable parameters.
- [x] Confirm compiled libraries: MIST + C3K (low-res) + DL07.
- [x] Generate SSP spectra (solar Z, 1 Gyr, Kroupa & Chabrier IMF) across AGB scenarios.
- [x] Compare NIR spectra in 1.4–1.8 micron (ratios vs fiducial).
- [x] Inspect raw AGB templates (`Orich.spec`, `Crich_Aringer.spec`) at native resolution.
- [x] Rebuild `python-fsps` with `C3K_HR` for higher NIR resolution.

### Review
- TP-AGB stars supply ~31% of the NIR (1.4–1.8 µm) flux at 1 Gyr — the `agb`
  weight is the dominant lever; `pagb` and `add_agb_dust_model` are second-order
  in the NIR (dust matters in the mid-IR instead).
- `tpagb_norm_type` and `fcstar` are currently inert for MIST builds; document
  this clearly to avoid wasted experimentation.
- The C3K_LR output grid (R=100) is coarse for the 1.6 µm region; the C3K_HR
  rebuild is the main lever to improve the spectral comparison.

## Phase 3 — Diagnostic analysis (requested 2026-09-24)
- [x] Q1: Is a model with vs without TP-AGB contribution (agb = 0 vs 2) clearly distinguishable in the three indices, for SSPs and for the CSP tracks and population, given realistic index uncertainties? QA figures required.
- [x] Q2: Assuming strong TP-AGB (agb = 2), can fast-quenching epochs be isolated in the index planes from star-forming, slowly quenching and old quiescent epochs? Test alternative SFH families if the delayed-tau family is inconclusive. QA figures required.

### Review (2026-09-24)
- Answers, with every number traced to a summary JSON key, are in `docs/ANALYSIS.md`;
  both TP-AGB template configurations (default C3K, empirical LW02) were run end to end.
- Q1 is template-dependent: with C3K the population bump offset (agb2 - agb0) at fixed
  D4000/HdeltaA is at most 0.009 mag (0 of 48 bins pass the "clearly" rule); with LW02 it
  is 0.032-0.059 mag, 6-16 times the per-model RMS intrinsic scatter (48 of 48 bins
  pass); with C3K it is 0.9-2.2 times that scatter, so the C3K "no" rests on the 0.01 mag
  precision (final-review fix wave, 2026-09-24).
- Q2 (after the final-review fix wave): partially, in D4000-HdeltaA only. With yardstick
  noise the kNN recovers about 41 percent of rapid-quenching epochs at about 61 percent
  purity (1 percent base rate); noise-free maps put 77-79 percent in cells above 0.5
  purity (full-range grid). The bump adds nothing TP-AGB-specific with C3K; with LW02 it
  adds +0.081 completeness / +0.039 purity at 0.005 mag and +0.041 / +0.028 at 0.010 mag
  (mean over 3 noise seeds, seed std 0.004-0.012).
- Alternative SFH families (300 bursty, 300 slowly fading histories): no contaminant epoch
  lands in any previously pure cell in either configuration; unbalanced kNN mislabels at
  most 16 of 144,600 contaminant epochs (neither family can reach R < 0.1, so this is
  a weak test).
- Methodology finding: the 0.5-99.5 percentile purity grid clips 29.6 percent of LW02 agb2
  rapid-quenching epochs in the bump planes; the full-range grid raises the D4000-bump
  isolable fraction from 0.134 to 0.291 (still far below the optical plane).
