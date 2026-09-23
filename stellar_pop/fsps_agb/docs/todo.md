# TODO

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
  is 0.032-0.059 mag, 1.7-1.9 times the intrinsic scatter (48 of 48 bins pass).
- Q2: fast-quenching epochs are isolable in D4000-HdeltaA alone (77-81 percent of them in
  cells above 0.5 purity); the bump adds nothing with C3K and about +0.05 completeness,
  +0.04 purity at 0.01 mag with LW02.
- Alternative SFH families (300 bursty, 300 slowly fading histories): no contaminant epoch
  lands in any previously pure cell in either configuration; unbalanced kNN mislabels at
  most 13 of 144,600 contaminant epochs.
- Methodology finding: the 0.5-99.5 percentile purity grid clips 29.6 percent of LW02 agb2
  rapid-quenching epochs in the bump planes; the full-range grid raises the D4000-bump
  isolable fraction from 0.134 to 0.291 (still far below the optical plane).
