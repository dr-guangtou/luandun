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
- [ ] Implementation plan (writing-plans).
- [ ] Patch nzinit, rebuild wheel, uv project, pre-commit.
- [ ] sfh_model, ssp_grid, broadening, csp_integrate, spectral_indices with tests.
- [ ] Step 1 driver, cross-check against FSPS tabular CSP, figures.
- [ ] Pilot timing, then Step 2 population and figures.
- [ ] Review section, lessons.

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
