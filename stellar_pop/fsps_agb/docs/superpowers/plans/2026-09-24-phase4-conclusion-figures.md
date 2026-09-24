# Phase 4 Conclusion Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) switch the population sSFR to the surviving-mass convention and refresh every number that depends on it; (2) produce figures that support or refute two conclusions: "TP-AGB treatment makes a large difference to the H-minus bump, so the population locus can test TP-AGB models" and "the three indices have different age sensitivities, so combining them adds information about the SFH and the quenching process"; (3) a figure isolating metallicity: the same SFH at four metallicities drawn as curves in the three index planes.

**Architecture:** One small change to the SSP grid (store the surviving stellar mass fraction per age and metallicity from FSPS), one change to the population driver (surviving mass per epoch from the epoch weight matrix; agb = 1 indices added), reruns of the two populations and the two dependent analyses, and one new analysis script for the figures. Both template configurations (default C3K `output/ssp_grid`, LW02 `output/ssp_grid_lw02`) are shown side by side at agb = 2 unless stated.

**Tech Stack:** numpy, scipy (cKDTree), matplotlib, python-fsps (four SSP builds per grid for the surviving mass), pytest.

**Spec:** `docs/SPEC.md` Phase 3 section for conventions; this plan adds a Phase 4 subsection to the spec in Task 3.

## Global Constraints

- English, `snake_case`, no camelCase, no comments restating names, Ruff (E, F, I, N, UP, B, line length 100) clean, imports first then `matplotlib.use("Agg")`, no `noqa`.
- `uv run ...`; `SPS_HOME=/Users/shuang/code/fsps` exported; never hard-coded.
- Never estimate: every number in a caption, summary JSON or `docs/ANALYSIS.md` is computed.
- Default code paths keep their behaviour unless the change is the point of the task; existing tests stay green; new behaviour gets a test.
- Population statistics use epochs with `epoch_gyr >= 1.0`; yardstick precisions bump 0.005/0.010/0.020 mag, D4000 0.05, HdeltaA 0.5 A.
- Figures: constrained layout, units on every axis, bump axes inverted (stronger bump up), a text line in `summary.json` per figure with the numbers that appear in it.
- Commit after each task with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013xv6Y8bZShdhU6rexHLhni`.

---

### Task 1: Surviving-mass sSFR and agb = 1 population outputs

**Files:**
- Modify: `ssp_grid.py` (add `build_surviving_mass_fractions(out_dir, extra_params=None)` writing `<out_dir>/surviving_mass.npz` with keys `log_age_yr`, `log_z_grid`, `fraction` of shape (4, 107) taken from `fsps.StellarPopulation(...).stellar_mass` with `tage = 0` for `agb = 1` at each metallicity, plus `provenance_surviving_mass.json`; and `load_surviving_mass(out_dir) -> (log_age_yr, log_z_grid, fraction)`; CLI flag `--surviving-mass` on the existing `__main__`)
- Modify: `csp_integrate.py` (add `surviving_mass_per_epoch(weight_matrix, fraction_by_age) -> (n_epochs,)` = `weight_matrix @ fraction_by_age`)
- Modify: `run_population.py` (`specific_sfr_windows(edges_gyr, masses, epoch_index, surviving_mass)` divides by `surviving_mass` instead of the formed mass; `_history_indices` loads the fraction, interpolates it in log Z with `interpolate_log_z`, computes the surviving mass per epoch and passes it; also emits `agb1` columns (`d4000_agb1`, `hdelta_a_agb1`, `h_minus_bump_sigma300_agb1`, `h_minus_bump_r100_agb1`) from `flux_agb1`; a new column `surviving_mass_fraction` = surviving / formed per epoch)
- Modify: `analysis_alternative_sfh.py` only if its call into `_history_indices` needs the new argument (it should not, since the fraction is loaded from the grid dir inside `_history_indices`; verify)
- Tests: `tests/test_run_population.py` (surviving-mass sSFR: with fraction = 1 the old numbers are reproduced; with fraction = 0.5 everywhere the sSFR doubles; toy grids get a `surviving_mass.npz` written by a helper), `tests/test_csp_integrate.py` (one test for `surviving_mass_per_epoch`), `tests/test_ssp_grid.py` (slow: the solar fraction is 1 at the youngest age within 1e-6 and between 0.5 and 0.7 at 10 Gyr)

**Steps:**
- [ ] Write the failing tests, run them, implement, run the fast suite.
- [ ] Build the fractions for both grids: `uv run python ssp_grid.py --surviving-mass` and `uv run python ssp_grid.py --out-dir output/ssp_grid_lw02 --use-lw-tpagb --surviving-mass` (4 builds each, about 45 s). Record in `docs/lessons.md` the fraction at 1 and 10 Gyr, solar Z, both grids.
- [ ] Rerun the populations: `uv run python run_population.py` and `... --grid-dir output/ssp_grid_lw02 --out-dir output/population_lw02` (about 6 minutes each). Record the new class counts (rapid-quenching, post-starburst, quiescent, star-forming, transitional) next to the old ones in lessons.
- [ ] Rerun `analysis_fast_quenching.py` and `analysis_alternative_sfh.py` for both configurations (check their CLIs), then refresh every number in `docs/ANALYSIS.md` that changed (class counts, purity table, classifier table, contaminant numbers) and the README Phase 3 headline numbers; add a sentence in the Setup section that sSFR is normalized by surviving stellar mass including remnants (FSPS `stellar_mass`). Q1 numbers do not depend on sSFR; confirm by diffing `q1_summary.json` before and after (the population offsets use the same index columns; if anything moved, say why).
- [ ] `uv run pytest -q`, `uv run pytest -m slow -q`, `uv run pre-commit run --all-files`; commit code, small outputs (draws, summaries, figures, provenance), docs. Subject: "Normalize population sSFR by surviving stellar mass and add agb = 1 indices".

---

### Task 2: Conclusion figures and the metallicity figure

**Files:**
- Create: `analysis_conclusion_figures.py` with CLI `--out-dir output/analysis` and `--pilot`; it reads both grids, both populations and the fiducial parameters.
- Output: `output/analysis/c1_tpagb_population_test.png`, `c2_age_clocks.png`, `c2_clock_planes.png`, `c2_sfh_recovery.png`, `c3_metallicity_planes.png`, `conclusion_summary.json`.

**Figure C1 (TP-AGB test at the population level).** Rows: bump versus D4000, bump versus HdeltaA. Columns or overlays: four prescriptions on the same 2000 histories, C3K agb = 0, C3K agb = 1, LW02 agb = 1, LW02 agb = 2 (the `agb1` columns come from Task 1). Draw each prescription as a filled 16–84 percentile band of the bump in 20 bins of the x index, with the median line, distinct colors, and the 0.01 mag yardstick as an error bar. A third row: histograms of the bump in three D4000 slices (1.3–1.5, 1.5–1.7, 1.7–1.9) for the four prescriptions. Record per slice the medians, the 16–84 widths, and the separation between neighbouring prescriptions in units of the wider band's half-width and of 0.01 mag.

**Figure C2a (different age clocks).** For the fiducial t_q = 3 Gyr and tau_q in {0.1, 0.3, 1, 3} Gyr (solar Z, agb = 2), and for tau_q = 0.3 and t_q in {1.5, 3, 4.5} Gyr: the three indices versus time since quenching from -1 to +6 Gyr, one panel per index, two template rows (C3K, LW02), tracks computed with `csp_track` on the `sigma300` product (bump on `sigma300`). Mark the epoch of the extreme (HdeltaA maximum, bump minimum) on each track and record it: the delay after quenching at which each index peaks, per tau_q, per template.

**Figure C2b (the same tracks in the planes).** The three planes with the tau_q family and the t_q family as curves, time markers every 0.5 Gyr after quenching, both template rows.

**Figure C2c (SFH recovery test).** Nearest-neighbour regression (k = 25, cKDTree, 5 folds grouped by history, features scaled by the yardstick noise, noise added with `SeedSequence`) predicting log10(time since quenching) and log10(tau_q) for post-quench epochs (0 < t - t_q < 6 Gyr, epochs >= 1 Gyr) from (D4000, HdeltaA) versus (D4000, HdeltaA, bump at 0.005 / 0.010 / 0.020 mag), for both templates at agb = 2 and for C3K agb = 0 as a control. Report the RMS error of each target, the paired per-fold difference and its standard error, over 3 noise seeds. Figure: RMS error versus bump precision, one panel per target, lines per template. This is the quantitative test of conclusion 2; report the result whichever way it goes.

**Figure C3 (metallicity only).** The fiducial SFH (t_q = 3, tau_q = 0.3 Gyr) at log Z = -0.5, -0.25, 0, +0.25 (agb = 2), curves in the three planes colored by metallicity (a sequential colormap), time markers at t_q and t_q + 0.5, 1, 2, 5 Gyr (distinct marker shapes, one legend), two template rows. Record the spread across metallicity of each index at those five epochs, and compare it with the agb 0 to 2 delta at the same epochs (from the same tracks at agb = 0).

**Steps:**
- [ ] Implement as small functions per figure; `--pilot` runs each figure on a subsample without saving; run the full script; look at every PNG with the Read tool and fix layout problems.
- [ ] Ruff; `uv run pytest -q`; commit the script, the PNGs and `conclusion_summary.json`. Subject: "Add conclusion and metallicity figures".

---

### Task 3: Write-up

**Files:** `docs/ANALYSIS.md` (new section "Supporting figures for the two conclusions", with one subsection per conclusion stating whether the figures support it, with numbers from `conclusion_summary.json` cited by key, and a subsection for the metallicity figure), `docs/SPEC.md` (short Phase 4 subsection describing the sSFR convention change and the new script), `docs/todo.md` (Phase 4 items and a Review), `README.md` (script and figures), `docs/lessons.md`.

- [ ] Write the sections; every claim tied to a figure and a number. Do not soften a negative result.
- [ ] `uv run pre-commit run --all-files`; commit. Subject: "Document the conclusion and metallicity figures".
