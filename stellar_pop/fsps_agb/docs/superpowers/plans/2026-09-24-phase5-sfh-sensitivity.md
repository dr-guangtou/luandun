# Phase 5 SFH Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the Phase 3 and 4 results depend on the functional form of the quenching history by rerunning the population with three alternative SFH families, two of them FSPS-native forms, paired history by history with the existing draws, for both TP-AGB template sets.

**Architecture:** New cumulative-mass functions in `sfh_model.py` feed the existing `epoch_weight_matrix_from_cumulative` hook; `run_population.py` gains an `--sfh-family` option and writes to a family-specific output directory; a cross-check script validates the two FSPS-native families against FSPS's own `sfh = 5` and `sfh = 4` with `sf_trunc`; a new analysis script compares the families.

**Tech Stack:** numpy, scipy, matplotlib, python-fsps (cross-check only), pytest.

**Spec:** `docs/SPEC.md` (Phases 2 to 4 conventions); Task 3 adds a Phase 5 subsection.

## Global Constraints

- English, `snake_case`, Ruff (E, F, I, N, UP, B, line length 100) clean, imports first then `matplotlib.use("Agg")`, no `noqa`; `uv run ...` with `SPS_HOME=/Users/shuang/code/fsps` exported.
- Never estimate; every number in docs comes from a summary JSON. Default code paths unchanged; existing tests green; new behaviour tested.
- Population statistics on epochs >= 1 Gyr; yardstick precisions as before; classes by the manuscript rules with surviving-mass sSFR (Phase 4).
- The four families, all with the delayed-tau rise `SFR = (t/tau) exp(-t/tau)` for `t < t_q` and the same 2000 draws of `(t_q, tau_q, log_z)` (seed 20260924):
  - `exponential` (existing): tau = t_q; `SFR(t >= t_q) = e^-1 exp(-(t - t_q)/tau_q)`.
  - `linear` (FSPS `sfh = 5` form): tau = t_q; `SFR(t >= t_q) = e^-1 max(0, 1 - (t - t_q)/delta_q)` with `delta_q = 2 ln 2 tau_q` (same SFR half-life as the exponential family).
  - `truncation` (FSPS `sfh = 4` with `sf_trunc`): tau = t_q; `SFR(t >= t_q) = 0`.
  - `decoupled`: exponential quench with tau_q as drawn but tau independent of t_q, log-uniform on [0.5, 5] Gyr drawn with seed 20260926 (new column `tau_gyr` in draws.npz). `SFR(t < t_q) = (t/tau) exp(-t/tau)` and `SFR(t >= t_q) = (t_q/tau) exp(-t_q/tau) exp(-(t - t_q)/tau_q)` (continuous at t_q).
- Output directories: `output/population_<family>` and `output/population_<family>_lw02` for the three new families (the exponential ones stay in `output/population` and `output/population_lw02`); same ignore rules as before.
- Commit after each task with the two trailer lines: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013xv6Y8bZShdhU6rexHLhni`.

---

### Task 1: SFH families and FSPS-native cross-checks

**Files:**
- Modify: `sfh_model.py` — generalize `cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr, tau_gyr=None)` (default `tau_gyr = t_q_gyr`, exact old behaviour) and `star_formation_rate` likewise; add `linear_quench_cumulative_mass(time_gyr, t_q_gyr, delta_q_gyr)` and `truncated_cumulative_mass(time_gyr, t_q_gyr)`, all analytic (the linear ramp integral is a quadratic; the ramp ends at `t_q + delta_q`); add `sfh_family_cumulative(family, t_q_gyr, tau_q_gyr, tau_gyr=None) -> callable(time_gyr)` returning the right partial for the four family names; add `draw_decoupled_tau(n_draws, seed=20260926)`.
- Modify: `run_population.py` — `--sfh-family` (default `exponential`), `--out-dir` default derived from the family, `draws.npz` gains `tau_gyr` for `decoupled`; `_history_indices` receives the family's callable via the existing `cumulative_mass_fn`; the `sfr` column uses the family's SFR (add `star_formation_rate_from_cumulative` by finite difference of the cumulative mass over the 0.05 Gyr bins, or analytic per family; state which). Default outputs must be bit-identical to before for the exponential family (test: run `--pilot` and compare to a stored pilot table; or a unit test with the toy grids).
- Create: `cross_check_fsps_families.py` — for the fiducial `(t_q = 3, tau_q = 0.3, solar Z, agb = 1)`: the `linear` family versus FSPS `sfh = 5` with `tau = 3`, `sf_start = 0`, `sf_trunc = 3`, `sf_slope = -1 / delta_q` (per Gyr; verify the sign convention empirically by checking that FSPS's `sfr` attribute at `t_q + delta_q / 2` is half of its value at `t_q`), and the `truncation` family versus `sfh = 4`, `tau = 3`, `sf_trunc = 3`; six epochs (1, 3, 3.5, 5, 8, 13 Gyr); maximum relative flux difference inside the three index windows and the index differences, JSON to `output/single_csp/fsps_family_cross_check.json`; threshold 2 percent; slow test in `tests/test_cross_check_fsps_families.py` writing to `tmp_path`.
- Tests: `tests/test_sfh_families.py` — each cumulative mass against `scipy.integrate.quad` of the corresponding SFR at several bins including one straddling `t_q` and, for `linear`, one straddling `t_q + delta_q`; `tau_gyr = None` reproduces the old values exactly; `truncation` mass is constant after `t_q`; `decoupled` continuity at `t_q`.

- [ ] TDD for the families; run the fast suite; implement the cross-check; run it (about 40 s of FSPS); if a family exceeds 2 percent, fix the family's formula or the FSPS parameter mapping, not the threshold.
- [ ] `uv run pytest -q`, `uv run pytest -m slow -q`, ruff; commit code, tests, the cross-check JSON. Subject: "Add linear, truncation and decoupled SFH families with FSPS cross-checks".

---

### Task 2: Population runs and the sensitivity analysis

**Files:**
- Create: `analysis_sfh_sensitivity.py` with `--out-dir output/analysis` and `--pilot`.
- Output: `output/analysis/s1_fiducial_families.png`, `s2_population_bands.png`, `s3_class_fractions.png`, `s4_classifier_by_family.png`, `s5_recovery_by_family.png`, `sfh_sensitivity_summary.json`.

- [ ] Run the six new populations: `uv run python run_population.py --sfh-family <linear|truncation|decoupled>` and the same with `--grid-dir output/ssp_grid_lw02 --out-dir output/population_<family>_lw02` (about 6 minutes each; run two at a time at most). Look at their index-plane figures.
- [ ] Figure S1: the fiducial history under the four forms (SFR versus time, then the three indices versus time since quenching, two template rows, agb = 2), tracks via `csp_track`-equivalent calls using `epoch_weight_matrix_from_cumulative` on the `sigma300` grids.
- [ ] Figure S2: for each template (rows) and each family (line style or column), the 16–84 band of the bump versus D4000 for agb = 2 and the rapid-quenching locus (contour of the class density); record per family the band median in the three D4000 slices of Phase 4 and the rapid-quenching locus centroid.
- [ ] Figure S3: class fractions per family (stacked bars), with the rapid-quenching and post-starburst counts recorded.
- [ ] Figure S4: rapid-quenching completeness and purity, optical-only versus with the bump at 0.010 mag (agb = 2, unbalanced, 3 noise seeds, paired fold SE as in Phase 3/4), per family and template; reuse `population_classes` and the feature-set and paired-difference helpers from `analysis_fast_quenching.py` (import them; do not copy).
- [ ] Figure S5: the SFH-recovery RMS gain from the bump (log10 time since quenching and log10 tau_q, at 0.005 and 0.010 mag) per family and template, reusing the Phase 4 regression helpers (import from `analysis_conclusion_figures.py`).
- [ ] Look at every PNG; ruff; `uv run pytest -q`; commit the script, PNGs, summary, and the small population outputs (draws, summary.json, figures) of the six new runs; `.gitignore` the large tables. Subject: "Add SFH-family sensitivity analysis".

---

### Task 3: Write-up

- [ ] `docs/ANALYSIS.md`: new section "Sensitivity to the star formation history model": which FSPS-native forms exist and were used, the cross-check numbers, then for each Phase 3/4 conclusion whether it survives across the four families, with numbers cited by key; a table of class fractions per family; an explicit statement of what changed and what did not. `docs/SPEC.md` Phase 5 subsection; `docs/todo.md` items and Review; `README.md`; `docs/lessons.md`.
- [ ] pre-commit; commit. Subject: "Document the SFH-family sensitivity test".
