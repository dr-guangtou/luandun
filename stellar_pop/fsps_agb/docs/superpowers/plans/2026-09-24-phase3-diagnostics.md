# Phase 3 Diagnostic Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer, with QA figures and measured numbers, (Q1) whether MIST + C3K models with and without TP-AGB contribution can be told apart in D4000, HdeltaA and the H-minus bump, and (Q2) whether, with a strong TP-AGB contribution, fast-quenching galaxies can be isolated in the index planes.

**Architecture:** Three analysis scripts on top of the Phase 2 modules and outputs. Q1 works at the SSP level (cached grids plus two extra FSPS builds for the empirical TP-AGB spectra) and at the population level (binned offsets against intrinsic scatter). Q2 classifies population epochs with the manuscript's sSFR rules and tests a nearest-neighbour classifier with and without the bump under measurement noise; a third script adds alternative SFH families as contaminants and writes the answer document.

**Tech Stack:** numpy, scipy (cKDTree for nearest neighbours, erf for burst masses), matplotlib, python-fsps (two extra SSP builds), pytest.

**Spec:** `docs/SPEC.md`, section "Phase 3 — Diagnostic analysis". Findings from the Phase 2 final review that shape this plan: the pure TP-AGB component spectrum S(1) - S(0) has a bump index of only -0.025 to -0.037 mag at every age; TP-AGB supplies 22–35 percent of the feature-band light at 0.3–1 Gyr, so agb 0 to 2 moves the index by 0.01–0.016 mag; the bump is identical on native and sigma300 and changes by under 0.001 mag at r100; the SSP bump changes by 0.006–0.013 mag over log Z = -0.5 to +0.25; at fixed D4000 the population bump scatter is mostly metallicity.

## Global Constraints

- English, `snake_case`, no camelCase, no comments restating names, Ruff (E, F, I, N, UP, B, line length 100) clean, all imports first then `matplotlib.use("Agg")`, no `noqa`.
- `uv run ...` for everything; `SPS_HOME=/Users/shuang/code/fsps` exported; never hard-coded.
- Never estimate: every number in a figure caption, `summary.json` or `docs/ANALYSIS.md` is computed by the scripts.
- Yardstick precisions: bump 0.005, 0.010, 0.020 mag; D4000 0.05; HdeltaA 0.5 A. Population statistics use epochs with `epoch_gyr >= 1.0` only.
- Classes (manuscript): R = sSFR(0–100 Myr)/sSFR(100–1000 Myr) with sSFR in per yr = table value / 1e9; star-forming R > 1; rapid-quenching sSFR(100–1000) > 1e-10 and R < 0.1; post-starburst subset sSFR(100–1000) > 1e-10 and sSFR(0–100) < 1e-11; quiescent sSFR(100–1000) < 1e-10 and sSFR(0–100) < 1e-11; else transitional. Guard division by zero with `np.divide(..., where=...)` and treat sSFR(100–1000) = 0 as R = 0.
- Every script has `--pilot` (subsample, no figures) and prints its timing; outputs go to `output/analysis/`; each script writes its own `q<N>_summary.json`.
- Figures: matplotlib defaults of the environment; every panel has axis labels with units; the bump axis is inverted (stronger bump up) as in `index_planes.py`; shade index bands where spectra are shown.
- Commit after each task with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013xv6Y8bZShdhU6rexHLhni`.

---

## File structure

| File | Responsibility |
| ---- | -------------- |
| `population_classes.py` | class assignment from the sSFR columns, noise injection, grouped k-fold, nearest-neighbour classifier, completeness/purity metrics |
| `analysis_agb_separability.py` | Q1: SSP-level and population-level separability figures and numbers |
| `analysis_fast_quenching.py` | Q2: purity maps and noise-aware classification with and without the bump |
| `analysis_alternative_sfh.py` | Q2 robustness: bursty and slowly fading families as contaminants; reuses the classifier |
| `sfh_model.py` (modify) | add `burst_cumulative_mass` and a generic `epoch_weight_matrix_from_cumulative` hook in `csp_integrate.py` |
| `docs/ANALYSIS.md` | the written answers |
| `tests/test_population_classes.py`, `tests/test_sfh_burst.py` | unit tests |

Shared interfaces (defined in Task 1, used by Tasks 3 and 4):

```python
# population_classes.py
CLASS_NAMES = ("star_forming", "rapid_quenching", "transitional", "quiescent")
def assign_classes(ssfr_recent_per_gyr, ssfr_previous_per_gyr) -> np.ndarray  # int codes 0..3 in CLASS_NAMES order
def is_post_starburst(ssfr_recent_per_gyr, ssfr_previous_per_gyr) -> np.ndarray  # bool
def add_measurement_noise(features, sigmas, rng) -> np.ndarray               # features (n, k), sigmas (k,)
def grouped_folds(group_ids, n_folds, rng) -> list[np.ndarray]                # fold index per sample, groups never split
def knn_predict(train_features, train_labels, test_features, k, feature_scales) -> np.ndarray
def completeness_purity(true_labels, predicted_labels, positive_class) -> tuple[float, float]
def cross_validated_metrics(features, labels, groups, feature_scales, k=25, n_folds=5, seed=0, positive_class=1) -> dict
    # keys: completeness_mean, completeness_std, purity_mean, purity_std, confusion (4x4 summed over folds)
```

---

### Task 1: Population classes and classifier helpers

**Files:**
- Create: `population_classes.py`
- Test: `tests/test_population_classes.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from population_classes import (
    CLASS_NAMES,
    add_measurement_noise,
    assign_classes,
    completeness_purity,
    cross_validated_metrics,
    grouped_folds,
    is_post_starburst,
    knn_predict,
)


def test_assign_classes_follows_manuscript_rules():
    recent = np.array([2e-10, 5e-12, 1e-12, 5e-11, 0.0]) * 1e9   # per Gyr
    previous = np.array([1e-10, 2e-10, 5e-11, 2e-10, 0.0]) * 1e9
    codes = assign_classes(recent, previous)
    assert [CLASS_NAMES[c] for c in codes] == [
        "star_forming", "rapid_quenching", "quiescent", "transitional", "quiescent"]
    assert is_post_starburst(recent, previous).tolist() == [False, True, False, False, False]


def test_noise_has_requested_scale():
    rng = np.random.default_rng(1)
    features = np.zeros((20000, 2))
    noisy = add_measurement_noise(features, np.array([0.05, 0.01]), rng)
    assert np.allclose(noisy.std(axis=0), [0.05, 0.01], rtol=0.05)


def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(40), 5)
    folds = grouped_folds(groups, 5, np.random.default_rng(0))
    assert folds.shape == groups.shape and set(folds) == set(range(5))
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1


def test_knn_predict_recovers_separated_blobs():
    rng = np.random.default_rng(2)
    a = rng.normal([0, 0], 0.1, size=(200, 2))
    b = rng.normal([1, 1], 0.1, size=(200, 2))
    features = np.vstack([a, b]); labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    test = np.array([[0.05, -0.05], [0.95, 1.05]])
    assert knn_predict(features, labels, test, k=5, feature_scales=np.array([1.0, 1.0])).tolist() == [0, 1]


def test_completeness_purity():
    true = np.array([1, 1, 1, 0, 0])
    pred = np.array([1, 1, 0, 1, 0])
    completeness, purity = completeness_purity(true, pred, positive_class=1)
    assert np.isclose(completeness, 2 / 3) and np.isclose(purity, 2 / 3)


def test_cross_validated_metrics_on_separable_data():
    rng = np.random.default_rng(3)
    n = 400
    features = np.vstack([rng.normal([0, 0], 0.1, (n, 2)), rng.normal([2, 2], 0.1, (n, 2))])
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]
    groups = np.arange(2 * n) // 4
    result = cross_validated_metrics(features, labels, groups, np.array([1.0, 1.0]), k=5)
    assert result["completeness_mean"] > 0.95 and result["purity_mean"] > 0.95
    assert np.asarray(result["confusion"]).shape == (4, 4)
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_population_classes.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `population_classes.py`**

Algorithms (write the code from these):
- `assign_classes`: convert per-Gyr sSFR to per-yr by dividing by 1e9; `ratio = np.divide(recent, previous, out=np.zeros_like(recent), where=previous > 0)`; star-forming where ratio > 1; rapid-quenching where previous > 1e-10 and ratio < 0.1; quiescent where previous < 1e-10 and recent < 1e-11; remaining transitional. Apply in the order quiescent, rapid-quenching, star-forming, transitional-as-default so a sample gets exactly one code; the test cases above are consistent with that order.
- `add_measurement_noise`: `features + rng.normal(0, sigmas, features.shape)`.
- `grouped_folds`: shuffle the unique group ids with `rng.permutation`, assign group i to fold i mod n_folds, map back to samples.
- `knn_predict`: divide features by `feature_scales`, build `scipy.spatial.cKDTree` on the scaled training set, query k neighbours, majority vote with `np.bincount` per row (ties: lowest code).
- `completeness_purity`: completeness = TP / (TP + FN), purity = TP / (TP + FP); return `float("nan")` when a denominator is zero.
- `cross_validated_metrics`: for each fold train on the rest, predict the fold, accumulate a 4x4 confusion matrix `confusion[true, pred]`, collect per-fold completeness and purity of `positive_class`; return means and standard deviations (`ddof=0`) and the summed confusion as a nested list.

- [ ] **Step 4: Run the tests** — expect 6 passed. Ruff check/format.

- [ ] **Step 5: Commit** — `git add population_classes.py tests/test_population_classes.py`, subject "Add population class assignment and nearest-neighbour classifier helpers".

---

### Task 2: Q1 — TP-AGB separability

**Files:**
- Create: `analysis_agb_separability.py`
- Output: `output/analysis/q1_ssp_delta_vs_age.png`, `q1_component_spectrum.png`, `q1_nir_spectra.png`, `q1_broadband_ratio.png`, `q1_population_offsets.png`, `q1_lw02_variant.png`, `q1_summary.json`

**Interfaces consumed:** `ssp_grid.load_ssp_grid`, `ssp_grid.build_ssp` (add an optional `extra_params: dict` argument to `build_ssp` that sets more `population.params` keys, default empty, recorded in provenance), `spectral_indices.measure_all / h_minus_bump / d4000 / hdelta_a / flux_nu_to_flux_lambda / H_MINUS_BANDS_A`, `broadening.make_resolution_product`, `run_single_csp.compute_track_indices`, the population table `output/population/indices.npz`.

- [ ] **Step 1: SSP-level deltas.** Load `sigma300` and `r100`. For each metallicity and age compute `delta_index = index(agb2) - index(agb0)` with `agb2 = 2 * flux[:, 1] - flux[:, 0]`, for the three indices (bump at both products). Figure `q1_ssp_delta_vs_age.png`: 4 rows (D4000, HdeltaA, bump sigma300, bump r100) versus age (log axis, 10 Myr to 20 Gyr), one line per metallicity, a gray band showing the spread across the four metallicities of the agb0 index at each age (max minus min, centered on zero), and dashed horizontal lines at ± the yardstick precisions. Record in `q1_summary.json`: for each index, the maximum |delta| over ages 0.1–13 Gyr and the age where it occurs (per Z and overall), and the maximum metallicity spread of the agb0 index over the same ages.

- [ ] **Step 2: Component spectrum diagnostic.** `component = flux[:, 1] - flux[:, 0]` (the pure TP-AGB light per Msun) at solar Z. Compute its bump index versus age (only where the component's feature-band mean flux is positive) and the fraction of the feature-band light supplied by TP-AGB, `component_band / flux_agb1_band`, versus age. Figure `q1_component_spectrum.png`: left panel both curves versus age; right panel the component's continuum-normalized F_lambda over 1.3–2.0 micron at 0.3, 1 and 3 Gyr with the bands shaded. Record the component bump range and the peak TP-AGB light fraction (value and age).

- [ ] **Step 3: NIR spectra QA.** Figure `q1_nir_spectra.png`: two rows (sigma300, r100) times four columns (ages 0.3, 1, 2, 5 Gyr at solar Z): the ratio S(agb2)/S(agb0) (top of each panel, or a twin axis) and the F_lambda spectra of agb0 and agb2 each normalized by their own linear pseudo-continuum (the `_feature_ratio_mean` construction: divide by the line through the blue and red band means), bands shaded, 1.3–2.0 micron. The point is to show that the TP-AGB light is a tilted continuum that the pseudo-continuum removes, and where CO band heads (1.578, 1.598, 1.619, 1.640, 1.661, 1.684, 1.707 micron; draw thin vertical ticks) fall inside the feature window.

- [ ] **Step 4: Broadband contrast.** For every SSP age and metallicity compute `ratio = F_nu(16000 A) / F_nu(4200 A)` (nearest pixel on the native grid) for agb0 and agb2. Figure `q1_broadband_ratio.png`: the ratio versus age for both agb settings (four Z as lines) and the relative difference. Record the relative difference at 0.3, 1, 2, 5 Gyr (solar Z). This is the "what a flux-calibrated measurement sees" comparison.

- [ ] **Step 5: Empirical TP-AGB spectra variant.** Build with `build_ssp(0.0, agb, extra_params={"use_lw_tpagb": 1})` for agb 0 and 1 (about 25 s), smooth to sigma300 with `make_resolution_product` on a one-metallicity `SspGrid` (construct it with `log_z_grid=np.array([0.0])` and `flux_nu` of shape (1, 2, 107, n_pix)), and repeat Step 1's bump delta versus age next to the default C3K-grid curve. Figure `q1_lw02_variant.png`. Record both maximum deltas. Cache the two variant SSPs to `output/ssp_grid/lw02_solar_native.npz` (git-ignored by the existing pattern) so re-runs are fast.

- [ ] **Step 6: Population-level offsets.** From the population table (epochs >= 1 Gyr): in 12 bins of `d4000_agb0` between its 1st and 99th percentiles, and separately in 12 bins of `hdelta_a_agb0`, compute for the bump (sigma300 and r100) the median and 16/84 percentiles for agb0 and agb2, the median paired difference (agb2 - agb0 per row), and the offset divided by the pooled 16–84 half-width. Figure `q1_population_offsets.png`: two rows (bin by D4000, by HdeltaA) times two columns (bump product): bands for agb0 and agb2 with medians, and a lower inset or twin axis with offset/scatter and the 0.01 mag yardstick converted to the same units. Record per-bin numbers and the maximum offset/scatter.

- [ ] **Step 7: Pilot, full run, inspect.** `--pilot` uses only solar Z and 3 ages and skips Step 5 and the figures. Run the full script, look at every PNG with the Read tool, fix layout problems. Ruff clean.

- [ ] **Step 8: Commit** — the script, `ssp_grid.py` change, the PNGs and `q1_summary.json`; subject "Add Q1 TP-AGB separability analysis".

---

### Task 3: Q2 — isolating fast quenching

**Files:**
- Create: `analysis_fast_quenching.py`
- Output: `output/analysis/q2_class_planes.png`, `q2_purity_maps.png`, `q2_classifier.png`, `q2_summary.json`

- [ ] **Step 1: Classes.** Load the population table, keep `epoch_gyr >= 1.0`, assign classes with `population_classes.assign_classes` from `ssfr_0_100_myr` and `ssfr_100_1000_myr`. Record class counts and fractions (and the post-starburst count) in `q2_summary.json`; they must reproduce the numbers in the ledger to the sample (star-forming 21.2 percent, rapid-quenching 0.98 percent, quiescent 54.8 percent, transitional 23.0 percent of the >= 1 Gyr epochs).

- [ ] **Step 2: Class planes.** Figure `q2_class_planes.png`: for agb2 (top row) and agb0 (bottom row), the three planes with a hexbin density of the whole population in light gray and the four classes as scatter with distinct colors (rapid-quenching drawn last and larger); legend with counts.

- [ ] **Step 3: Purity maps.** For each plane and each agb setting, a 2-D histogram on a 40 x 40 grid spanning the 0.5–99.5 percentiles of each axis; per cell the fraction of rapid-quenching points among all points (cells with fewer than 20 points masked). Figure `q2_purity_maps.png`: 2 rows (agb2, agb0) x 3 planes, color = purity, contour of the rapid-quenching density on top. Record the maximum cell purity per plane and the fraction of rapid-quenching points lying in cells with purity > 0.5 (the "isolable" fraction).

- [ ] **Step 4: Noise-aware classification.** Features: (D4000, HdeltaA) and (D4000, HdeltaA, bump) for bump at sigma300 and at r100, for agb2 and agb0. Noise: D4000 0.05, HdeltaA 0.5 A, bump at each of 0.005, 0.010, 0.020 mag; `feature_scales` equal to the noise sigmas. `cross_validated_metrics` with k = 25, 5 folds grouped by `history_id`, seed 20260924, positive class rapid-quenching. Because rapid-quenching is 1 percent of samples, also report results after down-weighting: repeat with a balanced training set (subsample each class to the rapid-quenching count in the training fold, fixed seed) and report both. Figure `q2_classifier.png`: completeness and purity (with fold scatter as error bars) versus bump precision, one line per feature set, panels for agb2/agb0 and for unbalanced/balanced. Record everything in `q2_summary.json` including the confusion matrices.

- [ ] **Step 5: Pilot, full run, inspect.** `--pilot` uses 1 of every 50 rows and 2 folds. Full run; look at the PNGs; Ruff.

- [ ] **Step 6: Commit** — subject "Add Q2 fast-quenching isolation analysis".

---

### Task 4: Alternative SFH families and the written answer

**Files:**
- Modify: `sfh_model.py` (add `burst_cumulative_mass(time_gyr, t_burst_gyr, width_gyr, mass_fraction, base_cumulative)`), `csp_integrate.py` (add `epoch_weight_matrix_from_cumulative(edges_gyr, cumulative_mass_fn, log_age_grid_yr)` and make `epoch_weight_matrix` call it with `functools.partial(cumulative_mass, t_q_gyr=..., tau_q_gyr=...)`)
- Create: `analysis_alternative_sfh.py`, `docs/ANALYSIS.md`
- Test: `tests/test_sfh_burst.py`
- Output: `output/analysis/q2_alternative_sfh.png`, `q2_alternative_summary.json`

- [ ] **Step 1: Failing tests.** `burst_cumulative_mass` of a Gaussian burst centred at 4 Gyr, width 0.1 Gyr, fraction 0.1 on top of the fiducial delayed-tau (no quench, i.e. `tau_q` very large) must satisfy: total mass at 13 Gyr equals base total / (1 - 0.1) times... define it precisely as `base(t) + mass_fraction * base(13) * 0.5 * (1 + erf((t - t_burst) / (sqrt(2) * width)))`, so the burst adds `mass_fraction` of the base's final mass; test that the added mass at 13 Gyr equals `0.1 * base(13)` within 1e-9 and that the derivative at t_burst equals the Gaussian peak within 1e-6. `epoch_weight_matrix_from_cumulative` with the delayed-tau partial must equal `epoch_weight_matrix` exactly (allclose, atol 0).

- [ ] **Step 2: Implement** the two functions; run the tests; run the existing suite (`uv run pytest -q`) to confirm nothing changed.

- [ ] **Step 3: Alternative families.** Draw 300 histories each with seed 20260925: (a) bursty star-forming: delayed-tau with `t_q` from the prior but no quench (`tau_q = 1e6` Gyr) plus a burst at `t_burst` uniform in [2, 10] Gyr, width 0.1 Gyr, fraction 0.1; (b) slowly fading: delayed-tau with `tau_q` log-uniform in [3, 6] Gyr; log Z from the same truncated normal. Trace all 260 epochs with the agb2 and agb0 settings using the same code path as `run_population._history_indices` (import it; pass a cumulative-mass callable, which requires `_history_indices` to accept one — add an optional `cumulative_mass_fn` parameter there with the default behaviour unchanged). Assign classes with the same rules. Figure `q2_alternative_sfh.png`: the three planes (agb2) with the delayed-tau rapid-quenching points, the bursty family (colored by time since burst) and the slow-fading family; then the purity maps of Task 3 recomputed with the contaminants added. Record: how many contaminant epochs fall inside the cells that were > 0.5 pure before, and the new purity there.

- [ ] **Step 4: Write `docs/ANALYSIS.md`.** Two sections, "Q1" and "Q2", each with: the answer in one sentence (yes / no / conditional), the evidence as a list of measured numbers pulled from the three summary JSON files (quote the file and key), the figures that show it (file names), and the caveats from the spec (model physics: hydrostatic C3K spectra for O-rich TP-AGB stars, no C-rich stars in MIST, no nebular emission or dust, four metallicities, uniform time weighting). Do not soften a negative answer.

- [ ] **Step 5: Update `docs/todo.md`** (tick Phase 3, add a Review), `README.md` (Phase 3 section: scripts, run order, key numbers), `docs/lessons.md` (dated bullets with timings and anything surprising). Run `uv run pytest -q`, `uv run pre-commit run --all-files`.

- [ ] **Step 6: Commit** — subject "Add alternative SFH robustness test and the Phase 3 answers".

---

## Self-review against the spec

- Q1 items 1 (SSP deltas, spectral QA, broadband) and 2 (track and population offsets): Task 2. The fiducial-track delta versus time since quenching is already in `output/single_csp/` (Task 10 figures); Task 2's Step 6 covers the population part and its summary quotes the track maximum from `output/single_csp/indices.csv`.
- Q2 items 1–3: Tasks 3 and 4. Class rules, yardstick, epoch cut: Global Constraints.
- Deliverables (figures, summary.json, ANALYSIS.md): Tasks 2, 3, 4.
- Interfaces used across tasks (`assign_classes`, `cross_validated_metrics`, `build_ssp(extra_params=...)`, `epoch_weight_matrix_from_cumulative`, `_history_indices(cumulative_mass_fn=...)`) are named identically where defined and where used.
