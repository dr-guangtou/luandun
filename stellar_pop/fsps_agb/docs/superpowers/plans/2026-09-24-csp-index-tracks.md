# CSP Index Tracks and Populations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trace D4000, HdeltaA and the 1.6 micron H-minus bump in time for delayed-tau plus exponential-quench galaxies built from MIST + C3K_HR + Chabrier SSPs with the TP-AGB weight `agb` = 0 and 2, first for one fiducial history and then for a 2000-history population.

**Architecture:** FSPS is used only to build and cache SSP grids (4 metallicities x 2 `agb` values x 107 ages). Everything else is numpy: SSPs are smoothed once to the two resolution products, CSPs are one matrix product of SFH bin masses against log-age- and log-Z-interpolated SSPs, and indices are measured on the fly. FSPS's own tabular-SFH CSP is used once as a cross-check.

**Tech Stack:** Python 3.12, uv, numpy, scipy, matplotlib, pytest, ruff; python-fsps 0.5.1.dev0 rebuilt from source (scikit-build-core, cmake, ninja, gfortran) against FSPS bd187a0 with the C3K_HR loader patch; data from `$SPS_HOME=/Users/shuang/code/fsps`.

**Spec:** `docs/SPEC.md` (section "Phase 2 — CSP index tracks and populations").

## Global Constraints

- All code, comments, docs and commits in English; `snake_case` everywhere, never camelCase; names are complete words.
- No comments that restate a name. Docstrings state units.
- Dependencies through `uv` only (never pip inside the project env); Ruff via pre-commit is the only linter/formatter.
- Work stays on branch `fsps-agb-experiment` in the parent repo `/Users/shuang/Dropbox/work/project/luandun`; never merge without permission. All paths below are relative to `stellar_pop/fsps_agb/` unless absolute.
- Phase 1 scripts (`agb_experiment.py`, `agb_pagb_response.py`, `inspect_agb_templates.py`, `plot_nir_comparison.py`, `norm_utils.py`) are not modified.
- Fixed configuration: MIST, `c3k_hr`, `imf_type = 1` (Chabrier), `agb` in {0, 1} built, `agb` = 2 derived by linearity, all other parameters at python-fsps defaults, `sfh = 0`, `zcontinuous = 0`, metallicities log(Z/Zsun) = -0.50, -0.25, 0.00, +0.25 (`zmet` = 9, 10, 11, 12).
- Wavelength window kept: 3400 A to 22000 A (vacuum). Log-wavelength grid step 30 km/s. Native library sigma: 42.4378 km/s below 10000 A, 254.6267 km/s above (read from `sp.resolutions`, never hard-coded in production code).
- Resolution products: `sigma300` (target total sigma 300 km/s, whole window), `r100` (target total sigma sqrt((c / (2.35482 x 100))^2 + 300^2) over 12500 to 21000 A). Native sigma is subtracted in quadrature.
- SFH: `SFR(t) = (t/t_q) exp(-t/t_q)` for `t < t_q`, `exp(-1) exp(-(t - t_q)/tau_q)` for `t >= t_q`; time bins 0.05 Gyr from 0 to 13 Gyr; epochs are the bin edges 0.05 ... 13.0 Gyr; bin masses are analytic integrals.
- Fiducial: `t_q = 3.0`, `tau_q = 0.3`, `log_z = 0.0`. Priors: `t_q` uniform [1, 5.9] Gyr, `tau_q` log-uniform [0.1, 3] Gyr, `log_z` normal (0.0, 0.2) truncated to [-0.5, 0.2]. 2000 draws, seed 20260924.
- Index bands: D4000 blue 3750–3950 A, red 4050–4250 A (air), ratio of mean F_nu red/blue. HdeltaA blue 4041.60–4079.75, feature 4083.50–4122.25, red 4128.50–4161.00 A (air), EW in A on F_lambda. H-minus blue 14940–15390, feature 15700–17340, red 17460–17910 A (vacuum), magnitudes on F_lambda, negative for a bump. Air bands are converted to vacuum with the Morton (1991) formula used by FSPS.
- Validation thresholds (fixed in advance): null indices below 1e-10; broadening sigma recovery within 1 percent; `agb` linearity within 1e-10 relative; integrator vs FSPS tabular flux within 2 percent inside the index windows.
- Small scale first: every driver has a `--pilot` flag that runs a sub-minute subset and prints timings before any full run.
- Ruff (rules E, F, I, N, UP, B) must pass. Where the plan shows `matplotlib.use("Agg")` between imports, place all imports first and call `matplotlib.use("Agg")` after them (valid as long as no figure exists yet); do not add `noqa` comments.

---

## File structure

| File | Responsibility |
| ---- | -------------- |
| `pyproject.toml`, `uv.lock`, `.python-version`, `.pre-commit-config.yaml` | uv project on Python 3.12 with the local fsps wheel; Ruff hook |
| `docs/patches/c3k_hr_nzinit_13.patch` | the one-line FSPS fix, applied to the python-fsps submodule |
| `sfh_model.py` | SFH function, bin edges, analytic bin masses, prior draws |
| `ssp_grid.py` | build SSPs with FSPS, cache to `output/ssp_grid/`, load as `SspGrid` |
| `broadening.py` | log-wavelength grid, resampling, quadrature-corrected Gaussian smoothing, resolution products |
| `csp_integrate.py` | age-interpolation weights, log-Z interpolation, CSP matrix product |
| `spectral_indices.py` | air-to-vacuum, band means, D4000, HdeltaA, H-minus bump |
| `index_planes.py` | shared figure helpers for the three 2-D index planes |
| `run_single_csp.py` | Step 1 driver (spectra, index table, FSPS cross-check, figures) |
| `run_population.py` | Step 2 driver (draws, index table, figures) |
| `tests/test_*.py` | unit tests per module; FSPS-dependent tests marked `slow` |

Interfaces shared by every task (defined in Task 3 to 7, repeated here so a reader of any single task knows them):

```python
# sfh_model.py
TIME_STEP_GYR = 0.05
TIME_END_GYR = 13.0
def time_bin_edges(step_gyr=TIME_STEP_GYR, end_gyr=TIME_END_GYR) -> np.ndarray   # shape (261,)
def star_formation_rate(time_gyr, t_q_gyr, tau_q_gyr) -> np.ndarray             # same shape as time_gyr
def bin_masses(edges_gyr, t_q_gyr, tau_q_gyr) -> np.ndarray                     # shape (len(edges) - 1,), Msun for SFR in Msun/Gyr
def draw_population(n_draws, seed) -> dict[str, np.ndarray]                     # keys t_q_gyr, tau_q_gyr, log_z

# ssp_grid.py
@dataclass SspGrid: wave_a (n_pix,), log_age_yr (107,), log_z_grid (4,), agb_weights (2,), flux_nu (4, 2, 107, n_pix), product (str), provenance (dict)
def build_ssp(log_z, agb) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]   # wave_a, log_age_yr, flux_nu (107, n_pix), provenance
def build_and_cache_grid(out_dir) -> Path                                       # writes native.npz + provenance.json
def load_ssp_grid(out_dir, product) -> SspGrid                                  # product in {"native", "sigma300", "r100"}
def save_ssp_grid(grid, out_dir) -> Path

# broadening.py
SPEED_OF_LIGHT_KM_S = 299792.458
def log_wavelength_grid(wave_min_a, wave_max_a, velocity_step_km_s=30.0) -> np.ndarray
def resample_flux(wave_a, flux, target_wave_a) -> np.ndarray                    # linear, flux may be (..., n_pix)
def gaussian_broaden(log_wave_a, flux_nu, sigma_km_s) -> np.ndarray             # convolves flux per log wavelength
def added_sigma_km_s(target_sigma_km_s, native_sigma_km_s) -> float
def make_resolution_product(grid: SspGrid, product, native_sigma_km_s_per_pix) -> SspGrid

# csp_integrate.py
def age_weights(log_age_grid_yr, lookback_gyr) -> np.ndarray                    # (n_lookback, n_age), rows sum to 1
def epoch_weight_matrix(edges_gyr, masses, log_age_grid_yr) -> np.ndarray       # (n_epochs, n_age); epoch k uses edges[k]
def interpolate_log_z(flux_by_z, log_z_grid, log_z) -> np.ndarray               # drops the leading z axis
def csp_spectra(weight_matrix, ssp_flux) -> np.ndarray                          # (n_epochs, n_pix), per Msun formed

# spectral_indices.py
def air_to_vacuum(wave_a) -> np.ndarray
def flux_nu_to_flux_lambda(wave_a, flux_nu) -> np.ndarray
def band_mean(wave_a, flux, lower_a, upper_a) -> np.ndarray                     # flux (..., n_pix) -> (...)
def d4000(wave_a, flux_nu) -> np.ndarray
def hdelta_a(wave_a, flux_nu) -> np.ndarray
def h_minus_bump(wave_a, flux_nu) -> np.ndarray
def measure_all(wave_a, flux_nu) -> dict[str, np.ndarray]                       # keys d4000, hdelta_a, h_minus_bump
```

---

### Task 1: Patch the C3K_HR loader and rebuild the python-fsps wheel

**Files:**
- Create: `docs/patches/c3k_hr_nzinit_13.patch`
- Create: `wheels/` (git-ignored; holds the built wheel)
- Modify: `.gitignore`
- Modify (outside the repo): `/Users/shuang/code/python-fsps/src/fsps/libfsps` (submodule checkout + patch), `/Users/shuang/code/python-fsps/src/fsps/CMakeLists.txt`, `/Users/shuang/code/fsps/src/sps_vars.f90` (same patch, keeps `$SPS_HOME` consistent with the compiled code)
- Test: manual verification script in this task (the pre-fix reference spectrum already exists at `output/pre_fix_zmet12_chabrier_1gyr.npz`)

**Interfaces:**
- Consumes: nothing.
- Produces: a wheel `wheels/fsps-0.5.1.dev0*-cp312-cp312-macosx_*.whl` whose `StellarPopulation().libraries == (b'mist', b'c3k_hr', b'DL07')` and which loads 13 C3K_HR metallicities.

- [ ] **Step 1: Check out the FSPS submodule at the commit matching `$SPS_HOME`**

```bash
cd /Users/shuang/code/python-fsps/src/fsps/libfsps
git fetch origin
git checkout bd187a0
git log --oneline -1   # expect: bd187a0 [ci skip] DOC: Change to documentation ...
diff <(ls src/*.f90 | xargs -n1 basename) <(ls /Users/shuang/code/fsps/src/*.f90 | xargs -n1 basename) && echo "same file list"
```

- [ ] **Step 2: Write the patch file and apply it to both copies**

Create `docs/patches/c3k_hr_nzinit_13.patch`:

```diff
--- a/src/sps_vars.f90
+++ b/src/sps_vars.f90
@@ -320,5 +320,5 @@
   REAL(SP), PARAMETER :: zsol_spec = 0.0185
   CHARACTER(7), PARAMETER :: spec_type = 'c3k_hr'
   INTEGER, PARAMETER      :: ndim_logt=80, ndim_logg=14
-  INTEGER, PARAMETER :: nzinit=11
+  INTEGER, PARAMETER :: nzinit=13
   INTEGER, PARAMETER :: nspec=10992
```

Apply:

```bash
cd /Users/shuang/code/python-fsps/src/fsps/libfsps
git apply --check /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/docs/patches/c3k_hr_nzinit_13.patch && git apply /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/docs/patches/c3k_hr_nzinit_13.patch
cd /Users/shuang/code/fsps
git apply --check /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/docs/patches/c3k_hr_nzinit_13.patch && git apply /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/docs/patches/c3k_hr_nzinit_13.patch
grep -n "nzinit=13" src/sps_vars.f90   # expect one hit in the c3k_hr block only
```

If `git apply --check` fails because the hunk line numbers moved, edit the `nzinit=11` line in the block that contains `spec_type = 'c3k_hr'` by hand and regenerate the patch with `git diff src/sps_vars.f90 > <patch path>`. Do not touch the `c3k_lr` block.

- [ ] **Step 3: Add the C3K_HR compile definitions to python-fsps**

Append to `/Users/shuang/code/python-fsps/src/fsps/CMakeLists.txt`, after the `python_add_library(...)` call:

```cmake
target_compile_options(_fsps PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-cpp;-DC3K_LR=0;-DC3K_HR=1>)
```

- [ ] **Step 4: Build the wheel with uv**

```bash
mkdir -p /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/wheels
cd /Users/shuang/code/python-fsps
uv build --wheel --python 3.12 --out-dir /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/wheels .
ls -la /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb/wheels
```

Expected: one file `fsps-0.5.1.dev0+<hash>.d<date>-cp312-cp312-macosx_*.whl`. If `uv build` cannot find gfortran, run it with `FC=/opt/homebrew/bin/gfortran`. If the build backend refuses because of the dirty submodule version string, that is fine; the `dev0+...` tag is expected.

- [ ] **Step 5: Verify the wheel in a throwaway environment**

```bash
cd /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb
uv run --isolated --no-project --python 3.12 --with wheels/fsps-*.whl --with numpy python - <<'EOF'
import numpy as np, fsps
sp = fsps.StellarPopulation(zcontinuous=0, zmet=12, imf_type=1)
print(fsps.__version__, sp.libraries)
w, s = sp.get_spectrum(tage=1.0)
old = np.load("output/pre_fix_zmet12_chabrier_1gyr.npz")
m = (w > 3400) & (w < 22000)
print("max |dF/F| zmet=12 old vs new:", np.max(np.abs(s[m] / old["flux_nu"][m] - 1)))
sp11 = fsps.StellarPopulation(zcontinuous=0, zmet=11, imf_type=1)
w11, s11 = sp11.get_spectrum(tage=1.0)
print("max |dF/F| zmet=11 vs zmet=12 new:", np.max(np.abs(s11[m] / s[m] - 1)))
EOF
```

Expected: libraries `(b'mist', b'c3k_hr', b'DL07')`; the zmet=12 difference old vs new is larger than 1e-3 (supersolar spectra now loaded); the zmet=11 vs zmet=12 difference is also larger than 1e-3. Record the two numbers in `docs/lessons.md`.

- [ ] **Step 6: Git-ignore the wheel directory and commit the patch**

Append to `.gitignore`:

```
wheels/
.venv/
.ruff_cache/
.pytest_cache/
output/ssp_grid/*.npz
```

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/.gitignore stellar_pop/fsps_agb/docs/patches/c3k_hr_nzinit_13.patch stellar_pop/fsps_agb/docs/lessons.md
git commit -m "Add C3K_HR nzinit patch and wheel build recipe"
```

---

### Task 2: uv project, pre-commit and test scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.pre-commit-config.yaml`, `tests/__init__.py`, `tests/conftest.py`
- Modify: `README.md` (setup section)

**Interfaces:**
- Produces: `uv run pytest` and `uv run python <script>` work with `fsps` importable and `$SPS_HOME` set.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "fsps-agb"
version = "0.2.0"
description = "TP-AGB sensitivity of D4000, HdeltaA and the 1.6 micron H-minus bump with FSPS"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "matplotlib>=3.9",
    "fsps",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "pre-commit>=3"]

[tool.uv.sources]
fsps = { path = "wheels/FSPS_WHEEL_FILENAME.whl" }

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs FSPS and takes more than a few seconds"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B"]
```

Replace `FSPS_WHEEL_FILENAME.whl` with the real filename produced in Task 1 (`ls wheels/`), for example `fsps-0.5.1.dev0+g7d202b8e0.d20260924-cp312-cp312-macosx_15_0_arm64.whl`. uv sources must point at the wheel file itself, not the directory.

- [ ] **Step 2: Pin Python and sync**

```bash
cd /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb
echo "3.12" > .python-version
uv sync
uv run python -c "import fsps, numpy, scipy, matplotlib; print(fsps.__version__, fsps.StellarPopulation().libraries)"
```

Expected: prints the dev version and `(b'mist', b'c3k_hr', b'DL07')`. If `SPS_HOME` is not visible, export it in the shell; do not hard-code it in the code.

- [ ] **Step 3: Write the pre-commit config and install the hook**

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: ruff passes on the existing files, or fixes only formatting; review the diff on Phase 1 scripts and revert any change to them (`git checkout -- <file>`) so the Phase 1 code stays untouched, then add `exclude: ^(agb_experiment|agb_pagb_response|inspect_agb_templates|plot_nir_comparison|norm_utils)\.py$` at the top level of the hook config.

- [ ] **Step 4: Write the test scaffold**

`tests/__init__.py`: empty.

`tests/conftest.py`:

```python
import numpy as np
import pytest


@pytest.fixture
def linear_spectrum():
    """A spectrum linear in wavelength: every pseudo-continuum index must be zero on it."""
    wave_a = np.linspace(3300.0, 22500.0, 40000)
    flux_lambda = 3.0 - 1e-4 * wave_a
    return wave_a, flux_lambda
```

```bash
uv run pytest -q
```

Expected: `no tests ran` (exit code 5 is acceptable here).

- [ ] **Step 5: Document setup in README and commit**

Add to `README.md` under Setup:

```markdown
## Phase 2 setup

    uv sync                     # installs numpy/scipy/matplotlib and the local fsps wheel from wheels/
    export SPS_HOME=/Users/shuang/code/fsps
    uv run pytest               # fast tests; add `-m slow` for the FSPS-dependent ones
    uv run pre-commit install

The wheel in `wheels/` is built from python-fsps 7d202b8 with the FSPS submodule at bd187a0 plus
`docs/patches/c3k_hr_nzinit_13.patch`, compiled with `-cpp -DC3K_LR=0 -DC3K_HR=1` (see docs/lessons.md).
```

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/pyproject.toml stellar_pop/fsps_agb/uv.lock stellar_pop/fsps_agb/.python-version stellar_pop/fsps_agb/.pre-commit-config.yaml stellar_pop/fsps_agb/tests stellar_pop/fsps_agb/README.md
git commit -m "Set up uv project, Ruff pre-commit and test scaffold"
```

---

### Task 3: SFH model, bin masses and prior draws

**Files:**
- Create: `sfh_model.py`
- Test: `tests/test_sfh_model.py`

**Interfaces:**
- Produces: `time_bin_edges`, `star_formation_rate`, `bin_masses`, `draw_population` as listed in the file structure section. SFR is in Msun per Gyr for a unit-normalized history (absolute scale irrelevant).

- [ ] **Step 1: Write the failing tests**

`tests/test_sfh_model.py`:

```python
import numpy as np
from scipy.integrate import quad

from sfh_model import bin_masses, draw_population, star_formation_rate, time_bin_edges


def test_time_bin_edges_default_grid():
    edges = time_bin_edges()
    assert edges.shape == (261,)
    assert edges[0] == 0.0
    assert np.isclose(edges[-1], 13.0)
    assert np.allclose(np.diff(edges), 0.05)


def test_star_formation_rate_is_continuous_at_t_q():
    t_q, tau_q = 3.0, 0.3
    before = star_formation_rate(np.array([t_q - 1e-9]), t_q, tau_q)
    after = star_formation_rate(np.array([t_q + 1e-9]), t_q, tau_q)
    assert np.isclose(before, after, rtol=1e-6)
    assert np.isclose(after, np.exp(-1.0), rtol=1e-6)


def test_bin_masses_match_numerical_integration():
    t_q, tau_q = 3.0, 0.3
    edges = time_bin_edges()
    masses = bin_masses(edges, t_q, tau_q)
    assert masses.shape == (260,)
    for i in (0, 10, 59, 60, 61, 100, 259):
        expected, _ = quad(lambda t: star_formation_rate(np.array([t]), t_q, tau_q)[0],
                           edges[i], edges[i + 1], points=[t_q])
        assert np.isclose(masses[i], expected, rtol=1e-8), i


def test_bin_masses_handle_bin_straddling_t_q():
    t_q, tau_q = 3.02, 0.5
    edges = time_bin_edges()
    masses = bin_masses(edges, t_q, tau_q)
    expected, _ = quad(lambda t: star_formation_rate(np.array([t]), t_q, tau_q)[0],
                       3.0, 3.05, points=[t_q])
    assert np.isclose(masses[60], expected, rtol=1e-8)


def test_draw_population_respects_priors():
    draws = draw_population(20000, seed=1)
    assert set(draws) == {"t_q_gyr", "tau_q_gyr", "log_z"}
    assert draws["t_q_gyr"].min() >= 1.0 and draws["t_q_gyr"].max() <= 5.9
    assert draws["tau_q_gyr"].min() >= 0.1 and draws["tau_q_gyr"].max() <= 3.0
    assert draws["log_z"].min() >= -0.5 and draws["log_z"].max() <= 0.2
    log_tau = np.log10(draws["tau_q_gyr"])
    counts, _ = np.histogram(log_tau, bins=5, range=(-1.0, np.log10(3.0)))
    assert counts.max() / counts.min() < 1.2
    assert abs(np.mean(draws["log_z"])) < 0.03


def test_draw_population_is_reproducible():
    a = draw_population(10, seed=5)
    b = draw_population(10, seed=5)
    assert np.array_equal(a["t_q_gyr"], b["t_q_gyr"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sfh_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sfh_model'`.

- [ ] **Step 3: Implement `sfh_model.py`**

```python
"""Delayed-tau star formation followed by exponential quenching, with tau = t_q.

Times are in Gyr since the start of star formation. The SFR is dimensionless
up to a constant; only relative masses matter for spectral indices.
"""

import numpy as np
from scipy.stats import truncnorm

TIME_STEP_GYR = 0.05
TIME_END_GYR = 13.0

T_Q_RANGE_GYR = (1.0, 5.9)
TAU_Q_RANGE_GYR = (0.1, 3.0)
LOG_Z_RANGE = (-0.5, 0.2)
LOG_Z_MEAN = 0.0
LOG_Z_SIGMA = 0.2


def time_bin_edges(step_gyr=TIME_STEP_GYR, end_gyr=TIME_END_GYR):
    n_bins = int(round(end_gyr / step_gyr))
    return np.linspace(0.0, n_bins * step_gyr, n_bins + 1)


def star_formation_rate(time_gyr, t_q_gyr, tau_q_gyr):
    time_gyr = np.asarray(time_gyr, dtype=float)
    forming = (time_gyr / t_q_gyr) * np.exp(-time_gyr / t_q_gyr)
    quenching = np.exp(-1.0) * np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr)
    return np.where(time_gyr < t_q_gyr, forming, quenching)


def _forming_cumulative_mass(time_gyr, t_q_gyr):
    """Integral of (t / t_q) exp(-t / t_q) from 0 to time_gyr."""
    return t_q_gyr - np.exp(-time_gyr / t_q_gyr) * (time_gyr + t_q_gyr)


def _quenching_cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr):
    """Integral of exp(-1) exp(-(t - t_q) / tau_q) from t_q to time_gyr."""
    return np.exp(-1.0) * tau_q_gyr * (1.0 - np.exp(-(time_gyr - t_q_gyr) / tau_q_gyr))


def cumulative_mass(time_gyr, t_q_gyr, tau_q_gyr):
    time_gyr = np.asarray(time_gyr, dtype=float)
    before = _forming_cumulative_mass(np.minimum(time_gyr, t_q_gyr), t_q_gyr)
    after = np.where(
        time_gyr > t_q_gyr,
        _quenching_cumulative_mass(np.maximum(time_gyr, t_q_gyr), t_q_gyr, tau_q_gyr),
        0.0,
    )
    return before + after


def bin_masses(edges_gyr, t_q_gyr, tau_q_gyr):
    cumulative = cumulative_mass(edges_gyr, t_q_gyr, tau_q_gyr)
    return np.diff(cumulative)


def draw_population(n_draws, seed):
    rng = np.random.default_rng(seed)
    t_q_gyr = rng.uniform(*T_Q_RANGE_GYR, size=n_draws)
    log_tau_q = rng.uniform(np.log10(TAU_Q_RANGE_GYR[0]), np.log10(TAU_Q_RANGE_GYR[1]), size=n_draws)
    lower = (LOG_Z_RANGE[0] - LOG_Z_MEAN) / LOG_Z_SIGMA
    upper = (LOG_Z_RANGE[1] - LOG_Z_MEAN) / LOG_Z_SIGMA
    log_z = truncnorm.rvs(lower, upper, loc=LOG_Z_MEAN, scale=LOG_Z_SIGMA, size=n_draws,
                          random_state=rng)
    return {"t_q_gyr": t_q_gyr, "tau_q_gyr": 10.0**log_tau_q, "log_z": log_z}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sfh_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/sfh_model.py stellar_pop/fsps_agb/tests/test_sfh_model.py
git commit -m "Add delayed-tau plus exponential-quench SFH model and priors"
```

---

### Task 4: Spectral indices

**Files:**
- Create: `spectral_indices.py`
- Test: `tests/test_spectral_indices.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `air_to_vacuum`, `flux_nu_to_flux_lambda`, `band_mean`, `d4000`, `hdelta_a`, `h_minus_bump`, `measure_all`. All accept `flux` of shape `(..., n_pix)` and return shape `(...)`. `wave_a` is vacuum Angstrom, increasing.

- [ ] **Step 1: Write the failing tests**

`tests/test_spectral_indices.py`:

```python
import numpy as np

from spectral_indices import (
    H_MINUS_BANDS_A,
    HDELTA_A_BANDS_A,
    air_to_vacuum,
    band_mean,
    d4000,
    flux_nu_to_flux_lambda,
    h_minus_bump,
    hdelta_a,
    measure_all,
)


def test_air_to_vacuum_matches_fsps_value():
    # FSPS (vacairconv.f90, Morton 1991): 5000 A air -> 5001.394 A vacuum
    assert np.isclose(air_to_vacuum(np.array([5000.0]))[0], 5001.394, atol=0.002)


def test_band_mean_of_linear_flux_is_midpoint_value():
    wave = np.linspace(4000.0, 4300.0, 301)
    flux = 2.0 + 0.01 * wave
    assert np.isclose(band_mean(wave, flux, 4050.0, 4250.0), 2.0 + 0.01 * 4150.0, rtol=1e-10)


def test_band_mean_uses_exact_band_limits():
    wave = np.linspace(4000.0, 4300.0, 31)  # 10 A pixels, band edges fall between pixels
    flux = np.ones_like(wave)
    assert np.isclose(band_mean(wave, flux, 4053.0, 4247.0), 1.0, rtol=1e-12)


def test_indices_vanish_on_linear_spectrum(linear_spectrum):
    wave_a, flux_lambda = linear_spectrum
    flux_nu = flux_lambda * wave_a**2
    assert abs(hdelta_a(wave_a, flux_nu)) < 1e-10
    assert abs(h_minus_bump(wave_a, flux_nu)) < 1e-10


def test_d4000_of_flat_flux_nu_is_one():
    wave_a = np.linspace(3300.0, 4500.0, 5000)
    assert np.isclose(d4000(wave_a, np.ones_like(wave_a)), 1.0, rtol=1e-12)


def test_d4000_is_ratio_of_mean_flux_nu():
    wave_a = np.linspace(3300.0, 4500.0, 12001)
    flux_nu = np.where(wave_a < 4000.0, 1.0, 2.0)
    assert np.isclose(d4000(wave_a, flux_nu), 2.0, rtol=1e-6)


def test_hdelta_a_gaussian_absorption_has_positive_equivalent_width():
    wave_a = np.linspace(4000.0, 4200.0, 20001)
    center = air_to_vacuum(np.array([4101.7]))[0]
    depth, sigma = 0.5, 3.0
    flux_lambda = 1.0 - depth * np.exp(-0.5 * ((wave_a - center) / sigma) ** 2)
    expected = depth * sigma * np.sqrt(2 * np.pi)
    measured = hdelta_a(wave_a, flux_lambda * wave_a**2)
    assert np.isclose(measured, expected, rtol=1e-3)


def test_h_minus_bump_is_negative_for_a_bump():
    wave_a = np.linspace(14000.0, 19000.0, 5001)
    flux_lambda = 1.0 + 0.1 * np.exp(-0.5 * ((wave_a - 16500.0) / 600.0) ** 2)
    value = h_minus_bump(wave_a, flux_lambda * wave_a**2)
    assert value < 0
    assert np.isclose(value, -2.5 * np.log10(band_mean(wave_a, flux_lambda, *H_MINUS_BANDS_A["feature"])), rtol=1e-3)


def test_measure_all_broadcasts_over_spectra(linear_spectrum):
    wave_a, flux_lambda = linear_spectrum
    flux_nu = np.stack([flux_lambda * wave_a**2, 2 * flux_lambda * wave_a**2])
    result = measure_all(wave_a, flux_nu)
    assert result["d4000"].shape == (2,)
    assert np.allclose(result["hdelta_a"], 0.0, atol=1e-10)
    assert np.allclose(result["h_minus_bump"], 0.0, atol=1e-10)


def test_band_definitions_are_vacuum_and_ordered():
    for bands in (HDELTA_A_BANDS_A, H_MINUS_BANDS_A):
        blue, feature, red = bands["blue"], bands["feature"], bands["red"]
        assert blue[0] < blue[1] <= feature[0] < feature[1] <= red[0] < red[1]
    assert np.isclose(HDELTA_A_BANDS_A["feature"][1], air_to_vacuum(np.array([4122.25]))[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_spectral_indices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spectral_indices'`.

- [ ] **Step 3: Implement `spectral_indices.py`**

```python
"""D4000, HdeltaA and the 1.6 micron H-minus bump on vacuum-wavelength model spectra.

Inputs are F_nu (any units) on a vacuum wavelength grid in Angstrom. Bands
defined in air (Lick/IDS, Bruzual D4000) are converted to vacuum with the
Morton (1991) relation used by FSPS. The pseudo-continuum is the straight
line through the band-mean F_lambda of the blue and red windows, anchored
at the band midpoints, following the Lick/IDS convention.
"""

import numpy as np

SPEED_OF_LIGHT_A_PER_S = 2.99792458e18


def air_to_vacuum(wave_a):
    wave_a = np.asarray(wave_a, dtype=float)
    sigma2 = (1e4 / wave_a) ** 2
    factor = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2)
    return np.where(wave_a < 2000.0, wave_a, wave_a * factor)


def _vacuum_band(lower_air_a, upper_air_a):
    return tuple(air_to_vacuum(np.array([lower_air_a, upper_air_a])))


D4000_BANDS_A = {"blue": _vacuum_band(3750.0, 3950.0), "red": _vacuum_band(4050.0, 4250.0)}
HDELTA_A_BANDS_A = {
    "blue": _vacuum_band(4041.60, 4079.75),
    "feature": _vacuum_band(4083.50, 4122.25),
    "red": _vacuum_band(4128.50, 4161.00),
}
H_MINUS_BANDS_A = {
    "blue": (14940.0, 15390.0),
    "feature": (15700.0, 17340.0),
    "red": (17460.0, 17910.0),
}


def flux_nu_to_flux_lambda(wave_a, flux_nu):
    return flux_nu * SPEED_OF_LIGHT_A_PER_S / np.asarray(wave_a) ** 2


def _interpolate(wave_a, flux, target_a):
    """Linear interpolation along the last axis for flux of shape (..., n_pix)."""
    flux = np.asarray(flux)
    flat = flux.reshape(-1, flux.shape[-1])
    values = np.stack([np.interp(target_a, wave_a, row) for row in flat])
    return values.reshape(flux.shape[:-1] + (len(target_a),))


def band_mean(wave_a, flux, lower_a, upper_a):
    """Mean of flux over [lower_a, upper_a] by trapezoid integration on the pixel
    grid augmented with the exact band edges."""
    inside = (wave_a > lower_a) & (wave_a < upper_a)
    nodes = np.concatenate([[lower_a], wave_a[inside], [upper_a]])
    values = _interpolate(wave_a, flux, nodes)
    return np.trapezoid(values, nodes, axis=-1) / (upper_a - lower_a)


def _feature_ratio_mean(wave_a, flux_lambda, bands):
    """Mean over the feature band of F_lambda / F_c, with F_c the line through the
    blue and red band means at their midpoints. Uses 4-point Gauss-Legendre
    quadrature inside every pixel interval, as in the ProGeny resolution study."""
    blue_mean = band_mean(wave_a, flux_lambda, *bands["blue"])
    red_mean = band_mean(wave_a, flux_lambda, *bands["red"])
    blue_mid = 0.5 * sum(bands["blue"])
    red_mid = 0.5 * sum(bands["red"])
    lower, upper = bands["feature"]
    inside = (wave_a > lower) & (wave_a < upper)
    edges = np.concatenate([[lower], wave_a[inside], [upper]])
    nodes, weights = np.polynomial.legendre.leggauss(4)
    half_width = 0.5 * np.diff(edges)
    centers = edges[:-1] + half_width
    target = (centers[:, None] + half_width[:, None] * nodes).ravel()
    quad_weights = (half_width[:, None] * weights / (upper - lower)).ravel()
    fraction = (target - blue_mid) / (red_mid - blue_mid)
    continuum = blue_mean[..., None] * (1.0 - fraction) + red_mean[..., None] * fraction
    return (_interpolate(wave_a, flux_lambda, target) / continuum) @ quad_weights


def d4000(wave_a, flux_nu):
    flux_nu = np.asarray(flux_nu, dtype=float)
    return band_mean(wave_a, flux_nu, *D4000_BANDS_A["red"]) / band_mean(
        wave_a, flux_nu, *D4000_BANDS_A["blue"]
    )


def hdelta_a(wave_a, flux_nu):
    flux_lambda = flux_nu_to_flux_lambda(wave_a, np.asarray(flux_nu, dtype=float))
    lower, upper = HDELTA_A_BANDS_A["feature"]
    return (upper - lower) * (1.0 - _feature_ratio_mean(wave_a, flux_lambda, HDELTA_A_BANDS_A))


def h_minus_bump(wave_a, flux_nu):
    flux_lambda = flux_nu_to_flux_lambda(wave_a, np.asarray(flux_nu, dtype=float))
    return -2.5 * np.log10(_feature_ratio_mean(wave_a, flux_lambda, H_MINUS_BANDS_A))


def measure_all(wave_a, flux_nu):
    return {
        "d4000": d4000(wave_a, flux_nu),
        "hdelta_a": hdelta_a(wave_a, flux_nu),
        "h_minus_bump": h_minus_bump(wave_a, flux_nu),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spectral_indices.py -v`
Expected: 10 passed. If `test_hdelta_a_gaussian_absorption...` fails at the 1e-3 level, the pseudo-continuum is being evaluated on F_nu instead of F_lambda; check `hdelta_a`.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/spectral_indices.py stellar_pop/fsps_agb/tests/test_spectral_indices.py
git commit -m "Add D4000, HdeltaA and H-minus bump index measurements"
```

---

### Task 5: SSP grid build and cache

**Files:**
- Create: `ssp_grid.py`
- Test: `tests/test_ssp_grid.py` (all `slow`)

**Interfaces:**
- Consumes: the fsps wheel.
- Produces: `SspGrid` dataclass, `build_ssp`, `build_and_cache_grid`, `save_ssp_grid`, `load_ssp_grid`. Cache layout: `output/ssp_grid/<product>.npz` with keys `wave_a`, `log_age_yr`, `log_z_grid`, `agb_weights`, `flux_nu` (shape `(4, 2, 107, n_pix)`), `native_sigma_km_s` (per pixel, only in `native.npz`), plus `output/ssp_grid/provenance.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ssp_grid.py`:

```python
import json

import numpy as np
import pytest

from ssp_grid import LOG_Z_GRID, ZMET_BY_LOG_Z, SspGrid, build_ssp, load_ssp_grid, save_ssp_grid

pytestmark = pytest.mark.slow


def test_zmet_lookup_matches_mist_zlegend():
    assert ZMET_BY_LOG_Z == {-0.5: 9, -0.25: 10, 0.0: 11, 0.25: 12}
    assert LOG_Z_GRID == (-0.5, -0.25, 0.0, 0.25)


def test_build_ssp_returns_window_and_provenance():
    wave_a, log_age_yr, flux_nu, provenance = build_ssp(log_z=0.0, agb=1.0)
    assert log_age_yr.shape == (107,)
    assert np.isclose(log_age_yr[0], 5.0) and np.isclose(log_age_yr[-1], 10.3)
    assert wave_a[0] >= 3400.0 and wave_a[-1] <= 22000.0
    assert flux_nu.shape == (107, wave_a.size)
    assert np.all(flux_nu >= 0)
    assert provenance["libraries"] == ["mist", "c3k_hr", "DL07"]
    assert provenance["params"]["imf_type"] == 1
    assert provenance["params"]["agb"] == 1.0
    assert provenance["params"]["zmet"] == 11


def test_agb_weight_is_exactly_linear():
    wave_a, _, flux_0, _ = build_ssp(log_z=0.0, agb=0.0)
    _, _, flux_1, _ = build_ssp(log_z=0.0, agb=1.0)
    _, _, flux_2, _ = build_ssp(log_z=0.0, agb=2.0)
    predicted = 2.0 * flux_1 - flux_0
    scale = np.max(flux_2)
    assert np.max(np.abs(flux_2 - predicted)) / scale < 1e-10


def test_save_and_load_round_trip(tmp_path):
    wave_a = np.linspace(3400.0, 22000.0, 100)
    grid = SspGrid(
        wave_a=wave_a,
        log_age_yr=np.linspace(5.0, 10.3, 107),
        log_z_grid=np.array(LOG_Z_GRID),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=np.ones((4, 2, 107, 100)),
        native_sigma_km_s=np.full(100, 42.4),
        product="native",
        provenance={"note": "test"},
    )
    save_ssp_grid(grid, tmp_path)
    loaded = load_ssp_grid(tmp_path, "native")
    assert loaded.flux_nu.shape == (4, 2, 107, 100)
    assert loaded.product == "native"
    assert json.loads((tmp_path / "provenance.json").read_text())["note"] == "test"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ssp_grid.py -m slow -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssp_grid'`.

- [ ] **Step 3: Implement `ssp_grid.py`**

```python
"""Build, cache and load the FSPS SSP grids used by the CSP integrator.

Fixed configuration: MIST isochrones, C3K high-resolution spectra, Chabrier
IMF, all other python-fsps parameters at their defaults. Only the TP-AGB
weight `agb` and the metallicity vary. Spectra are F_nu in Lsun/Hz per
solar mass formed, on the FSPS vacuum wavelength grid restricted to the
3400-22000 A window.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

WAVE_MIN_A = 3400.0
WAVE_MAX_A = 22000.0
LOG_Z_GRID = (-0.5, -0.25, 0.0, 0.25)
ZMET_BY_LOG_Z = {-0.5: 9, -0.25: 10, 0.0: 11, 0.25: 12}
AGB_WEIGHTS = (0.0, 1.0)
IMF_TYPE_CHABRIER = 1
PRODUCTS = ("native", "sigma300", "r100")
DEFAULT_GRID_DIR = Path(__file__).resolve().parent / "output" / "ssp_grid"


@dataclass
class SspGrid:
    wave_a: np.ndarray
    log_age_yr: np.ndarray
    log_z_grid: np.ndarray
    agb_weights: np.ndarray
    flux_nu: np.ndarray
    native_sigma_km_s: np.ndarray
    product: str
    provenance: dict = field(default_factory=dict)


def _git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_ssp(log_z, agb):
    import fsps

    population = fsps.StellarPopulation(
        zcontinuous=0, zmet=ZMET_BY_LOG_Z[log_z], imf_type=IMF_TYPE_CHABRIER, sfh=0
    )
    population.params["agb"] = agb
    wave_a, flux_nu = population.get_spectrum(tage=0.0, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    resolutions = np.asarray(population.resolutions)
    provenance = {
        "fsps_version": fsps.__version__,
        "libraries": [item.decode() for item in population.libraries],
        "sps_home": os.environ.get("SPS_HOME", "unset"),
        "sps_home_git_hash": _git_hash(os.environ.get("SPS_HOME", ".")),
        "params": {
            key: population.params[key]
            for key in ("imf_type", "zmet", "agb", "pagb", "add_agb_dust_model", "agb_dust",
                        "use_lw_tpagb", "add_neb_emission", "dust1", "dust2", "sfh")
        },
        "native_sigma_km_s_note": "from StellarPopulation.resolutions; negative means approximate",
    }
    return wave_a[window], np.asarray(population.ssp_ages), flux_nu[:, window], provenance | {
        "native_sigma_km_s": np.abs(resolutions[window]).tolist()
    }


def build_and_cache_grid(out_dir=DEFAULT_GRID_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flux = None
    provenance = {"per_ssp": {}}
    for i_z, log_z in enumerate(LOG_Z_GRID):
        for i_agb, agb in enumerate(AGB_WEIGHTS):
            wave_a, log_age_yr, flux_nu, ssp_provenance = build_ssp(log_z, agb)
            native_sigma = np.array(ssp_provenance.pop("native_sigma_km_s"))
            if flux is None:
                flux = np.empty((len(LOG_Z_GRID), len(AGB_WEIGHTS)) + flux_nu.shape)
            flux[i_z, i_agb] = flux_nu
            provenance["per_ssp"][f"log_z={log_z:+.2f},agb={agb:g}"] = ssp_provenance
    grid = SspGrid(
        wave_a=wave_a, log_age_yr=log_age_yr, log_z_grid=np.array(LOG_Z_GRID),
        agb_weights=np.array(AGB_WEIGHTS), flux_nu=flux, native_sigma_km_s=native_sigma,
        product="native", provenance=provenance,
    )
    return save_ssp_grid(grid, out_dir)


def save_ssp_grid(grid, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{grid.product}.npz"
    arrays = {key: value for key, value in asdict(grid).items() if key not in ("product", "provenance")}
    np.savez(path, product=grid.product, **arrays)
    if grid.product == "native" or not (out_dir / "provenance.json").exists():
        (out_dir / "provenance.json").write_text(json.dumps(grid.provenance, indent=2, default=str) + "\n")
    return path


def load_ssp_grid(out_dir, product):
    out_dir = Path(out_dir)
    with np.load(out_dir / f"{product}.npz") as data:
        arrays = {key: data[key] for key in data.files if key != "product"}
    provenance_path = out_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    return SspGrid(product=product, provenance=provenance, **arrays)


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    written = build_and_cache_grid()
    print(f"wrote {written} in {time.perf_counter() - start:.1f} s")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ssp_grid.py -m slow -v`
Expected: 4 passed in roughly one minute (three SSP builds).

- [ ] **Step 5: Build the full native grid and record the timing**

```bash
cd /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb
uv run python ssp_grid.py
ls -la output/ssp_grid/
uv run python -c "from ssp_grid import load_ssp_grid; g = load_ssp_grid('output/ssp_grid', 'native'); print(g.flux_nu.shape, g.wave_a.size, g.native_sigma_km_s.min(), g.native_sigma_km_s.max())"
```

Expected: 8 SSP builds in about 90 to 240 s; shape `(4, 2, 107, n_pix)` with n_pix near 6900; native sigma 42.44 and 254.63. Record the measured time in `docs/lessons.md`.

- [ ] **Step 6: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/ssp_grid.py stellar_pop/fsps_agb/tests/test_ssp_grid.py stellar_pop/fsps_agb/output/ssp_grid/provenance.json stellar_pop/fsps_agb/docs/lessons.md
git commit -m "Add FSPS SSP grid builder with cache and provenance"
```

---

### Task 6: Broadening and resolution products

**Files:**
- Create: `broadening.py`
- Test: `tests/test_broadening.py`

**Interfaces:**
- Consumes: `SspGrid` from Task 5.
- Produces: `log_wavelength_grid`, `resample_flux`, `gaussian_broaden`, `added_sigma_km_s`, `make_resolution_product`, constants `SIGMA_GALAXY_KM_S = 300.0`, `R100_FWHM = 100.0`, `R100_RANGE_A = (12500.0, 21000.0)`, `SPLICE_A = 10000.0`. A resolution product is an `SspGrid` on the log-wavelength grid with `product` set to `"sigma300"` or `"r100"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_broadening.py`:

```python
import numpy as np

from broadening import (
    SPEED_OF_LIGHT_KM_S,
    added_sigma_km_s,
    gaussian_broaden,
    log_wavelength_grid,
    make_resolution_product,
    resample_flux,
    target_sigma_km_s,
)
from ssp_grid import SspGrid


def test_log_wavelength_grid_has_constant_velocity_step():
    wave = log_wavelength_grid(3400.0, 22000.0, 30.0)
    steps = np.diff(np.log(wave)) * SPEED_OF_LIGHT_KM_S
    assert np.allclose(steps, 30.0, rtol=1e-9)
    assert wave[0] >= 3400.0 and wave[-1] <= 22000.0


def test_resample_flux_preserves_linear_spectrum():
    wave = np.linspace(3400.0, 22000.0, 3000)
    flux = np.stack([1.0 + 1e-4 * wave, 2.0 - 5e-5 * wave])
    target = log_wavelength_grid(3500.0, 21000.0)
    out = resample_flux(wave, flux, target)
    assert out.shape == (2, target.size)
    assert np.allclose(out[0], 1.0 + 1e-4 * target, rtol=1e-12)


def test_gaussian_broaden_recovers_quadrature_sum():
    wave = log_wavelength_grid(15000.0, 18000.0, 10.0)
    center = 16500.0
    sigma_line = 40.0
    velocity = SPEED_OF_LIGHT_KM_S * np.log(wave / center)
    flux_lambda = 1.0 - 0.5 * np.exp(-0.5 * (velocity / sigma_line) ** 2)
    flux_nu = flux_lambda * wave**2
    broadened = gaussian_broaden(wave, flux_nu[None, :], 300.0)[0] / wave**2
    depth = 1.0 - broadened
    second_moment = np.sum(depth * velocity**2) / np.sum(depth)
    recovered = np.sqrt(second_moment)
    expected = np.hypot(sigma_line, 300.0)
    assert abs(recovered / expected - 1.0) < 0.01


def test_gaussian_broaden_conserves_flux_per_log_wavelength():
    wave = log_wavelength_grid(15000.0, 18000.0, 10.0)
    flux_nu = np.exp(-0.5 * ((wave - 16500.0) / 100.0) ** 2)[None, :] / wave
    broadened = gaussian_broaden(wave, flux_nu, 500.0)
    assert np.isclose(np.sum(broadened[0] / wave), np.sum(flux_nu[0] / wave), rtol=1e-6)


def test_added_sigma_is_quadrature_difference():
    assert np.isclose(added_sigma_km_s(300.0, 42.4378), np.sqrt(300.0**2 - 42.4378**2))
    assert np.isclose(added_sigma_km_s(300.0, 254.6267), np.sqrt(300.0**2 - 254.6267**2))


def test_target_sigma_values():
    assert target_sigma_km_s("sigma300") == 300.0
    r100 = SPEED_OF_LIGHT_KM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)) * 100.0)
    assert np.isclose(target_sigma_km_s("r100"), np.hypot(r100, 300.0))


def _toy_grid():
    wave = np.linspace(3400.0, 22000.0, 4000)
    flux = np.ones((4, 2, 107, wave.size)) * (1.0 + 1e-5 * wave)
    return SspGrid(
        wave_a=wave, log_age_yr=np.linspace(5.0, 10.3, 107), log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
        agb_weights=np.array([0.0, 1.0]), flux_nu=flux,
        native_sigma_km_s=np.where(wave < 10000.0, 42.4378, 254.6267), product="native",
    )


def test_make_resolution_product_shapes_and_ranges():
    grid = _toy_grid()
    sigma300 = make_resolution_product(grid, "sigma300")
    assert sigma300.product == "sigma300"
    assert sigma300.flux_nu.shape[:3] == (4, 2, 107)
    assert sigma300.wave_a[0] >= 3400.0 and sigma300.wave_a[-1] <= 22000.0
    r100 = make_resolution_product(grid, "r100")
    assert r100.product == "r100"
    assert r100.wave_a[0] >= 12500.0 and r100.wave_a[-1] <= 21000.0
    assert np.allclose(sigma300.flux_nu[0, 0, 0] / sigma300.wave_a**2,
                       (1.0 + 1e-5 * sigma300.wave_a) / sigma300.wave_a**2 * 1.0, rtol=2e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_broadening.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'broadening'`.

- [ ] **Step 3: Implement `broadening.py`**

```python
"""Gaussian velocity broadening of SSP grids on a uniform log-wavelength grid.

The convolution acts on flux per unit log wavelength (lambda F_lambda, equal
to nu F_nu up to a constant), following the ProGeny resolution study. The
native library resolution is subtracted in quadrature before smoothing so
that each product has a known total resolution.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

SPEED_OF_LIGHT_KM_S = 299792.458
VELOCITY_STEP_KM_S = 30.0
SIGMA_GALAXY_KM_S = 300.0
R100_FWHM = 100.0
R100_RANGE_A = (12500.0, 21000.0)
SPLICE_A = 10000.0
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def log_wavelength_grid(wave_min_a, wave_max_a, velocity_step_km_s=VELOCITY_STEP_KM_S):
    step = velocity_step_km_s / SPEED_OF_LIGHT_KM_S
    log_wave = np.arange(np.log(wave_min_a), np.log(wave_max_a) + 0.5 * step, step)
    return np.exp(log_wave[np.exp(log_wave) <= wave_max_a])


def resample_flux(wave_a, flux, target_wave_a):
    flux = np.asarray(flux)
    flat = flux.reshape(-1, flux.shape[-1])
    out = np.stack([np.interp(target_wave_a, wave_a, row) for row in flat])
    return out.reshape(flux.shape[:-1] + (target_wave_a.size,))


def gaussian_broaden(log_wave_a, flux_nu, sigma_km_s):
    step_km_s = np.log(log_wave_a[1] / log_wave_a[0]) * SPEED_OF_LIGHT_KM_S
    per_log_wavelength = flux_nu / log_wave_a
    smoothed = gaussian_filter1d(
        per_log_wavelength, sigma_km_s / step_km_s, axis=-1, mode="nearest", truncate=6.0
    )
    return smoothed * log_wave_a


def added_sigma_km_s(target_sigma_km_s, native_sigma_km_s):
    if target_sigma_km_s <= native_sigma_km_s:
        raise ValueError(
            f"target sigma {target_sigma_km_s} km/s is not above native {native_sigma_km_s} km/s"
        )
    return float(np.sqrt(target_sigma_km_s**2 - native_sigma_km_s**2))


def target_sigma_km_s(product):
    if product == "sigma300":
        return SIGMA_GALAXY_KM_S
    if product == "r100":
        instrument = SPEED_OF_LIGHT_KM_S / (FWHM_PER_SIGMA * R100_FWHM)
        return float(np.hypot(instrument, SIGMA_GALAXY_KM_S))
    raise ValueError(f"unknown product {product!r}")


def _broaden_segment(grid, wave_lo, wave_hi, sigma_target, padding_km_s=6000.0):
    """Broaden one segment of constant native resolution with padding on both sides."""
    pad = np.exp(padding_km_s / SPEED_OF_LIGHT_KM_S)
    lo = max(grid.wave_a[0], wave_lo / pad)
    hi = min(grid.wave_a[-1], wave_hi * pad)
    log_wave = log_wavelength_grid(lo, hi)
    native = np.median(grid.native_sigma_km_s[(grid.wave_a >= wave_lo) & (grid.wave_a <= wave_hi)])
    flux = resample_flux(grid.wave_a, grid.flux_nu, log_wave)
    flux = gaussian_broaden(log_wave, flux, added_sigma_km_s(sigma_target, native))
    keep = (log_wave >= wave_lo) & (log_wave <= wave_hi)
    return log_wave[keep], flux[..., keep], float(native)


def make_resolution_product(grid, product):
    sigma_target = target_sigma_km_s(product)
    if product == "sigma300":
        segments = [(grid.wave_a[0], SPLICE_A), (SPLICE_A, grid.wave_a[-1])]
    else:
        segments = [R100_RANGE_A]
    waves, fluxes, natives = [], [], []
    for wave_lo, wave_hi in segments:
        wave, flux, native = _broaden_segment(grid, wave_lo, wave_hi, sigma_target)
        waves.append(wave)
        fluxes.append(flux)
        natives.append(np.full(wave.size, native))
    wave = np.concatenate(waves)
    order = np.argsort(wave)
    provenance = dict(grid.provenance) | {
        "product": product,
        "total_sigma_km_s": sigma_target,
        "velocity_step_km_s": VELOCITY_STEP_KM_S,
        "segments_a": [list(map(float, segment)) for segment in segments],
    }
    return replace(
        grid, wave_a=wave[order], flux_nu=np.concatenate(fluxes, axis=-1)[..., order],
        native_sigma_km_s=np.concatenate(natives)[order], product=product, provenance=provenance,
    )


if __name__ == "__main__":
    import time

    from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid, save_ssp_grid

    native_grid = load_ssp_grid(DEFAULT_GRID_DIR, "native")
    for name in ("sigma300", "r100"):
        start = time.perf_counter()
        path = save_ssp_grid(make_resolution_product(native_grid, name), DEFAULT_GRID_DIR)
        print(f"wrote {path} in {time.perf_counter() - start:.1f} s")
```

Note on the segment split: the two `sigma300` segments meet at 10000 A; each is padded by 6000 km/s (about 2 percent in wavelength) before convolution and trimmed afterwards, so the only inexact pixels are within a few kernel widths of 10000 A, far from every index band. The `wave_a[0]`/`wave_a[-1]` edges use `mode="nearest"`; the index bands are more than 300 A from them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_broadening.py -v`
Expected: 7 passed.

- [ ] **Step 5: Build the two products from the cached native grid and record timing**

```bash
cd /Users/shuang/Dropbox/work/project/luandun/stellar_pop/fsps_agb
uv run python broadening.py
ls -la output/ssp_grid/
```

Expected: `sigma300.npz` and `r100.npz` written in a few seconds each. Record the times in `docs/lessons.md`.

- [ ] **Step 6: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/broadening.py stellar_pop/fsps_agb/tests/test_broadening.py stellar_pop/fsps_agb/docs/lessons.md
git commit -m "Add quadrature-corrected Gaussian broadening and resolution products"
```

---

### Task 7: CSP integrator

**Files:**
- Create: `csp_integrate.py`
- Test: `tests/test_csp_integrate.py`

**Interfaces:**
- Consumes: `time_bin_edges`, `bin_masses` (Task 3); `SspGrid` (Task 5).
- Produces: `age_weights`, `epoch_weight_matrix`, `interpolate_log_z`, `csp_spectra`, and the convenience `csp_track(grid, log_z, agb_index, t_q_gyr, tau_q_gyr, epochs=None) -> (epoch_gyr, flux_nu (n_epochs, n_pix), mass_formed (n_epochs,))` where `agb_index` selects `grid.agb_weights`, and `agb_two_spectra(flux_agb0, flux_agb1) = 2 * flux_agb1 - flux_agb0`.

- [ ] **Step 1: Write the failing tests**

`tests/test_csp_integrate.py`:

```python
import numpy as np

from csp_integrate import (
    age_weights,
    agb_two_spectra,
    csp_spectra,
    csp_track,
    epoch_weight_matrix,
    interpolate_log_z,
)
from sfh_model import bin_masses, time_bin_edges
from ssp_grid import SspGrid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def test_age_weights_are_linear_in_log_age():
    lookback_gyr = np.array([10 ** (5.025 - 9), 1.0, 1e-6, 30.0])
    weights = age_weights(LOG_AGE, lookback_gyr)
    assert weights.shape == (4, LOG_AGE.size)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.isclose(weights[0, 0], 0.5) and np.isclose(weights[0, 1], 0.5)
    assert np.isclose(weights[1, np.argmin(np.abs(LOG_AGE - 9.0))], 1.0)
    assert np.isclose(weights[2, 0], 1.0)          # below 1e5 yr uses the youngest SSP
    assert np.isclose(weights[3, -1], 1.0)         # beyond the oldest SSP uses the oldest


def test_epoch_weight_matrix_uses_only_bins_before_each_epoch():
    edges = time_bin_edges()
    masses = bin_masses(edges, 3.0, 0.3)
    matrix = epoch_weight_matrix(edges, masses, LOG_AGE)
    assert matrix.shape == (260, LOG_AGE.size)
    assert np.allclose(matrix.sum(axis=1), np.cumsum(masses))
    first_epoch_lookback = np.log10(0.025e9)
    assert matrix[0].argmax() in (np.searchsorted(LOG_AGE, first_epoch_lookback) - 1,
                                  np.searchsorted(LOG_AGE, first_epoch_lookback))


def test_interpolate_log_z_is_linear_between_grid_points():
    grid = np.array([-0.5, -0.25, 0.0, 0.25])
    flux = np.arange(4.0)[:, None, None] * np.ones((4, 3, 5))
    assert np.allclose(interpolate_log_z(flux, grid, 0.0), 2.0)
    assert np.allclose(interpolate_log_z(flux, grid, 0.125), 2.5)
    assert np.allclose(interpolate_log_z(flux, grid, -0.4), 0.4)


def test_csp_spectra_is_normalized_per_mass_formed():
    weights = np.array([[1.0, 3.0], [2.0, 2.0]])
    ssp = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    out = csp_spectra(weights, ssp)
    assert np.allclose(out[0], (1 * 1 + 3 * 2) / 4.0)
    assert np.allclose(out[1], 1.5)


def _constant_grid():
    wave = np.linspace(3400.0, 22000.0, 50)
    flux = np.ones((4, 2, LOG_AGE.size, wave.size))
    flux[:, 1] = 3.0
    return SspGrid(wave_a=wave, log_age_yr=LOG_AGE, log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
                   agb_weights=np.array([0.0, 1.0]), flux_nu=flux,
                   native_sigma_km_s=np.full(wave.size, 42.4), product="native")


def test_csp_track_of_constant_ssps_is_constant():
    grid = _constant_grid()
    epochs, flux, mass = csp_track(grid, log_z=0.1, agb_index=1, t_q_gyr=3.0, tau_q_gyr=0.3)
    assert epochs.shape == (260,) and np.isclose(epochs[-1], 13.0)
    assert flux.shape == (260, 50)
    assert np.allclose(flux, 3.0)
    assert np.all(np.diff(mass) > 0)


def test_agb_two_spectra_is_linear_extrapolation():
    assert np.allclose(agb_two_spectra(np.array([1.0]), np.array([3.0])), 5.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_csp_integrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'csp_integrate'`.

- [ ] **Step 3: Implement `csp_integrate.py`**

```python
"""Composite stellar population spectra from cached SSP grids.

For an observation epoch t_obs, every SFH bin with center t_i < t_obs
contributes its formed mass at lookback age t_obs - t_i. The SSP at that age
is interpolated linearly in log age between the bracketing grid ages, the
same kernel FSPS uses (sfh_weight.f90, interpolation_type = 0). Metallicity
is interpolated linearly in log Z between grid SSPs, as zcontinuous = 1 does.
"""

import numpy as np

from sfh_model import bin_masses, time_bin_edges

MIN_LOG_AGE_YR = 5.0


def age_weights(log_age_grid_yr, lookback_gyr):
    log_lookback = np.log10(np.maximum(np.asarray(lookback_gyr, dtype=float) * 1e9, 1.0))
    log_lookback = np.clip(log_lookback, log_age_grid_yr[0], log_age_grid_yr[-1])
    upper = np.clip(np.searchsorted(log_age_grid_yr, log_lookback, side="right"), 1,
                    log_age_grid_yr.size - 1)
    lower = upper - 1
    fraction = (log_lookback - log_age_grid_yr[lower]) / (
        log_age_grid_yr[upper] - log_age_grid_yr[lower]
    )
    weights = np.zeros((log_lookback.size, log_age_grid_yr.size))
    rows = np.arange(log_lookback.size)
    weights[rows, lower] = 1.0 - fraction
    weights[rows, upper] += fraction
    return weights


def epoch_weight_matrix(edges_gyr, masses, log_age_grid_yr):
    centers = 0.5 * (edges_gyr[:-1] + edges_gyr[1:])
    n_epochs = edges_gyr.size - 1
    matrix = np.zeros((n_epochs, log_age_grid_yr.size))
    for k in range(n_epochs):
        t_obs = edges_gyr[k + 1]
        active = centers < t_obs
        lookback = t_obs - centers[active]
        matrix[k] = masses[active] @ age_weights(log_age_grid_yr, lookback)
    return matrix


def interpolate_log_z(flux_by_z, log_z_grid, log_z):
    if not log_z_grid[0] <= log_z <= log_z_grid[-1]:
        raise ValueError(f"log_z {log_z} outside grid [{log_z_grid[0]}, {log_z_grid[-1]}]")
    upper = int(np.clip(np.searchsorted(log_z_grid, log_z, side="right"), 1, log_z_grid.size - 1))
    lower = upper - 1
    fraction = (log_z - log_z_grid[lower]) / (log_z_grid[upper] - log_z_grid[lower])
    return (1.0 - fraction) * flux_by_z[lower] + fraction * flux_by_z[upper]


def csp_spectra(weight_matrix, ssp_flux):
    mass_formed = weight_matrix.sum(axis=1)
    return (weight_matrix @ ssp_flux) / mass_formed[:, None]


def agb_two_spectra(flux_agb0, flux_agb1):
    return 2.0 * flux_agb1 - flux_agb0


def csp_track(grid, log_z, agb_index, t_q_gyr, tau_q_gyr, edges_gyr=None):
    edges_gyr = time_bin_edges() if edges_gyr is None else edges_gyr
    masses = bin_masses(edges_gyr, t_q_gyr, tau_q_gyr)
    weights = epoch_weight_matrix(edges_gyr, masses, grid.log_age_yr)
    ssp_flux = interpolate_log_z(grid.flux_nu[:, agb_index], grid.log_z_grid, log_z)
    return edges_gyr[1:], csp_spectra(weights, ssp_flux), weights.sum(axis=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_csp_integrate.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/csp_integrate.py stellar_pop/fsps_agb/tests/test_csp_integrate.py
git commit -m "Add numpy CSP integrator with log-age and log-Z interpolation"
```

---

### Task 8: Cross-check the integrator against FSPS's tabular SFH

**Files:**
- Create: `cross_check_fsps_tabular.py`
- Test: `tests/test_cross_check_fsps_tabular.py` (`slow`)
- Output: `output/single_csp/fsps_cross_check.json`

**Interfaces:**
- Consumes: `csp_track` (Task 7), `load_ssp_grid` (Task 5), `star_formation_rate`, `time_bin_edges` (Task 3), `measure_all` (Task 4).
- Produces: `fsps_tabular_spectrum(t_q_gyr, tau_q_gyr, zmet, agb, tage_gyr, edges_gyr) -> (wave_a, flux_nu)` and `run_cross_check(epochs_gyr) -> dict` with per-epoch maximum relative flux differences inside the three index windows and the three index differences.

- [ ] **Step 1: Write the failing test**

`tests/test_cross_check_fsps_tabular.py`:

```python
import pytest

from cross_check_fsps_tabular import run_cross_check

pytestmark = pytest.mark.slow


def test_integrator_agrees_with_fsps_tabular_within_two_percent():
    report = run_cross_check(epochs_gyr=(1.0, 3.0, 3.5, 5.0, 13.0))
    for epoch, entry in report["epochs"].items():
        for window, value in entry["max_relative_flux_difference"].items():
            assert value < 0.02, (epoch, window, value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cross_check_fsps_tabular.py -m slow -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cross_check_fsps_tabular.py`**

```python
"""Compare the numpy CSP integrator with FSPS's own tabular-SFH CSP.

FSPS interpolates the SFR linearly between table nodes and integrates
against the same log-age kernel; the integrator uses exact bin masses. The
comparison is made on the native FSPS wavelength grid, inside the three
index windows, with the fiducial history at solar metallicity and agb = 1.
"""

import json
from pathlib import Path

import numpy as np

from csp_integrate import csp_track
from sfh_model import star_formation_rate, time_bin_edges
from spectral_indices import D4000_BANDS_A, H_MINUS_BANDS_A, HDELTA_A_BANDS_A, measure_all
from ssp_grid import DEFAULT_GRID_DIR, IMF_TYPE_CHABRIER, WAVE_MAX_A, WAVE_MIN_A, load_ssp_grid

FIDUCIAL = {"t_q_gyr": 3.0, "tau_q_gyr": 0.3, "log_z": 0.0}
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "single_csp" / "fsps_cross_check.json"
WINDOWS_A = {
    "d4000": (D4000_BANDS_A["blue"][0], D4000_BANDS_A["red"][1]),
    "hdelta_a": (HDELTA_A_BANDS_A["blue"][0], HDELTA_A_BANDS_A["red"][1]),
    "h_minus_bump": (H_MINUS_BANDS_A["blue"][0], H_MINUS_BANDS_A["red"][1]),
}


def fsps_tabular_spectrum(t_q_gyr, tau_q_gyr, zmet, agb, tage_gyr, edges_gyr):
    import fsps

    population = fsps.StellarPopulation(zcontinuous=0, zmet=zmet, imf_type=IMF_TYPE_CHABRIER, sfh=3)
    population.params["agb"] = agb
    nodes = edges_gyr[1:]
    population.set_tabular_sfh(nodes, star_formation_rate(nodes, t_q_gyr, tau_q_gyr))
    wave_a, flux_nu = population.get_spectrum(tage=tage_gyr, peraa=False)
    window = (wave_a >= WAVE_MIN_A) & (wave_a <= WAVE_MAX_A)
    return wave_a[window], flux_nu[window] / population.formed_mass


def run_cross_check(epochs_gyr=(1.0, 3.0, 3.5, 5.0, 8.0, 13.0), grid_dir=DEFAULT_GRID_DIR):
    grid = load_ssp_grid(grid_dir, "native")
    edges = time_bin_edges()
    epoch_grid, own_flux, _ = csp_track(grid, FIDUCIAL["log_z"], 1, FIDUCIAL["t_q_gyr"],
                                        FIDUCIAL["tau_q_gyr"], edges)
    report = {"fiducial": FIDUCIAL, "epochs": {}}
    for epoch in epochs_gyr:
        k = int(np.argmin(np.abs(epoch_grid - epoch)))
        wave_a, fsps_flux = fsps_tabular_spectrum(FIDUCIAL["t_q_gyr"], FIDUCIAL["tau_q_gyr"], 11,
                                                  1.0, float(epoch_grid[k]), edges)
        assert np.allclose(wave_a, grid.wave_a)
        mine = own_flux[k]
        differences = {}
        for name, (lo, hi) in WINDOWS_A.items():
            inside = (wave_a >= lo) & (wave_a <= hi)
            differences[name] = float(np.max(np.abs(mine[inside] / fsps_flux[inside] - 1.0)))
        own_indices = measure_all(wave_a, mine)
        fsps_indices = measure_all(wave_a, fsps_flux)
        report["epochs"][f"{epoch_grid[k]:.2f}"] = {
            "max_relative_flux_difference": differences,
            "index_difference": {key: float(own_indices[key] - fsps_indices[key]) for key in own_indices},
            "own_indices": {key: float(value) for key, value in own_indices.items()},
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run_cross_check(), indent=2))
```

- [ ] **Step 4: Run the test and the script**

Run: `uv run pytest tests/test_cross_check_fsps_tabular.py -m slow -v` then `uv run python cross_check_fsps_tabular.py`
Expected: PASS; the JSON lists differences. If any window exceeds 2 percent, first check the epoch right after quenching (3.05 to 3.5 Gyr), where the linear-interpolation of the exponential in FSPS differs most; if only those epochs fail, report the measured values in the spec's validation section rather than loosening the threshold, and flag it in the final summary.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/cross_check_fsps_tabular.py stellar_pop/fsps_agb/tests/test_cross_check_fsps_tabular.py stellar_pop/fsps_agb/output/single_csp/fsps_cross_check.json
git commit -m "Cross-check the CSP integrator against FSPS tabular SFH"
```

---

### Task 9: Shared index-plane figures

**Files:**
- Create: `index_planes.py`
- Test: `tests/test_index_planes.py`

**Interfaces:**
- Consumes: nothing from other modules (pure matplotlib).
- Produces: `PLANES = (("d4000", "hdelta_a"), ("d4000", "h_minus_bump"), ("hdelta_a", "h_minus_bump"))`, `AXIS_LABELS`, `plot_track(axes, indices, color_values, color_label, cmap="viridis")`, `plot_population(axes, indices, color_values, color_label, track_indices=None)`, `new_plane_figure(n_rows=1) -> (figure, axes of shape (n_rows, 3))`. `indices` is a dict with keys `d4000`, `hdelta_a`, `h_minus_bump` of equal-length arrays. The H-minus axis is inverted so a stronger bump is up.

- [ ] **Step 1: Write the failing test**

`tests/test_index_planes.py`:

```python
import matplotlib

matplotlib.use("Agg")
import numpy as np

from index_planes import PLANES, new_plane_figure, plot_population, plot_track


def test_planes_and_track_figure(tmp_path):
    indices = {"d4000": np.linspace(1.2, 1.9, 20), "hdelta_a": np.linspace(8, -1, 20),
               "h_minus_bump": np.linspace(-0.05, -0.1, 20)}
    figure, axes = new_plane_figure(n_rows=2)
    assert axes.shape == (2, 3)
    plot_track(axes[0], indices, np.arange(20), "time [Gyr]")
    plot_population(axes[1], indices, np.arange(20), "t_obs - t_q [Gyr]", track_indices=indices)
    assert axes[0, 1].yaxis_inverted()
    figure.savefig(tmp_path / "planes.png")
    assert (tmp_path / "planes.png").stat().st_size > 0
    assert len(PLANES) == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_index_planes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `index_planes.py`**

```python
"""Figure helpers for the three 2-D spectral-index planes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

PLANES = (("d4000", "hdelta_a"), ("d4000", "h_minus_bump"), ("hdelta_a", "h_minus_bump"))
AXIS_LABELS = {
    "d4000": "D4000",
    "hdelta_a": r"H$\delta_A$ [$\AA$]",
    "h_minus_bump": r"H$^-$ bump [mag]",
}
INVERTED_AXES = ("h_minus_bump",)


def new_plane_figure(n_rows=1):
    figure, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.2 * n_rows), squeeze=False)
    for row in axes:
        for axis, (x_key, y_key) in zip(row, PLANES, strict=True):
            axis.set_xlabel(AXIS_LABELS[x_key])
            axis.set_ylabel(AXIS_LABELS[y_key])
            if y_key in INVERTED_AXES and not axis.yaxis_inverted():
                axis.invert_yaxis()
    figure.tight_layout()
    return figure, axes


def plot_track(axes, indices, color_values, color_label, cmap="viridis"):
    norm = plt.Normalize(np.min(color_values), np.max(color_values))
    mappable = None
    for axis, (x_key, y_key) in zip(axes, PLANES, strict=True):
        points = np.column_stack([indices[x_key], indices[y_key]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.8)
        collection.set_array(np.asarray(color_values)[:-1])
        axis.add_collection(collection)
        axis.autoscale_view()
        mappable = collection
    axes[-1].figure.colorbar(mappable, ax=list(axes), label=color_label, pad=0.02)


def plot_population(axes, indices, color_values, color_label, track_indices=None, cmap="plasma"):
    mappable = None
    for axis, (x_key, y_key) in zip(axes, PLANES, strict=True):
        mappable = axis.scatter(indices[x_key], indices[y_key], c=color_values, s=2, alpha=0.35,
                                cmap=cmap, linewidths=0, rasterized=True)
        if track_indices is not None:
            axis.plot(track_indices[x_key], track_indices[y_key], color="black", lw=1.2,
                      label="fiducial track")
    axes[-1].figure.colorbar(mappable, ax=list(axes), label=color_label, pad=0.02)
    if track_indices is not None:
        axes[0].legend(frameon=False, loc="best")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_index_planes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/index_planes.py stellar_pop/fsps_agb/tests/test_index_planes.py
git commit -m "Add shared index-plane figure helpers"
```

---

### Task 10: Step 1 driver — fiducial CSP track

**Files:**
- Create: `run_single_csp.py`
- Test: `tests/test_run_single_csp.py`
- Output: `output/single_csp/` (`spectra_<product>.npz`, `indices.npz`, `indices.csv`, figures)

**Interfaces:**
- Consumes: `csp_track`, `agb_two_spectra` (Task 7); `load_ssp_grid` (Task 5); `measure_all` (Task 4); `star_formation_rate`, `time_bin_edges` (Task 3); `new_plane_figure`, `plot_track` (Task 9).
- Produces: `compute_track_indices(grid_dir, t_q_gyr, tau_q_gyr, log_z, agb_settings=(0.0, 2.0)) -> dict` with keys `epoch_gyr`, `sfr`, and for each `agb` setting and product the index arrays; plus the files above.

- [ ] **Step 1: Write the failing test (pilot mode on a toy grid directory)**

`tests/test_run_single_csp.py`:

```python
import numpy as np

from run_single_csp import compute_track_indices
from ssp_grid import SspGrid, save_ssp_grid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def _write_toy_grids(directory):
    wave = np.linspace(3400.0, 22000.0, 6000)
    base = np.ones((4, 2, LOG_AGE.size, wave.size)) * (1.0 + 1e-5 * wave) * wave**2
    for product, lo, hi in (("sigma300", 3400.0, 22000.0), ("r100", 12500.0, 21000.0)):
        keep = (wave >= lo) & (wave <= hi)
        grid = SspGrid(wave_a=wave[keep], log_age_yr=LOG_AGE, log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
                       agb_weights=np.array([0.0, 1.0]), flux_nu=base[..., keep],
                       native_sigma_km_s=np.full(keep.sum(), 42.4), product=product)
        save_ssp_grid(grid, directory)


def test_compute_track_indices_shapes(tmp_path):
    _write_toy_grids(tmp_path)
    result = compute_track_indices(tmp_path, t_q_gyr=3.0, tau_q_gyr=0.3, log_z=0.0)
    assert result["epoch_gyr"].shape == (260,)
    assert result["sfr"].shape == (260,)
    for agb in ("agb0", "agb2"):
        assert result[agb]["sigma300"]["d4000"].shape == (260,)
        assert result[agb]["sigma300"]["hdelta_a"].shape == (260,)
        assert result[agb]["sigma300"]["h_minus_bump"].shape == (260,)
        assert result[agb]["r100"]["h_minus_bump"].shape == (260,)
        assert np.allclose(result[agb]["sigma300"]["hdelta_a"], 0.0, atol=1e-9)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_run_single_csp.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `run_single_csp.py`**

```python
"""Step 1: trace the fiducial CSP in time and measure the three indices.

Products: D4000 and HdeltaA on the sigma300 grid; H-minus bump on sigma300
and r100. Both agb = 0 and agb = 2 (from linearity) are traced.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csp_integrate import agb_two_spectra, csp_track
from index_planes import new_plane_figure, plot_track
from sfh_model import star_formation_rate, time_bin_edges
from spectral_indices import H_MINUS_BANDS_A, d4000, flux_nu_to_flux_lambda, h_minus_bump, hdelta_a
from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid

FIDUCIAL = {"t_q_gyr": 3.0, "tau_q_gyr": 0.3, "log_z": 0.0}
AGB_SETTINGS = {"agb0": 0.0, "agb2": 2.0}
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "single_csp"
ZOOM_EPOCHS_GYR = (1.0, 3.0, 3.5, 4.0, 5.0, 8.0, 13.0)


def _agb_spectra(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr):
    epochs, flux_agb0, mass = csp_track(grid, log_z, 0, t_q_gyr, tau_q_gyr, edges_gyr)
    _, flux_agb1, _ = csp_track(grid, log_z, 1, t_q_gyr, tau_q_gyr, edges_gyr)
    return epochs, {"agb0": flux_agb0, "agb2": agb_two_spectra(flux_agb0, flux_agb1)}, mass


def compute_track_indices(grid_dir, t_q_gyr, tau_q_gyr, log_z, edges_gyr=None, keep_spectra=False):
    edges_gyr = time_bin_edges() if edges_gyr is None else edges_gyr
    grids = {product: load_ssp_grid(grid_dir, product) for product in ("sigma300", "r100")}
    result = {"epoch_gyr": edges_gyr[1:], "sfr": star_formation_rate(edges_gyr[1:], t_q_gyr, tau_q_gyr),
              "parameters": {"t_q_gyr": t_q_gyr, "tau_q_gyr": tau_q_gyr, "log_z": log_z}}
    spectra = {}
    for product, grid in grids.items():
        epochs, flux_by_agb, mass = _agb_spectra(grid, log_z, t_q_gyr, tau_q_gyr, edges_gyr)
        result["mass_formed"] = mass
        for agb_key, flux in flux_by_agb.items():
            entry = result.setdefault(agb_key, {}).setdefault(product, {})
            if product == "sigma300":
                entry["d4000"] = d4000(grid.wave_a, flux)
                entry["hdelta_a"] = hdelta_a(grid.wave_a, flux)
            entry["h_minus_bump"] = h_minus_bump(grid.wave_a, flux)
            spectra[(agb_key, product)] = (grid.wave_a, flux)
    if keep_spectra:
        result["spectra"] = spectra
    return result


def _write_tables(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = {"epoch_gyr": result["epoch_gyr"], "sfr": result["sfr"], "mass_formed": result["mass_formed"]}
    for agb_key in AGB_SETTINGS:
        for product, indices in result[agb_key].items():
            for name, values in indices.items():
                columns[f"{name}_{product}_{agb_key}"] = values
    np.savez(out_dir / "indices.npz", **columns)
    with (out_dir / "indices.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))
    for (agb_key, product), (wave_a, flux) in result["spectra"].items():
        np.savez(out_dir / f"spectra_{product}_{agb_key}.npz", wave_a=wave_a, flux_nu=flux,
                 epoch_gyr=result["epoch_gyr"])
    (out_dir / "parameters.json").write_text(json.dumps(result["parameters"], indent=2) + "\n")


def _figure_time_evolution(result, out_dir):
    figure, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
    epochs = result["epoch_gyr"]
    axes[0].plot(epochs, result["sfr"], color="black")
    axes[0].set_ylabel("SFR [arbitrary]")
    axes[0].axvline(result["parameters"]["t_q_gyr"], color="0.6", ls=":")
    for axis, name in zip(axes[1:], ("d4000", "hdelta_a", "h_minus_bump"), strict=True):
        for agb_key, color in (("agb0", "#1f77b4"), ("agb2", "#d62728")):
            axis.plot(epochs, result[agb_key]["sigma300"][name], color=color, label=f"{agb_key}, sigma300")
            if name == "h_minus_bump":
                axis.plot(epochs, result[agb_key]["r100"][name], color=color, ls="--", label=f"{agb_key}, R=100")
        axis.set_ylabel(name)
        axis.axvline(result["parameters"]["t_q_gyr"], color="0.6", ls=":")
    axes[3].invert_yaxis()
    axes[3].set_xlabel("time since start of star formation [Gyr]")
    axes[1].legend(frameon=False, fontsize=8)
    axes[3].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "time_evolution.png", dpi=150)
    plt.close(figure)


def _figure_nir_zoom(result, out_dir):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ZOOM_EPOCHS_GYR)))
    for column, agb_key in enumerate(AGB_SETTINGS):
        for row, product in enumerate(("sigma300", "r100")):
            wave_a, flux = result["spectra"][(agb_key, product)]
            axis = axes[row, column]
            zoom = (wave_a >= 14000.0) & (wave_a <= 18500.0)
            for epoch, color in zip(ZOOM_EPOCHS_GYR, colors, strict=True):
                k = int(np.argmin(np.abs(result["epoch_gyr"] - epoch)))
                flux_lambda = flux_nu_to_flux_lambda(wave_a, flux[k])
                axis.plot(wave_a[zoom] / 1e4, flux_lambda[zoom] / np.median(flux_lambda[zoom]),
                          color=color, lw=1.0, label=f"{result['epoch_gyr'][k]:.2f} Gyr")
            for name, color in (("blue", "tab:blue"), ("feature", "tab:green"), ("red", "tab:red")):
                axis.axvspan(H_MINUS_BANDS_A[name][0] / 1e4, H_MINUS_BANDS_A[name][1] / 1e4, color=color, alpha=0.08)
            axis.set_title(f"{agb_key}, {product}")
            axis.set_ylabel(r"normalized $F_\lambda$")
    axes[1, 0].set_xlabel(r"rest wavelength [$\mu$m]")
    axes[1, 1].set_xlabel(r"rest wavelength [$\mu$m]")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(out_dir / "nir_zoom.png", dpi=150)
    plt.close(figure)


def _figure_index_planes(result, out_dir):
    figure, axes = new_plane_figure(n_rows=2)
    for row, agb_key in enumerate(AGB_SETTINGS):
        indices = dict(result[agb_key]["sigma300"])
        plot_track(axes[row], indices, result["epoch_gyr"], "time [Gyr]")
        axes[row, 0].set_title(f"{agb_key} (sigma300, bump at sigma300)")
    figure.savefig(out_dir / "index_planes.png", dpi=150)
    plt.close(figure)
    figure, axes = new_plane_figure(n_rows=2)
    for row, agb_key in enumerate(AGB_SETTINGS):
        indices = dict(result[agb_key]["sigma300"]) | {"h_minus_bump": result[agb_key]["r100"]["h_minus_bump"]}
        plot_track(axes[row], indices, result["epoch_gyr"], "time [Gyr]")
        axes[row, 0].set_title(f"{agb_key} (bump at R=100)")
    figure.savefig(out_dir / "index_planes_r100.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Step 1: fiducial CSP index track.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--pilot", action="store_true", help="only 10 epochs, no figures")
    args = parser.parse_args()
    start = time.perf_counter()
    edges = time_bin_edges()
    if args.pilot:
        edges = edges[:11]
    result = compute_track_indices(Path(args.grid_dir), keep_spectra=True, edges_gyr=edges, **FIDUCIAL)
    print(f"indices for {result['epoch_gyr'].size} epochs in {time.perf_counter() - start:.2f} s")
    if args.pilot:
        return
    out_dir = Path(args.out_dir)
    _write_tables(result, out_dir)
    _figure_time_evolution(result, out_dir)
    _figure_nir_zoom(result, out_dir)
    _figure_index_planes(result, out_dir)
    print(f"wrote {out_dir} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the pilot, then the full Step 1**

```bash
uv run pytest tests/test_run_single_csp.py -v
uv run python run_single_csp.py --pilot
uv run python run_single_csp.py
ls output/single_csp/
```

Expected: test passes; pilot prints a sub-second timing; full run writes `indices.npz`, `indices.csv`, four `spectra_*.npz`, `time_evolution.png`, `nir_zoom.png`, `index_planes.png`, `index_planes_r100.png`. Open the three PNGs and check: SFR peaks at 3 Gyr; HdeltaA rises after quenching then declines; D4000 rises monotonically after quenching; the `agb2` bump is more negative than `agb0` around 0.5 to 2 Gyr after quenching. Record the run time and the largest `agb0` vs `agb2` bump difference in `docs/lessons.md`.

- [ ] **Step 5: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/run_single_csp.py stellar_pop/fsps_agb/tests/test_run_single_csp.py stellar_pop/fsps_agb/output/single_csp/*.png stellar_pop/fsps_agb/output/single_csp/indices.csv stellar_pop/fsps_agb/output/single_csp/parameters.json stellar_pop/fsps_agb/docs/lessons.md
git commit -m "Add Step 1 fiducial CSP index track with figures"
```

Spectra `.npz` files under `output/single_csp/` are committed only if each is below 5 MB; otherwise add `output/single_csp/spectra_*.npz` to `.gitignore` in this commit.

---

### Task 11: Step 2 driver — population

**Files:**
- Create: `run_population.py`
- Test: `tests/test_run_population.py`
- Output: `output/population/` (`draws.npz`, `indices.npz`, `indices.csv`, `summary.json`, figures)

**Interfaces:**
- Consumes: `draw_population`, `bin_masses`, `time_bin_edges`, `star_formation_rate` (Task 3); `epoch_weight_matrix`, `interpolate_log_z`, `csp_spectra`, `agb_two_spectra` (Task 7); `load_ssp_grid` (Task 5); index functions (Task 4); `new_plane_figure`, `plot_population` (Task 9); `compute_track_indices` (Task 10) for the fiducial overlay.
- Produces: `population_indices(grids, draws, edges_gyr) -> dict of flat arrays` with one entry per (history, epoch): `history_id`, `t_q_gyr`, `tau_q_gyr`, `log_z`, `epoch_gyr`, `time_since_quenching_gyr`, `sfr`, `ssfr_0_100_myr`, `ssfr_100_1000_myr`, and `d4000_agb0`, `hdelta_a_agb0`, `h_minus_bump_sigma300_agb0`, `h_minus_bump_r100_agb0` and the same four with `agb2`.

- [ ] **Step 1: Write the failing test**

`tests/test_run_population.py`:

```python
import numpy as np

from run_population import population_indices, specific_sfr_windows
from sfh_model import bin_masses, draw_population, time_bin_edges
from ssp_grid import SspGrid

LOG_AGE = np.round(np.arange(5.0, 10.3001, 0.05), 3)


def _toy_grids():
    wave = np.linspace(3400.0, 22000.0, 6000)
    base = np.ones((4, 2, LOG_AGE.size, wave.size)) * (1.0 + 1e-5 * wave) * wave**2
    grids = {}
    for product, lo, hi in (("sigma300", 3400.0, 22000.0), ("r100", 12500.0, 21000.0)):
        keep = (wave >= lo) & (wave <= hi)
        grids[product] = SspGrid(wave_a=wave[keep], log_age_yr=LOG_AGE,
                                 log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
                                 agb_weights=np.array([0.0, 1.0]), flux_nu=base[..., keep],
                                 native_sigma_km_s=np.full(keep.sum(), 42.4), product=product)
    return grids


def test_specific_sfr_windows_for_constant_sfr():
    edges = time_bin_edges()
    masses = np.full(260, 0.05)  # SFR = 1 per Gyr
    recent, previous = specific_sfr_windows(edges, masses, epoch_index=199)  # t_obs = 10 Gyr
    assert np.isclose(recent, 1.0 / 10.0, rtol=1e-6)
    assert np.isclose(previous, 1.0 / 10.0, rtol=1e-6)


def test_population_indices_layout():
    draws = draw_population(3, seed=2)
    table = population_indices(_toy_grids(), draws, time_bin_edges()[:21])
    assert table["history_id"].shape == (3 * 20,)
    assert set(table) >= {"t_q_gyr", "tau_q_gyr", "log_z", "epoch_gyr", "time_since_quenching_gyr",
                          "sfr", "ssfr_0_100_myr", "ssfr_100_1000_myr", "d4000_agb0", "hdelta_a_agb2",
                          "h_minus_bump_sigma300_agb0", "h_minus_bump_r100_agb2"}
    assert np.allclose(table["hdelta_a_agb0"], 0.0, atol=1e-9)
    assert np.allclose(table["time_since_quenching_gyr"], table["epoch_gyr"] - table["t_q_gyr"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_run_population.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `run_population.py`**

```python
"""Step 2: index distributions for a population of quenching histories.

Each drawn history is traced over every epoch. Only index tables are stored.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csp_integrate import agb_two_spectra, csp_spectra, epoch_weight_matrix, interpolate_log_z
from index_planes import new_plane_figure, plot_population
from run_single_csp import FIDUCIAL, compute_track_indices
from sfh_model import bin_masses, draw_population, star_formation_rate, time_bin_edges
from spectral_indices import d4000, h_minus_bump, hdelta_a
from ssp_grid import DEFAULT_GRID_DIR, load_ssp_grid

N_DRAWS = 2000
SEED = 20260924
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "population"


def specific_sfr_windows(edges_gyr, masses, epoch_index):
    """sSFR over the last 100 Myr and over 100 to 1000 Myr before the epoch, per Gyr."""
    t_obs = edges_gyr[epoch_index + 1]
    centers = 0.5 * (edges_gyr[:-1] + edges_gyr[1:])
    formed = masses[centers < t_obs].sum()
    lookback = t_obs - centers
    recent = masses[(lookback > 0) & (lookback <= 0.1)].sum() / 0.1
    previous = masses[(lookback > 0.1) & (lookback <= 1.0)].sum() / 0.9
    return recent / formed, previous / formed


def _history_indices(grids, t_q_gyr, tau_q_gyr, log_z, edges_gyr):
    masses = bin_masses(edges_gyr, t_q_gyr, tau_q_gyr)
    out = {}
    for product, grid in grids.items():
        weights = epoch_weight_matrix(edges_gyr, masses, grid.log_age_yr)
        flux_agb0 = csp_spectra(weights, interpolate_log_z(grid.flux_nu[:, 0], grid.log_z_grid, log_z))
        flux_agb1 = csp_spectra(weights, interpolate_log_z(grid.flux_nu[:, 1], grid.log_z_grid, log_z))
        for agb_key, flux in (("agb0", flux_agb0), ("agb2", agb_two_spectra(flux_agb0, flux_agb1))):
            if product == "sigma300":
                out[f"d4000_{agb_key}"] = d4000(grid.wave_a, flux)
                out[f"hdelta_a_{agb_key}"] = hdelta_a(grid.wave_a, flux)
            out[f"h_minus_bump_{product}_{agb_key}"] = h_minus_bump(grid.wave_a, flux)
    n_epochs = edges_gyr.size - 1
    ssfr = np.array([specific_sfr_windows(edges_gyr, masses, k) for k in range(n_epochs)])
    out["ssfr_0_100_myr"] = ssfr[:, 0]
    out["ssfr_100_1000_myr"] = ssfr[:, 1]
    out["sfr"] = star_formation_rate(edges_gyr[1:], t_q_gyr, tau_q_gyr)
    return out


def population_indices(grids, draws, edges_gyr):
    n_epochs = edges_gyr.size - 1
    n_draws = draws["t_q_gyr"].size
    columns = {}
    for i in range(n_draws):
        entry = _history_indices(grids, draws["t_q_gyr"][i], draws["tau_q_gyr"][i], draws["log_z"][i], edges_gyr)
        entry["history_id"] = np.full(n_epochs, i)
        entry["t_q_gyr"] = np.full(n_epochs, draws["t_q_gyr"][i])
        entry["tau_q_gyr"] = np.full(n_epochs, draws["tau_q_gyr"][i])
        entry["log_z"] = np.full(n_epochs, draws["log_z"][i])
        entry["epoch_gyr"] = edges_gyr[1:]
        entry["time_since_quenching_gyr"] = edges_gyr[1:] - draws["t_q_gyr"][i]
        for key, values in entry.items():
            columns.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in columns.items()}


def _write_tables(table, draws, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "draws.npz", **draws)
    np.savez(out_dir / "indices.npz", **table)
    with (out_dir / "indices.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table)
        writer.writerows(zip(*table.values(), strict=True))


def _figure_planes(table, track, out_dir, bump_product):
    figure, axes = new_plane_figure(n_rows=2)
    color = table["time_since_quenching_gyr"]
    for row, agb_key in enumerate(("agb0", "agb2")):
        indices = {"d4000": table[f"d4000_{agb_key}"], "hdelta_a": table[f"hdelta_a_{agb_key}"],
                   "h_minus_bump": table[f"h_minus_bump_{bump_product}_{agb_key}"]}
        track_indices = dict(track[agb_key]["sigma300"]) | {
            "h_minus_bump": track[agb_key][bump_product]["h_minus_bump"]}
        plot_population(axes[row], indices, color, r"$t_{\rm obs} - t_q$ [Gyr]", track_indices)
        axes[row, 0].set_title(f"{agb_key}, bump at {bump_product}")
    figure.savefig(out_dir / f"index_planes_{bump_product}.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Step 2: population of quenching histories.")
    parser.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pilot", action="store_true", help="10 histories, 10 epochs, timing only")
    args = parser.parse_args()
    start = time.perf_counter()
    grids = {product: load_ssp_grid(args.grid_dir, product) for product in ("sigma300", "r100")}
    edges = time_bin_edges()
    n_draws = args.n_draws
    if args.pilot:
        edges, n_draws = edges[:11], 10
    draws = draw_population(n_draws, args.seed)
    table = population_indices(grids, draws, edges)
    elapsed = time.perf_counter() - start
    print(f"{n_draws} histories x {edges.size - 1} epochs in {elapsed:.1f} s")
    if args.pilot:
        per_history_epoch = elapsed / (n_draws * (edges.size - 1))
        print(f"extrapolated full run: {per_history_epoch * N_DRAWS * 260 / 60:.1f} min")
        return
    out_dir = Path(args.out_dir)
    _write_tables(table, draws, out_dir)
    track = compute_track_indices(args.grid_dir, **FIDUCIAL)
    for bump_product in ("sigma300", "r100"):
        _figure_planes(table, track, out_dir, bump_product)
    summary = {"n_draws": n_draws, "seed": args.seed, "n_epochs": int(edges.size - 1),
               "elapsed_s": elapsed, "grid_provenance": grids["sigma300"].provenance}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {out_dir} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the pilot**

```bash
uv run pytest tests/test_run_population.py -v
uv run python run_population.py --pilot
```

Expected: test passes; the pilot prints the measured time and the extrapolated full-run time. Proceed to the full run only if the extrapolation is under 60 minutes. If it is longer, profile with `uv run python -X importtime` ruled out and instead vectorize `_history_indices` by measuring indices only on the pixels inside the index windows (slice `grid.wave_a` and `grid.flux_nu` to the union of the band ranges with 100 A margins before `csp_spectra`), re-run the pilot, and record both timings in `docs/lessons.md`.

- [ ] **Step 5: Run the full population and inspect the figures**

```bash
uv run python run_population.py
ls -la output/population/
```

Open `index_planes_sigma300.png` and `index_planes_r100.png`. Check: the D4000 versus HdeltaA plane is a narrow sequence; the H-minus planes fan out; the `agb2` row shows more negative bump values than `agb0` at 0.5 to 2 Gyr after quenching; the fiducial track lies inside the cloud. Record the run time, the table size and the bump range per AGB setting in `docs/lessons.md`.

- [ ] **Step 6: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/run_population.py stellar_pop/fsps_agb/tests/test_run_population.py stellar_pop/fsps_agb/output/population/*.png stellar_pop/fsps_agb/output/population/summary.json stellar_pop/fsps_agb/output/population/draws.npz stellar_pop/fsps_agb/docs/lessons.md
git commit -m "Add Step 2 population of quenching histories with index planes"
```

`indices.csv` and `indices.npz` (about 520,000 rows) are git-ignored: add `output/population/indices.*` to `.gitignore` in this commit.

---

### Task 12: Documentation, review and lessons

**Files:**
- Modify: `README.md`, `docs/todo.md`, `docs/lessons.md`, `docs/SPEC.md` (only if the implementation deviated; record the deviation, do not silently rewrite requirements)

- [ ] **Step 1: Update the README**

Add a "Phase 2" section listing each new module (one line each, matching the file structure table of this plan), how to run `ssp_grid.py`, `broadening.py`, `cross_check_fsps_tabular.py`, `run_single_csp.py --pilot`, `run_single_csp.py`, `run_population.py --pilot`, `run_population.py`, and a results table copied from the measured numbers in `docs/lessons.md`: the FSPS cross-check maximum flux difference, the timing of each step, and the range of each index for `agb0` and `agb2`.

- [ ] **Step 2: Update `docs/todo.md`**

Tick every Phase 2 plan item and add a "Review" subsection with three to five bullets: what the population figures show about the TP-AGB sensitivity of the bump relative to D4000 and HdeltaA, the validation numbers, and any threshold that was not met.

- [ ] **Step 3: Run the full test suite and the pre-commit hook**

```bash
uv run pytest -q
uv run pytest -m slow -q
uv run pre-commit run --all-files
```

Expected: all fast and slow tests pass; ruff clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/shuang/Dropbox/work/project/luandun
git add stellar_pop/fsps_agb/README.md stellar_pop/fsps_agb/docs
git commit -m "Document Phase 2 results, review and lessons"
```

---

## Self-review against the spec

- Fixed configuration, loader fix and rebuild: Task 1, Task 5.
- SFH form, bins, analytic masses, fiducial, priors, seed: Task 3, Task 10, Task 11.
- Own integrator with FSPS-matching kernels, `agb` = 2 by linearity: Task 7; linearity test: Task 5.
- FSPS tabular cross-check with the 2 percent threshold: Task 8.
- Resolution products, quadrature subtraction, piecewise native sigma, R = 100 as FWHM on top of 300 km/s, NIR only: Task 6.
- Index definitions, air-to-vacuum, F_nu D4000, F_lambda HdeltaA and bump, sign convention: Task 4.
- Outputs and figures for Step 1 and Step 2, spectra not stored for the population: Task 10, Task 11.
- Validation items (null, broadening, linearity, integrator, pilot timing): Tasks 4, 6, 5, 8, 10, 11.
- Code layout and style, uv, Ruff, Phase 1 untouched: Task 2 and throughout.
- Names used across tasks: `SspGrid` fields (`wave_a`, `log_age_yr`, `log_z_grid`, `agb_weights`, `flux_nu`, `native_sigma_km_s`, `product`, `provenance`), `csp_track`, `agb_two_spectra`, `epoch_weight_matrix`, `interpolate_log_z`, `csp_spectra`, `measure_all`, `compute_track_indices`, `population_indices` are consistent between definitions and uses.
