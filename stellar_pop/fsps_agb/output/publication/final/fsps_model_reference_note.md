# The FSPS TP-AGB model study behind the H-minus bump paper: reference note

Written 2026-09-25 for the drafting agent. This is the complete record of the modelling
work in `stellar_pop/fsps_agb` (monorepo `luandun`), not paper text. Read it before
drafting the model section. Every number quoted here is computed by a script in that
folder and stored in a JSON file named in the text; nothing is estimated. The project's
own documents (`docs/SPEC.md`, `docs/ANALYSIS.md`, `docs/PUBLICATION.md`,
`docs/lessons.md`, `docs/todo.md`, `README.md`) hold the same material in the order it
was produced; this note reorganises it around the final figures.

Contents

1. What the study is for and what it concludes
2. The stellar-population model in FSPS
3. How the TP-AGB stars enter the spectrum, read from the FSPS source
4. Simple stellar population grids and resolution products
5. Star formation histories and the mock population
6. The three spectral indices
7. Classes, the classifier and the recovery regression
8. Results, figure by figure
9. Metallicity: the FSPS facts, the caveat and the advantage
10. Caveats and model omissions
11. Reference guide: code, data, commands, JSON keys
12. Numbers at a glance

---

## 1. What the study is for and what it concludes

The paper (paper I on the 1.6 micron H-minus bump, `~/Dropbox/Apps/Overleaf/H-bump/main.tex`)
proposes the H-minus opacity bump at 1.6 micron as a spectral index for galaxy
evolution, alongside D4000 and HdeltaA. The model section has two messages, and the
study was built to test them with FSPS models rather than assume them:

- **Message 1.** The bump diagnoses the thermally pulsing asymptotic giant branch
  (TP-AGB) contribution. With TP-AGB light in the model ("TP-AGB on"), the population of
  quenching galaxies sits 0.04 to 0.08 mag deeper in the bump than without it ("AGB
  off"), far beyond the 0.005 to 0.02 mag measurement precisions considered.
- **Message 2.** With TP-AGB on, the bump adds information about the star formation
  history (SFH) that D4000 and HdeltaA do not carry. HdeltaA peaks 0.2 to 0.3 Gyr
  after quenching, the bump peaks 0.5 to 1.5 Gyr after quenching with a delay that
  grows with the quenching timescale, and D4000 only rises. In a nearest-neighbour
  test the bump raises the completeness of a rapid-quenching class from 0.40 to 0.49
  and its purity from 0.61 to 0.65 at 0.005 mag precision.

Two qualifications matter for the drafting:

- The size of the bump signal depends on which TP-AGB spectral templates FSPS uses.
  With the default C3K hydrostatic model atmospheres the TP-AGB signal in the bump is
  0.01 to 0.02 mag; with the empirical Lancon and Mouhcine (2002) O-rich templates it
  is 0.05 to 0.13 mag. The paper's "TP-AGB on" is the empirical-template configuration;
  the paper must say so. Section 3 explains why this happens.
- The 19 z ~ 1 quiescent galaxies of Lu+2026 have bump values of -0.037 to -0.073 mag,
  inside the TP-AGB-on locus and 0.02 to 0.05 mag deeper than anything the TP-AGB-off model
  produces at the same D4000. That is an empirical argument for the TP-AGB-on
  configuration, shown in the fourth final figure.

The four final figures are in this folder:

| File | Role | Section |
| --- | --- | --- |
| `index_planes_agb_on_off.{pdf,png}` | key figure 1, main text | 8.1 |
| `age_sensitivity_and_classifier_gain.{pdf,png}` | key figure 2, main text | 8.2 |
| `index_planes_sfh_families.{pdf,png}` | appendix | 8.3 |
| `d4000_hminus_plane_with_jwst_data.{pdf,png}` | data comparison | 8.4 |
| `observed_stack_vs_fsps_mocks.{pdf,png}` | spectral-shape comparison (replaces the draft's Figure 8) | 8.5 |

Captions are in the `.tex` files of the same names. The numbers drawn in the figures are
in `final_summary.json` (this folder) and, for the classifier and recovery sweeps and
the earlier candidates, in `../publication_summary.json`.

---

## 2. The stellar-population model in FSPS

### 2.1 Software and build

| Item | Value |
| --- | --- |
| FSPS | Fortran FSPS at commit `bd187a0` (v4.0 line), data in `$SPS_HOME = /Users/shuang/code/fsps` |
| python-fsps | 0.5.1.dev0 built from commit `7d202b8`, wheel in `stellar_pop/fsps_agb/wheels/` |
| Compiled libraries | MIST isochrones, C3K high-resolution spectra (`c3k_hr`), Draine and Li 2007 dust |
| Patches applied | `docs/patches/c3k_hr_nzinit_13.patch` (see below) and `docs/patches/python_fsps_cmake_c3k_hr.patch` (adds `-cpp -DC3K_LR=0 -DC3K_HR=1` so the high-resolution C3K branch is compiled) |
| Environment | `uv sync` installs numpy, scipy, matplotlib and the local wheel; `export SPS_HOME=/Users/shuang/code/fsps`; never `pip install -U fsps`, which would replace the patched wheel with the PyPI low-resolution build |

The `nzinit` patch fixes a real FSPS bug: the C3K_HR loader read only 11 of the 13
metallicity files, so MIST isochrones at log Z = +0.25 and +0.5 were silently paired
with solar-metallicity spectra. After the patch the log Z = +0.25 SSP spectrum differs
from the pre-patch one by up to 21.5 per cent in flux (`docs/lessons.md`, Task 1).

Provenance is stored with every SSP grid (`output/ssp_grid*/provenance*.json`): FSPS
version, library tuple, git hash and diff hash of `$SPS_HOME`, and every non-default
parameter.

### 2.2 Fixed configuration

| Setting | Value |
| --- | --- |
| Isochrones | MIST, solar Z = 0.0185, 107 ages, log(age/yr) = 5.00 to 10.30 in 0.05 dex |
| Spectral library | C3K high resolution: 10992 vacuum wavelengths; R = 3000 (sigma 42.4 km/s) over 3000 to 10000 A, R = 500 (sigma 254.6 km/s, 16 A pixels) over 1 to 2.5 micron |
| IMF | Chabrier (2003), `imf_type = 1` |
| Metallicities built | log(Z/Zsun) = -0.50, -0.25, 0.00, +0.25 (FSPS `zmet` = 9, 10, 11, 12) |
| Nebular emission, dust attenuation, IGM | off |
| Post-AGB weight `pagb` | 1 (default); post-AGB stars contribute under 0.5 per cent anywhere in the range and are negligible in the NIR |
| AGB circumstellar dust | on (default, `add_agb_dust_model = 1`, `agb_dust = 1`); changes the 1.4 to 1.8 micron flux by 0.4 per cent |
| TP-AGB weight `agb` | 0 and 1 built, 2 derived (Section 2.3) |
| TP-AGB template switch `use_lw_tpagb` | 0 (C3K) for the TP-AGB-off grid, 1 (LW02) for the TP-AGB-on grid |

Wavelength window kept: 3400 A to 2.2 micron (7263 native pixels).

### 2.3 The two AGB configurations and the `agb` weight

Naming: both knobs below act on the thermally pulsing AGB phase only (MIST `phase ==
5`); the early AGB (`phase == 4`, 5 to 7 per cent of the bolometric light at every age)
is never varied and keeps its C3K spectra in both configurations. The configurations
are therefore called "TP-AGB on" and "TP-AGB off", never "AGB on/off" (see
`naming_tp_agb_note.md`).

FSPS multiplies the IMF weight of every TP-AGB star (MIST phase 5) by `agb`
(`mod_gb.f90`); it does not change the stars' luminosities or temperatures. The SSP
spectrum is therefore exactly linear in `agb`, verified to 1e-15, so S(agb = 2) = 2 S(1)
minus S(0) with no extra FSPS call. Only `agb = 0` and `agb = 1` are built.

The two configurations used throughout the paper's figures:

| Name in figures | Template switch | `agb` | Meaning |
| --- | --- | --- | --- |
| TP-AGB off | `use_lw_tpagb = 0` (C3K) | 0 | no TP-AGB light at all; the clean control |
| TP-AGB on | `use_lw_tpagb = 1` (LW02) | 2 | empirical O-rich TP-AGB spectra at twice the fiducial weight |

Two intermediate prescriptions exist in the outputs and in the candidate figure
`fig2_bump_offsets`: C3K at `agb = 1` (the FSPS default) and LW02 at `agb = 1`. In the
D4000 = [1.3, 1.5) slice of the population the median bump steps are (from
`../publication_summary.json`, `fig2_bump_offsets.slice.neighbouring_separations`):

| Step | What changes | Median offset [mag] | In units of the wider 16-84 half width | In units of 0.01 mag |
| --- | --- | ---: | ---: | ---: |
| C3K agb 0 to agb 1 | TP-AGB weight inside C3K | -0.0052 | 1.1 | 0.5 |
| C3K agb 1 to LW02 agb 1 | template at fixed weight | -0.0266 | 6.4 | 2.7 |
| LW02 agb 1 to agb 2 | TP-AGB weight inside LW02 | -0.0234 | 3.9 | 2.3 |

So the population bump locus tests which TP-AGB spectral template is right far more
sharply than how much TP-AGB light there is, and doubling the weight inside C3K stays
under the 0.01 mag yardstick. C3K at agb = 1 differs from agb = 0 by 0.005 mag, which
is why agb = 0 is an adequate "off" (Section 10 lists an agb = 1 variant as an open
item).

Phase 1 (an SSP sandbox, 1 Gyr, solar Z) established the orders of magnitude: TP-AGB
stars supply about 31 per cent of the 1.4 to 1.8 micron flux, `agb = 0` lowers the NIR
F_nu by 31 per cent and `agb = 2` raises it by 31 per cent, `pagb` and the dust model
are second order, `tpagb_norm_type` and `fcstar` are inert for MIST, and `use_lw_tpagb
= 1` changes the NIR flux by only 3.4 per cent but changes its shape.

---

## 3. How the TP-AGB stars enter the spectrum, read from the FSPS source

Read from `getspec.f90`, `sps_setup.f90`, `sps_vars.f90`, `mod_gb.f90` and
`add_agb_dust.f90` at commit `bd187a0`; `$SPS_HOME/src` and the python-fsps `libfsps`
submodule are identical files.

- **Default, `use_lw_tpagb = 0`.** A phase-5 star falls through to the main library
  branch (`getspec.f90` lines 150 to 200): bilinear interpolation in log Teff and log g
  of the C3K grid slice at the star's own metallicity index `pset%zmet`, clamped to the
  grid edge (2500 K, log g = -1). If a corner spectrum is missing the code takes one
  available corner ("a very crude hack" in the source). The C3K templates are
  hydrostatic model atmospheres and fully metallicity dependent.
- **`use_lw_tpagb = 1`.** An O-rich phase-5 star with log Teff < 3.6 (about 3980 K)
  takes the branch at lines 90 to 102: a linear interpolation in Teff only, no log g,
  no metallicity index, of nine averaged empirical Lancon and Mouhcine (2002) spectra
  `Orich.spec` (`agb_spec_o(nspec, 9)`), scaled by the star's bolometric luminosity
  (the templates are normalised to unit integrated flux). The one Z-aware ingredient is
  the Teff label table `agb_logt_o(nz, 9)`, built in `sps_setup.f90` lines 438 to 461
  by interpolating `Orich.teff` (22 columns in log Z from -1.98 to +0.20) to each
  isochrone metallicity, without clamping, so log Z = +0.25 is extrapolated. The
  spectral shapes never change with Z; only which template a star of a given Teff
  blends does. O-rich TP-AGB stars hotter than the switch still get C3K spectra.
- **Carbon stars.** C/O > 1 sends a star to the Aringer et al. (2009) templates (Teff
  only, no Z), regardless of `use_lw_tpagb` (`cstar_aringer = 1` is a compile-time
  constant). In the MIST tables C/O > 1 occurs only at [Fe/H] <= -2, so no carbon
  star exists in this grid (log Z >= -0.5). The model therefore has no C-rich TP-AGB
  spectra at all.
- **Normalisation.** For MIST no metallicity-dependent TP-AGB normalisation is applied:
  `mod_gb.f90` guards the Conroy and Gunn (2010) corrections and the Villaume et al.
  (2015) weighting with `isoc_type == 'pdva'`. Only the user factors act: the
  multiplicative `agb`, and `dell`, `delt` (left at 0).
- **Circumstellar dust.** `add_agb_dust.f90` receives the metallicity but its only Z
  term is commented out; the dust depends on the C/O class, Teff and the optical depth
  from a Vassiliadis and Wood (1993) mass-loss rate.
- **What does depend on metallicity: the isochrone.** The coolest phase-5 log Teff in
  MIST is 3.412 at [Fe/H] = +0.5, 3.456 at 0.0, 3.533 at -1.0, so the fraction of TP-AGB
  stars below the fixed log Teff = 3.6 switch, hence the fraction assigned the LW02
  template, rises steeply with Z.

Why the LW02 templates produce a much larger bump index (Phase 3, Q1): the signal is
not a feature inside the 1.57 to 1.73 micron feature band but H2O absorption in the
two side bands of the empirical spectra. The pure TP-AGB component (S(1) minus S(0))
drops steeply at 1.33 to 1.50 micron next to the blue side band and at 1.75 to 1.80
micron next to the red one, so the pseudo-continuum is pulled down and the feature
band appears as a bump. The component has a bump index of -0.19 to -0.24 mag with
LW02 against -0.015 to -0.037 mag with C3K, at the same TP-AGB light fraction (37.5
per cent of the feature-band light at 0.79 Gyr). With C3K the TP-AGB light enters
mostly as a tilted continuum that the pseudo-continuum removes. The component
spectrum has straight interpolated segments at 1.34 to 1.42 and 1.81 to 1.93 micron
where the empirical spectra have telluric gaps, and its shape is age independent
because FSPS uses one fixed template set (`output/analysis/lw02_q1_component_spectrum.png`).

---

## 4. Simple stellar population grids and resolution products

### 4.1 Grids

`ssp_grid.py` builds the SSPs once with python-fsps (`sfh = 0`, `zcontinuous = 0`,
`tage = 0` returns all 107 ages) for the four metallicities and `agb` in {0, 1}, and
caches them (`output/ssp_grid/native.npz`, shape 4 x 2 x 107 x 7263; 49.9 MB) with
provenance. The LW02 grid is `output/ssp_grid_lw02/` with `use_lw_tpagb = 1`. Eight SSP
builds take 97 s (12 s each); changing `agb`, the IMF or the LSF invalidates the FSPS
cache, which is why the CSPs are assembled outside FSPS.

`ssp_grid.py --surviving-mass` stores FSPS `StellarPopulation.stellar_mass` (living
stars plus remnants, `agb = 1`) per solar mass formed for the 4 metallicities x 107
ages (`surviving_mass.npz`). It is used to normalise the specific star formation rates
(Section 7.1). The fraction is slightly above 1 below about 2 Myr because the youngest
MIST isochrones lack low-mass stars; that age range carries negligible mass and lies
far below the 1 Gyr floor of every population statistic.

### 4.2 Resolution products (`broadening.py`)

All smoothing is Gaussian in velocity on a uniform log-wavelength grid (30 km/s step),
applied to lambda F_lambda, after subtracting the native library resolution in
quadrature. The native resolution is piecewise constant (42.4 km/s below 10000 A, 254.6
km/s above), so the two segments are convolved separately with 6000 km/s padding; no
index window lies within 800 A of the 10000 A splice.

| Product | Range | Added kernel | Use |
| --- | --- | --- | --- |
| `native` | 3400 A to 2.2 micron | none | reference |
| `sigma300` | 3400 A to 2.2 micron | sqrt(300^2 - sigma_lib^2): 297 km/s optical, 159 km/s NIR | D4000, HdeltaA (and a bump variant) |
| `r100` | 1.25 to 2.1 micron | sqrt(sigma_R100^2 + 300^2 - 254.6^2) with sigma_R100 = c / (2.3548 x 100) = 1273 km/s (R defined as FWHM) | the bump in every final figure |

The `sigma300` and `r100` bumps differ by at most 0.001 mag, almost all of it a constant
offset (epoch-to-epoch scatter 0.0001 mag), so results at the two products are not
independent. Phases 3 to 5 quoted the `sigma300` bump; the final figures use `r100`
and every classifier and recovery number was recomputed with it (differences within
the seed scatter).

Because convolution and SFH integration are both linear, smoothing the SSPs once is
identical to smoothing every composite spectrum.

---

## 5. Star formation histories and the mock population

### 5.1 The SFH families (`sfh_model.py`)

Time t runs from the start of star formation. All families share a delayed-tau rise
SFR(t) = (t / tau) exp(-t / tau) for t < t_q, with tau = t_q unless stated. After t_q:

| Family | SFR(t >= t_q) | FSPS-native analogue | Role |
| --- | --- | --- | --- |
| `exponential` | e^-1 exp(-(t - t_q) / tau_q) | none (cross-checked with tabular `sfh = 3`) | the paper's default |
| `linear` | e^-1 max(0, 1 - (t - t_q) / delta_q), delta_q = 2 ln 2 tau_q (same half-life) | `sfh = 5` with `sf_slope` | robustness |
| `truncation` | 0 | `sfh = 4` with `sf_trunc` | robustness |
| `decoupled` | (t_q / tau) exp(-t_q / tau) exp(-(t - t_q) / tau_q), rise tau drawn independently, log-uniform on [0.5, 5] Gyr, seed 20260926 | none | robustness |

Priors, identical for all families (seed 20260924, 2000 draws, `draws.npz`):

| Parameter | Distribution | Range |
| --- | --- | --- |
| t_q | uniform | [1, 5.9] Gyr |
| tau_q | log-uniform | [0.1, 3] Gyr |
| log(Z/Zsun) | normal, mean 0, sigma 0.2 dex, truncated | [-0.5, +0.2] |

The fiducial history is t_q = 3 Gyr, tau_q = 0.3 Gyr, solar Z. Each history has one
metallicity for its whole life. Every history is observed at 260 epochs, t_obs = 0.05,
0.10, ..., 13.0 Gyr, and every population statistic keeps epochs t_obs >= 1 Gyr
(482,000 rows per family and template; the discrete epochs make stripes earlier and no
real sample is that young). Note that the population is uniform in observation time
along each history, which weights it differently from a mock table that draws one
t_obs per galaxy.

### 5.2 The integrator (`csp_integrate.py`)

For each epoch the SFH is integrated over lookback age on a log-spaced sub-grid (0.01
dex from 1e5 yr to t_obs, plus the sliver below 1e5 yr). The mass formed in each sub-bin
is the analytic integral of the SFR (closed forms for every family), assigned to the
two bracketing SSP ages by linear interpolation in log age, matching the FSPS kernel.
Metallicity is interpolated linearly in log Z between the two bracketing grid SSPs
(FSPS `zcontinuous = 1`). The composite spectrum is one matrix product (epoch weights
of shape epochs x ages, times SSP flux of shape ages x pixels). A first version with
0.05 Gyr bin centres disagreed with FSPS by up to 45 per cent in the D4000 window while
star formation was ongoing, because stars younger than 100 Myr change their blue flux
by large factors within one bin; the log sub-grid fixed it.

Cross-checks against FSPS's own engine, fiducial history, maximum relative flux
difference inside any index window:

| Family | FSPS route | Largest difference |
| --- | --- | --- |
| `exponential` | tabular `sfh = 3` | 0.312 per cent at 1 Gyr, 0.006 per cent at 13 Gyr (`output/single_csp/fsps_cross_check.json`) |
| `linear` | `sfh = 5` | 0.023 per cent (`output/single_csp/fsps_family_cross_check.json`) |
| `truncation` | `sfh = 4` + `sf_trunc` | 0.027 per cent |
| `decoupled` | none available | internal `quad` and continuity checks only (`tests/test_sfh_families.py`) |

The validation threshold was 2 per cent. The `sf_slope` sign convention of `sfh = 5`
was verified directly (SFR ratio 0.466 across half a ramp, declining).

### 5.3 The populations

`run_population.py` writes, per family and template, `draws.npz`, `indices.npz` (and
`.csv`) with one row per history and epoch, and `summary.json`. Columns: `history_id`,
`t_q_gyr`, `tau_q_gyr`, `log_z`, `epoch_gyr`, `time_since_quenching_gyr`, `sfr`,
`ssfr_0_100_myr`, `ssfr_100_1000_myr`, `surviving_mass_fraction`, and the indices
`d4000_agb{0,1,2}`, `hdelta_a_agb{0,1,2}`, `h_minus_bump_{sigma300,r100}_agb{0,1,2}`.
A full run takes 5 to 8 minutes. Directories:

| Family | C3K | LW02 |
| --- | --- | --- |
| exponential | `output/population/` | `output/population_lw02/` |
| linear | `output/population_linear/` | `output/population_linear_lw02/` |
| truncation | `output/population_truncation/` | `output/population_truncation_lw02/` |
| decoupled | `output/population_decoupled/` | `output/population_decoupled_lw02/` |

The `indices.npz` tables (about 340 MB per directory) are git-ignored and must be
regenerated from the SSP grids if absent.

---

## 6. The three spectral indices (`spectral_indices.py`)

FSPS wavelengths are vacuum. Lick/IDS and D4000 bands are defined in air and converted
with the Morton (1991) relation FSPS itself uses. The H-minus bands were defined on
model grids and are used as vacuum values.

| Index | Blue band | Feature band | Red band | Form |
| --- | --- | --- | --- | --- |
| D4000 | 3750 to 3950 A (air) | none | 4050 to 4250 A (air) | ratio of mean F_nu, red over blue (Bruzual 1983 form) |
| HdeltaA | 4041.60 to 4079.75 A (air) | 4083.50 to 4122.25 A (air) | 4128.50 to 4161.00 A (air) | equivalent width in A on F_lambda, integral of (1 - F / F_c) over the feature |
| H-minus bump | 1.494 to 1.539 micron | 1.570 to 1.734 micron | 1.746 to 1.791 micron | -2.5 log10 of the mean of F / F_c over the feature band, on F_lambda; negative for a bump |

F_c is the straight line through the band-mean F_lambda of the blue and red windows,
anchored at the band midpoints. Band means use exact band limits and Gauss-Legendre
quadrature inside pixels. Sign convention for the drafting: a stronger bump is a more
negative index; the figures invert the bump axis so that "up" is a stronger bump, and
the age-sensitivity figure plots the bump strength (minus the index).

Ranges over the fiducial track (`output/single_csp/indices.csv`): D4000 1.02 to 2.34,
HdeltaA -4.6 to 6.3 A, bump -0.022 to +0.011 mag (C3K). Over the C3K population the
`r100` bump spans -0.027 to +0.019 mag; over the LW02 population it reaches -0.10 mag.

Measurement yardsticks used in every noise test: 0.05 in D4000, 0.5 A in HdeltaA,
and 0.005, 0.010, 0.020 mag in the bump (the SPHEREx ARP 151 measurement reached 0.018
mag; JWST prism stacks are at the 0.01 level).

---

## 7. Classes, the classifier and the recovery regression

### 7.1 Classes (`population_classes.py`)

Two specific star formation rates per epoch, both divided by the surviving stellar
mass at t_obs (living stars plus remnants, Section 4.1): sSFR over the last 100 Myr and
sSFR over 100 Myr to 1 Gyr before the epoch. With R = sSFR(0 to 100 Myr) / sSFR(100 Myr
to 1 Gyr), the Zhang et al. (2023) style rules are

- star forming: R > 1
- rapid quenching: sSFR(100 Myr to 1 Gyr) > 1e-10 per yr and R < 0.1 (post-starburst
  is the subset with sSFR(0 to 100 Myr) < 1e-11 per yr)
- quiescent: sSFR(100 Myr to 1 Gyr) < 1e-10 per yr and sSFR(0 to 100 Myr) < 1e-11 per yr
- transitional: everything else

Rules are applied in that order, so quiescent overrides rapid quenching, which
overrides star forming. Class counts over the 482,000 epochs, identical for C3K and
LW02 within a family (`../../analysis/sfh_sensitivity_summary.json`,
`s3_class_fractions`):

| Family | Star forming | Rapid quenching | Transitional | Quiescent | Rapid-quenching fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| exponential | 102,100 | 5,907 | 126,762 | 247,231 | 1.23 per cent |
| linear | 102,394 | 17,262 | 45,296 | 317,048 | 3.58 per cent |
| truncation | 99,922 | 30,730 | 3,466 | 347,882 | 6.38 per cent |
| decoupled | 42,305 | 4,072 | 161,835 | 273,788 | 0.84 per cent |

A sharper post-quench cutoff holds R below 0.1 for longer, so the rapid-quenching share
rises from decoupled to truncation. In the exponential family the R < 0.1 rule admits
only histories with tau_q < 0.27 Gyr into the rapid-quenching class
(`final_summary.json`, `index_planes_agb_on_off.agb_on.rapid_quenching_tau_q_range_gyr`),
which is why that class cannot be coloured by quenching timescale.

### 7.2 The classifier

- Observables: the true indices plus Gaussian noise at the yardsticks. One noise
  realisation per seed is added to every epoch; training and test see the same noisy
  values. The optical-only baseline uses the same noisy (D4000, HdeltaA) pair as the
  bump sets, so every comparison is paired.
- Method: k nearest neighbours, k = 25, majority vote (ties to the lowest class code),
  Euclidean distance after dividing each feature by its noise sigma. No class
  balancing; training reflects the 1.2 per cent base rate, so absolute completeness is
  low by construction.
- Cross-validation: 5 folds grouped by history (all 241 epochs of a history in one
  fold), so a test epoch never finds its own history in the training set.
- Completeness = TP / (TP + FN) and purity = TP / (TP + FP) of the rapid-quenching
  class on the held-out fold.
- Three noise seeds (20260924, 20260925, 20260926); a seed sets both the noise and the
  fold assignment. Rapid-quenching epochs are confused with transitional and
  quiescent epochs, essentially never with star-forming ones.

### 7.3 The recovery regression

- Sample: post-quench epochs only, 0 < t - t_q < 6 Gyr and t_obs >= 1 Gyr, 240,000
  rows.
- Targets: log10(t - t_q) and log10(tau_q).
- Method: k nearest neighbours regression, k = 25, same noise-scaled features and
  grouped folds; the prediction is the mean target of the 25 nearest training epochs.
- Metric: RMS of (predicted minus true) on the held-out fold, in dex.

### 7.4 Error bars and significance

- Figure error bars: the fold-to-fold standard deviation divided by sqrt(5), averaged
  over the seeds. Baseline bands are the same quantity for the optical-only set.
- Paired gain: for each seed, "with bump minus without" is computed fold by fold on the
  same folds; its standard error is the ddof-1 standard deviation over the 5 folds
  divided by sqrt(5); the reported gain and standard error are the means over seeds.
  A gain divided by this standard error is what the candidate figures labelled as
  "sigma". The user asked that these labels not appear in the paper figure: the gain is
  consistent across folds, but the absolute improvement is moderate, and a
  three-feature nearest-neighbour test is a proxy for information content, not a
  proposed method.
- An earlier caveat still holds: kNN is non-parametric and the noise realisation is
  shared between training and test, so the test answers "how separable are the
  classes given these observables at this precision", not "how would a real fit
  behave".

---

## 8. Results, figure by figure

### 8.1 Key figure 1: `index_planes_agb_on_off` (main text)

Two rows (TP-AGB off, TP-AGB on) of three panels: D4000 against HdeltaA, D4000 against the
bump, HdeltaA against the bump, exponential family, epochs later than 1 Gyr. Layers:

- grey filled contours enclosing 68, 95 and 99.5 per cent of all 482,000 epochs
  (smoothed 2-D histograms, 90 bins per axis, Gaussian smoothing of 1.2 bins);
- a red filled contour for the rapid-quenching class (68 and 95 per cent);
- light-blue tracks of the fiducial history at log Z = -0.5, -0.25, 0, +0.25 from t_q
  to t_q + 5 Gyr with markers at t_q and 0.5, 1, 2, 5 Gyr later;
- "TP-AGB off" and "TP-AGB on" in the empty top-right corner of the D4000-HdeltaA panels.

What it shows:

- The D4000-HdeltaA plane is identical in the two rows; the optical indices do not know
  about the TP-AGB stars.
- In the bump planes the TP-AGB-on population sits 0.04 to 0.08 mag deeper than the
  TP-AGB-off one and its rapid-quenching class occupies the deepest part of the plane:
  median -0.075 mag, 16-84 range -0.083 to -0.066 mag. With TP-AGB off the class is
  confined to a 0.007 mag wide band near zero, median -0.004 mag
  (`final_summary.json`, `index_planes_agb_on_off.<config>.rapid_quenching`).
- The metallicity dependence reverses sign between the rows (Section 9).
- The rapid-quenching class contains only tau_q < 0.27 Gyr histories, so it cannot be
  coloured by quenching timescale; a version with tau_q contours of the recently
  quenched epochs was tried and dropped because it added nothing beyond figure 2 and
  crowded the panels.

Numbers for the text (exponential family, LW02 population, D4000 = [1.3, 1.5)): the
TP-AGB-on bump is -0.059 mag (16-84: -0.065 to -0.053) against -0.004 mag (-0.008 to
+0.001) with TP-AGB off (`../publication_summary.json`, `fig2_bump_offsets.slice`).

### 8.2 Key figure 2: `age_sensitivity_and_classifier_gain` (main text)

Left, panels (a) to (c): one panel per tau_q = 0.3, 1 and 3 Gyr at t_q = 3 Gyr, TP-AGB on,
solar metallicity, sharing the log10(t - t_q) axis from 0.05 to 6 Gyr. Each panel shows
HdeltaA (blue), the bump strength (vermilion, minus the index) and D4000 (green), each
track scaled to [0, 1] over its own post-quench range so the three indices share one
axis. Stars mark the HdeltaA and bump maxima; an open star means the maximum sits at
the window edge, which happens for HdeltaA at tau_q = 3 Gyr where it only declines
after quenching. The unscaled ranges are in `final_summary.json`
(`age_sensitivity_and_classifier_gain.scaled_tracks`):

| tau_q | HdeltaA range [A] | bump strength range [mag] | D4000 range | HdeltaA peak | bump peak |
| --- | --- | --- | --- | ---: | ---: |
| 0.3 Gyr | -3.19 to 6.27 | 0.052 to 0.075 | 1.26 to 2.14 | 0.25 Gyr | 0.8 Gyr |
| 1 Gyr | -2.44 to 5.97 | 0.052 to 0.065 | 1.26 to 2.04 | 0.2 Gyr | 1.25 Gyr |
| 3 Gyr | 2.33 to 5.90 | 0.052 to 0.059 | 1.25 to 1.53 | window edge | 1.5 Gyr |

The candidate figure `fig3_quenching_clocks` adds tau_q = 0.1 Gyr (HdeltaA peak 0.2
Gyr, bump peak 0.5 Gyr), a finer tau_q grid at three t_q values, and the TP-AGB-off bump,
which has no interior extremum for any tau_q (`../publication_summary.json`,
`fig3_quenching_clocks`). The delay of the bump minimum grows monotonically with tau_q
(0.5 Gyr at 0.1 Gyr to 1.5 Gyr at 3 Gyr for t_q = 3 Gyr) and shifts by at most 0.3 Gyr
between t_q = 1.5 and 4.5 Gyr, so the pair (HdeltaA, bump) constrains the quenching
timescale.

Right, panels (d) and (e): rapid-quenching completeness and purity against bump
precision, TP-AGB on, with the D4000-plus-HdeltaA baseline as a dashed line and its fold
standard error as a band (`final_summary.json`,
`age_sensitivity_and_classifier_gain.classifier`):

| | Optical only | + bump at 0.005 mag | + bump at 0.010 mag | + bump at 0.020 mag |
| --- | ---: | ---: | ---: | ---: |
| Completeness | 0.398 +/- 0.005 | 0.488 +/- 0.009 | 0.445 +/- 0.009 | 0.405 +/- 0.007 |
| Purity | 0.610 +/- 0.010 | 0.649 +/- 0.011 | 0.632 +/- 0.010 | 0.621 +/- 0.010 |

Paired gains and their fold standard errors (`../publication_summary.json`,
`fig4_information_gain.agb_on.classifier`): completeness +0.090 +/- 0.008, +0.047 +/-
0.008, +0.006 +/- 0.005; purity +0.039 +/- 0.005, +0.021 +/- 0.003, +0.010 +/- 0.006.
For TP-AGB off (not in the final figure) the completeness changes are +0.006, -0.010,
-0.013 and the purity changes +0.012, +0.008, +0.006: the bump adds essentially
nothing without TP-AGB light.

Recovery results, kept out of the paper figure because no real analysis would infer
these parameters from three indices, but useful for the text
(`fig4_information_gain.<config>.recovery`):

| RMS error [dex] | Optical only | + bump 0.005 mag | + bump 0.010 mag | + bump 0.020 mag |
| --- | ---: | ---: | ---: | ---: |
| log10(t - t_q), TP-AGB on | 0.240 | 0.222 | 0.231 | 0.237 |
| log10(tau_q), TP-AGB on | 0.311 | 0.298 | 0.305 | 0.309 |
| log10(t - t_q), TP-AGB off | 0.240 | 0.234 | 0.238 | 0.240 |
| log10(tau_q), TP-AGB off | 0.311 | 0.310 | 0.311 | 0.311 |

The tau_q gain (0.013 dex at 0.005 mag) exists only with TP-AGB on; the small TP-AGB-off age
gain is the age information any 1.6 micron continuum index carries.

### 8.3 Appendix figure: `index_planes_sfh_families`

Key figure 1's TP-AGB-on row for the linear, truncation and decoupled families (three rows,
population and rapid-quenching contours only, family named in the top-right corner of
the first panel). The rapid-quenching class lands in the deep-bump region in every
family: its median bump is -0.074 (linear), -0.076 (truncation) and -0.075 (decoupled)
mag against -0.075 mag for the exponential family (`final_summary.json`,
`index_planes_sfh_families.<family>.rapid_quenching`).

Per-family gains, TP-AGB on, from the candidate figure `fig6_robustness_gains`
(`../publication_summary.json`, `fig6_robustness_gains`):

| Family | Delta completeness (bump at 0.010 mag) | Delta purity (0.010 mag) | RMS gain log10(t - t_q) (0.005 mag) [dex] | RMS gain log10(tau_q) (0.005 mag) [dex] |
| --- | ---: | ---: | ---: | ---: |
| exponential | +0.047 +/- 0.008 | +0.021 +/- 0.003 | 0.018 | 0.013 |
| linear | +0.038 +/- 0.004 | +0.022 +/- 0.003 | 0.017 | 0.004 |
| truncation | +0.026 +/- 0.003 | +0.010 +/- 0.002 | 0.031 | none (no tau_q) |
| decoupled | +0.052 +/- 0.010 | +0.036 +/- 0.006 | 0.010 | 0.010 |

The TP-AGB-off completeness gain is within two standard errors of zero or negative in
every family. The optical-only baselines differ a lot between families (completeness
0.32 to 0.85) because the class base rate does (0.84 to 6.4 per cent), so the deltas,
not the absolute values, are comparable across families.

The population TP-AGB offset (agb 2 minus agb 0 at fixed D4000, slice [1.5, 1.7)) per
family: C3K -0.006 to -0.011 mag (1 to 3 times the per-model scatter), LW02 -0.046 to
-0.067 mag (4 to 12 times the scatter); the C3K-to-LW02 template separation is 0.049 to
0.061 mag in every family (`../../analysis/sfh_sensitivity_summary.json`, `s6_tpagb_offset_by_family`).

### 8.4 Data comparison: `d4000_hminus_plane_with_jwst_data`

The D4000 versus bump panels of key figure 1, TP-AGB off left and TP-AGB on right, on one
shared bump axis, with the 19 quiescent galaxies of Lu+2026 (`JWST_QG_indices.npz`:
`ID`, `z`, `D4000`, `D4000_err` as 16th and 84th percentile bounds, `Hbump`,
`Hbump_err` symmetric) overplotted. Sample: z = 1.016 to 1.955; D4000 = 1.353 to 1.940;
bump = -0.0369 to -0.0731 mag, median -0.0588 mag; bump errors 0.0002 to 0.0025 mag.
Over D4000 = 1.35 to 1.95 the TP-AGB-on population spans -0.081 to -0.041 mag (1st to
99th percentile) and the TP-AGB-off population -0.021 to +0.003 mag. Every galaxy lies
inside the TP-AGB-on locus, several near its rapid-quenching region, and 0.02 to 0.05 mag
deeper than anything the TP-AGB-off model produces at the same D4000. The comparison does
not test the TP-AGB weight (Section 2.3): it tests the template.

### 8.5 Spectral-shape comparison: `observed_stack_vs_fsps_mocks`

Built by `jwst_spectrum_comparison.py` (about 2 minutes, most of it the search),
numbers in `observed_stack_vs_fsps_mocks.json`, the full search tables in
`observed_stack_vs_fsps_mocks_search_{agb_off,agb_on}.npy` (columns log Z, t_q,
tau_q, mismatch).

Observed side: the 19 galaxies of the index table (the 8 other files in `qg_spec/`
have no redshift there and are not used), shifted to the rest frame with the table
redshifts, each divided by a straight line fitted to all pixels in the two side
windows (the pyphot degree-1 convention of the draft), and combined as an S/N-weighted
mean (weights (F/sigma)^2 per pixel) on a 45 A rest-frame grid, the median pixel width
of the spectra in this region (27 to 60 A). All spectra are drawn as steps.
Recomputing the bump index from the normalised spectra reproduces the table values to
0.001 to 0.002 mag for 18 galaxies; galaxy 20150 gives -0.050 against the stored
-0.059 (`observed.per_galaxy`), worth checking with the data owner. The stack peaks at
1.080 in the feature band and has a bump index of -0.057 mag.

Model side: one epoch, the cosmic age at the median redshift z = 1.357 (4.57 Gyr for a
flat LCDM cosmology with H0 = 70 and Omega_m = 0.3, star formation from t = 0; grid
epoch 4.55 Gyr), on the `r100` product, normalised the same way and binned to the 45 A
grid, displayed over 1.46 to 1.83 micron. Two layers:

- The panel curves: solar metallicity, t - t_q fixed at 1 Gyr (t_q = 3.55 Gyr), tau_q in
  {0.1, 0.3, 1, 3} Gyr.
- For TP-AGB on only (an TP-AGB-off search is not meaningful and is not drawn): the closest model in a grid search over log Z from -0.5 to +0.25 in 0.05 dex steps
  (interpolated between the four grid metallicities), t_q from 1 Gyr to the epoch in
  0.1 Gyr steps and 13 log-spaced tau_q values (7280 models per configuration), ranked
  by the RMS of (model minus stack) divided by the galaxy-to-galaxy scatter over 1.494
  to 1.791 micron ("mismatch"; 1 means the model deviates by one scatter on average).

| | TP-AGB off | TP-AGB on |
| --- | ---: | ---: |
| panel bump index, tau_q = 0.1 / 0.3 / 1 / 3 Gyr [mag] | -0.007 / -0.005 / -0.001 / +0.001 | -0.072 / -0.072 / -0.064 / -0.058 |
| panel mismatch, tau_q = 0.1 / 0.3 / 1 / 3 Gyr | 2.41 / 2.49 / 2.65 / 2.73 | 0.74 / 0.74 / 0.45 / 0.39 |
| best match | not drawn; a one-off search gave log Z = -0.25, t_q = 1.0, tau_q = 0.10 Gyr, mismatch 2.03 (search floor 2.03 to 3.06) | log Z = +0.15, t_q = 1.2, tau_q = 0.41 Gyr (delay 3.35 Gyr), index -0.059 mag |
| best mismatch (range over the search) | 2.03 | 0.36 (0.36 to 1.39) |
| median observed over best model in the feature band | 1.038 | 0.999 |

Reading: no TP-AGB-off model comes within two scatters of the stack; the best one is the
oldest, most abruptly quenched, sub-solar population the search allows, and it still
leaves a 4 per cent excess across the feature band. With TP-AGB on the tau_q = 3 Gyr
solar-metallicity curve at t - t_q = 1 Gyr is already within the scatter, and the
search finds a model that matches to 0.36 scatters, but the solution is degenerate:
the eight best TP-AGB-on models all have mismatch 0.36 and span t_q = 1.1 to 1.8 Gyr with
tau_q from 0.10 to 0.41 Gyr at log Z = +0.15, so the stack constrains the template and
roughly the metallicity, not the quenching history. Outside the side windows (below
1.45 and above 1.8 micron) the TP-AGB-on models sit 5 to 8 per cent above the data; that
region is not part of the normalisation or the search metric and reflects the H2O
absorption in the LW02 templates.

---

## 9. Metallicity: the FSPS facts, the caveat and the advantage

The bump at 1 Gyr after t_q for the fiducial history against log Z, decomposed
(`../publication_summary.json`, `fig7_metallicity.tracks.*.t_q+1_gyr`):

| Component | log Z = -0.5 | 0.0 | +0.25 | Slope [mag per dex] |
| --- | ---: | ---: | ---: | ---: |
| Non-AGB stars (C3K, agb 0) | -0.0075 | -0.0044 | +0.0004 | +0.011 |
| TP-AGB increment, C3K (agb 2 minus agb 0) | -0.0105 | -0.0096 | -0.0088 | +0.003 |
| TP-AGB increment, LW02 | -0.0478 | -0.0698 | -0.0844 | -0.049 |
| TP-AGB on total (LW02 agb 2) | -0.0552 | -0.0742 | -0.0840 | -0.038 |

Reading: the TP-AGB-off bump weakens slowly with metallicity, driven by the non-AGB stars
(the C3K TP-AGB increment is flat in Z). The TP-AGB-on bump deepens with metallicity four
times faster and in the opposite direction, and that trend is entirely the LW02
increment. Since the LW02 spectra carry no metallicity dependence (Section 3), the
trend is an isochrone plus Teff-cut effect: at higher Z more TP-AGB stars are cool
enough to receive the empirical template, and each receives a slightly cooler template
through the Z-dependent Teff labels. Any metallicity correction to the bump is only as
good as the MIST TP-AGB Teff distribution and the hard log Teff = 3.6 switch, and it
changes sign between the two template configurations, so no template-agnostic
correction exists. The paper must state this.

In the TP-AGB-on population, inside D4000 = [1.4, 1.7) (`fig7_metallicity.population_slice`):
the bump follows log Z with slope -0.031 mag per dex (all epochs) and -0.047 mag per
dex (rapid quenching); removing the linear trend shrinks the rapid-quenching 16-84 half
width from 0.0085 to 0.0049 mag, so 40 per cent of the class's bump scatter at fixed
D4000 is metallicity. The class stays offset from the rest of the slice at every
metallicity, by 0.013 mag at log Z in [-0.5, -0.3) rising to 0.024 mag at log Z in
[0.1, 0.25].

The advantage: supplying log Z with 0.1 dex noise as an extra feature (to both the
baseline and the bump set) does not dilute the bump's information but sharpens it
(`fig7_metallicity.known_z`, gains as a fraction of the baseline):

| Gain from the bump | Z unknown | Z known (0.1 dex) |
| --- | ---: | ---: |
| completeness | +11.8 +/- 2.1 per cent | +13.0 +/- 1.8 per cent |
| purity | +3.5 +/- 0.5 per cent | +3.8 +/- 0.7 per cent |
| RMS log10(t - t_q) | 7.6 +/- 0.2 per cent | 8.7 +/- 0.2 per cent |
| RMS log10(tau_q) | 4.1 +/- 0.2 per cent | 7.0 +/- 0.2 per cent |

With TP-AGB off, the same known metallicity makes the bump's completeness gain negative.
The metallicity dependence of the TP-AGB-on bump is thus a second handle on the TP-AGB
template: a sample with known metallicities tests the sign and slope of the table
above directly.

---

## 10. Caveats and model omissions

- The bump signal is template dependent: C3K gives a 0.02 mag SSP delta, LW02 0.11 mag.
  "TP-AGB on" in the paper means the LW02 empirical O-rich templates at agb = 2.
- No carbon-rich TP-AGB stars exist in the MIST tables above [Fe/H] = -2, so the model
  has no C-star spectra; real intermediate-age populations do.
- No nebular emission, no dust attenuation, no IGM; pure stellar continuum.
- Four metallicities interpolated linearly in log Z; one metallicity per history; the
  LW02 Teff labels are extrapolated at log Z = +0.25.
- The population is uniform in observation time along each history and the
  rapid-quenching base rate ranges from 0.84 to 6.4 per cent depending on the
  post-quench SFR form.
- The classifier and regression share the noise realisation between training and
  test and use only three features; they measure information content, not a method.
- The TP-AGB-off control at agb = 0 rather than the FSPS default agb = 1; the two differ
  by 0.005 mag in the bump. An agb = 1 variant of the figures is a one-constant change
  (`AGB_CONFIGS["agb_off"]["agb"] = "agb1"` in `publication_figures.py`).
- Two alternative contaminant families (bursty star forming with a 10 per cent late
  burst; slowly fading with tau_q = 3 to 6 Gyr; 300 histories each) never enter the
  rapid-quenching region of the planes, but neither can produce R < 0.1 by
  construction, so that test is weak (`docs/ANALYSIS.md`, Q2).

---

## 11. Reference guide: code, data, commands, JSON keys

All paths relative to `stellar_pop/fsps_agb/`.

### 11.1 Pipeline modules

| File | Purpose |
| --- | --- |
| `ssp_grid.py` | build, cache and load the FSPS SSP grids with provenance; `--surviving-mass` |
| `broadening.py` | log-wavelength resampling and quadrature-corrected Gaussian smoothing; the `sigma300` and `r100` products |
| `sfh_model.py` | the four SFH families, closed-form cumulative masses, priors and seeded draws |
| `csp_integrate.py` | log-age sub-grid weights, log-Z interpolation, the CSP matrix product, surviving mass per epoch |
| `spectral_indices.py` | D4000, HdeltaA, H-minus bump; air-to-vacuum conversion |
| `index_planes.py` | plane definitions and axis labels |
| `run_single_csp.py` | the fiducial track (`FIDUCIAL = {t_q 3, tau_q 0.3, log Z 0}`), `compute_track_indices` |
| `run_population.py` | the 2000-history populations (`--sfh-family`, `--grid-dir`, `--out-dir`, `--pilot`, `--figures-only`) |
| `population_classes.py` | class rules, noise, grouped folds, kNN classifier, completeness and purity |
| `cross_check_fsps_tabular.py`, `cross_check_fsps_families.py` | the FSPS cross-checks |
| `analysis_agb_separability.py` | Phase 3 Q1 (SSP deltas, component spectra, population offsets) |
| `analysis_fast_quenching.py` | Phase 3 Q2 (purity maps, classifier, balanced variant) |
| `analysis_alternative_sfh.py` | contaminant families |
| `analysis_conclusion_figures.py` | Phase 4 (C1 to C3: prescription test, age clocks, recovery regression, metallicity tracks) |
| `analysis_sfh_sensitivity.py` | Phase 5 (S1 to S6, per family) |
| `publication_figures.py` | Phase 6: the seven candidate figures and the final set |
| `jwst_spectrum_comparison.py` | the observed JWST stack against FSPS mock spectra at the sample epoch, with the best-match search (final figure 8.5) |
| `tests/` | 75 fast unit tests (`uv run pytest`); `-m slow` adds the FSPS-dependent ones |

### 11.2 Data products

| Path | Content |
| --- | --- |
| `output/ssp_grid/{native,sigma300,r100}.npz`, `surviving_mass.npz`, `provenance*.json` | C3K SSP grid, 4 Z x 2 agb x 107 ages, and its mass fractions |
| `output/ssp_grid_lw02/` | the same with `use_lw_tpagb = 1` |
| `output/single_csp/`, `output/single_csp_lw02/` | fiducial track spectra at every epoch, `indices.csv`, cross-check JSONs |
| `output/population*/` | the eight populations (Section 5.3) |
| `output/analysis/` | Phase 3 to 5 figures and summaries: `q1_summary.json`, `lw02_q1_summary.json`, `q2_summary.json`, `lw02_q2_summary.json`, `conclusion_summary.json`, `sfh_sensitivity_summary.json` |
| `output/publication/` | the seven candidate figures and `publication_summary.json` |
| `output/publication/final/` | the four final figures, their captions, `final_summary.json`, `JWST_QG_indices.npz`, this note |

### 11.3 Commands

```
uv sync
export SPS_HOME=/Users/shuang/code/fsps
uv run pytest                                   # 75 tests, 5 s
uv run python ssp_grid.py                       # 97 s, C3K grid
uv run python ssp_grid.py --surviving-mass
uv run python broadening.py                     # 2 s
uv run python run_population.py                 # 5 to 8 min per population
uv run python publication_figures.py --pilot    # 35 s dry run
uv run python publication_figures.py            # 10 min, all candidates and the sweeps
uv run python publication_figures.py --reuse-gains          # candidates from stored sweeps
uv run python publication_figures.py --final                # the final set, 10 s
```

The LW02 grid and populations are built with `--grid-dir output/ssp_grid_lw02` and the
`_lw02` output directories (the grid builder takes `extra_params={"use_lw_tpagb": 1}`).

### 11.4 Where each number lives

| Quantity | File and key |
| --- | --- |
| rapid-quenching bump statistics, key figure 1 | `final/final_summary.json`, `index_planes_agb_on_off.<agb_off|agb_on>.rapid_quenching` |
| metallicity track values at the marker epochs | same, `...metallicity_tracks.log_z_<value>` |
| scaled-track ranges and peak delays, key figure 2 | `final/final_summary.json`, `age_sensitivity_and_classifier_gain.scaled_tracks.tau_q_<value>` |
| completeness and purity against precision | same, `age_sensitivity_and_classifier_gain.classifier.<metric>` |
| paired gains with standard errors, both configurations | `publication_summary.json`, `fig4_information_gain.<config>.classifier.bump_<precision>` and `...recovery...` |
| per-family gains | `publication_summary.json`, `fig6_robustness_gains.<family>.<config>` |
| prescription offsets | `publication_summary.json`, `fig2_bump_offsets` |
| clock extrema and lag grid | `publication_summary.json`, `fig3_quenching_clocks` |
| metallicity decomposition, slice statistics, known-Z gains | `publication_summary.json`, `fig7_metallicity` |
| JWST sample statistics | `final/final_summary.json`, `d4000_hminus_plane_with_jwst_data` |
| observed stack and mock envelope statistics | `final/observed_stack_vs_fsps_mocks.json` |
| class counts per family, S6 offsets | `output/analysis/sfh_sensitivity_summary.json` |
| Phase 3 SSP deltas, component bumps, purity maps | `output/analysis/{q1,lw02_q1,q2,lw02_q2}_summary.json` |
| FSPS cross-checks | `output/single_csp/fsps_cross_check.json`, `fsps_family_cross_check.json` |

### 11.5 Figure conventions used in the final set

Computer Modern text through usetex; 7.1 inch double-column width; four major ticks per
axis, no minor ticks; the bump axis inverted so a stronger bump is up; TP-AGB off in blue
and TP-AGB on in vermilion where both appear; the rapid-quenching class in red; legends
below the panels with entries written as sentences; the fiducial SFH parameters in the
legend title.

---

## 12. Numbers at a glance

| Quantity | Value |
| --- | --- |
| Population per family and template | 2000 histories x 260 epochs; 482,000 epochs later than 1 Gyr |
| Rapid-quenching epochs, exponential family | 5,907 (1.23 per cent), all with tau_q < 0.27 Gyr |
| TP-AGB-on bump of the rapid-quenching class | median -0.075 mag (16-84: -0.083 to -0.066) |
| TP-AGB-off bump of the rapid-quenching class | median -0.004 mag (-0.007 to -0.000) |
| Template step C3K to LW02 at agb = 1, D4000 = [1.3, 1.5) | -0.027 mag (6.4 half widths) |
| Weight step agb 0 to 1 inside C3K | -0.005 mag (1.1 half widths) |
| HdeltaA peak after quenching | 0.2 to 0.3 Gyr, none for tau_q above about 0.6 Gyr |
| TP-AGB-on bump peak after quenching | 0.5 (tau_q 0.1) to 1.5 Gyr (tau_q 3) at t_q = 3 Gyr |
| Completeness, optical only to + bump at 0.005 mag | 0.398 to 0.488 (+0.090 +/- 0.008) |
| Purity, optical only to + bump at 0.005 mag | 0.610 to 0.649 (+0.039 +/- 0.005) |
| tau_q recovery RMS, TP-AGB on, + bump at 0.005 mag | 0.311 to 0.298 dex |
| TP-AGB-on bump slope with metallicity at t_q + 1 Gyr | -0.038 mag per dex (TP-AGB off: +0.011) |
| Lu+2026 sample | 19 galaxies, z = 1.0 to 2.0, bump -0.037 to -0.073 mag |
| Integrator against FSPS | at most 0.31 per cent flux difference in any index window |
