# Lessons

## 2026-08-31
- The pip-installed `fsps` (0.5.0) is compiled with **MIST** isochrones and the
  **low-resolution C3K** library (`c3k_lr`, 1936 wavelength points, ~80 Å/pix at
  1.6 µm), plus Draine & Li 2007 dust. Confirmed via `StellarPopulation.libraries`.
- Data files (isochrones, spectra) are read at runtime from `$SPS_HOME`
  (`/Users/shuang/code/fsps`); the pip package only ships the compiled Fortran.
- `tpagb_norm_type` only matters for **Padova** isochrones (`mod_gb.f90` guards it
  with `isoc_type == 'pdva'`), so it is inert for MIST.
- `fcstar` dilution is commented out in `ssp_gen.f90` → currently has no effect.
- `agb` (TP-AGB weight), `pagb` (post-AGB weight), `add_agb_dust_model` +
  `agb_dust`, and `use_lw_tpagb` are the meaningful MIST AGB knobs.
- The `python-fsps` `libfsps` submodule was checked out at `e2441ab` (v3.2-41),
  which lacks the `C3K_LR`/`C3K_HR` split; the parent repo records `05b5e550`
  (v4.0), which has it. Rebuilding the wheel therefore required checking out
  v4.0 first.
- gfortran does NOT preprocess lowercase `.f90` by default — the `#ifndef` in
  `sps_vars.f90` needs an explicit `-cpp` flag in the CMake build.
- To build `C3K_HR`: reset `libfsps` to v4.0 and add
  `target_compile_options(_fsps PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp;-DC3K_LR=0;-DC3K_HR=1>)`
  to `src/fsps/CMakeLists.txt`, then `pip wheel .` + install. `pip install -U fsps`
  restores the PyPI `c3k_lr` wheel.
- Integrated quantities (log Lbol, SSP mass) are identical between C3K_LR and
  C3K_HR — only the spectral sampling changes (1936 vs 10992 points; R=100 vs
  R=500 at 1.6 µm).
- `fcstar` is documented as "Currently has no effect" in the FSPS manual
  (`doc/sps.tex`); its only code reference is a commented-out dilution line in
  `ssp_gen.f90`. Verified empirically: `fcstar` = 0.0/0.3/0.5/1.0/1.5 all give
  identical spectra.
- Even if uncommented, `fcstar` only acts on C-rich TP-AGB stars
  (`ffco > 1.0`), and the MIST "Composition" column for `phase=5` is always
  < 1 (C/O = 0.02–0.34, all O-rich) at every metallicity checked. So the C-rich
  (Aringer 2009) TP-AGB branch never activates for MIST — `use_lw_tpagb` only
  swaps the O-rich template (C3K grid ↔ LW02 `Orich.spec`).
- `agb` and `pagb` are pure **multiplicative weights on the IMF weight**
  (`mod_gb.f90`: `wght(i) = wght(i)*agb` for phase=5, `*pagb` for phase=6) —
  they do NOT change stellar logL/logT. Hence the SSP spectrum is **exactly
  linear** in each knob (verified to ~1e-15).
- `agb`: TP-AGB supplies ~30% of the NIR flux at 1.6 µm (0.5% at 400 nm, 35% at
  3 µm). `pagb`: post-AGB contributes <0.5% everywhere (peaks in the UV), so it
  is negligible in practice — as the flat `response_pagb.png` shows.

## 2026-09-24
- C3K_HR native resolution is R=3000 (sigma 42.4 km/s) only over 3000–10000 A;
  from 1 to 2.5 micron it is R=500 (sigma 254.6 km/s) with 16 A pixels, per
  `SPECTRA/C3K/c3k_hr/readme.md` and `c3k_hr.res` (exposed as `sp.resolutions`).
  A "sigma = 300 km/s" NIR product therefore adds only 159 km/s in quadrature.
- The C3K_HR loader reads `nzinit = 11` metallicity files although 13 exist
  (`sps_vars.f90` c3k_hr block, `sps_setup.f90` read loop with `dz` clipped to
  [0, 1]). MIST +0.25 and +0.50 SSPs silently use solar-[Fe/H] spectra.
- No built-in FSPS SFH gives an exponential decline after truncation:
  `sf_trunc` is a hard cut (sfh=4) and `sf_slope` gives a linear ramp (sfh=5).
  Use tabular SFH (`sfh=3`), which interpolates the SFR linearly between nodes
  and is normalized to absolute Msun, not per Msun formed. Set `tage` on a node.
- python-fsps blocks `zcontinuous = 3` (time-varying Z with tabular SFH) with
  an assertion, and `zcontinuous = 2` (MDF) returns zeros because the summation
  loop in `ztinterp.f90` is commented out.
- python-fsps does not expose FSPS's Lick indices (`getindx` only runs for
  `write_compsp = 4`). FSPS's own HdeltaA feature band is 4083.50–4122.25 A and
  Lick bands are converted air -> vacuum (Morton 1991) in `sps_setup.f90`.
- Timings (C3K_HR, this laptop): one SSP metallicity build ~11 s; changing
  `agb`, IMF or `set_lsf` invalidates every cached SSP (~23 s for a
  zcontinuous=1 pair); CSP-only changes (tage, tabular SFH, `sigma_smooth`)
  cost milliseconds. `sigma_smooth` only acts inside
  [`min_wave_smooth`, `max_wave_smooth`] = [1e3, 1e4] A by default.
