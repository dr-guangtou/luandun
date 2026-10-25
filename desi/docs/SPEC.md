# SPEC: DESI DR1 x Legacy Surveys in COSMOS through Astro Data Lab

## Goal

1. All Legacy Surveys Tractor sources in the COSMOS field (about 2 deg^2).
2. Append DESI DR1 spectroscopic redshifts from `desi_dr1.zpix`.
3. Append the physical properties from the `desi_dr1.stellar_mass_emline` VAC (CIGALE and
   STARLIGHT fits, Dn4000, emission lines).

Phase 1 (learning Data Lab) is done. Phase 2: the full DR9 COSMOS catalog was built and
validated on 2026-10-25 with the defaults below; the remaining open decisions can change it.

## Architecture

- `scripts/datalab_utils.py`: login from `ASTRO_DATALAB_USER`/`ASTRO_DATALAB_PASS`, sync and
  async query wrappers, TAP_SCHEMA/`tbl_stat` helpers, the COSMOS box and a Q3C polygon
  condition. Paths: `RESULTS_DIRECTORY` (small outputs, in git) and `LARGE_DATA_DIRECTORY`
  = `/Users/shuang/work/desi/data/datalab` (anything over ~100 MB, outside the repo).
- `scripts/describe_tables.py`: table sizes and column descriptions to `results/schema/`.
- `scripts/explore_cosmos.py`: aggregate counts in COSMOS to `results/cosmos/*_counts.csv`.
- `scripts/build_cosmos_catalog.py`: the three-way join as one server-side SQL query, run
  sync (the SQL is too long for async); a test patch writes CSV to `results/cosmos/`, `--full`
  writes FITS to `LARGE_DATA_DIRECTORY`.
- `scripts/validate_cosmos_catalog.py`: checks the full DR9 FITS against independent counts
  and a re-query; writes `results/cosmos/cosmos_ls_dr9_summary.csv`.

## Current choices (defaults; open to change)

| Item | Default | Why |
|---|---|---|
| COSMOS footprint | 1.4 x 1.4 deg square on 10:00:28.6 +02:12:21.0 (1.96 deg^2) | Scoville et al. 2007 field definition |
| Legacy Surveys release | **DR9 (decided 2026-10-25)** | DESI targeting release; exact `ls_id` join through `desi_dr1.photometry` |
| DR10/DR11 link | nearest Tractor source within 1 arcsec of the DESI target position | IDs differ between releases |
| DESI rows | `zcat_primary` rows of `zpix`, all `zwarn` values kept (filter later) | one spectrum per TARGETID |
| Several TARGETIDs per LS object | keep one: `zwarn == 0` first, then highest `tsnr2_lrg`; record `n_desi_spectra`, `desi_z_spread` | 11 per cent of matched objects in COSMOS |
| Missing values | integers and strings get sentinels (`targetid = -1`, `zwarn = -1`, `''`); floats NaN | keeps 64-bit IDs exact |
| VAC columns | 24 physical-property columns + 13 strong lines x (flux, fluxerr, ew, ewerr); no `xj_sl` | 428 columns in total are too many by default |
| Output | test patch CSV in `results/cosmos/`; full field FITS in `LARGE_DATA_DIRECTORY` | size |

## Open decisions (ask the user)

1. ~~Legacy Surveys release~~: decided DR9 (2026-10-25), the release DESI targeted from. DR10
   (552 k sources, deeper DECam imaging plus i band) and DR11 stay available through
   `--release` for comparison.
2. Is the 1.4 x 1.4 deg square right, or should it follow a specific COSMOS footprint
   (COSMOS2020, UltraVISTA, HSC)?
3. Duplicate-spectrum rule and which DESI rows to keep (primary only, or all spectra in a
   separate long table).
4. Which VAC columns, and whether to add `emfit`, `agngal` or the LS `photo_z` table.
