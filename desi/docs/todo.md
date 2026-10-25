# TODO

## Phase 1: learn Astro Data Lab (2026-10-25)
- [x] Read the Data Lab docs, API reference and the DESI/crossmatch example notebooks
- [x] Set up the `desi/` uv project (Python 3.11, `astro-datalab` 2.24.0)
- [x] Log in from the environment variables; run a first SQL query
- [x] Save table sizes and column descriptions (`scripts/describe_tables.py`)
- [x] Count LS DR9/DR10/DR11 and DESI DR1 objects in COSMOS (`scripts/explore_cosmos.py`)
- [x] Check the join keys: `photometry.ls_id` to DR9, uniqueness of (targetid, survey,
      program) in the VAC, several TARGETIDs per LS object
- [x] Prototype the three-way join and validate it on a 0.1 x 0.1 deg patch
      (`scripts/build_cosmos_catalog.py`)
- [x] Exercise the async query path and a MyDB round trip
- [x] Write `docs/datalab_guide.md`

## Phase 2: full COSMOS catalog
- [x] Choose the LS release: DR9 (user, 2026-10-25)
- [ ] Confirm or change the footprint, duplicate rule and VAC columns (defaults in SPEC.md)
- [x] Run `build_cosmos_catalog.py --release dr9 --full` (sync; async rejects the long SQL)
      into `/Users/shuang/work/desi/data/datalab/cosmos_ls_dr9_desi_dr1.fits`
- [x] Validate the full output (`scripts/validate_cosmos_catalog.py`, 5 of 5 checks pass)
- [ ] Quick-look figures: redshift distribution, stellar mass versus redshift, sky map

## Review (Phase 1)
- SQL access works with the registered account: sync queries, async jobs and MyDB were all
  run. Aggregates over the whole COSMOS box take 2 to 4 s, and the full three-way join on a
  0.01 deg^2 patch takes about 3 s.
- COSMOS box: 170 373 / 552 055 / 243 889 LS DR9 / DR10 / DR11 sources; 35 720 DESI DR1
  `zpix` rows, 25 554 unique primary with `zwarn == 0`; 25 257 VAC rows.
- Test-patch outputs match an independent re-query of `zpix` and the VAC exactly, and DR9 (ID
  join), DR10 and DR11 (positional) pick the same DESI spectra.
- Not yet understood: why DR11 holds fewer COSMOS sources than DR10.

## Review (Phase 2, DR9 run)
- One sync query, 18.5 s: 170 373 LS DR9 sources, 23 669 with a DESI primary spectrum (22 217
  ZWARN == 0: 18 133 galaxies, 635 QSOs, 3 449 stars), 18 133 with stellar-mass/emission-line
  values, 2 675 collapsed from several TARGETIDs. Summary in
  `results/cosmos/cosmos_ls_dr9_summary.csv`.
- Validation: row count, unique `ls_id`, matched and collapsed counts against a fresh server
  aggregate, VAC coverage exactly on GALAXY & ZWARN == 0, and 300 random rows identical to a
  direct re-query.
- The first `--full` attempt used async and hit the 502 URL-length limit; the script now runs
  sync and `run_async_query` guards the limit.
