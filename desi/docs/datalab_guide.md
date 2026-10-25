# Astro Data Lab for DESI DR1 and Legacy Surveys: a working guide

Notes from learning NOIRLab Astro Data Lab (2026-10-25). Statements marked **(verified)** were
run in this folder; the rest come from the Data Lab documentation and its official example
notebooks ([astro-datalab/notebooks-latest](https://github.com/astro-datalab/notebooks-latest),
especially `04_HowTos/QueryClient/How_to_query_DESI_DR1_Data.ipynb` and
`04_HowTos/CrossmatchTables/How_to_crossmatch_tables.ipynb`). The manual pages under
`docs/manual/` are mostly tables of contents; the notebooks are the better reference.

## 1. Ways in

| Route | What it is | Used here |
|---|---|---|
| Web portal, [datalab.noirlab.edu](https://datalab.noirlab.edu) | Data Explorer (table browser; **bold** columns are indexed) and a query form | for browsing |
| Hosted JupyterLab | notebooks next to the database, DESI software preinstalled | no |
| Python client `astro-datalab` (module `dl`) | `authClient`, `queryClient`, `storeClient` from any machine | **yes** |
| TAP, `https://datalab.noirlab.edu/tap` | IVOA TAP/ADQL for TOPCAT or `pyvo`; works anonymously **(verified)** | no |
| SPARCL | spectra retrieval service | not yet |

`astro-datalab` 2.24.0 declares `requires_python <3.12` and pins old libraries (numpy < 1.26.4,
pandas < 2.2, astropy < 5.4, scipy < 1.12, pycurl). That is why this folder has its own uv
environment on Python 3.11 (`uv sync`).

## 2. Logging in

```python
from dl import authClient, queryClient
token = authClient.login(user, password)   # returns an error *string* on failure, no exception
authClient.isValidToken(token)             # check this instead of try/except
authClient.whoAmI()
```

- **(verified)** `login()` caches the session in `~/.datalab/dl.conf` and
  `~/.datalab/id_token.<user>`. Every later `queryClient` call, in any Python process, picks
  the token up automatically (`dl.Util.def_token`), so log in once and reuse it.
  `scripts/datalab_utils.py::login_from_environment()` does this with `ASTRO_DATALAB_USER` and
  `ASTRO_DATALAB_PASS` and only logs in again when the cached token is no longer valid.
- **(verified)** Both files are written world-readable (mode 644); `chmod 600 ~/.datalab/*`.
  The token sits in plain text in `dl.conf`, so never `cat` that file in a shared log.
- **(verified)** `authClient.logout(token)` invalidates the token on the server.
- Anonymous access is enough for SELECT queries on public tables. Saving results to MyDB or
  to virtual storage (VOSpace) needs a registered login.

## 3. Running queries

```python
df = queryClient.query(sql="SELECT ...", fmt="pandas", timeout=600)          # synchronous
df = queryClient.query(sql="SELECT ...", fmt="pandas", async_=True,
                       wait=True, poll=10, timeout=3600)                      # asynchronous
```

- `fmt`: `csv` (default), `csv-noheader`, `ascii`, `pandas`, `array`, `structarray`, `table`
  (astropy); `fits` and `votable` only when saving to virtual storage.
- Timeouts: sync default 300 s, maximum 600 s; async up to 86 400 s (24 h). The service can
  refuse a sync query whose projected run time is too long.
- **(verified)** With `async_=True, wait=True` the call polls and returns the result itself,
  here a DataFrame. Without `wait`, it returns a job ID for `queryClient.status(jobId)`,
  `queryClient.results(jobId)`, `queryClient.error(jobId)` and `queryClient.abort(jobId)`.
  `queryClient.jobs()` lists the last 30 days of jobs.
- **(verified)** The client sends the SQL inside the URL of a GET request. **Async submissions
  fail with `502 Bad Gateway` once the URL-encoded SQL exceeds about 1030 characters**
  (bisected: 1027 works, 1035 fails); sync queries of 6200 encoded characters work. For a long
  query, run it sync if it finishes within 600 s, or save it server-side with
  `out="mydb://name"` and fetch the short `SELECT * FROM mydb://name` asynchronously.
  `run_async_query` in `scripts/datalab_utils.py` refuses SQL over the limit.
- **(verified)** MyDB, a personal Postgres schema on the server:
  `query(sql=..., out="mydb://name", drop=True)` saves a result, SQL reads it back as
  `FROM mydb://name`, `queryClient.mydb_list()` lists tables (the string `'No tables'` when
  empty) and `queryClient.mydb_drop(name)` removes one. `mydb_import(name, dataframe_or_csv)`
  uploads a local table, and `mydb_index(name, column, q3c="ra,dec", cluster=True)` adds a
  spatial index. Use it to stage a long query in steps, or to crossmatch your own catalog
  against a Data Lab table.
- Use the PostgreSQL dialect (`sql=`), not ADQL. **(verified)** It supports Q3C, CTEs (`WITH`),
  `LATERAL`, `DISTINCT ON`, window functions and `::` casts.

### Table metadata, without the web page

```sql
SELECT schema, tbl_name, nrows FROM tbl_stat WHERE schema = 'desi_dr1';
SELECT column_name, datatype, indexed, unit, description
FROM tap_schema.columns WHERE table_name = 'desi_dr1.zpix';
```

`queryClient.schema()` and `queryClient.list()` are deprecated. `tbl_stat.nrows` is a planner
estimate, a little off from the counts on the data pages. `scripts/describe_tables.py` saves
both views to `results/schema/`.

## 4. Spatial queries with Q3C

| Function | Meaning |
|---|---|
| `q3c_radial_query(ra, dec, ra0, dec0, radius_deg)` | cone search |
| `q3c_poly_query(ra, dec, ARRAY[ra1, dec1, ra2, dec2, ...])` | polygon or box search |
| `q3c_join(ra_small, dec_small, ra_big, dec_big, radius_deg)` | all pairs within a radius |
| `q3c_dist(ra1, dec1, ra2, dec2)` | separation in **degrees** |

- Every radius is in degrees. In `q3c_join` the small table goes first and the big indexed
  table second, or the query crawls.
- Nearest neighbour only (**verified**, 2 s for 24 661 DESI targets against LS DR10):

  ```sql
  SELECT p.targetid, m.ls_id, m.sep_arcsec
  FROM desi_dr1.photometry AS p
  LEFT JOIN LATERAL (
      SELECT t.ls_id, q3c_dist(p.ra, p.dec, t.ra, t.dec) * 3600 AS sep_arcsec
      FROM ls_dr10.tractor AS t
      WHERE q3c_join(p.ra, p.dec, t.ra, t.dec, 1.0 / 3600)
      ORDER BY q3c_dist(p.ra, p.dec, t.ra, t.dec) ASC LIMIT 1
  ) AS m ON true
  WHERE q3c_poly_query(p.ra, p.dec, ARRAY[...]);
  ```

  `INNER JOIN LATERAL` drops the unmatched rows.
- Pre-computed crossmatch tables `<schema>.x1p5__<table1>__<schema2>__<table2>`: single
  nearest neighbour within 1.5 arcsec, columns `id1, ra1, dec1, id2, ra2, dec2, distance`
  (arcsec). `desi_dr1` has them for `zpix` against AllWISE, Gaia DR3, NSC DR2, SDSS DR17
  specObj and unWISE. **There is no `zpix` x Legacy Surveys crossmatch table.**
  `desi_dr1.photometry` plays that role (section 5).
- Speed **(verified)**: a grouped count over the 1.96 deg^2 COSMOS box takes 2 s on
  `ls_dr9.tractor` (2.0 billion rows) and 4 s on `ls_dr10.tractor` (3.1 billion rows).
- Data Lab tips: select only the columns you need (no `SELECT *`), filter on indexed columns,
  do not wrap an indexed column in a function (`CAST(col AS ...)` disables the index), drop
  `ORDER BY` when the order does not matter, and use `random_id` (0 to 100) for random subsets.

## 5. DESI DR1 (`desi_dr1`)

Core tables: `zpix` (23.1 M rows, redshifts from HEALPix coadds; the one to use), `ztile`
(per-tile coadds), `photometry` (76.0 M, LS DR9 Tractor photometry for every TARGETID),
`target` (targeting bits per tile), `fiberassign`, `potential`, `tile`, `exposure`, `frame`.
VACs: `stellar_mass_emline` (14.7 M), `emfit` (emission lines, z <= 0.45), `agngal` (AGN/galaxy
diagnostics), `mws` (Milky Way Survey). Column docs:
[data.desi.lbl.gov/doc/access/database](https://data.desi.lbl.gov/doc/access/database/).

`zpix`
- One row per (`targetid`, `survey`, `program`). `zcat_primary` picks one row per TARGETID.
  "Useful" spectra are `zcat_primary AND zwarn = 0` (the database already excludes sky fibers).
  Values: `survey` in {cmx, sv1, sv2, sv3, special, main}, `program` in {dark, bright,
  backup, other}, `spectype` in {GALAXY, QSO, STAR}.
- **(verified)** No `target_ra`/`target_dec`, only `mean_fiber_ra`/`mean_fiber_dec`. Those can
  sit far from the target: up to 75 arcsec in a COSMOS `special/other` spectrum. For
  positions, join `desi_dr1.photometry ON targetid` and use `p.ra, p.dec`.
- Targeting bitmasks (`desi_target`, `bgs_target`, `sv3_desi_target`, ...) are copied into
  `zpix`, so a join to `target` is rarely needed.

`photometry`
- **(verified)** `photometry.ls_id` equals `ls_dr9.tractor.ls_id` for every DESI target with
  `release > 0` (24 661 of 24 661 in COSMOS). Targets without Legacy Surveys photometry (for
  example secondary programs) have `release = 0` (892 in COSMOS). The same `ls_id` matches
  nothing in DR10 or DR11 (section 6).
- **(verified)** One Legacy Surveys object can carry several TARGETIDs: different surveys and
  programs, secondary-target IDs, the backup/Gaia ID range (`targetid >= 2^61`). Each TARGETID
  has its own primary spectrum. In COSMOS, 2 675 of 23 669 matched LS DR9 objects have two or
  three. Their redshifts agree (median |dz| = 1e-5), but 30 differ by more than 0.01, so any
  "one row per photometric object" catalog needs a rule for choosing among them.

`stellar_mass_emline`
- Galaxies with `SPECTYPE == GALAXY` and `ZWARN == 0`. **(verified)** One row per (`targetid`,
  `survey`, `program`), the same key as `zpix`. Non-primary spectra are included (5 205 of the
  25 257 COSMOS rows have `zcat_primary = false`), so join on all three columns and pick the
  spectrum you want in `zpix`.
- Has `target_ra`/`target_dec` (indexed), the CIGALE fit (`*_cg`: `mass_cg` and `sfr_cg` in
  linear M_sun and M_sun/yr, plus `age_cg`, `tau_cg`, `av_cg`, `z_cg`), the STARLIGHT fit
  (`*_sl`: `massini_sl`, `masscor_sl`, `age_sl`, `z_sl`, `av_sl`, `v0_sl`, `vd_sl`, and the
  45-element array `xj_sl`), `dn4000`, and 35 emission lines from Mg II 2796 to [S III] 9533,
  each with `_center`, `_flux`, `_sigma`, `_cont`, `_ew` and errors.
- **(verified)** `survey` and `program` are lower case (`main`, `dark`) in both `zpix` and the
  VAC, even though the VAC column description lists upper case.

## 6. Legacy Surveys (`ls_dr9`, `ls_dr10`, `ls_dr11`)

- DR9 is the DESI targeting release. DR10 combines new DR10-South DECam data (adding the
  i band) with DR9-North. DR11 adds more DECam imaging in the south.
- Object IDs: `ls_id` (DR9, DR10) and `ls_id_dr11` (DR11). They encode
  release, brickid and objid, and the release differs in COSMOS (9010, 10000, 11010), so
  **IDs do not carry over between releases**. Match by position instead.
- **(verified)** All COSMOS rows have `brick_primary = 1`, so these tables seem to hold only
  primary objects.
- Sources in the 1.96 deg^2 COSMOS box **(verified)**: DR9 170 373, DR10 552 055, DR11
  243 889. DR10 includes the deep DECam COSMOS imaging; why DR11 holds fewer is not yet
  understood.
- `photo_z` tables (DR9, DR10) are keyed by `ls_id`.

## 7. Gotchas met so far

1. **(verified)** Postgres booleans arrive as the strings `'t'`/`'f'` in `fmt="pandas"`, so
   `.sum()` concatenates them. Cast in SQL: `flag::int`.
2. **(verified)** A NULL in an integer column (any LEFT JOIN) turns it into float64 in pandas.
   TARGETID and DR9 `ls_id` exceed 2^53 and lose digits. Use `COALESCE(targetid, -1)` in SQL,
   or read `fmt="csv"` with explicit `Int64` dtypes.
3. A `LIMIT` on a join limits the output, not the scan. Test a heavy query on a small patch
   (0.1 x 0.1 deg here) before the full area.
4. `zpix` positions are fiber positions, not target positions (section 5).
5. **(verified)** Long SQL cannot go through async: 502 above about 1030 URL-encoded
   characters (section 3).

## 8. Recipe: COSMOS Legacy Surveys x DESI redshifts x stellar masses

`scripts/build_cosmos_catalog.py` builds the whole thing in one server-side query:

```sql
WITH desi AS (                    -- one primary DESI spectrum per Legacy Surveys object
    SELECT DISTINCT ON (p.ls_id) p.ls_id, z.targetid, z.survey, z.program, z.z, z.zwarn, ...,
           COUNT(*) OVER (PARTITION BY p.ls_id) AS n_desi_spectra
    FROM desi_dr1.photometry AS p
    JOIN desi_dr1.zpix AS z ON z.targetid = p.targetid AND z.zcat_primary
    WHERE q3c_poly_query(p.ra, p.dec, ARRAY[...]) AND p.release > 0
    ORDER BY p.ls_id, (z.zwarn = 0) DESC, z.tsnr2_lrg DESC)
SELECT t.ls_id, t.ra, t.dec, t.flux_g, ..., d.*, s.mass_cg, s.sfr_cg, s.halpha_flux, ...
FROM ls_dr9.tractor AS t
LEFT JOIN desi AS d ON d.ls_id = t.ls_id
LEFT JOIN desi_dr1.stellar_mass_emline AS s
       ON s.targetid = d.targetid AND s.survey = d.survey AND s.program = d.program
WHERE q3c_poly_query(t.ra, t.dec, ARRAY[...]);
```

For DR10 or DR11 the `p.ls_id` link becomes the `LATERAL` nearest-neighbour match of section 4.
On the 0.1 x 0.1 deg test patch the three releases pick the same 81 DESI spectra, and the
values match an independent re-query of `zpix` and the VAC exactly.

**(verified)** The full DR9 COSMOS box runs as one sync query in 18.5 s and returns 170 373
rows (176 MB FITS). 23 669 sources have a DESI primary spectrum (22 217 with ZWARN == 0) and
18 133 have stellar-mass/emission-line values. `scripts/validate_cosmos_catalog.py` checks it
against independent counts and a re-query of 300 random rows.

## 9. Acknowledgments to include in papers

Astro Data Lab text and citations (Fitzpatrick et al. 2014; Nikutta et al. 2020) are in the
DESI DR1 query notebook. DESI DR1 acknowledgments:
[data.desi.lbl.gov/doc/acknowledgments](https://data.desi.lbl.gov/doc/acknowledgments/).
Legacy Surveys acknowledgments: [legacysurvey.org/acknowledgment](https://www.legacysurvey.org/acknowledgment/).
