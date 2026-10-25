# Lessons

## 2026-10-25 (first Data Lab session)
- `astro-datalab` 2.24.0 needs Python < 3.12 and pins numpy < 1.26.4, pandas < 2.2 and
  astropy < 5.4, so the folder has its own uv environment on Python 3.11 rather than sharing
  a newer one.
- The Data Lab manual pages are mostly tables of contents (several deep links return 404).
  The example notebooks in `astro-datalab/notebooks-latest` are the real reference.
- `authClient.login` stores the token in plain text in `~/.datalab/dl.conf` (mode 644). I
  printed that file while inspecting it and leaked the token into the session log; the fix was
  `authClient.logout(old_token)` (verified invalid afterwards), a fresh login and
  `chmod 600 ~/.datalab/*`. Never `cat` `dl.conf` or the `id_token.*` file.
- Postgres booleans come back as `'t'`/`'f'` strings with `fmt="pandas"`; summing them
  produced a long string instead of a count. Cast to int in SQL.
- LEFT JOINs leave NULLs in integer columns, which pandas turns into float64. TARGETID and DR9
  `ls_id` exceed 2^53 and would lose digits, so COALESCE them to -1 in SQL.
- `desi_dr1.photometry.ls_id` is the DR9 `ls_id`. It matches nothing in `ls_dr10.tractor`
  (different `release` code), so DR10/DR11 need a positional match.
- `zpix` has only fiber positions (`mean_fiber_ra/dec`), which can be tens of arcsec off the
  target; use `photometry.ra/dec` for target positions.
- One LS object can have several DESI TARGETIDs, each with its own primary spectrum (11 per
  cent of matched objects in COSMOS). Check uniqueness of the join key before any LEFT JOIN
  that should keep one row per object, and assert it on the output.
- Async submissions return an nginx `502 Bad Gateway` page when the URL-encoded SQL exceeds
  about 1030 characters, because the client puts the SQL in a GET URL. The 3.7 KB catalog query
  failed async, worked sync. I first suspected a service outage; a bisection with a padded
  `IN (...)` list found the limit in a few calls. Isolate a failure with the smallest change
  (same SQL sync vs async, short vs long) before guessing at the cause.
