"""Shared helpers for querying NOIRLab Astro Data Lab from this sandbox.

Credentials come from the ``ASTRO_DATALAB_USER`` and ``ASTRO_DATALAB_PASS`` environment
variables. ``authClient.login`` caches the session token in ``~/.datalab/``, and every later
``queryClient`` call (in any process) picks it up automatically, so a login is only needed when
that cached token is missing or expired. Never print the token.
"""

import math
import os
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dl import authClient, queryClient
from dl.Util import def_token

DESI_DIRECTORY = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = DESI_DIRECTORY / "results"
# Catalogs larger than ~100 MB go here, outside the git repository.
LARGE_DATA_DIRECTORY = Path("/Users/shuang/work/desi/data/datalab")

# COSMOS: 1.4 x 1.4 deg square (about 1.96 deg^2) centred on the Scoville et al. (2007) field
# centre, J2000 10:00:28.6 +02:12:21.0.
COSMOS_CENTER_RA = 150.119167
COSMOS_CENTER_DEC = 2.205833
COSMOS_HALF_SIDE_DEG = 0.7

# The client sends the SQL in the URL of a GET request. Async submissions return
# "502 Bad Gateway" once the URL-encoded SQL exceeds about 1030 characters (bisected on
# 2026-10-25); sync queries of 6200 encoded characters still work.
ASYNC_SQL_MAX_ENCODED_LENGTH = 1000


def cosmos_box(half_side_deg: float = COSMOS_HALF_SIDE_DEG) -> tuple[float, float, float, float]:
    """Return (ra_min, ra_max, dec_min, dec_max) of a square on the sky centred on COSMOS."""
    half_width_ra = half_side_deg / math.cos(math.radians(COSMOS_CENTER_DEC))
    return (
        COSMOS_CENTER_RA - half_width_ra,
        COSMOS_CENTER_RA + half_width_ra,
        COSMOS_CENTER_DEC - half_side_deg,
        COSMOS_CENTER_DEC + half_side_deg,
    )


def q3c_box_condition(ra_column: str, dec_column: str, box: tuple[float, ...] | None = None) -> str:
    """SQL condition selecting rows inside a RA/Dec box through the Q3C polygon index."""
    ra_min, ra_max, dec_min, dec_max = box if box is not None else cosmos_box()
    corners = [ra_min, dec_min, ra_max, dec_min, ra_max, dec_max, ra_min, dec_max]
    polygon = ", ".join(f"{value:.6f}" for value in corners)
    return f"q3c_poly_query({ra_column}, {dec_column}, ARRAY[{polygon}])"


def login_from_environment(force: bool = False) -> str:
    """Make sure a valid Data Lab session exists and return the user name."""
    user = os.environ["ASTRO_DATALAB_USER"]
    cached_token = def_token(None)
    if (
        not force
        and authClient.isValidToken(cached_token)
        and authClient.isTokenLoggedIn(cached_token)
    ):
        return user
    token = authClient.login(user, os.environ["ASTRO_DATALAB_PASS"])
    if not authClient.isValidToken(token):
        # On failure login() returns an error message instead of a token.
        raise RuntimeError(f"Data Lab login failed for {user}: {token}")
    return user


def run_query(sql: str, timeout: int = 600, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Run a synchronous SQL query and return a DataFrame, printing the wall-clock time."""
    start = time.perf_counter()
    result = queryClient.query(sql=sql, fmt="pandas", timeout=timeout, **kwargs)
    if verbose:
        print(f"[{time.perf_counter() - start:6.1f} s] {len(result):>9d} rows <- {_one_line(sql)}")
    return result


def run_async_query(sql: str, timeout: int = 3600, poll: int = 10, **kwargs) -> pd.DataFrame:
    """Run an asynchronous SQL query (up to 24 h on the server) and return a DataFrame."""
    encoded_length = len(quote_plus(sql))
    if encoded_length > ASYNC_SQL_MAX_ENCODED_LENGTH:
        raise ValueError(
            f"SQL is {encoded_length} characters once URL-encoded; async submissions longer than "
            "about 1030 fail with '502 Bad Gateway'. Run it with run_query (sync, up to 600 s), "
            "or save it server-side with queryClient.query(sql=..., out='mydb://name') and fetch "
            "'SELECT * FROM mydb://name' asynchronously."
        )
    start = time.perf_counter()
    result = queryClient.query(
        sql=sql, fmt="pandas", async_=True, wait=True, poll=poll, timeout=timeout, **kwargs
    )
    print(f"[{time.perf_counter() - start:6.1f} s] {len(result):>9d} rows <- {_one_line(sql)}")
    return result


def describe_table(table: str) -> pd.DataFrame:
    """Column names, types, index flags, units and descriptions from TAP_SCHEMA."""
    sql = (
        "SELECT column_name, datatype, size, indexed, principal, unit, description "
        f"FROM tap_schema.columns WHERE table_name = '{table}'"
    )
    return run_query(sql, verbose=False)


def table_sizes(schemas: list[str]) -> pd.DataFrame:
    """Row counts of every table in the given schemas, from the Data Lab tbl_stat table."""
    schema_list = ", ".join(f"'{schema}'" for schema in schemas)
    sql = (
        "SELECT schema, tbl_name, nrows FROM tbl_stat "
        f"WHERE schema IN ({schema_list}) ORDER BY schema, tbl_name"
    )
    return run_query(sql, verbose=False)


def _one_line(sql: str, width: int = 90) -> str:
    text = " ".join(sql.split())
    return text if len(text) <= width else text[: width - 3] + "..."
