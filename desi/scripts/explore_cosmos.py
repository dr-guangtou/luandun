"""Count Legacy Surveys sources and DESI DR1 spectra inside the COSMOS box.

Only aggregate queries run here, so nothing large is downloaded. Writes
``results/cosmos/legacy_surveys_counts.csv``, ``results/cosmos/desi_zpix_counts.csv`` and
``results/cosmos/desi_join_check.csv``.

    uv run python scripts/explore_cosmos.py
"""

import pandas as pd
from datalab_utils import (
    RESULTS_DIRECTORY,
    cosmos_box,
    login_from_environment,
    q3c_box_condition,
    run_query,
)

LEGACY_SURVEYS_TABLES = ["ls_dr9.tractor", "ls_dr10.tractor", "ls_dr11.tractor"]


def count_legacy_surveys() -> pd.DataFrame:
    rows = []
    for table in LEGACY_SURVEYS_TABLES:
        counts = run_query(
            f"""SELECT type, COUNT(*) AS n_sources,
                       SUM(brick_primary::int) AS n_brick_primary,
                       SUM((maskbits = 0)::int) AS n_maskbits_zero
                FROM {table}
                WHERE {q3c_box_condition("ra", "dec")}
                GROUP BY type ORDER BY type"""
        )
        counts.insert(0, "table", table)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def count_desi_zpix() -> pd.DataFrame:
    # zpix has no target coordinates, so select on the LS DR9 position in desi_dr1.photometry.
    return run_query(
        f"""SELECT z.survey, z.program, z.spectype,
                   COUNT(*) AS n_zpix,
                   SUM(z.zcat_primary::int) AS n_zcat_primary,
                   SUM((z.zcat_primary AND z.zwarn = 0)::int) AS n_primary_zwarn0,
                   COUNT(s.targetid) AS n_in_stellar_mass_emline
            FROM desi_dr1.zpix AS z
            JOIN desi_dr1.photometry AS p ON p.targetid = z.targetid
            LEFT JOIN desi_dr1.stellar_mass_emline AS s
                ON s.targetid = z.targetid AND s.survey = z.survey AND s.program = z.program
            WHERE {q3c_box_condition("p.ra", "p.dec")}
            GROUP BY z.survey, z.program, z.spectype
            ORDER BY z.survey, z.program, z.spectype"""
    )


def check_desi_join() -> pd.DataFrame:
    """How the unique DESI targets with a good primary redshift link back to LS DR9."""
    return run_query(
        f"""SELECT COUNT(*) AS n_primary_zwarn0,
                   SUM((p.release = 0)::int) AS n_without_legacy_surveys_photometry,
                   COUNT(t.ls_id) AS n_ls_id_in_ls_dr9_tractor,
                   COUNT(DISTINCT z.targetid) AS n_distinct_targetid
            FROM desi_dr1.zpix AS z
            JOIN desi_dr1.photometry AS p ON p.targetid = z.targetid
            LEFT JOIN ls_dr9.tractor AS t ON t.ls_id = p.ls_id
            WHERE {q3c_box_condition("p.ra", "p.dec")}
              AND z.zcat_primary AND z.zwarn = 0"""
    )


def main() -> None:
    login_from_environment()
    output_directory = RESULTS_DIRECTORY / "cosmos"
    output_directory.mkdir(parents=True, exist_ok=True)
    print("COSMOS box (ra_min, ra_max, dec_min, dec_max):", [round(v, 4) for v in cosmos_box()])

    legacy_surveys = count_legacy_surveys()
    legacy_surveys.to_csv(output_directory / "legacy_surveys_counts.csv", index=False)
    print(legacy_surveys.to_string(index=False))
    print(legacy_surveys.groupby("table")[["n_sources", "n_brick_primary"]].sum())

    zpix = count_desi_zpix()
    zpix.to_csv(output_directory / "desi_zpix_counts.csv", index=False)
    print(zpix.to_string(index=False))
    print(zpix[["n_zpix", "n_zcat_primary", "n_primary_zwarn0", "n_in_stellar_mass_emline"]].sum())

    join_check = check_desi_join()
    join_check.to_csv(output_directory / "desi_join_check.csv", index=False)
    print(join_check.to_string(index=False))


if __name__ == "__main__":
    main()
