"""Check the full COSMOS catalog written by ``build_cosmos_catalog.py --full`` (DR9).

Each check compares the FITS file with an independent number: the Legacy Surveys count from
``explore_cosmos.py``, a fresh aggregate query on the server, or a re-query of ``zpix`` and the
VAC for random rows. Writes ``results/cosmos/cosmos_ls_dr9_summary.csv``.

    uv run python scripts/validate_cosmos_catalog.py
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from datalab_utils import (
    LARGE_DATA_DIRECTORY,
    RESULTS_DIRECTORY,
    login_from_environment,
    q3c_box_condition,
    run_query,
)

CATALOG_PATH = LARGE_DATA_DIRECTORY / "cosmos_ls_dr9_desi_dr1.fits"
SPOT_CHECK_SIZE = 300
SPOT_CHECK_COLUMNS = ["z", "zwarn", "mass_cg", "sfr_cg", "dn4000", "halpha_flux"]


def report(name: str, passed: bool, detail: str) -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def main() -> None:
    login_from_environment()
    table = Table.read(CATALOG_PATH)
    catalog = table.to_pandas()
    for column in ["survey", "program", "spectype", "subtype", "type"]:
        catalog[column] = catalog[column].str.decode("ascii").str.strip()
    has_spectrum = catalog["targetid"] > 0
    results = []

    counts = pd.read_csv(RESULTS_DIRECTORY / "cosmos" / "legacy_surveys_counts.csv")
    expected_rows = counts.loc[counts["table"] == "ls_dr9.tractor", "n_sources"].sum()
    results.append(
        report("row count", len(catalog) == expected_rows, f"{len(catalog)} vs {expected_rows}")
    )
    results.append(report("unique ls_id", catalog["ls_id"].is_unique, f"{len(catalog)} rows"))

    server = run_query(
        f"""SELECT COUNT(*) AS n_objects, SUM((n_targetid > 1)::int) AS n_several_targetid
            FROM (SELECT p.ls_id, COUNT(DISTINCT p.targetid) AS n_targetid
                  FROM desi_dr1.photometry AS p
                  JOIN desi_dr1.zpix AS z ON z.targetid = p.targetid AND z.zcat_primary
                  WHERE {q3c_box_condition("p.ra", "p.dec")} AND p.release > 0
                  GROUP BY p.ls_id) AS q"""
    ).iloc[0]
    n_several = int((catalog["n_desi_spectra"] > 1).sum())
    results.append(
        report(
            "LS objects with a DESI primary spectrum",
            has_spectrum.sum() == server["n_objects"] and n_several == server["n_several_targetid"],
            f"{has_spectrum.sum()} vs {server['n_objects']} on the server; several TARGETIDs "
            f"{n_several} vs {server['n_several_targetid']}",
        )
    )

    good_galaxy = has_spectrum & (catalog["spectype"] == "GALAXY") & (catalog["zwarn"] == 0)
    has_vac = catalog["has_stellar_mass_emline"] == 1
    results.append(
        report(
            "stellar_mass_emline exactly on GALAXY & ZWARN == 0",
            (good_galaxy == has_vac).all(),
            f"{good_galaxy.sum()} good galaxies, {has_vac.sum()} rows with VAC values",
        )
    )

    sample = catalog[has_spectrum].sample(SPOT_CHECK_SIZE, random_state=20261025)
    targetid_list = ", ".join(str(targetid) for targetid in sample["targetid"])
    truth = run_query(
        f"""SELECT z.targetid, z.z, z.zwarn, s.mass_cg, s.sfr_cg, s.dn4000, s.halpha_flux
            FROM desi_dr1.zpix AS z
            LEFT JOIN desi_dr1.stellar_mass_emline AS s
                ON s.targetid = z.targetid AND s.survey = z.survey AND s.program = z.program
            WHERE z.zcat_primary AND z.targetid IN ({targetid_list})"""
    )
    merged = sample.merge(truth, on="targetid", suffixes=("", "_truth"))
    identical = all(
        np.array_equal(
            merged[column].to_numpy(float), merged[f"{column}_truth"].to_numpy(float), True
        )
        for column in SPOT_CHECK_COLUMNS
    )
    results.append(
        report(
            "spot check against zpix and the VAC",
            len(merged) == SPOT_CHECK_SIZE and identical,
            f"{len(merged)} of {SPOT_CHECK_SIZE} random rows, {', '.join(SPOT_CHECK_COLUMNS)}",
        )
    )

    summary = (
        catalog[has_spectrum]
        .assign(zwarn_zero=lambda frame: frame["zwarn"] == 0)
        .groupby(["spectype", "zwarn_zero"])
        .agg(
            n_objects=("ls_id", "size"),
            n_with_stellar_mass_emline=("has_stellar_mass_emline", "sum"),
            n_several_targetid=("n_desi_spectra", lambda values: int((values > 1).sum())),
            median_z=("z", "median"),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIRECTORY / "cosmos" / "cosmos_ls_dr9_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(summary.to_string(index=False))
    print(f"{sum(results)} of {len(results)} checks passed; wrote {summary_path}")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
