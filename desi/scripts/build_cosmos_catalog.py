"""Legacy Surveys sources in COSMOS with DESI DR1 redshifts and stellar-mass/emission-line values.

One server-side SQL query per run:

1. every Legacy Surveys Tractor source inside the box;
2. LEFT JOIN the DESI DR1 primary redshifts (``desi_dr1.zpix`` with ``zcat_primary``), linked to
   Legacy Surveys through ``desi_dr1.photometry`` -- by ``ls_id`` for DR9 (the DESI targeting
   release), or by the nearest Tractor source within ``--match-radius`` for DR10 and DR11, whose
   object IDs differ from DR9;
3. LEFT JOIN ``desi_dr1.stellar_mass_emline`` on (targetid, survey, program).

About 11 per cent of the DESI-matched Legacy Surveys objects in COSMOS carry two or three
TARGETIDs, each with its own primary spectrum. One spectrum is kept per Legacy Surveys object:
ZWARN == 0 first, then the highest TSNR2_LRG. ``n_desi_spectra`` and ``desi_z_spread`` record
what was collapsed.

    uv run python scripts/build_cosmos_catalog.py --release dr9               # 0.1 x 0.1 deg test
    uv run python scripts/build_cosmos_catalog.py --release dr9 --full        # whole COSMOS box
"""

import argparse
import math

from astropy.table import Table
from datalab_utils import (
    COSMOS_CENTER_DEC,
    COSMOS_CENTER_RA,
    COSMOS_HALF_SIDE_DEG,
    LARGE_DATA_DIRECTORY,
    RESULTS_DIRECTORY,
    cosmos_box,
    login_from_environment,
    q3c_box_condition,
    run_query,
)

LEGACY_SURVEYS_TABLES = {
    "dr9": ("ls_dr9.tractor", "ls_id", ["g", "r", "z"]),
    "dr10": ("ls_dr10.tractor", "ls_id", ["g", "r", "i", "z"]),
    "dr11": ("ls_dr11.tractor", "ls_id_dr11", ["g", "r", "i", "z"]),
}
WISE_BANDS = ["w1", "w2"]

ZPIX_COLUMNS = [
    "survey", "program", "spectype", "subtype", "z", "zerr", "zwarn", "deltachi2",
    "tsnr2_lrg", "coadd_numexp", "coadd_exptime", "desi_target", "bgs_target",
]  # fmt: skip
STELLAR_MASS_COLUMNS = [
    "mass_cg", "masserr_cg", "sfr_cg", "sfrerr_cg", "age_cg", "ageerr_cg", "tau_cg",
    "tauerr_cg", "av_cg", "averr_cg", "z_cg", "zerr_cg", "massini_sl", "masscor_sl",
    "age_sl", "z_sl", "av_sl", "v0_sl", "vd_sl", "snr_sl", "dn4000", "dn4000_err",
    "snr_med", "flux_scale",
]  # fmt: skip
EMISSION_LINES = [
    "oii3726", "oii3729", "hdelta", "hgamma", "hbeta", "oiii4959", "oiii5007",
    "oi6300", "nii6548", "halpha", "nii6583", "sii6716", "sii6731",
]  # fmt: skip
EMISSION_LINE_QUANTITIES = ["flux", "fluxerr", "ew", "ewerr"]


def legacy_surveys_columns(id_column: str, optical_bands: list[str]) -> list[str]:
    columns = [f"t.{id_column} AS ls_id", "t.release", "t.brickid", "t.objid", "t.ra", "t.dec"]
    columns += ["t.type", "t.ebv", "t.maskbits", "t.fitbits", "t.ref_cat"]
    columns += ["t.shape_r", "t.shape_e1", "t.shape_e2", "t.sersic"]
    for band in optical_bands + WISE_BANDS:
        columns += [f"t.flux_{band}", f"t.flux_ivar_{band}", f"t.mw_transmission_{band}"]
        columns += [f"t.nobs_{band}"]
    columns += [f"t.fiberflux_{band}" for band in optical_bands]
    return columns


def desi_match_sql(release: str, box_condition: str, match_radius_arcsec: float) -> str:
    """One DESI primary spectrum per Legacy Surveys object, keyed by the object's ls_id."""
    if release == "dr9":
        match = "CROSS JOIN LATERAL (SELECT p.ls_id, 0.0::double precision AS sep_arcsec) AS m"
        release_condition = "AND p.release > 0"
    else:
        table, id_column, _ = LEGACY_SURVEYS_TABLES[release]
        match = f"""JOIN LATERAL (
                SELECT t.{id_column} AS ls_id,
                       q3c_dist(p.ra, p.dec, t.ra, t.dec) * 3600 AS sep_arcsec
                FROM {table} AS t
                WHERE q3c_join(p.ra, p.dec, t.ra, t.dec, {match_radius_arcsec / 3600:.8f})
                ORDER BY q3c_dist(p.ra, p.dec, t.ra, t.dec) ASC LIMIT 1
            ) AS m ON true"""
        release_condition = ""
    zpix_columns = ", ".join(f"z.{column}" for column in ZPIX_COLUMNS)
    return f"""
        SELECT DISTINCT ON (m.ls_id)
               m.ls_id, m.sep_arcsec AS desi_sep_arcsec, z.targetid, {zpix_columns},
               COUNT(*) OVER (PARTITION BY m.ls_id) AS n_desi_spectra,
               MAX(z.z) OVER (PARTITION BY m.ls_id) - MIN(z.z) OVER (PARTITION BY m.ls_id)
                   AS desi_z_spread
        FROM desi_dr1.photometry AS p
        JOIN desi_dr1.zpix AS z ON z.targetid = p.targetid AND z.zcat_primary
        {match}
        WHERE {box_condition} {release_condition}
        ORDER BY m.ls_id, (z.zwarn = 0) DESC, z.tsnr2_lrg DESC"""


def catalog_sql(release: str, box: tuple[float, ...], match_radius_arcsec: float) -> str:
    table, id_column, optical_bands = LEGACY_SURVEYS_TABLES[release]
    # NULL integers would turn into float64 in pandas and lose TARGETID precision (> 2^53),
    # and NULL strings break FITS output, so both get sentinels.
    desi_columns = ["COALESCE(d.targetid, -1) AS targetid"]
    for column in ZPIX_COLUMNS:
        if column in ("survey", "program", "spectype", "subtype"):
            desi_columns.append(f"COALESCE(d.{column}, '') AS {column}")
        elif column in ("zwarn", "coadd_numexp", "desi_target", "bgs_target"):
            desi_columns.append(f"COALESCE(d.{column}, -1) AS {column}")
        else:
            desi_columns.append(f"d.{column}")
    desi_columns += ["COALESCE(d.n_desi_spectra, 0) AS n_desi_spectra", "d.desi_z_spread"]
    desi_columns += [
        "d.desi_sep_arcsec",
        "(s.targetid IS NOT NULL)::int AS has_stellar_mass_emline",
    ]
    vac_columns = [f"s.{column}" for column in STELLAR_MASS_COLUMNS]
    vac_columns += [
        f"s.{line}_{quantity}" for line in EMISSION_LINES for quantity in EMISSION_LINE_QUANTITIES
    ]
    select_list = ",\n               ".join(
        legacy_surveys_columns(id_column, optical_bands) + desi_columns + vac_columns
    )
    desi_sql = desi_match_sql(release, q3c_box_condition("p.ra", "p.dec", box), match_radius_arcsec)
    return f"""
        WITH desi AS ({desi_sql})
        SELECT {select_list}
        FROM {table} AS t
        LEFT JOIN desi AS d ON d.ls_id = t.{id_column}
        LEFT JOIN desi_dr1.stellar_mass_emline AS s
            ON s.targetid = d.targetid AND s.survey = d.survey AND s.program = d.program
        WHERE {q3c_box_condition("t.ra", "t.dec", box)}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", choices=sorted(LEGACY_SURVEYS_TABLES), default="dr9")
    parser.add_argument("--full", action="store_true", help="whole COSMOS box")
    parser.add_argument("--half-side", type=float, default=0.05, help="test patch, deg")
    parser.add_argument("--match-radius", type=float, default=1.0, help="DR10/DR11, arcsec")
    parser.add_argument("--print-sql", action="store_true")
    arguments = parser.parse_args()

    if arguments.full:
        box = cosmos_box(COSMOS_HALF_SIDE_DEG)
        output_path = LARGE_DATA_DIRECTORY / f"cosmos_ls_{arguments.release}_desi_dr1.fits"
    else:
        half_width_ra = arguments.half_side / math.cos(math.radians(COSMOS_CENTER_DEC))
        box = (
            COSMOS_CENTER_RA - half_width_ra,
            COSMOS_CENTER_RA + half_width_ra,
            COSMOS_CENTER_DEC - arguments.half_side,
            COSMOS_CENTER_DEC + arguments.half_side,
        )
        output_path = RESULTS_DIRECTORY / "cosmos" / f"patch_ls_{arguments.release}_desi_dr1.csv"

    sql = catalog_sql(arguments.release, box, arguments.match_radius)
    if arguments.print_sql:
        print(sql)
    login_from_environment()
    # Sync on purpose: the SQL is too long for an async submission (see datalab_utils).
    catalog = run_query(sql)

    if catalog["ls_id"].duplicated().any():
        raise RuntimeError("ls_id is not unique in the output; the DESI de-duplication failed")
    has_spectrum = catalog["targetid"] > 0
    good_redshift = has_spectrum & (catalog["zwarn"] == 0)
    print(
        f"{len(catalog)} Legacy Surveys {arguments.release} sources; "
        f"{has_spectrum.sum()} with a DESI DR1 primary spectrum "
        f"({good_redshift.sum()} ZWARN == 0); "
        f"{catalog['has_stellar_mass_emline'].sum()} with stellar_mass_emline values; "
        f"{(catalog['n_desi_spectra'] > 1).sum()} collapsed from several TARGETIDs"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".fits":
        Table.from_pandas(catalog).write(output_path, overwrite=True)
    else:
        catalog.to_csv(output_path, index=False)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
