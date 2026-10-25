"""Save the row counts and column descriptions of the DESI DR1 and Legacy Surveys tables.

Writes ``results/schema/table_sizes.csv`` and one ``results/schema/<schema>.<table>.csv`` per
table, so the column lists can be read offline instead of through the Data Explorer.

    uv run python scripts/describe_tables.py
"""

from datalab_utils import RESULTS_DIRECTORY, describe_table, login_from_environment, table_sizes

SCHEMAS = ["desi_dr1", "ls_dr9", "ls_dr10", "ls_dr11"]
TABLES = [
    "desi_dr1.zpix",
    "desi_dr1.ztile",
    "desi_dr1.photometry",
    "desi_dr1.target",
    "desi_dr1.stellar_mass_emline",
    "desi_dr1.emfit",
    "desi_dr1.agngal",
    "ls_dr9.tractor",
    "ls_dr9.photo_z",
    "ls_dr10.tractor",
    "ls_dr10.photo_z",
    "ls_dr11.tractor",
]


def main() -> None:
    login_from_environment()
    schema_directory = RESULTS_DIRECTORY / "schema"
    schema_directory.mkdir(parents=True, exist_ok=True)

    sizes = table_sizes(SCHEMAS)
    sizes.to_csv(schema_directory / "table_sizes.csv", index=False)
    print(sizes.to_string(index=False))

    for table in TABLES:
        columns = describe_table(table)
        columns.to_csv(schema_directory / f"{table}.csv", index=False)
        indexed = columns.loc[columns["indexed"] == 1, "column_name"].tolist()
        print(f"\n{table}: {len(columns)} columns; indexed: {', '.join(indexed)}")


if __name__ == "__main__":
    main()
