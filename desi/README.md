# DESI DR1 through NOIRLab Astro Data Lab

Sandbox for learning to query the public DESI DR1 spectroscopic catalogs and their value-added
catalogs (VACs) on [Astro Data Lab](https://datalab.noirlab.edu), together with the Legacy
Surveys imaging catalogs. First target: Legacy Surveys sources in COSMOS with DESI redshifts and
stellar-mass/emission-line values appended.

Start with [`docs/datalab_guide.md`](docs/datalab_guide.md), which covers logging in, sync and
async queries, Q3C spatial joins, MyDB, the DESI DR1 and Legacy Surveys tables, and the gotchas
found so far.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Python scripts; `datalab_utils.py` holds the shared login/query helpers |
| `docs/` | `datalab_guide.md` (how-to), `SPEC.md` (goal, choices, open questions), `lessons.md`, `todo.md` |
| `results/schema/` | table sizes and column descriptions saved from TAP_SCHEMA |
| `results/cosmos/` | COSMOS counts; test-patch catalogs (`patch_*.csv`, git-ignored, rebuilt in seconds) |
| `/Users/shuang/work/desi/data/datalab/` | large downloads, kept outside the repo; `cosmos_ls_dr9_desi_dr1.fits` (176 MB) is the COSMOS catalog |

## Setup

    cd desi
    uv sync                                   # Python 3.11; astro-datalab needs < 3.12
    export ASTRO_DATALAB_USER=... ASTRO_DATALAB_PASS=...   # already in ~/.zshrc

The first query logs in and caches the token in `~/.datalab/` (keep it `chmod 600`).

## Scripts

    uv run python scripts/describe_tables.py                    # table sizes, column lists
    uv run python scripts/explore_cosmos.py                     # counts in the COSMOS box
    uv run python scripts/build_cosmos_catalog.py --release dr9 # 0.1 x 0.1 deg test patch
    uv run python scripts/build_cosmos_catalog.py --release dr9 --full    # whole box, ~20 s
    uv run python scripts/validate_cosmos_catalog.py            # checks on the full DR9 file

Lint and format:

    uv run ruff check scripts && uv run ruff format scripts
    uv run pre-commit run --config desi/.pre-commit-config.yaml --all-files   # from the repo root
