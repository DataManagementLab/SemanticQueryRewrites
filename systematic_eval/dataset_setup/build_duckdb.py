"""Build a DuckDB database from a directory of scaled CSV files.

Generalizes `basketball_setup.py` to any learned_db dataset. Loads every `*.csv` in
the source directory into a table named after the file (table name = CSV stem), using
`read_csv_auto` with the dataset's `<!NULL-?>` NULL token. Optionally dumps an
authoritative `schema.txt` sidecar (the `- <table>: col,col,...` block consumed by the
prompt loader), read straight from the built tables.

Self-contained — no project imports — so it can be scp'd to the execution host (c06/c07)
and run inside the remote .venv, exactly like `stages/execution.py`.

Usage (on the remote host, inside the .venv):
    python3 build_duckdb.py \
        --csv-dir   /mnt/labstore/SIGs/ML/learned_db/datasets/financial_scaled4 \
        --db-file   /mnt/labstore/psiegler/c06_multi_query_comparison/financial.duckdb \
        --schema-out financial_schema.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

NULL_TOKEN = "<!NULL-?>"


def _load_table(con: "duckdb.DuckDBPyConnection", table: str, csv_lit: str, null_token: str) -> str:
    """Create one table from a CSV, escalating read_csv_auto robustness on failure.

    Returns a short tag describing which strategy succeeded (for the build report).
    """
    attempts = [
        ("auto", f"read_csv_auto('{csv_lit}', nullstr='{null_token}')"),
        # Full-file type inference: fixes columns whose type only becomes clear deep in
        # the file (e.g. an int column that turns out to hold a stray non-numeric value).
        ("sample=-1", f"read_csv_auto('{csv_lit}', nullstr='{null_token}', sample_size=-1)"),
        # Last resort: everything as text. Never fails on type inference; queries that
        # need numeric comparisons would cast, but at least the table loads.
        ("all_varchar", f"read_csv_auto('{csv_lit}', nullstr='{null_token}', all_varchar=true)"),
    ]
    last_err: Exception | None = None
    for tag, reader in attempts:
        try:
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM {reader}')
            return tag
        except Exception as e:  # noqa: BLE001 - report and escalate
            last_err = e
    raise RuntimeError(f"failed to load table {table!r}: {last_err}")


def _dump_schema(con: "duckdb.DuckDBPyConnection", schema_out: Path) -> None:
    """Write the `- <table>: col,col,...` schema block from the built tables."""
    rows = con.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    by_table: dict[str, list[str]] = {}
    for table_name, column_name in rows:
        by_table.setdefault(table_name, []).append(column_name)
    lines = [f"- {t}: {','.join(cols)}" for t, cols in by_table.items()]
    schema_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, help="directory of *.csv files (one per table)")
    ap.add_argument("--db-file", required=True, help="output .duckdb path")
    ap.add_argument("--null-token", default=NULL_TOKEN, help=f"NULL token (default {NULL_TOKEN!r})")
    ap.add_argument("--schema-out", default=None, help="optional schema.txt sidecar path")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSV files found in {csv_dir}")

    # rebuild from scratch for reproducibility
    if os.path.exists(args.db_file):
        os.remove(args.db_file)

    con = duckdb.connect(args.db_file)
    for csv in csvs:
        table = csv.stem
        csv_lit = csv.as_posix().replace("'", "''")
        tag = _load_table(con, table, csv_lit, args.null_token)
        n = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        note = "" if tag == "auto" else f"  [{tag}]"
        print(f"{table}: {n} rows{note}")

    if args.schema_out:
        _dump_schema(con, Path(args.schema_out))
        print(f"wrote schema -> {args.schema_out}")

    con.close()
    print(f"built {args.db_file}")


if __name__ == "__main__":
    main()
