"""Build the basketball DuckDB database from the scaled CSV files.

Loads every `*.csv` in the source directory into a table named after the file, using
`read_csv_auto` with the dataset's `<!NULL-?>` NULL token. Intended to be run once on
the execution host (c06), next to `imdb.duckdb`.

Usage (on c06, inside the remote .venv):
    python3 basketball_setup.py \
        --csv-dir /mnt/labstore/psiegler/c06_multi_query_comparison/basketball_scaled200 \
        --db-file /mnt/labstore/psiegler/c06_multi_query_comparison/basketball.duckdb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

NULL_TOKEN = "<!NULL-?>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-dir",
        default="/mnt/labstore/psiegler/c06_multi_query_comparison/basketball_scaled200",
    )
    ap.add_argument(
        "--db-file",
        default="/mnt/labstore/psiegler/c06_multi_query_comparison/basketball.duckdb",
    )
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
        con.execute(
            f'CREATE TABLE "{table}" AS '
            f"SELECT * FROM read_csv_auto('{csv_lit}', nullstr='{NULL_TOKEN}')"
        )
        n = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        print(f"{table}: {n} rows")

    con.close()
    print(f"built {args.db_file}")


if __name__ == "__main__":
    main()
