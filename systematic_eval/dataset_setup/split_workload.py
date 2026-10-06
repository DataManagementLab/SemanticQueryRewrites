"""Materialize a per-query SQL directory from a single-file workload.

Generalizes `basketball_split_workload.py` to any learned_db workload file. Reads the
first `--count` queries (one query per line), strips a leading Postgres `/*+ ... */`
planner-hint comment (a no-op for hint-free workloads), and writes one query per file
into `--out-dir` as `q000000.sql … q000NNN.sql`.

The per-file layout matches what `stages/generation.py::parse_sqls` expects (it globs
`*.sql`, sorts by name, and treats each file as one query).

The workload files on the lab mount are 60-90 MB, so the orchestrator streams only the
first N lines (`ssh ... head -n N`) into this script via stdin:

    ssh c06 'head -n 1000 .../workload_200k_s1.sql' | \
        python3 split_workload.py --workload-file - --out-dir sql/financial_200k

Or read a local file directly:

    python3 split_workload.py --workload-file path/to/workload.sql --out-dir sql/foo
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Leading planner-hint comment: /*+ ... */ at the very start of the line.
HINT_RE = re.compile(r"^\s*/\*\+.*?\*/\s*")


def strip_hint(line: str) -> str:
    return HINT_RE.sub("", line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workload-file",
        required=True,
        help="path to the single-file workload, or '-' to read from stdin",
    )
    ap.add_argument("--out-dir", required=True, help="output directory for q*.sql files")
    ap.add_argument("--count", type=int, default=1000, help="number of queries to take (sequential, from the top)")
    ap.add_argument("--width", type=int, default=6, help="zero-padding width for q<idx>.sql filenames")
    args = ap.parse_args()

    if args.workload_file == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.workload_file).read_text(encoding="utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    chosen = lines[: args.count]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear any previous split
    for old in out_dir.glob("*.sql"):
        old.unlink()

    written = 0
    for idx, raw in enumerate(chosen):
        sql = strip_hint(raw)
        if not sql:
            continue
        if not sql.endswith(";"):
            sql += ";"
        (out_dir / f"q{idx:0{args.width}d}.sql").write_text(sql + "\n", encoding="utf-8")
        written += 1

    print(f"available lines: {len(lines)}; written: {written} -> {out_dir}")


if __name__ == "__main__":
    main()
