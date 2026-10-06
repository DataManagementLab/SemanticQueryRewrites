"""Materialize a per-query SQL directory for the basketball workload.

Reads the single-file workload `sql/complex_workload_many_joins_hints.sql` (one query
per line, each prefixed with a Postgres `/*+ ... */` planner hint), samples an evenly
spaced subset, strips the hint comment, and writes one query per file into
`sql/basketball/` as `q000.sql … q0NN.sql`.

The per-file layout matches what `stages/generation.py::parse_sqls` expects (it globs
`*.sql` and treats each file as one query).

Usage:
    python3 systematic_eval/scripts/basketball_split_workload.py [--count 100]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/ -> systematic_eval/ -> repo root
WORKLOAD = REPO_ROOT / "sql" / "complex_workload_many_joins_hints.sql"
OUT_DIR = REPO_ROOT / "sql" / "basketball"

# Leading planner-hint comment: /*+ ... */ at the very start of the line.
HINT_RE = re.compile(r"^\s*/\*\+.*?\*/\s*")


def strip_hint(line: str) -> str:
    return HINT_RE.sub("", line.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100, help="number of queries to sample")
    args = ap.parse_args()

    lines = [ln for ln in WORKLOAD.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    if args.count >= total:
        chosen = list(range(total))
    else:
        # evenly spaced sample across the whole workload for diversity
        step = total / args.count
        chosen = sorted({int(i * step) for i in range(args.count)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # clear any previous split
    for old in OUT_DIR.glob("*.sql"):
        old.unlink()

    width = max(3, len(str(len(chosen) - 1)))
    written = 0
    for new_idx, src_idx in enumerate(chosen):
        sql = strip_hint(lines[src_idx])
        if not sql.endswith(";"):
            sql += ";"
        (OUT_DIR / f"q{new_idx:0{width}d}.sql").write_text(sql + "\n", encoding="utf-8")
        written += 1

    print(f"workload lines: {total}; sampled: {written}; written to {OUT_DIR}")


if __name__ == "__main__":
    main()
