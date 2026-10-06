#!/usr/bin/env python3
"""Debug script: test Umbra EXPLAIN variants.

Run on the remote server alongside umbra_setup.py.
Starts the Umbra container, runs experiments, then tears it down.
"""

import json
import sys
from pathlib import Path

# umbra_setup.py lives in systematic_eval/; this script lives in systematic_eval/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import umbra_setup

PORT = 5432
DB = "imdb_sf1"


def main():
    # ── Start Umbra container ──────────────────────────────────────────────
    print("Starting Umbra container...")
    umbra_setup.start(port=PORT)

    try:
        import psycopg2
        con = psycopg2.connect(
            host="127.0.0.1", port=PORT,
            user="postgres", password="postgres",
            dbname=DB,
        )
        con.autocommit = True
        cur = con.cursor()

        test_query = "SELECT * FROM title LIMIT 10"

        # ─── Test A: EXPLAIN format variants ───────────────────────────
        print("\n=== Test A: EXPLAIN format variants (no execution) ===")
        for label, stmt in [
            ("EXPLAIN (plain)",            f"EXPLAIN {test_query}"),
            ("EXPLAIN (FORMAT TEXT)",       f"EXPLAIN (FORMAT TEXT) {test_query}"),
            ("EXPLAIN (FORMAT JSON)",       f"EXPLAIN (FORMAT JSON) {test_query}"),
            ("EXPLAIN (FORMAT XML)",        f"EXPLAIN (FORMAT XML) {test_query}"),
            ("EXPLAIN (FORMAT DOT)",        f"EXPLAIN (FORMAT DOT) {test_query}"),
            ("EXPLAIN (FORMAT YAML)",       f"EXPLAIN (FORMAT YAML) {test_query}"),
            ("EXPLAIN (VERBOSE)",           f"EXPLAIN (VERBOSE) {test_query}"),
            ("EXPLAIN (VERBOSE,FORMAT JSON)", f"EXPLAIN (VERBOSE, FORMAT JSON) {test_query}"),
        ]:
            try:
                cur.execute(stmt)
                rows = cur.fetchall()
                raw = rows[0][0] if rows else ""
                # Try parsing as JSON
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(parsed, dict):
                        print(f"  [{label}] JSON keys: {list(parsed.keys())}")
                        print(f"    Full output (first 2000 chars):\n{json.dumps(parsed, indent=2)[:2000]}")
                    elif isinstance(parsed, list):
                        print(f"  [{label}] JSON array, len={len(parsed)}")
                        print(f"    Full output (first 2000 chars):\n{json.dumps(parsed, indent=2)[:2000]}")
                    else:
                        print(f"  [{label}] Parsed type: {type(parsed)}")
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, print as text
                    text = raw if isinstance(raw, str) else str(raw)
                    print(f"  [{label}] Text output (first 1000 chars):\n{text[:1000]}")
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")

        # ─── Test B: EXPLAIN ANALYZE variants (executes the query) ─────
        print("\n=== Test B: EXPLAIN ANALYZE variants (with execution) ===")
        for label, stmt in [
            ("EXPLAIN ANALYZE",                     f"EXPLAIN ANALYZE {test_query}"),
            ("EXPLAIN (ANALYZE)",                    f"EXPLAIN (ANALYZE) {test_query}"),
            ("EXPLAIN (ANALYZE, FORMAT TEXT)",        f"EXPLAIN (ANALYZE, FORMAT TEXT) {test_query}"),
            ("EXPLAIN (ANALYZE, FORMAT JSON)",        f"EXPLAIN (ANALYZE, FORMAT JSON) {test_query}"),
            ("EXPLAIN (ANALYZE, VERBOSE)",            f"EXPLAIN (ANALYZE, VERBOSE) {test_query}"),
            ("EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON)", f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {test_query}"),
        ]:
            try:
                cur.execute(stmt)
                rows = cur.fetchall()
                raw = rows[0][0] if rows else ""
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(parsed, dict):
                        print(f"  [{label}] JSON keys: {list(parsed.keys())}")
                        print(f"    Full output (first 2000 chars):\n{json.dumps(parsed, indent=2)[:2000]}")
                    elif isinstance(parsed, list):
                        print(f"  [{label}] JSON array, len={len(parsed)}")
                        print(f"    Full output (first 2000 chars):\n{json.dumps(parsed, indent=2)[:2000]}")
                    else:
                        print(f"  [{label}] Parsed type: {type(parsed)}")
                except (json.JSONDecodeError, TypeError):
                    text = raw if isinstance(raw, str) else str(raw)
                    print(f"  [{label}] Text output (first 1000 chars):\n{text[:1000]}")
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")

        print()
        cur.close()
        con.close()

    finally:
        # ── Teardown Umbra container ───────────────────────────────────
        print("Tearing down Umbra container...")
        umbra_setup.teardown()


if __name__ == "__main__":
    main()
