#!/usr/bin/env python3
"""Probe: does WHERE-clause conjunct order flip DuckDB's cast_info scan type?

Runs the two textual variants of the JOB 16a rule-rewritten query (differ only
in whether `t.kind_id IN (...)` or `n.id > 0` comes first in the WHERE clause)
against a local imdb.duckdb. Uses the same pragmas as
``systematic_eval/stages/execution.py:execute_query`` so the optimizer state
mirrors the pipeline:

    SET threads TO 1
    SET disabled_optimizers = 'join_order,build_side_probe_side'
    PRAGMA enable_profiling = 'json'

For each variant, prints per-attempt latencies plus the cast_info SEQ_SCAN's
``Type`` (Sequential Scan / Index Scan) and output cardinality across runs.
If the two variants disagree on Type or cardinality, the WHERE-clause order
is observably influencing access-path selection.

    python3 duckdb_where_order.py
    python3 duckdb_where_order.py imdb.duckdb 10
"""

import json
import sys
from pathlib import Path

import duckdb

DB = Path(sys.argv[1] if len(sys.argv) > 1 else "imdb.duckdb")
ATTEMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROFILE_PATH = "/tmp/duckdb_where_order_eval.json"

# ---- The two variants under test ------------------------------------------
# Both are the fixed_join_order_sql for JOB 16a from
# experiment_imdb_job_12_join_p; identical except WHERE-clause conjunct order.

_QUERY_BODY = (
    "SELECT an.name AS cool_actor_pseudonym, t.title AS series_named_after_char "
    "FROM name AS n INNER JOIN (aka_name AS an INNER JOIN (cast_info AS ci INNER JOIN "
    "(company_name AS cn INNER JOIN (movie_companies AS mc INNER JOIN (title AS t INNER JOIN "
    "(movie_keyword AS mk INNER JOIN keyword AS k ON mk.keyword_id = k.id) ON t.id = mk.movie_id) "
    "ON mc.movie_id = mk.movie_id AND t.id = mc.movie_id) ON mc.company_id = cn.id) "
    "ON ci.movie_id = mk.movie_id AND ci.movie_id = mc.movie_id AND ci.movie_id = t.id) "
    "ON an.person_id = ci.person_id) ON n.id = ci.person_id AND an.person_id = n.id "
    "WHERE cn.country_code ='[us]' AND k.keyword ='character-name-in-title' "
    "AND t.episode_nr >= 50 AND t.episode_nr < 100 "
)

SQL_A = _QUERY_BODY + "AND t.kind_id IN (2, 3, 6, 7) AND n.id > 0"
SQL_B = _QUERY_BODY + "AND n.id > 0 AND t.kind_id IN (2, 3, 6, 7)"


def find_cast_info_scan(node):
    if not isinstance(node, dict):
        return None
    extra = node.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("Table") == "cast_info":
        return node
    for c in node.get("children", []) or []:
        r = find_cast_info_scan(c)
        if r is not None:
            return r
    return None


def describe_db(read_only):
    """Print indexes/constraints visible to this connection mode."""
    con = duckdb.connect(str(DB), read_only=read_only)
    try:
        mode = "read_only" if read_only else "read_write"
        try:
            idx = con.execute(
                "SELECT database_name, schema_name, table_name, index_name, is_unique, is_primary, expressions "
                "FROM duckdb_indexes() ORDER BY table_name, index_name"
            ).fetchall()
            print(f"  [{mode}] duckdb_indexes(): {len(idx)} rows")
            for row in idx:
                print(f"    {row}")
        except Exception as e:
            print(f"  [{mode}] duckdb_indexes() failed: {e}")
        try:
            cons = con.execute(
                "SELECT table_name, constraint_type, constraint_column_names "
                "FROM duckdb_constraints() "
                "WHERE table_name IN ('cast_info','name','title','movie_keyword','keyword',"
                "'movie_companies','company_name','aka_name') "
                "ORDER BY table_name, constraint_type"
            ).fetchall()
            print(f"  [{mode}] duckdb_constraints() on join tables: {len(cons)} rows")
            for row in cons:
                print(f"    {row}")
        except Exception as e:
            print(f"  [{mode}] duckdb_constraints() failed: {e}")
    finally:
        con.close()


def run(label, sql, read_only):
    con = duckdb.connect(str(DB), read_only=read_only)
    try:
        con.execute("SET threads TO 1")
        con.execute("SET disabled_optimizers = 'join_order,build_side_probe_side'")
        con.execute("PRAGMA enable_profiling = 'json'")
        con.execute(f"PRAGMA profiling_output = '{PROFILE_PATH}'")

        con.execute(sql).fetchall()  # discarded warmup, mirrors execute_query

        latencies, types, cards = [], [], []
        for _ in range(ATTEMPTS):
            con.execute(sql).fetchall()
            with open(PROFILE_PATH) as f:
                plan = json.load(f)
            latencies.append(plan.get("latency"))
            node = find_cast_info_scan(plan)
            if node is not None:
                types.append((node.get("extra_info") or {}).get("Type"))
                cards.append(node.get("operator_cardinality"))
    finally:
        con.close()

    mode = "read_only" if read_only else "read_write"
    print(f"--- {label}  [{mode}] ---")
    print(f"  latencies (s)           : {[round(t, 4) for t in latencies]}")
    print(f"  cast_info scan Type     : {sorted({t for t in types if t is not None})}")
    print(f"  cast_info scan cardinal.: {sorted({c for c in cards if c is not None})}")
    print()


print(f"duckdb version : {duckdb.__version__}")
print(f"db file        : {DB.resolve()}")
print(f"attempts       : {ATTEMPTS}")
print()

print("== indexes & constraints visible in each connection mode ==")
describe_db(read_only=True)
describe_db(read_only=False)
print()

# Pipeline opens the DB read-write; test that first.
print("== timing: read-write (matches systematic_eval/stages/execution.py) ==")
run("A: ... AND t.kind_id IN (...) AND n.id > 0", SQL_A, read_only=False)
run("B: ... AND n.id > 0 AND t.kind_id IN (...)", SQL_B, read_only=False)

print("== timing: read-only (control) ==")
run("A: ... AND t.kind_id IN (...) AND n.id > 0", SQL_A, read_only=True)
run("B: ... AND n.id > 0 AND t.kind_id IN (...)", SQL_B, read_only=True)
