#!/usr/bin/env python3
"""Manually inspect Postgres runtime + EXPLAIN output for a single SQL.

Brings up the pg_lab container via pg_lab_setup.py, runs EXPLAIN and
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) on the hardcoded SQL below,
prints median runtime over a warmup + N attempts, plan-flip detection,
and a sorted top-N of the slowest plan nodes. Then tears the container
down again.

Edit SQL_TO_TEST below to investigate a different query.

    python3 demo_pg_runtime.py
    python3 demo_pg_runtime.py --keep-up
    python3 demo_pg_runtime.py --no-start
    python3 demo_pg_runtime.py --attempts 5 --top-n 15
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import psycopg2

HERE = Path(__file__).resolve().parent.parent  # systematic_eval/ (this script lives in scripts/)
PG_LAB_SETUP = HERE / "pg_lab_setup.py"
PG_CONF = HERE / "postgres" / "postgresql16.conf"

# ---- The query under test -------------------------------------------------
# JOB 15c — adjust manually to test a different query.
SQL_TO_TEST = """
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS modern_american_internet_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mi.note like '%internet%'
  AND mi.info is not NULL
  AND (mi.info like 'USA:% 199%' or mi.info like 'USA:% 200%')
  AND t.production_year > 1990
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;
"""


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def plan_signature(node: dict) -> tuple:
    parts = (node.get("Node Type"), node.get("Relation Name"), node.get("Join Type"))
    children = tuple(plan_signature(c) for c in node.get("Plans", []))
    return (parts, children)


def explain_analyze_once(cur, sql: str) -> tuple[dict, float, tuple, float]:
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
    raw = cur.fetchone()[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    top = raw[0] if isinstance(raw, list) else raw
    exec_ms = top.get("Execution Time", 0.0)
    plan_ms = top.get("Planning Time", 0.0)
    sig = plan_signature(top.get("Plan", {}))
    return top, exec_ms / 1000.0, sig, plan_ms / 1000.0


def measure_runtime(cur, sql: str, attempts: int) -> dict:
    print(f"  warmup run...")
    explain_analyze_once(cur, sql)

    runs = []
    for i in range(attempts):
        print(f"  attempt {i + 1}/{attempts}...")
        runs.append(explain_analyze_once(cur, sql))

    sig_counts = Counter(sig for _, _, sig, _ in runs)
    majority_sig, majority_count = sig_counts.most_common(1)[0]
    majority = [(top, t, plan_t) for top, t, sig, plan_t in runs if sig == majority_sig]
    times = sorted(t for _, t, _ in majority)
    median_time_s = times[len(times) // 2]
    chosen_top = next(top for top, t, _ in majority if t == median_time_s)
    return {
        "median_time_s": median_time_s,
        "all_times_s": [t for _, t, _, _ in runs],
        "majority_count": majority_count,
        "distinct_plans": len(sig_counts),
        "planning_time_s": next(plan_t for _, t, plan_t in majority if t == median_time_s),
        "plan_top": chosen_top,
    }


def collect_nodes(node: dict, depth: int = 0, out: list | None = None) -> list[dict]:
    """Flatten plan tree, recording per-node timing/row info."""
    if out is None:
        out = []
    out.append({
        "depth": depth,
        "node_type": node.get("Node Type"),
        "relation": node.get("Relation Name"),
        "alias": node.get("Alias"),
        "join_type": node.get("Join Type"),
        "index_name": node.get("Index Name"),
        "actual_total_time_ms": node.get("Actual Total Time"),
        "actual_startup_time_ms": node.get("Actual Startup Time"),
        "actual_rows": node.get("Actual Rows"),
        "plan_rows": node.get("Plan Rows"),
        "actual_loops": node.get("Actual Loops"),
        "filter": node.get("Filter"),
        "rows_removed_by_filter": node.get("Rows Removed by Filter"),
        "hash_cond": node.get("Hash Cond"),
        "merge_cond": node.get("Merge Cond"),
        "index_cond": node.get("Index Cond"),
        "recheck_cond": node.get("Recheck Cond"),
        "shared_hit": node.get("Shared Hit Blocks"),
        "shared_read": node.get("Shared Read Blocks"),
    })
    for c in node.get("Plans", []):
        collect_nodes(c, depth + 1, out)
    return out


def print_slowest_nodes(plan_top: dict, top_n: int) -> None:
    nodes = collect_nodes(plan_top.get("Plan", {}))
    # Actual Total Time is per-loop cumulative — multiply by loops for total work.
    for n in nodes:
        t = n.get("actual_total_time_ms") or 0.0
        loops = n.get("actual_loops") or 1
        n["total_ms"] = t * loops
    ranked = sorted(nodes, key=lambda n: n["total_ms"], reverse=True)[:top_n]

    print(f"\n--- Top {top_n} slowest nodes (Actual Total Time * loops) ---")
    print(f"{'rank':>4}  {'total_ms':>12}  {'loops':>6}  {'act_rows':>10}  "
          f"{'plan_rows':>10}  node")
    for i, n in enumerate(ranked, 1):
        label_parts = [n["node_type"] or "?"]
        if n["relation"]:
            label_parts.append(n["relation"] + (f" {n['alias']}" if n['alias'] and n['alias'] != n['relation'] else ""))
        if n["join_type"]:
            label_parts.append(f"[{n['join_type']}]")
        if n["index_name"]:
            label_parts.append(f"idx={n['index_name']}")
        label = " ".join(label_parts)
        cond = n["hash_cond"] or n["merge_cond"] or n["index_cond"] or n["recheck_cond"] or n["filter"]
        if cond:
            label += f"  ({cond})"
        if n["rows_removed_by_filter"]:
            label += f"  [filtered out {n['rows_removed_by_filter']}]"
        print(f"{i:>4}  {n['total_ms']:>12.2f}  {n['actual_loops'] or 1:>6}  "
              f"{n['actual_rows'] or 0:>10}  {n['plan_rows'] or 0:>10}  {label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--dbname", default="imdb")
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--attempts", type=int, default=3,
                    help="Number of EXPLAIN ANALYZE timing runs (after a warmup).")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Show this many slowest plan nodes.")
    ap.add_argument("--no-start", action="store_true",
                    help="Skip starting pg_lab (assume already running).")
    ap.add_argument("--keep-up", action="store_true",
                    help="Don't tear down the container at the end.")
    args = ap.parse_args()

    if not args.no_start:
        if not PG_LAB_SETUP.exists():
            print(f"ERROR: {PG_LAB_SETUP} not found.")
            return 1
        cmd = [sys.executable, str(PG_LAB_SETUP),
               "--start", "--port", str(args.port)]
        if PG_CONF.exists():
            cmd += ["--conf", str(PG_CONF)]
        run(cmd)

    try:
        con = psycopg2.connect(host=args.host, port=args.port,
                               user=args.user, password=args.password,
                               dbname=args.dbname)
        con.autocommit = True
        cur = con.cursor()

        print("\n=== SQL under test ===")
        print(SQL_TO_TEST.strip())

        print("\n=== EXPLAIN (planner only) ===")
        cur.execute(f"EXPLAIN {SQL_TO_TEST}")
        for (line,) in cur.fetchall():
            print(line)

        print(f"\n=== EXPLAIN (ANALYZE, BUFFERS): warmup + {args.attempts} attempts ===")
        meas = measure_runtime(cur, SQL_TO_TEST, attempts=args.attempts)
        print(f"\n  attempt times (s) = {[round(t, 4) for t in meas['all_times_s']]}")
        print(f"  distinct plans    = {meas['distinct_plans']}  "
              f"(majority {meas['majority_count']}/{args.attempts})")
        print(f"  planning time (s) = {meas['planning_time_s']:.4f}")
        print(f"  median exec  (s)  = {meas['median_time_s']:.4f}   <-- reported time")

        print_slowest_nodes(meas["plan_top"], args.top_n)

        print("\n--- EXPLAIN (ANALYZE, BUFFERS) text ---")
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {SQL_TO_TEST}")
        for (line,) in cur.fetchall():
            print(line)

        print("\n--- Full EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ---")
        print(json.dumps(meas["plan_top"], indent=2))

        cur.close()
        con.close()
        return 0
    finally:
        if not args.no_start and not args.keep_up:
            run([sys.executable, str(PG_LAB_SETUP),
                 "--teardown", "--port", str(args.port)], check=False)


if __name__ == "__main__":
    sys.exit(main())
