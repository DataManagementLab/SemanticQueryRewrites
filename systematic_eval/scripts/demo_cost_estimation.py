#!/usr/bin/env python3
"""Demonstrate the cost-estimation step (stage 3 / cost-aggregate).

Brings up the pg_lab container via pg_lab_setup.py, runs EXPLAIN (FORMAT JSON)
on an original JOB-style query and on a rule-rewritten variant, prints the
root Total Cost for each, and tears the container down again.

Run from the remote working dir that already contains pg_lab_setup.py and an
imdb database loaded into pg_lab (i.e. after a prior pipeline run).

    python3 demo_cost_estimation.py
    python3 demo_cost_estimation.py --keep-up   # leave the container running
    python3 demo_cost_estimation.py --no-start  # assume container is already up
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import psycopg2

HERE = Path(__file__).resolve().parent.parent  # systematic_eval/ (this script lives in scripts/)
PG_LAB_SETUP = HERE / "pg_lab_setup.py"
PG_CONF = HERE / "postgres" / "postgresql16.conf"

# ---- The query + rule we use for the demo ---------------------------------
# A small JOB-style query (q1a-shaped) — adjust if your DB schema differs.
ORIGINAL_SQL = """
SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%')
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;
"""

# The rule (semantic predicate) we add: assume the data shows that for the
# 'top 250 rank' info type, all rows are post-1950. This is the kind of
# instance-specific predicate the LLM proposes.
REFINED_SQL = ORIGINAL_SQL.rstrip().rstrip(";") + """
  AND t.production_year > 1950;
"""


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def plan_signature(node: dict) -> tuple:
    """Mirrors execution.py:_postgres_plan_signature — structural fingerprint."""
    parts = (node.get("Node Type"), node.get("Relation Name"), node.get("Join Type"))
    children = tuple(plan_signature(c) for c in node.get("Plans", []))
    return (parts, children)


def explain_analyze_once(cur, sql: str) -> tuple[dict, float, tuple]:
    """Mirrors execution.py:_postgres_run_explain_analyze_once."""
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
    raw = cur.fetchone()[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    top = raw[0] if isinstance(raw, list) else raw
    exec_ms = top.get("Execution Time", 0.0)
    plan_ms = top.get("Planning Time", 0.0)
    sig = plan_signature(top.get("Plan", {}))
    return top, exec_ms / 1000.0, sig, plan_ms / 1000.0


def measure_runtime(cur, sql: str, attempts: int = 3) -> dict:
    """Mirrors execution.py:_run_query_postgres — warmup + attempts, median of
    majority-plan runs."""
    from collections import Counter

    # Warmup (discarded) — primes buffers / JIT.
    explain_analyze_once(cur, sql)

    runs = [explain_analyze_once(cur, sql) for _ in range(attempts)]
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


def estimate_cost(cur, sql: str) -> tuple[float, dict]:
    """Mirrors execution.py:_estimate_cost_postgres — root Plan Total Cost."""
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    raw = cur.fetchone()[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    top = raw[0] if isinstance(raw, list) else raw
    root_plan = top.get("Plan", {})
    cost = root_plan.get("Total Cost")
    if cost is None:
        # Fallback identical to the production code path.
        def _sum(n):
            total = n.get("Plan Rows", 0) or 0
            for c in n.get("Plans", []):
                total += _sum(c)
            return total
        cost = _sum(root_plan)
    return float(cost), top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--dbname", default="imdb")
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--attempts", type=int, default=3,
                    help="Number of EXPLAIN ANALYZE timing runs (after a warmup).")
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

        print("\n=== Original query ===")
        cost_orig, plan_orig = estimate_cost(cur, ORIGINAL_SQL)
        print(f"  root Total Cost = {cost_orig:.2f}")
        print(f"  root Plan Rows  = {plan_orig['Plan'].get('Plan Rows')}")
        print(f"  root Node Type  = {plan_orig['Plan'].get('Node Type')}")

        print("\n--- Full EXPLAIN (FORMAT JSON) for original ---")
        print(json.dumps(plan_orig, indent=2))
        print("\n--- EXPLAIN (text) for original ---")
        cur.execute(f"EXPLAIN {ORIGINAL_SQL}")
        for (line,) in cur.fetchall():
            print(line)

        print("\n=== Rule-rewritten query (added: t.production_year > 1950) ===")
        cost_ref, plan_ref = estimate_cost(cur, REFINED_SQL)
        print(f"  root Total Cost = {cost_ref:.2f}")
        print(f"  root Plan Rows  = {plan_ref['Plan'].get('Plan Rows')}")
        print(f"  root Node Type  = {plan_ref['Plan'].get('Node Type')}")

        print("\n--- Full EXPLAIN (FORMAT JSON) for rewritten ---")
        print(json.dumps(plan_ref, indent=2))
        print("\n--- EXPLAIN (text) for rewritten ---")
        cur.execute(f"EXPLAIN {REFINED_SQL}")
        for (line,) in cur.fetchall():
            print(line)

        delta = cost_orig - cost_ref
        verdict = "KEEP (cost reduced)" if cost_ref < cost_orig else "REJECT (no reduction)"
        print(f"\n=== Cost-estimation verdict (planner-only, the step-5 metric) ===")
        print(f"  cost reduction  = {delta:.2f}  ({100.0 * delta / cost_orig:.2f} %)")
        print(f"  decision        = {verdict}")
        print("  rule:  this is the exact metric run_cost_aggregate() compares;")
        print("         a candidate rule (or rule subset) is kept iff")
        print("         refined Total Cost < original Total Cost.")

        # =========================================================
        # Real runtime measurement: EXPLAIN (ANALYZE, BUFFERS, JSON)
        # =========================================================
        print("\n\n#####################################################")
        print(f"# Real runtime measurement (warmup + {args.attempts} attempts)")
        print("#####################################################")
        print("Mirrors stages/execution.py:_run_query_postgres — what the")
        print("execution / refinement / final stages actually report.")
        print("Per call: EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) <sql>.")
        print("Collected per query:")
        print("  * Execution Time (ms, server-side, excludes client fetch)")
        print("    -> converted to seconds, median over majority-plan runs")
        print("  * Planning Time (ms)")
        print("  * plan signature = (Node Type, Relation Name, Join Type) tree")
        print("    -> detects plan flips across runs; majority wins")
        print("  * per-node Actual Total Time, Actual Rows, Plan Rows,")
        print("    Hash/Merge/Index/Recheck Cond, Filter, Index Name,")
        print("    Shared Hit/Read/Dirtied/Written Blocks (from BUFFERS).")

        print("\n=== Original query: EXPLAIN ANALYZE ===")
        meas_orig = measure_runtime(cur, ORIGINAL_SQL, attempts=args.attempts)
        print(f"  attempt times (s) = {[round(t, 4) for t in meas_orig['all_times_s']]}")
        print(f"  distinct plans    = {meas_orig['distinct_plans']}  "
              f"(majority {meas_orig['majority_count']}/{args.attempts})")
        print(f"  planning time (s) = {meas_orig['planning_time_s']:.4f}")
        print(f"  median exec  (s)  = {meas_orig['median_time_s']:.4f}   <-- reported time")
        print("\n--- Full EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for original ---")
        print(json.dumps(meas_orig["plan_top"], indent=2))
        print("\n--- EXPLAIN (ANALYZE, BUFFERS) text for original ---")
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {ORIGINAL_SQL}")
        for (line,) in cur.fetchall():
            print(line)

        print("\n=== Rule-rewritten query: EXPLAIN ANALYZE ===")
        meas_ref = measure_runtime(cur, REFINED_SQL, attempts=args.attempts)
        print(f"  attempt times (s) = {[round(t, 4) for t in meas_ref['all_times_s']]}")
        print(f"  distinct plans    = {meas_ref['distinct_plans']}  "
              f"(majority {meas_ref['majority_count']}/{args.attempts})")
        print(f"  planning time (s) = {meas_ref['planning_time_s']:.4f}")
        print(f"  median exec  (s)  = {meas_ref['median_time_s']:.4f}   <-- reported time")
        print("\n--- Full EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for rewritten ---")
        print(json.dumps(meas_ref["plan_top"], indent=2))
        print("\n--- EXPLAIN (ANALYZE, BUFFERS) text for rewritten ---")
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {REFINED_SQL}")
        for (line,) in cur.fetchall():
            print(line)

        t1 = meas_orig["median_time_s"]
        t2 = meas_ref["median_time_s"]
        threshold = 0.001  # matches experiment_T_imdb_job_12_oracle_c07_4-2.yaml
        improvement = (t1 - t2) > threshold * t1
        print("\n=== Real-runtime verdict (the final-execution metric) ===")
        print(f"  time_1 (orig)        = {t1:.4f} s")
        print(f"  time_2 (rewritten)   = {t2:.4f} s")
        print(f"  delta                = {t1 - t2:.4f} s "
              f"({100.0 * (t1 - t2) / t1:.2f} %)")
        print(f"  threshold            = {threshold} (fraction)")
        print(f"  performance_improve  = {improvement}    "
              "(time_1 - time_2 > threshold * time_1)")

        cur.close()
        con.close()
        return 0
    finally:
        if not args.no_start and not args.keep_up:
            run([sys.executable, str(PG_LAB_SETUP),
                 "--teardown", "--port", str(args.port)], check=False)


if __name__ == "__main__":
    sys.exit(main())
