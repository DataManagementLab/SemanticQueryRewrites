#!/usr/bin/env python3
"""
thesis_numbers.py -- single reproducible source for every number quoted in
Chapter 5 (Evaluation) of the thesis.

Design goals (per thesis workflow):
  * ONE label per number. The LaTeX carries a comment `% thesis_numbers.py -> [label]`.
    Search this file's output for that exact `[label]` to find the value.
  * NEVER guess. If a run has not been collected yet, the dependent label prints
    `MISSING RUN (<dir>)` at exactly that position, so a missing result is always
    visible and can be back-filled by re-running the experiment.
  * Trust / CV measurement metrics are deliberately NOT computed or printed.
  * Existing runs yield real values immediately.

Data foundations:
  * PRIMARY : the c07 `_T_` imdb_job runs (see RUNS below). Carry all of Part 1
              (except generality) and all of Part 2.
  * SECONDARY: the nineteen `experiment_T_gen_<dataset>` runs, used ONLY for
              eval_generality, ONLY the oracle ceiling in percent_saved.

Usage:
    python3 systematic_eval/scripts/thesis_numbers.py
    python3 systematic_eval/scripts/thesis_numbers.py | grep '\\[eval_oracle'
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from statistics import mean, median, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.normpath(os.path.join(HERE, "..", "saved_results"))
TRANSFER = os.path.normpath(os.path.join(HERE, "..", "transfer_data"))
JOB_SQL_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "sql", "job"))

# ---------------------------------------------------------------------------
# Run registry. Keys are the stable selector labels used throughout Chapter 5.
# ---------------------------------------------------------------------------
# Listed in the canonical chapter order (DuckDB, Umbra, Postgres native,
# Postgres ZeroShot), each engine followed by its own variants. See
# ENGINE_SELECTORS below for why Postgres comes last.
#
# These eight are exactly the runs the chapter reports, one per experimental
# condition, and `config/` holds exactly their eight configs plus the nineteen
# T_gen_ ones. Keep the two sets in sync: a key here without a config, or a
# config without a key, is the defect this registry is meant to make visible.
# Section 6.1 states the count as eight, so `setup.run_status.*` must print
# eight primary rows (plus the twenty GEN: rows registered below).
#
# Deliberately NOT registered: experiment_T_imdb_job_12_oracle_umbra_c07_1-1,
# the 1-1 ablation on Umbra. It was executed but is not comparable to its 4-2
# sibling, whose baselines are roughly nine times faster on the same queries
# (median original runtime 59 ms against 509 ms over the 39 shared queries), so
# its apparently higher yield is a property of the substrate state at
# measurement time and not of the mining budget. The budget ablation is
# therefore reported on DuckDB alone, where the two baselines agree to within
# two percent. Its config and results were archived out of the repository.
RUNS = {
    "DDB":     "experiment_T_imdb_job_12_oracle_c07_4-2",           # DuckDB, join order free, 4-2
    "DDB_JP":  "experiment_T_imdb_job_12_oracle_join_p_c07_4-2",    # DuckDB, join order pinned
    "DDB_1_1": "experiment_T_imdb_job_12_oracle_c07_1-1",           # DuckDB, 1-1 (generation ablation)
    "UMB":     "experiment_T_imdb_job_12_oracle_umbra_c07_4-2",     # Umbra, 4-2
    "PG":      "experiment_T_imdb_job_12_oracle_pg_c07_4-2",        # Postgres, random_page_cost 4.0
    "PG_JP":   "experiment_T_imdb_job_12_oracle_pg_join_p_c07_4-2", # Postgres, join order pinned
    "PG_RPC":  "experiment_T_imdb_job_12_oracle_pg_c07_4-2_rpc1-1", # Postgres, random_page_cost 1.1
    "LRN":     "experiment_T_imdb_job_12_zeroshot_pg_c07_4-2",      # learned ZeroShot cost model, Postgres substrate
}

# World-knowledge judge output. It is scored on the DDB run itself, so the
# single-rule speedups annotated on each rule come from the same measurements
# as every other number in the chapter. An earlier version read the judge output
# of a separate run, which mixed two measurement sets; do not reintroduce that.
WK_RUN = RUNS["DDB"]


# ---------------------------------------------------------------------------
# Generality sweep (SECONDARY foundation): the nineteen further datasets, each
# mined and measured by the same pipeline on DuckDB with the join order free.
# ---------------------------------------------------------------------------
# Registered into RUNS under GEN:<name> keys so that ceiling(), load_oracle()
# and load_all_runtimes() apply to them unchanged. That is the point of doing it
# this way: the per-dataset numbers are then literally the same computation as
# the IMDB ceiling of eval_ceiling, not a lookalike that could drift from it.
# GEN:imdb_job resolves to the primary DuckDB run itself, so IMDB enters the
# sweep as the twentieth workload on identical terms and its row must reproduce
# eval_ceiling.DDB exactly (1.180 improved, 1.089 workload). If it ever does not,
# the two code paths have diverged and one of them is wrong.
#
# The predecessor of this sweep was the experiment_<name>_oracle family, which
# is still on disk. Those runs are superseded and must not be mixed in: they
# predate the current prompt and generation budget, and they carry no IMDB
# counterpart, so no number from them is comparable to Chapter 5.
def _discover_generality() -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(os.path.join(SAVED, "experiment_T_gen_*"))):
        if os.path.isdir(d):
            name = os.path.basename(d)[len("experiment_T_gen_"):]
            out[f"GEN:{name}"] = os.path.basename(d)
    out[GEN_IMDB_KEY] = RUNS["DDB"]
    return out


GEN_IMDB_KEY = "GEN:imdb_job"
GEN_RUNS = _discover_generality()
RUNS.update(GEN_RUNS)



def wk_eval_path() -> str:
    return os.path.join(SAVED, WK_RUN, "world_knowledge", "wk_eval.json")


# Selectors that make up the engine comparison (eval_engines), free join order.
# DDB/UMB/PG vary the ENGINE; LRN varies the COST MODEL at a fixed engine
# (Postgres), so it is never read as a fourth engine.
#
# CANONICAL ORDER, used by every enumeration in Chapter 5: the two analytical
# engines first, then the row-store, and under the row-store the two cost models
# that share it. PG is therefore adjacent to LRN, which is what makes the pair
# readable as a difference: the two runs share engine, data and rule pool and
# differ only in the cost model. PG has a double role here (third level of the
# engine axis AND baseline of the cost-model axis) and the chapter says so
# explicitly; do not reorder this list without reordering the prose.
ENGINE_SELECTORS = ["DDB", "UMB", "PG", "LRN"]

# Free/pinned pairs for the join-order attribution. Pinning is available on
# DuckDB (disabled_optimizers) and on Postgres (pg_lab JoinOrder hint); the two
# mechanisms are NOT equivalent, see the note in do_attribution.
JOINORDER_PAIRS = [("ddb", "DDB", "DDB_JP"), ("pg", "PG", "PG_JP")]

# How many queries at the head of the free oracle distribution get an individual
# free -> pinned label in eval_ceiling_pinned. Six covers every per-query pair
# 5.2.3 quotes (6b/6c/6a/6e on DuckDB, 29b/29a on Postgres) with room to spare;
# raise it rather than hand-listing queries, so the label set stays data-driven
# and does not silently go stale when a run is replaced.
HEAD_N_PINNED = 6

# ---------------------------------------------------------------------------
# Output helpers. Every printed line is `[label] = value` or a MISSING marker.
# ---------------------------------------------------------------------------
_MISSING_SEEN: list[str] = []


class Missing:
    """Sentinel: a value that cannot be computed because a run is absent."""

    def __init__(self, where: str):
        self.where = where


def emit(label: str, value) -> None:
    if isinstance(value, Missing):
        _MISSING_SEEN.append(label)
        print(f"[{label}] = MISSING RUN ({value.where})")
    elif isinstance(value, float):
        print(f"[{label}] = {value:.1f}")
    else:
        print(f"[{label}] = {value}")


def section(title: str) -> None:
    print()
    print(f"# ===== {title} " + "=" * max(0, 60 - len(title)))


# ---------------------------------------------------------------------------
# Loaders. Return None (and register the reason) when a file is absent.
# ---------------------------------------------------------------------------
def _run_dir(key: str) -> str:
    return os.path.join(SAVED, RUNS[key])


def _load_stats_csv(path: str):
    """Read a *_stats.csv into {prefix: {orig, improved, pct}}. None if absent."""
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out[row["prefix"]] = {
                    "orig": float(row["original_runtime"]),
                    "improved": float(row["improved_runtime"]),
                    "pct": float(row["percent_saved"]),
                }
            except (KeyError, ValueError):
                # A malformed row is skipped rather than silently coerced.
                continue
    return out


def load_oracle(key: str):
    return _load_stats_csv(os.path.join(_run_dir(key), "oracle_stats.csv"))


def load_optimizer(key: str):
    return _load_stats_csv(os.path.join(_run_dir(key), "rule_summary_result_stats.csv"))


def load_transfer(key: str):
    path = os.path.join(_run_dir(key), "rule_summary_transfer.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def run_status(key: str) -> str:
    d = _run_dir(key)
    if not os.path.isdir(d) or not os.listdir(d):
        return "MISSING"
    if load_optimizer(key) is None and load_oracle(key) is None:
        return "EMPTY"
    return "present"


# ---------------------------------------------------------------------------
# Small statistics helpers (stdlib only, no numpy dependency).
# ---------------------------------------------------------------------------
def pct_list(stats: dict) -> list[float]:
    return [v["pct"] for v in stats.values()]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation, average-rank ties, Pearson on ranks."""
    if len(a) < 2 or len(a) != len(b):
        return float("nan")

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


# ===========================================================================
# eval_setup : foundation sizes
# ===========================================================================
def do_setup():
    section("eval_setup : data foundation")
    n_job = len(
        [
            f
            for f in glob.glob(os.path.join(JOB_SQL_DIR, "*.sql"))
            if os.path.basename(f) not in ("schema.sql", "fkindexes.sql")
        ]
    )
    emit("setup.workload_n_queries", n_job if n_job else Missing(JOB_SQL_DIR))
    for key in RUNS:
        emit(f"setup.run_status.{key}", run_status(key))


# ===========================================================================
# eval_coverage (DDB) : how many queries get an improving rule
# ===========================================================================
def do_coverage():
    section("eval_coverage (DDB)")
    n_job = len(
        [
            f
            for f in glob.glob(os.path.join(JOB_SQL_DIR, "*.sql"))
            if os.path.basename(f) not in ("schema.sql", "fkindexes.sql")
        ]
    )
    transfer = load_transfer("DDB")
    oracle = load_oracle("DDB")
    if transfer is None:
        emit("eval_coverage.n_with_rules", Missing(RUNS["DDB"]))
    else:
        emit("eval_coverage.n_with_rules", len(transfer))
        rule_counts = [len(v.get("rules", [])) for v in transfer.values()]
        emit("eval_coverage.rules_per_query_median", median(rule_counts) if rule_counts else Missing(RUNS["DDB"]))
    if oracle is None:
        emit("eval_coverage.n_with_improvement", Missing(RUNS["DDB"]))
    else:
        emit("eval_coverage.n_with_improvement", len(oracle))
        if n_job:
            emit("eval_coverage.pct_with_improvement", 100.0 * len(oracle) / n_job)
        # The oracle keeps the fastest of all executed subsets, so the count of
        # "improving" queries is a maximum over noise at its lower edge. The
        # thresholds make that edge visible; 5.2.1 quotes them next to the raw
        # count and does not let the count carry the claim alone.
        pcts = pct_list(oracle)
        for th in (1, 5, 10):
            emit(f"eval_coverage.n_improving_above_{th}pct",
                 sum(1 for p in pcts if p > th))


# ===========================================================================
# eval_transfer (DDB) : cross-query fan-out
# ===========================================================================
def do_transfer():
    section("eval_transfer (DDB)")
    transfer = load_transfer("DDB")
    if transfer is None:
        emit("eval_transfer.fanout_median_all", Missing(RUNS["DDB"]))
        return
    # Invert: rule name -> set of queries it is applied to.
    fan: dict[str, set] = {}
    for q, v in transfer.items():
        for r in v.get("rules", []):
            fan.setdefault(r["name"], set()).add(q)
    sizes = [len(s) for s in fan.values()]
    emit("eval_transfer.n_distinct_rules", len(fan))
    # The applicable rule repository is heterogeneous: merging combines rules with
    # identical `requires` into a composite whose components come from several
    # source queries, so a composite has no single source query. 5.2.2 names the
    # two kinds rather than reporting a fan-out split, which would carry the
    # section past its scope.
    emit("eval_transfer.n_merged_rules",
         sum(1 for k in fan if str(k).startswith("merged:")))
    emit("eval_transfer.n_atomic_rules",
         sum(1 for k in fan if not str(k).startswith("merged:")))
    # Fan-out is summarised on TWO populations and every label names the one it
    # belongs to, because mixing them is easy and changes the claim. The _all
    # labels cover every applicable rule, including the 54 that are applied to exactly one
    # query and therefore contribute a fan-out of 1 by construction. The *_gt1
    # labels further down cover only the rules that actually transfer, and the
    # two differ substantially (mean 2.5 against 3.9).
    #
    # 5.2.2 quotes the _all labels and names that population explicitly, so the
    # fan-out summary describes the same 111 rules as the 57/54 split beside it
    # and the reader never has to infer which set a number belongs to. The *_gt1
    # labels exist as a cross-check and are deliberately NOT quoted there: a
    # conditional mean placed next to an unconditional split reads as one
    # statistic over one population when it is two over two. Median and mean are
    # emitted together per the aggregation convention of 5.1.
    emit("eval_transfer.fanout_median_all", float(median(sizes)) if sizes else Missing(RUNS["DDB"]))
    emit("eval_transfer.fanout_mean_all", float(mean(sizes)) if sizes else Missing(RUNS["DDB"]))
    emit("eval_transfer.fanout_max", max(sizes) if sizes else Missing(RUNS["DDB"]))
    n_gt1 = sum(1 for s in sizes if s > 1)
    emit("eval_transfer.n_rules_transferred", n_gt1)
    emit("eval_transfer.n_fanout_1", sum(1 for s in sizes if s == 1))
    if sizes:
        emit("eval_transfer.pct_rules_transferred", 100.0 * n_gt1 / len(sizes))
    # Population of the transferring rules alone. Mean AND median are emitted
    # here for the reason 5.1 gives for outcome aggregates: the distribution is
    # right-skewed (median 3 against a mean of 3.9, with a maximum of 15 carried
    # by two rules), so the mean alone overstates the typical transferring rule
    # while the median alone hides the tail that the maximum belongs to. The two
    # threshold counts make that tail concrete without introducing a quantile
    # the prose would then have to explain.
    gt1 = [s for s in sizes if s > 1]
    if gt1:
        emit("eval_transfer.fanout_median_gt1", float(median(gt1)))
        emit("eval_transfer.fanout_mean_gt1", float(mean(gt1)))
        emit("eval_transfer.n_fanout_ge3", sum(1 for s in sizes if s >= 3))
        emit("eval_transfer.n_fanout_ge4", sum(1 for s in sizes if s >= 4))


# ===========================================================================
# eval_runtime (DDB oracle) : achievable speedups + hero examples
# ===========================================================================
def do_runtime():
    section("eval_runtime (DDB, oracle ceiling)")
    oracle = load_oracle("DDB")
    if oracle is None:
        emit("eval_runtime.median_speedup", Missing(RUNS["DDB"]))
        return
    ps = pct_list(oracle)
    emit("eval_runtime.n_improved", len(ps))
    emit("eval_runtime.median_speedup", float(median(ps)))
    emit("eval_runtime.mean_speedup", float(mean(ps)))
    emit("eval_runtime.max_speedup", float(max(ps)))
    heroes = sorted(oracle.items(), key=lambda kv: kv[1]["pct"], reverse=True)[:5]
    for i, (q, v) in enumerate(heroes, 1):
        print(f"[eval_runtime.hero.{i}] = {q}: {v['pct']:.1f}% ({v['orig']}s -> {v['improved']}s)")


# ===========================================================================
# eval_ceiling : the oracle ceiling per ENGINE, on two explicit populations.
#
# Two denominators, kept strictly apart, because conflating them is the easiest
# way to overstate the result:
#   * IMPROVED  -- the queries the rules actually touch (an oracle row exists).
#                  Answers "how large is the win where there is one".
#   * WORKLOAD  -- every query with a measured original runtime, including the
#                  ones no rule improves, which enter at factor 1.0. Answers
#                  "what does the whole benchmark gain". This is the honest
#                  headline denominator and it is always the smaller number.
# Aggregates: geometric mean of the per-query speedup FACTOR (the correct mean
# for ratios), plus the runtime-weighted total factor sum(orig)/sum(improved).
# ===========================================================================
def _cache_by_key(fn):
    """Memoise a per-run loader. rule_summary_result.json is large and both the
    numbers script and the figures script ask for the same run repeatedly; the
    files are read-only inputs for the length of a run, so caching cannot change
    a result, only the time it takes to produce one."""
    store: dict = {}

    def wrapped(key: str):
        if key not in store:
            store[key] = fn(key)
        return store[key]

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _basename(prefix: str) -> str:
    """'12c NoMIN' -> '12c'. oracle_stats.csv carries the variant suffix,
    baseline_runtimes.json does not, so both are normalised before joining."""
    return prefix.split(" ", 1)[0]


@_cache_by_key
def load_all_runtimes(key: str):
    """{basename: original runtime} for EVERY measured query of a run, whether
    or not a rule improved it. Two disjoint sources, mirroring statistics.py:
    the per-query summary (queries that carry a rule) and baseline_runtimes.json
    (queries no rule applies to, measured by the dedicated baseline pass).
    Returns None when neither file exists."""
    out: dict[str, float] = {}
    summary_path = os.path.join(_run_dir(key), "rule_summary_result.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as fh:
            for name, entry in json.load(fh).items():
                if not isinstance(entry, dict):
                    continue
                rt = entry.get("summary", {}).get("execution_time", {}).get("original_query")
                if isinstance(rt, (int, float)) and rt > 0:
                    out.setdefault(_basename(name), float(rt))
    baseline_path = os.path.join(_run_dir(key), "baseline_runtimes.json")
    if os.path.isfile(baseline_path):
        with open(baseline_path) as fh:
            for name, entry in json.load(fh).items():
                rt = entry.get("original_query") if isinstance(entry, dict) else entry
                if isinstance(rt, (int, float)) and rt > 0:
                    # The dedicated baseline pass wins over a stale summary value.
                    out[_basename(name)] = float(rt)
    return out or None


def unmeasured_queries(key: str) -> list[str]:
    """Queries the run recorded but for which every attempt came back 0.0, i.e.
    the measurement failed rather than the query being fast. They cannot enter a
    speedup factor (the ratio is undefined) and are excluded from the ceiling
    population, so their number has to be reportable."""
    out: set[str] = set()
    for fn, get in (("rule_summary_result.json",
                     lambda e: e.get("summary", {}).get("execution_time", {}).get("original_query")),
                    ("baseline_runtimes.json",
                     lambda e: e.get("original_query") if isinstance(e, dict) else e)):
        path = os.path.join(_run_dir(key), fn)
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            for name, entry in json.load(fh).items():
                if not isinstance(entry, dict):
                    continue
                rt = get(entry)
                if isinstance(rt, (int, float)) and rt <= 0:
                    out.add(_basename(name))
    measured = load_all_runtimes(key) or {}
    return sorted(out - set(measured))


def geomean(factors: list[float]) -> float:
    """Geometric mean, computed in log space so a long product cannot underflow."""
    if not factors:
        return float("nan")
    from math import exp, log
    return exp(sum(log(f) for f in factors) / len(factors))


def ceiling_common():
    """{engine key: total original runtime} restricted to the queries EVERY
    engine measured. The cross-engine runtime comparison has to run on one
    shared population so no engine is flattered by measuring fewer queries. In
    the current runs all three engines measure the full 113, so this shared
    population equals the full workload; the intersection is kept as a safeguard
    for future runs where an engine might measure fewer. Returns None if any run
    is absent."""
    per_engine = {}
    for key, _label, _pin in CEILING_ENGINES:
        oracle, runtimes = load_oracle(key), load_all_runtimes(key)
        if oracle is None or runtimes is None:
            return None
        per_engine[key] = ({_basename(q): v for q, v in oracle.items()}, runtimes)
    shared = set.intersection(
        *(set(rt) | set(by_q) for by_q, rt in per_engine.values()))
    return {key: sum(by_q[q]["orig"] if q in by_q else rt[q] for q in shared)
            for key, (by_q, rt) in per_engine.items()}


@_cache_by_key
def ceiling(key: str):
    """Oracle ceiling of one run on both populations. Returns None if absent."""
    oracle = load_oracle(key)
    runtimes = load_all_runtimes(key)
    if oracle is None or runtimes is None:
        return None
    improved = {_basename(q): v for q, v in oracle.items()}
    # Population = every measured query. An improved query without a baseline
    # entry still counts (its original runtime is in the oracle row itself).
    pop = sorted(set(runtimes) | set(improved))
    factors, tot_orig, tot_best = [], 0.0, 0.0
    for q in pop:
        if q in improved:
            v = improved[q]
            factors.append(v["orig"] / v["improved"])
            tot_orig += v["orig"]
            tot_best += v["improved"]
        else:
            factors.append(1.0)
            tot_orig += runtimes[q]
            tot_best += runtimes[q]
    win_factors = [v["orig"] / v["improved"] for v in improved.values()]
    win_pcts = [v["pct"] for v in improved.values()]
    # Runtime-weighted factor on the improved population only (the counterpart to
    # geo_improved, and to eval_ceiling_weighted.{key}.wtd_factor_improved). Its
    # sums run over the improved queries alone, not the full-workload tot_orig /
    # tot_best above, which dilute with the unchanged queries entering at 1.0.
    imp_orig = sum(v["orig"] for v in improved.values())
    imp_best = sum(v["improved"] for v in improved.values())
    return {
        "n_workload": len(pop),
        "n_improved": len(improved),
        "total_original_s": tot_orig,
        "total_oracle_s": tot_best,
        "saved_s": tot_orig - tot_best,
        "geo_improved": geomean(win_factors),
        "geo_workload": geomean(factors),
        "improved_factor": imp_orig / imp_best if imp_best > 0 else float("nan"),
        "workload_factor": tot_orig / tot_best if tot_best > 0 else float("nan"),
        "workload_pct": 100.0 * (tot_orig - tot_best) / tot_orig if tot_orig > 0 else float("nan"),
        "median_improved_pct": float(median(win_pcts)),
        "mean_improved_pct": float(mean(win_pcts)),
        "max_improved_pct": float(max(win_pcts)),
    }


# Engines whose ceiling is reported, each with its pinned sibling where one
# exists. Umbra has NO pinned entry on purpose: execution.py raises for any
# engine other than duckdb/postgres because Umbra exposes no join-order pinning
# mechanism, so its ceiling cannot be decomposed. That is a limitation, not a
# missing run, and it must not be silently filled in later.
CEILING_ENGINES = [("DDB", "DuckDB", "DDB_JP"),
                   ("UMB", "Umbra", None),
                   ("PG", "Postgres", "PG_JP")]


def generality_rows() -> list[dict]:
    """One ceiling() per workload of the generality sweep, IMDB included,
    sorted by the full-workload factor descending. Rows carry the same keys
    ceiling() returns, so geo_improved and geo_workload mean exactly what they
    mean in eval_ceiling."""
    rows = []
    for key in GEN_RUNS:
        c = ceiling(key)
        if c is None:
            continue
        rows.append({"key": key, "name": key.split(":", 1)[1], **c})
    rows.sort(key=lambda r: -r["geo_workload"])
    return rows


@_cache_by_key
def generality_pool(key: str):
    """Per-query rows of one generality workload as (orig_seconds, factor or
    None), factor None meaning no improving subset exists. Same two sources as
    ceiling(), so the population is identical to the one the ceiling is
    computed on; this only exposes it per query instead of aggregated."""
    oracle, rts = load_oracle(key), load_all_runtimes(key)
    if oracle is None or rts is None:
        return None
    improved = {_basename(q): v for q, v in oracle.items()}
    out = []
    for q in sorted(set(rts) | set(improved)):
        if q in improved:
            v = improved[q]
            out.append((v["orig"], v["orig"] / v["improved"]))
        else:
            out.append((rts[q], None))
    return out


def generality_sec_share(key: str) -> float:
    """Percent of a workload's total original seconds held by the queries an
    improving subset exists for. This is the variable that explains the
    runtime-weighted factor: a workload whose wins sit on its cheap queries
    cannot move its own wall clock however many queries improve."""
    pool = generality_pool(key)
    if not pool:
        return float("nan")
    tot = sum(o for o, _ in pool)
    return 100.0 * sum(o for o, f in pool if f) / tot if tot > 0 else float("nan")


def generality_floor(key: str, floor_s: float):
    """geo_workload and the runtime-weighted factor recomputed with queries
    below floor_s dropped from BOTH populations. The generality workloads are
    machine-generated and far cheaper than JOB, so a reader is entitled to ask
    whether their ceiling is sub-millisecond measurement noise; this answers it
    without appealing to the trust labels."""
    pool = [(o, f) for o, f in (generality_pool(key) or []) if o >= floor_s]
    if not pool:
        return None
    factors = [f if f else 1.0 for o, f in pool]
    to = sum(o for o, _ in pool)
    tb = sum((o / f) if f else o for o, f in pool)
    return {"n": len(pool), "geo_workload": geomean(factors),
            "workload_factor": to / tb if tb > 0 else float("nan")}


def do_ceiling():
    section("eval_ceiling : oracle ceiling per engine (improved vs full workload)")
    for key, label, pin in CEILING_ENGINES:
        for sub, k in ((key, key), *(((pin, pin),) if pin else ())):
            c = ceiling(k)
            if c is None:
                emit(f"eval_ceiling.{sub}.geo_workload", Missing(RUNS[k]))
                continue
            emit(f"eval_ceiling.{sub}.n_workload", c["n_workload"])
            emit(f"eval_ceiling.{sub}.n_improved", c["n_improved"])
            # Two decimals, and printed directly rather than via emit(): emit()
            # formats every float with .1f, which would collapse 69.35 -> 69.3
            # and 2.95 -> 3.0. The wall-clock prose in eval_runtime quotes these
            # to two decimals, so the label has to carry them at that precision.
            print(f"[eval_ceiling.{sub}.total_original_s] = {c['total_original_s']:.2f}")
            print(f"[eval_ceiling.{sub}.saved_s] = {c['saved_s']:.2f}")
            emit(f"eval_ceiling.{sub}.median_improved_pct", c["median_improved_pct"])
            print(f"[eval_ceiling.{sub}.geo_improved] = {c['geo_improved']:.3f}x")
            print(f"[eval_ceiling.{sub}.geo_workload] = {c['geo_workload']:.3f}x")
            print(f"[eval_ceiling.{sub}.workload_factor] = {c['workload_factor']:.3f}x")
            emit(f"eval_ceiling.{sub}.workload_pct", c["workload_pct"])
            emit(f"eval_ceiling.{sub}.max_improved_pct", c["max_improved_pct"])
            # A run whose population is smaller than the workload is only
            # comparable to the others once the shortfall is named.
            unmeasured = unmeasured_queries(k)
            emit(f"eval_ceiling.{sub}.n_unmeasured", len(unmeasured))
            if unmeasured:
                print(f"[eval_ceiling.{sub}.unmeasured] = {unmeasured}")

    # Per-query versus runtime-weighted aggregation. The geometric means above
    # weight every query equally; the total factor weights by runtime. Reporting
    # only one of the two would let the chapter pick whichever is larger, so both
    # are emitted together, along with the evidence for WHY they differ.
    section("eval_ceiling_weighted : per-query versus runtime-weighted view")
    # The pinned runs are included, not just the three free engines, because the
    # Rt-wtd. column of the ceiling table has a cell on every row it prints,
    # pinned rows included. Emitting only the free engines left the two pinned
    # cells (1.110 on DuckDB, 1.018 on Postgres) as the only numbers in that
    # table without a label. Only wtd_factor_improved is meaningful for a pinned
    # run; the tail diagnostics below (spearman, top10, top_saved) stay on the
    # free engines, where the prose actually argues from them.
    weighted_keys = [k for key, _l, pin in CEILING_ENGINES
                     for k in ((key,) + ((pin,) if pin else ()))]
    for key in weighted_keys:
        oracle, c = load_oracle(key), ceiling(key)
        if oracle is None or c is None:
            emit(f"eval_ceiling_weighted.{key}.wtd_mean_pct", Missing(RUNS[key]))
            continue
        if key not in [e[0] for e in CEILING_ENGINES]:
            total_orig = sum(v["orig"] for v in oracle.values())
            total_improved = sum(v["improved"] for v in oracle.values())
            print(f"[eval_ceiling_weighted.{key}.wtd_factor_improved] = "
                  f"{total_orig / total_improved if total_improved else float('nan'):.3f}x")
            continue
        origs = [v["orig"] for v in oracle.values()]
        pcts = [v["pct"] for v in oracle.values()]
        total_orig = sum(origs)
        emit(f"eval_ceiling_weighted.{key}.unwtd_mean_pct", float(mean(pcts)))
        emit(f"eval_ceiling_weighted.{key}.wtd_mean_pct",
             sum(t * p for t, p in zip(origs, pcts)) / total_orig if total_orig else float("nan"))
        # Runtime-weighted total factor on the SAME (improved) population, as a
        # factor rather than a percent. This is the runtime-weighted counterpart
        # to eval_ceiling.{key}.geo_improved (the query-weighted factor on the
        # improved queries, the dashed line of the ceiling figure). It equals
        # 1 / (1 - wtd_mean_pct/100); reported as a factor so the ceiling
        # paragraph can stay entirely in factors on both populations.
        total_improved = sum(v["improved"] for v in oracle.values())
        print(f"[eval_ceiling_weighted.{key}.wtd_factor_improved] = "
              f"{total_orig / total_improved if total_improved else float('nan'):.3f}x")
        # A negative rank correlation would mean cheap queries systematically
        # gain more. It is near zero on two of three engines, so the divergence
        # between the two aggregates is a tail effect, not a general trend.
        # Two decimals: the prose reports this correlation to two places
        # (e.g. -0.39 vs -0.01), and emit() would otherwise round a float to
        # one, collapsing -0.01 to -0.0 and breaking the label/prose match.
        emit(f"eval_ceiling_weighted.{key}.spearman_runtime_speedup",
             f"{spearman(origs, pcts):.2f}")
        saved = sorted(((v["orig"] - v["improved"], q) for q, v in oracle.items()),
                       reverse=True)
        total_saved = sum(s for s, _ in saved)
        top10 = saved[:10]
        emit(f"eval_ceiling_weighted.{key}.top10_share",
             100.0 * sum(s for s, _ in top10) / total_saved if total_saved else float("nan"))
        for i, (s, q) in enumerate(top10, 1):
            v = oracle[q]
            print(f"[eval_ceiling_weighted.{key}.top_saved.{i}] = {_basename(q)}: "
                  f"{s:.3f}s ({v['pct']:.1f}% of {v['orig']*1000:.0f}ms)")

    # The cross-engine runtime comparison ("Umbra is the fastest substrate and
    # still has the most headroom") must run on ONE population, otherwise an
    # engine that measured fewer queries would be compared on a smaller
    # denominator. In the current runs all three engines measure the full 113,
    # so this restriction is a safeguard; it keeps the comparison honest if a
    # future run measures fewer. Report the totals on the shared population.
    section("eval_ceiling_common : workload runtime on the shared population")
    totals = ceiling_common()
    if totals is None:
        emit("eval_ceiling_common.n_shared", Missing("one of the engine runs"))
    else:
        emit("eval_ceiling_common.n_shared",
             min(ceiling(k)["n_workload"] for k in totals if ceiling(k)))
        for key, total in totals.items():
            emit(f"eval_ceiling_common.{key}.total_original_s", round(total, 1))

    # How much of the ceiling is genuine selectivity rather than plan
    # reordering. Reported here (not only in the Part 2 attribution) because a
    # ceiling quoted without it is not defensible on Postgres, where the two
    # largest wins are pure reordering.
    section("eval_ceiling_pinned : share of the ceiling that survives pinning")
    for key, label, pin in CEILING_ENGINES:
        if pin is None:
            print(f"[eval_ceiling_pinned.{key}.retained_pct] = "
                  f"NOT MEASURABLE (engine has no join-order pinning mechanism)")
            continue
        free, pinned = load_oracle(key), load_oracle(pin)
        if free is None or pinned is None:
            emit(f"eval_ceiling_pinned.{key}.retained_pct", Missing(RUNS[pin]))
            continue
        pinned_by_q = {_basename(q): v for q, v in pinned.items()}
        s_free = sum(v["orig"] - v["improved"] for v in free.values())
        s_pin = sum(v["orig"] - v["improved"] for v in pinned.values())
        big = [_basename(q) for q, v in free.items() if v["pct"] > 10]
        survive = [q for q in big if pinned_by_q.get(q, {}).get("pct", 0.0) > 10]
        emit(f"eval_ceiling_pinned.{key}.saved_s_free", round(s_free, 2))
        emit(f"eval_ceiling_pinned.{key}.saved_s_pinned", round(s_pin, 2))
        emit(f"eval_ceiling_pinned.{key}.retained_pct",
             100.0 * s_pin / s_free if s_free > 0 else float("nan"))
        emit(f"eval_ceiling_pinned.{key}.n_big_free", len(big))
        emit(f"eval_ceiling_pinned.{key}.n_big_survive", len(survive))

        # Per-query free -> pinned pairs at the head of the free distribution.
        # 5.2.3 quotes these individually (6a/6c/6e on DuckDB, 29b/29a on
        # Postgres) and they had no label before, so they were read off
        # oracle_stats.csv by hand.
        #
        # CAUTION, this is the ORACLE view. eval_casestudies below emits
        # per-query free/pinned pairs too, but in the OPTIMIZER view, and the two
        # disagree: 6a is 58.9 -> 58.9 here and 58.5 -> 58.8 there, 6c is
        # 64.7 -> 64.2 here and 62.5 -> 64.2 there. The 6c pinned value happens
        # to coincide, which makes the wrong label look right. Never cite
        # eval_casestudies for a ceiling number.
        head = sorted(free.items(), key=lambda kv: -kv[1]["pct"])[:HEAD_N_PINNED]
        for q, v in head:
            qb = _basename(q)
            p = pinned_by_q.get(qb, {}).get("pct")
            print(f"[eval_ceiling_pinned.{key}.q.{qb}] = "
                  f"{v['pct']:.1f}% -> "
                  f"{'NOT IMPROVED' if p is None else f'{p:.1f}%'} (oracle view)")

    # Which queries carry the ceiling is engine dependent. Stated explicitly so
    # the per-engine numbers above are not read as the same win measured thrice.
    section("eval_ceiling_overlap : do the engines improve the same queries?")
    sets, big_sets = {}, {}
    for key, _label, _pin in CEILING_ENGINES:
        o = load_oracle(key)
        if o is None:
            emit(f"eval_ceiling_overlap.{key}.n", Missing(RUNS[key]))
            return
        sets[key] = {_basename(q) for q in o}
        big_sets[key] = {_basename(q) for q, v in o.items() if v["pct"] > 10}
    keys = list(sets)
    union = set().union(*sets.values())
    common = set.intersection(*sets.values())
    big_union = set().union(*big_sets.values())
    big_common = set.intersection(*big_sets.values())
    emit("eval_ceiling_overlap.n_union", len(union))
    emit("eval_ceiling_overlap.n_common_all3", len(common))
    emit("eval_ceiling_overlap.big_n_union", len(big_union))
    emit("eval_ceiling_overlap.big_n_common_all3", len(big_common))
    print(f"[eval_ceiling_overlap.big_common_all3] = {sorted(big_common)}")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            emit(f"eval_ceiling_overlap.pair.{a}_{b}", len(sets[a] & sets[b]))


# ===========================================================================
# eval_worldknowledge (judge run on the DDB run, see WK_RUN)
# ===========================================================================
# The seven rules quoted verbatim in the 5.2.4 table, as (score printed in the
# table, rule_name in wk_eval.json), in the table's own row order: at least one
# per score level, with score 4 shown three times because it is the modal bucket
# of the repository (61 of the 111 rated rules, emitted as eval_wk.score_4).
# Used by do_worldknowledge to give each printed rule's score a label of its own
# and to assert it against the table.
WK_TABLE_RULES = [
    (1, "4a NoMIN_r3_2_IMDB_R3"),      # >2005 entails >=2006 on an integer year
    (2, "26b NoMIN_r3_1_IMDB_R2"),     # IMDb ratings carry one decimal
    (3, "23a NoMIN_r0_1_IMDB_R2"),     # a film is not an episode
    (4, "28b NoMIN_r0_1_IMDB_R2"),     # nationality adjective vs country value
    (4, "3b NoMIN_r1_0_IMDB_R1"),      # sequel-tagged post-2010 title is a film
    # Entity disambiguation. '%Downey%Robert%' matches TWO people on this instance,
    # 'Downey Jr., Robert' (id 414711) and 'Downey Sr., Robert' (id 414712); see the
    # base-table validation of '6d NoMIN_r2_error_rerun_s1' in
    # transfer_data/<DDB run>/result2.json, which runs the pattern against `name`
    # alone and reports requires_row_count = 2 with Downey Sr. as the counterexample.
    # Only the post-2014 comic-keyword context makes the exact-name implication sound,
    # and that same counterexample is what rejected the over-broad query-6d siblings.
    # This is a MERGED rule: the exact-name implies exists on the merge alone and on no
    # 6b singleton, so the full name is pinned here rather than a prefix of it.
    (4, "merged: 6b NoMIN_r0_1_IMDB_R2 + 6b NoMIN_r0_2_IMDB_R3 + "
        "6b NoMIN_r1_1_IMDB_R2 + 6b NoMIN_r0_0_IMDB_R1_rerun_s1"),
    (5, "29b NoMIN_r2_2_IMDB_R3"),     # Shrek 2 principal voice role, nr_order<=20
]


def do_worldknowledge():
    section("eval_worldknowledge (LLM judge, 1-5)")
    path = wk_eval_path()
    if not os.path.isfile(path):
        emit("eval_wk.mean_score", Missing(path))
        return
    with open(path) as fh:
        rules = json.load(fh)["rules"]
    scores = [r["verdict"]["score"] for r in rules if "verdict" in r and "score" in r["verdict"]]
    emit("eval_wk.n_rules", len(scores))
    for s in range(1, 6):
        emit(f"eval_wk.score_{s}", sum(1 for x in scores if x == s))
    if scores:
        emit("eval_wk.mean_score", float(mean(scores)))
        emit("eval_wk.pct_score_ge4", 100.0 * sum(1 for x in scores if x >= 4) / len(scores))
    # Score vs single-rule speedup (agg.median_percent_saved).
    pairs = [
        (r["verdict"]["score"], r["agg"]["median_percent_saved"])
        for r in rules
        if r.get("verdict", {}).get("score") is not None
        and r.get("agg", {}).get("median_percent_saved") is not None
    ]
    if len(pairs) >= 2:
        # Two decimals for the same reason as eval_ceiling_weighted above: the
        # prose reports this near-zero correlation as 0.03, which emit() would
        # round to 0.0.
        emit("eval_wk.corr_score_speedup",
             f"{spearman([p[0] for p in pairs], [p[1] for p in pairs]):.2f}")
    # Identity of the enabling tail with the top of the ceiling (5.2.4 prose):
    # every rule whose best single-rule speedup exceeds 50 % is applied only to the
    # JOB query-6 group (the Downey/MCU family). Reported as "n_q6 of n_big" so
    # the prose "six of six ... are variants of JOB query 6" carries a label.
    def _group(q: str):
        m = re.match(r"\s*(\d+)", str(q))
        return m.group(1) if m else None
    big = [r for r in rules if (r.get("agg", {}).get("max_percent_saved") or -1e9) > 50]
    if big:
        n_q6 = sum(
            1 for r in big
            if all(_group(pq.get("query")) == "6" for pq in r.get("per_query", []))
        )
        emit("eval_wk.big50_n", len(big))
        emit("eval_wk.big50_all_q6", n_q6)
    # Single-rule speedup by world-knowledge score. median_percent_saved is
    # already stored in percent, so it is emitted as-is (do NOT rescale sub-1
    # values: a rule with -0.11 means -0.11 %, not -11 %). The buckets show that
    # the large wins concentrate in the high-WK scores (>=4), while the rank
    # correlation above stays weak because most rules, score 4 included, sit
    # near zero in the median. Buckets 3 and 5 are small (n<=6), so the mean is
    # reported next to the more robust median rather than on its own.
    by_score: dict = {}
    for r in rules:
        s = r.get("verdict", {}).get("score")
        v = r.get("agg", {}).get("median_percent_saved")
        if s is None or v is None:
            continue
        by_score.setdefault(s, []).append(v)
    for s in range(1, 6):
        vals = by_score.get(s, [])
        emit(f"eval_wk.speedup_by_score.{s}.n", len(vals))
        if vals:
            emit(f"eval_wk.speedup_by_score.{s}.mean", float(mean(vals)))
            emit(f"eval_wk.speedup_by_score.{s}.median", float(median(vals)))
    le3 = [v for s, vs in by_score.items() if s <= 3 for v in vs]
    ge4 = [v for s, vs in by_score.items() if s >= 4 for v in vs]
    if le3 and ge4:
        emit("eval_wk.speedup_ge4_vs_le3.le3_n", len(le3))
        emit("eval_wk.speedup_ge4_vs_le3.ge4_n", len(ge4))
        emit("eval_wk.speedup_ge4_vs_le3.le3_mean", float(mean(le3)))
        emit("eval_wk.speedup_ge4_vs_le3.ge4_mean", float(mean(ge4)))
        emit("eval_wk.speedup_ge4_vs_le3.le3_median", float(median(le3)))
        emit("eval_wk.speedup_ge4_vs_le3.ge4_median", float(median(ge4)))

    # The six rules printed verbatim in the Table of 5.2.4, one per score level
    # (two at score 4). Only the score DISTRIBUTION had a label before, so the
    # individual score attached to each printed rule was uncheckable. These
    # labels close that gap: if the judge is ever re-run, a changed score shows
    # up here and the table's \wkscore{} column can be corrected against it.
    #
    # The rule names are pinned rather than matched on the predicate text,
    # because several near-identical variants exist and they do NOT share a
    # score. 4a_r3_2 (score 1) has a sibling 4a_r1_0_rerun_s1 with an extra
    # mi_idx bound, and 26b_r3_1 (score 2) has a sibling 26b_r0_2 that scores 5.
    # Matching by predicate would silently pick the wrong one.
    section("eval_wk_table : the seven rules shown in the 5.2.4 table")
    by_name = {r["rule_name"]: r for r in rules}
    for tbl_score, name in WK_TABLE_RULES:
        r = by_name.get(name)
        if r is None:
            emit(f"eval_wk_table.{name}.score", Missing(f"{path} :: {name}"))
            continue
        got = r.get("verdict", {}).get("score")
        flag = "" if got == tbl_score else f"  <<< MISMATCH, table shows {tbl_score}"
        print(f"[eval_wk_table.{name}.score] = {got}{flag}")


# ===========================================================================
# eval_oracle (DDB) : oracle vs optimizer gap  --- HEADLINE of Part 2
# ===========================================================================
def _unified(key: str):
    """Unified per-selector view over ONE population: every query that has at
    least one applicable rule (transfer keys, plus any oracle/optimizer prefix
    for safety). Ceiling = oracle percent_saved (0 if no improving subset);
    Realized = optimizer percent_saved (0 if no rewrite applied / no-change).
    Percentages and absolute seconds are both aggregated on this population.
    Returns Missing if any required file is absent.

    Three families of aggregate come out of here and they are NOT
    interchangeable.

    The FACTOR family (geo_*, wtd_*, median_*_factor) is the reporting default,
    because 5.1 fixes the geometric mean of speedup factors as the aggregate for
    ratios and percent_saved is a ratio. Averaging percent_saved arithmetically
    is not merely a second convention, it is biased: the quantity is bounded at
    +100 but unbounded below, so a 6x speedup and a 6x slowdown enter as +83 and
    -500 and the mean is pulled against the rewrite by any cheap query that
    doubles in runtime.

    The PERCENTAGE family (mean_*/median_*) is kept because the per-query
    values, the worst-regression labels and the mean gap are stated on that
    scale, and because removing it would silently move numbers 5.3.1 already
    carries. Do not promote it back into an aggregate headline.

    The SECONDS family (realized_fraction_abs) is the runtime-weighted answer to
    "how much of the ceiling was realized" and is immune to cheap-query
    artefacts. It is the one RQ4 should lean on.

    The factor family is reported on the FULL WORKLOAD denominator (113 on JOB),
    not on the 86-query population, and the difference is presentational only.
    RQ4 asks how much of *that* ceiling a selector realizes, and the ceiling the
    chapter states is eval_ceiling.*.geo_workload, the 113-query figure; on this
    denominator geo_ceiling_workload reproduces it exactly, so the figure can be
    laid against Table 5.1 with no conversion. The two denominators are one
    quantity, since log(geo_n) = S/n with S summed over improved queries alone
    and every untouched query contributing log(1) = 0, hence
    geo@86 = geo@113 ** (113/86). Counts and composition stay on the 86, which
    is where a selection decision arises at all (fig_engines_composition)."""
    oracle = load_oracle(key)
    opt = load_optimizer(key)
    transfer = load_transfer(key)
    runtimes = load_all_runtimes(key)
    if oracle is None or opt is None or transfer is None:
        return Missing(RUNS[key])
    # Sorted, not set order: float addition is not associative, so an unordered
    # sum makes the last printed digit depend on PYTHONHASHSEED.
    pop = sorted(set(transfer.keys()) | set(oracle.keys()) | set(opt.keys()))
    ceil_pct, real_pct, gaps = [], [], []
    ceil_fac, real_fac = [], []
    total_ceil_s, total_real_s = 0.0, 0.0
    # Workload seconds for the runtime-weighted factors. These sum the ACTUAL
    # runtimes rather than the seconds saved, and every query of the population
    # enters, including one that neither view touched. That is why the original
    # runtime falls back to load_all_runtimes(): a query with an applicable rule
    # that the oracle never improved and the selector declined appears in
    # neither stats CSV, yet it ran and its seconds belong in a workload total.
    # _impact() drops exactly those queries, which is why its workload
    # percentages sit on a smaller denominator than the ones computed here.
    w_orig = w_ceil = w_real = 0.0
    worst = None
    outside = 0  # oracle/optimizer prefixes not present as a transfer key
    for q in pop:
        if q not in transfer:
            outside += 1
        orig = oracle[q]["orig"] if q in oracle else (opt[q]["orig"] if q in opt else None)
        if orig is None and runtimes is not None:
            # stats CSVs carry the variant suffix, load_all_runtimes does not.
            orig = runtimes.get(_basename(q))
        cp = oracle[q]["pct"] if q in oracle else 0.0
        rp = opt[q]["pct"] if q in opt else 0.0
        ceil_pct.append(cp)
        real_pct.append(rp)
        gaps.append(cp - rp)
        # Factor = t_orig / t_rewrite, the convention of ceiling(). A query that
        # neither view changed enters at 1.0 and contributes log(1) = 0.
        ceil_fac.append(oracle[q]["orig"] / oracle[q]["improved"]
                        if q in oracle and oracle[q]["improved"] > 0 else 1.0)
        real_fac.append(opt[q]["orig"] / opt[q]["improved"]
                        if q in opt and opt[q]["improved"] > 0 else 1.0)
        if orig is not None:
            w_orig += orig
            w_ceil += oracle[q]["improved"] if q in oracle else orig
            w_real += opt[q]["improved"] if q in opt else orig
        if q in oracle:
            total_ceil_s += oracle[q]["orig"] - oracle[q]["improved"]
        if q in opt and orig is not None:
            total_real_s += orig - opt[q]["improved"]
        if rp < 0 and (worst is None or rp < worst[1]):
            worst = (q, rp)
    # A query in the population with no optimizer row means the cost model
    # declined EVERY rule for it (statistics.build_optimizer_cost_rows skips an
    # empty winner list). That is a real decision, not missing data, and it is
    # counted as realized 0.0 above (over the full population, not dropped). It
    # is reported separately because abstention only protects the mean against a
    # regression while lowering it when a gain is passed up, so a heavily
    # abstaining selector can post a competitive mean and a low regression count
    # mainly by avoiding negatives rather than by selecting well.
    n_declined = len([q for q in pop if q not in opt])
    # Widen both factor lists from the population to the full workload. A query
    # with no applicable rule cannot be rewritten under either view, so it is a
    # 1.0 in both and adds its own runtime to all three workload sums unchanged.
    # This is what makes geo_ceiling_workload identical to
    # eval_ceiling.<key>.geo_workload, which do_engines asserts.
    n_workload = len(runtimes) if runtimes else len(pop)
    pad = max(0, n_workload - len(pop))
    pop_bases = {_basename(q) for q in pop}
    outside_s = (sum(rt for b, rt in runtimes.items() if b not in pop_bases)
                 if runtimes else 0.0)
    ceil_fac_w = ceil_fac + [1.0] * pad
    real_fac_w = real_fac + [1.0] * pad
    geo_ceil_w = geomean(ceil_fac_w)
    geo_real_w = geomean(real_fac_w)
    from math import log
    sum_log_ceil = sum(log(f) for f in ceil_fac)
    sum_log_real = sum(log(f) for f in real_fac)
    return {
        "n_pop": len(pop),
        "n_workload": n_workload,
        "n_declined_all": n_declined,
        "n_decided": len(pop) - n_declined,
        "pct_declined": 100.0 * n_declined / len(pop) if pop else float("nan"),
        "n_outside_transfer": outside,
        "median_ceiling": float(median(ceil_pct)),
        "median_realized": float(median(real_pct)),
        "mean_ceiling": float(mean(ceil_pct)),
        "mean_realized": float(mean(real_pct)),
        "median_gap": float(median(gaps)),
        "mean_gap": float(mean(gaps)),
        # Factor family, full-workload denominator. These are what the chapter
        # and fig_engines report.
        "geo_ceiling_workload": geo_ceil_w,
        "geo_realized_workload": geo_real_w,
        # Same quantity on the 86-query population, kept so that a figure or a
        # sentence stating the population denominator has the value to hand
        # without recomputing the exponent.
        "geo_ceiling_pop": geomean(ceil_fac),
        "geo_realized_pop": geomean(real_fac),
        "median_ceiling_factor": float(median(ceil_fac)),
        "median_realized_factor": float(median(real_fac)),
        "min_realized_factor": float(min(real_fac)) if real_fac else float("nan"),
        "max_realized_factor": float(max(real_fac)) if real_fac else float("nan"),
        # Runtime-weighted twins of the two geometric means, on the full
        # workload. The pair is the query-weighted / runtime-weighted contrast of
        # 5.1 and the two can disagree in SIGN, which is the ZeroShot case.
        "wtd_ceiling_workload": ((w_orig + outside_s) / (w_ceil + outside_s)
                                 if w_ceil + outside_s > 0 else float("nan")),
        "wtd_realized_workload": ((w_orig + outside_s) / (w_real + outside_s)
                                  if w_real + outside_s > 0 else float("nan")),
        "workload_pct_realized_full": (100.0 * (w_orig - w_real) / (w_orig + outside_s)
                                       if w_orig + outside_s > 0 else float("nan")),
        # Share of the available log-speedup the selector captures. Being a
        # quotient of two log sums it carries no denominator of its own, so it
        # is identical on 86 and on 113, and it is the query-weighted
        # counterpart of realized_fraction_abs.
        "log_share_of_ceiling": (100.0 * sum_log_real / sum_log_ceil
                                 if sum_log_ceil > 0 else float("nan")),
        "n_improved": sum(1 for x in real_pct if x > 0),
        "n_regress": sum(1 for x in real_pct if x < 0),
        "n_flat": sum(1 for x in real_pct if x == 0),
        "worst_regress": worst,
        "total_ceiling_s": total_ceil_s,
        "total_realized_s": total_real_s,
        "realized_fraction_abs": (100.0 * total_real_s / total_ceil_s) if total_ceil_s > 0 else float("nan"),
    }


def do_oracle():
    section("eval_oracle (DDB) : oracle vs optimizer gap [HEADLINE, unified pop]")
    m = _unified("DDB")
    if isinstance(m, Missing):
        emit("eval_oracle.median_gap", m)
        return
    emit("eval_oracle.n_population", m["n_pop"])
    emit("eval_oracle.n_declined_all", m["n_declined_all"])
    emit("eval_oracle.median_ceiling", m["median_ceiling"])
    emit("eval_oracle.median_realized", m["median_realized"])
    emit("eval_oracle.median_gap", m["median_gap"])
    emit("eval_oracle.mean_gap", m["mean_gap"])
    emit("eval_oracle.n_improved", m["n_improved"])
    emit("eval_oracle.n_regress", m["n_regress"])
    emit("eval_oracle.n_flat", m["n_flat"])
    emit("eval_oracle.pct_regress", 100.0 * m["n_regress"] / m["n_pop"])
    emit("eval_oracle.realized_fraction_abs", m["realized_fraction_abs"])
    print(f"[eval_oracle.geo_ceiling_workload] = {m['geo_ceiling_workload']:.3f}x")
    print(f"[eval_oracle.geo_realized_workload] = {m['geo_realized_workload']:.3f}x")
    print(f"[eval_oracle.median_realized_factor] = {m['median_realized_factor']:.3f}x")
    emit("eval_oracle.log_share_of_ceiling", m["log_share_of_ceiling"])
    if m["worst_regress"]:
        print(f"[eval_oracle.worst_regress] = {m['worst_regress'][0]}: {m['worst_regress'][1]:.1f}%")
    if m["n_outside_transfer"]:
        emit("eval_oracle.WARN_prefixes_outside_transfer", m["n_outside_transfer"])


# ===========================================================================
# eval_joinorder (DDB free vs DDB_JP pinned)
# ===========================================================================
def _spread(stats):
    xs = pct_list(stats)
    return {
        "min": float(min(xs)),
        "max": float(max(xs)),
        "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
        "n_regress": sum(1 for x in xs if x < 0),
        "p5": float(percentile(xs, 5)),
        "p95": float(percentile(xs, 95)),
    }


def do_joinorder():
    """Spread of the speedup distribution, join order free vs pinned, per engine.

    Label scheme: the DuckDB pair keeps the original flat labels
    (eval_joinorder.free.*) for backward compatibility with the chapter; every
    further engine is namespaced (eval_joinorder.pg.free.*)."""
    for ns, free_key, pin_key in JOINORDER_PAIRS:
        section(f"eval_joinorder [{ns}] ({free_key} free vs {pin_key} pinned)")
        pre = "eval_joinorder." if ns == "ddb" else f"eval_joinorder.{ns}."
        for view, loader in (("optimizer", load_optimizer), ("oracle", load_oracle)):
            for tag, key in (("free", free_key), ("pinned", pin_key)):
                stats = loader(key)
                if stats is None:
                    emit(f"{pre}{tag}.{view}.min", Missing(RUNS[key]))
                    continue
                s = _spread(stats)
                emit(f"{pre}{tag}.{view}.min", s["min"])
                emit(f"{pre}{tag}.{view}.max", s["max"])
                emit(f"{pre}{tag}.{view}.std", s["std"])
                emit(f"{pre}{tag}.{view}.n_regress", s["n_regress"])


# ===========================================================================
# eval_engines : per-selector realisation of the ceiling  (Table T2.A)
# ===========================================================================
def do_engines():
    section("eval_engines : per-selector profile (Table T2.A, unified pop)")
    for key in ENGINE_SELECTORS:
        m = _unified(key)
        if isinstance(m, Missing):
            emit(f"eval_engines.{key}.n_improved", m)
            continue
        emit(f"eval_engines.{key}.n_population", m["n_pop"])
        emit(f"eval_engines.{key}.n_decided", m["n_decided"])
        emit(f"eval_engines.{key}.n_declined_all", m["n_declined_all"])
        emit(f"eval_engines.{key}.pct_declined", m["pct_declined"])
        emit(f"eval_engines.{key}.n_improved", m["n_improved"])
        emit(f"eval_engines.{key}.n_regressed", m["n_regress"])
        # Mean AND median per selector, per the aggregation convention of 5.1,
        # and the pair is not decoration here: the median realized speedup is
        # 0.0 on all four selectors, while the means run from +3.1 to -4.2. The
        # spread between the selectors is therefore carried entirely by a tail
        # of few queries, and the typical query in the population is untouched
        # no matter which selector decides. Quoting the mean alone would read as
        # a per-query gain that no median query sees, which is the opposite of
        # the finding. Whichever the prose quotes, it must quote both.
        emit(f"eval_engines.{key}.median_ceiling", m["median_ceiling"])
        emit(f"eval_engines.{key}.median_realized", m["median_realized"])
        emit(f"eval_engines.{key}.mean_ceiling", m["mean_ceiling"])
        emit(f"eval_engines.{key}.mean_realized", m["mean_realized"])
        emit(f"eval_engines.{key}.realized_fraction_abs", m["realized_fraction_abs"])
        # THE REPORTED AGGREGATE. Geometric mean of the per-query speedup
        # factor on the full workload, the same statistic and the same
        # denominator as eval_ceiling.<key>.geo_workload, so the ceiling value
        # here IS the Table 5.1 figure and the two are directly comparable.
        # Quote these, not mean_ceiling/mean_realized: percent_saved is a ratio
        # and 5.1 fixes the geometric mean for ratios.
        print(f"[eval_engines.{key}.geo_ceiling_workload] = "
              f"{m['geo_ceiling_workload']:.3f}x")
        print(f"[eval_engines.{key}.geo_realized_workload] = "
              f"{m['geo_realized_workload']:.3f}x")
        print(f"[eval_engines.{key}.geo_ceiling_pop] = {m['geo_ceiling_pop']:.3f}x")
        print(f"[eval_engines.{key}.geo_realized_pop] = {m['geo_realized_pop']:.3f}x")
        print(f"[eval_engines.{key}.median_realized_factor] = "
              f"{m['median_realized_factor']:.3f}x")
        print(f"[eval_engines.{key}.wtd_realized_workload] = "
              f"{m['wtd_realized_workload']:.3f}x")
        emit(f"eval_engines.{key}.workload_pct_realized_full",
             m["workload_pct_realized_full"])
        emit(f"eval_engines.{key}.log_share_of_ceiling", m["log_share_of_ceiling"])
        emit(f"eval_engines.{key}.n_workload", m["n_workload"])
        # The ceiling on this denominator must reproduce the Part 1 figure
        # exactly. A divergence means the unified population and the ceiling
        # path no longer read the same oracle rows, and every geo_* number in
        # 5.3.1 would then be quoting a ceiling the chapter never stated.
        c = ceiling(key)
        if c is not None:
            drift = abs(m["geo_ceiling_workload"] - c["geo_workload"])
            print(f"[eval_engines.{key}.CHECK_ceiling_matches_part1] = "
                  + ("YES" if drift < 5e-4
                     else f"NO, {m['geo_ceiling_workload']:.4f} vs "
                          f"{c['geo_workload']:.4f}"))
        # Selector accuracy: how often the cost model picks exactly the rule subset
        # the oracle found best. Restricted to queries that HAVE an improving
        # subset and on which the selector actually decided, so neither declining
        # everything nor a query with nothing to win can count as a correct pick.
        w_opt = _winning_subsets(key)
        w_ora = _winning_subsets_oracle(key)
        picks = sorted(set(w_ora) & set(w_opt))
        if picks:
            hit = sum(1 for q in picks if w_ora[q] == w_opt[q])
            emit(f"eval_engines.{key}.n_oracle_optimal_pick", hit)
            emit(f"eval_engines.{key}.n_pick_population", len(picks))
            emit(f"eval_engines.{key}.pct_oracle_optimal_pick",
                 100.0 * hit / len(picks))
        if m["worst_regress"]:
            print(f"[eval_engines.{key}.worst_regress] = "
                  f"{m['worst_regress'][0]}: {m['worst_regress'][1]:.1f}%")


# ===========================================================================
# eval_errors : regression counts + worst offenders per selector
# ===========================================================================
def do_errors():
    section("eval_errors : regressions per selector")
    for key in ENGINE_SELECTORS + ["DDB_JP", "PG_JP"]:
        opt = load_optimizer(key)
        if opt is None:
            emit(f"eval_errors.{key}.n_regress", Missing(RUNS[key]))
            continue
        regress = sorted(
            ((p, v) for p, v in opt.items() if v["pct"] < 0),
            key=lambda kv: kv[1]["pct"],
        )
        emit(f"eval_errors.{key}.n_regress", len(regress))
        emit(f"eval_errors.{key}.n_severe_regress",
             sum(1 for _, v in regress if v["pct"] < -5))
        # Regressions beyond 10 %, counted and named. 5.3.4 says what survives
        # plan pinning on DuckDB, which is the DDB_JP row here; the count was
        # prose ("a handful") because no label carried it. Naming the queries
        # matters as much as counting them, since the count is small enough
        # (n=2 on DDB_JP) that "a handful" overstates it.
        big_reg = [(p, v) for p, v in regress if v["pct"] < -10]
        emit(f"eval_errors.{key}.n_regress_gt10", len(big_reg))
        print(f"[eval_errors.{key}.regress_gt10] = "
              + (", ".join(f"{p}: {v['pct']:.1f}%" for p, v in big_reg)
                 if big_reg else "(none)"))
        # Absolute milliseconds alongside the percentage: a large negative
        # percentage on a sub-10ms query is a ratio artefact, not an impact.
        for i, (p, v) in enumerate(regress[:3], 1):
            lost_ms = 1000.0 * (v["improved"] - v["orig"])
            print(f"[eval_errors.{key}.top_regress.{i}] = {p}: {v['pct']:.1f}% "
                  f"({v['orig']*1000:.1f}ms -> {v['improved']*1000:.1f}ms, "
                  f"{lost_ms:+.1f}ms)")


# ===========================================================================
# eval_explored : random_page_cost 4.0 vs 1.1 (Postgres)
# ===========================================================================
def do_explored():
    section("eval_explored : Postgres random_page_cost 4.0 vs 1.1")
    for tag, key in (("rpc4", "PG"), ("rpc1", "PG_RPC")):
        opt = load_optimizer(key)
        ora = load_oracle(key)
        emit(f"eval_explored.{tag}.optimizer_median",
             float(median(pct_list(opt))) if opt else Missing(RUNS[key]))
        emit(f"eval_explored.{tag}.oracle_median",
             float(median(pct_list(ora))) if ora else Missing(RUNS[key]))
    o4 = load_optimizer("PG")
    o1 = load_optimizer("PG_RPC")
    if o4 and o1:
        emit("eval_explored.optimizer_median_delta",
             float(median(pct_list(o1)) - median(pct_list(o4))))


# ---------------------------------------------------------------------------
# Mirror of stages/refinement.py::_is_failed_entry, which decides which
# generation candidates the refinement loop is run on. It is NOT the negation of
# base-table soundness: an entry also counts as failed if its key carries
# "error" or if summary.outputs_match is False. Both eval_funnel and
# eval_ablation size the refinement request count from this, so it must stay in
# sync with that function.
# ---------------------------------------------------------------------------
def _is_failed_entry(k: str, e: dict) -> bool:
    if "error" in k:
        return True
    summary = e.get("summary")
    if summary is not None and summary.get("outputs_match") is False:
        return True
    bt = e.get("base_table_validation")
    return (not bt.get("all_valid", False)) if isinstance(bt, dict) else False


# ===========================================================================
# eval_ablation (generation breadth) : DDB 4-2 vs 1-1
# ===========================================================================
def do_ablation_gen():
    section("eval_ablation.gen : DuckDB 4-2 vs 1-1")
    for tag, key in (("4_2", "DDB"), ("1_1", "DDB_1_1")):
        oracle = load_oracle(key)
        transfer = load_transfer(key)
        emit(f"eval_ablation.gen.{tag}.n_covered",
             len(oracle) if oracle is not None else Missing(RUNS[key]))
        emit(f"eval_ablation.gen.{tag}.n_with_rules",
             len(transfer) if transfer is not None else Missing(RUNS[key]))
        emit(f"eval_ablation.gen.{tag}.median_yield",
             float(median(pct_list(oracle))) if oracle else Missing(RUNS[key]))
    o42 = load_oracle("DDB")
    o11 = load_oracle("DDB_1_1")
    if o42 and o11 and len(o11):
        emit("eval_ablation.gen.covered_ratio_4_2_over_1_1", 100.0 * len(o42) / len(o11) - 100.0)

    # -----------------------------------------------------------------------
    # Request cost and marginal yield of the two budget knobs.
    #
    # The "4-2" label names TWO different quantities and they must not be
    # multiplied into a single factor: 4 is the number of GENERATION rounds
    # (one request per query per round), 2 is the number of REFINEMENT samples
    # per failed candidate. The refinement request count therefore scales with
    # the number of failures rather than with the round count, which is why the
    # total request ratio is not 4 and not 8 but is measured here.
    # -----------------------------------------------------------------------
    section("eval_ablation.cost / .round / .yield : request cost and marginal yield")

    def _valid(v) -> bool:
        bt = v.get("base_table_validation")
        return isinstance(bt, dict) and bt.get("all_valid") is True

    def _budget_stats(key):
        base = os.path.join(TRANSFER, RUNS[key])
        gp, rp = os.path.join(base, "result.json"), os.path.join(base, "result2.json")
        if not (os.path.isfile(gp) and os.path.isfile(rp)):
            return None
        with open(gp) as fh:
            gen = json.load(fh)
        with open(rp) as fh:
            ref = json.load(fh)
        queries, rounds, per_round = set(), set(), {}
        for k, v in gen.items():
            m = re.match(r"^(.*?)\s+NoMIN", k)
            queries.add(m.group(1) if m else k)
            r = re.search(r"NoMIN_r(\d+)_", k)
            r = r.group(1) if r else "0"
            rounds.add(r)
            c, s = per_round.get(r, (0, 0))
            per_round[r] = (c + 1, s + (1 if _valid(v) else 0))
        n_gen_req = len(queries) * len(rounds)

        # DO NOT read len(result2.json) as a request count. refinement.py issues
        # `samples` requests per failed candidate (its line "for m in range(samples)")
        # but writes a DEDUPLICATED dict: with samples > 1 it keeps at most one
        # non-rule outcome per candidate and collapses refined rules by exact
        # signature per SQL. On the 4-2 run that turns 1416 issued requests into
        # 837 retained entries, which is why the retained count is odd. With
        # samples == 1 the dedup branches are inactive and the two coincide.
        n_ref_samples = 1
        for k in ref:
            m = re.search(r"_rerun_s(\d+)$", k)
            if m:
                n_ref_samples = max(n_ref_samples, int(m.group(1)) + 1)
        n_failed = sum(1 for k, v in gen.items() if _is_failed_entry(k, v))
        n_ref_req = n_ref_samples * n_failed

        # Entries that survived dedup and carry a rule. This is a count of
        # DISTINCT refined rules, not of requests that returned something, so it
        # must never serve as the denominator of a per-request rate.
        distinct_returned = [k for k, v in ref.items() if v.get("status") != "no_rule_returned"]
        sound_rules = [k for k in distinct_returned if _valid(ref[k])]
        rec = {re.sub(r"_rerun(_s\d+)?$", "", k) for k in sound_rules}
        return {
            "n_queries": len(queries), "n_rounds": len(rounds), "per_round": per_round,
            "n_cand": len(gen), "n_sound": sum(1 for v in gen.values() if _valid(v)),
            "n_failed": n_failed, "n_ref_samples": n_ref_samples,
            "n_gen_req": n_gen_req, "n_ref_req": n_ref_req,
            "n_ref_entries_retained": len(ref),
            "n_req_total": n_gen_req + n_ref_req,
            "n_distinct_returned": len(distinct_returned),
            "n_sound_rules": len(sound_rules), "n_recovered": len(rec),
        }

    st = {}
    for tag, key in (("4_2", "DDB"), ("1_1", "DDB_1_1")):
        s = _budget_stats(key)
        if s is None:
            emit(f"eval_ablation.cost.{tag}.n_requests_total", Missing(RUNS[key]))
            continue
        st[tag] = s
        emit(f"eval_ablation.cost.{tag}.n_generation_rounds", s["n_rounds"])
        emit(f"eval_ablation.cost.{tag}.n_generation_requests", s["n_gen_req"])
        emit(f"eval_ablation.cost.{tag}.n_refinement_samples", s["n_ref_samples"])
        emit(f"eval_ablation.cost.{tag}.n_failed_candidates", s["n_failed"])
        emit(f"eval_ablation.cost.{tag}.n_refinement_requests", s["n_ref_req"])
        emit(f"eval_ablation.cost.{tag}.n_refinement_entries_retained",
             s["n_ref_entries_retained"])
        emit(f"eval_ablation.cost.{tag}.n_requests_total", s["n_req_total"])
        # The only comparison the two stages admit per request is sound rules per
        # issued request. Rates over retained entries are not comparable across
        # the two budgets, because dedup applies at 4-2 and not at 1-1.
        emit(f"eval_ablation.yield.{tag}.pct_generation_sound",
             100.0 * s["n_sound"] / s["n_cand"])
        # Reported per 100 requests, because the per-request figures round to
        # 0.7 and 0.0 and would hide the very gap they are there to show.
        emit(f"eval_ablation.yield.{tag}.sound_rules_per_100_generation_requests",
             100.0 * s["n_sound"] / s["n_gen_req"] if s["n_gen_req"] else float("nan"))
        emit(f"eval_ablation.yield.{tag}.sound_rules_per_100_refinement_requests",
             100.0 * s["n_sound_rules"] / s["n_ref_req"] if s["n_ref_req"] else float("nan"))
        emit(f"eval_ablation.yield.{tag}.pct_refine_recovered_per_failed_candidate",
             100.0 * s["n_recovered"] / s["n_failed"] if s["n_failed"] else float("nan"))
        emit(f"eval_ablation.yield.{tag}.n_recovered", s["n_recovered"])
        emit(f"eval_ablation.yield.{tag}.n_sound_refined_rules", s["n_sound_rules"])
        if s["n_ref_req"]:
            emit(f"eval_ablation.yield.{tag}.gen_over_refine_request_factor",
                 (s["n_sound"] / s["n_gen_req"]) / (s["n_sound_rules"] / s["n_ref_req"])
                 if s["n_sound_rules"] else float("nan"))
    if "4_2" in st and "1_1" in st and st["1_1"]["n_req_total"]:
        emit("eval_ablation.cost.request_ratio_4_2_over_1_1",
             st["4_2"]["n_req_total"] / st["1_1"]["n_req_total"])

    # Per-round RETAINED entry counts (332, 267, 230, 213). Kept for provenance
    # only. These are NOT production volume: generation.py drops any candidate
    # whose signature was already kept for that query in an earlier round, so the
    # sequence is a novelty count under one arbitrary order of exchangeable
    # rounds. The chapter quotes the permutation-averaged curves from do_novelty()
    # instead, and an earlier draft that read this sequence as the model producing
    # less per round was wrong. Do not reintroduce that reading.
    if "4_2" in st:
        for r in sorted(st["4_2"]["per_round"]):
            c, s_ = st["4_2"]["per_round"][r]
            emit(f"eval_ablation.round.{r}.n_candidates", c)
            emit(f"eval_ablation.round.{r}.n_sound", s_)
            emit(f"eval_ablation.round.{r}.pct_sound", 100.0 * s_ / c)


# ===========================================================================
# eval_novelty : what each additional generation round actually adds
#
# READS THE LLM CACHE, NOT result.json. This is deliberate and the reason the
# function is this long.
#
# WHY. The per-round entry counts in result.json (332, 267, 230, 213) are NOT
# production volume and must never be read as the model running dry. Two
# distortions sit between the model and that file:
#   1. stages/generation.py issues the rounds as INDEPENDENT requests (identical
#      prompt, only `seed` differs, no history) and then drops any rule whose
#      rule_signature was already kept FOR THAT QUERY in an earlier round. The
#      declining sequence is therefore a novelty count under an arbitrary round
#      order, not a falling output rate.
#   2. A response that fails to parse still writes one `_error` entry, so
#      len(result.json) counts 19 non-rules as rules on this run.
# The raw model output survives only in llm_cache/, which holds every response
# keyed by the hash of its request. Reconstructing the requests and looking them
# up there recovers all 452 responses, including the 274 duplicate rules that
# generation.py discarded and the 45 rules inside responses that left no entry
# at all. An earlier version of this function reconstructed from the
# `original_output` field of result.json instead, which silently lost those 45
# rules and put the emitted count at 1252 instead of 1297. Do not go back to it.
#
# Because the rounds are exchangeable, a single order is meaningless. Every
# accumulation figure is averaged over all 4! = 24 round orders and reported with
# the min and max across those orders as a band. The k = n_rounds row is
# order-independent by construction, which is why only it comes out integral.
# ===========================================================================
def _openai_cache_path(cache_dir: str, request: dict) -> str:
    """Path of the cache entry for `request`.

    Mirrors llm_helpers.llms.BaseProvider.compute_hash with prefix
    "execute-request" and provider name "openai". Key ORDER inside `request`
    is part of the hash, so the caller must build the dict in the same order
    llm_helpers.run.construct_request_dummy does.
    """
    import hashlib
    blob = f"execute-request-openai-{json.dumps(request)}"
    return os.path.join(cache_dir, hashlib.sha256(blob.encode("utf-8")).hexdigest() + ".json")


def _generation_request(model: str, system_prompt: str, user_message: str, seed):
    """Mirror of construct_request_dummy(...) + inject_seed(...) for gpt-5 models.

    Kept as a local mirror so this script stays importable without the LLM stack.
    Drift is caught rather than absorbed: every lookup built from this must hit
    the cache, and do_novelty() reports MISSING if any does not.
    """
    request = {
        "model": model,
        "messages": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_completion_tokens": 10000,
    }
    if seed is not None:
        request["seed"] = seed
    return request


_NOVELTY_CURVES: dict = {}


def novelty_curves() -> dict:
    """{name: [(mean, min, max), ...]} per k, for "distinct", "sound", "queries".

    Populated as a side effect of do_novelty(), which is the single place the
    cache is read. Returns {} when the cache or the run directory is unavailable,
    in which case the caller must draw a placeholder rather than invent data.
    """
    if not _NOVELTY_CURVES:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            do_novelty()
    return _NOVELTY_CURVES


def do_novelty():
    section("eval_novelty (DDB 4-2) : marginal yield of an additional round")
    import itertools
    import sys

    repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from systematic_eval.parsing import parse_llm_json
        from systematic_eval.prompt_loader import PromptLoader
        from systematic_eval.stages.sampling import rule_signature
    except Exception as exc:
        emit("eval_novelty", Missing(f"cannot import project modules: {exc}"))
        return

    cache_dir = os.path.join(repo_root, "llm_cache")
    if not os.path.isdir(cache_dir):
        emit("eval_novelty", Missing(cache_dir))
        return

    # Run parameters. Hardcoded to the DDB run's config rather than parsed from
    # the YAML, so that this function cannot silently follow a config edit and
    # start reporting a different run than the rest of the chapter.
    model, n_rounds = "gpt-5.4-2026-03-05", 4
    dataset, sys_prompt_name, gen_prompt_name = "imdb_job", "prompt01", "prompt12"
    excluded = {"fkindexes.sql", "schema.sql"}

    sql_dir = os.path.join(repo_root, "sql", "job")
    if not os.path.isdir(sql_dir):
        emit("eval_novelty", Missing(sql_dir))
        return
    # parse_sqls with strip_min=True: MIN() wrappers removed, " NoMIN" suffix.
    sqls = {}
    for fn in sorted(f for f in os.listdir(sql_dir) if f.endswith(".sql") and f not in excluded):
        with open(os.path.join(sql_dir, fn), encoding="utf-8") as fh:
            sqls[fn.split(".sql")[0] + " NoMIN"] = re.sub(r"MIN\((.*?)\)", r"\1", fh.read())

    prompts = PromptLoader(dataset)
    system_prompt = prompts.load_system_prompt(sys_prompt_name)
    template = prompts.load_generation_prompt(gen_prompt_name)
    rule_example = prompts.load_rule_example()

    queries = sorted(sqls)
    rounds = list(range(n_rounds))

    # ---- fetch every response from the cache -----------------------------
    responses, n_missing = {}, 0
    for q in queries:
        message = template.format(sql=sqls[q], rule_example=rule_example)
        for r in rounds:
            path = _openai_cache_path(cache_dir, _generation_request(model, system_prompt, message, r))
            if not os.path.isfile(path):
                n_missing += 1
                continue
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            try:
                responses[(q, r)] = cached["response"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                n_missing += 1
    n_expected = len(queries) * n_rounds
    if n_missing:
        # A partial hit means the mirrored request construction has drifted from
        # llm_helpers, so every figure below would be computed on a biased subset.
        # Refuse rather than report.
        emit("eval_novelty", Missing(
            f"{n_missing} of {n_expected} generation responses not in {cache_dir}; "
            "request construction likely drifted from llm_helpers.run"))
        return
    emit("eval_novelty.n_responses", len(responses))

    # ---- parse: raw model output, nothing deduplicated -------------------
    emitted = {}          # (query, round) -> [signature] with duplicates kept
    n_parse_fail = 0
    for (q, r), text in responses.items():
        parsed = parse_llm_json(text)
        if not isinstance(parsed, dict) or "error" in parsed:
            n_parse_fail += 1
            emitted[(q, r)] = []
            continue
        sigs = []
        for rule in parsed.get("determined_ruleset") or []:
            try:
                sigs.append(rule_signature(rule))
            except Exception:
                continue
        emitted[(q, r)] = sigs

    n_emitted = sum(len(v) for v in emitted.values())
    pairs = {(q, s) for (q, r), sigs in emitted.items() for s in sigs}
    global_sigs = {s for sigs in emitted.values() for s in sigs}
    emit("eval_novelty.n_rules_emitted", n_emitted)
    emit("eval_novelty.n_responses_unparseable", n_parse_fail)
    emit("eval_novelty.n_distinct_pairs", len(pairs))
    # Generation dedup is PER QUERY, so the same rule proposed for two queries
    # survives twice. The chapter must not call n_distinct_pairs "distinct rules".
    emit("eval_novelty.n_distinct_signatures_global", len(global_sigs))
    emit("eval_novelty.n_dropped_by_generation_dedup", n_emitted - len(pairs))
    emit("eval_novelty.redundancy_factor", n_emitted / len(pairs) if pairs else float("nan"))

    # Production per request: the direct test of whether later rounds return less.
    for r in rounds:
        nresp = sum(1 for (_, rr) in responses if rr == r)
        nrules = sum(len(emitted[(q, r)]) for q in queries)
        emit(f"eval_novelty.production.r{r}.n_rules_emitted", nrules)
        emit(f"eval_novelty.production.r{r}.rules_per_response",
             nrules / nresp if nresp else float("nan"))

    # ---- soundness, taken from the validated run -------------------------
    # Only the deduplicated survivors were ever validated. A dropped duplicate has
    # the same signature, hence the same rewrite, hence the same outcome, so the
    # map from the retained entries covers every distinct pair.
    base = os.path.join(TRANSFER, RUNS["DDB"])
    gen_path = os.path.join(base, "result.json")
    if not os.path.isfile(gen_path):
        emit("eval_novelty.accum", Missing(gen_path))
        return
    with open(gen_path) as fh:
        gen_d = json.load(fh)
    sound = set()
    for k, v in gen_d.items():
        m = re.match(r"^(.*?)\s+NoMIN_r(\d+)_", k)
        if not m:
            continue
        bt = v.get("base_table_validation")
        if not (isinstance(bt, dict) and bt.get("all_valid") is True):
            continue
        resp_rule = v.get("response")
        if isinstance(resp_rule, dict) and ("requires" in resp_rule or "implies" in resp_rule):
            sound.add((m.group(1) + " NoMIN", rule_signature(resp_rule)))
    emit("eval_novelty.n_sound_pairs", len(sound))

    # ---- accumulation curves, averaged over all round orders -------------
    def _accumulate(collect):
        curves = []
        for order in itertools.permutations(rounds):
            seen, row = set(), []
            for r in order:
                for q in queries:
                    for s in emitted[(q, r)]:
                        collect(seen, q, s)
                row.append(len(seen))
            curves.append(row)
        return curves

    def _emit_curve(name, curves):
        prev, cum_rows, marg_rows = 0.0, [], []
        for i in range(n_rounds):
            vals = [c[i] for c in curves]
            m = mean(vals)
            emit(f"eval_novelty.accum.{name}.k{i + 1}.mean", float(m))
            emit(f"eval_novelty.accum.{name}.k{i + 1}.min", min(vals))
            emit(f"eval_novelty.accum.{name}.k{i + 1}.max", max(vals))
            emit(f"eval_novelty.accum.{name}.k{i + 1}.marginal", float(m - prev))
            # Spread of the MARGINAL, which is not the spread of the cumulative:
            # it is taken per ordering first and only then reduced, because the
            # k-th round of a lucky ordering can be small while its running total
            # is large. Drawn as the whisker in fig_novelty.
            deltas = [c[i] - (c[i - 1] if i else 0) for c in curves]
            emit(f"eval_novelty.accum.{name}.k{i + 1}.marginal_min", min(deltas))
            emit(f"eval_novelty.accum.{name}.k{i + 1}.marginal_max", max(deltas))
            prev = m
            cum_rows.append((float(m), min(vals), max(vals)))
            marg_rows.append((float(mean(deltas)), min(deltas), max(deltas)))
        # Stashed so fig_novelty can draw exactly the emitted numbers instead of
        # recomputing them from a second copy of the cache-reading logic.
        _NOVELTY_CURVES[name] = {"cum": cum_rows, "marg": marg_rows}

    def _add_pair(seen, q, s):
        seen.add((q, s))

    def _add_sound_pair(seen, q, s):
        if (q, s) in sound:
            seen.add((q, s))

    def _add_covered_query(seen, q, s):
        if (q, s) in sound:
            seen.add(q)

    _emit_curve("distinct", _accumulate(_add_pair))
    _emit_curve("sound", _accumulate(_add_sound_pair))
    _emit_curve("queries", _accumulate(_add_covered_query))

    # Share of each round's NEW candidates that validates. Flat across rounds
    # means the later rounds are not proposing worse rules, only fewer new ones,
    # which is what separates "the pool is thinning" from "the model is degrading".
    cand, snd = _NOVELTY_CURVES["distinct"]["marg"], _NOVELTY_CURVES["sound"]["marg"]
    for i in range(n_rounds):
        if cand[i][0]:
            emit(f"eval_novelty.pct_marginal_validating.k{i + 1}",
                 100.0 * snd[i][0] / cand[i][0])

    # ---- incidence and Chao2 --------------------------------------------
    incidence = {}
    for (q, r), sigs in emitted.items():
        for s in set(sigs):
            incidence[(q, s)] = incidence.get((q, s), 0) + 1
    freq = {}
    for c in incidence.values():
        freq[c] = freq.get(c, 0) + 1
    s_obs = len(incidence)
    for c in sorted(freq):
        emit(f"eval_novelty.pct_in_exactly_{c}_rounds", 100.0 * freq[c] / s_obs)
    # Chao2, incidence-based. A LOWER bound that assumes homogeneous detection
    # across rules, so the chapter quotes it as a bound and never as the truth.
    q1, q2 = freq.get(1, 0), freq.get(2, 0)
    if q2:
        chao2 = s_obs + ((n_rounds - 1) / n_rounds) * (q1 * q1) / (2.0 * q2)
        emit("eval_novelty.chao2_lower_bound", float(chao2))
        emit("eval_novelty.pct_of_chao2_recovered", 100.0 * s_obs / chao2)


# ===========================================================================
# eval_generality : SECONDARY foundation (non-c07 dataset oracle runs, % only)
# ===========================================================================
def do_generality():
    section("eval_generality (SECONDARY foundation, oracle ceiling only)")
    rows = generality_rows()
    if not rows:
        emit("eval_generality.n_workloads", Missing(os.path.join(SAVED, "experiment_T_gen_*")))
        return

    # Per workload, both denominators, in the same quantities as eval_ceiling.
    for r in rows:
        n = r["name"]
        emit(f"eval_generality.{n}.n_improved", r["n_improved"])
        emit(f"eval_generality.{n}.n_workload", r["n_workload"])
        print(f"[eval_generality.{n}.geo_improved] = {r['geo_improved']:.3f}x")
        print(f"[eval_generality.{n}.geo_workload] = {r['geo_workload']:.3f}x")
        print(f"[eval_generality.{n}.workload_factor] = {r['workload_factor']:.3f}x")

    # The IMDB row is the primary DuckDB run re-entered as the twentieth
    # workload. It has to reproduce eval_ceiling.DDB exactly; a mismatch means
    # the two paths have diverged, so it is checked rather than trusted.
    imdb = next((r for r in rows if r["key"] == GEN_IMDB_KEY), None)
    ddb = ceiling("DDB")
    if imdb and ddb:
        ok = (abs(imdb["geo_improved"] - ddb["geo_improved"]) < 5e-4
              and abs(imdb["geo_workload"] - ddb["geo_workload"]) < 5e-4)
        print(f"[eval_generality.imdb_matches_eval_ceiling] = {'YES' if ok else 'NO -- PATHS DIVERGED'}")

    emit("eval_generality.n_workloads", len(rows))
    emit("eval_generality.n_further_datasets", len(rows) - (1 if imdb else 0))
    gw = [r["geo_workload"] for r in rows]
    gi = [r["geo_improved"] for r in rows]
    print(f"[eval_generality.median_geo_workload] = {median(gw):.3f}x")
    print(f"[eval_generality.min_geo_workload] = {min(gw):.3f}x")
    print(f"[eval_generality.max_geo_workload] = {max(gw):.3f}x")
    print(f"[eval_generality.median_geo_improved] = {median(gi):.3f}x")
    if imdb:
        # IMDB's position in the field is the whole point of the sweep, so the
        # rank is emitted rather than left to be read off the figure.
        emit("eval_generality.imdb_rank_geo_workload",
             sorted(gw, reverse=True).index(imdb["geo_workload"]) + 1)
        emit("eval_generality.n_above_imdb_geo_workload",
             sum(1 for v in gw if v > imdb["geo_workload"]))

    # The runtime-weighted counterpart is where the sweep is weakest, so it is
    # emitted with the count that names the weakness instead of only the median.
    rt = [r["workload_factor"] for r in rows]
    print(f"[eval_generality.median_workload_factor] = {median(rt):.3f}x")
    emit("eval_generality.n_workload_factor_below_1_01",
         sum(1 for v in rt if v < 1.01))
    # Sign is definitional: the oracle keeps the fastest subset and therefore
    # never selects a regression, so geo_workload > 1.0 on every workload is a
    # property of the oracle, NOT a finding. Emitted only so the chapter can
    # state it as the tautology it is rather than claim it as a result.
    emit("eval_generality.n_geo_workload_above_1",
         sum(1 for v in gw if v > 1.0))
    # The count that carries the "large" qualifier. Sign is definitional (above),
    # so any claim stronger than "a ceiling exists" has to name a magnitude
    # threshold and the number of workloads clearing it.
    emit("eval_generality.n_geo_workload_at_least_1_05",
         sum(1 for v in gw if v >= 1.05))

    # ---- why the runtime-weighted factor is near 1.0 on some workloads ----
    # The tempting explanation, "those workloads are cheap", cannot be right:
    # the factor is a ratio and does not shrink with absolute runtime. The
    # actual variable is WHERE the wins sit inside the workload. Both candidate
    # explanations are emitted side by side so the chapter argues from the
    # stronger one rather than from the plausible-sounding one.
    section("eval_generality_composition : where the wins sit")
    shares, med_rts = [], []
    for r in rows:
        sh = generality_sec_share(r["key"])
        pool = generality_pool(r["key"]) or []
        mrt = median([o for o, _ in pool]) if pool else float("nan")
        shares.append(sh)
        med_rts.append(mrt)
        emit(f"eval_generality_composition.{r['name']}.sec_share_improved", sh)
        print(f"[eval_generality_composition.{r['name']}.median_query_ms] = {1000 * mrt:.1f}")
    print("[eval_generality_composition.spearman_secshare_vs_workload_factor] = "
          f"{spearman(shares, rt):.3f}")
    print("[eval_generality_composition.spearman_medianruntime_vs_workload_factor] = "
          f"{spearman(med_rts, rt):.3f}")

    # ---- does query cost govern whether, or how much, a rule helps? ----
    # Pooled over every query of every workload, binned by original runtime.
    # Two separate questions that answer differently, so they must never be
    # collapsed into one claim: whether an improving subset EXISTS, and how
    # large the speedup IS where one does.
    section("eval_generality_cost : query cost versus improvability")
    BINS = [(0.0, 0.002, "lt2ms"), (0.002, 0.005, "2_5ms"), (0.005, 0.010, "5_10ms"),
            (0.010, 0.025, "10_25ms"), (0.025, 0.050, "25_50ms"),
            (0.050, 0.100, "50_100ms"), (0.100, 0.250, "100_250ms"),
            (0.250, 1.0, "250ms_1s"), (1.0, float("inf"), "gt1s")]
    allq = [qr for r in rows for qr in (generality_pool(r["key"]) or [])]
    emit("eval_generality_cost.n_queries_pooled", len(allq))
    for lo, hi, lbl in BINS:
        sel = [(o, f) for o, f in allq if lo <= o < hi]
        if not sel:
            continue
        imp = [f for _, f in sel if f]
        emit(f"eval_generality_cost.{lbl}.n", len(sel))
        emit(f"eval_generality_cost.{lbl}.pct_improved", 100.0 * len(imp) / len(sel))
        if imp:
            print(f"[eval_generality_cost.{lbl}.geo_improved] = {geomean(imp):.3f}x")

    # Within-workload rank correlations, so that a workload-level confound
    # (cheap workloads happening to be low-yield for unrelated reasons) cannot
    # manufacture the pooled trend. The per-workload sign counts carry more
    # weight than the medians: they are what makes the direction a finding.
    c_exist, c_mag = [], []
    for r in rows:
        pool = generality_pool(r["key"]) or []
        if not pool:
            continue
        c_exist.append(spearman([o for o, _ in pool],
                                [1.0 if f else 0.0 for _, f in pool]))
        imp = [(o, f) for o, f in pool if f]
        if len(imp) >= 3:
            c_mag.append(spearman([o for o, _ in imp], [f for _, f in imp]))
    print(f"[eval_generality_cost.median_spearman_runtime_vs_improvable] = {median(c_exist):.3f}")
    emit("eval_generality_cost.n_workloads_positive_runtime_vs_improvable",
         sum(1 for x in c_exist if x > 0))
    emit("eval_generality_cost.n_workloads_runtime_vs_improvable", len(c_exist))
    print(f"[eval_generality_cost.median_spearman_runtime_vs_speedup] = {median(c_mag):.3f}")
    emit("eval_generality_cost.n_workloads_negative_runtime_vs_speedup",
         sum(1 for x in c_mag if x < 0))
    emit("eval_generality_cost.n_workloads_runtime_vs_speedup", len(c_mag))

    # ---- robustness: is the ceiling sub-millisecond measurement noise? ----
    # JOB has almost no cheap queries, so IMDB barely moves under a floor while
    # the generated workloads deflate. IMDB's rank under each floor is the point:
    # the conservative reading is the one that favours the chapter, which is why
    # it is stated rather than omitted.
    section("eval_generality_floor : ceiling with cheap queries excluded")
    for floor_s, lbl in ((0.010, "10ms"), (0.050, "50ms")):
        vals, imdb_v = [], None
        for r in rows:
            f = generality_floor(r["key"], floor_s)
            if f is None:
                continue
            vals.append(f["geo_workload"])
            emit(f"eval_generality_floor.{lbl}.{r['name']}.n", f["n"])
            print(f"[eval_generality_floor.{lbl}.{r['name']}.geo_workload] = {f['geo_workload']:.3f}x")
            if r["key"] == GEN_IMDB_KEY:
                imdb_v = f["geo_workload"]
        print(f"[eval_generality_floor.{lbl}.median_geo_workload] = {median(vals):.3f}x")
        if imdb_v is not None:
            print(f"[eval_generality_floor.{lbl}.imdb_geo_workload] = {imdb_v:.3f}x")
            emit(f"eval_generality_floor.{lbl}.imdb_rank",
                 sorted(vals, reverse=True).index(imdb_v) + 1)

    # ---- coverage, the exponent linking the two denominators ----
    section("eval_generality_coverage")
    covs = [100.0 * r["n_improved"] / r["n_workload"] for r in rows]
    emit("eval_generality_coverage.median_pct", median(covs))
    if imdb:
        ic = 100.0 * imdb["n_improved"] / imdb["n_workload"]
        emit("eval_generality_coverage.imdb_pct", ic)
        emit("eval_generality_coverage.imdb_rank",
             sorted(covs, reverse=True).index(ic) + 1)


# ===========================================================================
# eval_casestudies : reproducibly surface plan-level examples for 5.2
# Selection is data-driven (free-vs-pinned divergence), never hand-picked.
# For each candidate the exact plan_comparisons PNG path(s) are printed so the
# figures can be sourced without guessing.
# ===========================================================================
def _plan_pngs(run_key: str, prefix: str, kind: str) -> list[str]:
    """Return plan_comparisons PNGs for a query. kind in {optimizer, oracle}."""
    qdir = os.path.join(_run_dir(run_key), "plan_comparisons", prefix.replace(" ", "_"))
    if not os.path.isdir(qdir):
        return []
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(qdir, f"{kind}__*.png")))


def _emit_case(cat: str, q: str, f: float, p: float) -> None:
    print(f"[eval_casestudies.{cat}] = {q}: free={f:.1f}% pinned={p:.1f}% (delta={p - f:+.1f})")
    for run_key, tag in (("DDB", "free"), ("DDB_JP", "pinned")):
        pngs = _plan_pngs(run_key, q, "optimizer")
        rel = os.path.join(RUNS[run_key], "plan_comparisons", q.replace(" ", "_"))
        if pngs:
            print(f"    {tag}: {rel}/{pngs[0]}")
        else:
            print(f"    {tag}: NO PLAN PNG in {rel}")


def _subsets_from(path: str) -> dict[str, str]:
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["prefix"]] = r["winning_rules"].strip()
    return out


def _winning_subsets(run_key: str) -> dict[str, str]:
    """The optimizer view's cost-winning rule subset per query."""
    return _subsets_from(os.path.join(_run_dir(run_key),
                                      "rule_summary_result_stats.csv"))


def _winning_subsets_oracle(run_key: str) -> dict[str, str]:
    """The oracle's best-measured rule subset per query."""
    return _subsets_from(os.path.join(_run_dir(run_key), "oracle_stats.csv"))


def do_attribution():
    """Join-order attribution per engine (needs the matching pinned run).

    CLEAN-COMPARISON PRECONDITION. free-minus-pinned isolates join order only
    where the cost-winning rule SUBSET is identical in both runs. Pinning changes
    the estimated cost of every candidate plan, so the argmin over subsets can
    move; where it does, the difference confounds join order with a different set
    of injected predicates. Those queries are therefore EXCLUDED from the
    attribution and reported separately as a second, distinct mechanism
    (pinning-induced selection change). On DuckDB the confounded set is empty, so
    the flat eval_attribution.* labels are unaffected by this split."""
    for ns, free_key, pin_key in JOINORDER_PAIRS:
        section(f"eval_attribution [{ns}] : how much is join-order? "
                f"({free_key} free vs {pin_key} pinned)")
        pre = "eval_attribution." if ns == "ddb" else f"eval_attribution.{ns}."
        free = load_optimizer(free_key)
        pin = load_optimizer(pin_key)
        if free is None or pin is None:
            emit(f"{pre}n_both", Missing(f"{free_key} or {pin_key}"))
            continue

        wf, wp = _winning_subsets(free_key), _winning_subsets(pin_key)
        both = [(q, free[q]["pct"], pin[q]["pct"]) for q in free if q in pin]
        trips = [t for t in both if wf.get(t[0]) == wp.get(t[0])]
        confounded = [t for t in both if wf.get(t[0]) != wp.get(t[0])]

        emit(f"{pre}n_both", len(both))
        emit(f"{pre}subset_mismatch_count", len(confounded))
        emit(f"{pre}n_clean", len(trips))
        if confounded:
            emit(f"{pre}WARN_confounded_queries", len(confounded))

        def mag(x):  # regression magnitude (0 if not a regression)
            return -x if x < 0 else 0.0

        for T in (5, 10):
            reg = [t for t in trips if t[1] < -T]
            # Regression more than halved by pinning => join order dominates.
            halved = [t for t in reg if mag(t[2]) < 0.5 * mag(t[1])]
            # Regression essentially gone when pinned (<2pp residual).
            gone = [t for t in reg if t[2] > -2]
            emit(f"{pre}reg_gt{T}.n", len(reg))
            emit(f"{pre}reg_gt{T}.n_halved_by_pinning", len(halved))
            emit(f"{pre}reg_gt{T}.n_fully_resolved", len(gone))

            win = [t for t in trips if t[1] > T]
            survive = [t for t in win if t[2] > T]   # genuine selectivity win
            vanish = [t for t in win if t[2] <= 0]   # win was join-order luck
            emit(f"{pre}win_gt{T}.n", len(win))
            emit(f"{pre}win_gt{T}.n_survives_pinning", len(survive))
            emit(f"{pre}win_gt{T}.n_vanishes_pinning", len(vanish))

        # Sign flips purely from join order (clean population).
        emit(f"{pre}n_flip_regress_to_ok",
             sum(1 for _, f, p in trips if f < 0 and p >= 0))
        emit(f"{pre}n_flip_win_to_regress",
             sum(1 for _, f, p in trips if f > 5 and p < 0))
        # Aggregate regression mass (summed pp) removed by pinning.
        tot_free = sum(mag(f) for _, f, _ in trips)
        tot_pin = sum(mag(p) for _, _, p in trips)
        emit(f"{pre}regress_mass_pp_free", float(tot_free))
        emit(f"{pre}regress_mass_pp_pinned", float(tot_pin))
        if tot_free > 0:
            emit(f"{pre}pct_regress_mass_removed", 100.0 * (1 - tot_pin / tot_free))
        emit(f"{pre}n_severe_free", sum(1 for _, f, _ in trips if f < -5))
        emit(f"{pre}n_severe_pinned", sum(1 for _, _, p in trips if p < -5))

        # Second mechanism: pinning moved the cost argmin. Not join-order
        # attribution, but a real effect and reported as its own quantity.
        if confounded:
            cf_free = sum(mag(f) for _, f, _ in confounded)
            cf_pin = sum(mag(p) for _, _, p in confounded)
            emit(f"{pre}selchange.n", len(confounded))
            emit(f"{pre}selchange.regress_mass_pp_free", float(cf_free))
            emit(f"{pre}selchange.regress_mass_pp_pinned", float(cf_pin))
            emit(f"{pre}selchange.n_severe_avoided",
                 sum(1 for _, f, p in confounded if f < -5 and p >= -5))
            for q, f, p in sorted(confounded, key=lambda t: t[1]):
                print(f"[{pre}selchange.q] = {q}: free={f:.1f}% pinned={p:.1f}% "
                      f"(subset changed)")
            # Unbereinigt: what the naive read over all `both` would have claimed.
            all_free = sum(mag(f) for _, f, _ in both)
            all_pin = sum(mag(p) for _, _, p in both)
            if all_free > 0:
                emit(f"{pre}UNCLEAN_pct_regress_mass_removed",
                     100.0 * (1 - all_pin / all_free))


def do_casestudies():
    section("eval_casestudies : plan-level examples for 5.2 (free vs pinned)")
    free = load_optimizer("DDB")
    pin = load_optimizer("DDB_JP")
    if free is None or pin is None:
        emit("eval_casestudies.join_order_regression", Missing("DDB or DDB_JP"))
        return
    both = [(q, free[q]["pct"], pin[q]["pct"]) for q in free if q in pin]

    # (A) Regression removed by pinning => proves join-order is the cause.
    reg_fixed = sorted(
        [(q, f, p) for q, f, p in both if f < -5 and p >= -5],
        key=lambda x: x[2] - x[1], reverse=True,
    )
    # (B) Win removed by pinning => the speedup was join-order luck, not selectivity.
    win_fragile = sorted(
        [(q, f, p) for q, f, p in both if f > 5 and p <= 0],
        key=lambda x: x[1] - x[2], reverse=True,
    )
    # (C) Win survives pinning => genuine predicate-selectivity gain (positive control).
    win_robust = sorted(
        [(q, f, p) for q, f, p in both if f > 5 and p > 5],
        key=lambda x: x[2], reverse=True,
    )
    # (D) Regression survives pinning => NOT a join-order artefact (mis-estimation).
    reg_persist = sorted(
        [(q, f, p) for q, f, p in both if f < -5 and p < -5],
        key=lambda x: x[2],
    )

    for cat, items in (
        ("join_order_regression", reg_fixed),
        ("win_fragile_joinorder", win_fragile),
        ("win_robust_selectivity", win_robust),
        ("regression_not_joinorder", reg_persist),
    ):
        if not items:
            print(f"[eval_casestudies.{cat}] = (none)")
        for q, f, p in items[:3]:
            _emit_case(cat, q, f, p)


# ===========================================================================
# eval_soundness : empirical correctness across all executed rewrites
# ===========================================================================
def do_soundness():
    section("eval_soundness : output equality across all executed subsets")
    for key in ("DDB", "UMB", "PG", "LRN", "PG_JP"):
        path = os.path.join(_run_dir(key), "rule_summary_result.json")
        if not os.path.isfile(path):
            emit(f"eval_soundness.{key}.n_subsets", Missing(RUNS[key]))
            continue
        with open(path) as fh:
            d = json.load(fh)
        ok = bad = 0
        for e in d.values():
            for s in (e.get("per_subset_results") or []):
                om = s.get("summary", {}).get("outputs_match")
                if om is True:
                    ok += 1
                elif om is False:
                    bad += 1
        emit(f"eval_soundness.{key}.n_subsets", ok + bad)
        emit(f"eval_soundness.{key}.n_output_mismatch", bad)


# ===========================================================================
# eval_planspace : which physical operators an engine actually uses.
#
# Backs the claim in 5.2.3 that the DuckDB and the Postgres pinned runs name a
# different set of optimizer decisions because the two engines OFFER a different
# set, not because the two experiments were designed differently. The census is
# taken over every plan recorded in a run, which covers the original query and
# every executed subset, so it measures the plan space the workload actually
# reaches rather than the one the engine documents.
# ===========================================================================

# Operators that constitute a JOIN choice and a BASE-TABLE ACCESS choice, per
# engine dialect. Membership decides what is counted; anything a run emits that
# is in neither set is reported under `.other` rather than dropped, so a future
# run that reaches a new operator cannot silently weaken the claim.
_DDB_JOIN_OPS = {
    "HASH_JOIN", "NESTED_LOOP_JOIN", "BLOCKWISE_NL_JOIN", "PIECEWISE_MERGE_JOIN",
    "IE_JOIN", "CROSS_PRODUCT", "DELIM_JOIN", "POSITIONAL_JOIN", "ASOF_JOIN",
}
# COLUMN_DATA_SCAN is deliberately absent: it reads an in-memory chunk collection,
# not a base table, so counting it would inflate the access-path variety.
# SEQ_SCAN and TABLE_SCAN are the SAME operator under two names. DuckDB's profiling
# JSON carries `operator_name` "SEQ_SCAN " (note the trailing space) alongside
# `operator_type` "TABLE_SCAN" on every base-table node, and the plan plots label it
# by the former. Both are listed so the census stays correct if a DuckDB version
# populates operator_type with the display name, matching _SCAN_OPS in
# sql_predicate_converter.py. They never co-occur in one node's operator_type, so
# listing both cannot double-count.
_DDB_SCAN_OPS = {"TABLE_SCAN", "SEQ_SCAN", "INDEX_SCAN"}

_PG_JOIN_OPS = {"Nested Loop", "Hash Join", "Merge Join"}
# "Hash" is the build-side node of a Hash Join, not an access path or a join.
_PG_SCAN_OPS = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
                "Bitmap Index Scan", "Tid Scan"}

# DuckDB profiling names the operator `operator_type`; Postgres EXPLAIN names it
# `Node Type`. Both runs store raw plans, so the census reads the plan key that
# matches the engine the run used.
_CENSUS_DIALECT = {
    "DDB":    ("operator_type", _DDB_JOIN_OPS, _DDB_SCAN_OPS),
    "DDB_JP": ("operator_type", _DDB_JOIN_OPS, _DDB_SCAN_OPS),
    "PG":     ("Node Type", _PG_JOIN_OPS, _PG_SCAN_OPS),
    "PG_JP":  ("Node Type", _PG_JOIN_OPS, _PG_SCAN_OPS),
}

# CRITICAL for the access-path count on DuckDB. DuckDB does NOT expose the index
# scan as its own operator: an index scan and a sequential scan are both reported
# as operator_type TABLE_SCAN, and only `extra_info.Type` distinguishes them
# ("Index Scan" against "Sequential Scan"). Counting access paths from
# operator_type alone therefore reports one access path where the run used two.
# In this data the bare "Type" key occurs exactly once per TABLE_SCAN node and on
# no other node, so censusing it is equivalent to censusing the scan nodes.
# Postgres needs no counterpart because its Node Type already names the access path.
_DDB_ACCESS_PATH_KEY = "Type"


def _json_key_census(path: str, json_key: str):
    """{value: count} for every `"<json_key>": "<value>"` pair in a JSON file.

    Streamed with a regex over fixed-size chunks rather than json.load: the file
    reaches 2 GB, the census needs only the string values, and a full parse
    would cost an order of magnitude more memory than the answer is worth. The
    chunk boundary is handled by carrying the tail of each chunk into the next,
    so a key split across a read cannot be missed.
    """
    pat = re.compile(r'"' + re.escape(json_key) + r'":\s*"([^"]*)"')
    counts: dict[str, int] = {}
    # Longer than any operator name plus the key, so no match can straddle the
    # carry boundary undetected.
    overlap = 256
    buf = ""
    with open(path, "r", errors="replace") as fh:
        while True:
            chunk = fh.read(1 << 24)  # 16 MiB
            if not chunk:
                break
            buf += chunk
            # Count only matches that START before the carry region. Anything at
            # or after `limit` is left for the next pass, which sees it once and
            # in full, so no match is counted twice and none is lost at a seam.
            limit = max(0, len(buf) - overlap)
            consumed = 0
            for m in pat.finditer(buf):
                if m.start() >= limit:
                    break
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
                consumed = m.end()
            buf = buf[max(consumed, limit):]
        # Final flush: no more data can arrive, so the tail is counted whole.
        for m in pat.finditer(buf):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


@_cache_by_key
def _operator_census(key: str):
    """{operator name: count} over every plan node of a run, plus, on DuckDB, the
    access-path breakdown that operator_type alone does not expose (see
    _DDB_ACCESS_PATH_KEY). Returns (operators, access_paths); access_paths is None
    on engines whose operator name already carries the access path."""
    if key not in _CENSUS_DIALECT:
        return None
    path = os.path.join(_run_dir(key), "rule_summary_result.json")
    if not os.path.isfile(path):
        return None
    plan_key = _CENSUS_DIALECT[key][0]
    operators = _json_key_census(path, plan_key)
    access = (_json_key_census(path, _DDB_ACCESS_PATH_KEY)
              if plan_key == "operator_type" else None)
    return operators, access


def do_planspace():
    section("eval_planspace : physical operators each engine actually uses")
    for key in ("DDB", "DDB_JP", "PG", "PG_JP"):
        census = _operator_census(key)
        if census is None:
            emit(f"eval_planspace.{key}.n_join_algorithms", Missing(RUNS[key]))
            continue
        counts, access = census
        _, join_ops, scan_ops = _CENSUS_DIALECT[key]
        joins = {o: n for o, n in counts.items() if o in join_ops}
        # On DuckDB the access path lives in extra_info.Type, not in the operator
        # name, so the scan breakdown is taken from there when it is available.
        # Falling back to the operator names would report one access path for a
        # run that used two.
        scans = access if access else {o: n for o, n in counts.items() if o in scan_ops}
        other = {o: n for o, n in counts.items()
                 if o not in join_ops and o not in scan_ops}

        emit(f"eval_planspace.{key}.n_join_algorithms", len(joins))
        emit(f"eval_planspace.{key}.n_scan_strategies", len(scans))
        # The population the two counts above are taken over. This is the number
        # the chapter quotes when it says the variety is measured and not assumed.
        emit(f"eval_planspace.{key}.n_join_scan_ops",
             sum(joins.values()) + sum(scans.values()))
        for op, n in sorted(joins.items(), key=lambda t: -t[1]):
            emit(f"eval_planspace.{key}.join.{op}", n)
        for op, n in sorted(scans.items(), key=lambda t: -t[1]):
            emit(f"eval_planspace.{key}.scan.{op}", n)
        for op, n in sorted(other.items(), key=lambda t: -t[1]):
            emit(f"eval_planspace.{key}.other.{op}", n)


# ===========================================================================
# eval_impact : ABSOLUTE workload runtime (counters the "median % looks tiny")
# Population: queries with an applicable rule. Untouched queries keep original.
# ===========================================================================
def _impact(key: str):
    oracle = load_oracle(key)
    opt = load_optimizer(key)
    transfer = load_transfer(key)
    if oracle is None or opt is None or transfer is None:
        return None
    # Sorted, not set order: float addition is not associative, so an unordered
    # sum makes the last printed digit depend on PYTHONHASHSEED.
    pop = sorted(set(transfer.keys()) | set(oracle.keys()) | set(opt.keys()))
    tot_orig = tot_oracle = tot_real = 0.0
    saved = []  # (prefix, oracle seconds saved)
    for q in pop:
        orig = oracle[q]["orig"] if q in oracle else (opt[q]["orig"] if q in opt else None)
        if orig is None:
            continue
        best = oracle[q]["improved"] if q in oracle else orig
        real = opt[q]["improved"] if q in opt else orig
        tot_orig += orig
        tot_oracle += best
        tot_real += real
        saved.append((q, orig - best))
    return tot_orig, tot_oracle, tot_real, saved


def do_impact():
    """Absolute workload seconds. Reported for every selector, not just DuckDB:
    a percentage regression on a sub-10ms query is a ratio artefact, and the
    chapter must apply the absolute view to losses as well as to gains."""
    section("eval_impact (DDB) : absolute workload runtime, seconds")
    m = _impact("DDB")
    if m is None:
        emit("eval_impact.total_original_s", Missing(RUNS["DDB"]))
    else:
        tot_orig, tot_oracle, tot_real, saved = m
        emit("eval_impact.n_population", len(saved))
        emit("eval_impact.total_original_s", round(tot_orig, 3))
        emit("eval_impact.total_oracle_s", round(tot_oracle, 3))
        emit("eval_impact.total_realized_s", round(tot_real, 3))
        if tot_orig > 0:
            emit("eval_impact.workload_pct_oracle", 100.0 * (tot_orig - tot_oracle) / tot_orig)
            emit("eval_impact.workload_pct_realized", 100.0 * (tot_orig - tot_real) / tot_orig)
        for i, (q, s) in enumerate(sorted(saved, key=lambda kv: kv[1], reverse=True)[:5], 1):
            print(f"[eval_impact.top_saved_s.{i}] = {q}: {s:.3f}s saved (oracle)")

    section("eval_impact_all : absolute workload runtime per selector, seconds")
    for key in ENGINE_SELECTORS + ["DDB_JP", "PG_JP"]:
        m = _impact(key)
        if m is None:
            emit(f"eval_impact_all.{key}.total_original_s", Missing(RUNS[key]))
            continue
        tot_orig, tot_oracle, tot_real, saved = m
        emit(f"eval_impact_all.{key}.n_population", len(saved))
        emit(f"eval_impact_all.{key}.total_original_s", round(tot_orig, 3))
        emit(f"eval_impact_all.{key}.total_oracle_s", round(tot_oracle, 3))
        emit(f"eval_impact_all.{key}.total_realized_s", round(tot_real, 3))
        if tot_orig > 0:
            emit(f"eval_impact_all.{key}.workload_pct_oracle",
                 100.0 * (tot_orig - tot_oracle) / tot_orig)
            emit(f"eval_impact_all.{key}.workload_pct_realized",
                 100.0 * (tot_orig - tot_real) / tot_orig)
        # Worst absolute loss, the honest counterpart to the worst percentage.
        opt = load_optimizer(key)
        losses = sorted(((v["improved"] - v["orig"], q, v["pct"])
                         for q, v in opt.items() if v["improved"] > v["orig"]),
                        reverse=True)
        for i, (loss, q, pct) in enumerate(losses[:3], 1):
            print(f"[eval_impact_all.{key}.top_lost_ms.{i}] = {q}: "
                  f"{loss*1000:+.1f}ms ({pct:.1f}%)")


# ===========================================================================
# eval_replication : measurement stability from two INDEPENDENT full runs.
# PG and LRN differ only in the rule-subset selector (cost_model), so the
# ORIGINAL query is executed twice under identical conditions (same host, same
# engine, same warmup mode, same frozen rule pool). The spread between the two
# is an empirical replicate estimate of timing noise. This is NOT a CV study
# over K repetitions; it is the evidence that happens to be on hand.
# ===========================================================================
def do_replication():
    section("eval_replication : PG vs LRN, two independent runs of the same queries")
    a_opt, b_opt = load_optimizer("PG"), load_optimizer("LRN")
    a_ora, b_ora = load_oracle("PG"), load_oracle("LRN")
    if a_opt is None or b_opt is None:
        emit("eval_replication.n_queries", Missing("PG or LRN"))
        return
    # Baseline agreement over EVERY query both runs timed (oracle rows included).
    # Restricting this to the optimizer rows would silently drop queries that one
    # run declined, and those are not a random subset.
    def origs(opt, ora):
        out = {}
        for src in (ora or {}, opt or {}):
            for q, v in src.items():
                if v["orig"] > 0:
                    out.setdefault(q, v["orig"])
        return out

    oa, ob = origs(a_opt, a_ora), origs(b_opt, b_ora)
    common = sorted(set(oa) & set(ob))
    rel = [100.0 * (ob[q] - oa[q]) / oa[q] for q in common]
    if rel:
        absrel = sorted(abs(x) for x in rel)
        emit("eval_replication.n_queries", len(rel))
        emit("eval_replication.baseline_median_rel_diff_pct", float(median(rel)))
        emit("eval_replication.baseline_mean_abs_rel_diff_pct",
             float(mean(absrel)))
        emit("eval_replication.baseline_median_abs_rel_diff_pct",
             float(median(absrel)))
        emit("eval_replication.baseline_p90_abs_rel_diff_pct",
             float(absrel[int(0.9 * (len(absrel) - 1))]))
        emit("eval_replication.baseline_max_abs_rel_diff_pct", float(absrel[-1]))
        emit("eval_replication.baseline_n_within_5pct",
             sum(1 for x in absrel if x <= 5.0))
        for q in sorted(common, key=lambda q: -abs(ob[q] - oa[q]) / oa[q])[:3]:
            print(f"[eval_replication.baseline_outlier] = {q}: "
                  f"PG={oa[q]*1000:.1f}ms LRN={ob[q]*1000:.1f}ms "
                  f"({100.0*(ob[q]-oa[q])/oa[q]:+.1f}%)")
    if a_ora is not None and b_ora is not None:
        cc = sorted(set(a_ora) & set(b_ora))
        d = [b_ora[q]["pct"] - a_ora[q]["pct"] for q in cc]
        if d:
            emit("eval_replication.ceiling_n_common", len(cc))
            emit("eval_replication.ceiling_median_delta_pp", float(median(d)))
            emit("eval_replication.ceiling_p90_abs_delta_pp",
                 float(sorted(abs(x) for x in d)[int(0.9 * (len(d) - 1))]))
            emit("eval_replication.ceiling_max_abs_delta_pp",
                 float(max(abs(x) for x in d)))
            # Outliers are named, never averaged away.
            for q, dd in sorted(zip(cc, d), key=lambda t: -abs(t[1]))[:3]:
                print(f"[eval_replication.ceiling_outlier] = {q}: "
                      f"PG={a_ora[q]['pct']:.1f}% LRN={b_ora[q]['pct']:.1f}% "
                      f"(delta {dd:+.1f}pp, orig {a_ora[q]['orig']*1000:.1f}ms / "
                      f"{b_ora[q]['orig']*1000:.1f}ms)")


# ===========================================================================
# eval_transfer_win : is transfer part of the WINNING selection? (not chaining)
# A winning subset "uses transfer" when it contains a rule mined from ANOTHER
# query (origin prefix != target prefix), or a merged rule composed of several.
# ===========================================================================
def _origins(winning_rules: str) -> list[str]:
    """Origin query prefixes of every component rule in a winning_rules cell."""
    origins = []
    for part in winning_rules.split(";"):
        part = part.strip()
        if not part:
            continue
        body = part[len("merged:"):] if part.startswith("merged:") else part
        for comp in body.split("+"):
            comp = comp.strip()
            if " NoMIN" in comp:
                origins.append(comp.split(" NoMIN")[0].split()[-1])
    return origins


def _n_components(winning_rules: str) -> int:
    n = 0
    for part in winning_rules.split(";"):
        part = part.strip()
        if not part:
            continue
        body = part[len("merged:"):] if part.startswith("merged:") else part
        n += len([c for c in body.split("+") if c.strip()])
    return n


def do_transfer_win():
    section("eval_transfer_win (DDB oracle) : transfer's role in the winning subset")
    path = os.path.join(_run_dir("DDB"), "oracle_stats.csv")
    if not os.path.isfile(path):
        emit("eval_transfer_win.n_wins", Missing(RUNS["DDB"]))
        return
    n = multi = with_transfer = only_transfer = 0
    single = single_transfer = 0
    comps = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            target = r["prefix"].split()[0]
            wr = r["winning_rules"]
            k = _n_components(wr)
            origins = _origins(wr)
            comps.append(k)
            n += 1
            if k > 1:
                multi += 1
            if any(o != target for o in origins):
                with_transfer += 1
            # Containing a transferred rule does not prove the transferred rule
            # contributed. These two counts do: a win with no component from the
            # target query rests on transfer entirely, and a single-rule win by a
            # rule mined elsewhere needs no counterfactual argument at all.
            if origins and all(o != target for o in origins):
                only_transfer += 1
            if k == 1:
                single += 1
                if origins and origins[0] != target:
                    single_transfer += 1
    emit("eval_transfer_win.n_wins", n)
    emit("eval_transfer_win.n_multi_rule", multi)
    emit("eval_transfer_win.pct_multi_rule", 100.0 * multi / n if n else float("nan"))
    emit("eval_transfer_win.n_uses_transferred_rule", with_transfer)
    emit("eval_transfer_win.pct_uses_transferred_rule", 100.0 * with_transfer / n if n else float("nan"))
    emit("eval_transfer_win.n_only_transferred", only_transfer)
    emit("eval_transfer_win.pct_only_transferred", 100.0 * only_transfer / n if n else float("nan"))
    emit("eval_transfer_win.n_single_rule_wins", single)
    emit("eval_transfer_win.n_single_rule_transferred", single_transfer)
    emit("eval_transfer_win.components_median", float(median(comps)) if comps else float("nan"))
    emit("eval_transfer_win.components_max", max(comps) if comps else 0)


# ===========================================================================
# eval_transfer_tmpl (DDB) : is the transfer of 5.2.2 an artefact of the JOB
# template structure? The 113 queries derive from 33 templates whose variants
# share a join graph and differ in their predicate constants, so a fan-out of
# 2.5 queries is numerically consistent with pure intra-template reuse and a
# reviewer can construct that objection from the query names alone. Every count
# of eval_transfer / eval_transfer_win is therefore recomputed at TEMPLATE
# granularity, which is the conservative reading, and the two readings are
# reported side by side rather than one replacing the other.
# ===========================================================================
def _template(q: str) -> str:
    """JOB template of a query name: the leading digit run of `17f` is `17`.

    This grouping is derived here from the query name and is NOT carried by the
    pipeline, so it is a definition of this analysis, not recovered data.
    """
    m = re.match(r"(\d+)", q.strip())
    return m.group(1) if m else q.strip()


def do_transfer_tmpl():
    section("eval_transfer_tmpl (DDB) : transfer counted across JOB templates")
    transfer = load_transfer("DDB")
    if transfer is None:
        emit("eval_transfer_tmpl.firing_rate_same_template", Missing(RUNS["DDB"]))
        return
    workload = [
        os.path.basename(f)[:-4]
        for f in glob.glob(os.path.join(JOB_SQL_DIR, "*.sql"))
        if os.path.basename(f) not in ("schema.sql", "fkindexes.sql")
    ]
    if not workload:
        emit("eval_transfer_tmpl.firing_rate_same_template", Missing(JOB_SQL_DIR))
        return
    emit("eval_transfer_tmpl.n_templates", len({_template(q) for q in workload}))

    # Rule -> queries it is applied to. The transfer JSON keys carry a run suffix
    # ("11a NoMIN"), so the prefix is taken exactly as _origins takes it on the
    # oracle side; without this the two sides would not be joinable.
    fan: dict[str, set] = {}
    for q, v in transfer.items():
        for r in v.get("rules", []):
            fan.setdefault(r["name"], set()).add(q.split()[0])

    # Grouping is by origin TEMPLATE, not by origin query, which keeps the base
    # at all 111 applicable rules of eval_transfer. Merging combines rules with
    # identical `requires`, so a composite can carry several source queries, but
    # what matters here is only whether those queries fall in one template: a
    # rule merged from 6a and 6c still has the single well-defined origin
    # template 6 and is evaluated like any atomic rule. Only the
    # 4 rules whose origins span SEVERAL templates have no "own template", and
    # those have demonstrably crossed a template boundary by construction, so
    # they are counted as transferred rather than dropped from the base.
    origin_tmpls = {k: {_template(o) for o in _origins(k)} for k in fan}
    single = [k for k in fan if len(origin_tmpls[k]) == 1]
    emit("eval_transfer_tmpl.n_rules_single_origin_template", len(single))
    emit("eval_transfer_tmpl.n_rules_multi_origin_template", len(fan) - len(single))

    # The denominator is the FULL 113-query workload, not the 86 rule-covered
    # ones: a rule that is not applied to a sibling is a real outcome of the
    # applicability check and must stay in the denominator, otherwise the rate
    # conditions on the very event it is meant to measure. A rule's own origin
    # queries are excluded from its siblings, since applying it there is not reuse.
    same_hit = same_tot = diff_hit = diff_tot = 0
    no_sibling = all_siblings = 0
    for name in single:
        applied_to = fan[name]
        origins = set(_origins(name))
        tmpl = next(iter(origin_tmpls[name]))
        siblings = [q for q in workload if _template(q) == tmpl and q not in origins]
        others = [q for q in workload if _template(q) != tmpl]
        hits = sum(1 for q in siblings if q in applied_to)
        same_hit += hits
        same_tot += len(siblings)
        diff_hit += sum(1 for q in others if q in applied_to)
        diff_tot += len(others)
        if siblings and hits == 0:
            no_sibling += 1
        if siblings and hits == len(siblings):
            all_siblings += 1
    p_same = 100.0 * same_hit / same_tot if same_tot else float("nan")
    p_diff = 100.0 * diff_hit / diff_tot if diff_tot else float("nan")
    emit("eval_transfer_tmpl.firing_rate_same_template", p_same)
    emit("eval_transfer_tmpl.firing_rate_diff_template", p_diff)
    emit("eval_transfer_tmpl.firing_rate_ratio", p_same / p_diff if p_diff else float("nan"))
    # The two rates say opposite things and 5.2.2 quotes both. The ratio shows
    # that a shared join graph is what makes transfer likely at all; the
    # pct_no_sibling below shows that it is nowhere near sufficient, because the
    # `requires` conditions are stated over the constants that distinguish one
    # variant from the next. Quoting either alone misrepresents the mechanism.
    #
    # Both percentages take ALL 111 applicable rules as their base, so they are
    # directly comparable to every other rule count in 5.2.2. The 4 rules with
    # several origin templates sit in the base without sitting in either
    # numerator, which is the conservative placement: counting them as
    # "applied to no sibling" would inflate the very number that concedes the
    # confound, and counting them as "applied to all" would inflate the rebuttal.
    emit("eval_transfer_tmpl.n_rules_no_sibling", no_sibling)
    emit("eval_transfer_tmpl.pct_rules_no_sibling", 100.0 * no_sibling / len(fan))
    emit("eval_transfer_tmpl.pct_rules_all_siblings", 100.0 * all_siblings / len(fan))
    # Application EVENTS, not rules: the share of actual reuse that crosses a
    # template boundary. This is the number the confound objection has to beat.
    total_ev = same_hit + diff_hit
    emit("eval_transfer_tmpl.n_firing_events", total_ev)
    emit("eval_transfer_tmpl.pct_events_cross_template",
         100.0 * diff_hit / total_ev if total_ev else float("nan"))

    # Fan-out at template granularity, the direct counterpart of
    # eval_transfer.n_rules_transferred / fanout_mean_gt1.
    tmpl_fan = {k: {_template(q) for q in v} for k, v in fan.items()}
    gt1_q = [k for k in fan if len(fan[k]) > 1]
    gt1_t = [k for k in fan if len(tmpl_fan[k]) > 1]
    emit("eval_transfer_tmpl.n_rules_cross_template", len(gt1_t))
    emit("eval_transfer_tmpl.pct_transferring_rules_cross_template",
         100.0 * len(gt1_t) / len(gt1_q) if gt1_q else float("nan"))
    if gt1_q:
        emit("eval_transfer_tmpl.fanout_mean_gt1_queries",
             float(mean(len(fan[k]) for k in gt1_q)))
    if gt1_t:
        emit("eval_transfer_tmpl.fanout_mean_gt1_templates",
             float(mean(len(tmpl_fan[k]) for k in gt1_t)))
        emit("eval_transfer_tmpl.fanout_max_templates",
             max(len(tmpl_fan[k]) for k in gt1_t))

    # Win side, mirroring do_transfer_win one-for-one at template granularity.
    path = os.path.join(_run_dir("DDB"), "oracle_stats.csv")
    if not os.path.isfile(path):
        emit("eval_transfer_tmpl.n_wins", Missing(RUNS["DDB"]))
        return
    n = with_x = only_x = single_win_x = 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            target = _template(r["prefix"].split()[0])
            origins = [_template(o) for o in _origins(r["winning_rules"])]
            n += 1
            if any(o != target for o in origins):
                with_x += 1
            if origins and all(o != target for o in origins):
                only_x += 1
            if len(origins) == 1 and origins[0] != target:
                single_win_x += 1
    emit("eval_transfer_tmpl.n_wins", n)
    emit("eval_transfer_tmpl.n_wins_uses_cross_template", with_x)
    emit("eval_transfer_tmpl.pct_wins_uses_cross_template",
         100.0 * with_x / n if n else float("nan"))
    emit("eval_transfer_tmpl.n_wins_only_cross_template", only_x)
    emit("eval_transfer_tmpl.pct_wins_only_cross_template",
         100.0 * only_x / n if n else float("nan"))
    emit("eval_transfer_tmpl.n_single_rule_wins_cross_template", single_win_x)


# ===========================================================================
# eval_crossengine : do the engines regress on the same or disjoint queries?
# ===========================================================================
def do_crossengine():
    """Severe-regression overlap. LRN is included because the SHARPEST version of
    the claim is the pair PG vs LRN: same engine, same data, same frozen rule
    pool, different cost model. Without it the section's headline ("a property of
    the cost model rather than of the engine") has no figure behind it.

    The three-engine labels keep their exact original meaning (n_common_all3 and
    n_union are over DDB/UMB/PG only) because the prose cites them; the
    four-selector view is emitted under its own labels next to them."""
    section("eval_crossengine : regression overlap across selectors (severe <-5%)")
    sets = {}
    for key in ENGINE_SELECTORS:
        opt = load_optimizer(key)
        if opt is None:
            emit(f"eval_crossengine.{key}.n_severe_regress", Missing(RUNS[key]))
            sets[key] = None
        else:
            sets[key] = {q for q, v in opt.items() if v["pct"] < -5}
            emit(f"eval_crossengine.{key}.n_severe_regress", len(sets[key]))
    if all(sets[k] is not None for k in ("DDB", "UMB", "PG")):
        d, u, p = sets["DDB"], sets["UMB"], sets["PG"]
        emit("eval_crossengine.n_common_all3", len(d & u & p))
        emit("eval_crossengine.n_union", len(d | u | p))
        emit("eval_crossengine.n_unique_DDB", len(d - u - p))
        emit("eval_crossengine.n_unique_UMB", len(u - d - p))
        emit("eval_crossengine.n_unique_PG", len(p - d - u))
    if all(sets[k] is not None for k in ENGINE_SELECTORS):
        d, u, p, zs = (sets["DDB"], sets["UMB"], sets["PG"], sets["LRN"])
        emit("eval_crossengine.n_common_all4", len(d & u & p & zs))
        emit("eval_crossengine.n_union_all4", len(d | u | p | zs))
        # The cost-model pair on one engine: how much of the failure set is
        # shared when the engine is held fixed and only the cost model changes.
        emit("eval_crossengine.n_shared_PG_LRN", len(p & zs))
        emit("eval_crossengine.n_only_PG", len(p - zs))
        emit("eval_crossengine.n_only_LRN", len(zs - p))
        print("[eval_crossengine.shared_PG_LRN] = " + ", ".join(sorted(p & zs)))
        print("[eval_crossengine.only_LRN] = " + ", ".join(sorted(zs - p)))


# ===========================================================================
# eval_showcase : a few representative rules for a qualitative table
# ===========================================================================
# The showcase is PINNED, not ranked. Sorting by speedup returns the Downey/MCU
# family five times over, because that one semantic cluster owns the whole tail;
# a table of five variants of the running example demonstrates nothing. The rules
# below are chosen so that each shows a different KIND of world knowledge, and
# every one of them survives join-order pinning (checked in do_ceiling_pinned),
# so none of them is a reordering artefact being sold as selectivity.
SHOWCASE_RULES = [
    # (rule_name, the query whose oracle win it carries, what makes it world knowledge)
    ("28b NoMIN_r0_1_IMDB_R2", "28b",
     "nationality adjectives are not country values under info_type 'countries'"),
    ("3b NoMIN_r1_0_IMDB_R1", "3b",
     "a sequel-tagged post-2010 title is a feature film, not a TV episode"),
    ("29b NoMIN_r2_2_IMDB_R3", "29b",
     "a principal voiced role in Shrek 2 is billed within the first 20 cast positions"),
    ("merged: 6a NoMIN_r1_1_IMDB_R2 + 6c NoMIN_r1_1_IMDB_R2 + 6e NoMIN_r0_2_IMDB_R3 "
     "+ 6e NoMIN_r2_1_IMDB_R2 + 6a NoMIN_r1_0_IMDB_R1_rerun_s0", "6a",
     "the running example: fuzzy name match resolves to one male acting entity"),
]


def do_showcase():
    section("eval_showcase : representative rules (name, WK score, speedup)")
    path = wk_eval_path()
    if not os.path.isfile(path):
        emit("eval_showcase", Missing(path))
        return
    with open(path) as fh:
        rules = {r["rule_name"]: r for r in json.load(fh)["rules"]}

    # Query-level oracle speedup, join order free (DDB) vs pinned (DDB_JP),
    # keyed by query. This is the free/pinned figure the prose quotes for the
    # showcase rules; emitting it here gives that number a reproducible label
    # instead of it being read off oracle_stats.csv by hand. It is a DIFFERENT
    # quantity from max_saved below (query oracle, not the single rule alone).
    free = load_oracle("DDB")
    pinned = load_oracle("DDB_JP")
    free_by_q = {_basename(k): v for k, v in free.items()} if free else {}
    pinned_by_q = {_basename(k): v for k, v in pinned.items()} if pinned else {}

    for i, (name, query, gist) in enumerate(SHOWCASE_RULES, 1):
        r = rules.get(name)
        if r is None:
            emit(f"eval_showcase.{i}", Missing(f"{path} :: {name}"))
            continue
        agg = r.get("agg", {})
        f_pct = free_by_q.get(query, {}).get("pct")
        p_pct = pinned_by_q.get(query, {}).get("pct")
        emit(f"eval_showcase.{i}.q_free_pct",
             round(f_pct, 1) if f_pct is not None else Missing(f"DDB oracle :: {query}"))
        emit(f"eval_showcase.{i}.q_pinned_pct",
             round(p_pct, 1) if p_pct is not None else Missing(f"DDB_JP oracle :: {query}"))
        print(f"[eval_showcase.{i}] = {name} | carries={query} | "
              f"score={r['verdict']['score']} | "
              f"median_saved={agg.get('median_percent_saved', float('nan')):.1f}% | "
              f"max_saved={agg.get('max_percent_saved', float('nan')):.1f}% | "
              f"q_free={'NA' if f_pct is None else round(f_pct, 1)}% | "
              f"q_pinned={'NA' if p_pct is None else round(p_pct, 1)}% | "
              f"fired={agg.get('n_queries_fired')} | {gist}")


# ===========================================================================
# eval_funnel : LLM candidate -> sound rule -> repository (generation yield)
# Reads transfer_data/<run>/ (generation + refinement stage outputs).
# ===========================================================================
def do_funnel():
    section("eval_funnel (DDB) : generation -> validation -> repository")
    base = os.path.join(TRANSFER, RUNS["DDB"])

    def _valid(v) -> bool:
        bt = v.get("base_table_validation")
        return isinstance(bt, dict) and bt.get("all_valid") is True

    def load(fn):
        p = os.path.join(base, fn)
        if not os.path.isfile(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    gen_d = load("result.json")
    ref_d = load("result2.json")
    if gen_d is None:
        emit("eval_funnel.n_candidates", Missing(base + "/result.json"))
        return

    # len(result.json) is NOT a rule count and must never be quoted as one. A
    # response that fails to parse still writes one `_error` entry that carries
    # no rule, so on this run the file holds 1023 candidate rules plus 19 such
    # entries. It is also a POST-DEDUP figure: generation.py drops any rule whose
    # signature is already known for that query, so the model actually emitted
    # 1297 rules (eval_novelty.n_rules_emitted) to leave these 1023 behind. An
    # earlier draft quoted len(result.json) = 1042 as "candidate rules the model
    # produced", which was wrong on both counts. Do not reintroduce that.
    n_entries = len(gen_d)
    n_parse_fail = sum(1 for k in gen_d if k.endswith("_error"))
    g_gen = n_entries - n_parse_fail
    g_inj = sum(1 for v in gen_d.values() if v.get("status") == "new predicates applied")
    g_val = sum(1 for v in gen_d.values() if _valid(v))
    emit("eval_funnel.n_result_entries", n_entries)
    emit("eval_funnel.n_parse_failures", n_parse_fail)
    emit("eval_funnel.n_candidates", g_gen)
    emit("eval_funnel.n_validated_generation", g_val)

    # Injection and soundness are INDEPENDENT properties, not successive funnel
    # stages: a rule can be sound on the full instance and still inject nothing
    # into its source query, because the predicate is already implied there.
    # n_validated_no_injection is the direct evidence that the two counts are not
    # nested; 5.2.1 states this explicitly and fig_funnel omits the 647 bar.
    emit("eval_funnel.n_inject_predicates", g_inj)
    emit("eval_funnel.n_validated_no_injection",
         sum(1 for v in gen_d.values()
             if _valid(v) and v.get("status") != "new predicates applied"))

    n_val_total = g_val
    if ref_d is not None:
        # HOW MANY REQUESTS THE LOOP ACTUALLY ISSUES.
        #
        # len(result2.json) is NOT that number and must never be used as one.
        # stages/refinement.py issues `refinement.samples` requests for every
        # entry that _is_failed_entry accepts, but writes a DEDUPLICATED dict
        # whenever samples > 1: at most one non-rule outcome is kept per
        # candidate, and refined rules are collapsed by exact signature per SQL.
        # On this run that turns 1416 issued requests into 837 retained entries.
        # With samples == 1 both dedup branches are inactive and the two counts
        # coincide, which is why the discrepancy is invisible on the 1-1 ablation
        # run. An earlier version of this function read the retained entries that
        # carried a rule (191) as the "attempts" denominator, which inflated the
        # loop's recovery rate from 7.1% to 26%; do not reintroduce that.
        n_ref_samples = 1
        for k in ref_d:
            m = re.search(r"_rerun_s(\d+)$", k)
            if m:
                n_ref_samples = max(n_ref_samples, int(m.group(1)) + 1)
        n_failed = sum(1 for k, v in gen_d.items() if _is_failed_entry(k, v))
        emit("eval_funnel.n_refine_samples", n_ref_samples)
        emit("eval_funnel.n_failed_candidates", n_failed)
        emit("eval_funnel.n_refine_requests", n_ref_samples * n_failed)
        # Retained entries, kept for provenance only. These are post-dedup counts
        # and are not a denominator for any rate the chapter reports.
        emit("eval_funnel.n_refine_entries_retained", len(ref_d))
        returned = [k for k, v in ref_d.items() if v.get("status") != "no_rule_returned"]
        emit("eval_funnel.n_refine_distinct_rules_returned", len(returned))
        r_val_keys = [k for k in returned if _valid(ref_d[k])]
        r_val = len(r_val_keys)
        emit("eval_funnel.n_validated_refinement", r_val)
        # Several candidates validate in both samples, so the rule count above
        # exceeds the number of candidates actually rescued.
        recovered = {re.sub(r"_rerun_s\d+$", "", k) for k in r_val_keys}
        emit("eval_funnel.n_refine_candidates_recovered", len(recovered))
        # With refinement.samples = 2 a candidate has at most two slots, so the
        # difference is exactly the candidates repaired in both samples. This is
        # why the validated-rule count exceeds the recovered-candidate count.
        emit("eval_funnel.n_refine_double_recovered", r_val - len(recovered))
        # Recovery rate over the population the loop was actually run on, that is
        # every failed candidate, not over the subset that returned something.
        if n_failed:
            emit("eval_funnel.pct_refine_recovery_rate",
                 100.0 * len(recovered) / n_failed)
        n_val_total = g_val + r_val
        emit("eval_funnel.n_validated_total", n_val_total)
        if n_val_total:
            emit("eval_funnel.pct_repository_from_refinement",
                 100.0 * r_val / n_val_total)

    # aggregation.concatenate_rule_entries drops only sound rules with no implies
    # (none in this run). Non-injecting no-op rewrites are retained for possible
    # cross-query transfer, so "base-table sound" maps directly to "distinct after
    # merging" in fig_funnel with no intermediate filter bar.

    # Post-merge pool. With cost_estimation on, aggregation writes the merged
    # rule pool to cost_aggregate_input.json instead of rule_summary_transfer.json.
    pool_path = os.path.join(base, "cost_aggregate_input.json")
    pool_names = None
    if os.path.isfile(pool_path):
        with open(pool_path) as fh:
            pool = json.load(fh).get("rules", [])
        pool_names = {r["name"] for r in pool}
        emit("eval_funnel.n_after_merge", len(pool))
        emit("eval_funnel.n_merged_composites",
             sum(1 for r in pool if str(r.get("name", "")).startswith("merged:")))
    else:
        emit("eval_funnel.n_after_merge", Missing(pool_path))

    transfer = load_transfer("DDB")
    if transfer is not None:
        names = set()
        for e in transfer.values():
            for r in e.get("rules", []):
                names.add(r["name"])
        emit("eval_funnel.n_distinct_repository", len(names))
        if pool_names is not None:
            # Sound and merged, but applied to no query of this workload.
            emit("eval_funnel.n_never_fires", len(pool_names - names))


# ===========================================================================
def main():
    print("# thesis_numbers.py output -- Chapter 5 (Evaluation)")
    print(f"# saved_results: {SAVED}")
    do_setup()
    # Part 1
    do_coverage()
    do_funnel()
    do_transfer()
    do_transfer_win()
    do_transfer_tmpl()
    do_runtime()
    do_ceiling()
    do_impact()
    do_replication()
    do_soundness()
    do_worldknowledge()
    do_showcase()
    do_generality()
    # Part 2
    do_oracle()
    do_joinorder()
    do_engines()
    do_crossengine()
    do_errors()
    do_explored()
    do_attribution()
    do_planspace()
    do_casestudies()
    # Ablation
    do_ablation_gen()
    do_novelty()

    section("MISSING SUMMARY")
    if _MISSING_SEEN:
        print(f"# {len(_MISSING_SEEN)} label(s) could not be computed (run not collected):")
        for lbl in _MISSING_SEEN:
            print(f"#   [{lbl}]")
    else:
        print("# all labels computed; no missing runs.")


if __name__ == "__main__":
    main()
