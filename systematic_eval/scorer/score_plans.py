#!/usr/bin/env python3
"""Standalone ZeroShot learned-cost scorer (runs in the isolated py3.12 scorer venv).

Two entry modes:

  predict-parsed     Read an already-parsed v3 plans file ({parsed_plans, database_stats,
                     run_kwargs}) and predict per-plan runtime. No parsing — the reliable
                     path used for verification gate 1 (validate env + model + stats).

  predict-candidates Read candidates.json produced by execution.py (each candidate carries
                     a Postgres `EXPLAIN (VERBOSE)` text plan), parse to the v3 schema,
                     then predict. The parse step uses the vendored cross_db_benchmark
                     parser + a v3 adapter (see _planop_to_v3).

Output (both modes): predictions.json  { "<id>": predicted_runtime_seconds, ... }.

NOTE ON DRIFT: the v3 node schema (op_name categories like "Simple Aggregate" /
"Index Nested Loop", the `filter`/`join` split, per-node `is_index_cond`) is produced
by the lab's internal cross_db_benchmark fork. The adapter here reproduces it best-effort
from the public parser and MUST be validated at gate 2 (diff against a real parsed_plans3
node) — or replaced by dropping the lab's exact parser into vendor/cross_db_benchmark/.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import orjson

# A predicate literal that is itself a column reference (table.col) is a JOIN condition,
# not a filter. v3 encodes joins in a separate `join` field (col_id1/col_id2); the public
# parser cannot produce that split, so such predicates are dropped (the est featurization
# only uses `operator`, so an unencoded join is a featurization gap, not a crash). Closing
# this gap requires the lab's exact cross_db_benchmark parser. See README.
_COLREF_RE = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*\s*$")

# Vendored parser (plain dir on sys.path; no install).
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))


# ───────────────────────── model load + predict (notebook recipe) ──────────────────────────
def predict_parsed_file(parsed_path, *, model_type, model_dir, seed, statistics_file,
                        device, batch_size=256):
    """Load the ZeroShot checkpoint and predict per-plan runtime (seconds) for a v3
    parsed-plans file. Returns { plan_id_or_index: seconds }.

    Follows the reference loading recipe of the group's ldb_models implementation.
    """
    import torch  # noqa: F401  (ensures CUDA init happens in this venv)
    import numpy as np
    from ldb_models.training.dataset.dataset_creation import create_zeroshot_dataloader
    from ldb_models.classes.workload_runs import WorkloadRuns
    from ldb_models.classes.configs.base import DataLoaderOptions
    from ldb_models.classes.model_config_helper import get_model_config
    from ldb_models.models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
    from ldb_models.training.training.checkpoint import load_checkpoint

    model_config = get_model_config(
        model_type=model_type,
        m_args=dict(
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_workers=0,
            skip_top_aggregation=False,
            limit_plans_per_file=10_000_000,
        ),
    )

    workload = WorkloadRuns(
        train_workload_runs=[Path(parsed_path)],
        test_workload_runs=[],
        target_test_csv_paths=[],
    )
    opts = DataLoaderOptions(val_ratio=0.0)
    label_norm, feature_stats, train_loader, _val, _test, _out = create_zeroshot_dataloader(
        workload_runs=workload,
        statistics_file=Path(statistics_file),
        model_config=model_config,
        data_loader_options=opts,
        quick_run=False,
        assemble_graphs_up_front=False,
    )

    model = PostgresZeroShotModel(
        model_config=model_config,
        feature_statistics=feature_stats,
        label_norm=label_norm,
    )
    model.to(model_config.device)
    load_checkpoint(model=model, config=model_config, optimizer=None,
                    target_path=Path(model_dir), filetype=".pt", load_last_checkpoint=True)
    model.eval()

    # id list, written in plan order by _write_workload (top-level "plan_id" per plan).
    plan_ids = [p.get("plan_id") for p in _load(parsed_path)["parsed_plans"]]

    preds_by_idx = {}
    with torch.no_grad():
        for batch in train_loader:
            input_model, label, query_stats = model_config.batch_to_func(
                batch, model.device, model.label_norm)
            output = model(input_model)
            pred = output.plan_predictions.detach().cpu().numpy()
            if model.label_norm is not None:
                pred = model.label_norm.inverse_transform(pred)
            pred = np.asarray(pred).reshape(-1)
            # sample_idxs maps each prediction back to its position in parsed_plans.
            sample_idxs = query_stats["sample_idxs"]
            for j, s_idx in enumerate(sample_idxs):
                preds_by_idx[int(s_idx)] = float(pred[j])

    out = {}
    for idx, seconds in preds_by_idx.items():
        key = plan_ids[idx] if idx < len(plan_ids) and plan_ids[idx] is not None else str(idx)
        out[str(key)] = seconds
    return out


# ───────────────────────── parsing: EXPLAIN text → v3 parsed plans ──────────────────────────
# v3 op_name categories the model knows (from feature_statistics). Any value outside
# this set makes ldb_models' encode() assert-fail, so _normalize_op_name must always
# return one of these. The pg_lab conf disables parallelism/bitmap/memoize, so the raw
# Postgres node types already fall within this set in practice.
_KNOWN_OP_NAMES = {
    "Index Only Scan", "Sort", "Merge Join", "Nested Loop", "Materialize", "Hash",
    "Hash Join", "Index Nested Loop", "Seq Scan", "Index Scan", "Simple Aggregate",
}


def _normalize_op_name(op_name, children):
    """Map a raw Postgres node type to the lab's v3 op_name categories.

    VALIDATE at gate 2 against real parsed_plans3 op_names — esp. the two synthesized
    categories (Simple Aggregate, Index Nested Loop). Falls back to the nearest known
    category for any unexpected type so one odd plan cannot abort the whole batch.
    """
    if op_name in ("Aggregate", "Simple Aggregate", "Finalize Aggregate",
                   "Partial Aggregate", "GroupAggregate", "HashAggregate"):
        return "Simple Aggregate"
    if op_name == "Nested Loop":
        # Lab distinguishes a nested loop whose inner side is an index scan.
        inner = children[1] if len(children) > 1 else None
        inner_op = inner.plan_parameters.get("op_name") if inner is not None else None
        return "Index Nested Loop" if inner_op in ("Index Scan", "Index Only Scan") else "Nested Loop"
    if op_name in _KNOWN_OP_NAMES:
        return op_name
    fallback = ("Seq Scan" if "Scan" in (op_name or "")
                else "Hash Join" if "Join" in (op_name or "")
                else "Materialize")
    print(f"WARNING: unknown op_name {op_name!r} -> {fallback} (not in trained categories)",
          file=sys.stderr)
    return fallback


def _pred_to_v3(node, is_index_cond=False):
    """Convert a public PredicateNode.to_dict() tree to the v3 `filter` tree.

    Public node: {column(->col_id after lookup), operator(str), literal, literal_feature, children}
    v3 leaf:     {col_id, operator, literal, filter_complexity, is_index_cond}
    v3 logical:  {operator: 'AND'|'OR', is_index_cond, children:[...]}

    Returns None for predicates that cannot be encoded as filters (join conditions,
    unresolved columns); callers drop None.
    """
    raw_children = node.get("children", []) or []
    children = [c for c in (_pred_to_v3(ch, is_index_cond) for ch in raw_children) if c is not None]
    op = node.get("operator")
    complexity = node.get("literal_feature", 0) or 0

    if not raw_children:  # leaf predicate
        col = node.get("column")
        if not isinstance(col, int):
            return None  # column not resolved to an id
        literal = node.get("literal")
        if isinstance(literal, str):
            literal = literal.strip()
        if isinstance(literal, str) and _COLREF_RE.match(literal):
            return None  # join condition (col = col) — see _COLREF_RE note
        return {"operator": op, "is_index_cond": is_index_cond,
                "filter_complexity": complexity, "col_id": col, "literal": literal}

    # logical node (AND/OR)
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return {"operator": op, "is_index_cond": is_index_cond,
            "filter_complexity": complexity, "children": children}


def _planop_to_v3(node, dataset):
    """Serialize a parsed PlanOperator into a plain v3 dict the dataloader consumes."""
    pp = dict(node.plan_parameters)
    children = [_planop_to_v3(c, dataset) for c in node.children]

    op_name = _normalize_op_name(pp.get("op_name"), node.children)

    v3 = {
        "op_name": op_name,
        "est_card": pp.get("est_card"),
        "est_width": pp.get("est_width"),
        "pg_cost": pp.get("est_cost"),
        "est_loops": pp.get("est_loops", 1) or 1,
        "est_children_card": pp.get("est_children_card", 1) or 1,
        # est featurization ignores act_*; dummies keep the root-aggregation-enforcement
        # branch in postgres_plan_batching.py from KeyError-ing on EXPLAIN-only input.
        "act_card": 1, "act_time": 0.0, "act_loops": 1,
        "act_children_card": 1,
        "above_udf_filter": False, "is_udf_filter": False,
    }
    if pp.get("table") is not None:
        v3["table_id"] = pp["table"]  # already mapped to id by parse_columns_bottom_up
    if pp.get("output_columns") is not None:
        v3["output_columns"] = [
            {"aggregation": oc.get("aggregation"), "columns": [
                {"col_id": cid} for cid in oc.get("columns", []) if isinstance(cid, int)
            ]}
            for oc in pp["output_columns"]
        ]
    if pp.get("filter_columns") is not None:
        f = _pred_to_v3(pp["filter_columns"])
        if f is not None:
            v3["filter"] = f
    return {"plan_parameters": v3, "children": children}


def _count_tables_filters(v3_node):
    """Count scan nodes (table instances) and leaf filter predicates in a v3 plan tree."""
    pp = v3_node["plan_parameters"]
    n_tables = 1 if pp.get("table_id") is not None else 0

    def _leaves(f):
        if not f:
            return 0
        if "children" in f:
            return sum(_leaves(c) for c in f["children"])
        return 1

    n_filters = _leaves(pp.get("filter"))
    for c in v3_node["children"]:
        ct, cf = _count_tables_filters(c)
        n_tables += ct
        n_filters += cf
    return n_tables, n_filters


def parse_candidates_to_v3(candidates, database_stats):
    """Parse each candidate's EXPLAIN(VERBOSE) text into a v3 parsed plan.

    `candidates`: { id: {"sql": str, "verbose_plan": [line, ...]} }
    Returns the workload dict {parsed_plans, database_stats, run_kwargs}.
    """
    from types import SimpleNamespace
    from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plans

    def _ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_ns(v) for v in obj]
        return obj

    parsed_plans = []
    for cand_id, cand in candidates.items():
        lines = cand["verbose_plan"]
        run_stats = SimpleNamespace(
            database_stats=_ns(database_stats),  # fresh copy per query (parser mutates table_size)
            query_list=[SimpleNamespace(
                verbose_plan=[[ln] for ln in lines],
                analyze_plans=None, sql=cand.get("sql"), timeout=False,
            )],
            run_kwargs={},
        )
        parsed_runs, _ = parse_plans(
            run_stats, min_runtime=None, max_runtime=None,
            parse_baseline=False, parse_join_conds=True,
            include_zero_card=True, explain_only=True,
        )
        plans = parsed_runs["parsed_plans"]
        if not plans:
            print(f"WARNING: no plan parsed for candidate {cand_id}", file=sys.stderr)
            continue
        v3 = _planop_to_v3(plans[0], dataset="imdb")
        n_tables, n_filters = _count_tables_filters(v3)
        # Full v3 top-level field set the dataloader / extract_query_stats expects.
        v3.update({
            "plan_runtime_ms": 0, "plan_id": str(cand_id),
            "parent_plan_id": None, "subplan_id": None, "dataset": "imdb",
            "num_tables": n_tables, "num_filters": n_filters,
            "hint": None, "invalid_hint": False, "timeout": False,
            "timeout_threshold": None, "zero_card": False, "sql": cand.get("sql"),
        })
        parsed_plans.append(v3)

    return {"parsed_plans": parsed_plans, "database_stats": database_stats, "run_kwargs": {}}


# ───────────────────────────────────────── io ──────────────────────────────────────────────
def _load(path):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _write_workload(workload, path):
    with open(path, "wb") as f:
        f.write(orjson.dumps(workload))


def main():
    ap = argparse.ArgumentParser(description="ZeroShot learned-cost scorer.")
    ap.add_argument("--mode", choices=["predict-parsed", "predict-candidates"], required=True)
    ap.add_argument("--input", required=True, help="parsed v3 file (predict-parsed) or candidates.json")
    ap.add_argument("--output", required=True, help="predictions.json")
    ap.add_argument("--database-stats", default=None,
                    help="JSON with {'database_stats': {...}} (predict-candidates only).")
    ap.add_argument("--model-type", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--statistics-file", required=True)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "predict-parsed":
        parsed_path = args.input
    else:
        candidates = _load(args.input)
        db_stats_blob = _load(args.database_stats)
        database_stats = db_stats_blob["database_stats"] if "database_stats" in db_stats_blob else db_stats_blob
        workload = parse_candidates_to_v3(candidates, database_stats)
        fd, parsed_path = tempfile.mkstemp(suffix=".json", prefix="zs_candidates_")
        os.close(fd)
        _write_workload(workload, parsed_path)
        print(f"Parsed {len(workload['parsed_plans'])} candidates to {parsed_path}")

    preds = predict_parsed_file(
        parsed_path,
        model_type=args.model_type, model_dir=args.model_dir, seed=args.seed,
        statistics_file=args.statistics_file, device=args.device, batch_size=args.batch_size,
    )

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
