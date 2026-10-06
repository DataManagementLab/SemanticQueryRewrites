# ZeroShot learned-cost scorer

Standalone, **isolated** scorer for the semantic-rewrite pipeline. Runs in its own
py3.12 venv (torch 2.3 + dgl cu121 + `ldb_models`) so the main pipeline (py3.13) stays
clean. Communicates with `execution.py` only via JSON files.

Used by `execution.py --mode zeroshot-aggregate`: that stage writes `candidates.json`
(one Postgres `EXPLAIN (VERBOSE)` per candidate rule-subset) and invokes this scorer to
predict each candidate's runtime; the lowest-predicted subset is selected.

It replaces the ranking signal only, not the subset search, and leaves execution untouched.
The model is a pre-trained ZeroShot checkpoint applied without fine-tuning.

## Layout
- `score_plans.py` — CLI. Modes: `predict-parsed` (score an existing v3 parsed-plans file —
  the reliable path, used for verification gate 1) and `predict-candidates` (parse fresh
  EXPLAINs to v3, then score).
- `vendor/cross_db_benchmark/` — vendored Postgres plan parser (public
  `github.com/DataManagementLab/zero-shot-cost-estimation`). Local deltas vs upstream:
  `generate_workload.py` trimmed to enums + `Operator.GT/LT` added; `parse_filter.py`
  patched so `>`/`<` stay distinct from `>=`/`<=` (the model's `operator` categorical
  distinguishes them).
- `database_stats_imdb.json` — `{ "database_stats": {...} }` lifted from an
  `imdb_scaled1/parsed_plans3` file (col_id/table_id are indices into these lists).
- `pyproject.toml` — deps (private `ldb_models` repo: needs git credentials on the host, or `LDB_MODELS_TOKEN` exported for the pip fallback).

## Setup (on the remote execution host)
```
cd scorer && uv venv && uv sync          # or python3.12 -m venv + pip install
```

`controller.sh` does this automatically for `cost_model: zeroshot` runs and verifies
`import ldb_models, torch, dgl` afterwards. **uv is required on the remote** — the pip
fallback cannot install the private `ldb_models` repo reliably. The venvs live on the
shared labstore mount, so an existing working one can be reused across servers instead
of rebuilding:
```
ZS_SCORER_PYTHON=/mnt/labstore/psiegler/c06_zeroshot_pg/scorer/.venv/bin/python \
  ./controller.sh <experiment> --stage 3
```

## Gate 1 — env + model (no parser)
```
.venv/bin/python score_plans.py --mode predict-parsed \
  --input  /mnt/labstore/SIGs/ML/learned_db/runs/parsed_plans3/imdb_scaled1/access_path_selection.json \
  --output /tmp/preds.json \
  --model-type "pm-zeroshot-include_hint_idx_workloads-limit_plans_per_file10001" \
  --model-dir  /mnt/labstore/SIGs/ML/learned_db/models/pm-zeroshot-hi-limitpf10001/imdb \
  --seed 9 \
  --statistics-file /mnt/labstore/SIGs/ML/learned_db/feature_statistics/parsed_plans3_feature_statistics_v20260309.json
```
Sanity-check predictions vs the shipped `pm-zeroshot-hi-limitpf10001_9.csv`.

## ⚠ Parser drift (gate 2)
The v3 schema (op_name categories `Simple Aggregate` / `Index Nested Loop`, the
`filter`/`join` split, per-node `is_index_cond`) comes from the lab's **internal**
cross_db_benchmark fork. `score_plans.py:_planop_to_v3 / _normalize_op_name / _pred_to_v3`
reproduce it best-effort. Validate by diffing a freshly-parsed node against a real
`parsed_plans3` node. Given access to the lab's exact parser, drop it into
`vendor/cross_db_benchmark/` (replacing the public copy) and remove the adapter shims.
