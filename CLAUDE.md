# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Code artifact of the master thesis *Semantic Query Rewrites*. Rewrites are correct on the
stored database instance rather than on every instance; the implemented rewrite type is
**filter-predicate injection** (a predicate added conjunctively to the WHERE clause).

Two halves the repo keeps apart, and so should you: **mining** (rule generation +
base-table validation + transfer — a correctness question) and **selection** (which
applicable rules to inject — a performance question only).

See `README.md` for the project overview. The thesis itself carries all rationale; code
comments here should stay short and not restate it.

**Repository scope**: this repo holds only productive code. The thesis document, the query
workloads, the databases and the LLM cache are kept separately.

## Commands

```bash
uv sync          # install dependencies (Python 3.13, managed by uv)
```

Full orchestration (local stages + remote execution over SSH):

```bash
cd systematic_eval/
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2                       # full run
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2 --version v01         # suffix transfer_data/ + saved_results/ with __v01
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2 --stage 3             # resume: 1=full, 2=refine, 3=aggregate+final exec, 4=stats only
```

Individual stages (local only):

```bash
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage generate
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage refine --iteration 0
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage aggregate
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage stats
```

No test suite or linter is configured.

## Architecture

### 4-Stage Pipeline

The four stages *are* the `controller.sh --stage` resume points. Stages 1–3 additionally
invoke the execution backend on a lab server over SSH; stage 4 is local only.

1. **Generation** (`stages/generation.py`): multi-sampled LLM rule generation, dedup by rule
   signature, rule completion (`joins`/`alias_map` from the source query), then validation.
2. **Refinement** (`stages/refinement.py`): failed candidates back to the model with
   violation rows; repairs are re-validated.
3. **Aggregation** (`stages/aggregation.py`): validity gate, merging, cross-query transfer,
   then the final timed execution.
4. **Statistics** (`stages/statistics.py`): per-run CSV reports and plots.

`stages/execution.py` is not a stage but the **execution substrate**: base-table validation
in stages 1–2 (parallel workers, shared node) and the timed measurement in stage 3 (serial,
node-exclusive, one core). Self-contained — `controller.sh` scps it to the remote host.

### Execution Backends

- **DuckDB** — default (configs without a suffix), and the mining/reporting substrate.
- **PostgreSQL** — `*_pg.yaml` configs; setup via `pg_lab_setup.py`, ad-hoc query via `run_pg_query.sh`.
- **Umbra** — `*_umbra.yaml` configs; setup via `umbra_setup.py`, debug helper in `scripts/debug_umbra_explain.py`.

Rules are mined and validated **once** on DuckDB; the identical set is re-executed on the
other engines.

### Learned-Cost Scorer: `scorer/`

Standalone ZeroShot learned-cost scorer, isolated in its own py3.12 venv (torch/dgl) so the
main py3.13 pipeline stays clean. Invoked by `execution.py --mode zeroshot-aggregate` via
JSON files to predict candidate rule-subset runtimes and pick the cheapest. See
`scorer/README.md`.

### Predicate Engine: `sql_predicate_converter.py`

Applies a validated rule to a query. Application is a *logical* operation, not a textual one,
and this is where the runtime side of the soundness argument lives:
- `extract_predicates(sql)` — the query's guaranteed predicates; descends AND/parens, stops
  at OR. Gotcha: a negated group `NOT (a AND b)` slips through (no corpus query has one).
- `entails(query_pred, rule_pred)` — operator-aware subsumption (numeric bounds,
  BETWEEN ↔ paired `>=`/`<=`, IN-list subset, `=`/LIKE/IS/IS NOT); deliberately incomplete.
- `apply_rules(predicates, rules, ...)` — fixed point; checks (C3) and (C4). Terminates
  because the loop only ever adds (`drops` collected separately, reduction runs after).
- `rewrite_sql(original_sql, predicates, ...)` — textual top-level-AND append, no AST
  round-trip.
- `apply_sql_rules(sql, rules)` — end-to-end: parse → apply → reduce → rewrite.
- `derive_rule_joins(sql, rule)` — the equi-joins validation runs over. **Mandatory**: an
  uncompleted rule has no `alias_map`, fails the table guard in `apply_rules`, never fires.

### World-Knowledge Evaluation: `world_knowledge_eval/`

Post-hoc analysis (not a pipeline stage): an OpenAI judge rates each **validated** rule of an
oracle run 1–5 on how strongly it depends on real-world domain knowledge, annotated with the
rule's single-rule oracle performance. See its `README.md`.

```bash
python3 -m systematic_eval.world_knowledge_eval --experiment <name> [--reuse-extract]
```

### Rule Format

```json
{
  "id": "IMDB_R1",
  "requires": {"op": "AND", "conditions": [{"column": "t.col", "op": "=", "value": "..."}]},
  "implies": [{"column": "t.col", "op": "=", "value": "..."}],
  "joins": [{"left": "a.col", "right": "b.col"}]
}
```

`requires` is a recursive AND/OR tree (the context a query must guarantee); `implies` the
flat list of injected predicates. Only these two come from the LLM — `joins`/`alias_map` are
derived from the source query afterwards, which is why `rule_schema.py` does not validate them.

### Validation Modes

- **"base_tables"** — the soundness gate; required for cross-query transfer.
- **"query"** — per-query output-equality check; does not transfer.
- **"both"** — both checks.

### Key Supporting Modules

- `config_loader.py` — YAML config → `ExperimentConfig` dataclass hierarchy
- `rule_schema.py` — structural validation of rule JSON
- `prompt_loader.py` — loads system/generation/refinement prompts, substitutes schema
- `llm_helpers/` — the group's LLM API wrapper (caching, token counting, cost tracking);
  external to this thesis, do not restructure

### Data Flow

Stages communicate through files in `systematic_eval/transfer_data/<experiment>/`; each pair
is "what a stage sends to the execution backend" → "what comes back":
- `transfer.json` → `result.json` (stage 1: generation, then validation)
- `transfer<N>.json` → `result<N>.json` (stage 2: one pair per refinement iteration)
- `rule_summary_transfer.json` → `rule_summary_result.json` (stage 3: aggregation → final
  timed execution; `cost_aggregate_input.json` precedes it when cost estimation is on, and
  `baseline_input` → `baseline_runtimes` measures the no-rule queries for the
  whole-workload denominator)
- `statistics_summary.json`, `results.csv` (stage 4 output)

LLM responses are cached on disk (`llm_helpers/llms.py` sets `CACHE_PATH = "llm_cache"`,
relative to the repo root where the pipeline is invoked). The **active cache is the
top-level `llm_cache/`** — preserve it for reproducibility. `systematic_eval/llm_cache/` is
an unused location kept only as an empty `.gitkeep` placeholder.

### `scripts/`: reporting layer + ad-hoc helpers

- **Reporting layer** (post-hoc, over finished runs): `thesis_numbers.py` emits every number
  quoted in the thesis's evaluation chapter under a named label; `thesis_figures.py` every
  figure and generated table. Also holds the *cross-run* aggregates, as opposed to the
  per-run statistics of stage 4.
- **Ad-hoc helpers**, not part of the pipeline: `performance_overview.py`,
  `demo_cost_estimation.py`, `demo_pg_runtime.py`, `basketball_setup.py`,
  `basketball_split_workload.py`, `debug_umbra_explain.py`, `duckdb_where_order.py`
  (+ `run_on_c06.sh`).

All resolve project paths relative to `systematic_eval/` (one level up).

## Configuration

YAML configs in `systematic_eval/config/`. `config_loader.py` is the authoritative schema
(dataclass defaults + `__post_init__` validation); `config/experiment_template.yaml` documents
**every** accepted key with its default and is kept in sync with the loader — update both when
adding a parameter. Unknown keys inside a group raise a `TypeError` from the dataclass
constructor; unknown **top-level** keys are silently ignored, because `load_config` picks
those explicitly.

An experiment is identified by its **config file name**, not by any field inside the file:
`experiment_<name>.yaml` drives `transfer_data/experiment_<name>/` and
`saved_results/experiment_<name>/`. `controller.sh`, `run_pipeline.py` and
`world_knowledge_eval` all use that stem.

The 27 shipped configs are exactly the runs the thesis reports: eight `T_imdb_job_12_*` runs
(one per experimental condition) and nineteen `T_gen_*` generality runs. Name suffixes:
`_join_p` = join-order **pinned** (`fix_join_order: true`, a confounder control — not a
selection method), `_oracle` (ceiling next to the selector view), `_pg` / `_umbra` (engine),
`_zeroshot` (learned cost model on Postgres), `_c07` (lab server), `4-2` / `1-1`
(`generation.rounds`–`refinement.samples`), `_rpc1-1` (`random_page_cost`).

Traps that the template alone does not make obvious:

- `statistics.mode: both` makes `controller.sh` pass `--keep-full-pool` to final execution, so
  the oracle bound and the optimizer pick come from the *same* measured data (no subset is
  executed twice). The `oracle` view reports improvements only (the empty subset is always a
  candidate); `optimizer` can regress.
- `statistics.show_stats` and `statistics.runtime_aggregator` are **silently inert** under
  `mode: oracle`/`both` or `cost_estimation: true`: those views read `per_subset_results`, which
  stores the execution stage's median and no per-attempt list. Only the legacy all-rules
  optimizer view (`build_rows`) honours them.
- `execution.attempts` is collapsed by the execution stage itself, always to the median
  (per-pipeline median summed on Umbra) — `runtime_aggregator` does not govern that step.
- `execution.performance_threshold` is inert: the flag it computes is recorded in the result
  artifact but never acted on. Rule acceptance is governed by `aggregation.time_filtering`.
- `controller.sh` re-reads most `execution.*`, `remote.*` and `statistics.mode` keys itself (via
  `read_config`) to build the remote command line — a new execution parameter usually needs a
  matching `read_config` line there, not just the dataclass field.
- Postgres planner knobs come from `postgres/postgresql16.conf` (applied via `ALTER SYSTEM`),
  except `random_page_cost`, which is per-experiment: set `execution.postgres_random_page_cost`
  and `controller.sh` forwards it as `pg_lab_setup.py --set random_page_cost=…` (applied after
  the conf file, so it wins). Omit the key to leave postgres' default of 4.0. Verify end-to-end
  with `postgres/verify_pg_conf.sh <server> <remote_path> <port> [value]`.
