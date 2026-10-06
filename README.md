# Semantic SQL Rewrites

Code artifact of the paper *Towards Semantic Query Rewrites from LLM-Mined Rules*.

A **semantic query rewrite** returns the same result as the original query on the database
instance that is actually stored, without being equivalent on every possible instance. The
rewrite type implemented here is **filter-predicate injection**: an LLM proposes a
world-knowledge fact about the query's literal values, and the corresponding predicate is
added conjunctively to the `WHERE` clause. Running example: `n.name LIKE '%Downey%Robert%'`
denotes one real person, so `n.gender = 'm'` may be injected.

Each rule is validated against the base tables before it may be applied, and admitted only
if every row satisfying its `requires` also satisfies its `implies`. Since that check
concerns the data and not a query, an admitted rule transfers across the workload.

The code keeps the two halves apart, as the paper does: *mining* sound rules, and
*selecting* which applicable rules to inject (a performance question only).

> The paper carries the rationale for these design decisions; this README and the code
> comments only say what things are, not why.

## Repository layout

```
.
├── systematic_eval/        # The evaluation pipeline (the main deliverable)
│   ├── run_pipeline.py         # Pipeline entrypoint (python -m systematic_eval.run_pipeline)
│   ├── controller.sh           # Orchestrates local + remote (SSH) runs, resume stages
│   ├── sql_predicate_converter.py  # Predicate engine: matching, injection, rewriting
│   ├── config_loader.py        # YAML config → ExperimentConfig dataclasses
│   ├── rule_schema.py          # Structural validation of rule JSON
│   ├── prompt_loader.py        # Loads/parameterizes prompts
│   ├── parsing.py              # LLM-output parsing
│   ├── pg_lab_setup.py         # PostgreSQL backend setup
│   ├── umbra_setup.py          # Umbra backend setup
│   ├── run_pg_query.sh         # Ad-hoc Postgres query helper
│   ├── remote_requirements.txt # Minimal deps for the self-contained remote executor
│   ├── stages/                 # Pipeline stages + the shared execution substrate
│   ├── scorer/                 # Standalone ZeroShot learned-cost scorer (isolated venv)
│   ├── dataset_setup/          # Tooling to onboard new datasets
│   ├── prompts/                # Prompt library, one folder per dataset
│   ├── config/                 # Experiment configs (YAML), one per reported run
│   ├── postgres/               # PostgreSQL tuning config for the lab server
│   ├── world_knowledge_eval/   # Post-hoc: LLM judge rating each rule's world-knowledge dependence
│   ├── scripts/                # Post-hoc reporting (thesis_numbers/_figures) + one-off helpers
│   ├── transfer_data/          # Per-experiment intermediate stage I/O (gitignored)
│   └── saved_results/          # Final experiment outputs (gitignored)
├── llm_helpers/            # The group's LLM API wrapper (external to this thesis)
├── sql/                    # Per-dataset SQL workloads — NOT shipped, see "Setup"
├── llm_cache/              # On-disk LLM response cache — NOT shipped, see "Setup"
├── pyproject.toml          # Project + dependencies (Python 3.13, managed by uv)
├── .env.example            # Template for the API key; copy to .env
├── CLAUDE.md               # Guidance for the Claude Code assistant
└── README.md               # This file
```

> This repository contains **only the productive code**: the pipeline, the prompt folder and
> the config of every experiment reported in the paper. The paper document, the query
> workloads, the databases and the LLM cache are maintained separately and are not part of
> this code submission.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python 3.13):

```bash
uv sync
```

Copy `.env.example` to `.env` (not committed) and fill in `OPENAI_API_KEY`. Responses are
cached on disk under `llm_cache/`, so re-running an experiment with the same inputs incurs
no API cost.

### Data you must supply yourself

Neither the query workloads (`sql/`), the databases (`*.duckdb`) nor the LLM cache
(`llm_cache/`) are part of this repository — they are too large to distribute and, in the
case of the source data, not ours to redistribute. **A fresh clone therefore contains the
pipeline and every prompt and experiment config, but cannot execute a run until the
following are in place.**

| What | Where it goes | Where to get it |
|---|---|---|
| **JOB** — 113 analytical queries over IMDB, the primary workload | `sql/job/` | The Join Order Benchmark of Leis et al., *How Good Are Query Optimizers, Really?* (VLDB 2015); queries at [gregrahn/join-order-benchmark](https://github.com/gregrahn/join-order-benchmark) |
| **IMDB database** | `imdb.duckdb` on the execution host | Build from the IMDB dump the JOB repository points to |
| **19 further workloads + schemas** for the generality study | `sql/<name>_200k/` | The twenty-database collection published with the zero-shot cost model of Hilprecht & Binnig — 17 real-world schemas from the CTU Prague Relational Learning Repository plus the synthetic SSB and TPC-H. Each dataset ships a generated workload |
| **LLM cache** (optional) | `llm_cache/` | Only needed to reproduce a run *without* paying for API calls; regenerated automatically otherwise; archived copy: see [Archive](#archive) |

JOB is run without its `MIN(...)` wrappers: `dataset.strip_min: true` removes them as the
queries are read, so a rewrite's effect on the actual result rows stays observable. The
generality workloads keep their scalar aggregate (`strip_min: false`).

Once the raw data is available, `dataset_setup/` onboards a dataset end to end —
`add_dataset.sh` builds the DuckDB file (`build_duckdb.py`), splits the shipped workload
into one query per file (`split_workload.py`), and renders the prompt folder and config
(`render_prompts.py`, `make_config.py`). Paths in these scripts point at the lab mount and
need adjusting for another environment.

## Running the pipeline

**Full orchestration** (local generation + remote execution over SSH):

```bash
cd systematic_eval/
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2                  # full run
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2 --version v01    # suffix transfer_data/ + saved_results/ with __v01
./controller.sh experiment_T_imdb_job_12_oracle_c07_4-2 --stage 3        # resume from a later stage
```

`controller.sh --stage` resume points:

| Stage | Starts at | Requires |
|-------|-----------|----------|
| `1` (default) | Full pipeline: generate → exec → refine → aggregate → stats | — |
| `2` | Refinement loop | `transfer.json` + `result.json` |
| `3` | Aggregation + final execution | final refine outputs |
| `4` | Statistics only (no remote work) | `rule_summary_result.json` |

**Individual stages** (local only):

```bash
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage generate
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage refine --iteration 0
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage aggregate
python3 -m systematic_eval.run_pipeline --config config/<experiment>.yaml --stage stats
```

## Pipeline stages

The pipeline is four stages, and they are exactly the `controller.sh --stage` resume points.

1. **Generation** (`stages/generation.py`) — multi-sampled LLM rule generation, dedup, rule
   completion (`joins`/`alias_map` derived from the source query), then validation.
2. **Refinement** (`stages/refinement.py`) — failed candidates go back to the model with the
   violation rows; repairs are re-validated.
3. **Aggregation** (`stages/aggregation.py`) — validity gate, merging, cross-query transfer,
   then the final timed execution.
4. **Statistics** (`stages/statistics.py`) — CSV reports and plots for one run.

`stages/execution.py` is not a stage but the **execution substrate**: base-table validation
in stages 1–2 (parallel workers, shared node) and the timed measurement in stage 3 (serial,
node-exclusive, one core per engine). It carries no project imports, so `controller.sh` can
scp it to the remote host standalone.

## Execution backends

| Backend | Configs | Setup |
|---------|---------|-------|
| **DuckDB** (default) | configs without a backend suffix | — |
| **PostgreSQL** | `*_pg.yaml` | `pg_lab_setup.py`, ad-hoc queries via `run_pg_query.sh` |
| **Umbra** | `*_umbra.yaml` | `umbra_setup.py`, debug via `scripts/debug_umbra_explain.py` |

Rules are mined and validated once on DuckDB; the identical rule set is re-executed on the
other two engines.

PostgreSQL server settings live in `systematic_eval/postgres/postgresql16.conf`. The planner's
`random_page_cost` is the exception: set it per experiment via `execution.postgres_random_page_cost`
in the config YAML. Omitting the key leaves it at postgres' default (4.0).

An optional **learned-cost scorer** (`scorer/`, ZeroShot cost estimation) can rank candidate
rule subsets instead of the planner's estimate; it runs in its own isolated venv and is
invoked by `execution.py --mode zeroshot-aggregate`.

## Datasets

Prompts exist for 20 benchmarks:

`accidents`, `airline`, `baseball`, `basketball`, `carcinogenesis`, `consumer`, `credit`,
`employee`, `fhnk`, `financial`, `geneea`, `genome`, `hepatitis`, `imdb_job`, `movielens`,
`seznam`, `ssb`, `tournament`, `tpc_h`, `walmart`.

`imdb_job` (the JOB benchmark over IMDB) is the primary workload and the mining substrate;
the other nineteen serve the generality check. A workload contributes only a prompt folder
and a config file — no dataset-specific code — and is onboarded with the tooling in
`dataset_setup/`.

## Configuration

Each experiment is one YAML file in `systematic_eval/config/`. **The file name is the
experiment's identity** — it selects `transfer_data/<name>/` and `saved_results/<name>/`; the
config carries no name field of its own.

The 27 shipped configs are exactly the runs the paper reports:

- eight `experiment_T_imdb_job_12_*` runs, one per experimental condition — the three engines
  with the plan free, DuckDB and Postgres with the plan pinned, the learned ranking signal on
  Postgres, the `random_page_cost` variant, and the generation-budget ablation;
- nineteen `experiment_T_gen_<dataset>` runs for the generality study.

Name suffixes: `_oracle` emits the oracle ceiling next to the selector view
(`statistics.mode: both`); `_join_p` is **join-order pinned** (`fix_join_order: true`), a
confounder control rather than a second selection method; `_pg` / `_umbra` swap the engine at
a fixed rule set; `_zeroshot` swaps the planner estimate for the learned ZeroShot cost model;
`_c07` names the lab server; `4-2` / `1-1` records `generation.rounds`–`refinement.samples`;
`_rpc1-1` records `random_page_cost`.

`config/experiment_template.yaml` is the annotated reference: it lists **every** accepted
parameter with its default, grouped into `dataset`, `prompts`, `generation`, `refinement`,
`execution`, `remote`, `aggregation` and `statistics`. `config_loader.py` is the authoritative
schema (unknown keys inside a group are rejected).

Two points worth knowing before reading a result:

- `statistics.mode` selects the reported view: `oracle` = fastest output-matching subset per
  query (the ceiling; a measurement construct, not deployable), `optimizer` = the subset the
  engine's estimate or the learned cost model picked (deployable, can regress), `both` = the
  two side by side from a single run.
- `statistics.show_stats` and `statistics.runtime_aggregator` are inert for every run reported
  in the paper. Under `mode: oracle`/`both` or `cost_estimation: true` the runtimes come from
  `per_subset_results`, which stores the execution stage's median and no per-attempt list, so
  no spread columns are written. `execution.attempts` is likewise collapsed by the execution
  stage itself, always to the median.

Memory/time guards worth setting on large workloads: `execution.query_timeout_s` aborts a
runaway query and discards its rule, while `max_temp_size` (DuckDB), `umbra_memory_gb` (Umbra
container) and `driver_memory_gb` (client process) make a run fail catchably instead of
OOM-killing the lab node.

## Rule format

```json
{
  "id": "IMDB_R1",
  "requires": {"op": "AND", "conditions": [{"column": "t.col", "op": "=", "value": "..."}]},
  "implies": [{"column": "t.col", "op": "=", "value": "..."}],
  "joins": [{"left": "a.col", "right": "b.col"}]
}
```

`requires` is a recursive AND/OR tree (the context a query must guarantee); `implies` the
flat list of injected predicates. Only these two are written by the LLM — `joins` and
`alias_map` are derived from the source query by the pipeline.

## Validation modes

- **`base_tables`** — the soundness gate: the rule is evaluated over its own tables (joined
  on the derived `joins`) and admitted only if no row satisfies `requires` without
  satisfying `implies`. Required for cross-query transfer.
- **`query`** — per-query comparison of original vs. rewritten output. Does not transfer.
- **`both`** — run both checks.

## Outputs & data

- `systematic_eval/transfer_data/<experiment>/` — intermediate stage I/O
  (`transfer.json` → `result.json` → refinement → `rule_summary_*.json`).
- `systematic_eval/saved_results/<experiment>/` — final results, `results.csv`, plots.
- `llm_cache/` — cached LLM responses.

All three are gitignored and not shipped (large / regenerable). Within a working copy the
LLM cache is worth preserving: it makes a re-run reproduce the same rules at no API cost.

### Archive

The data behind the 27 reported runs is archived on the TU Darmstadt Systems Group labstore
under `/mnt/labstore/psiegler/thesis_archive/` (layout mirrors this repository; restore notes
in its `ARCHIVE_README.md`): `saved_results/` and the non-duplicate part of `transfer_data/`
for every config in `systematic_eval/config/`, the complete `llm_cache/`, the `sql/`
workloads, the DuckDB files and a git bundle of this repository. Created with
`systematic_eval/scripts/archive_to_labstore.sh`.

## Post-hoc analyses

Two pieces of tooling sit outside the four stages and run after them:

- `scripts/thesis_numbers.py`, `scripts/thesis_figures.py` — the reporting layer: every
  number and figure of the paper's evaluation chapter, emitted from the saved run artifacts
  under a named label. The remaining files in `scripts/` are one-off helpers (demos, debug
  tools, data probes), **not** part of the pipeline; all resolve project paths relative to
  `systematic_eval/`.
- `world_knowledge_eval/` — an LLM judge rates each validated rule 1–5 on world-knowledge
  dependence, annotated with that rule's single-rule speedup.

## Notes

- No automated test suite or linter is configured.
- Development notes for the Claude Code assistant live in `CLAUDE.md`.
