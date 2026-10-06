# world_knowledge_eval

Post-hoc analysis (not a pipeline stage): for each **validated** rewrite rule of
an oracle run, an OpenAI judge rates on a **1–5 Likert scale** how strongly the
rule depends on real-world domain knowledge versus being a property of the
database instance that a data-profiling tool could discover on its own
(functional/approximate dependency, near-constant column, etc.). Each rule is
annotated with its **individual (single-rule) oracle performance**.

- `1` = fully DB-derivable · `5` = fully world-knowledge-dependent.

The judge shares a model family with the author of the rules; the prompt controls for the
resulting self-referential bias by requiring the *actual* dependence to be scored rather
than how confidently the rationale is phrased.

## What it reads (source data)

All under `systematic_eval/transfer_data/<experiment>/`:

| Piece | File | Field |
|---|---|---|
| Validated, merged rules (survived aggregation + `base_tables` validation) | `rule_summary_result.json` | per-query `rules[].{name,rule}` |
| Individual rule performance | `rule_summary_result.json` | `per_subset_results` **singleton** subsets (`rule_names` of length 1) → `subset_transformed_query` vs `original_query` |
| Model's reasoning (`short_rationale`) | `result.json`, `result2.json` | `<entry>.original_output` (raw LLM JSON), matched to each rule by its `R<n>` id |

Only rules present in `rule_summary_result.json` are considered — i.e. exactly
the rules that were validity-tested. A merged rule name (`merged: A + B`) is
decomposed into its source keys to gather every component's rationale.

## Judge input

A generated prompt (`prompt.txt`) with the assessment task, the rule JSON
(`requires → implies`), and the model's stated rationale. The source SQL is kept
in `rules_extracted.json` but not sent by default.

The prompt is dataset-agnostic: its `{dataset}` and `{schema}` placeholders are
filled from `config.dataset.name` and that dataset's `PromptLoader` schema, so a
non-IMDb run is never framed as IMDb. Both are substituted literally — keep the
JSON response template in `prompt.txt` written with single braces.

## Usage

```bash
# 1) extraction only — no API calls (streams the multi-GB oracle file, ~minutes)
uv run python3 -m systematic_eval.world_knowledge_eval \
    --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --extract-only

# 2) smoke test — judge 2 rules, reusing the extraction
uv run python3 -m systematic_eval.world_knowledge_eval \
    --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --reuse-extract --limit 2

# 3) full run
uv run python3 -m systematic_eval.world_knowledge_eval \
    --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --reuse-extract --budget 2.0
```

Key flags: `--model` (default: config's generation model), `--limit N`,
`--reuse-extract` (skip re-streaming), `--budget` (LLM cost gate; negative =
no prompt), `--no-cache`.

## Outputs → `saved_results/<experiment>/world_knowledge/`

- `rules_extracted.json` — rule + rationale + per-query performance + aggregate (always).
- `wk_eval.json` — the above plus each rule's `verdict` (`score`, `label`, `justification`).
- `wk_eval.csv` — one flat row per rule: score, label, individual performance, justification, rationale.

LLM calls go through `llm_helpers` (OpenAI, disk-cached, cost-tracked), so
re-runs are free and reproducible.
