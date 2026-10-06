"""Render a per-dataset prompt directory from one shared scaffold + domain_specs.

Produces, under ``systematic_eval/prompts/<short>/``:
    generation/prompt_wk01.txt   - rendered from the scaffold below (mirrors the hand-written
                                   basketball/imdb prompts); keeps literal {schema}/{sql}/{rule_example}
    examples/rule_example.json   - the domain example rule
    system/prompt01.txt          - copied verbatim from prompts/imdb_job (identical across datasets)
    refinement/refine_prompt02.txt - copied verbatim from prompts/imdb_job

``schema.txt`` is written separately by the orchestrator (from the DB-derived sidecar);
this module only fills it in if ``--schema-file`` is supplied.

The scaffold uses ``<<TOKEN>>`` placeholders (filled here) and leaves ``{schema}``,
``{sql}``, ``{rule_example}`` untouched — those are substituted downstream by
``prompt_loader.load_generation_prompt`` and ``stages/generation.py``.

Usage:
    python3 -m systematic_eval.dataset_setup.render_prompts --short financial
    python3 -m systematic_eval.dataset_setup.render_prompts --short financial --schema-file /tmp/financial_schema.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from systematic_eval.dataset_setup.domain_specs import get_spec
except ImportError:  # allow `python3 render_prompts.py` from this dir
    from domain_specs import get_spec  # type: ignore

PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
GENERIC_SOURCE = "imdb_job"  # system + refinement prompts are identical across datasets

SCAFFOLD = """You are a senior database performance engineer. Your task is to come up with rewriting rules for a given SQL query over <<DISPLAY>>, to turn it into a faster variant by applying <<ADJ>>-specific shortcuts and pragmatic heuristics. These are expressed by filters that can be added, because the context of <<DATA_REF>> allows it. You are encouraged to produce semantically non-equivalent queries (when your rules are applied to the original query) as long as it remains valid on <<DATA_REF>> and preserves the intent. Leverage <<KNOWLEDGE>> to justify simplifications.

<<SCHEMA_LABEL>> schema (tables and columns):
{schema}

Optimization focus
- Only apply shortcuts that are realistic for <<DATA_REF>> and likely to speed up queries. You may not drop rare/edge cases to simplify the plan.
- Do not invent tables/columns. Use exactly the column names from the schema (they are case-sensitive and used quoted).
- Prefer sargable predicates (e.g. range bounds) that the optimizer can use to prune scans.
- Use your knowledge about the data and the requested info to make sure that the result stays the same.
- Focus on finding semantically non-equivalent queries that return the same result. A prominent type of shortcuts can be the application of further filters, if some filters already exist. Either on the same table, or, in case of a join, also on the joined table.
- Make sure the rules make sense in the context of the given SQL. If there is no context for a certain rule, do not invent it. Do not come up with more than 3 rules.

<<ADJ_CAP>>-specific, world-knowledge-driven shortcuts you should consider when applicable
<<BULLETS>>
- Think of more and other filters that can be applied, depending on the query's tables and filters.

A rule has one or multiple conditions (connected solely by AND or OR in an upper-level statement) that must be part of the SQL (so should be fulfilled in the current SQL) and implies one more more additional filters that can be applied and are likely to speed up the query. Do not invent any other fields in a rule set. Use exactly the operators that are used in the current SQL query for the conditions. The value of a condition or implied answer can only be of primitive datatypes or lists. You can not run subqueries with that. You can not put a value of another column there (join like). Be aware that your rules must contain all necessary preconditions for applying a rule. This is very important. If in doubt, better include more predicates to make sure the rule is not applied in cases where it is not valid. It must work with arbitrary SQL statements as long as the conditions are fulfilled. Stick exactly to these rules and this format. An example looks like:
{rule_example}

Output requirements
- Return in JSON format for automatic postprocessing:
    'determined_ruleset': an array of rules. If applied to the original SQL query, the result will contain more filters but still return the same result. Max 3 rules.
    'short_rationale': a bulleted list of the rules you discovered and why they are safe for <<DATA_REF>>

Input SQL:
{sql}

Process
- Briefly infer the likely intent of the input SQL.
- Identify the highest-cost parts (joins, sorts, DISTINCT/GROUP BY, wide scans).
- Apply <<ADJ>>-specific shortcuts and real-world heuristics to trim joins, narrow scans and incorporate your knowledge to filter wisely.
- Note down your shortcuts in rules of the defined format.
- Ensure the final SQL, if rules are applied, remains valid on the given schema and matches the result of the original query for <<DATA_REF>>.
"""


def _schema_label(adjective: str) -> str:
    return adjective[:1].upper() + adjective[1:]


def render_generation_prompt(short: str) -> str:
    spec = get_spec(short)
    bullets = "\n".join(f"- {b}" for b in spec["bullets"])
    text = SCAFFOLD
    for token, value in {
        "<<DISPLAY>>": spec["display"],
        "<<ADJ>>": spec["adjective"],
        "<<KNOWLEDGE>>": spec["knowledge"],
        "<<DATA_REF>>": spec["data_ref"],
        "<<SCHEMA_LABEL>>": _schema_label(spec["adjective"]),
        "<<ADJ_CAP>>": _schema_label(spec["adjective"]),
        "<<BULLETS>>": bullets,
    }.items():
        text = text.replace(token, value)
    if "<<" in text:
        raise RuntimeError(f"unfilled token remains in scaffold for {short}: {text[text.index('<<'):][:40]!r}")
    return text


def write_dataset_prompts(short: str, schema_file: Path | None = None) -> Path:
    spec = get_spec(short)
    base = PROMPTS_ROOT / short
    for sub in ("generation", "examples", "system", "refinement"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    (base / "generation" / "prompt_wk01.txt").write_text(render_generation_prompt(short), encoding="utf-8")
    (base / "examples" / "rule_example.json").write_text(
        json.dumps(spec["example_rule"], indent=4) + "\n", encoding="utf-8"
    )

    # Copy the generic system + refinement prompts verbatim (identical across datasets).
    src = PROMPTS_ROOT / GENERIC_SOURCE
    (base / "system" / "prompt01.txt").write_text(
        (src / "system" / "prompt01.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (base / "refinement" / "refine_prompt02.txt").write_text(
        (src / "refinement" / "refine_prompt02.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    if schema_file is not None:
        (base / "schema.txt").write_text(Path(schema_file).read_text(encoding="utf-8"), encoding="utf-8")

    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short", required=True, help="project-local dataset short name (e.g. financial)")
    ap.add_argument("--schema-file", default=None, help="optional schema.txt to copy into prompts/<short>/")
    args = ap.parse_args()
    base = write_dataset_prompts(args.short, Path(args.schema_file) if args.schema_file else None)
    print(f"wrote prompts -> {base}")


if __name__ == "__main__":
    main()
