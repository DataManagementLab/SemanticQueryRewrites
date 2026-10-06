"""OpenAI judge: rate a rule's world-knowledge dependence on a 1-5 Likert scale.

Reuses the project's cached OpenAI wrapper (``llm_helpers``) so verdicts are
disk-cached and cost-tracked, consistent with the generation/refinement stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import json as _json

from llm_helpers.llms import execute
from llm_helpers.run import construct_request_dummy
from systematic_eval.parsing import cut_string, parse_llm_json


def _parse_json_object(content: str) -> dict | None:
    """Parse a JSON object from LLM content.

    Tries strict ``json.loads`` on the outermost ``{...}`` first — reasoning
    models emit well-formed JSON, and the shared ``parse_llm_json`` would
    corrupt legitimately-escaped quotes (``\\"``) via its ``replace_backslash``
    step. Falls back to the lenient helper only if strict parsing fails.
    """
    try:
        obj = _json.loads(cut_string(content))
        if isinstance(obj, dict):
            return obj
    except _json.JSONDecodeError:
        pass
    parsed = parse_llm_json(content)
    return parsed if isinstance(parsed, dict) else None

_PROMPT_DIR = Path(__file__).resolve().parent
VALID_LABELS = {
    "db_derivable",
    "mostly_db_derivable",
    "mixed",
    "mostly_world_knowledge",
    "world_knowledge",
}


def load_system_prompt(schema: str, dataset: str) -> str:
    """Fill the judge prompt template with the dataset's name and schema.

    Both placeholders are substituted literally (no ``str.format``), so the
    JSON template in the prompt keeps its single braces.
    """
    template = (_PROMPT_DIR / "prompt.txt").read_text(encoding="utf-8")
    return template.replace("{dataset}", dataset).replace("{schema}", schema)


def build_user_message(record: dict) -> str:
    """Render one rule record (rule JSON + rationale) as the judge's user turn."""
    rule_json = json.dumps(record["rule"], indent=2, ensure_ascii=False)
    rationales = record.get("rationales") or []
    if rationales:
        rationale_block = "\n".join(rationales)
    else:
        rationale_block = "(no rationale recorded for this rule)"
    return (
        f"Rule id: {record.get('rule_id', '')}\n\n"
        f"Rule (requires -> implies):\n{rule_json}\n\n"
        f"Model's stated rationale:\n{rationale_block}\n"
    )


def build_requests(records: list[dict], schema: str, dataset: str, model: str) -> list[dict]:
    """One request per rule record, aligned by index with *records*."""
    system_prompt = load_system_prompt(schema, dataset)
    requests: list[dict] = []
    for rec in records:
        req = construct_request_dummy(
            model=model,
            system_prompt=system_prompt,
            first_message=build_user_message(rec),
        )
        # construct_request_dummy returns a list for reasoning models, a dict otherwise.
        requests.append(req[0] if isinstance(req, list) else req)
    return requests


def parse_verdict(response: dict) -> dict:
    """Extract a normalised verdict from one API response.

    Always returns a dict with ``score`` (int|None), ``label`` (str|None),
    ``justification`` (str), and ``parse_error`` (str|None) so a single bad
    response never aborts a batch.
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        return {"score": None, "label": None, "justification": "", "parse_error": f"no content: {exc}"}

    parsed = _parse_json_object(content)
    if parsed is None:
        return {"score": None, "label": None, "justification": content[:500], "parse_error": "not JSON"}

    score = parsed.get("score")
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None
    if score is not None and not (1 <= score <= 5):
        score = None

    label = parsed.get("label")
    if not isinstance(label, str) or label not in VALID_LABELS:
        label = label if isinstance(label, str) else None

    justification = parsed.get("justification")
    justification = justification if isinstance(justification, str) else ""

    return {
        "score": score,
        "label": label,
        "justification": justification,
        "parse_error": None if score is not None else "missing/invalid score",
    }


def judge_records(
    records: list[dict],
    schema: str,
    dataset: str,
    model: str,
    budget: float | None,
    use_cache: bool = True,
) -> list[dict]:
    """Run the judge over records; return a verdict list aligned by index."""
    if not records:
        return []
    requests = build_requests(records, schema, dataset, model)
    responses = execute(requests, budget=budget, use_cache=use_cache, silent=False)
    if isinstance(responses, dict):  # single-request path
        responses = [responses]
    return [parse_verdict(r) for r in responses]
