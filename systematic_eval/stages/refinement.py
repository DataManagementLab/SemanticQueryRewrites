"""Stage 2: refinement loop — repair candidates that failed validation.

Feedback per failure: its class (execution/parse error, requires too broad, vacuous) plus
sampled violation rows. The model returns a sharpened rule or ``no_rule``.
Depth = ``refinement.iterations`` (looped by controller.sh), breadth =
``refinement.samples``, deduped by signature. Repairs are re-validated like any candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from llm_helpers.llms import execute
from llm_helpers.run import construct_request_dummy
from systematic_eval.config_loader import ExperimentConfig
from systematic_eval.prompt_loader import PromptLoader
from systematic_eval.sql_predicate_converter import apply_sql_rules, derive_rule_joins
from systematic_eval.parsing import parse_llm_json, cut_string, replace_backslash
from systematic_eval.rule_schema import validate_rule
from systematic_eval.stages.sampling import inject_seed, rule_signature


def _format_failed_rule(rule: Any) -> str:
    try:
        return json.dumps(rule, indent=2, ensure_ascii=False)
    except TypeError:
        return str(rule)


def _base_table_failure_reason(entry: dict) -> str:
    """Build a failure description for base_table_validation failures."""
    bt = entry.get("base_table_validation", {})
    details = bt.get("details", [])

    failed_rules: list[str] = []
    for d in details:
        status = d.get("status", "")
        rule_id = d.get("rule_id", "unknown")
        if status == "invalid":
            violations = d.get("violation_count", 0)
            sample = d.get("violations_sample", [])[:3]
            failed_rules.append(
                f"  - Rule {rule_id}: {violations} violations (rows where requires holds but implies does not). "
                f"Sample violations: {sample}"
            )
        elif status == "skipped_empty_requires":
            failed_rules.append(
                f"  - Rule {rule_id}: the rule's conditions (requires) matched 0 rows in the database, "
                f"meaning the rule could not be validated at all."
            )

    query_passed = entry.get("summary", {}).get("outputs_match") is True

    if query_passed:
        header = (
            "Problem: the rule produces correct results for the specific SQL query it was derived from, "
            "but it has FALSE POSITIVES when checked against the full database tables. "
            "This means the rule's 'requires' conditions are too broad — they match rows in the database "
            "where the 'implies' predicate does NOT hold. "
            "You need to SHARPEN the 'requires' conditions to make them more specific, "
            "so that the implication truly holds for ALL matching rows in the database, not just "
            "the subset that appears in this particular query."
        )
    else:
        header = (
            "Problem: the rule failed base-table validation against the full database. "
            "The 'implies' predicate does not hold for all rows matching the 'requires' conditions."
        )

    detail_text = "\n".join(failed_rules) if failed_rules else "  (no specific details available)"
    return f"{header}\nValidation details:\n{detail_text}"


def _failure_reason(key: str, entry: dict) -> str:
    if "error" in key:
        response = entry.get("response")
        return f"Problem: error in original output or execution. Details: {_format_failed_rule(response)}"
    summary = entry.get("summary", {})
    if summary.get("outputs_match") is False:
        diff = entry.get("results", {}).get("queries", {}).get("diff", {})
        only_orig = len(diff.get("only_in_original", []) or []) if len(diff.get("only_in_original", []) or []) < 100 else "100+"
        only_new = len(diff.get("only_in_llm_made", []) or []) if len(diff.get("only_in_llm_made", []) or []) < 100 else "100+"
        return (
            "Problem: outputs did not match after applying the rule. "
            f"Diff sizes -> only_in_original: {only_orig}, only_in_llm_made: {only_new}.\n"
            f"Examples of differences (up to 5 each) ->\n"
            f"only_in_original: {diff.get('only_in_original', [])[:5]}\n"
            f"only_in_llm_made: {diff.get('only_in_llm_made', [])[:5]}"
        )
    if _is_base_table_failed(entry):
        return _base_table_failure_reason(entry)
    return "Problem: failure detected (unknown reason)."


def _is_base_table_failed(entry: dict) -> bool:
    """Return True if base_table_validation exists and is not fully valid."""
    bt = entry.get("base_table_validation")
    if not isinstance(bt, dict):
        return False
    return not bt.get("all_valid", False)


def _is_failed_entry(key: str, entry: dict) -> bool:
    if "error" in key:
        return True
    summary = entry.get("summary")
    if summary is not None and summary.get("outputs_match") is False:
        return True
    if _is_base_table_failed(entry):
        return True
    return False


def _parse_refined_rule(output: str) -> tuple[dict | None, str]:
    if "no_rule" in output.lower():
        return None, "no_rule"

    cleaned = replace_backslash(cut_string(output))
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, "parse_failed"

    if isinstance(parsed, dict) and parsed.get("determined_ruleset"):
        rules = parsed.get("determined_ruleset")
        if isinstance(rules, list) and rules:
            return rules[0], "rule_extracted_from_ruleset"
    if isinstance(parsed, dict) and {"id", "requires", "implies"}.issubset(parsed.keys()):
        return parsed, "rule_extracted"

    return None, "parse_failed"


def _build_refine_prompt(
    generation_prompt: str,
    sql: str,
    rule_example: str,
    original_output: str,
    failed_rule: Any,
    failure_reason: str,
    refinement_prompt_template: str,
) -> str:
    original_user_prompt = generation_prompt.format(sql=sql, rule_example=rule_example)
    failed_rule_text = _format_failed_rule(failed_rule)
    refine_prompt = refinement_prompt_template.format(failed_rule=failed_rule_text)
    return (
        f"{original_user_prompt}\n\n"
        f"Original model output:\n{original_output}\n\n"
        f"{failure_reason}\n\n"
        f"{refine_prompt}"
    )


def run_refinement_iteration(
    result_path: Path,
    output_path: Path,
    config: ExperimentConfig,
    prompts: PromptLoader,
    iteration: int,
) -> dict:
    """Run one refinement iteration.

    Reads failed entries from result_path, sends refinement prompts to LLM,
    and writes the new transfer dict to output_path.
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))

    system_prompt = prompts.load_system_prompt(config.prompts.system)
    generation_prompt = prompts.load_generation_prompt(config.prompts.generation)
    rule_example = prompts.load_rule_example()
    refinement_prompt_template = prompts.load_refinement_prompt(config.refinement.prompt)

    samples = max(1, config.refinement.samples)

    # Collect all failed entries and build requests in batch. With samples > 1 we draw
    # several independent corrections per failed rule (identical prompt, distinct seed so
    # they cache separately) and dedup the overlap before re-execution.
    failed_entries: list[tuple[str, dict, str, int]] = []  # (key, entry, reason, sample)
    requests: list = []

    for key, entry in data.items():
        if not _is_failed_entry(key, entry):
            continue

        failure_reason = _failure_reason(key, entry)

        refine_message = _build_refine_prompt(
            generation_prompt=generation_prompt,
            sql=entry["sql"],
            rule_example=rule_example,
            original_output=entry.get("original_output", ""),
            failed_rule=entry.get("response", {}),
            failure_reason=failure_reason,
            refinement_prompt_template=refinement_prompt_template,
        )

        for m in range(samples):
            failed_entries.append((key, entry, failure_reason, m))
            reqs = construct_request_dummy(
                model=config.model,
                system_prompt=system_prompt,
                first_message=refine_message,
            )
            if samples > 1:
                inject_seed(reqs, m)
            requests.extend(reqs)

    # Execute all refinement requests in parallel
    responses = execute(requests, budget=config.budget, silent=False, use_cache=config.use_llm_cache) if requests else []

    # Process responses
    results: dict[str, dict] = {}
    # Exact-match dedup (only when sampling >1): collapse identical refined rules per
    # query, and keep a single representative per source key for non-rule outcomes.
    seen_signatures: dict[str, set[str]] = {}
    seen_norule_keys: set[str] = set()
    dropped_duplicates = 0

    for i, (key, entry, failure_reason, m) in enumerate(failed_entries):
        sql = entry["sql"]
        response = responses[i]
        if not isinstance(response, dict) or "choices" not in response:
            print(f"WARNING: Skipping {key} (sample {m}) — invalid API response (likely transient error)")
            continue
        llm_output = response["choices"][0]["message"]["content"]

        refined_rule, parse_status = _parse_refined_rule(llm_output)

        if refined_rule is None:
            if samples > 1:
                if key in seen_norule_keys:
                    continue
                seen_norule_keys.add(key)
            refined_sql = sql
            status = "no_rule_returned" if parse_status == "no_rule" else "refine_parse_failed"
        else:
            if samples > 1:
                sig = rule_signature(refined_rule)
                signatures = seen_signatures.setdefault(sql, set())
                if sig in signatures:
                    dropped_duplicates += 1
                    continue
                signatures.add(sig)

            valid, msg = validate_rule(refined_rule)
            if not valid:
                print(f"Invalid refined rule for {key}: {msg}")
                refined_sql = sql
                status = f"refine_invalid_rule: {msg}"
            else:
                derive_rule_joins(sql, refined_rule)
                refined_sql, has_new_predicates = apply_sql_rules(sql, [refined_rule])
                status = "new predicates applied" if has_new_predicates else "no new predicates applied"

        result_key = f"{key}_rerun" if samples == 1 else f"{key}_rerun_s{m}"
        results[result_key] = {
            "prompt_name": entry["prompt_name"],
            "sql": sql,
            "response": refined_rule if refined_rule is not None else "no_rule",
            "refined_sql": refined_sql,
            "status": status,
            "refine_output": llm_output,
            "original_failed_rule": entry.get("response", {}),
            "original_output": entry.get("original_output", ""),
            "failure_reason": failure_reason,
            "system_prompt_name": config.prompts.system,
            "refine_prompt_name": config.refinement.prompt,
            "parse_status": parse_status,
        }

    if samples > 1:
        print(f"Refinement: {samples} samples/failed-rule, dropped {dropped_duplicates} duplicate rules (exact match)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    return results
