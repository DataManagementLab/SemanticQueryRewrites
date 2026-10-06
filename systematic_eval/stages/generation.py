"""Stage 1: LLM rule generation, multi-sampled, plus rule completion.

``generation.rounds`` independent samples per query, deduped by rule signature, then
completed via ``derive_rule_joins`` (adds ``joins``/``alias_map`` from the source query).
Schema-check failures are logged only; admission is decided by base-table validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llm_helpers.llms import execute
from llm_helpers.run import construct_request_dummy
from systematic_eval.config_loader import ExperimentConfig
from systematic_eval.prompt_loader import PromptLoader
from systematic_eval.sql_predicate_converter import apply_sql_rules, derive_rule_joins
from systematic_eval.parsing import parse_llm_json, cut_string
from systematic_eval.rule_schema import validate_rule
from systematic_eval.stages.sampling import inject_seed, rule_signature


def parse_sqls(
    sql_filedir: Path,
    sql_filenames: list[str],
    strip_min: bool,
) -> tuple[list[str], dict[str, str]]:
    """Parse SQL files into a dict of sql_name -> sql_string.

    If strip_min is True, MIN() wrappers are removed and names get ' NoMIN' suffix.
    """
    sqls: dict[str, str] = {}
    sql_names: list[str] = []
    for sql_file in sql_filenames:
        sql_name = sql_file.split(".sql")[0]
        current_path = sql_filedir / sql_file
        sql_as_string = current_path.read_text(encoding="utf-8")

        if strip_min:
            sql_name_variant = sql_name + " NoMIN"
            sqls[sql_name_variant] = re.sub(r"MIN\((.*?)\)", r"\1", sql_as_string)
            sql_names.append(sql_name_variant)
        else:
            sqls[sql_name] = sql_as_string
            sql_names.append(sql_name)

    return sql_names, sqls


def run_generation(
    config: ExperimentConfig,
    prompts: PromptLoader,
    transfer_dir: Path,
) -> dict:
    """Run LLM rule generation for all SQL files.

    Writes transfer.json and test_size.json to transfer_dir.
    Returns the transfer dict.
    """
    root = Path(__file__).resolve().parents[2]
    sql_filedir = root / config.dataset.sql_dir
    sql_filenames = sorted(
        p.name
        for p in sql_filedir.glob("*.sql")
        if p.name not in set(config.dataset.excluded_files)
    )

    if config.dataset.query_limit is not None:
        sql_filenames = sql_filenames[: config.dataset.query_limit]

    sql_names, sqls = parse_sqls(
        sql_filedir=sql_filedir,
        sql_filenames=sql_filenames,
        strip_min=config.dataset.strip_min,
    )

    system_prompt = prompts.load_system_prompt(config.prompts.system)
    prompt_template = prompts.load_generation_prompt(config.prompts.generation)
    rule_example = prompts.load_rule_example()

    rounds = max(1, config.generation.rounds)

    # Build one request per (query, sample round). Each round gets a distinct seed so
    # the otherwise-identical requests cache separately; for a single round we omit the
    # seed entirely to stay byte-identical to earlier runs (cache hits preserved).
    requests = []
    request_meta: list[tuple[str, int]] = []  # (sql_name, round) aligned with requests
    for sql_name in sql_names:
        sql_prompt = prompt_template.format(sql=sqls[sql_name], rule_example=rule_example)
        for r in range(rounds):
            reqs = construct_request_dummy(
                model=config.model,
                system_prompt=system_prompt,
                first_message=sql_prompt,
            )
            if rounds > 1:
                inject_seed(reqs, r)
            requests.extend(reqs)
            request_meta.extend((sql_name, r) for _ in reqs)

    responses = execute(requests, budget=config.budget, silent=False, use_cache=config.use_llm_cache)

    def _round_tag(r: int) -> str:
        return f"_r{r}" if rounds > 1 else ""

    result_dict: dict = {}
    # Track rule signatures already kept per query, to drop exact duplicates across
    # rounds before they reach the (expensive) execution/validation stage.
    seen_signatures: dict[str, set[str]] = {}
    dropped_duplicates = 0

    for i, (sql_name, r) in enumerate(request_meta):
        response = responses[i]
        if not isinstance(response, dict) or "choices" not in response:
            print(f"WARNING: Skipping {sql_name} (round {r}) — invalid API response (likely transient error)")
            continue
        resp = response["choices"][0]["message"]["content"]
        resp_json = parse_llm_json(resp)

        if resp_json is None:
            result_dict[sql_name + _round_tag(r) + "_error"] = {
                "prompt_name": config.prompts.generation,
                "sql": sqls[sql_name],
                "response": {
                    "status": "json formatting failed!",
                    "original_output": resp,
                    "cut output": cut_string(resp),
                },
                "refined_sql": resp,
                "original_output": resp,
            }
            continue

        if "error" in resp_json:
            result_dict[sql_name + _round_tag(r) + "_error"] = {
                "prompt_name": config.prompts.generation,
                "sql": sqls[sql_name],
                "response": resp_json,
                "refined_sql": resp_json,
                "original_output": resp,
            }
            continue

        for j, rule in enumerate(resp_json.get("determined_ruleset", [])):
            valid, msg = validate_rule(rule)
            if not valid:
                print(f"Invalid rule for {sql_name}: {msg}")

            signatures = seen_signatures.setdefault(sql_name, set())
            sig = rule_signature(rule)
            if sig in signatures:
                dropped_duplicates += 1
                continue
            signatures.add(sig)

            derive_rule_joins(sqls[sql_name], rule)
            new_sql, has_new_predicates = apply_sql_rules(sqls[sql_name], [rule])
            key_base = sql_name + _round_tag(r) + "_" + str(j) + "_" + rule["id"]
            if has_new_predicates:
                result_dict[key_base] = {
                    "prompt_name": config.prompts.generation,
                    "sql": sqls[sql_name],
                    "response": rule,
                    "refined_sql": new_sql,
                    "status": "new predicates applied",
                    "original_output": resp,
                }
            else:
                result_dict[key_base + "_no_change"] = {
                    "prompt_name": config.prompts.generation,
                    "sql": sqls[sql_name],
                    "response": rule,
                    "status": "no new predicates applied",
                    "original_output": resp,
                }

    if rounds > 1:
        print(f"Generation: {rounds} rounds/query, dropped {dropped_duplicates} duplicate rules (exact match)")

    transfer_dir.mkdir(parents=True, exist_ok=True)
    with open(transfer_dir / "transfer.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    with open(transfer_dir / "test_size.json", "w") as f:
        json.dump({"test_size": len(sql_filenames)}, f, indent=4)

    return result_dict
