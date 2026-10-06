"""Stage 3: aggregation — build the rule repository and transfer it across the workload.

Three steps: the validity gate (base-table-validated rules only), merging of rules with
identical ``requires``, and cross-query transfer (every rule tried against every query).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from systematic_eval.config_loader import ExperimentConfig
from systematic_eval.sql_predicate_converter import apply_sql_rules
from systematic_eval.stages.sampling import normalise_node


def load_result_files(transfer_dir: Path) -> list[dict]:
    """Load all result JSON files from the transfer directory."""
    # Match only the per-iteration input result files (result.json, result2.json,
    # …) — NOT rule_summary_result.json, which is the *final-execution output*
    # produced downstream of aggregation. Loading that here is circular, and a
    # stale/partial copy left by an interrupted run would break the load.
    result_files = sorted(
        p for p in transfer_dir.iterdir()
        if p.is_file() and re.fullmatch(r"result\d*\.json", p.name)
    )
    all_entries: list[dict] = []
    for file_path in result_files:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            all_entries.append(data)
    return all_entries


def _check_base_table_validation(entry: dict) -> tuple[bool, str | None, float | None]:
    """Check base_table_validation for validity and time improvement.

    Returns (is_valid, reason, time_saved).  When *is_valid* is True, *reason*
    is None and *time_saved* is the sum across validated rules.  When False,
    *reason* is a short human-readable explanation and *time_saved* is None.
    """
    bt = entry.get("base_table_validation")
    if not isinstance(bt, dict):
        return False, "no base_table_validation present (required by validation_mode)", None

    if not bt.get("all_valid", False):
        return False, "base_table_validation reported all_valid=False", None

    total_saved = 0.0
    for detail in bt.get("details", []):
        rule_id = detail.get("rule_id", "?")
        status = detail.get("status")
        if status == "skipped_empty_requires":
            # Vacuous case: requires condition matched 0 rows — nothing to
            # validate against.  This often indicates contradictory conditions
            # on the same base table (e.g. info_type.info = 'genres' AND
            # info_type.info = 'votes' collapsed onto a single alias).
            # Reject to avoid accepting unverified rules.
            return False, f"base-table validation skipped for {rule_id} (requires matched 0 rows — vacuous)", None
        if status == "skipped_no_joins":
            # Rule references multiple tables but lacks join information — cannot be
            # validated on base tables.  Reject rather than silently accept.
            return False, f"base-table validation skipped for {rule_id} (could not connect tables via joins)", None
        if not detail.get("outputs_match", False):
            return False, f"base-table outputs did not match for {rule_id}", None
        et = detail.get("execution_time", {})
        total_saved += et.get("time_saved", 0.0)

    return True, None, total_saved


def concatenate_rule_entries(
    all_data: list[dict],
    require_base_table: bool = False,
    time_filtering: bool = True,
) -> list[dict]:
    """The validity gate: filter to rules admissible into the repository.

    *require_base_table* (set whenever validation_mode includes base_tables) admits only
    base-table-validated rules — the precondition for cross-query transfer.
    *time_filtering* additionally demands a query-level runtime improvement (a
    performance filter, not a correctness one; oracle runs disable it).
    """
    concatenated: list[dict] = []
    for data in all_data:
        for rule_name, entry in data.items():
            response = entry.get("response")
            if not isinstance(response, dict) or not response.get("implies"):
                print(f"Skipped rule {rule_name}: no valid response/implies")
                continue
            # No-op rewrites on the source query are intentionally NOT dropped
            # here: a rule that adds nothing to its own source query may still
            # fire on another query via cross-query transfer, exactly like the
            # generation-stage no-ops (which never carried a refined_sql field).
            # Whether a rule fires anywhere is decided by apply_rules_to_all_queries.

            # --- Base-table correctness check ---
            if require_base_table:
                bt_valid, bt_reason, _ = _check_base_table_validation(entry)
                if not bt_valid:
                    print(f"Skipped rule {rule_name}: {bt_reason}")
                    continue

            # --- Query-level execution time check ---
            summary = entry.get("summary", {})
            exec_time = summary.get("execution_time", {})
            orig_time = exec_time.get("original_query")
            llm_time = exec_time.get("llm_transformed_query")
            if time_filtering and not (
                isinstance(orig_time, (int, float))
                and isinstance(llm_time, (int, float))
                and orig_time - llm_time > 0
            ):
                print(f"Skipped rule {rule_name}: no query-level time improvement")
                continue

            # --- Query-level output match (when available) ---
            if summary.get("outputs_match") is False:
                print(f"Skipped rule {rule_name}: query outputs do not match")
                continue

            concatenated.append({
                "name": rule_name,
                "rule": response,
                "sql": entry.get("sql"),
            })
            if isinstance(orig_time, (int, float)) and isinstance(llm_time, (int, float)):
                print(f"Added rule {rule_name} with query time improvement {orig_time - llm_time:.4f} seconds")
            else:
                print(f"Added rule {rule_name} (no query-level timing available)")
    return concatenated


def _canonicalize_requires(requires: dict | list | str) -> str:
    """Return a canonical JSON string for a requires condition tree.

    Normalises operator case and sorts conditions lists so that
    structurally equivalent requires trees produce the same string.
    """
    return json.dumps(normalise_node(requires), sort_keys=True)


def _merge_implies(implies_lists: list[list[dict]]) -> list[dict]:
    """Merge the implies lists of rules validated under equivalent requires.

    - Same (column, IN): intersect value lists
    - Same (column, =): keep only if the values agree
    - Different (column, op): keep all (union)

    An empty IN intersection or contradictory equalities drop the predicate.
    """
    from collections import defaultdict

    # Group by (column, op_upper)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for implies in implies_lists:
        for impl in implies:
            key = (impl["column"], impl.get("op", "").upper().strip())
            groups[key].append(impl)

    merged: list[dict] = []
    for (column, op), entries in groups.items():
        if op == "IN":
            # Intersect value lists across all rules
            value_sets = [set(map(str, e["value"])) for e in entries]
            intersection = value_sets[0]
            for vs in value_sets[1:]:
                intersection &= vs
            if intersection:
                # Preserve original value types from the first entry
                orig_values = entries[0]["value"]
                ordered = [v for v in orig_values if str(v) in intersection]
                merged.append({"column": column, "op": "IN", "value": ordered})
            else:
                print(f"  Warning: IN-list intersection is empty for {column}, dropping implies entry")
        elif op == "=":
            unique_values = {json.dumps(e.get("value"), sort_keys=True) for e in entries}
            if len(unique_values) == 1:
                merged.append(entries[0])
            else:
                print(f"  Warning: contradictory = values for {column}, dropping implies entry")
        else:
            # For other ops, deduplicate by full entry content
            seen = set()
            for e in entries:
                key_str = json.dumps(e, sort_keys=True)
                if key_str not in seen:
                    seen.add(key_str)
                    merged.append(e)
    return merged


def _merge_equivalent_rules(rule_entries: list[dict]) -> list[dict]:
    """Merge rules with structurally identical requires conditions.

    Combines their implies sections (intersect IN-lists, union others).
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in rule_entries:
        req = entry["rule"].get("requires", {})
        canon = _canonicalize_requires(req)
        groups[canon].append(entry)

    merged: list[dict] = []
    for canon_key, entries in groups.items():
        if len(entries) == 1:
            merged.append(entries[0])
            continue

        names = [e["name"] for e in entries]
        print(f"Merging {len(entries)} rules with identical requires: {names}")

        implies_lists = [e["rule"]["implies"] for e in entries]
        merged_implies = _merge_implies(implies_lists)

        if not merged_implies:
            print(f"  Skipping merge: no implies entries survived merging")
            continue

        base_rule = dict(entries[0]["rule"])
        base_rule["implies"] = merged_implies

        merged.append({
            "name": "merged: " + " + ".join(names),
            "rule": base_rule,
            "sql": entries[0].get("sql"),
        })

    print(f"Rule count after merging: {len(merged)} (was {len(rule_entries)})")
    return merged


def _load_sql_files(
    sql_filedir: Path,
    sql_filenames: list[str],
    strip_min: bool,
) -> tuple[list[str], dict[str, str]]:
    """Load SQL files into a name→string dict (same logic as generation.parse_sqls)."""
    sqls: dict[str, str] = {}
    sql_names: list[str] = []
    for sql_file in sql_filenames:
        sql_name = sql_file.split(".sql")[0]
        sql_as_string = (sql_filedir / sql_file).read_text(encoding="utf-8")
        if strip_min:
            sql_name_variant = sql_name + " NoMIN"
            sqls[sql_name_variant] = re.sub(r"MIN\((.*?)\)", r"\1", sql_as_string)
            sql_names.append(sql_name_variant)
        else:
            sqls[sql_name] = sql_as_string
            sql_names.append(sql_name)
    return sql_names, sqls


def apply_rules_to_all_queries(
    rule_entries: list[dict],
    sql_names: list[str],
    sqls: dict[str, str],
    cross_query_transfer: bool = True,
) -> dict[str, dict]:
    """Cross-query transfer: apply the repository to the whole workload.

    *cross_query_transfer* (default) tries every rule against every query; off confines
    each rule to its source query (the within-query ablation).

    Only queries where at least one rule fires are included in the output.
    """
    if not rule_entries:
        return {}

    result: dict[str, dict] = {}
    for sql_name in sql_names:
        sql = sqls[sql_name]

        # Select which rules are eligible for this query
        if cross_query_transfer:
            eligible = rule_entries
        else:
            eligible = [e for e in rule_entries if e.get("sql") == sql]
            if not eligible:
                continue

        eligible_rules = [e["rule"] for e in eligible if e.get("rule")]
        if not eligible_rules:
            continue

        # Apply eligible rules at once (fixed-point handles transitive deps)
        refined_sql, has_new = apply_sql_rules(sql, eligible_rules)
        if not has_new:
            continue

        # Determine which individual rules fired for this query
        fired_rules: list[dict] = []
        for entry in eligible:
            rule = entry.get("rule")
            if not rule:
                continue
            _, single_fired = apply_sql_rules(sql, [rule])
            if single_fired:
                fired_rules.append({
                    "name": entry.get("name"),
                    "rule": rule,
                })

        result[sql_name] = {
            "sql": sql,
            "rules": fired_rules,
            "refined_sql": refined_sql,
        }
        print(f"Rules applied to {sql_name}: {len(fired_rules)} of {len(eligible_rules)}")

    print(f"\nTotal: {len(result)} of {len(sql_names)} queries have rules applied")
    return result


def run_aggregation(config: ExperimentConfig, transfer_dir: Path) -> dict:
    """Run the aggregation stage: filter rules, apply across all queries.

    When cost_estimation is enabled, outputs an intermediate file
    (cost_aggregate_input.json) containing rules and queries for remote
    cost-filtered aggregation, instead of the final rule_summary_transfer.json.
    """
    # Load validated rules from result files
    all_data = load_result_files(transfer_dir)
    require_bt = config.execution.validation_mode in ("base_tables", "both")
    rule_entries = concatenate_rule_entries(
        all_data,
        require_base_table=require_bt,
        time_filtering=config.aggregation.time_filtering,
    )
    rule_entries = _merge_equivalent_rules(rule_entries)

    # Filter by allowed rule types (if configured)
    if config.aggregation.allowed_rule_types is not None:
        allowed = set(config.aggregation.allowed_rule_types)
        before = len(rule_entries)
        rule_entries = [
            e for e in rule_entries
            if e.get("rule", {}).get("type", "filter") in allowed
        ]
        print(f"Rule type filter {sorted(allowed)}: kept {len(rule_entries)} of {before} rules")

    # Load ALL SQL queries from the dataset
    root = Path(__file__).resolve().parents[2]
    sql_filedir = root / config.dataset.sql_dir
    sql_filenames = sorted(
        p.name
        for p in sql_filedir.glob("*.sql")
        if p.name not in set(config.dataset.excluded_files)
    )
    if config.dataset.query_limit is not None:
        sql_filenames = sql_filenames[: config.dataset.query_limit]

    sql_names, sqls = _load_sql_files(
        sql_filedir=sql_filedir,
        sql_filenames=sql_filenames,
        strip_min=config.dataset.strip_min,
    )

    if config.execution.cost_estimation:
        # Output intermediate data for remote cost-filtered aggregation.
        # Rule matching + EXPLAIN cost comparison happens on the remote server.
        intermediate = {
            "rules": rule_entries,
            "queries": {name: sqls[name] for name in sql_names},
        }
        output_path = transfer_dir / "cost_aggregate_input.json"
        output_path.write_text(
            json.dumps(intermediate, indent=4, ensure_ascii=False), encoding="utf-8",
        )
        print(f"Cost estimation enabled: wrote {len(rule_entries)} rules "
              f"and {len(sql_names)} queries to {output_path.name}")
        return intermediate

    grouped = apply_rules_to_all_queries(
        rule_entries, sql_names, sqls,
        cross_query_transfer=config.aggregation.cross_query_transfer,
    )

    output_path = transfer_dir / "rule_summary_transfer.json"
    output_path.write_text(json.dumps(grouped, indent=4, ensure_ascii=False), encoding="utf-8")

    return grouped
