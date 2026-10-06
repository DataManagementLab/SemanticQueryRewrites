"""Rule format definition and validation.

This module is the single source of truth for the rewrite rule JSON structure.
Rules have the form:
{
    "id": "IMDB_R1",
    "requires": {
        "op": "AND",
        "conditions": [
            {"column": "table.col", "op": "=", "value": "x"},
            ...
        ]
    },
    "implies": [
        {"column": "table.col", "op": "=", "value": "y"}
    ]
}

``requires`` and ``implies`` are written by the model; nothing else is.  ``joins`` and
``alias_map`` are deliberately not part of this schema — they are derived from the source
query by ``derive_rule_joins`` (``sql_predicate_converter.py``) after validation.

A structural gate only: a rule failing it is logged, not rejected.  Admission is decided
by base-table validation.

``type`` defaults to ``filter``; ``join_elimination`` and ``drops`` are not evaluated in
the thesis.
"""

from __future__ import annotations

VALID_COMPARISON_OPS = {
    "=", "!=", ">", ">=", "<", "<=",
    "IN", "NOT IN", "IS", "IS NOT",
    "LIKE", "NOT LIKE", "BETWEEN",
}

VALID_LOGICAL_OPS = {"AND", "OR"}

VALID_RULE_TYPES = {"filter", "join_elimination"}


def validate_condition(cond: dict) -> tuple[bool, str]:
    """Validate a single condition dict recursively.

    A condition is either:
    - A leaf: {"column": "t.c", "op": "=", "value": ...}
    - A logical node: {"op": "AND"|"OR", "conditions": [...]}

    Returns (valid, error_message).
    """
    if not isinstance(cond, dict):
        return False, f"Condition must be a dict, got {type(cond).__name__}"

    if "column" in cond:
        # Leaf condition
        if "op" not in cond:
            return False, f"Leaf condition missing 'op': {cond}"
        if "value" not in cond:
            return False, f"Leaf condition missing 'value': {cond}"
        if not isinstance(cond["column"], str) or "." not in cond["column"]:
            return False, f"Column must be 'table.column' format, got: {cond['column']}"
        return True, ""

    if "op" in cond and "conditions" in cond:
        # Logical node
        if cond["op"].upper() not in VALID_LOGICAL_OPS:
            return False, f"Logical op must be AND/OR, got: {cond['op']}"
        if not isinstance(cond["conditions"], list) or len(cond["conditions"]) == 0:
            return False, f"Logical node must have non-empty 'conditions' list"
        for sub in cond["conditions"]:
            ok, msg = validate_condition(sub)
            if not ok:
                return False, msg
        return True, ""

    return False, f"Invalid condition structure (need 'column' or 'op'+'conditions'): {cond}"


def validate_implies(implies: list) -> tuple[bool, str]:
    """Validate the implies field of a rule."""
    if not isinstance(implies, list) or len(implies) == 0:
        return False, "'implies' must be a non-empty list"
    for impl in implies:
        if not isinstance(impl, dict):
            return False, f"Each implies entry must be a dict, got {type(impl).__name__}"
        for field in ("column", "op", "value"):
            if field not in impl:
                return False, f"Implies entry missing '{field}': {impl}"
        if not isinstance(impl["column"], str) or "." not in impl["column"]:
            return False, f"Implies column must be 'table.column' format, got: {impl['column']}"
    return True, ""


def validate_drops(drops: list) -> tuple[bool, str]:
    """Validate the optional drops field of a rule.

    Each drop is a leaf predicate (same shape as an implies entry) — no nested AND/OR.
    """
    if not isinstance(drops, list):
        return False, f"'drops' must be a list, got {type(drops).__name__}"
    for d in drops:
        if not isinstance(d, dict):
            return False, f"Each drops entry must be a dict, got {type(d).__name__}"
        for field in ("column", "op", "value"):
            if field not in d:
                return False, f"Drops entry missing '{field}': {d}"
        if not isinstance(d["column"], str) or "." not in d["column"]:
            return False, f"Drops column must be 'table.column' format, got: {d['column']}"
    return True, ""


def validate_eliminates(eliminates: list) -> tuple[bool, str]:
    """Validate the eliminates field of a join_elimination rule."""
    if not isinstance(eliminates, list) or len(eliminates) == 0:
        return False, "'eliminates' must be a non-empty list"
    for entry in eliminates:
        if not isinstance(entry, str):
            return False, f"Each eliminates entry must be a string, got {type(entry).__name__}"
    return True, ""


def validate_rule(rule: dict) -> tuple[bool, str]:
    """Validate a complete rule dict.

    Returns (valid, error_message).
    """
    if not isinstance(rule, dict):
        return False, f"Rule must be a dict, got {type(rule).__name__}"
    if "id" not in rule:
        return False, "Rule missing 'id'"
    # Validate optional type field
    if "type" in rule:
        if rule["type"] not in VALID_RULE_TYPES:
            return False, f"Invalid rule type '{rule['type']}', must be one of {VALID_RULE_TYPES}"

    if "requires" not in rule:
        return False, "Rule missing 'requires'"
    if "implies" not in rule:
        return False, "Rule missing 'implies'"

    # Validate requires
    req = rule["requires"]
    if isinstance(req, list):
        # Legacy flat list format
        for r in req:
            ok, msg = validate_condition(r)
            if not ok:
                return False, f"In requires: {msg}"
    elif isinstance(req, dict):
        ok, msg = validate_condition(req)
        if not ok:
            return False, f"In requires: {msg}"
    else:
        return False, f"'requires' must be dict or list, got {type(req).__name__}"

    # Validate implies
    ok, msg = validate_implies(rule["implies"])
    if not ok:
        return False, msg

    # Validate drops (optional field — predicates to remove from WHERE clause)
    if "drops" in rule:
        ok, msg = validate_drops(rule["drops"])
        if not ok:
            return False, msg

    # Validate eliminates (required for join_elimination, ignored for filter)
    if rule.get("type") == "join_elimination":
        if "eliminates" not in rule:
            return False, "join_elimination rule missing 'eliminates'"
        ok, msg = validate_eliminates(rule["eliminates"])
        if not ok:
            return False, msg

    return True, ""
