"""Predicate engine: applies validated rules to a query.

parse → guaranteed context → match → inject → reduce → rewrite, with match/inject
iterated to a fixed point. Discharges (C3) and (C4).

The ``join_elimination`` / ``drops`` paths are not evaluated in the thesis.
"""

from dataclasses import dataclass
from typing import Any, Tuple

@dataclass(frozen=True)
class Predicate:
    table: str
    column: str
    op: str
    value: Any

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.table, self.column, self.op)


import sqlglot
from sqlglot import exp


OP_SYMBOLS = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "is": "IS",
    "isnot": "IS NOT",
    "like": "LIKE",
    "ilike": "NOT LIKE",
    "between": "BETWEEN",
}

OP_TO_SQLGLOT = {
    "=": "eq",
    "!=": "neq",
    "<>": "neq",
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
    "IS": "is",
    "IS NOT": "isnot",
    "LIKE": "like",
    "NOT LIKE": "ilike",
    "BETWEEN": "between",
}


def extract_logical_structure(expr, alias_map: dict[str, str]) -> dict | None:
    """
    Extract OR/AND logical structure from a sqlglot expression.
    Returns a dict matching rule format: {"op": "AND|OR", "conditions": [...]}
    or a single condition dict, or None if not a logical expression.
    
    SQLGlot uses a binary tree structure for AND/OR (this/expression), not a flat list.
    """
    # Unwrap parentheses
    if isinstance(expr, exp.Paren):
        return extract_logical_structure(expr.this, alias_map)
    
    if isinstance(expr, exp.Or):
        conditions = []
        # Process left side (this)
        left_cond = extract_logical_structure(expr.this, alias_map)
        if left_cond:
            conditions.append(left_cond)
        # Process right side (expression)
        right_cond = extract_logical_structure(expr.expression, alias_map)
        if right_cond:
            conditions.append(right_cond)
        if conditions:
            return {"op": "OR", "conditions": conditions}
    
    elif isinstance(expr, exp.And):
        conditions = []
        # Process left side (this)
        left_cond = extract_logical_structure(expr.this, alias_map)
        if left_cond:
            conditions.append(left_cond)
        # Process right side (expression)
        right_cond = extract_logical_structure(expr.expression, alias_map)
        if right_cond:
            conditions.append(right_cond)
        if conditions:
            return {"op": "AND", "conditions": conditions}
    
    # Leaf conditions
    elif isinstance(expr, (exp.Binary, exp.In, exp.Is, exp.Like, exp.Not, exp.Between)):
        # Extract as predicate and convert back to rule format
        if isinstance(expr, exp.Binary) and not isinstance(expr, exp.Like):
            if isinstance(expr.left, exp.Column) and not isinstance(expr.right, exp.Column):
                table = alias_map.get(expr.left.table, expr.left.table)
                column = expr.left.name
                op = OP_SYMBOLS.get(expr.key, expr.key)
                value = expr.right.name or expr.right.this
                return {"column": f"{table}.{column}", "op": op, "value": value}
        elif isinstance(expr, exp.In):
            col = expr.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                values = [v.this for v in expr.expressions]
                return {"column": f"{table}.{column}", "op": "IN", "value": values}
        elif isinstance(expr, exp.Is):
            col = expr.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                value = expr.expression.name if expr.expression else None
                return {"column": f"{table}.{column}", "op": "IS", "value": value}
        elif isinstance(expr, exp.Like):
            col = expr.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                value = expr.expression.this if hasattr(expr.expression, 'this') else str(expr.expression)
                # sqlglot >=30 represents "NOT LIKE" as a bare Like node with
                # negate=True (older versions wrapped it in exp.Not, handled below).
                op = "NOT LIKE" if expr.args.get("negate") else "LIKE"
                return {"column": f"{table}.{column}", "op": op, "value": value}
        elif isinstance(expr, exp.Not):
            inner = expr.this
            if isinstance(inner, exp.Like):
                col = inner.this
                if isinstance(col, exp.Column):
                    table = alias_map.get(col.table, col.table)
                    column = col.name
                    value = inner.expression.this if hasattr(inner.expression, 'this') else str(inner.expression)
                    return {"column": f"{table}.{column}", "op": "NOT LIKE", "value": value}
            elif isinstance(inner, exp.Is):
                col = inner.this
                if isinstance(col, exp.Column):
                    table = alias_map.get(col.table, col.table)
                    column = col.name
                    value = inner.expression.name if inner.expression else None
                    return {"column": f"{table}.{column}", "op": "IS NOT", "value": value}
        elif isinstance(expr, exp.Between):
            col = expr.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                low = expr.args.get("low")
                high = expr.args.get("high")
                low_val = low.this if hasattr(low, 'this') else str(low)
                high_val = high.this if hasattr(high, 'this') else str(high)
                return {"column": f"{table}.{column}", "op": "BETWEEN", "value": (low_val, high_val)}
    
    return None

def extract_predicates(sql: str) -> tuple[set[Predicate], dict[str, str], dict | None]:
    """Extract the query's guaranteed predicates (descends AND/parens, stops at OR).

    Caveat: a negated group ``NOT (a AND b)`` is descended into and its parts wrongly
    treated as guaranteed. No query in the twenty workloads contains one.

    Returns:
        (predicates, alias_map alias→base table, full AND/OR structure of the WHERE clause)
    """
    tree = sqlglot.parse_one(sql)
    alias_map = {}

    # Map aliases (e.g., "t") to their underlying tables (e.g., "title_basics").
    for tbl in tree.find_all(exp.Table):
        base_name = tbl.this.name  # actual table identifier
        alias_expr = tbl.args.get("alias")
        alias_name = alias_expr.name if alias_expr else None

        if alias_name:
            alias_map[alias_name] = base_name

        # Also allow direct lookup by base name for non-aliased columns
        alias_map.setdefault(base_name, base_name)
    
    def extract_predicates_recursive(expr):
        """Recursively extract predicates from nested AND/OR/NOT/etc structures."""
        if expr is None:
            return
            
        if isinstance(expr, exp.And):
            # Recursively extract from left and right
            extract_predicates_recursive(expr.this)
            extract_predicates_recursive(expr.expression)
        elif isinstance(expr, exp.Or):
            # Do NOT descend into OR branches for flat predicate extraction.
            # Predicates inside OR branches are not guaranteed for all rows,
            # so they must not be used to satisfy rule 'requires' conditions.
            pass
        elif isinstance(expr, exp.Paren):
            extract_predicates_recursive(expr.this)
        elif isinstance(expr, exp.Not):
            # For NOT, check what's inside and handle it
            inner = expr.this
            if isinstance(inner, (exp.And, exp.Or, exp.Paren)):
                # NOT of a complex expression, recurse into it
                extract_predicates_recursive(inner)
            else:
                # NOT of a simple predicate (LIKE, Is, etc), add it as a predicate
                _add_predicate(expr)
        elif isinstance(expr, (exp.Binary, exp.In, exp.Is, exp.Like, exp.Between)):
            _add_predicate(expr)
    
    def _add_predicate(cond):
        """Add a single predicate to the set."""
        if isinstance(cond, exp.Binary) and not isinstance(cond, exp.Like):
            if isinstance(cond.left, exp.Column) and not isinstance(cond.right, exp.Column):
                table = alias_map.get(cond.left.table, cond.left.table)
                column = cond.left.name
                op = OP_SYMBOLS.get(cond.key, cond.key)
                value = cond.right.name or cond.right.this
                predicates.add(Predicate(table, column, op, value))

        elif isinstance(cond, exp.In):
            col = cond.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                values = [v.this for v in cond.expressions]
                predicates.add(Predicate(table, column, "IN", tuple(sorted(values, key=str))))

        elif isinstance(cond, exp.Is):
            col = cond.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                # Handle IS NULL or IS NOT NULL
                value = cond.expression.name if cond.expression else None
                predicates.add(Predicate(table, column, "IS", value))

        elif isinstance(cond, exp.Like):
            col = cond.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                # Extract the LIKE pattern
                value = cond.expression.this if hasattr(cond.expression, 'this') else str(cond.expression)
                # sqlglot >=30 represents "NOT LIKE" as a bare Like node with
                # negate=True (older versions wrapped it in exp.Not, handled below).
                op = "NOT LIKE" if cond.args.get("negate") else "LIKE"
                predicates.add(Predicate(table, column, op, value))

        elif isinstance(cond, exp.Not):
            # Handle NOT expressions - check specific types first
            inner = cond.this
            if isinstance(inner, exp.Like):
                # This is NOT LIKE
                col = inner.this
                if isinstance(col, exp.Column):
                    table = alias_map.get(col.table, col.table)
                    column = col.name
                    # Extract the LIKE pattern
                    value = inner.expression.this if hasattr(inner.expression, 'this') else str(inner.expression)
                    predicates.add(Predicate(table, column, "NOT LIKE", value))
            elif isinstance(inner, exp.Is):
                # This is IS NOT
                col = inner.this
                if isinstance(col, exp.Column):
                    table = alias_map.get(col.table, col.table)
                    column = col.name
                    value = inner.expression.name if inner.expression else None
                    predicates.add(Predicate(table, column, "IS NOT", value))

        elif isinstance(cond, exp.Between):
            col = cond.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                # Extract low and high bounds
                low = cond.args.get("low")
                high = cond.args.get("high")
                low_val = low.this if hasattr(low, 'this') else str(low) # type: ignore
                high_val = high.this if hasattr(high, 'this') else str(high) # type: ignore
                predicates.add(Predicate(table, column, "BETWEEN", (low_val, high_val)))
    
    predicates = set()
    logical_structure = None

    for where in tree.find_all(exp.Where):
        # Extract logical structure (OR/AND) from the WHERE clause
        logical_structure = extract_logical_structure(where.this, alias_map)

        # Recursively extract all predicates from the WHERE clause
        extract_predicates_recursive(where.this)

    predicates = _merge_gte_lte_to_between(predicates)
    return predicates, alias_map, logical_structure


def _merge_gte_lte_to_between(predicates: set[Predicate]) -> set[Predicate]:
    """Merge ``col >= X`` AND ``col <= Y`` pairs into ``col BETWEEN (X, Y)``."""
    gte_map: dict[tuple[str, str], Predicate] = {}
    lte_map: dict[tuple[str, str], Predicate] = {}
    for p in predicates:
        key = (p.table, p.column)
        if p.op == ">=":
            gte_map[key] = p
        elif p.op == "<=":
            lte_map[key] = p

    merged = set()
    consumed = set()
    for key in gte_map.keys() & lte_map.keys():
        lo = gte_map[key]
        hi = lte_map[key]
        merged.add(Predicate(lo.table, lo.column, "BETWEEN", (lo.value, hi.value)))
        consumed.add(lo)
        consumed.add(hi)

    return (predicates - consumed) | merged


def _merge_rule_conditions_between(condition: dict) -> dict:
    """Merge ``>= X`` AND ``<= Y`` leaf conditions in a rule into BETWEEN."""
    if "column" in condition:
        return condition  # leaf, nothing to merge
    if "op" not in condition or "conditions" not in condition:
        return condition
    op = condition["op"].upper()
    children = [_merge_rule_conditions_between(c) for c in condition["conditions"]]
    if op != "AND":
        return {"op": condition["op"], "conditions": children}

    # Collect >= and <= leaves by column
    gte_map: dict[str, dict] = {}
    lte_map: dict[str, dict] = {}
    for c in children:
        if "column" in c:
            col = c["column"]
            cop = c.get("op", "").upper().strip()
            if cop == ">=":
                gte_map[col] = c
            elif cop == "<=":
                lte_map[col] = c

    consumed = set()
    new_children = []
    for col in gte_map.keys() & lte_map.keys():
        lo = gte_map[col]
        hi = lte_map[col]
        new_children.append({
            "column": col,
            "op": "BETWEEN",
            "value": [lo["value"], hi["value"]],
        })
        consumed.add(id(lo))
        consumed.add(id(hi))

    for c in children:
        if id(c) not in consumed:
            new_children.append(c)

    if len(new_children) == 1:
        return new_children[0]
    return {"op": condition["op"], "conditions": new_children}


def entails(query_pred: Predicate, rule_pred: dict, alias_map: dict[str, str] = None) -> bool:
    """Operator-aware entailment: does the query predicate imply the rule condition?

    Covers numeric subsumption, BETWEEN vs paired >=/<=, IN-list subset, =, LIKE, IS,
    IS NOT. Deliberately incomplete.
    """
    
    # Normalize column comparison: handle both alias.column and full_table.column
    query_col = f"{query_pred.table}.{query_pred.column}"
    rule_col = rule_pred["column"]
    
    # Check if columns match (exact or by expanding aliases)
    cols_match = query_col == rule_col
    if not cols_match and alias_map:
        # Try expanding rule column if it uses an alias
        rule_parts = rule_col.split(".")
        if len(rule_parts) == 2:
            rule_alias, rule_col_name = rule_parts
            rule_expanded = f"{alias_map.get(rule_alias, rule_alias)}.{rule_col_name}"
            cols_match = query_col == rule_expanded
    
    if not cols_match:
        return False

    # Normalize operators (case-insensitive, handle spaces)
    query_op = query_pred.op.upper()
    rule_op = rule_pred["op"].upper().replace(" ", " ")

    # Cross-match: query BETWEEN vs rule >= or <=
    if query_op == "BETWEEN" and rule_op in (">=", "<="):
        try:
            q_lo = float(query_pred.value[0])
            q_hi = float(query_pred.value[1])
            rv = float(rule_pred["value"])
        except (ValueError, TypeError):
            return False
        if rule_op == ">=":
            return q_lo >= rv  # BETWEEN(5,10) satisfies >= 3
        else:  # <=
            return q_hi <= rv  # BETWEEN(5,10) satisfies <= 12

    # Cross-match: query >= or <= vs rule BETWEEN
    if query_op in (">=", "<=") and rule_op == "BETWEEN":
        try:
            qv = float(query_pred.value)
            r_lo = float(rule_pred["value"][0])
            r_hi = float(rule_pred["value"][1])
        except (ValueError, TypeError):
            return False
        if query_op == ">=":
            return qv >= r_lo  # col >= 5 satisfies BETWEEN(3,10) lower bound only
        else:  # <=
            return qv <= r_hi
        # Note: a single >= or <= can only satisfy one bound of a BETWEEN.
        # Both bounds together are needed to fully satisfy a rule BETWEEN.
        # This is handled by the AND merging above; this branch covers
        # residual cases where only one side appears.

    if query_op != rule_op:
        return False

    # Numeric comparisons (operator-aware entailment)
    if isinstance(rule_pred["value"], (int, float)):
        try:
            qv = float(query_pred.value)
            rv = float(rule_pred["value"])
        except (ValueError, TypeError):
            return False
        if query_op in (">=", ">"):
            # query: col >= 8.5, rule: col >= 8.0 → 8.5 >= 8.0 → satisfied
            res = qv >= rv
        elif query_op in ("<=", "<"):
            # query: col < 100, rule: col < 200 → 100 <= 200 → satisfied
            res = qv <= rv
        elif query_op == "=":
            res = qv == rv
        else:
            res = qv == rv
        return res

    # Equality
    if isinstance(rule_pred["value"], str):
        res = query_pred.value == rule_pred["value"]
        return res

    # IN: query's values must be a subset of rule's values
    # (rule validated for its IN set; query must stay within that set)
    if query_op == "IN" and rule_op == "IN":
        res = set(query_pred.value).issubset(set(rule_pred["value"]))
        return res

    # IS (e.g., IS NULL)
    if query_op == "IS" and rule_op == "IS":
        res = str(query_pred.value).upper() == str(rule_pred["value"]).upper()
        return res

    # IS NOT (e.g., IS NOT NULL)
    if query_op == "IS NOT" and rule_op == "IS NOT":
        res = str(query_pred.value).upper() == str(rule_pred["value"]).upper()
        return res

    # LIKE
    if query_op == "LIKE" and rule_op == "LIKE":
        res = query_pred.value == rule_pred["value"]
        return res

    # NOT LIKE
    if query_pred.op == "NOT LIKE" and rule_pred["op"] == "NOT LIKE":
        res = query_pred.value == rule_pred["value"]
        return res

    # BETWEEN - query entails rule if query's range is within or equal to rule's range
    if query_pred.op == "BETWEEN" and rule_pred["op"] == "BETWEEN":
        query_low, query_high = query_pred.value
        rule_low, rule_high = rule_pred["value"]
        try:
            ql = float(query_low)
            qh = float(query_high)
            rl = float(rule_low)
            rh = float(rule_high)
            res = ql >= rl and qh <= rh
            return res
        except (ValueError, TypeError):
            res = query_low == rule_low and query_high == rule_high
            return res

    print("[entails] no matching handler; returning False")
    return False

def evaluate_condition(condition, predicates: set[Predicate], query_structure: dict | None = None, alias_map: dict[str, str] = None) -> bool:
    """
    Recursively evaluate a condition structure against predicates.
    Also checks against the query's logical structure if available.
    
    Args:
        condition: Can be a dict with "column", "op", "value" (leaf condition)
                   or a dict with "op" (AND/OR) and "conditions" (nested structure)
        predicates: Set of predicates to check against
        query_structure: The logical structure extracted from the query (for matching ORs/ANDs)
        alias_map: Mapping from alias to table name for column resolution
    
    Returns:
        True if the condition is satisfied, False otherwise
    """
    # First, check if condition matches the query's structure exactly
    if query_structure and _conditions_match(condition, query_structure, alias_map):
        return True
    
    # Leaf condition: {"column": "...", "op": "...", "value": ...}
    if "column" in condition:
        return any(entails(p, condition, alias_map) for p in predicates)
    
    # Logical operator: {"op": "AND"|"OR", "conditions": [...]}
    if "op" in condition and "conditions" in condition:
        op = condition["op"].upper()
        conditions = condition["conditions"]
        
        if op == "AND":
            return all(evaluate_condition(cond, predicates, query_structure, alias_map) for cond in conditions)
        elif op == "OR":
            return any(evaluate_condition(cond, predicates, query_structure, alias_map) for cond in conditions)
        else:
            raise ValueError(f"Unknown logical operator: {op}. Expected 'AND' or 'OR'. Condition: {condition}")
    
    raise ValueError(f"Invalid condition structure: {condition}")

def _conditions_match(cond1: dict, cond2: dict, alias_map: dict[str, str] = None) -> bool:
    """Recursively check if two condition structures match."""
    # Both are leaf conditions
    if "column" in cond1 and "column" in cond2:
        col1 = cond1.get("column")
        col2 = cond2.get("column")
        
        # Normalize columns: expand aliases in col1 to match col2's format
        cols_match = col1 == col2
        if not cols_match and alias_map:
            # Try expanding col1 if it uses an alias
            parts1 = col1.split(".")
            if len(parts1) == 2:
                alias, col_name = parts1
                col1_expanded = f"{alias_map.get(alias, alias)}.{col_name}"
                cols_match = col1_expanded == col2
        
        # Normalize operators (case-insensitive)
        op1 = cond1.get("op", "").upper()
        op2 = cond2.get("op", "").upper()
        
        return cols_match and op1 == op2 and cond1.get("value") == cond2.get("value")
    
    # Both are logical operators
    if cond1.get("op", "").upper() in ("AND", "OR") and cond2.get("op", "").upper() == cond1.get("op", "").upper():
        cond1_list = cond1.get("conditions", [])
        cond2_list = cond2.get("conditions", [])
        if len(cond1_list) != len(cond2_list):
            return False
        return all(_conditions_match(c1, c2, alias_map) for c1, c2 in zip(cond1_list, cond2_list))
    
    return False


# ---------------------------------------------------------------------------
# Predicate reduction: remove redundant predicates after rule application
# ---------------------------------------------------------------------------

def reduce_predicates(predicates: set[Predicate]) -> set[Predicate]:
    """Simplify a predicate set by removing logically subsumed predicates.

    Runs once after the fixed point; the only step that removes a predicate.

    Groups predicates by (table, column) and within each group:
    - IN ∩ IN → intersection of value sets
    - = + IN → keep only = if value is in the IN set
    - Multiple upper/lower bounds → keep only the tightest
    - = + comparison → drop comparison if = satisfies it
    - BETWEEN decomposed into >=/<= for unified comparison handling, re-merged after
    """
    from collections import defaultdict
    groups: dict[tuple[str, str], list[Predicate]] = defaultdict(list)
    for p in predicates:
        groups[(p.table, p.column)].append(p)

    result: set[Predicate] = set()
    for key, group in groups.items():
        if len(group) == 1:
            result.add(group[0])
        else:
            result.update(_reduce_column_group(group))
    return result


def _try_float(value) -> float | None:
    """Try to convert a value to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _reduce_column_group(preds: list[Predicate]) -> set[Predicate]:
    """Reduce a group of predicates that share the same (table, column)."""
    table = preds[0].table
    column = preds[0].column

    in_preds: list[Predicate] = []
    eq_preds: list[Predicate] = []
    # Bounds store (float_value, is_strict, original_string_value)
    upper_bounds: list[tuple[float, bool, Any]] = []
    lower_bounds: list[tuple[float, bool, Any]] = []
    between_preds: list[Predicate] = []
    others: list[Predicate] = []

    for p in preds:
        op = p.op.upper().strip()
        if op == "IN":
            in_preds.append(p)
        elif op == "=":
            eq_preds.append(p)
        elif op == "BETWEEN":
            between_preds.append(p)
        elif op in ("<", "<=", ">", ">="):
            v = _try_float(p.value)
            if v is not None:
                if op == "<":
                    upper_bounds.append((v, True, p.value))
                elif op == "<=":
                    upper_bounds.append((v, False, p.value))
                elif op == ">":
                    lower_bounds.append((v, True, p.value))
                elif op == ">=":
                    lower_bounds.append((v, False, p.value))
            else:
                others.append(p)
        else:
            others.append(p)

    # Expand BETWEEN into >= and <= for unified handling
    for p in between_preds:
        lo_val, hi_val = p.value
        lo_f = _try_float(lo_val)
        hi_f = _try_float(hi_val)
        if lo_f is not None and hi_f is not None:
            lower_bounds.append((lo_f, False, lo_val))   # >= lo
            upper_bounds.append((hi_f, False, hi_val))   # <= hi
        else:
            others.append(p)  # non-numeric BETWEEN, can't reduce

    result: set[Predicate] = set(others)

    # --- IN reduction ---
    reduced_in_value: tuple | None = None
    if in_preds:
        intersection = set(in_preds[0].value)
        for ip in in_preds[1:]:
            intersection &= set(ip.value)

        if not intersection:
            print(f"WARNING: Empty IN intersection for {table}.{column}, keeping original predicates")
            result.update(in_preds)
        else:
            reduced_in_value = tuple(sorted(intersection, key=str))

    # --- Equality + IN interaction ---
    if eq_preds and reduced_in_value is not None:
        eq_val = eq_preds[0].value
        if eq_val in set(reduced_in_value) or str(eq_val) in {str(v) for v in reduced_in_value}:
            # = is in the IN set, keep only =
            result.add(eq_preds[0])
            reduced_in_value = None  # consumed
        else:
            # Contradiction: = value not in IN intersection
            print(f"WARNING: Equality value '{eq_val}' not in IN set for {table}.{column}, keeping both")
            result.update(in_preds)
            result.add(eq_preds[0])
            reduced_in_value = None
    elif eq_preds:
        result.add(eq_preds[0])
        # Keep additional = with different values (contradiction, let DB handle)
        for ep in eq_preds[1:]:
            if ep.value != eq_preds[0].value:
                result.add(ep)

    # Emit reduced IN if not consumed by equality
    if reduced_in_value is not None:
        if len(reduced_in_value) == 1:
            result.add(Predicate(table, column, "=", reduced_in_value[0]))
        else:
            result.add(Predicate(table, column, "IN", reduced_in_value))

    # --- Comparison reduction ---
    def _find_tightest_bound(bounds, upper=True):
        """Find the tightest bound from a list of (float_val, is_strict, str_val).
        For upper bounds: smallest value is tightest.
        For lower bounds: largest value is tightest.
        At equal values, strict (< or >) beats non-strict (<= or >=).
        """
        if not bounds:
            return None, None, None
        if upper:
            best_val = min(b[0] for b in bounds)
        else:
            best_val = max(b[0] for b in bounds)
        # Check if any bound at best_val is strict
        best_strict = any(b[1] for b in bounds if b[0] == best_val)
        # Get original string value from any bound at best_val
        best_str = next(b[2] for b in bounds if b[0] == best_val)
        return best_val, best_strict, best_str

    best_upper: Predicate | None = None
    if upper_bounds:
        _, ub_strict, ub_str = _find_tightest_bound(upper_bounds, upper=True)
        op = "<" if ub_strict else "<="
        best_upper = Predicate(table, column, op, ub_str)

    best_lower: Predicate | None = None
    if lower_bounds:
        _, lb_strict, lb_str = _find_tightest_bound(lower_bounds, upper=False)
        op = ">" if lb_strict else ">="
        best_lower = Predicate(table, column, op, lb_str)

    # --- Equality absorbs comparisons ---
    if eq_preds:
        eq_f = _try_float(eq_preds[0].value)
        if eq_f is not None:
            if best_upper is not None:
                ub_f = _try_float(best_upper.value)
                if ub_f is not None:
                    satisfies = (eq_f < ub_f) if best_upper.op == "<" else (eq_f <= ub_f)
                    if satisfies:
                        best_upper = None  # = absorbs upper bound
            if best_lower is not None:
                lb_f = _try_float(best_lower.value)
                if lb_f is not None:
                    satisfies = (eq_f > lb_f) if best_lower.op == ">" else (eq_f >= lb_f)
                    if satisfies:
                        best_lower = None  # = absorbs lower bound

    # Try to re-merge >= and <= back into BETWEEN
    if (best_upper is not None and best_lower is not None
            and best_upper.op == "<=" and best_lower.op == ">="):
        result.add(Predicate(table, column, "BETWEEN", (best_lower.value, best_upper.value)))
    else:
        if best_upper is not None:
            result.add(best_upper)
        if best_lower is not None:
            result.add(best_lower)

    return result


def _base_join_edge(left: str, right: str, alias_to_base: dict[str, str]) -> frozenset:
    """Unordered, base-table-level representation of an equi-join edge.

    ``left``/``right`` are ``"table_or_alias.column"`` strings; *alias_to_base*
    resolves alias names to base table names (identity for names already in base
    form).  Returning a frozenset makes the edge direction-independent, so
    ``t.id = mc.movie_id`` and ``mc.movie_id = t.id`` compare equal.
    """
    lt, lc = left.split(".", 1)
    rt, rc = right.split(".", 1)
    return frozenset({
        f"{alias_to_base.get(lt, lt)}.{lc}",
        f"{alias_to_base.get(rt, rt)}.{rc}",
    })


def apply_rules(predicates: set[Predicate], rules: list[dict], query_structure: dict | None = None, alias_map: dict[str, str] = None, collect_drops: bool = False, query_joins: list[dict] | None = None):
    """Inject the implies of matching rules to a fixed point.

    Checks entailment (C3) and, for multi-table rules, join compatibility (C4).
    Reduction runs once afterwards, outside the loop.
    """
    changed = True
    predicates = set(predicates)
    dropped: set[Predicate] = set()

    # Base-table-level equi-join edges of the target query, for the (C4) check below.
    # None => check disabled.
    query_edge_set = None
    if query_joins is not None:
        query_edge_set = {
            _base_join_edge(j["left"], j["right"], {}) for j in query_joins
        }

    while changed:
        changed = False

        for rule in rules:
            if "requires" not in rule or "implies" not in rule:
                print(f"WARNING: Skipping invalid rule (missing 'requires' or 'implies'): {rule}")
                continue

            # 'implies' must be a flat list of leaf predicates (each with a 'column').
            # Nested AND/OR trees are only legal in 'requires'; guard against the LLM
            # putting one in 'implies', which would otherwise KeyError below.
            if not isinstance(rule["implies"], list) or any(
                not isinstance(impl, dict) or "column" not in impl for impl in rule["implies"]
            ):
                print(f"WARNING: Skipping rule with malformed 'implies' (expected list of leaf predicates with 'column'): {rule['implies']}")
                continue

            # Check that all tables in the rule's joins and implies exist in the query
            if alias_map:
                query_tables = set(alias_map.values())
                rule_alias_map = rule.get("alias_map", {})
                rule_tables = set()
                for j in rule.get("joins", []):
                    lt = j["left"].split(".")[0]
                    rt = j["right"].split(".")[0]
                    # Resolve alias→base via rule's own alias_map
                    rule_tables.add(rule_alias_map.get(lt, lt))
                    rule_tables.add(rule_alias_map.get(rt, rt))
                for impl in rule["implies"]:
                    t = impl["column"].split(".")[0]
                    rule_tables.add(rule_alias_map.get(t, t))
                missing = rule_tables - query_tables
                if missing:
                    continue

                # (C4) Join compatibility: fire only if the query connects the rule's
                # tables via exactly the validated equi-join keys.
                if query_edge_set is not None and rule.get("joins"):
                    required_edges = {
                        _base_join_edge(j["left"], j["right"], rule_alias_map)
                        for j in rule["joins"]
                    }
                    if not required_edges <= query_edge_set:
                        continue

            requires = rule["requires"]

            # Merge >= / <= pairs into BETWEEN in rule conditions
            if isinstance(requires, dict):
                requires = _merge_rule_conditions_between(requires)

            # Backward compatibility: if requires is a list, treat as AND
            if isinstance(requires, list):
                satisfied = all(
                    any(entails(p, req, alias_map) for p in predicates)
                    for req in requires
                )
            # New structure: nested AND/OR
            elif isinstance(requires, dict):
                satisfied = evaluate_condition(requires, predicates, query_structure, alias_map)
            else:
                raise ValueError(f"Invalid requires structure: {requires}")

            if satisfied:
                for impl in rule["implies"]:
                    table, column = impl["column"].split(".")
                    # Expand alias to actual table name if it exists
                    expanded_table = alias_map.get(table, table) if alias_map else table

                    # Normalize operator variants the LLM may produce
                    op = impl["op"]
                    value = impl["value"]
                    op_upper = op.strip().upper()
                    if op_upper == "IS NOT NULL":
                        op = "IS NOT"
                        value = None
                    elif op_upper == "IS NULL":
                        op = "IS"
                        value = None

                    new_pred = Predicate(
                        table=expanded_table,
                        column=column,
                        op=op,
                        value=tuple(sorted(value, key=str)) if isinstance(value, list) else value
                    )

                    if new_pred not in predicates:
                        predicates.add(new_pred)
                        changed = True

                for drop in rule.get("drops", []):
                    table, column = drop["column"].split(".")
                    expanded_table = alias_map.get(table, table) if alias_map else table

                    op = drop["op"]
                    value = drop["value"]
                    op_upper = op.strip().upper()
                    if op_upper == "IS NOT NULL":
                        op = "IS NOT"
                        value = None
                    elif op_upper == "IS NULL":
                        op = "IS"
                        value = None

                    # extract_predicates stores scalar literals as strings; coerce drop
                    # values the same way so set-membership matches correctly.
                    if isinstance(value, list):
                        norm_value = tuple(sorted((str(v) for v in value)))
                    elif value is None:
                        norm_value = None
                    else:
                        norm_value = str(value)

                    dropped.add(Predicate(
                        table=expanded_table,
                        column=column,
                        op=op,
                        value=norm_value
                    ))

    # Reduce once after fixed-point completes to remove redundant predicates.
    # Reduction only makes predicates more restrictive (can't enable new rules),
    # so doing it after the loop is equivalent to doing it inside.
    reduced = reduce_predicates(predicates)
    if collect_drops:
        return reduced, dropped
    return reduced


def _ast_node_to_predicate(node, alias: str, base_table: str) -> Predicate | None:
    """Convert a sqlglot AST condition node to a Predicate for a specific alias.

    Returns None if the node isn't a simple filter predicate on the given alias.
    """
    if isinstance(node, exp.Binary) and not isinstance(node, exp.Like):
        if isinstance(node.left, exp.Column) and not isinstance(node.right, exp.Column):
            if node.left.table == alias:
                op = OP_SYMBOLS.get(node.key, node.key)
                value = node.right.name or node.right.this
                return Predicate(base_table, node.left.name, op, value)
    elif isinstance(node, exp.In):
        col = node.this
        if isinstance(col, exp.Column) and col.table == alias:
            values = tuple(sorted((v.this for v in node.expressions), key=str))
            return Predicate(base_table, col.name, "IN", values)
    elif isinstance(node, exp.Is):
        col = node.this
        if isinstance(col, exp.Column) and col.table == alias:
            value = node.expression.name if node.expression else None
            return Predicate(base_table, col.name, "IS", value)
    elif isinstance(node, exp.Like):
        col = node.this
        if isinstance(col, exp.Column) and col.table == alias:
            value = node.expression.this if hasattr(node.expression, 'this') else str(node.expression)
            # sqlglot >=30 represents "NOT LIKE" as a bare Like node with
            # negate=True (older versions wrapped it in exp.Not, handled below).
            op = "NOT LIKE" if node.args.get("negate") else "LIKE"
            return Predicate(base_table, col.name, op, value)
    elif isinstance(node, exp.Not):
        inner = node.this
        if isinstance(inner, exp.Like):
            col = inner.this
            if isinstance(col, exp.Column) and col.table == alias:
                value = inner.expression.this if hasattr(inner.expression, 'this') else str(inner.expression)
                return Predicate(base_table, col.name, "NOT LIKE", value)
        elif isinstance(inner, exp.Is):
            col = inner.this
            if isinstance(col, exp.Column) and col.table == alias:
                value = inner.expression.name if inner.expression else None
                return Predicate(base_table, col.name, "IS NOT", value)
    return None


def _collect_alias_specific_predicates(
    tree,
    target_aliases: list[str],
    alias_map: dict[str, str],
) -> dict[str, set[Predicate]]:
    """Walk WHERE AST and collect predicates keyed by specific alias name.

    For each leaf condition in the WHERE clause, checks if it references one of
    the *target_aliases*.  If so, converts it to a Predicate (with base-table
    name from *alias_map*) using _ast_node_to_predicate.

    Args:
        tree: A sqlglot AST (already parsed via sqlglot.parse_one).
        target_aliases: Alias names to collect predicates for.
        alias_map: Mapping alias → base table name.

    Returns:
        Dict mapping each target alias to its set of Predicates.
    """
    result: dict[str, set[Predicate]] = {a: set() for a in target_aliases}
    where_clause = tree.args.get("where")
    if not where_clause:
        return result

    alias_set = set(target_aliases)

    def _walk(node):
        if isinstance(node, (exp.And, exp.Or)):
            _walk(node.this)
            _walk(node.expression)
            return
        if isinstance(node, exp.Paren):
            _walk(node.this)
            return
        # Leaf condition — check if it references one of our aliases
        for col in node.find_all(exp.Column):
            if col.table in alias_set:
                base_table = alias_map.get(col.table, col.table)
                p = _ast_node_to_predicate(node, col.table, base_table)
                if p:
                    result[col.table].add(p)

    _walk(where_clause.this)
    return result


def _predicate_from_where_node(node, alias_map: dict[str, str]) -> Predicate | None:
    """Convert a WHERE clause AST leaf node to a Predicate for matching.

    Uses alias_map to expand alias→base table, mirroring how extract_predicates
    stores predicates.  Returns None if the node isn't a recognizable filter.
    """
    if isinstance(node, exp.Paren):
        return _predicate_from_where_node(node.this, alias_map)

    if isinstance(node, exp.Binary) and not isinstance(node, (exp.And, exp.Or, exp.Like)):
        if isinstance(node.left, exp.Column) and not isinstance(node.right, exp.Column):
            table = alias_map.get(node.left.table, node.left.table)
            column = node.left.name
            op = OP_SYMBOLS.get(node.key, node.key)
            value = node.right.name or node.right.this
            return Predicate(table, column, op, value)

    if isinstance(node, exp.In):
        col = node.this
        if isinstance(col, exp.Column):
            table = alias_map.get(col.table, col.table)
            column = col.name
            values = tuple(sorted((v.this for v in node.expressions), key=str))
            return Predicate(table, column, "IN", values)

    if isinstance(node, exp.Between):
        col = node.this
        if isinstance(col, exp.Column):
            table = alias_map.get(col.table, col.table)
            column = col.name
            low = node.args.get("low")
            high = node.args.get("high")
            low_val = low.this if hasattr(low, 'this') else str(low)
            high_val = high.this if hasattr(high, 'this') else str(high)
            return Predicate(table, column, "BETWEEN", (low_val, high_val))

    if isinstance(node, exp.Is):
        col = node.this
        if isinstance(col, exp.Column):
            table = alias_map.get(col.table, col.table)
            column = col.name
            value = node.expression.name if node.expression else None
            return Predicate(table, column, "IS", value)

    if isinstance(node, exp.Like):
        col = node.this
        if isinstance(col, exp.Column):
            table = alias_map.get(col.table, col.table)
            column = col.name
            value = node.expression.this if hasattr(node.expression, 'this') else str(node.expression)
            # sqlglot >=30 represents "NOT LIKE" as a bare Like node with
            # negate=True (older versions wrapped it in exp.Not, handled below).
            op = "NOT LIKE" if node.args.get("negate") else "LIKE"
            return Predicate(table, column, op, value)

    if isinstance(node, exp.Not):
        inner = node.this
        if isinstance(inner, exp.Like):
            col = inner.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                value = inner.expression.this if hasattr(inner.expression, 'this') else str(inner.expression)
                return Predicate(table, column, "NOT LIKE", value)
        if isinstance(inner, exp.Is):
            col = inner.this
            if isinstance(col, exp.Column):
                table = alias_map.get(col.table, col.table)
                column = col.name
                value = inner.expression.name if inner.expression else None
                return Predicate(table, column, "IS NOT", value)

    return None


def _filter_where_by_predicates(where_expr, remove_preds: set[Predicate], alias_map: dict[str, str]):
    """Remove WHERE conditions whose Predicate is in remove_preds.

    Walks the AND/OR tree analogously to _filter_where_conditions.
    """
    if where_expr is None:
        return None

    if isinstance(where_expr, exp.Paren):
        inner = _filter_where_by_predicates(where_expr.this, remove_preds, alias_map)
        return exp.Paren(this=inner) if inner else None

    if isinstance(where_expr, exp.And):
        left = _filter_where_by_predicates(where_expr.this, remove_preds, alias_map)
        right = _filter_where_by_predicates(where_expr.expression, remove_preds, alias_map)
        if left and right:
            return exp.And(this=left, expression=right)
        return left or right

    if isinstance(where_expr, exp.Or):
        left = _filter_where_by_predicates(where_expr.this, remove_preds, alias_map)
        right = _filter_where_by_predicates(where_expr.expression, remove_preds, alias_map)
        if left and right:
            return exp.Or(this=left, expression=right)
        return None  # drop entire OR if either branch removed

    # Leaf condition — check if it matches a predicate to remove
    pred = _predicate_from_where_node(where_expr, alias_map)
    if pred is not None and pred in remove_preds:
        return None
    return where_expr


def _references_alias(node, aliases: set[str]) -> bool:
    """Check if an expression AST node references any of the given aliases."""
    for col in node.find_all(exp.Column):
        if col.table in aliases:
            return True
    return False


def _filter_where_conditions(where_expr, eliminate_aliases: set[str]):
    """Remove WHERE conditions that reference eliminated aliases.

    Walks the AND/OR tree and drops any leaf condition that references
    an eliminated alias.  Returns the cleaned expression or None if
    everything was removed.
    """
    if where_expr is None:
        return None

    if isinstance(where_expr, exp.Paren):
        inner = _filter_where_conditions(where_expr.this, eliminate_aliases)
        return exp.Paren(this=inner) if inner else None

    if isinstance(where_expr, exp.And):
        left = _filter_where_conditions(where_expr.this, eliminate_aliases)
        right = _filter_where_conditions(where_expr.expression, eliminate_aliases)
        if left and right:
            return exp.And(this=left, expression=right)
        return left or right

    if isinstance(where_expr, exp.Or):
        left = _filter_where_conditions(where_expr.this, eliminate_aliases)
        right = _filter_where_conditions(where_expr.expression, eliminate_aliases)
        if left and right:
            return exp.Or(this=left, expression=right)
        # If either side of OR is removed, the whole OR is unsafe to keep
        # (removing one branch changes semantics).  Keep the full OR only
        # if both sides survive.
        return None

    # Leaf condition — check if it references an eliminated alias
    if _references_alias(where_expr, eliminate_aliases):
        return None
    return where_expr


def _eliminate_tables_from_ast(tree, eliminate_aliases: set[str]):
    """Remove eliminated table aliases from FROM/JOIN and clean WHERE conditions.

    Handles both comma-separated FROM (implicit joins) and explicit JOIN syntax.
    """
    # --- Remove tables from FROM clause (comma-separated / implicit joins) ---
    from_clause = tree.find(exp.From)
    if from_clause:
        # In sqlglot, comma-separated tables in FROM are represented as
        # the first table in From.this and additional tables as Join nodes
        # with join_type="" (implicit cross joins).

        # Check if the main FROM table should be eliminated
        main_table = from_clause.this
        if isinstance(main_table, exp.Table):
            alias_expr = main_table.args.get("alias")
            main_alias = alias_expr.name if alias_expr else main_table.this.name
            if main_alias in eliminate_aliases:
                # Need to promote the first JOIN to be the FROM table
                joins = list(tree.find_all(exp.Join))
                if joins:
                    first_join = joins[0]
                    from_clause.set("this", first_join.this)
                    first_join.pop()

    # Remove JOIN nodes for eliminated aliases
    for join_node in list(tree.find_all(exp.Join)):
        join_table = join_node.this
        if isinstance(join_table, exp.Table):
            alias_expr = join_table.args.get("alias")
            join_alias = alias_expr.name if alias_expr else join_table.this.name
            if join_alias in eliminate_aliases:
                join_node.pop()

    # --- Clean WHERE conditions referencing eliminated aliases ---
    where = tree.args.get("where")
    if where:
        cleaned = _filter_where_conditions(where.this, eliminate_aliases)
        if cleaned:
            tree.set("where", exp.Where(this=cleaned))
        else:
            tree.set("where", None)

    return tree


def _quote_ident(name: str) -> str:
    """Double-quote a single SQL identifier (doubling any embedded quote).

    Injected predicates must be quoted for the same reason the base-table
    validation path quotes (see ``execution._quote_ident``): DuckDB folds
    unquoted identifiers case-insensitively, but Umbra/Postgres fold them to
    lowercase, so an unquoted mixed-case name like
    ``On_Time_On_Time_Performance_2016_1`` silently fails to resolve on Umbra
    and the whole rewritten query errors out (rule wrongly declined). Quoting
    is a no-op for already-lowercase schemas (e.g. JOB) and case-insensitive on
    DuckDB, so it is safe across all engines.
    """
    return '"' + str(name).replace('"', '""') + '"'


def _predicate_to_sql(pred: Predicate, reverse_alias_map: dict[str, str]) -> str:
    """Convert a Predicate to SQL text without going through sqlglot AST serialization.

    This avoids normalizations like IS NOT NULL → NOT ... IS NULL.
    """
    table_ref = reverse_alias_map.get(pred.table, pred.table)
    prefix = f"{_quote_ident(table_ref)}.{_quote_ident(pred.column)}"
    op_upper = pred.op.upper().strip()

    if op_upper == "IN":
        vals = ", ".join(
            str(v) if isinstance(v, (int, float)) else f"'{v}'" for v in pred.value
        )
        return f"{prefix} IN ({vals})"
    elif op_upper == "NOT IN":
        vals = ", ".join(
            str(v) if isinstance(v, (int, float)) else f"'{v}'" for v in pred.value
        )
        return f"{prefix} NOT IN ({vals})"
    elif op_upper == "IS":
        if pred.value is None or str(pred.value).upper() == "NULL":
            return f"{prefix} IS NULL"
        return f"{prefix} IS {pred.value}"
    elif op_upper == "IS NOT":
        if pred.value is None or str(pred.value).upper() == "NULL":
            return f"{prefix} IS NOT NULL"
        return f"{prefix} IS NOT {pred.value}"
    elif op_upper == "LIKE":
        return f"{prefix} LIKE '{pred.value}'"
    elif op_upper == "NOT LIKE":
        return f"{prefix} NOT LIKE '{pred.value}'"
    elif op_upper == "BETWEEN":
        lo, hi = pred.value
        lo_s = str(lo) if isinstance(lo, (int, float)) else f"'{lo}'"
        hi_s = str(hi) if isinstance(hi, (int, float)) else f"'{hi}'"
        return f"{prefix} BETWEEN {lo_s} AND {hi_s}"
    else:
        if pred.value is None:
            return f"{prefix} IS NULL"
        val_s = str(pred.value) if isinstance(pred.value, (int, float)) else f"'{pred.value}'"
        return f"{prefix} {pred.op} {val_s}"


def _is_identifier_char(ch: str) -> bool:
    """True for characters that may occur inside an unquoted SQL identifier.

    The underscore must count here: without it the FROM in an alias such as
    ``SELECT cn.name AS from_company`` reads as the FROM keyword.
    """
    return ch.isalnum() or ch == '_'


def _keyword_at(sql: str, upper_sql: str, i: int, keyword: str,
                *, start: int = 0, end: int | None = None) -> bool:
    """True when *keyword* (upper-case) starts at *i* as a standalone word.

    *start*/*end* bound the region being scanned; text outside it is treated as
    a word boundary.
    """
    if end is None:
        end = len(sql)
    stop = i + len(keyword)
    if upper_sql[i:stop] != keyword:
        return False
    if i > start and _is_identifier_char(sql[i - 1]):
        return False
    return stop >= end or not _is_identifier_char(sql[stop])


def _find_where_clause_end(sql: str) -> tuple[int, int]:
    """Find the boundaries of the WHERE clause in a SQL string.

    Returns (where_keyword_pos, where_clause_end_pos).
    where_keyword_pos is the index of 'W' in WHERE, or -1 if no WHERE.
    where_clause_end_pos is the index where the WHERE clause ends (before GROUP BY,
    ORDER BY, HAVING, LIMIT, UNION, or semicolon/end of string).
    """
    import re
    # Find WHERE keyword (not inside quotes or parens)
    # Simple approach: tokenize by tracking quote/paren state
    upper_sql = sql.upper()
    i = 0
    where_pos = -1
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            # Skip quoted string
            i += 1
            while i < len(sql) and sql[i] != "'":
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2  # escaped quote
                else:
                    i += 1
            i += 1  # skip closing quote
            continue
        if ch == '(':
            # Skip parenthesized expression
            depth = 1
            i += 1
            while i < len(sql) and depth > 0:
                if sql[i] == "'":
                    i += 1
                    while i < len(sql) and sql[i] != "'":
                        i += 1
                    i += 1
                    continue
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                i += 1
            continue
        # Check for WHERE keyword at top level
        if where_pos == -1 and _keyword_at(sql, upper_sql, i, 'WHERE'):
            where_pos = i
            i += 5
            continue
        i += 1

    if where_pos == -1:
        return -1, len(sql)

    # Now find where the WHERE clause ends (scan from after WHERE keyword)
    clause_enders = {'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT', 'WINDOW', 'FETCH'}
    i = where_pos + 5  # skip past WHERE
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < len(sql) and sql[i] != "'":
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if ch == '(':
            depth = 1
            i += 1
            while i < len(sql) and depth > 0:
                if sql[i] == "'":
                    i += 1
                    while i < len(sql) and sql[i] != "'":
                        i += 1
                    i += 1
                    continue
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                i += 1
            continue
        if ch == ';':
            return where_pos, i
        # Check for clause-ending keywords at top level
        if ch.isalpha():
            word_start = i
            while i < len(sql) and (sql[i].isalnum() or sql[i] == '_'):
                i += 1
            word = upper_sql[word_start:i]
            if word in clause_enders:
                # Back up to before any whitespace before the keyword
                end_pos = word_start
                while end_pos > where_pos + 5 and sql[end_pos - 1] in (' ', '\t', '\n', '\r'):
                    end_pos -= 1
                return where_pos, end_pos
            continue
        i += 1

    # WHERE clause extends to end of string (strip trailing whitespace/semicolons)
    end = len(sql)
    while end > where_pos + 5 and sql[end - 1] in (' ', '\t', '\n', '\r', ';'):
        end -= 1
    return where_pos, end


def _find_from_clause_start(sql: str) -> tuple[int, int]:
    """Find the boundaries of the FROM clause in a SQL string.

    Returns (from_keyword_pos, from_body_end).
    from_keyword_pos is the index of 'F' in FROM, or -1 if no top-level FROM.
    from_body_end is the index where the FROM body ends (before WHERE,
    GROUP BY, ORDER BY, HAVING, LIMIT, UNION, or semicolon/end of string).
    """
    upper_sql = sql.upper()
    i = 0
    from_pos = -1
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < len(sql) and sql[i] != "'":
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if ch == '(':
            depth = 1
            i += 1
            while i < len(sql) and depth > 0:
                if sql[i] == "'":
                    i += 1
                    while i < len(sql) and sql[i] != "'":
                        i += 1
                    i += 1
                    continue
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                i += 1
            continue
        if from_pos == -1 and _keyword_at(sql, upper_sql, i, 'FROM'):
            from_pos = i
            i += 4
            continue
        i += 1

    if from_pos == -1:
        return -1, len(sql)

    # If a WHERE exists, FROM ends where WHERE begins.
    where_pos, _ = _find_where_clause_end(sql)
    if where_pos >= 0:
        end_pos = where_pos
        while end_pos > from_pos + 4 and sql[end_pos - 1] in (' ', '\t', '\n', '\r'):
            end_pos -= 1
        return from_pos, end_pos

    # Otherwise scan forward for the next clause-ending keyword.
    clause_enders = {'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT', 'WINDOW', 'FETCH'}
    i = from_pos + 4
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < len(sql) and sql[i] != "'":
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if ch == '(':
            depth = 1
            i += 1
            while i < len(sql) and depth > 0:
                if sql[i] == "'":
                    i += 1
                    while i < len(sql) and sql[i] != "'":
                        i += 1
                    i += 1
                    continue
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                i += 1
            continue
        if ch == ';':
            return from_pos, i
        if ch.isalpha():
            word_start = i
            while i < len(sql) and (sql[i].isalnum() or sql[i] == '_'):
                i += 1
            word = upper_sql[word_start:i]
            if word in clause_enders:
                end_pos = word_start
                while end_pos > from_pos + 4 and sql[end_pos - 1] in (' ', '\t', '\n', '\r'):
                    end_pos -= 1
                return from_pos, end_pos
            continue
        i += 1

    end = len(sql)
    while end > from_pos + 4 and sql[end - 1] in (' ', '\t', '\n', '\r', ';'):
        end -= 1
    return from_pos, end


def _split_where_at_top_level_and(sql: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a WHERE clause body into segments at top-level AND keywords.

    Args:
        sql: The full SQL string
        start: Start of WHERE body (after 'WHERE ')
        end: End of WHERE clause

    Returns list of (seg_start, seg_end) tuples — positions in the original string.
    Each segment is one AND-separated condition (preserving original text).
    """
    upper_sql = sql.upper()
    segments = []
    seg_start = start
    i = start

    while i < end:
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < end and sql[i] != "'":
                if sql[i] == "'" and i + 1 < end and sql[i + 1] == "'":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if ch == '(':
            depth = 1
            i += 1
            while i < end and depth > 0:
                if sql[i] == "'":
                    i += 1
                    while i < end and sql[i] != "'":
                        i += 1
                    i += 1
                    continue
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                i += 1
            continue
        # Check for top-level AND keyword
        if _keyword_at(sql, upper_sql, i, 'AND', start=start, end=end):
            # End of current segment is before any whitespace before AND
            seg_end = i
            while seg_end > seg_start and sql[seg_end - 1] in (' ', '\t', '\n', '\r'):
                seg_end -= 1
            if seg_end > seg_start:
                segments.append((seg_start, seg_end))
            # New segment starts after AND + whitespace
            seg_start = i + 3
            while seg_start < end and sql[seg_start] in (' ', '\t', '\n', '\r'):
                seg_start += 1
            i = seg_start
            continue
        i += 1

    # Last segment
    seg_end = end
    while seg_end > seg_start and sql[seg_end - 1] in (' ', '\t', '\n', '\r'):
        seg_end -= 1
    if seg_end > seg_start:
        segments.append((seg_start, seg_end))

    return segments


def rewrite_sql(original_sql: str, predicates: set[Predicate], original_predicates: set[Predicate] = None,
                alias_map: dict[str, str] = None, eliminate_aliases: set[str] | None = None,
                remove_predicates: set[Predicate] | None = None) -> str:
    """Emit the rewritten SQL, appending injected predicates as a top-level AND.

    Textual, not AST-based, so the rest of the query stays verbatim (no sqlglot
    re-serialization artifacts such as IS NOT NULL → NOT IS NULL).

    Three code paths:
      A) Add-only (most common): pure string append, no AST re-serialization.
      B) Removal (predicate reduction): string surgery to remove specific conditions.
      C) Join elimination: AST-based (rewrite type not evaluated in the thesis).

    Args:
        original_sql: The original SQL query
        predicates: The expanded set of predicates (original + inferred)
        original_predicates: The original predicates extracted from the query (to identify new ones)
        alias_map: Mapping from alias to table name (e.g., {"t": "title_basics"})
        eliminate_aliases: Set of alias names to remove from the query (join elimination)
        remove_predicates: Original predicates to remove from the WHERE clause (subsumed by reduction)
    """
    # Create reverse alias map (table name -> alias) for display purposes
    reverse_alias_map = {}
    if alias_map:
        for alias, table in alias_map.items():
            if table not in reverse_alias_map or len(alias) < len(reverse_alias_map[table]):
                reverse_alias_map[table] = alias

    if original_predicates is None:
        original_predicates = predicates

    # Get all valid table names from the query
    valid_tables = set(reverse_alias_map.keys()) if reverse_alias_map else set()
    if alias_map:
        valid_tables.update(alias_map.keys())

    # Compute new predicates to add.
    # Sort deterministically before emitting SQL: `predicates - original_predicates`
    # is a set, whose iteration order depends on PYTHONHASHSEED (randomized per
    # process). Without this sort, two otherwise-identical runs emit the injected
    # predicates in different textual order, making refined_sql non-reproducible.
    # str(p.value) keeps the key type-safe (value may be a list, str, or number).
    new_inferred_predicates = sorted(
        predicates - original_predicates,
        key=lambda p: (p.table, p.column, p.op, str(p.value)),
    )
    new_pred_sqls = []
    for p in new_inferred_predicates:
        if reverse_alias_map and p.table not in valid_tables:
            print(f"WARNING: Skipping inferred predicate '{p.table}.{p.column} {p.op} {p.value}' - table '{p.table}' not found in query")
            continue
        new_pred_sqls.append(_predicate_to_sql(p, reverse_alias_map))

    # ── Path C: Join elimination (rare, uses AST) ─────────────────────────
    if eliminate_aliases:
        return _rewrite_sql_ast_fallback(
            original_sql, new_pred_sqls, remove_predicates, alias_map,
            eliminate_aliases, reverse_alias_map, valid_tables,
        )

    # ── Path B: Predicate removal (reduction) + addition ──────────────────
    if remove_predicates:
        return _rewrite_sql_with_removal(
            original_sql, new_pred_sqls, remove_predicates, alias_map,
        )

    # ── Path A: Add-only (most common) ────────────────────────────────────
    if not new_pred_sqls:
        return original_sql  # nothing changed — return verbatim

    sql = original_sql.rstrip()
    has_semicolon = sql.endswith(';')
    if has_semicolon:
        sql = sql[:-1].rstrip()

    where_pos, where_end = _find_where_clause_end(original_sql)
    if where_pos >= 0:
        # Append new predicates at end of WHERE clause
        suffix = " AND ".join(new_pred_sqls)
        result = original_sql[:where_end] + " AND " + suffix + original_sql[where_end:]
    else:
        # No WHERE clause — insert one (before trailing clauses or at end)
        suffix = " AND ".join(new_pred_sqls)
        # Find insertion point (before GROUP BY, ORDER BY, etc.)
        _, insert_pos = _find_where_clause_end(original_sql)
        result = original_sql[:insert_pos] + " WHERE " + suffix + original_sql[insert_pos:]

    return result


def _rewrite_sql_with_removal(
    original_sql: str,
    new_pred_sqls: list[str],
    remove_predicates: set[Predicate],
    alias_map: dict[str, str] | None,
) -> str:
    """Path B: Remove specific predicates from WHERE clause, add new ones.

    Splits the WHERE clause at top-level AND boundaries, identifies segments
    matching predicates to remove, keeps the rest verbatim, and appends new predicates.
    """
    where_pos, where_end = _find_where_clause_end(original_sql)
    if where_pos < 0:
        # No WHERE clause — just add new predicates if any
        if new_pred_sqls:
            return original_sql.rstrip().rstrip(';') + " WHERE " + " AND ".join(new_pred_sqls)
        return original_sql

    # Find WHERE body start (skip 'WHERE' keyword + whitespace)
    body_start = where_pos + 5  # len('WHERE')
    while body_start < where_end and original_sql[body_start] in (' ', '\t', '\n', '\r'):
        body_start += 1

    segments = _split_where_at_top_level_and(original_sql, body_start, where_end)

    # Determine which segments to keep
    kept_segments = []
    for seg_start, seg_end in segments:
        seg_text = original_sql[seg_start:seg_end]
        if alias_map and remove_predicates:
            # Try to parse this segment into a Predicate to see if it should be removed
            try:
                wrapper = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {seg_text}")
                where_node = wrapper.find(exp.Where)
                if where_node:
                    pred = _predicate_from_where_node(where_node.this, alias_map)
                    if pred is not None and pred in remove_predicates:
                        continue  # skip this segment
            except Exception:
                pass  # unparseable — keep it
        kept_segments.append(seg_text)

    # Add new predicates
    kept_segments.extend(new_pred_sqls)

    if not kept_segments:
        # All conditions removed and no new ones — drop WHERE clause
        before = original_sql[:where_pos].rstrip()
        after = original_sql[where_end:]
        return before + after

    new_where_body = " AND ".join(kept_segments)
    # Reconstruct: everything before WHERE body + new body + everything after WHERE clause
    result = original_sql[:body_start] + new_where_body + original_sql[where_end:]
    return result


def _rewrite_sql_ast_fallback(
    original_sql: str,
    new_pred_sqls: list[str],
    remove_predicates: set[Predicate] | None,
    alias_map: dict[str, str] | None,
    eliminate_aliases: set[str],
    reverse_alias_map: dict[str, str],
    valid_tables: set[str],
) -> str:
    """Path C: AST-based rewrite for join elimination (rare).

    Uses the existing AST approach but post-processes to undo known normalizations.
    """
    tree = sqlglot.parse_one(original_sql)

    # Join elimination
    tree = _eliminate_tables_from_ast(tree, eliminate_aliases)

    # Predicate reduction
    if remove_predicates and alias_map:
        where = tree.args.get("where")
        if where:
            cleaned = _filter_where_by_predicates(where.this, remove_predicates, alias_map)
            if cleaned:
                tree.set("where", exp.Where(this=cleaned))
            else:
                tree.set("where", None)

    # Build new predicate AST nodes for addition
    op_to_exp_map = {
        "=": exp.EQ, "!=": exp.NEQ, "<>": exp.NEQ,
        ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE,
    }

    new_conditions = []
    for pred_sql in new_pred_sqls:
        # Parse each new predicate via a wrapper query to get the AST node
        try:
            wrapper = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {pred_sql}")
            where_node = wrapper.find(exp.Where)
            if where_node:
                new_conditions.append(where_node.this)
        except Exception:
            pass

    where = tree.args.get("where")
    if new_conditions:
        if where:
            if isinstance(where.this, exp.And):
                all_conditions = list(where.this.flatten())
                actual_conditions = [c for c in all_conditions
                                    if isinstance(c, (exp.Binary, exp.In, exp.Is, exp.Like,
                                                     exp.Not, exp.Between, exp.And, exp.Or))]
                combined = exp.and_(*actual_conditions, *new_conditions)
            else:
                combined = exp.and_(where.this, *new_conditions)
            tree.set("where", exp.Where(this=combined))
        else:
            combined = exp.and_(*new_conditions)
            tree.set("where", exp.Where(this=combined))

    result = tree.sql()

    # Post-process: undo known sqlglot normalizations based on original SQL
    result = _undo_sqlglot_normalizations(original_sql, result)

    return result


def _undo_sqlglot_normalizations(original_sql: str, result_sql: str) -> str:
    """Reverse known sqlglot normalizations when the original SQL used different forms."""
    import re

    # 1. NOT <col> IS NULL → <col> IS NOT NULL (if original had IS NOT NULL)
    if 'is not null' in original_sql.lower():
        def _fix_is_not_null(m):
            col = m.group(1)
            return f"{col} IS NOT NULL"
        result_sql = re.sub(r'NOT (\w+\.\w+) IS NULL', _fix_is_not_null, result_sql)

    # 2. NOT <col> LIKE → <col> NOT LIKE (if original had NOT LIKE)
    if 'not like' in original_sql.lower():
        def _fix_not_like(m):
            col = m.group(1)
            pattern = m.group(2)
            return f"{col} NOT LIKE {pattern}"
        result_sql = re.sub(r"NOT (\w+\.\w+) LIKE ('(?:[^']*)')", _fix_not_like, result_sql)

    # 3. <> → != (if original used !=)
    if '!=' in original_sql and '<>' in result_sql:
        result_sql = result_sql.replace('<>', '!=')

    return result_sql


def _resolve_elimination_aliases(
    sql: str,
    rules: list[dict],
    original_predicates: set[Predicate],
    alias_map: dict[str, str],
    query_structure: dict | None,
) -> tuple[set[str], set[Predicate]]:
    """Determine which query aliases should be eliminated based on fired join_elimination rules.

    For each fired join_elimination rule, resolves the base table names in
    ``eliminates`` to specific query aliases.  When a base table has multiple
    aliases (e.g. info_type AS it1, info_type AS it2) the requires conditions
    are matched against query predicates to pick the correct alias.

    Returns ``(eliminate_aliases, extra_implies)`` where *extra_implies* contains
    alias-specific predicates for cases where a rule eliminates multiple aliases
    of the same base table (e.g. both it1 and it2).  In such cases the standard
    ``apply_rules`` path only produces one base-table-level predicate which
    ``rewrite_sql`` maps to a single alias; the extra predicates ensure all
    remaining aliases receive the correct filter.
    """
    tree = sqlglot.parse_one(sql)

    # Build base_table → [alias, ...] mapping from the query
    base_to_aliases: dict[str, list[str]] = {}
    for tbl in tree.find_all(exp.Table):
        base_name = tbl.this.name
        alias_expr = tbl.args.get("alias")
        alias_name = alias_expr.name if alias_expr else base_name
        base_to_aliases.setdefault(base_name, []).append(alias_name)

    # Collect aliases referenced in SELECT expressions / GROUP BY / ORDER BY / HAVING
    # — these must NOT be eliminated.
    protected_aliases: set[str] = set()
    # Only check projected columns (tree.expressions), NOT the full Select subtree
    for select_expr in tree.expressions:
        for col in select_expr.find_all(exp.Column):
            if col.table:
                protected_aliases.add(col.table)
    for clause_key in ("group", "order", "having"):
        clause = tree.args.get(clause_key)
        if clause:
            for col in clause.find_all(exp.Column):
                if col.table:
                    protected_aliases.add(col.table)

    eliminate_aliases: set[str] = set()
    # Track per-rule eliminations: [(rule, {base_table: [aliases]}), ...]
    rule_eliminations: list[tuple[dict, dict[str, list[str]]]] = []

    for rule in rules:
        if rule.get("type") != "join_elimination":
            continue
        eliminates = rule.get("eliminates", [])
        if not eliminates:
            continue

        # Check if this rule's requires are satisfied
        requires = rule["requires"]
        if isinstance(requires, dict):
            satisfied = evaluate_condition(requires, original_predicates, query_structure, alias_map)
        elif isinstance(requires, list):
            satisfied = all(
                any(entails(p, req, alias_map) for p in original_predicates)
                for req in requires
            )
        else:
            continue

        if not satisfied:
            continue

        # Collect base table names referenced in requires conditions
        def _collect_requires_tables(cond) -> set[str]:
            tables = set()
            if "column" in cond:
                tbl = cond["column"].split(".")[0]
                resolved = alias_map.get(tbl, tbl)
                tables.add(resolved)
            for sub in cond.get("conditions", []):
                tables.update(_collect_requires_tables(sub))
            return tables

        if isinstance(requires, dict):
            requires_tables = _collect_requires_tables(requires)
        else:
            requires_tables = set()
            for r in requires:
                requires_tables.update(_collect_requires_tables(r))

        this_rule_elims: dict[str, list[str]] = {}

        for base_table in eliminates:
            aliases_for_table = base_to_aliases.get(base_table, [])
            if not aliases_for_table:
                continue

            if len(aliases_for_table) == 1:
                alias = aliases_for_table[0]
                if alias not in protected_aliases:
                    eliminate_aliases.add(alias)
                    this_rule_elims.setdefault(base_table, []).append(alias)
            else:
                # Multiple aliases for same base table — match via requires predicates.
                # Extract alias-specific predicates by walking the WHERE AST
                # to find conditions that use each specific alias.
                if base_table not in requires_tables:
                    continue

                # Build alias-specific predicate sets from the AST
                # This preserves alias identity (it1 vs it2) unlike extracted predicates
                alias_specific_preds = _collect_alias_specific_predicates(
                    tree, aliases_for_table, alias_map
                )

                for alias in aliases_for_table:
                    if alias in protected_aliases:
                        continue
                    preds_for_alias = alias_specific_preds.get(alias, set())
                    if not preds_for_alias:
                        continue

                    def _check_requires_for_alias(cond, _preds=preds_for_alias) -> bool:
                        if "column" in cond:
                            tbl = cond["column"].split(".")[0]
                            resolved = alias_map.get(tbl, tbl)
                            if resolved != base_table:
                                return True  # Not about this table, skip
                            return any(entails(p, cond, alias_map) for p in _preds)
                        op = cond.get("op", "").upper()
                        subs = cond.get("conditions", [])
                        if op == "AND":
                            return all(_check_requires_for_alias(s, _preds) for s in subs)
                        elif op == "OR":
                            return any(_check_requires_for_alias(s, _preds) for s in subs)
                        return False

                    if isinstance(requires, dict):
                        if _check_requires_for_alias(requires):
                            eliminate_aliases.add(alias)
                            this_rule_elims.setdefault(base_table, []).append(alias)
                    elif isinstance(requires, list):
                        if all(_check_requires_for_alias(r) for r in requires):
                            eliminate_aliases.add(alias)
                            this_rule_elims.setdefault(base_table, []).append(alias)

        if this_rule_elims:
            rule_eliminations.append((rule, this_rule_elims))

    # --- Compute alias-specific implies for multi-alias eliminations -----------
    # When a rule eliminates multiple aliases of the same base table (e.g. both
    # it1 and it2 for info_type), apply_rules only produces one base-table-level
    # implies predicate which rewrite_sql maps to a single alias.  We use the
    # query's equi-join conditions to pair each eliminated alias with the correct
    # target alias for the implies predicate.
    extra_implies: set[Predicate] = set()

    for rule, elim_map in rule_eliminations:
        for base_table, eliminated_list in elim_map.items():
            if len(eliminated_list) <= 1:
                continue

            equi_joins, _ = extract_equi_joins(sql, keep_aliases=True)
            rule_alias_map = rule.get("alias_map", {})

            for eliminated_alias in eliminated_list:
                for impl in rule.get("implies", []):
                    impl_table_raw = impl["column"].split(".")[0]
                    impl_base_table = rule_alias_map.get(
                        impl_table_raw, alias_map.get(impl_table_raw, impl_table_raw)
                    )
                    impl_column = impl["column"].split(".")[1]

                    # Find the query alias of the implies target table that is
                    # joined to the eliminated alias
                    target_alias = None
                    for join in equi_joins:
                        l_alias, l_col = join["left"].split(".")
                        r_alias, r_col = join["right"].split(".")

                        if l_alias == eliminated_alias:
                            r_base = alias_map.get(r_alias, r_alias)
                            if r_base == impl_base_table:
                                target_alias = r_alias
                                break
                        elif r_alias == eliminated_alias:
                            l_base = alias_map.get(l_alias, l_alias)
                            if l_base == impl_base_table:
                                target_alias = l_alias
                                break

                    if target_alias:
                        op = impl["op"]
                        value = impl["value"]
                        extra_implies.add(Predicate(
                            table=target_alias,
                            column=impl_column,
                            op=op,
                            value=tuple(value) if isinstance(value, list) else value,
                        ))

    return eliminate_aliases, extra_implies


def apply_sql_rules(sql: str, rules: list[dict]) -> tuple[str, bool]:
    original_predicates, alias_map, query_structure = extract_predicates(sql)
    query_joins, _ = extract_equi_joins(sql, keep_aliases=False)
    expanded, rule_drops = apply_rules(
        original_predicates, rules, query_structure, alias_map, collect_drops=True,
        query_joins=query_joins,
    )

    # Determine aliases to eliminate from fired join_elimination rules
    eliminate_aliases, extra_implies = _resolve_elimination_aliases(
        sql, rules, original_predicates, alias_map, query_structure
    )

    # When a rule eliminates multiple aliases of the same base table, apply_rules
    # only produces one base-table-level implies predicate.  Replace it with the
    # alias-specific predicates so each remaining alias gets the correct filter.
    if extra_implies:
        superseded = set()
        for ep in extra_implies:
            for p in expanded - original_predicates:
                if p.column == ep.column and p.op == ep.op and p.value == ep.value:
                    superseded.add(p)
        expanded = (expanded - superseded) | extra_implies

    # Compute original predicates that were removed by reduction
    removed_originals = original_predicates - expanded
    # Restrict rule-driven drops to predicates that actually exist in the query
    effective_drops = rule_drops & original_predicates
    all_removed = removed_originals | effective_drops

    new_sql = rewrite_sql(sql, expanded, original_predicates, alias_map, eliminate_aliases,
                          remove_predicates=all_removed if all_removed else None)
    has_new_predicates = (
        len(expanded - original_predicates) > 0
        or bool(eliminate_aliases)
        or bool(all_removed)
    )
    return new_sql, has_new_predicates


# ---------------------------------------------------------------------------
# Fixed join order: reconstruct SQL with explicit JOIN tree from plan
# ---------------------------------------------------------------------------

# ── Join types recognised as "real" joins in the plan tree ────────────────
_JOIN_OPS = {"HASH_JOIN", "NESTED_LOOP_JOIN", "PIECEWISE_MERGE_JOIN",
             "BLOCKWISE_NL_JOIN", "CROSS_PRODUCT"}
_SCAN_OPS = {"SEQ_SCAN", "TABLE_SCAN", "INDEX_SCAN"}


def _extract_join_tree(profiling_data: dict) -> dict | None:
    """Extract an abstract join tree from a DuckDB profiling JSON.

    Returns a nested dict representing the join tree:
      - Leaf (scan):  {"type": "scan", "table": "...", "filters": "..."}
      - Join:         {"type": "join", "left": <subtree>, "right": <subtree>}

    Returns None if the plan cannot be parsed.
    """

    def _op_name(node: dict) -> str:
        return (node.get("operator_name") or node.get("name") or "").strip().upper()

    def _walk(node: dict) -> dict | None:
        if not isinstance(node, dict):
            return None

        op = _op_name(node)
        children = node.get("children", [])
        extra = node.get("extra_info", {})
        if isinstance(extra, str):
            extra = {}

        # ── Scan leaf ─────────────────────────────────────────────────
        if op in _SCAN_OPS:
            table = extra.get("Table", "")
            filters = extra.get("Filters", "")
            # Filters can be a string or a list in some DuckDB versions
            if isinstance(filters, list):
                filters = " AND ".join(str(f) for f in filters)
            return {"type": "scan", "table": table, "filters": filters}

        # ── Join node ─────────────────────────────────────────────────
        if op in _JOIN_OPS:
            join_type = extra.get("Join Type", "INNER").upper()

            # MARK joins are optimizer artifacts (for IN-list evaluation).
            # Flatten: recurse into the non-COLUMN_DATA_SCAN child only.
            if join_type == "MARK":
                for child in children:
                    child_op = _op_name(child)
                    if child_op not in ("COLUMN_DATA_SCAN",):
                        return _walk(child)
                return None

            if len(children) != 2:
                return None
            left = _walk(children[0])
            right = _walk(children[1])
            if left is None or right is None:
                return None
            return {"type": "join", "left": left, "right": right}

        # ── Pass-through (FILTER, PROJECTION, AGGREGATE, etc.) ───────
        if len(children) == 1:
            return _walk(children[0])

        # Multiple children on a non-join node: unexpected — bail out.
        # Exception: root wrapper may have one useful child.
        if len(children) > 1:
            # Try to find the one subtree that contains joins/scans
            results = [_walk(c) for c in children]
            results = [r for r in results if r is not None]
            if len(results) == 1:
                return results[0]
            return None

        return None  # no children, not a scan

    # Navigate into root wrapper. DuckDB sometimes returns the plan as a
    # list of root nodes instead of a {"children": [...]} dict — handle both.
    if isinstance(profiling_data, list):
        top_children = profiling_data
    elif isinstance(profiling_data, dict):
        top_children = profiling_data.get("children", [])
    else:
        return None
    if not top_children:
        return None
    return _walk(top_children[0])


def _collect_leaves(join_tree: dict) -> list[dict]:
    """Collect all scan leaf nodes from a join tree (in-order)."""
    if join_tree["type"] == "scan":
        return [join_tree]
    leaves = []
    if "left" in join_tree:
        leaves.extend(_collect_leaves(join_tree["left"]))
    if "right" in join_tree:
        leaves.extend(_collect_leaves(join_tree["right"]))
    return leaves


def _collect_subtree_aliases(node: dict) -> set[str]:
    """Collect all assigned aliases from a join (sub)tree."""
    if node["type"] == "scan":
        alias = node.get("alias")
        return {alias} if alias else set()
    return _collect_subtree_aliases(node["left"]) | _collect_subtree_aliases(node["right"])


def _map_plan_leaves_to_aliases(
    join_tree: dict,
    alias_map: dict[str, str],
    equi_joins: list[dict],
    refined_tree,
) -> bool:
    """Map each scan leaf in the join tree to a SQL alias.

    Mutates the join tree in-place, adding an ``"alias"`` key to each leaf.
    Returns True on success, False if mapping fails.

    Strategy (multi-pass):
      1. Unique base tables → direct assignment
      2. Filter string matching against per-alias predicates
      3. Join-graph topology propagation
      4. Fallback: arbitrary assignment for symmetric cases
    """
    leaves = _collect_leaves(join_tree)

    # Build base_table → [aliases] (only real aliases, not self-references)
    base_to_aliases: dict[str, list[str]] = {}
    for alias, base in alias_map.items():
        if alias != base:  # skip self-referencing base_name→base_name entries
            base_to_aliases.setdefault(base, []).append(alias)
    # Also handle non-aliased tables
    for alias, base in alias_map.items():
        if alias == base and base not in base_to_aliases:
            base_to_aliases[base] = [base]

    # Build base_table → [leaf_nodes]
    base_to_leaves: dict[str, list[dict]] = {}
    for leaf in leaves:
        base_to_leaves.setdefault(leaf["table"], []).append(leaf)

    # ── Pass 1: unique tables ─────────────────────────────────────────
    for base_table, aliases in base_to_aliases.items():
        leaf_list = base_to_leaves.get(base_table, [])
        if len(aliases) == 1 and len(leaf_list) == 1:
            leaf_list[0]["alias"] = aliases[0]

    unmatched_leaves = [l for l in leaves if "alias" not in l]
    if not unmatched_leaves:
        return True

    # ── Pass 2: filter string matching ────────────────────────────────
    # Collect the set of ambiguous aliases (those not yet assigned)
    assigned_aliases = {l["alias"] for l in leaves if "alias" in l}
    ambiguous_aliases = []
    for base_table in {l["table"] for l in unmatched_leaves}:
        for a in base_to_aliases.get(base_table, []):
            if a not in assigned_aliases:
                ambiguous_aliases.append(a)

    if ambiguous_aliases:
        alias_preds = _collect_alias_specific_predicates(
            refined_tree, ambiguous_aliases, alias_map
        )

        for leaf in unmatched_leaves:
            if "alias" in leaf:
                continue
            candidates = [a for a in base_to_aliases.get(leaf["table"], [])
                          if a not in assigned_aliases]
            if not candidates:
                continue

            leaf_filter = leaf.get("filters", "")
            if not leaf_filter:
                continue

            # Score each candidate by how many of its predicates appear in the filter
            best_alias = None
            best_score = -1
            for alias in candidates:
                score = 0
                for pred in alias_preds.get(alias, set()):
                    # Build recognisable fragments from the predicate
                    fragments = _predicate_to_filter_fragments(pred)
                    for frag in fragments:
                        if frag in leaf_filter:
                            score += 1
                            break
                if score > best_score:
                    best_score = score
                    best_alias = alias

            # Only assign if the best candidate has a strictly higher score
            if best_alias and best_score > 0:
                scores = []
                for alias in candidates:
                    s = 0
                    for pred in alias_preds.get(alias, set()):
                        for frag in _predicate_to_filter_fragments(pred):
                            if frag in leaf_filter:
                                s += 1
                                break
                    scores.append(s)
                # Assign only if best is unique
                if scores.count(best_score) == 1:
                    leaf["alias"] = best_alias
                    assigned_aliases.add(best_alias)

    unmatched_leaves = [l for l in leaves if "alias" not in l]
    if not unmatched_leaves:
        return True

    # ── Pass 3: topology propagation ──────────────────────────────────
    # Build parent pointers and sibling references
    def _set_parents(node, parent=None, side=None):
        node["_parent"] = parent
        node["_side"] = side
        if node["type"] == "join":
            _set_parents(node["left"], node, "left")
            _set_parents(node["right"], node, "right")

    _set_parents(join_tree)

    changed = True
    while changed:
        changed = False
        for leaf in unmatched_leaves:
            if "alias" in leaf:
                continue
            parent = leaf.get("_parent")
            if parent is None:
                continue
            side = leaf["_side"]
            sibling_side = "right" if side == "left" else "left"
            sibling = parent[sibling_side]
            sibling_aliases = _collect_subtree_aliases(sibling)

            if not sibling_aliases:
                continue

            candidates = [a for a in base_to_aliases.get(leaf["table"], [])
                          if a not in assigned_aliases]
            if len(candidates) != 1:
                # Try to narrow using equi_joins: which candidate connects to a sibling alias?
                narrowed = []
                for a in candidates:
                    for ej in equi_joins:
                        l_alias = ej["left"].split(".")[0]
                        r_alias = ej["right"].split(".")[0]
                        if (a == l_alias and r_alias in sibling_aliases) or \
                           (a == r_alias and l_alias in sibling_aliases):
                            narrowed.append(a)
                            break
                if len(narrowed) == 1:
                    candidates = narrowed

            if len(candidates) == 1:
                leaf["alias"] = candidates[0]
                assigned_aliases.add(candidates[0])
                changed = True

        unmatched_leaves = [l for l in leaves if "alias" not in l]
        if not unmatched_leaves:
            break

    if not unmatched_leaves:
        # Clean up temporary parent pointers
        _cleanup_parent_pointers(join_tree)
        return True

    # ── Pass 4: fallback — arbitrary assignment ───────────────────────
    for base_table in {l["table"] for l in unmatched_leaves}:
        remaining_leaves = [l for l in unmatched_leaves
                            if l["table"] == base_table and "alias" not in l]
        remaining_aliases = [a for a in base_to_aliases.get(base_table, [])
                             if a not in assigned_aliases]
        for leaf, alias in zip(remaining_leaves, remaining_aliases):
            leaf["alias"] = alias
            assigned_aliases.add(alias)

    _cleanup_parent_pointers(join_tree)

    # Check all leaves are assigned
    return all("alias" in l for l in leaves)


def _cleanup_parent_pointers(node: dict):
    """Remove temporary _parent/_side keys from tree nodes."""
    node.pop("_parent", None)
    node.pop("_side", None)
    if node["type"] == "join":
        _cleanup_parent_pointers(node["left"])
        _cleanup_parent_pointers(node["right"])


def _predicate_to_filter_fragments(pred: Predicate) -> list[str]:
    """Convert a Predicate to possible DuckDB filter string fragments.

    DuckDB's plan filter strings use formats like:
      col=val, col>val, col<val, col!=val
      contains(col, 'val')  (for LIKE '%val%')
      prefix(col, 'val')    (for LIKE 'val%')
    """
    col = pred.column
    val = str(pred.value) if pred.value is not None else "NULL"
    fragments = []

    if pred.op in ("=", "eq"):
        fragments.append(f"{col}={val}")
        fragments.append(f"{col}='{val}'")
    elif pred.op in (">", "gt"):
        fragments.append(f"{col}>{val}")
    elif pred.op in (">=", "gte"):
        fragments.append(f"{col}>={val}")
    elif pred.op in ("<", "lt"):
        fragments.append(f"{col}<{val}")
    elif pred.op in ("<=", "lte"):
        fragments.append(f"{col}<={val}")
    elif pred.op in ("!=", "<>", "neq"):
        fragments.append(f"{col}!={val}")
        fragments.append(f"{col}<>{val}")
    elif pred.op in ("LIKE", "like"):
        fragments.append(f"contains({col}")
        fragments.append(f"prefix({col}")
        fragments.append(f"suffix({col}")
        fragments.append(f"{col}")  # DuckDB may inline the pattern
    elif pred.op in ("IN", "in"):
        fragments.append(f"{col}=")  # DuckDB may convert IN to =
        fragments.append(f"{col} IN")
    elif pred.op in ("IS", "is"):
        fragments.append(f"{col} IS {val}".upper())
    elif pred.op == "BETWEEN":
        if isinstance(pred.value, tuple) and len(pred.value) == 2:
            fragments.append(f"{col}>={pred.value[0]}")
            fragments.append(f"{col}<={pred.value[1]}")

    return fragments


def _prune_eliminated_aliases(join_tree: dict, eliminated_aliases: set[str]) -> dict | None:
    """Remove leaves for eliminated aliases, collapsing their parent join nodes.

    Returns the pruned tree, or None if the entire tree is eliminated.
    """
    if not eliminated_aliases:
        return join_tree

    if join_tree["type"] == "scan":
        if join_tree.get("alias") in eliminated_aliases:
            return None
        return join_tree

    # Recursively prune children
    left = _prune_eliminated_aliases(join_tree["left"], eliminated_aliases)
    right = _prune_eliminated_aliases(join_tree["right"], eliminated_aliases)

    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left

    return {"type": "join", "left": left, "right": right}


def _build_fixed_join_sql(
    join_tree: dict,
    equi_joins: list[dict],
    refined_sql: str,
    alias_map: dict[str, str],
) -> str:
    """Build SQL with explicit JOIN ... ON syntax matching the join tree.

    Moves equi-join conditions from WHERE into ON clauses at the appropriate
    join nodes.  Filter predicates (and rule-added predicates) stay in WHERE.
    """
    # Parse the equi-join list into a set of (left_alias.col, right_alias.col) tuples
    ej_set: list[tuple[str, str, str]] = []  # (left_ref, right_ref, sql_fragment)
    for ej in equi_joins:
        left_ref = ej["left"]   # e.g. "cn1.id"
        right_ref = ej["right"]  # e.g. "mc1.company_id"
        sql_frag = f"{left_ref} = {right_ref}"
        ej_set.append((left_ref, right_ref, sql_frag))

    used_conditions: set[int] = set()  # indices into ej_set

    # ── Annotate subtree alias sets ───────────────────────────────────
    def _annotate_aliases(node):
        if node["type"] == "scan":
            node["_aliases"] = {node["alias"]}
        else:
            _annotate_aliases(node["left"])
            _annotate_aliases(node["right"])
            node["_aliases"] = node["left"]["_aliases"] | node["right"]["_aliases"]

    _annotate_aliases(join_tree)

    # ── Assign conditions to join nodes (bottom-up via recursion) ─────
    def _assign_conditions(node):
        """Assign equi-join conditions to the lowest applicable join node."""
        if node["type"] == "scan":
            return
        _assign_conditions(node["left"])
        _assign_conditions(node["right"])

        left_aliases = node["left"]["_aliases"]
        right_aliases = node["right"]["_aliases"]
        on_conds = []
        for i, (lref, rref, frag) in enumerate(ej_set):
            if i in used_conditions:
                continue
            l_alias = lref.split(".")[0]
            r_alias = rref.split(".")[0]
            if (l_alias in left_aliases and r_alias in right_aliases) or \
               (r_alias in left_aliases and l_alias in right_aliases):
                on_conds.append(frag)
                used_conditions.add(i)
        node["_on_conditions"] = on_conds

    _assign_conditions(join_tree)

    # ── Build FROM clause string ──────────────────────────────────────
    def _build_from(node) -> str:
        if node["type"] == "scan":
            alias = node["alias"]
            base = alias_map.get(alias, alias)
            if alias != base:
                return f"{base} AS {alias}"
            return base

        left_str = _build_from(node["left"])
        right_str = _build_from(node["right"])

        # Wrap right side in parens if it's a join (bushy tree)
        if node["right"]["type"] == "join":
            right_str = f"({right_str})"

        on_conds = node.get("_on_conditions", [])
        if on_conds:
            on_clause = " AND ".join(on_conds)
            return f"{left_str} INNER JOIN {right_str} ON {on_clause}"
        else:
            return f"{left_str} CROSS JOIN {right_str}"

    from_clause = _build_from(join_tree)

    # ── Rebuild WHERE via string surgery: preserve original text, drop
    #    equi-joins that moved to ON. Mirrors rewrite_sql Path A discipline
    #    (avoids sqlglot IS NOT NULL → NOT IS NULL canonicalization).
    used_pairs: set[tuple[str, str]] = set()
    for i in used_conditions:
        lref, rref, _ = ej_set[i]
        used_pairs.add((lref, rref))
        used_pairs.add((rref, lref))  # both directions

    def _segment_is_used_equi_join(seg_text: str) -> bool:
        s = seg_text.strip()
        try:
            wrapper = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {s}")
        except Exception:
            return False
        w = wrapper.find(exp.Where)
        if not w:
            return False
        node = w.this
        while isinstance(node, exp.Paren):
            node = node.this
        if not isinstance(node, exp.EQ):
            return False
        left, right = node.this, node.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            return False
        if not left.table or not right.table or left.table == right.table:
            return False
        return (f"{left.table}.{left.name}", f"{right.table}.{right.name}") in used_pairs

    # Locate verbatim regions in refined_sql
    from_pos, from_end = _find_from_clause_start(refined_sql)
    where_pos, where_end = _find_where_clause_end(refined_sql)

    select_prefix = refined_sql[:from_pos].rstrip() if from_pos >= 0 else refined_sql.rstrip()

    tail_start = where_end if where_pos >= 0 else from_end
    tail = refined_sql[tail_start:].strip().rstrip(';').rstrip()

    where_str = ""
    if where_pos >= 0:
        body_start = where_pos + 5
        while body_start < where_end and refined_sql[body_start] in (' ', '\t', '\n', '\r'):
            body_start += 1
        segments = _split_where_at_top_level_and(refined_sql, body_start, where_end)
        kept = [refined_sql[s:e] for s, e in segments
                if not _segment_is_used_equi_join(refined_sql[s:e])]
        if kept:
            where_str = " WHERE " + " AND ".join(kept)

    # Clean up temporary keys from tree nodes
    def _cleanup(node):
        node.pop("_aliases", None)
        node.pop("_on_conditions", None)
        if node["type"] == "join":
            _cleanup(node["left"])
            _cleanup(node["right"])

    _cleanup(join_tree)

    trailing_str = (" " + tail) if tail else ""
    return f"{select_prefix} FROM {from_clause}{where_str}{trailing_str}"


def reconstruct_with_fixed_join_order(
    original_sql: str,
    refined_sql: str,
    profiling_data: dict,
) -> str | None:
    """Reconstruct refined SQL with the original query's join tree structure.

    Takes the join tree from the original query's execution plan and rebuilds
    the refined SQL (which has additional predicates from rules) using explicit
    JOIN ... ON syntax that matches the original plan's join order.

    Args:
        original_sql:   The original query (for determining eliminated aliases).
        refined_sql:    The rewritten query from apply_sql_rules.
        profiling_data: DuckDB profiling JSON from running the original query.

    Returns:
        Reconstructed SQL string, or None if reconstruction fails (caller
        should fall back to normal execution).
    """
    try:
        # 1. Extract join tree from profiling data
        join_tree = _extract_join_tree(profiling_data)
        if join_tree is None:
            return None

        # Single-table query: no join tree to fix
        if join_tree["type"] == "scan":
            return refined_sql

        # 2. Determine eliminated aliases
        orig_tree = sqlglot.parse_one(original_sql)
        refined_parsed = sqlglot.parse_one(refined_sql)

        orig_aliases = set()
        for tbl in orig_tree.find_all(exp.Table):
            alias_expr = tbl.args.get("alias")
            alias_name = alias_expr.name if alias_expr else tbl.this.name
            orig_aliases.add(alias_name)

        refined_aliases = set()
        for tbl in refined_parsed.find_all(exp.Table):
            alias_expr = tbl.args.get("alias")
            alias_name = alias_expr.name if alias_expr else tbl.this.name
            refined_aliases.add(alias_name)

        eliminated_aliases = orig_aliases - refined_aliases

        # 3. Get equi-joins and alias map from the refined SQL
        equi_joins, alias_map = extract_equi_joins(refined_sql, keep_aliases=True)

        # 4. Map plan leaves to SQL aliases (using original SQL's alias map
        #    for the mapping, since the plan was generated from the original)
        orig_equi_joins, orig_alias_map = extract_equi_joins(original_sql, keep_aliases=True)
        if not _map_plan_leaves_to_aliases(join_tree, orig_alias_map, orig_equi_joins, orig_tree):
            return None

        # 5. Prune eliminated aliases
        if eliminated_aliases:
            join_tree = _prune_eliminated_aliases(join_tree, eliminated_aliases)
            if join_tree is None:
                return None

        # 6. Build SQL with fixed join order
        return _build_fixed_join_sql(join_tree, equi_joins, refined_sql, alias_map)

    except Exception as e:
        print(f"  Warning: join order reconstruction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Join extraction: derive the 'joins' field for a rule from the SQL query
# ---------------------------------------------------------------------------

def extract_equi_joins(
    sql: str, *, keep_aliases: bool = False,
) -> tuple[list[dict], dict[str, str]]:
    """Extract equi-join conditions (column = column) from a SQL query.

    Looks in both WHERE clause conditions and explicit JOIN ON clauses.
    Returns (joins_list, alias_map) where each join is
    {"left": "table.column", "right": "table.column"}.

    When *keep_aliases* is False (default) table references in the returned
    joins are resolved to their base table names.  When True, the original
    alias names used in the SQL are preserved so that multiple aliases of
    the same base table remain distinguishable.
    """
    tree = sqlglot.parse_one(sql)
    alias_map: dict[str, str] = {}

    for tbl in tree.find_all(exp.Table):
        base_name = tbl.this.name
        alias_expr = tbl.args.get("alias")
        alias_name = alias_expr.name if alias_expr else None
        if alias_name:
            alias_map[alias_name] = base_name
        alias_map.setdefault(base_name, base_name)

    joins: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _try_add_eq(eq_expr):
        left = eq_expr.this
        right = eq_expr.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            return
        lt = left.table
        rt = right.table
        if not lt or not rt:
            return
        if keep_aliases:
            left_table = lt
            right_table = rt
            # Skip self-joins on the *same alias* (same instance)
            if lt == rt:
                return
        else:
            left_table = alias_map.get(lt, lt)
            right_table = alias_map.get(rt, rt)
            if left_table == right_table:
                return
        key = (f"{left_table}.{left.name}", f"{right_table}.{right.name}")
        rev_key = (key[1], key[0])
        if key not in seen and rev_key not in seen:
            seen.add(key)
            joins.append({"left": key[0], "right": key[1]})

    # WHERE clause
    where = tree.args.get("where")
    if where:
        for eq in where.find_all(exp.EQ):
            _try_add_eq(eq)

    # Explicit JOIN ON clauses
    for join_node in tree.find_all(exp.Join):
        on = join_node.args.get("on")
        if on:
            for eq in ([on] if isinstance(on, exp.EQ) else on.find_all(exp.EQ)):
                _try_add_eq(eq)

    return joins, alias_map


def _collect_tables_from_rule(rule: dict) -> set[str]:
    """Get all table names referenced in a rule's requires and implies."""
    tables: set[str] = set()

    def _visit(cond: dict):
        if "column" in cond:
            tables.add(cond["column"].split(".")[0])
        for sub in cond.get("conditions", []):
            _visit(sub)

    requires = rule.get("requires", {})
    if isinstance(requires, list):
        for r in requires:
            _visit(r)
    elif isinstance(requires, dict):
        _visit(requires)

    for impl in rule.get("implies", []):
        if "column" in impl:
            tables.add(impl["column"].split(".")[0])

    return tables


def _find_connecting_joins(
    all_joins: list[dict], target_tables: set[str],
) -> list[dict]:
    """Find the minimal set of joins that connect all target_tables.

    Uses BFS on the join graph.  May include intermediate tables.
    """
    from collections import defaultdict, deque

    if len(target_tables) <= 1:
        return []

    # Build adjacency list
    graph: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for j in all_joins:
        lt = j["left"].split(".")[0]
        rt = j["right"].split(".")[0]
        graph[lt].append((rt, j))
        graph[rt].append((lt, j))

    target_list = list(target_tables)
    needed_keys: set[tuple[str, str]] = set()
    needed_joins: list[dict] = []

    # Connect every target table to the first one via BFS
    for end in target_list[1:]:
        start = target_list[0]
        visited = {start}
        queue: deque[tuple[str, list[dict]]] = deque([(start, [])])
        found = False
        while queue and not found:
            current, path = queue.popleft()
            for neighbor, join_dict in graph[current]:
                if neighbor in visited:
                    continue
                new_path = path + [join_dict]
                if neighbor == end:
                    for j in new_path:
                        key = (j["left"], j["right"])
                        if key not in needed_keys:
                            needed_keys.add(key)
                            needed_joins.append(j)
                    found = True
                    break
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return needed_joins


def _determine_alias_assignment_from_sql(
    sql: str,
    rule: dict,
    alias_map: dict[str, str],
) -> dict[str, str]:
    """Determine alias assignment by re-parsing SQL to get alias-level predicates.

    This is more reliable than matching against already-resolved predicates
    because it preserves which alias each predicate came from.
    """
    from collections import defaultdict

    tree = sqlglot.parse_one(sql)

    # Build reverse map: base_table → [alias, ...]
    base_to_aliases: dict[str, list[str]] = defaultdict(list)
    for alias, base in alias_map.items():
        if alias != base:
            base_to_aliases[base].append(alias)
    for base in set(alias_map.values()):
        if base not in base_to_aliases or not base_to_aliases[base]:
            base_to_aliases[base] = [base]

    # Extract predicates with ALIAS names (not resolved to base)
    alias_predicates: dict[str, list[tuple[str, str, Any]]] = defaultdict(list)
    where = tree.args.get("where")
    if where:
        for node in where.walk():
            node = node[0] if isinstance(node, tuple) else node
            if isinstance(node, exp.Binary) and not isinstance(node, (exp.And, exp.Or, exp.Like)):
                if isinstance(node.left, exp.Column) and not isinstance(node.right, exp.Column):
                    alias_name = node.left.table or ""
                    col_name = node.left.name
                    op = OP_SYMBOLS.get(node.key, node.key)
                    val = node.right.name or str(node.right.this)
                    alias_predicates[alias_name].append((col_name, op, val))
            elif isinstance(node, exp.In):
                col = node.this
                if isinstance(col, exp.Column):
                    alias_name = col.table or ""
                    col_name = col.name
                    values = tuple(v.this for v in node.expressions)
                    alias_predicates[alias_name].append((col_name, "IN", values))
            elif isinstance(node, exp.Like):
                col = node.this
                if isinstance(col, exp.Column):
                    alias_name = col.table or ""
                    col_name = col.name
                    val = node.expression.this if hasattr(node.expression, 'this') else str(node.expression)
                    # sqlglot >=30 represents "NOT LIKE" as a bare Like node with negate=True.
                    op = "NOT LIKE" if node.args.get("negate") else "LIKE"
                    alias_predicates[alias_name].append((col_name, op, val))

    # Collect leaf conditions from rule's requires
    def _leaf_conditions(cond: dict) -> list[dict]:
        if "column" in cond and "op" in cond and "conditions" not in cond:
            return [cond]
        leaves = []
        for sub in cond.get("conditions", []):
            leaves.extend(_leaf_conditions(sub))
        return leaves

    requires = rule.get("requires", {})
    if isinstance(requires, list):
        req_leaves = requires
    elif isinstance(requires, dict):
        req_leaves = _leaf_conditions(requires)
    else:
        req_leaves = []

    assignment: dict[str, str] = {}

    for base_table, aliases in base_to_aliases.items():
        if len(aliases) == 1:
            assignment[base_table] = aliases[0]
            continue

        # Multiple aliases — score each by how many rule conditions match
        rule_conds = [
            c for c in req_leaves
            if c["column"].split(".")[0] == base_table
        ]

        if not rule_conds:
            # Table not in requires — pick any alias for now; the join
            # connectivity check later will determine the right one.
            assignment[base_table] = aliases[0]
            continue

        best_alias = None
        best_score = -1
        for alias in aliases:
            preds = alias_predicates.get(alias, [])
            score = 0
            for rc in rule_conds:
                rc_col = rc["column"].split(".")[1]
                rc_op = rc["op"].upper()
                rc_val = rc["value"]
                for (pcol, pop, pval) in preds:
                    if pcol == rc_col and pop.upper() == rc_op:
                        # Value match (loose — string or set comparison)
                        if isinstance(rc_val, list):
                            if isinstance(pval, tuple) and set(pval) == set(str(v) for v in rc_val):
                                score += 2
                            else:
                                score += 1
                        elif str(pval) == str(rc_val):
                            score += 2
                        else:
                            score += 1
            if score > best_score:
                best_score = score
                best_alias = alias

        assignment[base_table] = best_alias if best_alias else aliases[0]

    return assignment


def derive_rule_joins(sql: str, rule: dict) -> dict:
    """Derive the ``joins`` field for a rule from the SQL query.

    Extracts equi-join conditions from the SQL **in alias form** so that
    multiple aliases of the same base table remain distinguishable.
    The rule dict is updated **in-place** with:
      - ``joins``: minimal connecting joins using alias names
      - ``alias_map``: mapping from alias → base table name
    """
    # Get joins in alias form
    alias_joins, alias_map = extract_equi_joins(sql, keep_aliases=True)

    # Determine which alias corresponds to each base table in the rule
    base_assignment = _determine_alias_assignment_from_sql(sql, rule, alias_map)

    # Collect base table names from the rule
    raw_tables = _collect_tables_from_rule(rule)

    # Resolve any aliases the LLM might have used and convert to alias names
    target_aliases: set[str] = set()
    for t in raw_tables:
        base = alias_map.get(t, t)
        alias = base_assignment.get(base, base_assignment.get(t, t))
        target_aliases.add(alias)

    connecting = _find_connecting_joins(alias_joins, target_aliases)
    rule["joins"] = connecting
    rule["alias_map"] = alias_map
    return rule


if __name__ == "__main__":

    sql = """
    SELECT t.tconst, r.averageRating
    FROM title_basics t
    JOIN title_ratings r ON t.tconst = r.tconst
    WHERE
    t.titleType = 'movie'
    AND r.averageRating >= 8.5
    """

    RULES = [
        # Simple rule with implicit AND (backward compatible)
        {
            "id": "IMDB_R1",
            "requires": [
                {
                    "column": "title_ratings.averageRating",
                    "op": ">=",
                    "value": 8.5
                }
            ],
            "implies": [
                {
                    "column": "title_ratings.numVotes",
                    "op": ">=",
                    "value": 50000
                }
            ],
        },
        # Complex rule with nested AND/OR
        {
            "id": "IMDB_R2",
            "requires": {
                "op": "AND",
                "conditions": [
                    {
                        "column": "title_basics.titleType",
                        "op": "=",
                        "value": "movie"
                    },
                    {
                        "op": "OR",
                        "conditions": [
                            {
                                "op": "AND",
                                "conditions": [
                                    {
                                        "column": "title_basics.startYear",
                                        "op": "<",
                                        "value": 1970
                                    },
                                    {
                                        "column": "title_ratings.averageRating",
                                        "op": ">=",
                                        "value": 8.0
                                    }
                                ]
                            },
                            {
                                "column": "title_ratings.averageRating",
                                "op": ">=",
                                "value": 9.0
                            }
                        ]
                    }
                ]
            },
            "implies": [
                {
                    "column": "title_basics.genres",
                    "op": "IN",
                    "value": ["Drama", "Film-Noir", "History"]
                }
            ],
        },
        # Another example: (highRating AND many votes) OR (perfect rating)
        {
            "id": "IMDB_R3",
            "requires": {
                "op": "OR",
                "conditions": [
                    {
                        "op": "AND",
                        "conditions": [
                            {
                                "column": "title_ratings.averageRating",
                                "op": ">=",
                                "value": 8.0
                            },
                            {
                                "column": "title_ratings.numVotes",
                                "op": ">=",
                                "value": 100000
                            }
                        ]
                    },
                    {
                        "column": "title_ratings.averageRating",
                        "op": ">=",
                        "value": 9.5
                    }
                ]
            },
            "implies": [
                {
                    "column": "title_basics.isPopular",
                    "op": "=",
                    "value": "true"
                }
            ],
        },
    ]

    preds, alias_map, query_structure = extract_predicates(sql)
    print(preds)
    expanded = apply_rules(preds, RULES, query_structure, alias_map)
    print(expanded)
    new_sql = rewrite_sql(sql, expanded, preds, alias_map)

    print(new_sql)
