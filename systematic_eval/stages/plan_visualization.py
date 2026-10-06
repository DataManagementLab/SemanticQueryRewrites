"""Generate side-by-side query plan comparison graphs from rule_summary_result.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import graphviz
from html import escape as _esc

# Above this many rule subsets for one query, only the oracle-picked and
# optimizer-picked subsets are plotted. The execution stage enumerates the full
# powerset of a query's fired rules (2^k - 1), so a 13-rule query would otherwise
# emit 8191 PNGs that nobody ever opens.
PLAN_COMPARISON_SUBSET_LIMIT = 50


def _format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.3f}s"


def _format_rows(n: int | float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def _node_label(node: dict) -> str:
    """Build a compact multi-line label for a plan node."""
    name = node.get("operator_name", "?").strip()
    timing = node.get("operator_timing", 0)
    cardinality = node.get("operator_cardinality", 0)
    rows_scanned = node.get("operator_rows_scanned", 0)
    extra = node.get("extra_info", {})

    lines = [f"<b>{_esc(name)}</b>"]

    # Key details from extra_info
    if "Table" in extra:
        lines.append(f"Table: {_esc(str(extra['Table']))}")
    if "Filters" in extra:
        filt = str(extra["Filters"])
        if len(filt) > 50:
            filt = filt[:47] + "..."
        lines.append(f"Filter: {_esc(filt)}")
    if "Conditions" in extra:
        cond = str(extra["Conditions"])
        if len(cond) > 50:
            cond = cond[:47] + "..."
        lines.append(f"Cond: {_esc(cond)}")
    if "Join Type" in extra:
        lines.append(f"Join: {_esc(str(extra['Join Type']))}")
    if "Expression" in extra:
        expr = str(extra["Expression"])
        if len(expr) > 50:
            expr = expr[:47] + "..."
        lines.append(f"Expr: {_esc(expr)}")

    if "Estimate" in extra:
        lines.append(f"Est: {_format_rows(extra['Estimate'])}")

    pid = node.get("pipeline_id")
    time_label = f"Time: {_format_time(timing)}"
    if pid is not None:
        time_label += f" (P{pid})"
    lines.append(time_label)
    lines.append(f"Rows: {_format_rows(cardinality)} (scanned: {_format_rows(rows_scanned)})")

    return "<" + "<br/>".join(lines) + ">"


def _node_color(node: dict, pipeline_colors: dict[int, str] | None = None) -> str:
    """Color nodes by pipeline membership (if available) or timing."""
    pid = node.get("pipeline_id")
    if pipeline_colors is not None and pid is not None:
        return pipeline_colors[pid]
    timing = node.get("operator_timing", 0)
    if timing > 0.1:
        return "#ff6b6b"  # red — hot
    if timing > 0.01:
        return "#ffa94d"  # orange
    if timing > 0.001:
        return "#ffd43b"  # yellow
    return "#d3f9d8"  # green — fast


# Palette for distinguishing pipelines — visually distinct, pastel-ish.
_PIPELINE_PALETTE = [
    "#a8d8ea",  # light blue
    "#f9c5d1",  # light pink
    "#c3e8bd",  # light green
    "#f5d5a0",  # light orange
    "#d4b8e0",  # light purple
    "#f9e79f",  # light yellow
    "#a0d2db",  # teal
    "#f0b8b8",  # salmon
    "#b8d4e3",  # steel blue
    "#d5e8d4",  # sage
    "#e8cfe8",  # lavender
    "#f5e0c3",  # peach
]


def _collect_pipeline_ids(node: dict, ids: set[int]) -> None:
    """Recursively collect all pipeline_id values from a plan tree."""
    pid = node.get("pipeline_id")
    if pid is not None:
        ids.add(pid)
    for child in node.get("children", []):
        _collect_pipeline_ids(child, ids)


def _build_pipeline_colors(*plans: dict) -> dict[int, str] | None:
    """Build a pipeline_id → color mapping from one or more plan trees.

    Returns None if no pipeline_id fields are present (non-Umbra plans).
    """
    ids: set[int] = set()
    for plan in plans:
        if plan:
            _collect_pipeline_ids(plan, ids)
    if not ids:
        return None
    sorted_ids = sorted(ids)
    return {pid: _PIPELINE_PALETTE[i % len(_PIPELINE_PALETTE)]
            for i, pid in enumerate(sorted_ids)}


def _normalize_condition(cond: str | list | None) -> str | None:
    """Normalize a join condition so that 'a = b' and 'b = a' compare equal."""
    if cond is None:
        return None
    # DuckDB may emit Conditions as a list of strings or a single string.
    if isinstance(cond, list):
        cond = " AND ".join(str(c) for c in cond)
    # Split on " AND ", normalize each part, then sort for stable order
    parts = [p.strip() for p in cond.split(" AND ")]
    normalized = []
    for part in parts:
        sides = part.split("=", 1)
        if len(sides) == 2:
            normalized.append(" = ".join(sorted(s.strip() for s in sides)))
        else:
            normalized.append(part)
    normalized.sort()
    return " AND ".join(normalized)


def _node_content_key(node: dict) -> tuple:
    """Extract content-relevant fields for comparison (ignoring timing/cardinality).

    Only compares operator type, table, filters, conditions, join type, and
    expression — NOT runtime stats like timing, cardinality, estimated
    cardinality, or rows scanned.
    """
    extra = node.get("extra_info", {})
    # Normalize Filters to a frozenset so order doesn't matter
    filters = extra.get("Filters")
    if isinstance(filters, list):
        filters = frozenset(filters)
    return (
        node.get("operator_name", "").strip(),
        extra.get("Table"),
        filters,
        _normalize_condition(extra.get("Conditions")),
        extra.get("Join Type"),
        extra.get("Expression"),
    )


def _sum_operator_timings(node: dict) -> float:
    """Sum operator_timing for a node and all its descendants."""
    total = node.get("operator_timing", 0)
    for child in node.get("children", []):
        total += _sum_operator_timings(child)
    return total


def _find_changed_nodes(
    original: dict | None,
    transformed: dict | None,
    counter: list[int],
    changed: set[int],
    unchanged_timings: list[tuple[float, float]] | None = None,
    unmatched_orig_timings: list[float] | None = None,
    changed_trans_timings: list[float] | None = None,
) -> None:
    """Walk the transformed tree top-down, matching each node to the original.

    For each transformed node, we check whether the original tree has a node
    with the same content at the same position in the tree (same parent
    relationship).  Children are matched by content key — not by index — so
    reordered children are handled correctly.

    Collects timing data into the provided lists:
    - *unchanged_timings*: (orig, trans) pairs for matched nodes
    - *changed_trans_timings*: operator_timing for changed/new transformed nodes
    - *unmatched_orig_timings*: operator_timing for original nodes with no match
    """
    if transformed is None:
        return
    idx = counter[0]
    counter[0] += 1

    is_match = original is not None and _node_content_key(original) == _node_content_key(transformed)

    if not is_match:
        changed.add(idx)
        if changed_trans_timings is not None:
            changed_trans_timings.append(transformed.get("operator_timing", 0))
    elif unchanged_timings is not None:
        unchanged_timings.append((
            original.get("operator_timing", 0),
            transformed.get("operator_timing", 0),
        ))

    orig_children = original.get("children", []) if original else []
    trans_children = transformed.get("children", [])

    # Match each transformed child to an original child by content key.
    # Greedy: first match wins; unmatched transformed children get None.
    used_orig: set[int] = set()
    for t_child in trans_children:
        t_key = _node_content_key(t_child)
        matched_orig = None
        for j, o_child in enumerate(orig_children):
            if j not in used_orig and _node_content_key(o_child) == t_key:
                matched_orig = o_child
                used_orig.add(j)
                break
        _find_changed_nodes(
            matched_orig, t_child, counter, changed,
            unchanged_timings, unmatched_orig_timings, changed_trans_timings,
        )

    # Collect timings for original children that had no match
    if unmatched_orig_timings is not None:
        for j, o_child in enumerate(orig_children):
            if j not in used_orig:
                unmatched_orig_timings.append(_sum_operator_timings(o_child))


def _add_plan_nodes(
    graph: graphviz.Digraph,
    node: dict,
    prefix: str,
    counter: list[int],
    highlight: set[int] | None = None,
    pipeline_colors: dict[int, str] | None = None,
) -> str:
    """Recursively add nodes to the graph. Returns the node id."""
    node_id = f"{prefix}_{counter[0]}"
    is_highlighted = highlight is not None and counter[0] in highlight
    counter[0] += 1

    attrs = {
        "label": _node_label(node),
        "fillcolor": _node_color(node, pipeline_colors),
        "shape": "box",
        "fontsize": "9",
        "fontname": "Helvetica",
    }
    if is_highlighted:
        attrs["style"] = "filled,bold"
        attrs["color"] = "#7b2ff7"  # purple border
        attrs["penwidth"] = "3"
    else:
        attrs["style"] = "filled"

    graph.node(node_id, **attrs)

    for child in node.get("children", []):
        child_id = _add_plan_nodes(graph, child, prefix, counter, highlight, pipeline_colors)
        graph.edge(child_id, node_id)

    return node_id


def _format_predicate(pred: dict) -> str:
    """Format a single implies predicate as a readable string."""
    col = pred.get("column", "?")
    op = pred.get("op", "?")
    val = pred.get("value", "?")
    if isinstance(val, list):
        val = "(" + ", ".join(str(v) for v in val) + ")"
    return f"{col} {op} {val}"


def render_plan_comparison(
    query_name: str,
    original_plan: dict,
    transformed_plan: dict,
    original_latency: float,
    transformed_latency: float,
    output_path: Path,
    applied_rules: list[dict] | None = None,
) -> None:
    """Render a side-by-side query plan comparison and save as PNG."""
    speedup = (original_latency - transformed_latency) / original_latency * 100 if original_latency > 0 else 0
    sign = "+" if speedup < 0 else ""

    # Build pipeline color mapping (for Umbra plans with pipeline_id fields)
    pipeline_colors = _build_pipeline_colors(original_plan, transformed_plan)

    g = graphviz.Digraph(
        name="plan_comparison",
        format="png",
        graph_attr={
            "rankdir": "BT",
            "label": (
                f"Query Plan Comparison: {query_name}\\n"
                f"Original: {_format_time(original_latency)} → "
                f"Transformed: {_format_time(transformed_latency)} "
                f"({sign}{speedup:.1f}% reduction)"
            ),
            "labelloc": "t",
            "fontsize": "14",
            "fontname": "Helvetica Bold",
            "compound": "true",
            "nodesep": "0.3",
            "ranksep": "0.4",
        },
    )

    # Original plan subgraph (left)
    with g.subgraph(name="cluster_original") as orig:
        orig.attr(
            label=f"Original Plan\\nLatency: {_format_time(original_latency)}",
            style="dashed",
            color="#4C78A8",
            fontcolor="#4C78A8",
            fontsize="12",
            fontname="Helvetica Bold",
        )
        _add_plan_nodes(orig, original_plan, "orig", [0], pipeline_colors=pipeline_colors)

    # Identify changed/new nodes and collect timing data
    changed_nodes: set[int] = set()
    unchanged_timings: list[tuple[float, float]] = []
    unmatched_orig_timings: list[float] = []
    changed_trans_timings: list[float] = []
    _find_changed_nodes(
        original_plan, transformed_plan, [0], changed_nodes,
        unchanged_timings, unmatched_orig_timings, changed_trans_timings,
    )

    # Runtime breakdown
    total_diff = transformed_latency - original_latency
    variance_diff = sum(t - o for o, t in unchanged_timings)
    change_diff = sum(changed_trans_timings) - sum(unmatched_orig_timings)

    # Transformed plan subgraph (right)
    with g.subgraph(name="cluster_transformed") as trans:
        trans.attr(
            label=f"Transformed Plan\\nLatency: {_format_time(transformed_latency)}",
            style="dashed",
            color="#F58518",
            fontcolor="#F58518",
            fontsize="12",
            fontname="Helvetica Bold",
        )
        _add_plan_nodes(trans, transformed_plan, "trans", [0], highlight=changed_nodes, pipeline_colors=pipeline_colors)

    # Add runtime breakdown as a bottom label
    def _signed(seconds: float) -> str:
        sign = "+" if seconds >= 0 else ""  # _format_time already has '-' for negatives
        return f"{sign}{_format_time(abs(seconds))}" if seconds >= 0 else f"-{_format_time(abs(seconds))}"

    g.attr(
        label=(
            f"Query Plan Comparison: {query_name}\\n"
            f"Original: {_format_time(original_latency)} → "
            f"Transformed: {_format_time(transformed_latency)} "
            f"({sign}{speedup:.1f}% reduction)\\n"
            f"Total diff: {_signed(total_diff)}  |  "
            f"Exec. variance (unchanged nodes): {_signed(variance_diff)}  |  "
            f"Changed nodes impact: {_signed(change_diff)}"
        ),
    )

    # Applied predicates panel (right side)
    if applied_rules:
        with g.subgraph(name="cluster_predicates") as preds:
            preds.attr(
                label="Applied Predicates",
                style="dashed",
                color="#7b2ff7",
                fontcolor="#7b2ff7",
                fontsize="12",
                fontname="Helvetica Bold",
            )
            for i, rule in enumerate(applied_rules):
                rule_obj = rule.get("rule", rule)
                rule_id = rule_obj.get("id", f"Rule {i + 1}")
                implies = rule_obj.get("implies", [])
                lines = [f"<b>{_esc(rule_id)}</b>"]
                for pred in implies:
                    lines.append(_esc(_format_predicate(pred)))
                label = "<" + "<br/>".join(lines) + ">"
                preds.node(
                    f"pred_{i}",
                    label=label,
                    style="filled",
                    fillcolor="#e8d5f5",
                    shape="box",
                    fontsize="9",
                    fontname="Helvetica",
                )
                if i > 0:
                    preds.edge(f"pred_{i - 1}", f"pred_{i}", style="invis")

    # Pipeline legend (for Umbra plans where nodes are colored by pipeline)
    if pipeline_colors:
        with g.subgraph(name="cluster_pipelines") as leg:
            leg.attr(
                label="Pipelines",
                style="dashed",
                color="#666666",
                fontcolor="#666666",
                fontsize="12",
                fontname="Helvetica Bold",
            )
            sorted_pids = sorted(pipeline_colors)
            for pid in sorted_pids:
                leg.node(
                    f"legend_p{pid}",
                    label=f"P{pid}",
                    style="filled",
                    fillcolor=pipeline_colors[pid],
                    shape="box",
                    fontsize="9",
                    fontname="Helvetica",
                    width="0.4",
                    height="0.25",
                )
            # invisible edges to keep legend items in order
            for i in range(len(sorted_pids) - 1):
                leg.edge(f"legend_p{sorted_pids[i]}", f"legend_p{sorted_pids[i + 1]}", style="invis")

    # Render to file (graphviz appends the format extension)
    output_stem = str(output_path).removesuffix(".png")
    g.render(output_stem, cleanup=True)


def _select_pick_subsets(
    per_subset_results: list[dict],
    picks: dict[str, list[str]],
) -> list[tuple[str, dict]]:
    """Pick out the oracle-chosen and optimizer-chosen subsets, labelled.

    Returns (label, subset_entry) pairs — oracle first, at most two. When both
    views picked the same subset it is returned once, labelled "oracle+optimizer".
    A pick that is absent or has no matching subset is silently left out, so the
    result may be empty.
    """
    by_key = {
        tuple(sorted(e.get("rule_names", []) or [])): e
        for e in per_subset_results
    }

    labels_by_key: dict[tuple[str, ...], list[str]] = {}
    for view in ("oracle", "optimizer"):
        names = picks.get(view)
        if not names:
            continue
        key = tuple(sorted(names))
        if key not in by_key:
            continue
        labels_by_key.setdefault(key, []).append(view)

    return [("+".join(labels), by_key[key]) for key, labels in labels_by_key.items()]


def generate_plan_comparisons(
    rule_summary_path: Path,
    output_dir: Path,
    picks: dict[str, dict[str, list[str]]] | None = None,
    full: bool = False,
    subset_limit: int = PLAN_COMPARISON_SUBSET_LIMIT,
) -> int:
    """Generate plan comparison images for all queries with applied rules.

    By default a query with more than *subset_limit* rule subsets is reduced to
    its oracle-picked and optimizer-picked subsets (at most two images), since
    plotting the full powerset costs a lot of storage for plots nobody reads.
    *picks* maps query name -> {"oracle": [rule names], "optimizer": [rule names]};
    either view may be missing. Pass full=True (or omit *picks*) to plot every
    subset regardless.

    Returns the number of images generated.
    """
    summary = json.loads(rule_summary_path.read_text(encoding="utf-8"))

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    count = 0
    reduced_queries = 0
    for query_name, entry in summary.items():
        if not entry.get("rule_applied", False):
            continue

        queries = entry.get("results", {}).get("queries", {})
        original = queries.get("original_query", {})
        transformed = queries.get("llm_transformed_query", {})

        original_plan = original.get("query_plan")
        transformed_plan = transformed.get("query_plan")

        if not original_plan or not transformed_plan:
            continue

        original_latency = original_plan.get("latency", 0)
        transformed_latency = transformed_plan.get("latency", 0)

        safe_name = query_name.replace(" ", "_").replace("/", "_")
        applied_rules = entry.get("rules", [])

        per_subset_results = entry.get("per_subset_results", [])

        if per_subset_results:
            if not full and picks is not None and len(per_subset_results) > subset_limit:
                selected = _select_pick_subsets(
                    per_subset_results, picks.get(query_name, {}))
                reduced_queries += 1
            else:
                selected = [(None, e) for e in per_subset_results]

            if not selected:
                continue

            # One folder per query, one plot per selected rule subset
            query_dir = output_dir / safe_name
            query_dir.mkdir(parents=True, exist_ok=True)

            for label, subset_entry in selected:
                rule_names = subset_entry["rule_names"]
                subset_queries = subset_entry.get("results", {}).get("queries", {})
                subset_orig_plan = subset_queries.get("original_query", {}).get("query_plan")
                subset_trans_plan = subset_queries.get("subset_transformed_query", {}).get("query_plan")
                if not subset_orig_plan or not subset_trans_plan:
                    continue

                matching_rules = [r for r in applied_rules if r.get("name") in rule_names]
                safe_rule_part = "+".join(
                    n.replace(" ", "_").replace("/", "_").replace(":", "_")
                    for n in rule_names
                )
                if len(safe_rule_part) > 200:
                    # Truncating to a fixed prefix collides distinct subsets that
                    # share their first 200 chars, silently overwriting plots.
                    # Append a hash of the full (untruncated) rule list so every
                    # subset maps to a unique file. 191 + 2 + 8 = 201 chars.
                    digest = hashlib.sha1(
                        "+".join(rule_names).encode()
                    ).hexdigest()[:8]
                    safe_rule_part = f"{safe_rule_part[:191]}__{digest}"

                file_stem = safe_rule_part if label is None else f"{label}__{safe_rule_part}"
                title = f"{query_name} [{', '.join(rule_names)}]"
                if label is not None:
                    title = f"{title} — {label} pick"

                render_plan_comparison(
                    query_name=title,
                    original_plan=subset_orig_plan,
                    transformed_plan=subset_trans_plan,
                    original_latency=subset_orig_plan.get("latency", 0),
                    transformed_latency=subset_trans_plan.get("latency", 0),
                    output_path=query_dir / f"{file_stem}.png",
                    applied_rules=matching_rules,
                )
                count += 1

        else:
            # Backward compat: old JSON without per_subset_results → single file
            output_path = output_dir / f"plan_{safe_name}.png"
            render_plan_comparison(
                query_name=query_name,
                original_plan=original_plan,
                transformed_plan=transformed_plan,
                original_latency=original_latency,
                transformed_latency=transformed_latency,
                output_path=output_path,
                applied_rules=applied_rules,
            )
            count += 1

    if reduced_queries:
        print(f"Plan comparisons: reduced {reduced_queries} query/queries with more "
              f"than {subset_limit} rule subsets to their oracle/optimizer picks "
              f"(use --full-plan-comparisons to plot every subset).")

    return count
