"""Helpers for multi-sample LLM stages (generation & refinement).

Two concerns:
- ``inject_seed``: make each of N otherwise-identical requests unique so they cache
  separately. The LLM cache keys on the full request JSON
  (``llm_helpers.llms.compute_hash``); without a distinguishing field, N identical
  requests collide to a single cache entry. ``seed`` lives outside ``messages``, so
  the prompt the model reads is byte-identical across samples.
- ``rule_signature``: a canonical key for exact-match dedup of rules (same ``requires``
  AND ``implies``), used to drop redundant candidates before the expensive validation
  runs. Builds on :func:`normalise_node`, which is also reused by aggregation.
"""

from __future__ import annotations

import json
from typing import Any


def inject_seed(requests: list[dict], seed: int) -> list[dict]:
    """Set ``request["seed"] = seed`` on each request dict (in place) and return them.

    Used only when more than one sample is drawn; for a single sample the seed is
    omitted so requests stay byte-identical to earlier runs and keep hitting the cache.
    """
    for request in requests:
        request["seed"] = seed
    return requests


def normalise_node(node: Any) -> Any:
    """Normalise a requires/implies condition (sub)tree for canonical comparison.

    Upper-cases and strips operators and sorts ``conditions`` lists so that
    structurally equivalent trees produce identical output. Shared with
    ``aggregation._canonicalize_requires``.
    """
    if isinstance(node, dict):
        if "column" in node:
            # Leaf predicate
            return {
                "column": node["column"],
                "op": node.get("op", "").upper().strip(),
                "value": node.get("value"),
            }
        if "conditions" in node:
            children = [normalise_node(c) for c in node["conditions"]]
            children.sort(key=lambda c: json.dumps(c, sort_keys=True))
            return {
                "op": node.get("op", "").upper().strip(),
                "conditions": children,
            }
    return node


def rule_signature(rule: dict) -> str:
    """Canonical string over a rule's ``requires`` and ``implies`` for exact-match dedup.

    Two rules with the same signature inject the same predicates under the same
    conditions, so they produce identical rewritten SQL and are redundant to validate.
    """
    requires = normalise_node(rule.get("requires", {}))
    implies = sorted(
        (normalise_node(impl) for impl in rule.get("implies", []) or []),
        key=lambda c: json.dumps(c, sort_keys=True),
    )
    return json.dumps({"requires": requires, "implies": implies}, sort_keys=True)
