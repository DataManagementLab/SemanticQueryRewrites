"""Build per-rule evaluation records from an oracle experiment's transfer data.

Each record bundles the three things the judge needs — the rule itself, the
model's stated rationale, and the rule's individual oracle performance — for
every rule that passed validation (i.e. survived aggregation into
``rule_summary_result.json``).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from systematic_eval.parsing import parse_llm_json


# --------------------------------------------------------------------------- #
# Streaming reader for the (multi-GB) rule_summary_result.json
# --------------------------------------------------------------------------- #
def stream_top_level_items(path: Path, chunk_size: int = 8 << 20) -> Iterator[tuple[str, dict]]:
    """Yield ``(key, value)`` pairs of a top-level JSON object without loading
    the whole file into memory.

    ``rule_summary_result.json`` is a single JSON object keyed by query prefix,
    each value a self-contained dict. We decode one entry at a time with
    ``json.JSONDecoder.raw_decode`` over a sliding buffer, so peak memory is
    roughly one entry plus one chunk — not the full file.
    """
    dec = json.JSONDecoder()

    def _decode(buf: str, fh) -> tuple[object, str]:
        """Decode one JSON value at the start of *buf* (after whitespace),
        reading more from *fh* if the buffer holds only a partial value.
        Returns (value, remaining_buffer)."""
        while True:
            s = buf.lstrip()
            try:
                val, end = dec.raw_decode(s)
                return val, s[end:]
            except ValueError:
                more = fh.read(chunk_size)
                if not more:
                    raise
                buf = s + more

    with path.open("r", encoding="utf-8") as fh:
        buf = ""
        # Advance to the opening brace of the top-level object.
        while "{" not in buf:
            more = fh.read(chunk_size)
            if not more:
                return
            buf += more
        buf = buf[buf.index("{") + 1:]

        while True:
            # Skip separators/whitespace before the next key (or closing brace).
            while True:
                buf = buf.lstrip()
                if not buf:
                    more = fh.read(chunk_size)
                    if not more:
                        return
                    buf += more
                    continue
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                break
            if buf[0] == "}":
                return

            key, buf = _decode(buf, fh)
            buf = buf.lstrip()
            while not buf:
                more = fh.read(chunk_size)
                if not more:
                    raise ValueError("Unexpected EOF after key, expected ':'")
                buf += more
            assert buf[0] == ":", f"Expected ':' after key {key!r}, got {buf[:20]!r}"
            buf = buf[1:]
            value, buf = _decode(buf, fh)
            yield str(key), value  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Rationale index over result*.json
# --------------------------------------------------------------------------- #
_SUFFIX_RE = re.compile(r"(?:_rerun_s\d+|_no_change)$")


def build_rationale_index(transfer_dir: Path) -> dict[str, list[str]]:
    """Index every generation/refinement record's rationale bullets by key.

    Returns ``{result_key: short_rationale_bullets}``. ``short_rationale`` is a
    list of strings like ``"- R1: ..."``; kept as-is so the caller can pick the
    bullet matching a rule id.
    """
    index: dict[str, list[str]] = {}
    for result_file in sorted(transfer_dir.glob("result*.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            raw = entry.get("original_output")
            if not isinstance(raw, str):
                continue
            parsed = parse_llm_json(raw)
            if not isinstance(parsed, dict):
                continue
            rationale = parsed.get("short_rationale")
            bullets: list[str]
            if isinstance(rationale, list):
                bullets = [str(b) for b in rationale]
            elif isinstance(rationale, str):
                bullets = [rationale]
            else:
                continue
            index[key] = bullets
    return index


def _rule_number(rule_id: str) -> int | None:
    """Extract the trailing rule number from an id like ``IMDB_R3`` -> 3."""
    m = re.search(r"R(\d+)$", rule_id or "")
    return int(m.group(1)) if m else None


def _component_keys(rule_name: str) -> list[str]:
    """Split a (possibly merged) rule name into its source result*.json keys.

    ``"merged: A + B"`` -> ``["A", "B"]``; a plain name -> ``[name]``.
    """
    name = rule_name
    if name.startswith("merged:"):
        name = name[len("merged:"):]
    return [part.strip() for part in name.split(" + ") if part.strip()]


def rationale_for_rule(
    rule_name: str,
    rule_id: str,
    rationale_index: dict[str, list[str]],
) -> list[str]:
    """Collect the model's rationale bullet(s) for a (merged) rule.

    For each source component we look up its record and pick the bullet whose
    ``R<n>`` matches the rule id; failing that we fall back to the positional
    index encoded in the key (``..._<j>_IMDB_R..``), then to all bullets.
    """
    wanted = _rule_number(rule_id)
    collected: list[str] = []
    seen: set[str] = set()

    for comp in _component_keys(rule_name):
        bullets = rationale_index.get(comp)
        if bullets is None:
            stripped = _SUFFIX_RE.sub("", comp)
            bullets = rationale_index.get(stripped)
        if not bullets:
            continue

        chosen: list[str] = []
        if wanted is not None:
            pat = re.compile(rf"\bR0*{wanted}\b")
            chosen = [b for b in bullets if pat.search(b)]
        if not chosen:
            m = re.search(r"_(\d+)_IMDB_R", comp)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(bullets):
                    chosen = [bullets[idx]]
        if not chosen:
            chosen = bullets

        for b in chosen:
            if b not in seen:
                seen.add(b)
                collected.append(b)
    return collected


# --------------------------------------------------------------------------- #
# Per-rule record assembly
# --------------------------------------------------------------------------- #
@dataclass
class QueryPerf:
    query: str
    original_runtime: float
    improved_runtime: float
    percent_saved: float
    outputs_match: bool


@dataclass
class RuleRecord:
    rule_name: str
    rule: dict
    rationales: list[str] = field(default_factory=list)
    per_query: list[QueryPerf] = field(default_factory=list)

    @property
    def rule_id(self) -> str:
        return str(self.rule.get("id", ""))

    def agg(self) -> dict:
        """Aggregate individual performance across the queries the rule fired on.

        Only beneficial, output-matching single applications count toward the
        speedup aggregates; ``n_queries_fired`` counts every firing.
        """
        good = [p.percent_saved for p in self.per_query if p.outputs_match]
        beneficial = [p for p in good if p > 0]
        return {
            "n_queries_fired": len(self.per_query),
            "n_queries_beneficial": len(beneficial),
            "median_percent_saved": statistics.median(good) if good else None,
            "mean_percent_saved": statistics.fmean(good) if good else None,
            "max_percent_saved": max(good) if good else None,
        }

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "rule_id": self.rule_id,
            "rule": self.rule,
            "rationales": self.rationales,
            "per_query": [vars(p) for p in self.per_query],
            "agg": self.agg(),
        }


def _singleton_perf(entry: dict) -> dict[str, QueryPerf]:
    """Map ``rule_name -> QueryPerf`` from a query entry's singleton subsets."""
    out: dict[str, QueryPerf] = {}
    exec_time = (entry.get("summary") or {}).get("execution_time") or {}
    orig = exec_time.get("original_query")
    for subset in entry.get("per_subset_results") or []:
        names = subset.get("rule_names") or []
        if len(names) != 1:
            continue
        s_summary = subset.get("summary") or {}
        s_exec = s_summary.get("execution_time") or {}
        s_time = s_exec.get("subset_transformed_query")
        s_orig = s_exec.get("original_query", orig)
        if not isinstance(s_time, (int, float)) or not isinstance(s_orig, (int, float)) or s_orig <= 0:
            continue
        out[names[0]] = QueryPerf(
            query="",  # filled by caller
            original_runtime=float(s_orig),
            improved_runtime=float(s_time),
            percent_saved=(s_orig - s_time) / s_orig * 100.0,
            outputs_match=bool(s_summary.get("outputs_match", False)),
        )
    return out


def extract_rule_records(transfer_dir: Path) -> list[RuleRecord]:
    """Assemble one :class:`RuleRecord` per unique validated rule.

    A rule may fire on several queries; its per-query singleton performances are
    collected under a single record keyed by rule name.
    """
    result_path = transfer_dir / "rule_summary_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"No rule_summary_result.json in {transfer_dir}")

    rationale_index = build_rationale_index(transfer_dir)

    records: dict[str, RuleRecord] = {}
    for query_name, entry in stream_top_level_items(result_path):
        if not isinstance(entry, dict):
            continue
        perf_by_name = _singleton_perf(entry)
        for rule_entry in entry.get("rules") or []:
            name = rule_entry.get("name")
            rule = rule_entry.get("rule")
            if not name or not isinstance(rule, dict):
                continue
            rec = records.get(name)
            if rec is None:
                rec = RuleRecord(
                    rule_name=name,
                    rule=rule,
                    rationales=rationale_for_rule(name, str(rule.get("id", "")), rationale_index),
                )
                records[name] = rec
            perf = perf_by_name.get(name)
            if perf is not None:
                perf.query = query_name
                rec.per_query.append(perf)

    return list(records.values())
