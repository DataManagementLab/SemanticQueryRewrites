"""Cross-dataset performance overview for the semantic-SQL-rewrite experiments.

Reads the per-dataset statistics already produced by stage 4 (``stages/statistics.py``)
and condenses them into a single human-readable report that answers, at a glance:

  * Does the approach generalize?        -> coverage (queries with a rule / workload)
  * How big is the realistic win?         -> optimizer view (cost-picked subset, incl. slowdowns)
  * How big is the best-case win?          -> oracle view (best measured subset = upper bound)
  * *Why* does it work?                    -> the 2-3 most impactful rules, rendered as IF/THEN

Exactly one run per dataset is included, plus exactly one IMDB/JOB run as the IMDB
sample; every other IMDB constellation (_pg, _umbra, _join_p, _within_query,
_zeroshot, ...) and every ``*_complex_*`` run is ignored.

Four run families are selectable via ``--variant`` (each writes its own set of files
so they coexist):
  * ``thesis`` (default) — the current ``experiment_T_gen_<dataset>`` generality runs
    (DuckDB, free join order), IMDB sample ``experiment_T_imdb_job_12_oracle_c07_4-2``.
    Files: ``performance_overview_T_gen*``.
  * ``flexible`` — legacy base ``experiment_<name>_oracle`` runs, DB optimizer
    re-orders joins freely after the rewrite. Files: ``performance_overview*``.
  * ``fixed`` — legacy ``experiment_<name>_oracle_join_p`` runs (``fix_join_order:
    true``, joins replayed in the original order). Files: ``performance_overview_join_p*``.
  * ``umbra`` — legacy ``experiment_<name>_oracle_umbra`` runs (Umbra backend instead
    of DuckDB, flexible join order). Files: ``performance_overview_umbra*``.

Outputs (under ``saved_results/`` by default; ``<stem>`` per --variant above):
  * <stem>.md   - the report (summary table + per-dataset rules)
  * <stem>.png  - cross-dataset grouped bar chart (% workload runtime saved)
  * <stem>_relevant.png - the shared-denominator ("relevant") view
  * <stem>_oracle.png - oracle upper bound only, geo-mean speedup per dataset
                        (bars annotated with coverage)
  * <stem>_oracle_workload.png - oracle speedup over the WHOLE query set
                        (untouched queries counted at 1.00×)

Usage:
  python3 scripts/performance_overview.py             # from systematic_eval/ (thesis)
  python3 scripts/performance_overview.py --variant flexible  # legacy *_oracle runs
  python3 scripts/performance_overview.py --variant fixed     # legacy _join_p runs
  python3 scripts/performance_overview.py --variant umbra     # legacy _umbra runs
  python3 scripts/performance_overview.py --results-dir saved_results --out-dir saved_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IMDB is represented by exactly one run per variant (``canonical_imdb`` below);
# every other imdb_* / *_complex_* directory is excluded from the overview.
RULE_ID_RE = re.compile(r"[A-Z][A-Z0-9_]*_R\d+")

# Run families. "thesis" is the default: the current ``experiment_T_gen_<dataset>``
# generality runs (DuckDB, free join order) with the thesis' primary IMDB run as the
# IMDB sample. The other three select the older run generation: "flexible" = base
# ``experiment_<name>_oracle``, "fixed" = the ``_join_p`` runs (``fix_join_order:
# true`` — refined queries are replayed with the original query's join order),
# "umbra" = the Umbra-backend runs. Each variant selects a different set of
# experiment directories, its own canonical IMDB run, and its own output filenames
# so the overviews sit side by side without overwriting each other.
#
# The canonical IMDB run does NOT have to match the glob — ``discover_experiments``
# adds it explicitly (the thesis variant's IMDB run is named nothing like
# ``experiment_T_gen_*``).
VARIANTS = {
    "thesis": {
        "glob": "experiment_T_gen_*",
        "canonical_imdb": "experiment_T_imdb_job_12_oracle_c07_4-2",
        "out_stem": "performance_overview_T_gen",
        "title_suffix": "",
    },
    "flexible": {
        "glob": "experiment_*_oracle",
        "canonical_imdb": "experiment_imdb_job_12_oracle",
        "out_stem": "performance_overview",
        "title_suffix": "",
    },
    "fixed": {
        "glob": "experiment_*_oracle_join_p",
        "canonical_imdb": "experiment_imdb_job_12_oracle_join_p",
        "out_stem": "performance_overview_join_p",
        "title_suffix": " (fixed join order)",
    },
    "umbra": {
        "glob": "experiment_*_oracle_umbra",
        "canonical_imdb": "experiment_imdb_job_12_oracle_umbra",
        "out_stem": "performance_overview_umbra",
        "title_suffix": " (Umbra backend)",
    },
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ViewStats:
    """Aggregate metrics for one view (oracle or optimizer) of one dataset."""

    n_rows: int = 0                  # queries touched by this view
    n_improved: int = 0              # rows that got faster (speedup > 1)
    n_slowdown: int = 0             # rows that got slower (speedup < 1)
    total_orig: float = 0.0
    total_impr: float = 0.0
    geo_mean: float | None = None    # geometric mean of per-query speedup factors
    best_prefix: str = ""
    best_speedup: float = 1.0

    @property
    def pct_saved(self) -> float | None:
        """Runtime-weighted % of total workload time saved (the primary metric)."""
        if self.total_orig <= 0:
            return None
        return (self.total_orig - self.total_impr) / self.total_orig * 100.0

    @property
    def speedup_factor(self) -> float | None:
        if self.total_impr <= 0:
            return None
        return self.total_orig / self.total_impr


@dataclass
class DatasetStats:
    name: str                        # short label, e.g. "baseball", "imdb_job"
    experiment: str                  # full dir name, e.g. "experiment_baseball_oracle"
    test_size: int | None
    oracle: ViewStats
    optimizer: ViewStats
    top_rules: list[dict] = field(default_factory=list)  # rendered impactful rules
    # "Relevant" view: both % values computed over the shared union denominator.
    oracle_pct_relevant: float | None = None
    optimizer_pct_relevant: float | None = None
    relevant_size: int = 0           # |union of oracle- and optimizer-touched queries|
    # Whole-workload oracle view: untouched queries weigh in unchanged.
    oracle_workload_speedup: float | None = None   # Σ orig / Σ improved over ALL queries
    oracle_workload_pct: float | None = None
    workload_size: int = 0           # queries with a known original runtime


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def geometric_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _view_from_rows(rows: list[dict]) -> ViewStats:
    v = ViewStats()
    factors: list[float] = []
    for row in rows:
        try:
            orig = float(row["original_runtime"])
            impr = float(row["improved_runtime"])
        except (KeyError, ValueError):
            continue
        if orig <= 0 or impr <= 0:
            continue
        v.n_rows += 1
        v.total_orig += orig
        v.total_impr += impr
        speedup = orig / impr
        factors.append(speedup)
        if speedup > 1.0:
            v.n_improved += 1
        elif speedup < 1.0:
            v.n_slowdown += 1
        if speedup > v.best_speedup:
            v.best_speedup = speedup
            v.best_prefix = row.get("prefix", "")
    v.geo_mean = geometric_mean(factors)
    return v


def _relevant_pct(oracle_rows: list[dict],
                  optimizer_rows: list[dict]) -> tuple[float | None, float | None, int]:
    """Runtime-weighted % saved for each view over the SHARED union denominator.

    The universe is the union of queries touched by either view (matched on the
    leading token of ``prefix``, like stages/statistics.py's relevant plots). A
    query's original runtime is shared across views; a query absent from a view
    contributes ``improved = original`` (0% saved) there. Both percentages use the
    same Σ original over the union, so oracle and optimizer are directly comparable.

    Returns ``(oracle_pct, optimizer_pct, union_size)``.
    """
    def _base(prefix: str) -> str:
        return (prefix or "").split(" ", 1)[0]

    def _collect(rows):
        orig, impr = {}, {}
        for row in rows:
            try:
                o = float(row["original_runtime"])
                i = float(row["improved_runtime"])
            except (KeyError, ValueError):
                continue
            if o <= 0 or i <= 0:
                continue
            b = _base(row.get("prefix", ""))
            orig[b] = o
            impr[b] = i
        return orig, impr

    o_orig, o_impr = _collect(oracle_rows)
    p_orig, p_impr = _collect(optimizer_rows)

    # Union original runtime: prefer the oracle's baseline where both exist (same
    # query -> same original, modulo measurement noise), matching the setdefault
    # convention in stages/statistics.py.
    orig = dict(p_orig)
    orig.update(o_orig)
    union = sorted(orig)
    if not union:
        return None, None, 0

    total_orig = sum(orig[b] for b in union)
    if total_orig <= 0:
        return None, None, len(union)
    total_oracle = sum(o_impr.get(b, orig[b]) for b in union)
    total_optim = sum(p_impr.get(b, orig[b]) for b in union)
    oracle_pct = (total_orig - total_oracle) / total_orig * 100.0
    optim_pct = (total_orig - total_optim) / total_orig * 100.0
    return oracle_pct, optim_pct, len(union)


# --------------------------------------------------------------------------- #
# Rule rendering (IF requires THEN implies)
# --------------------------------------------------------------------------- #
def _fmt_value(value) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _render_requires(node) -> str:
    """Render the recursive AND/OR requires-tree into a readable predicate string."""
    if not isinstance(node, dict):
        return str(node)
    op = node.get("op")
    if op in ("AND", "OR") and "conditions" in node:
        parts = [_render_requires(c) for c in node["conditions"]]
        joiner = f" {op} "
        if len(parts) == 1:
            return parts[0]
        rendered = joiner.join(parts)
        return f"({rendered})" if op == "OR" else rendered
    col = node.get("column", "?")
    return f"{col} {node.get('op', '?')} {_fmt_value(node.get('value'))}"


def _render_implies(implies: list[dict]) -> str:
    """Render implied predicates, collapsing redundant bounds per column.

    The generator often emits several overlapping bounds for the same column
    (e.g. ``yearID >= 1871`` and ``yearID >= 1914``). Keep only the tightest:
    the max lower bound and the min upper bound, rendered as ``lo <= col <= hi``.
    """
    by_col: dict[str, dict] = {}
    order: list[str] = []
    for pred in implies or []:
        col = pred.get("column", "?")
        if col not in by_col:
            by_col[col] = {"eq": None, "lo": None, "lo_op": ">=", "hi": None, "hi_op": "<=", "other": []}
            order.append(col)
        slot = by_col[col]
        op, val = pred.get("op"), pred.get("value")
        num = isinstance(val, (int, float))
        if op == "=":
            slot["eq"] = val
        elif op in (">=", ">") and num:
            if slot["lo"] is None or val > slot["lo"]:
                slot["lo"], slot["lo_op"] = val, op
        elif op in ("<=", "<") and num:
            if slot["hi"] is None or val < slot["hi"]:
                slot["hi"], slot["hi_op"] = val, op
        else:
            slot["other"].append(f"{col} {op} {_fmt_value(val)}")

    pieces: list[str] = []
    for col in order:
        slot = by_col[col]
        if slot["eq"] is not None:
            pieces.append(f"{col} = {_fmt_value(slot['eq'])}")
        elif slot["lo"] is not None and slot["hi"] is not None:
            lo_sym = "≤" if slot["lo_op"] == ">=" else "<"
            hi_sym = "≤" if slot["hi_op"] == "<=" else "<"
            pieces.append(f"{_fmt_value(slot['lo'])} {lo_sym} {col} {hi_sym} {_fmt_value(slot['hi'])}")
        elif slot["lo"] is not None:
            pieces.append(f"{col} {slot['lo_op']} {_fmt_value(slot['lo'])}")
        elif slot["hi"] is not None:
            pieces.append(f"{col} {slot['hi_op']} {_fmt_value(slot['hi'])}")
        pieces.extend(slot["other"])
    return " AND ".join(pieces)


def _render_rule(rule: dict) -> str:
    req = _render_requires(rule.get("requires", {}))
    impl = _render_implies(rule.get("implies", []))
    text = f"IF {req} THEN {impl}"
    joins = rule.get("joins") or []
    if joins:
        jtxt = ", ".join(f"{j.get('left')} = {j.get('right')}" for j in joins)
        text += f"  [join: {jtxt}]"
    return text


def _rule_id_from_name(name: str) -> str:
    m = RULE_ID_RE.search(name or "")
    return m.group(0) if m else (name or "?")


# --------------------------------------------------------------------------- #
# LLM rationale extraction (transfer.json -> original_output.short_rationale)
# --------------------------------------------------------------------------- #
# The generator's raw output for each query/attempt is stored as a JSON string in
# transfer.json under ``original_output``; it carries ``determined_ruleset`` and a
# ``short_rationale`` list (one free-text explanation per rule, sometimes preceded
# by 1-2 preamble items). Each winning-rule name (e.g. "q19_r3_2_CREDIT_R3") is a
# direct key into transfer.json, so we resolve the exact generating attempt and
# then pick the matching rationale line.

def _strip_merged(token: str) -> str:
    """First component of a 'merged: a + b + ...' name, else the token itself."""
    token = token.strip()
    if token.lower().startswith("merged:"):
        token = token.split(":", 1)[1]
        token = token.split("+", 1)[0]
    return token.strip()


def _parse_token(token: str) -> tuple[str, str | None, int | None, int | None, str]:
    """Decompose a winning-rule name into (clean, query, gen, rule_index, rule_id).

    Two name schemes occur:
      * ``<query>_r<gen>_<idx>_<ID...>``  (most datasets; ID may be long/descriptive)
      * ``<query>_<idx>_<ID...>``         (e.g. basketball; no generation index)
    ``query`` may itself contain spaces/underscores (e.g. "6b NoMIN").
    """
    t = _strip_merged(token)
    rid = _rule_id_from_name(t)
    m = re.match(r"^(.*?)_r(\d+)_(\d+)_", t)
    if m:
        return t, m.group(1), int(m.group(2)), int(m.group(3)), rid
    m = re.match(r"^(.*?)_(\d+)_[A-Za-z]", t)
    if m:
        return t, m.group(1), None, int(m.group(2)), rid
    return t, None, None, None, rid


def _clean_rationale(text: str) -> str:
    """Strip only a leading markdown bullet; otherwise keep the LLM text verbatim."""
    return re.sub(r"^[\s]*[-*]\s*", "", text).strip()


def _norm_implies(rule) -> frozenset:
    """Normalized set of a rule's implied predicates, for content matching."""
    if not isinstance(rule, dict):
        return frozenset()
    out = set()
    for p in (rule.get("implies") or []):
        if isinstance(p, dict):
            out.add((p.get("column"), p.get("op"),
                     json.dumps(p.get("value"), sort_keys=True)))
    return frozenset(out)


def _select_rationale(short_rationale, ruleset_ids: list[str], rid: str,
                      rule_index: int | None) -> str | None:
    """Pick the rationale line for rule ``rid`` from a generation's short_rationale.

    Tries, in order: a unique full-id mention; a unique line *anchored* on a rule
    reference (``Rule CREDIT_R3`` / ``R3`` / ``Rule 3``); then index alignment of
    the trailing per-rule items (skipping leading preamble) using either the
    ruleset position of ``rid`` or the index encoded in the rule name.
    """
    if not isinstance(short_rationale, list):
        return None
    items = [s for s in short_rationale if isinstance(s, str)]
    if not items:
        return None

    rnum_m = re.search(r"R(\d+)$", rid or "")
    rnum = rnum_m.group(1) if rnum_m else None

    # 1) Unique full-id mention anywhere in the line.
    if rid:
        full = [s for s in items if re.search(r"\b" + re.escape(rid) + r"\b", s)]
        if len(full) == 1:
            return full[0]

    # 2) Unique line anchored on a rule reference for this rule.
    if rnum is not None:
        pats = [
            r"^[\s]*[-*]?\s*(Rule\s+)?" + re.escape(rid) + r"\b",
            r"^[\s]*[-*]?\s*(Rule\s+)?R" + rnum + r"\b",
            r"^[\s]*[-*]?\s*Rule\s+" + rnum + r"\b",
        ]
        anchored = [s for s in items if any(re.match(p, s) for p in pats)]
        if len(anchored) == 1:
            return anchored[0]

    # 3) Index alignment: keep only lines that *start* with a rule reference
    # (dropping preamble like "Likely intent:" and postamble like "No other
    # heuristic applies"), then index into them in ruleset order. Far more robust
    # than a trailing window when filler lines bracket the per-rule explanations.
    # A per-rule line starts with "Rule ..." or a bare "R<n>" token; preamble
    # ("Likely intent:", "Highest cost...") and postamble ("No other heuristic...")
    # do not. Kept deliberately loose at the start so long descriptive ids
    # (e.g. "ACCIDENTS_R1_ADMIN_AREA_CARRY") still register.
    rule_line = re.compile(r"^[\s]*[-*]?\s*(Rule\b|R\d)")
    rule_lines = [s for s in items if rule_line.match(s)]
    idx = None
    if ruleset_ids and rid in ruleset_ids:
        idx = ruleset_ids.index(rid)
    elif rule_index is not None:
        idx = rule_index
    if idx is not None and rule_lines and 0 <= idx < len(rule_lines):
        # Only trust positional mapping when the rule-line count lines up with the
        # ruleset (or with the index we need); otherwise stay silent.
        if not ruleset_ids or len(rule_lines) == len(ruleset_ids):
            return rule_lines[idx]
    return None


class RationaleIndex:
    """Lazily loads a dataset's transfer.json and resolves per-rule rationales."""

    def __init__(self, transfer_dir: Path) -> None:
        self._path = transfer_dir / "transfer.json"
        self._entries: dict | None = None

    def _load(self) -> dict:
        if self._entries is None:
            try:
                self._entries = json.loads(self._path.read_text(encoding="utf-8")) \
                    if self._path.exists() else {}
            except (json.JSONDecodeError, OSError):
                self._entries = {}
        return self._entries

    @staticmethod
    def _parse_original_output(entry: dict):
        oo = entry.get("original_output")
        if isinstance(oo, dict):
            return oo
        if not isinstance(oo, str):
            return None
        s = oo.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*", "", s).strip().rstrip("`").strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _resolve_key(self, clean: str, query: str | None, rule_def) -> str | None:
        """Find the transfer.json key for a winning-rule name, *with confidence*.

        Only two attributions are trusted, to avoid showing a rationale that
        belongs to a different rule:
          1. The name is itself a transfer.json key (exact generation+rule).
          2. Some entry of the same query has a ``response`` (the single generated
             rule) whose implied predicates match the displayed rule exactly.
        Anything else (e.g. a renamed/refined rule with no clean provenance, like
        basketball's no-generation names) returns None and shows no rationale.
        """
        entries = self._load()
        if clean in entries:
            return clean
        if not query:
            return None
        want = _norm_implies(rule_def)
        if not want:
            return None
        for k, e in entries.items():
            if not isinstance(e, dict) or not k.startswith(query + "_"):
                continue
            if _norm_implies(e.get("response")) == want:
                return k
        return None

    def rationale_for(self, token: str, rule_def=None) -> str | None:
        clean, query, _gen, _idx, _rid = _parse_token(token)
        key = self._resolve_key(clean, query, rule_def)
        if key is None:
            return None
        entry = self._load().get(key)
        if not isinstance(entry, dict):
            return None
        oo = self._parse_original_output(entry)
        if oo is None:
            return None
        # Use the resolved key (authoritative name) for the rule id and index.
        _c, _q, _g, kidx, krid = _parse_token(key)
        ruleset = oo.get("determined_ruleset") or oo.get("ruleset") or []
        ruleset_ids = [r.get("id") for r in ruleset if isinstance(r, dict)]
        text = _select_rationale(oo.get("short_rationale"), ruleset_ids, krid, kidx)
        return _clean_rationale(text) if text else None


def collect_top_rules(oracle_rows: list[dict], name_to_rule: dict[str, dict],
                      limit: int = 3) -> list[dict]:
    """Pick the most impactful distinct rules from the oracle view.

    Ranks queries by % saved (best first); for each query resolves its winning
    rule subset(s) back to rule definitions, and keeps the first ``limit``
    *distinct* rule ids encountered. Also counts how many oracle queries each
    rule id fired on.
    """
    # Frequency: how many oracle-beneficial queries reference each rule id.
    freq: dict[str, int] = {}
    for row in oracle_rows:
        ids = set()
        for token in (row.get("winning_rules") or "").split(";"):
            ids.add(_rule_id_from_name(token.strip()))
        for rid in ids:
            freq[rid] = freq.get(rid, 0) + 1

    ranked = sorted(oracle_rows, key=lambda r: float(r.get("percent_saved", 0)), reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for row in ranked:
        try:
            orig = float(row["original_runtime"])
            impr = float(row["improved_runtime"])
            pct = float(row["percent_saved"])
        except (KeyError, ValueError):
            continue
        speedup = orig / impr if impr > 0 else None
        for token in (row.get("winning_rules") or "").split(";"):
            token = token.strip()
            rid = _rule_id_from_name(token)
            if rid in seen:
                continue
            rule_def = name_to_rule.get(token)
            rendered = _render_rule(rule_def) if rule_def else "(rule definition unavailable)"
            seen.add(rid)
            out.append({
                "id": rid,
                "token": token,            # raw winning-rule name, for rationale lookup
                "rendered": rendered,
                "example_prefix": row.get("prefix", ""),
                "example_speedup": speedup,
                "example_pct": pct,
                "fired_on": freq.get(rid, 1),
                "rationale": None,         # filled in by load_dataset
            })
            if len(out) >= limit:
                return out
    return out


# --------------------------------------------------------------------------- #
# Per-dataset loading
# --------------------------------------------------------------------------- #
def _short_name(experiment: str) -> str:
    name = experiment
    if name.startswith("experiment_"):
        name = name[len("experiment_"):]
    if name.startswith("T_gen_"):       # thesis generality runs
        name = name[len("T_gen_"):]
    elif name.startswith("T_"):         # other thesis runs (e.g. the IMDB sample)
        name = name[len("T_"):]
    if name.endswith("_umbra"):         # Umbra-backend variant suffix
        name = name[: -len("_umbra")]
    if name.endswith("_join_p"):        # fixed-join-order variant suffix
        name = name[: -len("_join_p")]
    if name.endswith("_oracle"):
        name = name[: -len("_oracle")]
    if name.startswith("imdb_job"):
        name = "imdb_job"
    return name


def _load_test_size(transfer_dir: Path) -> int | None:
    path = transfer_dir / "test_size.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        ts = data.get("test_size")
        return int(ts) if isinstance(ts, (int, float)) else None
    if isinstance(data, (int, float)):
        return int(data)
    return None


def _build_name_to_rule(results_dir: Path, transfer_dir: Path) -> dict[str, dict]:
    """Map every rule *name* to its rule definition, across all queries."""
    for base in (results_dir, transfer_dir):
        path = base / "rule_summary_result.json"
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            mapping: dict[str, dict] = {}
            for entry in summary.values():
                for r in (entry.get("rules") or []):
                    if "name" in r and isinstance(r.get("rule"), dict):
                        mapping[r["name"]] = r["rule"]
            return mapping
    return {}


def _workload_original_runtimes(results_dir: Path, transfer_dir: Path) -> dict[str, float]:
    """Original runtime per query over the WHOLE workload, keyed by prefix token.

    Mirrors the two disjoint sources ``stages/statistics.py`` uses to draw its
    whole-workload line: every measured query's ``original_query`` from
    ``rule_summary_result.json``, plus the *no-rule* queries measured separately
    into ``baseline_runtimes.json`` (which wins on the rare overlap). Together
    these cover the full ``query_limit`` workload, including queries no rule
    ever fired on.
    """
    orig: dict[str, float] = {}
    for base in (results_dir, transfer_dir):
        path = base / "rule_summary_result.json"
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key, entry in summary.items():
            if not isinstance(entry, dict):
                continue
            et = (entry.get("summary") or {}).get("execution_time") or {}
            ov = et.get("original_query")
            if isinstance(ov, (int, float)) and ov > 0:
                orig.setdefault(key.split(" ", 1)[0], float(ov))
        break

    for base in (results_dir, transfer_dir):
        path = base / "baseline_runtimes.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, entry in raw.items():
            rt = entry.get("original_query") if isinstance(entry, dict) else entry
            if isinstance(rt, (int, float)) and rt > 0:
                orig[name] = float(rt)   # freshly measured, beats a stale summary value
        break
    return orig


def _workload_view(rows: list[dict],
                   workload_orig: dict[str, float]) -> tuple[float | None, float | None, int]:
    """Speedup factor and % saved of one view over the *whole* workload.

    Queries the view changed contribute their measured original/improved runtimes;
    every other query of the workload contributes its original runtime to BOTH
    totals — pure denominator weight that dilutes the gain without changing the
    absolute time saved. Returns ``(factor, pct_saved, workload_size)``.
    """
    total_orig = 0.0
    total_impr = 0.0
    shown: set[str] = set()
    for row in rows:
        try:
            o = float(row["original_runtime"])
            i = float(row["improved_runtime"])
        except (KeyError, ValueError):
            continue
        if o <= 0 or i <= 0:
            continue
        shown.add((row.get("prefix") or "").split(" ", 1)[0])
        total_orig += o
        total_impr += i

    n_total = len(shown)
    for name, rt in workload_orig.items():
        if name in shown or rt <= 0:
            continue
        total_orig += rt
        total_impr += rt
        n_total += 1

    if total_orig <= 0 or total_impr <= 0:
        return None, None, n_total
    return (total_orig / total_impr,
            (total_orig - total_impr) / total_orig * 100.0,
            n_total)


def load_dataset(experiment: str, results_root: Path, transfer_root: Path) -> DatasetStats | None:
    res_dir = results_root / experiment
    trans_dir = transfer_root / experiment
    oracle_csv = res_dir / "oracle_stats.csv"
    optimizer_csv = res_dir / "rule_summary_result_stats.csv"
    if not optimizer_csv.exists():
        return None  # incomplete run

    oracle_rows = _read_csv_rows(oracle_csv)
    optimizer_rows = _read_csv_rows(optimizer_csv)
    name_to_rule = _build_name_to_rule(res_dir, trans_dir)

    top_rules = collect_top_rules(oracle_rows, name_to_rule)
    rationales = RationaleIndex(trans_dir)
    for r in top_rules:
        r["rationale"] = rationales.rationale_for(r["token"], name_to_rule.get(r["token"]))

    oracle_pct_rel, optim_pct_rel, relevant_size = _relevant_pct(oracle_rows, optimizer_rows)
    wl_speedup, wl_pct, wl_size = _workload_view(
        oracle_rows, _workload_original_runtimes(res_dir, trans_dir))

    return DatasetStats(
        name=_short_name(experiment),
        experiment=experiment,
        test_size=_load_test_size(trans_dir),
        oracle=_view_from_rows(oracle_rows),
        optimizer=_view_from_rows(optimizer_rows),
        top_rules=top_rules,
        oracle_pct_relevant=oracle_pct_rel,
        optimizer_pct_relevant=optim_pct_rel,
        relevant_size=relevant_size,
        oracle_workload_speedup=wl_speedup,
        oracle_workload_pct=wl_pct,
        workload_size=wl_size,
    )


def discover_experiments(results_root: Path, glob_pat: str,
                         canonical_imdb: str) -> tuple[list[str], list[str]]:
    """Return (complete, pending) canonical experiment names.

    Canonical = ``glob_pat`` matches minus ``*complex*`` minus every IMDB run,
    plus ``canonical_imdb`` added explicitly (it need not match the glob).
    Complete = has the optimizer stats CSV. The glob scopes the selection to one
    run family (``experiment_T_gen_*`` for thesis, ``experiment_*_oracle`` for
    flexible, ``experiment_*_oracle_join_p`` for fixed).
    """
    complete: list[str] = []
    pending: list[str] = []

    dirs = [d for d in sorted(results_root.glob(glob_pat)) if d.is_dir()]
    imdb_dir = results_root / canonical_imdb
    if imdb_dir.is_dir() and imdb_dir not in dirs:
        dirs.append(imdb_dir)

    for d in dirs:
        name = d.name
        if "complex" in name:
            continue
        if "imdb" in name and name != canonical_imdb:
            continue
        if (d / "rule_summary_result_stats.csv").exists():
            complete.append(name)
        else:
            pending.append(name)
    return sorted(complete), sorted(pending)


# --------------------------------------------------------------------------- #
# Rendering: chart + markdown
# --------------------------------------------------------------------------- #
def _fmt_pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "n/a"


def _fmt_x(v: float | None) -> str:
    return f"{v:.2f}×" if v is not None else "n/a"


def _render_grouped_barh(ds: list[DatasetStats], oracle_vals: list[float],
                         opt_vals: list[float], oracle_labels: list[str],
                         opt_labels: list[str], title: str, xlabel: str,
                         footer: str, path: Path) -> None:
    """Shared renderer for the grouped oracle-vs-optimizer horizontal bar charts.

    ``ds`` is assumed pre-sorted (ascending -> best on top). Each bar carries its
    own text label, placed just past the bar end and colored to match it.
    """
    y = range(len(ds))
    h = 0.38
    fig_height = max(4.0, 0.55 * len(ds) + 1.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    ax.barh([i + h / 2 for i in y], oracle_vals, height=h,
            color="#4C78A8", label="Oracle (best subset = upper bound)")
    opt_colors = ["#54A24B" if v >= 0 else "#D62728" for v in opt_vals]
    ax.barh([i - h / 2 for i in y], opt_vals, height=h,
            color=opt_colors, label="Optimizer (cost-picked, realistic)")

    ax.axvline(0.0, color="#333", linewidth=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels([d.name for d in ds])
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for i in range(len(ds)):
        ax.text(oracle_vals[i] + 0.5, i + h / 2, oracle_labels[i],
                va="center", ha="left", fontsize=7, color="#4C78A8")
        if opt_vals[i] >= 0:
            ax.text(opt_vals[i] + 0.5, i - h / 2, opt_labels[i],
                    va="center", ha="left", fontsize=7, color="#54A24B")
        else:
            ax.text(opt_vals[i] - 0.5, i - h / 2, opt_labels[i],
                    va="center", ha="right", fontsize=7, color="#D62728")

    # green = realistic gain, red = realistic net slowdown
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="Oracle (upper bound)"),
        plt.Rectangle((0, 0), 1, 1, color="#54A24B", label="Optimizer gain (realistic)"),
        plt.Rectangle((0, 0), 1, 1, color="#D62728", label="Optimizer net slowdown"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.text(0.99, 0.01, footer, fontsize=7, ha="right", color="#777")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_chart(datasets: list[DatasetStats], path: Path,
               title_suffix: str = "") -> None:
    """Grouped horizontal bars: oracle vs optimizer % workload runtime saved.

    Each bar is runtime-weighted over its OWN query basket (different denominators
    for oracle and optimizer). Sorted by oracle % saved (best at top).
    """
    ds = [d for d in datasets if d.oracle.pct_saved is not None]
    ds.sort(key=lambda d: d.oracle.pct_saved or 0.0)  # ascending -> best on top in barh
    if not ds:
        return

    oracle_vals = [d.oracle.pct_saved or 0.0 for d in ds]
    opt_vals = [d.optimizer.pct_saved if d.optimizer.pct_saved is not None else 0.0 for d in ds]
    # Per-bar label = how many queries that bar is computed over / workload, i.e.
    # the size of each view's query basket (n_rows). Keeps the label denominator-
    # consistent with the runtime-weighted bar.
    oracle_labels = [f"{d.oracle.n_rows}/{d.test_size}" if d.test_size
                     else f"{d.oracle.n_rows}" for d in ds]
    opt_labels = [f"{d.optimizer.n_rows}/{d.test_size}" if d.test_size
                  else f"{d.optimizer.n_rows}" for d in ds]

    _render_grouped_barh(
        ds, oracle_vals, opt_vals, oracle_labels, opt_labels,
        title="Semantic SQL rewrites — runtime saved per dataset" + title_suffix
              + "\n(base: 100 candidate queries per dataset)",
        xlabel="% of workload runtime saved (runtime-weighted)",
        footer=("labels = queries the bar covers / workload size "
                "(blue = oracle beneficial, green/red = optimizer rewritten incl. slowdowns)"),
        path=path)


def make_chart_relevant(datasets: list[DatasetStats], path: Path,
                        title_suffix: str = "") -> None:
    """Grouped horizontal bars over the SHARED "relevant" denominator.

    Both bars are runtime-weighted over the SAME set — the union of queries touched
    by either view (oracle-beneficial OR optimizer-picked); a query missing from a
    view counts as unchanged (0% saved) there. This puts oracle and optimizer on
    one denominator so they are directly comparable and oracle >= optimizer always
    holds (mirrors the ``_relevant`` speedup plots in stages/statistics.py).
    """
    ds = [d for d in datasets if d.oracle_pct_relevant is not None]
    ds.sort(key=lambda d: d.oracle_pct_relevant or 0.0)  # ascending -> best on top
    if not ds:
        return

    oracle_vals = [d.oracle_pct_relevant or 0.0 for d in ds]
    opt_vals = [d.optimizer_pct_relevant if d.optimizer_pct_relevant is not None else 0.0
                for d in ds]
    # Shared denominator = union size; label numerator = queries each view acted on.
    oracle_labels = [f"{d.oracle.n_rows}/{d.relevant_size}" for d in ds]
    opt_labels = [f"{d.optimizer.n_rows}/{d.relevant_size}" for d in ds]

    _render_grouped_barh(
        ds, oracle_vals, opt_vals, oracle_labels, opt_labels,
        title="Semantic SQL rewrites — runtime saved on relevant queries "
              "(union of oracle & optimizer)" + title_suffix
              + "\n(base: 100 candidate queries per dataset)",
        xlabel="% of relevant-query runtime saved (runtime-weighted, shared denominator)",
        footer=("labels = queries the view acted on / union size (shared denominator; "
                "blue = oracle, green/red = optimizer)"),
        path=path)


def _render_factor_barh(names: list[str], vals: list[float], labels: list[str],
                        title: str, xlabel: str, footer: str, path: Path) -> None:
    """Shared renderer for the single-series speedup-factor bar charts.

    ``names``/``vals``/``labels`` are parallel and pre-sorted ascending (best on
    top in ``barh``). Bars are drawn from 1.0× (= no change) rather than 0, so the
    differences between datasets stay visible; the axis therefore starts at 1.0×.
    """
    base = 1.0
    span = max(max(vals) - base, 0.01)

    y = range(len(names))
    fig_height = max(4.0, 0.42 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(list(y), [v - base for v in vals], left=base, height=0.62, color="#4C78A8")

    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Headroom on the right so the longest label still fits inside the axes.
    ax.set_xlim(base, base + span * 1.42)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.grid(axis="x", color="#DDD", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for i, v in enumerate(vals):
        ax.text(v + span * 0.012, i, labels[i],
                va="center", ha="left", fontsize=7.5, color="#2A4A6B")

    med = median(vals)
    if med is not None:
        ax.axvline(med, color="#D62728", linestyle="--", linewidth=1.0,
                   label=f"median {_fmt_x(med)}")
        ax.legend(loc="lower right", fontsize=8)

    fig.text(0.99, 0.008, footer, fontsize=7, ha="right", color="#777")
    # Reserve a bottom strip for the footer so it cannot collide with the x label.
    fig.tight_layout(rect=(0.0, 0.028, 1.0, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_chart_oracle(datasets: list[DatasetStats], path: Path,
                      title_suffix: str = "") -> None:
    """Single-series horizontal bars: the oracle upper bound only.

    Unlike ``make_chart``, the bar is the **geometric mean of the per-query speedup
    factors** over the queries a beneficial rule was found for — the typical
    per-query speedup, not the runtime-weighted workload share. Every bar carries
    that factor plus the coverage ``X/<workload>``: how many of the dataset's
    queries the geo-mean is computed over.
    """
    ds = [d for d in datasets if d.oracle.geo_mean is not None]
    ds.sort(key=lambda d: d.oracle.geo_mean or 0.0)  # ascending -> best on top in barh
    if not ds:
        return

    labels = [
        f"{_fmt_x(d.oracle.geo_mean)}  ({d.oracle.n_rows}/{d.test_size} queries)"
        if d.test_size else f"{_fmt_x(d.oracle.geo_mean)}  ({d.oracle.n_rows} queries)"
        for d in ds
    ]
    _render_factor_barh(
        [d.name for d in ds], [d.oracle.geo_mean or 1.0 for d in ds], labels,
        title="Semantic SQL rewrites — oracle upper bound per dataset" + title_suffix
              + "\n(geometric mean over the queries a beneficial rule was found for)",
        xlabel="geometric mean of per-query speedup factors (oracle subset)",
        footer=("bars start at 1.00× (no change); labels = geo-mean speedup "
                "(queries with a beneficial rule / workload size)"),
        path=path)


def make_chart_oracle_workload(datasets: list[DatasetStats], path: Path,
                               title_suffix: str = "") -> None:
    """Single-series horizontal bars: oracle speedup over the WHOLE query set.

    Complements ``make_chart_oracle``: instead of averaging over the covered
    queries only, this is Σ original / Σ improved across **every** query of the
    workload — queries no rule fired on (or that the oracle found no beneficial
    subset for) contribute their original runtime to both totals. That makes it
    the honest end-to-end number for the full benchmark, always ≤ the per-query
    geo-mean, and it is exactly the factor ``stages/statistics.py`` prints as
    "WHOLE-WORKLOAD (runtime-weighted)".
    """
    ds = [d for d in datasets if d.oracle_workload_speedup is not None]
    ds.sort(key=lambda d: d.oracle_workload_speedup or 0.0)  # ascending -> best on top
    if not ds:
        return

    labels = [
        f"{_fmt_x(d.oracle_workload_speedup)}  "
        f"({_fmt_pct(d.oracle_workload_pct)} of {d.workload_size} queries' runtime)"
        for d in ds
    ]
    _render_factor_barh(
        [d.name for d in ds], [d.oracle_workload_speedup or 1.0 for d in ds], labels,
        title="Semantic SQL rewrites — oracle speedup over the whole workload"
              + title_suffix
              + "\n(all queries, unchanged ones included at 1.00×)",
        xlabel="total workload speedup factor (Σ original / Σ improved, all queries)",
        footer=("bars start at 1.00× (no change); labels = workload speedup "
                "(% of total workload runtime saved / workload size)"),
        path=path)


def make_report(datasets: list[DatasetStats], pending: list[str],
                chart_name: str, path: Path,
                relevant_chart_name: str | None = None,
                oracle_chart_name: str | None = None,
                oracle_workload_chart_name: str | None = None) -> None:
    ordered = sorted(datasets, key=lambda d: d.oracle.pct_saved or -1e9, reverse=True)

    # Headline aggregates.
    oracle_pcts = [d.oracle.pct_saved for d in datasets if d.oracle.pct_saved is not None]
    opt_pcts = [d.optimizer.pct_saved for d in datasets if d.optimizer.pct_saved is not None]
    total_rule_queries = sum(d.oracle.n_rows for d in datasets)
    total_queries = sum(d.test_size for d in datasets if d.test_size)
    best = ordered[0] if ordered else None

    lines: list[str] = []
    lines.append("# Semantic SQL Rewrites — Cross-Dataset Performance Overview")
    lines.append("")
    lines.append(f"*Generated {date.today().isoformat()} from {len(datasets)} completed datasets.*")
    lines.append("")

    # --- Headline (oracle-led) ---
    lines.append("## Headline")
    lines.append("")
    if oracle_pcts:
        med_o = median(oracle_pcts)
        med_opt = median(opt_pcts) if opt_pcts else None
        lines.append(
            f"- **Oracle (upper bound):** median **{_fmt_pct(med_o)}** of workload runtime "
            f"saved per dataset"
            + (f", up to **{_fmt_pct(best.oracle.pct_saved)}** on **{best.name}**." if best else ".")
        )
        lines.append(
            f"- **Optimizer (realistic, cost-picked):** median **{_fmt_pct(med_opt)}** of "
            f"workload runtime saved per dataset (negative = net slowdown from cost mispredicts)."
        )
    lines.append(
        f"- **Coverage:** a beneficial rule was found for **{total_rule_queries}"
        + (f" / {total_queries}" if total_queries else "")
        + "** queries across all datasets."
    )
    lines.append("")
    lines.append(
        "> *Oracle* = the fastest output-matching rule subset per query (the ceiling an ideal "
        "selector could reach). *Optimizer* = the subset a cost-based optimizer actually picks, "
        "including mispredicts. The gap between them is selection headroom, not a rule-quality "
        "problem. % saved is runtime-weighted (robust to noisy sub-10 ms queries)."
    )
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary table")
    lines.append("")
    lines.append("Sorted by oracle % workload runtime saved.")
    lines.append("")
    lines.append("| Dataset | Coverage | Oracle % saved | Oracle geo-mean | Oracle best query | Optimizer % saved | Optimizer up/down |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in ordered:
        cov = f"{d.oracle.n_rows}/{d.test_size}" if d.test_size else str(d.oracle.n_rows)
        best_q = f"{d.oracle.best_prefix} ({_fmt_x(d.oracle.best_speedup)})" if d.oracle.best_prefix else "—"
        updown = f"{d.optimizer.n_improved}↑ / {d.optimizer.n_slowdown}↓"
        lines.append(
            f"| **{d.name}** | {cov} | {_fmt_pct(d.oracle.pct_saved)} | "
            f"{_fmt_x(d.oracle.geo_mean)} | {best_q} | "
            f"{_fmt_pct(d.optimizer.pct_saved)} | {updown} |"
        )
    lines.append("")
    lines.append(f"![Runtime saved per dataset]({chart_name})")
    lines.append("")

    # --- Oracle-only view ---
    if oracle_chart_name is not None:
        lines.append("## Oracle upper bound only")
        lines.append("")
        lines.append(
            "The oracle view without the optimizer bars — the ceiling an ideal subset "
            "selector could reach. Bars show the **geometric mean of the per-query speedup "
            "factors** (the typical per-query speedup, unweighted), drawn from 1.00× = no "
            "change; each is annotated with the coverage (queries a beneficial rule was "
            "found for / workload size). Note this is a different metric from the "
            "runtime-weighted % saved in the tables above."
        )
        lines.append("")
        lines.append(f"![Oracle upper bound per dataset]({oracle_chart_name})")
        lines.append("")

    if oracle_workload_chart_name is not None:
        lines.append("### Over the whole query set")
        lines.append("")
        lines.append(
            "The same oracle view as one end-to-end number per dataset: Σ original / "
            "Σ improved over **all** queries of the workload, with untouched queries "
            "counted at 1.00×. This is always below the per-query geo-mean above — "
            "the difference is exactly the dilution by queries no rule fired on."
        )
        lines.append("")
        lines.append("| Dataset | Whole-workload speedup | % of total runtime saved | Workload size |")
        lines.append("|---|---|---|---|")
        for d in sorted(datasets, key=lambda x: x.oracle_workload_speedup or -1e9,
                        reverse=True):
            lines.append(
                f"| **{d.name}** | {_fmt_x(d.oracle_workload_speedup)} | "
                f"{_fmt_pct(d.oracle_workload_pct)} | {d.workload_size} |"
            )
        lines.append("")
        lines.append(f"![Oracle speedup over the whole workload]({oracle_workload_chart_name})")
        lines.append("")

    # --- Relevant-query view (shared denominator) ---
    if relevant_chart_name is not None:
        rel_ordered = sorted(datasets,
                             key=lambda d: d.oracle_pct_relevant or -1e9, reverse=True)
        lines.append("## Relevant-query view (shared denominator)")
        lines.append("")
        lines.append(
            "Both views computed over the **same** denominator — the union of queries "
            "touched by either the oracle or the optimizer (a query absent from a view "
            "counts as unchanged there). Unlike the summary table above, this makes "
            "oracle and optimizer directly comparable, so oracle ≥ optimizer always holds."
        )
        lines.append("")
        lines.append("| Dataset | Union size | Oracle % saved (relevant) | Optimizer % saved (relevant) |")
        lines.append("|---|---|---|---|")
        for d in rel_ordered:
            lines.append(
                f"| **{d.name}** | {d.oracle.n_rows}∪{d.optimizer.n_rows}→{d.relevant_size} | "
                f"{_fmt_pct(d.oracle_pct_relevant)} | {_fmt_pct(d.optimizer_pct_relevant)} |"
            )
        lines.append("")
        lines.append(f"![Runtime saved on relevant queries]({relevant_chart_name})")
        lines.append("")

    # --- Per-dataset rules ---
    lines.append("## Most impactful rules per dataset")
    lines.append("")
    lines.append(
        "The 2–3 highest-impact rules per dataset, rendered as `IF <conditions in the query> "
        "THEN <predicates injected>`. These encode the world knowledge the optimizer cannot derive."
    )
    lines.append("")
    for d in ordered:
        lines.append(f"### {d.name}")
        lines.append("")
        cov = f"{d.oracle.n_rows}/{d.test_size}" if d.test_size else str(d.oracle.n_rows)
        lines.append(
            f"- Oracle: **{_fmt_pct(d.oracle.pct_saved)}** saved "
            f"({_fmt_x(d.oracle.speedup_factor)} total, geo-mean {_fmt_x(d.oracle.geo_mean)}); "
            f"rules helped **{cov}** queries."
        )
        lines.append(
            f"- Optimizer: **{_fmt_pct(d.optimizer.pct_saved)}** saved "
            f"({d.optimizer.n_improved} faster, {d.optimizer.n_slowdown} slower)."
        )
        lines.append("")
        if d.top_rules:
            for r in d.top_rules:
                ex = ""
                if r["example_speedup"]:
                    ex = (f" — best on `{r['example_prefix']}`: {_fmt_x(r['example_speedup'])} "
                          f"({r['example_pct']:.0f}% faster)")
                fired = f"; fired on {r['fired_on']} queries" if r["fired_on"] > 1 else ""
                lines.append(f"- **{r['id']}** — {r['rendered']}{ex}{fired}")
                if r.get("rationale"):
                    lines.append(f"  - *LLM rationale:* {r['rationale']}")
        else:
            lines.append("- *(no beneficial rules in the oracle view)*")
        lines.append("")

    if pending:
        lines.append("## Pending datasets (not yet complete)")
        lines.append("")
        for name in pending:
            lines.append(f"- {_short_name(name)} (`{name}`)")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve().parent.parent  # systematic_eval/ (this script lives in scripts/)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=here / "saved_results",
                    help="directory holding experiment_<name>_oracle result folders")
    ap.add_argument("--transfer-dir", type=Path, default=here / "transfer_data",
                    help="directory holding experiment_<name>_oracle transfer folders (test_size)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the report + chart (default: --results-dir)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="thesis",
                    help="run family: 'thesis' = experiment_T_gen_* runs + the thesis IMDB "
                         "run (default), 'flexible' = legacy base *_oracle runs, "
                         "'fixed' = legacy *_oracle_join_p runs (fix_join_order), "
                         "'umbra' = legacy *_oracle_umbra runs")
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    results_root: Path = args.results_dir
    transfer_root: Path = args.transfer_dir
    out_dir: Path = args.out_dir or results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    complete, pending = discover_experiments(
        results_root, variant["glob"], variant["canonical_imdb"])
    datasets: list[DatasetStats] = []
    for name in complete:
        ds = load_dataset(name, results_root, transfer_root)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        print(f"No completed datasets found under {results_root}")
        return

    stem = variant["out_stem"]
    tsuffix = variant["title_suffix"]
    chart_path = out_dir / f"{stem}.png"
    relevant_chart_path = out_dir / f"{stem}_relevant.png"
    oracle_chart_path = out_dir / f"{stem}_oracle.png"
    oracle_workload_chart_path = out_dir / f"{stem}_oracle_workload.png"
    report_path = out_dir / f"{stem}.md"
    make_chart(datasets, chart_path, title_suffix=tsuffix)
    make_chart_relevant(datasets, relevant_chart_path, title_suffix=tsuffix)
    make_chart_oracle(datasets, oracle_chart_path, title_suffix=tsuffix)
    make_chart_oracle_workload(datasets, oracle_workload_chart_path, title_suffix=tsuffix)
    make_report(datasets, pending, chart_path.name, report_path,
                relevant_chart_name=relevant_chart_path.name,
                oracle_chart_name=oracle_chart_path.name,
                oracle_workload_chart_name=oracle_workload_chart_path.name)

    # Terminal summary. "opt %" is the own-basket optimizer number; "rel o/o" is
    # the shared-denominator oracle/optimizer pair (directly comparable).
    print(f"\n{'=' * 78}")
    print(f"PERFORMANCE OVERVIEW [{args.variant}] — {len(datasets)} datasets "
          f"({len(pending)} pending: {', '.join(_short_name(p) for p in pending) or 'none'})")
    print("=" * 78)
    print(f"{'dataset':<16}{'cov':>9}{'oracle %':>11}{'oracle x':>10}{'opt %':>10}"
          f"{'rel orac%':>11}{'rel opt%':>10}")
    for d in sorted(datasets, key=lambda x: x.oracle.pct_saved or -1e9, reverse=True):
        cov = f"{d.oracle.n_rows}/{d.test_size}" if d.test_size else str(d.oracle.n_rows)
        print(f"{d.name:<16}{cov:>9}{_fmt_pct(d.oracle.pct_saved):>11}"
              f"{_fmt_x(d.oracle.speedup_factor):>10}{_fmt_pct(d.optimizer.pct_saved):>10}"
              f"{_fmt_pct(d.oracle_pct_relevant):>11}{_fmt_pct(d.optimizer_pct_relevant):>10}")
    print("=" * 78)
    print(f"Report: {report_path}")
    print(f"Chart:  {chart_path}")
    print(f"Chart:  {relevant_chart_path}")
    print(f"Chart:  {oracle_chart_path}")
    print(f"Chart:  {oracle_workload_chart_path}\n")


if __name__ == "__main__":
    main()
