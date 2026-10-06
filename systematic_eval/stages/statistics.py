"""Stage 4: per-run statistics and plots.

Per query: median runtime and percent saved; speedup factors aggregated by geometric mean.
Cross-run aggregates and the thesis's reported numbers live in ``scripts/``, not here.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as stats_stdlib
from pathlib import Path

import matplotlib.pyplot as plt

from systematic_eval.stages.plan_visualization import generate_plan_comparisons


# ── Trust scoring thresholds ────────────────────────────────────────────────
# A query's timing is only as trustworthy as it is reproducible. We gauge that
# from the coefficient of variation (std/mean) of the repeated per-attempt
# timings of the *same* query — pure measurement noise, no signal. On a quiet
# node this is ~0.2%; on a contended shared node it routinely exceeds 10%.
TRUST_CV_WARN = 0.02   # CV below this → "trusted"
TRUST_CV_FAIL = 0.10   # CV above this → "untrusted" (between the two → "suspect")
# Extrinsic backstop (populated by resource_monitor.py; absent on older runs):
# even if the N attempts agreed with each other, sustained contention during the
# measurement window makes the number untrustworthy.
TRUST_SIBLING_BUSY_FAIL = 0.30  # sibling-core busy fraction during window → "untrusted"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[float], aggregator: str) -> float:
    if aggregator == "mean":
        return sum(values) / len(values)
    # median (default): match execution-stage convention (sorted middle, no two-element averaging)
    s = sorted(values)
    return s[len(s) // 2]


def _drop_sentinels(values) -> list[float]:
    """Keep only valid positive per-attempt timings.

    The execution stage records ``-1`` as a placeholder when an attempt's
    DuckDB profiling JSON can't be read back (a rare write/read race); it means
    "timing unknown", not a real runtime. Aggregating it (especially with
    runtime_aggregator=mean) yields negative "improved" runtimes and impossible
    percent-saved values (>100%). Drop any non-positive timing here so the
    sentinel never reaches aggregation.
    """
    if not isinstance(values, list):
        return []
    return [t for t in values if isinstance(t, (int, float)) and t > 0]


def build_rows(summary: dict, runtime_aggregator: str = "median") -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped_empty = 0
    for prefix, entry in summary.items():
        exec_time = entry.get("summary", {}).get("execution_time", {})
        orig = exec_time.get("original_query")
        improved = exec_time.get("llm_transformed_query")
        # The original and improved runtimes come from two separate execute_query
        # calls, so attempt i of one does not correspond to attempt i of the
        # other — filter each list independently rather than dropping paired
        # indices.
        orig_runs = _drop_sentinels(exec_time.get("original_runtimes"))
        improved_runs = _drop_sentinels(exec_time.get("llm_transformed_runtimes"))
        runs_valid = len(orig_runs) >= 1 and len(improved_runs) >= 1
        if runs_valid:
            orig = _aggregate(orig_runs, runtime_aggregator)
            improved = _aggregate(improved_runs, runtime_aggregator)
        # When a list was fully sentinel-filtered we fall back to the stored
        # scalar median (original_query / llm_transformed_query). Require both to
        # be positive: a scalar median is itself -1 when >=2 attempts failed.
        if (isinstance(orig, (int, float)) and isinstance(improved, (int, float))
                and orig > 0 and improved > 0):
            saved_pct = (orig - improved) / orig * 100.0
            row = {
                "prefix": prefix,
                "original_runtime": orig,
                "improved_runtime": improved,
                "percent_saved": saved_pct,
            }
            # Paired range/variance columns assume equal-length lists. After
            # independent sentinel-filtering the two sides can differ in length
            # (one lost an attempt to a profiling-read failure); only emit the
            # paired stats when they still line up, otherwise skip just these
            # optional columns while keeping a correct percent_saved.
            if runs_valid and len(orig_runs) == len(improved_runs) >= 2:
                paired_pct = [(o - i) / o * 100.0 for o, i in zip(orig_runs, improved_runs)]
                row["original_min"] = min(orig_runs)
                row["original_max"] = max(orig_runs)
                row["improved_min"] = min(improved_runs)
                row["improved_max"] = max(improved_runs)
                row["percent_saved_min"] = min(paired_pct)
                row["percent_saved_max"] = max(paired_pct)
                row["n_attempts"] = len(orig_runs)
                row["original_stddev"] = math.sqrt(stats_stdlib.variance(orig_runs))
                row["improved_stddev"] = math.sqrt(stats_stdlib.variance(improved_runs))
                row["percent_saved_stddev"] = math.sqrt(stats_stdlib.variance(paired_pct))
            rows.append(row)
    return rows, skipped_empty


# ── Trust scoring ───────────────────────────────────────────────────────────

def _cv(runtimes) -> float | None:
    """Coefficient of variation (population std / mean) of clean per-attempt
    timings, or None when fewer than two valid attempts are available."""
    vals = _drop_sentinels(runtimes)
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return None
    return math.sqrt(stats_stdlib.pvariance(vals)) / mean


def compute_trust(original_runtimes, transformed_runtimes=None,
                  contention: dict | None = None) -> dict:
    """Per-query trust from timing reproducibility (intrinsic) plus, when
    available, measured server contention during the window (extrinsic).

    Returns {original_cv, transformed_cv, worst_cv, trust, sibling_busy_max,
    pin_khz_min} with cv values as fractions (None when unknown). ``trust`` is
    one of "trusted" / "suspect" / "untrusted" / "unknown".
    """
    ocv = _cv(original_runtimes)
    tcv = _cv(transformed_runtimes) if transformed_runtimes is not None else None
    cvs = [c for c in (ocv, tcv) if c is not None]
    worst = max(cvs) if cvs else None

    if worst is None:
        label = "unknown"
    elif worst < TRUST_CV_WARN:
        label = "trusted"
    elif worst < TRUST_CV_FAIL:
        label = "suspect"
    else:
        label = "untrusted"

    sibling_busy_max = pin_khz_min = None
    if contention:
        sibling_busy_max = contention.get("sibling_busy_max")
        pin_khz_min = contention.get("pin_khz_min")
        # Extrinsic backstop: sustained sibling contention overrides a clean CV
        # (the blind spot where all N attempts were equally slowed).
        if (isinstance(sibling_busy_max, (int, float))
                and sibling_busy_max > TRUST_SIBLING_BUSY_FAIL
                and label in ("trusted", "suspect")):
            label = "untrusted"

    return {
        "original_cv": ocv,
        "transformed_cv": tcv,
        "worst_cv": worst,
        "trust": label,
        "sibling_busy_max": sibling_busy_max,
        "pin_khz_min": pin_khz_min,
    }


def build_trust_rows(summary: dict) -> dict[str, dict]:
    """Map prefix → trust dict (see compute_trust) for every query in a
    rule_summary_result.json. Reads the top-level per-attempt arrays and the
    optional ``contention`` block written by the resource monitor."""
    out: dict[str, dict] = {}
    for prefix, entry in summary.items():
        summ = entry.get("summary", {}) if isinstance(entry, dict) else {}
        exec_time = summ.get("execution_time", {})
        out[prefix] = compute_trust(
            exec_time.get("original_runtimes"),
            exec_time.get("llm_transformed_runtimes"),
            contention=summ.get("contention"),
        )
    return out


def summarize_trust(trust_by_prefix: dict[str, dict]) -> dict:
    """Run-level trust summary from per-query trust dicts."""
    labels = [t["trust"] for t in trust_by_prefix.values()]
    known_cvs = sorted(t["worst_cv"] for t in trust_by_prefix.values()
                       if t["worst_cv"] is not None)
    n = len(labels)
    n_scored = len(known_cvs)
    n_trusted = labels.count("trusted")
    return {
        "n_queries": n,
        "n_scored": n_scored,
        "n_trusted": n_trusted,
        "n_suspect": labels.count("suspect"),
        "n_untrusted": labels.count("untrusted"),
        "n_unknown": labels.count("unknown"),
        "pct_trusted": (100.0 * n_trusted / n_scored) if n_scored else None,
        "median_cv_pct": (100.0 * known_cvs[len(known_cvs) // 2]) if known_cvs else None,
        "worst_cv_pct": (100.0 * known_cvs[-1]) if known_cvs else None,
        "thresholds": {
            "cv_warn_pct": 100.0 * TRUST_CV_WARN,
            "cv_fail_pct": 100.0 * TRUST_CV_FAIL,
            "sibling_busy_fail_pct": 100.0 * TRUST_SIBLING_BUSY_FAIL,
        },
    }


# Trust columns appended to the per-query CSV/table views. Kept in sync with
# _attach_trust below.
_TRUST_CORE_FIELDS = ["trust", "worst_cv_pct", "original_cv_pct"]
_TRUST_CONTENTION_FIELDS = ["sibling_busy_max_pct", "pin_khz_min"]


def _attach_trust(rows: list[dict], trust_by_prefix: dict[str, dict]) -> list[str]:
    """Add trust columns to each row (matched by ``prefix``) in place; return
    the list of field names actually added (contention columns only when any
    row carries them)."""
    any_contention = False
    for row in rows:
        t = trust_by_prefix.get(row["prefix"])
        if not t:
            continue
        row["trust"] = t["trust"]
        if t["worst_cv"] is not None:
            row["worst_cv_pct"] = 100.0 * t["worst_cv"]
        if t["original_cv"] is not None:
            row["original_cv_pct"] = 100.0 * t["original_cv"]
        if t["sibling_busy_max"] is not None:
            row["sibling_busy_max_pct"] = 100.0 * t["sibling_busy_max"]
            any_contention = True
        if t["pin_khz_min"] is not None:
            row["pin_khz_min"] = t["pin_khz_min"]
            any_contention = True
    return _TRUST_CORE_FIELDS + (_TRUST_CONTENTION_FIELDS if any_contention else [])


def print_trust_summary(summary: dict, label: str = "TRUST SUMMARY") -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    thr = summary["thresholds"]
    print(f"thresholds: trusted CV < {thr['cv_warn_pct']:.0f}%, "
          f"untrusted CV > {thr['cv_fail_pct']:.0f}%  "
          f"(sibling-busy backstop > {thr['sibling_busy_fail_pct']:.0f}%)")
    mcv = summary["median_cv_pct"]
    wcv = summary["worst_cv_pct"]
    pct = summary["pct_trusted"]
    print(f"queries: {summary['n_queries']} "
          f"({summary['n_scored']} scored, {summary['n_unknown']} unknown)")
    print(f"median attempt CV: {mcv:.2f}%" if mcv is not None else "median attempt CV: n/a")
    print(f"worst attempt CV:  {wcv:.2f}%" if wcv is not None else "worst attempt CV:  n/a")
    print(f"trusted: {summary['n_trusted']}  suspect: {summary['n_suspect']}  "
          f"untrusted: {summary['n_untrusted']}"
          + (f"  ({pct:.0f}% of scored trusted)" if pct is not None else ""))
    print("=" * 60 + "\n")


def write_csv(rows: list[dict], path: Path, show_stats: str = "none",
              extra_fieldnames: list[str] | None = None) -> None:
    fieldnames = ["prefix", "original_runtime", "improved_runtime", "percent_saved"]
    if show_stats == "range":
        fieldnames += [
            "original_min", "original_max",
            "improved_min", "improved_max",
            "percent_saved_min", "percent_saved_max",
            "n_attempts",
        ]
    elif show_stats == "variance":
        fieldnames += [
            "original_stddev",
            "improved_stddev",
            "percent_saved_stddev",
            "n_attempts",
        ]
    if extra_fieldnames:
        fieldnames += extra_fieldnames
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Serialize list-typed cells (e.g. winning_rules) as ';'-joined strings.
            out = dict(row)
            for k, v in list(out.items()):
                if isinstance(v, list):
                    out[k] = ";".join(str(x) for x in v)
            writer.writerow(out)


def save_table_figure(rows: list[dict], path: Path, skipped_empty: int, show_stats: str = "none") -> None:
    if not rows:
        return

    def _range_str(lo, hi, fmt):
        if lo is None or hi is None:
            return ""
        return f"[{lo:{fmt}}..{hi:{fmt}}]"

    def _sigma_str(value, fmt):
        if value is None:
            return ""
        return f"±{value:{fmt}}"

    if show_stats == "range":
        cell_text = [
            [
                row["prefix"],
                f"{row['original_runtime']:.4f}",
                _range_str(row.get("original_min"), row.get("original_max"), ".4f"),
                f"{row['improved_runtime']:.4f}",
                _range_str(row.get("improved_min"), row.get("improved_max"), ".4f"),
                f"{row['percent_saved']:.2f}%",
                _range_str(row.get("percent_saved_min"), row.get("percent_saved_max"), ".2f"),
            ]
            for row in rows
        ]
        col_labels = [
            "Prefix",
            "Original runtime", "orig [min..max]",
            "Improved runtime", "improved [min..max]",
            "% saved", "% saved [min..max]",
        ]
        fig_width = 12.0
    elif show_stats == "variance":
        cell_text = [
            [
                row["prefix"],
                f"{row['original_runtime']:.4f}",
                _sigma_str(row.get("original_stddev"), ".4f"),
                f"{row['improved_runtime']:.4f}",
                _sigma_str(row.get("improved_stddev"), ".4f"),
                f"{row['percent_saved']:.2f}%",
                _sigma_str(row.get("percent_saved_stddev"), ".2f"),
            ]
            for row in rows
        ]
        col_labels = [
            "Prefix",
            "Original runtime", "orig σ",
            "Improved runtime", "improved σ",
            "% saved", "% saved σ",
        ]
        fig_width = 12.0
    else:
        cell_text = [
            [
                row["prefix"],
                f"{row['original_runtime']:.4f}",
                f"{row['improved_runtime']:.4f}",
                f"{row['percent_saved']:.2f}%",
            ]
            for row in rows
        ]
        col_labels = ["Prefix", "Original runtime", "Improved runtime", "% saved"]
        fig_width = 8.0
    fig_height = max(2.0, 0.4 + 0.4 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    if skipped_empty > 0:
        fig.text(0.01, 0.01, f"Skipped {skipped_empty} entries with empty original results", fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def runtime_weighted_percent_saved(rows: list[dict]) -> float | None:
    total_orig = sum(row["original_runtime"] for row in rows)
    total_impr = sum(row["improved_runtime"] for row in rows)
    if total_orig <= 0:
        return None
    return (total_orig - total_impr) / total_orig * 100.0


def geometric_mean(values: list[float]) -> float | None:
    """Geometric mean of positive values; the scale-consistent average for
    multiplicative quantities like speedup factors. Returns None if empty."""
    vals = [v for v in values if v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def save_bar_plot(rows: list[dict], path: Path, test_size: int | None,
                  show_stats: str = "none", runtime_aggregator: str = "median",
                  title: str = "Runtime saved by optimized SQL query") -> None:
    if not rows:
        return

    prefixes = [row["prefix"].split(" ", 1)[0] for row in rows]
    saved = [row["percent_saved"] for row in rows]
    avg_saved = sum(saved) / len(saved) if saved else 0.0
    weighted_saved = runtime_weighted_percent_saved(rows)

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    yerr = None
    if show_stats == "range":
        lower: list[float] = []
        upper: list[float] = []
        any_range = False
        for row in rows:
            lo = row.get("percent_saved_min")
            hi = row.get("percent_saved_max")
            mid = row["percent_saved"]
            if lo is not None and hi is not None:
                lower.append(max(0.0, mid - lo))
                upper.append(max(0.0, hi - mid))
                any_range = True
            else:
                lower.append(0.0)
                upper.append(0.0)
        if any_range:
            yerr = [lower, upper]
    elif show_stats == "variance":
        sigmas: list[float] = []
        any_sigma = False
        for row in rows:
            s = row.get("percent_saved_stddev")
            if s is not None:
                sigmas.append(float(s))
                any_sigma = True
            else:
                sigmas.append(0.0)
        if any_sigma:
            yerr = sigmas
    if yerr is not None:
        ax.bar(prefixes, saved, color="#4C78A8", yerr=yerr, capsize=3,
               ecolor="#333", error_kw={"linewidth": 1.0})
    else:
        ax.bar(prefixes, saved, color="#4C78A8")
    ax.axhline(avg_saved, color="#F58518", linestyle="--", linewidth=1.5,
               label=f"unweighted mean: {avg_saved:.1f}%")
    if weighted_saved is not None:
        ax.axhline(weighted_saved, color="#54A24B", linestyle="--", linewidth=1.5,
                   label=f"runtime-weighted mean: {weighted_saved:.1f}%")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("runtime reduction (%)")
    ax.set_xlabel("Prefix")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    if isinstance(test_size, int) and test_size > 0:
        fig.text(
            0.01,
            0.01,
            f"For {len(rows)} of {test_size} SQL queries an optimization rule was found",
            fontsize=11,
        )
    fig.text(
        0.99,
        0.01,
        f"runtime_aggregator={runtime_aggregator}  |  show_stats={show_stats}",
        fontsize=8,
        ha="right",
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_speedup_bar_plot(rows: list[dict], path: Path, test_size: int | None,
                          runtime_aggregator: str = "mean",
                          baseline_basenames: list[str] | None = None,
                          baseline_runtimes: dict[str, float] | None = None,
                          title: str = "Speedup factor by optimized SQL query (ordered ascending)",
                          overall_total: int | None = None) -> None:
    """Bar plot of per-query speedup factor (original / improved), sorted ascending.

    Factor 1.0 = unchanged. Red bars are slowdowns (<1), blue bars speedups (>=1).
    Error bars come from first-order propagation of original/improved stddevs.

    If `baseline_basenames` is provided, queries from that list that are NOT in
    `rows` are added as gray 1.0x bars (no rule was applied — runtime unchanged).
    Matching is done on the leading whitespace-separated token of each row's
    prefix (e.g. "1c NoMIN" -> "1c").

    If `overall_total` is provided, the footer reports `test_size` as the number
    of *relevant* queries (the plot's universe) and additionally states
    `overall_total` as the full dataset size — used by the union/"relevant" plots
    where `test_size` is the union size rather than the whole dataset.
    """
    if not rows:
        return

    enriched: list[dict] = []
    seen_basenames: set[str] = set()
    for row in rows:
        orig = row["original_runtime"]
        impr = row["improved_runtime"]
        if orig <= 0 or impr <= 0:
            continue
        speedup = orig / impr
        s_o = row.get("original_stddev")
        s_i = row.get("improved_stddev")
        if s_o is not None and s_i is not None:
            rel = math.sqrt((s_o / orig) ** 2 + (s_i / impr) ** 2)
            sigma = speedup * rel
        else:
            sigma = 0.0
        prefix = row["prefix"]
        seen_basenames.add(prefix.split(" ", 1)[0])
        enriched.append({
            "prefix": prefix, "speedup": speedup, "sigma": sigma,
            "orig": orig, "impr": impr, "kind": "rule",
        })

    if baseline_basenames is not None:
        for name in baseline_basenames:
            if name in seen_basenames:
                continue
            row = {
                "prefix": name, "speedup": 1.0, "sigma": 0.0,
                "kind": "baseline",
            }
            # A baseline query is unchanged in this view (orig == impr). If its
            # runtime is known, record it so it can join the workload total.
            if baseline_runtimes is not None and name in baseline_runtimes:
                rt = baseline_runtimes[name]
                row["orig"] = rt
                row["impr"] = rt
            enriched.append(row)

    enriched.sort(key=lambda r: (r["speedup"], r["kind"] != "rule", r["prefix"]))
    prefixes = [r["prefix"].split(" ", 1)[0] for r in enriched]
    speedups = [r["speedup"] for r in enriched]
    sigmas = [r["sigma"] for r in enriched]
    kinds = [r["kind"] for r in enriched]

    def bar_color(s: float, kind: str) -> str:
        if kind == "baseline":
            return "#BBBBBB"
        return "#D62728" if s < 1.0 else "#4C78A8"

    colors = [bar_color(s, k) for s, k in zip(speedups, kinds)]
    # Geometric mean over EVERY displayed bar (synthetic baselines count as
    # factor 1.0), so the line reflects exactly the queries shown on this plot
    # rather than always summarising the rule-only subset.
    geo_mean = geometric_mean(speedups)
    # Total-time factor sum(orig)/sum(impr) and the equivalent % runtime saved
    # are the only runtime-weighted aggregates with a physical meaning. They are
    # computed over every displayed query that has runtimes, including unchanged
    # baselines (orig == impr) — so the percentage reflects the whole shown set.
    #
    # When `baseline_runtimes` was NOT provided (e.g. the plain *_all plot before
    # a baseline run), *no* baseline has a runtime, so the workload total would be
    # meaningless — suppress the line. When it WAS provided the caller intends a
    # whole-workload total; a handful of baselines may still lack a runtime (a
    # query whose measurement failed, so it can't be shown as a real bar either).
    # Rather than drop the line entirely, compute over the queries with known
    # runtimes and note how many were excluded.
    n_missing_baselines = sum(
        1 for r in enriched if r["kind"] == "baseline" and "orig" not in r
    )
    if baseline_runtimes is None and n_missing_baselines:
        workload_factor = None
        percent_saved = None
    else:
        total_orig = sum(r["orig"] for r in enriched if "orig" in r)
        total_impr = sum(r["impr"] for r in enriched if "impr" in r)
        if total_impr > 0 and total_orig > 0:
            workload_factor = total_orig / total_impr
            percent_saved = (total_orig - total_impr) / total_orig * 100.0
        else:
            workload_factor = None
            percent_saved = None

    fig_width = max(10.0, 0.18 * len(enriched))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    any_sigma = any(s > 0 for s in sigmas)
    ax.bar(
        prefixes, speedups, color=colors,
        yerr=sigmas if any_sigma else None,
        capsize=3, ecolor="#333", error_kw={"linewidth": 1.0},
    )
    ax.axhline(1.0, color="#333", linewidth=1.0)
    if geo_mean is not None:
        ax.axhline(geo_mean, color="#F58518", linestyle="--", linewidth=1.5)
    if workload_factor is not None:
        ax.axhline(workload_factor, color="#54A24B", linestyle="--", linewidth=1.5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#D62728", label="slowdown (<1)"),
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="speedup (≥1)"),
    ]
    if baseline_basenames is not None:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color="#BBBBBB", label="no rule applied (=1)")
        )
    if geo_mean is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color="#F58518", linestyle="--",
                       label=f"geometric mean: {geo_mean:.3f}x")
        )
    if workload_factor is not None:
        pct_txt = f" ({percent_saved:.1f}% saved)" if percent_saved is not None else ""
        miss_txt = (f" — {n_missing_baselines} query w/o runtime excluded"
                    if n_missing_baselines else "")
        legend_handles.append(
            plt.Line2D([0], [0], color="#54A24B", linestyle="--",
                       label=f"total runtime saved: {workload_factor:.3f}x{pct_txt}{miss_txt}")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_ylabel("speedup factor (original / improved)")
    xlabel = "Prefix (sorted by speedup ascending)"
    if baseline_basenames is not None:
        xlabel += " — gray bars: no rule applied"
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    label_fontsize = 6 if len(enriched) > 60 else 8
    ax.tick_params(axis="x", rotation=0, labelsize=label_fontsize)

    label_offset = max(speedups) * 0.04
    text_fontsize = 6 if len(enriched) > 60 else 7
    for i, (s, k) in enumerate(zip(speedups, kinds)):
        if k == "baseline":
            continue
        ax.text(i, s - label_offset, f"{s:.2f}", ha="center", va="top",
                fontsize=text_fontsize, color="white")

    if isinstance(test_size, int) and test_size > 0:
        n_rule = sum(1 for k in kinds if k == "rule")
        if isinstance(overall_total, int) and overall_total > 0:
            footer_text = (
                f"For {n_rule} of {test_size} relevant SQL queries an optimization "
                f"rule was found ({overall_total} queries total)"
            )
        else:
            footer_text = (
                f"For {n_rule} of {test_size} SQL queries an optimization rule was found"
            )
        fig.text(0.01, 0.01, footer_text, fontsize=11)
    fig.text(0.99, 0.01,
             f"runtime_aggregator={runtime_aggregator}  |  factor 1.00 = unchanged",
             fontsize=8, ha="right", color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def list_sql_basenames(
    sql_dir: Path,
    excluded_files: list[str] | None = None,
    query_limit: int | None = None,
) -> list[str]:
    """SQL query basenames (e.g. "1c", "9d") from <sql_dir>/*.sql.

    Mirrors the selection in generation/aggregation: drop *excluded_files*,
    sort, then take the first *query_limit*. This keeps the statistics baseline
    set identical to the queries the experiment actually ran (important when
    sql_dir holds a larger pool than the run, e.g. the unpacked 200k workload).
    """
    excluded = set(excluded_files or [])
    names = sorted(
        p.name for p in Path(sql_dir).glob("*.sql") if p.name not in excluded
    )
    if query_limit is not None:
        names = names[:query_limit]
    return [Path(n).stem for n in names]


def build_baseline_input(
    transfer_dir: Path,
    sql_dir: Path,
    excluded_files: list[str] | None = None,
    query_limit: int | None = None,
) -> Path | None:
    """Write baseline_input.json = {basename: sql} for the *no-rule* queries.

    The no-rule set is the experiment's query universe (`list_sql_basenames`,
    honoring *excluded_files*/*query_limit*) minus the queries already measured in
    `rule_summary_result.json`. Those are exactly the queries drawn as gray 1.0×
    bars on the `*_speedup_bar_all` plots; measuring their original runtime lets
    the runtime-weighted whole-workload line be drawn there.

    Returns the written path, or None when there is nothing to measure (every
    query already had a rule applied, or no matching .sql files were found).
    """
    all_names = list_sql_basenames(Path(sql_dir), excluded_files, query_limit)

    # A query counts as "already measured" only if it has a *valid* original
    # runtime. A query whose final-execution measurement failed (missing / <=0
    # original_query) is re-measured here so the runtime-weighted *_all line can
    # cover it instead of dropping it as an unknown baseline.
    measured: set[str] = set()
    result_path = transfer_dir / "rule_summary_result.json"
    if result_path.exists():
        summary = load_summary(result_path)
        for k, entry in summary.items():
            et = entry.get("summary", {}).get("execution_time", {}) if isinstance(entry, dict) else {}
            ov = et.get("original_query")
            if isinstance(ov, (int, float)) and ov > 0:
                measured.add(k.split(" ", 1)[0])

    todo = [n for n in all_names if n not in measured]
    queries: dict[str, str] = {}
    for name in todo:
        sql_file = Path(sql_dir) / f"{name}.sql"
        if sql_file.exists():
            queries[name] = sql_file.read_text(encoding="utf-8")

    if not queries:
        print("baseline-prep: no no-rule queries to measure "
              f"({len(all_names)} in workload, {len(measured)} already measured).")
        return None

    out_path = transfer_dir / "baseline_input.json"
    out_path.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(f"baseline-prep: wrote {len(queries)} no-rule query/queries to {out_path}.")
    return out_path


def print_workload_weighted_improvement(
    rows: list[dict], baseline_runtimes: dict[str, float], label: str) -> None:
    """Runtime-weighted improvement over the WHOLE workload.

    The queries shown as real bars (in *rows*) contribute their measured
    original/improved runtimes; every other query is *unchanged* in this view
    (no rule applied, or the view declined/found no beneficial subset) and
    contributes its original runtime to *both* totals — pure denominator weight,
    diluting the percentage without changing the absolute time saved. Mirrors the
    green "total runtime saved" line on the `*_speedup_bar_all` plots.
    """
    if not baseline_runtimes:
        return
    shown = {r["prefix"].split(" ", 1)[0] for r in rows}
    total_orig = sum(r["original_runtime"] for r in rows)
    total_impr = sum(r["improved_runtime"] for r in rows)
    n_extra = 0
    for name, rt in baseline_runtimes.items():
        if name in shown or rt <= 0:
            continue
        total_orig += rt
        total_impr += rt
        n_extra += 1
    if total_orig <= 0:
        return
    factor = total_orig / total_impr if total_impr > 0 else float("inf")
    pct = (total_orig - total_impr) / total_orig * 100.0
    print("\n" + "-" * 60)
    print(f"{label} — WHOLE-WORKLOAD (runtime-weighted)")
    print("-" * 60)
    print(f"Queries: {len(rows)} changed + {n_extra} unchanged (no-rule/declined)")
    print(f"Total original runtime (all): {total_orig:.4f}s")
    print(f"Total improved runtime (all): {total_impr:.4f}s")
    print(f"Runtime-weighted improvement: {pct:.2f}%  ({factor:.4f}x)")
    print("-" * 60 + "\n")


def print_runtime_statistics(rows: list[dict], runtime_aggregator: str = "median",
                             show_stats: str = "none",
                             label: str = "RUNTIME STATISTICS") -> None:
    if not rows:
        print("No valid data to calculate runtime statistics.")
        return

    total_runtime_wo_rewrites = sum(row["original_runtime"] for row in rows)
    total_runtime_w_rewrites = sum(row["improved_runtime"] for row in rows)

    if total_runtime_w_rewrites > 0:
        speedup = total_runtime_wo_rewrites / total_runtime_w_rewrites
    else:
        speedup = float("inf")

    avg_pct = sum(row["percent_saved"] for row in rows) / len(rows)

    factors = [r["original_runtime"] / r["improved_runtime"]
               for r in rows if r["improved_runtime"] > 0]
    geo_mean = geometric_mean(factors)

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"runtime_aggregator: {runtime_aggregator}   show_stats: {show_stats}")
    print(f"Total runtime without rewrites: {total_runtime_wo_rewrites:.4f}s")
    print(f"Total runtime with rewrites:    {total_runtime_w_rewrites:.4f}s")
    print(f"Speedup factor (total time):    {speedup:.4f}x")
    print(f"Geometric mean speedup:         {geo_mean:.4f}x" if geo_mean is not None else "Geometric mean speedup:         n/a")
    print(f"Total time saved:               {total_runtime_wo_rewrites - total_runtime_w_rewrites:.4f}s")
    print(f"Weighted improvement (by runtime): {((total_runtime_wo_rewrites - total_runtime_w_rewrites) / total_runtime_wo_rewrites * 100):.2f}%")
    print(f"Unweighted mean improvement:       {avg_pct:.2f}%")
    print("=" * 60 + "\n")


def build_oracle_rows(summary: dict) -> tuple[list[dict], int]:
    """Build per-query rows where each query's "improved" runtime is the fastest
    output-matching subset of its firing rules — the upper bound on speedup that
    an ideal optimizer could achieve given the discovered rules.

    Reads per_subset_results (populated by stages/execution.py during final
    execution) and, per query, picks the candidate with minimum measured runtime
    among:
      - a synthetic "no rules" candidate (original_query timing, trivial match), AND
      - every entry in per_subset_results whose ``summary.outputs_match`` is true.

    A subset with ``outputs_match: false`` is a real correctness bug for an
    individually validated rule combination — excluded and reported with a loud
    WARN line (does not silently skip).

    Tie-break for equal runtimes: smaller subset first, then lex order on sorted
    rule names. The "no rules" candidate (empty subset) wins all ties of equal
    runtime by virtue of size 0.

    Per-subset entries only store the median runtime (not per-attempt lists), so
    the chosen runtime is what the execution stage reported as the median —
    independent of ``runtime_aggregator``. The original-query side mirrors that
    by using the stored median ``original_query`` value for parity.

    Returns (rows, skipped_empty) with the same shape as ``build_rows`` plus
    ``winning_rules`` (list of rule names; ``[]`` when "no rules" wins).
    """
    rows: list[dict] = []
    skipped_empty = 0
    for prefix, entry in summary.items():
        exec_time = entry.get("summary", {}).get("execution_time", {})
        orig = exec_time.get("original_query")
        llm_improved = exec_time.get("llm_transformed_query")
        if not (isinstance(orig, (int, float)) and orig > 0):
            continue

        # Candidate list: each tuple = (size, sorted_rule_names_tuple, runtime, rule_names_list)
        candidates: list[tuple[int, tuple[str, ...], float, list[str]]] = [
            (0, (), float(orig), []),  # "no rules"
        ]

        per_subset = entry.get("per_subset_results")
        if isinstance(per_subset, list) and per_subset:
            for subset in per_subset:
                rule_names = subset.get("rule_names", []) or []
                s_summary = subset.get("summary", {}) or {}
                s_exec = s_summary.get("execution_time", {}) or {}
                s_time = s_exec.get("subset_transformed_query")
                outputs_match = s_summary.get("outputs_match")
                # Reject the -1 "timing unknown" sentinel (and any non-positive
                # value): the oracle picks the minimum runtime, so a -1 would
                # look like the fastest candidate and win every query.
                if not isinstance(s_time, (int, float)) or s_time <= 0:
                    continue
                if outputs_match is False:
                    print(
                        f"WARN: oracle excluded subset {rule_names} for query "
                        f"{prefix!r}: outputs_match=False on a validated rule "
                        f"combination — this should be impossible; inspect the "
                        f"rules / refined SQL in rule_summary_result.json."
                    )
                    continue
                candidates.append(
                    (len(rule_names), tuple(sorted(rule_names)),
                     float(s_time), list(rule_names))
                )
        else:
            # Backward compat: pre-per_subset_results runs only have the
            # all-rules-combined timing. Treat that as the sole candidate.
            if isinstance(llm_improved, (int, float)):
                print(
                    f"INFO: oracle fallback for {prefix!r} — no per_subset_results "
                    f"in rule_summary_result.json; using llm_transformed_query "
                    f"as the only candidate."
                )
                candidates.append(
                    (1, ("__llm_combined__",), float(llm_improved), ["<all-rules-combined>"])
                )

        # Pick by (runtime, size, sorted_names) — smallest runtime wins; ties
        # resolved by fewer rules, then lex on sorted rule names.
        candidates.sort(key=lambda c: (c[2], c[0], c[1]))
        _, _, best_time, best_rules = candidates[0]

        # Exclude queries where no rule subset beats the original — the oracle
        # view should only report queries with a beneficial rule combination.
        if not best_rules:
            skipped_empty += 1
            continue

        saved_pct = (orig - best_time) / orig * 100.0
        rows.append({
            "prefix": prefix,
            "original_runtime": float(orig),
            "improved_runtime": best_time,
            "percent_saved": saved_pct,
            "winning_rules": best_rules,
        })
    return rows, skipped_empty


def resolve_optimizer_winners(transfer_dir: Path, summary: dict) -> dict[str, list[str]] | None:
    """Return {prefix: winner_rule_names} for the cost-based optimizer view, or None.

    Two sources, in priority order:
      1. A sidecar ``optimizer_cost_winners.json`` in transfer_dir — written by the
         retrofit ``--mode optimizer-winners`` pass over an existing run.
      2. ``cost_estimation.combo_search.winner_rules`` recorded per query during a
         cost-estimation pipeline run (``--mode cost-aggregate --keep-full-pool``).

    Returns None when neither source is present, so callers fall back to the legacy
    all-rules optimizer view (``build_rows``). An empty winner list ([]) is a real
    value — the optimizer declined every rule for that query.
    """
    sidecar = transfer_dir / "optimizer_cost_winners.json"
    if sidecar.exists():
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        winners: dict[str, list[str]] = {}
        for prefix, val in raw.items():
            if isinstance(val, dict):
                winners[prefix] = list(val.get("winner_rules", []) or [])
            elif isinstance(val, list):
                winners[prefix] = list(val)
        return winners

    winners = {}
    found = False
    for prefix, entry in summary.items():
        ce = entry.get("cost_estimation") if isinstance(entry, dict) else None
        if not isinstance(ce, dict):
            continue
        # Only full-pool (oracle) cost runs drive the optimizer view from metadata;
        # pruned optimizer-only runs fall back to the legacy build_rows view.
        if not ce.get("full_pool_kept"):
            continue
        wr = ce.get("combo_search", {}).get("winner_rules")
        if wr is not None:
            winners[prefix] = list(wr)
            found = True
    return winners if found else None


def build_optimizer_cost_rows(summary: dict, winners: dict[str, list[str]]) -> tuple[list[dict], int]:
    """Build optimizer-view rows whose runtime is that of the cost-winning rule
    subset — the subset a cost-based optimizer would pick — read from
    per_subset_results.

    Mirrors ``build_oracle_rows`` but selects by cost winner instead of minimum
    measured runtime. A query whose winner is empty/missing (optimizer declines all
    rules) is skipped (counted). Slowdowns are NOT excluded: a cost winner that
    measures slower than the original is a real mispredict and must be reported —
    that gap versus the oracle is the whole point of the comparison.

    Like ``build_oracle_rows``, the runtime comes from ``per_subset_results``,
    which stores the execution stage's median only (no per-attempt lists), so the
    reported runtime is that median — ``runtime_aggregator`` does not apply here.

    Returns (rows, skipped) with ``build_oracle_rows``' shape (plus "winning_rules").
    """
    rows: list[dict] = []
    skipped = 0
    for prefix, entry in summary.items():
        exec_time = entry.get("summary", {}).get("execution_time", {})
        orig = exec_time.get("original_query")
        if not (isinstance(orig, (int, float)) and orig > 0):
            continue

        wr = winners.get(prefix)
        if not wr:  # None (no info) or [] (optimizer declines all rules)
            skipped += 1
            continue
        wr_key = tuple(sorted(wr))

        chosen_time = None
        per_subset = entry.get("per_subset_results")
        if isinstance(per_subset, list):
            for subset in per_subset:
                names = subset.get("rule_names", []) or []
                if tuple(sorted(names)) != wr_key:
                    continue
                s_time = (subset.get("summary", {})
                          .get("execution_time", {})
                          .get("subset_transformed_query"))
                # Skip the -1 "timing unknown" sentinel (and any non-positive
                # value) so it can't be reported as the cost-winner's runtime.
                if isinstance(s_time, (int, float)) and s_time > 0:
                    chosen_time = float(s_time)
                break

        if chosen_time is None:
            # Winner equals the full fired set → reuse the all-rules call #2 timing.
            all_names = tuple(sorted(
                e.get("name") for e in (entry.get("rules") or []) if isinstance(e, dict)
            ))
            llm_improved = exec_time.get("llm_transformed_query")
            if wr_key == all_names and isinstance(llm_improved, (int, float)) and llm_improved > 0:
                chosen_time = float(llm_improved)
            else:
                print(
                    f"WARN: optimizer winner {list(wr)} for query {prefix!r} has no "
                    f"matching per_subset_results entry — skipping. Re-run the winner "
                    f"search against the same rule pool as the execution."
                )
                skipped += 1
                continue

        saved_pct = (orig - chosen_time) / orig * 100.0
        rows.append({
            "prefix": prefix,
            "original_runtime": float(orig),
            "improved_runtime": chosen_time,
            "percent_saved": saved_pct,
            "winning_rules": list(wr),
        })
    return rows, skipped


def build_plan_comparison_picks(
    summary: dict,
    oracle_rows: list[dict],
    optimizer_rows: list[dict] | None,
) -> dict[str, dict[str, list[str]]]:
    """Map each query to the rule subset its oracle / optimizer view selected.

    Feeds ``generate_plan_comparisons`` so that queries with too many subsets can
    be reduced to the two subsets the stats actually report. Both row builders
    already carry ``winning_rules``; this pivots them per query.

    *optimizer_rows* is None for the legacy optimizer view (``build_rows``, no
    cost-winner info), whose reported runtime is ``llm_transformed_query`` — i.e.
    all fired rules. That full set is always present in per_subset_results, so it
    is the correct pick for that view.

    A view is omitted for a query when it selected no rules or the query was
    dropped from that view, so a query may map to 0, 1, or 2 entries.
    """
    picks: dict[str, dict[str, list[str]]] = {}

    for row in oracle_rows:
        if row.get("winning_rules"):
            picks.setdefault(row["prefix"], {})["oracle"] = list(row["winning_rules"])

    if optimizer_rows is not None:
        for row in optimizer_rows:
            if row.get("winning_rules"):
                picks.setdefault(row["prefix"], {})["optimizer"] = list(row["winning_rules"])
    else:
        for prefix, entry in summary.items():
            all_rules = [
                e.get("name") for e in (entry.get("rules") or [])
                if isinstance(e, dict) and e.get("name")
            ]
            if all_rules:
                picks.setdefault(prefix, {})["optimizer"] = all_rules

    return picks


def run_statistics(transfer_dir: Path, results_dir: Path | None = None,
                   show_stats: str = "none", runtime_aggregator: str = "median",
                   mode: str = "optimizer",
                   sql_dir: Path | None = None,
                   excluded_files: list[str] | None = None,
                   query_limit: int | None = None,
                   skip_plan_comparisons: bool = False,
                   full_plan_comparisons: bool = False) -> None:
    """Run statistics calculation and generate CSV + visualizations.

    *mode* selects which view(s) to emit:
      - "optimizer": current behaviour — what the engine's planner picked.
      - "oracle": upper-bound from the per-subset measurements.
      - "both": emit both side-by-side from the same rule_summary_result.json.

    *full_plan_comparisons* plots every rule subset instead of reducing queries
    with more than PLAN_COMPARISON_SUBSET_LIMIT subsets to their oracle/optimizer
    picks.

    *runtime_aggregator* and *show_stats* only reach the legacy all-rules
    optimizer view (``build_rows``), the sole view that reads the per-attempt
    timing lists. The oracle view and the cost-winner optimizer view both read
    ``per_subset_results``, which stores the execution stage's median and no
    per-attempt list, so for those two the runtime is always that median and the
    spread columns are always empty — independent of what the config asks for.

    If results_dir is provided, all outputs (CSV, PNGs, final rule set) are
    saved there instead of transfer_dir.
    """
    input_path = transfer_dir / "rule_summary_result.json"
    test_size_path = transfer_dir / "test_size.json"

    out_dir = results_dir if results_dir is not None else transfer_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(input_path)

    # Per-query + run-level trust from timing reproducibility (and, when the
    # resource monitor ran, measured server contention). Emitted once and joined
    # into every per-query view below.
    trust_by_prefix = build_trust_rows(summary)
    trust_summary = summarize_trust(trust_by_prefix)
    (out_dir / "trust_summary.json").write_text(
        json.dumps(trust_summary, indent=2), encoding="utf-8")
    print_trust_summary(trust_summary)

    try:
        test_size_data = json.loads(test_size_path.read_text(encoding="utf-8"))
        if isinstance(test_size_data, dict):
            test_size = test_size_data.get("test_size")
        elif isinstance(test_size_data, (int, float)):
            test_size = int(test_size_data)
        else:
            test_size = None
    except FileNotFoundError:
        test_size = None

    emit_optimizer = mode in ("optimizer", "both")
    emit_oracle = mode in ("oracle", "both")

    baseline_basenames: list[str] | None = None
    if sql_dir is not None:
        baseline_basenames = list_sql_basenames(
            Path(sql_dir), excluded_files=excluded_files, query_limit=query_limit
        )

    # Runtimes for every query that shows up as an *unchanged* (gray 1.0×) bar on
    # the `*_speedup_bar_all` plots, so the runtime-weighted "total runtime saved"
    # line can be drawn over the whole workload. Two disjoint sources:
    #   1. no-rule queries — measured by `--mode baseline` into baseline_runtimes.json;
    #   2. measured-but-unchanged queries — a query that HAS a measurement but is
    #      dropped from a view (optimizer declined all rules / oracle found no
    #      beneficial subset) still renders as a gray bar; its original runtime is
    #      already in the summary and is the right weight (improved == original).
    # A query shown as a real bar is never a baseline (it's excluded downstream).
    all_baseline_runtimes: dict[str, float] = {}
    for key, entry in summary.items():
        et = entry.get("summary", {}).get("execution_time", {}) if isinstance(entry, dict) else {}
        ov = et.get("original_query")
        if isinstance(ov, (int, float)) and ov > 0:
            all_baseline_runtimes.setdefault(key.split(" ", 1)[0], ov)

    no_rule_runtimes: dict[str, float] = {}
    baseline_rt_path = transfer_dir / "baseline_runtimes.json"
    if baseline_rt_path.exists():
        raw_baseline = load_summary(baseline_rt_path)
        for name, entry in raw_baseline.items():
            rt = entry.get("original_query") if isinstance(entry, dict) else entry
            if isinstance(rt, (int, float)) and rt > 0:
                no_rule_runtimes[name] = rt
        print(f"Loaded {len(no_rule_runtimes)} no-rule baseline runtime(s) from "
              f"{baseline_rt_path.name}.")
    # no-rule (measured now) wins over any stale summary value on the rare overlap.
    all_baseline_runtimes.update(no_rule_runtimes)

    # Kept across the view branches to drive the plan-comparison picks below.
    # oracle_rows stays None until built; optimizer_cost_rows stays None for the
    # legacy (non-cost-winner) optimizer view.
    oracle_rows: list[dict] | None = None
    optimizer_cost_rows: list[dict] | None = None

    if emit_optimizer:
        csv_path = out_dir / "rule_summary_result_stats.csv"
        fig_path = out_dir / "rule_summary_result_stats.png"
        bar_fig_path = out_dir / "rule_summary_result_stats_bar.png"
        speedup_bar_path = out_dir / "rule_summary_result_speedup_bar.png"
        speedup_bar_all_path = out_dir / "rule_summary_result_speedup_bar_all.png"

        # When cost-winner info is available (sidecar or pipeline metadata), the
        # optimizer view reports each query's cost-chosen subset runtime, read from
        # per_subset_results. Otherwise fall back to the legacy all-rules view.
        winners = resolve_optimizer_winners(transfer_dir, summary)
        if winners is not None:
            rows, skipped_empty = build_optimizer_cost_rows(summary, winners)
            optimizer_cost_rows = rows
            # per_subset_results stores medians only — no per-attempt variance.
            opt_show_stats = "none"
            # For the same reason `runtime_aggregator` cannot apply to this view:
            # there is no per-attempt list to collapse, the runtime IS the
            # execution stage's median. Label the artefacts with what was
            # actually computed, never with the config value — otherwise a run
            # configured `mean` prints "mean" over a column of medians.
            opt_runtime_aggregator = "median"
            opt_extra_fieldnames = ["winning_rules"]
            speedup_rows = list(rows)
            print(f"Optimizer view: cost-based subset selection over {len(rows)} "
                  f"queries ({skipped_empty} declined/skipped).")
        else:
            rows, skipped_empty = build_rows(summary, runtime_aggregator=runtime_aggregator)
            opt_show_stats = show_stats
            # The legacy all-rules view is the only one that reads the per-attempt
            # lists, so it is the only one `runtime_aggregator` governs.
            opt_runtime_aggregator = runtime_aggregator
            opt_extra_fieldnames = None
            # Speedup plots of this view use mean-aggregated runtimes.
            speedup_rows, _ = build_rows(summary, runtime_aggregator="mean")
        rows.sort(key=lambda r: r["prefix"])
        speedup_rows.sort(key=lambda r: r["prefix"])

        trust_fields = _attach_trust(rows, trust_by_prefix)
        opt_extra_fieldnames = (opt_extra_fieldnames or []) + trust_fields

        write_csv(rows, csv_path, show_stats=opt_show_stats,
                  extra_fieldnames=opt_extra_fieldnames)
        save_table_figure(rows, fig_path, skipped_empty, show_stats=opt_show_stats)
        save_bar_plot(rows, bar_fig_path, test_size, show_stats=opt_show_stats,
                      runtime_aggregator=opt_runtime_aggregator)

        save_speedup_bar_plot(speedup_rows, speedup_bar_path, test_size,
                              runtime_aggregator="mean")
        if baseline_basenames is not None:
            save_speedup_bar_plot(speedup_rows, speedup_bar_all_path, test_size,
                                  runtime_aggregator="mean",
                                  baseline_basenames=baseline_basenames,
                                  baseline_runtimes=all_baseline_runtimes or None)

        print_runtime_statistics(rows, runtime_aggregator=opt_runtime_aggregator,
                                 show_stats=opt_show_stats,
                                 label="RUNTIME STATISTICS (optimizer)")
        if no_rule_runtimes:
            print_workload_weighted_improvement(
                speedup_rows, all_baseline_runtimes,
                label="RUNTIME STATISTICS (optimizer)")

    if emit_oracle:
        oracle_csv_path = out_dir / "oracle_stats.csv"
        oracle_fig_path = out_dir / "oracle_stats.png"
        oracle_bar_fig_path = out_dir / "oracle_stats_bar.png"
        oracle_speedup_bar_path = out_dir / "oracle_speedup_bar.png"
        oracle_speedup_bar_all_path = out_dir / "oracle_speedup_bar_all.png"

        oracle_rows, oracle_skipped = build_oracle_rows(summary)
        oracle_rows.sort(key=lambda r: r["prefix"])

        oracle_trust_fields = _attach_trust(oracle_rows, trust_by_prefix)

        # The oracle picks per-subset medians as-is (per_subset_results only
        # stores median, not per-attempt lists), so range/variance columns are
        # not meaningful here — pass show_stats="none" for the oracle artefacts.
        # The trust columns come from the query-level original-attempt CV.
        write_csv(oracle_rows, oracle_csv_path, show_stats="none",
                  extra_fieldnames=["winning_rules"] + oracle_trust_fields)
        save_table_figure(oracle_rows, oracle_fig_path, oracle_skipped,
                          show_stats="none")
        # runtime_aggregator="median" is a statement of fact, not a setting: the
        # oracle reads per-subset medians, so the config value never applies here.
        save_bar_plot(oracle_rows, oracle_bar_fig_path, test_size,
                      show_stats="none", runtime_aggregator="median",
                      title="Oracle: max possible runtime saved (best subset per query)")
        save_speedup_bar_plot(oracle_rows, oracle_speedup_bar_path, test_size,
                              runtime_aggregator="median",
                              title="Oracle speedup factor by SQL query (ordered ascending)")
        if baseline_basenames is not None:
            save_speedup_bar_plot(oracle_rows, oracle_speedup_bar_all_path, test_size,
                                  runtime_aggregator="median",
                                  baseline_basenames=baseline_basenames,
                                  baseline_runtimes=all_baseline_runtimes or None,
                                  title="Oracle speedup factor by SQL query (ordered ascending)")
        print_runtime_statistics(oracle_rows, runtime_aggregator="median",
                                 show_stats="none",
                                 label="RUNTIME STATISTICS (oracle — upper bound)")
        if no_rule_runtimes:
            print_workload_weighted_improvement(
                oracle_rows, all_baseline_runtimes,
                label="RUNTIME STATISTICS (oracle — upper bound)")

    # "Relevant"-queries speedup plots: when both views are computed, restrict
    # the x-axis to the union of queries touched by either run (optimizer-picked
    # OR oracle-beneficial). Each plot shows one view's real speedups; queries
    # relevant only to the *other* view appear as gray 1.0 bars. This drops
    # irrelevant queries while keeping both plots on a shared, comparable x-axis.
    if emit_optimizer and emit_oracle:
        def _basename(prefix: str) -> str:
            return prefix.split(" ", 1)[0]

        union = sorted(
            {_basename(r["prefix"]) for r in speedup_rows}
            | {_basename(r["prefix"]) for r in oracle_rows}
        )
        union_size = len(union)

        # Original runtime per query (same query → same baseline regardless of
        # view), so a query shown as a 1.0 baseline in one plot still counts in
        # that plot's workload total / % saved.
        orig_runtimes: dict[str, float] = {}
        for r in (*speedup_rows, *oracle_rows):
            orig_runtimes.setdefault(_basename(r["prefix"]), r["original_runtime"])

        save_speedup_bar_plot(
            speedup_rows, out_dir / "rule_summary_result_speedup_bar_relevant.png",
            union_size, runtime_aggregator="mean", baseline_basenames=union,
            baseline_runtimes=orig_runtimes,
            overall_total=test_size,
            title="Speedup factor — relevant queries (optimizer)")
        save_speedup_bar_plot(
            oracle_rows, out_dir / "oracle_speedup_bar_relevant.png",
            union_size, runtime_aggregator="median", baseline_basenames=union,
            baseline_runtimes=orig_runtimes,
            overall_total=test_size,
            title="Oracle speedup factor — relevant queries")

    # Generate query plan comparison visualizations (optimizer view only —
    # the oracle does not change which plans were captured).
    if emit_optimizer and not skip_plan_comparisons:
        plans_dir = out_dir / "plan_comparisons"
        # The oracle pick selects which subsets to plot even when the oracle view
        # itself is not emitted; build_oracle_rows is pure computation over the
        # already-loaded summary.
        if oracle_rows is None:
            oracle_rows, _ = build_oracle_rows(summary)
        picks = build_plan_comparison_picks(summary, oracle_rows, optimizer_cost_rows)
        n_plans = generate_plan_comparisons(
            input_path, plans_dir, picks=picks, full=full_plan_comparisons)
        if n_plans > 0:
            print(f"Generated {n_plans} query plan comparison(s) in {plans_dir}")
    elif skip_plan_comparisons:
        print("Skipping plan comparison generation (--no-plan-comparisons).")

    # Copy final rule set and execution results to results dir
    if results_dir is not None:
        import shutil
        for name in ("rule_summary_transfer.json", "rule_summary_result.json",
                     "baseline_runtimes.json"):
            src = transfer_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)


def run_trust_only(result_path: Path, out_dir: Path | None = None) -> dict:
    """Retroactive trust scoring for an existing rule_summary_result.json.

    Prints the run-level summary, writes trust_summary.json and a per-query
    trust_stats.csv next to the input (or into *out_dir*), and returns the
    run-level summary dict. No re-execution — works on any past run.
    """
    summary = load_summary(result_path)
    trust_by_prefix = build_trust_rows(summary)
    run_summary = summarize_trust(trust_by_prefix)

    out_dir = out_dir or result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trust_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8")

    rows: list[dict] = [{"prefix": p} for p in sorted(trust_by_prefix)]
    fields = _attach_trust(rows, trust_by_prefix)
    # Standalone CSV: only the prefix + trust columns (no runtime columns, which
    # this retroactive path does not recompute).
    with (out_dir / "trust_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prefix"] + fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print_trust_summary(run_summary, label=f"TRUST SUMMARY — {result_path.parent.name}")
    return run_summary


def _resolve_trust_input(arg: str) -> Path:
    """Accept a full path to a rule_summary_result.json, a directory containing
    one, or a bare experiment name (resolved under saved_results/, then
    transfer_data/)."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "rule_summary_result.json"
    # Treat as an experiment name relative to the systematic_eval/ root.
    base = Path(__file__).resolve().parent.parent  # .../systematic_eval
    for sub in ("saved_results", "transfer_data"):
        cand = base / sub / arg / "rule_summary_result.json"
        if cand.is_file():
            return cand
    # Fall back to the literal path so the error message is clear.
    return p


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retroactive trust scoring for a run's rule_summary_result.json.")
    parser.add_argument("--trust", required=True, metavar="EXPERIMENT_OR_PATH",
                        help="Experiment name (resolved under saved_results/ then "
                             "transfer_data/), or a path to a rule_summary_result.json "
                             "or its directory.")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help="Output directory (default: alongside the input file).")
    args = parser.parse_args()
    result_path = _resolve_trust_input(args.trust)
    if not result_path.is_file():
        parser.error(f"could not find rule_summary_result.json for {args.trust!r} "
                     f"(looked for {result_path})")
    run_trust_only(result_path, Path(args.out) if args.out else None)
