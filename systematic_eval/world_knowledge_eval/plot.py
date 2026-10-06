"""Local matplotlib rendering of the world-knowledge evaluation.

Produces static figures (PNG + PDF) alongside the JSON/CSV outputs, so the
analysis is usable in the thesis without the interactive HTML artifact. Mirrors
the artifact's two views: the score distribution and the score-vs-speedup strip
plot (with per-score medians).
"""

from __future__ import annotations

import statistics
from pathlib import Path

# Blue ordinal ramp (light -> dark): the same "score is the entity" identity the
# HTML report uses; 1 = data-derivable, 5 = world-knowledge.
_RAMP = {1: "#9ec5f4", 2: "#6da7ec", 3: "#3987e5", 4: "#1c5cab", 5: "#0d366b"}
_SHORT = {1: "1\ndb-\nderivable", 2: "2\nmostly\ndb-deriv.", 3: "3\nmixed",
          4: "4\nmostly\nworld-kn.", 5: "5\nworld-\nknowledge"}
_SLOWDOWN = "#c0392b"


def _by_score(records: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {s: [] for s in range(1, 6)}
    for r in records:
        s = (r.get("verdict") or {}).get("score")
        if isinstance(s, int) and 1 <= s <= 5:
            out[s].append(r)
    return out


def render_plots(records: list[dict], out_dir: Path, experiment: str = "", model: str = "") -> list[Path]:
    """Write score-distribution and score-vs-speedup figures. Returns paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = _by_score(records)
    counts = {s: len(grouped[s]) for s in range(1, 6)}
    written: list[Path] = []

    def _save(fig, stem: str) -> None:
        for ext in ("png", "pdf"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            if ext == "png":
                written.append(p)
        plt.close(fig)

    sub = f"{experiment}  ·  judge {model}" if experiment or model else ""

    # ---- 1. score distribution -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    xs = list(range(1, 6))
    ax.bar(xs, [counts[s] for s in xs], color=[_RAMP[s] for s in xs], width=0.72, zorder=3)
    for s in xs:
        ax.text(s, counts[s], str(counts[s]), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([_SHORT[s] for s in xs], fontsize=8.5)
    ax.set_ylabel("rules")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    fig.suptitle(f"World-knowledge dependence of {len(records)} validated rules",
                 fontsize=11, fontweight="bold", y=1.04)
    if sub:
        ax.set_title(sub, fontsize=7.5, color="#898781")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "wk_score_distribution")

    # ---- 2. score vs individual median speedup ---------------------------
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    all_meds = [d["agg"]["median_percent_saved"] for d in records
                if d["agg"].get("median_percent_saved") is not None]
    if not all_meds:
        all_meds = [0.0]
    ymin = min(-5.0, min(all_meds)) * 1.08
    ymax = max(5.0, max(all_meds)) * 1.08

    if ymin < 0:
        ax.axhspan(ymin, 0, color=_SLOWDOWN, alpha=0.055, zorder=0)
    ax.axhline(0, color="#c3c2b7", linewidth=1.3, zorder=2)

    for s in xs:
        meds = [d["agg"]["median_percent_saved"] for d in grouped[s]
                if d["agg"].get("median_percent_saved") is not None]
        if meds:
            jitter = (rng.random(len(meds)) - 0.5) * 0.5
            ax.scatter(s + jitter, meds, s=42, color=_RAMP[s],
                       edgecolor="white", linewidth=0.7, alpha=0.9, zorder=4)
            mn = statistics.fmean(meds)
            ax.hlines(mn, s - 0.28, s + 0.28, color="#0b0b0b", linewidth=2.0, zorder=5)
            ax.annotate(f"{mn:+.1f}", (s + 0.30, mn), fontsize=8.5, fontweight="bold",
                        va="center", ha="left", color="#0b0b0b")
        ax.text(s, ymax, f"n={counts[s]}", ha="center", va="top", fontsize=8, color="#52514e")

    ax.set_xticks(xs)
    ax.set_xticklabels([_SHORT[s] for s in xs], fontsize=8.5)
    ax.set_xlim(0.5, 5.6)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("world-knowledge dependence  (1 = data-derivable  →  5 = world-knowledge)")
    ax.set_ylabel("median % runtime saved (rule applied alone)")
    fig.suptitle("World-knowledge dependence vs. individual speedup",
                 fontsize=11, fontweight="bold", y=1.02)
    if sub:
        ax.set_title(sub, fontsize=7.5, color="#898781")
    ax.text(0.995, 0.015, "solid line = per-score mean · shaded = slowdown",
            transform=ax.transAxes, fontsize=7.5, color="#898781", ha="right", va="bottom")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "wk_score_vs_speedup")

    return written
