#!/usr/bin/env python3
"""
thesis_figures.py -- publication figures and tables for Chapter 5.

Reads the same data as thesis_numbers.py (imported, DRY) and renders vector
PDF figures plus booktabs .tex tables into the thesis img/ directory.

Colour follows the TU Darmstadt corporate palette and is assigned a ROLE; see
the palette block below. Figure width is fixed to the width the figure is
PRINTED at, so every figure lands on the page at scale 1.0 and a 9 pt label is
9 pt on paper; see the geometry block below.

Missing data never crashes: the figure (or table) is written as an empty,
dashed placeholder box under the SAME filename, so \\includegraphics / \\input
always resolves and the pending spot stays visible.

Run:
    python3 systematic_eval/scripts/thesis_figures.py            # -> thesis img/
    python3 systematic_eval/scripts/thesis_figures.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median, mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import thesis_numbers as tn  # noqa: E402  (same directory; sys.path[0] is script dir)

# ---------------------------------------------------------------------------
# PALETTE. Sourced from the TU Darmstadt corporate palette (tuda-ci,
# tudacolors.def), so the figures and the document body share one colour
# system. Raw tokens keep their TUDa names; everything below is an alias, so
# the provenance of every hex stays checkable against the .def file.
#
# The organising rule is that colour is assigned a ROLE, and a figure uses AT
# MOST ONE role set. Colour previously encoded engine, condition, series,
# status and ordinal position at once, with no separation, which is what made
# one hue mean several things across the chapter (blue was DuckDB in
# fig_oracle_ceiling and "oracle ceiling" in fig_engines, two figures with the
# same x axis).
#
#   DATA       one hue. For every figure where colour carries NO categorical
#              meaning because the category is already spelled out by a panel
#              title, an axis label or a tick label. This covers the engines:
#              fig_oracle_ceiling, fig_oracle_workload and fig_crossengine all
#              name the engine in text, so a per-engine hue was a redundant
#              channel. Retiring it is what frees the palette.
#   CONTRAST   one dark/light pair for two quantities where ONE BOUNDS THE
#              OTHER and the reader is meant to read them against each other.
#              DARK is always the bound, LIGHT what is measured against it, in
#              every figure that uses the pair:
#                  fig_engines     ceiling      >= realized
#                  fig_joinorder   free          (pinned restricts it)
#                  fig_oracle_*    geo_improved >= geo_workload
#              The bound relation is the whole criterion. Sharing one axes
#              system is NOT required: fig_oracle_ceiling and
#              fig_oracle_workload are two stacked figures printed as one
#              float, and the pair is what tells the reader that the two rows
#              are one measurement on two denominators rather than two
#              unrelated results. Conversely two panels that merely measure
#              different things (fig_novelty: rules against queries) do NOT
#              get it, since with no bound the dark/light order would assert
#              a ranking the data do not have.
#              Because the two differ in lightness rather than in hue, the
#              pair may never be drawn with alpha; see fig_joinorder.
#   STATUS     improved / neutral / regressed / declined. Reserved; these four
#              appear in no other role.
#   AGGREGATE  reference and summary markers (geometric mean, median, y=x).
#              Achromatic and dashed, never a fill, so an annotation can never
#              be mistaken for a category.
#   ORDINAL    one sequential ramp, for genuinely ordered quantities only.
#
# WHY THE CONTRAST PAIR IS TWO BLUES AND NOT A SECOND HUE. It was orange
# (TUDa-8b) first, which separates from the blue beautifully (dE 53.5) and is
# still wrong, because separation WITHIN a figure is not the only constraint:
# fig_engines (orange bars) and fig_engines_composition (red "regressed"
# segments) are printed on facing pages, and orange against the status red
# scores dE 8.8 under deuteranopia. All 48 TUDa colours were then searched for
# a hue that clears the data blue, the green, the red AND the ordinal ramp at
# once. None does; the best candidate bottoms out at dE 13.1. This is
# structural rather than bad luck: green and red are spent on the status pair
# and blue on the data, so every remaining colour sits next to one of them.
# Cyan (TUDa-2a) is the trap, since it looks free but lands inside the ramp at
# dE 3.8 from step 3.
#
# So an overlap is unavoidable and the only question is which one is cheapest.
# The pair overlaps the ORDINAL ramp (light is exactly ramp step 2). That costs
# nothing, because neither use ever carries meaning through colour alone: the
# ramp is always accompanied by the score as a digit or an axis tick, and the
# contrast pair always by a legend. Overlapping the STATUS pair would instead
# cost the one association the reader is asked to carry through the chapter
# unaided, green is better and red is worse.
#
# Every pair a reader must actually separate was checked with CIEDE2000 under
# normal vision and under simulated protanopia and deuteranopia, floor dE 20;
# check_palette() below is the executable version of this list.
# The status pair is the load-bearing one: the previous green/red (#0ca30c,
# #d03b3b) scored dE 11.8 under deuteranopia and was not separable for roughly
# 8% of male readers, which mattered because improved and regressed are the two
# largest segments of fig_engines_composition. TUDa-3b is a blue-green, so the
# same green-is-good convention now survives the simulation at 31.9.
# Do not eyeball-substitute; rerun the check instead.
# ---------------------------------------------------------------------------
TUDA_1B = "#005AA9"                               # corporate blue, base
TUDA_3B, TUDA_9B = "#009D81", "#E6001A"           # status green / status red
TUDA_0A, TUDA_0C, TUDA_0D = "#DCDCDC", "#898989", "#535353"   # greys

DATA = TUDA_1B                       # bars, points, histograms
DATA_PALE = "#99BDDD"                # de-emphasised bars behind a highlight
C1, C2 = TUDA_1B, DATA_PALE          # CONTRAST: C1 dark = bound, C2 light
GOOD, NEUTRAL, CRIT = TUDA_3B, TUDA_0A, TUDA_9B   # STATUS (declined = hatch)
REF = "#2B2B2B"                      # AGGREGATE lines and their labels
BASE = TUDA_0C                       # the no-change baseline: chrome, not a finding
INK, INK2, SURFACE = "#0B0B0B", TUDA_0D, "#ffffff"
GRID = "#E8E8E8"                     # lightened TUDa-0a, must recede behind bars
# ORDINAL: TUDa-1b tinted toward white at 20/40/60/80/100 %.
SEQ = ["#CCDEEE", "#99BDDD", "#669CCB", "#337BBA", "#005AA9"]

# ---------------------------------------------------------------------------
# GEOMETRY. \\textwidth of the thesis is 398.34 pt = 5.512 in (tuda-ci, a4,
# 11 pt base, BCOR=10mm; read off thesis.log).
#
# Each figure is authored at the width it is PRINTED at, and Evaluation.tex
# includes it with NO width= option, so \\includegraphics places it at natural
# size and the scale factor is exactly 1.0. Previously figsize and the
# \\linewidth factor were chosen independently, so the on-page scale ranged from
# 0.62 to 0.93, a 1.5x spread: fig_engines printed its axis labels at 5.6 pt and
# its value labels at 4.3 pt while fig_worldknowledge printed the same nominal
# sizes at 8.4 and 6.5 pt. That spread, not the palette, was the strongest
# signal that the figures were not one family.
#
# Only two widths exist, so a figure cannot drift to a bespoke size. If a
# figure does not fit its width, fix its content, do not scale it on the page.
# bbox_inches="tight" is deliberately NOT used in save(): it crops the canvas
# and would make the saved width differ from figsize again (5.5 in -> 5.40 in),
# which is what made the scale factors hard to reason about in the first place.
# ---------------------------------------------------------------------------
W_FULL = 5.50                        # full text width (2 pt of slack)
W_MED = 4.40                         # the narrower standard slot


# ---------------------------------------------------------------------------
# Palette checker. The comment above says "do not eyeball-substitute"; this is
# what makes that enforceable rather than aspirational. Run:
#     python3 systematic_eval/scripts/thesis_figures.py --check-palette
# Pure stdlib on purpose, so the check has no dependency the figures do not
# already have and cannot be skipped for being inconvenient to run.
# ---------------------------------------------------------------------------
# Pairs a reader must actually be able to separate, with the floor each must
# clear under normal vision AND under simulated protanopia and deuteranopia.
# Sequential-ramp steps are deliberately absent: adjacent steps of an ordinal
# ramp are supposed to be close, and bar length carries the quantity anyway.
#
# The CROSS-FIGURE block is the one that matters most, and the one whose
# absence let the original orange contrast pair through: separation inside a
# figure is not sufficient when two figures sit on facing pages and the reader
# carries an association from one to the next. Orange passed every within-
# figure check at dE 53.5 and still collided with the status red at dE 8.8.
PALETTE_PAIRS = [
    # within a figure
    ("C1", "C2", 20, "CONTRAST pair (fig_engines, fig_joinorder)"),
    ("GOOD", "CRIT", 20, "STATUS pair (fig_engines_composition, fig_oracle_gap)"),
    ("GOOD", "NEUTRAL", 20, "STATUS vs neutral segment"),
    ("CRIT", "NEUTRAL", 20, "STATUS vs neutral segment"),
    ("DATA", "DATA_PALE", 20, "IMDB highlight vs the other workloads"),
    ("REF", "DATA_PALE", 20, "AGGREGATE line on the generality bars"),
    ("REF", "BASE", 20, "AGGREGATE line vs the no-change baseline"),
    ("REF", "C1", 20, "median line on the dark box (fig_joinorder)"),
    ("REF", "C2", 20, "median line on the light box (fig_joinorder)"),
    # across adjacent figures: 5.8 CONTRAST faces 5.9 STATUS
    ("C1", "GOOD", 20, "CROSS-FIGURE: contrast vs status (5.8 next to 5.9)"),
    ("C1", "CRIT", 20, "CROSS-FIGURE: contrast vs status (5.8 next to 5.9)"),
    # Floor 15, not 20, and this is the one deliberate exception in the list.
    # The pale blue sits at dE 16.9 from the status green. It is the best any
    # available colour reaches (the exhaustive search over all 48 TUDa colours
    # is summarised in the palette block), and the two differ in lightness by
    # 17 L* with the blue almost unsaturated, so a pale wash and a saturated
    # teal are not plausibly swapped. Raising this to 20 makes the list
    # unsatisfiable rather than making the figures better.
    ("C2", "GOOD", 15, "CROSS-FIGURE: contrast vs status (5.8 next to 5.9)"),
    ("C2", "CRIT", 20, "CROSS-FIGURE: contrast vs status (5.8 next to 5.9)"),
    # a light fill still has to be visible on the page at all
    ("C2", "SURFACE", 20, "light fill against the white page (bars, silhouette)"),
]


def _lab(hex_):
    def chan(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (chan(int(hex_.lstrip("#")[i:i + 2], 16)) for i in (0, 2, 4))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _de2000(c1, c2):
    import math
    L1, a1, b1 = c1
    L2, a2, b2 = c2
    C1, C2_ = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2_) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1 = math.degrees(math.atan2(b1, a1p)) % 360
    h2 = math.degrees(math.atan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dh = 0
    elif abs(h2 - h1) <= 180:
        dh = h2 - h1
    else:
        dh = h2 - h1 - 360 if h2 - h1 > 180 else h2 - h1 + 360
    dH = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh) / 2)
    Lb, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hb = h1 + h2
    elif abs(h1 - h2) <= 180:
        hb = (h1 + h2) / 2
    else:
        hb = (h1 + h2 + 360) / 2 if h1 + h2 < 360 else (h1 + h2 - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hb - 30))
         + 0.24 * math.cos(math.radians(2 * hb))
         + 0.32 * math.cos(math.radians(3 * hb + 6))
         - 0.20 * math.cos(math.radians(4 * hb - 63)))
    SL = 1 + 0.015 * (Lb - 50) ** 2 / math.sqrt(20 + (Lb - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = (-2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-((hb - 275) / 25) ** 2)))) if Cbp > 0 else 0
    return math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2
                     + RT * (dC / SC) * (dH / SH))


# Standard linear-RGB dichromacy matrices (Vienot/Brettel).
_CVD = {"protan": ((0.1121, 0.8853, -0.0005), (0.1127, 0.8897, -0.0001), (0.0045, 0.0085, 1.0)),
        "deutan": ((0.2920, 0.7054, -0.0003), (0.2934, 0.7089, 0.0), (-0.0209, 0.4055, 0.6156))}


def _sim(hex_, kind):
    def chan(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def unchan(c):
        c = min(1.0, max(0.0, c))
        return (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055) * 255
    rgb = [chan(int(hex_.lstrip("#")[i:i + 2], 16)) for i in (0, 2, 4)]
    m = _CVD[kind]
    out = [sum(m[i][j] * rgb[j] for j in range(3)) for i in range(3)]
    return _lab("#%02X%02X%02X" % tuple(round(unchan(v)) for v in out))


def check_palette() -> int:
    """Print the separation of every pair that must hold. Returns an exit code."""
    tokens = {"DATA": DATA, "DATA_PALE": DATA_PALE,
              "C1": C1, "C2": C2, "GOOD": GOOD, "NEUTRAL": NEUTRAL, "CRIT": CRIT,
              "REF": REF, "BASE": BASE, "SURFACE": SURFACE}
    print(f"{'pair':26}{'normal':>8}{'protan':>8}{'deutan':>8}  floor  verdict")
    bad = 0
    for a, b, floor, why in PALETTE_PAIRS:
        la, lb = _lab(tokens[a]), _lab(tokens[b])
        n = _de2000(la, lb)
        p = _de2000(_sim(tokens[a], "protan"), _sim(tokens[b], "protan"))
        d = _de2000(_sim(tokens[a], "deutan"), _sim(tokens[b], "deutan"))
        ok = min(n, p, d) >= floor
        bad += 0 if ok else 1
        print(f"{a + '/' + b:26}{n:8.1f}{p:8.1f}{d:8.1f}{floor:7}  "
              f"{'ok' if ok else 'FAIL'}   {why}")
    print("\n" + (f"{bad} pair(s) below the floor" if bad else "all pairs clear the floor"))
    return 1 if bad else 0


# The chapter under active revision lives in masterthesis_writeup_latex, which
# is the tree the thesis is compiled from. masterthesis_workspace_latex is the
# older copy. Render into the writeup tree by default and pass --out explicitly
# when targeting the other one.
DEFAULT_OUT = os.path.normpath(
    os.path.join(tn.HERE, "..", "..", "..", "thesis", "masterthesis_writeup_latex", "img", "generated")
)


def set_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "legend.frameon": False,
    })


def save(fig, out: str, name: str, **tight_kw) -> None:
    # tight_kw is forwarded to tight_layout, which is the only way to widen the
    # gutter between subplots: tight_layout recomputes the subplot params and
    # would silently discard a gridspec wspace set at creation time. Used by
    # fig_generality, which parks text in that gutter, and by
    # fig_engines_composition, which reserves room for a legend below the axes.
    fig.tight_layout(**tight_kw)
    path = os.path.join(out, name + ".pdf")
    # No bbox_inches="tight": see the GEOMETRY block. The saved width must equal
    # figsize, because figsize is the printed width. Anything drawn outside the
    # axes must be given room via tight_layout(rect=...) instead of by cropping.
    fig.savefig(path)
    # With the crop gone, anything drawn past the canvas is now silently cut
    # instead of being absorbed, and a clipped legend or title is easy to miss
    # in a 13-figure batch. Compare the tight bbox against the canvas and say so.
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    over = max(-tb.x0, -tb.y0, tb.x1 - w, tb.y1 - h)
    if over > 0.01:
        print(f"  WARNING {name}: content extends {over:.2f} in past the canvas; "
              f"give it room via tight_layout(rect=...) or shorten the labels")
    plt.close(fig)
    print("figure:", name + ".pdf")


def placeholder(out: str, name: str, reason: str, size=(W_MED, 2.6)) -> None:
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=ax.transAxes,
                               fill=False, ls=(0, (6, 4)), ec=INK2, lw=1.2))
    ax.text(0.5, 0.56, "Figure pending", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color=INK2)
    ax.text(0.5, 0.40, reason, ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=INK2, wrap=True)
    save(fig, out, name)


def on_color(bg: str) -> str:
    """Ink or white for a label sitting ON a filled patch, by WCAG contrast.

    Hardcoding "white on every coloured segment" broke when the status fills
    moved to the TUDa palette: white on TUDa-3b is only 3.4:1, under the 4.5:1
    floor, while white on TUDa-9b is 4.8:1. The two segments therefore need
    different label inks, and picking that by eye is exactly the kind of thing
    that silently rots the next time a fill changes.
    """
    def chan(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)
    return "white" if 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.05 else INK


def write_table(out: str, name: str, body: str) -> None:
    with open(os.path.join(out, name + ".tex"), "w") as fh:
        fh.write(body)
    print("table: ", name + ".tex")


def pending_table(out: str, name: str, reason: str) -> None:
    write_table(out, name,
                "% pending: " + reason + "\n"
                "\\begin{tabular}{l}\n\\toprule\n"
                "\\emph{pending: " + reason + "} \\\\\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Shared per-query view (mirror of thesis_numbers unified population).
# ---------------------------------------------------------------------------
def unified_rows(key: str):
    o, p, t = tn.load_oracle(key), tn.load_optimizer(key), tn.load_transfer(key)
    if o is None or p is None or t is None:
        return None
    # sorted, not a bare set: set iteration order over strings varies per process
    # and would reorder the scatter points, so the figure would never reproduce.
    pop = sorted(set(t) | set(o) | set(p))
    return [(q,
             o[q]["pct"] if q in o else 0.0,
             p[q]["pct"] if q in p else 0.0) for q in pop]


def funnel_counts():
    """The genuine nested chain from candidate to applied rule.

    Returns (candidates, validated, after_merge, applied).

    The predicate-injecting count (647) is NOT part of this chain and is not
    returned: injection and soundness are independent properties, and 100 of the
    base-table-sound rules do not inject into their source query. Drawing them as
    successive funnel stages asserts a nesting the data do not have.

    The first stage is NOT len(result.json). A response that fails to parse still
    writes one `_error` entry that carries no rule, so the raw file length
    overstates the candidate count by 19 on this run. It is also a post-dedup
    figure: the model emitted 1297 rules to leave these 1023 behind. See the
    comment in thesis_numbers.do_funnel().
    """
    base = os.path.join(tn.TRANSFER, tn.RUNS["DDB"])

    def valid(v):
        bt = v.get("base_table_validation")
        return isinstance(bt, dict) and bt.get("all_valid") is True

    def load(fn):
        path = os.path.join(base, fn)
        return json.load(open(path)) if os.path.isfile(path) else None

    gen_d = load("result.json")
    if gen_d is None:
        return None
    ref_d = load("result2.json") or {}

    validated = [v for v in list(gen_d.values()) + list(ref_d.values()) if valid(v)]

    pool_path = os.path.join(base, "cost_aggregate_input.json")
    after_merge = None
    if os.path.isfile(pool_path):
        after_merge = len(json.load(open(pool_path)).get("rules", []))

    t = tn.load_transfer("DDB")
    applied = len({r["name"] for e in t.values() for r in e.get("rules", [])}) if t else None
    candidates = len(gen_d) - sum(1 for k in gen_d if k.endswith("_error"))
    return candidates, len(validated), after_merge, applied


# ===========================================================================
# PART 1 FIGURES
# ===========================================================================
def fig_funnel(out):
    fc = funnel_counts()
    if fc is None:
        return placeholder(out, "fig_funnel", "generation outputs (transfer_data) not found")
    candidates, validated, after_merge, applied = fc
    stages = [("distinct candidates", candidates),
              ("base-table sound", validated),
              ("distinct after merging", after_merge or 0),
              ("applied to a query", applied or 0)]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    fig, ax = plt.subplots(figsize=(W_MED, 2.6))
    y = range(len(stages))
    # ORDINAL: the stages are a nested chain, so the ramp is ordered, not
    # categorical. Adjacent ramp steps are deliberately close (dE ~10); the
    # dE 20 categorical floor does not apply, and bar length carries the
    # quantity anyway.
    ax.barh(list(y), vals, color=SEQ[::-1], height=0.62)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlabel("rules / candidates")
    ax.set_xlim(0, max(vals) * 1.12)
    save(fig, out, "fig_funnel")


def fig_novelty(out):
    """What each additional generation round newly discovers, in absolute counts.

    Two panels, not two lines in one panel: "new sound rules" and "newly covered
    queries" are different kinds of thing on different scales (tens of rules
    against single-digit queries by the fourth round), and putting them on one
    axis would either flatten the query panel to nothing or require a second
    y-axis, which is never correct. Small multiples are the sanctioned way out.

    Bars rather than a line, because the quantity is a magnitude per discrete
    round rather than a trajectory through a continuum. The whisker is the min
    and max over all 24 round orders. It is taken on the MARGINAL per ordering,
    not derived from the spread of the cumulative curve, since a lucky ordering
    can have a small k-th round and a large running total.
    """
    curves = tn.novelty_curves()
    if not curves:
        return placeholder(out, "fig_novelty", "LLM cache or DDB run not found")

    # DATA role: one hue for both panels, and deliberately NOT the contrast
    # pair. The panels do measure different things, rules against queries, on
    # different scales, so there is a real case for separating them by colour.
    # The contrast pair is the wrong instrument for it: it is a dark/light pair
    # whose whole meaning is "the dark one bounds the light one", and neither
    # of these two quantities bounds the other. Using it here would assert an
    # ordering that does not exist, in exchange for a distinction the two y
    # labels and the two y scales already make. The panels also carry no shared
    # legend and no shared axis, so nothing invites reading one bar against the
    # other in the first place.
    panels = [
        ("sound", "newly discovered sound rules", DATA),
        ("queries", "newly covered queries", DATA),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    for ax, (key, ylabel, color) in zip(axes, panels):
        rows = curves[key]["marg"]
        ks = list(range(1, len(rows) + 1))
        m = [r[0] for r in rows]
        lo = [r[0] - r[1] for r in rows]
        hi = [r[2] - r[0] for r in rows]
        ax.bar(ks, m, width=0.62, color=color, zorder=3)
        ax.errorbar(ks, m, yerr=[lo, hi], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2.5, zorder=4)
        top = max(r[2] for r in rows)
        for k, v in zip(ks, m):
            ax.text(k, v + top * 0.06, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
        ax.set_xticks(ks)
        ax.set_xlabel("generation round")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, top * 1.22)
        ax.grid(axis="x", visible=False)
    save(fig, out, "fig_novelty")


def fig_transfer(out):
    t = tn.load_transfer("DDB")
    if t is None:
        return placeholder(out, "fig_transfer", "DDB transfer data not found")
    fan = {}
    for q, e in t.items():
        for r in e.get("rules", []):
            fan.setdefault(r["name"], set()).add(q)
    sizes = [len(s) for s in fan.values()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"width_ratios": [3, 2]})
    mx = max(sizes)
    a1.hist(sizes, bins=range(1, mx + 2), color=DATA, align="left", rwidth=0.85)
    a1.set_xlabel("queries a rule is applied to (fan-out)")
    a1.set_ylabel("number of rules")
    a1.set_title("Cross-query fan-out", fontsize=9)
    # winning-subset transfer share (recomputed here from oracle_stats winning_rules).
    # Both bars are transfer statements. The multi-rule share is deliberately not
    # shown here: combination is composition, not transfer, and mixing the two
    # under one title claims a relation the data do not carry.
    path = os.path.join(tn._run_dir("DDB"), "oracle_stats.csv")
    import csv
    n = trans = only_trans = 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            target = r["prefix"].split()[0]
            origins = tn._origins(r["winning_rules"])
            n += 1
            if any(o != target for o in origins):
                trans += 1
            if origins and all(o != target for o in origins):
                only_trans += 1
    bars = [("uses a\ntransferred rule", 100 * trans / n),
            ("transferred\nrules only", 100 * only_trans / n)]
    # One hue, not two. The second bar is a SUBSET of the first (wins whose
    # rules are all transferred are a subset of wins that use any transferred
    # rule), so they are one quantity at two thresholds and not a contrast pair.
    # The previous green here also spent the reserved status hue on a plain
    # category, which is exactly what the role split exists to prevent.
    a2.bar([b[0] for b in bars], [b[1] for b in bars], color=DATA, width=0.6)
    a2.set_ylim(0, 100)
    a2.set_ylabel("share of wins (%)")
    a2.set_title("Transfer in the winning subset", fontsize=9)
    for i, b in enumerate(bars):
        a2.text(i, b[1] + 2, f"{b[1]:.1f}%", ha="center", fontsize=9, color=INK)
    save(fig, out, "fig_transfer")


# ---------------------------------------------------------------------------
# Oracle ceiling, two figures on two explicit denominators.
#
# Both are drawn as a filled silhouette of the sorted per-query speedup factor
# rather than as individual bars: at 113 queries a bar is under one point wide,
# and the filled area reads the same while allowing the log y-axis the ratios
# require (a 6x win and a 1.05x win cannot share a linear axis legibly).
# The fill baseline is 1.0, which is "no change", so area = benefit.
# ---------------------------------------------------------------------------
# Canonical chapter order (tn.ENGINE_SELECTORS): the two analytical engines,
# then the row-store under which the ranking-signal axis is varied.
#
# No per-engine hue. Each panel names its engine in the title, so colour was a
# redundant channel here, and spending three scarce distinguishable hues on it
# forced the status and contrast pairs into collisions elsewhere. It also
# actively misled: the three engine hues had L* 50 / 64 / 32, and because these
# panels are FILLED silhouettes, lightness reads as area reads as magnitude.
# Postgres looked heaviest and Umbra lightest while Umbra has the highest
# ceiling (1.239x), so the colour fought the finding. One fill for all three
# panels also gives the small multiples the equal visual weight they need to be
# compared, which is the entire point of drawing them as small multiples.
CEILING_ORDER = [("DDB", "DuckDB"), ("UMB", "Umbra"), ("PG", "Postgres")]


def _ceiling_panel(ax, factors, geo, geo_label, n_label, fill):
    """Filled sorted-speedup silhouette with the geometric mean annotated."""
    x = range(len(factors))
    ax.fill_between(x, 1.0, factors, step="mid", color=fill, lw=0)
    # The 1.0 baseline is chrome: it is the frame the fill is measured from, not
    # a result. It recedes so that the AGGREGATE line, which IS a result, is the
    # only emphatic horizontal in the panel. Previously both were mid-weight and
    # competed, which is why the mean had to be chromatic to win.
    ax.axhline(1.0, color=BASE, lw=0.8)
    ax.axhline(geo, color=REF, ls=(0, (4, 2.5)), lw=1.1)
    ax.set_yscale("log")
    ax.set_yticks([1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0])
    ax.set_yticklabels(["1.0x", "1.25x", "1.5x", "2x", "3x", "5x", "8x"])
    # The log locator otherwise prints its own minor labels (6e0, 5e0) straight
    # through the custom major labels.
    ax.minorticks_off()
    ax.set_ylim(0.99, 9.0)
    ax.set_xlim(-0.5, len(factors) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.text(0.97, 0.93, f"{geo_label} {geo:.3f}x", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=REF)
    ax.set_xlabel(n_label, fontsize=8)


def fig_oracle_ceiling(out):
    """Figure 1 of 2: the ceiling ON THE QUERIES THE RULES TOUCH.

    Answers "how large is the win where there is one". The workload runtime in
    each panel title is the load-bearing detail: Umbra runs the same benchmark
    roughly eight times faster than DuckDB and still offers the largest
    headroom, so the benefit does not rest on a weak execution substrate."""
    cs = {k: tn.ceiling(k) for k, _ in CEILING_ORDER}
    if not any(cs.values()):
        return placeholder(out, "fig_oracle_ceiling", "no oracle runs found")
    # The runtime in the panel titles is a CROSS-ENGINE comparison, so it must
    # come from the shared population. Each engine's own total is in
    # tab_ceiling, next to that engine's own n. All three engines now measure
    # the full 113 queries, so the shared population equals the full workload;
    # ceiling_common() stays as a safeguard in case a future run measures fewer.
    shared = tn.ceiling_common()
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.5), sharey=True)
    for ax, (key, label) in zip(axes, CEILING_ORDER):
        c = cs[key]
        if c is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color=INK2)
            continue
        o = tn.load_oracle(key)
        factors = sorted((v["orig"] / v["improved"] for v in o.values()), reverse=True)
        # C1, the dark half of the contrast pair: this row is the bound that
        # the row below is measured against. See fig_oracle_workload.
        _ceiling_panel(ax, factors, c["geo_improved"], "geo. mean",
                       f"{c['n_improved']} improved queries", C1)
        # One decimal, matching tab_ceiling: rounding 8.55 to "9 s" here while
        # the table prints "8.5" reads as two different measurements.
        secs = (shared or {}).get(key, c["total_original_s"])
        # Two lines: at true scale one third of the text width does not hold
        # "Postgres, 151.5 s workload" on one line and the three titles collide.
        ax.set_title(f"{label}\n{secs:.1f}\u2009s workload", fontsize=8.5)
    axes[0].set_ylabel("oracle speedup factor")
    save(fig, out, "fig_oracle_ceiling")


def fig_oracle_workload(out):
    """Figure 2 of 2: the same ceiling spread over the FULL workload.

    Every measured query enters, including the ones no rule improves, which
    enter at factor 1.0 and are drawn gray. This is the honest headline
    denominator and always yields the smaller number; showing it next to the
    improved-only view is what keeps the improved-only view defensible."""
    cs = {k: tn.ceiling(k) for k, _ in CEILING_ORDER}
    if not any(cs.values()):
        return placeholder(out, "fig_oracle_workload", "no oracle runs found")
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.5), sharey=True)
    for ax, (key, label) in zip(axes, CEILING_ORDER):
        c = cs[key]
        if c is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color=INK2)
            continue
        o = tn.load_oracle(key)
        rt = tn.load_all_runtimes(key) or {}
        improved = {tn._basename(q): v["orig"] / v["improved"] for q, v in o.items()}
        n_flat = len(set(rt) | set(improved)) - len(improved)
        factors = sorted(improved.values(), reverse=True) + [1.0] * n_flat
        # C2, the light half. The two figures are stacked as one float
        # (Figure 5.3) and are the SAME measurement on two denominators, with
        # geo_workload <= geo_improved by construction, so the pair convention
        # applies exactly: dark bounds light. Colour therefore varies along the
        # denominator axis, which is what the float actually varies, and not
        # along the engine axis, which the panel titles already name.
        _ceiling_panel(ax, factors, c["geo_workload"], "geo. mean",
                       f"{c['n_improved']} of {c['n_workload']} improved", C2)
        # The untouched tail, shaded: those queries enter the mean at 1.0.
        # Left unlabelled on purpose. The caption of Figure 5.3 already names
        # the shaded band, and an in-panel label would sit in only one of the
        # three panels while the band appears in all three.
        ax.axvspan(len(improved) - 0.5, len(factors) - 0.5, color=TUDA_0C,
                   alpha=0.20, lw=0)
        ax.set_title(label, fontsize=9)
    axes[0].set_ylabel("oracle speedup factor")
    save(fig, out, "fig_oracle_workload")


def fig_worldknowledge(out):
    path = tn.wk_eval_path()
    if not os.path.isfile(path):
        return placeholder(out, "fig_worldknowledge", "wk_eval.json not found")
    rules = json.load(open(path))["rules"]
    scores = [r["verdict"]["score"] for r in rules if r.get("verdict", {}).get("score")]
    pairs = [(r["verdict"]["score"], r["agg"]["median_percent_saved"]) for r in rules
             if r.get("verdict", {}).get("score") and isinstance(r.get("agg", {}).get("median_percent_saved"), (int, float))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"width_ratios": [2, 3]})
    counts = [scores.count(s) for s in range(1, 6)]
    # ORDINAL, keyed to the score and not to the count. This is a deliberate
    # redundant encoding of the x axis, which is normally noise, but the same
    # 1-5 score is printed as a colour-ramp badge (\wkscore) in the rule table a
    # page later. Before, the badge ran on an orange ramp and this panel on flat
    # blue, so one variable had two visual languages on facing pages. The ramp
    # here is the same one the badge now uses, so the two match on sight.
    a1.bar(range(1, 6), counts, color=SEQ, width=0.7)
    a1.set_xlabel("world-knowledge score")
    a1.set_ylabel("number of rules")
    a1.set_xticks(range(1, 6))
    a1.set_title("Judge score distribution", fontsize=9)
    a1.grid(axis="x", visible=False)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    a2.axhline(0, color=BASE, lw=0.8)
    # Deterministic jitter. Python salts str hashing per process, so hash() here
    # would move every point on every run and the figure would never reproduce.
    a2.scatter([x + ((i * 37) % 100 - 50) / 400 for i, x in enumerate(xs)], ys,
               s=18, color=DATA, alpha=0.5, edgecolor="white", linewidth=0.4)
    # Per-bucket median (bar) and mean (open diamond): the bulk sits near zero
    # across scores, but the large wins concentrate in scores 4 and 5. Both are
    # AGGREGATE markers, so both are achromatic and separated by shape rather
    # than by hue; the previous magenta median read as a third data category.
    for s in range(1, 6):
        sv = [p[1] for p in pairs if p[0] == s]
        if not sv:
            continue
        a2.plot([s - 0.3, s + 0.3], [median(sv), median(sv)], color=REF, lw=2.2, zorder=3)
        a2.scatter([s], [mean(sv)], marker="D", s=22, facecolor="white",
                   edgecolor=REF, linewidth=1.1, zorder=4)
    a2.set_xlabel("world-knowledge score")
    a2.set_ylabel("single-rule speedup (%)")
    a2.set_xticks(range(1, 6))
    # Neutral title on purpose: the score does not predict the speedup, it only
    # admits the extreme tail. A title claiming a relation would overstate it.
    a2.set_title("Score vs single-rule speedup", fontsize=9)
    save(fig, out, "fig_worldknowledge")


def fig_generality(out):
    """The oracle ceiling on both denominators, one bar per workload.

    Deliberately parallel to fig_oracle_ceiling / fig_oracle_workload: those
    show the two denominators for IMDB across three engines, this shows the
    same two denominators across twenty workloads on DuckDB. IMDB is drawn in
    the DuckDB hue with a bold tick label and every other workload in the pale
    step of the same ramp, so the question the figure exists to answer, whether
    IMDB is an outlier, is readable without consulting the numbers.

    Both panels share one logarithmic x axis anchored at 1.0, which is not
    cosmetic. Because the unimproved queries enter the full-workload geometric
    mean at exactly 1.0,

        log(geo_workload) = (n_improved / n_workload) * log(geo_improved),

    so on a log axis each right-hand bar is exactly the coverage fraction of its
    left-hand bar. The dilution between the two denominators IS the shortening,
    which is why the axis must be shared and must be log."""
    rows = tn.generality_rows()
    if not rows:
        return placeholder(out, "fig_generality", "no generality runs found")

    is_imdb = [r["key"] == tn.GEN_IMDB_KEY for r in rows]
    names = ["IMDB (JOB)" if m else r["name"].replace("_", "‑")
             for r, m in zip(rows, is_imdb)]
    colors = [DATA if m else DATA_PALE for m in is_imdb]
    y = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 0.23 * len(rows) + 1.3), sharey=True)
    panels = (
        (axes[0], "geo_improved", "improved queries", "median"),
        (axes[1], "geo_workload", "full workload", "median"),
    )
    hi = max(r["geo_improved"] for r in rows)
    for ax, field, title, med_label in panels:
        vals = [r[field] for r in rows]
        ax.barh(y, [v - 1.0 for v in vals], left=1.0, height=0.72,
                color=colors, lw=0)
        med = median(vals)
        ax.axvline(med, color=REF, ls=(0, (4, 2.5)), lw=1.1, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(1.0, hi * 1.06)
        ax.set_xticks([1.0, 1.1, 1.25, 1.5, 1.75])
        ax.set_xticklabels(["1.0x", "1.1x", "1.25x", "1.5x", "1.75x"], fontsize=7.5)
        ax.minorticks_off()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("oracle speedup factor", fontsize=8)
        ax.text(0.97, 0.02, f"{med_label} {med:.3f}x", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=REF)

    # Coverage annotates the ROW, not either bar: it is the exponent that turns
    # the left bar into the right one. It is parked in the gutter BETWEEN the
    # panels rather than in the right margin of the second one. The numerator is
    # the population of the left panel and the denominator that of the right, so
    # sitting against the right edge of the full-workload panel misread it as
    # that panel's own label. The gutter touches both panels and is the placement
    # the quantity actually has. It cannot go inside the left panel: that panel's
    # longest bar (airline) runs to within a few percent of its right edge, and
    # the xlim is shared with the right panel and must stay so, since the
    # coverage-fraction identity below is only readable on one common log axis.
    for i, r in enumerate(rows):
        axes[0].text(1.19, i, f"{r['n_improved']}/{r['n_workload']}",
                     transform=axes[0].get_yaxis_transform(),
                     va="center", ha="right", fontsize=6.5, color=INK2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7.5)
    for lbl, m in zip(axes[0].get_yticklabels(), is_imdb):
        if m:
            # Colour alone must not be the only channel identifying IMDB.
            lbl.set_fontweight("bold")
            lbl.set_color(DATA)
    axes[0].invert_yaxis()
    # w_pad widens the gutter that carries the coverage annotation. The default
    # leaves it about as wide as the text itself, so the numbers would touch the
    # right panel's spine.
    save(fig, out, "fig_generality", w_pad=3.0)


# ===========================================================================
# PART 2 FIGURES
# ===========================================================================
def fig_oracle_gap(out):
    rows = unified_rows("DDB")
    if rows is None:
        return placeholder(out, "fig_oracle_gap", "DDB oracle/optimizer data not found")
    ceil = [c for _, c, _ in rows]
    real = [r for _, _, r in rows]
    fig, ax = plt.subplots(figsize=(W_MED, 3.4))
    lim_lo = min(min(real), 0) - 5
    lim_hi = max(ceil) + 5
    # y=x is an AGGREGATE reference, so it shares the dash and the ink of every
    # other reference line in the chapter; the 0 line is chrome and recedes.
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=REF, lw=1.0,
            ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(0, color=BASE, lw=0.8)
    # STATUS role. Sign is also carried by position relative to the 0 line, so
    # the encoding stays redundant, but the pair is now separable under
    # deuteranopia as well (dE 31.9 against the previous 11.8).
    cols = [CRIT if r < 0 else (GOOD if r > 0 else BASE) for r in real]
    ax.scatter(ceil, real, s=26, c=cols, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("oracle ceiling (%)")
    ax.set_ylabel("selector realized (%)")
    ax.set_title("Oracle vs selector: realized falls below the ceiling", fontsize=9)
    ax.text(lim_hi, lim_hi, " y=x (fully realized)", fontsize=7, color=REF, va="bottom", ha="right")
    save(fig, out, "fig_oracle_gap")


def _clean_pairs(free_key: str, pin_key: str):
    """Free/pinned pairs restricted to queries whose cost-winning subset is
    IDENTICAL in both runs. Mirrors thesis_numbers.do_attribution: where pinning
    moved the cost argmin, the difference confounds join order with a different
    predicate set, so those queries cannot enter the attribution."""
    free, pin = tn.load_optimizer(free_key), tn.load_optimizer(pin_key)
    if free is None or pin is None:
        return None, None
    wf, wp = tn._winning_subsets(free_key), tn._winning_subsets(pin_key)
    clean = [q for q in free if q in pin and wf.get(q) == wp.get(q)]
    conf = [q for q in free if q in pin and wf.get(q) != wp.get(q)]
    return ([(q, free[q]["pct"], pin[q]["pct"]) for q in clean],
            [(q, free[q]["pct"], pin[q]["pct"]) for q in conf])


def fig_joinorder(out):
    """Two engines side by side. Both panels use the CLEAN population only, so
    the bars are an attribution and not a mixture of two mechanisms."""
    engines = [("DuckDB", "DDB", "DDB_JP"), ("Postgres", "PG", "PG_JP")]
    data = []
    for label, fk, pk in engines:
        clean, conf = _clean_pairs(fk, pk)
        if clean is None:
            return placeholder(out, "fig_joinorder",
                               f"{fk} / {pk} optimizer data not found")
        data.append((label, clean, conf))

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.9),
                             gridspec_kw={"width_ratios": [2.2, 2.2, 2.4]})
    for ax, (label, clean, _) in zip(axes[:2], data):
        f = [t[1] for t in clean]
        p = [t[2] for t in clean]
        bp = ax.boxplot([f, p], orientation="vertical", widths=0.55, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2,
                                        alpha=0.4, markeredgecolor="none"))
        # CONTRAST role: free vs pinned is a paired comparison and colour is the
        # only carrier, which is what the pair is reserved for. Dark is free,
        # the unconstrained case, and light is the pinned restriction of it.
        #
        # Full opacity, deliberately. These boxes used to be drawn at alpha=0.55,
        # which an orange/blue pair survived because its separation came from the
        # hue. This pair separates by LIGHTNESS, and blending both toward white
        # compresses exactly that: the pair drops from dE 34.6 to 17.0 and the
        # light box falls to dE 12.4 against the white page, which is close to
        # invisible in print. Never reintroduce alpha here.
        for patch, col in zip(bp["boxes"], [C1, C2]):
            patch.set_facecolor(col)
        for med in bp["medians"]:
            med.set_color(REF)
        ax.set_xticks([1, 2], ["free", "pinned"])
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_title(f"{label} (n={len(clean)})", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("selector speedup (%)")

    ax = axes[2]
    import numpy as np
    x = np.arange(len(data))
    w = 0.36
    mf = [sum(-t[1] for t in c if t[1] < 0) for _, c, _ in data]
    mp = [sum(-t[2] for t in c if t[2] < 0) for _, c, _ in data]
    ax.bar(x - w / 2, mf, w, color=C1, label="free")
    ax.bar(x + w / 2, mp, w, color=C2, label="pinned")
    for i in range(len(data)):
        if mf[i] > 0:
            ax.text(x[i], max(mf[i], mp[i]), f"-{100*(1-mp[i]/mf[i]):.1f}%",
                    va="bottom", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [d[0] for d in data])
    ax.set_ylabel("regression mass (summed pp)")
    ax.set_title("Removed by pinning", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, out, "fig_joinorder")


# Selector labels for the Part 2 figures. The two Postgres runs are named by
# their RANKING SIGNAL, because in Part 2 a "selector" is named by the estimate
# it ranks by and the engine is only the substrate it runs on. Postgres keeps
# planning and executing under its own cost model in both runs, so labelling
# the pair by the cost model would misstate what is varied. Leaving the native
# run labelled just "Postgres" would in addition leave one of the two factor
# levels unnamed and would reuse the Part 1 meaning of "Postgres" (the engine)
# for something narrower.
ENGINE_LABELS = {"DDB": "DuckDB", "UMB": "Umbra",
                 "PG": "Postgres\nnative", "LRN": "Postgres\nZeroShot"}

# x positions: the two Postgres signals are pushed together and away from
# the engines, so the grouping is visible before any label is read. This is the
# one-bar-per-selector layout; fig_engines needs its own (MEANS_X below) because
# it draws one shared ceiling bar for the Postgres pair instead of two.
ENGINE_X = [0.0, 1.0, 2.15, 2.95]

# fig_engines layout. DuckDB and Umbra are ceiling/realized pairs as usual; the
# Postgres group is a TRIO, one shared ceiling followed by the two realized bars,
# so its ticks name the bars rather than the selectors and the band note carries
# the engine name.
MEANS_W = 0.40
# The Postgres trio is spaced 0.60 rather than 0.50 apart: at true scale its
# tick labels are set at 8 pt for real, and at 0.50 "ceiling (shared)", "native"
# and "ZeroShot" overlap each other.
MEANS_X = {"DDB": 0.0, "UMB": 1.0, "PG_CEIL": 2.30, "PG": 2.90, "LRN": 3.50}
MEANS_TICKS = [("DDB", "DuckDB"), ("UMB", "Umbra"), ("PG_CEIL", "ceiling\n(shared)"),
               ("PG", "native"), ("LRN", "ZeroShot")]


def _engine_rows():
    selectors = list(tn.ENGINE_SELECTORS)
    rows = []
    for k in selectors:
        m = tn._unified(k)
        rows.append(None if isinstance(m, tn.Missing) else m)
    return selectors, ENGINE_LABELS, rows, list(ENGINE_X)


def _postgres_band(ax, lo, hi, xmin, note=None):
    """Shade the Postgres positions as one group. The band says what the
    ordering alone cannot: those bars share an engine and differ only in the
    ranking signal, whereas the first two groups differ in the engine."""
    # Achromatic: the band is a grouping device, not a category, so it must not
    # read as a fifth colour with a meaning of its own.
    ax.axvspan(lo, hi, color=TUDA_0C, alpha=0.13, lw=0, zorder=0)
    if note:
        ax.text((lo + hi) / 2, 0.985, note,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=INK2)
    ax.set_xlim(xmin, hi + 0.05)


def fig_engines(out):
    """Oracle ceiling versus selector-realized speedup per selector, as the
    GEOMETRIC MEAN of the per-query speedup factor over the full workload.
    Isolates the ceiling-minus-realized gap, which is the chapter headline. The
    composition of the population (improved / neutral / regressed / declined),
    which no central aggregate shows, is the separate fig_engines_composition;
    the two are meant to be read together but stand as their own figures.

    STATISTIC. This figure used to plot the arithmetic mean of percent_saved and
    must not go back to it. percent_saved is a ratio, and 5.1 fixes the
    geometric mean as the aggregate for ratios; averaging a quantity that is
    bounded at +100 and unbounded below prices a 6x speedup and a 6x slowdown as
    +83 and -500, which biases the aggregate against the rewrite and lets one
    cheap query decide its sign. It did: on the old scale the ZeroShot bar read
    -1.1 and turned positive on removing a single 6.6 ms query, whereas the
    geometric mean of that selector is 1.008x and moves by 0.01 on the same
    removal.

    DENOMINATOR. Full workload (113 on JOB), NOT the 86-query population that
    fig_engines_composition uses. RQ4 asks how much of *that* ceiling a selector
    realizes, and the ceiling the chapter states is Table 5.1's geo_workload; on
    this denominator the ceiling bars reproduce it exactly (1.089 / 1.125 /
    1.061), so the figure can be laid against that table with no conversion. On
    any other denominator the reader meets a ceiling value stated nowhere else
    and tries, correctly and fruitlessly, to reconcile it. The two denominators
    are one quantity: geo@86 = geo@113 ** (113/86).

    AXIS. Linear, anchored at 1.0. fig_oracle_ceiling uses a log axis because it
    spans 1.0x to 8x, where equal multiplicative steps must occupy equal
    distance. Here the range is 0.98x to 1.13x, over which log and linear differ
    by under one percent of the axis, so a log axis would buy nothing and cost
    legibility. Bars are drawn from the 1.0 baseline, so a bar below the line is
    a selector that leaves the workload slower than it found it.

    The Postgres pair gets ONE ceiling bar, not two. The native and the ZeroShot
    run measure the same oracle sweep on the same substrate, the same data and
    the same rule pool, so their ceilings are one quantity measured twice. On
    the factor scale those two measurements agree to 1.061x against 1.055x, half
    a percent, a good deal tighter than the 4.9 against 4.3 of the old
    arithmetic scale and further reason to draw one bar. The ceiling shown is
    the native run's, and only the realized bars, which is where the two ranking
    signals actually differ, are drawn per selector."""
    selectors, _labels, rows, _ = _engine_rows()
    if not any(r is not None for r in rows):
        return placeholder(out, "fig_engines", "no engine runs found")
    m = dict(zip(selectors, rows))
    x, w = MEANS_X, MEANS_W
    # W_FULL, and fig_engines_composition matches it: the two are companions
    # (see the docstring) and reading them together only works if the selectors
    # sit at the same size on both. The five tick groups also need the width.
    fig, a1 = plt.subplots(figsize=(W_FULL, 3.0))
    _postgres_band(a1, x["PG_CEIL"] - w / 2 - 0.25, x["LRN"] + w / 2 + 0.25,
                   xmin=x["DDB"] - 0.75,
                   note="Postgres: one engine, one ceiling,\ntwo ranking signals")
    # 1.0x is the no-change baseline and every bar is measured from it.
    a1.axhline(1.0, color=BASE, lw=0.9)

    def bar(xc, value, color, label=""):
        if value is None:  # run absent: keep the slot visible as pending
            a1.bar(xc, 0.02, w, bottom=1.0, color="none", edgecolor=INK2,
                   hatch="///", lw=0.8)
            a1.text(xc, 1.024, "pending", ha="center", va="bottom",
                    fontsize=7, color=INK2)
            return
        # bottom=1.0 with a signed height, so a factor below 1.0 draws downward
        # from the baseline instead of upward from zero.
        a1.bar(xc, value - 1.0, w, bottom=1.0, color=color, label=label)
        up = value >= 1.0
        a1.text(xc, value + (0.004 if up else -0.004), f"{value:.3f}",
                ha="center", va="bottom" if up else "top", fontsize=7, color=INK)

    # CONTRAST role. Ceiling vs realized is the paired comparison the figure
    # exists to show, and the engine is already named by the tick labels, so
    # colour is spent on the comparison rather than on the engine. Dark is the
    # ceiling, which bounds the light realized bar, matching free/pinned in
    # fig_joinorder: dark is always the bound, light always what is measured
    # against it.
    for k in ("DDB", "UMB"):  # engine axis: ceiling and realized as a pair
        r = m.get(k)
        bar(x[k] - w / 2, None if r is None else r["geo_ceiling_workload"], C1,
            "oracle ceiling" if k == "DDB" else "")
        bar(x[k] + w / 2, None if r is None else r["geo_realized_workload"], C2,
            "selector realized" if k == "DDB" else "")
    # Ranking-signal axis: shared ceiling first, then one realized bar each.
    ceil_src = m.get("PG") if m.get("PG") is not None else m.get("LRN")
    bar(x["PG_CEIL"], None if ceil_src is None else ceil_src["geo_ceiling_workload"],
        C1)
    if ceil_src is not None:
        # Carry the ceiling level across both realized bars: without it the one
        # bar reads as the ceiling of the selector standing next to it.
        a1.hlines(ceil_src["geo_ceiling_workload"],
                  x["PG_CEIL"] - w / 2, x["LRN"] + w / 2,
                  color=C1, ls=(0, (2, 2)), lw=0.9, zorder=3)
    for k in ("PG", "LRN"):
        r = m.get(k)
        bar(x[k], None if r is None else r["geo_realized_workload"], C2)

    a1.set_xticks([x[k] for k, _ in MEANS_TICKS], [t for _, t in MEANS_TICKS],
                  fontsize=8)
    # Limits from the data, so the baseline never sits at the very edge and the
    # value labels stay inside the axes.
    vals = [v for r in rows if r is not None
            for v in (r["geo_ceiling_workload"], r["geo_realized_workload"])] + [1.0]
    a1.set_ylim(min(vals) - 0.024, max(vals) + 0.030)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}x")
    a1.set_ylabel("speedup factor (geometric mean)")
    a1.set_title("Ceiling versus realized, full workload", fontsize=9)
    a1.grid(axis="x", visible=False)
    # Lower left: the only quadrant no bar and no group note occupies.
    a1.legend(loc="lower left", fontsize=8)
    save(fig, out, "fig_engines")


def fig_engines_composition(out):
    """Stacked composition of the unified 86-query population per selector:
    improved, decided-but-neutral, regressed, and declined (every rule
    abstained). The four segments are mutually exclusive and sum to the
    population, so the bars are directly comparable and the abstention share is
    read against the decisions the selector actually made. This replaces the old
    single-metric abstention panel and the per-selector table; every count shown
    here was previously in tab_engines."""
    selectors, labels, rows, x = _engine_rows()
    if not any(r is not None for r in rows):
        return placeholder(out, "fig_engines_composition", "no engine runs found")
    fig, ax = plt.subplots(figsize=(W_FULL, 2.9))   # matches fig_engines
    # the stack reaches the top, so no room for the note
    _postgres_band(ax, x[2] - 0.5, x[3] + 0.5, xmin=x[0] - 0.75)
    # STATUS role, the one figure that needs all four tokens. improved (good) |
    # neutral decided (grey) | regressed (bad) | declined (abstained, hatched so
    # it reads as "did not act"). improved and regressed are the two largest
    # segments, so this is the figure the deuteranopia check was run for.
    segs = [
        ("improved", GOOD, {}),
        ("neutral", NEUTRAL, {}),
        ("regressed", CRIT, {}),
        ("declined", "none", {"edgecolor": INK2, "hatch": "///", "linewidth": 0.8}),
    ]
    n_pop = None
    labeled = False
    for i, m in enumerate(rows):
        if m is None:
            ax.bar(x[i], 1, 0.62, color="none", edgecolor=INK2, hatch="xx", lw=0.8)
            ax.text(x[i], 1.5, "pending", ha="center", va="bottom", fontsize=7, color=INK2)
            continue
        n_pop = m["n_pop"]
        vals = {
            "improved": m["n_improved"],
            "neutral": m["n_decided"] - m["n_improved"] - m["n_regress"],
            "regressed": m["n_regress"],
            "declined": m["n_declined_all"],
        }
        bottom = 0.0
        for name, color, kw in segs:
            v = vals[name]
            ax.bar(x[i], v, 0.62, bottom=bottom, color=color,
                   label=(name if not labeled else "_nolegend_"), **kw)
            if v >= 2:  # skip labels on 0/1-count slivers
                tc = INK if name == "declined" else on_color(color)
                # The declined segment has no fill, only hatching, so its label
                # sits on the diagonal lines and loses contrast. Back it with an
                # opaque patch; the filled segments need none.
                tbox = (dict(facecolor=SURFACE, edgecolor="none",
                             boxstyle="square,pad=0.18")
                        if name == "declined" else None)
                ax.text(x[i], bottom + v / 2, str(v), ha="center", va="center",
                        fontsize=7, color=tc, bbox=tbox)
            bottom += v
        labeled = True
    ax.set_xticks(x, [labels[k] for k in selectors], fontsize=8)
    ax.set_ylabel(f"queries (of {n_pop})" if n_pop else "queries")
    ax.set_ylim(0, (n_pop or 86) * 1.02)
    ax.set_title("How each selector spends the population", fontsize=9)
    ax.grid(axis="x", visible=False)
    # FIGURE-level legend pinned to the bottom edge, not an axes-level legend
    # offset by bbox_to_anchor. An axes-relative offset has to be hand-tuned to
    # clear the tick labels, which here are two lines tall for Postgres, and
    # tight_layout does not know the legend exists either way: too small an
    # offset drops it onto the labels, too large leaves a dead band. Anchoring
    # to the figure and reserving exactly that band with rect makes the two
    # agree by construction. Note that the canvas-overflow guard in save() does
    # NOT catch the collision case, since the legend stays inside the canvas and
    # merely overlaps the labels.
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               ncol=4, fontsize=7.5, frameon=False)
    save(fig, out, "fig_engines_composition", rect=(0, 0.09, 1, 1))


def fig_crossengine(out):
    """Severe regressors per selector, INCLUDING the learned ranking signal.

    The three engines alone show only the weaker claim (failure sets do not
    follow the engine). The stronger claim the section actually makes is that
    they do not follow the engine even when the engine is held fixed, and that
    rests entirely on the Postgres pair. Leaving the fourth bar out left the
    section's headline without a figure."""
    sel = list(tn.ENGINE_SELECTORS)
    sets = {}
    for k in sel:
        opt = tn.load_optimizer(k)
        sets[k] = None if opt is None else {q for q, v in opt.items() if v["pct"] < -5}
    if any(v is None for v in sets.values()):
        return placeholder(out, "fig_crossengine", "one of the selector runs is missing")
    counts = [len(sets[k]) for k in sel]
    common3 = len(sets["DDB"] & sets["UMB"] & sets["PG"])
    shared_pair = len(sets["PG"] & sets["LRN"])
    x = list(ENGINE_X)
    fig, ax = plt.subplots(figsize=(W_MED, 2.7))
    _postgres_band(ax, x[2] - 0.5, x[3] + 0.5, xmin=x[0] - 0.75)
    # DATA role: every selector is named by its tick label, so one hue. The
    # ZeroShot bar keeps its hatch, which was always the channel that carried
    # "same engine, different ranking signal" and does not depend on colour.
    for i, k in enumerate(sel):
        ax.bar(x[i], counts[i], 0.6, color=DATA,
               hatch="///" if k == "LRN" else None,
               edgecolor="white" if k == "LRN" else "none", linewidth=0.0)
        ax.text(x[i], counts[i] + 0.1, str(counts[i]), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [ENGINE_LABELS[k] for k in sel], fontsize=8)
    ax.set_ylabel("severe regressions (< -5%)")
    ax.set_title("Failure sets barely overlap\n"
                 f"{common3} shared by all three engines, "
                 f"{shared_pair} by the two Postgres signals", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(counts) * 1.30 + 0.5)
    save(fig, out, "fig_crossengine")


def fig_casestudy_29a(out):
    src = os.path.join(tn.RUNS["DDB"], "plan_comparisons", "29a_NoMIN")
    placeholder(out, "fig_casestudy_29a",
                "manual crop of plan_comparisons/29a_NoMIN (free -73.7% vs pinned +0.9%)",
                size=(W_MED, 2.9))


# ===========================================================================
# TABLES (booktabs .tex includes)
# ===========================================================================
def _tex(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("&", "\\&"),
                 ("%", "\\%"), ("#", "\\#"), ("$", "\\$")):
        s = s.replace(a, b)
    return s


def tab_showcase(out):
    """Pinned selection, see tn.SHOWCASE_RULES for why it is not a ranking."""
    path = tn.wk_eval_path()
    if not os.path.isfile(path):
        return pending_table(out, "tab_showcase", "wk_eval.json not found")
    rules = {r["rule_name"]: r for r in json.load(open(path))["rules"]}

    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{lp{62mm}rrr}", "\\toprule",
             "Carries & World knowledge injected & Score & Applied to & Max (\\%) \\\\",
             "\\midrule"]
    for name, query, gist in tn.SHOWCASE_RULES:
        r = rules.get(name)
        if r is None:
            lines.append(f"{_tex(query)} & \\multicolumn{{4}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        agg = r.get("agg", {})
        lines.append(
            f"{_tex(query)} & {_tex(gist)} & {r['verdict']['score']} & "
            f"{agg.get('n_queries_fired')} & {agg.get('max_percent_saved', float('nan')):.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_showcase", "\n".join(lines) + "\n")


def tab_ceiling(out):
    """Oracle ceiling per engine on both denominators, with the pinned sibling
    inline. The pinned rows are what make the free rows defensible: on Postgres
    the two largest wins are plan reordering, not selectivity, and a ceiling
    quoted without that is a number the chapter itself later retracts. Umbra has
    no pinned row because the engine exposes no join-order pinning mechanism
    (execution.py raises for any engine other than duckdb/postgres)."""
    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "& Workload & \\multicolumn{3}{c}{Improved queries} & "
             "\\multicolumn{3}{c}{Full workload} & \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Engine & (s) & $n$ & Geo. mean & Rt-wtd. & "
             "$n$ & Geo. mean & Rt-wtd. & Max \\\\",
             "\\midrule"]
    for key, label, pin in tn.CEILING_ENGINES:
        for k, disp in ((key, label), *(((pin, "\\quad plan pinned"),) if pin else ())):
            c = tn.ceiling(k)
            if c is None:
                lines.append(f"{disp} & \\multicolumn{{8}}{{c}}{{\\emph{{pending}}}} \\\\")
                continue
            lines.append(
                f"{disp} & {c['total_original_s']:.1f} & {c['n_improved']} & "
                f"{c['geo_improved']:.3f}$\\times$ & {c['improved_factor']:.3f}$\\times$ & "
                f"{c['n_workload']} & {c['geo_workload']:.3f}$\\times$ & "
                f"{c['workload_factor']:.3f}$\\times$ & "
                f"{c['max_improved_pct']:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_ceiling", "\n".join(lines) + "\n")


def tab_joinorder(out):
    """Join-order attribution per engine, clean population only, with the
    confounded queries reported as a separate row so the exclusion is visible."""
    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Engine & $n$ clean & $n$ conf. & Regr. mass free (pp) & "
             "Regr. mass pinned (pp) & Removed \\\\", "\\midrule"]
    for label, fk, pk in (("DuckDB", "DDB", "DDB_JP"), ("Postgres", "PG", "PG_JP")):
        clean, conf = _clean_pairs(fk, pk)
        if clean is None:
            lines.append(f"{label} & \\multicolumn{{5}}{{c}}{{\\emph{{pending}}}} \\\\")
            continue
        mf = sum(-t[1] for t in clean if t[1] < 0)
        mp = sum(-t[2] for t in clean if t[2] < 0)
        rem = f"{100*(1-mp/mf):.1f}\\%" if mf > 0 else "--"
        lines.append(f"{label} & {len(clean)} & {len(conf)} & {mf:.1f} & {mp:.1f} & {rem} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_joinorder", "\n".join(lines) + "\n")


def tab_ablation(out):
    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Config & Queries covered & Queries with rule & Median yield (\\%) \\\\", "\\midrule"]
    ok = True
    for tag, key, disp in (("4-2", "DDB", "4 rounds / 2 samples"), ("1-1", "DDB_1_1", "1 round / 1 sample")):
        o = tn.load_oracle(key)
        t = tn.load_transfer(key)
        if o is None or t is None:
            lines.append(f"{disp} & \\multicolumn{{3}}{{c}}{{\\emph{{pending}}}} \\\\")
            ok = False
            continue
        yield_ = median([v["pct"] for v in o.values()])
        lines.append(f"{disp} & {len(o)} & {len(t)} & {yield_:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_ablation", "\n".join(lines) + "\n")


def tab_rpc(out):
    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{lrr}", "\\toprule",
             "random\\_page\\_cost & Selector median (\\%) & Oracle median (\\%) \\\\", "\\midrule"]
    for tag, key in (("4.0 (default)", "PG"), ("1.1 (SSD)", "PG_RPC")):
        opt = tn.load_optimizer(key)
        ora = tn.load_oracle(key)
        if opt is None or ora is None:
            lines.append(f"{tag} & \\multicolumn{{2}}{{c}}{{\\emph{{pending}}}} \\\\")
            continue
        om = median([v["pct"] for v in opt.values()])
        orm = median([v["pct"] for v in ora.values()])
        lines.append(f"{tag} & {om:.1f} & {orm:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_rpc", "\n".join(lines) + "\n")


def tab_wk_byscore(out):
    path = tn.wk_eval_path()
    if not os.path.isfile(path):
        return pending_table(out, "tab_wk_byscore", "wk_eval.json not found")
    rules = json.load(open(path))["rules"]
    # median_percent_saved is already in percent; use it as-is, no rescaling.
    by_score = {}
    for r in rules:
        s = r.get("verdict", {}).get("score")
        v = r.get("agg", {}).get("median_percent_saved")
        if s is None or not isinstance(v, (int, float)):
            continue
        by_score.setdefault(s, []).append(v)
    def f1(x):  # one decimal, no negative zero
        return f"{x + 0.0:.1f}" if abs(x) >= 0.05 else "0.0"
    lines = ["% generated by thesis_figures.py",
             "\\begin{tabular}{crrr}", "\\toprule",
             "World-knowledge score & Rules & Mean speedup (\\%) & Median speedup (\\%) \\\\", "\\midrule"]
    for s in range(1, 6):
        vals = by_score.get(s, [])
        if vals:
            lines.append(f"{s} & {len(vals)} & {f1(mean(vals))} & {f1(median(vals))} \\\\")
        else:
            lines.append(f"{s} & 0 & \\multicolumn{{2}}{{c}}{{\\textemdash}} \\\\")
    le3 = [v for s, vs in by_score.items() if s <= 3 for v in vs]
    ge4 = [v for s, vs in by_score.items() if s >= 4 for v in vs]
    lines.append("\\midrule")
    lines.append(f"$\\leq 3$ & {len(le3)} & {f1(mean(le3))} & {f1(median(le3))} \\\\")
    lines.append(f"$\\geq 4$ & {len(ge4)} & {f1(mean(ge4))} & {f1(median(ge4))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_wk_byscore", "\n".join(lines) + "\n")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory for PDFs/tables")
    ap.add_argument("--check-palette", action="store_true",
                    help="verify colour separation (CIEDE2000, incl. simulated "
                         "protanopia/deuteranopia) and exit; renders nothing")
    args = ap.parse_args()
    if args.check_palette:
        raise SystemExit(check_palette())
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    set_style()
    print("writing to:", out)
    # Part 1
    fig_funnel(out)
    fig_novelty(out)
    fig_transfer(out)
    fig_oracle_ceiling(out)
    fig_oracle_workload(out)
    fig_worldknowledge(out)
    fig_generality(out)
    # Part 2
    fig_oracle_gap(out)
    fig_joinorder(out)
    fig_engines(out)
    fig_engines_composition(out)
    fig_crossengine(out)
    fig_casestudy_29a(out)
    # Tables
    tab_showcase(out)
    tab_ceiling(out)
    tab_joinorder(out)
    tab_ablation(out)
    tab_rpc(out)
    tab_wk_byscore(out)


if __name__ == "__main__":
    main()
