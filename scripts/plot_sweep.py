#!/usr/bin/env python3
"""Render the particle-count sweep as a per-stage scaling chart.

Reads docs/sweep.csv (from `profiler --sweep --csv docs/sweep.csv`) and
writes docs/scaling.svg + docs/scaling.png.

Two stacked panels sharing the x axis:
  A) force dispatch + frame total on log-log, with slope-1 / slope-2 guides
     -- a straight slope-s line means time ~ N^s, so the O(N^2) force kernel
     bends to slope ~2 once the GPU is compute-bound (past N ~ 8k).
  B) the fixed-cost stages (barriers, integrate, readback) on log-x /
     linear-y -- tens of microseconds, roughly flat in N.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "docs" / "sweep.csv"

# Validated categorical palette (dataviz skill reference, light mode), fixed
# slot order. Contrast WARN on slots 3-5 is relieved by direct end-labels
# plus the CSV serving as the table view.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

FIXED = [
    ("barrier1_us", "barrier 1", ORANGE),
    ("integrate_us", "integrate dispatch", AQUA),
    ("barrier2_us", "barrier 2", YELLOW),
    ("copy_us", "readback copy", MAGENTA),
]


def declutter(ys, min_gap):
    """Nudge label y-positions apart, preserving order, by >= min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    adj = list(ys)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if adj[hi] - adj[lo] < min_gap:
            adj[hi] = adj[lo] + min_gap
    return adj


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}; run: profiler --sweep --csv {CSV}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(CSV.open()))
    n = [int(r["n"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(8.4, 7.4), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: the stages that scale -------------------------------------
    force, frame = col("force_us"), col("frame_total_us")
    axA.plot(n, force, color=BLUE, lw=2.2, marker="o", ms=5, zorder=3,
             label="force dispatch")
    axA.plot(n, frame, color=MUTED, lw=1.6, ls="--", marker="s", ms=4, zorder=2,
             label="frame total")

    # Reference slopes anchored to the compute-bound segment (from N = 8192);
    # labels sit at the segment midpoint to stay clear of the curve's end.
    a = n.index(8192)
    fy = force[a]
    for slope, dash, tag in ((2.0, (0, (5, 3)), "slope 2  (~N^2)"),
                             (1.0, (0, (2, 2)), "slope 1  (~N^1)")):
        xs = [n[a], n[-1]]
        ys = [fy, fy * (n[-1] / n[a]) ** slope]
        axA.plot(xs, ys, color=MUTED, lw=1.0, ls=dash, zorder=1)
        mx = (xs[0] * xs[1]) ** 0.5
        my = fy * (mx / n[a]) ** slope
        axA.annotate(tag, (mx, my), xytext=(4, -4 if slope == 1 else 6),
                     textcoords="offset points", fontsize=7.5, color=MUTED,
                     va="center")

    axA.set_yscale("log")
    axA.set_ylim(force[0] * 0.55, force[-1] * 3.0)
    axA.set_ylabel("time per step  (us)", fontsize=9, color=INK)
    axA.set_title(
        "Per-stage GPU time vs workload size  (Apple M1 via MoltenVK)",
        fontsize=11, color=INK, pad=10,
    )
    axA.legend(frameon=False, fontsize=8, loc="upper left")

    # ---- Panel B: the fixed-cost stages -----------------------------------
    end_y = [col(k)[-1] for k, _, _ in FIXED]
    span = max(end_y) - min(end_y) or 1.0
    lab_y = declutter(end_y, min_gap=span * 0.42)
    for (key, label, color), ly in zip(FIXED, lab_y):
        y = col(key)
        axB.plot(n, y, color=color, lw=2.0, marker="o", ms=4.5, zorder=3,
                 label=label)
        axB.annotate(label, (n[-1], ly), xytext=(8, 0),
                     textcoords="offset points", va="center", fontsize=8.5,
                     color=color, fontweight="bold")
    axB.set_ylabel("time per step  (us)", fontsize=9, color=INK)
    axB.set_ylim(0, max(max(col(k)) for k, _, _ in FIXED) * 1.18)
    axB.set_xlabel("particle count  N", fontsize=9, color=INK)
    axB.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    # ---- shared x + chrome ------------------------------------------------
    for ax in (axA, axB):
        ax.set_xscale("log", base=2)
        ax.set_xticks(n)
        ax.set_xticklabels([str(v) for v in n], fontsize=8)
        ax.set_xlim(n[0] * 0.9, n[-1] * 2.8)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED)

    fig.subplots_adjust(left=0.10, right=0.82, top=0.93, bottom=0.08, hspace=0.12)
    for ext in ("svg", "png"):
        out = ROOT / "docs" / f"scaling.{ext}"
        fig.savefig(out, facecolor=SURFACE)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
