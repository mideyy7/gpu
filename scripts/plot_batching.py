#!/usr/bin/env python3
"""Before/after chart for the step-batching fix (Phase 8).

Reads docs/sweep.csv (batch=1) and docs/sweep_batch32.csv (batch=32) and
writes docs/batching.svg + .png: mean frame time per step vs N for both.
At small/medium N batch=1 pays a fixed ~0.5 ms submit+fence round-trip
every step; batching 32 steps per submit amortises it and the two lines
converge at large N once the O(N^2) force kernel dominates.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load(name):
    p = ROOT / "docs" / name
    if not p.exists():
        print(f"missing {p}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(p.open()))
    return ([int(r["n"]) for r in rows],
            [float(r["frame_total_us"]) for r in rows])

def main():
    n, f1 = load("sweep.csv")
    _, f32 = load("sweep_batch32.csv")

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(n, f1, color=ORANGE, lw=2.2, marker="o", ms=5, zorder=3,
            label="batch = 1   (one submit per step)")
    ax.plot(n, f32, color=BLUE, lw=2.2, marker="o", ms=5, zorder=3,
            label="batch = 32  (one submit per 32 steps)")

    for x, a, b in zip(n, f1, f32):
        if a / b >= 1.15:
            ax.annotate(f"{a/b:.1f}x faster", (x, b), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=BLUE, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n], fontsize=8)
    ax.set_xlabel("particle count  N", fontsize=9, color=INK)
    ax.set_ylabel("mean frame time per step  (us)", fontsize=9, color=INK)
    ax.set_title("Step batching vs per-step submit  (Apple M1 via MoltenVK)",
                 fontsize=11, color=INK, pad=10)
    ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    for ext in ("svg", "png"):
        out = ROOT / "docs" / f"batching.{ext}"
        fig.savefig(out, facecolor=SURFACE)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
