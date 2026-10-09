"""Plot learned alignment for all eight selected diagnostic counts.

Lines and bands are medians and interquartile ranges over 30 fits at each
pilot length. Red curves include mode-estimation error, conditional on
the recorded numerical margins and known true slow-mode span.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from main_alignment_row import CASES

COLORS = {1: "#0072B2", 2: "#009E73", 9: "#7B4AB0"}
MARKERS = {1: "o", 2: "s", 9: "D"}

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "paper/figures/data/r2_real_alignment"
OUT = ROOT / "paper/figures/fig_r2_learned_alignment"
LENGTHS = (4096, 16384)
STYLES = {4096: (0, (3.0, 1.5)), 16384: "-"}
BOUND_COLOR = "#C62828"
YLIM = (0.008, 1.5)


def draw():
    with (FOLDER / "angles.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["span_kind"] == "learned"]
    with (FOLDER / "learned_summary.csv").open() as stream:
        saved = {(r["dataset"], int(r["m"]), int(r["pilot_length"]), int(r["t"])): r
                 for r in csv.DictReader(stream)}
    selection = json.loads((FOLDER / "selection.json").read_text())["datasets"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(row["m"]), int(row["pilot_length"]), int(row["t"]))].append(row)
    assert len(rows) == 5280 and len(groups) == 176
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "axes.linewidth": 0.55,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2,
        "ytick.major.size": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
    })
    fig, axes = plt.subplots(1, 5, figsize=(6.75, 1.92), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.865, bottom=0.32, wspace=0.25)
    facts = {"sources": [str((FOLDER / name).relative_to(ROOT))
                         for name in ("angles.csv", "learned_summary.csv")],
             "curves": {},
             "meaning": "Median and IQR of learned-span angle over 30 independent fits. "
                        "Long-run plateaus reflect estimation of the true embedded slow-mode span. "
                        "The red envelope adds the measured true-span estimation error to the "
                        "Section 4 envelope; it is conditional on numerical margins and uses the "
                        "known slow-mode span, not only the learned fit."}
    for ax, (dataset, title) in zip(axes, CASES):
        counts = sorted(selection[dataset]["selected_m"])
        facts["curves"][dataset] = {}
        for m in counts:
            facts["curves"][dataset][str(m)] = {}
            for length in LENGTHS:
                ts = sorted(t for (ds, count, budget, t) in groups
                            if ds == dataset and count == m and budget == length)
                assert ts == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                points = []
                for t in ts:
                    cell = groups[(dataset, m, length, t)]
                    assert len(cell) == 30 and sorted(int(r["fit"]) for r in cell) == list(range(30))
                    assert all(r["estimate_status"] == "RESOLVED" for r in cell)
                    vals = [float(r["sin_theta"]) for r in cell]
                    bounds = [float(r["learned_sin_upper"]) for r in cell]
                    uncertainties = [float(r["sin_uncertainty"]) for r in cell]
                    assert all(s <= g + e for s, g, e in zip(vals, bounds, uncertainties))
                    q25, median, q75 = map(float, np.quantile(vals, [0.25, 0.5, 0.75]))
                    upper = float(np.median(bounds))
                    expected = saved[(dataset, m, length, t)]
                    for key, value in (("sin_theta_q25", q25), ("sin_theta_median", median),
                                       ("sin_theta_q75", q75), ("learned_sin_upper_median", upper)):
                        assert np.isclose(value, float(expected[key]), rtol=2e-14, atol=0), (dataset, m, length, t, key)
                        assert YLIM[0] <= value <= YLIM[1], (dataset, m, length, t, key, value)
                    points.append({"t": t, "q25": q25, "median": median, "q75": q75,
                                   "envelope_median": upper, "n": len(vals)})
                lower, med, higher, bound = ([p[key] for p in points] for key in
                                             ("q25", "median", "q75", "envelope_median"))
                ax.fill_between(ts, lower, higher, color=COLORS[m], alpha=0.10 if length == 4096 else 0.18,
                                lw=0, zorder=1)
                ax.plot(ts, bound, color=BOUND_COLOR, lw=0.85, alpha=0.75,
                        ls=STYLES[length], zorder=2)
                ax.plot(ts, med, color=COLORS[m], lw=0.95, ls=STYLES[length],
                        marker=MARKERS[m], ms=2.5, mfc="white" if length == 4096 else COLORS[m],
                        mew=0.55, zorder=4)
                facts["curves"][dataset][str(m)][str(length)] = points
        ax.set_title(title, pad=3)
        ax.set_xscale("symlog", linthresh=1, base=2)
        ax.set_xlim(-0.15, 650)
        ax.set_xticks([0, 8, 64, 512], labels=["0", "8", "64", "512"])
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.set_yticks([0.01, 0.1, 1], labels=["0.01", "0.1", "1"])
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e7e7", lw=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(pad=1.5)
    axes[0].set_ylabel(r"$\sin\theta_{\max}$")
    fig.text(0.535, 0.205, "Sampling iterations", ha="center", fontsize=7)
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=3, lw=0.9,
                      label=f"{m} diagnostic" + ("s" if m > 1 else "")) for m in (1, 2, 9)]
    handles += [Patch(fc="#888888", alpha=0.2, label="IQR over 30 fits"),
                Line2D([], [], color="black", ls=STYLES[4096], lw=0.9, label="4,096 steps / chain"),
                Line2D([], [], color="black", ls=STYLES[16384], lw=0.9, label="16,384 steps / chain"),
                Line2D([], [], color=BOUND_COLOR, lw=0.9, label="Bound with mode-estimation error")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.535, -0.003), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.0, handletextpad=0.4,
               labelspacing=0.35, borderaxespad=0)
    fig.canvas.draw()
    for extension in ("pdf", "png"):
        fig.savefig(f"{OUT}.{extension}", dpi=300)
    plt.close(fig)
    return facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="output stem (PDF and PNG)")
    args = parser.parse_args()
    OUT = args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    facts = draw()
    print(f"Wrote {OUT}.pdf/.png; checked all 176 summaries against 5,280 saved fit checkpoints.")
