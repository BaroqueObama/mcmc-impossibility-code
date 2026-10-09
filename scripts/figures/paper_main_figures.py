"""Draw the main experiment and residual appendix figures from saved tables.

Use --collect to rebuild the residual input tables and --out-dir to choose
the output directory.
"""

from __future__ import annotations
import argparse
import csv
import math
import pathlib
import sys
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import main_figures_data as mfd  # noqa: E402

ROOT = mfd.ROOT

DATA = ROOT / "paper/figures/data/main_figures"

FIGS = ROOT / "paper/figures"

GROUP_COLOR = {"analytic": "#1f1f1f", "synthetic": "#0072b2", "regression": "#009e73", "logistic": "#d55e00"}

GROUP_LABEL = {
    "analytic": "analytic controls (known answer)",
    "synthetic": "synthetic targets",
    "regression": "Gaussian regression (exact posterior)",
    "logistic": "logistic regression (NUTS reference)",
}

INK, MUTED, GRID, VIOL, REF = "#1f1f1f", "#6b6a66", "#e7e6e2", "#f6e3dc", "#a3a29d"

FAMILY_MARKER = {"gauss": "o", "mixture": "s", "studentt": "^", "funnel": "D"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "axes.linewidth": 0.55,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    })  # fmt: skip


def _ax(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)


def _title(ax, letter: str, text: str) -> None:
    ax.set_title(rf"$\bf{{({letter})}}$ {text}" if letter else text, loc="left", pad=3)


def read(name: str) -> list[dict]:
    f = DATA / f"{name}.csv"
    with f.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No residual data in {f}")
    return rows


def num(x) -> float:
    return mfd.fnum(x)


def _log2_m(ax, ticks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(min(ticks) / 1.25, max(ticks) * 1.25)


def _group_handles(groups: list[str], pend: set[str]) -> list[Line2D]:
    hs = []
    for g in groups:
        lab = GROUP_LABEL[g] + (" — pending" if g in pend else "")
        hs.append(Line2D([], [], color=GROUP_COLOR[g] if g not in pend else "#b9b8b3", lw=1.4, label=lab))
    return hs


def _save(fig, stem: str, out_dir: pathlib.Path, preview: bool) -> None:
    if preview:
        fig.text(0.5, 0.5, "Preview: pilot data", ha="center", va="center",
                 rotation=20, fontsize=22, color="#d55e00", alpha=0.25, zorder=100)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out_dir / stem}.pdf/.png")

R1B_LO, R1B_HI = 4e-3, 30.0


def _ref_line(ax, y: float, label: str, x_text: float, va: str = "bottom") -> None:
    ax.axhline(y, color=REF, lw=2.0, zorder=2, solid_capstyle="butt")
    ax.text(x_text, y * (1.03 if va == "bottom" else 0.97), label, va=va, ha="left", fontsize=5.9,
            color=MUTED, zorder=3)  # fmt: skip


def _group_legend(fig, present: set[str], y: float = 1.0, groups=("analytic", "synthetic", "regression", "logistic")) -> None:
    hs = _group_handles(list(groups), set(groups) - present)
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(hs), frameon=False,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.4)  # fmt: skip


def _r1_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda r: int(r["m"]))
    return np.array([int(r["m"]) for r in rows]), np.array([num(r[key]) for r in rows])


def _r1_marker(g: str, ds: str) -> str:
    if g == "logistic":
        return "s" if ds.startswith("horseshoe") else "o"
    return "o" if g in ("analytic", "regression") else FAMILY_MARKER.get(ds.split("-")[0], "o")

STRESS_TINT = "#7db6dd"  # lighter synthetic blue: stress variants (one factor changed), appendix only

STRESS_SHORT = {
    "gauss-d10-c100__rwm": "cond. 100", "gauss-d10-c1__rwm": "cond. 1", "mixture-d10-s10-w0.5__rwm": "sep. 10",
    "mixture-d10-s2-w0.5__rwm": "sep. 2", "mixture-d10-s6-w0.1__rwm": "weight 0.1", "funnel-d10-sy1__rwm": r"$\sigma_y=1$",
    "studentt-d10-nu5__rwm": r"$\nu=5$", "studentt-d10-nu30__rwm": r"$\nu=30$", "gauss-d50-c25__rwm": "$d=50$",
}  # fmt: skip


def _r1_projected(ax, rows: list[dict], groups: tuple[str, ...], law_sets_for: dict, label_stress: bool = False) -> bool:
    """Per dataset (and law set): worst resolved ratio over initial laws and checkpoints; if nothing is
    resolved at that m, the largest +2 SE value, hollow. Returns whether any hollow point was drawn."""
    any_hollow = False
    end_labels: list[tuple[float, float, str, str]] = []  # (x, y, text, colour), placed after the loop
    for g in groups:
        rr = [r for r in rows if r["group"] == g]
        sets = law_sets_for.get(g, {"": None})
        for ds in sorted({r["dataset"] for r in rr}):
            for law_key, laws in sets.items():
                sel = [r for r in rr if r["dataset"] == ds and (laws is None or r["setting"].split("/")[-1] in laws)]
                if not sel:
                    continue
                ms = sorted({int(r["m"]) for r in sel})
                ys, hollow = [], []
                for mm in ms:
                    cell = [r for r in sel if int(r["m"]) == mm]
                    res = [r for r in cell if r["resolved"] == "True"]
                    # worst resolved value; if none is resolved, the largest +2 SE value (an upper end), hollow
                    ys.append(max(num(r["ratio"]) for r in res) if res else max(num(r["ratio_hi"]) for r in cell))
                    hollow.append(not res)
                ys, hollow, ms = np.maximum(np.array(ys), R1B_LO), np.array(hollow), np.array(ms)
                any_hollow |= bool(hollow.any())
                stress = sel[0].get("tier") == "stress" and ds not in mfd.R1B_MAIN_EXTRA
                col, mk = (STRESS_TINT if stress else GROUP_COLOR[g]), _r1_marker(g, ds)
                mala = g == "synthetic" and ds.endswith("__mala")
                ls = (0, (3, 1.6)) if (law_key == "I1/I2" or mala) else "-"
                ax.plot(ms, ys, color=col, ls=ls, lw=0.85, alpha=0.9, zorder=3)
                ax.plot(ms[~hollow], ys[~hollow], ls="none", marker=mk, ms=2.7, color=col, mec="white", mew=0.3, zorder=4)
                ax.plot(ms[hollow], ys[hollow], ls="none", marker=mk, ms=2.9, mfc="white", mec=col, mew=0.6, zorder=4)
                if "-d50-" in ds or (label_stress and stress):
                    end_labels.append((float(ms[-1]), float(ys[-1]), STRESS_SHORT.get(ds, ds), col))
    # end-of-line labels, pushed apart in log10(y) so that none overlap (thin leader to the line end)
    end_labels.sort(key=lambda e: e[1])
    placed: list[float] = []
    for x, y, txt, col in end_labels:
        if not label_stress:  # main panel: the single d = 50 label sits above-left of its line end
            ax.annotate(txt, (x, y), xytext=(-2, 5), textcoords="offset points", fontsize=5.2, color=col,
                        ha="right", va="bottom")  # fmt: skip
            continue
        ly = math.log10(y)
        if placed and ly < placed[-1] + 0.13:
            ly = placed[-1] + 0.13
        placed.append(ly)
        ax.annotate(txt, (x, y), xytext=(x * 1.45, 10**ly), textcoords="data", fontsize=5.0, color=col, va="center",
                    ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.4, shrinkA=0, shrinkB=0))  # fmt: skip
    _log2_m(ax, [2, 4, 8, 16, 32])
    ax.set_yscale("log")
    return any_hollow


def fig_r1(out_dir: pathlib.Path, preview: bool) -> None:
    a, b = read("r1a"), read("r1b")
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45), gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.83, wspace=0.33)
    # (a) the worst case R lambda_{m+1}^r itself (lines) and the constructed laws (markers), log-log
    ax = axes[0]
    _ax(ax)
    for g in ("synthetic", "regression", "analytic"):  # analytic last: drawn on top
        rows = [r for r in a if r["group"] == g and (g == "analytic" or num(r["R"]) == 0.5)]
        for ds in sorted({r["dataset"] for r in rows}):
            for rr in sorted({r["r"] for r in rows if r["dataset"] == ds}):
                sel = [r for r in rows if r["dataset"] == ds and r["r"] == rr]
                m, env = _r1_series(sel, "envelope")
                _, res = _r1_series(sel, "residual")
                col, z = GROUP_COLOR[g], (6 if g == "analytic" else 3)
                ax.plot(m, env, color=col, lw=1.1 if g == "analytic" else 0.8, alpha=0.95 if g == "analytic" else 0.6,
                        zorder=z)  # fmt: skip
                ax.plot(m, res, ls="none", marker=_r1_marker(g, ds), ms=2.8 if g == "analytic" else 2.5, color=col,
                        mec="white", mew=0.3, zorder=z + 1)  # fmt: skip
                if g == "analytic":
                    lab = r"circle, $r=\frac{1}{2}$" if num(rr) == 0.5 else "circle, $r=1$"
                    ax.annotate(lab, (m[-1], env[-1]), xytext=(-3, -4), textcoords="offset points", ha="right",
                                va="top", fontsize=5.6, color=INK, zorder=8)  # fmt: skip
    circ = [r for r in a if r["group"] == "analytic"]
    if circ:  # light reference slopes for the circle's exact rate m^{-alpha r}, alpha = 2
        mm = np.array([2.0, 64.0])
        for rr, k, lab in ((0.5, 1, r"$\propto m^{-1}$"), (1.0, 2, r"$\propto m^{-2}$")):
            sel = sorted((r for r in circ if num(r["r"]) == rr), key=lambda r: int(r["m"]))
            c = 2.0 * num(sel[-1]["envelope"]) * float(sel[-1]["m"]) ** k  # anchored above the last circle point
            ax.plot(mm, c * mm ** (-k), color=REF, lw=0.8, ls=(0, (1.2, 1.4)), zorder=2)
            ax.text(mm[-1], c * mm[-1] ** (-k) * 1.7, lab, fontsize=5.6, color=MUTED, ha="right", va="bottom")
    _log2_m(ax, [2, 4, 8, 16, 32, 64])
    ax.set_yscale("log")
    ax.set_ylim(8e-6, 0.3)
    ax.set_ylabel("remaining MMD")
    ax.set_xlabel("satisfied diagnostics $m$", labelpad=1)
    _title(ax, "a", r"worst case $R\lambda_{m+1}^{r}$ (lines), laws (dots)")
    # (b) attainment: constructed-law residual / envelope
    ax = axes[1]
    _ax(ax)
    _ref_line(ax, 1.0, "envelope (worst case)", 1.75)
    for g in ("synthetic", "regression", "analytic"):
        rows = [r for r in a if r["group"] == g]
        for ds in sorted({r["dataset"] for r in rows}):
            for R in sorted({r["R"] for r in rows if r["dataset"] == ds}, reverse=True):
                for rr in sorted({r["r"] for r in rows if r["dataset"] == ds and r["R"] == R}):
                    sel = [r for r in rows if r["dataset"] == ds and r["R"] == R and r["r"] == rr]
                    m, y = _r1_series(sel, "ratio")
                    assert np.all(y <= 1 + 1e-3), (ds, R, y)  # nothing above the envelope beyond numerics
                    ls = "-" if (g == "analytic" or num(R) == 0.5) else (0, (3, 1.6))
                    z = 6 if g == "analytic" else 4
                    ax.plot(m, y, color=GROUP_COLOR[g], ls=ls, lw=0.85, marker=_r1_marker(g, ds), ms=2.6,
                            mfc=GROUP_COLOR[g], mec="white", mew=0.3, alpha=0.9, zorder=z)  # fmt: skip
    _log2_m(ax, [2, 4, 8, 16, 32, 64])
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("remaining MMD / envelope")
    ax.set_xlabel("satisfied diagnostics $m$", labelpad=1)
    _title(ax, "b", "attainment by positive laws")
    hs = [Line2D([], [], color=MUTED, lw=0.9, label="$R=0.5$"),
          Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 1.6)), label="$R=0.2$")]  # fmt: skip
    hs += [Line2D([], [], color=MUTED, ls="none", marker=mk, ms=3, label=lab)
           for lab, mk in (("Gaussian", "o"), ("mixture", "s"), ("Student-$t$", "^"), ("funnel", "D"))]  # fmt: skip
    ax.legend(handles=hs, loc="lower left", ncol=2, frameon=False, fontsize=5.4, handlelength=1.6,
              columnspacing=0.8, labelspacing=0.3, borderaxespad=0.2)  # fmt: skip
    # (c) projected residuals of actual chain laws whose radius is known (synthetic) or exact (regression)
    ax = axes[2]
    _ax(ax)
    _ref_line(ax, 1.0, "bound", 16.5, va="bottom")  # above the line, right: clear of the data near m = 2
    b_main = [r for r in b if r["group"] != "synthetic" or r["in_main"] == "True"]
    hollow = _r1_projected(ax, b_main, ("synthetic", "regression"), {})
    ax.set_ylim(R1B_LO, 2.5)
    ax.set_ylabel(r"$\max\,\|P_{V_m^\perp}\delta_t\|$ / bound")
    ax.set_xlabel("diagnostics $m$", labelpad=1)
    _title(ax, "c", "chain laws (criteria not satisfied)")
    hs = [Line2D([], [], color=MUTED, lw=0.9, label="RWM / Gibbs"), Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 1.6)),
          label="MALA")]  # fmt: skip
    if hollow:
        hs.append(Line2D([], [], color=MUTED, ls="none", marker="o", ms=3, mfc="white", label="unresolved ($+2$ SE)"))
    ax.legend(handles=hs, loc="lower left", ncol=len(hs), frameon=False, fontsize=5.4, handlelength=1.8)
    _group_legend(fig, {r["group"] for r in a} | {r["group"] for r in b if r["group"] != "logistic"},
                  groups=("analytic", "synthetic", "regression"))  # fmt: skip
    _save(fig, "fig_main_r1", out_dir, preview)


def fig_main_experiments(out_dir: pathlib.Path, preview: bool) -> dict:
    """Three real-data comparisons: residuals, slow-mode alignment, and bounds."""
    from main_residual_row import draw_residual_row
    from main_anytime_row import draw_anytime_row, SECTION3_COLOR, SECTION4_COLOR
    from main_alignment_row import draw_alignment_row

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7})
    figure_height = 4.50
    fig = plt.figure(figsize=(6.75, figure_height))
    # Vertical positions in inches from the top.
    layout = (
        dict(title=0.03, legend=0.19, plot=0.53, height=0.73, xlabel=1.45),
        dict(title=1.62, legend=1.78, plot=2.12, height=0.57, xlabel=2.88),
        dict(title=3.06, legend=3.22, plot=3.47, height=0.73, xlabel=4.39),
    )

    def from_top(inches: float) -> float:
        return 1 - inches / figure_height

    axes = []
    left, right, gap = 0.067, 0.985, 0.037
    width = (right - left - 4 * gap) / 5
    for positions in layout:
        bottom = from_top(positions["plot"] + positions["height"])
        height = positions["height"] / figure_height
        row = [fig.add_axes([left + i * (width + gap), bottom, width, height]) for i in range(5)]
        for ax in row:
            _ax(ax)
        axes.append(row)
    facts = {"a": draw_residual_row(fig, axes[0], ROOT),
             "b": draw_alignment_row(fig, axes[1], ROOT),
             "c": draw_anytime_row(fig, axes[2], ROOT)}
    fig.text(left, from_top(layout[0]["title"]), "(A) How much discrepancy do the diagnostics leave unmeasured?", va="top", fontsize=8, weight="bold")
    fig.text(left, from_top(layout[1]["title"]), "(B) Do the optimal diagnostics approach the slow modes?", va="top", fontsize=8, weight="bold")
    fig.text(left, from_top(layout[2]["title"]), "(C) Does using the sampler tighten the bound?", va="top", fontsize=8, weight="bold")
    red, blue = "#C62828", "#0072B2"
    handles_a = [Line2D([], [], color=red, lw=1.5, label="Worst-case residual"),
                 Line2D([], [], color=blue, lw=1.0, marker="o", ms=2.5, ls="--", label="Slow start"),
                 Line2D([], [], color="#009e73", lw=1.0, marker="s", ms=2.5, ls=":", label="Fast start")]
    fig.legend(handles=handles_a, loc="upper center", bbox_to_anchor=(0.53, from_top(layout[0]["legend"])), frameon=False,
               ncol=3, handlelength=1.8, columnspacing=1.2, handletextpad=0.4, fontsize=7, borderaxespad=0)
    handles_b = [Line2D([], [], color=red, lw=1.35, label="Theoretical bound"),
                 Line2D([], [], color=blue, lw=1.0, marker="o", ms=2.5, label="Subspace angle"),
                 Line2D([], [], color=blue, lw=0, marker="v", mfc="white", ms=3.4,
                        label="Unresolved angle: upper limit")]
    fig.legend(handles=handles_b, loc="upper center", bbox_to_anchor=(0.53, from_top(layout[1]["legend"])), frameon=False,
               ncol=3, handlelength=1.6, columnspacing=1.2, handletextpad=0.4, fontsize=7, borderaxespad=0)
    handles_c = [Line2D([], [], color=SECTION3_COLOR, lw=1.35, label="Section 3"),
                 Line2D([], [], color=SECTION4_COLOR, lw=1.35, label="Section 4"),
                 Line2D([], [], color=blue, lw=1.1, marker="o", ms=2.5, label="Actual discrepancy"),
                 Line2D([], [], color="black", lw=1.15, ls="-", label="Supplied covariance"),
                 Line2D([], [], color="black", lw=1.15, ls="--", label="Estimated covariance")]
    fig.legend(handles=handles_c, loc="upper center", bbox_to_anchor=(0.53, from_top(layout[2]["legend"])), frameon=False,
               ncol=5, handlelength=1.8, columnspacing=0.8, handletextpad=0.4, fontsize=7, borderaxespad=0)
    axes[0][0].set_ylabel("Unmonitored discrepancy", labelpad=2)
    axes[1][0].set_ylabel("Angle (degrees)", labelpad=2)
    axes[2][0].set_ylabel("MMD and bound", labelpad=2)
    fig.text(0.535, from_top(layout[0]["xlabel"]), "Number of diagnostics $m$", ha="center", va="top", fontsize=7)
    fig.text(0.535, from_top(layout[1]["xlabel"]), "Sampling iterations", ha="center", va="top", fontsize=7)
    fig.text(0.535, from_top(layout[2]["xlabel"]), "Sampling iterations", ha="center", va="top", fontsize=7)
    # A begins with the circle control; B/C retain all five regression posteriors.
    for row in axes[2:]:
        for ax in row:
            ax.set_title("")
    for txt in fig.findobj(match=matplotlib.text.Text):
        txt.set_color("black")
    _save(fig, "fig_main_experiments", out_dir, preview)
    _style()
    return facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", choices=["r1", "exp"],
                    help="r1: residual appendix; exp: main experiment figure")
    ap.add_argument("--collect", action="store_true", help="rebuild residual tables from the included results")
    ap.add_argument("--out-dir", type=pathlib.Path, default=FIGS)
    args = ap.parse_args()
    selected = set(args.only or ["r1", "exp"])
    missing_cache = any(not (DATA / name).is_file() for name in ("r1a.csv", "r1b.csv"))
    if args.collect or ("r1" in selected and missing_cache):
        mfd.collect(DATA)
        print("Collected residual tables.")
    _style()
    if "r1" in selected:
        fig_r1(args.out_dir, False)
    if "exp" in selected:
        fig_main_experiments(args.out_dir, False)

if __name__ == "__main__":
    main()
