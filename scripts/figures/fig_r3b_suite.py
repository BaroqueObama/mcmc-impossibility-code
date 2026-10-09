"""Plot synthetic anytime bounds from saved analysis or compact summaries.

Solid lines are bounds on the stated events; dashed lines use a plug-in
reference operator. The grey band is the independent reference interval
[L_t, U_t]. The compact paper export uses J=32768, m=8 and a sampled target
reference. See README.md for the plotting command.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHODS = [  # (name pattern, label, color, linestyle, linewidth): Sec. 3 drawn wider beneath Sec. 4
    ("direct", "direct full MMD", "#2a78d6", "-", 1.6),
    ("B3_N{N}", "Sec. 3, estimated $\\Sigma$", "#eb6834", "-", 3.2),
    ("B4own_N{N}", "Sec. 4, estimated $\\Sigma_t$", "#1baf7a", "-", 1.4),
    ("B3ref", "Sec. 3, plug-in (not a guarantee)", "#eda100", "--", 1.6),
    ("B4ref_own", "Sec. 4, plug-in (not a guarantee)", "#e87ba4", "--", 1.6),
]
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#d9d8d4"


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis-dir")
    source.add_argument("--compact-json", help="saved primary-cell export from extract_r3b_paper_data.py")
    ap.add_argument("--out", required=True, help="output path without extension (pdf + png)")
    ap.add_argument("--J", type=int, default=32768)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--m", type=int, default=8)
    a = ap.parse_args()
    if a.compact_json:
        if (a.J, a.m) != (32768, 8):
            ap.error("--compact-json contains only the primary cell J=32768, m=8")
        items = list(json.loads(pathlib.Path(a.compact_json).read_text()).items())
    else:
        files = sorted(pathlib.Path(a.analysis_dir).glob("*.summary.json"))
        if not files:
            raise FileNotFoundError(f"no summaries in {a.analysis_dir}")
        items = [(f.name.removesuffix(".summary.json"), json.loads(f.read_text())) for f in files]
    families = {"gauss": (0, "Gaussian"), "mixture": (1, "Mixture"),
                "studentt": (2, "Student-$t$"), "funnel": (3, "Funnel")}
    # RWM across the first row, MALA across the second; one family per column.
    items.sort(key=lambda p: ({"rwm": 0, "mala": 1}.get(p[0].split("__")[1], 2),
                              families.get(p[0].split("-")[0], (4, ""))[0], p[0]))
    family_variants = {}
    for cid, _ in items:
        parts = cid.split("__")
        family_variants.setdefault(cid.split("-")[0], set()).add((parts[0], *parts[2:]))
    summ = [s for _, s in items]
    n = len(items)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
                         "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.75, 1.4 * nrow + 0.7), squeeze=False)
    for panel, (ax, (cid, s)) in enumerate(zip(axes.flat, items)):
        t = np.array(s["t_grid"] if a.compact_json else s["setup"]["t_grid"], dtype=float)
        x = t + 1.0
        L, U, D = (np.array(s["truth"][k]) for k in ("L", "U", "D"))
        ax.fill_between(x, np.maximum(L, 1e-4), U, color=BAND, lw=0, label="truth interval")
        ax.plot(x, np.maximum(D, 1e-4), color=MUTED, lw=0.8, label="reference estimate")
        key = f"J{a.J}|m{a.m}|sampled"
        for pat, label, col, ls, lw in METHODS:
            name = pat.format(N=a.N)
            record = s["methods"][name]["primary"] if a.compact_json else s["methods"][name][key]
            B = np.array(record["B_median"])
            if len(B) != len(t):
                raise ValueError(f"{cid}/{name}: time and bound arrays differ in length")
            ax.plot(x, B, color=col, ls=ls, lw=lw / 2, label=label)
        for tol in (0.05, 0.10):
            ax.axhline(tol, color=MUTED, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.0)
        family = families.get(cid.split("-")[0], (4, cid.split("__")[0]))[1]
        sampler = cid.split("__")[1].upper()
        title = f"{family} / {sampler}"
        if not a.compact_json and len(family_variants[cid.split("-")[0]]) > 1:
            title = cid.replace("__", " / ")
        ax.set_title(title, pad=4)
        if panel // ncol == nrow - 1:
            ax.set_xlabel("$t+1$", labelpad=1)
        if panel % ncol == 0:
            ax.set_ylabel("MMD and bound", labelpad=2)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(labelsize=7, pad=2, width=0.5, length=2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, which="major", color="#ecebe8", lw=0.5)
    for ax in axes.flat[n:]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.005), handlelength=2, handletextpad=0.5,
               columnspacing=1, labelspacing=0.4)
    top = 0.94
    init = [s.get("init") or {} for s in summ]
    if any(i and not i.get("A3_holds", True) for i in init):
        # conventional starts: every bound except the direct one uses the ASSUMED radius
        kind = sorted({str(i.get("kind")) for i in init})
        fig.suptitle(
            f"Conventional start ({', '.join(kind)}): all bounds except the direct one use the "
            f"ASSUMED radius R = {init[0].get('assumed_R')}, not a verified one (not guarantees)",
            fontsize=8,
            color=INK,
        )
        top = 0.90
    fig.subplots_adjust(left=0.075, right=0.99, top=top, bottom=0.22, hspace=0.38, wspace=0.16)
    for txt in fig.findobj(match=matplotlib.text.Text):
        txt.set_color("black")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
