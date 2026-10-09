"""Figure 1: chain histograms and the bimodal target density.

All 20,000 draws per chain contribute to density-normalized bins of width 0.25.
Diagnostics use the original coordinates.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=root / "paper/figures")
    parser.add_argument("--data-dir", type=Path, default=root / "paper/figures/data")
    args = parser.parse_args()
    output, data_dir = args.out_dir, args.data_dir
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "intro_mixture_seed0_draws.npz") as data:
        projections = data["projections"]
    with np.load(data_dir / "intro_mixture_seed0_diagnostics.npz") as data:
        rhat_max = float(data["rhat_max"][-1])
        ess_min = float(data["ess_min"][-1])
    metadata = json.loads((data_dir / "intro_mixture_seed0.json").read_text())
    assert projections.shape == (4, 20000)
    assert projections.size == metadata["draws_total"]
    assert rhat_max == metadata["final_rhat_max"]
    assert ess_min == metadata["final_ess_min"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "text.color": "black",
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.linewidth": 0.55, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = ["#0072b2", "#c35d17", "#00846b", "#aa5f94"]
    styles = [(0, (0.7, 1.3)), (0, (4, 2)), (0, (3, 1.4, 0.8, 1.4)), (0, (1.8, 1.5))]
    fig, ax = plt.subplots(figsize=(3.25, 1.43))
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.225, top=0.825)
    z = np.linspace(-13, 13, 801)
    sep = float(metadata["config"]["separation"]) / 2
    target = (np.exp(-0.5*(z-sep)**2) + np.exp(-0.5*(z+sep)**2)) / (2*np.sqrt(2*np.pi))
    edges = np.linspace(-13, 13, 105)
    counts = np.stack([np.histogram(x, bins=edges)[0] for x in projections])
    assert np.all(counts.sum(axis=1) == projections.shape[1])
    densities = counts / (projections.shape[1] * np.diff(edges))
    assert np.allclose((densities * np.diff(edges)).sum(axis=1), 1)
    target_color = "#467f9e"
    # Draw Chain 1 in front.
    for index in reversed(range(len(densities))):
        layer = 2 * (len(densities) - index) - 1
        ax.stairs(densities[index], edges, fill=True, color=colors[index], alpha=0.11,
                  linewidth=0, zorder=layer)
        ax.stairs(densities[index], edges, color=colors[index], ls=styles[index], lw=1.35,
                  baseline=None, zorder=layer + 1)
    ax.fill_between(z, target, color="#c2e0ef", alpha=0.9, zorder=9)
    ax.plot(z, target, color=target_color, lw=1.15, ls="-", zorder=10)
    ax.set(xlim=(-13, 13), ylim=(0, 0.48), ylabel="Density",
           xlabel="Projection along separation direction")
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([0, 0.2, 0.4])
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.tick_params(length=2, width=0.55, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    # Keep the axes visible over the target's near-zero tails and filled baseline.
    ax.spines[["bottom", "left"]].set_zorder(12)
    ax.xaxis.set_zorder(12)
    ax.yaxis.set_zorder(12)

    assert rhat_max < 1.01 and ess_min > 200
    ess_label = f"{ess_min:,.0f}".replace(",", r"{,}")
    ax.text(0.025, 0.95, rf"Max $\hat R$: ${rhat_max:.4f}<1.01$",
            transform=ax.transAxes, ha="left", va="top")
    ax.text(0.025, 0.76, rf"Min ESS: ${ess_label}>200$",
            transform=ax.transAxes, ha="left", va="top")
    handles = [Line2D([0], [0], color=target_color, lw=1.15, ls="-", label="Target")]
    handles += [Line2D([0], [0], color=c, lw=1.35, ls=s, label=f"Chain {i+1}")
                for i, (c, s) in enumerate(zip(colors, styles))]
    fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.51, 0.995),
               frameon=False, handlelength=1.2, columnspacing=0.55,
               handletextpad=0.35, borderaxespad=0)
    fig.savefig(output / "fig_intro_trapped_mixture.pdf")
    fig.savefig(output / "fig_intro_trapped_mixture.png", dpi=300)
    plt.close(fig)
    print("Saved unsmoothed histograms: 20,000 draws per chain, shared bin width 0.25.")


if __name__ == "__main__":
    main()
