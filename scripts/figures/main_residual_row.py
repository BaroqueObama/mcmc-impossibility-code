"""Residuals of fixed chain laws as diagnostic counts increase.

The circle uses two piecewise-uniform starts; regression panels use Gaussian
starts. Counts are plotted directly, including the finite-count curvature
of the circle bound.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator

BOUND_COLOR = "#c62828"
RESIDUAL_COLOR = "#0072b2"
FAST_COLOR = "#009e73"
CASES = [("concrete", "Concrete"), ("energy", "Energy"),
         ("airfoil", "Airfoil"), ("wine_red", "Wine"), ("kin8nm", "kin8nm")]
Y_LIMITS = (1e-10, 0.09)


def _read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))




def _axis(ax, title, first):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([1e-9, 1e-6, 1e-3])
    ax.set_yticklabels([r"$10^{-9}$", r"$10^{-6}$", r"$10^{-3}$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=7, colors="black", length=2.2, width=.5, pad=1.5)
    ax.tick_params(axis="y", labelleft=first)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7e7e7", lw=.45)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7, color="black", pad=3)


def draw_regressions(axes, root, cases):
    """Plot all evaluated diagnostic counts for the Gaussian posteriors."""
    root = Path(root)
    folder = root / "paper/figures/data/r1_dense"
    path = folder / "regression_curves.csv"
    raw = _read(path)
    counts = list(range(1, 201))
    panels = []
    for ax, (dataset, title) in zip(axes, cases):
        curves = []
        for start, color, linestyle, marker in (
            ("slow", RESIDUAL_COLOR, "--", "o"),
            ("fast", FAST_COLOR, ":", "s"),
        ):
            rows = sorted((r for r in raw if r["dataset"] == dataset and r["start"] == start),
                          key=lambda r: int(r["m"]))
            assert [int(r["m"]) for r in rows] == counts
            assert all(float(r["R"]) == .2 and float(r["r"]) == .5 and int(r["t"]) == 1 for r in rows)
            assert all(r["status"] == "RESOLVED" for r in rows), "Unresolved residual in the plotted range"
            residual = np.asarray([float(r["residual"]) for r in rows])
            bound = np.asarray([float(r["bound"]) for r in rows])
            assert np.all(residual > Y_LIMITS[0]) and np.all(residual < bound)
            assert np.all(np.diff(residual) <= 0) and np.all(np.diff(bound) <= 0)
            assert np.ptp([float(r["full_mmd"]) for r in rows]) == 0
            for r in rows:
                assert float(r["residual_squared_raw"]) > float(r["numerical_allowance_squared"])
            ax.plot(counts, residual, color=color, lw=1.05, ls=linestyle, marker=marker,
                    markevery=[0, 3, 7, 15, 31, 63, 127, 199], ms=2.3, mew=.4, zorder=3)
            curves.append({"start": start, "points": [
                {"m": int(r["m"]), "residual": float(r["residual"]),
                 "bound": float(r["bound"]), "status": r["status"]} for r in rows]})
        ax.plot(counts, bound, color=BOUND_COLOR, lw=1.4, zorder=2)
        _axis(ax, title, False)
        ax.set_xlim(.9, 225)
        ax.set_xticks([1, 10, 200], labels=["1", "10", "200"])
        panels.append({"dataset": dataset, "curves": curves})
    return {"csv": str(path.relative_to(root)),
            "scope": "Numerically refined Gaussian/RBF expansion; fixed time-one laws at R=.2",
            "counts": "Every integer 1 through 200",
            "panels": panels}


def draw_polynomial(ax, root):
    root = Path(root)
    folder = root / "paper/figures/data/r1_dense"
    path = folder / "circle_fixed_laws.csv"
    rows = _read(path)
    curves = []
    for start, color, style, marker in (("slow", RESIDUAL_COLOR, "--", "o"),
                                        ("fast", FAST_COLOR, ":", "s")):
        cell = sorted((row for row in rows if row["start"] == start
                       and row["complete_pairs"] == "True"), key=lambda row: int(row["m"]))
        assert [int(row["m"]) for row in cell] == list(range(2, 201, 2))
        x = np.array([int(row["m"]) for row in cell])
        y = np.array([float(row["bound"]) for row in cell])
        residual = np.array([float(row["residual"]) for row in cell])
        assert all(int(row["effective_count"]) == int(row["m"])+2 for row in cell)
        assert all(float(row["R"]) == .1 and float(row["r"]) == .5 for row in cell)
        assert all(int(row["inspection_time"]) == 10 for row in cell)
        assert np.all(residual > Y_LIMITS[0]) and np.all(residual < y)
        assert np.all(y < Y_LIMITS[1]) and np.all(np.diff(residual) <= 0)
        assert np.ptp([float(row["full_mmd"]) for row in cell]) == 0
        # Fourier eigenvalues occur in sine/cosine pairs, so at even m the
        # exact bound is proportional to 1/(m+2), asymptotically 1/m.
        assert abs(np.polyfit(np.log(x + 2), np.log(y), 1)[0] + 1) < 1e-12
        ax.plot(x, residual, color=color, lw=1.05, ls=style, marker=marker,
                markevery=[0, 3, 7, 15, 31, 63, 99], ms=2.3, mew=.4, zorder=3)
        for row in cell:
            assert abs(float(row["relative_quadrature_error"])) < 2e-6
            assert float(row["min_initial_relative_density"]) == .9
        curves.append({"start": start, "points": [
            {"m": int(row["m"]), "x": int(row["m"]),
             "bound": float(row["bound"]), "residual": float(row["residual"]),
             "full_mmd": float(row["full_mmd"])} for row in cell]})
    ax.plot(x, y, color=BOUND_COLOR, lw=1.4, zorder=2)
    _axis(ax, "Synthetic circle", True)
    ax.set_xlim(1.8, 225)
    ax.set_xticks([2, 10, 200], labels=["2", "10", "200"])
    return {"csv": str(path.relative_to(root)),
            "target": "Uniform circle", "x_axis": "Actual diagnostic count m (even counts)",
            "scope": "Two fixed time-ten laws from piecewise-uniform initial densities; r=.5,R=.1",
            "asymptotic_slope": -1,
            "exact_bound": "R*sqrt(12/pi^2)/(m+2), because Fourier eigenvalues occur in pairs",
            "curves": curves}


def draw_residual_row(fig, axes, root):
    if len(axes) != 5:
        raise ValueError("One polynomial control and four regression plots are required")
    return {"panel_order": ["circle", "energy", "airfoil", "wine_red", "kin8nm"],
            "polynomial": draw_polynomial(axes[0], root),
            "regression": draw_regressions(axes[1:], root, CASES[1:]),
            "concrete": "Computed and retained in regression_curves.csv",
            "no_smoothing": True}
