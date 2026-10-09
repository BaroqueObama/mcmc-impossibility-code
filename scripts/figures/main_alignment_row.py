"""Plot principal angles to the slow-mode span in the saved feature geometry.

Red curves are the Section 4 envelope, conditional on numerical margins.
Unresolved angles leave gaps; hollow markers give numerical upper limits.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ANGLE_COLOR = "#0072B2"
DISPLAY_COUNTS = {"concrete": 2, "energy": 2, "airfoil": 1, "wine_red": 2, "kin8nm": 9}
BOUND_COLOR = "#C62828"
SINE_LIMITS = (1e-24, 1.0)
Y_LIMITS = (-3.0, 93.0)
CASES = [("concrete", "Concrete"), ("energy", "Energy"),
         ("airfoil", "Airfoil"), ("wine_red", "Wine"), ("kin8nm", "kin8nm")]


def _number(value):
    return float(value) if value != "" else None


def _inside(values, description):
    """Check that all finite sine values fit the displayed range."""
    for value in values:
        if value is not None:
            assert np.isfinite(value) and SINE_LIMITS[0] <= value <= SINE_LIMITS[1], (
                description, value, SINE_LIMITS)


def _degrees(value):
    return None if value is None else float(np.degrees(np.arcsin(value)))


def draw_alignment_row(fig, axes, root):
    """Draw into five existing axes; the caller provides shared labels/legend."""
    if len(axes) != 5:
        raise ValueError("The alignment row requires five axes")
    root = Path(root)
    folder = root / "paper/figures/data/r2_real_alignment"
    paths = {name: folder / name for name in ("angles.csv", "selection.json")}
    with paths["angles.csv"].open() as stream:
        all_rows = list(csv.DictReader(stream))
    selection = json.loads(paths["selection.json"].read_text())

    # Diagnostic counts were selected before evaluating the alignment curves.
    frozen = dict(selection["datasets"])
    rows = [row for row in all_rows if row["span_kind"] == "true"]
    statuses = Counter(row["estimate_status"] for row in rows)
    assert statuses == {"RESOLVED": 73, "BELOW_NUMERICAL_SENSITIVITY": 15}
    assert len(rows) == 88
    counts = {ds: sorted(frozen[ds]["selected_m"]) for ds, _ in CASES}
    assert counts == {"concrete": [1, 2], "energy": [1, 2], "airfoil": [1],
                      "wine_red": [1, 2], "kin8nm": [9]}
    facts = {
        "sampler": "Controlled reversible Gaussian autoregressive sampler",
        "targets": "Gaussian regression posteriors of five real datasets",
        "quantity": "theta_max(W_m, V*_m(t)), in degrees",
        "theorem_bound": "min(1, kappa_squared / c_m_squared * (rate_a / rate_b)**(2*t))",
        "display_conversion": "Saved sine values and sine bounds are converted by degrees(arcsin(value)); no data are changed",
        "bound_status": "CONDITIONAL_NUMERICAL_MARGIN, not an interval-certified or chain-only bound",
        "count_selection": "Counts were selected by an independent pilot; the main figure shows "
                           "m=2 where both 1 and 2 were selected, otherwise the sole count.",
        "displayed_counts": DISPLAY_COUNTS,
        "numerical_display": "Resolved angles are shown after exact unit conversion. Hollow downward "
                             "markers show degrees(arcsin(11*sin_uncertainty)) for unresolved angles; these are numerical "
                             "upper limits, not measured violations. Lines break across those checkpoints. "
                             "No zero substitution, smoothing, or interpolation across missing points.",
        "ylim": list(Y_LIMITS), "cases": {}, "checkpoint_status_counts": dict(statuses),
    }
    totals = Counter()
    for ax, (dataset, title) in zip(axes, CASES):
        facts["cases"][dataset] = {}
        observed_counts = sorted({int(row["m"]) for row in rows if row["dataset"] == dataset})
        assert observed_counts == counts[dataset]
        for m in counts[dataset]:
            cell = sorted((row for row in rows if row["dataset"] == dataset and int(row["m"]) == m),
                          key=lambda row: int(row["t"]))
            times = [int(row["t"]) for row in cell]
            assert times == frozen[dataset]["time_grid"]
            assert all(int(row["feature_rank"]) == 512 for row in cell)
            assert all(row["sampler"] == "controlled_reversible_gaussian_ar" for row in cell)
            points, first_cutoff = [], None
            for row in cell:
                t = int(row["t"])
                status = row["estimate_status"]
                if status in ("OPERATOR_UNRESOLVED", "PRECISION_UNSTABLE") and first_cutoff is None:
                    first_cutoff = t
                s, e, bound = _number(row["sin_theta"]), float(row["sin_uncertainty"]), float(row["sin_angle_upper"])
                assert "CONDITIONAL_NUMERICAL_MARGIN" in row["theorem_bound_status"]
                assert 0 < bound <= 1 and e >= 0
                on_curve = first_cutoff is None and not row["cutoff_reason"].startswith("after")
                if status == "RESOLVED" and on_curve:
                    assert s is not None and s > 10 * e and s <= bound
                    totals["resolved_s_le_bound"] += 1
                    if bound < 1:
                        # Check numerical uncertainty when the bound is nontrivial.
                        assert s + e <= bound
                        totals["resolved_nontrivial_s_plus_e_le_bound"] += 1
                else:
                    assert s is None
                below = status == "BELOW_NUMERICAL_SENSITIVITY" and on_curve
                level = 11 * e if below else None
                if below and level <= bound:
                    totals["below_sensitivity_level_le_bound"] += 1
                totals["nontrivial_envelope_checkpoints"] += int(bound < 1)
                points.append({"t": t, "sin_theta": s, "numerical_uncertainty": e,
                               "sensitivity_upper_level": level, "bound": bound,
                               "status": status, "cutoff_reason": row["cutoff_reason"],
                               "numerical_tier": "exact-integer" if row["cutoff_reason"].startswith("exact-integer") else "float64",
                               "theorem_bound_status": row["theorem_bound_status"]})
            sine = [p["sin_theta"] for p in points]
            bound = [p["bound"] for p in points]
            levels = [p["sensitivity_upper_level"] for p in points]
            _inside(sine, (dataset, m, "resolved angles"))
            _inside(bound, (dataset, m, "theorem bound"))
            _inside(levels, (dataset, m, "sensitivity levels"))
            if m == DISPLAY_COUNTS[dataset]:
                ax.plot(times, [_degrees(value) for value in bound], color=BOUND_COLOR, lw=1.2, ls="-", zorder=2)
                # None leaves a gap at every unresolved checkpoint.
                ax.plot(times, [_degrees(value) for value in sine], color=ANGLE_COLOR, lw=1.0, ls="-",
                        marker="o", ms=2.6, mew=0.35, mec="white", zorder=4)
                unresolved = [p for p in points if p["sensitivity_upper_level"] is not None]
                ax.plot([p["t"] for p in unresolved],
                        [_degrees(p["sensitivity_upper_level"]) for p in unresolved],
                        ls="none", marker="v", ms=3.4, mfc="white",
                        mec=ANGLE_COLOR, mew=.7, zorder=5)
            facts["cases"][dataset][str(m)] = {
                "selection_lag": frozen[dataset]["selected_lag"],
                "displayed": m == DISPLAY_COUNTS[dataset],
                "first_numerical_cutoff": first_cutoff, "points": points,
            }
        m = DISPLAY_COUNTS[dataset]
        assert m in counts[dataset]
        count_label = f"{m} diagnostic" + ("s" if m > 1 else "")
        ax.set_title(f"{title} ({count_label})", pad=3, fontsize=7, color="black")
        ax.set_xscale("symlog", linthresh=1, base=2)
        ax.set_xlim(-0.15, 650)
        ax.set_xticks([0, 8, 64, 512], labels=["0", "8", "64", "512"])
        ax.set_yscale("linear")
        ax.set_ylim(*Y_LIMITS)
        ax.set_yticks([0, 45, 90], labels=["0", "45", "90"])
        ax.minorticks_off()
        ax.tick_params(axis="both", colors="black", labelsize=7, length=2.2, width=0.5, pad=1.5)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", which="major", color="#e7e7e7", lw=0.45)
        ax.set_axisbelow(True)
    assert totals == {"resolved_s_le_bound": 73, "resolved_nontrivial_s_plus_e_le_bound": 29,
                      "below_sensitivity_level_le_bound": 4, "nontrivial_envelope_checkpoints": 44}
    facts["comparison_counts"] = dict(totals)
    displayed_points = [point for dataset in facts["cases"].values()
                        for span in dataset.values() if span["displayed"]
                        for point in span["points"]]
    facts["displayed_checkpoint_status_counts"] = dict(Counter(p["status"] for p in displayed_points))
    assert facts["displayed_checkpoint_status_counts"] == {
        "RESOLVED": 46, "BELOW_NUMERICAL_SENSITIVITY": 9}
    return facts
