"""Plot median anytime bounds and exact MMD from saved regression results.

Each dataset has 500 ensembles on the same inspection grid. Float64 MMD
underflow is marked using the saved log-scale values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator


SECTION3_COLOR = "#D55E00"
SECTION4_COLOR = "#7B3294"
DISCREPANCY_BLUE = "#0072b2"
METHOD_KEYS = {
    "sec3": "B3b_N2048",
    "sec3_ideal": "B3a",
    "sec4": "B4b_own_N2048",
    "sec4_ideal": "B4a_own",
    "direct": "direct",
}
PLOTTED_METHODS = (
    ("sec3", SECTION3_COLOR, "--", "Section 3, estimated covariance"),
    ("sec4", SECTION4_COLOR, "--", "Section 4, estimated covariance"),
    ("sec3_ideal", SECTION3_COLOR, "-", "Section 3, supplied covariance"),
    ("sec4_ideal", SECTION4_COLOR, "-", "Section 4, supplied covariance"),
)
DATASETS = (
    ("concrete", "Concrete"),
    ("energy", "Energy"),
    ("airfoil", "Airfoil"),
    ("wine_red", "Wine"),
    ("kin8nm", "kin8nm"),
)


CHECKPOINTS = [0] + [2**j for j in range(17)]
TICKS = [0, 16, 1024, 65536]
TICK_LABELS = ["0", "16", r"$2^{10}$", r"$2^{16}$"]


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def draw_anytime_row(fig, axes, root) -> dict:
    """Draw medians over 500 ensembles; the caller supplies labels and legend."""
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if len(axes) != len(DATASETS):
        raise ValueError("The anytime row requires exactly five axes.")
    if any(ax.figure is not fig for ax in axes):
        raise ValueError("All supplied axes must belong to the supplied figure.")

    root = Path(root)
    relative = Path("paper/figures/data/c_common_horizon")
    curve_path, coverage_path = root / relative / "curves.csv", root / relative / "coverage.csv"
    rows, coverage = _read(curve_path), _read(coverage_path)
    facts = {
        "filter": {"ref": "sampled", "J": 16384, "m": 8,
                   "summary_cell": "0|16384|8|sampled"},
        "export_status": "final",
        "n_repetitions": 500,
        "planned_repetitions": 500,
        "ensemble_ids": [2000000, 2000499],
        "initialization": "slow mean shift, R_init=0.8 (regression setting 0)",
        "target_reference": "sampled, 65536 independent exact target draws",
        "aggregation": "pointwise median bound across repetitions; exact population MMD",
        "bands": "omitted; available CSV quantiles describe repetition variability, not confidence bands",
        "plot": {
            "x": "common Gibbs-sweep checkpoints 0, 1, 2, ..., 65536; symlog base 2 with linthresh=1",
            "y": "MMD, common linear range [0, 0.5]",
            "interpolation": "straight segments between saved inspection times; no smoothing",
            "orange_dashed": "Section 3, covariance estimated from N=2048 blocks (sec3 / B3b_N2048)",
            "purple_dashed": "Section 4, covariance estimated from N=2048 blocks (sec4 / B4b_own_N2048)",
            "orange_solid": "Section 3, analytically supplied covariance with landmark compression and omitted-trace bound (sec3_ideal / B3a)",
            "purple_solid": "Section 4, analytically supplied covariance with landmark compression and omitted-trace bound (sec4_ideal / B4a_own)",
            "supporting_comparisons": "Direct full-MMD bound remains in facts and appendix table",
            "blue_circles": "exact MMD of the current chain law; hollow circles mark floating-point underflow, with log-scale values preserved",
            "operator_floor": "reported in provenance only; omitted from the plot to avoid a nearly overlapping additional line",
        },
        "guarantee_scope": [
            "The confidence level is 95% simultaneously over inspection times within a setting, not jointly across datasets.",
            "A plotted median is a summary of repeated bounds, not one realized confidence sequence.",
            "Dashed Section 3 and Section 4 bounds include covariance estimation uncertainty; solid bounds use supplied analytic covariance operators.",
            "The direct bound uses no covariance operator or initialization radius; it retains the same replica and target-reference confidence allowances.",
            "Numerical coverage qualifications for Gram-matrix formation remain as described in the appendix.",
            "The Section 3 and Section 4 bounds use a supplied initialization radius; all four plotted bounds require independent exact-target reference sampling in this experiment.",
        ],
        "failure_budget": {"total": 0.05, "replicas": 0.025,
                           "target_reference": 0.0125, "estimated_operator": 0.0125},
        "validation": "Medians, exact MMD and all-times coverage counts are checked against the saved summaries.",
        "datasets": {},
    }

    for index, (ax, (dataset, title)) in enumerate(zip(axes, DATASETS)):
        selected = {
            method: sorted(
                (
                    row for row in rows
                    if row["dataset"] == dataset and row["ref"] == "sampled"
                    and row["method"] == METHOD_KEYS[method]
                ),
                key=lambda row: int(row["t"]),
            )
            for method in METHOD_KEYS
        }
        if any(not series for series in selected.values()):
            raise ValueError(f"Missing saved anytime results for {dataset}.")
        times = [int(row["t"]) for row in selected["sec4"]]
        if times != CHECKPOINTS or any(
            times != [int(row["t"]) for row in series] for series in selected.values()
        ):
            raise ValueError(f"Duplicate or mismatched inspection times for {dataset}.")
        truth = [float(row["truth_fp64"]) for row in selected["sec4"]]
        if any(truth != [float(row["truth_fp64"]) for row in series]
               for series in selected.values()):
            raise ValueError(f"Inconsistent exact discrepancy for {dataset}.")
        summary_relative = relative / f"summary_{dataset}.json"
        summary_path = root / summary_relative
        summary = json.loads(summary_path.read_text())
        setup = summary["setup"]
        if times != setup["checkpoints"] or truth != setup["truth"][0]:
            raise ValueError(f"Figure and original experiment disagree for {dataset}.")
        if (setup["r_init"] != [0.8] or setup["J_grid"] != [16384]
                or setup["m_grid"] != [8] or setup["N_grid"] != [2048]
                or summary["n_ensembles"] != 500
                or summary["reps_range"] != [2000000, 2000499]):
            raise ValueError(f"Unexpected final design for {dataset}.")

        for method, color, linestyle, label in PLOTTED_METHODS:
            ax.plot(times, [float(row["B_median"]) for row in selected[method]],
                    color=color, linestyle=linestyle, linewidth=1.25,
                    label=label, zorder=3 if method == "sec4" else 4)
        exact_rows = selected["sec4"]
        statuses = [row["truth_status"] for row in exact_rows]
        exact_log10 = [float(row["log10_truth"]) for row in exact_rows]
        analytic = summary["meta"]["analytic_horizon"]
        if (statuses != [row["status"] for row in analytic]
                or exact_log10 != [row["log10_mmd"] for row in analytic]
                or any(status not in {"RESOLVED", "MMD2_UNDERFLOW", "BELOW_FLOAT64"}
                       for status in statuses)):
            raise ValueError(f"MMD underflow records disagree for {dataset}.")
        # Use log-scale values when MMD^2 underflows in float64.
        display_truth = np.array([
            float(row["truth_fp64"]) if row["truth_status"] == "RESOLVED"
            else float(row["truth_from_log"] or 0.0) for row in exact_rows
        ])
        resolved = np.array([status == "RESOLVED" for status in statuses])
        ax.plot(times, display_truth, color=DISCREPANCY_BLUE, linewidth=1.1,
                label="Actual discrepancy", zorder=5)
        time_array = np.array(times)
        ax.plot(time_array[resolved], display_truth[resolved], linestyle="none",
                color=DISCREPANCY_BLUE, marker="o", markersize=2.0,
                markeredgewidth=0, zorder=6, clip_on=False)
        ax.plot(time_array[~resolved], display_truth[~resolved], linestyle="none",
                marker="o", markerfacecolor="white", markeredgecolor=DISCREPANCY_BLUE,
                markersize=2.2, markeredgewidth=0.55, zorder=6, clip_on=False)

        ax.set_xscale("symlog", base=2, linthresh=1, linscale=1)
        ax.set_xlim(0, times[-1])
        ax.set_xticks(TICKS)
        ax.set_xticklabels(TICK_LABELS, fontsize=7, color="black")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yscale("linear")
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.2, 0.4])
        ax.tick_params(axis="both", labelsize=7, colors="black", pad=1.5)
        ax.tick_params(axis="y", labelleft=index == 0)
        if index == 0:
            ax.set_ylabel("MMD", fontsize=7, color="black", labelpad=2)
        ax.set_title(title, fontsize=7, color="black", pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        for name in ("bottom", "left"):
            ax.spines[name].set_color("black")
            ax.spines[name].set_linewidth(0.5)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.4)
        ax.set_axisbelow(True)

        dataset_facts = {
            "title": title,
            "checkpoints": times,
            "exact_MMD": truth,
            "exact_MMD_status": statuses,
            "log10_exact_MMD": exact_log10,
            "displayed_MMD": display_truth.tolist(),
            "n_repetitions": int(summary["n_ensembles"]),
            "summary_source": str(summary_relative),
            "operator_error_radius": [row[setup["N_grid"].index(2048)] for row in setup["e_tN"]],
            "operator_floor_R_sqrt_e": [
                float(0.8 * np.sqrt(row[setup["N_grid"].index(2048)]))
                for row in setup["e_tN"]
            ],
            "methods": {},
        }
        for method, series in selected.items():
            method_key = METHOD_KEYS[method]
            if any(row["method"] != method_key for row in series):
                raise ValueError(f"Unexpected source method for {dataset}/{method}.")
            original = summary["methods"][method_key]["0|16384|8|sampled"]
            bound = [float(row["B_median"]) for row in series]
            if bound != original["B_median"]:
                raise ValueError(f"Figure and original bound disagree for {dataset}/{method}.")
            if not np.isfinite(bound).all() or not np.isfinite(truth).all():
                raise ValueError(f"Non-finite bound or discrepancy for {dataset}/{method}.")
            if any(value < target for value, target in zip(bound, truth)):
                raise ValueError(f"Median bound below exact MMD for {dataset}/{method}.")
            if any(int(row["n_ens"]) != dataset_facts["n_repetitions"] for row in series):
                raise ValueError(f"Inconsistent repetition counts for {dataset}/{method}.")
            matches = [
                row for row in coverage
                if row["dataset"] == dataset and row["ref"] == "sampled"
                and row["method"] == method_key
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous coverage for {dataset}/{method}.")
            record = matches[0]
            n_repetitions = dataset_facts["n_repetitions"]
            if (int(record["covered_all_t"]) != original["covered_all_t"]
                    or int(record["n_ens"]) != n_repetitions
                    or int(record["any_violation"]) != n_repetitions - original["covered_all_t"]):
                raise ValueError(f"Figure and original all-times coverage disagree for {dataset}/{method}.")
            if [float(record["cp_lo"]), float(record["cp_hi"])] != original["cp"]:
                raise ValueError(f"Coverage intervals disagree for {dataset}/{method}.")
            dataset_facts["methods"][method] = {
                "method_key": method_key,
                "operator": "none" if method == "direct" else ("supplied covariance" if method.endswith("_ideal") else "estimated covariance, N=2048"),
                "drawn": method in {entry[0] for entry in PLOTTED_METHODS},
                "bound_median": bound,
                "final_bound_median": bound[-1],
                "covered_all_inspections": int(record["covered_all_t"]),
                "any_violation": int(record["any_violation"]),
                "unresolved": 0,
                "coverage_interval_95": [float(record["cp_lo"]), float(record["cp_hi"])],
            }
        facts["datasets"][dataset] = dataset_facts

    return facts
