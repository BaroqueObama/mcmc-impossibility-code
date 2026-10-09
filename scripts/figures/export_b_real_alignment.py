"""Export regression alignment tables from the runs listed in RUNS.

The angle table uses exact-integer refinements at selected checkpoints.
Unresolved estimates are blank. Learned-fit summaries report medians and
interquartile ranges across fits.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib

import numpy as np

from mcmc.paperexp.runio import output_root, resolve_output_path

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/data/r2_real_alignment"
STUDY = "b_real_alignment"

# Source runs under $MCMC_PAPER_OUT/b_real_alignment/pilot/.
RUNS = {'concrete': {'population': 's4-concrete-pop',
              'learned': 's4-concrete-learned',
              'precision': 's4-concrete-precision',
              'learned_refine': 's4-concrete-learned-refine'},
 'airfoil': {'population': 's4-airfoil-pop', 'learned': 's4-airfoil-learned'},
 'energy': {'population': 's4-energy-pop',
            'learned': 's4-energy-learned',
            'precision': 's4-energy-precision',
            'learned_refine': 's4-energy-learned-refine'},
 'wine_red': {'population': 's4-wine_red-pop',
              'learned': 's4-wine_red-learned',
              'precision': 's4-wine_red-precision',
              'learned_refine': 's4-wine_red-learned-refine'},
 'kin8nm': {'population': 's4-kin8nm-pop', 'learned': 's4-kin8nm-learned'}}
FIELDS = [
    "model", "dataset", "sampler", "feature_rank", "m", "t", "span_kind", "fit", "pilot_length", "lag",
    "sin_theta", "sin_uncertainty", "angle_deg", "sin_angle_upper", "angle_upper", "theorem_bound_status",
    "mode_error_deg", "learned_sin_upper", "learned_angle_upper", "lambda_m", "lambda_m1", "eigengap",
    "rate_a", "rate_b", "c_m_squared", "kappa_squared", "numerical_allowance", "empirical_sensitivity",
    "estimate_status", "cutoff_reason",
]  # fmt: skip
MODEL = "bayesian_linear_regression_exact_gaussian_posterior"
SAMPLER = "controlled_reversible_gaussian_ar"
OK = ("RESOLVED", "BELOW_NUMERICAL_SENSITIVITY")


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def fmt(x) -> str:
    """Full-precision text for floats (repr round-trips), empty for None/NaN."""
    if x is None or x == "":
        return ""
    v = float(x)
    if math.isnan(v):
        return ""
    return repr(v)


def run_dir(root: pathlib.Path, rid: str) -> pathlib.Path:
    p = root / STUDY / "pilot" / rid
    st = json.loads((p / "RUN-STATUS.json").read_text())
    if st["status"] != "COMPLETED":
        raise RuntimeError(f"{p} is {st['status']}")
    return p


def close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def true_base(r: dict, entry: dict, b: dict) -> dict:
    """Format a population row; retain raw estimates until applying the cutoff."""
    return {
        "model": MODEL, "dataset": r["dataset"], "sampler": SAMPLER, "feature_rank": r["feature_rank"],
        "m": int(r["m"]), "t": int(r["t"]), "span_kind": "true", "fit": -1, "pilot_length": "",
        "lag": entry["selected_lag"], "sin_theta": "", "sin_uncertainty": fmt(r["sin_uncertainty"]),
        "angle_deg": "", "sin_angle_upper": fmt(r["sin_angle_upper"]), "angle_upper": fmt(r["angle_upper"]),
        "theorem_bound_status": r["theorem_bound_status"],
        "mode_error_deg": "", "learned_sin_upper": "", "learned_angle_upper": "",
        "lambda_m": fmt(r["lambda_m"]), "lambda_m1": fmt(r["lambda_m1"]), "eigengap": fmt(r["eigengap"]),
        "rate_a": fmt(b["a_upper"]), "rate_b": fmt(b["b_lower"]),
        "c_m_squared": fmt(b["c_m_squared_lower"]), "kappa_squared": fmt(b["kappa2_upper"]),
        "numerical_allowance": fmt(r["numerical_allowance"]), "empirical_sensitivity": "",
        "estimate_status": r["estimate_status"], "cutoff_reason": "",
        "_s": r["sin_theta"], "_a": r["angle_deg"], "_reason": r["cutoff_reason"],
    }  # fmt: skip


def learned_status(upper: float) -> str:
    # Learned rows describe the envelope including mode-estimation error.
    # The true row at the same (m, t) gives the status of g_t itself.
    return (
        "TRIVIAL(CONDITIONAL_NUMERICAL_MARGIN)" if upper >= 1.0 else "CONDITIONAL_NUMERICAL_MARGIN"
    )


def learned_base(r: dict, entry: dict, b: dict) -> dict:
    failed = r["estimate_status"] == "FIT_FAILED"
    e = (lambda k: "") if failed else (lambda k: fmt(r[k]))
    return {
        "model": MODEL, "dataset": r["dataset"], "sampler": SAMPLER, "feature_rank": r["feature_rank"],
        "m": int(r["m"]), "t": int(r["t"]), "span_kind": "learned", "fit": int(r["fit"]),
        "pilot_length": int(r["pilot_length"]), "lag": entry["selected_lag"],
        "sin_theta": "", "sin_uncertainty": e("sin_uncertainty"), "angle_deg": "",
        "sin_angle_upper": e("sin_angle_upper"), "angle_upper": e("angle_upper"),
        "theorem_bound_status": "" if failed else learned_status(float(r["learned_sin_upper"])),
        "mode_error_deg": e("mode_error_deg"), "learned_sin_upper": e("learned_sin_upper"),
        "learned_angle_upper": e("learned_angle_upper"), "lambda_m": e("lambda_m"), "lambda_m1": e("lambda_m1"),
        "eigengap": e("eigengap"), "rate_a": fmt(b["a_upper"]), "rate_b": fmt(b["b_lower"]),
        "c_m_squared": fmt(b["c_m_squared_lower"]), "kappa_squared": fmt(b["kappa2_upper"]),
        "numerical_allowance": e("numerical_allowance"), "empirical_sensitivity": "",
        "estimate_status": r["estimate_status"], "cutoff_reason": "",
        "_s": "" if failed else r["sin_theta"], "_a": "" if failed else r["angle_deg"],
        "_reason": r["cutoff_reason"],
    }  # fmt: skip


def tier_reason(tier: dict, dps: tuple) -> str:
    own = tier.get("cutoff_reason", "")
    out = f"exact-integer whitening tier ({dps[1]} vs {dps[0]} digits)" + (
        f": {own}" if own else ""
    )
    return out


def promote_true(row: dict, tr: dict, dps: tuple) -> None:
    if not close(float(row["sin_angle_upper"]), tr["sin_angle_upper"], 1e-12):
        raise ValueError(f"g_t mismatch at {row['dataset']} m={row['m']} t={row['t']}")
    row.update({
        "sin_uncertainty": fmt(tr["sin_uncertainty"]), "lambda_m": fmt(tr["lambda_m"]),
        "lambda_m1": fmt(tr["lambda_m1"]), "eigengap": fmt(tr["eigengap"]),
        "numerical_allowance": fmt(tr["numerical_allowance"]), "estimate_status": tr["estimate_status"],
        "_s": tr["sin_theta"], "_a": tr["angle_deg"],
        "_reason": tier_reason(tr, dps),
    })  # fmt: skip


def promote_learned(row: dict, lr: dict, dps: tuple) -> None:
    if not (
        close(float(row["mode_error_deg"]), float(lr["mode_error_deg"]), 1e-9)
        and close(float(row["learned_sin_upper"]), float(lr["learned_sin_upper"]), 1e-9)
    ):
        raise ValueError(
            f"learned refine mismatch at {row['dataset']} fit={row['fit']} t={row['t']}"
        )
    row.update({
        "sin_uncertainty": fmt(lr["sin_uncertainty"]), "lambda_m": fmt(lr["lambda_m"]),
        "lambda_m1": fmt(lr["lambda_m1"]), "eigengap": fmt(lr["eigengap"]),
        "numerical_allowance": fmt(lr["numerical_allowance"]), "estimate_status": lr["estimate_status"],
        "_s": lr["sin_theta"], "_a": lr["angle_deg"],
        "_reason": tier_reason(lr, dps),
    })  # fmt: skip


def finalize(rows: list[dict]) -> list[dict]:
    """Show resolved estimates up to the first failed true-span checkpoint."""
    true_sorted = sorted((r for r in rows if r["span_kind"] == "true"), key=lambda r: r["t"])
    cut = next((r["t"] for r in true_sorted if r["estimate_status"] not in OK), None)
    out = []
    for r in rows:
        r = dict(r)
        s, a, reason = r.pop("_s"), r.pop("_a"), r.pop("_reason")
        beyond = cut is not None and r["t"] >= cut
        if beyond and r["estimate_status"] in OK:
            what = "true-span checkpoint" if r["span_kind"] == "learned" else "checkpoint"
            reason = f"after the first failed {what} t={cut}" + (f"; {reason}" if reason else "")
        show = r["estimate_status"] == "RESOLVED" and not beyond
        r["sin_theta"] = fmt(s) if show else ""
        r["angle_deg"] = fmt(a) if show else ""
        r["cutoff_reason"] = reason
        out.append(r)
    return out


def finalize_all(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["m"]), []).append(r)
    out = []
    for key in groups:
        out += finalize(groups[key])
    return out


def quantiles(v: list[float]) -> tuple:
    a = np.asarray([x for x in v if math.isfinite(x)])
    if a.size == 0:
        return (0, "", "", "")
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return (int(a.size), repr(float(q[1])), repr(float(q[0])), repr(float(q[2])))


def learned_summary(rows: list[dict], raw_by_ds: dict, entries: dict) -> list[dict]:
    summ = []
    for ds, lrows in raw_by_ds.items():
        lr = [r for r in rows if r["dataset"] == ds and r["span_kind"] == "learned"]
        for m in entries[ds]["selected_m"]:
            for L in sorted({r["pilot_length"] for r in lr}):
                for t in entries[ds]["time_grid"]:
                    cell = [r for r in lr if r["m"] == m and r["pilot_length"] == L and r["t"] == t]
                    q_s = quantiles([float(r["sin_theta"]) for r in cell if r["sin_theta"] != ""])
                    q_m = quantiles(
                        [float(r["mode_error_deg"]) for r in cell if r["mode_error_deg"] != ""]
                    )
                    q_u = quantiles(
                        [
                            float(r["learned_sin_upper"])
                            for r in cell
                            if r["learned_sin_upper"] != ""
                        ]
                    )
                    raw = [
                        r
                        for r in lrows
                        if int(r["m"]) == m and int(r["pilot_length"]) == L and int(r["t"]) == t
                    ]
                    q_r = quantiles(
                        [
                            float(r["residual_ratio"])
                            for r in raw
                            if r.get("residual_ratio_status") == "RESOLVED"
                        ]
                    )
                    summ.append({
                        "dataset": ds, "m": m, "pilot_length": L, "t": t, "n_fits": len(cell),
                        "n_failed": sum(r["estimate_status"] == "FIT_FAILED" for r in cell),
                        "n_resolved": q_s[0], "sin_theta_median": q_s[1], "sin_theta_q25": q_s[2],
                        "sin_theta_q75": q_s[3], "mode_error_deg_median": q_m[1], "mode_error_deg_q25": q_m[2],
                        "mode_error_deg_q75": q_m[3], "learned_sin_upper_median": q_u[1],
                        "learned_sin_upper_q25": q_u[2], "learned_sin_upper_q75": q_u[3],
                        "residual_ratio_n_resolved": q_r[0], "residual_ratio_median": q_r[1],
                        "residual_ratio_q25": q_r[2], "residual_ratio_q75": q_r[3],
                    })  # fmt: skip
    return summ


def main() -> None:
    root = output_root()
    selection = json.loads((OUT / "selection.json").read_text())
    entries = selection["datasets"]
    refined = {
        (r["dataset"], int(r["m"])): sorted(int(t) for t in r["times"])
        for r in selection["numerical_refinements"]
    }
    base_rows, raw_learned = [], {}
    promoted_true, promoted_learned = {}, {}
    for ds, runs in RUNS.items():
        entry = entries[ds]
        paths = {k: run_dir(root, v) for k, v in runs.items()}
        pop = json.loads((paths["population"] / "population.json").read_text())
        prow = [r for r in read_csv(paths["population"] / "population_rows.csv")
                if r["landmark_set"] == "primary"]
        lrows = read_csv(paths["learned"] / "learned_rows.csv")
        if not prow or not lrows:
            raise ValueError(f"{ds}: empty population or learned rows")
        for row in prow:
            if row["estimate_status"] != "OPERATOR_UNRESOLVED":
                if float(row.get("eigvec_residual_ratio") or 0) >= 1:
                    raise ValueError(f"{ds}: eigensolver residual exceeds the declared allowance")
        raw_learned[ds] = lrows
        bi = pop["per_landmarks"]["feat"]["bound_inputs"]
        base_rows += [true_base(r, entry, bi[r["m"]]) for r in prow]
        base_rows += [learned_base(r, entry, bi[r["m"]]) for r in lrows]
        if "precision" in paths:
            precision = json.loads((paths["precision"] / "precision.json").read_text())
            m = int(precision["m"])
            if sorted(int(t) for t in precision["results"]) != refined[(ds, m)]:
                raise ValueError(f"{ds}: precision run times differ from the selected design")
            dps = tuple(sorted(int(x) for x in next(iter(precision["results"].values()))["tiers"]))
            for t, result in precision["results"].items():
                promoted_true[(ds, m, int(t))] = (result["row"], dps)
            lref = read_csv(paths["learned_refine"] / "learned_refined_rows.csv")
            lrj = json.loads((paths["learned_refine"] / "learned_refine.json").read_text())
            if resolve_output_path(lrj["learned_run"]) != paths["learned"]:
                raise ValueError(f"{ds}: learned refinement used a different set of pilot spans")
            for row in lref:
                promoted_learned[(ds, m, int(row["fit"]), int(row["pilot_length"]), int(row["t"]))] = (row, dps)
    primary = []
    n_true = n_learned = 0
    for row in base_rows:
        row = dict(row)
        key = (row["dataset"], row["m"], row["t"])
        if row["span_kind"] == "true" and key in promoted_true:
            promote_true(row, *promoted_true[key])
            n_true += 1
        elif row["span_kind"] == "learned":
            key = (row["dataset"], row["m"], row["fit"], row["pilot_length"], row["t"])
            if key in promoted_learned:
                if row["estimate_status"] == "FIT_FAILED":
                    raise ValueError("refined row for a failed fit")
                promote_learned(row, *promoted_learned[key])
                n_learned += 1
        primary.append(row)
    expected_true = sum(len(t) for t in refined.values())
    if n_true != expected_true or n_learned != expected_true * 30 * 2:
        raise ValueError(f"refinement counts {n_true}/{n_learned} differ from the selected design")
    rows = finalize_all(primary)
    summary = learned_summary(rows, raw_learned, entries)
    write_csv(OUT / "angles.csv", rows, FIELDS)
    write_csv(OUT / "learned_summary.csv", summary, list(summary[0]))
    print(f"wrote {len(rows)} angles and {len(summary)} learned-fit summaries")


if __name__ == "__main__":
    main()
