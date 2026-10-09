"""Extract synthetic figure data from analyze_r3b.py summaries."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np



def compact(s: dict, J: int, m: int) -> dict:
    t = s["setup"]["t_grid"]
    ji, mi = s["setup"]["J_grid"].index(J), s["setup"]["m_grid"].index(m)
    out = {
        "n_ensembles": s["n_ensembles"],
        "t_grid": t,
        "R": s["setup"]["r_init"],
        "b_t": s["setup"]["b_t"],
        "b_pi": s["setup"]["b_pi"],
        "e_tN": s["setup"]["e_tN"],
        "J_grid": s["setup"]["J_grid"],
        "N_grid": s["setup"]["N_grid"],
        "truth": {k: s["truth"][k] for k in ("D", "L", "U", "b_ref_t", "beta_ref")},
        "joint_guaranteed_sampled": s["joint_guaranteed_sampled"],
        **{
            k: s[k]
            for k in (
                "applicability",
                "joint_assumed_radius_sampled",
                "init_chi2",
                "at_true_radius",
            )
            if k in s
        },
        "embedding_status": s["embedding_status"],
        "corrections": {
            k: s["corrections"][k] for k in ("quad_margin_known", "omega_max", "inflation_max")
        },
        "methods": {},
        "paired": {},
        "transition_matched": [
            {**tm, "mean_B3_minus_B4_last_primary": tm["mean_B3_minus_B4"][-1][mi][0]}
            for tm in s["transition_matched"]
        ],
        "cost_units": s.get("cost_units"),
        "cost_comparison_sampled": [
            c for c in s.get("cost_comparison", []) if c["ref"] == "sampled"
        ],
        "cost": s["cost"],
    }
    for tm in out["transition_matched"]:
        tm.pop("mean_B3_minus_B4")
        tm.pop("se")
    for name, md in s["methods"].items():
        rec = md[f"J{J}|m{m}|sampled"]
        vals = list(md.values())
        out["methods"][name] = {
            "label": s["labels"][name],
            "primary": {
                k: rec[k]
                for k in (
                    "n",
                    "covered_all",
                    "any_violation",
                    "unresolved",
                    "cp_if_unresolved_fail",
                    "cp_if_unresolved_succeed",
                    "B_median",
                    "B_q05",
                    "B_q95",
                    "ratio_to_D_median",
                    "H_median",
                    "U_median",
                    "cross_0.1",
                    "cross_0.05",
                )
                if k in rec
            },
            "all_cells": {
                "min_covered_all": min(v["covered_all"] for v in vals),
                "max_any_violation": max(v["any_violation"] for v in vals),
                "max_unresolved": max(v["unresolved"] for v in vals),
                "n_cells": len(vals),
                "ensemble_invariant": rec["ensemble_invariant"],
                "t_unresolved_any_cell": sorted(
                    {t_ for v in vals for t_ in v.get("t_unresolved", [])}
                ),
            },
        }
        if "cost_last" in rec:
            out["methods"][name]["primary"]["cost_last"] = rec["cost_last"]
    for key, pr in s["paired"].items():
        mean, se = np.array(pr["mean"]), np.array(pr["se"])
        out["paired"][key] = {
            "mean_primary": mean[:, ji, mi, 0].tolist(),
            "se_primary": se[:, ji, mi, 0].tolist(),
            "min_all_cells": pr["min"],
            "argmin": pr["argmin"],
            "frac_cells_B3_below_B4": pr["frac_negative_cells"],
            **{
                k: pr[k]
                for k in (
                    "n_entries",
                    "n_negative",
                    "n_entries_t_pos",
                    "n_negative_t_pos",
                    "min_t_pos",
                    "n_negative_last_t",
                    "n_negative_t_pos_1e5",
                    "n_negative_last_t_1e5",
                    "n_negative_by_t_1e5",
                )
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--J", type=int, default=32768)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument(
        "--analyze-cmd",
        default="scripts/paper_experiments/analyze_r3b.py --phase main",
        help="the analyze_r3b.py command that produced --analysis-dir",
    )
    ap.add_argument("--figure", default="paper/figures/fig_r3b_main", help="figure path")
    ap.add_argument("--name", default="r3b_main_compact.json", help="compact data file name")
    a = ap.parse_args()
    src = pathlib.Path(a.analysis_dir)
    files = sorted(src.glob("*.summary.json"))
    if not files:
        raise FileNotFoundError(f"no summaries in {src}")
    data = {}
    for f in files:
        s = json.loads(f.read_text())
        if s["run_status"] != "COMPLETED":
            raise RuntimeError(f"{f.name} is {s['run_status']}: extract only COMPLETED samples")
        data["__".join(f.name.split("__")[:2])] = compact(s, a.J, a.m)  # <target>__<sampler>
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / a.name).write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"wrote {out}/{a.name} ({len(data)} configurations)")


if __name__ == "__main__":
    main()
