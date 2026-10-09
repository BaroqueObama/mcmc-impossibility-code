"""Summarize the saved logistic alignment CSVs.

Computes displayed windows, median angles and envelopes across 30 fits at
L=16384, half-budget comparisons and status counts. These are empirical
NUTS-reference comparisons. Use --out to choose the output JSON path.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
D = REPO / "paper/figures/data/r2_real_alignment"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=D / "report_numbers_logistic.json")
    args = parser.parse_args()
    a = pd.read_csv(D / "angles_logistic.csv")
    r = pd.read_csv(D / "reference_logistic.csv")
    s = pd.read_csv(D / "status_logistic.csv")
    a["unit"] = a["model"] + "__" + a["sampler"]
    out = {
        "rows": int(len(a)),
        "rows_by_budget": a["budget"].value_counts().to_dict(),
        "status_counts": a.groupby("budget")["estimate_status"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict("index"),
        "failed_fits": int((a.estimate_status == "FAILED_FIT").sum()),
        "pairs": [],
    }
    r["e_ratio"] = r["e"] / r["projected_e_fresh"]
    out["e_realized_over_projected"] = {
        b: [float(g.e_ratio.min()), float(g.e_ratio.median()), float(g.e_ratio.max())]
        for b, g in r.groupby("budget")
    }
    out["witness_ratio"] = {
        b: [
            float(g.witness_ratio.min()),
            float(g.witness_ratio.median()),
            float(g.witness_ratio.max()),
        ]
        for b, g in r.groupby("budget")
    }
    for (b, u, m), g in a.groupby(["budget", "unit", "m"], sort=False):
        rr = r[(r.budget == b) & (r.unit == u) & (r.m == m)].set_index("t")
        shown = sorted(t for t in rr.index if not isinstance(rr.loc[t, "cutoff"], str))
        rec = {
            "budget": b,
            "unit": u,
            "m": int(m),
            "n_checkpoints": int(len(rr)),
            "displayed": shown,
        }
        if shown:
            g16 = g[(g.pilot_length == 16384) & (g.t.isin(shown))]
            g4 = g[(g.pilot_length == 4096) & (g.t.isin(shown))]
            med = g16.groupby("t")["angle_deg"].median()
            env = g16.groupby("t")["learned_angle_upper"].median()
            med4 = g4.groupby("t")["angle_deg"].median()
            rec.update(
                angle_median_range=[float(med.min()), float(med.max())],
                envelope_median_range=[float(env.min()), float(env.max())],
                angle_median_range_L4096=[float(med4.min()), float(med4.max())],
                n_envelope_trivial_rows=int(
                    (g16.estimate_status == "TRIVIAL(EMPIRICAL_SENSITIVITY_ONLY)").sum()
                ),
                n_displayed_half_budget_inconsistent=int((rr.loc[shown, "stable"] != True).sum()),  # noqa: E712
                dk_term_range=[
                    float(rr.loc[shown, "dk_deg"].min()),
                    float(rr.loc[shown, "dk_deg"].max()),
                ],
            )
        out["pairs"].append(rec)
        print(rec)
    print(
        {
            k: out[k]
            for k in (
                "rows",
                "rows_by_budget",
                "failed_fits",
                "e_realized_over_projected",
                "witness_ratio",
            )
        }
    )
    print(out["status_counts"])
    print(s.to_string())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
