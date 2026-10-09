"""Tabulate synthetic sensitivity checks and conventional-start results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib

import numpy as np


CELL = "J32768|m8|sampled"


def _cr(rec: dict, eps: str) -> str:
    c = rec[f"cross_{eps}"]
    t = c["t_median_crossed"]
    return f"{c['n_crossed']}@{int(t)}" if t is not None else "0"


def row(arm: str, s: dict) -> dict:
    N = s["setup"]["N_grid"][-1]
    meth = s["methods"]
    lab = s["labels"]
    # Labels distinguish valid guarantees, assumed-radius bounds and plug-in estimates.
    g = [n for n in meth if lab[n].startswith("guarantee")]
    ar = [n for n in meth if lab[n].startswith("assumed radius")]
    p = [n for n in meth if lab[n].startswith("plug-in")]
    if sorted(g + ar + p) != sorted(meth):
        raise ValueError(f"unclassified method labels: {lab}")
    gcells = [v for n in g for v in meth[n].values()]
    pun = {
        n: (
            max(v["unresolved"] for v in meth[n].values()),
            next(iter(meth[n].values()))["ensemble_invariant"],
        )
        for n in p
    }

    def _fs(names):
        return sum(
            v[f"cross_{e}"]["false_stops_definite"]
            for n in names
            for v in meth[n].values()
            for e in ("0.05", "0.1")
        )

    js = s["joint_guaranteed_sampled"]
    jk = s.get("joint_guaranteed_known")
    p_own, p_same = s["paired"][f"B3_N{N}-B4own_N{N}"], s["paired"][f"B3_N{N}-B4same_N{N}"]
    mi = s["setup"]["m_grid"].index(8)
    ji = s["setup"]["J_grid"].index(32768)
    co = s["corrections"]
    tr = s["truth"]
    r = {
        "arm": arm,
        "config": s["run_dir"].split("/")[-1].split("__")[0]
        + "__"
        + s["run_dir"].split("/")[-1].split("__")[1],
        "n": s["n_ensembles"],
        "status": s["run_status"],
        "T3": s["setup"]["t_grid"][-1],
        "embedding": s["embedding_status"]["kind"],
        "joint_sampled": f"{js['covered_all']}/{js['any_violation']}/{js['unresolved']}",
        "joint_known": f"{jk['covered_all']}/{jk['any_violation']}/{jk['unresolved']}"
        if jk
        else "",
        "guar_min_covered": min(v["covered_all"] for v in gcells),
        "guar_max_violation": max(v["any_violation"] for v in gcells),
        "guar_max_unresolved": max(v["unresolved"] for v in gcells),
        "plugin_max_violation": max(max(v["any_violation"] for v in meth[n].values()) for n in p),
        "applicable_guarantees": ";".join(g),
        "plugin_unresolved": ";".join(
            f"{n}:{u}{'(inv)' if inv else ''}" for n, (u, inv) in pun.items() if u
        ),
        "false_stops": _fs(g),  # guaranteed (applicable) methods
        "false_stops_other": _fs(ar + p),
        "fallback_events": len(s["union"].get("fallback_events", [])),
        "D_last": tr["D"][-1],
        "L_last": tr["L"][-1],
        "U_last": tr["U"][-1],
    }
    for name, key in (
        ("direct", "direct"),
        (f"B3_N{N}", "B3"),
        (f"B4same_N{N}", "B4same"),
        (f"B4own_N{N}", "B4own"),
        (f"NO4_N{N}", "NO4"),
        ("B4ref_own", "B4ref_own"),
    ):
        rec = meth[name][CELL]
        r[f"{key}_x010"] = _cr(rec, "0.1")
        r[f"{key}_x005"] = _cr(rec, "0.05")
        r[f"{key}_B_last"] = rec["B_median"][-1]
    for key, pr in (("own", p_own), ("same", p_same)):
        r[f"B3mB4{key}_last"] = float(np.array(pr["mean"])[-1, ji, mi, 0])
        r[f"B3mB4{key}_neg_tpos"] = f"{pr['n_negative_t_pos']}/{pr['n_entries_t_pos']}"
        r[f"B3mB4{key}_neg_tpos_1e5"] = pr["n_negative_t_pos_1e5"]
        r[f"B3mB4{key}_neg_by_t_1e5"] = ";".join(
            f"{t}:{v}" for t, v in pr["n_negative_by_t_1e5"].items() if v
        )
        r[f"B3mB4{key}_min_tpos"] = pr["min_t_pos"]
        r[f"B3mB4{key}_neg_last"] = pr["n_negative_last_t"]
    r["omega_max"] = co["omega_max"]
    r["wall_med_s"] = s["cost"]["wall_per_ensemble_median"]
    ja = s.get("joint_assumed_radius_sampled")
    if ja is not None:
        r["assumed_R_joint_sampled_cov_exceed_unres"] = (
            f"{ja['covered_all']}/{ja['radius_misspecification_exceedance']}/{ja['unresolved']}"
        )
        r["assumed_R_max_exceedance_cells"] = max(
            max(v["any_violation"] for v in meth[n].values()) for n in ar
        )
    if "init_chi2" in s:
        c = s["init_chi2"]
        r["chi2"] = c["chi2"]
        r["R_true"] = c["R"]
        if "at_true_radius" in s:
            t = s["at_true_radius"]
            jt = t["joint_guaranteed_sampled"]
            r["true_R_joint_sampled"] = (
                f"{jt['covered_all']}/{jt['any_violation']}/{jt['unresolved']}"
            )
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs", nargs="+", required=True, help="analysis dir names under r3b_suite/analysis"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = pathlib.Path(os.environ["MCMC_PAPER_OUT"]) / "r3b_suite" / "analysis"
    rows = []
    for d in a.dirs:
        for f in sorted((root / d).glob("*.summary.json")):
            rows.append(row(d, json.loads(f.read_text())))
    if not rows:
        raise FileNotFoundError("no summaries")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "overview.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    show = [
        "arm",
        "config",
        "n",
        "status",
        "T3",
        "joint_sampled",
        "joint_known",
        "guar_max_violation",
        "plugin_max_violation",
        "plugin_unresolved",
        "false_stops",
        "false_stops_other",
        "fallback_events",
        "direct_x010",
        "direct_x005",
        "B3_x010",
        "B4same_x010",
        "B4own_x010",
        "NO4_x010",
        "B4ref_own_x010",
        "B4ref_own_x005",
        "D_last",
        "U_last",
        "B3_B_last",
        "B4own_B_last",
        "B3mB4own_last",
        "B3mB4own_neg_tpos",
        "B3mB4own_neg_tpos_1e5",
        "B3mB4own_neg_by_t_1e5",
        "B3mB4own_min_tpos",
        "B3mB4same_neg_tpos",
        "B3mB4same_neg_tpos_1e5",
        "B3mB4same_neg_by_t_1e5",
        "B3mB4same_min_tpos",
        "omega_max",
        "wall_med_s",
    ]
    show += [
        k
        for k in (
            "applicable_guarantees",
            "assumed_R_joint_sampled_cov_exceed_unres",
            "assumed_R_max_exceedance_cells",
            "chi2",
            "R_true",
            "true_R_joint_sampled",
        )
        if k in keys
    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 1e3 else f"{v:.4f}"
        return "" if v is None else str(v)

    lines = [
        "# Synthetic sensitivity checks\n",
        "Primary cell: J=32768, N=2048, m=8, sampled reference; bounds include numerical corrections.\n\n"
        "joint reports covered/violated/unresolved for applicable guarantees over all J/m cells. "
        "For conventional starts at an invalid assumed radius, only the direct bound qualifies. "
        "assumed_R_* reports coverage/radius-misspecification exceedances/unresolved without a "
        "guarantee. true_R_joint uses the true radius for dispersed Student-t starts. "
        "false_stops covers applicable guarantees; false_stops_other covers the rest.\n\n"
        "x010/x005 gives first crossings of 0.10/0.05 (count @ median t). B3mB4* is the paired "
        "mean at the last checkpoint. neg_tpos counts differences below -1e-12 over ensembles, "
        "t>0, J, m and reference; _1e5 uses -1e-5, and by_t gives t:count. "
        "(inv) marks one plug-in curve repeated across ensembles. Plug-ins have no guarantee.\n",
        "| " + " | ".join(show) + " |",
        "|" + "---|" * len(show),
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k in show) + " |")
    (out / "overview.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/overview.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
