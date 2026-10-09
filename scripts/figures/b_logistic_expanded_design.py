"""Choose block counts and time windows from the logistic calibration results.

For a calibration gap g(t), need(t) = 8192 * (2 e_saved(t) / g(t))^2
extrapolates the blocks needed for gap > 2 e_t under 1/sqrt(N) error scaling.
Use lag-2t gaps at t=2,8,32,128 and branched gaps elsewhere. Start from the
projected window at 4*8192 blocks; if empty, use the contiguous lag-2t window
with gap > 3 eps, or {1,2} if that is empty too.

Take 1.5 times the largest need(t) in this window, round to four equal
batches in multiples of 16384, and extend the window while that budget
suffices. Simulate one additional checkpoint. Costs are capped at 4 GPU-hours
per dataset. Above 100,000 blocks, use a new NUTS pool with at least
2*N_batch states in each of its four disjoint quarters.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
D = REPO / "paper/figures/data/r2_real_alignment"
SAFETY = 1.5
SUB = 16384
COST = {"mala": 3.0e-7, "rwm": 2.0e-7}  # s per chain-step; measured 1.4e-7 to 2.5e-7 asymptotically
CAP_H = 4.0
NEW_POOL_THRESHOLD = 100_000
DRAWS_PER_RUN = 25_000


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    sel_p = D / "selection_logistic.json"
    win_p = D / "window_projection_logistic_v2.csv"
    l2_p = D / "lag2t_check_logistic_v2.csv"
    sel = json.loads(sel_p.read_text())
    win = collections.defaultdict(dict)
    for r in csv.DictReader(open(win_p)):
        win[(r["unit"], int(r["m"]))][int(r["t"])] = r
    l2 = {(r["unit"], int(r["m"]), int(r["t"])): r for r in csv.DictReader(open(l2_p))}
    pairs = {}
    for u, v in sel["units"].items():
        for m in v["selected_m"]:
            w = win[(u, m)]
            grid = sorted(t for t in w if t >= 1)

            def gap(t):
                k = (u, m, t)
                return (
                    (float(l2[k]["gap_lag2t"]), "lag2t")
                    if k in l2
                    else (float(w[t]["gap"]), "branched")
                )

            need = {}
            for t in grid:
                g, src = gap(t)
                need[t] = {
                    "blocks": 8192 * (2 * float(w[t]["e_saved"]) / g) ** 2 if g > 0 else math.inf,
                    "gap": g,
                    "gap_source": src,
                    "e_saved": float(w[t]["e_saved"]),
                }
            W0 = list(v["reference_window"][str(m)]["t_dk_applicable_fresh"])
            basis = "Projected Davis-Kahan window at 4 x 8192 blocks"
            if not W0:
                treal = None
                for t in (2, 8, 32, 128):
                    k = (u, m, t)
                    if k in l2 and float(l2[k]["gap_lag2t"]) > 3 * float(l2[k]["eps_lag2t"]):
                        treal = t
                    else:
                        break
                if treal is None:
                    W0, basis = [1, 2], "fallback {1, 2}: lag-2t gap <= 3 eps at t = 2"
                else:
                    W0, basis = (
                        [t for t in grid if t <= treal],
                        f"lag-2t evidence window t <= {treal}",
                    )
            pairs[(u, m)] = {
                "W0": W0,
                "basis": basis,
                "required_blocks": SAFETY * max(need[t]["blocks"] for t in W0),
                "need": need,
                "grid": grid,
            }
    units = collections.OrderedDict()
    for u, m in pairs:
        units.setdefault(u, []).append(m)
    plan, per_ds = {}, collections.defaultdict(float)
    for u, ms in units.items():
        req = max(pairs[(u, m)]["required_blocks"] for m in ms)
        nb = max(8192, SUB * math.ceil(req / 4 / SUB))
        ext = {}
        for m in ms:
            p = pairs[(u, m)]
            last = max(p["W0"])
            for t in p["grid"]:
                if t <= last:
                    continue
                if p["need"][t]["blocks"] <= 4 * nb:
                    last = t
                else:
                    break
            ext[m] = last
        later = [t for t in pairs[(u, ms[0])]["grid"] if t > max(ext.values())]
        t_max = later[0] if later else max(ext.values())
        cost = COST[u.split("__")[1]] * 4 * 4 * nb * t_max
        ds = u.split("__")[0]
        per_ds[ds] += cost
        plan[u] = {
            "m": ms,
            "N_batch": nb,
            "N_total": 4 * nb,
            "t_max": t_max,
            "new_pool": 4 * nb > NEW_POOL_THRESHOLD,
            "pool_states_per_subset_needed": 2 * nb,
            "projected_gpu_s": cost,
            "pairs": {
                str(m): {
                    "W0": pairs[(u, m)]["W0"],
                    "basis": pairs[(u, m)]["basis"],
                    "required_blocks": pairs[(u, m)]["required_blocks"],
                    "extended_to": ext[m],
                    "need_blocks": {str(t): x["blocks"] for t, x in pairs[(u, m)]["need"].items()},
                    "gap_source": {
                        str(t): x["gap_source"] for t, x in pairs[(u, m)]["need"].items()
                    },
                }
                for m in ms
            },
        }
    over = {k: v / 3600 for k, v in per_ds.items() if v / 3600 > CAP_H}
    if over:
        raise SystemExit(f"cap exceeded (GPU-h): {over}")
    pools = {}
    for model in sorted({u.split("__")[0] for u, p in plan.items() if p["new_pool"]}):
        need = max(
            p["pool_states_per_subset_needed"]
            for u, p in plan.items()
            if u.split("__")[0] == model and p["new_pool"]
        )
        q = math.ceil(need / DRAWS_PER_RUN)
        pools[model] = {
            "runs_per_subset": q,
            "n_runs": 4 * q,
            "states_per_subset": q * DRAWS_PER_RUN,
        }
    out = {
        "status": "Empirical NUTS-reference design, fixed before expanded sampling",
        "rule": __doc__,
        "safety": SAFETY,
        "sub_batch_blocks": SUB,
        "cost_s_per_chain_step": COST,
        "cap_gpu_h_per_dataset": CAP_H,
        "new_pool_threshold_blocks": NEW_POOL_THRESHOLD,
        "inputs_sha256": {p.name: sha(p) for p in (sel_p, win_p, l2_p)},
        "units": plan,
        "new_pools": pools,
        "projected_gpu_h_per_dataset": {k: v / 3600 for k, v in per_ds.items()},
        "projected_gpu_h_total": sum(per_ds.values()) / 3600,
    }
    dest = D / "expanded_design_logistic_x1.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(dest)
    for u, p in plan.items():
        print(
            u,
            p["m"],
            p["N_batch"],
            p["t_max"],
            p["new_pool"],
            round(p["projected_gpu_s"]),
            {m: (x["W0"][-1], x["extended_to"]) for m, x in p["pairs"].items()},
        )
    print(pools, out["projected_gpu_h_per_dataset"], out["projected_gpu_h_total"])


if __name__ == "__main__":
    main()
