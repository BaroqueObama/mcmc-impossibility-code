"""Compare lag-2t and branched covariance gaps on saved logistic pilots.

For reversible RWM/MALA, Cov(P^t f) equals the stationary lag-2t covariance.
Both estimates use half-split errors ||S_A-S_B||_op/2. This calibration check
sets fresh-sampling budgets; it does not evaluate learned angles.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib

import numpy as np
from mcmc.paperexp.runio import resolve_output_path

REPO = pathlib.Path(__file__).resolve().parents[2]
LAGS = (4, 16, 64, 256)  # t = lag / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument(
        "--selection",
        default=None,
        help="selection file name (default selection_logistic_<tag>.json)",
    )
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    root = pathlib.Path(os.environ["MCMC_PAPER_OUT"])
    cal = root / "b_logistic" / "pilot" / f"calibrate_{args.tag}"
    out_dir = REPO / "paper/figures/data/r2_real_alignment"
    sel_name = args.selection or f"selection_logistic_{args.tag}.json"
    out_tag = args.out_tag or args.tag
    sel = json.loads((out_dir / sel_name).read_text())
    rows = []
    for unit, info in sel["units"].items():
        if not info["selected_m"]:
            continue
        cf = cal / f"calib_{unit}.npz"
        sf = resolve_output_path(json.loads((cal / f"calib_{unit}.json").read_text())["meta"]["run_v1"])
        sf = sf / "sigma.npz"
        z, sig = np.load(cf), np.load(sf)
        grid = [int(t) for t in sig["grid"]]
        for lag in LAGS:
            t = lag // 2
            if t not in grid:
                continue
            i = grid.index(t)
            Cf, CA, CB = (z[f"Clag__{s}__lag{lag}"] for s in ("full", "A", "B"))
            lam_c = np.linalg.eigvalsh(Cf)[::-1]
            eps_c = float(np.linalg.norm(CA - CB, 2) / 2)
            op = 0.5 * (sig["O"][i] + sig["O"][i].T)
            lam_o = np.linalg.eigvalsh(op)[::-1]
            eps_o = float(np.linalg.norm(sig["OA"][i] - sig["OB"][i], 2) / 2)
            diff = float(np.linalg.norm(Cf - op, 2))
            for m in info["selected_m"]:
                rows.append(
                    {
                        "unit": unit,
                        "m": m,
                        "t": t,
                        "pilot_lag": lag,
                        "lambda_m_lag2t": float(lam_c[m - 1]),
                        "gap_lag2t": float(lam_c[m - 1] - lam_c[m]),
                        "eps_lag2t": eps_c,
                        "lambda_m_branched": float(lam_o[m - 1]),
                        "gap_branched": float(lam_o[m - 1] - lam_o[m]),
                        "eps_branched": eps_o,
                        "diff_op": diff,
                        "diff_over_combined_eps": diff / float(np.hypot(eps_c, eps_o)),
                    }
                )
    dest = out_dir / f"lag2t_check_logistic_{out_tag}.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print(dest, len(rows))


if __name__ == "__main__":
    main()
