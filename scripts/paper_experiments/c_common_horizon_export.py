"""Pool completed common-horizon runs and write the Figure 2C data.

Runs must share a setup, contain disjoint ensemble IDs, and cover all
18 checkpoints. Writes curves, coverage, per-dataset summaries and
per-ensemble arrays under paper/figures/data/c_common_horizon/.

Bounds use the supplied initialization radius R=0.8. Coverage is simultaneous
across checkpoints within each dataset; median curves summarize repeated bounds.
The sampled target reference uses 65,536 exact draws unless ref="known".
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mcmc.paperexp.runio import resolve_output_path
from mcmc.experiments import exp_c_common_horizon as chz  # noqa: E402
from mcmc.experiments import exp_r3_concrete as r3  # noqa: E402

STUDY = pathlib.Path(os.environ.get("MCMC_PAPER_OUT", ROOT / "output/paper_experiments")) / "r3_common_horizon"
# None exports all ensembles; otherwise take the first N_MAX by seed.
N_MAX: int | None = None
OUT = ROOT / "paper" / "figures" / "data" / "c_common_horizon"
DATASETS = ("concrete", "energy", "airfoil", "wine_red", "kin8nm")
ROLES = {
    "B3a": "main", "B4a_own": "main", "B4b_own_N2048": "main",
    "B4a_same": "supporting", "B3b_N2048": "supporting", "B4b_same_N2048": "supporting",
    "direct": "appendix",
}  # fmt: skip
QS = (0.0, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 1.0)
QN = ("min", "q025", "q05", "q25", "median", "q75", "q95", "q975", "max")
PER_ENS_KEYS = ("proj", "proj_true", "direct", "repl_event", "tgt_event", "U3b_unit", "NO3b_unit", "U4b_same_unit",
                "U4b_own_unit", "NO4b_unit", "op_rank", "op_npos", "op_delta_trunc", "op_orth_err", "op_top",
                "op_min_eig", "op_head_slack_max", "span_orth_err_max", "kernel_evals_cum", "wall_cum", "wall",
                "peak_gpu_bytes", "peak_gpu_bytes_ensemble", "peak_rss_kb", "transitions_simulated",
                "sweeps_represented", "timing", "direct_err2")  # fmt: skip


def _orth_module():
    spec = importlib.util.spec_from_file_location(
        "r3_orthonormality_check", ROOT / "scripts" / "analysis" / "r3_orthonormality_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runs_for(phase: str, ds: str, tag: str, expect_commit: str | None) -> list[pathlib.Path]:
    """Find completed runs with a common problem fingerprint."""
    runs = sorted(p for p in (STUDY / phase).glob(f"{ds}_{tag}*") if p.is_dir())
    if N_MAX is not None:
        if phase != "main":
            raise ValueError("the ensemble limit applies to main runs only")
        runs = [p for p in runs if int(p.name.rsplit("_e", 1)[1]) < N_MAX]
    if not runs:
        raise FileNotFoundError(f"no {phase} runs for {ds}")
    fingerprints = set()
    for p in runs:
        st = json.loads((p / "RUN-STATUS.json").read_text())
        if st["status"] != "COMPLETED":
            raise RuntimeError(f"{p} is {st['status']}")
        prov = json.loads((p / "PROVENANCE.json").read_text())
        if expect_commit and prov.get("commit") != expect_commit:
            raise RuntimeError(f"{p}: expected commit {expect_commit}, got {prov.get('commit')}")
        fp = json.loads((p / "setup.json").read_text())["setup_cache"]["fingerprint"]
        if not fp:
            raise RuntimeError(f"{p}: missing problem fingerprint")
        fingerprints.add(fp)
        res_path = p / "RESUMES.jsonl"
        for line in res_path.read_text().splitlines() if res_path.exists() else []:
            rec = json.loads(line)
            if rec["problem_fingerprint"] != fp:
                raise RuntimeError(f"{p}: resume record has a different problem fingerprint")
    if len(fingerprints) != 1:
        raise RuntimeError(f"{ds}: runs have different problem fingerprints")
    return runs


def _q(x: np.ndarray) -> dict:
    return dict(zip(QN, np.quantile(x, QS, axis=0)))


def export_dataset(ds: str, phase: str, tag: str, out: pathlib.Path, expect_commit: str | None, orth) -> dict:
    runs = runs_for(phase, ds, tag, expect_commit)
    setup, data = r3.load_runs(runs)
    meta = json.loads((runs[0] / "setup.json").read_text())
    for p in runs[1:]:
        m2 = json.loads((p / "setup.json").read_text())
        for key in ("analytic_horizon", "transitions", "source", "ell", "landmark_closed_form", "seed_namespace"):
            if m2[key] != meta[key]:
                raise ValueError(f"{ds}: setup.json {key} differs between runs")
    ck = setup["checkpoints"].tolist()
    if ck != chz.GRID:
        raise ValueError(f"{ds}: checkpoints {ck} != common grid")
    E = int(data["reps"].shape[0])
    T = len(ck)
    if data["proj"].shape[2] != T or data["wall_cum"].shape[1] != T:
        raise ValueError(f"{ds}: arrays do not cover all {T} checkpoints")
    lo, hi = meta["rep_range"]
    if N_MAX is not None:
        hi = lo + N_MAX
    if sorted(data["reps"].tolist()) != list(range(lo, hi)):
        raise ValueError(f"{ds}: ensemble ids are not exactly [{lo}, {hi})")
    if setup["r_init"].tolist() != [0.8] or not np.array_equal(setup["shifts"][0], np.array(meta["source"]["shift"])):
        raise ValueError(f"{ds}: setting is not the saved slow_R0.8 shift")
    for k in PER_ENS_KEYS:
        if k in data and not np.isfinite(data[k]).all():
            raise FloatingPointError(f"{ds}: non-finite {k}")
    acc = meta["transitions"]
    if (data["transitions_simulated"] != acc["ensemble_transitions_simulated"]).any() or (
        data["sweeps_represented"] != acc["ensemble_sweeps_represented"]
    ).any():
        raise ValueError(f"{ds}: an ensemble has unexpected transition counts")
    cache = resolve_output_path(meta["setup_cache"]["path"])
    problem = pickle.loads(cache.read_bytes())["problem"]
    arrays = r3.problem_arrays(problem)
    if set(arrays) != set(setup) or any(not np.array_equal(arrays[k], setup[k]) for k in setup):
        raise ValueError(f"{ds}: setup cache differs from the runs' setup.npz")
    design = {"N_pi": int(problem.N_pi), "M": int(problem.basis.Z.shape[0])}
    summ = r3.summarize(setup, data, design=design)
    asm = r3.assemble_bounds(setup, data)
    B, H, U = asm["B"], asm["H"], asm["U"]
    truth = setup["truth"][0]
    horizon = meta["analytic_horizon"]
    if [h["t"] for h in horizon] != ck or any(h["mmd_fp64"] != float(truth[i]) for i, h in enumerate(horizon)):
        raise ValueError(f"{ds}: analytic horizon in setup.json disagrees with setup.npz truth")
    # curves
    rows = []
    for name in sorted(B):
        role = ROLES.get(name, "other")
        for ri, ref in enumerate(r3.REF_NAMES):
            Bv = B[name][:, 0, :, 0, 0, ri]  # (E, T)
            qb = _q(Bv)
            qh = _q(H[name][:, 0, :, 0, 0, ri]) if name in H else None
            qu = _q(U[name][:, 0, :, 0, 0, ri]) if name in U else None
            for ti, t in enumerate(ck):
                h = horizon[ti]
                rec = {"dataset": ds, "method": name, "role": role, "ref": ref, "t": t, "n_ens": E}
                rec.update({f"B_{k}": float(v[ti]) for k, v in qb.items()})
                rec["B_mean"] = float(Bv[:, ti].mean())
                for lab, qq in (("H", qh), ("U", qu)):
                    for k in ("q05", "median", "q95"):
                        rec[f"{lab}_{k}"] = float(qq[k][ti]) if qq is not None else ""
                rec.update(
                    {
                        "truth_fp64": h["mmd_fp64"],
                        "log10_truth": h["log10_mmd"],
                        "truth_from_log": "" if h["mmd_from_log"] is None else h["mmd_from_log"],
                        "truth_status": h["status"],
                        "n_violations_t": int((Bv[:, ti] < truth[ti]).sum()),
                    }
                )
                rows.append(rec)
    # coverage (uncorrected + Gram-corrected variants)
    lm = orth.landmark_diagnostics(problem)
    corr = {}
    X0 = orth.crossings(B)
    for var, w in (("A", lm["omega_A_recorded"]), ("B", lm["omega_B_rederived"]), ("C", lm["omega_C_observed"])):
        s2, d2, om = orth.corrected(setup, data, lm, w)
        cmp_ = orth.compare(setup, B, X0, s2, d2)
        B1 = r3.assemble_bounds(s2, d2)["B"]
        corr[var] = {"omegas": om, "summary": orth.summarize_cmp(cmp_), "B": B1}
    cov_rows = []
    for name in sorted(B):
        for ri, ref in enumerate(r3.REF_NAMES):
            key = f"0|{int(setup['J_grid'][0])}|{int(setup['m_grid'][0])}|{ref}"
            rec_s = summ["methods"][name][key]
            Bv = B[name][:, 0, :, 0, 0, ri]
            k_cov = int((Bv >= truth[None]).all(1).sum())
            if k_cov != rec_s["covered_all_t"]:
                raise AssertionError(f"{ds}/{name}: coverage recount mismatch")
            row = {
                "dataset": ds, "method": name, "role": ROLES.get(name, "other"), "ref": ref, "n_ens": E,
                "covered_all_t": k_cov, "any_violation": E - k_cov,
                "cp_lo": rec_s["cp"][0], "cp_hi": rec_s["cp"][1], "min_margin": rec_s["min_margin"],
                "violations_by_t": ";".join(str(int((Bv[:, i] < truth[i]).sum())) for i in range(T)),
            }  # fmt: skip
            for var in ("A", "B", "C"):
                Bc = corr[var]["B"][name][:, 0, :, 0, 0, ri]
                kc = int((Bc >= truth[None]).all(1).sum())
                row[f"covered_all_t_gram{var}"] = kc
                row[f"max_abs_change_gram{var}"] = float((Bc - Bv).max())
            cov_rows.append(row)
    # Per-ensemble values retain the numerical allowances and timing measurements.
    tk = [str(x) for x in data["timing_keys"]]
    pe = out / "per_ensemble"
    pe.mkdir(parents=True, exist_ok=True)
    arrs = {f"setup__{k}": v for k, v in setup.items()}
    arrs.update({k: data[k] for k in PER_ENS_KEYS if k in data})
    arrs["reps"] = data["reps"]
    arrs["timing_keys"] = np.array(tk)
    arrs["truth_log10"] = np.array([h["log10_mmd"] for h in horizon])
    arrs["truth_status"] = np.array([h["status"] for h in horizon])
    np.savez_compressed(pe / f"{ds}.npz", **arrs)
    (out / f"summary_{ds}.json").write_text(json.dumps({**summ, "setup": {k: v.tolist() for k, v in setup.items()}, "meta": meta}, indent=1, default=str))
    return {"curves": rows, "coverage": cov_rows,
            "n_ensembles": E, "gpu_hours": float(data["wall"].sum() / 3600.0)}


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    for r in rows:
        if list(r) != keys:
            raise ValueError(f"{path.name}: inconsistent columns")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    global STUDY
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main", choices=("smoke", "pilot", "main"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--tag", default=None, help="run-id tag after '<dataset>_' (default: the phase)")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--study-root", default=str(STUDY), help="directory containing the raw runs")
    ap.add_argument("--n-max", type=int, default=None, help="export the first N ensembles per dataset by id")
    args = ap.parse_args()
    STUDY = pathlib.Path(args.study_root)
    global N_MAX
    N_MAX = args.n_max
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    orth = _orth_module()
    tag = args.tag or args.phase
    res = {ds: export_dataset(ds, args.phase, tag, out, args.expect_commit, orth) for ds in args.datasets}
    _write_csv(out / "curves.csv", [r for v in res.values() for r in v["curves"]])
    _write_csv(out / "coverage.csv", [r for v in res.values() for r in v["coverage"]])
    for ds, v in res.items():
        main_cov = {r["method"]: r["covered_all_t"] for r in v["coverage"] if r["ref"] == "sampled" and r["role"] == "main"}
        print(ds, "E", v["n_ensembles"], "main coverage", main_cov, "gpu-h", round(v["gpu_hours"], 3))
    print("wrote", out)


if __name__ == "__main__":
    main()
