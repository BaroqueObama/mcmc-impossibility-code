"""Orthonormality corrections for the saved regression bounds.

Saved monitored norms are ||c||_2, c = V^* delta_hat: equal to ||P_V delta_hat|| only for an exactly orthonormal span.
With span Gram G_V = V^* V, ||G_V - I||_2 <= omega_V < 1:  ||P_V delta_hat||^2 = c^T G_V^{-1} c <= ||c||^2/(1 - omega_V),
so every monitored norm is inflated by 1/sqrt(1 - omega_V). Compressed norms that assumed orthonormal bases get additive
corrections to the squared U term:

  landmark (ideal) basis W, Gram G = W^T K W, ||G - I||_2 <= w; S_t = W^T M_t W; V orthonormal coordinates:
    lambda_max(true compression onto Z minus span(W V)) <= computed + 3 w/(1-w) ||S_t||      (U3a, U4a same/own)
    lambda_1(true P_Z Sigma_t P_Z)                       <= computed +   w/(1-w) ||S_t||      (NO3a, NO4a)
    omitted trace tau_true                               <= tau      +   w/(1-w) ||S_t||_*    (all ideal U, NO)
  estimated operator O_+ = U Theta U^* (U: n_pos eigenvectors, ||U^*U - I|| <= w_U), span V (||V^*V - I|| <= w_V):
    lambda_max(true compression) <= computed + theta_1 (w_U + (1+w_U)(1+w_V) w_V/(1-w_V))   (U3b, U4b same/own)
    NO3b/NO4b use lambda_1(O_+) only (an eigenvalue of R^T C R; no span, no eigenvector orthonormality): unchanged.
  direct bound: no span, unchanged.

omega per span type, (A) from RECORDED max-entry Gram errors e and the number k of vectors (||F||_2 <= k max|F_ij|):
  landmark: w = r e_W + w_eig (r = retained landmark rank; w_eig = ||V^T V - I||_2 of the S_t eigenvector coordinates,
            recomputed with an a-priori product-rounding bound); this also bounds every landmark H span;
  operator spans V3b/V4b: w_V = m e_op (e_op = op_orth_err: max-entry Gram error of the leading 64 eigenvectors,
            recorded per ensemble, checkpoint and N; V3b uses the t = 0 value at every t);
  operator eigenvectors in compressed norms: w_U = n_pos e_op (EXTRAPOLATION: only the leading 64 were checked).
(B) landmark only, re-derived here (no recorded value used): Gram recomputed in 80-bit long double with an a-priori
  product-rounding bound (rigorous for the stored fp64 kernel matrix) plus a model-based bound on the kernel-entry
  error ||W||_2^2 ||Delta K||_2 (|Delta d^2| <= (d+10) u (|u_i|+|u_j|)^2, exp within 4 ulp: an error MODEL, not
  certified). Operator spans cannot be re-derived post hoc (block endpoints are not saved).
(C) landmark only, OBSERVATIONAL (not a bound): the measured spectral defect (long-double Gram of the stored (W, K);
  agrees with fp64) + the observed kernel-formula sensitivity ||W^T (K_direct - K) W||_2 + w_eig; operator as in (A).
(D) breakdown: largest uniform omega (all spans/bases) with no first-crossing change at eps 0.05/0.10.

    python scripts/analysis/r3_orthonormality_check.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mcmc.experiments import exp_r3_concrete as r3  # noqa: E402

OUT = pathlib.Path(os.environ.get("MCMC_PAPER_OUT", ROOT / "output/paper_experiments"))
CONCRETE_MAIN = ("main01", "main02_p460", "main02_p550", "main02_p700", "main02_p850a", "main02_p950")
GROUPS = {
    "concrete_main": (
        OUT / "r3_concrete/setup_cache/main_ell1.0.pkl",
        [OUT / "r3_concrete/main" / n for n in CONCRETE_MAIN],
    ),
    "concrete_stress_ell0.5": (
        OUT / "r3_concrete/setup_cache/stress_ell0.5.pkl",
        [OUT / "r3_concrete/stress/stress_ell0.5"],
    ),
    "concrete_stress_ell2.0": (
        OUT / "r3_concrete/setup_cache/stress_ell2.0.pkl",
        [OUT / "r3_concrete/stress/stress_ell2.0"],
    ),
}
for _ds, _cache in (
    ("kin8nm", "kin8nm.pkl"),
    ("airfoil", "airfoil_n2048.pkl"),
    ("wine_red", "wine_red_n2048.pkl"),
    ("energy", "energy_n2048.pkl"),
):
    GROUPS[_ds] = (
        OUT / "r3_regression_cache" / _cache,
        [OUT / f"r3_{_ds}/main/main_p{a:03d}" for a in range(0, 500, 100)],
    )
F64 = torch.float64
U64 = 2.0**-53
EPS = (0.05, 0.10)


def _spec(F: np.ndarray) -> float:
    return float(np.abs(np.linalg.eigvalsh(0.5 * (F + F.T))).max())


def _gamma(n: int, u: float) -> float:
    return n * u / (1.0 - n * u)


def landmark_diagnostics(problem) -> dict:
    """Recorded, recomputed and re-derived Gram defects of the landmark basis and of the S_t eigenvector coordinates."""
    b, k = problem.basis, problem.kernel
    W, M, r = b.W, b.W.shape[0], b.W.shape[1]
    Zt = torch.as_tensor(b.Z, dtype=F64)
    K = k(Zt, Zt).numpy()
    K = 0.5 * (K + K.T)  # exactly as in landmark_basis
    F64g = W.T @ K @ W - np.eye(r)
    # 80-bit long-double Gram of the stored (W, K) with an a-priori bound on its own rounding
    u_ld = float(np.finfo(np.longdouble).eps) / 2.0
    Wl, Kl = W.astype(np.longdouble), K.astype(np.longdouble)
    F_ld = Wl.T @ (Kl @ Wl) - np.eye(r, dtype=np.longdouble)
    absW = np.abs(W)
    bound_mat = absW.T @ (np.abs(K) @ absW)  # fp64, >= exact up to rel 2 gamma_M(u64): absorbed below
    prod_bound = _gamma(2 * M, u_ld) * float(np.linalg.norm(bound_mat)) * (1.0 + 4 * _gamma(M, U64))
    F_ld64 = F_ld.astype(np.float64)
    conv = U64 * float(np.linalg.norm(F_ld64)) * 2.0 + 10 * r * U64 * float(np.linalg.norm(F_ld64))
    omega_ld = _spec(F_ld64) + conv + prod_bound
    # kernel-entry error MODEL (true kernel vs computed fp64 K): see module docstring
    inv = 1.0 / (np.asarray(k.scales) * k.ell)
    c = b.Z.mean(0)
    un = np.sqrt((((b.Z - c) * inv) ** 2).sum(1))
    d = b.Z.shape[1]
    dK = K * (0.5 * (d + 10) * U64 * (un[:, None] + un[None, :]) ** 2 + 4 * U64) * 1.01
    normW2 = float(np.linalg.norm(W, 2)) ** 2 * (1.0 + 1e-10)
    omega_K = normW2 * float(np.linalg.eigvalsh(dK)[-1]) * (1.0 + 1e-10)
    # observed sensitivity: same Gram with the kernel from direct coordinate differences
    D2 = (((b.Z[:, None, :] - b.Z[None, :, :]) * inv) ** 2).sum(-1)
    K2 = np.exp(-0.5 * D2)
    obs_kernel_formula = _spec(W.T @ (K2 - K) @ W)
    # S_t eigenvector coordinates (V3 and V4 at every t): ||V^T V - I|| plus product rounding
    w_eig = 0.0
    for t in problem.checkpoints:
        V = problem.ideal_ops[t].top_basis(problem.n_dirs)
        e = _spec(V.T @ V - np.eye(V.shape[1]))
        e += _gamma(r, U64) * float(np.linalg.norm(np.abs(V).T @ np.abs(V)))
        w_eig = max(w_eig, e)
    S_norm = [float(np.abs(problem.ideal_ops[t].eigvals).max()) for t in problem.checkpoints]
    S_nuc = [float(np.abs(problem.ideal_ops[t].eigvals).sum()) for t in problem.checkpoints]
    return {
        "M": int(M),
        "rank_r": int(r),
        "d": int(d),
        "recorded_max_entry": float(b.orthonormality_error),
        "recomputed_fp64_max_entry": float(np.abs(F64g).max()),
        "recomputed_fp64_spectral": _spec(F64g),
        "longdouble_spectral": _spec(F_ld64),
        "longdouble_product_rounding_bound": prod_bound,
        "kernel_entry_model_bound": omega_K,
        "kernel_formula_observed_spectral": obs_kernel_formula,
        "W_norm2_sq": normW2,
        "W_abs_colsum_max": float(absW.sum(0).max()),
        "eigvec_coord_defect": w_eig,
        "omega_A_recorded": r * float(b.orthonormality_error) + w_eig,
        "omega_B_rederived": omega_ld + omega_K + w_eig,
        "omega_C_observed": _spec(F_ld64) + obs_kernel_formula + w_eig,
        "S_norm": S_norm,
        "S_nuclear": S_nuc,
    }


def corrected(setup: dict, data: dict, lm: dict, omega_W: float, op_uniform: float | None = None):
    """Correct span-dependent terms; op_uniform overrides the operator Gram defects."""
    s2 = {k: np.array(v, copy=True) for k, v in setup.items()}
    d2 = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    ms = np.asarray(setup["m_grid"], dtype=np.float64)
    nN = len(setup["N_grid"])
    w = omega_W
    if not 0.0 <= w < 1.0:
        raise ValueError(f"landmark omega {w} outside [0, 1)")
    Sn, Snuc = np.asarray(lm["S_norm"]), np.asarray(lm["S_nuclear"])
    q = w / (1.0 - w)
    add_cmp = 3.0 * q * Sn + q * Snuc  # (T,) compressed (U3a / U4a)
    add_no = q * Sn + q * Snuc  # (T,) no-observation (NO3a / NO4a)
    s2["U3a_unit"] = np.sqrt(setup["U3a_unit"] ** 2 + add_cmp[0])
    s2["U4a_same_unit"] = np.sqrt(setup["U4a_same_unit"] ** 2 + add_cmp[:, None])
    s2["U4a_own_unit"] = np.sqrt(setup["U4a_own_unit"] ** 2 + add_cmp[:, None])
    s2["NO3a_unit"] = np.sqrt(setup["NO3a_unit"] ** 2 + add_no[0])
    s2["NO4a_unit"] = np.sqrt(setup["NO4a_unit"] ** 2 + add_no)
    proj = d2["proj"]  # (E, S, T, J, F, m, ref)
    proj[:, :, :, :, 0:2] /= math.sqrt(1.0 - w)
    err = data["op_orth_err"]  # (E, T, nN)
    if op_uniform is None:
        wV = err[..., None] * ms  # (E, T, nN, m)
        wU = err * data["op_npos"]  # (E, T, nN)
    else:
        wV = np.full(err.shape + (len(ms),), op_uniform)
        wU = np.full(err.shape, op_uniform)
    if wV.max() >= 1.0 or wU.max() >= 1.0:
        raise ValueError("operator omega >= 1")
    th1 = data["op_top"][..., 0]  # (E, T, nN)

    def margin(wu, wv):  # broadcast (.., m)
        return wu + (1.0 + wu) * (1.0 + wv) * wv / (1.0 - wv)

    for n in range(nN):
        f3, f4 = 2 + n, 2 + nN + n
        proj[:, :, :, :, f3] /= np.sqrt(1.0 - wV[:, 0, n, :])[:, None, None, None, :, None]
        proj[:, :, :, :, f4] /= np.sqrt(1.0 - wV[:, :, n, :])[:, None, :, None, :, None]
        wu = wU[:, :, n, None]  # (E, T, 1)
        add_same = th1[:, :, n, None] * margin(wu, wV[:, 0:1, n, :])  # span from t = 0
        add_own = th1[:, :, n, None] * margin(wu, wV[:, :, n, :])
        d2["U3b_unit"][:, n, :] = np.sqrt(data["U3b_unit"][:, n, :] ** 2 + add_same[:, 0, :])
        d2["U4b_same_unit"][:, :, n, :] = np.sqrt(data["U4b_same_unit"][:, :, n, :] ** 2 + add_same)
        d2["U4b_own_unit"][:, :, n, :] = np.sqrt(data["U4b_own_unit"][:, :, n, :] ** 2 + add_own)
    meta = {
        "omega_W": w,
        "omega_op_span_max": float(wV.max()),
        "omega_op_eigvec_max": float(wU.max()),
    }
    return s2, d2, meta


def crossings(B: dict) -> dict:
    return {
        (name, eps): r3.first_crossing(np.moveaxis(b, 2, -1), eps)
        for name, b in B.items()
        for eps in EPS
    }


def compare(setup: dict, B0: dict, X0: dict, s2: dict, d2: dict) -> dict:
    B1 = r3.assemble_bounds(s2, d2)["B"]
    truth = setup["truth"][None, :, :, None, None, None]
    res = {}
    for name in B0:
        b0, b1 = B0[name], B1[name]
        if np.any(b1 < b0):
            raise AssertionError(f"{name}: corrected bound decreased")
        diff = b1 - b0
        rec = {
            "max_abs_change": float(diff.max()),
            "max_rel_change": float((diff / b0).max()),
            "covered_cells_before": int((b0 >= truth).all(2).sum()),
            "covered_cells_after": int((b1 >= truth).all(2).sum()),
            "cells": int((b0 >= truth).all(2).size),
            "min_margin_before": float((b0 - truth).min()),
        }
        ck = setup["checkpoints"]
        for eps in EPS:
            f1 = r3.first_crossing(np.moveaxis(b1, 2, -1), eps)  # (E, S, J, m, ref)
            f0 = X0[(name, eps)]
            ch = f0 != f1
            rec[f"crossing_changed_{eps}"] = int(ch.sum())
            rec[f"crossing_lost_{eps}"] = int(((f0 >= 0) & (f1 < 0)).sum())
            rec[f"crossed_before_{eps}"] = int((f0 >= 0).sum())
            # Record the original crossing margin and the change in crossing summaries.
            cells, keys = [], {}
            for e, s, j, mi, ri in zip(*np.nonzero(ch)):
                cells.append(
                    {
                        "ens": int(e),
                        "key": f"{s}|{setup['J_grid'][j]}|{setup['m_grid'][mi]}|{r3.REF_NAMES[ri]}",
                        "t_before": int(ck[f0[e, s, j, mi, ri]]) if f0[e, s, j, mi, ri] >= 0 else None,
                        "t_after": int(ck[f1[e, s, j, mi, ri]]) if f1[e, s, j, mi, ri] >= 0 else None,
                        "eps_minus_B_before": float(eps - b0[e, s, f0[e, s, j, mi, ri], j, mi, ri]),
                    }
                )
                keys[cells[-1]["key"]] = (s, j, mi, ri)
            rec[f"changed_cells_{eps}"] = cells
            stats = {}
            for key, (s, j, mi, ri) in keys.items():
                st = {}
                for lab, f in (("before", f0), ("after", f1)):
                    fc = f[:, s, j, mi, ri]
                    st[lab] = {
                        "fraction": float((fc >= 0).mean()),
                        "t_median_crossed": float(np.median(ck[fc[fc >= 0]])) if (fc >= 0).any() else None,
                    }
                stats[key] = st
            rec[f"changed_keys_{eps}"] = stats
        res[name] = rec
    return res


def n_crossing_changes(setup, data, X0, lm, w) -> int:
    s2, d2, _ = corrected(setup, data, lm, w, op_uniform=w)
    B1 = r3.assemble_bounds(s2, d2)["B"]
    return sum(
        int((X0[(nm, eps)] != r3.first_crossing(np.moveaxis(B1[nm], 2, -1), eps)).sum())
        for nm in B1
        for eps in EPS
    )


def breakdown(setup, data, X0, lm) -> float:
    """Largest uniform omega (all spans/bases; bisection in log10 to 0.02 decades) with no first-crossing change."""
    lo, hi = -16.0, math.log10(0.5)
    if n_crossing_changes(setup, data, X0, lm, 10**hi) == 0:
        return 10**hi
    if n_crossing_changes(setup, data, X0, lm, 10**lo) > 0:
        return 0.0
    while hi - lo > 0.02:
        mid = 0.5 * (lo + hi)
        if n_crossing_changes(setup, data, X0, lm, 10**mid) > 0:
            hi = mid
        else:
            lo = mid
    return 10**lo


def summarize_cmp(cmp: dict) -> dict:
    v = list(cmp.values())
    worst_abs = max(cmp, key=lambda k: cmp[k]["max_abs_change"])
    worst_rel = max(cmp, key=lambda k: cmp[k]["max_rel_change"])
    return {
        "max_abs_change": cmp[worst_abs]["max_abs_change"],
        "max_abs_method": worst_abs,
        "max_rel_change": cmp[worst_rel]["max_rel_change"],
        "max_rel_method": worst_rel,
        "uncovered_before": sum(x["cells"] - x["covered_cells_before"] for x in v),
        "uncovered_after": sum(x["cells"] - x["covered_cells_after"] for x in v),
        **{f"crossing_changed_{e}": sum(x[f"crossing_changed_{e}"] for x in v) for e in EPS},
        **{f"crossed_before_{e}": sum(x[f"crossed_before_{e}"] for x in v) for e in EPS},
    }


def main() -> None:
    torch.set_num_threads(4)
    only = sys.argv[1:] or list(GROUPS)
    results = {}
    for group in only:
        cache, runs = GROUPS[group]
        t0 = time.perf_counter()
        problem = pickle.loads(cache.read_bytes())["problem"]
        setup, data = r3.load_runs(runs)
        arrays = r3.problem_arrays(problem)
        if set(arrays) != set(setup) or any(not np.array_equal(arrays[k], setup[k]) for k in setup):
            raise ValueError(f"{group}: setup cache differs from the runs' setup.npz")
        lm = landmark_diagnostics(problem)
        A0 = r3.assemble_bounds(setup, data)["B"]
        X0 = crossings(A0)
        sA, dA, mA = corrected(setup, data, lm, lm["omega_A_recorded"])
        sB, dB, mB = corrected(setup, data, lm, lm["omega_B_rederived"])
        sC, dC, mC = corrected(setup, data, lm, lm["omega_C_observed"])
        cmpA, cmpB = compare(setup, A0, X0, sA, dA), compare(setup, A0, X0, sB, dB)
        cmpC = compare(setup, A0, X0, sC, dC)
        w_star = breakdown(setup, data, X0, lm)
        results[group] = {
            "runs": [str(p) for p in runs],
            "n_ensembles": int(len(data["reps"])),
            "N_grid": [int(n) for n in setup["N_grid"]],
            "landmark": lm,
            "operator": {
                "op_orth_err_max": float(data["op_orth_err"].max()),
                "op_npos_max": int(data["op_npos"].max()),
                "span_orth_err_max": float(np.max(data["span_orth_err_max"])),
            },
            "A_recorded": {"omegas": mA, "summary": summarize_cmp(cmpA), "methods": cmpA},
            "B_landmark_rederived": {"omegas": mB, "summary": summarize_cmp(cmpB), "methods": cmpB},
            "C_landmark_observed": {"omegas": mC, "summary": summarize_cmp(cmpC), "methods": cmpC},
            "breakdown_uniform_omega": w_star,
            "seconds": time.perf_counter() - t0,
        }
        sa = results[group]["A_recorded"]["summary"]
        sb = results[group]["B_landmark_rederived"]["summary"]
        sc = results[group]["C_landmark_observed"]["summary"]
        print(
            f"{group}: E={len(data['reps'])} r={lm['rank_r']} omega_W A {lm['omega_A_recorded']:.2e} "
            f"B {lm['omega_B_rederived']:.2e} (ld {lm['longdouble_spectral']:.1e} "
            f"+ prod {lm['longdouble_product_rounding_bound']:.1e} + kernel {lm['kernel_entry_model_bound']:.1e}); "
            f"op span {mA['omega_op_span_max']:.1e} eigvec {mA['omega_op_eigvec_max']:.1e}; "
            f"A: abs {sa['max_abs_change']:.1e} ({sa['max_abs_method']}) rel {sa['max_rel_change']:.1e} "
            f"uncov {sa['uncovered_before']}->{sa['uncovered_after']} "
            f"cross {sa['crossing_changed_0.05']},{sa['crossing_changed_0.1']}; "
            f"B: abs {sb['max_abs_change']:.1e} rel {sb['max_rel_change']:.1e} "
            f"cross {sb['crossing_changed_0.05']},{sb['crossing_changed_0.1']}; "
            f"C: omega_W {lm['omega_C_observed']:.1e} abs {sc['max_abs_change']:.1e} "
            f"cross {sc['crossing_changed_0.05']},{sc['crossing_changed_0.1']}; "
            f"omega* {w_star:.2e}  [{results[group]['seconds']:.0f}s]",
            flush=True,
        )
    tag = "" if only == list(GROUPS) else "_partial"
    out = ROOT / "paper" / "figures" / "data" / "r3_regression" / f"orthonormality_check{tag}.json"
    out.write_text(json.dumps(results, indent=1))
    table = ROOT / "paper" / "tables" / f"r3_orthonormality_check{tag}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            ["group", "E", "landmark_rank", "recorded_landmark_max_entry", "omega_W_recorded",
             "omega_W_rederived", "omega_op_span_max", "omega_op_eigvec_max", "A_max_abs", "A_max_rel",
             "A_crossing_changes_0.05", "A_crossing_changes_0.10", "B_max_abs", "B_max_rel",
             "B_crossing_changes_0.05", "B_crossing_changes_0.10", "omega_W_observed", "C_max_abs",
             "C_crossing_changes_0.05", "C_crossing_changes_0.10", "uncovered_before", "uncovered_after",
             "breakdown_uniform_omega"]
        )  # fmt: skip
        for g, rr in results.items():
            sa, sb = rr["A_recorded"]["summary"], rr["B_landmark_rederived"]["summary"]
            sc = rr["C_landmark_observed"]["summary"]
            lm, mA = rr["landmark"], rr["A_recorded"]["omegas"]
            wr.writerow(
                [g, rr["n_ensembles"], lm["rank_r"], f"{lm['recorded_max_entry']:.3e}",
                 f"{lm['omega_A_recorded']:.3e}", f"{lm['omega_B_rederived']:.3e}",
                 f"{mA['omega_op_span_max']:.3e}", f"{mA['omega_op_eigvec_max']:.3e}",
                 f"{sa['max_abs_change']:.3e}", f"{sa['max_rel_change']:.3e}", sa["crossing_changed_0.05"],
                 sa["crossing_changed_0.1"], f"{sb['max_abs_change']:.3e}", f"{sb['max_rel_change']:.3e}",
                 sb["crossing_changed_0.05"], sb["crossing_changed_0.1"], f"{lm['omega_C_observed']:.3e}",
                 f"{sc['max_abs_change']:.3e}", sc["crossing_changed_0.05"], sc["crossing_changed_0.1"],
                 sa["uncovered_before"],
                 sa["uncovered_after"], f"{rr['breakdown_uniform_omega']:.3e}"]
            )  # fmt: skip
    print("wrote", out)


if __name__ == "__main__":
    main()
