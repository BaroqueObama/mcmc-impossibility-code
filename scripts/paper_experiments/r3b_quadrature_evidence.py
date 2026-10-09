"""Compare Student-t and funnel embedding quadrature with independent rules.

Use refined RKHS norms, Gauss-Kronrod quadrature and Monte Carlo to assess
the declared tolerance eps_quad=1e-10. These comparisons support the margin
sqrt(3 eps_quad); they do not certify the quadrature error. Results go to
$MCMC_PAPER_OUT/r3b_suite/diag/quadrature_evidence/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch

from mcmc.paperexp import r3_suite as r3
from mcmc.paperexp.embeddings import (
    FunnelEmbedding,
    StudentTEmbedding,
    default_probe_points,
    quadpack_mu,
)
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.targets import load_frozen_suite

F64 = torch.float64
LD = np.longdouble
DEFAULT_COMBOS = [
    "studentt-d10-nu3:ell",
    "funnel-d10-sy3:ell",
    "studentt-d10-nu3:ell_half",
    "studentt-d10-nu3:ell_double",
    "funnel-d10-sy3:ell_half",
    "funnel-d10-sy3:ell_double",
    "studentt-d10-nu5:ell",
    "studentt-d10-nu30:ell",
    "studentt-d2-nu3:ell",
    "funnel-d10-sy1:ell",
    "funnel-d2-sy3:ell",
]


def node_gram_row(emb, nodes_a, nodes_b, lam) -> np.ndarray:
    """<mu_a, mu_b>_H for node laws (longdouble): Student-t nodes tau (N(0, tau I)); funnel nodes y
    (delta_y x N(0, e^y I))."""
    a = np.asarray(nodes_a, LD)[:, None]
    b = np.asarray(nodes_b, LD)[None, :]
    lam = np.asarray(lam, LD)
    if isinstance(emb, StudentTEmbedding):
        s = a + b
        return np.exp(-0.5 * np.log1p(s[..., None] / lam).sum(-1))
    s = np.exp(a) + np.exp(b)
    return np.exp(-0.5 * (a - b) ** 2 / lam[0] - 0.5 * np.log1p(s[..., None] / lam[1:]).sum(-1))


def rkhs_distance(emb, rule_a, rule_b, rows: int = 256) -> dict:
    """||sum w_a mu_a - sum w_b mu_b||_H^2 in longdouble (signed weights on the union of nodes), with
    a rounding floor (pairwise sums of <= rows N terms, sequential chunk totals, d-term log-sums in
    each Gram entry: (log2(rows N) + n_chunks + 2 + 2 d) u_ld sum |w_i w_j| K_ij) and the upper
    bound eta_upper = sqrt(max(eta2, 0) + floor)."""
    nodes = np.concatenate([rule_a[0], rule_b[0]])
    w = np.concatenate([np.asarray(rule_a[1], LD), -np.asarray(rule_b[1], LD)])
    total, total_abs, n_chunks = LD(0), LD(0), 0
    for i0 in range(0, len(nodes), rows):
        G = node_gram_row(emb, nodes[i0 : i0 + rows], nodes, emb.lam)
        P = w[i0 : i0 + rows, None] * G * w[None, :]
        total += P.sum()
        total_abs += np.abs(P).sum()
        n_chunks += 1
    u_ld = float(np.finfo(LD).eps) / 2
    d = len(emb.lam)
    floor = (math.log2(rows * len(nodes)) + n_chunks + 2 + 2 * d) * u_ld * float(total_abs)
    eta2 = float(total)
    return {
        "eta2": eta2,
        "rounding_floor": floor,
        "eta_upper": math.sqrt(max(eta2, 0.0) + floor),
        "n_nodes": int(len(nodes)),
    }


def rule(emb, n: int, kind: str | None = None):
    """(nodes, weights) of the embedding's rule at order n (or the funnel Gauss-Hermite rule)."""
    if kind == "gauss_hermite_y":
        old = emb.rule_name
        emb.rule_name = kind
        emb._set_rule(n)
        out = (emb._nodes.copy(), emb._weights.copy())
        emb.rule_name = old
    else:
        emb._set_rule(n)
        out = (emb._nodes.copy(), emb._weights.copy())
    return out


def eval_rule(emb, r, X: torch.Tensor) -> np.ndarray:
    emb._nodes, emb._weights = r
    emb._dev = {}
    return emb._eval(X).numpy()


def evaluation_points(suite, tid: str, gen_seed: int = 2026) -> tuple[torch.Tensor, list]:
    tgt = suite.target(tid)
    g = torch_gen(gen_seed, "cpu")
    parts = []
    parts.append(("target draws", tgt.sample(400, g)))
    X0, _ = r3.sample_init_counted(tgt, 200, 0.8, g)
    parts.append(("known-radius init draws", X0))
    m = torch.as_tensor(np.asarray(tgt.mean(), np.float64), dtype=F64)
    parts.append(("3x dispersed draws", m + 3.0 * (tgt.sample(100, g) - m)))
    s = np.asarray(suite.scales(tid), np.float64)
    x = m.clone()
    x[int(np.argmax(s))] += 3.0 * float(s[int(np.argmax(s))])
    parts.append(("point start", x[None, :]))
    cid = f"{tid}__rwm"
    step = r3.MHStep(suite.sampler(cid))
    T3 = r3.r3_schedule(int(suite.config(cid)["t_max"]))[-1]
    X0c, _ = r3.sample_init_counted(tgt, 50, 0.8, g)
    ch = r3.GroupedChains(step, [("c", X0c, torch_gen(gen_seed + 1, "cpu"))])
    for t in sorted({1, 16, 256, T3}):
        ch.advance_to(t)
        parts.append((f"RWM replica states t={t}", ch.get("c").clone()))
    y = tgt.sample(100, g)
    parts.append(("4x / 8x scaled far points", torch.cat([4.0 * y[:50], 8.0 * y[50:]])))
    X = torch.cat([p[1] for p in parts])
    labels = [lab for lab, p in parts for _ in range(p.shape[0])]
    return X, labels


def mc_embedding(kernel, tgt, X: torch.Tensor, n: int, seed: int, chunk: int = 1 << 16):
    g = torch_gen(seed, "cpu")
    s1 = torch.zeros(X.shape[0], dtype=F64)
    s2 = torch.zeros(X.shape[0], dtype=F64)
    done = 0
    while done < n:
        mb = min(chunk, n - done)
        Y = tgt.sample(mb, g)
        K = kernel(X, Y)
        s1 += K.sum(1)
        s2 += (K * K).sum(1)
        done += mb
    mean = s1 / n
    var = (s2 / n - mean**2) * n / (n - 1)
    return mean.numpy(), np.sqrt(np.clip(var.numpy(), 0, None) / n)


def norm2_dblquad(emb, tgt) -> tuple[float, float]:
    """||mu_pi||^2 by 2-D adaptive quadrature over the mixing variables (scipy dblquad)."""
    from scipy import integrate, stats

    lam = emb.lam
    if isinstance(emb, StudentTEmbedding):
        nu = tgt.nu
        pdf = stats.chi2(nu).pdf

        def f(w2, w1):
            s = nu / w1 + nu / w2
            return math.exp(-0.5 * np.log1p(s / lam).sum()) * pdf(w1) * pdf(w2)

        return integrate.dblquad(f, 0, np.inf, 0, np.inf, epsabs=1e-14, epsrel=1e-12)
    sig = tgt.sigma_y
    L = sig * math.sqrt(-2.0 * math.log(1e-35))

    def g(y2, y1):
        s = math.exp(y1) + math.exp(y2)
        return math.exp(
            -0.5 * (y1 / sig) ** 2
            - 0.5 * (y2 / sig) ** 2
            - 0.5 * (y1 - y2) ** 2 / lam[0]
            - 0.5 * np.log1p(s / lam[1:]).sum()
        ) / (2 * math.pi * sig * sig)

    return integrate.dblquad(g, -L, L, -L, L, epsabs=1e-14, epsrel=1e-12)


def evidence(suite, tid: str, bw: str, n_mc: int) -> dict:
    t0 = time.perf_counter()
    tgt = suite.target(tid)
    emb = suite.embedding(tid, bw=bw)
    kernel = suite.kernel(tid, kind="rbf", bw=bw)
    n = int(emb.info["order"])
    prod = (emb._nodes.copy(), emb._weights.copy())
    out = {
        "target": tid,
        "bw": bw,
        "rule": emb.info["rule"],
        "order": n,
        "declared_eps_quad": 1e-10,
        "q": math.sqrt(3e-10),
        "probe_points_production": int(emb.info["n_probe"]),
    }
    # RKHS distance bounds the difference between rules at every x.
    eta = {}
    for f in (2 * n, 4 * n):
        r_f = rule(emb, f)
        eta[f"trapezoid_{f}"] = rkhs_distance(emb, prod, r_f)
    if isinstance(emb, FunnelEmbedding):
        r_gh = rule(emb, 8192, "gauss_hermite_y")
        eta["gauss_hermite_8192"] = rkhs_distance(emb, prod, r_gh)
    out["rkhs_distance"] = eta
    X, labels = evaluation_points(suite, tid)
    mu_n = eval_rule(emb, prod, X)
    r4 = rule(emb, 4 * n)
    mu_4n = eval_rule(emb, r4, X)
    emb._nodes, emb._weights = prod
    emb._dev = {}
    lam_ok = np.array_equal(emb(X).numpy(), mu_n)
    qp = np.array([quadpack_mu(tgt, kernel, x) for x in X.numpy()])
    mc, se = mc_embedding(kernel, tgt, X, n_mc, seed=777)
    # Rare-event estimates have unreliable standard errors; require mu * n_mc >= 1000.
    zok = mu_n * n_mc >= 1000.0
    z = np.where(zok, (mu_n - mc) / np.where(zok, se, 1.0), 0.0)
    by_group = {}
    for lab in dict.fromkeys(labels):
        idx = np.array([i for i, x in enumerate(labels) if x == lab])
        by_group[lab] = {
            "n": int(len(idx)),
            "mu_range": [float(mu_n[idx].min()), float(mu_n[idx].max())],
            "max_abs_vs_4n": float(np.abs(mu_n[idx] - mu_4n[idx]).max()),
            "max_abs_vs_quadpack": float(np.abs(mu_n[idx] - qp[idx, 0]).max()),
            "max_abs_z_mc": float(np.abs(z[idx]).max()),
            "n_z": int(zok[idx].sum()),
        }
    out["points"] = {
        "n": int(X.shape[0]),
        "production_eval_reproduced": bool(lam_ok),
        "max_abs_vs_4n": float(np.abs(mu_n - mu_4n).max()),
        "max_abs_vs_quadpack": float(np.abs(mu_n - qp[:, 0]).max()),
        "quadpack_err_estimate_max": float(qp[:, 1].max()),
        "max_rel_vs_quadpack_mu_gt_1e-6": float(
            (np.abs(mu_n - qp[:, 0]) / np.abs(qp[:, 0]))[np.abs(qp[:, 0]) > 1e-6].max()
        ),
        "mc_draws": int(n_mc),
        "max_abs_vs_mc": float(np.abs(mu_n - mc).max()),
        "mc_se_max": float(se.max()),
        "n_z_mc": int(zok.sum()),
        "max_abs_z_mc": float(np.abs(z[zok]).max()),
        "mean_z_mc": float(z[zok].mean()),
        "z_rule": "points with mu * n_mc >= 1000",
        "by_group": by_group,
    }
    n2 = float(emb.norm2())
    emb._nodes, emb._weights = r4
    n2_4n = float(emb._norm2_rule())
    emb._nodes, emb._weights = prod
    emb._dev = {}
    dq, dq_err = norm2_dblquad(emb, tgt)
    out["norm2"] = {
        "production": n2,
        "abs_vs_4n": abs(n2 - n2_4n),
        "dblquad": dq,
        "dblquad_err_estimate": dq_err,
        "abs_vs_dblquad": abs(n2 - dq),
    }
    eta_max = max(v["eta_upper"] for v in out["rkhs_distance"].values())
    out["verdict"] = {
        "max_pointwise_vs_independent_quadrature": out["points"]["max_abs_vs_quadpack"],
        "max_rkhs_distance_to_refined_rules": eta_max,
        "norm2_max_abs_diff": max(out["norm2"]["abs_vs_4n"], out["norm2"]["abs_vs_dblquad"]),
        "within_declared_1e-10": bool(
            out["points"]["max_abs_vs_quadpack"] <= 1e-10
            and out["points"]["max_abs_vs_4n"] <= 1e-10
            and max(out["norm2"]["abs_vs_4n"], out["norm2"]["abs_vs_dblquad"]) <= 1e-10
        ),
        "eta_below_q": bool(eta_max <= math.sqrt(3e-10)),
    }
    out["wall_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", nargs="*", default=DEFAULT_COMBOS)
    ap.add_argument("--n-mc", type=int, default=1 << 22)
    a = ap.parse_args()
    out_dir = (
        pathlib.Path(os.environ["MCMC_PAPER_OUT"]) / "r3b_suite" / "diag" / "quadrature_evidence"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = load_frozen_suite()
    summary = {}
    for combo in a.combos:
        tid, bw = combo.split(":")
        ev = evidence(suite, tid, bw, a.n_mc)
        (out_dir / f"{tid}__{bw}.json").write_text(json.dumps(ev, indent=1))
        summary[combo] = {
            "order": ev["order"],
            **ev["verdict"],
            "max_abs_vs_4n": ev["points"]["max_abs_vs_4n"],
            "max_abs_z_mc": ev["points"]["max_abs_z_mc"],
            "n_z_mc": ev["points"]["n_z_mc"],
            "mc_se_max": ev["points"]["mc_se_max"],
            "n_points": ev["points"]["n"],
            "wall_s": ev["wall_s"],
        }
        print(combo, json.dumps(summary[combo]), flush=True)
    prev = out_dir / "summary.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old.update(summary)
    prev.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    _ = default_probe_points
    main()
