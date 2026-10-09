"""Anytime MMD bounds for systematic-scan Gibbs on UCI Concrete.

From pi_0 = N(mu + a, C), the law at time t is N(mu + A^t a, C), so MMD
and coverage can be evaluated analytically. Each ensemble regenerates the
replicas, exact posterior reference draws and branched operator samples.

B3 uses a static span; B4 uses either that span or the time-t span. Known
operators use landmark compression with an omitted-trace bound. Estimated
operators include sampling and truncation errors. Both are compared with
the direct MMD bound and bounds without observations.

The failure budget is split across replicas, the target reference and the
operators. The replica event covers all spans, so m needs no budget split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import resource
import time
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import bounds as bd
from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import operators as op
from mcmc.paperexp.runio import RunDir, utc_now

log = logging.getLogger(__name__)
F64 = torch.float64
FAMILIES = ("V3a", "V4a", "V3b", "V4b")  # V3b/V4b expand over the N grid


def seed_for(namespace: str, stage: str, rep: int) -> int:
    """sha256-derived 63-bit seed; every estimation stage has its own stream."""
    h = hashlib.sha256(f"{namespace}|{stage}|{rep}".encode()).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def torch_gen(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


class CountingKernel:
    """Wraps a kernel callable and counts evaluated entries per named stage."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.kappa2 = kernel.kappa2
        self.counts: dict[str, int] = {}
        self.stage = "other"

    def __call__(self, X, Y):
        self.counts[self.stage] = self.counts.get(self.stage, 0) + X.shape[0] * Y.shape[0]
        return self.kernel(X, Y)

    def rowsum(self, X, Y):
        self.counts[self.stage] = self.counts.get(self.stage, 0) + X.shape[0] * Y.shape[0]
        f = getattr(self.kernel, "rowsum", None)
        return f(X, Y) if f is not None else self.kernel(X, Y).sum(1)

    def __getattr__(self, name):
        return getattr(self.kernel, name)


@dataclass
class Setting:
    name: str
    direction: str
    r_init: float
    shift: np.ndarray


@dataclass
class Problem:
    chain: gl.LinearGaussianChain
    kernel: gl.AnisoRBF
    settings: list
    checkpoints: list
    m_grid: list
    J_grid: list
    N_grid: list
    N_pi: int
    budget: bd.BudgetAllocation
    basis: gl.LandmarkBasis
    numerical_margin: float
    op_rtol: float
    head: int
    n_dirs: int  # number of leading directions kept per span family (>= max m)
    landmark_closed_form: str = "standard"  # "stable": cancellation-free Sigma_t moments (opt-in)

    def __post_init__(self):
        assert self.n_dirs >= max(self.m_grid)
        assert self.checkpoints == sorted(self.checkpoints) and self.checkpoints[0] == 0
        self.truth = np.array(
            [
                [
                    math.sqrt(
                        gl.mmd2_shifted_gaussians(
                            self.kernel, self.chain.C, self.chain.power(t) @ s.shift
                        )
                    )
                    for t in self.checkpoints
                ]
                for s in self.settings
            ]
        )
        T = len(self.checkpoints)
        self.b_t = np.array(
            [
                bd.replica_radius(
                    1.0, J, self.budget.beta("replicas"), self.budget.weight("replicas")
                )
                for J in self.J_grid
            ]
        )
        self.b_pi = bd.target_radius(1.0, self.N_pi, self.budget.beta("target"))
        self.e_tN = np.array(
            [
                [
                    op.operator_radius(
                        1.0, N, self.budget.beta("operator"), self.budget.weight("operator")
                    )
                    for N in self.N_grid
                ]
                for _ in range(T)
            ]
        )
        self._build_ideal()

    def _build_ideal(self) -> None:
        k, nu = self.kernel, self.numerical_margin
        T, nm = len(self.checkpoints), len(self.m_grid)
        self.ideal_ops = {
            t: gl.landmark_operator(k, self.basis, self.chain, t, self.landmark_closed_form)
            for t in self.checkpoints
        }
        L0 = self.ideal_ops[0]
        V3 = L0.top_basis(self.n_dirs)
        self.U3a_unit = np.array(
            [math.sqrt(L0.compressed_top(V3[:, :m]) + L0.tau + nu) for m in self.m_grid]
        )
        self.U4a_same_unit = np.zeros((T, nm))
        self.U4a_own_unit = np.zeros((T, nm))
        self.NO4a_unit = np.zeros(T)
        self.ideal_tau = np.zeros(T)
        self.ideal_top = np.zeros((T, self.n_dirs + 1))
        self.V4a_coefs = {}
        for ti, t in enumerate(self.checkpoints):
            Lt = self.ideal_ops[t]
            V4 = Lt.top_basis(self.n_dirs)
            self.V4a_coefs[t] = self.basis.W @ V4
            for mi, m in enumerate(self.m_grid):
                self.U4a_same_unit[ti, mi] = math.sqrt(Lt.compressed_top(V3[:, :m]) + Lt.tau + nu)
                self.U4a_own_unit[ti, mi] = math.sqrt(Lt.compressed_top(V4[:, :m]) + Lt.tau + nu)
            self.NO4a_unit[ti] = math.sqrt(max(Lt.lam(1), 0.0) + Lt.tau + nu)
            self.ideal_tau[ti] = Lt.tau
            self.ideal_top[ti] = [Lt.lam(j) for j in range(1, self.n_dirs + 2)]
        self.NO3a_unit = self.NO4a_unit[0]
        self.V3a_coefs = self.basis.W @ V3


def mu_pi_torch(problem: Problem, P: torch.Tensor, mean: np.ndarray | None = None) -> torch.Tensor:
    """Closed-form mean embedding of N(mean, C) at points P (torch, on P's device)."""
    chain, k = problem.chain, problem.kernel
    Lam = gl.lengthscale_matrix(k)
    mean = chain.mu if mean is None else mean
    LS = Lam + chain.C
    c = math.exp(-0.5 * (np.linalg.slogdet(LS)[1] - np.linalg.slogdet(Lam)[1]))
    Pinv = torch.as_tensor(np.linalg.inv(LS), dtype=F64, device=P.device)
    diff = P - torch.as_tensor(mean, dtype=F64, device=P.device)[None, :]
    return c * torch.exp(-0.5 * ((diff @ Pinv) * diff).sum(1))


def delta_shift_torch(problem: Problem, P: torch.Tensor, b: np.ndarray) -> torch.Tensor:
    """mu_{N(mu + b, C)}(P) - mu_{N(mu, C)}(P) without cancellation (same c and Pinv as mu_pi_torch).

    = c exp(-q0/2) expm1(-(q_b - q0)/2) with q_b - q0 = -2 b^T Pinv (P - mu) + b^T Pinv b, so the
    difference keeps its relative accuracy for tiny shifts b (late checkpoints).
    """
    chain, k = problem.chain, problem.kernel
    Lam = gl.lengthscale_matrix(k)
    LS = Lam + chain.C
    c = math.exp(-0.5 * (np.linalg.slogdet(LS)[1] - np.linalg.slogdet(Lam)[1]))
    Pinv_np = np.linalg.inv(LS)
    Pinv = torch.as_tensor(Pinv_np, dtype=F64, device=P.device)
    u = P - torch.as_tensor(chain.mu, dtype=F64, device=P.device)[None, :]
    q0 = ((u @ Pinv) * u).sum(1)
    b = np.asarray(b, dtype=np.float64)
    Pb = torch.as_tensor(Pinv_np @ b, dtype=F64, device=P.device)
    dq = -2.0 * (u @ Pb) + float(b @ Pinv_np @ b)
    return c * torch.exp(-0.5 * q0) * torch.expm1(-0.5 * dq)


def run_ensemble(problem: Problem, rep: int, namespace: str, device) -> dict:
    """Regenerate replicas, target draws and operator blocks for one ensemble."""
    t_start = time.perf_counter()
    ch, S, T = problem.chain, len(problem.settings), len(problem.checkpoints)
    nJ, nm, nN = len(problem.J_grid), len(problem.m_grid), len(problem.N_grid)
    Jmax, Nmax, nd = problem.J_grid[-1], problem.N_grid[-1], problem.n_dirs
    cj = min(1024, problem.J_grid[0])  # replica chunk (prefixes must be multiples of it)
    k = CountingKernel(problem.kernel)
    params = ch.torch_params(device)
    timing: dict[str, float] = {}

    def tic(name, t0):
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0

    t0 = time.perf_counter()
    g_tgt = torch_gen(seed_for(namespace, "ens-target", rep), device)
    g_op = torch_gen(seed_for(namespace, "ens-operator", rep), device)
    g_rep = [
        torch_gen(seed_for(namespace, f"ens-replicas-{s.name}", rep), device)
        for s in problem.settings
    ]
    Xpi = ch.sample_stationary(problem.N_pi, g_tgt, device)
    X0 = ch.sample_stationary(Nmax, g_op, device)
    X0p = ch.sample_stationary(Nmax, g_op, device)
    branches = torch.cat([X0, X0p, X0, X0p], 0)  # Y, Y', Z, Z' (t = 0: Y = Z = X)
    reps = [
        ch.sample_stationary(Jmax, g_rep[i], device, shift=s.shift)
        for i, s in enumerate(problem.settings)
    ]
    tic("sampling", t0)

    # target-side quantities that do not change with t
    Zl = torch.as_tensor(problem.basis.Z, dtype=F64, device=device)
    E0 = branches.clone()
    t0 = time.perf_counter()
    k.stage = "target_means"
    tgt_fixed = bd.kernel_prefix_means(
        k, torch.cat([Zl, E0], 0), Xpi, [problem.N_pi], chunk=min(4096, problem.N_pi)
    )[0]
    ref_sampled = bd.TargetReference.sampled(Xpi, 1.0, problem.budget.beta("target"))
    k.stage = "direct"
    norm2_pi_hat = ref_sampled.norm2(k)
    n2_pi = gl.mean_embedding_norm2(problem.kernel, ch.C)
    ref_known = bd.TargetReference.known(lambda P: mu_pi_torch(problem, P), n2_pi)
    tgt_event = math.sqrt(
        max(norm2_pi_hat - 2.0 * float(mu_pi_torch(problem, Xpi).mean()) + n2_pi, 0.0)
    )
    known_fixed = mu_pi_torch(problem, torch.cat([Zl, E0], 0))
    tic("target", t0)

    V3a = torch.as_tensor(problem.V3a_coefs, dtype=F64, device=device)
    nP = Zl.shape[0] + 8 * Nmax

    def pad_landmark(coefs):
        full = torch.zeros(nP, coefs.shape[1], dtype=F64, device=device)
        full[: Zl.shape[0]] = coefs
        return full

    out = {
        "proj": np.zeros(
            (S, T, nJ, 2 + 2 * nN, nm, 2)
        ),  # families: V3a, V4a, V3b[N..], V4b[N..]; ref: sampled, known
        "proj_true": np.zeros(
            (S, T, 2 + 2 * nN, nm)
        ),  # exact ||P_V delta_t|| (closed-form delta_t)
        "direct": np.zeros((S, T, nJ, 2)),
        "repl_event": np.zeros((S, T, nJ)),
        "direct_err2": np.zeros((S, T, 2)),  # fp32 squared-norm error bounds (0 in the fp64 path)
        "tgt_event": tgt_event,
        "U3b_unit": np.zeros((nN, nm)),
        "NO3b_unit": np.zeros(nN),
        "U4b_same_unit": np.zeros((T, nN, nm)),
        "U4b_own_unit": np.zeros((T, nN, nm)),
        "NO4b_unit": np.zeros((T, nN)),
        "op_rank": np.zeros((T, nN), dtype=np.int64),
        "op_npos": np.zeros((T, nN), dtype=np.int64),
        "op_delta_trunc": np.zeros((T, nN)),
        "op_orth_err": np.zeros((T, nN)),
        "op_top": np.zeros((T, nN, nd + 1)),
        "op_min_eig": np.zeros((T, nN)),
        "op_head_slack_max": np.zeros((T, nN)),
        "span_orth_err_max": 0.0,
        "kernel_evals_cum": np.zeros(
            (T, 5), dtype=np.float64
        ),  # target_means, repl_means, operator, direct, total
        "wall_cum": np.zeros(T),
    }
    V3b_spans = {}
    t_prev = 0
    for ti, t in enumerate(problem.checkpoints):
        gap = t - t_prev
        t0 = time.perf_counter()
        if gap:
            branches = ch.sweep(branches, g_op, gap, params)
            reps = [ch.sweep(reps[i], g_rep[i], gap, params) for i in range(S)]
        tic("sampling", t0)
        t_prev = t
        Y, Yp, Z, Zp = branches.split(Nmax)

        # estimated operators (nested N)
        t0 = time.perf_counter()
        k.stage = "operator"
        coef_cols = [
            pad_landmark(V3a),
            pad_landmark(torch.as_tensor(problem.V4a_coefs[t], dtype=F64, device=device)),
        ]
        for ni, N in enumerate(problem.N_grid):
            est = op.estimate_operator(k, Y[:N], Z[:N], Yp[:N], Zp[:N], rtol=problem.op_rtol)
            e = problem.e_tN[ti, ni] + est.delta_trunc
            out["op_rank"][ti, ni], out["op_npos"][ti, ni] = est.rank, est.n_pos
            out["op_delta_trunc"][ti, ni], out["op_orth_err"][ti, ni] = (
                est.delta_trunc,
                est.eigvec_orthonormality_error,
            )
            out["op_top"][ti, ni] = [est.lambda_plus(j) for j in range(1, nd + 2)]
            out["op_min_eig"][ti, ni] = float(est.eigvals[-1])
            if ti == 0:
                V3b_spans[N] = est.top_span(nd, label=f"V3b_N{N}")
                for mi, (val, info) in enumerate(
                    est.compressed_plus_norms(k, V3b_spans[N], problem.m_grid, problem.head)
                ):
                    out["U3b_unit"][ni, mi] = math.sqrt(val + e)
                out["NO3b_unit"][ni] = math.sqrt(est.lambda_plus(1) + e)
            own = est.top_span(nd, label=f"V4b_t{t}_N{N}")
            same_vals = est.compressed_plus_norms(k, V3b_spans[N], problem.m_grid, problem.head)
            own_vals = est.compressed_plus_norms(k, own, problem.m_grid, problem.head)
            slack = 0.0
            for mi in range(nm):
                (v_same, info_s), (v_own, info_o) = same_vals[mi], own_vals[mi]
                slack = max(slack, info_s["slack_bound"], info_o["slack_bound"])
                out["U4b_same_unit"][ti, ni, mi] = math.sqrt(v_same + e)
                out["U4b_own_unit"][ti, ni, mi] = math.sqrt(v_own + e)
            out["op_head_slack_max"][ti, ni] = slack
            out["NO4b_unit"][ti, ni] = math.sqrt(est.lambda_plus(1) + e)
            # coefficient columns over the global point set [landmarks, E0 (4 Nmax), Et (4 Nmax)]
            for spec, span_coefs in (("E0", V3b_spans[N].coefs), ("Et", own.coefs)):
                col = torch.zeros(nP, nd, dtype=F64, device=device)
                base = Zl.shape[0] + (0 if spec == "E0" else 4 * Nmax)
                for b in range(4):
                    col[base + b * Nmax : base + b * Nmax + N] = span_coefs[b * N : (b + 1) * N]
                coef_cols.append((spec, ni, col))
            del est, own
        tic("operator", t0)
        cols = [coef_cols[0], coef_cols[1]]
        cols += [c for (spec, ni, c) in coef_cols[2:] if spec == "E0"]
        cols += [c for (spec, ni, c) in coef_cols[2:] if spec == "Et"]
        Coef = torch.cat(cols, 1)  # (nP, (2 + 2 nN) * nd)

        # delta_hat at the global point set; projected norms
        t0 = time.perf_counter()
        Et = branches
        P = torch.cat([Zl, E0, Et], 0)
        k.stage = "target_means"
        tgt_Et = bd.kernel_prefix_means(k, Et, Xpi, [problem.N_pi], chunk=min(4096, problem.N_pi))[
            0
        ]
        tgt_s = torch.cat([tgt_fixed, tgt_Et])
        tgt_k = torch.cat([known_fixed, mu_pi_torch(problem, Et)])
        tic("target", t0)
        t0 = time.perf_counter()
        for si, s in enumerate(problem.settings):
            k.stage = "repl_means"
            rm = bd.kernel_prefix_means(k, P, reps[si], problem.J_grid, chunk=cj)  # (nJ, nP)
            for ri, tg in enumerate((tgt_s, tgt_k)):
                coords = (rm - tg[None, :]) @ Coef  # (nJ, F * nd)
                cum = torch.cumsum(coords.view(nJ, -1, nd) ** 2, 2)  # (nJ, F, nd)
                for mi, m in enumerate(problem.m_grid):
                    out["proj"][si, ti, :, :, mi, ri] = cum[:, :, m - 1].sqrt().cpu().numpy()
            mean_t = ch.mu + ch.power(t) @ s.shift
            if getattr(problem, "stable_delta_true", False):
                # Avoid cancellation for tiny shifts at late checkpoints.
                delta_true = delta_shift_torch(problem, P, ch.power(t) @ s.shift)
            else:
                delta_true = mu_pi_torch(problem, P, mean=mean_t) - tgt_k
            cum_true = torch.cumsum((delta_true @ Coef).view(-1, nd) ** 2, 1)  # (F, nd)
            for mi, m in enumerate(problem.m_grid):
                out["proj_true"][si, ti, :, mi] = cum_true[:, m - 1].sqrt().cpu().numpy()
            # direct bound and the exact replica-event check
            k.stage = "direct"
            rs = bd.ReplicaSet(reps[si], problem.J_grid, chunk=cj)
            cross_k = rs.cross(k, ref_known)
            if getattr(problem, "direct_fp32", False):
                # fp32 kernel entries with fp64 sums; add the squared-norm error bound.
                n2, e_n2, cross_s, e_c = bd.fp32_vstat_and_cross(
                    problem.kernel, reps[si], problem.J_grid, Xpi, chunk=cj
                )
                Jm = problem.J_grid[-1]
                k.counts["direct"] = k.counts.get("direct", 0) + Jm * Jm + Jm * Xpi.shape[0]
                err_s, err_k = e_n2 + 2.0 * e_c, e_n2
                out["direct_err2"][si, ti] = (err_s, err_k)
            else:
                n2 = rs.norm2(k)
                cross_s = rs.cross(k, ref_sampled)
                err_s = err_k = 0.0
            d_s = n2 - 2.0 * cross_s + norm2_pi_hat
            d_k = n2 - 2.0 * cross_k + n2_pi
            for dd, ee in ((d_s, err_s), (d_k, err_k)):
                if float(dd.min()) < -1e-12 - ee:
                    raise FloatingPointError(f"negative squared distance {float(dd.min())}")
            out["direct"][si, ti, :, 0] = (d_s.clamp_min(0) + err_s).sqrt().cpu().numpy()
            out["direct"][si, ti, :, 1] = (d_k.clamp_min(0) + err_k).sqrt().cpu().numpy()
            mut_vals = mu_pi_torch(problem, reps[si][:Jmax], mean=mean_t)
            cs = torch.cumsum(mut_vals, 0)
            cross_t = torch.stack([cs[J - 1] / J for J in problem.J_grid])
            ev = (
                n2 - 2.0 * cross_t + n2_pi
            )  # ||mu_hat_t - mu_{pi_t}||^2 (same covariance => same norm)
            out["repl_event"][si, ti] = (ev.clamp_min(0) + err_k).sqrt().cpu().numpy()
        tic("projections_direct", t0)
        counts = k.counts
        out["kernel_evals_cum"][ti] = [
            counts.get("target_means", 0),
            counts.get("repl_means", 0),
            counts.get("operator", 0),
            counts.get("direct", 0),
            sum(counts.values()),
        ]
        out["wall_cum"][ti] = time.perf_counter() - t_start
    # Check each t=0 span once per ensemble.
    out["span_orth_err_max"] = max(
        op.require_orthonormal(problem.kernel, V3b_spans[N], tol=1e-6) for N in problem.N_grid
    )
    out["timing"] = timing
    out["wall"] = time.perf_counter() - t_start
    out["peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    out["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return out


def build_linear_problem(chain, kernel, settings, cfg_like: dict, namespace: str) -> Problem:
    """Construct the shared Gaussian-chain problem from its model and grids."""
    checkpoints = [int(t) for t in cfg_like["checkpoints"]]
    budget = bd.BudgetAllocation(
        float(cfg_like["budget_total"]), dict(cfg_like["budget_components"])
    )
    budget.declare_grid("replicas", {"t": checkpoints, "J": list(cfg_like["J_grid"])})
    budget.declare_grid("operator", {"t": checkpoints, "N": list(cfg_like["N_grid"])})
    g_lm = torch_gen(seed_for(namespace, "landmarks", 0), "cpu")
    Zl = chain.sample_stationary(int(cfg_like["M"]), g_lm, "cpu").numpy()
    basis = gl.landmark_basis(kernel, Zl, rtol=float(cfg_like["landmark_rtol"]))
    return Problem(
        chain=chain,
        kernel=kernel,
        settings=settings,
        checkpoints=checkpoints,
        m_grid=[int(m) for m in cfg_like["m_grid"]],
        J_grid=[int(J) for J in cfg_like["J_grid"]],
        N_grid=[int(N) for N in cfg_like["N_grid"]],
        N_pi=int(cfg_like["N_pi"]),
        budget=budget,
        basis=basis,
        numerical_margin=float(cfg_like["numerical_margin"]),
        op_rtol=float(cfg_like["op_rtol"]),
        head=int(cfg_like["head"]),
        n_dirs=int(cfg_like["n_dirs"]),
        landmark_closed_form=str(cfg_like.get("landmark_closed_form", "standard")),
    )


def build_concrete_problem(cfg: DictConfig, device) -> tuple[Problem, dict]:
    from mcmc.paperexp.concrete import ConcretePosterior, load_concrete

    X, y, prov = load_concrete()
    post = ConcretePosterior(X, y, prior_var=float(cfg.model.prior_var))
    if abs(post.sigma2 - float(cfg.model.sigma2_declared)) > 0:
        raise ValueError(
            f"declared sigma2 {cfg.model.sigma2_declared} != OLS rounding {post.sigma2}"
        )
    ch = post.chain
    ns = str(cfg.seed_namespace)
    # bandwidth: median scaled pairwise distance of independent exact posterior draws (frozen)
    g_bw = torch_gen(seed_for(ns, "pilot-bw", 0), "cpu")
    D = ch.sample_stationary(int(cfg.kernel.bandwidth_draws), g_bw, "cpu") / torch.as_tensor(
        post.coord_sd
    )
    dist = torch.cdist(D, D)
    iu = torch.triu_indices(D.shape[0], D.shape[0], 1)
    ell_median = float(dist[iu[0], iu[1]].median())
    if cfg.kernel.ell_frozen is not None and abs(ell_median - float(cfg.kernel.ell_frozen)) > 1e-12:
        raise ValueError(f"median bandwidth {ell_median!r} != frozen {cfg.kernel.ell_frozen!r}")
    ell = ell_median * float(cfg.kernel.ell_factor)
    kernel = gl.AnisoRBF(post.coord_sd, ell)
    dirs = post.directions()
    settings = []
    for s in cfg.settings:
        v = dirs[str(s.direction)]["vec"]
        settings.append(
            Setting(str(s.name), str(s.direction), float(s.r_init), post.shift(v, float(s.r_init)))
        )
    checkpoints = [int(t) for t in cfg.checkpoints]
    # t_max rule: first power of two with analytic MMD(slow, R=0.8) < threshold (cap)
    if float(cfg.kernel.ell_factor) == 1.0:
        a = post.shift(dirs["slow"]["vec"], 0.8)
        t, tmax = 1, None
        while t <= int(cfg.t_max_cap):
            if math.sqrt(gl.mmd2_shifted_gaussians(kernel, ch.C, ch.power(t) @ a)) < float(
                cfg.t_max_threshold
            ):
                tmax = t
                break
            t *= 2
        expected = [0] + [2**j for j in range(int(math.log2(tmax)) + 1)]
        if checkpoints != expected:
            raise ValueError(f"checkpoints {checkpoints} != t_max rule {expected}")
    cfg_like = {
        "checkpoints": checkpoints,
        "budget_total": cfg.budget.total,
        "budget_components": dict(cfg.budget.components),
        "J_grid": list(cfg.J_grid),
        "N_grid": list(cfg.N_grid),
        "M": cfg.landmarks.M,
        "landmark_rtol": cfg.landmarks.rtol,
        "m_grid": list(cfg.m_grid),
        "N_pi": cfg.N_pi,
        "numerical_margin": cfg.landmarks.numerical_margin,
        "op_rtol": cfg.operator.rtol,
        "head": cfg.operator.head,
        "n_dirs": cfg.n_dirs,
    }
    problem = build_linear_problem(ch, kernel, settings, cfg_like, ns)
    basis = problem.basis
    meta = {
        "data_provenance": prov,
        "sigma2": post.sigma2,
        "ols_resid_var": post.ols_resid_var,
        "mu": post.mu.tolist(),
        "coord_sd": post.coord_sd.tolist(),
        "gibbs_eigvals_real": dirs["eigvals_real"],
        "gibbs_eigvals_imag": dirs["eigvals_imag"],
        "directions": {
            k_: {"eigval": dirs[k_]["eigval"], "vec": dirs[k_]["vec"].tolist()}
            for k_ in ("slow", "fast")
        },
        "reversibility_error": ch.reversibility_error(),
        "stationarity_error": ch.stationarity_error(),
        "ell_median": ell_median,
        "ell": ell,
        "landmark_rank": basis.rank,
        "landmark_orthonormality_error": basis.orthonormality_error,
    }
    return problem, meta


def problem_arrays(problem: Problem) -> dict:
    """Deterministic per-run arrays saved once (setup.npz)."""
    return {
        "checkpoints": np.array(problem.checkpoints),
        "m_grid": np.array(problem.m_grid),
        "J_grid": np.array(problem.J_grid),
        "N_grid": np.array(problem.N_grid),
        "r_init": np.array([s.r_init for s in problem.settings]),
        "shifts": np.array([s.shift for s in problem.settings]),
        "truth": problem.truth,
        "b_t": problem.b_t,
        "b_pi": np.array(problem.b_pi),
        "e_tN": problem.e_tN,
        "U3a_unit": problem.U3a_unit,
        "U4a_same_unit": problem.U4a_same_unit,
        "U4a_own_unit": problem.U4a_own_unit,
        "NO3a_unit": np.array(problem.NO3a_unit),
        "NO4a_unit": problem.NO4a_unit,
        "ideal_tau": problem.ideal_tau,
        "ideal_top": problem.ideal_top,
    }


REF_NAMES = ("sampled", "known")


def load_run(run_path: pathlib.Path) -> tuple[dict, dict]:
    setup = dict(np.load(run_path / "setup.npz"))
    chunks = sorted(run_path.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {run_path}")
    parts = [dict(np.load(c)) for c in chunks]
    for pp in parts:  # Older fp64 chunks have no direct_err2 field.
        if "direct_err2" not in pp:
            pp["direct_err2"] = np.zeros(pp["direct"].shape[:3] + (2,))
    data = {k: np.concatenate([pp[k] for pp in parts]) for k in parts[0] if k != "timing_keys"}
    data["timing_keys"] = parts[0]["timing_keys"]
    reps = data["reps"]
    if len(np.unique(reps)) != len(reps):
        raise ValueError("duplicate replicate ids across chunks")
    return setup, data


def load_runs(run_paths) -> tuple[dict, dict]:
    """Concatenate runs with identical setup arrays and disjoint replicate IDs."""
    loaded = [load_run(pathlib.Path(p)) for p in run_paths]
    setup = loaded[0][0]
    for s2, _ in loaded[1:]:
        if set(s2) != set(setup) or any(not np.array_equal(setup[k], s2[k]) for k in setup):
            raise ValueError("runs do not share an identical frozen problem (setup.npz differs)")
    parts = [d for _, d in loaded]
    data = {k: np.concatenate([pp[k] for pp in parts]) for k in parts[0] if k != "timing_keys"}
    data["timing_keys"] = parts[0]["timing_keys"]
    if len(np.unique(data["reps"])) != len(data["reps"]):
        raise ValueError("duplicate replicate ids across runs")
    return setup, data


def assemble_bounds(setup: dict, data: dict) -> dict:
    """All bounds B[method] with shape (E, S, T, J, m, ref) (direct/no-obs broadcast over missing axes)."""
    R = setup["r_init"]  # (S,)
    b_t, b_pi = setup["b_t"], float(setup["b_pi"])
    nN = len(setup["N_grid"])
    proj = data["proj"]  # (E, S, T, J, F, m, ref)
    bpi = np.array([b_pi, 0.0])
    H = (
        proj
        + b_t[None, None, None, :, None, None, None]
        + bpi[None, None, None, None, None, None, :]
    )
    E, S, T, nJ, F, nm, nref = H.shape
    Rb = R[None, :, None, None, None, None]

    def comb(Hf, U_unit):  # Hf (E,S,T,J,m,ref); U_unit broadcastable to (E,T,m)
        U = Rb * U_unit[:, None, :, None, :, None]
        return np.sqrt(Hf**2 + U**2), np.broadcast_to(U, Hf.shape)

    ones_E = np.ones((E, 1, 1))
    out, Hs, Us = {}, {}, {}
    U3a = ones_E * setup["U3a_unit"][None, None, :]
    U4a_same = ones_E * setup["U4a_same_unit"][None]
    U4a_own = ones_E * setup["U4a_own_unit"][None]
    for name, fam, Uu in (("B3a", 0, U3a), ("B4a_same", 0, U4a_same), ("B4a_own", 1, U4a_own)):
        out[name], Us[name] = comb(H[:, :, :, :, fam], Uu)
        Hs[name] = H[:, :, :, :, fam]
    for ni, N in enumerate(setup["N_grid"]):
        U3b = np.broadcast_to(data["U3b_unit"][:, ni][:, None, :], (E, T, nm))
        for name, fam, Uu in (
            (f"B3b_N{N}", 2 + ni, U3b),
            (f"B4b_same_N{N}", 2 + ni, data["U4b_same_unit"][:, :, ni]),
            (f"B4b_own_N{N}", 2 + nN + ni, data["U4b_own_unit"][:, :, ni]),
        ):
            out[name], Us[name] = comb(H[:, :, :, :, fam], Uu)
            Hs[name] = H[:, :, :, :, fam]
    direct = data["direct"] + b_t[None, None, None, :, None] + bpi[None, None, None, None, :]
    out["direct"] = np.broadcast_to(direct[:, :, :, :, None, :], (E, S, T, nJ, nm, nref))

    # no-observation bounds (no H): broadcast over J, m, ref
    def noobs(unit_ET):
        return np.broadcast_to(
            (Rb[..., 0] * unit_ET[:, None, :, None, None])[..., None], (E, S, T, nJ, nm, nref)
        )

    out["NO3a"] = noobs(ones_E[:, :, 0] * setup["NO3a_unit"] * np.ones((E, T)))
    out["NO4a"] = noobs(ones_E[:, :, 0] * setup["NO4a_unit"][None, :])
    for ni, N in enumerate(setup["N_grid"]):
        out[f"NO3b_N{N}"] = noobs(np.repeat(data["NO3b_unit"][:, ni][:, None], T, 1))
        out[f"NO4b_N{N}"] = noobs(data["NO4b_unit"][:, :, ni])
    return {"B": out, "H": Hs, "U": Us}


def method_cost(method: str, k: int, J: int, setup: dict, ref: str, design: dict) -> dict:
    """Operation counts for one method through checkpoint k, for one setting.

    Each method pays the full cost of the draws, sweeps, kernel evaluations and
    eigensolves it needs. Costs are comparable across methods but not additive.
    Shared setup (landmarks, analytic operators and bandwidth) is excluded.
    """
    ck = list(setup["checkpoints"])
    t, n_insp = int(ck[k]), k + 1
    M = int(design["M"])
    N = int(method.split("_N")[-1]) if "_N" in method else 0
    Npi = int(design["N_pi"]) if ref == "sampled" else 0
    c = {
        "sweep_replicas": 0.0,
        "operator_sweeps": 0.0,
        "exact_draws": 0.0,
        "kernel_evals": 0.0,
        "eigh_2N": 0.0,
    }
    observes = not method.startswith("NO")
    if observes:
        c["sweep_replicas"] = float(J * t)
        c["exact_draws"] += Npi
    if method.startswith(("B3b", "NO3b")):
        c["exact_draws"] += 2 * N
        c["kernel_evals"] += 3.0 * (4 * N) ** 2  # t=0 operator Gram + compressed norms
        c["eigh_2N"] += 2
    if method.startswith(("B4b", "NO4b")):
        c["exact_draws"] += 2 * N
        c["operator_sweeps"] = float(4 * N * t)
        c["kernel_evals"] += n_insp * 3.0 * (4 * N) ** 2 + (
            3.0 * (4 * N) ** 2 if "same" in method else 0.0
        )
        c["eigh_2N"] += 2 * n_insp + (2 if "same" in method else 0)
    if method in ("B3a", "B4a_same", "B4a_own"):
        c["kernel_evals"] += n_insp * M * J + M * Npi
    if method.startswith(("B3b", "B4b_same")):
        c["kernel_evals"] += n_insp * 4 * N * J + 4 * N * Npi
    if method.startswith("B4b_own"):
        c["kernel_evals"] += n_insp * 4 * N * (J + Npi)
    if method == "direct":
        c["kernel_evals"] += n_insp * (J * J + J * Npi) + Npi * Npi
    return c


def first_crossing(B_t: np.ndarray, eps: float) -> np.ndarray:
    """Index of the first checkpoint with B <= eps along the last axis; -1 if never (censored)."""
    hit = B_t <= eps
    idx = np.where(hit.any(-1), hit.argmax(-1), -1)
    return idx


def summarize(
    setup: dict,
    data: dict,
    tolerances=(0.05, 0.10),
    alpha: float = 0.05,
    design: dict | None = None,
) -> dict:
    """Summarize coverage and crossings; design={N_pi, M} also enables method costs."""
    from mcmc.paperexp import reference as ref

    asm = assemble_bounds(setup, data)
    B = asm["B"]
    truth = setup["truth"]  # (S, T)
    E = data["proj"].shape[0]
    S, T = truth.shape
    J_grid, m_grid = setup["J_grid"].tolist(), setup["m_grid"].tolist()
    ck = setup["checkpoints"]
    summ = {
        "n_ensembles": int(E),
        "reps_range": [int(data["reps"].min()), int(data["reps"].max())],
        "methods": {},
    }
    # coverage per method / setting / J / m / ref
    joint_ok = np.ones((E,), dtype=bool)
    joint_ok_by_ref = {r: np.ones((E,), dtype=bool) for r in REF_NAMES}
    for name, Bm in B.items():
        cov = Bm >= truth[None, :, :, None, None, None]  # (E,S,T,J,m,ref)
        cov_all_t = cov.all(2)  # (E,S,J,m,ref)
        joint_ok &= cov_all_t.all((1, 2, 3, 4))
        for ri, r in enumerate(REF_NAMES):
            joint_ok_by_ref[r] &= cov_all_t[..., ri].all((1, 2, 3))
        md = {}
        for si in range(S):
            for ji, J in enumerate(J_grid):
                for mi, m in enumerate(m_grid):
                    for ri, r in enumerate(REF_NAMES):
                        k_cov = int(cov_all_t[:, si, ji, mi, ri].sum())
                        Bv = Bm[:, si, :, ji, mi, ri]
                        rec = {
                            "covered_all_t": k_cov,
                            "cp": ref.clopper_pearson(k_cov, E, alpha),
                            "min_margin": float((Bv - truth[si][None, :]).min()),
                            "B_median": np.median(Bv, 0).tolist(),
                            "B_q05": np.quantile(Bv, 0.05, 0).tolist(),
                            "B_q95": np.quantile(Bv, 0.95, 0).tolist(),
                        }
                        if name in asm["H"]:
                            rec["H_median"] = np.median(
                                asm["H"][name][:, si, :, ji, mi, ri], 0
                            ).tolist()
                            rec["U_median"] = np.median(
                                asm["U"][name][:, si, :, ji, mi, ri], 0
                            ).tolist()
                        for eps in tolerances:
                            fc = first_crossing(Bv, eps)
                            crossed = fc >= 0
                            if design is not None:
                                costs = [
                                    method_cost(name, int(k_), J, setup, r, design)
                                    for k_ in fc[crossed]
                                ]
                                rec[f"cost_{eps}"] = (
                                    {
                                        key: float(np.median([c_[key] for c_ in costs]))
                                        for key in costs[0]
                                    }
                                    if costs
                                    else None
                                )
                            rec[f"cross_{eps}"] = {
                                "fraction": float(crossed.mean()),
                                "n_censored": int((~crossed).sum()),
                                "t_median_crossed": float(np.median(ck[fc[crossed]]))
                                if crossed.any()
                                else None,
                                "t_values": {
                                    int(t): int((fc == i).sum()) for i, t in enumerate(ck)
                                },
                            }
                        md[f"{si}|{J}|{m}|{r}"] = rec
        summ["methods"][name] = md
    summ["joint_all_methods_all_refs"] = {
        "covered": int(joint_ok.sum()),
        "cp": ref.clopper_pearson(int(joint_ok.sum()), E, alpha),
    }
    summ["joint_by_ref"] = {
        r: {"covered": int(v.sum()), "cp": ref.clopper_pearson(int(v.sum()), E, alpha)}
        for r, v in joint_ok_by_ref.items()
    }
    # component events (exact): replica event, target event, H-component, U-component
    b_t, b_pi = setup["b_t"], float(setup["b_pi"])
    rep_ok = (data["repl_event"] <= b_t[None, None, None, :]).all((1, 2, 3))
    tgt_ok = data["tgt_event"] <= b_pi
    summ["events"] = {
        "replica_event_all": int(rep_ok.sum()),
        "replica_event_max_ratio": float((data["repl_event"] / b_t[None, None, None, :]).max()),
        "target_event": int(tgt_ok.sum()),
        "target_event_max_ratio": float((data["tgt_event"] / b_pi).max()),
    }
    # H component: ||P_V delta_t|| <= H for every family/J/m/ref
    proj, pt = data["proj"], data["proj_true"]
    bpi = np.array([b_pi, 0.0])
    Hfull = (
        proj
        + b_t[None, None, None, :, None, None, None]
        + bpi[None, None, None, None, None, None, :]
    )
    H_ok = (pt[:, :, :, None, :, :, None] <= Hfull).all((1, 2, 3, 4, 5, 6))
    summ["events"]["H_component_all"] = int(H_ok.sum())
    # U component: ||P_{V-perp} delta_t|| = sqrt(MMD^2 - ||P_V delta_t||^2) <= U for each method's own span
    unmon = np.sqrt(np.clip(truth[None, :, :, None, None] ** 2 - pt**2, 0.0, None))  # (E,S,T,F,m)
    fam_of = {"B3a": 0, "B4a_same": 0, "B4a_own": 1}
    nN = len(setup["N_grid"])
    for ni, N in enumerate(setup["N_grid"]):
        fam_of[f"B3b_N{N}"] = 2 + ni
        fam_of[f"B4b_same_N{N}"] = 2 + ni
        fam_of[f"B4b_own_N{N}"] = 2 + nN + ni
    summ["events"]["U_component"] = {}
    for name, fam in fam_of.items():
        U = asm["U"][name][:, :, :, 0, :, 0]  # U does not depend on J/ref: (E,S,T,m)
        ok = (unmon[:, :, :, fam, :] <= U * (1 + 1e-12)).all((1, 2, 3))
        summ["events"]["U_component"][name] = {
            "covered": int(ok.sum()),
            "min_slack": float((U - unmon[:, :, :, fam, :]).min()),
        }
    # A time-t span can lower U while raising H, so B3 - B4 may be negative.
    pairs = [("B3a", "B4a_same"), ("B3a", "B4a_own"), ("B4a_same", "B4a_own")]
    for N in setup["N_grid"]:
        pairs += [
            (f"B3b_N{N}", f"B4b_same_N{N}"),
            (f"B3b_N{N}", f"B4b_own_N{N}"),
            (f"B4b_same_N{N}", f"B4b_own_N{N}"),
        ]
    summ["paired"] = {}
    for a, b in pairs:
        D = B[a] - B[b]  # (E,S,T,J,m,ref)
        mean, sd = D.mean(0), D.std(0, ddof=1) if E > 1 else np.zeros(D.shape[1:])
        summ["paired"][f"{a}-{b}"] = {
            "mean": mean.tolist(),
            "se": (sd / math.sqrt(E)).tolist(),
            "min": float(D.min()),
            "frac_negative_cells": float((D < -1e-12).mean()),
            "n_negative_cells": int((D < -1e-12).sum()),
            "n_cells": int(D.size),
        }
    summ["operator"] = {
        "rank_min": int(data["op_rank"].min()),
        "npos_min": int(data["op_npos"].min()),
        "delta_trunc_max": float(data["op_delta_trunc"].max()),
        "orth_err_max": float(data["op_orth_err"].max()),
        "head_slack_max": float(data["op_head_slack_max"].max()),
        "min_eig_min": float(data["op_min_eig"].min()),
        "span_orth_err_max": float(data["span_orth_err_max"].max()),
        "top_eig_median": np.median(data["op_top"], 0).tolist(),
    }
    tk = [str(x) for x in data["timing_keys"]]
    summ["cost"] = {
        "wall_per_ensemble_median": float(np.median(data["wall"])),
        "wall_total": float(data["wall"].sum()),
        "timing_median": dict(zip(tk, np.median(data["timing"], 0).tolist())),
        "kernel_evals_per_ensemble": float(np.median(data["kernel_evals_cum"][:, -1, 4])),
        "kernel_evals_cum_median": np.median(data["kernel_evals_cum"], 0).tolist(),
        "wall_cum_median": np.median(data["wall_cum"], 0).tolist(),
        "peak_gpu_bytes_max": int(data["peak_gpu_bytes"].max()),
        "peak_rss_kb_max": int(data["peak_rss_kb"].max()),
    }
    return summ


def _save_chunk(path: pathlib.Path, results: list, reps: list) -> None:
    keys = [k for k in results[0] if k not in ("timing",)]
    arrs = {k: np.stack([np.asarray(r[k]) for r in results]) for k in keys}
    arrs["reps"] = np.array(reps)
    timing_keys = sorted(results[0]["timing"])
    arrs["timing"] = np.array([[r["timing"][tk] for tk in timing_keys] for r in results])
    arrs["timing_keys"] = np.array(timing_keys)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


PROBLEM_KEYS = (
    "seed_namespace",
    "model",
    "kernel",
    "settings",
    "t_max_threshold",
    "t_max_cap",
    "checkpoints",
    "m_grid",
    "n_dirs",
    "J_grid",
    "N_grid",
    "N_pi",
    "landmarks",
    "operator",
    "budget",
)


def problem_fingerprint(cfg: DictConfig) -> str:
    """sha256 of the problem-defining config subtree (independent of phase/run id/range/device)."""
    sub = {
        k: OmegaConf.to_container(cfg[k], resolve=True) if OmegaConf.is_config(cfg[k]) else cfg[k]
        for k in PROBLEM_KEYS
    }
    blas = {
        v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    return hashlib.sha256(
        json.dumps({"cfg": sub, "blas": blas}, sort_keys=True, default=str).encode()
    ).hexdigest()


def cached_problem(cfg: DictConfig, device) -> tuple[Problem, dict]:
    """Load or build a CPU setup cache.

    The cache must match the problem configuration and BLAS thread settings.
    Use setup_only=true to build it before running GPU ensembles.
    """
    import pickle

    path = cfg.get("setup_cache", None)
    fp = problem_fingerprint(cfg)
    if path is not None and pathlib.Path(str(path)).exists():
        blob = pickle.loads(pathlib.Path(str(path)).read_bytes())
        if blob["fingerprint"] != fp:
            raise ValueError(
                f"setup cache {path} fingerprint {blob['fingerprint'][:12]} != config {fp[:12]}"
            )
        meta = dict(blob["meta"])
        meta["setup_cache"] = {"path": str(path), "fingerprint": fp}
        return blob["problem"], meta
    t0 = time.perf_counter()
    problem, meta = build_concrete_problem(cfg, device)
    meta["setup_seconds"] = time.perf_counter() - t0
    if path is not None:
        tmp = pathlib.Path(str(path) + f".tmp{os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pickle.dumps({"fingerprint": fp, "problem": problem, "meta": meta}))
        os.replace(tmp, str(path))
        meta["setup_cache"] = {"path": str(path), "fingerprint": fp}
    return problem, meta


def record_resume(run: RunDir, problem_fp: str) -> None:
    """Record the command, code version, device and saved chunks when resuming."""
    import socket
    import sys

    from mcmc.paperexp.runio import git_state

    rec = {
        **git_state(),
        "resumed_utc": utc_now(),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "problem_fingerprint": problem_fp,
        "chunks_present_at_resume": sorted(p.name for p in run.path.glob("chunk_*.npz")),
    }
    with open(run.path / "RESUMES.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def run_study(cfg: DictConfig) -> RunDir | None:
    phase, run_id = str(cfg.phase), cfg.run_id
    if bool(cfg.get("setup_only", False)):
        if cfg.get("setup_cache", None) is None:
            raise ValueError("setup_only requires +setup_cache=<path>")
        problem, meta = cached_problem(cfg, torch.device("cpu"))
        log.info(
            f"setup cached at {cfg.setup_cache} (fingerprint {meta['setup_cache']['fingerprint'][:12]})"
        )
        return None
    if run_id is None:
        raise ValueError("run_id is required")
    pcfg = cfg.phases[phase]
    n_ens, rep0, chunk = int(pcfg.n_ensembles), int(pcfg.rep_offset), int(cfg.chunk_size)
    # +ens_range=[a,b] splits a phase across processes without changing replicate seeds.
    rng = cfg.get("ens_range", None)
    start, stop = (0, n_ens) if rng is None else (int(rng[0]), int(rng[1]))
    if not (0 <= start < stop <= n_ens) or start % chunk or (stop % chunk and stop != n_ens):
        raise ValueError(
            f"ens_range [{start},{stop}) invalid for n_ensembles={n_ens}, chunk={chunk}"
        )
    if phase in ("main", "stress") and cfg.kernel.ell_frozen is None:
        raise ValueError("main/stress phases require a frozen bandwidth (kernel.ell_frozen)")
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = OmegaConf.to_container(cfg, resolve=True)
    config_text = OmegaConf.to_yaml(OmegaConf.create(frozen))
    if bool(cfg.resume):
        run = RunDir.resume(str(cfg.study), phase, str(run_id))
        record_resume(run, problem_fingerprint(cfg))
    else:
        run = RunDir.create(
            str(cfg.study),
            phase,
            str(run_id),
            config_text,
            requested_units=stop - start,
            allow_dirty=phase in ("smoke", "pilot"),
        )
    try:
        problem, meta = cached_problem(cfg, device)
        meta["rep_range"] = [rep0, rep0 + n_ens]
        meta["ens_range"] = [start, stop]
        # BLAS thread counts affect landmark rounding; pooled runs must use the same counts.
        meta["blas_threads"] = {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        meta["budget"] = problem.budget.to_dict()
        if not (run.path / "setup.npz").exists():
            np.savez_compressed(run.path / "setup.npz", **problem_arrays(problem))
            (run.path / "setup.json").write_text(json.dumps(meta, indent=2, default=str))
        log.info(
            f"setup done in {meta.get('setup_seconds', 0.0):.1f}s; ell={meta['ell']:.6f}; truth[:,0]={problem.truth[:, 0]}"
        )
        done = 0
        for c0 in range(start, stop, chunk):
            path = run.path / f"chunk_{c0:05d}.npz"
            c1 = min(stop, c0 + chunk)
            if path.exists():
                done += c1 - c0
                run.update(completed_units=done)
                continue
            results, reps = [], []
            for e in range(c0, c1):
                rep = rep0 + e
                results.append(run_ensemble(problem, rep, str(cfg.seed_namespace), device))
                reps.append(rep)
                log.info(f"ensemble {e} (rep {rep}) wall {results[-1]['wall']:.2f}s")
            _save_chunk(path, results, reps)
            done += c1 - c0
            run.update(completed_units=done)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return run


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_r3_concrete", version_base=None
)
def main(cfg: DictConfig) -> None:
    # Setup pickles need the importable module name.
    from mcmc.experiments import exp_r3_concrete as mod

    run = mod.run_study(cfg)
    if run is not None:
        log.info(f"{utc_now()} run finished: {run.path}")


if __name__ == "__main__":
    main()
