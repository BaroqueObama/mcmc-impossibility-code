"""Anytime MMD bounds for the synthetic experiments (Section 6).

Each repetition independently samples target references, chain replicas and
operator blocks. Replica and operator budgets are shared across checkpoints
and the nested sample-size grids. All bound calculations use float64.

With B = sqrt(H^2 + U^2), H contains the monitored discrepancy and the
replica/target error radii. The unmonitored terms are:
    B3:     R sqrt(lambda_{m+1}(O_{0,+}) + e_0)
    B4same: R sqrt(||P_perp O_{t,+} P_perp|| + e_t), using the time-zero span
    B4own:  R sqrt(lambda_{m+1}(O_{t,+}) + e_t).
Here e_t includes operator estimation and rank truncation. NO3/NO4 use only
the leading eigenvalue; the direct bound uses the full empirical discrepancy.

Separate reference draws give truth intervals for coverage checks. Variants
using a high-budget reference operator are plug-in comparisons, without
finite-sample guarantees. Reference intervals that straddle a bound are
classified as unresolved.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from mcmc.paperexp import bounds as bd
from mcmc.paperexp import operators as op
from mcmc.paperexp import reference as rf
from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import kernel_sum
from mcmc.paperexp.rng import seed_for, torch_gen

F64 = torch.float64
KERNEL_STAGES = ("target", "replicas", "operator", "direct", "reference")


def r3_schedule(t_max: int, factor: int = 4, floor: int = 64, cap: int = 8192) -> list[int]:
    """{0, 1, 2, 4, ..., T3}, T3 = min(cap, max(floor, factor * t_max))."""
    if t_max < 1 or t_max & (t_max - 1):
        raise ValueError(f"t_max must be a power of two >= 1, got {t_max}")
    T3 = min(cap, max(floor, factor * t_max))
    if T3 & (T3 - 1):
        raise ValueError(f"T3={T3} is not a power of two")
    return [0] + [2**j for j in range(int(math.log2(T3)) + 1)]


class CountingKernel:
    """Count kernel entries by stage.

    For RBF kernels, fused=True supplies AnisoRBF.rowsum with the same
    lengthscales. bounds._rowsum uses this fused float64 reduction when present.
    """

    def __init__(self, kernel, fused: bool = False):
        self.kernel = kernel
        self.counts: dict[str, int] = {}
        self._stage = "other"
        if fused:
            if getattr(kernel, "kind", None) != "rbf":
                raise ValueError(
                    f"the fused row sum exists only for the RBF kernel, not {kernel!r}"
                )
            from mcmc.paperexp.gaussian_linear import AnisoRBF

            self._fused = AnisoRBF(kernel.scales, kernel.ell)
            self.rowsum = self._fused_rowsum

    def _fused_rowsum(self, X, Y):
        self.counts[self._stage] = self.counts.get(self._stage, 0) + X.shape[0] * Y.shape[0]
        return self._fused.rowsum(X, Y)

    def set_stage(self, name: str) -> None:
        self._stage = name

    def __call__(self, X, Y, compute_dtype=F64):
        self.counts[self._stage] = self.counts.get(self._stage, 0) + X.shape[0] * Y.shape[0]
        return self.kernel(X, Y, compute_dtype=compute_dtype)

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def prefix_means(vals: torch.Tensor, J_grid) -> torch.Tensor:
    """Means of vals[:J] for each J, using a separate reduction per prefix.

    CUDA prefix scans can combine sums in different orders across runs;
    separate reductions keep these values reproducible.
    """
    return torch.stack([vals[:J].sum() / J for J in J_grid])


def paired_kernel(kernel, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """k(X_i, Y_i) row by row (n,), float64, exact differences (no GEMM expansion)."""
    kernel._check(X, "X")
    kernel._check(Y, "Y")
    if X.shape != Y.shape:
        raise ValueError(f"paired shapes {tuple(X.shape)} vs {tuple(Y.shape)}")
    diff = (X - Y) * kernel._inv(X.device)
    return kernel.from_sqdist((diff * diff).sum(1))


# Batched transitions
class MHStep:
    """MHSampler transition with cached log density and gradient."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.d = sampler.d
        self.grad_per_eval = 1 if sampler.kind == "mala" else 0

    def init(self, x):
        lp, g = self.sampler._eval(x)
        if not torch.isfinite(lp).all():
            raise FloatingPointError("initial states have non-finite log density")
        return (lp, g)

    def transition(self, x, cache, xi, u):
        x_new, lp, g, acc, bad = self.sampler._transition(x, cache[0], cache[1], xi, u)
        return x_new, (lp, g), acc, bad

    def to_dict(self) -> dict:
        return self.sampler.to_dict()


class LinearGaussianStep:
    """X' = A X + L xi; stationary N(0, C) when A C A^T + L L^T = C."""

    grad_per_eval = 0

    def __init__(self, A: np.ndarray, L: np.ndarray):
        A, L = np.asarray(A, np.float64), np.asarray(L, np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or L.shape != A.shape:
            raise ValueError(f"A {A.shape}, L {L.shape}")
        self.A_np, self.L_np = A, L
        self.d = A.shape[0]
        self._dev: dict[str, tuple] = {}

    def _p(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = (
                torch.as_tensor(self.A_np.T.copy(), dtype=F64, device=device),
                torch.as_tensor(self.L_np.T.copy(), dtype=F64, device=device),
            )
        return self._dev[key]

    def init(self, x):
        return None

    def transition(self, x, cache, xi, u):
        At, Lt = self._p(x.device)
        x_new = x @ At + xi @ Lt
        acc = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        bad = ~torch.isfinite(x_new).all(1)
        return x_new, None, acc, bad

    def to_dict(self) -> dict:
        return {"kind": "linear_gaussian", "A": self.A_np.tolist(), "L": self.L_np.tolist()}


class GroupedChains:
    """Advance independent chain groups in one batch with separate generators.

    Each group draws xi (n_g, d), then u (n_g,) per step. A single group
    therefore uses the same random stream as MHSampler.run.
    """

    def __init__(self, step, groups: list[tuple[str, torch.Tensor, torch.Generator]]):
        if not groups:
            raise ValueError("no chain groups")
        self.step = step
        self.names = [g[0] for g in groups]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate group names {self.names}")
        dev = groups[0][1].device
        self.slices, self.gens, i0 = {}, [], 0
        for name, x0, gen in groups:
            if x0.dtype != F64 or x0.ndim != 2 or x0.shape[1] != step.d or x0.device != dev:
                raise ValueError(f"group {name}: x0 {tuple(x0.shape)} {x0.dtype} {x0.device}")
            if torch.device(gen.device).type != dev.type:
                raise ValueError(f"group {name}: generator on {gen.device}, states on {dev}")
            self.slices[name] = slice(i0, i0 + x0.shape[0])
            self.gens.append((x0.shape[0], gen))
            i0 += x0.shape[0]
        self.x = torch.cat([g[1] for g in groups], 0)
        self.cache = step.init(self.x)
        self.t = 0
        self.acc_chain = torch.zeros(self.x.shape[0], dtype=torch.int64, device=dev)
        self._bad = torch.zeros((), dtype=torch.int64, device=dev)

    def advance_to(self, t_new: int) -> None:
        if t_new < self.t:
            raise ValueError(f"cannot go back from t={self.t} to {t_new}")
        d = self.step.d
        while self.t < t_new:
            xis, us = [], []
            for n, gen in self.gens:
                xis.append(torch.randn((n, d), generator=gen, dtype=F64, device=gen.device))
                us.append(torch.rand((n,), generator=gen, dtype=F64, device=gen.device))
            xi = xis[0] if len(xis) == 1 else torch.cat(xis, 0)
            u = us[0] if len(us) == 1 else torch.cat(us, 0)
            self.x, self.cache, acc, bad = self.step.transition(self.x, self.cache, xi, u)
            self.acc_chain += acc
            self._bad += bad.sum()
            self.t += 1
        nbad = int(self._bad.item())
        if nbad:
            raise FloatingPointError(
                f"{nbad} proposals with NaN/+inf density or ratio by t={self.t}"
            )

    def get(self, name: str) -> torch.Tensor:
        return self.x[self.slices[name]]

    def n(self, name: str) -> int:
        s = self.slices[name]
        return s.stop - s.start

    def accept_rate(self, name: str) -> float:
        if self.t == 0:
            return float("nan")
        return float(self.acc_chain[self.slices[name]].sum().item()) / (self.n(name) * self.t)


def sample_init_counted(target, n: int, eps: float, gen: torch.Generator):
    """Target.sample_init with the same random stream, also counting rejection proposals."""
    if not 0.0 <= float(eps) <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    chunks, got, proposals = [], 0, 0
    while got < n:
        m = int(math.ceil((n - got) * (1.0 + eps) * 1.15)) + 64
        prop = target.sample(m, gen)
        u = torch.rand((m,), generator=gen, dtype=F64, device=gen.device)
        acc = u * (1.0 + eps) < target.init_density_ratio(prop, eps)
        chunks.append(prop[acc])
        got += int(acc.sum().item())
        proposals += m
    out = torch.cat(chunks, 0)[:n]
    assert out.shape == (n, target.d)
    return out, proposals


def draw_init(problem, n: int, gen: torch.Generator):
    """Draw from the initial law; return states (n, d) and target proposal count."""
    tgt = problem.target
    kind = problem.init_kind
    if kind == "known_radius":
        return sample_init_counted(tgt, n, problem.eps, gen)
    m = torch.as_tensor(np.asarray(tgt.mean(), np.float64), dtype=F64, device=gen.device)
    if kind == "point":
        s = np.asarray(problem.init_scales, np.float64)
        if s.shape != (tgt.d,) or not (s > 0).all():
            raise ValueError(f"point start needs positive init_scales of shape ({tgt.d},), got {s}")
        j = int(np.argmax(s))  # first index on ties
        x = m.clone()
        x[j] += problem.init_scale * float(s[j])
        return x[None, :].repeat(n, 1).contiguous(), 0
    if kind == "dispersed":
        x = tgt.sample(n, gen)
        return m[None, :] + problem.init_scale * (x - m[None, :]), n
    raise ValueError(f"unknown init kind {kind!r}")


def init_chi2(target, kind: str, scale: float, eps: float) -> dict:
    """chi^2(pi_0 || pi) of an initial law (R = sqrt(chi^2) is the (A3) radius), with its argument.

    known_radius: d pi0/d pi = 1 + eps(2 1_A - 1), pi(A) = 1/2, so chi^2 = eps^2.
    point: a point mass has no density w.r.t. pi: chi^2 = infinity.
    dispersed (law of m + a (X - m), X ~ pi, m = mean; every target here has m as its centre):
      gauss N(m, C): q = N(m, a^2 C), q^2/pi ~ exp(-(1/a^2 - 1/2) |x|_C^2) is integrable iff a^2 < 2;
      mixture (components N(mu_i, I)): q has components N(., a^2 I), same tail exponent: iff a^2 < 2;
      funnel: y' = a y ~ N(0, a^2 sigma_y^2) alone has chi^2(q_y || pi_y) = infinity for a^2 >= 2, and
        chi^2 of a marginal is <= chi^2 of the joint (data processing);
      student_t (standard t_nu, mean 0): q/pi(x) = a^-d ((1 + r^2/nu)/(1 + r^2/(a^2 nu)))^((nu+d)/2)
        <= a^nu is bounded, so chi^2 = E_pi[(q/pi)^2] - 1 is finite; computed by adaptive quadrature over
        f = r^2/d ~ F(d, nu) (scipy.integrate.quad) and cross-checked over r with the t radial density.
    """
    a = float(scale)
    if kind == "known_radius":
        return {"chi2": float(eps) ** 2, "R": float(eps), "finite": True, "argument": "eps^2"}
    if kind == "point":
        return {"chi2": math.inf, "R": math.inf, "finite": False, "argument": "point mass"}
    if kind != "dispersed":
        raise ValueError(f"unknown init kind {kind!r}")
    fam = target.family
    if fam in ("gauss", "mixture", "funnel"):
        if a * a < 2.0:
            raise NotImplementedError(
                f"finite chi^2 for dispersed {fam} with a^2 < 2 not implemented"
            )
        why = {
            "gauss": "Gaussian tails: q^2/pi not integrable for a^2 >= 2",
            "mixture": "Gaussian component tails: q^2/pi not integrable for a^2 >= 2",
            "funnel": "y-marginal N(0, a^2 sigma_y^2) vs N(0, sigma_y^2) has chi^2 = inf for a^2 >= 2",
        }[fam]
        return {"chi2": math.inf, "R": math.inf, "finite": False, "argument": why}
    if fam != "student_t":
        raise ValueError(f"no chi^2 rule for family {fam!r}")
    from scipy import integrate, special, stats

    d, nu = int(target.d), float(target.nu)
    if np.abs(np.asarray(target.mean(), np.float64)).max() != 0.0:
        raise ValueError("the Student-t rule assumes mean 0")

    def log_ratio(r2):
        return -d * math.log(a) + 0.5 * (nu + d) * (
            math.log1p(r2 / nu) - math.log1p(r2 / (a * a * nu))
        )

    fdist = stats.f(d, nu)
    e_f, err_f = integrate.quad(
        lambda f: math.exp(2.0 * log_ratio(d * f)) * fdist.pdf(f),
        0.0,
        np.inf,
        epsabs=0.0,
        epsrel=1e-12,
        limit=1000,
    )
    # Check the F-distribution integral against the radial parametrization.
    log_c = (
        math.lgamma((nu + d) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * d * math.log(nu * math.pi)
        + math.log(2.0)
        + 0.5 * d * math.log(math.pi)
        - special.gammaln(d / 2)
    )
    e_r, err_r = integrate.quad(
        lambda r: (
            math.exp(
                2.0 * log_ratio(r * r)
                + log_c
                + (d - 1) * math.log(r)
                - 0.5 * (nu + d) * math.log1p(r * r / nu)
            )
            if r > 0
            else 0.0
        ),
        0.0,
        np.inf,
        epsabs=0.0,
        epsrel=1e-12,
        limit=1000,
    )
    if abs(e_f - e_r) > 1e-9 * e_f:
        raise FloatingPointError(f"chi^2 quadratures disagree: {e_f} vs {e_r}")
    chi2 = e_f - 1.0
    return {
        "chi2": chi2,
        "R": math.sqrt(chi2),
        "finite": True,
        "sup_ratio": a**nu,
        "argument": "bounded ratio <= a^nu; quadrature over F(d, nu) and over r agree",
        "quad": {"over_F": [e_f, err_f], "over_r": [e_r, err_r]},
    }


def certified_span_gram(kernel, points: torch.Tensor, coefs: torch.Tensor, block: int):
    """Span Gram matrix S = fl(C^T K C) and its entrywise rounding bound ES.

    Uses the certified_span_overlap bound in operators, with K evaluated in
    row blocks. Kernel values are treated as input data. span_gram_defect(S, ES)
    then bounds ||V^* V - I||_2 and gives lambda_min(V^* V) >= 1 - omega.
    """
    n, m = coefs.shape
    if points.shape[0] != n:
        raise ValueError(f"points {tuple(points.shape)} vs coefs {tuple(coefs.shape)}")
    As = coefs.abs()
    T2 = torch.empty(n, m, dtype=F64, device=coefs.device)
    A2 = torch.empty(n, m, dtype=F64, device=coefs.device)
    for i0 in range(0, n, block):
        Kb = kernel(points[i0 : i0 + block], points)
        if not torch.isfinite(Kb).all():
            raise FloatingPointError("non-finite span kernel block")
        T2[i0 : i0 + block] = Kb @ coefs
        A2[i0 : i0 + block] = Kb.abs() @ As
        del Kb
    S = coefs.T @ T2
    infl = 1.0 + 2.0 * op._gamma(n + 2)
    ES = (op._gamma(n) * (As.T @ A2) + op._gamma(n) * (As.T @ T2.abs())) * infl
    ES = ES + (2 * n + 4) * op.TINY
    return S, ES


@dataclass
class SuiteProblem:
    namespace: str  # independent random streams per configuration
    config_id: str
    target: object
    kernel: object
    kernel_tag: str
    step: object  # MHStep | LinearGaussianStep
    eps: float
    t_grid: list
    J_grid: list
    N_grid: list
    m_grid: list
    n_dirs: int
    N_pi: int
    budget: bd.BudgetAllocation
    embedding: object = None  # known mu_pi (RBF) or None
    op_rtol: float = 1e-11
    head: int = 1024
    kernel_chunk: int = 1024
    refop: dict | None = None  # plug-in reference-operator spans/values (from the reference stage)
    op_method: str = "fp64"  # operators.estimate_operator method: fp64 | fp32-enclosed
    # Retry failed eigensolves with this method; record each retry in op_fallback.
    op_fallback: str | None = None
    n_refined: int = 128  # enclosed methods only
    fused_rowsum: bool = False  # fp64 fused RBF row sums (gaussian_linear.AnisoRBF.rowsum)
    # known_radius satisfies (A3) with R = eps. Point and dispersed starts
    # use that radius without verifying it; their R-dependent bounds are conditional.
    init_kind: str = "known_radius"
    init_scale: float = 3.0
    init_scales: np.ndarray | None = None  # frozen coordinate scales (point start)
    # Include Gram rounding bounds in gram_cert_* diagnostics when enabled.
    certify_gram: bool = False
    b_t: np.ndarray = field(init=False)
    b_pi: float = field(init=False)
    e_tN: np.ndarray = field(init=False)

    def __post_init__(self):
        assert self.t_grid == sorted(self.t_grid) and self.t_grid[0] == 0, self.t_grid
        assert self.J_grid == sorted(self.J_grid) and self.N_grid == sorted(self.N_grid)
        assert self.m_grid == sorted(self.m_grid) and self.n_dirs >= self.m_grid[-1]
        assert all(J % self.kernel_chunk == 0 for J in self.J_grid), "J grid vs kernel chunk"
        if self.N_pi % self.kernel_chunk:
            raise ValueError(f"N_pi={self.N_pi} must be a multiple of {self.kernel_chunk}")
        kappa = math.sqrt(self.kernel.kappa2)
        bud = self.budget
        if set(bud.components) != {"replicas", "target", "operator"}:
            raise ValueError(f"budget components {sorted(bud.components)}")
        if "replicas" not in bud.grids:
            bud.declare_grid("replicas", {"t": list(self.t_grid), "J": list(self.J_grid)})
            bud.declare_grid("operator", {"t": list(self.t_grid), "N": list(self.N_grid)})
        self.b_t = np.array(
            [
                bd.replica_radius(kappa, J, bud.beta("replicas"), bud.weight("replicas"))
                for J in self.J_grid
            ]
        )
        self.b_pi = bd.target_radius(kappa, self.N_pi, bud.beta("target"))
        self.e_tN = np.array(
            [
                [
                    op.operator_radius(
                        self.kernel.kappa2, N, bud.beta("operator"), bud.weight("operator")
                    )
                    for N in self.N_grid
                ]
                for _ in self.t_grid
            ]
        )

    @property
    def r_init(self) -> float:
        return self.target.r_init(self.eps)

    @property
    def refs(self) -> list[str]:
        return ["sampled", "known"] if self.embedding is not None else ["sampled"]

    def stage_seed(self, stage: str, rep: int) -> int:
        return seed_for(self.namespace, stage, rep)

    def budget_check(self) -> dict:
        """Check component budgets and per-grid allocation weights."""
        bud = self.budget
        comp_sum = sum(bud.components.values())
        out = {
            "total": bud.total,
            "components": dict(bud.components),
            "components_sum": comp_sum,
            "sum_equals_total": abs(comp_sum - bud.total) <= 1e-15,
        }
        for g in ("replicas", "operator"):
            out[f"{g}_cells"] = bud.grids[g]["n_cells"]
            out[f"{g}_weight_sum"] = bud.weight(g) * bud.grids[g]["n_cells"]
        return out

    def setup_arrays(self) -> dict:
        return {
            "t_grid": np.array(self.t_grid),
            "J_grid": np.array(self.J_grid),
            "N_grid": np.array(self.N_grid),
            "m_grid": np.array(self.m_grid),
            "r_init": np.array(self.r_init),
            "b_t": self.b_t,
            "b_pi": np.array(self.b_pi),
            "e_tN": self.e_tN,
            "N_pi": np.array(self.N_pi),
            "refs": np.array(self.refs),
        }


# Independent reference estimates
def _block_vstat(k, X: torch.Tensor, mem_bytes: int) -> float:
    """sum_ij k(X_i, X_j) (fp64, blocked; fused row sums when the kernel provides them)."""
    f = getattr(k, "rowsum", None)
    if f is None:
        return float(kernel_sum(k, X, X, mem_bytes=mem_bytes).item())
    n = X.shape[0]
    cols = max(1, min(n, mem_bytes // (8 * 4 * n)))
    total = torch.zeros((), dtype=F64, device=X.device)
    for j0 in range(0, n, cols):
        total += f(X, X[j0 : j0 + cols]).sum()
    return float(total.item())


def feature_operator_accumulate(feats, Y, Z, Yp, Zp, chunk: int):
    """Sym. feature-space O = (D1^T D2 + D2^T D1)/(4N) and its two independent halves, plus the
    full-RKHS trace estimate mean <D1, D2>/2 and the feature trace (for the omitted-trace tau)."""
    N = Y.shape[0]
    if N % 2:
        raise ValueError("N must be even for the split-half operator")
    r = feats.rank
    acc = [torch.zeros(r, r, dtype=F64, device=Y.device) for _ in range(2)]
    half = N // 2
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        for h, (a, b) in enumerate(((s, min(e, half)), (max(s, half), e))):
            if a >= b:
                continue
            D1 = feats(Y[a:b]) - feats(Yp[a:b])
            D2 = feats(Z[a:b]) - feats(Zp[a:b])
            acc[h] += D1.T @ D2
    k = feats.kernel
    full_tr = (
        paired_kernel(k, Y, Z)
        - paired_kernel(k, Y, Zp)
        - paired_kernel(k, Yp, Z)
        + paired_kernel(k, Yp, Zp)
    ).mean() / 2.0
    S = (acc[0] + acc[1] + acc[0].T + acc[1].T) / (4.0 * N)
    Sa = (acc[0] + acc[0].T) / (4.0 * half)
    Sb = (acc[1] + acc[1].T) / (4.0 * half)
    return S, Sa, Sb, float(full_tr.item())


def plugin_from_operators(S_list: list[torch.Tensor], m_grid, n_dirs: int) -> dict:
    """Plug-in leading spaces (feature coordinates) and tails from feature operators S_t (r x r)."""
    T = len(S_list)
    r = S_list[0].shape[0]
    if n_dirs + 1 > r:
        raise ValueError(f"feature rank {r} < n_dirs + 1 = {n_dirs + 1}")
    lam = np.zeros((T, r))
    W = []
    for ti, S in enumerate(S_list):
        w, V = torch.linalg.eigh(0.5 * (S + S.T))
        lam[ti] = w.flip(0).cpu().numpy()
        W.append(V.flip(1))
    nm = len(m_grid)
    own = np.zeros((T, nm))
    same = np.zeros((T, nm))
    W0 = W[0]
    for ti in range(T):
        lp = np.clip(lam[ti], 0.0, None)
        S_plus = (W[ti] * torch.as_tensor(lp, dtype=F64, device=W[ti].device)) @ W[ti].T
        for mi, m in enumerate(m_grid):
            own[ti, mi] = lp[m]
            V0 = W0[:, :m]
            Pp = torch.eye(r, dtype=F64, device=V0.device) - V0 @ V0.T
            M = Pp @ S_plus @ Pp
            same[ti, mi] = float(torch.linalg.eigvalsh(0.5 * (M + M.T))[-1].clamp_min(0.0))
    return {
        "lam": lam,
        "lam1": np.clip(lam[:, 0], 0.0, None),
        "own": own,
        "same": same,
        "W": torch.stack([w[:, :n_dirs] for w in W]),  # (T, r, n_dirs)
    }


def run_reference(
    problem: SuiteProblem,
    J_ref: int,
    N_refop: int,
    n_landmarks: int,
    beta_ref: float,
    device,
    quad_margin: float,
    N_ref_target: int | None = None,
    landmark_rtol: float = 1e-10,
    mem_bytes: int = 1 << 30,
    feat_chunk: int = 8192,
    log=print,
) -> dict:
    """Reference truth intervals and a plug-in operator for one configuration.

    Uses independent ref-* streams. The estimated reference operator is used
    only in the plug-in comparisons. Returns arrays for saving.
    """
    t_start = time.perf_counter()
    tgt, T = problem.target, len(problem.t_grid)
    k = CountingKernel(problem.kernel, fused=problem.fused_rowsum)
    known = problem.embedding is not None
    if known:
        b_ref_t, b_ref_pi = rf.reference_radii(
            math.sqrt(k.kappa2), J_ref, None, beta_ref, beta_ref, T
        )
        beta_split = {"replicas": beta_ref, "target": 0.0}
    else:
        if N_ref_target is None:
            raise ValueError("a sampled reference target needs N_ref_target")
        b_ref_t, b_ref_pi = rf.reference_radii(
            math.sqrt(k.kappa2), J_ref, N_ref_target, beta_ref / 2, beta_ref / 2, T
        )
        beta_split = {"replicas": beta_ref / 2, "target": beta_ref / 2}
    timing: dict[str, float] = {}

    def tic(name, t0):
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0

    t0 = time.perf_counter()
    X0, n_prop = draw_init(
        problem, J_ref, torch_gen(problem.stage_seed("ref-replica-init", 0), device)
    )
    g_op0 = torch_gen(problem.stage_seed("ref-op-start", 0), device)
    Xs, Xsp = tgt.sample(N_refop, g_op0), tgt.sample(N_refop, g_op0)
    branches = torch.cat([Xs, Xsp, Xs, Xsp], 0)  # Y, Y', Z, Z'
    del Xs, Xsp
    Zl = tgt.sample(n_landmarks, torch_gen(problem.stage_seed("ref-landmarks", 0), device))
    Xref_pi = None
    if not known:
        Xref_pi = tgt.sample(N_ref_target, torch_gen(problem.stage_seed("ref-target", 0), device))
    chains = GroupedChains(
        problem.step,
        [
            ("rep", X0, torch_gen(problem.stage_seed("ref-replica-chain", 0), device)),
            ("op", branches, torch_gen(problem.stage_seed("ref-op-chain", 0), device)),
        ],
    )
    del X0, branches
    tic("sampling", t0)
    t0 = time.perf_counter()
    feats = NystromFeatures(problem.kernel, Zl, rel_cutoff=landmark_rtol)
    whitening_err = feats.whitening_error()
    if whitening_err > 1e-6:
        raise FloatingPointError(f"Nystrom whitening error {whitening_err:.2e}")
    if known:
        mu_norm2 = float(problem.embedding.norm2())
    else:
        k.set_stage("reference")
        mu_norm2 = _block_vstat(k, Xref_pi, mem_bytes) / N_ref_target**2
    tic("setup", t0)

    D = np.zeros(T)
    mmd2_u = np.zeros(T)
    pA = np.zeros(T)
    S_list, Sa_list, Sb_list = [], [], []
    full_tr = np.zeros(T)
    for ti, t in enumerate(problem.t_grid):
        t0 = time.perf_counter()
        chains.advance_to(t)
        tic("sampling", t0)
        Xr = chains.get("rep")
        t0 = time.perf_counter()
        k.set_stage("reference")
        s_vv = _block_vstat(k, Xr, mem_bytes)
        if known:
            cross = float(problem.embedding(Xr).mean().item())
        else:
            # Block both axes to fit the large reference populations.
            cross = float(kernel_sum(k, Xr, Xref_pi, mem_bytes=mem_bytes).item()) / (
                J_ref * N_ref_target
            )
        d2 = s_vv / J_ref**2 - 2.0 * cross + mu_norm2
        if d2 < -1e-12:
            raise FloatingPointError(f"negative reference squared distance {d2} at t={t}")
        D[ti] = math.sqrt(max(d2, 0.0))
        mmd2_u[ti] = (s_vv - J_ref * k.kappa2) / (J_ref * (J_ref - 1)) - 2.0 * cross + mu_norm2
        pA[ti] = float(tgt.event_A(Xr).to(F64).mean().item())
        tic("truth", t0)
        t0 = time.perf_counter()
        Y, Yp, Z, Zp = chains.get("op").split(N_refop)
        S, Sa, Sb, ftr = feature_operator_accumulate(feats, Y, Z, Yp, Zp, feat_chunk)
        S_list.append(S)
        Sa_list.append(Sa)
        Sb_list.append(Sb)
        full_tr[ti] = ftr
        tic("refop", t0)
        log(
            f"reference t={t}: D={D[ti]:.5f} mmd2_u={mmd2_u[ti]:.3e} pA={pA[ti]:.4f} "
            f"tr={ftr:.4e} feat_tr={float(torch.trace(S)):.4e} wall={time.perf_counter() - t_start:.1f}s"
        )
    L, U = rf.truth_intervals(D, np.full(T, b_ref_t), b_ref_pi)
    L = np.maximum(0.0, L - quad_margin)
    U = U + quad_margin
    t0 = time.perf_counter()
    plug = plugin_from_operators(S_list, problem.m_grid, problem.n_dirs)
    plug_a = plugin_from_operators(Sa_list, problem.m_grid, problem.n_dirs)
    plug_b = plugin_from_operators(Sb_list, problem.m_grid, problem.n_dirs)
    feat_tr = np.array([float(torch.trace(S).item()) for S in S_list])
    tau = full_tr - feat_tr
    B = feats.B
    coefs_t = torch.stack([B @ plug["W"][ti] for ti in range(T)])  # (T, M, n_dirs)
    # Check the plug-in spans in the landmark RKHS metric.
    Kzz = problem.kernel(Zl, Zl)
    orth = 0.0
    for ti in range(T):
        G = coefs_t[ti].T @ Kzz @ coefs_t[ti]
        orth = max(orth, float((G - torch.eye(G.shape[0], dtype=F64, device=G.device)).abs().max()))
    if orth > 1e-6:
        raise FloatingPointError(f"plug-in spans not orthonormal (err {orth:.2e})")
    tic("plugin", t0)
    out = {
        "t_grid": np.array(problem.t_grid),
        "D": D,
        "mmd2_u": mmd2_u,
        "L": L,
        "U": U,
        "b_ref_t": np.array(b_ref_t),
        "b_ref_pi": np.array(b_ref_pi),
        "quad_margin": np.array(quad_margin),
        "beta_ref": np.array(beta_ref),
        "beta_ref_replicas": np.array(beta_split["replicas"]),
        "beta_ref_target": np.array(beta_split["target"]),
        "J_ref": np.array(J_ref),
        "N_ref_target": np.array(-1 if N_ref_target is None else N_ref_target),
        "pA": pA,
        "init_proposals": np.array(n_prop),
        "accept_rep": np.array(chains.accept_rate("rep")),
        "accept_op": np.array(chains.accept_rate("op")),
        "mu_norm2": np.array(mu_norm2),
        # Reference operator for the plug-in comparisons.
        "refop_N": np.array(N_refop),
        "refop_M": np.array(n_landmarks),
        "refop_rank": np.array(feats.rank),
        "refop_whitening_err": np.array(whitening_err),
        "refop_span_orth_err": np.array(orth),
        "landmarks": Zl.cpu().numpy(),
        "coefs_t": coefs_t.cpu().numpy(),
        "lam": plug["lam"],
        "lam1": plug["lam1"],
        "own": plug["own"],
        "same": plug["same"],
        "own_half_a": plug_a["own"],
        "own_half_b": plug_b["own"],
        "same_half_a": plug_a["same"],
        "same_half_b": plug_b["same"],
        "lam1_half_a": plug_a["lam1"],
        "lam1_half_b": plug_b["lam1"],
        "full_tr": full_tr,
        "feat_tr": feat_tr,
        "tau": tau,
        "kernel_evals": np.array([k.counts.get(s, 0) for s in KERNEL_STAGES], dtype=np.float64),
        "timing_keys": np.array(sorted(timing)),
        "timing": np.array([timing[x] for x in sorted(timing)]),
        "wall": np.array(time.perf_counter() - t_start),
        "peak_gpu_bytes": np.array(
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
    }
    return out


def load_refop(ref: dict, device) -> dict:
    """Plug-in spans/values (from a saved reference dict) on ``device`` for the ensemble stage."""
    return {
        "landmarks": torch.as_tensor(ref["landmarks"], dtype=F64, device=device),
        "coefs_t": torch.as_tensor(ref["coefs_t"], dtype=F64, device=device),
        "own": np.asarray(ref["own"], np.float64),
        "same": np.asarray(ref["same"], np.float64),
        "lam1": np.asarray(ref["lam1"], np.float64),
        "tau": np.asarray(ref["tau"], np.float64),
    }


# Independent repetitions
class _Ensemble:
    """One repetition with independent generators for each sampling stage."""

    def __init__(self, problem: SuiteProblem, rep: int, phase: str, device, analytic=None):
        if problem.refop is None:
            raise ValueError("the ensemble stage needs the reference-operator plug-in spans")
        self.p, self.rep, self.device, self.analytic = problem, rep, device, analytic
        p = problem
        T = len(p.t_grid)
        nJ, nN, nm = len(p.J_grid), len(p.N_grid), len(p.m_grid)
        self.Jmax, self.Nmax, nd = p.J_grid[-1], p.N_grid[-1], p.n_dirs
        self.refs = p.refs
        self.k = CountingKernel(p.kernel, fused=p.fused_rowsum)
        self.timing: dict[str, float] = {}

        def seed(stage):
            return p.stage_seed(f"{phase}-ens-{stage}", rep)

        t0 = time.perf_counter()
        tgt = p.target
        self.Xpi = tgt.sample(p.N_pi, torch_gen(seed("tref"), device))
        X0, n_prop = draw_init(p, self.Jmax, torch_gen(seed("rep-init"), device))
        g_ops = torch_gen(seed("op-start"), device)
        Xs, Xsp = tgt.sample(self.Nmax, g_ops), tgt.sample(self.Nmax, g_ops)
        self.E0 = torch.cat([Xs, Xsp, Xs, Xsp], 0)  # Y, Y', Z, Z' at t = 0
        self.groups = [
            (f"rep{rep}", X0, torch_gen(seed("rep-chain"), device)),
            (f"op{rep}", self.E0.clone(), torch_gen(seed("op-chain"), device)),
        ]
        self._tic("sampling", t0)

        # Reuse target means at landmarks and time-zero endpoints.
        Zl = p.refop["landmarks"]
        self.M = Zl.shape[0]
        self.nP = self.M + 8 * self.Nmax
        t0 = time.perf_counter()
        self.k.set_stage("target")
        self.fixed_pts = torch.cat([Zl, self.E0], 0)
        self.tgt_fixed = bd.kernel_prefix_means(
            self.k, self.fixed_pts, self.Xpi, [p.N_pi], chunk=p.kernel_chunk
        )[0]
        self.norm2_pi_hat = _block_vstat(self.k, self.Xpi, 1 << 30) / p.N_pi**2
        self.known_fixed = p.embedding(self.fixed_pts) if "known" in self.refs else None
        self.n2_pi = float(p.embedding.norm2()) if "known" in self.refs else None
        self._tic("target", t0)

        self.F = F = 2 * nN + 2  # families: V3[N..], V4own[N..], Vref0, Vref_t
        nref = len(self.refs)
        self.out = {
            "proj": np.zeros((T, nJ, F, nm, nref)),
            "direct": np.zeros((T, nJ, nref)),
            "U3_unit": np.zeros((nN, nm)),
            "NO3_unit": np.zeros(nN),
            "U4same_unit": np.zeros((T, nN, nm)),
            "U4own_unit": np.zeros((T, nN, nm)),
            "NO4_unit": np.zeros((T, nN)),
            "e_tot": np.zeros((T, nN)),
            "op_rank": np.zeros((T, nN), dtype=np.int64),
            "op_npos": np.zeros((T, nN), dtype=np.int64),
            "op_delta_trunc": np.zeros((T, nN)),
            "op_orth_err": np.zeros((T, nN)),
            "op_top": np.zeros((T, nN, nd + 1)),
            "op_min_eig": np.zeros((T, nN)),
            "op_trace": np.zeros((T, nN)),
            "op_trace_paired": np.zeros((T, nN)),
            "op_head_slack_max": np.zeros((T, nN)),
            "op_own_check": np.zeros((nN,)),
            "op_kdirs": np.zeros((T, nN), dtype=np.int64),
            "op_fallback": np.zeros((T, nN), dtype=np.int64),
            "op_time": np.zeros((T, nN)),
            "pA_rep": np.zeros(T),
            "accept_rep": np.zeros(T),
            "accept_op": np.zeros(T),
            "kernel_evals_cum": np.zeros((T, len(KERNEL_STAGES))),
            "wall_cum": np.zeros(T),
            "init_proposals": np.array(n_prop),
            "replica_event": np.full((T, nJ), np.nan),
            "trace_true": np.full(T, np.nan),
            "proj_true": np.full((T, F, nm), np.nan),
        }
        if p.certify_gram:
            for key in ("gram_cert_omega", "gram_cert_fro_dev", "gram_cert_fro_es"):
                self.out[key] = np.full((T, nN), np.nan)
        self.V3: dict = {}
        self.V3_cols = torch.zeros(self.nP, nN * nd, dtype=F64, device=device)
        self.ref0_cols = torch.zeros(self.nP, nd, dtype=F64, device=device)
        self.ref0_cols[: self.M] = p.refop["coefs_t"][0]

    def _tic(self, name, t0):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing[name] = self.timing.get(name, 0.0) + time.perf_counter() - t0

    def checkpoint(self, ti: int, t: int, chains: GroupedChains) -> None:
        p, out, k, device = self.p, self.out, self.k, self.device
        nJ, nd, F, M, Nmax, Jmax = len(p.J_grid), p.n_dirs, self.F, self.M, self.Nmax, self.Jmax
        cj, refs, nP = p.kernel_chunk, self.refs, self.nP
        Et = chains.get(f"op{self.rep}")
        Y, Yp, Z, Zp = Et.split(Nmax)
        Xr = chains.get(f"rep{self.rep}")
        out["accept_rep"][ti] = chains.accept_rate(f"rep{self.rep}")
        out["accept_op"][ti] = chains.accept_rate(f"op{self.rep}")
        out["pA_rep"][ti] = float(p.target.event_A(Xr).to(F64).mean().item())

        # estimated operators (nested N)
        nN = len(p.N_grid)
        V4_cols = torch.zeros(nP, nN * nd, dtype=F64, device=device)
        k.set_stage("operator")
        for ni, N in enumerate(p.N_grid):
            t1 = time.perf_counter()
            blocks = (Y[:N], Z[:N], Yp[:N], Zp[:N])
            try:
                est = op.estimate_operator(
                    k, *blocks, rtol=p.op_rtol, method=p.op_method, n_refined=p.n_refined
                )
            except op.EigensolveNotConverged:
                if p.op_fallback is None:
                    raise
                out["op_fallback"][ti, ni] = 1
                est = op.estimate_operator(
                    k, *blocks, rtol=p.op_rtol, method=p.op_fallback, n_refined=p.n_refined
                )
            e = float(p.e_tN[ti, ni]) + est.delta_trunc
            out["e_tot"][ti, ni] = e
            out["op_rank"][ti, ni], out["op_npos"][ti, ni] = est.rank, est.n_pos
            out["op_delta_trunc"][ti, ni] = est.delta_trunc
            out["op_orth_err"][ti, ni] = est.eigvec_orthonormality_error
            out["op_top"][ti, ni] = [est.lambda_plus(j) for j in range(1, nd + 2)]
            out["op_min_eig"][ti, ni] = float(est.eigvals[-1])
            diag = getattr(est, "diagnostics", None)
            out["op_trace"][ti, ni] = (
                float(diag["trace"]) if diag is not None else float(est.eigvals.sum())
            )
            out["op_trace_paired"][ti, ni] = float(
                (
                    paired_kernel(k, Y[:N], Z[:N])
                    - paired_kernel(k, Y[:N], Zp[:N])
                    - paired_kernel(k, Yp[:N], Z[:N])
                    + paired_kernel(k, Yp[:N], Zp[:N])
                ).mean()
                / 2.0
            )
            # Use only positive, orthonormal directions. If kd < m, the bound
            # uses lambda^+_{kd+1}; it is zero (plus enclosure) when kd = n_pos.
            kd = min(nd, est.n_pos, getattr(est, "n_orthonormal", nd))
            if kd < 1:
                raise FloatingPointError(f"no positive operator direction at t={t}, N={N}")
            out["op_kdirs"][ti, ni] = kd
            own = est.top_span(kd, label=f"V4own_t{t}_N{N}")
            if p.certify_gram:  # diagnostic: uncounted kernel stage, outputs gram_cert_*
                k.set_stage("certify")
                S, ES = certified_span_gram(k, own.points, own.coefs, p.kernel_chunk)
                out["gram_cert_omega"][ti, ni] = op.span_gram_defect(S, ES)
                eye = torch.eye(kd, dtype=F64, device=S.device)
                out["gram_cert_fro_dev"][ti, ni] = op._fro(S - eye)
                out["gram_cert_fro_es"][ti, ni] = op._fro(ES)
                del S, ES
                k.set_stage("operator")
            for mi, m in enumerate(p.m_grid):
                out["U4own_unit"][ti, ni, mi] = math.sqrt(est.lambda_plus(min(m, kd) + 1) + e)
            out["NO4_unit"][ti, ni] = math.sqrt(est.lambda_plus(1) + e)
            if ti == 0:
                self.V3[N] = own
                out["U3_unit"][ni] = out["U4own_unit"][0, ni]
                out["NO3_unit"][ni] = out["NO4_unit"][0, ni]
                # own-space check: compressed norm on the own leading span vs lambda_{m+1}
                # (equal for the fp64 path; differs by the enclosure terms for enclosed estimates)
                ms = [min(m, kd) for m in p.m_grid]
                vals = est.compressed_plus_norms(k, own, ms, p.head)
                out["op_own_check"][ni] = max(
                    abs(v - est.lambda_plus(m + 1)) for (v, _), m in zip(vals, ms)
                )
                for b in range(4):
                    rows = slice(M + b * Nmax, M + b * Nmax + N)
                    cols = slice(ni * nd, ni * nd + kd)  # columns beyond kd stay zero
                    self.V3_cols[rows, cols] = own.coefs[b * N : (b + 1) * N]
            V3 = self.V3[N]
            same = est.compressed_plus_norms(k, V3, [min(m, V3.dim) for m in p.m_grid], p.head)
            slack = 0.0
            for mi, (val, info) in enumerate(same):
                slack = max(slack, info["slack_bound"])
                out["U4same_unit"][ti, ni, mi] = math.sqrt(val + e)
            out["op_head_slack_max"][ti, ni] = slack
            for b in range(4):
                rows = slice(M + 4 * Nmax + b * Nmax, M + 4 * Nmax + b * Nmax + N)
                V4_cols[rows, ni * nd : ni * nd + kd] = own.coefs[b * N : (b + 1) * N]
            del est, own
            if device.type == "cuda":
                torch.cuda.synchronize()
            out["op_time"][ti, ni] = time.perf_counter() - t1
        self.timing["operator"] = self.timing.get("operator", 0.0) + float(out["op_time"][ti].sum())
        reft_cols = torch.zeros(nP, nd, dtype=F64, device=device)
        reft_cols[:M] = p.refop["coefs_t"][ti]
        Coef = torch.cat([self.V3_cols, V4_cols, self.ref0_cols, reft_cols], 1)  # (nP, F * nd)
        assert Coef.shape == (nP, F * nd)

        # delta_hat at the point set and projections
        t0 = time.perf_counter()
        P = torch.cat([self.fixed_pts, Et], 0)
        k.set_stage("target")
        tgt_Et = bd.kernel_prefix_means(k, Et, self.Xpi, [p.N_pi], chunk=cj)[0]
        tg = [torch.cat([self.tgt_fixed, tgt_Et])]
        if "known" in refs:
            tg.append(torch.cat([self.known_fixed, p.embedding(Et)]))
        self._tic("target", t0)
        t0 = time.perf_counter()
        k.set_stage("replicas")
        rm = bd.kernel_prefix_means(k, P, Xr, p.J_grid, chunk=cj)  # (nJ, nP)
        for ri in range(len(refs)):
            sq = ((rm - tg[ri][None, :]) @ Coef).view(nJ, F, nd) ** 2
            for mi, m in enumerate(p.m_grid):
                out["proj"][ti, :, :, mi, ri] = sq[:, :, :m].sum(2).sqrt().cpu().numpy()
        if self.analytic is not None:  # closed-form projection for the Gaussian check
            delta_true = self.analytic.mu_t(t, P) - p.embedding(P)
            sq_true = (delta_true @ Coef).view(F, nd) ** 2
            for mi, m in enumerate(p.m_grid):
                out["proj_true"][ti, :, mi] = sq_true[:, :m].sum(1).sqrt().cpu().numpy()
        self._tic("projections", t0)
        # direct bound (the target-target sum is computed once, in __init__)
        t0 = time.perf_counter()
        k.set_stage("direct")
        n2 = bd.kernel_gram_prefix_means(k, Xr, p.J_grid, chunk=cj)
        vals = bd.kernel_prefix_means(k, Xr[:Jmax], self.Xpi, [p.N_pi], chunk=cj)[0]
        cross_s = prefix_means(vals, p.J_grid)
        dd = [n2 - 2.0 * cross_s + self.norm2_pi_hat]
        if "known" in refs:
            dd.append(n2 - 2.0 * prefix_means(p.embedding(Xr[:Jmax]), p.J_grid) + self.n2_pi)
        for ri, d2 in enumerate(dd):
            if float(d2.min()) < -1e-12:
                raise FloatingPointError(f"negative squared distance {float(d2.min())}")
            out["direct"][ti, :, ri] = d2.clamp_min(0.0).sqrt().cpu().numpy()
        self._tic("direct", t0)
        if self.analytic is not None:
            out["replica_event"][ti] = self.analytic.replica_event(t, Xr, p.J_grid, n2)
            out["trace_true"][ti] = self.analytic.trace(t)
        out["kernel_evals_cum"][ti] = [k.counts.get(s, 0) for s in KERNEL_STAGES]
        out["wall_cum"][ti] = sum(self.timing.values())

    def finish(self, chains: GroupedChains, batch_size: int, peak_gpu_bytes: int) -> dict:
        p, out = self.p, self.out
        out["n_logprob_rep"] = np.array(chains.n(f"rep{self.rep}") * (p.t_grid[-1] + 1))
        out["n_logprob_op"] = np.array(chains.n(f"op{self.rep}") * (p.t_grid[-1] + 1))
        out["grad_per_eval"] = np.array(p.step.grad_per_eval)
        out["batch_size"] = np.array(batch_size)
        out["timing"] = self.timing
        out["wall"] = np.array(sum(self.timing.values()))
        out["peak_gpu_bytes"] = np.array(peak_gpu_bytes)
        out["peak_rss_kb"] = np.array(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return out


def run_ensembles(problem: SuiteProblem, reps, phase: str, device, analytic=None) -> list[dict]:
    """Run repetitions with batched sampling and separate checkpoint analysis.

    Each repetition retains its own random streams. Sampling time is shared
    equally; peak GPU memory is reported for the whole batch. analytic can
    supply an ARGaussWitness for checks against closed-form values.
    """
    reps = list(reps)
    if not reps or len(set(reps)) != len(reps):
        raise ValueError(f"reps must be distinct and non-empty, got {reps}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ens = [_Ensemble(problem, r, phase, device, analytic) for r in reps]
    chains = GroupedChains(problem.step, [g for e in ens for g in e.groups])
    for e in ens:
        e.groups = None  # the chains own the states now
    for ti, t in enumerate(problem.t_grid):
        t0 = time.perf_counter()
        chains.advance_to(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        share = (time.perf_counter() - t0) / len(ens)
        for e in ens:
            e.timing["sampling"] = e.timing.get("sampling", 0.0) + share
            e.checkpoint(ti, t, chains)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return [e.finish(chains, len(ens), peak) for e in ens]


def run_ensemble(problem: SuiteProblem, rep: int, phase: str, device, analytic=None) -> dict:
    """Run one repetition."""
    return run_ensembles(problem, [rep], phase, device, analytic)[0]


# Bounds from saved components
def span_gram_omega(setup: dict, data: dict, refop: dict) -> np.ndarray:
    """Gram-defect estimates omega of shape (E, T, F, nm) for each monitored span.

    For k = min(m, kd) directions and recorded entrywise defect e,
        ||G_k - I||_2 <= ||G_k - I||_F <= k e.
    Thus ||P_V delta_hat|| <= ||V^* delta_hat|| / sqrt(1 - omega).

    This estimate is conditional on the rounding in the recorded Gram matrix:
    e excludes that rounding, and the saved results lack the coefficients
    needed to reconstruct it. certified_span_gram includes the rounding term
    for newly computed spans. Families are V3[N], V4own[N], Vref0 and Vref_t.
    """
    E, T, nJ, F, nm, nref = data["proj"].shape
    N_grid = [int(n) for n in setup["N_grid"]]
    nN = len(N_grid)
    m_grid = np.array([int(m) for m in setup["m_grid"]])
    n_dirs = int(np.asarray(refop["coefs_t"]).shape[2])
    e = np.asarray(data["op_orth_err"], np.float64)  # (E, T, nN)
    kd = (
        np.asarray(data["op_kdirs"], np.int64)
        if "op_kdirs" in data
        else np.full(e.shape, n_dirs, dtype=np.int64)  # pre-kd runs: top_span(n_dirs) or raise
    )
    k = np.minimum(m_grid[None, None, None, :], kd[..., None])  # (E, T, nN, nm)
    om = np.zeros((E, T, F, nm))
    for ni in range(nN):
        om[:, :, ni] = (k[:, 0, ni] * e[:, 0, ni, None])[:, None, :]  # V3 = own span at t = 0
        om[:, :, nN + ni] = k[:, :, ni] * e[:, :, ni, None]
    e_ref = float(refop["refop_span_orth_err"])
    om[:, :, 2 * nN :] = (np.minimum(m_grid, n_dirs) * e_ref)[None, None, None, :]
    if not (np.isfinite(om).all() and om.min() >= 0.0 and om.max() < 0.5):
        raise FloatingPointError(f"span Gram defect outside [0, 1/2): max {om.max()}")
    return om


def assemble_bounds(
    setup: dict,
    data: dict,
    refop: dict,
    *,
    quad_margin_known: float,
    gram_correction: bool = True,
    r_init: float | None = None,
) -> dict:
    """Assemble B, H and U arrays with shape (E, T, nJ, nm, nref).

    B3/B4/NO3/NO4 use operator error radii; *ref variants use plug-in operators
    and carry no finite-sample guarantee. Their _tr variants add the estimated
    omitted trace.

    Gram correction multiplies monitor norms by 1/sqrt(1 - omega); see
    span_gram_omega for its rounding assumption. quad_margin_known bounds
    ||mu_tilde - mu_pi||_H and is added to the known-reference monitor and
    direct distances. q = sqrt(3 eps_quad) suffices if both the pointwise
    embedding error and squared-norm error are <= eps_quad. r_init overrides
    the saved source radius.
    """
    R = float(setup["r_init"]) if r_init is None else float(r_init)
    if not (math.isfinite(R) and R > 0.0):
        raise ValueError(f"radius must be finite and positive, got {R}")
    q = float(quad_margin_known)
    if not (math.isfinite(q) and q >= 0.0):
        raise ValueError(f"quad_margin_known must be finite and >= 0, got {q}")
    b_t, b_pi = setup["b_t"], float(setup["b_pi"])
    refs = [str(r) for r in setup["refs"]]
    bpi = np.array([b_pi if r == "sampled" else 0.0 for r in refs])
    qv = np.array([q if r == "known" else 0.0 for r in refs])
    proj = data["proj"]  # (E, T, nJ, F, nm, nref)
    E, T, nJ, F, nm, nref = proj.shape
    N_grid = [int(n) for n in setup["N_grid"]]
    nN = len(N_grid)
    assert F == 2 * nN + 2
    if gram_correction:
        infl = 1.0 / np.sqrt(1.0 - span_gram_omega(setup, data, refop))  # (E, T, F, nm)
        proj = proj * infl[:, :, None, :, :, None]
    H = (
        proj
        + qv[None, None, None, None, None, :]
        + b_t[None, None, :, None, None, None]
        + bpi[None, None, None, None, None, :]
    )
    shape = (E, T, nJ, nm, nref)

    def comb(Hf, U_ETm):  # Hf (E,T,nJ,nm,nref); U_ETm (E,T,nm) unit (without R)
        U = np.broadcast_to(R * U_ETm[:, :, None, :, None], shape)
        return np.sqrt(Hf**2 + U**2), U

    def noobs(U_ET):
        return np.broadcast_to(R * U_ET[:, :, None, None, None], shape)

    B, Hs, Us, guaranteed = {}, {}, {}, {}
    for ni, N in enumerate(N_grid):
        U3 = np.broadcast_to(data["U3_unit"][:, None, ni, :], (E, T, nm))
        for name, fam, Uu in (
            (f"B3_N{N}", ni, U3),
            (f"B4same_N{N}", ni, data["U4same_unit"][:, :, ni]),
            (f"B4own_N{N}", nN + ni, data["U4own_unit"][:, :, ni]),
        ):
            B[name], Us[name] = comb(H[:, :, :, fam], Uu)
            Hs[name] = H[:, :, :, fam]
            guaranteed[name] = True
        B[f"NO3_N{N}"] = noobs(np.broadcast_to(data["NO3_unit"][:, None, ni], (E, T)))
        B[f"NO4_N{N}"] = noobs(data["NO4_unit"][:, :, ni])
        guaranteed[f"NO3_N{N}"] = guaranteed[f"NO4_N{N}"] = True
    direct = (
        data["direct"]
        + qv[None, None, None, :]
        + b_t[None, None, :, None]
        + bpi[None, None, None, :]
    )
    B["direct"] = np.broadcast_to(direct[:, :, :, None, :], shape)
    guaranteed["direct"] = True
    # Plug-in comparisons omit operator error radii.
    own, same, lam1, tau = refop["own"], refop["same"], refop["lam1"], refop["tau"]
    for suffix, tt in (("", np.zeros(T)), ("_tr", np.clip(tau, 0.0, None))):
        U3r = np.broadcast_to(np.sqrt(own[0][None, :] + tt[0]), (T, nm))
        for name, fam, Uu in (
            (f"B3ref{suffix}", 2 * nN, U3r),
            (f"B4ref_same{suffix}", 2 * nN, np.sqrt(same + tt[:, None])),
            (f"B4ref_own{suffix}", 2 * nN + 1, np.sqrt(own + tt[:, None])),
        ):
            B[name], Us[name] = comb(H[:, :, :, fam], np.broadcast_to(Uu[None], (E, T, nm)))
            Hs[name] = H[:, :, :, fam]
            guaranteed[name] = False
        B[f"NOref3{suffix}"] = noobs(
            np.broadcast_to(np.sqrt(lam1[0] + tt[0]), (E, T)).astype(np.float64)
        )
        B[f"NOref4{suffix}"] = noobs(np.broadcast_to(np.sqrt(lam1 + tt)[None], (E, T)))
        guaranteed[f"NOref3{suffix}"] = guaranteed[f"NOref4{suffix}"] = False
    for name, b in B.items():
        assert b.shape == shape, (name, b.shape)
        if not np.isfinite(b).all():
            raise FloatingPointError(f"non-finite bound values for {name}")
    return {"B": B, "H": Hs, "U": Us, "guaranteed": guaranteed}


def first_crossing(B_t: np.ndarray, eps: float) -> np.ndarray:
    """First checkpoint index with B <= eps along the last axis; -1 if none."""
    hit = B_t <= eps
    return np.where(hit.any(-1), hit.argmax(-1), -1)


def transition_count(setup: dict, data: dict, method: str, J: int, N: int | None) -> np.ndarray:
    """Cumulative transitions (E, T): J t for replicas and 4 N t for time-t operators.

    Kernel evaluations, solves and exact draws are counted in method_cost.
    """
    t = setup["t_grid"].astype(np.float64)
    E = data["proj"].shape[0]
    rep = J * t
    if method.startswith("NO"):
        rep = 0.0 * t
    opc = np.zeros_like(t)
    if N is not None and (method.startswith("B4") or method.startswith("NO4")):
        opc = 4.0 * N * t
    return np.broadcast_to((rep + opc)[None, :], (E, len(t))).copy()


COST_UNITS = (
    "transitions",
    "exact_draws",
    "kernel_proj",
    "kernel_direct",
    "kernel_target",
    "solves",
)


def cost_units(setup: dict, data: dict) -> dict:
    """Estimate seconds per operation from median stage timings across repetitions.

    These shared-GPU timings define a cost model. Sampling covers replica and
    operator transitions; kernel costs follow their recorded stages. Each
    operator solve includes its Gram matrix, enclosure and compressed norms.
    Exact draws are counted but charged zero because they were not timed separately.
    """
    tk = [str(x) for x in data["timing_keys"]]
    tim = {k_: np.asarray(data["timing"])[:, i] for i, k_ in enumerate(tk)}
    ke = np.asarray(data["kernel_evals_cum"])[:, -1]  # (E, stages)
    st = {s_: i for i, s_ in enumerate(KERNEL_STAGES)}
    J, N, T3 = int(setup["J_grid"][-1]), int(setup["N_grid"][-1]), float(setup["t_grid"][-1])
    # Smaller N values use prefixes of the same operator-chain population.
    n_tr = (J + 4.0 * N) * T3

    def rate(sec, cnt):
        cnt = np.asarray(cnt, np.float64)
        if (cnt <= 0).any():
            raise ValueError("zero operation count in a timed stage")
        return float(np.median(np.asarray(sec) / cnt))

    return {
        "transitions": rate(tim["sampling"], np.full(len(ke), n_tr)),
        "exact_draws": 0.0,
        "kernel_proj": rate(tim["projections"], ke[:, st["replicas"]]),
        "kernel_direct": rate(tim["direct"], ke[:, st["direct"]]),
        "kernel_target": rate(tim["target"], ke[:, st["target"]]),
        "solves": float(np.median(np.asarray(data["op_time"]))),
        "note": "median stage seconds / operations over ensembles; shared GPUs (2 workers each)",
    }


def method_cost(method: str, k: int, J: int, ref: str, setup: dict, design: dict) -> dict:
    """Operation counts through checkpoint k for a method run on its own.

    Shared replicas, targets and operators are charged in full to each method.
    The counts are comparable across methods but cannot be added to recover
    joint run cost. Excludes one-time reference, landmark and bandwidth setup.
    design supplies N_pi, M and init_draws_per_replica.
    """
    t, n_insp = float(setup["t_grid"][k]), k + 1
    fam, N = method_axes(method)
    N = 0 if N is None else int(N)
    sampled = ref == "sampled"
    Npi = int(design["N_pi"]) if sampled else 0
    M = int(design["M"])
    c = {u: 0.0 for u in COST_UNITS}
    observes = not fam.startswith("NO")
    if observes:
        c["transitions"] += J * t
        c["exact_draws"] += J * float(design["init_draws_per_replica"]) + Npi
    if fam in ("B3", "B4same", "B4own", "NO3", "NO4"):
        c["exact_draws"] += 2 * N
        static = fam in ("B3", "NO3")
        c["solves"] += 1 if static else n_insp
        if not static:
            c["transitions"] += 4 * N * t
    if fam in ("B3", "B4same"):  # replica / target means at the 4N t = 0 endpoints
        c["kernel_proj"] += n_insp * 4 * N * J
        c["kernel_target"] += 4 * N * max(Npi, 1)
    if fam == "B4own":  # new span points at every inspection
        c["kernel_proj"] += n_insp * 4 * N * J
        c["kernel_target"] += n_insp * 4 * N * max(Npi, 1)
    if fam.startswith(("B3ref", "B4ref")):
        c["kernel_proj"] += n_insp * M * J
        c["kernel_target"] += M * max(Npi, 1)
    if fam == "direct":
        c["kernel_direct"] += n_insp * (J * J + J * max(Npi, 1))
        c["kernel_target"] += Npi * Npi  # target self-sum ||mu_hat_pi||^2: stage/timing 'target'
    return c


def modeled_seconds(c: dict, units: dict) -> float:
    return float(sum(c[u] * units[u] for u in COST_UNITS))


def method_axes(name: str) -> tuple[str, int | None]:
    """(family, N) from a method name like 'B4same_N1024'."""
    if "_N" in name:
        fam, n = name.rsplit("_N", 1)
        return fam, int(n)
    return name, None


REVERSAL_THRESHOLDS = {"": 1e-12, "_1e5": 1e-5}  # key suffix: B3 < B4 by more than this


def _paired_record(Dd: np.ndarray, t_grid, J_grid, m_grid, refs) -> dict:
    """Mean, standard error and negative counts for paired differences.

    Dd has shape (E, T, nJ, nm, nref). Count reversals at thresholds -1e-12
    and -1e-5, both overall and by checkpoint.
    """
    E = Dd.shape[0]
    sd = Dd.std(0, ddof=1) if E > 1 else np.zeros(Dd.shape[1:])
    neg = Dd < -REVERSAL_THRESHOLDS[""]
    i = np.unravel_index(int(np.argmin(Dd)), Dd.shape)
    return {
        "mean": Dd.mean(0).tolist(),
        "se": (sd / math.sqrt(E)).tolist(),
        "min": float(Dd.min()),
        "argmin": {
            "ensemble_index": int(i[0]),
            "t": int(t_grid[i[1]]),
            "J": int(J_grid[i[2]]),
            "m": int(m_grid[i[3]]),
            "ref": refs[i[4]],
        },
        "frac_negative_cells": float(neg.mean()),
        "n_entries": int(Dd.size),
        "n_negative": int(neg.sum()),
        "n_entries_t_pos": int(Dd[:, 1:].size),
        "n_negative_t_pos": int(neg[:, 1:].sum()),
        "min_t_pos": float(Dd[:, 1:].min()) if Dd.shape[1] > 1 else None,
        "n_negative_by_t": {int(t): int(neg[:, ti].sum()) for ti, t in enumerate(t_grid)},
        "n_negative_last_t": int(neg[:, -1].sum()),
        **{
            k: v
            for sfx, thr in REVERSAL_THRESHOLDS.items()
            if sfx
            for k, v in (
                (f"n_negative{sfx}", int((Dd < -thr).sum())),
                (f"n_negative_t_pos{sfx}", int((Dd[:, 1:] < -thr).sum())),
                (
                    f"n_negative_by_t{sfx}",
                    {int(t): int((Dd[:, ti] < -thr).sum()) for ti, t in enumerate(t_grid)},
                ),
                (f"n_negative_last_t{sfx}", int((Dd[:, -1] < -thr).sum())),
            )
        },
    }


def summarize(
    setup: dict,
    data: dict,
    refop: dict,
    truth: dict,
    tolerances=(0.05, 0.10),
    alpha: float = 0.05,
    *,
    quad_margin_known: float,
    r_init: float | None = None,
    design: dict | None = None,
    applicable: list[str] | None = None,
) -> dict:
    """Summarize coverage, tightness, paired differences and stopping costs.

    Uses Gram and quadrature corrections from assemble_bounds; corrections
    records their effect on coverage codes and stopping counts. design enables
    the per-method cost model.

    applicable lists methods whose assumptions hold for this run (None means
    all). Bounds evaluated at an assumed radius are kept out of the joint
    guarantee counts and reported under joint_assumed_radius_* instead.
    """
    asm = assemble_bounds(setup, data, refop, quad_margin_known=quad_margin_known, r_init=r_init)
    B = asm["B"]
    L, U, D = truth["L"], truth["U"], truth["D"]
    t_grid = setup["t_grid"]
    J_grid = [int(j) for j in setup["J_grid"]]
    m_grid = [int(m) for m in setup["m_grid"]]
    refs = [str(r) for r in setup["refs"]]
    E = data["proj"].shape[0]
    if not np.array_equal(truth["t_grid"], t_grid):
        raise ValueError("truth intervals were computed on a different schedule")
    summ = {"n_ensembles": int(E), "methods": {}, "labels": {}}
    summ["r_init_used"] = float(setup["r_init"]) if r_init is None else float(r_init)
    units = cost_units(setup, data) if design is not None else None
    if units is not None:
        summ["cost_units"] = units
    joint_codes = {}
    if applicable is not None and not set(applicable) <= {
        n for n, g in asm["guaranteed"].items() if g
    }:
        raise ValueError(f"applicable methods {applicable} are not all guaranteed-by-construction")
    holds = {
        n: bool(g and (applicable is None or n in applicable)) for n, g in asm["guaranteed"].items()
    }
    for name, Bm in B.items():
        summ["labels"][name] = (
            "guarantee"
            if holds[name]
            else "assumed radius (NOT a guarantee)"
            if asm["guaranteed"][name]
            else "plug-in (NOT a guarantee)"
        )
        fam, N = method_axes(name)
        invariant = bool((Bm == Bm[:1]).all())  # no ensemble dependence (plug-in no-obs bounds)
        md = {}
        for ji, J in enumerate(J_grid):
            for mi, m in enumerate(m_grid):
                for ri, r in enumerate(refs):
                    Bv = Bm[:, :, ji, mi, ri]  # (E, T)
                    codes = rf.classify(Bv, L, U)
                    joint_codes[(name, J, m, r)] = codes
                    rec = rf.coverage_summary(codes, alpha=alpha, beta_ref=float(truth["beta_ref"]))
                    rec["cells_violation"] = int((codes == rf.VIOLATION).sum())
                    rec["cells_unresolved"] = int((codes == rf.UNRESOLVED).sum())
                    rec["ensemble_invariant"] = invariant
                    if invariant:  # one curve repeated E times: not E independent validations
                        rec["t_unresolved"] = [
                            int(t) for t, c in zip(t_grid, codes[0]) if c == rf.UNRESOLVED
                        ]
                        rec["t_violation"] = [
                            int(t) for t, c in zip(t_grid, codes[0]) if c == rf.VIOLATION
                        ]
                    rec["B_median"] = np.median(Bv, 0).tolist()
                    rec["B_q05"] = np.quantile(Bv, 0.05, 0).tolist()
                    rec["B_q95"] = np.quantile(Bv, 0.95, 0).tolist()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = Bv / np.where(D > 0, D, np.nan)[None, :]
                    rec["ratio_to_D_median"] = np.nanmedian(ratio, 0).tolist()
                    if name in asm["H"]:
                        rec["H_median"] = np.median(asm["H"][name][:, :, ji, mi, ri], 0).tolist()
                        rec["U_median"] = np.median(asm["U"][name][:, :, ji, mi, ri], 0).tolist()
                    trans = transition_count(setup, data, fam, J, N)
                    for eps in tolerances:
                        fc = first_crossing(Bv, eps)
                        crossed = fc >= 0
                        idx = np.where(crossed, fc, 0)
                        # a stop with MMD > eps is certain if L_t > eps at the stop (false stop)
                        false_stop = crossed & (L[idx] > eps)
                        confirmed_ok = crossed & (U[idx] <= eps)
                        cr = {
                            "n_crossed": int(crossed.sum()),
                            "n_censored": int((~crossed).sum()),
                            "t_counts": {
                                int(t): int((fc == i).sum()) for i, t in enumerate(t_grid)
                            },
                            "t_median_crossed": float(np.median(t_grid[fc[crossed]]))
                            if crossed.any()
                            else None,
                            "transitions_median_crossed": float(
                                np.median(trans[np.arange(E), idx][crossed])
                            )
                            if crossed.any()
                            else None,
                            # Elapsed time includes all methods in the run.
                            "pipeline_wall_cum_median_crossed": float(
                                np.median(data["wall_cum"][np.arange(E), idx][crossed])
                            )
                            if crossed.any()
                            else None,
                            "false_stops_definite": int(false_stop.sum()),
                            "stops_confirmed_below_eps": int(confirmed_ok.sum()),
                        }
                        if units is not None and crossed.any():
                            ks = np.unique(fc[crossed])
                            sec = {
                                int(k_): modeled_seconds(
                                    method_cost(name, int(k_), J, r, setup, design), units
                                )
                                for k_ in ks
                            }
                            cr["modeled_seconds_median_crossed"] = float(
                                np.median([sec[int(k_)] for k_ in fc[crossed]])
                            )
                        rec[f"cross_{eps}"] = cr
                    if units is not None:
                        last = method_cost(name, len(t_grid) - 1, J, r, setup, design)
                        rec["cost_last"] = {**last, "modeled_seconds": modeled_seconds(last, units)}
                    md[f"J{J}|m{m}|{r}"] = rec
        summ["methods"][name] = md
    # Joint coverage across methods and grids, separating assumed-radius runs.
    for r in refs:
        for key, pick in (
            ("joint_guaranteed", lambda n: holds[n]),
            ("joint_assumed_radius", lambda n: asm["guaranteed"][n] and not holds[n]),
        ):
            sel = [c for (name, J, m, rr), c in joint_codes.items() if rr == r and pick(name)]
            if not sel:
                continue
            stack = np.stack(sel)  # (n_methods, E, T)
            all_cov = (stack == rf.COVERED).all((0, 2))
            any_vio = (stack == rf.VIOLATION).any((0, 2))
            unres = ~all_cov & ~any_vio
            rec = {
                "covered_all": int(all_cov.sum()),
                "any_violation": int(any_vio.sum()),
                "unresolved": int(unres.sum()),
                "cp_if_unresolved_fail": rf.clopper_pearson(int(all_cov.sum()), E, alpha),
                "cp_if_unresolved_succeed": rf.clopper_pearson(
                    int(all_cov.sum() + unres.sum()), E, alpha
                ),
            }
            if applicable is not None:  # which methods the joint event covers (known-radius: all)
                rec["methods"] = sorted({n for (n, J, m, rr) in joint_codes if rr == r and pick(n)})
            if key == "joint_assumed_radius":  # B < L here is a radius-misspecification exceedance
                rec["radius_misspecification_exceedance"] = rec["any_violation"]
            summ[f"{key}_{r}"] = rec
    # paired differences B3 - B4 (same replicas, same events)
    pairs = []
    for N in setup["N_grid"]:
        N = int(N)
        pairs += [(f"B3_N{N}", f"B4same_N{N}"), (f"B3_N{N}", f"B4own_N{N}")]
    pairs += [("B3ref", "B4ref_same"), ("B3ref", "B4ref_own")]
    summ["paired"] = {
        f"{a}-{b}": _paired_record(B[a] - B[b], t_grid, J_grid, m_grid, refs) for a, b in pairs
    }
    # Match transition counts: (J + 4N) t for both B3 and B4. Other costs differ.
    summ["transition_matched"] = []
    for N in setup["N_grid"]:
        N = int(N)
        for ji, J in enumerate(J_grid):
            if J + 4 * N in J_grid:
                j3 = J_grid.index(J + 4 * N)
                for kind in ("B4same", "B4own"):
                    Dd = B[f"B3_N{N}"][:, :, j3] - B[f"{kind}_N{N}"][:, :, ji]
                    summ["transition_matched"].append(
                        {
                            "B3": f"B3_N{N}@J{J + 4 * N}",
                            "B4": f"{kind}_N{N}@J{J}",
                            "mean_B3_minus_B4": Dd.mean(0).tolist(),
                            "se": (Dd.std(0, ddof=1) / math.sqrt(E)).tolist() if E > 1 else None,
                        }
                    )
    if units is not None:
        summ["cost_comparison"] = _cost_comparison(B, asm, setup, design, units, refs)
    summ["operator"] = {
        "rank_min": int(data["op_rank"].min()),
        "npos_min": int(data["op_npos"].min()),
        "delta_trunc_max": float(data["op_delta_trunc"].max()),
        "orth_err_max": float(data["op_orth_err"].max()),
        "head_slack_max": float(data["op_head_slack_max"].max()),
        "min_eig_min": float(data["op_min_eig"].min()),
        "own_check_max": float(data["op_own_check"].max()),
        "trace_consistency_max": float(np.abs(data["op_trace"] - data["op_trace_paired"]).max()),
        "e_tot_median": np.median(data["e_tot"], 0).tolist(),
        "top_eig_median": np.median(data["op_top"], 0).tolist(),
    }
    summ["corrections"] = _corrections(
        B, setup, data, refop, truth, tolerances, quad_margin_known, r_init, asm["guaranteed"]
    )
    tk = [str(x) for x in data["timing_keys"]]
    summ["cost"] = {
        "wall_per_ensemble_median": float(np.median(data["wall"])),
        "wall_per_ensemble_max": float(np.max(data["wall"])),
        "wall_total": float(data["wall"].sum()),
        "timing_median": dict(zip(tk, np.median(data["timing"], 0).tolist())),
        "op_time_median_per_checkpoint": np.median(data["op_time"], 0).tolist(),
        "kernel_evals_median_total": np.median(data["kernel_evals_cum"][:, -1], 0).tolist(),
        "kernel_stages": list(KERNEL_STAGES),
        "init_proposals_median": float(np.median(data["init_proposals"])),
        "peak_gpu_bytes_max": int(data["peak_gpu_bytes"].max()),
        "peak_rss_kb_max": int(data["peak_rss_kb"].max()),
        "accept_rep_final_median": float(np.median(data["accept_rep"][:, -1])),
        "accept_op_final_median": float(np.median(data["accept_op"][:, -1])),
    }
    return summ


def _corrections(B, setup, data, refop, truth, tolerances, q, r_init, guaranteed) -> dict:
    """Changes in bounds, coverage codes and first crossings due to numerical corrections."""
    raw = assemble_bounds(
        setup, data, refop, quad_margin_known=0.0, gram_correction=False, r_init=r_init
    )["B"]
    gram = assemble_bounds(
        setup, data, refop, quad_margin_known=0.0, gram_correction=True, r_init=r_init
    )["B"]
    om = span_gram_omega(setup, data, refop)
    L, U = truth["L"], truth["U"]
    J_grid = [int(j) for j in setup["J_grid"]]
    m_grid = [int(m) for m in setup["m_grid"]]
    refs = [str(r) for r in setup["refs"]]
    out = {
        "quad_margin_known": float(q),
        "omega_max": float(om.max()),
        "inflation_max": float(1.0 / math.sqrt(1.0 - om.max())),
        "methods": {},
    }
    for name in B:
        d_all = B[name] - raw[name]
        d_gram = gram[name] - raw[name]
        if (d_all < 0).any() or (d_gram < 0).any():
            raise FloatingPointError(f"a correction lowered {name}")
        rec = {
            "guaranteed": bool(guaranteed[name]),
            "max_change": float(d_all.max()),
            "max_change_gram": float(d_gram.max()),
            "max_rel_change_gram": float((d_gram / raw[name]).max()),
            "cells_code_changed": 0,
            "crossings_changed": {},
        }
        for ji, J in enumerate(J_grid):
            for mi, m in enumerate(m_grid):
                for ri, r in enumerate(refs):
                    a, b = B[name][:, :, ji, mi, ri], raw[name][:, :, ji, mi, ri]
                    rec["cells_code_changed"] += int(
                        (rf.classify(a, L, U) != rf.classify(b, L, U)).sum()
                    )
                    for eps in tolerances:
                        na = int((first_crossing(a, eps) >= 0).sum())
                        nb = int((first_crossing(b, eps) >= 0).sum())
                        if na != nb:
                            rec["crossings_changed"][f"J{J}|m{m}|{r}|{eps}"] = [nb, na]
        out["methods"][name] = rec
    return out


def _cost_comparison(B, asm, setup, design, units, refs) -> dict:
    """Compare B4 with the largest B3 replica budget no more costly on the grid.

    Also check R U3 > B4, which rules out matching B4 with any B3 replica
    budget. Grid-limited B3 runs may still use less than the available cost.
    """
    J_grid = [int(j) for j in setup["J_grid"]]
    m_grid = [int(m) for m in setup["m_grid"]]
    k = len(setup["t_grid"]) - 1
    out = []
    for N in [int(n) for n in setup["N_grid"]]:
        U3 = asm["U"][f"B3_N{N}"][:, -1, 0]  # (E, nm, nref) at the last t (J-independent)
        for kind in ("B4same", "B4own"):
            for ri, r in enumerate(refs):
                c3 = [
                    modeled_seconds(method_cost(f"B3_N{N}", k, J, r, setup, design), units)
                    for J in J_grid
                ]
                for j4, J4 in enumerate(J_grid):
                    c4 = modeled_seconds(
                        method_cost(f"{kind}_N{N}", k, J4, r, setup, design), units
                    )
                    ok = [j for j in range(len(J_grid)) if c3[j] <= c4]
                    rec = {
                        "B4": f"{kind}_N{N}@J{J4}",
                        "ref": r,
                        "modeled_seconds_B4": c4,
                        "modeled_seconds_B3_by_J": dict(zip(J_grid, c3)),
                        "B3_floor_RU3_exceeds_B4_frac_by_m": {
                            int(m): float(
                                (U3[:, mi, ri] > B[f"{kind}_N{N}"][:, -1, j4, mi, ri]).mean()
                            )
                            for mi, m in enumerate(m_grid)
                        },
                    }
                    if ok:
                        j3 = max(ok)
                        Dd = B[f"B3_N{N}"][:, -1, j3, :, ri] - B[f"{kind}_N{N}"][:, -1, j4, :, ri]
                        rec["B3_at_most_B4_cost"] = f"B3_N{N}@J{J_grid[j3]}"
                        rec["cost_ratio_B4_over_B3"] = c4 / c3[j3]
                        rec["mean_B3_minus_B4_last_by_m"] = {
                            int(m): float(Dd[:, mi].mean()) for mi, m in enumerate(m_grid)
                        }
                    out.append(rec)
    return out


# Analytic Gaussian checks
class ARGaussWitness:
    """X' = A X + L xi on pi = N(0, C) with A C A^T + L L^T = C; init d pi0/d pi = 1 + eps s(x),
    s = sign(q^T x) (pi(A) = 1/2). For the RBF kernel with Lam = diag((s ell)^2), A_t = A^t,
    Q_t = C - A_t C A_t^T and X, X' iid pi, U = X - X' ~ N(0, 2C):

        MMD^2(pi_t, pi) = eps^2 E[s(X) s(X') K_t(X, X')]
            = eps^2 c_K w [(4/pi) arctan(sigma_a / s_g) - 1],
        K_t(x, x') = c_K exp(-u^T B u / 2), u = x - x', c_K = det(I + 2 Q_t Lam^-1)^{-1/2},
        B = A_t^T (Lam + 2 Q_t)^{-1} A_t, w = det(I + 2 C B)^{-1/2}, S' = ((2C)^{-1} + B)^{-1},
        s_g^2 = q^T S' q, sigma_a^2 = 2 q^T C q,

    using E_V[sign(a^2 - b^2) | b] = 4 Phi(-|b|/sigma_a) - 1 (V = X + X' independent of U) and
    E[4 Phi(-|g|/c) - 1] = (4/pi) arctan(c/s_g) - 1 for g ~ N(0, s_g^2). The current-law embedding
    mu_{pi_t}(z) = c_t n(z) [1 + eps (2 Phi(q^T m(z)/sqrt(q^T Pi^-1 q)) - 1)] (closed form), and
    ||mu_{pi_t}||^2 = ||mu_pi||^2 + MMD^2 (the cross term vanishes by the x -> -x symmetry).
    tr Sigma_t = det(I + 2 Q_t Lam^-1)^{-1/2} - ||mu_pi||^2 (two branches from one start)."""

    def __init__(self, C, A, lam_kernel, q, eps):
        self.C = np.asarray(C, np.float64)
        self.A = np.asarray(A, np.float64)
        self.lam = np.asarray(lam_kernel, np.float64)  # (s ell)^2 per coordinate
        self.q = np.asarray(q, np.float64)
        self.eps = float(eps)
        d = self.C.shape[0]
        self.d = d
        self.Lam = np.diag(self.lam)
        Q = self.C - self.A @ self.C @ self.A.T
        if np.linalg.eigvalsh(0.5 * (Q + Q.T)).min() < -1e-12:
            raise ValueError("A C A^T exceeds C: not a valid stationary linear-Gaussian kernel")
        self.mu_norm2 = float(
            np.exp(-0.5 * np.linalg.slogdet(np.eye(d) + 2.0 * self.C @ np.diag(1 / self.lam))[1])
        )

    def At(self, t: int) -> np.ndarray:
        return np.linalg.matrix_power(self.A, t)

    def Qt(self, t: int) -> np.ndarray:
        At = self.At(t)
        Q = self.C - At @ self.C @ At.T
        return 0.5 * (Q + Q.T)

    def mmd2(self, t: int) -> float:
        d, C, q = self.d, self.C, self.q
        At, Qt = self.At(t), self.Qt(t)
        iLam = np.diag(1.0 / self.lam)
        cK = math.exp(-0.5 * np.linalg.slogdet(np.eye(d) + 2.0 * Qt @ iLam)[1])
        Bm = At.T @ np.linalg.solve(self.Lam + 2.0 * Qt, At)
        Bm = 0.5 * (Bm + Bm.T)
        w = math.exp(-0.5 * np.linalg.slogdet(np.eye(d) + 2.0 * C @ Bm)[1])
        Sp = np.linalg.inv(np.linalg.inv(2.0 * C) + Bm)
        s_g = math.sqrt(float(q @ Sp @ q))
        sig_a = math.sqrt(2.0 * float(q @ C @ q))
        val = self.eps**2 * cK * w * ((4.0 / math.pi) * math.atan(sig_a / s_g) - 1.0)
        if val < -1e-15:
            raise FloatingPointError(f"negative analytic MMD^2 {val}")
        return max(val, 0.0)

    def trace(self, t: int) -> float:
        d = self.d
        Qt = self.Qt(t)
        iLam = np.diag(1.0 / self.lam)
        return math.exp(-0.5 * np.linalg.slogdet(np.eye(d) + 2.0 * Qt @ iLam)[1]) - self.mu_norm2

    def mu_t(self, t: int, Zp: torch.Tensor) -> torch.Tensor:
        """mu_{pi_t}(z) at the rows of Zp (closed form, float64, on Zp's device)."""
        d, C, q = self.d, self.C, self.q
        At, Qt = self.At(t), self.Qt(t)
        P = np.linalg.inv(self.Lam + Qt)
        c_t = math.exp(-0.5 * np.linalg.slogdet(np.eye(d) + Qt @ np.diag(1.0 / self.lam))[1])
        Pi = np.linalg.inv(C) + At.T @ P @ At
        Pi_inv = np.linalg.inv(Pi)
        logdet_term = -0.5 * (np.linalg.slogdet(C)[1] + np.linalg.slogdet(Pi)[1])
        Mz = Pi_inv @ At.T @ P  # m(z) = Mz z
        Kq = P - P @ At @ Pi_inv @ At.T @ P  # exponent: -1/2 z^T Kq z
        dev = Zp.device
        Mz_t = torch.as_tensor(Mz, dtype=F64, device=dev)
        Kq_t = torch.as_tensor(0.5 * (Kq + Kq.T), dtype=F64, device=dev)
        q_t = torch.as_tensor(q, dtype=F64, device=dev)
        sq = math.sqrt(float(q @ Pi_inv @ q))
        quad = ((Zp @ Kq_t) * Zp).sum(1)
        norm = c_t * torch.exp(logdet_term - 0.5 * quad)
        m = Zp @ Mz_t.T
        arg = (m @ q_t) / sq
        Phi = 0.5 * torch.erfc(-arg / math.sqrt(2.0))
        return norm * (1.0 + self.eps * (2.0 * Phi - 1.0))

    def replica_event(self, t: int, X: torch.Tensor, J_grid, n2_hat: torch.Tensor) -> np.ndarray:
        """||mu_hat_{t,J} - mu_{pi_t}|| for each J (exact; n2_hat = ||mu_hat_J||^2 nested)."""
        cross = prefix_means(self.mu_t(t, X[: J_grid[-1]]), J_grid)
        n2_t = self.mu_norm2 + self.mmd2(t)
        ev = n2_hat - 2.0 * cross + n2_t
        if float(ev.min()) < -1e-10:
            raise FloatingPointError(f"negative replica-event distance {float(ev.min())}")
        return ev.clamp_min(0.0).sqrt().cpu().numpy()


def ar_witness_chain(
    C: np.ndarray, rho: np.ndarray, Qrot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A = Q diag(rho) Q^T, L = Q diag(sqrt(lam (1 - rho^2))) for C = Q diag(lam) Q^T (reversible)."""
    lamC = np.diag(Qrot.T @ C @ Qrot)
    if np.abs(Qrot.T @ C @ Qrot - np.diag(lamC)).max() > 1e-10:
        raise ValueError("Qrot does not diagonalize C")
    rho = np.asarray(rho, np.float64)
    if (np.abs(rho) >= 1).any():
        raise ValueError(f"|rho| must be < 1, got {rho}")
    A = (Qrot * rho[None, :]) @ Qrot.T
    L = Qrot * np.sqrt(lamC * (1.0 - rho**2))[None, :]
    return A, L


def witness_checks(setup: dict, data: dict, ref: dict, mmd_true: np.ndarray, budget: dict) -> dict:
    """Check reference intervals, coverage and component events against analytic MMD.

    For a Gaussian target, the embedding needs no quadrature margin. Stored
    projection values pt use nearly orthonormal coefficients, giving
        pt/sqrt(1 + omega) <= ||P_V delta_t|| <= pt/sqrt(1 - omega).
    Use the upper end for H and the lower end for the unmonitored remainder.
    Also report operator-trace z-scores and budget checks.
    """
    mmd_true = np.asarray(mmd_true, np.float64)
    T = len(setup["t_grid"])
    assert mmd_true.shape == (T,)
    asm = assemble_bounds(setup, data, ref, quad_margin_known=0.0)
    E = data["proj"].shape[0]
    out = {
        "n_ensembles": int(E),
        "i_truth_interval_contains_analytic": bool(
            ((ref["L"] <= mmd_true) & (mmd_true <= ref["U"])).all()
        ),
        "i_interval": {
            "L": ref["L"].tolist(),
            "analytic": mmd_true.tolist(),
            "U": ref["U"].tolist(),
            "D": ref["D"].tolist(),
            "D_minus_analytic_over_halfwidth": (
                (ref["D"] - mmd_true) / (ref["U"] - ref["D"])
            ).tolist(),
        },
        "iii_budget": budget,
        "ii_methods": {},
    }
    for name, B in asm["B"].items():
        cov = (B >= mmd_true[None, :, None, None, None]).all(1)  # (E, nJ, nm, nref)
        codes = np.stack(
            [
                rf.classify(B[:, :, ji, mi, ri], ref["L"], ref["U"])
                for ji in range(B.shape[2])
                for mi in range(B.shape[3])
                for ri in range(B.shape[4])
            ]
        )
        out["ii_methods"][name] = {
            "guaranteed": asm["guaranteed"][name],
            "covered_all_t_all_cells": int(cov.all((1, 2, 3)).sum()),
            "covered_all_t_fraction_cells": float(cov.mean()),
            "min_margin": float((B - mmd_true[None, :, None, None, None]).min()),
            "interval_violation_cells": int((codes == rf.VIOLATION).sum()),
            "interval_unresolved_cells": int((codes == rf.UNRESOLVED).sum()),
        }
    b_t, b_pi = setup["b_t"], float(setup["b_pi"])
    rep_ok = (data["replica_event"] <= b_t[None, None, :]).all((1, 2))
    out["ii_replica_event"] = {
        "holds": int(rep_ok.sum()),
        "max_ratio_to_b_t": float((data["replica_event"] / b_t[None, None, :]).max()),
    }
    refs = [str(r) for r in setup["refs"]]
    bpi = np.array([b_pi if r == "sampled" else 0.0 for r in refs])
    om = span_gram_omega(setup, data, ref)  # (E, T, F, nm)
    infl = 1.0 / np.sqrt(1.0 - om)
    H = (
        data["proj"] * infl[:, :, None, :, :, None]
        + b_t[None, None, :, None, None, None]
        + bpi[None, None, None, None, None, :]
    )
    pt = data["proj_true"]  # (E, T, F, nm)
    pt_hi, pt_lo = pt * infl, pt / np.sqrt(1.0 + om)
    H_ok = (pt_hi[:, :, None, :, :, None] <= H).all((1, 2, 3, 4, 5))
    out["ii_H_component"] = {
        "holds": int(H_ok.sum()),
        "max_ratio": float((pt_hi[:, :, None, :, :, None] / H).max()),
        "gram_omega_max": float(om.max()),
    }
    unmon = np.sqrt(np.clip(mmd_true[None, :, None, None] ** 2 - pt_lo**2, 0.0, None))  # (E,T,F,nm)
    nN = len(setup["N_grid"])
    fam_of = {}
    for ni, N in enumerate(setup["N_grid"]):
        fam_of[f"B3_N{int(N)}"] = ni
        fam_of[f"B4same_N{int(N)}"] = ni
        fam_of[f"B4own_N{int(N)}"] = nN + ni
    for suf in ("", "_tr"):
        fam_of[f"B3ref{suf}"] = 2 * nN
        fam_of[f"B4ref_same{suf}"] = 2 * nN
        fam_of[f"B4ref_own{suf}"] = 2 * nN + 1
    out["ii_U_component"] = {}
    for name, fam in fam_of.items():
        Uv = asm["U"][name][:, :, 0, :, 0]  # (E, T, nm); U does not depend on J / ref
        ok = (unmon[:, :, fam, :] <= Uv * (1 + 1e-12)).all((1, 2))
        out["ii_U_component"][name] = {
            "guaranteed": asm["guaranteed"][name],
            "holds": int(ok.sum()),
            "min_slack": float((Uv - unmon[:, :, fam, :]).min()),
        }
    tr_true = data["trace_true"][0]
    z = (data["op_trace_paired"] - tr_true[None, :, None]) / data["op_trace_paired"].std(0, ddof=1)
    out["ii_operator_trace"] = {
        "analytic": tr_true.tolist(),
        "mean_estimate": data["op_trace_paired"].mean(0).tolist(),
        "mean_z_of_mean": (
            (data["op_trace_paired"].mean(0) - tr_true[:, None])
            / (data["op_trace_paired"].std(0, ddof=1) / math.sqrt(E))
        ).tolist(),
        "max_abs_z_single": float(np.abs(z).max()),
    }
    return out
