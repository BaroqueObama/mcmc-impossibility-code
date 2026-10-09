"""Positive density perturbations and projected residuals.

The two-dimensional examples use dQ/dpi = 1 + h, with h constant on
approximately equal-mass cells. A constrained eigenproblem finds perturbations
orthogonal to the chosen covariance eigenfunctions; scaling enforces positivity.
These are feasible examples, without a claim of global optimality.

Construction and evaluation use separate quadrature rules. The evaluated
residual is sqrt(a^T G_eval a - ||z_eval||^2), where z_eval contains the monitor
readings. The width R sqrt(lambda_{m+1}) is computed separately for comparison.
Kernel PCA and independent chain replicas provide the higher-dimensional checks.
"""

from __future__ import annotations

import concurrent.futures
import copy
import functools
import math
import multiprocessing
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.special as sps
import torch

from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import ScaledRBF, _ScaledStationaryKernel
from mcmc.paperexp.targets import (
    FunnelTarget,
    GaussTarget,
    MixtureTarget,
    StudentTTarget,
    Target,
)

F64 = torch.float64
SQRT2PI = math.sqrt(2.0 * math.pi)
DEFAULT_BLOCK_BYTES = 1 << 29


# One-dimensional quadrature
@functools.lru_cache(maxsize=None)
def gl_rule(q: int) -> tuple[np.ndarray, np.ndarray]:
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    t, w = np.polynomial.legendre.leggauss(q)
    return t, w


def composite_gl(lo: float, hi: float, n_sub: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights of the n_sub-panel q-point Gauss-Legendre rule for int_lo^hi f(u) du."""
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"bad interval [{lo}, {hi}]")
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}")
    edges = np.linspace(lo, hi, n_sub + 1)
    t, w = gl_rule(q)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * t[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def _n_sub(length: float, cap: float, jac: float, h_x: float) -> int:
    """Panels so that each is <= cap long (latent units) and <= h_x bandwidths in x-space."""
    return max(1, int(math.ceil(length / cap)), int(math.ceil(length * jac / h_x)))


def graded_rule(lo: float, hi: float, step, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite q-point GL on [lo, hi] with panel lengths <= step(left edge) (graded mesh)."""
    edges = [lo]
    while edges[-1] < hi:
        h = float(step(edges[-1]))
        if not h > 0:
            raise ValueError(f"non-positive panel length {h}")
        edges.append(min(hi, edges[-1] + h))
    e = np.asarray(edges)
    if e.shape[0] > 2 and e[-1] - e[-2] < 1e-3 * (e[-2] - e[-3]):
        e = np.delete(e, -2)  # merge a sliver panel into its neighbour
    t, w = gl_rule(q)
    half, mid = 0.5 * np.diff(e), 0.5 * (e[1:] + e[:-1])
    return (mid[:, None] + half[:, None] * t[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def panel_rule(lo: float, hi: float, cap: float, jac: float, h_x: float, q: int, q_min: int):
    """Composite Gauss-Legendre rule with panel length <= min(cap, h_x / jac).

    Short panels use proportionally fewer points, with at least q_min per panel.
    """
    n = _n_sub(hi - lo, cap, jac, h_x)
    l_ref = min(cap, h_x / jac) if jac > 0 else cap
    qq = max(q_min, min(q, int(math.ceil(q * (hi - lo) / n / l_ref))))
    return composite_gl(lo, hi, n, qq)


# Bivariate normal probabilities
def phi2(h, k, rho, sq):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho in [0, 1).

    ``sq`` must equal sqrt(1 - rho^2), passed separately so callers can compute it without
    cancellation. Owen (1956):  Phi2 = Phi(h)/2 + Phi(k)/2 - T(h, a_h) - T(k, a_k) - 1/2 [hk < 0],
    a_h = (k - rho h)/(h sq), a_k = (h - rho k)/(k sq); the h = 0 or k = 0 limits are used
    explicitly (Phi2(0, k) = Phi(k)/2 + T(k, rho/sq), Phi2(0, 0) = 1/4 + asin(rho)/(2 pi)).
    Infinite arguments are allowed.
    """
    h, k, rho, sq = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (h, k, rho, sq)))
    if np.isnan(h).any() or np.isnan(k).any():
        raise FloatingPointError("NaN argument to phi2")
    if (rho < 0).any() or (rho >= 1).any() or (sq <= 0).any():
        raise ValueError("phi2 needs rho in [0, 1) and sq > 0")
    out = np.empty(h.shape, dtype=np.float64)
    neg_inf = np.isneginf(h) | np.isneginf(k)
    out[neg_inf] = 0.0
    hp = np.isposinf(h) & ~neg_inf
    kp = np.isposinf(k) & ~neg_inf
    both = hp & kp
    out[both] = 1.0
    sel = hp & ~kp
    out[sel] = sps.ndtr(k[sel])
    sel = kp & ~hp
    out[sel] = sps.ndtr(h[sel])
    fin = ~(neg_inf | hp | kp)
    hf, kf, rf, sf = h[fin], k[fin], rho[fin], sq[fin]
    val = np.empty(hf.shape, dtype=np.float64)
    h0, k0 = hf == 0.0, kf == 0.0
    s = h0 & k0
    val[s] = 0.25 + np.arcsin(rf[s]) / (2.0 * math.pi)
    s = h0 & ~k0
    val[s] = 0.5 * sps.ndtr(kf[s]) + sps.owens_t(kf[s], rf[s] / sf[s])
    s = k0 & ~h0
    val[s] = 0.5 * sps.ndtr(hf[s]) + sps.owens_t(hf[s], rf[s] / sf[s])
    s = ~h0 & ~k0
    hs, ks, rs, ss = hf[s], kf[s], rf[s], sf[s]
    ah = (ks - rs * hs) / (hs * ss)
    ak = (hs - rs * ks) / (ks * ss)
    val[s] = (
        0.5 * sps.ndtr(hs)
        + 0.5 * sps.ndtr(ks)
        - sps.owens_t(hs, ah)
        - sps.owens_t(ks, ak)
        - 0.5 * (hs * ks < 0)
    )
    out[fin] = val
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite phi2")
    return out


def phi_interval(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Phi(hi) - Phi(lo) computed on the side of the origin that avoids cancellation."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, np.float64), np.asarray(hi, np.float64))
    pos = lo > 0
    return np.where(pos, sps.ndtr(-lo) - sps.ndtr(-hi), sps.ndtr(hi) - sps.ndtr(lo))


# Latent coordinate distributions
class NormalAxis:
    """N(0, 1) in its own coordinate; truncated to [-cut, cut]."""

    kind = "normal"

    def __init__(self, tail_mass: float):
        self.cut = float(-sps.ndtri(tail_mass))
        self.lo, self.hi = -self.cut, self.cut

    def edges(self, n: int) -> np.ndarray:
        return sps.ndtri(np.arange(n + 1) / n)

    def density(self, v):
        return np.exp(-0.5 * v * v) / SQRT2PI

    def cdf(self, v):
        return sps.ndtr(v)

    def mass(self, lo, hi):
        return phi_interval(lo, hi)

    def to_dict(self):
        return {"kind": self.kind, "cut": self.cut}


class MixtureAxis:
    """w N(-s/2, 1) + (1 - w) N(+s/2, 1); quantiles by bracketed root finding."""

    kind = "mixture"

    def __init__(self, sep: float, w: float, tail_mass: float):
        self.sep, self.w = float(sep), float(w)
        cut = float(-sps.ndtri(tail_mass))
        self.lo, self.hi = -self.sep / 2 - cut, self.sep / 2 + cut
        self.cut = cut

    def density(self, v):
        h = self.sep / 2
        return (
            self.w * np.exp(-0.5 * (v + h) ** 2) + (1 - self.w) * np.exp(-0.5 * (v - h) ** 2)
        ) / SQRT2PI

    def cdf(self, v):
        h = self.sep / 2
        return self.w * sps.ndtr(v + h) + (1 - self.w) * sps.ndtr(v - h)

    def mass(self, lo, hi):
        h = self.sep / 2
        return self.w * phi_interval(lo + h, hi + h) + (1 - self.w) * phi_interval(lo - h, hi - h)

    def edges(self, n: int) -> np.ndarray:
        e = np.empty(n + 1)
        e[0], e[n] = -np.inf, np.inf
        for i in range(1, n):
            u = i / n
            e[i] = sopt.brentq(
                lambda v: self.cdf(v) - u, -self.sep - 40, self.sep + 40, xtol=1e-15, rtol=8.9e-16
            )
        return e

    def to_dict(self):
        return {"kind": self.kind, "sep": self.sep, "w": self.w, "cut": self.cut}


class RadialTAxis:
    """Radius of a 2-D t_nu(0, I): density r (1 + r^2/nu)^{-nu/2-1}, cdf 1 - (1 + r^2/nu)^{-nu/2}."""

    kind = "t_radial"

    def __init__(self, nu: float, r_max: float):
        self.nu = float(nu)
        self.lo, self.hi = 0.0, float(r_max)

    def density(self, r):
        return r * (1.0 + r * r / self.nu) ** (-self.nu / 2 - 1)

    def surv(self, r):
        return (1.0 + np.asarray(r) ** 2 / self.nu) ** (-self.nu / 2)

    def cdf(self, r):
        return 1.0 - self.surv(r)

    def mass(self, lo, hi):
        return self.surv(lo) - self.surv(hi)

    def edges(self, n: int) -> np.ndarray:
        u = np.arange(n + 1) / n
        with np.errstate(divide="ignore"):
            e = np.sqrt(self.nu * ((1.0 - u) ** (-2.0 / self.nu) - 1.0))
        e[-1] = np.inf
        return e

    def to_dict(self):
        return {"kind": self.kind, "nu": self.nu, "r_max": self.hi}


class AngleAxis:
    kind = "angle"

    def __init__(self):
        self.lo, self.hi = -math.pi, math.pi

    def density(self, v):
        return np.full_like(np.asarray(v, dtype=np.float64), 1.0 / (2.0 * math.pi))

    def mass(self, lo, hi):
        return (np.asarray(hi) - np.asarray(lo)) / (2.0 * math.pi)

    def edges(self, n: int) -> np.ndarray:
        return -math.pi + 2.0 * math.pi * np.arange(n + 1) / n

    def to_dict(self):
        return {"kind": self.kind}


@dataclass(frozen=True)
class QuadSpec:
    """Quadrature resolution and truncation.

    h_x limits spacing in kernel bandwidths; cap_out/cap_in limit panel lengths
    in latent coordinates. q_out/q_in are points per panel, reduced on short
    panels to at least q_min. tail_mass is the truncation per unbounded end.
    For Student-t, r_max bounds the radius and cap_radial grows as r/sqrt(nu)
    away from the origin. The funnel uses panels of width funnel_h_z in
    y/sigma_y, with funnel_q_y points per panel.
    """

    h_x: float
    q_out: int
    q_in: int
    cap_out: float
    cap_in: float
    tail_mass: float
    r_max: float
    funnel_h_z: float = 0.25
    funnel_q_y: int = 6
    q_min: int = 2
    cap_radial: float = 0.25

    def to_dict(self):
        return dict(self.__dict__)


# Target integration
def _kscale(kernel) -> np.ndarray:
    return kernel.scales * kernel.ell


def _blocks(n: int, width: int, block_bytes: int):
    rows = max(1, block_bytes // (8 * 4 * max(1, width)))
    for i0 in range(0, n, rows):
        yield i0, min(n, i0 + rows)


class LatentIntegrator:
    """Nested Gauss-Legendre quadrature in latent product coordinates.

    X has shape (N, 2), w contains the pi weights, cell maps nodes to cells,
    and masses contains the K exact cell masses. Kernel sections decay in the
    tails; moment constraints use the exact masses despite node truncation.
    """

    def __init__(self, family: "LatentFamily", n1: int, n2: int, spec: QuadSpec):
        self.family, self.n1, self.n2, self.spec = family, int(n1), int(n2), spec
        self.K = self.n1 * self.n2
        self.kernel = family.kernel
        outer, inner = family.outer(spec), family.inner(spec)
        oe, ie = outer.edges(self.n1), inner.edges(self.n2)
        self.outer_edges, self.inner_edges = oe, ie
        oe_t, ie_t = np.clip(oe, outer.lo, outer.hi), np.clip(ie, inner.lo, inner.hi)
        if (np.diff(oe_t) <= 0).any() or (np.diff(ie_t) <= 0).any():
            raise ValueError("truncation removes a whole cell; enlarge the truncation range")
        jac_out = family.jac_out()
        us, vs, ws, cs = [], [], [], []
        for k in range(self.n1):
            lo, hi = oe_t[k], oe_t[k + 1]
            if family.fam == "student_t":  # the density's own scale grows like r / sqrt(nu)
                u, wu = graded_rule(
                    lo,
                    hi,
                    lambda r: min(
                        spec.cap_radial * max(1.0, r / math.sqrt(outer.nu)), spec.h_x / jac_out
                    ),
                    spec.q_out,
                )
            else:
                u, wu = panel_rule(lo, hi, spec.cap_out, jac_out, spec.h_x, spec.q_out, spec.q_min)
            wu = wu * outer.density(u)
            for ui, wui in zip(u, wu):
                jin = family.jac_in(ui)
                for li in range(self.n2):
                    lo2, hi2 = ie_t[li], ie_t[li + 1]
                    v, wv = panel_rule(lo2, hi2, spec.cap_in, jin, spec.h_x, spec.q_in, spec.q_min)
                    us.append(np.full(v.shape, ui))
                    vs.append(v)
                    ws.append(wui * wv * inner.density(v))
                    cs.append(np.full(v.shape, k * self.n2 + li, dtype=np.int64))
        u = np.concatenate(us)
        v = np.concatenate(vs)
        w = np.concatenate(ws)
        cell = np.concatenate(cs)
        X = family.to_x(u, v)
        assert X.shape == (u.shape[0], 2)
        if not (np.isfinite(X).all() and np.isfinite(w).all() and (w >= 0).all()):
            raise FloatingPointError("bad quadrature nodes/weights")
        self.X = torch.as_tensor(X, dtype=F64)
        self.w = torch.as_tensor(w, dtype=F64)
        self.cell = torch.as_tensor(cell, dtype=torch.int64)
        # Compare node weights with the exact truncated cell masses.
        mo, mi = outer.mass(oe[:-1], oe[1:]), inner.mass(ie[:-1], ie[1:])
        self.masses = np.outer(mo, mi).ravel()
        mo_t, mi_t = outer.mass(oe_t[:-1], oe_t[1:]), inner.mass(ie_t[:-1], ie_t[1:])
        trunc_masses = np.outer(mo_t, mi_t).ravel()
        quad_masses = np.bincount(cell, weights=w, minlength=self.K)
        self.info = {
            "n_nodes": int(u.shape[0]),
            "spec": spec.to_dict(),
            "outer": outer.to_dict(),
            "inner": inner.to_dict(),
            "mass_exact_minus_uniform_max": float(np.abs(self.masses - 1.0 / self.K).max()),
            "truncated_mass_total": float(1.0 - trunc_masses.sum()),
            "quad_cell_mass_err_max": float(np.abs(quad_masses - trunc_masses).max()),
        }

    def cell_integrals(
        self, P: torch.Tensor, block_bytes: int = DEFAULT_BLOCK_BYTES
    ) -> torch.Tensor:
        """(n_P, K): int_{A_j} k(p, x) dpi(x)."""
        out = torch.zeros((P.shape[0], self.K), dtype=F64)
        N = self.X.shape[0]
        for i0, i1 in _blocks(P.shape[0], N, block_bytes):
            Kb = self.kernel(P[i0:i1], self.X) * self.w[None, :]
            out[i0:i1].index_add_(1, self.cell, Kb)
        return out

    def mean(self, P: torch.Tensor) -> torch.Tensor:
        return self.cell_integrals(P).sum(1)

    def cell_gram(self, block_bytes: int = DEFAULT_BLOCK_BYTES) -> torch.Tensor:
        """(K, K): int_{A_j} int_{A_l} k dpi dpi."""
        N = self.X.shape[0]
        G = torch.zeros((self.K, self.K), dtype=F64)
        for i0, i1 in _blocks(N, N, block_bytes):
            Kb = self.kernel(self.X[i0:i1], self.X) * self.w[None, :]
            T = torch.zeros((i1 - i0, self.K), dtype=F64).index_add_(1, self.cell, Kb)
            G.index_add_(0, self.cell[i0:i1], self.w[i0:i1, None] * T)
        G = 0.5 * (G + G.T)
        return G

    def cell_gram_lowrank(
        self, feats: NystromFeatures, block_bytes: int = DEFAULT_BLOCK_BYTES
    ) -> tuple[torch.Tensor, np.ndarray, dict]:
        """G_lr = H H^T, H_j = sum_{a in A_j} w_a f(x_a), f the Nystrom features of ``feats``.

        The residual kernel k_r = k - f.f is positive semi-definite with |k_r(x, y)| <=
        sqrt(r(x) r(y)), r = k(x, x) - ||f(x)||^2, so for every coefficient vector a
            0 <= a^T (G - G_lr) a <= (sum_j |a_j| s_j)^2,   s_j = sum_{a in A_j} w_a r(x_a)^{1/2}
        (G the exact node-quadrature cell Gram). Returns (G_lr, s, info)."""
        N = self.X.shape[0]
        H = torch.zeros((self.K, feats.rank), dtype=F64)
        s = torch.zeros(self.K, dtype=F64)
        r_min, r_max, r_mean = math.inf, -math.inf, 0.0
        rows = max(1, block_bytes // (8 * 4 * max(1, feats.n_landmarks)))
        for i0 in range(0, N, rows):
            Xb, wb, cb = self.X[i0 : i0 + rows], self.w[i0 : i0 + rows], self.cell[i0 : i0 + rows]
            F = feats(Xb)
            r = self.kernel.diag(Xb) - (F * F).sum(1)
            r_min, r_max = min(r_min, float(r.min())), max(r_max, float(r.max()))
            r_mean += float((wb * r.clamp_min(0.0)).sum())
            H.index_add_(0, cb, wb[:, None] * F)
            s.index_add_(0, cb, wb * torch.sqrt(r.clamp_min(0.0)))
        if r_min < -1e-9:
            raise FloatingPointError(f"negative omitted energy {r_min}")
        G = H @ H.T
        info = {
            "rank": int(feats.rank),
            "n_landmarks": int(feats.n_landmarks),
            "omitted_energy_weighted_mean": r_mean,
            "omitted_energy_max": r_max,
            "s_sum": float(s.sum()),
        }
        return 0.5 * (G + G.T), s.numpy(), info

    def second_moment(
        self, Z: torch.Tensor, block_bytes: int = DEFAULT_BLOCK_BYTES
    ) -> torch.Tensor:
        """(L, L): int k(z_a, x) k(z_b, x) dpi(x)."""
        L = Z.shape[0]
        P = torch.zeros((L, L), dtype=F64)
        N = self.X.shape[0]
        cols = max(1, block_bytes // (8 * 4 * max(1, L)))
        for j0 in range(0, N, cols):
            j1 = min(N, j0 + cols)
            Kb = self.kernel(Z, self.X[j0:j1])
            P += (Kb * self.w[None, j0:j1]) @ Kb.T
        return 0.5 * (P + P.T)

    def cell_index(self, x: np.ndarray) -> np.ndarray:
        u, v = self.family.to_latent(x)
        k = np.searchsorted(self.outer_edges, u, side="right") - 1
        li = np.searchsorted(self.inner_edges, v, side="right") - 1
        if (k < 0).any() or (k >= self.n1).any() or (li < 0).any() or (li >= self.n2).any():
            raise ValueError("point outside the cell grid")
        return k * self.n2 + li


class LatentFamily:
    """Product-coordinate maps for Gaussian, mixture and Student-t targets."""

    def __init__(self, target: Target, kernel):
        if target.d != 2:
            raise ValueError(f"population constructions are 2-D, got d={target.d}")
        if kernel.d != 2:
            raise ValueError(f"kernel d={kernel.d}")
        self.target, self.kernel = target, kernel
        self.ks = _kscale(kernel)
        fam = target.family
        if fam == "gauss":
            self.A = target.rotation * np.sqrt(target.lam)[None, :]
        elif fam == "mixture":
            self.A = target.rotation.copy()
        elif fam != "student_t":
            raise ValueError(f"no latent product structure for {fam}; use FunnelIntegrator")
        self.fam = fam

    def outer(self, spec: QuadSpec):
        if self.fam == "gauss":
            return NormalAxis(spec.tail_mass)
        if self.fam == "mixture":
            return MixtureAxis(self.target.sep, self.target.w, spec.tail_mass)
        return RadialTAxis(self.target.nu, spec.r_max)

    def inner(self, spec: QuadSpec):
        return AngleAxis() if self.fam == "student_t" else NormalAxis(spec.tail_mass)

    def to_x(self, u, v) -> np.ndarray:
        if self.fam == "student_t":
            return np.stack([u * np.cos(v), u * np.sin(v)], axis=1)
        return np.stack([u, v], axis=1) @ self.A.T

    def to_latent(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if self.fam == "student_t":
            return np.hypot(x[:, 0], x[:, 1]), np.arctan2(x[:, 1], x[:, 0])
        z = np.linalg.solve(self.A, x.T).T
        return z[:, 0], z[:, 1]

    def jac_out(self) -> float:
        if self.fam == "student_t":
            return 1.0 / self.ks.min()
        return float(np.linalg.norm(self.A[:, 0] / self.ks))

    def jac_in(self, u: float) -> float:
        if self.fam == "student_t":
            return float(u) / self.ks.min()
        return float(np.linalg.norm(self.A[:, 1] / self.ks))


# Funnel integrals
class FunnelIntegrator:
    """Semi-analytic RBF integration for y ~ N(0, sigma^2), x1 | y ~ N(0, e^y).

    y-slabs follow normal quantiles. Each slab's x1 boundaries are conditional
    quantiles from a fixed Gauss-Legendre rule (panel width 0.01 in y/sigma,
    8 points per panel). Reusing these boundaries and masses keeps the cells
    identical across quadrature levels. Integrals in x1 are analytic.
    """

    DEF_H_Z, DEF_Q_Y, DEF_TAIL = 0.01, 8, 1e-17

    def __init__(self, target: FunnelTarget, kernel, n1: int, n2: int, spec: QuadSpec):
        if not isinstance(kernel, ScaledRBF):
            raise TypeError("the funnel integrator is semi-analytic for ScaledRBF only")
        if target.d != 2:
            raise ValueError("2-D funnel only")
        self.target, self.kernel, self.spec = target, kernel, spec
        self.n1, self.n2, self.K = int(n1), int(n2), int(n1) * int(n2)
        ks = _kscale(kernel)
        self.by, self.b1 = float(ks[0]), float(ks[1])
        self.sig = target.sigma_y
        cut = float(-sps.ndtri(spec.tail_mass))
        self.ylo, self.yhi = -self.sig * cut, self.sig * cut
        e = self.sig * sps.ndtri(np.arange(self.n1 + 1) / self.n1)
        self.slab_edges = e
        self.slab_edges_t = np.clip(e, self.ylo, self.yhi)
        # Fix cell boundaries independently of the integration level.
        def_cut = self.sig * float(-sps.ndtri(self.DEF_TAIL))
        def_edges = np.clip(e, -def_cut, def_cut)
        def_rules = [
            self._slab_rule(k, self.DEF_H_Z, self.DEF_Q_Y, def_edges) for k in range(self.n1)
        ]
        self.bounds = np.empty((self.n1, self.n2 + 1))
        for k, (yk, wk) in enumerate(def_rules):
            self.bounds[k] = self._slab_quantiles(yk, wk)
        self.masses = np.concatenate([self._slab_masses(k, *def_rules[k]) for k in range(self.n1)])
        self.rules = [
            self._slab_rule(k, spec.funnel_h_z, spec.funnel_q_y, q_min=spec.q_min)
            for k in range(self.n1)
        ]
        lvl_masses = np.concatenate([self._slab_masses(k, *self.rules[k]) for k in range(self.n1)])
        n_y = sum(r[0].shape[0] for r in self.rules)
        self.info = {
            "n_y_nodes": int(n_y),
            "spec": spec.to_dict(),
            "y_cut": self.yhi,
            "by": self.by,
            "b1": self.b1,
            "mass_def_minus_uniform_max": float(np.abs(self.masses - 1.0 / self.K).max()),
            "mass_level_minus_def_max": float(np.abs(lvl_masses - self.masses).max()),
            "truncated_mass_total": float(2 * sps.ndtr(-cut)),
        }

    def _slab_rule(
        self, k: int, h_z: float, q: int, edges: np.ndarray | None = None, q_min: int | None = None
    ):
        """Gauss-Legendre rule on y-slab k, with panel width <= h_z in y/sigma.

        If q_min is set, short panels use fewer than q points, down to q_min.
        """
        edges = self.slab_edges_t if edges is None else edges
        lo, hi = edges[k], edges[k + 1]
        n_sub = max(1, int(math.ceil((hi - lo) / (self.sig * h_z))))
        qq = q
        if q_min is not None:
            qq = max(q_min, min(q, int(math.ceil(q * (hi - lo) / n_sub / (self.sig * h_z)))))
        y, w = composite_gl(lo, hi, n_sub, qq)
        w = w * np.exp(-0.5 * (y / self.sig) ** 2) / (self.sig * SQRT2PI)
        return y, w

    def _slab_quantiles(self, y, w) -> np.ndarray:
        tot = w.sum()
        sd = np.exp(0.5 * y)
        b = np.empty(self.n2 + 1)
        b[0], b[-1] = -np.inf, np.inf
        for i in range(1, self.n2):
            u = i / self.n2
            if 2 * i == self.n2:
                b[i] = 0.0
                continue
            f = lambda t: (w * sps.ndtr(t / sd)).sum() / tot - u  # noqa: E731
            span = 40.0 * sd.max()
            b[i] = sopt.brentq(f, -span, span, xtol=1e-15 * sd.min(), rtol=8.9e-16)
        # Enforce the symmetry of the conditional law.
        b[1:-1] = 0.5 * (b[1:-1] - b[1:-1][::-1])
        return b

    def _slab_masses(self, k: int, y, w) -> np.ndarray:
        b = self.bounds[k]
        sd = np.exp(0.5 * y)
        lo = b[:-1][None, :] / sd[:, None]
        hi = b[1:][None, :] / sd[:, None]
        return (w[:, None] * phi_interval(lo, hi)).sum(0)

    def _one_kernel_x1(self, c: np.ndarray, v: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """int_lo^hi exp(-(x - c)^2/(2 b1^2)) N(x; 0, v) dx for c (P,1,1), v (1,Y,1), lo/hi (1,1,B)."""
        b2 = self.b1**2
        pref = self.b1 / np.sqrt(b2 + v) * np.exp(-(c * c) / (2.0 * (b2 + v)))
        m = c * v / (v + b2)
        s = np.sqrt(v * b2 / (v + b2))
        return pref * phi_interval((lo - m) / s, (hi - m) / s)

    def cell_integrals(
        self, P: torch.Tensor, block_bytes: int = DEFAULT_BLOCK_BYTES
    ) -> torch.Tensor:
        Pn = P.detach().cpu().numpy()
        out = np.zeros((Pn.shape[0], self.K))
        for k in range(self.n1):
            y, w = self.rules[k]
            b = self.bounds[k]
            width = y.shape[0] * (self.n2 + 1)
            for i0, i1 in _blocks(Pn.shape[0], width, block_bytes):
                py, px = Pn[i0:i1, 0], Pn[i0:i1, 1]
                ky = np.exp(-((y[None, :] - py[:, None]) ** 2) / (2.0 * self.by**2)) * w[None, :]
                inner = self._one_kernel_x1(
                    px[:, None, None],
                    np.exp(y)[None, :, None],
                    b[None, None, :-1],
                    b[None, None, 1:],
                )  # (P, Y, n2)
                out[i0:i1, k * self.n2 : (k + 1) * self.n2] = np.einsum("py,pyl->pl", ky, inner)
        return torch.as_tensor(out, dtype=F64)

    def mean(self, P: torch.Tensor) -> torch.Tensor:
        Pn = P.detach().cpu().numpy()
        y = np.concatenate([r[0] for r in self.rules])
        w = np.concatenate([r[1] for r in self.rules])
        v = np.exp(y)
        b2 = self.b1**2
        out = np.zeros(Pn.shape[0])
        for i0, i1 in _blocks(Pn.shape[0], y.shape[0], DEFAULT_BLOCK_BYTES):
            py, px = Pn[i0:i1, 0:1], Pn[i0:i1, 1:2]
            ky = np.exp(-((y[None, :] - py) ** 2) / (2.0 * self.by**2))
            x1 = self.b1 / np.sqrt(b2 + v[None, :]) * np.exp(-(px * px) / (2.0 * (b2 + v[None, :])))
            out[i0:i1] = (w[None, :] * ky * x1).sum(1)
        return torch.as_tensor(out, dtype=F64)

    def cell_gram(self) -> torch.Tensor:
        n2 = self.n2
        G = np.zeros((self.K, self.K))
        for k in range(self.n1):
            yk, wk = self.rules[k]
            vk = np.exp(yk)
            Bk = self.bounds[k]
            for kk in range(k, self.n1):
                yl, wl = self.rules[kk]
                vl = np.exp(yl)
                Bl = self.bounds[kk]
                blk = np.zeros((n2, n2))
                rows = max(1, int(4e6 // (yl.shape[0] * (n2 + 1) ** 2)))
                for i0 in range(0, yk.shape[0], rows):
                    blk += self._gram_block(
                        yk[i0 : i0 + rows],
                        wk[i0 : i0 + rows],
                        vk[i0 : i0 + rows],
                        Bk,
                        yl,
                        wl,
                        vl,
                        Bl,
                    )
                G[k * n2 : (k + 1) * n2, kk * n2 : (kk + 1) * n2] = blk
                if kk != k:
                    G[kk * n2 : (kk + 1) * n2, k * n2 : (k + 1) * n2] = blk.T
        G = 0.5 * (G + G.T)
        return torch.as_tensor(G, dtype=F64)

    def _gram_block(self, yk, wk, vk, Bk, yl, wl, vl, Bl) -> np.ndarray:
        """sum over y-node pairs of w w' k_y(y - y') int_{rect} k_x1 N N dx1 dx1' -> (n2, n2)."""
        b2 = self.b1**2
        V, Vp = vk[:, None], vl[None, :]
        den = b2 + V + Vp
        a1, a2 = b2 + V, b2 + Vp
        norm = (
            (self.b1 / np.sqrt(den))
            * np.exp(-((yk[:, None] - yl[None, :]) ** 2) / (2.0 * self.by**2))
            * wk[:, None]
            * wl[None, :]
        )
        s1 = np.sqrt(V * a2 / den)
        s2 = np.sqrt(Vp * a1 / den)
        rho = np.sqrt(V * Vp / (a1 * a2))
        sq = self.b1 * np.sqrt(den / (a1 * a2))
        H = Bk[None, None, :, None] / s1[:, :, None, None]
        Kc = Bl[None, None, None, :] / s2[:, :, None, None]
        F = phi2(H, Kc, rho[:, :, None, None], sq[:, :, None, None])
        Rr = F[..., 1:, 1:] - F[..., :-1, 1:] - F[..., 1:, :-1] + F[..., :-1, :-1]
        return np.einsum("ij,ijlm->lm", norm, Rr)

    def second_moment(self, Z: torch.Tensor) -> torch.Tensor:
        """P_ab = int k(z_a, x) k(z_b, x) dpi, with analytic integration in x1.

        For c = z_x1, cbar = (c_a + c_b)/2 and tau = b1^2/2 + e^y, the x1 integral is
            exp(-(c_a-c_b)^2/(4 b1^2)) sqrt(b1^2/(2 tau)) exp(-cbar^2/(2 tau)).
        On the landmark lattice c = k delta, the y-sum depends only on the two
        landmark rows and k_a + k_b. This reduces it to one matrix product and
        a gather. Raises if the x1 coordinates do not share a lattice.
        """
        Zn = Z.detach().cpu().numpy()
        y = np.concatenate([r[0] for r in self.rules])
        w = np.concatenate([r[1] for r in self.rules])
        b2h = self.b1**2 / 2.0
        tau = b2h + np.exp(y)
        rows, r_idx = np.unique(Zn[:, 0], return_inverse=True)
        zx = Zn[:, 1]
        nz = np.abs(zx[zx != 0])
        delta = float(nz.min()) if nz.size else 1.0
        kk = np.rint(zx / delta).astype(np.int64)
        if np.abs(zx - kk * delta).max() > 1e-9 * max(1.0, np.abs(zx).max()):
            raise ValueError("landmark x1 values are not on a common lattice")
        smin, smax = 2 * int(kk.min()), 2 * int(kk.max())
        cbar = 0.5 * delta * np.arange(smin, smax + 1)
        Ky = np.exp(-((rows[:, None] - y[None, :]) ** 2) / (2.0 * self.by**2))  # (R, Y)
        Gy = np.sqrt(b2h / tau)[:, None] * np.exp(-(cbar[None, :] ** 2) / (2.0 * tau[:, None]))
        RR = (Ky[:, None, :] * Ky[None, :, :] * w[None, None, :]).reshape(-1, y.shape[0])
        T = torch.as_tensor(RR @ Gy)  # (R*R, n_sums)
        nR = rows.shape[0]
        ri = torch.as_tensor(r_idx)
        ks = torch.as_tensor(kk - kk.min())
        flat = (ri[:, None] * nR + ri[None, :]) * T.shape[1] + (
            ks[:, None] + ks[None, :] + 2 * int(kk.min()) - smin
        )
        P = T.reshape(-1)[flat]
        zt = torch.as_tensor(zx)
        P = P * torch.exp(-((zt[:, None] - zt[None, :]) ** 2) / (4.0 * self.b1**2))
        return 0.5 * (P + P.T)

    def second_moment_direct(self, Z: torch.Tensor) -> torch.Tensor:
        """Reference implementation (loop over y-nodes, no lattice assumption; slow)."""
        Zn = Z.detach().cpu().numpy()
        y = np.concatenate([r[0] for r in self.rules])
        w = np.concatenate([r[1] for r in self.rules])
        v = np.exp(y)
        b2h = self.b1**2 / 2.0
        zy, zx = torch.as_tensor(Zn[:, 0]), torch.as_tensor(Zn[:, 1])
        cb2 = (0.5 * (zx[:, None] + zx[None, :])) ** 2
        P = torch.zeros((Zn.shape[0], Zn.shape[0]), dtype=F64)
        ky = torch.exp(-((torch.as_tensor(y)[None, :] - zy[:, None]) ** 2) / (2.0 * self.by**2))
        for i in range(y.shape[0]):
            tau = b2h + v[i]
            g = (math.sqrt(b2h) / math.sqrt(tau)) * torch.exp(-cb2 / (2.0 * tau))
            a = ky[:, i]
            P += w[i] * (a[:, None] * a[None, :]) * g
        P = P * torch.exp(-((zx[:, None] - zx[None, :]) ** 2) / (4.0 * self.b1**2))
        return 0.5 * (P + P.T)

    def cell_index(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        k = np.searchsorted(self.slab_edges, x[:, 0], side="right") - 1
        if (k < 0).any() or (k >= self.n1).any():
            raise ValueError("point outside the slab grid")
        li = np.empty(x.shape[0], dtype=np.int64)
        for kk in np.unique(k):
            sel = k == kk
            li[sel] = np.searchsorted(self.bounds[kk], x[sel, 1], side="right") - 1
        if (li < 0).any() or (li >= self.n2).any():
            raise ValueError("point outside the x1 boundaries")
        return k * self.n2 + li


# Tapered Matern-3/2 approximation for the funnel
MATERN32_SHAPE = 1.5  # (1 + sqrt3 r) e^{-sqrt3 r} = int exp(-r^2/(2 lam)) Gamma(lam; 3/2, rate 3/2)


def matern32_scale_mixture(
    h: float, log_lam0: float, tau: float, log_lam_max: float, offset: float = 0.0,
    w_min: float = 1e-17,
):  # fmt: skip
    """Trapezoid rule in u = log(lambda) for a tapered Matern-3/2 mixture.

        k(r) = int exp(-r^2/(2 lambda)) Phi((log lambda-log_lam0)/tau)
                   Gamma(lambda; 3/2, rate 3/2) dlambda.

    Without the taper, k(r) = (1 + sqrt(3) r) exp(-sqrt(3) r). The taper
    removes narrow Gaussian components, so this kernel approximates Matern-3/2.
    Nodes start at log_lam0 - 8 tau + offset with spacing h; weights below
    w_min are dropped. Returns (lambda, weights).
    """
    if not (0.0 <= offset < h):
        raise ValueError(f"offset must be in [0, h), got {offset}")
    u = np.arange(log_lam0 - 8.0 * tau + offset, log_lam_max + 1e-12, h)
    lam = np.exp(u)
    a = MATERN32_SHAPE
    logdens = a * math.log(a) - math.lgamma(a) + (a - 1.0) * u - a * lam
    w = h * np.exp(logdens + u) * sps.ndtr((u - log_lam0) / tau)
    keep = w >= w_min
    return lam[keep], w[keep]


def matern32(r: np.ndarray) -> np.ndarray:
    a = math.sqrt(3.0) * np.asarray(r, dtype=np.float64)
    return (1.0 + a) * np.exp(-a)


def mixture_kernel_values(lam: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (w[None, :] * np.exp(-(np.asarray(r)[:, None] ** 2) / (2.0 * lam[None, :]))).sum(1)


class ScaledGaussMixture(_ScaledStationaryKernel):
    """k(x, y) = sum_i w_i exp(-||(x - y)/(s ell)||^2 / (2 lam_i)): a Gaussian scale mixture (here
    a truncated Matern-3/2 surrogate); k(x, x) = sum_i w_i."""

    kind = "gauss_mixture"

    def __init__(self, scales, ell: float, lam: np.ndarray, w: np.ndarray, label: str = ""):
        super().__init__(scales, ell)
        lam, w = np.asarray(lam, dtype=np.float64), np.asarray(w, dtype=np.float64)
        if lam.ndim != 1 or lam.shape != w.shape or (lam <= 0).any() or (w <= 0).any():
            raise ValueError("lam, w must be positive 1-D arrays of equal length")
        self.lam, self.w, self.label = lam, w, label
        self.kappa2 = float(w.sum())

    def from_sqdist(self, d2):
        out = torch.zeros_like(d2)
        for li, wi in zip(self.lam, self.w):
            out += wi * torch.exp(-d2 / (2.0 * li))
        return out

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        self._check(X, "X")
        return torch.full((X.shape[0],), self.kappa2, dtype=F64, device=X.device)

    def component(self, i: int) -> ScaledRBF:
        return ScaledRBF(self.scales, self.ell * math.sqrt(float(self.lam[i])))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "scales": self.scales.tolist(),
            "ell": self.ell,
            "kappa2": self.kappa2,
            "n_components": int(self.lam.shape[0]),
            "lam_min": float(self.lam.min()),
            "lam_max": float(self.lam.max()),
        }


def _pool_init():
    torch.set_num_threads(1)


def _component_call(args):
    comp, method, P = args
    if method == "cell_gram":
        return comp.cell_gram().numpy()
    if method == "cell_integrals":
        return comp.cell_integrals(P).numpy()
    if method == "mean":
        return comp.mean(P).numpy()
    raise ValueError(method)


@dataclass
class NodeSpec:
    """Node rule for the second moment of FunnelMixtureIntegrator: y panels <= h_y (y units) with
    breakpoints at the landmark rows, q_y GL nodes; x1 panels <= min(h_x b1, cap sd(y)) with
    breakpoints on the landmark x1 lattice, q_x nodes; x1 restricted to |x1| <= max|z_x1| +
    margin_bw b1 and to the conditional range |x1| <= s_cut sd(y)."""

    h_y: float
    q_y: int
    h_x: float
    q_x: int
    cap: float
    margin_bw: float
    s_cut: float

    def to_dict(self):
        return dict(self.__dict__)


class FunnelMixtureIntegrator:
    """Funnel integration for sum_i w_i RBF(ell sqrt(lambda_i)).

    Cells and masses match FunnelIntegrator. Cell integrals, means and Gram
    matrices are weighted sums of its semi-analytic RBF integrals. The second
    moment uses node quadrature near the landmarks; the omitted region
    contributes at most k(margin)^2 per entry.
    """

    def __init__(self, target: FunnelTarget, mix: ScaledGaussMixture, n1: int, n2: int,
                 spec: QuadSpec, node: NodeSpec, workers: int = 1):  # fmt: skip
        geo = FunnelIntegrator(target, ScaledRBF(mix.scales, mix.ell), n1, n2, spec)
        self.geo, self.mix, self.node, self.workers = geo, mix, node, int(workers)
        self.target, self.spec = target, spec
        self.n1, self.n2, self.K = geo.n1, geo.n2, geo.K
        self.masses = geo.masses
        self.comps = []
        for i in range(mix.lam.shape[0]):
            c = copy.copy(geo)
            c.kernel = mix.component(i)
            c.by, c.b1 = (float(v) for v in _kscale(c.kernel))
            self.comps.append(c)
        self.info = {
            **geo.info,
            "mixture": mix.to_dict(),
            "node_spec": node.to_dict(),
            "workers": self.workers,
        }

    def _sum(self, method: str, P=None) -> np.ndarray:
        """Sum weighted component integrals in a fixed order across worker counts."""
        jobs = [(c, method, P) for c in self.comps]
        acc = 0.0
        if self.workers <= 1:
            for i, job in enumerate(jobs):
                acc = acc + self.mix.w[i] * _component_call(job)
            return acc
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(self.workers, ctx, _pool_init) as ex:
            for i, part in enumerate(ex.map(_component_call, jobs)):
                acc = acc + self.mix.w[i] * part
        return acc

    def cell_integrals(self, P: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self._sum("cell_integrals", P.cpu()), dtype=F64)

    def mean(self, P: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self._sum("mean", P.cpu()), dtype=F64)

    def cell_gram(self) -> torch.Tensor:
        G = self._sum("cell_gram")
        return torch.as_tensor(0.5 * (G + G.T), dtype=F64)

    def cell_index(self, x: np.ndarray) -> np.ndarray:
        return self.geo.cell_index(x)

    def second_moment_nodes(self, Z: torch.Tensor):
        """(X (N, 2), w (N,)) restricted to the landmark neighbourhood of Z."""
        ns, geo = self.node, self.geo
        Zn = Z.detach().cpu().numpy()
        b1 = float(_kscale(self.mix)[1])
        rows = np.unique(Zn[:, 0])
        zx = Zn[:, 1]
        nz = np.abs(zx[zx != 0])
        delta = float(nz.min()) if nz.size else ns.h_x * b1
        kk = np.rint(zx / delta)
        if np.abs(zx - kk * delta).max() > 1e-9 * max(1.0, np.abs(zx).max()):
            raise ValueError("landmark x1 values are not on a common lattice")
        xD = float(np.abs(zx).max()) + ns.margin_bw * b1
        ylo, yhi = geo.ylo, geo.yhi
        bp = np.unique(np.concatenate([[ylo, yhi], rows[(rows > ylo) & (rows < yhi)]]))
        ys, wys = [], []
        for lo, hi in zip(bp[:-1], bp[1:]):
            y, w = composite_gl(lo, hi, max(1, int(math.ceil((hi - lo) / ns.h_y))), ns.q_y)
            ys.append(y)
            wys.append(w)
        y, wy = np.concatenate(ys), np.concatenate(wys)
        wy = wy * np.exp(-0.5 * (y / geo.sig) ** 2) / (geo.sig * SQRT2PI)
        Xs, Ws = [], []
        for yi, wyi in zip(y, wy):
            sd = math.exp(0.5 * yi)
            X1 = min(ns.s_cut * sd, xD)
            lat = delta * np.arange(math.ceil(-X1 / delta), math.floor(X1 / delta) + 1)
            bx = np.unique(np.concatenate([[-X1, X1], lat[(lat > -X1) & (lat < X1)]]))
            plen = min(ns.h_x * b1, ns.cap * sd)
            for lo, hi in zip(bx[:-1], bx[1:]):
                x, wx = composite_gl(lo, hi, max(1, int(math.ceil((hi - lo) / plen))), ns.q_x)
                Xs.append(np.stack([np.full(x.shape, yi), x], 1))
                Ws.append(wyi * wx * np.exp(-0.5 * (x / sd) ** 2) / (sd * SQRT2PI))
        X, w = np.concatenate(Xs), np.concatenate(Ws)
        info = {"n_nodes": int(X.shape[0]), "n_y": int(y.shape[0]), "x1_lattice": delta,
                "x1_half_width": xD, "node_mass": float(w.sum())}  # fmt: skip
        return torch.as_tensor(X, dtype=F64), torch.as_tensor(w, dtype=F64), info

    def second_moment(self, Z: torch.Tensor, block_bytes: int = DEFAULT_BLOCK_BYTES):
        X, w, info = self.second_moment_nodes(Z)
        self.info["second_moment_nodes"] = info
        L = Z.shape[0]
        P = torch.zeros((L, L), dtype=F64)
        cols = max(1, block_bytes // (8 * 4 * max(1, L)))
        for j0 in range(0, X.shape[0], cols):
            Kb = self.mix(Z, X[j0 : j0 + cols])
            P += (Kb * w[None, j0 : j0 + cols]) @ Kb.T
        return 0.5 * (P + P.T)

    def mean_nodes(self, Z: torch.Tensor) -> torch.Tensor:
        """Mean embedding from node quadrature, for comparison with the RBF integrals."""
        X, w, _ = self.second_moment_nodes(Z)
        out = torch.zeros(Z.shape[0], dtype=F64)
        cols = max(1, DEFAULT_BLOCK_BYTES // (8 * 4 * max(1, Z.shape[0])))
        for j0 in range(0, X.shape[0], cols):
            out += self.mix(Z, X[j0 : j0 + cols]) @ w[j0 : j0 + cols]
        return out


def make_integrator(target: Target, kernel, n1: int, n2: int, spec: QuadSpec):
    if isinstance(target, FunnelTarget):
        return FunnelIntegrator(target, kernel, n1, n2, spec)
    if isinstance(target, (GaussTarget, MixtureTarget, StudentTTarget)):
        return LatentIntegrator(LatentFamily(target, kernel), n1, n2, spec)
    raise TypeError(f"no integrator for {type(target).__name__}")


# Landmarks and covariance spectrum
@dataclass(frozen=True)
class LandmarkSpec:
    """Grid spacing h_L (kernel bandwidths), keep mu_pi(z) >= eta_rel max mu_pi, box half-width
    box_sd (coordinate standard deviations, or radius r_box for student_t, sigma_y units in y for
    funnel), x_cap: max |x1| for funnel rows, rel_cutoff: K_ZZ eigen-truncation. box_bw: extra margin
    (kernel bandwidths) added to the gauss / mixture boxes."""

    h_L: float
    eta_rel: float
    box_sd: float
    r_box: float
    x_cap: float
    rel_cutoff: float = 1e-12
    box_bw: float = 0.0

    def to_dict(self):
        return dict(self.__dict__)


def landmark_candidates(target: Target, kernel, spec: LandmarkSpec) -> np.ndarray:
    ks = _kscale(kernel)
    if isinstance(target, FunnelTarget):
        sig = target.sigma_y
        rows = np.arange(-spec.box_sd * sig, spec.box_sd * sig + 1e-12, spec.h_L * ks[0])
        pts = []
        for yr in rows:
            ext = min(spec.x_cap, spec.box_sd * math.exp(0.5 * yr) + 3.0 * ks[1])
            n = int(math.floor(ext / (spec.h_L * ks[1])))
            x1 = spec.h_L * ks[1] * np.arange(-n, n + 1)
            pts.append(np.stack([np.full(x1.shape, yr), x1], axis=1))
        return np.concatenate(pts)
    if isinstance(target, StudentTTarget):
        half = np.full(2, spec.r_box)
        center = np.zeros(2)
    else:
        center = target.mean()
        if isinstance(target, MixtureTarget):
            cm = target.component_means()
            half = np.abs(cm - center).max(0) + spec.box_sd
        else:
            half = spec.box_sd * target.coordinate_std()
        half = half + spec.box_bw * ks
    axes = []
    for i in range(2):
        n = int(math.floor(half[i] / (spec.h_L * ks[i])))
        axes.append(center[i] + spec.h_L * ks[i] * np.arange(-n, n + 1))
    g0, g1 = np.meshgrid(axes[0], axes[1], indexing="ij")
    return np.stack([g0.ravel(), g1.ravel()], axis=1)


def make_landmarks(target: Target, kernel, integ, spec: LandmarkSpec) -> tuple[torch.Tensor, dict]:
    cand = torch.as_tensor(landmark_candidates(target, kernel, spec), dtype=F64)
    mu = integ.mean(cand)
    keep = mu >= spec.eta_rel * mu.max()
    Z = cand[keep]
    info = {
        "n_candidates": int(cand.shape[0]),
        "n_landmarks": int(Z.shape[0]),
        "mu_max": float(mu.max()),
        "mu_min_kept": float(mu[keep].min()),
        "mu_max_dropped": float(mu[~keep].max()) if (~keep).any() else 0.0,
        "edge_mu_max": _edge_mu(cand.numpy(), mu.numpy(), target),
        "spec": spec.to_dict(),
    }
    return Z, info


def _edge_mu(cand: np.ndarray, mu: np.ndarray, target) -> float:
    """Largest mu_pi on the candidate box boundary (the box must contain the kept region)."""
    if isinstance(target, FunnelTarget):
        ys = np.unique(cand[:, 0])
        edge = np.isin(cand[:, 0], [ys.min(), ys.max()])
        for yr in ys:
            row = cand[:, 0] == yr
            xm = np.abs(cand[row, 1]).max()
            edge |= row & (np.abs(cand[:, 1]) == xm)
    else:
        edge = np.zeros(cand.shape[0], dtype=bool)
        for i in range(2):
            edge |= (cand[:, i] == cand[:, i].min()) | (cand[:, i] == cand[:, i].max())
    return float(mu[edge].max())


@dataclass
class Spectrum:
    """Monitors e_i = sum_a alpha[a, i] phi(Z[a]) (orthonormal), eigenvalues lam (descending)."""

    Z: torch.Tensor
    alpha: torch.Tensor
    lam: np.ndarray
    info: dict = field(default_factory=dict)

    def take(self, m: int) -> torch.Tensor:
        if m > self.alpha.shape[1]:
            raise ValueError(f"only {self.alpha.shape[1]} monitors available, asked for {m}")
        return self.alpha[:, :m]

    def criteria(self, X: torch.Tensor, m: int) -> torch.Tensor:
        """(n, m) values c_i(x) = <e_i, phi(x)> (uncentered)."""
        return self.kernel_fn(X, self.Z) @ self.take(m)


def _covariance_in_features(
    feats: NystromFeatures, Z: torch.Tensor, integ
) -> tuple[torch.Tensor, str]:
    """C = int (f - fbar)(f - fbar)^T dpi in the orthonormal basis of span phi(Z).

    Node-based integrators: features at the quadrature nodes, F = k(X, Z) B (error ~ eps
    sqrt(L / e_i) per entry). Semi-analytic integrators (funnel): B^T P B - fbar fbar^T with
    P = int k(Z, x) k(Z, x)^T dpi (error ~ eps / e_i; use a larger rel_cutoff)."""
    Bm = feats.B
    if hasattr(integ, "X"):
        r = Bm.shape[1]
        S1 = torch.zeros(r, dtype=F64)
        S2 = torch.zeros((r, r), dtype=F64)
        N = integ.X.shape[0]
        rows = max(1, DEFAULT_BLOCK_BYTES // (8 * 4 * max(1, Z.shape[0])))
        for i0 in range(0, N, rows):
            F = feats(integ.X[i0 : i0 + rows])
            wb = integ.w[i0 : i0 + rows]
            S1 += (wb[:, None] * F).sum(0)
            S2 += (F * wb[:, None]).T @ F
        fbar = S1
        C = S2 - torch.outer(fbar, fbar)
        method = "features_at_nodes"
    else:
        muZ = integ.mean(Z)
        P = integ.second_moment(Z)
        fbar = Bm.T @ muZ
        C = Bm.T @ P @ Bm - torch.outer(fbar, fbar)
        method = "second_moment"
    return C, method


def nystrom_spectrum(kernel, Z: torch.Tensor, integ, rel_cutoff: float) -> Spectrum:
    feats = NystromFeatures(kernel, Z, rel_cutoff=rel_cutoff)
    C, method = _covariance_in_features(feats, Z, integ)
    asym = float((C - C.T).abs().max())
    C = 0.5 * (C + C.T)
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0), U.flip(1)
    alpha = feats.B @ U
    Kzz = kernel(Z, Z)
    ortho = float((alpha.T @ Kzz @ alpha - torch.eye(alpha.shape[1], dtype=F64)).abs().max())
    spec = Spectrum(
        Z=Z,
        alpha=alpha,
        lam=ev.numpy().copy(),
        info={
            "method": method,
            "n_landmarks": int(Z.shape[0]),
            "rank": int(feats.rank),
            "rel_cutoff": rel_cutoff,
            "kzz_eig_min": feats.min_eig,
            "kzz_discarded_eig_sum": feats.discarded_eig_sum,
            "C_asym": asym,
            "trace_C": float(ev.sum()),
            "min_eig_C": float(ev[-1]),
            "orthonormality_err": ortho,
        },
    )
    spec.kernel_fn = kernel
    spec.feats = feats
    return spec


def compressed_top_eigs(kernel, spec: Spectrum, m_list, Zp: torch.Tensor, integ, rel_cutoff: float):
    """Estimate lambda_max(P_perp Sigma P_perp) for each monitored dimension m.

    Rayleigh-Ritz on span phi(Z union Zp) gives a lower estimate, converging
    as landmarks are added. Sigma is integrated by integ. Returns {m: (value, info)}.
    """
    Zall = torch.cat([spec.Z, Zp])
    feats = NystromFeatures(kernel, Zall, rel_cutoff=rel_cutoff)
    C, method = _covariance_in_features(feats, Zall, integ)
    C = 0.5 * (C + C.T)
    Eall = feats.B.T @ kernel(Zall, spec.Z) @ spec.take(max(m_list))  # coordinates of e_i
    out = {}
    for m in m_list:
        E = Eall[:, :m]
        gram_err = float((E.T @ E - torch.eye(m, dtype=F64)).abs().max())
        Qe, _ = torch.linalg.qr(E)
        Pp = torch.eye(C.shape[0], dtype=F64) - Qe @ Qe.T
        Cc = Pp @ C @ Pp
        top = float(torch.linalg.eigvalsh(0.5 * (Cc + Cc.T))[-1])
        out[m] = (
            top,
            {"rank": int(feats.rank), "monitor_embedding_gram_err": gram_err, "method": method},
        )
    return out


# Positive perturbations
def moment_rows(spec: Spectrum, m: int, Mcells: torch.Tensor) -> np.ndarray:
    """(m, K): int_{A_j} c_i dpi = alpha_i^T Mcells[:, j]."""
    return (spec.take(m).T @ Mcells).numpy()


@dataclass(frozen=True)
class ConstructionSpec:
    """Search settings for positive perturbations with L2(pi) radius R.

    Try n_dir generalized eigenvectors with both signs. With n_combo > 1,
    Nelder-Mead also searches their unit combinations after positivity scaling.
    """

    R: float
    n_dir: int
    floor: float
    n_combo: int = 0
    rank_tol: float | None = None  # None: raise on dependent rows; else rank-revealing null space
    # A partial eigensolve can change the basis used by the combination search.
    eig_subset: bool = False

    def to_dict(self):
        return dict(self.__dict__)


def construct(G: np.ndarray, masses: np.ndarray, rows: np.ndarray, cs: ConstructionSpec) -> dict:
    """Feasible h = sum_j a_j 1_{A_j}: null(B) generalized eigen-directions scaled for positivity."""
    K = masses.shape[0]
    m = rows.shape[0]
    if G.shape != (K, K) or rows.shape != (m, K):
        raise ValueError(f"shapes G {G.shape}, rows {rows.shape}, K {K}")
    if K <= m + 1:
        raise ValueError(f"need K > m + 1 cells, got K={K}, m={m}")
    Bfull = np.vstack([masses[None, :], rows])
    norms = np.linalg.norm(Bfull, axis=1)
    n_zero_rows = 0
    if cs.rank_tol is not None:
        # Identically zero rows impose no constraint (e.g. odd integrated-out monitors).
        live = norms > 1e-12 * norms[1:].max() if m > 0 else np.ones(1, dtype=bool)
        live[0] = True
        n_zero_rows = int((~live).sum())
        Bfull, norms = Bfull[live], norms[live]
    Bn = Bfull / norms[:, None]
    _, S, Vt = np.linalg.svd(Bn, full_matrices=True)
    cond = float(S[-1] / S[0])
    if cs.rank_tol is None:
        if cond < 1e-12:
            raise np.linalg.LinAlgError(
                f"constraint rows nearly dependent (sigma_min/sigma_max={cond})"
            )
        rank = m + 1
    else:
        rank = int((S > cs.rank_tol * S[0]).sum())
    N = Vt[rank:].T
    assert N.shape == (K, K - rank)
    null_err = float(np.abs(Bn @ N).max())
    D = np.diag(masses)
    Gn = N.T @ G @ N
    Dn = N.T @ D @ N
    Gn, Dn = 0.5 * (Gn + Gn.T), 0.5 * (Dn + Dn.T)
    if cs.eig_subset:
        n_null = Gn.shape[0]
        k_top = min(n_null, max(cs.n_dir, cs.n_combo, 1))
        nu, Y = sla.eigh(Gn, Dn, subset_by_index=[n_null - k_top, n_null - 1])  # top k only
    else:
        nu, Y = sla.eigh(Gn, Dn)
    order = np.argsort(nu)[::-1][: cs.n_dir]
    cands = []
    for dir_rank, idx in enumerate(order):
        a0 = N @ Y[:, idx]
        a0 = a0 * (cs.R / math.sqrt(a0 @ D @ a0))
        for sign in (1.0, -1.0):
            a = sign * a0
            neg = float(-a.min())
            tau = min(1.0, (1.0 - cs.floor) / neg) if neg > 0 else 1.0
            ac = tau * a
            cands.append(
                {
                    "dir_rank": dir_rank,
                    "sign": sign,
                    "nu": float(nu[idx]),
                    "tau": tau,
                    "a": ac,
                    "mmd2_construction": float(ac @ G @ ac),
                }
            )
    best = max(cands, key=lambda c: c["mmd2_construction"])
    a = best["a"]
    combo = None
    if cs.n_combo > 1:
        combo = _combination_search(G, N, Y, nu, cs)
        combo["readings_construction_max"] = float(np.abs(rows @ combo["a"]).max())
        combo["R_realized"] = float(math.sqrt(combo["a"] @ D @ combo["a"]))
        combo["min_density"] = float(1.0 + combo["a"].min())
    return {
        "combo": combo,
        "a": a,
        "nu_top": [float(nu[i]) for i in order],
        "chosen": {k: v for k, v in best.items() if k != "a"},
        "candidates": [{k: v for k, v in c.items() if k != "a"} for c in cands],
        "tau": best["tau"],
        "R_nominal": cs.R,
        "R_realized": float(math.sqrt(a @ D @ a)),
        "min_density": float(1.0 + a.min()),
        "max_density": float(1.0 + a.max()),
        "mmd_construction": math.sqrt(max(best["mmd2_construction"], 0.0)),
        "width_unconstrained_construction": cs.R * math.sqrt(max(nu[order[0]], 0.0)),
        "readings_construction_max": float(np.abs(rows @ a).max()),
        "normalization_construction": float(masses @ a),
        "constraint_cond": cond,
        "constraint_rank": rank,
        "n_zero_rows": n_zero_rows,
        "null_space_err": null_err,
        "K": K,
        "m": m,
    }


def _combination_search(G, N, Y, nu, cs: ConstructionSpec) -> dict:
    """max over unit c of tau(c)^2 R^2 c^T diag(nu_k) c, a(c) = R N Y_k c (D-orthonormal columns),
    tau(c) = min(1, (1 - floor)/max_j(-a_j)). Starts: +-e_i and 24 fixed-seed random points."""
    k = min(cs.n_combo, nu.shape[0])
    idx = np.argsort(nu)[::-1][:k]
    A = N @ Y[:, idx]
    lam_k = nu[idx]

    def scaled(c):
        c = c / np.linalg.norm(c)
        a = cs.R * (A @ c)
        neg = float(-a.min())
        tau = min(1.0, (1.0 - cs.floor) / neg) if neg > 0 else 1.0
        return tau * a, tau, c

    def negobj(c):
        if not np.isfinite(c).all() or np.linalg.norm(c) == 0:
            return 0.0
        a, tau, cn = scaled(c)
        return -(tau**2) * cs.R**2 * float((cn * cn) @ lam_k)

    rng = np.random.default_rng(20261005)
    starts = [s * e for e in np.eye(k) for s in (1.0, -1.0)] + list(rng.normal(size=(24, k)))
    best = None
    for c0 in starts:
        r = sopt.minimize(
            negobj,
            c0,
            method="Nelder-Mead",
            options={"maxiter": 400 * k, "xatol": 1e-6, "fatol": 1e-14},
        )
        if best is None or r.fun < best.fun:
            best = r
    a, tau, c = scaled(best.x)
    return {
        "a": a,
        "tau": tau,
        "coef": c.tolist(),
        "k": int(k),
        "mmd2_construction": float(a @ G @ a),
        "mmd_construction": math.sqrt(max(float(a @ G @ a), 0.0)),
    }


def evaluate(
    a: np.ndarray,
    masses: np.ndarray,
    G_eval: np.ndarray,
    rows_eval: np.ndarray,
    s_omit: np.ndarray | None = None,
) -> dict:
    """Evaluate a perturbation with the finer quadrature rule.

    For low-rank G_eval, s_omit bounds the omitted kernel energy. The squared
    MMD lies in [a^T G_lr a, a^T G_lr a + (sum_j |a_j| s_j)^2]; report both ends.
    """
    z = rows_eval @ a
    mmd2 = float(a @ G_eval @ a)
    omit = 0.0 if s_omit is None else float(np.abs(a) @ s_omit) ** 2
    res2 = mmd2 - float(z @ z)
    return {
        "readings_eval": z.tolist(),
        "readings_eval_max_abs": float(np.abs(z).max()),
        "monitored_norm_eval": float(np.linalg.norm(z)),
        "normalization_eval": float(masses @ a),
        "mmd2_eval": mmd2,
        "mmd2_eval_omission_bound": omit,
        "residual2_eval": res2,
        "residual_eval": math.sqrt(res2) if res2 > 0 else float("nan"),
        "residual_eval_upper": math.sqrt(res2 + omit) if res2 + omit > 0 else float("nan"),
        "h_l2": float(math.sqrt((a * a) @ masses)),
    }


# Independent Monte Carlo checks
def sample_construction(
    target: Target, integ, a: np.ndarray, n: int, gen: torch.Generator
) -> torch.Tensor:
    """IID draws from dQ = (1 + h) dpi, h = a[cell(x)], by rejection (accept w.p. (1+h)/(1+max h))."""
    top = 1.0 + float(a.max())
    out, got = [], 0
    while got < n:
        m = int(math.ceil((n - got) * top * 1.2)) + 256
        x = target.sample(m, gen)
        hx = a[integ.cell_index(x.cpu().numpy())]
        u = torch.rand((m,), generator=gen, dtype=F64, device=gen.device).cpu().numpy()
        acc = u * top < 1.0 + hx
        out.append(x[torch.as_tensor(acc, device=x.device)])
        got += int(acc.sum())
    X = torch.cat(out)[:n]
    assert X.shape == (n, target.d)
    return X


def mc_mmd2_unbiased(
    kernel, X: torch.Tensor, mu_at_X: torch.Tensor, mu_norm2: float, n_batches: int
):
    """U-statistic MMD^2(Q, pi) with the exact target embedding; jackknife SE over batches."""
    J = X.shape[0]
    if J % n_batches:
        raise ValueError(f"J={J} not divisible by n_batches={n_batches}")
    b = J // n_batches
    S = torch.zeros((n_batches, n_batches), dtype=F64)
    for i in range(n_batches):
        for j in range(i, n_batches):
            Kb = kernel(X[i * b : (i + 1) * b], X[j * b : (j + 1) * b])
            s = float(Kb.sum())
            if i == j:
                s -= float(torch.diagonal(Kb).sum())
            S[i, j] = S[j, i] = s
    mu_b = mu_at_X.reshape(n_batches, b).sum(1)

    def est(keep):
        idx = torch.as_tensor(keep)
        n = len(keep) * b
        kxx = float(S[idx][:, idx].sum()) / (n * (n - 1))
        return kxx - 2.0 * float(mu_b[idx].sum()) / n + mu_norm2

    full = est(list(range(n_batches)))
    jk = np.array([est([j for j in range(n_batches) if j != i]) for i in range(n_batches)])
    se = math.sqrt((n_batches - 1) / n_batches * ((jk - jk.mean()) ** 2).sum())
    return full, se


# Analytic Gaussian spectrum
def mehler_1d(sigma2: float, n_max: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """1-D N(0, sigma2) with k = exp(-(x - y)^2/2): eigenvalues sqrt(2a/A) B^n of the uncentered
    integral operator (Rasmussen & Williams 4.3.1, a = 1/(4 sigma2), b = 1/2) and the means
    m_n = E[psi_n] of the L2-normalized eigenfunctions psi_n = exp(-(c-a)x^2) H_n(sqrt(2c) x)/N_n:
        m_{2k} = (a/c)^{1/4} alpha^{-1/2} sqrt((2k)!)/(k! 2^k) ((c-a)/(c+a))^k, alpha = (c+a)/(2c),
    m_odd = 0 (from int e^{-alpha t^2} H_{2k}(t) dt = sqrt(pi/alpha) (2k)!/k! (1/alpha - 1)^k)."""
    a, b = 1.0 / (4.0 * sigma2), 0.5
    c = math.sqrt(a * a + 2 * a * b)
    A = a + b + c
    B = b / A
    n = np.arange(n_max + 1)
    lam = math.sqrt(2 * a / A) * B**n
    alpha = (c + a) / (2 * c)
    r = (c - a) / (c + a)
    mvals = np.zeros(n_max + 1)
    for nn in range(0, n_max + 1, 2):
        k = nn // 2
        logm = (
            0.25 * math.log(a / c)
            - 0.5 * math.log(alpha)
            + 0.5 * math.lgamma(2 * k + 1)
            - math.lgamma(k + 1)
            - k * math.log(2.0)
            + k * math.log(r)
        )
        mvals[nn] = math.exp(logm)
    return lam, mvals, {"a": a, "b": b, "c": c, "A": A, "B": B}


def mehler_psi_1d(sigma2: float, n_max: int, x: np.ndarray) -> np.ndarray:
    """(n_max+1, len(x)) L2(N(0, sigma2))-normalized eigenfunctions psi_n(x) (stable recurrence
    for the orthonormal Hermite functions)."""
    a, b = 1.0 / (4.0 * sigma2), 0.5
    c = math.sqrt(a * a + 2 * a * b)
    t = math.sqrt(2 * c) * np.asarray(x, dtype=np.float64)
    # h_n(t) = H_n(t) / sqrt(2^n n!) via h_{n+1} = sqrt(2/(n+1)) t h_n - sqrt(n/(n+1)) h_{n-1}
    H = np.zeros((n_max + 1, t.shape[0]))
    H[0] = 1.0
    if n_max >= 1:
        H[1] = math.sqrt(2.0) * t
    for nn in range(1, n_max):
        H[nn + 1] = math.sqrt(2.0 / (nn + 1)) * t * H[nn] - math.sqrt(nn / (nn + 1)) * H[nn - 1]
    # ||psi_n||^2 = sqrt(a/c) 2^n n!  ->  psi_n / norm = exp(-(c-a) x^2) h_n(t) (a/c)^{-1/4}
    return H * np.exp(-(c - a) * np.asarray(x) ** 2)[None, :] * (a / c) ** (-0.25)


def gaussian_centered_spectrum(
    cov: np.ndarray, kernel: ScaledRBF, n_max: int = 60, n_out: int = 80
):
    """Exact centered covariance-operator spectrum for N(0, cov) in d=2 with a ScaledRBF kernel.

    In y = x/(s ell) the kernel is exp(-|y - y'|^2/2) and the target N(0, Cy), Cy = R diag(sig^2) R^T;
    in the rotated frame both factorize, so the uncentered operator has product eigenpairs. With
    v_n = lam_n^{1/2} m_n the centered operator is diag(lam) - v v^T (only even-even modes mix);
    its eigenvalues are those of that finite matrix (modes with lam > 1e-40)."""
    ks = _kscale(kernel)
    Cy = cov / np.outer(ks, ks)
    sig2, Rot = np.linalg.eigh(Cy)
    l1, m1, i1 = mehler_1d(sig2[0], n_max)
    l2, m2, i2 = mehler_1d(sig2[1], n_max)
    lam = np.outer(l1, l2)
    mm = np.outer(m1, m2)
    keep = lam > 1e-40
    idx = np.argwhere(keep)
    lv = lam[keep]
    v = np.sqrt(lv) * mm[keep]
    M = np.diag(lv) - np.outer(v, v)
    ev, U = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    return {
        "eig": ev[:n_out],
        "U": U[:, :n_out],
        "modes": idx,
        "lam_modes": lv,
        "m_modes": mm[keep],
        "sig2": sig2,
        "rot": Rot,
        "uncentered_trace": float(lam.sum()),
        "mu_norm2_from_modes": float((v * v).sum()),
        "info_axes": [i1, i2],
    }


def mehler_rkhs_angles(spec: Spectrum, gcs: dict, kernel, m_list, n_gh: int = 120) -> dict:
    """Largest principal angles to the analytic Gaussian eigenspaces, in degrees.

    Compute <e_i, e_j^M> = lambda_j^-1/2 int c_i psi_j^M dpi using an
    independent Gauss-Hermite rule in the rotated, whitened coordinates.
    """
    ks = _kscale(kernel)
    t, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    sig = np.sqrt(gcs["sig2"])
    Y1, Y2 = np.meshgrid(sig[0] * t, sig[1] * t, indexing="ij")
    W = np.outer(w, w).ravel()
    Yr = np.stack([Y1.ravel(), Y2.ravel()], axis=1)  # rotated frame (kernel units)
    X = (Yr @ gcs["rot"].T) * ks[None, :]  # back to x
    n_max = int(gcs["modes"].max())
    P1 = mehler_psi_1d(gcs["sig2"][0], n_max, Y1.ravel())
    P2 = mehler_psi_1d(gcs["sig2"][1], n_max, Y2.ravel())
    modes = gcs["modes"]
    Psi = P1[modes[:, 0]] * P2[modes[:, 1]]  # (n_modes, n_nodes)
    Psi_c = Psi - gcs["m_modes"][:, None]
    mmax = max(m_list)
    U = gcs["U"][:, :mmax]
    ev = gcs["eig"][:mmax]
    ev_next = gcs["eig"][: mmax + 1]
    # psi^M_j = ev_j^{-1/2} sum_n U_nj lam_n^{1/2} (psi_n - m_n)
    PsiM = (U * np.sqrt(gcs["lam_modes"])[:, None]).T @ Psi_c / np.sqrt(ev)[:, None]
    cvals = spec.criteria(torch.as_tensor(X, dtype=F64), mmax).numpy()  # (n_nodes, mmax)
    Gx = (cvals * W[:, None]).T @ PsiM.T / np.sqrt(ev)[None, :]  # <e_i, e^M_j>
    out = {}
    for m in m_list:
        s = np.linalg.svd(Gx[:m, :m], compute_uv=False)
        out[int(m)] = {
            "max_angle_deg": float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0)))),
            "sv_min": float(s.min()),
            "sv_max": float(s.max()),
            "gap_rel": float((ev_next[m - 1] - ev_next[m]) / ev_next[m - 1]),
        }
    return out


# Projected residuals from independent chain replicas
class KernelPCA:
    """Eigenpairs of the empirical covariance operator from iid target draws.

    Kc = H K H = U diag(g) U^T gives lambda_hat = g/n and orthonormal
    RKHS vectors with coefficients U g^-1/2. For positive eigenvalues,
    1^T U = 0, so c_i(x) = sum_a coef_ai k(Y_a, x) needs no centering term.
    These estimate the population eigenspaces.
    """

    def __init__(self, kernel, Y: torch.Tensor, n_keep: int):
        n = Y.shape[0]
        K = kernel(Y, Y)
        r = K.mean(0)
        Kc = K - r[:, None] - r[None, :] + r.mean()
        Kc = 0.5 * (Kc + Kc.T)
        g, U = torch.linalg.eigh(Kc)
        g, U = g.flip(0)[:n_keep], U.flip(1)[:, :n_keep]
        if (g <= 0).any():
            raise FloatingPointError(
                f"non-positive retained kernel-PCA eigenvalue {float(g.min())}"
            )
        self.Y, self.kernel = Y, kernel
        self.lam = (g / n).cpu().numpy()
        self.coef = U / torch.sqrt(g)[None, :]
        self.colsum_max = float((U.sum(0).abs() / math.sqrt(n)).max())
        G = self.coef.T @ Kc @ self.coef
        self.ortho_err = float((G - torch.eye(n_keep, dtype=F64, device=G.device)).abs().max())

    def criteria(self, X: torch.Tensor, m: int, block: int = 1 << 14) -> torch.Tensor:
        out = torch.empty((X.shape[0], m), dtype=F64, device=X.device)
        for i0 in range(0, X.shape[0], block):
            out[i0 : i0 + block] = self.kernel(X[i0 : i0 + block], self.Y) @ self.coef[:, :m]
        return out


def heldout_compressed_top(pca: KernelPCA, Yh: torch.Tensor, m: int) -> float:
    """Plug-in lambda_max(P_perp Sigma_hat P_perp) from independent target draws Yh."""
    n = Yh.shape[0]
    K = pca.kernel(Yh, Yh)
    r = K.mean(0)
    Kc = K - r[:, None] - r[None, :] + r.mean()
    if m == 0:
        M = Kc / n
    else:
        c = pca.criteria(Yh, m)
        c = c - c.mean(0, keepdim=True)
        M = (Kc - c @ c.T) / n
    M = 0.5 * (M + M.T)
    return float(torch.linalg.eigvalsh(M)[-1])


class ProjectedAccumulator:
    """Unbiased squared MMD, monitor readings and projected residuals.

    Uses iid replicas of the chain law, a known target embedding and
    delete-one-batch jackknife errors. Negative estimates are retained.
    """

    def __init__(self, n_batches: int, n_mon: int):
        self.nb, self.M = int(n_batches), int(n_mon)
        self.Kbb = np.zeros((self.nb, self.nb))
        self.s_mu = np.zeros(self.nb)
        self.S = np.zeros((self.nb, self.M))
        self.Q = np.zeros((self.nb, self.M))
        self.n = np.zeros(self.nb, dtype=np.int64)

    def add(self, kernel, X_batches: list, mu_at: list, g_at: list):
        nb = self.nb
        assert len(X_batches) == nb
        for b in range(nb):
            self.n[b] = X_batches[b].shape[0]
            self.s_mu[b] = float(mu_at[b].sum())
            self.S[b] = g_at[b].sum(0).cpu().numpy()
            self.Q[b] = (g_at[b] ** 2).sum(0).cpu().numpy()
            for bb in range(b, nb):
                Kb = kernel(X_batches[b], X_batches[bb])
                s = float(Kb.sum())
                if bb == b:
                    s -= float(torch.diagonal(Kb).sum())
                self.Kbb[b, bb] = self.Kbb[bb, b] = s

    def _est(self, keep: np.ndarray, mu_norm2: float, m_list) -> dict:
        n = int(self.n[keep].sum())
        full = float(self.Kbb[np.ix_(keep, keep)].sum()) / (n * (n - 1))
        full += -2.0 * float(self.s_mu[keep].sum()) / n + mu_norm2
        S = self.S[keep].sum(0)
        Q = self.Q[keep].sum(0)
        Ui = (S * S - Q) / (n * (n - 1))
        out = {"mmd2": full}
        for m in m_list:
            mon = float(Ui[:m].sum())
            out[f"mon2_{m}"] = mon
            out[f"proj2_{m}"] = full - mon
        return out

    def estimates(self, mu_norm2: float, m_list) -> dict:
        idx = np.arange(self.nb)
        full = self._est(idx, mu_norm2, m_list)
        jk = [self._est(np.delete(idx, i), mu_norm2, m_list) for i in range(self.nb)]
        out = {}
        for key, val in full.items():
            v = np.array([j[key] for j in jk])
            se = math.sqrt((self.nb - 1) / self.nb * ((v - v.mean()) ** 2).sum())
            out[key] = {"est": val, "se": se}
        return out
