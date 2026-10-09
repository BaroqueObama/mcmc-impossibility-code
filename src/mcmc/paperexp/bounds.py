"""Anytime MMD bounds and failure-probability allocation (Section 6, Appendix E).

The monitored radius H_t includes replica and target-embedding uncertainty.
The source-class bound U_t controls the unmonitored component, giving
MMD(pi_t, pi) <= sqrt(H_t^2 + U_t^2). Uniform embedding bounds allow the
monitored span to be chosen from the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

F64 = torch.float64


# Radii
def replica_radius(kappa: float, J: int, beta: float, weight: float = 1.0) -> float:
    """b = kappa/sqrt(J) + kappa sqrt(2 log(1/(beta w))/J) (bounded differences, App. E.2)."""
    if not (0 < beta < 1 and 0 < weight <= 1 and J >= 1 and kappa > 0):
        raise ValueError(f"bad radius inputs kappa={kappa} J={J} beta={beta} w={weight}")
    return kappa / math.sqrt(J) + kappa * math.sqrt(2.0 * math.log(1.0 / (beta * weight)) / J)


def target_radius(kappa: float, N_pi: int, beta_pi: float) -> float:
    """One-time radius for a target reference of N_pi exact IID draws."""
    return replica_radius(kappa, N_pi, beta_pi, 1.0)


# Failure-budget allocation
@dataclass
class BudgetAllocation:
    """Split failure probability among components, then uniformly over each component's grid."""

    total: float
    components: dict
    grids: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (0 < self.total < 1):
            raise ValueError(f"total beta {self.total} not in (0,1)")
        s = sum(self.components.values())
        if any(v <= 0 for v in self.components.values()):
            raise ValueError(f"non-positive component in {self.components}")
        if s > self.total * (1 + 1e-12):
            raise ValueError(f"components sum {s} exceeds total {self.total}")

    def beta(self, name: str) -> float:
        if name not in self.components:
            raise KeyError(f"no budget component {name!r}; have {sorted(self.components)}")
        return float(self.components[name])

    def declare_grid(self, name: str, axes: dict) -> None:
        if name in self.grids:
            raise ValueError(f"grid for {name!r} already declared")
        self.beta(name)
        n = 1
        for ax, vals in axes.items():
            if len(vals) == 0:
                raise ValueError(f"empty axis {ax}")
            n *= len(vals)
        self.grids[name] = {"axes": {k: list(v) for k, v in axes.items()}, "n_cells": n}

    def weight(self, name: str) -> float:
        """Uniform per-cell weight 1/n_cells (weights sum to exactly one over the declared grid)."""
        if name not in self.grids:
            raise KeyError(f"declare the grid for {name!r} first")
        return 1.0 / self.grids[name]["n_cells"]

    def cell_beta(self, name: str) -> float:
        return self.beta(name) * self.weight(name)

    def to_dict(self) -> dict:
        return {"total": self.total, "components": dict(self.components), "grids": self.grids}


# Kernel means and embedding differences (exact, blocked, nested prefixes)
def _rowsum(kernel, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """sum_j k(X_i, Y_j), using the fused reduction if available."""
    f = getattr(kernel, "rowsum", None)
    return f(X, Y) if f is not None else kernel(X, Y).sum(1)


def kernel_prefix_means(
    kernel, P: torch.Tensor, X: torch.Tensor, prefixes, chunk: int = 1024
) -> torch.Tensor:
    """(len(prefixes), |P|) with row q = mean_{i < prefixes[q]} k(P, X_i). Prefixes must be multiples of chunk."""
    prefixes = list(prefixes)
    n = X.shape[0]
    if prefixes != sorted(prefixes) or prefixes[-1] > n or any(p % chunk for p in prefixes):
        raise ValueError(f"prefixes {prefixes} invalid for n={n}, chunk={chunk}")
    out = torch.empty(len(prefixes), P.shape[0], dtype=F64, device=P.device)
    acc = torch.zeros(P.shape[0], dtype=F64, device=P.device)
    q = 0
    for s in range(0, prefixes[-1], chunk):
        acc += _rowsum(kernel, P, X[s : s + chunk])
        while q < len(prefixes) and prefixes[q] == s + chunk:
            out[q] = acc / prefixes[q]
            q += 1
    assert q == len(prefixes)
    return out


def kernel_gram_prefix_means(kernel, X: torch.Tensor, prefixes, chunk: int = 1024) -> torch.Tensor:
    """(len(prefixes),) with ||mu_hat_J||^2 = J^{-2} sum_{i,j<J} k(X_i, X_j) (V-statistic)."""
    prefixes = list(prefixes)
    nmax = prefixes[-1]
    if any(p % chunk for p in prefixes):
        raise ValueError("prefixes must be multiples of chunk")
    nb = nmax // chunk
    Sb = torch.zeros(nb, nb, dtype=F64, device=X.device)
    for i in range(nb):
        # column-block sums of rows i: sum over the chunk i of k(X[:nmax], X_chunk_i), grouped by block
        Sb[:, i] = _rowsum(kernel, X[:nmax], X[i * chunk : (i + 1) * chunk]).view(nb, chunk).sum(1)
    out = torch.empty(len(prefixes), dtype=F64, device=X.device)
    for q, p in enumerate(prefixes):
        b = p // chunk
        out[q] = Sb[:b, :b].sum() / (p * p)
    return out


@dataclass
class TargetReference:
    """Target-side information: sampled (N_pi exact draws, radius b_pi) or known (b_pi = 0)."""

    kind: str  # "sampled" | "known"
    b_pi: float
    X: torch.Tensor | None = None  # sampled draws
    mean_fn: object = None  # known: callable (n, d) tensor -> (n,) tensor mu_pi(x)
    norm2_value: float | None = None  # known ||mu_pi||^2
    _norm2_cache: float | None = None

    @staticmethod
    def sampled(X: torch.Tensor, kappa: float, beta_pi: float) -> "TargetReference":
        return TargetReference("sampled", target_radius(kappa, X.shape[0], beta_pi), X=X)

    @staticmethod
    def known(mean_fn, norm2: float) -> "TargetReference":
        return TargetReference("known", 0.0, mean_fn=mean_fn, norm2_value=float(norm2))

    def mean_at(self, kernel, P: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        if self.kind == "known":
            return self.mean_fn(P)
        n = self.X.shape[0]
        return kernel_prefix_means(kernel, P, self.X, [n], chunk=min(chunk, n))[0]

    def norm2(self, kernel, chunk: int = 2048) -> float:
        if self.kind == "known":
            return self.norm2_value
        if self._norm2_cache is None:
            n = self.X.shape[0]
            c = min(chunk, n)
            if n % c:
                raise ValueError(f"N_pi={n} not a multiple of chunk {c}")
            self._norm2_cache = float(kernel_gram_prefix_means(kernel, self.X, [n], chunk=c)[0])
        return self._norm2_cache


@dataclass
class ReplicaSet:
    """Replica states at one inspection time; nested prefixes J_grid share one array."""

    X: torch.Tensor  # (J_max, d)
    J_grid: list
    chunk: int = 1024

    def __post_init__(self):
        assert self.J_grid == sorted(self.J_grid) and self.J_grid[-1] <= self.X.shape[0]

    def mean_at(self, kernel, P: torch.Tensor) -> torch.Tensor:
        return kernel_prefix_means(kernel, P, self.X, self.J_grid, chunk=self.chunk)

    def norm2(self, kernel) -> torch.Tensor:
        return kernel_gram_prefix_means(kernel, self.X, self.J_grid, chunk=self.chunk)

    def cross(self, kernel, ref: TargetReference) -> torch.Tensor:
        """<mu_hat_J, mu_hat_pi> for each J (mean over the first J replicas of mu_hat_pi(X_i))."""
        Jmax = self.J_grid[-1]
        if ref.kind == "known":
            vals = ref.mean_fn(self.X[:Jmax])
        else:
            vals = torch.empty(Jmax, dtype=F64, device=self.X.device)
            for s in range(0, Jmax, self.chunk):
                Xs = self.X[s : s + self.chunk]
                n = ref.X.shape[0]
                acc = torch.zeros(Xs.shape[0], dtype=F64, device=Xs.device)
                for u in range(0, n, 8192):
                    acc += _rowsum(kernel, Xs, ref.X[u : u + 8192])
                vals[s : s + self.chunk] = acc / n
        cs = torch.cumsum(vals, 0)
        return torch.stack([cs[J - 1] / J for J in self.J_grid])


def fp32_vstat_and_cross(
    kernel,
    X: torch.Tensor,
    J_grid,
    Y_ref: torch.Tensor | None,
    chunk: int = 1024,
    ref_block: int = 8192,
):
    """Nested V-statistics ||mu_hat_J||^2 and cross terms <mu_hat_J, mu_hat_ref> with fp32 kernel
    entries (``kernel.rowsum32``), fp64 accumulation and rigorous error bounds.

    Returns (n2 (nJ,), err_n2, cross (nJ,) or None, err_cross): every reported value is within its err
    of the exact fp64-arithmetic quantity (a mean of entries each within the per-entry bound).
    """
    J_grid = list(J_grid)
    nmax = J_grid[-1]
    if any(J % chunk for J in J_grid):
        raise ValueError("prefixes must be multiples of chunk")
    nb = nmax // chunk
    Sb = torch.zeros(nb, nb, dtype=F64, device=X.device)
    err_n2 = 0.0
    for i in range(nb):
        rs, e = kernel.rowsum32(X[:nmax], X[i * chunk : (i + 1) * chunk])
        Sb[:, i] = rs.view(nb, chunk).sum(1)
        err_n2 = max(err_n2, e)
    n2 = torch.stack([Sb[: J // chunk, : J // chunk].sum() / (J * J) for J in J_grid])
    if Y_ref is None:
        return n2, err_n2, None, 0.0
    n = Y_ref.shape[0]
    vals = torch.empty(nmax, dtype=F64, device=X.device)
    err_c = 0.0
    for s0 in range(0, nmax, chunk):
        acc = torch.zeros(min(chunk, nmax - s0), dtype=F64, device=X.device)
        for u in range(0, n, ref_block):
            rs, e = kernel.rowsum32(X[s0 : s0 + chunk], Y_ref[u : u + ref_block])
            acc += rs
            err_c = max(err_c, e)
        vals[s0 : s0 + chunk] = acc / n
    cs = torch.cumsum(vals, 0)
    return n2, err_n2, torch.stack([cs[J - 1] / J for J in J_grid]), err_c


def direct_norm(kernel, reps: ReplicaSet, ref: TargetReference) -> torch.Tensor:
    """||mu_hat_J - mu_hat_pi|| for each J (exact V-statistic; clamped only at exact-zero roundoff)."""
    n2 = reps.norm2(kernel) - 2.0 * reps.cross(kernel, ref) + ref.norm2(kernel)
    if float(n2.min()) < -1e-12:
        raise FloatingPointError(f"negative squared embedding distance {float(n2.min())}")
    return n2.clamp_min(0.0).sqrt()


def span_coordinates(span_coefs: torch.Tensor, delta_at_points: torch.Tensor) -> torch.Tensor:
    """<v_j, delta_hat> for each J: (n_J, |P|) @ (|P|, m) -> (n_J, m)."""
    return delta_at_points @ span_coefs


def projected_norm(span_coefs: torch.Tensor, delta_at_points: torch.Tensor) -> torch.Tensor:
    """||P_V delta_hat|| for an orthonormal span."""
    a = span_coordinates(span_coefs, delta_at_points)
    return a.norm(dim=-1)


# Assembly (float64 only: bound values enter coverage decisions)
def _f64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.dtype != F64:
            raise TypeError(f"bound inputs must be float64, got {x.dtype}")
        return x
    return torch.as_tensor(x, dtype=F64)


def H_bound(proj_norm, b_t, b_pi):
    return _f64(proj_norm) + float(b_t) + float(b_pi)


def U_bound(r_init: float, lam_or_compressed, e: float = 0.0):
    """R sqrt(lam + e); lam is an eigenvalue/compressed norm (or a rigorous upper bound of one)."""
    lam = _f64(lam_or_compressed)
    if bool((lam < -1e-12).any()):
        raise ValueError(f"negative eigenvalue bound {lam}")
    return r_init * torch.sqrt(lam.clamp_min(0.0) + e)


def combined_bound(H, U):
    return torch.sqrt(_f64(H) ** 2 + _f64(U) ** 2)


def direct_bound(norm, b_t, b_pi):
    return _f64(norm) + float(b_t) + float(b_pi)
