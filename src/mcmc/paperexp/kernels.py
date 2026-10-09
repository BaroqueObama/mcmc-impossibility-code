"""Coordinate-scaled RBF and Matern-3/2 kernels with blocked reductions.

Inputs are float64 tensors; k(X,Y) has shape (n,p) on X.device and diagonal
one. Kernel sums and products always accumulate in float64. The optional
compute_dtype=torch.float32 evaluates kernel entries at lower precision;
its error is not covered by the float64 roundoff analysis.
"""

from __future__ import annotations

import math

import numpy as np
import torch

F64 = torch.float64
SQRT3 = math.sqrt(3.0)
DEFAULT_MEM_BYTES = 1 << 30
_BYTES_PER_ENTRY_FACTOR = 4  # distance block + kernel block + temporaries, conservatively
_NEAR_REL = 1e-2  # pairs with d2 <= _NEAR_REL (|u|^2 + |v|^2) are recomputed exactly


class _ScaledStationaryKernel:
    kind: str = ""
    kappa2: float = 1.0

    def __init__(self, scales, ell: float):
        scales = np.asarray(scales, dtype=np.float64)
        if scales.ndim != 1 or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(f"scales must be a finite positive 1-D array, got {scales}")
        if not (math.isfinite(ell) and ell > 0):
            raise ValueError(f"ell must be finite and > 0, got {ell}")
        self.scales = scales
        self.ell = float(ell)
        self.d = scales.shape[0]
        self._inv_cache: dict[str, torch.Tensor] = {}

    def _inv(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._inv_cache:
            self._inv_cache[key] = torch.as_tensor(
                1.0 / (self.scales * self.ell), dtype=F64, device=device
            )
        return self._inv_cache[key]

    def _check(self, X: torch.Tensor, name: str) -> None:
        if not isinstance(X, torch.Tensor) or X.dtype != F64:
            raise TypeError(f"{name} must be a float64 tensor, got {getattr(X, 'dtype', type(X))}")
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"{name} shape {tuple(X.shape)}, expected (n, {self.d})")

    def sqdist(self, X: torch.Tensor, Y: torch.Tensor, compute_dtype=F64) -> torch.Tensor:
        """||(x - y)/(s ell)||^2 for all pairs, shape (n, p), dtype ``compute_dtype``.

        Fast path: the GEMM expansion |u|^2 + |v|^2 - 2 u.v about the Y-block mean c. Its absolute
        error is ~eps (|u|^2 + |v|^2), which is large RELATIVE to d2 only for near pairs, so every
        entry with d2 <= 1e-2 (|u|^2 + |v|^2) (including all diagonal entries) is recomputed in
        float64 from the exact difference x - y. Every returned entry is therefore >= 0 with a
        relative error of O(100 eps) in the compute dtype (no clamping).
        """
        self._check(X, "X")
        self._check(Y, "Y")
        if X.device != Y.device:
            raise ValueError(f"X on {X.device}, Y on {Y.device}")
        inv = self._inv(X.device)
        c = Y.mean(0, keepdim=True) if Y.shape[0] > 0 else torch.zeros_like(Y[:1])
        u = ((X - c) * inv).to(compute_dtype)
        v = ((Y - c) * inv).to(compute_dtype)
        nu = (u * u).sum(1, keepdim=True)
        nv = (v * v).sum(1, keepdim=True)
        mag = nu + nv.T
        d2 = mag - 2.0 * (u @ v.T)
        near = d2 <= _NEAR_REL * mag
        ii, jj = near.nonzero(as_tuple=True)
        if ii.numel():
            diff = (X[ii] - Y[jj]) * inv
            d2[ii, jj] = (diff * diff).sum(1).to(compute_dtype)
        return d2

    def from_sqdist(self, d2: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, compute_dtype=F64) -> torch.Tensor:
        K = self.from_sqdist(self.sqdist(X, Y, compute_dtype))
        assert K.shape == (X.shape[0], Y.shape[0])
        return K

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        """k(x, x) = 1 for every row."""
        self._check(X, "X")
        return torch.ones(X.shape[0], dtype=F64, device=X.device)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "scales": self.scales.tolist(), "ell": self.ell, "kappa2": 1.0}


class ScaledRBF(_ScaledStationaryKernel):
    kind = "rbf"

    def from_sqdist(self, d2):
        return torch.exp(-0.5 * d2)


class ScaledMatern32(_ScaledStationaryKernel):
    kind = "matern32"

    def from_sqdist(self, d2):
        a = SQRT3 * torch.sqrt(d2)
        return (1.0 + a) * torch.exp(-a)


def make_kernel(kind: str, scales, ell: float) -> _ScaledStationaryKernel:
    if kind == "rbf":
        return ScaledRBF(scales, ell)
    if kind == "matern32":
        return ScaledMatern32(scales, ell)
    raise ValueError(f"unknown kernel kind {kind!r}")


def kernel_from_dict(spec: dict) -> _ScaledStationaryKernel:
    return make_kernel(spec["kind"], spec["scales"], spec["ell"])


# blocked reductions
def _block_sizes(n: int, p: int, mem_bytes: int) -> tuple[int, int]:
    if mem_bytes <= 0:
        raise ValueError(f"mem_bytes must be > 0, got {mem_bytes}")
    max_entries = max(1, mem_bytes // (8 * _BYTES_PER_ENTRY_FACTOR))
    pb = max(1, min(p, max_entries))
    nb = max(1, min(n, max_entries // pb))
    return nb, pb


def _blocks(k, X, Y, mem_bytes, compute_dtype):
    k._check(X, "X")
    k._check(Y, "Y")
    n, p = X.shape[0], Y.shape[0]
    nb, pb = _block_sizes(n, p, mem_bytes)
    for i0 in range(0, n, nb):
        for j0 in range(0, p, pb):
            Kb = k(X[i0 : i0 + nb], Y[j0 : j0 + pb], compute_dtype=compute_dtype).to(F64)
            yield i0, j0, Kb


def kernel_sum(k, X, Y, mem_bytes: int = DEFAULT_MEM_BYTES, compute_dtype=F64) -> torch.Tensor:
    """sum_{i,j} k(X_i, Y_j) as a 0-dim float64 tensor (blocked, float64 accumulation)."""
    total = torch.zeros((), dtype=F64, device=X.device)
    for _, _, Kb in _blocks(k, X, Y, mem_bytes, compute_dtype):
        total += Kb.sum()
    return total


def kernel_rowsum(k, X, Y, mem_bytes: int = DEFAULT_MEM_BYTES, compute_dtype=F64) -> torch.Tensor:
    """(n,) row sums sum_j k(X_i, Y_j) (blocked, float64 accumulation)."""
    out = torch.zeros(X.shape[0], dtype=F64, device=X.device)
    for i0, _, Kb in _blocks(k, X, Y, mem_bytes, compute_dtype):
        out[i0 : i0 + Kb.shape[0]] += Kb.sum(1)
    return out


def kernel_matvec(
    k, X, Y, V: torch.Tensor, mem_bytes: int = DEFAULT_MEM_BYTES, compute_dtype=F64
) -> torch.Tensor:
    """K(X, Y) @ V for V of shape (p,) or (p, c) (blocked, float64)."""
    if V.dtype != F64:
        raise TypeError(f"V must be float64, got {V.dtype}")
    if V.shape[0] != Y.shape[0] or V.ndim not in (1, 2):
        raise ValueError(f"V shape {tuple(V.shape)} incompatible with Y {tuple(Y.shape)}")
    out = torch.zeros((X.shape[0],) + tuple(V.shape[1:]), dtype=F64, device=X.device)
    for i0, j0, Kb in _blocks(k, X, Y, mem_bytes, compute_dtype):
        out[i0 : i0 + Kb.shape[0]] += Kb @ V[j0 : j0 + Kb.shape[1]]
    return out


def mmd2_from_embedding(
    k,
    X: torch.Tensor,
    mu_at_X: torch.Tensor,
    mu_norm2: float,
    unbiased: bool = False,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    compute_dtype=F64,
) -> float:
    """MMD^2(empirical law of X, pi) with the target embedding known exactly.

    V-statistic (unbiased=False):  (1/J^2) sum_ij k(X_i,X_j) - (2/J) sum_i mu(X_i) + ||mu||^2.
    U-statistic (unbiased=True):   off-diagonal pairs only, (1/(J(J-1))) sum_{i!=j} k(X_i, X_j);
                                   unbiased for MMD^2(law of X_i, pi) with IID rows.
    """
    J = X.shape[0]
    if mu_at_X.shape != (J,):
        raise ValueError(f"mu_at_X shape {tuple(mu_at_X.shape)}, expected ({J},)")
    s = kernel_sum(k, X, X, mem_bytes, compute_dtype).item()
    if unbiased:
        if J < 2:
            raise ValueError("U-statistic needs J >= 2")
        kxx = (s - k.diag(X).sum().item()) / (J * (J - 1))
    else:
        kxx = s / J**2
    val = kxx - 2.0 * mu_at_X.mean().item() + float(mu_norm2)
    if not math.isfinite(val):
        raise FloatingPointError(f"non-finite MMD^2 {val}")
    return val
