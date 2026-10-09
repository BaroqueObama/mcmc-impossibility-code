"""Cubic Hermite interpolation of the Matern-3/2 Gaussian-mixture surrogate.

Values and derivatives of sum_i w_i exp(-r^2 / (2 lam_i)) are tabulated on a
uniform radial grid. The caller measures interpolation error against the direct
mixture with max_abs_err_vs.
"""

from __future__ import annotations

import numpy as np
import torch

from mcmc.paperexp.kernels import _ScaledStationaryKernel

F64 = torch.float64


class TabulatedRadialKernel(_ScaledStationaryKernel):
    """k(x, y) = f(||(x - y)/(s ell)||), tabulated on [0, r_max].

    Values beyond r_max are zero; the mixture must be below roundoff there.
    """

    kind = "matern32_surrogate_tab"

    def __init__(self, scales, ell: float, lam: np.ndarray, w: np.ndarray, r_max: float,
                 n_grid: int, label: str = ""):  # fmt: skip
        super().__init__(scales, ell)
        self.lam, self.w = np.asarray(lam, dtype=np.float64), np.asarray(w, dtype=np.float64)
        self.r_max, self.n_grid, self.label = float(r_max), int(n_grid), label
        r = np.linspace(0.0, self.r_max, self.n_grid)
        self.h = float(r[1] - r[0])
        E = np.exp(-(r[:, None] ** 2) / (2.0 * self.lam[None, :]))
        f = (self.w[None, :] * E).sum(1)
        df = (self.w[None, :] * (-r[:, None] / self.lam[None, :]) * E).sum(1)
        if f[-1] > 1e-17:
            raise ValueError(f"r_max = {r_max} too small: f(r_max) = {f[-1]}")
        self._F = torch.as_tensor(f, dtype=F64)
        self._D = torch.as_tensor(df * self.h, dtype=F64)
        self.kappa2 = float(f[0])
        self._dev: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _tables(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = (self._F.to(device), self._D.to(device))
        return self._dev[key]

    def from_sqdist(self, d2):
        F, D = self._tables(d2.device)
        u = torch.sqrt(d2) / self.h
        inside = u < (self.n_grid - 1)
        u = torch.clamp(u, max=self.n_grid - 1 - 1e-9)
        i = u.to(torch.int64)
        t = u - i
        s = 1.0 - t
        out = (1.0 + 2.0 * t) * s * s * F[i] + t * s * s * D[i]
        out += t * t * (3.0 - 2.0 * t) * F[i + 1] - t * t * s * D[i + 1]
        return torch.where(inside, out, torch.zeros_like(out))

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        self._check(X, "X")
        return torch.full((X.shape[0],), self.kappa2, dtype=F64, device=X.device)

    def max_abs_err_vs(self, n: int = 2_000_000, seed: int = 0) -> float:
        """sup over random r in [0, r_max] (plus the grid midpoints near 0) of |tab - mixture|."""
        rng = np.random.default_rng(seed)
        r = np.concatenate([rng.uniform(0.0, self.r_max, n), (np.arange(2000) + 0.5) * self.h])
        exact = (self.w[None, :] * np.exp(-(r[:, None] ** 2) / (2.0 * self.lam[None, :]))).sum(1)
        tab = self.from_sqdist(torch.as_tensor(r * r, dtype=F64)).numpy()
        return float(np.abs(tab - exact).max())

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "scales": self.scales.tolist(),
            "ell": self.ell,
            "kappa2": self.kappa2,
            "n_components": int(self.lam.shape[0]),
            "r_max": self.r_max,
            "n_grid": self.n_grid,
        }
