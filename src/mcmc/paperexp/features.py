"""Rank-truncated Nystrom features f(x) = B^T k(Z, x).

For K_ZZ = U diag(e) U^T, retain e_i > rel_cutoff * e_max and set
B = U_r diag(e_r^-1/2). Since B^T K_ZZ B = I, feature coordinates represent
an orthonormal RKHS basis. The omitted energy k(x, x) - ||f(x)||^2 must be
bounded separately when using these features in a full-RKHS certificate.
"""

from __future__ import annotations

import torch

from mcmc.paperexp.kernels import DEFAULT_MEM_BYTES
from mcmc.paperexp.rng import seed_for, torch_gen

F64 = torch.float64


class NystromFeatures:
    def __init__(self, kernel, Z: torch.Tensor, rel_cutoff: float = 1e-12):
        if not (0.0 < rel_cutoff < 1.0):
            raise ValueError(f"rel_cutoff must be in (0, 1), got {rel_cutoff}")
        kernel._check(Z, "Z")
        self.kernel = kernel
        self.Z = Z
        self.rel_cutoff = float(rel_cutoff)
        K = kernel(Z, Z)
        asym = (K - K.T).abs().max().item()
        assert asym < 1e-12, f"K_ZZ asymmetric by {asym}"
        K = 0.5 * (K + K.T)
        evals, evecs = torch.linalg.eigh(K)
        evals, evecs = evals.flip(0), evecs.flip(1)  # descending
        e_max = evals[0].item()
        assert e_max > 0, f"K_ZZ has no positive eigenvalue: {e_max}"
        keep = evals > self.rel_cutoff * e_max
        self.rank = int(keep.sum().item())
        self.eigvals = evals  # all eigenvalues of K_ZZ, descending (fp64, may include roundoff < 0)
        self.B = evecs[:, keep] / torch.sqrt(evals[keep])[None, :]
        self.discarded_eig_sum = float(evals[~keep].clamp_min(0.0).sum().item())
        self.min_eig = float(evals[-1].item())
        assert self.B.shape == (Z.shape[0], self.rank)

    @classmethod
    def from_target(
        cls,
        kernel,
        target,
        n_landmarks: int,
        config_id: str,
        rep: int,
        device="cpu",
        rel_cutoff: float = 1e-12,
    ) -> "NystromFeatures":
        """Landmarks = n_landmarks exact target draws from stream (config_id, "landmarks", rep)."""
        gen = torch_gen(seed_for(config_id, "landmarks", rep), device)
        return cls(kernel, target.sample(n_landmarks, gen), rel_cutoff)

    @property
    def n_landmarks(self) -> int:
        return self.Z.shape[0]

    def __call__(self, X: torch.Tensor, mem_bytes: int = DEFAULT_MEM_BYTES) -> torch.Tensor:
        self.kernel._check(X, "X")
        B = self.B.to(X.device)
        Z = self.Z.to(X.device)
        rows = max(1, mem_bytes // (8 * 4 * max(1, self.n_landmarks)))
        out = torch.empty((X.shape[0], self.rank), dtype=F64, device=X.device)
        for i0 in range(0, X.shape[0], rows):
            out[i0 : i0 + rows] = self.kernel(X[i0 : i0 + rows], Z) @ B
        return out

    def residual_diag(self, X: torch.Tensor, mem_bytes: int = DEFAULT_MEM_BYTES) -> torch.Tensor:
        """k(x, x) - ||f(x)||^2 (omitted feature energy); raises if below -1e-9 (roundoff)."""
        F = self(X, mem_bytes)
        r = self.kernel.diag(X) - (F * F).sum(1)
        worst = r.min().item() if r.numel() else 0.0
        if worst < -1e-9:
            raise FloatingPointError(
                f"negative omitted energy {worst}: whitening is ill-conditioned"
            )
        return r

    def whitening_error(self) -> float:
        """max |B^T K_ZZ B - I_r| (the orthonormality of the retained RKHS basis)."""
        K = self.kernel(self.Z, self.Z)
        G = self.B.T @ K @ self.B
        return (G - torch.eye(self.rank, dtype=F64, device=G.device)).abs().max().item()

    def to_dict(self) -> dict:
        return {
            "n_landmarks": self.n_landmarks,
            "rel_cutoff": self.rel_cutoff,
            "rank": self.rank,
            "eig_max": float(self.eigvals[0].item()),
            "eig_min": self.min_eig,
            "discarded_eig_sum": self.discarded_eig_sum,
            "kernel": self.kernel.to_dict(),
        }
