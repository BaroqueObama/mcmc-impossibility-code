"""Linear-Gaussian Markov kernels and closed-form RBF integrals.

The transition X' = mu + A(X-mu) + xi, xi ~ N(0,Q), preserves N(mu,C)
when ACA^T + Q = C. Two independent t-step branches from a stationary start
have cross-covariance A^t C (A^t)^T, including for nonreversible chains.
Their centered kernel moment gives Sigma_t.

Landmark compression S_t retains part of Sigma_t; the omitted trace tau_t
bounds the lost operator norm. For any monitored span V within the landmark
span Z, the full residual operator norm is at most the compressed norm plus
tau_t. To see this, write u = a+b in Z and Z-perp: positivity gives
<u,Su> <= (||a|| sqrt(alpha) + ||b|| sqrt(beta))^2 <= alpha+beta
for unit u, with beta <= tau_t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

F64 = torch.float64


# Kernel
class AnisoRBF:
    """k(x, y) = exp(-||(x - y)/(ell s)||^2 / 2); same formula/attributes as kernels.ScaledRBF."""

    kind = "rbf"
    kappa2 = 1.0

    def __init__(self, scales, ell: float):
        scales = np.asarray(scales, dtype=np.float64)
        if scales.ndim != 1 or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(f"scales must be finite positive 1-D, got {scales}")
        if not (math.isfinite(ell) and ell > 0):
            raise ValueError(f"ell must be finite and > 0, got {ell}")
        self.scales = scales
        self.ell = float(ell)
        self.d = scales.shape[0]

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.dtype != F64 or Y.dtype != F64:
            raise TypeError(f"kernel inputs must be float64, got {X.dtype}, {Y.dtype}")
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != self.d or Y.shape[1] != self.d:
            raise ValueError(f"kernel input shapes {tuple(X.shape)}, {tuple(Y.shape)}; d={self.d}")
        inv = torch.as_tensor(1.0 / (self.scales * self.ell), dtype=F64, device=X.device)
        c = Y.mean(0, keepdim=True) if Y.shape[0] > 0 else torch.zeros_like(Y[:1])
        u = (X - c) * inv
        v = (Y - c) * inv
        nu, nv = (u * u).sum(1, keepdim=True), (v * v).sum(1)[None, :]
        d2 = torch.addmm(nu + nv, u, v.T, beta=1.0, alpha=-2.0)
        if d2.numel():
            # roundoff can make d2 slightly negative; anything beyond 1e-9 (1 + max norms) is a bug
            floor = -1e-9 * (1.0 + float(nu.max()) + float(nv.max()))
            if float(d2.min()) < floor:
                raise FloatingPointError(
                    f"squared distance below roundoff floor: {float(d2.min())}"
                )
        return d2.clamp_(min=0.0).mul_(-0.5).exp_()

    def rowsum(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Kernel row sums, shape (n,), in float64.

        CUDA fuses clamp, exponential, and reduction after the matrix product.
        CPU uses the dense kernel sum. Both check the minimum squared distance.
        """
        if X.device.type != "cuda":
            return self(X, Y).sum(1)
        if X.dtype != F64 or Y.dtype != F64:
            raise TypeError(f"kernel inputs must be float64, got {X.dtype}, {Y.dtype}")
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != self.d or Y.shape[1] != self.d:
            raise ValueError(f"kernel input shapes {tuple(X.shape)}, {tuple(Y.shape)}; d={self.d}")
        inv = torch.as_tensor(1.0 / (self.scales * self.ell), dtype=F64, device=X.device)
        c = Y.mean(0, keepdim=True)
        u = (X - c) * inv
        v = (Y - c) * inv
        nu, nv = (u * u).sum(1, keepdim=True), (v * v).sum(1)[None, :]
        rs, d2min = _fused_rbf_rowsum()(u @ v.T, nu, nv)
        floor = -1e-9 * (1.0 + float(nu.max()) + float(nv.max()))
        if float(d2min) < floor:
            raise FloatingPointError(f"squared distance below roundoff floor: {float(d2min)}")
        return rs

    def rowsum32(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, float]:
        """fp32 kernel entries with fp64 accumulation and a per-entry error bound (CUDA).

        Returns (rowsums (n,) float64, err) with |k32(x_i, y_j) - k(x_i, y_j)| <= err for every entry.
        Scaled points u, v are formed in fp64, rounded once to fp32; g = u32 v32^T (fp32 GEMM, TF32
        must be disabled); d2 = nu + nv - 2 g and exp(-d2/2) in fp32; row sums in fp64. With eps = 2^-24:
          |d2_32 - d2| <= E := (2 d + 12) eps (Umax + Vmax)^2   (input rounding, length-d dot product,
                                                                 norm rounding, final add/subtract),
          |k32 - k| <= E/2 + 16 eps                             (exp(-x/2) is 1/2-Lipschitz on x >= 0;
                                                                 fp32 exp error <= 16 eps since k <= 1).
        """
        if X.device.type != "cuda":
            raise ValueError("rowsum32 is the CUDA fast path only")
        if (
            torch.backends.cuda.matmul.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
        ):
            raise RuntimeError(
                "rowsum32 requires TF32 disabled and float32 matmul precision 'highest'"
            )
        if (
            X.dtype != F64
            or Y.dtype != F64
            or X.ndim != 2
            or Y.ndim != 2
            or X.shape[1] != self.d
            or Y.shape[1] != self.d
        ):
            raise ValueError(f"kernel inputs must be float64 (n, {self.d})")
        inv = torch.as_tensor(1.0 / (self.scales * self.ell), dtype=F64, device=X.device)
        c = Y.mean(0, keepdim=True)
        u = (X - c) * inv
        v = (Y - c) * inv
        nu, nv = (u * u).sum(1, keepdim=True), (v * v).sum(1)[None, :]
        umax, vmax = math.sqrt(float(nu.max())), math.sqrt(float(nv.max()))
        eps = 2.0**-24
        E = (2 * self.d + 12) * eps * (umax + vmax) ** 2
        rs, d2min = _fused_rbf_rowsum32()(u.float() @ v.float().T, nu.float(), nv.float())
        if float(d2min) < -E - 1e-12:
            raise FloatingPointError(
                f"fp32 squared distance {float(d2min)} below its error bound -{E}"
            )
        return rs, 0.5 * E + 16.0 * eps


_FUSED = None
_FUSED32 = None


def _rbf_rowsum32_epilogue(g: torch.Tensor, nu: torch.Tensor, nv: torch.Tensor):
    d2 = nu + nv - 2.0 * g
    return torch.exp(-0.5 * torch.clamp(d2, min=0.0)).to(torch.float64).sum(1), d2.min()


def _fused_rbf_rowsum32():
    global _FUSED32
    if _FUSED32 is None:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FUSED32 = torch.compile(_rbf_rowsum32_epilogue, dynamic=False)
    return _FUSED32


def _rbf_rowsum_epilogue(g: torch.Tensor, nu: torch.Tensor, nv: torch.Tensor):
    d2 = nu + nv - 2.0 * g
    return torch.exp(-0.5 * torch.clamp(d2, min=0.0)).sum(1), d2.min()


def _fused_rbf_rowsum():
    global _FUSED
    if _FUSED is None:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FUSED = torch.compile(_rbf_rowsum_epilogue, dynamic=False)
    return _FUSED


def lengthscale_matrix(kernel) -> np.ndarray:
    """Lambda = diag((ell * s)^2) for an RBF kernel with attributes ``scales`` and ``ell``."""
    if getattr(kernel, "kind", "rbf") != "rbf":
        raise TypeError(f"closed-form Gaussian integrals need an RBF kernel, got {kernel.kind}")
    ls = np.asarray(kernel.scales, dtype=np.float64) * float(kernel.ell)
    return np.diag(ls**2)


# Closed-form Gaussian integrals (numpy float64)
def _logdet_spd(M: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError(f"matrix not positive definite (sign {sign})")
    return float(ld)


def gauss_kernel_mean(kernel, m: np.ndarray, S: np.ndarray, z: np.ndarray) -> np.ndarray:
    """E_{W ~ N(m, S)} k(z_i, W) for each row z_i; returns (n,)."""
    Lam = lengthscale_matrix(kernel)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    d = Lam.shape[0]
    assert z.shape[1] == d and m.shape == (d,) and S.shape == (d, d)
    LS = Lam + S
    logfac = -0.5 * (_logdet_spd(LS) - _logdet_spd(Lam))
    diff = z - m[None, :]
    q = np.einsum("ij,ij->i", diff, np.linalg.solve(LS, diff.T).T)
    return np.exp(logfac - 0.5 * q)


def mean_embedding(kernel, mu: np.ndarray, C: np.ndarray, z: np.ndarray) -> np.ndarray:
    """mu_P(z) = E_{X~N(mu,C)} k(z, X)."""
    return gauss_kernel_mean(kernel, mu, C, z)


def cross_embedding_inner(kernel, m1, C1, m2, C2) -> float:
    """<mu_P, mu_Q> = E k(X, Y), X ~ N(m1, C1), Y ~ N(m2, C2) independent."""
    return float(
        gauss_kernel_mean(
            kernel,
            np.asarray(m2) - np.asarray(m1),
            np.asarray(C1) + np.asarray(C2),
            np.zeros((1, len(m1))),
        )[0]
    )


def mean_embedding_norm2(kernel, C: np.ndarray) -> float:
    """||mu_pi||^2 = |I + 2 C Lambda^{-1}|^{-1/2} for pi = N(mu, C)."""
    Lam = lengthscale_matrix(kernel)
    return float(math.exp(-0.5 * (_logdet_spd(Lam + 2.0 * C) - _logdet_spd(Lam))))


def mmd2_shifted_gaussians(kernel, C: np.ndarray, b: np.ndarray) -> float:
    """MMD^2(N(mu + b, C), N(mu, C)) = 2 |I + 2 C Lambda^{-1}|^{-1/2} (1 - exp(-b^T (Lambda + 2C)^{-1} b / 2))."""
    Lam = lengthscale_matrix(kernel)
    b = np.asarray(b, dtype=np.float64)
    q = float(b @ np.linalg.solve(Lam + 2.0 * C, b))
    return 2.0 * mean_embedding_norm2(kernel, C) * (-math.expm1(-0.5 * q))


def mmd2_gaussians(kernel, m1, C1, m2, C2) -> float:
    """General closed-form MMD^2 between N(m1, C1) and N(m2, C2) (cancellation-prone; tests only)."""
    return (
        cross_embedding_inner(kernel, m1, C1, m1, C1)
        + cross_embedding_inner(kernel, m2, C2, m2, C2)
        - 2.0 * cross_embedding_inner(kernel, m1, C1, m2, C2)
    )


def trace_sigma_t(kernel, C: np.ndarray, Gamma: np.ndarray) -> float:
    """tr(Sigma_t) = E k(Y, Z) - ||mu_pi||^2, (Y, Z) branched pair with cross-covariance Gamma.

    Y - Z ~ N(0, 2C - 2Gamma), so E k(Y, Z) = |I + (2C - 2Gamma) Lambda^{-1}|^{-1/2}. Computed as
    ||mu_pi||^2 * expm1(.) to avoid cancellation when Gamma is small.
    """
    Lam = lengthscale_matrix(kernel)
    a = _logdet_spd(Lam + 2.0 * C)
    b = _logdet_spd(Lam + 2.0 * C - 2.0 * Gamma)
    return mean_embedding_norm2(kernel, C) * math.expm1(0.5 * (a - b))


def branched_pair_centered_moment(
    kernel,
    Z: np.ndarray,
    mu: np.ndarray,
    C: np.ndarray,
    Gamma: np.ndarray,
    Z2: np.ndarray | None = None,
) -> np.ndarray:
    """E[(k(z_a, Y) - mu_pi(z_a)) (k(z'_b, Z) - mu_pi(z'_b))] for the branched pair (Y, Z).

    = <phi(z_a), Sigma_t phi(z'_b)>. Computed as m_a m_b expm1(log-ratio) (no cancellation).
    """
    Z2 = Z if Z2 is None else Z2
    Lam = lengthscale_matrix(kernel)
    d = Lam.shape[0]
    S = np.block([[C, Gamma], [Gamma.T, C]])
    Lam2 = np.block([[Lam, np.zeros((d, d))], [np.zeros((d, d)), Lam]])
    P = np.linalg.inv(Lam2 + S)
    P = 0.5 * (P + P.T)
    P11, P12, P22 = P[:d, :d], P[:d, d:], P[d:, d:]
    ua = Z - mu[None, :]
    ub = Z2 - mu[None, :]
    logfac_t = -0.5 * (_logdet_spd(Lam2 + S) - 2.0 * _logdet_spd(Lam))
    # independent-pair (Gamma = 0) reference: m_a m_b
    P0 = np.linalg.inv(Lam + C)
    P0 = 0.5 * (P0 + P0.T)
    logfac_0 = -(_logdet_spd(Lam + C) - _logdet_spd(Lam))
    qa = np.einsum("ij,jk,ik->i", ua, P11, ua)
    qb = np.einsum("ij,jk,ik->i", ub, P22, ub)
    q0a = np.einsum("ij,jk,ik->i", ua, P0, ua)
    q0b = np.einsum("ij,jk,ik->i", ub, P0, ub)
    cross = ua @ P12 @ ub.T
    expo_t = logfac_t - 0.5 * (qa[:, None] + qb[None, :] + 2.0 * cross)
    expo_0 = logfac_0 - 0.5 * (q0a[:, None] + q0b[None, :])
    return np.exp(expo_0) * np.expm1(expo_t - expo_0)


def _sym_whitened_eigs(B: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Eigenvalues of L^{-1} G L^{-T} (B = L L^T Cholesky; G symmetric): those of B^{-1} G."""
    L = np.linalg.cholesky(B)
    K = np.linalg.solve(L, np.linalg.solve(L, G).T)
    return np.linalg.eigvalsh(0.5 * (K + K.T))


def trace_sigma_t_stable(kernel, C: np.ndarray, Gamma: np.ndarray) -> float:
    """``trace_sigma_t`` with the log-determinant difference formed without cancellation.

    logdet(E) - logdet(E - 2 Gamma) = -sum log1p(-nu_i), nu_i the eigenvalues of 2 E^{-1} Gamma
    (E = Lambda + 2C), so tr Sigma_t keeps its relative accuracy for small Gamma (late checkpoints).
    """
    Lam = lengthscale_matrix(kernel)
    nu = _sym_whitened_eigs(Lam + 2.0 * C, 2.0 * Gamma)
    if not (nu < 1.0).all():
        raise FloatingPointError(f"Lambda + 2C - 2 Gamma not positive definite (max nu {nu.max()})")
    return mean_embedding_norm2(kernel, C) * math.expm1(-0.5 * float(np.log1p(-nu).sum()))


def branched_pair_centered_moment_stable(
    kernel,
    Z: np.ndarray,
    mu: np.ndarray,
    C: np.ndarray,
    Gamma: np.ndarray,
    Z2: np.ndarray | None = None,
) -> np.ndarray:
    """Branched-pair moment with a cancellation-free exponent difference.

    With B = Lambda + C and the Schur complement S_c = B - Gamma^T B^{-1} Gamma (S_c' with Gamma and
    Gamma^T exchanged), the block inverse of [[B, Gamma], [Gamma^T, B]] minus diag(B^{-1}, B^{-1}) is
        D11 = B^{-1} Gamma S_c^{-1} Gamma^T B^{-1},  D22 = B^{-1} Gamma^T S_c'^{-1} Gamma B^{-1},
        D12 = -B^{-1} Gamma S_c^{-1},
    and logdet([[B, Gamma], [Gamma^T, B]]) - 2 logdet B = sum log1p(-k_i^2), k_i the eigenvalues of
    B^{-1/2} Gamma B^{-1/2} (Gamma symmetric). Hence
        expo_t - expo_0 = -1/2 sum log1p(-k_i^2) - 1/2 (u_a^T D11 u_a + u_b^T D22 u_b + 2 u_a^T D12 u_b)
    is a sum of products of Gamma (no difference of O(1) numbers), so the moment keeps its relative
    accuracy when Gamma = A^t C (A^t)^T is small; the reference factor m_a m_b is unchanged.
    """
    Z2 = Z if Z2 is None else Z2
    Gamma = np.asarray(Gamma, dtype=np.float64)
    if np.abs(Gamma - Gamma.T).max() > 0:
        raise ValueError("Gamma must be exactly symmetric")
    Lam = lengthscale_matrix(kernel)
    B = Lam + C
    Binv = np.linalg.inv(B)
    Binv = 0.5 * (Binv + Binv.T)
    kap = _sym_whitened_eigs(B, Gamma)
    if not (np.abs(kap) < 1.0).all():
        raise FloatingPointError(f"branched-pair covariance not positive definite (max |k| {np.abs(kap).max()})")
    Sc = B - Gamma @ Binv @ Gamma
    Sc_inv = np.linalg.inv(0.5 * (Sc + Sc.T))
    Sc_inv = 0.5 * (Sc_inv + Sc_inv.T)
    BG = Binv @ Gamma
    D11 = BG @ Sc_inv @ BG.T
    D11 = 0.5 * (D11 + D11.T)  # Gamma symmetric: D22 = D11
    D12 = -BG @ Sc_inv
    ua = Z - mu[None, :]
    ub = Z2 - mu[None, :]
    da = np.einsum("ij,jk,ik->i", ua, D11, ua)
    db = np.einsum("ij,jk,ik->i", ub, D11, ub)
    cross = ua @ D12 @ ub.T
    dexpo = -0.5 * float(np.log1p(-(kap**2)).sum()) - 0.5 * (da[:, None] + db[None, :] + 2.0 * cross)
    logfac_0 = -(_logdet_spd(B) - _logdet_spd(Lam))
    q0a = np.einsum("ij,jk,ik->i", ua, Binv, ua)
    q0b = np.einsum("ij,jk,ik->i", ub, Binv, ub)
    expo_0 = logfac_0 - 0.5 * (q0a[:, None] + q0b[None, :])
    return np.exp(expo_0) * np.expm1(dexpo)


# Linear-Gaussian chains
@dataclass
class LinearGaussianChain:
    """X' = mu + A (X - mu) + L_Q z, z ~ N(0, I); stationary N(mu, C) with C = A C A^T + L_Q L_Q^T."""

    A: np.ndarray
    LQ: np.ndarray  # noise factor (Q = LQ LQ^T)
    mu: np.ndarray
    C: np.ndarray
    name: str = "linear_gaussian"
    _pow_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = self.mu.shape[0]
        for nm, M in (("A", self.A), ("LQ", self.LQ), ("C", self.C)):
            assert M.shape == (d, d), f"{nm} shape {M.shape}, expected ({d},{d})"
            assert np.isfinite(M).all(), f"non-finite {nm}"
        err = self.stationarity_error()
        if err > 1e-10 * max(1.0, np.abs(self.C).max()):
            raise ValueError(f"A C A^T + Q != C (max abs err {err:.3e})")
        self.LC = np.linalg.cholesky(self.C)

    @property
    def d(self) -> int:
        return self.mu.shape[0]

    @property
    def Q(self) -> np.ndarray:
        return self.LQ @ self.LQ.T

    def stationarity_error(self) -> float:
        return float(np.abs(self.A @ self.C @ self.A.T + self.Q - self.C).max())

    def reversibility_error(self) -> float:
        """max |A C - C A^T| (zero iff reversible w.r.t. N(mu, C))."""
        return float(np.abs(self.A @ self.C - self.C @ self.A.T).max())

    def power(self, t: int) -> np.ndarray:
        if t not in self._pow_cache:
            self._pow_cache[t] = np.linalg.matrix_power(self.A, t)
        return self._pow_cache[t]

    def cross_cov(self, t: int) -> np.ndarray:
        """Gamma_t = Cov(Y_t, Z_t) for two branches from X ~ pi: A^t C (A^t)^T."""
        At = self.power(t)
        G = At @ self.C @ At.T
        return 0.5 * (G + G.T)

    def noise_cov(self, t: int) -> np.ndarray:
        """Q_t = sum_{s<t} A^s Q (A^s)^T (= C - A^t C A^t^T), computed without cancellation."""
        Qt = np.zeros_like(self.C)
        for _ in range(t):
            Qt = self.A @ Qt @ self.A.T + self.Q
        return 0.5 * (Qt + Qt.T)

    def eig_modes(self, resid_tol: float = 1e-10):
        """Eigenvalues of A sorted by decreasing modulus, with unit eigenvectors (columns).

        LAPACK geev returned eigenvectors with residual ~3e-2 on the Concrete Gibbs map (near-zero
        columns), so each eigenvector is recomputed as the SVD null vector of A - lambda I and its
        residual ||A v - lambda v|| is verified (raises above resid_tol * max(1, ||A||)).
        """
        w = np.linalg.eigvals(self.A)
        order = np.argsort(-np.abs(w), kind="stable")
        w = w[order]
        d = self.d
        V = np.zeros((d, d), dtype=np.complex128)
        scale = max(1.0, float(np.linalg.norm(self.A, 2)))
        for i, lam in enumerate(w):
            if lam.imag == 0:
                _, _, Vh = np.linalg.svd(self.A - lam.real * np.eye(d))
                v = Vh[-1].astype(np.complex128)
            else:
                _, _, Vh = np.linalg.svd(self.A.astype(np.complex128) - lam * np.eye(d))
                v = Vh[-1].conj()
            res = float(np.abs(self.A @ v - lam * v).max())
            if res > resid_tol * scale:
                raise np.linalg.LinAlgError(f"eigenvector {i} (lambda={lam}) residual {res:.2e}")
            V[:, i] = v / np.linalg.norm(v)
        return w, V

    # simulation (torch, any device)
    def torch_params(self, device) -> dict:
        return {
            "A": torch.as_tensor(self.A, dtype=F64, device=device),
            "LQ": torch.as_tensor(self.LQ, dtype=F64, device=device),
            "mu": torch.as_tensor(self.mu, dtype=F64, device=device),
            "LC": torch.as_tensor(self.LC, dtype=F64, device=device),
        }

    def sample_stationary(
        self, n: int, gen: torch.Generator, device, shift: np.ndarray | None = None
    ):
        """n exact draws from N(mu + shift, C)."""
        p = self.torch_params(device)
        z = torch.randn(n, self.d, generator=gen, dtype=F64, device=device)
        x = p["mu"][None, :] + z @ p["LC"].T
        if shift is not None:
            x = x + torch.as_tensor(shift, dtype=F64, device=device)[None, :]
        return x

    def sweep(
        self, x: torch.Tensor, gen: torch.Generator, n_steps: int = 1, params: dict | None = None
    ):
        """Apply n_steps transitions (one Gibbs sweep each, for a Gibbs chain) to a (n, d) batch."""
        p = params if params is not None else self.torch_params(x.device)
        A, LQ, mu = p["A"], p["LQ"], p["mu"]
        for _ in range(n_steps):
            z = torch.randn(x.shape, generator=gen, dtype=F64, device=x.device)
            x = mu + (x - mu) @ A.T + z @ LQ.T
        return x


def gibbs_chain(
    P: np.ndarray, mu: np.ndarray, name: str = "gibbs_systematic"
) -> LinearGaussianChain:
    """Systematic-scan Gibbs (coordinates 1..d in order, one sweep = one step) for N(mu, P^{-1}).

    In centered coordinates the sweep solves (D + L) x' = -L^T x + D^{1/2} z with P = L + D + L^T
    (L strictly lower), so A = -(D + L)^{-1} L^T and the noise factor is (D + L)^{-1} D^{1/2}.
    The map is pathwise identical to ``gibbs_sweep_coordinatewise`` given the same normals z.
    """
    P = 0.5 * (P + P.T)
    d = P.shape[0]
    Dg = np.diag(np.diag(P))
    L = np.tril(P, -1)
    DL = Dg + L
    A = -np.linalg.solve(DL, L.T)
    LQ = np.linalg.solve(DL, np.diag(np.sqrt(np.diag(P))))
    C = np.linalg.inv(P)
    C = 0.5 * (C + C.T)
    assert A.shape == (d, d)
    return LinearGaussianChain(A=A, LQ=LQ, mu=np.asarray(mu, dtype=np.float64), C=C, name=name)


def gibbs_sweep_coordinatewise(
    P: np.ndarray, mu: np.ndarray, x: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Reference scalar implementation: one systematic-scan Gibbs sweep with given normals z.

    x_i <- mu_i - (1/P_ii) sum_{j != i} P_ij (x_j - mu_j) + z_i / sqrt(P_ii), i = 1..d in order.
    x, z: (n, d). Used only to validate ``gibbs_chain`` pathwise.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    d = P.shape[0]
    for i in range(d):
        others = [j for j in range(d) if j != i]
        cond_mean = mu[i] - (x[:, others] - mu[others]) @ P[i, others] / P[i, i]
        x[:, i] = cond_mean + z[:, i] / math.sqrt(P[i, i])
    return x


def reversible_ar(
    C: np.ndarray, U: np.ndarray, rhos, mu: np.ndarray | None = None
) -> LinearGaussianChain:
    """Reversible linear-Gaussian chain A = C^{1/2} U diag(rho) U^T C^{-1/2}, Q = C - A C A^T."""
    w, V = np.linalg.eigh(C)
    Ch = V @ np.diag(np.sqrt(w)) @ V.T
    Chi = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    A = Ch @ U @ np.diag(np.asarray(rhos, dtype=np.float64)) @ U.T @ Chi
    Q = C - A @ C @ A.T
    Q = 0.5 * (Q + Q.T)
    LQ = np.linalg.cholesky(Q)
    mu = np.zeros(C.shape[0]) if mu is None else mu
    return LinearGaussianChain(A=A, LQ=LQ, mu=mu, C=C, name="reversible_ar")


# Landmark compression and the rigorous ideal-operator bound
@dataclass
class LandmarkBasis:
    """Orthonormal basis e_i = sum_a W[a, i] phi(z_a) of the retained landmark span."""

    Z: np.ndarray  # (M, d)
    W: np.ndarray  # (M, r)
    gram_eigs: np.ndarray  # all eigenvalues of K_ZZ (descending)
    rtol: float
    orthonormality_error: float

    @property
    def rank(self) -> int:
        return self.W.shape[1]


def landmark_basis(kernel, Z: np.ndarray, rtol: float = 1e-10) -> LandmarkBasis:
    Zt = torch.as_tensor(Z, dtype=F64)
    K = kernel(Zt, Zt).numpy()
    K = 0.5 * (K + K.T)
    s, U = np.linalg.eigh(K)
    order = np.argsort(-s)
    s, U = s[order], U[:, order]
    keep = s > rtol * s[0]
    W = U[:, keep] / np.sqrt(s[keep])[None, :]
    err = float(np.abs(W.T @ K @ W - np.eye(W.shape[1])).max())
    if err > 1e-6:
        raise FloatingPointError(f"landmark basis not orthonormal (max err {err:.2e}); raise rtol")
    return LandmarkBasis(
        Z=np.asarray(Z, dtype=np.float64), W=W, gram_eigs=s, rtol=rtol, orthonormality_error=err
    )


@dataclass
class LandmarkOperator:
    """S = P_Z Sigma_t P_Z in the landmark basis, plus the omitted trace tau = tr(P_{Z-perp} Sigma_t P_{Z-perp})."""

    t: int
    S: np.ndarray  # (r, r)
    eigvals: np.ndarray  # descending
    eigvecs: np.ndarray  # (r, r) columns
    trace_full: float
    trace_kept: float

    @property
    def tau(self) -> float:
        return self.trace_full - self.trace_kept

    def lam(self, j: int) -> float:
        """j-th eigenvalue (1-indexed) of the compression; 0 beyond its rank."""
        return float(self.eigvals[j - 1]) if j <= len(self.eigvals) else 0.0

    def top_basis(self, m: int) -> np.ndarray:
        """(r, m) orthonormal coordinates of the leading m eigenvectors."""
        return self.eigvecs[:, :m]

    def compressed_top(self, Vb: np.ndarray) -> float:
        """lambda_max((I - Vb Vb^T) S (I - Vb Vb^T)) for orthonormal Vb (r, m) in the landmark basis."""
        r = self.S.shape[0]
        Pp = np.eye(r) - Vb @ Vb.T
        M = Pp @ self.S @ Pp
        return float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])

    def upper_lambda(self, j: int) -> float:
        """Rigorous upper bound on lambda_j(Sigma_t): lambda_j(S) + tau."""
        return max(self.lam(j), 0.0) + self.tau


CLOSED_FORMS = ("standard", "stable")


def landmark_operator(
    kernel, basis: LandmarkBasis, chain: LinearGaussianChain, t: int, closed_form: str = "standard"
) -> LandmarkOperator:
    """Compress Sigma_t to a landmark basis.

    ``closed_form="stable"`` avoids cancellation in the moment and trace
    at long horizons, where the branch cross-covariance is small.
    """
    if closed_form not in CLOSED_FORMS:
        raise ValueError(f"closed_form {closed_form!r} not in {CLOSED_FORMS}")
    stable = closed_form == "stable"
    Gam = chain.cross_cov(t)
    moment = branched_pair_centered_moment_stable if stable else branched_pair_centered_moment
    Mc = moment(kernel, basis.Z, chain.mu, chain.C, Gam)
    S = basis.W.T @ Mc @ basis.W
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    order = np.argsort(-w)
    tr_full = (trace_sigma_t_stable if stable else trace_sigma_t)(kernel, chain.C, Gam)
    tr_kept = float(np.trace(S))
    if tr_kept > tr_full * (1 + 1e-9) + 1e-14:
        raise FloatingPointError(f"t={t}: compressed trace {tr_kept} exceeds full trace {tr_full}")
    return LandmarkOperator(
        t=t, S=S, eigvals=w[order], eigvecs=V[:, order], trace_full=tr_full, trace_kept=tr_kept
    )


# Full-RKHS Hilbert-Schmidt integrals
def branched_pair_centered_moment_pairs(
    kernel, A: np.ndarray, B: np.ndarray, mu: np.ndarray, C: np.ndarray, Gamma: np.ndarray
) -> np.ndarray:
    """Row-wise s(a_i, b_i) = <phi(a_i), Sigma_t phi(b_i)> (same formula as the matrix version)."""
    Lam = lengthscale_matrix(kernel)
    d = Lam.shape[0]
    S = np.block([[C, Gamma], [Gamma.T, C]])
    Lam2 = np.block([[Lam, np.zeros((d, d))], [np.zeros((d, d)), Lam]])
    P = np.linalg.inv(Lam2 + S)
    P = 0.5 * (P + P.T)
    P0 = np.linalg.inv(Lam + C)
    P0 = 0.5 * (P0 + P0.T)
    ua, ub = A - mu[None, :], B - mu[None, :]
    logfac_t = -0.5 * (_logdet_spd(Lam2 + S) - 2.0 * _logdet_spd(Lam))
    logfac_0 = -(_logdet_spd(Lam + C) - _logdet_spd(Lam))
    q_t = (
        np.einsum("ij,jk,ik->i", ua, P[:d, :d], ua)
        + np.einsum("ij,jk,ik->i", ub, P[d:, d:], ub)
        + 2.0 * np.einsum("ij,jk,ik->i", ua, P[:d, d:], ub)
    )
    q_0 = np.einsum("ij,jk,ik->i", ua, P0, ua) + np.einsum("ij,jk,ik->i", ub, P0, ub)
    e_t, e_0 = logfac_t - 0.5 * q_t, logfac_0 - 0.5 * q_0
    return np.exp(e_0) * np.expm1(e_t - e_0)


def sigma_t_hs_norm2(kernel, C: np.ndarray, Gamma: np.ndarray) -> float:
    """||Sigma_t||_HS^2 = E[k_c(Y1, Y2) k_c(Z1, Z2)] for independent branched pairs (Y1,Z1), (Y2,Z2).

    k_c(a, b) = k(a, b) - mu_pi(a) - mu_pi(b) + ||mu_pi||^2. Each of the 16 product terms is
    const * E exp(-W^T Q W / 2) = const * |I + S4 Q|^{-1/2}, W = (Y1, Z1, Y2, Z2) - mean ~ N(0, S4).
    """
    Lam = lengthscale_matrix(kernel)
    d = Lam.shape[0]
    Li = np.linalg.inv(Lam)
    P0 = np.linalg.inv(Lam + C)
    c0 = math.exp(-0.5 * (_logdet_spd(Lam + C) - _logdet_spd(Lam)))
    n2 = mean_embedding_norm2(kernel, C)
    S2 = np.block([[C, Gamma], [Gamma.T, C]])
    S4 = np.zeros((4 * d, 4 * d))
    S4[: 2 * d, : 2 * d] = S2
    S4[2 * d :, 2 * d :] = S2

    def blk(i, j, M):
        Q = np.zeros((4 * d, 4 * d))
        Q[i * d : (i + 1) * d, j * d : (j + 1) * d] = M
        return Q

    def terms(a, b):
        return [
            (1.0, blk(a, a, Li) + blk(b, b, Li) - blk(a, b, Li) - blk(b, a, Li)),
            (-c0, blk(a, a, P0)),
            (-c0, blk(b, b, P0)),
            (n2, np.zeros((4 * d, 4 * d))),
        ]

    total = 0.0
    eye = np.eye(4 * d)
    for cf, Qf in terms(0, 2):  # Y1, Y2 (blocks 0, 2)
        for cg, Qg in terms(1, 3):  # Z1, Z2 (blocks 1, 3)
            sign, ld = np.linalg.slogdet(eye + S4 @ (Qf + Qg))
            if sign <= 0:
                raise np.linalg.LinAlgError("non-positive determinant in HS integral")
            total += cf * cg * math.exp(-0.5 * ld)
    return total
