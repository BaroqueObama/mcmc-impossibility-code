"""Slow-mode alignment for exact Gaussian regression posteriors.

A reversible Gaussian AR chain has known Hermite eigenfunctions. Their kernel
embeddings are compared with the leading spaces of the branched covariance
operator and with modes learned by TICA. Angles use the Euclidean geometry of
the fixed Nystrom coordinates; the RKHS metric defect is reported separately.

The default arithmetic is float64. The "ld" evaluation uses mpmath for d x d
algebra and x87 long double for landmark calculations, rounding after whitening.
Cross-precision differences provide numerical margins, not interval enclosures.
Exact integer Gram matrices are used when comparing stored subspaces.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from mcmc.paperexp import slowmodes as sm

LD = np.longdouble
U64 = 2.0**-53  # float64 unit roundoff
ULD = 2.0**-64  # x87 long double unit roundoff
BASE_RATE = 0.95
ROTATION_SEED = 20261006
MP_DPS = 50


def sha256_array(x: np.ndarray) -> str:
    """SHA256 of dtype, shape and raw bytes of a C-contiguous copy."""
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(x.dtype.str.encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _require_ld() -> None:
    if np.finfo(LD).eps > 1.1e-19:
        raise RuntimeError("np.longdouble is not the 80-bit extended type on this platform")


# Reversible Gaussian sampler
@dataclass(frozen=True)
class ControlledAR:
    """Reversible Gaussian AR transition X' = mu + A(X - mu) + xi.

    L = C^{1/2} U, A = L diag(rho) L^{-1} and Cov(xi) = L diag(1-rho^2) L^T.
    In coordinates y = L^{-1}(x-mu), the Hermite modes have eigenvalues rho^alpha.
    """

    name: str
    mu: np.ndarray
    C: np.ndarray
    corr_eigs: np.ndarray  # s_1 >= ... >= s_d
    h: float
    rho: np.ndarray  # frozen coordinate AR coefficients (float64; rho_1 = 0.95)
    U: np.ndarray  # frozen rotation (float64)
    qr_R_diag: np.ndarray  # diagonal of the sign-fixed R factor (all > 0)
    C_half: np.ndarray
    C_half_inv: np.ndarray
    L: np.ndarray  # C^{1/2} U
    A: np.ndarray
    LQ: np.ndarray  # C^{1/2} U diag(sqrt(1 - rho^2)); LQ LQ^T = Q
    LC: np.ndarray  # Cholesky factor of C (exact target draws)

    @property
    def d(self) -> int:
        return int(self.mu.shape[0])

    def sample_target(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n exact draws from N(mu, C) via the Cholesky factor (independent of the mode map)."""
        return self.mu[None, :] + rng.standard_normal((n, self.d)) @ self.LC.T

    def path(self, rng: np.random.Generator, n_transitions: int, chunk: int = 4096) -> np.ndarray:
        """Stationary trajectory of shape (n_transitions + 1, d).

        Draws the d normals for x_0, followed by noise in blocks of chunk steps.
        """
        if n_transitions < 1:
            raise ValueError(f"n_transitions must be >= 1, got {n_transitions}")
        X = np.empty((n_transitions + 1, self.d))
        X[0] = self.sample_target(rng, 1)[0]
        x = X[0] - self.mu
        At, LQt = self.A.T, self.LQ.T
        s = 0
        while s < n_transitions:
            b = min(chunk, n_transitions - s)
            xi = rng.standard_normal((b, self.d)) @ LQt
            for i in range(b):
                x = x @ At + xi[i]
                X[s + 1 + i] = x
            s += b
        X[1:] += self.mu[None, :]
        assert np.isfinite(X).all(), "non-finite AR path"
        return X

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "d": self.d,
            "base_rate": BASE_RATE,
            "rotation_seed": ROTATION_SEED,
            "rotation_rng": "numpy.random.default_rng(20261006).standard_normal((d, d)); QR; R diag > 0",
            "corr_eigs": self.corr_eigs.tolist(),
            "h": self.h,
            "rho": self.rho.tolist(),
            "rho_hex": [float(x).hex() for x in self.rho],
            "U": self.U.tolist(),
            "U_sha256": sha256_array(self.U),
            "qr_R_diag": self.qr_R_diag.tolist(),
            "A_sha256": sha256_array(self.A),
            "mu_sha256": sha256_array(self.mu),
            "C_sha256": sha256_array(self.C),
        }


def frozen_rotation(d: int, seed: int = ROTATION_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Gaussian QR with positive R diagonal; the seed is reset for every call."""
    G = np.random.default_rng(seed).standard_normal((d, d))
    Qm, Rm = np.linalg.qr(G)
    sgn = np.sign(np.diag(Rm))
    if (sgn == 0).any():
        raise np.linalg.LinAlgError("singular Gaussian matrix in the rotation draw")
    U = Qm * sgn[None, :]
    Rp = Rm * sgn[:, None]
    assert (np.diag(Rp) > 0).all()
    recon = float(np.abs(U @ Rp - G).max())
    if recon > 1e-12 * max(1.0, float(np.abs(G).max())):
        raise FloatingPointError(f"QR reconstruction error {recon:.3e}")
    return U, np.diag(Rp).copy()


def build_controlled_ar(name: str, mu: np.ndarray, C: np.ndarray) -> ControlledAR:
    """Set rho_j = exp(-h/s_j), h = -s_1 log(0.95), with s the correlation eigenvalues.

    Uses the same seeded rotation rule for every dataset.
    """
    mu = sm.as_f64("mu", np.asarray(mu), 1)
    C = sm.as_f64("C", np.asarray(C), 2)
    d = mu.shape[0]
    if C.shape != (d, d):
        raise ValueError(f"C shape {C.shape} for d={d}")
    if relative := sm.relative_asymmetry(C):
        if relative > 1e-14:
            raise ValueError(f"C asymmetric (relative {relative:.2e})")
    C = 0.5 * (C + C.T)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    s = linalg.eigvalsh(0.5 * (R + R.T))[::-1].copy()
    if s[-1] <= 0:
        raise np.linalg.LinAlgError(f"correlation matrix not positive definite: {s}")
    h = -float(s[0]) * math.log(BASE_RATE)
    rho = np.exp(-h / s)
    if abs(rho[0] - BASE_RATE) > 4 * U64:
        raise FloatingPointError(f"rho_1 = {rho[0]!r} != {BASE_RATE}")
    U, rdiag = frozen_rotation(d)
    return controlled_ar_from(name, mu, C, U, rho, corr_eigs=s, h=h, qr_R_diag=rdiag)


def controlled_ar_from(
    name: str,
    mu: np.ndarray,
    C: np.ndarray,
    U: np.ndarray,
    rho: np.ndarray,
    corr_eigs: np.ndarray | None = None,
    h: float = float("nan"),
    qr_R_diag: np.ndarray | None = None,
) -> ControlledAR:
    """The reversible AR for given (mu, C, U, rho); ``build_controlled_ar`` applies the frozen rule."""
    d = mu.shape[0]
    rho = np.asarray(rho, dtype=np.float64)
    if U.shape != (d, d) or rho.shape != (d,) or C.shape != (d, d):
        raise ValueError(f"shapes U {U.shape}, rho {rho.shape}, C {C.shape} for d={d}")
    if (rho < 0).any() or (rho >= 1).any():
        raise ValueError(f"rho must lie in [0, 1): {rho}")
    C = 0.5 * (C + C.T)
    w, V = linalg.eigh(C)
    if w[0] <= 0:
        raise np.linalg.LinAlgError("C not positive definite")
    C_half = (V * np.sqrt(w)) @ V.T
    C_half = 0.5 * (C_half + C_half.T)
    C_half_inv = (V / np.sqrt(w)) @ V.T
    C_half_inv = 0.5 * (C_half_inv + C_half_inv.T)
    L = C_half @ U
    A = (L * rho[None, :]) @ (U.T @ C_half_inv)
    one_minus_rho2 = -np.expm1(2.0 * np.log(np.where(rho > 0, rho, 1.0)))
    one_minus_rho2 = np.where(rho > 0, one_minus_rho2, 1.0)
    LQ = L * np.sqrt(one_minus_rho2)[None, :]
    LC = linalg.cholesky(C, lower=True)
    return ControlledAR(
        name=name,
        mu=mu.copy(),
        C=C,
        corr_eigs=np.full(d, np.nan) if corr_eigs is None else corr_eigs,
        h=h,
        rho=rho,
        U=U,
        qr_R_diag=np.full(d, np.nan) if qr_R_diag is None else qr_R_diag,
        C_half=C_half,
        C_half_inv=C_half_inv,
        L=L,
        A=A,
        LQ=LQ,
        LC=LC,
    )


def ar_witnesses(ar: ControlledAR, lags=(1, 2, 8, 64)) -> dict:
    """Stationarity, reversibility and mode-map residuals, relative to ||C||_max."""
    C, A = ar.C, ar.A
    cmax = float(np.abs(C).max())
    Q = ar.LQ @ ar.LQ.T  # compute independently of C - A C A^T
    out = {
        "stationarity_rel": float(np.abs(A @ C @ A.T + Q - C).max()) / cmax,
        "reversibility_rel": float(np.abs(A @ C - C @ A.T).max()) / cmax,
        "noise_min_eig": float(linalg.eigvalsh(0.5 * (Q + Q.T))[0]),
        "noise_vs_C_minus_ACAt_rel": float(np.abs(Q - (C - A @ C @ A.T)).max()) / cmax,
        "U_orthogonality": float(np.abs(ar.U.T @ ar.U - np.eye(ar.d)).max()),
        "qr_R_diag_min": float(np.min(ar.qr_R_diag)),
        "LLt_vs_C_rel": float(np.abs(ar.L @ ar.L.T - C).max()) / cmax,
        "C_half_sq_rel": float(np.abs(ar.C_half @ ar.C_half - C).max()) / cmax,
    }
    ev = np.sort(np.linalg.eigvals(A).real)[::-1]
    out["A_eigs_vs_rho_max_abs"] = float(np.abs(ev - np.sort(ar.rho)[::-1]).max())
    out["A_eigs_max_imag"] = float(np.abs(np.linalg.eigvals(A).imag).max())
    branch = {}
    for t in lags:
        At = np.linalg.matrix_power(A, t)
        A2t = np.linalg.matrix_power(A, 2 * t)
        mode = (ar.L * ar.rho ** (2 * t)) @ ar.L.T
        g_branch = At @ C @ At.T
        g_lag = A2t @ C
        branch[int(t)] = {
            "branch_vs_lag2t_rel": float(np.abs(g_branch - g_lag).max()) / cmax,
            "branch_vs_modes_rel": float(np.abs(g_branch - mode).max()) / cmax,
            "lag2t_asymmetry_rel": float(np.abs(g_lag - g_lag.T).max()) / cmax,
        }
    out["cross_covariance"] = branch
    return out


# Hermite spectrum
@dataclass(frozen=True)
class HermiteSpectrum:
    """Leading non-constant Hermite modes in descending eigenvalue order with tie clusters."""

    indices: np.ndarray  # (K, d) int multi-indices
    log_eig: np.ndarray  # (K,) float64, descending (from mpmath values of the frozen rho)
    log_eig_str: list  # (K,) mpmath strings (MP_DPS digits)
    cluster: np.ndarray  # (K,) cluster id (consecutive entries within tie_tol share an id)
    tie_tol: float
    n_top: int

    @property
    def K(self) -> int:
        return int(self.log_eig.size)

    @property
    def eig(self) -> np.ndarray:
        return np.exp(self.log_eig)

    def label(self, j: int) -> str:
        a = self.indices[j]
        return "*".join(f"y{i + 1}^{k}" if k > 1 else f"y{i + 1}" for i, k in enumerate(a) if k)

    def degree(self, j: int) -> int:
        return int(self.indices[j].sum())

    def to_dict(self) -> dict:
        return {
            "n_top": self.n_top,
            "tie_tol_log": self.tie_tol,
            "modes": [
                {
                    "rank": j + 1,
                    "alpha": self.indices[j].tolist(),
                    "label": self.label(j),
                    "degree": self.degree(j),
                    "eigenvalue": float(self.eig[j]),
                    "log_eigenvalue": self.log_eig_str[j],
                    "cluster": int(self.cluster[j]),
                }
                for j in range(self.K)
            ],
        }


def _mp_log_rho(rho: np.ndarray):
    import mpmath as mp

    out = []
    for r in rho:
        if not (0.0 <= r < 1.0):
            raise ValueError(f"rho must lie in [0, 1), got {r}")
        out.append(None if r == 0.0 else mp.log(mp.mpf(float(r))))
    return out


def enumerate_hermite(rho: np.ndarray, n_top: int = 17, tie_tol: float = 1e-12) -> HermiteSpectrum:
    """Top n_top Hermite eigenvalues rho^alpha, including complete tie clusters.

    A priority queue visits alpha + e_j in descending eigenvalue order. Stop
    once the next log eigenvalue is more than tie_tol below the last one,
    so the final cluster is complete. Coordinates with rho_j = 0 are omitted.
    """
    import mpmath as mp

    if n_top < 1:
        raise ValueError("n_top must be >= 1")
    with mp.workdps(MP_DPS):
        lr = _mp_log_rho(np.asarray(rho, dtype=np.float64))
        live = [j for j, v in enumerate(lr) if v is not None]
        d = len(lr)
        heap, seen, out = [], set(), []
        for j in live:
            a = tuple(1 if i == j else 0 for i in range(d))
            heapq.heappush(heap, (-lr[j], a))
            seen.add(a)
        tol = mp.mpf(tie_tol)
        while heap:
            if len(out) >= n_top and out[-1][0] - (-heap[0][0]) > tol:
                break
            neg, a = heapq.heappop(heap)
            out.append((-neg, a))
            for j in live:
                b = tuple(a[i] + (1 if i == j else 0) for i in range(d))
                if b not in seen:
                    seen.add(b)
                    heapq.heappush(heap, (-mp.fsum(b[i] * lr[i] for i in live if b[i]), b))
        lv = [v for v, _ in out]
        cl = [0]
        for k in range(1, len(lv)):
            cl.append(cl[-1] + (1 if lv[k - 1] - lv[k] > tol else 0))
        return HermiteSpectrum(
            indices=np.array([a for _, a in out], dtype=np.int64),
            log_eig=np.array([float(v) for v in lv]),
            log_eig_str=[mp.nstr(v, MP_DPS) for v in lv],
            cluster=np.array(cl, dtype=np.int64),
            tie_tol=float(tie_tol),
            n_top=int(n_top),
        )


def true_gap(spec: HermiteSpectrum, m: int) -> dict:
    """Strict true gap after the top-m modes: b = rho^mode_m, a = rho^mode_{m+1} (exact rho values).

    ``strict`` requires log b - log a > tie_tol, i.e. the top-m set is a union of complete clusters.
    ``a_upper`` / ``b_lower`` are the float64 values rounded outward by 4 ulp from the mpmath values.
    """
    import mpmath as mp

    if not (1 <= m < spec.K):
        raise ValueError(f"m={m} needs 1 <= m < K={spec.K}")
    with mp.workdps(MP_DPS):
        lb, la = mp.mpf(spec.log_eig_str[m - 1]), mp.mpf(spec.log_eig_str[m])
        b, a = mp.exp(lb), mp.exp(la)
        strict = bool(lb - la > mp.mpf(spec.tie_tol))
        return {
            "m": int(m),
            "strict": strict,
            "b": float(b),
            "a": float(a),
            "b_lower": float(np.nextafter(np.nextafter(float(b), 0.0), 0.0)) * (1 - 2 * U64),
            "a_upper": float(np.nextafter(np.nextafter(float(a), 1.0), 1.0)) * (1 + 2 * U64),
            "log_gap": float(lb - la),
            "rel_gap": float(1 - a / b),
            "cluster_m": int(spec.cluster[m - 1]),
            "cluster_m1": int(spec.cluster[m]),
            "a_over_b": float(a / b),
        }


def enumerate_bruteforce(rho: np.ndarray, max_degree: int) -> tuple[np.ndarray, np.ndarray]:
    """All alpha with 1 <= |alpha| <= max_degree sorted by descending rho^alpha (tests only)."""
    d = len(rho)
    idx = [a for a in multi_indices(d, max_degree) if sum(a) > 0]
    lr = np.log(np.asarray(rho))
    lv = np.array([float(np.dot(a, lr)) for a in idx])
    order = np.argsort(-lv, kind="stable")
    return np.array([idx[i] for i in order]), lv[order]


def multi_indices(d: int, max_degree: int) -> list:
    """All d-dim multi-indices with total degree <= max_degree, by degree then lexicographic."""
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for k in range(remaining, -1, -1):
            rec(prefix + [k], remaining - k, slots - 1)

    for deg in range(max_degree + 1):
        rec([], deg, d)
    return out


# Fixed landmark features
def rbf_aniso(X: np.ndarray, Q: np.ndarray, ls: np.ndarray, block: int = 2048, dtype=np.float64):
    """k(x, q) = exp(-|(x - q) / ls|^2 / 2) from explicit differences of scaled points, blocked."""
    X, Q = np.asarray(X), np.asarray(Q)
    if X.ndim != 2 or Q.ndim != 2 or X.shape[1] != Q.shape[1] or ls.shape != (X.shape[1],):
        raise ValueError(f"shapes X {X.shape}, Q {Q.shape}, ls {ls.shape}")
    Xs = X.astype(dtype) / ls.astype(dtype)
    Qs = Q.astype(dtype) / ls.astype(dtype)
    out = np.empty((X.shape[0], Q.shape[0]), dtype=dtype)
    for i in range(0, X.shape[0], block):
        diff = Xs[i : i + block, None, :] - Qs[None, :, :]
        out[i : i + block] = np.exp(-0.5 * np.einsum("nmk,nmk->nm", diff, diff))
    return out


@dataclass(frozen=True)
class LandmarkFeatures:
    """z(x) = W^T k(Q, x): whitened Nystrom coordinates of the retained landmark span."""

    Q: np.ndarray  # (M, d) landmarks
    ls: np.ndarray  # (d,) kernel lengthscales ell * sqrt(C_ii)
    rel_cutoff: float
    W: np.ndarray  # (M, r)
    gram_eigs: np.ndarray  # all eigenvalues of K, descending

    @property
    def rank(self) -> int:
        return int(self.W.shape[1])

    @property
    def M(self) -> int:
        return int(self.Q.shape[0])

    def __call__(self, X: np.ndarray, block: int = 2048) -> np.ndarray:
        out = np.empty((X.shape[0], self.rank))
        for i in range(0, X.shape[0], block):
            out[i : i + block] = rbf_aniso(X[i : i + block], self.Q, self.ls) @ self.W
        return out

    def to_dict(self) -> dict:
        s = self.gram_eigs
        return {
            "n_landmarks": self.M,
            "rel_cutoff": self.rel_cutoff,
            "rank": self.rank,
            "gram_eig_max": float(s[0]),
            "gram_eig_min_kept": float(s[self.rank - 1]),
            "gram_eig_first_dropped": float(s[self.rank]) if self.rank < s.size else None,
            "gram_eig_min": float(s[-1]),
            "kept_condition": float(s[0] / s[self.rank - 1]),
            "Q_sha256": sha256_array(self.Q),
            "W_sha256": sha256_array(self.W),
            "ls": self.ls.tolist(),
        }


def landmark_features(Q: np.ndarray, ls: np.ndarray, rel_cutoff: float) -> LandmarkFeatures:
    if not (0.0 < rel_cutoff < 1.0):
        raise ValueError(f"rel_cutoff must be in (0, 1), got {rel_cutoff}")
    K = rbf_aniso(Q, Q, ls)
    if not np.array_equal(K, K.T):
        raise FloatingPointError("explicit-difference landmark Gram is not exactly symmetric")
    s, V = sm.eigh_desc(K, "K_QQ", max_rel_asym=0.0)
    keep = s > rel_cutoff * s[0]
    if not keep[: int(keep.sum())].all():
        raise AssertionError("kept eigenvalues not a leading block")
    W = V[:, keep] / np.sqrt(s[keep])[None, :]
    return LandmarkFeatures(
        Q=Q.copy(), ls=ls.copy(), rel_cutoff=float(rel_cutoff), W=W, gram_eigs=s
    )


def metric_defect(feat: LandmarkFeatures) -> dict:
    """Compute W^T K W - I in long double, treating float64 W as exact input.

    This measures the RKHS metric defect. The fixed-feature angle comparison
    uses Euclidean coordinates and does not apply this metric.
    """
    _require_ld()
    K = rbf_aniso(feat.Q, feat.Q, feat.ls, dtype=LD)
    Wl = feat.W.astype(LD)
    E = (Wl.T @ K @ Wl - np.eye(feat.rank, dtype=LD)).astype(np.float64)
    E = 0.5 * (E + E.T)
    ev = linalg.eigvalsh(E)
    return {
        "metric_defect_op": float(max(abs(ev[0]), abs(ev[-1]))),
        "metric_defect_max_abs": float(np.abs(E).max()),
        "metric_eig_min": float(1.0 + ev[0]),
        "metric_eig_max": float(1.0 + ev[-1]),
    }


# Gaussian geometry
@dataclass(frozen=True)
class Geometry:
    """Landmark-level inputs to the closed forms in one precision ("fp64" or "ld")."""

    precision: str
    dtype: type
    ut: np.ndarray  # (M, d) K1^{-1/2} (q - mu)
    m: np.ndarray  # (M,) mu_pi(q)
    N: np.ndarray  # (d, d) K1^{-1/2} L
    u: np.ndarray  # (M, d) L^T K1^{-1} (q - mu)      (generating function)
    B: np.ndarray  # (d, d) -L^T K1^{-1} L            (generating function)
    W: np.ndarray  # (M, r)
    K1_isqrt: np.ndarray  # (d, d) (Lambda + C)^{-1/2}
    log_rho: object  # list of mpmath values (ld) or float64 array (fp64); None/-inf for rho = 0
    rho: np.ndarray  # frozen float64 rho
    N_mp: object = None  # ld only: K1^{-1/2} L as an mpmath matrix (MP_DPS digits)
    kept_condition: float = float("nan")  # s_1 / s_r of the kept landmark Gram eigenvalues


def _mp_sym_fn(mp, S, fn):
    E, Q = mp.eigsy(S)
    n = S.rows
    out = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            out[i, j] = mp.fsum(Q[i, k] * fn(E[k]) * Q[j, k] for k in range(n))
    return out


def _mp_to_ld(mp, X) -> np.ndarray:
    return np.array(
        [[LD(mp.nstr(X[i, j], 30)) for j in range(X.cols)] for i in range(X.rows)], dtype=LD
    )


def make_geometry(ar: ControlledAR, feat: LandmarkFeatures, precision: str) -> Geometry:
    """Precompute quantities for Gaussian moments and Hermite embeddings.

    fp64 uses NumPy. ld uses mpmath for C^{1/2}, the polar factor of U,
    K1^{-1/2} and d x d products, then long double for landmark arithmetic.
    The polar factor ensures L L^T = C in the reference arithmetic.
    """
    d = ar.d
    lam = feat.ls**2
    if precision == "fp64":
        K1 = np.diag(lam) + ar.C
        w, V = linalg.eigh(K1)
        K1_isqrt = (V / np.sqrt(w)) @ V.T
        K1_isqrt = 0.5 * (K1_isqrt + K1_isqrt.T)
        diff = feat.Q - ar.mu[None, :]
        ut = diff @ K1_isqrt  # rows K1^{-1/2} (q - mu) (K1^{-1/2} symmetric)
        ld_lam = float(np.sum(np.log(lam)))
        ld_K1 = float(np.sum(np.log(w)))
        m = np.exp(0.5 * (ld_lam - ld_K1) - 0.5 * np.einsum("ij,ij->i", ut, ut))
        N = K1_isqrt @ ar.L
        K1_inv = (V / w) @ V.T
        K1_inv = 0.5 * (K1_inv + K1_inv.T)
        u = diff @ (K1_inv @ ar.L)
        B = -(ar.L.T @ K1_inv @ ar.L)
        B = 0.5 * (B + B.T)
        with np.errstate(divide="ignore"):
            log_rho = np.log(ar.rho)
        return Geometry(
            "fp64",
            np.float64,
            ut,
            m,
            N,
            u,
            B,
            feat.W,
            K1_isqrt,
            log_rho,
            ar.rho,
            kept_condition=float(feat.gram_eigs[0] / feat.gram_eigs[feat.rank - 1]),
        )
    if precision != "ld":
        raise ValueError(f"precision must be 'fp64' or 'ld', got {precision!r}")
    _require_ld()
    import mpmath as mp

    with mp.workdps(MP_DPS):
        Cm = mp.matrix(ar.C.tolist())
        Um = mp.matrix(ar.U.tolist())
        C_half = _mp_sym_fn(mp, Cm, mp.sqrt)
        UtU = Um.T * Um
        Ubar = Um * _mp_sym_fn(mp, UtU, lambda x: 1 / mp.sqrt(x))  # polar factor of U
        Lm = C_half * Ubar
        K1 = mp.matrix(ar.C.tolist())
        for i in range(d):
            K1[i, i] += mp.mpf(float(feat.ls[i])) ** 2  # Lambda = ls^2 exactly
        K1_isqrt = _mp_sym_fn(mp, K1, lambda x: 1 / mp.sqrt(x))
        K1_inv = _mp_sym_fn(mp, K1, lambda x: 1 / x)
        Nm = K1_isqrt * Lm
        P = Lm.T * K1_inv  # L^T K1^{-1}
        Bm = -(P * Lm)
        ld_lam = mp.fsum(2 * mp.log(mp.mpf(float(x))) for x in feat.ls)
        ld_K1 = mp.log(mp.det(K1))
        logc = (ld_lam - ld_K1) / 2
        K1_isqrt_l, N_l, P_l, B_l = (_mp_to_ld(mp, X) for X in (K1_isqrt, Nm, P, Bm))
        logc_l = LD(mp.nstr(logc, 30))
        log_rho = _mp_log_rho(ar.rho)
    B_l = (B_l + B_l.T) / 2
    diff = feat.Q.astype(LD) - ar.mu.astype(LD)[None, :]
    ut = diff @ K1_isqrt_l
    m = np.exp(logc_l - LD(0.5) * np.einsum("ij,ij->i", ut, ut))
    u = diff @ P_l.T
    return Geometry(
        "ld",
        LD,
        ut,
        m,
        N_l,
        u,
        B_l,
        feat.W.astype(LD),
        K1_isqrt_l,
        log_rho,
        ar.rho,
        N_mp=Nm,
        kept_condition=float(feat.gram_eigs[0] / feat.gram_eigs[feat.rank - 1]),
    )


def _rho_pow(geom: Geometry, k: int) -> np.ndarray:
    """rho_j^k in the geometry's precision (0 where rho_j = 0 and k > 0)."""
    if geom.precision == "fp64":
        return geom.rho.astype(np.float64) ** k
    import mpmath as mp

    with mp.workdps(MP_DPS):
        return np.array(
            [
                (LD(1) if k == 0 else LD(0)) if lr is None else LD(mp.nstr(mp.exp(k * lr), 30))
                for lr in geom.log_rho
            ],
            dtype=LD,
        )


# Population covariance operator
def _gamma_terms_fp64(G: np.ndarray):
    G = 0.5 * (G + G.T)
    g = linalg.eigvalsh(G)
    if np.abs(g).max() >= 1.0:
        raise FloatingPointError(f"|eig(G)| >= 1: {np.abs(g).max()}")
    d = G.shape[0]
    H1 = linalg.solve(np.eye(d) - G @ G, G, assume_a="sym").T  # G (I - G^2)^{-1} (commuting)
    H1 = 0.5 * (H1 + H1.T)
    H2 = 0.5 * ((G @ H1) + (G @ H1).T)
    ldet = float(np.sum(np.log1p(-(g**2))))
    return H1, H2, ldet


def _gamma_terms_mp(N, rho2t: list):
    """H1, H2, logdet(I - G^2) in mpmath for G = N diag(rho^{2t}) N^T (N an mpmath matrix),
    rounded to long double."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        d = N.rows
        G = mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                G[i, j] = mp.fsum(N[i, k] * rho2t[k] * N[j, k] for k in range(d))
        E, Q = mp.eigsy(G)
        H1 = mp.matrix(d, d)
        H2 = mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                H1[i, j] = mp.fsum(Q[i, k] * (E[k] / (1 - E[k] ** 2)) * Q[j, k] for k in range(d))
                H2[i, j] = mp.fsum(
                    Q[i, k] * (E[k] ** 2 / (1 - E[k] ** 2)) * Q[j, k] for k in range(d)
                )
        ldet = mp.fsum(mp.log1p(-(E[k] ** 2)) for k in range(d))
        gmax = max(abs(E[k]) for k in range(d))
        if gmax >= 1:
            raise FloatingPointError("|eig(G)| >= 1")
        return _mp_to_ld(mp, H1), _mp_to_ld(mp, H2), LD(mp.nstr(ldet, 30))


def centered_moments(geom: Geometry, t: int) -> np.ndarray:
    """Branched covariance D_t[a,b] = Cov(k(q_a,Y), k(q_b,Z)), shape (M,M).

    The branch cross-covariance is Gamma_t = L diag(rho^{2t}) L^T.
    With K1 = Lambda+C, u_a = K1^-1/2(q_a-mu), G = K1^-1/2 Gamma_t K1^-1/2,
    H1 = G(I-G^2)^-1 and H2 = G H1,
        D_t[a,b] = m_a m_b expm1(u_a^T H1 u_b
            - (u_a^T H2 u_a + u_b^T H2 u_b)/2 - logdet(I-G^2)/2).
    This avoids subtracting nearly equal moments as Gamma_t tends to zero.
    """
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if geom.precision == "fp64":
        r2 = _rho_pow(geom, 2 * t)
        G = (geom.N * r2[None, :]) @ geom.N.T
        H1, H2, ldet = _gamma_terms_fp64(G)
    else:
        import mpmath as mp

        with mp.workdps(MP_DPS):
            # rho_j = 0 (rate-rule underflow): rho_j^0 = 1, rho_j^{2t} = 0 for t >= 1
            r2 = [
                (mp.mpf(1) if t == 0 else mp.mpf(0)) if lr is None else mp.exp(2 * t * lr)
                for lr in geom.log_rho
            ]
            H1, H2, ldet = _gamma_terms_mp(geom.N_mp, r2)
    ut = geom.ut
    q2 = np.einsum("ij,ij->i", ut @ H2, ut)
    x = (ut @ H1) @ ut.T - 0.5 * (q2[:, None] + q2[None, :]) - 0.5 * ldet
    D = np.outer(geom.m, geom.m) * np.expm1(x)
    assert np.isfinite(D.astype(np.float64)).all(), "non-finite centered moments"
    return D


def centered_moments_from_gamma(geom: Geometry, Gamma: np.ndarray) -> np.ndarray:
    """Branched covariance for an arbitrary symmetric Gamma, for float64 checks."""
    if geom.precision != "fp64":
        raise ValueError("fp64 geometry required")
    G = geom.K1_isqrt @ (0.5 * (Gamma + Gamma.T)) @ geom.K1_isqrt
    H1, H2, ldet = _gamma_terms_fp64(G)
    ut = geom.ut
    q2 = np.einsum("ij,ij->i", ut @ H2, ut)
    x = (ut @ H1) @ ut.T - 0.5 * (q2[:, None] + q2[None, :]) - 0.5 * ldet
    return np.outer(geom.m, geom.m) * np.expm1(x)


def whiten_info(geom: Geometry, D: np.ndarray) -> tuple[np.ndarray, float]:
    """Return symmetrized W^T D W in float64 and its pre-symmetrization asymmetry.

    Raises if relative asymmetry exceeds max(1e-6, 100 u cond(K_kept)).
    """
    S = (geom.W.T @ D @ geom.W).astype(np.float64)
    asym = sm.relative_asymmetry(S)
    u = U64 if geom.precision == "fp64" else ULD
    cond = geom.kept_condition if math.isfinite(geom.kept_condition) else 1.0
    if asym > max(1e-6, 100.0 * u * cond):
        raise FloatingPointError(
            f"whitened operator asymmetric (relative {asym:.2e}, cond {cond:.2e})"
        )
    return 0.5 * (S + S.T), asym


def whiten(geom: Geometry, D: np.ndarray) -> np.ndarray:
    return whiten_info(geom, D)[0]


def operator_S_info(geom: Geometry, t: int) -> tuple[np.ndarray, float]:
    return whiten_info(geom, centered_moments(geom, t))


def operator_S(geom: Geometry, t: int) -> np.ndarray:
    return whiten(geom, centered_moments(geom, t))


# Embedded Hermite modes
def downward_closure(indices) -> list:
    """Every beta <= alpha (componentwise, including 0) for alpha in ``indices``, ordered by total
    degree (so each recurrence input precedes its output)."""
    need, stack = set(), [tuple(int(v) for v in a) for a in indices]
    while stack:
        a = stack.pop()
        if a in need:
            continue
        need.add(a)
        for j, k in enumerate(a):
            if k > 0:
                stack.append(a[:j] + (k - 1,) + a[j + 1 :])
    return sorted(need, key=lambda a: (sum(a), tuple(-v for v in a)))


def hermite_coefficients(geom: Geometry, closure: list) -> tuple[dict, np.ndarray]:
    """Hermite embedding coefficients at each landmark, ordered by closure.

    E[k(q,X) exp(s^T y - |s|^2/2)] = mu_pi(q) exp(s^T u_q + s^T B s/2),
    where u_q = L^T K1^-1(q-mu) and B = -L^T K1^-1 L. Hence
    E[k(q,X) psi_alpha(X)] = mu_pi(q) a_alpha(q), with a_0 = 1 and
        a_{beta+e_j} = (u_j a_beta + sum_k B_jk sqrt(beta_k) a_{beta-e_k})
                       / sqrt(beta_j+1).
    closure must be downward-closed and degree-ordered. The recurrence uses
    the first nonzero coordinate j of alpha.
    """
    d = geom.u.shape[1]
    if tuple(closure[0]) != (0,) * d:
        raise ValueError("closure must start with the zero multi-index")
    index = {tuple(a): i for i, a in enumerate(closure)}
    if len(index) != len(closure):
        raise ValueError("duplicate multi-indices")
    max_deg = max(sum(a) for a in closure)
    sq = np.sqrt(np.arange(max_deg + 2, dtype=geom.dtype))
    A = np.empty((len(closure), geom.u.shape[0]), dtype=geom.dtype)
    A[0] = 1
    for i in range(1, len(closure)):
        a = closure[i]
        j = next(k for k in range(d) if a[k] > 0)
        beta = a[:j] + (a[j] - 1,) + a[j + 1 :]
        val = geom.u[:, j] * A[index[beta]]
        for k in range(d):
            if beta[k] > 0:
                val = (
                    val
                    + geom.B[j, k]
                    * sq[beta[k]]
                    * A[index[beta[:k] + (beta[k] - 1,) + beta[k + 1 :]]]
                )
        A[i] = val / sq[a[j]]
    if not np.isfinite(A.astype(np.float64)).all():
        raise FloatingPointError("non-finite Hermite coefficients")
    return index, A


def embedded_modes(geom: Geometry, indices) -> np.ndarray:
    """w_alpha = W^T (mu_pi * a_alpha) for each alpha in ``indices`` (columns), float64 (r, K)."""
    idx = [tuple(int(v) for v in a) for a in indices]
    if any(sum(a) == 0 for a in idx):
        raise ValueError("the constant mode is not a slow mode")
    index, A = hermite_coefficients(geom, downward_closure(idx))
    G = A[[index[a] for a in idx]] * geom.m[None, :]
    w = (geom.W.T @ G.T).astype(np.float64)
    assert w.shape == (geom.W.shape[1], len(idx))
    return w


def low_degree_checks(ar: ControlledAR, feat: LandmarkFeatures, geom: Geometry) -> dict:
    """Constant, linear and quadratic landmark moments from direct Gaussian-tilting formulas
    (slogdet / solve, independent of the eigen-based geometry and the recurrence)."""
    d = ar.d
    Lam = np.diag(feat.ls**2)
    K1 = Lam + ar.C
    diff = feat.Q - ar.mu[None, :]
    sgn, ld1 = np.linalg.slogdet(np.eye(d) + ar.C @ np.linalg.inv(Lam))
    assert sgn > 0
    m_dir = np.exp(-0.5 * ld1 - 0.5 * np.einsum("ij,ij->i", diff, np.linalg.solve(K1, diff.T).T))
    # tilted law of x under k(q, x) pi(x) / mu_pi(q): N(mu + C K1^{-1}(q - mu), C - C K1^{-1} C)
    tilt_mean_y = np.linalg.solve(ar.L, (ar.C @ np.linalg.solve(K1, diff.T))).T  # (M, d)
    tilt_cov_x = ar.C - ar.C @ np.linalg.solve(K1, ar.C)
    Linv = np.linalg.inv(ar.L)
    V = Linv @ tilt_cov_x @ Linv.T
    deg2 = [a for a in multi_indices(d, 2) if sum(a) == 2]
    deg1 = [a for a in multi_indices(d, 1) if sum(a) == 1]
    index, A = hermite_coefficients(geom, downward_closure(deg1 + deg2))
    A = A.astype(np.float64)
    m_rec = geom.m.astype(np.float64)
    out = {"constant_rel": float(np.abs(m_rec - m_dir).max() / np.abs(m_dir).max())}
    lin_err = 0.0
    for j, a in enumerate(deg1):
        got = m_rec * A[index[a]]
        ref = m_dir * tilt_mean_y[:, j]
        lin_err = max(lin_err, float(np.abs(got - ref).max() / np.abs(ref).max()))
    out["linear_rel"] = lin_err
    quad_err = 0.0
    for a in deg2:
        nz = [k for k in range(d) if a[k]]
        if len(nz) == 1:
            i = nz[0]
            ref = m_dir * (tilt_mean_y[:, i] ** 2 + V[i, i] - 1.0) / math.sqrt(2.0)
        else:
            i, j = nz
            ref = m_dir * (tilt_mean_y[:, i] * tilt_mean_y[:, j] + V[i, j])
        got = m_rec * A[index[a]]
        quad_err = max(quad_err, float(np.abs(got - ref).max() / np.abs(ref).max()))
    out["quadratic_rel"] = quad_err
    return out


def hermite_1d(y: np.ndarray, nmax: int) -> np.ndarray:
    """He_n(y)/sqrt(n!) for n = 0..nmax (stable normalised recurrence)."""
    out = np.empty((nmax + 1,) + y.shape)
    out[0] = 1.0
    if nmax >= 1:
        out[1] = y
    for n in range(1, nmax):
        out[n + 1] = (y * out[n] - math.sqrt(n) * out[n - 1]) / math.sqrt(n + 1)
    return out


def mc_moment_check(
    ar: ControlledAR,
    feat: LandmarkFeatures,
    geom: Geometry,
    rng: np.random.Generator,
    n: int,
    n_landmarks: int,
    max_degree: int,
    chunk: int = 65536,
) -> dict:
    """Monte Carlo E[k(q, X) psi_alpha(X)] (exact draws, y = U^T C^{-1/2}(x - mu) evaluated directly)
    versus the generating function, for the first ``n_landmarks`` landmarks and all modes with
    1 <= |alpha| <= max_degree. Returns the largest |z| and its Bonferroni two-sided 0.001 level."""
    from scipy import stats

    d = ar.d
    alphas = [a for a in multi_indices(d, max_degree) if sum(a) > 0]
    index, A = hermite_coefficients(geom, downward_closure(alphas))
    Qs = feat.Q[:n_landmarks]
    exact = (
        geom.m.astype(np.float64)[None, :n_landmarks]
        * A.astype(np.float64)[[index[a] for a in alphas]][:, :n_landmarks]
    )
    s1 = np.zeros_like(exact)
    s2 = np.zeros_like(exact)
    Ly = ar.U.T @ ar.C_half_inv  # y = Ly (x - mu)
    done = 0
    while done < n:
        b = min(chunk, n - done)
        X = ar.sample_target(rng, b)
        y = (X - ar.mu[None, :]) @ Ly.T
        H = hermite_1d(y.T, max_degree)  # (deg+1, d, b)
        k = rbf_aniso(X, Qs, feat.ls)  # (b, L)
        for i, a in enumerate(alphas):
            psi = np.ones(b)
            for j in range(d):
                if a[j]:
                    psi = psi * H[a[j], j]
            v = k * psi[:, None]
            s1[i] += v.sum(0)
            s2[i] += (v * v).sum(0)
        done += b
    mean = s1 / n
    se = np.sqrt(np.maximum(s2 / n - mean**2, 0.0) / (n - 1))
    if (se <= 0).any():
        raise FloatingPointError("zero Monte Carlo standard error")
    z = (mean - exact) / se
    n_tests = z.size
    return {
        "n": int(n),
        "n_landmarks": int(n_landmarks),
        "max_degree": int(max_degree),
        "n_tests": int(n_tests),
        "max_abs_z": float(np.abs(z).max()),
        "bonferroni_z_0.001": float(stats.norm.isf(0.0005 / n_tests)),
        "rms_z": float(np.sqrt((z**2).mean())),
        "max_abs_err": float(np.abs(mean - exact).max()),
    }


def hermite_reconstruction(
    geom: Geometry, max_degree: int, times, S_closed: dict, log_rho: np.ndarray
) -> dict:
    """Truncated Hermite reconstruction S_t^(D) = sum_{0<|alpha|<=D} rho_alpha^{2t} w_alpha w_alpha^T
    versus the closed form. The omitted part is PSD with trace
    tau_t = sum_{|alpha|>D} rho_alpha^{2t} |w_alpha|^2 <= rho_1^{2t(D+1)} (tr S_0 - sum_{|alpha|<=D}|w_alpha|^2)
    (an a-priori bound on ||S_t - S_t^(D)||_op given exact inputs)."""
    d = geom.u.shape[1]
    closure = multi_indices(d, max_degree)
    index, A = hermite_coefficients(geom, closure)
    del index
    Wm = (geom.W.T @ (A[1:] * geom.m[None, :]).T).astype(np.float64)  # (r, K)
    deg = np.array([sum(a) for a in closure[1:]])
    lr = np.array([sum(k * log_rho[j] for j, k in enumerate(a) if k) for a in closure[1:]])
    norms2 = np.einsum("ij,ij->j", Wm, Wm)
    tr0 = float(np.trace(S_closed[0]))
    by_degree = [float(norms2[deg <= k].sum()) for k in range(1, max_degree + 1)]
    tail0 = tr0 - by_degree[-1]
    rho1 = float(np.exp(max(log_rho)))
    out = {
        "max_degree": int(max_degree),
        "n_modes": int(len(closure) - 1),
        "trace_S0": tr0,
        "trace_by_max_degree": by_degree,
        "tail_trace_t0": tail0,
        "times": {},
    }
    for t in times:
        wt = np.exp(2 * t * lr)
        SD = (Wm * wt[None, :]) @ Wm.T
        diff = S_closed[t] - SD
        out["times"][int(t)] = {
            "op_diff": sm_opnorm(diff),
            "op_S": sm_opnorm(S_closed[t]),
            "trace_diff": float(np.trace(diff)),
            "tail_bound": max(tail0, 0.0) * rho1 ** (2 * t * (max_degree + 1)),
        }
    return out


def sm_opnorm(M: np.ndarray) -> float:
    M = 0.5 * (M + M.T)
    ev = linalg.eigvalsh(M)
    return float(max(abs(ev[0]), abs(ev[-1])))


# TICA calibration
SELECTION_VERSION = "b-real-selection-v1"


def pooled_lag_covariances(paths: list, lag: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair-symmetrised C0, C_lag pooled over trajectories, centred by the pooled empirical mean."""
    if lag < 1:
        raise ValueError("lag must be >= 1")
    mean = np.concatenate(paths, axis=0).mean(axis=0)
    r = paths[0].shape[1]
    S00 = np.zeros((r, r))
    S01 = np.zeros((r, r))
    n = 0
    for F in paths:
        sm.as_f64("F", F, 2)
        if F.shape[0] <= lag:
            raise ValueError(f"path of length {F.shape[0]} too short for lag {lag}")
        Y0 = F[:-lag] - mean
        Y1 = F[lag:] - mean
        S00 += Y0.T @ Y0 + Y1.T @ Y1
        S01 += Y0.T @ Y1
        n += F.shape[0] - lag
    C0 = S00 / (2.0 * n)
    Clag = (S01 + S01.T) / (2.0 * n)
    return C0, Clag, sm.relative_asymmetry(S01)


def tica_fit(paths: list, lag: int, rel_cutoff: float) -> sm.SlowModes:
    C0, Clag, asym = pooled_lag_covariances(paths, lag)
    return sm.tica_symmetric(C0, Clag, rel_cutoff, n_modes=None, order="abs", lag_asymmetry=asym)


def half_split_sensitivity(rho_a: np.ndarray, rho_b: np.ndarray, m: int) -> float:
    """Half the maximum difference in |rho_j| between fits, through j = m+2."""
    k = min(m + 2, rho_a.size, rho_b.size)
    if k < m + 1:
        raise ValueError("need at least m + 1 eigenvalues in each half")
    a = np.sort(np.abs(rho_a))[::-1][:k]
    b = np.sort(np.abs(rho_b))[::-1][:k]
    return float(0.5 * np.max(np.abs(a - b)))


def subspace_sine(A: np.ndarray, B: np.ndarray) -> float:
    """sin of the largest principal angle between two equal-dimensional column spans (= ||P_A - P_B||)."""
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"equal dimensions required, got {A.shape[1]} and {B.shape[1]}")
    Qa, Qb = sm.orthonormal_basis(A), sm.orthonormal_basis(B)
    s = linalg.svdvals(Qb - Qa @ (Qa.T @ Qb))
    assert s[0] <= 1.0 + 1e-12, s
    return float(min(s[0], 1.0))


def angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(sm.largest_principal_angle(A, B)))


@dataclass(frozen=True)
class SelectionRules:
    gap_units: float = 5.0  # gap > gap_units * half-split sensitivity
    max_half_angle_deg: float = 10.0
    max_stability_angle_deg: float = 10.0
    max_counts: int = 2
    m_max: int = 16


def calibration_candidates(
    fits: dict, candidate_lags: list, partner: dict, rules: SelectionRules
) -> list:
    """Gap, split-half and lag-stability checks for each (lag, m).

    fits[lag] contains pooled, A and B fits; partner[lag] supplies the longer
    lag used for the stability comparison.
    """
    rows = []
    for lag in candidate_lags:
        P, Fa, Fb = fits[lag]["pooled"], fits[lag]["A"], fits[lag]["B"]
        Pn = fits[partner[lag]]["pooled"]
        rank = min(P.whitening.rank, Fa.whitening.rank, Fb.whitening.rank, Pn.whitening.rank)
        rho = np.abs(P.eigenvalues)
        for m in range(1, rules.m_max + 1):
            row = {
                "lag": int(lag),
                "partner_lag": int(partner[lag]),
                "m": int(m),
                "rank": int(rank),
            }
            reasons = []
            if m + 1 > rank:
                row["eligible"] = False
                row["reasons"] = [f"m+1={m + 1} exceeds whitening rank {rank}"]
                rows.append(row)
                continue
            gap = float(rho[m - 1] - rho[m])
            eps = half_split_sensitivity(Fa.eigenvalues, Fb.eigenvalues, m)
            ratio = gap / eps if eps > 0 else float("inf")
            half = angle_deg(Fa.embedded[:, :m], Fb.embedded[:, :m])
            stab = angle_deg(P.embedded[:, :m], Pn.embedded[:, :m])
            rho_n = np.abs(Pn.eigenvalues)
            row.update(
                {
                    "rho_m": float(rho[m - 1]),
                    "rho_m1": float(rho[m]),
                    "gap": gap,
                    "half_split_sensitivity": eps,
                    "gap_ratio": ratio,
                    "half_split_angle_deg": half,
                    "stability_angle_deg": stab,
                    "partner_rho_m": float(rho_n[m - 1]),
                    "partner_rho_m1": float(rho_n[m]),
                    "per_step_rate_m": float(rho[m - 1] ** (1.0 / lag)),
                    "per_step_rate_m1": float(rho[m] ** (1.0 / lag)),
                }
            )
            if not ratio > rules.gap_units:
                reasons.append(f"gap/sensitivity {ratio:.3g} <= {rules.gap_units}")
            if not half <= rules.max_half_angle_deg:
                reasons.append(f"half-split angle {half:.3g} deg > {rules.max_half_angle_deg}")
            if not stab <= rules.max_stability_angle_deg:
                reasons.append(
                    f"lag-{partner[lag]} stability angle {stab:.3g} deg > {rules.max_stability_angle_deg}"
                )
            row["eligible"] = not reasons
            row["reasons"] = reasons
            rows.append(row)
    return rows


def select_counts(rows: list, rules: SelectionRules) -> dict:
    """Select the smallest eligible lag, then up to max_counts largest gap ratios."""
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return {"outcome": "NONE_ELIGIBLE", "lag": None, "selected": []}
    lag = min(r["lag"] for r in eligible)
    cands = sorted(
        (r for r in eligible if r["lag"] == lag), key=lambda r: (-r["gap_ratio"], r["m"])
    )
    chosen = cands[: rules.max_counts]
    return {
        "outcome": "SELECTED"
        if len(chosen) == rules.max_counts
        else f"ONLY_{len(chosen)}_ELIGIBLE",
        "lag": int(lag),
        "selected": [int(r["m"]) for r in chosen],
        "eligible_at_lag": [int(r["m"]) for r in cands],
    }


# Principal angles from exact Gram matrices
SINE_PREC_BITS = (256, 512)
SUM_PREC_BITS = 1200  # retain terms from about 1e-300 to 1


def _mp():
    import mpmath as mp

    return mp


def dyadic_integers(X) -> tuple[np.ndarray, int]:
    """Exact X = X_int 2^E for a float64 / long-double array or an mpmath matrix (object ints).

    frexp mantissas scaled by 2^53 (float64) or 2^64 (x87 long double) are integers; mpf values
    carry (man, exp). Entries of an mpmath matrix are read without re-rounding.
    """
    if isinstance(X, np.ndarray):
        bits = {np.dtype(np.float64): 53, np.dtype(LD): 64}.get(X.dtype)
        if bits is None:
            raise TypeError(f"float64 or long double required, got {X.dtype}")
        if not np.isfinite(X).all():
            raise FloatingPointError("non-finite entries")
        frac, ex = np.frexp(X)
        man = [int(v) for v in np.ldexp(frac, bits).ravel()]
        exp = [int(v) - bits for v in ex.ravel()]
        shape = X.shape
    else:
        mp = _mp()
        shape = (X.rows, X.cols)
        man, exp = [], []
        for i in range(X.rows):
            for j in range(X.cols):
                v = X[i, j]
                if not isinstance(v, mp.mpf) or not mp.isfinite(v):
                    raise TypeError(f"finite mpf entry required, got {v!r} ({type(v)})")
                sign, mm, ee, _ = v._mpf_  # (mpf.man_exp drops the sign)
                man.append(-int(mm) if sign else int(mm))
                exp.append(int(ee))
    nz = [e for mm, e in zip(man, exp) if mm]
    if not nz:
        raise ValueError("zero matrix")
    E = min(nz)
    out = np.empty(len(man), dtype=object)
    for k, (mm, e) in enumerate(zip(man, exp)):
        out[k] = mm << (e - E) if mm else 0
    return out.reshape(shape), E


def mp_from_dyadic(X_int: np.ndarray, E: int):
    """The exact mpmath matrix X_int 2^E (built with make_mpf, so no rounding to the context)."""
    mp = _mp()
    from mpmath import libmp

    rows, cols = X_int.shape
    out = mp.matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            v = int(X_int[i, j])
            out[i, j] = mp.mp.make_mpf(libmp.from_man_exp(v, E)) if v else mp.mpf(0)
    return out


def _fraction_inverse(G: list) -> list:
    """Exact inverse of a positive definite integer matrix by rational Gauss-Jordan.

    Positive pivots make row exchanges unnecessary.
    """
    from fractions import Fraction

    m = len(G)
    A = [[Fraction(int(G[i][j])) for j in range(m)] + [Fraction(int(i == j)) for j in range(m)]
         for i in range(m)]  # fmt: skip
    for k in range(m):
        p = A[k][k]
        if p <= 0:
            raise sm.RankDeficientError(f"Gram matrix not positive definite (pivot {k} = {p})")
        A[k] = [v / p for v in A[k]]
        for i in range(m):
            if i != k and A[i][k] != 0:
                f = A[i][k]
                A[i] = [a - f * b for a, b in zip(A[i], A[k])]
    return [row[m:] for row in A]


@dataclass(frozen=True)
class ExactSpan:
    """Column span as integers, with exact Gram matrix and rational inverse.

    The common power-of-two scale is dropped because it does not affect the span.
    """

    X: np.ndarray  # (r, m) object ints
    gram: list  # m x m ints
    gram_inv: list  # m x m Fractions

    @property
    def m(self) -> int:
        return int(self.X.shape[1])


def exact_span(X) -> ExactSpan:
    Xi, _ = dyadic_integers(X)
    if Xi.ndim != 2 or not (1 <= Xi.shape[1] <= Xi.shape[0]):
        raise ValueError(f"need an r x m matrix with 1 <= m <= r, got shape {Xi.shape}")
    G = Xi.T.dot(Xi).tolist()
    return ExactSpan(Xi, G, _fraction_inverse(G))


def float_nearest(x) -> float:
    """float64 nearest to an mpmath value (read without re-rounding)."""
    from mpmath import libmp

    return libmp.to_float(x._mpf_, rnd=libmp.round_nearest)


def float_up(x) -> float:
    """Smallest float64 >= x for a nonnegative mpmath value or float."""
    mp = _mp()
    from mpmath import libmp

    v = x if isinstance(x, mp.mpf) else mp.mpf(float(x))
    if v < 0:
        raise ValueError(f"float_up needs a nonnegative value, got {v}")
    return libmp.to_float(v._mpf_, rnd=libmp.round_ceiling)


def span_sine(A: ExactSpan, B: ExactSpan, precs: tuple = SINE_PREC_BITS) -> dict:
    """Largest principal-angle sine between exact stored spans of equal dimension.

    Form N = Ga - Gab Gb^-1 Gab^T over rational numbers, where Ga = A^T A,
    Gb = B^T B and Gab = A^T B. With Ga = L L^T, the squared sine is
    lambda_max(L^-1 N L^-T). Evaluate this m x m problem at both bit precisions.

    Returns the higher-precision sin_mp, its nearest float64 sin, an arithmetic
    allowance 2|s(P1)-s(P2)| + 2^-P1, and serialization error |fl(s)-s|.
    Both errors are rounded up to float64.
    """
    from fractions import Fraction

    mp = _mp()
    if A.m != B.m or A.X.shape[0] != B.X.shape[0]:
        raise ValueError(f"spans of shapes {A.X.shape} and {B.X.shape}")
    m = A.m
    Gab = A.X.T.dot(B.X).tolist()
    Ginv = B.gram_inv
    Xs = [[sum((Ginv[i][k] * int(Gab[j][k]) for k in range(m)), Fraction(0)) for j in range(m)]
          for i in range(m)]  # Gb^{-1} Gab^T  # fmt: skip
    N = [[int(A.gram[i][j]) - sum((int(Gab[i][k]) * Xs[k][j] for k in range(m)), Fraction(0))
          for j in range(m)] for i in range(m)]  # fmt: skip
    if any(N[i][j] != N[j][i] for i in range(m) for j in range(i)):
        raise ArithmeticError("exact Schur complement is not symmetric")
    vals = []
    for P in precs:
        with mp.workprec(int(P)):
            Ga = mp.matrix([[mp.mpf(int(v)) for v in row] for row in A.gram])
            Nm = mp.matrix([[mp.mpf(f.numerator) / f.denominator for f in row] for row in N])
            Li = mp.inverse(mp.cholesky(Ga))
            M = Li * Nm * Li.T
            lam = max(mp.eigsy((M + M.T) / 2, eigvals_only=True))
            s = mp.sqrt(lam) if lam > 0 else mp.mpf(0)
            if s > 1 + mp.ldexp(1, -int(precs[0])):
                raise ArithmeticError(f"sine {s} exceeds one")
            vals.append(min(s, mp.mpf(1)))
    s1, s2 = vals
    f = float_nearest(s2)
    with mp.workprec(SUM_PREC_BITS):
        arith = 2 * abs(s1 - s2) + mp.ldexp(1, -int(precs[0]))
        ser = abs(mp.mpf(f) - s2)
    return {
        "sin_mp": s2,
        "sin": f,
        "arith": float_up(arith),
        "serialization": float_up(ser),
        "arith_mp": arith,
        "serialization_mp": ser,
    }


def dk_sine_allowance(e: float, gap: float):
    """Davis-Kahan sin theta allowance min{1, e / (gap - e)} (1 if gap <= e), as an mpmath value
    from the float64 inputs (evaluated at high precision, so it is not itself rounded down)."""
    mp = _mp()
    with mp.workprec(SUM_PREC_BITS):
        if not gap > e:
            return mp.mpf(1)
        v = mp.mpf(float(e)) / (mp.mpf(float(gap)) - mp.mpf(float(e)))
        return v if v < 1 else mp.mpf(1)


def mode_term(sv_min: float, dW: float):
    """True-mode allowance dW / (sigma_min - dW) (1 if sigma_min <= dW), mpmath value: bounds
    sin theta(span W_m^computed, span W_m) when ||W_m^computed - W_m|| <= dW (declared margin)."""
    mp = _mp()
    with mp.workprec(SUM_PREC_BITS):
        if not sv_min > dW:
            return mp.mpf(1)
        v = mp.mpf(float(dW)) / (mp.mpf(float(sv_min)) - mp.mpf(float(dW)))
        return v if v < 1 else mp.mpf(1)


def sine_uncertainty(*terms) -> float:
    """min{1, sum of the terms} rounded up to float64 (terms: mpmath values or floats)."""
    mp = _mp()
    with mp.workprec(SUM_PREC_BITS):
        tot = mp.mpf(0)
        for x in terms:
            tot += x if isinstance(x, mp.mpf) else mp.mpf(float(x))
        return 1.0 if tot >= 1 else float_up(tot)


def asin_deg_change(a, b) -> float:
    """|asin(a) - asin(b)| in degrees for two mpmath sines (evaluated at high precision)."""
    mp = _mp()
    with mp.workprec(SUM_PREC_BITS):
        return float_up(abs(mp.degrees(mp.asin(a)) - mp.degrees(mp.asin(b))))


def eigvec_residual_check(S_int: np.ndarray, Es: int, V: np.ndarray, lam: np.ndarray) -> dict:
    """A-posteriori check of the float64 eigensolver on one computed block (exact arithmetic).

    For the stored float64 S = S_int 2^Es, vectors V (r x m) and values lam (m,), R = S V - V diag(lam)
    and F = V^T V - I are formed exactly. With Q an exact orthonormal basis of span(V),
    ||S Q - Q (Q^T S Q)|| <= ||R|| / sqrt(1 - ||F||) =: rq. To first order in the eigengap, span(V)
    deviates from the exact top eigenspace of S by rq / gap, which the declared eigensolver term
    r u ||S|| of e_t covers when rq / (r u ||S||) <= 1 (the caller records that ratio).
    """
    mp = _mp()
    if S_int.shape[0] != V.shape[0] or lam.shape != (V.shape[1],):
        raise ValueError(f"shapes S {S_int.shape}, V {V.shape}, lam {lam.shape}")
    m = V.shape[1]
    Vi, Ev = dyadic_integers(V)
    li, El = dyadic_integers(np.asarray(lam, dtype=np.float64).reshape(1, m))
    SV = S_int.dot(Vi)  # exponent Es + Ev
    VL = Vi * li  # exponent Ev + El (broadcast over rows)
    Ec = min(Es + Ev, Ev + El)
    R = SV * (1 << (Es + Ev - Ec)) - VL * (1 << (Ev + El - Ec))
    RtR = R.T.dot(R).tolist()
    VtV = Vi.T.dot(Vi)
    one = 1 << (-2 * Ev) if Ev < 0 else None
    if one is None:
        raise ValueError("unexpected nonnegative exponent for eigenvector entries")
    F = (VtV - np.eye(m, dtype=np.int64).astype(object) * one).tolist()
    with mp.workprec(256):
        lr = max(mp.eigsy(mp.matrix(RtR), eigvals_only=True))
        rn = mp.sqrt(max(lr, mp.mpf(0))) * mp.ldexp(1, Ec)
        fev = mp.eigsy(mp.matrix(F), eigvals_only=True)
        fn = max(abs(x) for x in fev) * mp.ldexp(1, 2 * Ev)
        if fn >= 1:
            raise ArithmeticError(f"computed eigenvectors far from orthonormal ({fn})")
        rq = rn / mp.sqrt(1 - fn)
        # m x m eigenvalues at 256 bits from exact integer inputs (relative error ~1e-70)
        return {"residual": float_up(rn), "orth_defect": float_up(fn), "rq": float_up(rq)}


# Alignment and numerical margins
def theorem_bound(kappa2_upper: float, cm2_lower: float, a_upper: float, b_lower: float, t: int):
    """g_t = min{1, kappa^2 / c_m^2 (a/b)^{2t}} evaluated in mpmath and rounded up."""
    import mpmath as mp

    if not (cm2_lower > 0 and kappa2_upper > 0 and 0 <= a_upper < b_lower <= 1):
        return 1.0, float("inf")
    with mp.workdps(MP_DPS):
        g = (
            mp.mpf(kappa2_upper)
            / mp.mpf(cm2_lower)
            * (mp.mpf(a_upper) / mp.mpf(b_lower)) ** (2 * int(t))
        )
        graw = float(g) * (1 + 4 * U64)
        return (1.0 if g >= 1 else min(1.0, graw)), graw


def eig_desc(S: np.ndarray):
    lam, Q = sm.eigh_desc(S, "S_t", max_rel_asym=1e-12)
    return lam, Q


@dataclass(frozen=True)
class NumericsRules:
    allowance_factor: float = 2.0
    op_factor: float = 20.0
    sin_factor: float = 10.0
    max_precision_change_deg: float = 0.1
    embedded_rank_rtol: float = 1e-8


def operator_allowance(S64: np.ndarray, S_hi: np.ndarray, factor: float) -> dict:
    """e_t = factor ||S_fp64 - S_hi||_op + r u ||S_fp64||_op.

    The first term estimates arithmetic error from the precision comparison;
    the second allows for the eigensolver's backward error. This is a
    conditional numerical margin, not an interval enclosure. No Gram metric
    correction is needed in fixed feature coordinates.
    """
    diff = sm_opnorm(S64 - S_hi)
    nrm = sm_opnorm(S64)
    r = S64.shape[0]
    return {
        "cross_precision_op": diff,
        "op_norm": nrm,
        "eig_backward": r * U64 * nrm,
        "e_t": factor * diff + r * U64 * nrm,
    }


def bound_inputs(
    S0_64: np.ndarray,
    e0: float,
    Wm64: np.ndarray,
    Wm_hi: np.ndarray,
    gap: dict,
    rules: NumericsRules,
) -> dict:
    """Upper kappa^2 and a, lower c_m^2 and b, under the numerical margins."""
    lam1 = float(linalg.eigvalsh(S0_64)[-1])
    sv = linalg.svdvals(Wm64)
    dW = rules.allowance_factor * float(linalg.svdvals(Wm64 - Wm_hi)[0])
    c_low = float(sv[-1]) - dW
    rank_ok = bool(sv[-1] > rules.embedded_rank_rtol * sv[0] and c_low > 0)
    return {
        "m": int(Wm64.shape[1]),
        "kappa2": lam1,
        "kappa2_upper": lam1 + e0,
        "c_m": float(sv[-1]),
        "c_m_squared": float(sv[-1]) ** 2,
        "c_m_margin": dW,
        "c_m_squared_lower": c_low**2 if c_low > 0 else 0.0,
        "sigma_max_Wm": float(sv[0]),
        "embedded_rank_ok": rank_ok,
        "a": gap["a"],
        "b": gap["b"],
        "a_upper": gap["a_upper"],
        "b_lower": gap["b_lower"],
        "strict_true_gap": gap["strict"],
    }


def bound_value(inputs: dict, t: int) -> tuple[float, str, float]:
    """(g_t, status, raw value) for the true-mode envelope."""
    if not (inputs["embedded_rank_ok"] and inputs["strict_true_gap"]):
        return 1.0, "UNAVAILABLE", float("nan")
    g, raw = theorem_bound(
        inputs["kappa2_upper"], inputs["c_m_squared_lower"], inputs["a_upper"], inputs["b_lower"], t
    )
    status = "CONDITIONAL_NUMERICAL_MARGIN" if g < 1.0 else "TRIVIAL(CONDITIONAL_NUMERICAL_MARGIN)"
    return g, status, raw


def operator_eigen(S64: np.ndarray, S_hi: np.ndarray) -> dict:
    """Float64 eigendecompositions of both tiers and the exact integer form of S64, shared by every
    count and span evaluated at one checkpoint (exact eigenvector spans are cached per m)."""
    lam, V = eig_desc(S64)
    lam_hi, V_hi = eig_desc(S_hi)
    S_int, Es = dyadic_integers(S64)
    return {
        "lam": lam,
        "V": V,
        "lam_hi": lam_hi,
        "V_hi": V_hi,
        "S_int": S_int,
        "Es": Es,
        "spans": {},
    }


def eigen_spans(eig: dict, m: int) -> tuple:
    """Exact spans of the computed top-m eigenvectors (float64 tier, higher tier); cached."""
    if m not in eig["spans"]:
        eig["spans"][m] = (exact_span(eig["V"][:, :m]), exact_span(eig["V_hi"][:, :m]))
    return eig["spans"][m]


def true_mode_term(Wm64: np.ndarray, Wm_hi: np.ndarray, rules: "NumericsRules"):
    """dW / (sigma_min(W_m) - dW) with the declared dW = factor ||W_m^fp64 - W_m^hi||_2 (mpmath)."""
    sv = linalg.svdvals(Wm64)
    dW = rules.allowance_factor * float(linalg.svdvals(Wm64 - Wm_hi)[0])
    return mode_term(float(sv[-1]), dW)


def capped_sum_up(*terms) -> float:
    """min{1, sum of the terms} (mpmath sum, rounded up to float64)."""
    return sine_uncertainty(*terms)


def _status(
    row: dict, s: float, e_sin: float, rules: "NumericsRules", unresolved_reason=None
) -> None:
    """Classify operator resolution, precision stability and angle sensitivity."""
    if unresolved_reason is not None:
        row["estimate_status"] = "OPERATOR_UNRESOLVED"
        row["cutoff_reason"] = unresolved_reason
    elif not row["precision_change_deg"] < rules.max_precision_change_deg:
        row["estimate_status"] = "PRECISION_UNSTABLE"
        row["cutoff_reason"] = (
            f"higher-precision angle change {row['precision_change_deg']:.3g} deg >= "
            f"{rules.max_precision_change_deg:g}"
        )
    elif s > rules.sin_factor * e_sin:
        row["estimate_status"] = "RESOLVED"
        row["cutoff_reason"] = ""
    else:
        row["estimate_status"] = "BELOW_NUMERICAL_SENSITIVITY"
        row["cutoff_reason"] = f"s_t={s:.3e} <= {rules.sin_factor:g} e_sin={e_sin:.3e}"


def checkpoint(
    S64: np.ndarray,
    S_hi: np.ndarray,
    Wm64: np.ndarray,
    Wm_hi: np.ndarray,
    allowance: dict,
    rules: NumericsRules,
    eig: dict | None = None,
) -> dict:
    """Compare the embedded true modes with the time-t leading eigenspace.

    span_sine evaluates the stored float64 spans; the higher tier checks
    precision sensitivity. The sine uncertainty sums Davis-Kahan, true-mode,
    angle-arithmetic and serialization terms, rounded up. Eigensolver
    residuals are checked against the allowance in e_t. _status classifies
    the point using the operator gap and precision/sensitivity thresholds.
    """
    mp = _mp()
    m = Wm64.shape[1]
    if eig is None:
        eig = operator_eigen(S64, S_hi)
    lam, V, lam_hi = eig["lam"], eig["V"], eig["lam_hi"]
    e = allowance["e_t"]
    lam_m, lam_m1 = float(lam[m - 1]), float(lam[m])
    gap = lam_m - lam_m1
    Vx, Vhx = eigen_spans(eig, m)
    sn = span_sine(exact_span(Wm64), Vx)
    sh = span_sine(exact_span(Wm_hi), Vhx)
    s64 = sn["sin"]
    e_dk = dk_sine_allowance(e, gap)
    e_w = true_mode_term(Wm64, Wm_hi, rules)
    e_sin = sine_uncertainty(e_dk, e_w, sn["arith_mp"], sn["serialization_mp"])
    route = subspace_sine(Wm64, V[:, :m])
    with mp.workprec(SUM_PREC_BITS):
        route_err = float_up(abs(mp.mpf(route) - sn["sin_mp"]))
    res = eigvec_residual_check(eig["S_int"], eig["Es"], V[:, :m], lam[:m])
    r = S64.shape[0]
    row = {
        "lambda_m": lam_m,
        "lambda_m1": lam_m1,
        "eigengap": gap,
        "lambda_m_hi": float(lam_hi[m - 1]),
        "lambda_m1_hi": float(lam_hi[m]),
        "numerical_allowance": e,
        "cross_precision_op": allowance["cross_precision_op"],
        "op_norm": allowance["op_norm"],
        "sin_theta": s64,
        "sin_theta_hi": sh["sin"],
        "angle_deg": math.degrees(math.asin(s64)),
        "angle_deg_hi": math.degrees(math.asin(sh["sin"])),
        "precision_change_deg": asin_deg_change(sn["sin_mp"], sh["sin_mp"]),
        "sin_uncertainty": e_sin,
        "sin_uncertainty_dk": float_up(e_dk),
        "sin_uncertainty_modes": float_up(e_w),
        "sin_uncertainty_arith": sn["arith"],
        "sin_uncertainty_serialization": sn["serialization"],
        "sin_theta_float64_route": route,
        "float64_route_abs_err": route_err,
        "eigvec_residual": res["rq"],
        "eigvec_orth_defect": res["orth_defect"],
        "eigvec_residual_ratio": res["rq"] / (r * U64 * allowance["op_norm"]),
        "lambda_ok": lam_m > rules.op_factor * e,
        "gap_ok": gap > rules.op_factor * e,
    }
    row["precision_ok"] = row["precision_change_deg"] < rules.max_precision_change_deg
    why = None
    if not (row["lambda_ok"] and row["gap_ok"]):
        parts = []
        if not row["lambda_ok"]:
            parts.append(f"lambda_m={lam_m:.3e} <= {rules.op_factor:g} e_t={e:.3e}")
        if not row["gap_ok"]:
            parts.append(f"gap={gap:.3e} <= {rules.op_factor:g} e_t={e:.3e}")
        why = "; ".join(parts)
    _status(row, s64, e_sin, rules, why)
    return row


def main_curve_and_gate(rows: list, min_checkpoints: int, max_sine_bound: float) -> dict:
    """End the main curve at its first failed numerical checkpoint.

    Require min_checkpoints points with nontrivial bounds, including one below
    max_sine_bound. The positive-points variant also requires resolved angles.
    """
    rows = sorted(rows, key=lambda r: r["t"])
    first_fail = None
    for r in rows:
        ok = r["estimate_status"] in ("RESOLVED", "BELOW_NUMERICAL_SENSITIVITY")
        if first_fail is None and not ok:
            first_fail = r["t"]
        r["on_main_curve"] = first_fail is None
    main = [r for r in rows if r["on_main_curve"]]
    nontriv = [r for r in main if r["sin_angle_upper"] < 1.0]
    strict = [r for r in nontriv if r["estimate_status"] == "RESOLVED"]
    return {
        "first_failed_checkpoint": first_fail,
        "n_main": len(main),
        "n_resolved_nontrivial": len(nontriv),
        "any_bound_below": any(r["sin_angle_upper"] < max_sine_bound for r in nontriv),
        "gate_pass": len(nontriv) >= min_checkpoints
        and any(r["sin_angle_upper"] < max_sine_bound for r in nontriv),
        "n_resolved_nontrivial_positive": len(strict),
        "gate_pass_positive_points": len(strict) >= min_checkpoints
        and any(r["sin_angle_upper"] < max_sine_bound for r in strict),
        "nontrivial_times": [r["t"] for r in nontriv],
        "positive_nontrivial_times": [r["t"] for r in strict],
    }


def mp_spot_check(
    ar: ControlledAR, feat: LandmarkFeatures, pairs: np.ndarray, t: int, dps: int = 60
) -> np.ndarray:
    """Evaluate E[k k] - m_a m_b directly in mpmath for selected landmark pairs.

    Independent check of the expm1 formula; uses the polar factor of U so
    L L^T = C. At t=512 subtraction loses roughly 23 digits. Returns mpmath
    values for comparison before rounding in spot_check_errors.
    """
    import mpmath as mp

    d = ar.d
    with mp.workdps(dps):
        Cm = mp.matrix(ar.C.tolist())
        Um = mp.matrix(ar.U.tolist())
        C_half = _mp_sym_fn(mp, Cm, mp.sqrt)
        Ubar = Um * _mp_sym_fn(mp, Um.T * Um, lambda x: 1 / mp.sqrt(x))
        Lm = C_half * Ubar
        r2 = [mp.mpf(float(r)) ** (2 * t) for r in ar.rho]  # mpmath: 0 ** 0 == 1
        Gam = mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                Gam[i, j] = mp.fsum(Lm[i, k] * r2[k] * Lm[j, k] for k in range(d))
        lam = [mp.mpf(float(x)) ** 2 for x in feat.ls]
        S2 = mp.matrix(2 * d, 2 * d)
        K1 = mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                S2[i, j] = S2[d + i, d + j] = Cm[i, j]
                S2[i, d + j] = S2[d + i, j] = Gam[i, j]
                K1[i, j] = Cm[i, j]
            S2[i, i] += lam[i]
            S2[d + i, d + i] += lam[i]
            K1[i, i] += lam[i]
        S2i, K1i = mp.inverse(S2), mp.inverse(K1)
        ldlam = mp.fsum(mp.log(x) for x in lam)
        logc2 = ldlam - mp.log(mp.det(S2)) / 2  # log (|Lam2| / |Lam2 + S|)^{1/2}, |Lam2| = |Lam|^2
        logc1 = (ldlam - mp.log(mp.det(K1))) / 2
        mu = [mp.mpf(float(x)) for x in ar.mu]
        out = []
        for a, b in pairs:
            va = [mp.mpf(float(feat.Q[a, i])) - mu[i] for i in range(d)]
            vb = [mp.mpf(float(feat.Q[b, i])) - mu[i] for i in range(d)]
            v = va + vb
            q = mp.fsum(v[i] * S2i[i, j] * v[j] for i in range(2 * d) for j in range(2 * d))
            qa = mp.fsum(va[i] * K1i[i, j] * va[j] for i in range(d) for j in range(d))
            qb = mp.fsum(vb[i] * K1i[i, j] * vb[j] for i in range(d) for j in range(d))
            Ekk = mp.exp(logc2 - q / 2)
            out.append(Ekk - mp.exp(2 * logc1 - (qa + qb) / 2))
        return out


def exact_mpf(x: np.ndarray) -> list:
    """Exact mpmath values of the entries of a float64 or long-double vector."""
    mp = _mp()
    from mpmath import libmp

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"vector required, got shape {x.shape}")
    out = []
    for v in x:
        if v == 0:
            out.append(mp.mpf(0))
            continue
        xi, E = dyadic_integers(np.array([[v]], dtype=x.dtype))
        out.append(mp.mp.make_mpf(libmp.from_man_exp(int(xi[0, 0]), E)))  # exact, no re-rounding
    return out


def spot_check_errors(ref: list, values: np.ndarray) -> dict:
    """max |value - ref| / max |ref| with the subtraction done in mpmath on the exact stored values
    (float64 or long double) and the reference at full precision; only the final statistic is
    rounded (up) to float64."""
    mp = _mp()
    vals = exact_mpf(values)
    if len(vals) != len(ref):
        raise ValueError(f"{len(vals)} values for {len(ref)} references")
    with mp.workprec(SUM_PREC_BITS):
        scale = max(abs(x) for x in ref)
        err = max(abs(v - x) for v, x in zip(vals, ref))
        return {"max_abs_ref": float_nearest(scale), "rel_err": float_up(err / scale)}


def learned_mode_error(What_x: ExactSpan, W_x: ExactSpan, e_w) -> dict:
    """Mode error sin theta(W^, W_m) of one learned span against the stored float64 embedded true
    modes (exact sine) and its numerical uncertainty e_mode = true-mode term + angle arithmetic,
    which bounds |measured - sin theta(W^, W_m)| (triangle inequality for projector distances)."""
    mp = _mp()
    md = span_sine(What_x, W_x)
    with mp.workprec(SUM_PREC_BITS):
        unc = (e_w if isinstance(e_w, mp.mpf) else mp.mpf(float(e_w))) + md["arith_mp"]
    return {
        "mode_error_sin": md["sin"],
        "mode_error_deg": math.degrees(math.asin(md["sin"])),
        "mode_error_mp": md["sin_mp"],
        "mode_uncertainty_mp": unc,
        "mode_uncertainty": float_up(unc),
    }


def learned_checkpoint(
    What: np.ndarray,
    What_x: ExactSpan,
    Vx: ExactSpan,
    Vhx: ExactSpan,
    V64: np.ndarray,
    S64: np.ndarray,
    lam64: np.ndarray,
    mode: dict,
    true_row: dict,
    g_t: float,
    rules: NumericsRules,
) -> dict:
    """Compare a learned span with the time-t optimum and the true slow modes.

    The learned envelope is min(1, mode error + e_mode + g_t), rounded up.
    Operator checks reuse the true-span checkpoint at the same (t,m).
    The residual ratio w(W_hat)/sqrt(lambda_{m+1}) is resolved only if
    lambda_{m+1} > op_factor e_t and
    w^2 > op_factor e_t (1 + amplification^2).
    """
    mp = _mp()
    m = What.shape[1]
    e = true_row["numerical_allowance"]
    gap = true_row["eigengap"]
    sn = span_sine(What_x, Vx)
    sh = span_sine(What_x, Vhx)
    s = sn["sin"]
    e_dk = dk_sine_allowance(e, gap)
    e_sin = sine_uncertainty(e_dk, sn["arith_mp"], sn["serialization_mp"])
    upper = capped_sum_up(mode["mode_error_mp"], mode["mode_uncertainty_mp"], g_t)
    route = subspace_sine(What, V64[:, :m])
    with mp.workprec(SUM_PREC_BITS):
        route_err = float_up(abs(mp.mpf(route) - sn["sin_mp"]))
    row = {
        "sin_theta": s,
        "sin_theta_hi": sh["sin"],
        "angle_deg": math.degrees(math.asin(s)),
        "precision_change_deg": asin_deg_change(sn["sin_mp"], sh["sin_mp"]),
        "sin_uncertainty": e_sin,
        "sin_uncertainty_dk": float_up(e_dk),
        "sin_uncertainty_arith": sn["arith"],
        "sin_uncertainty_serialization": sn["serialization"],
        "sin_theta_float64_route": route,
        "float64_route_abs_err": route_err,
        "mode_error_sin": mode["mode_error_sin"],
        "mode_error_deg": mode["mode_error_deg"],
        "mode_uncertainty": mode["mode_uncertainty"],
        "learned_sin_upper": upper,
        "learned_angle_upper": math.degrees(math.asin(upper)),
    }
    row["bound_satisfied"] = s <= upper + e_sin
    unresolved = (
        true_row["cutoff_reason"] if true_row["estimate_status"] == "OPERATOR_UNRESOLVED" else None
    )
    _status(row, s, e_sin, rules, unresolved)
    lam_m1 = float(lam64[m])
    row["d_m"] = math.sqrt(lam_m1) if lam_m1 > 0 else 0.0
    try:
        wc = sm.constrained_width_cov(S64, What)
    except sm.RankDeficientError as exc:
        row.update(
            {
                "residual_ratio": float("nan"),
                "residual_ratio_status": "RANK_DEFICIENT",
                "residual_ratio_note": str(exc)[:160],
            }
        )
        return row
    row["width"] = wc["width"]
    row["amplification"] = wc["amplification"]
    ok = (
        true_row["estimate_status"] != "OPERATOR_UNRESOLVED"
        and lam_m1 > rules.op_factor * e
        and wc["width_sq"] > rules.op_factor * e * (1.0 + wc["amplification"] ** 2)
    )
    row["residual_ratio"] = wc["width"] / row["d_m"] if row["d_m"] > 0 else float("nan")
    row["residual_ratio_status"] = "RESOLVED" if ok else "UNRESOLVED"
    return row


# Exact integer whitening
def _float_matrix_to_int(Wf: np.ndarray) -> tuple[np.ndarray, int]:
    """Exact integer representation W = W_int * 2^-E of a float64 matrix (dyadic rationals)."""
    E = 0
    for x in Wf.ravel():
        if x != 0.0:
            den = float(x).as_integer_ratio()[1]
            E = max(E, den.bit_length() - 1)
    out = np.empty(Wf.shape, dtype=object)
    for idx, x in np.ndenumerate(Wf):
        num, den = float(x).as_integer_ratio()
        out[idx] = num * ((1 << E) // den)
    return out, E


_EXACT_STATE: dict = {}


def _exact_block(cols: np.ndarray) -> np.ndarray:
    D, W = _EXACT_STATE["D"], _EXACT_STATE["W"]
    T = D.dot(W[:, cols])
    return W.T.dot(T)


def exact_integer_operator(
    ar: ControlledAR, feat: LandmarkFeatures, t: int, dps: int, workers: int = 1
) -> dict:
    """Whiten high-precision centered moments using exact integer arithmetic.

    Round the mpmath D_t to D_int 2^-F, with entrywise error <= 2^-(F+1).
    Write the stored float64 basis as W_int 2^-E. The integer product then
    gives S = S_int 2^-(F+2E). Also return the inputs for high-precision
    Ritz vectors and embedded true modes.
    """
    import multiprocessing as mpc

    import mpmath as mp

    d, M = ar.d, feat.M
    with mp.workdps(dps):
        Cm = mp.matrix(ar.C.tolist())
        Um = mp.matrix(ar.U.tolist())
        C_half = _mp_sym_fn(mp, Cm, mp.sqrt)
        Ubar = Um * _mp_sym_fn(mp, Um.T * Um, lambda x: 1 / mp.sqrt(x))
        Lm = C_half * Ubar
        K1 = mp.matrix(ar.C.tolist())
        for i in range(d):
            K1[i, i] += mp.mpf(float(feat.ls[i])) ** 2
        K1_isqrt = _mp_sym_fn(mp, K1, lambda x: 1 / mp.sqrt(x))
        K1_inv = _mp_sym_fn(mp, K1, lambda x: 1 / x)
        N = K1_isqrt * Lm
        r2 = [mp.mpf(float(r)) ** (2 * t) for r in ar.rho]
        G = mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                G[i, j] = mp.fsum(N[i, k] * r2[k] * N[j, k] for k in range(d))
        E_, Q_ = mp.eigsy(G)
        H1, H2 = mp.matrix(d, d), mp.matrix(d, d)
        for i in range(d):
            for j in range(d):
                H1[i, j] = mp.fsum(
                    Q_[i, k] * (E_[k] / (1 - E_[k] ** 2)) * Q_[j, k] for k in range(d)
                )
                H2[i, j] = mp.fsum(
                    Q_[i, k] * (E_[k] ** 2 / (1 - E_[k] ** 2)) * Q_[j, k] for k in range(d)
                )
        ldet = mp.fsum(mp.log1p(-(E_[k] ** 2)) for k in range(d))
        ldlam = mp.fsum(2 * mp.log(mp.mpf(float(x))) for x in feat.ls)
        logc = (ldlam - mp.log(mp.det(K1))) / 2
        mu = [mp.mpf(float(x)) for x in ar.mu]
        diff = [[mp.mpf(float(feat.Q[a, i])) - mu[i] for i in range(d)] for a in range(M)]
        ut = [[mp.fdot(K1_isqrt[i, :], diff[a]) for i in range(d)] for a in range(M)]
        P = [[mp.fdot(H1[i, :], ut[a]) for i in range(d)] for a in range(M)]
        q2 = [mp.fdot(ut[a], [mp.fdot(H2[i, :], ut[a]) for i in range(d)]) for a in range(M)]
        m = [mp.exp(logc - mp.fdot(ut[a], ut[a]) / 2) for a in range(M)]
        D = [[None] * M for _ in range(M)]
        dmax = mp.mpf(0)
        for a in range(M):
            for b in range(a, M):
                x = mp.fdot(P[a], ut[b]) - (q2[a] + q2[b]) / 2 - ldet / 2
                D[a][b] = D[b][a] = m[a] * m[b] * mp.expm1(x)
                dmax = max(dmax, abs(D[a][b]))
        bits = int(dps * 3.33) + 8
        F = bits - int(mp.floor(mp.log(dmax, 2)))
        scale = mp.mpf(2) ** F
        D_int = np.empty((M, M), dtype=object)
        for a in range(M):
            for b in range(a, M):
                D_int[a, b] = D_int[b, a] = int(mp.nint(D[a][b] * scale))
        # Inputs to the Hermite recurrence in the same precision.
        Pt = Lm.T * K1_inv
        u = [[mp.fdot(Pt[i, :], diff[a]) for i in range(d)] for a in range(M)]
        Bm = -(Pt * Lm)
    W_int, E = _float_matrix_to_int(feat.W)
    _EXACT_STATE.update({"D": D_int, "W": W_int})
    r = feat.rank
    blocks = [c for c in np.array_split(np.arange(r), max(1, workers * 4)) if c.size]
    if workers > 1:
        with mpc.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_exact_block, blocks)
    else:
        parts = [_exact_block(c) for c in blocks]
    S_int = np.empty((r, r), dtype=object)
    for c, part in zip(blocks, parts):
        S_int[:, c] = part
    _EXACT_STATE.clear()
    if any(S_int[i, j] != S_int[j, i] for i in range(r) for j in range(i)):
        raise ArithmeticError("exact integer whitening product is not symmetric")
    return {
        "t": int(t),
        "dps": int(dps),
        "S_int": S_int,
        "exponent": F + 2 * E,
        "F": F,
        "E": E,
        "D_round_abs": float(mp.mpf(2) ** -(F + 1)),
        "D_max_abs": float(dmax),
        "m_mp": m,
        "u_mp": u,
        "B_mp": Bm,
    }


def exact_to_float(S_int: np.ndarray, exponent: int) -> np.ndarray:
    """Correctly rounded float64 of S_int * 2^-exponent (entrywise)."""
    out = np.empty(S_int.shape)
    for idx, v in np.ndenumerate(S_int):
        out[idx] = float(Fraction_(v, exponent))
    return out


def Fraction_(v: int, exponent: int):
    from fractions import Fraction

    return Fraction(v, 1 << exponent)


def exact_true_modes(ex: dict, feat: LandmarkFeatures, alphas, dps: int):
    """w_alpha = W^T (mu_pi * a_alpha) in mpmath for the exact tier (W's float64 values exact)."""
    import mpmath as mp

    with mp.workdps(dps):
        u, B, m = ex["u_mp"], ex["B_mp"], ex["m_mp"]
        M, d = len(u), len(u[0])
        closure = downward_closure(alphas)
        index = {a: i for i, a in enumerate(closure)}
        A = [[mp.mpf(1)] * M]
        for a in closure[1:]:
            j = next(k for k in range(d) if a[k] > 0)
            beta = a[:j] + (a[j] - 1,) + a[j + 1 :]
            vec = [u[q][j] * A[index[beta]][q] for q in range(M)]
            for k in range(d):
                if beta[k] > 0:
                    lower = A[index[beta[:k] + (beta[k] - 1,) + beta[k + 1 :]]]
                    c = B[j, k] * mp.sqrt(beta[k])
                    vec = [vec[q] + c * lower[q] for q in range(M)]
            s = mp.sqrt(a[j])
            A.append([v / s for v in vec])
        Wm = mp.matrix([[mp.mpf(float(x)) for x in row] for row in feat.W.tolist()])
        cols = []
        for a in alphas:
            g = [m[q] * A[index[tuple(a)]][q] for q in range(M)]
            cols.append([mp.fdot(Wm[:, i], g) for i in range(feat.rank)])
        return mp.matrix([[cols[c][i] for c in range(len(cols))] for i in range(feat.rank)])


def mp_subspace_eigen(ex: dict, k: int, iters: int, dps: int, V0: np.ndarray) -> dict:
    """Top-k Ritz pairs of the exact S by block subspace iteration + Rayleigh-Ritz in mpmath, from
    float64 starting vectors. Returns Ritz values (descending), vectors (mp), residual norms."""
    import mpmath as mp

    with mp.workdps(dps):
        r = ex["S_int"].shape[0]
        sc = mp.mpf(2) ** (-ex["exponent"])
        S = mp.matrix(r, r)
        for i in range(r):
            for j in range(r):
                S[i, j] = mp.mpf(int(ex["S_int"][i, j])) * sc
        Y = mp.matrix(V0[:, :k].tolist())

        def orth(Y):
            Q = Y.copy()
            for _ in range(2):
                for j in range(k):
                    for i in range(j):
                        c = mp.fdot(Q[:, i], Q[:, j])
                        for q in range(r):
                            Q[q, j] -= c * Q[q, i]
                    nrm = mp.sqrt(mp.fdot(Q[:, j], Q[:, j]))
                    for q in range(r):
                        Q[q, j] /= nrm
            return Q

        def matmul(S, Y):
            Z = mp.matrix(r, k)
            for i in range(r):
                row = S[i, :]
                for j in range(k):
                    Z[i, j] = mp.fdot(row, Y[:, j])
            return Z

        hist = []
        Y = orth(Y)
        for _ in range(iters):
            Z = matmul(S, Y)
            H = Y.T * Z
            H = (H + H.T) / 2
            th, Qh = mp.eigsy(H)
            order = sorted(range(k), key=lambda i: -th[i])
            Qh = mp.matrix([[Qh[i, j] for j in order] for i in range(k)])
            th = [th[i] for i in order]
            Y = Y * Qh
            Z = Z * Qh
            R = [
                mp.sqrt(mp.fdot(Z[:, j] - th[j] * Y[:, j], Z[:, j] - th[j] * Y[:, j]))
                for j in range(k)
            ]
            hist.append([float(x) for x in R])
            Y = orth(Z)
        Z = matmul(S, Y)
        H = Y.T * Z
        H = (H + H.T) / 2
        th, Qh = mp.eigsy(H)
        order = sorted(range(k), key=lambda i: -th[i])
        Qh = mp.matrix([[Qh[i, j] for j in order] for i in range(k)])
        th = [th[i] for i in order]
        Y = Y * Qh
        Z = Z * Qh
        R = [
            mp.sqrt(mp.fdot(Z[:, j] - th[j] * Y[:, j], Z[:, j] - th[j] * Y[:, j])) for j in range(k)
        ]
        return {"theta": th, "Y": Y, "residuals": R, "residual_history": hist, "S_mp": S}
