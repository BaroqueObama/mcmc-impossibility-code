"""TICA slow modes and diagnostic subspaces in finite RKHS feature coordinates.

Feature paths have shape (T, r), with float64 rows f(X_s). The feature basis
must be orthonormal in the RKHS. TICA solves C_lag u = rho C_0 u by symmetric
whitening, with u^T C_0 u = 1 and embedded RKHS direction w = C_0 u.

The lag-2t path estimate of Sigma_t requires a stationary reversible sampler.
It does not apply to nonreversible transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

EPS = np.finfo(np.float64).eps


class RankDeficientError(ValueError):
    """A basis or constraint matrix has lower numerical rank than required."""


def as_f64(name: str, x: np.ndarray, ndim: int) -> np.ndarray:
    """Return ``x`` after checking type, dtype, dimensionality and finiteness (no conversion)."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(x).__name__}")
    if x.dtype != np.float64:
        raise TypeError(f"{name} must be float64, got {x.dtype}")
    if x.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name} has non-finite entries")
    return x


def relative_asymmetry(S: np.ndarray) -> float:
    """``||S - S^T||_F / ||S||_F`` (0 for the zero matrix)."""
    nrm = np.linalg.norm(S)
    return 0.0 if nrm == 0.0 else float(np.linalg.norm(S - S.T) / nrm)


def symmetric_part(S: np.ndarray, name: str = "S", max_rel_asym: float | None = None):
    """Return ``((S + S^T)/2, relative asymmetry)``; raise if asymmetry exceeds ``max_rel_asym``."""
    S = as_f64(name, S, 2)
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"{name} must be square, got {S.shape}")
    asym = relative_asymmetry(S)
    if max_rel_asym is not None and asym > max_rel_asym:
        raise ValueError(f"{name} relative asymmetry {asym:.3e} exceeds {max_rel_asym:.3e}")
    return (S + S.T) / 2.0, asym


def eigh_desc(S: np.ndarray, name: str = "S", max_rel_asym: float = 1e-12):
    """Eigen-decomposition of a symmetric matrix with eigenvalues in descending order."""
    S, _ = symmetric_part(S, name, max_rel_asym)
    lam, Q = linalg.eigh(S, check_finite=False)
    lam, Q = lam[::-1].copy(), Q[:, ::-1].copy()
    assert (lam[:-1] >= lam[1:]).all(), "eigenvalues not sorted"
    return lam, Q


@dataclass(frozen=True)
class Whitening:
    """Eigen-whitening of a PSD matrix ``C0`` with an explicit relative cutoff.

    Kept eigenpairs satisfy ``lam > threshold = max(rel_cutoff * lam_max, abs_floor)``.
    ``transform = basis / sqrt(eigenvalues)`` maps feature vectors to whitened coordinates
    (``transform.T @ C0 @ transform = I``); ``sqrt_factor = basis * sqrt(eigenvalues)``.
    """

    basis: np.ndarray
    eigenvalues: np.ndarray
    all_eigenvalues: np.ndarray
    rel_cutoff: float
    abs_floor: float
    threshold: float

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def n_dropped(self) -> int:
        return int(self.all_eigenvalues.size - self.eigenvalues.size)

    @property
    def most_negative(self) -> float:
        """Most negative eigenvalue of C0 (0 if none): a PSD-violation measure."""
        return float(min(self.all_eigenvalues.min(), 0.0))

    @property
    def transform(self) -> np.ndarray:
        return self.basis / np.sqrt(self.eigenvalues)

    @property
    def sqrt_factor(self) -> np.ndarray:
        return self.basis * np.sqrt(self.eigenvalues)


def psd_whitening(
    C0: np.ndarray, rel_cutoff: float, abs_floor: float = 0.0, max_rel_asym: float = 1e-12
) -> Whitening:
    """Whiten C0, retaining eigenvalues above max(rel_cutoff * lam_max, abs_floor).

    Raises RankDeficientError if no eigenvalue is retained.
    """
    if not (0.0 < rel_cutoff < 1.0):
        raise ValueError(f"rel_cutoff must be in (0, 1), got {rel_cutoff}")
    if abs_floor < 0.0:
        raise ValueError(f"abs_floor must be >= 0, got {abs_floor}")
    lam, Q = eigh_desc(C0, "C0", max_rel_asym)
    if lam[0] <= 0.0:
        raise RankDeficientError(f"C0 has zero usable rank (largest eigenvalue {lam[0]:.3e})")
    threshold = max(rel_cutoff * lam[0], abs_floor)
    keep = lam > threshold
    if not keep.any():
        raise RankDeficientError(
            f"cutoff {threshold:.3e} retains no eigenvalue of C0 (largest {lam[0]:.3e})"
        )
    return Whitening(
        basis=Q[:, keep],
        eigenvalues=lam[keep],
        all_eigenvalues=lam,
        rel_cutoff=float(rel_cutoff),
        abs_floor=float(abs_floor),
        threshold=float(threshold),
    )


def lagged_pairs_covariances(F: np.ndarray, lag: int, center: bool = True):
    """Reversible (pair-symmetrised) covariance estimates from one feature path.

    With ``Y0 = F[:-lag]`` and ``Y1 = F[lag:]`` (centred by the full-path mean if ``center``),
    returns ``(C0, Clag, lag_asymmetry)`` where

        C0   = (Y0^T Y0 + Y1^T Y1) / (2 (T - lag)),
        Clag = (Y0^T Y1 + Y1^T Y0) / (2 (T - lag)),

    and ``lag_asymmetry`` is the relative asymmetry of ``Y0^T Y1`` before symmetrisation. This pair
    form guarantees ``|v^T Clag v| <= v^T C0 v``, so whitened eigenvalues lie in ``[-1, 1]``.
    The symmetrisation presumes a reversible sampler.
    """
    F = as_f64("F", F, 2)
    T = F.shape[0]
    if not (1 <= lag < T):
        raise ValueError(f"lag must satisfy 1 <= lag < T={T}, got {lag}")
    Y = F - F.mean(axis=0) if center else F
    Y0, Y1 = Y[: T - lag], Y[lag:]
    n = T - lag
    cross = Y0.T @ Y1
    C0 = (Y0.T @ Y0 + Y1.T @ Y1) / (2.0 * n)
    Clag = (cross + cross.T) / (2.0 * n)
    return C0, Clag, relative_asymmetry(cross)


def static_covariance(F: np.ndarray, center: bool = True) -> np.ndarray:
    """Empirical covariance ``Y^T Y / T`` of a feature path (time-zero operator Sigma_0)."""
    F = as_f64("F", F, 2)
    Y = F - F.mean(axis=0) if center else F
    return (Y.T @ Y) / F.shape[0]


def dynamic_covariance_from_path(F: np.ndarray, t: int, center: bool = True):
    """Estimate ``Sigma_t`` by the symmetrised lag-2t covariance of one stationary path.

    Valid only for a reversible sampler started (approximately) in stationarity.
    Returns ``(Sigma_t_hat, lag_asymmetry)``; ``t = 0`` returns the static covariance.
    """
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if t == 0:
        return static_covariance(F, center), 0.0
    _, Clag, asym = lagged_pairs_covariances(F, 2 * t, center)
    return Clag, asym


@dataclass(frozen=True)
class SlowModes:
    """Result of the symmetric whitened TICA solver.

    ``eigenvalues[j]`` estimates rho_j; ``coefficients[:, j] = u_j`` gives
    ``psi_j(x) = u_j^T (f(x) - mean)`` with ``u_j^T C0 u_j = 1``; ``embedded[:, j] = C0 u_j``
    is the RKHS direction w_j. ``whitening`` records the C0 cutoff and retained rank.
    """

    eigenvalues: np.ndarray
    coefficients: np.ndarray
    embedded: np.ndarray
    whitening: Whitening
    lag_asymmetry: float
    max_abs_eigenvalue: float
    whitened_asymmetry: float


def tica_symmetric(
    C0: np.ndarray,
    Clag: np.ndarray,
    rel_cutoff: float,
    n_modes: int | None = None,
    order: str = "abs",
    abs_floor: float = 0.0,
    lag_asymmetry: float = 0.0,
) -> SlowModes:
    """Solve Clag u = rho C0 u after symmetrizing Clag and whitening C0.

    Sort by descending |rho| for order="abs", or rho for order="signed".
    Requesting more modes than the whitening rank raises RankDeficientError.
    """
    if order not in ("abs", "signed"):
        raise ValueError(f"order must be 'abs' or 'signed', got {order!r}")
    C0 = as_f64("C0", C0, 2)
    Clag = as_f64("Clag", Clag, 2)
    if C0.shape != Clag.shape or C0.shape[0] != C0.shape[1]:
        raise ValueError(f"shape mismatch C0 {C0.shape} vs Clag {Clag.shape}")
    Clag_sym, asym_in = symmetric_part(Clag, "Clag")
    wh = psd_whitening(C0, rel_cutoff, abs_floor)
    k = wh.rank if n_modes is None else int(n_modes)
    if k > wh.rank:
        raise RankDeficientError(f"requested {k} modes but whitening retained rank {wh.rank}")
    T = wh.transform
    M = T.T @ Clag_sym @ T
    # roundoff asymmetry of T^T Clag T scales with the whitening condition number
    cond = float(wh.eigenvalues[0] / wh.eigenvalues[-1])
    M, w_asym = symmetric_part(M, "whitened Clag", max_rel_asym=max(1e-10, 100.0 * EPS * cond))
    ev, V = linalg.eigh(M, check_finite=False)
    idx = (
        np.argsort(-np.abs(ev), kind="stable") if order == "abs" else np.argsort(-ev, kind="stable")
    )
    ev, V = ev[idx][:k], V[:, idx][:, :k]
    coefficients = T @ V
    embedded = wh.sqrt_factor @ V
    assert coefficients.shape == (C0.shape[0], k) and embedded.shape == (C0.shape[0], k)
    return SlowModes(
        eigenvalues=ev,
        coefficients=coefficients,
        embedded=embedded,
        whitening=wh,
        lag_asymmetry=float(max(asym_in, lag_asymmetry)),
        max_abs_eigenvalue=float(np.abs(ev).max()) if ev.size else 0.0,
        whitened_asymmetry=float(w_asym),
    )


def tica_from_path(
    F: np.ndarray,
    lag: int,
    rel_cutoff: float,
    n_modes: int | None = None,
    order: str = "abs",
    abs_floor: float = 0.0,
) -> SlowModes:
    """TICA slow modes from one (reversible-sampler) feature path, pair-symmetrised estimator."""
    C0, Clag, asym = lagged_pairs_covariances(F, lag)
    return tica_symmetric(C0, Clag, rel_cutoff, n_modes, order, abs_floor, lag_asymmetry=asym)


def leading_eigenspace(S: np.ndarray, m: int, max_rel_asym: float = 1e-12):
    """Return ``(eigenvalues_desc, V_m)`` with ``V_m`` the leading-m orthonormal eigenvectors."""
    lam, Q = eigh_desc(S, "S", max_rel_asym)
    if not (1 <= m < lam.size):
        raise ValueError(f"m={m} must satisfy 1 <= m < dim={lam.size}")
    return lam, Q[:, :m]


def orthonormal_basis(W: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis of ``span(W)`` (SVD); raise ``RankDeficientError`` if rank < W.shape[1].

    The rank test is ``s_min > rel_tol * s_max`` on the singular values of ``W``.
    """
    W = as_f64("W", W, 2)
    if W.shape[1] == 0 or W.shape[1] > W.shape[0]:
        raise ValueError(f"W must have 1 <= m <= r columns, got shape {W.shape}")
    U, s, _ = linalg.svd(W, full_matrices=False, check_finite=False)
    if s[0] == 0.0 or s[-1] <= rel_tol * s[0]:
        raise RankDeficientError(
            f"W has numerical rank < {W.shape[1]} (singular values {s[0]:.3e} .. {s[-1]:.3e})"
        )
    return U


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles in radians, ascending, with dim span(B) <= dim span(A).

    Both bases are orthonormalized. Use singular values of Qa^T Qb for cosines
    and (I - Qa Qa^T) Qb for sines. Switching at 45 degrees retains precision
    for small angles (Knyazev and Argentati, 2002).
    """
    Qa, Qb = orthonormal_basis(A), orthonormal_basis(B)
    if Qb.shape[1] > Qa.shape[1]:
        raise ValueError("principal_angles needs dim span(B) <= dim span(A)")
    cos = linalg.svdvals(Qa.T @ Qb, check_finite=False)  # descending
    sin = linalg.svdvals(Qb - Qa @ (Qa.T @ Qb), check_finite=False)[::-1]
    # singular values of products of orthonormal bases exceed 1 only by roundoff
    assert cos.max() <= 1.0 + 1e-12 and sin.max() <= 1.0 + 1e-12, (cos, sin)
    cos, sin = np.minimum(cos, 1.0), np.minimum(sin, 1.0)
    from_cos = np.arccos(cos)  # ascending
    from_sin = np.arcsin(sin)  # ascending
    return np.where(sin < np.sqrt(0.5), from_sin, from_cos)


def largest_principal_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Largest principal angle (radians) between two equal-dimensional spans."""
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"equal dimensions required, got {A.shape[1]} and {B.shape[1]}")
    return float(principal_angles(A, B)[-1])


def largest_angle_cosine_route(A: np.ndarray, B: np.ndarray) -> float:
    """Largest principal angle from arccos(sigma_min(Qa^T Qb)), for comparison."""
    Qa, Qb = orthonormal_basis(A), orthonormal_basis(B)
    s = linalg.svdvals(Qa.T @ Qb, check_finite=False)
    assert s[0] <= 1.0 + 1e-12, s
    return float(np.arccos(min(s[-1], 1.0)))


def constrained_width_factor(L: np.ndarray, W: np.ndarray, radius: float = 1.0) -> float:
    """Constrained width ``R ||L N||_op`` with ``N`` an orthonormal basis of ``ker(W^T L)``.

    ``L`` is any factor with ``L L^T = Sigma`` (``(r, k)``); ``W`` spans the diagnostic space
    (it is orthonormalised). This is the largest discrepancy in the ellipsoid
    ``{L u : ||u|| <= R}`` that every diagnostic in ``span(W)`` passes exactly. The null space
    is the orthogonal complement of ``range(L^T Q_W)`` from a complete Householder QR; the
    constraint must have full row rank ``m`` (else ``RankDeficientError``).
    """
    L = as_f64("L", L, 2)
    Qw = orthonormal_basis(W)
    if L.shape[0] != Qw.shape[0]:
        raise ValueError(f"L has {L.shape[0]} rows, W has {Qw.shape[0]}")
    m, k = Qw.shape[1], L.shape[1]
    G = L.T @ Qw  # (k, m); ker(W^T L) = range(G)^perp in R^k
    if k <= m:
        raise RankDeficientError(f"factor has {k} columns <= m={m}: constraint leaves no freedom")
    Qfull, R = linalg.qr(G, mode="full", check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.min() <= 1e3 * EPS * max(np.linalg.norm(G), np.finfo(float).tiny):
        raise RankDeficientError(
            f"constraint W^T L is rank deficient (|R_ii| min {diag.min():.3e})"
        )
    N = Qfull[:, m:]
    return float(radius * linalg.svdvals(L @ N, check_finite=False)[0])


def constrained_width_cov(S: np.ndarray, W: np.ndarray, radius: float = 1.0) -> dict:
    """Same width from the covariance: ``R sqrt(lam_max(S - S W (W^T S W)^{-1} W^T S))``.

    Equivalent to :func:`constrained_width_factor` for PSD ``S`` (conditional-covariance form).
    Also returns ``amplification = ||(W^T S W)^{-1} W^T S P_perp||``, which gives the first-order
    sensitivity ``|d(width^2)| <= eps (1 + amplification^2)`` to an operator perturbation of norm
    ``eps``. Raises if ``W^T S W`` is singular to working precision.
    """
    S, _ = symmetric_part(S, "S", max_rel_asym=1e-10)
    Qw = orthonormal_basis(W)
    SW = S @ Qw
    G = Qw.T @ SW
    G = (G + G.T) / 2.0
    lam_g = linalg.eigvalsh(G, check_finite=False)
    if lam_g[0] <= 0.0 or lam_g[0] <= 1e3 * EPS * np.abs(lam_g).max():
        raise RankDeficientError(f"W^T S W is singular to working precision (eigs {lam_g})")
    X = linalg.solve(G, SW.T, assume_a="pos", check_finite=False)  # (m, r) = G^{-1} W^T S
    resid = S - SW @ X
    resid = (resid + resid.T) / 2.0
    top = float(linalg.eigvalsh(resid, check_finite=False)[-1])
    if top < -1e3 * EPS * np.linalg.norm(S, 2):
        raise FloatingPointError(f"conditional covariance has negative top eigenvalue {top:.3e}")
    P_perp = np.eye(S.shape[0]) - Qw @ Qw.T
    amplification = float(np.linalg.norm(X @ P_perp, 2))
    return {
        # a top eigenvalue within roundoff of zero (possibly negative) means width zero;
        # the raw value is kept in width_sq
        "width": float(radius * np.sqrt(top)) if top > 0.0 else 0.0,
        "width_sq": top,
        "amplification": amplification,
    }


def psd_factor(S: np.ndarray, max_rel_asym: float = 1e-12):
    """Factor ``L`` with ``L L^T = P_+(S)`` from ``eigh``, dropping nonpositive eigenvalues.

    Returns ``(L, most_negative_eigenvalue)``. The dropped part has operator norm
    ``|most_negative|``, which callers must carry into their error budget.
    """
    lam, Q = eigh_desc(S, "S", max_rel_asym)
    keep = lam > 0.0
    return Q[:, keep] * np.sqrt(lam[keep]), float(min(lam.min(), 0.0))


def davis_kahan_sin_bound(eps: float, gap: float) -> float:
    """Upper bound on sin(angle) between true and computed leading eigenspaces.

    ``eps`` bounds the operator-norm perturbation and ``gap = lam_m - lam_{m+1}`` (exact or
    computed); returns ``eps / (gap - eps)`` capped at 1, or 1 if ``gap <= eps``.
    """
    if gap <= eps:
        return 1.0
    return float(min(1.0, eps / (gap - eps)))
