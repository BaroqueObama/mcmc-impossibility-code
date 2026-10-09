"""Reversible Gaussian AR(1) model with known Hermite eigenfunctions.

For target N(0, C), A = C^1/2 U diag(rho) U^T C^-1/2 and noise covariance
C - A C A^T. In coordinates U^T C^-1/2 x, the transition eigenfunctions are
products of normalized Hermite polynomials with eigenvalues rho1^a1 rho2^a2.

The population operator in Nystrom coordinates is evaluated in two ways:
Gaussian landmark integrals at lag 2t, and a truncated Hermite expansion.
The latter has tail estimate rho_out^(2t) ||R_0||, where R_0 is approximate.
Numerical error estimates are empirical sensitivities, not certified bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, special

from mcmc.paperexp.slowmodes import EPS, as_f64, eigh_desc, relative_asymmetry

# the sampler


def rotation(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


@dataclass(frozen=True)
class GaussAR:
    """Reversible Gaussian AR(1) with target N(0, diag(c_diag)) and mode rotation U."""

    c_diag: tuple[float, float]
    rotation_deg: float
    rho: tuple[float, float]
    C: np.ndarray = field(init=False, repr=False)
    U: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)
    C_half: np.ndarray = field(init=False, repr=False)
    C_half_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        c = np.asarray(self.c_diag, dtype=np.float64)
        r = np.asarray(self.rho, dtype=np.float64)
        if c.shape != (2,) or (c <= 0).any():
            raise ValueError(f"c_diag must be two positive numbers, got {self.c_diag}")
        if r.shape != (2,) or (np.abs(r) >= 1).any():
            raise ValueError(f"rho must be two numbers in (-1, 1), got {self.rho}")
        C = np.diag(c)
        Ch, Chi = np.diag(np.sqrt(c)), np.diag(1.0 / np.sqrt(c))
        U = rotation(self.rotation_deg)
        A = Ch @ U @ np.diag(r) @ U.T @ Chi
        Q = C - A @ C @ A.T
        Q = (Q + Q.T) / 2.0
        for name, val in (
            ("C", C),
            ("U", U),
            ("A", A),
            ("Q", Q),
            ("C_half", Ch),
            ("C_half_inv", Chi),
        ):
            object.__setattr__(self, name, val)
        if linalg.eigvalsh(Q)[0] <= 0.0:
            raise ValueError("noise covariance C - A C A^T is not positive definite")

    @property
    def rho_arr(self) -> np.ndarray:
        return np.asarray(self.rho, dtype=np.float64)

    def A_power(self, s: int) -> np.ndarray:
        """``A^s`` through the mode diagonalisation (stable for large s)."""
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")
        return self.C_half @ self.U @ np.diag(self.rho_arr**s) @ self.U.T @ self.C_half_inv

    def to_modes(self, X: np.ndarray) -> np.ndarray:
        """Rows ``y = U^T C^{-1/2} x``."""
        return as_f64("X", X, 2) @ self.C_half_inv @ self.U

    def from_modes(self, Y: np.ndarray) -> np.ndarray:
        """Rows ``x = C^{1/2} U y``."""
        return as_f64("Y", Y, 2) @ self.U.T @ self.C_half

    def sample_target(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """IID target draws via the Cholesky factor of C (independent of the mode map)."""
        return rng.standard_normal((n, 2)) @ linalg.cholesky(self.C, lower=True).T

    def step(self, rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
        return X @ self.A.T + rng.standard_normal(X.shape) @ linalg.cholesky(self.Q, lower=True).T

    def path(self, rng: np.random.Generator, T: int, x0: np.ndarray | None = None) -> np.ndarray:
        """One trajectory of length T (row 0 = start; stationary draw if ``x0`` is None)."""
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        Lq = linalg.cholesky(self.Q, lower=True)
        X = np.empty((T, 2))
        X[0] = self.sample_target(rng, 1)[0] if x0 is None else np.asarray(x0, dtype=np.float64)
        noise = rng.standard_normal((T - 1, 2)) @ Lq.T
        for s in range(T - 1):
            X[s + 1] = self.A @ X[s] + noise[s]
        return X

    def witnesses(self, lags=(1, 2, 8, 64)) -> dict:
        """Stationarity, reversibility, noise PSD and the two-branch / lag-2t identity."""
        Q_pred = self.C_half @ self.U @ np.diag(1.0 - self.rho_arr**2) @ self.U.T @ self.C_half
        out = {
            "stationarity_resid": float(np.abs(self.A @ self.C @ self.A.T + self.Q - self.C).max()),
            "noise_vs_closed_form": float(np.abs(self.Q - Q_pred).max()),
            "noise_min_eig": float(linalg.eigvalsh(self.Q)[0]),
            "reversibility_resid": float(np.abs(self.C @ self.A.T - self.A @ self.C).max()),
            "A_power_vs_matrix_power": float(
                np.abs(self.A_power(8) - np.linalg.matrix_power(self.A, 8)).max()
            ),
        }
        branch = []
        for t in lags:
            At = self.A_power(t)
            # Cov(Y_t, Z_t) for two branches from one X ~ pi versus Cov(X_0, X_2t)^T
            branch.append(float(np.abs(At @ self.C @ At.T - self.A_power(2 * t) @ self.C).max()))
        out["two_branch_vs_lag2t_max"] = max(branch)
        return out


# kernel / Nystrom


def rbf_kernel(X: np.ndarray, Z: np.ndarray, ell: float, block: int = 4096) -> np.ndarray:
    """``exp(-|x - z|^2 / (2 ell^2))`` from explicit differences (no clamping), blocked."""
    X, Z = as_f64("X", X, 2), as_f64("Z", Z, 2)
    out = np.empty((X.shape[0], Z.shape[0]))
    for s in range(0, X.shape[0], block):
        d = X[s : s + block, None, :] - Z[None, :, :]
        out[s : s + block] = np.exp(-np.einsum("nmk,nmk->nm", d, d) / (2.0 * ell**2))
    return out


@dataclass(frozen=True)
class Nystrom:
    """Whitened Nystrom feature map ``f(x) = H^T k(Z, x)``."""

    Z: np.ndarray
    ell: float
    rel_cutoff: float
    H: np.ndarray
    K_eigenvalues: np.ndarray  # all, descending

    @property
    def rank(self) -> int:
        return int(self.H.shape[1])

    @property
    def kept_condition(self) -> float:
        lam = self.K_eigenvalues[: self.rank]
        return float(lam[0] / lam[-1])

    def features(self, X: np.ndarray, block: int = 4096) -> np.ndarray:
        out = np.empty((X.shape[0], self.rank))
        for s in range(0, X.shape[0], block):
            out[s : s + block] = rbf_kernel(X[s : s + block], self.Z, self.ell) @ self.H
        return out


def nystrom_whitening(Z: np.ndarray, ell: float, rel_cutoff: float) -> Nystrom:
    K = rbf_kernel(Z, Z, ell)
    lam, Q = eigh_desc(K, "K_ZZ", max_rel_asym=1e-15)
    keep = lam > rel_cutoff * lam[0]
    H = Q[:, keep] / np.sqrt(lam[keep])
    return Nystrom(Z=Z, ell=float(ell), rel_cutoff=float(rel_cutoff), H=H, K_eigenvalues=lam)


# Gaussian landmark integrals


def landmark_mean(Z: np.ndarray, C: np.ndarray, ell: float) -> np.ndarray:
    """``E_pi k(z_a, X) = |I + C/ell^2|^{-1/2} exp(-z^T (ell^2 I + C)^{-1} z / 2)``."""
    d = Z.shape[1]
    Minv = np.linalg.inv(ell**2 * np.eye(d) + C)
    return np.linalg.det(np.eye(d) + C / ell**2) ** -0.5 * np.exp(
        -0.5 * np.einsum("ni,ij,nj->n", Z, Minv, Z)
    )


def landmark_cross_moment(Z: np.ndarray, C: np.ndarray, A_s: np.ndarray, ell: float) -> np.ndarray:
    """Uncentred E[k(z_a,X_0) k(z_b,X_s)] with Cov(X_0,X_s) = C A_s^T."""
    d = Z.shape[1]
    G = np.block([[C, C @ A_s.T], [A_s @ C, C]])
    Binv = np.linalg.inv(ell**2 * np.eye(2 * d) + G)
    q1 = np.einsum("ni,ij,nj->n", Z, Binv[:d, :d], Z)
    q2 = np.einsum("ni,ij,nj->n", Z, Binv[d:, d:], Z)
    cross = Z @ Binv[:d, d:] @ Z.T
    return np.linalg.det(np.eye(2 * d) + G / ell**2) ** -0.5 * np.exp(
        -0.5 * (q1[:, None] + q2[None, :] + 2.0 * cross)
    )


def landmark_lag_covariance(Z: np.ndarray, C: np.ndarray, A_s: np.ndarray, ell: float):
    """Centred ``E[k(z_a,X_0) k(z_b,X_s)] - m_a m_b`` without cancellation.

    Writing ``K_inf = blockdiag(ell^2 I + C, ell^2 I + C)`` and ``E`` the off-diagonal lag blocks,
    ``M_ab / (m_a m_b) = exp(z^T P z / 2 - logdet(I + K_inf^{-1/2} E K_inf^{-1/2}) / 2)`` with
    ``P = (K_inf + E)^{-1} E K_inf^{-1}``, so the centred moment is ``m_a m_b expm1(.)``.
    Returns ``(D, info)`` where ``info`` records the asymmetry of ``P``.
    """
    d = Z.shape[1]
    blk = ell**2 * np.eye(d) + C
    zero = np.zeros((d, d))
    K_inf = np.block([[blk, zero], [zero, blk]])
    E = np.block([[zero, C @ A_s.T], [A_s @ C, zero]])
    K_inf_inv = np.linalg.inv(K_inf)
    P = np.linalg.solve(K_inf + E, E @ K_inf_inv)
    p_asym = relative_asymmetry(P)
    P = (P + P.T) / 2.0
    lam_b, V_b = linalg.eigh(blk)
    blk_isqrt = V_b @ np.diag(lam_b**-0.5) @ V_b.T
    K_isqrt = np.block([[blk_isqrt, zero], [zero, blk_isqrt]])
    mu = linalg.eigvalsh(K_isqrt @ E @ K_isqrt)
    if mu.min() <= -1.0:
        raise FloatingPointError(f"lag covariance determinant is not positive: {mu}")
    logdet = float(np.sum(np.log1p(mu)))
    q1 = np.einsum("ni,ij,nj->n", Z, P[:d, :d], Z)
    q2 = np.einsum("ni,ij,nj->n", Z, P[d:, d:], Z)
    cross = Z @ P[:d, d:] @ Z.T
    expo = 0.5 * (q1[:, None] + q2[None, :] + 2.0 * cross) - 0.5 * logdet
    m = landmark_mean(Z, C, ell)
    D = np.outer(m, m) * np.expm1(expo)
    return D, {"P_rel_asym": p_asym}


def sigma_t_landmark(ar: GaussAR, nys: Nystrom, t: int, form: str = "stable"):
    """Whitened Sigma_t from Gaussian landmark integrals at lag 2t.

    Returns ``(Sigma_t, info)`` with the relative asymmetry of the landmark matrix before
    symmetrisation (reversibility makes it symmetric in exact arithmetic).
    """
    A_s = ar.A_power(2 * t)
    if form == "stable":
        D, info = landmark_lag_covariance(nys.Z, ar.C, A_s, nys.ell)
    elif form == "naive":
        m = landmark_mean(nys.Z, ar.C, nys.ell)
        D = landmark_cross_moment(nys.Z, ar.C, A_s, nys.ell) - np.outer(m, m)
        info = {}
    else:
        raise ValueError(f"form must be 'stable' or 'naive', got {form!r}")
    info["landmark_rel_asym"] = relative_asymmetry(D)
    S = nys.H.T @ D @ nys.H
    S = (S + S.T) / 2.0
    assert np.isfinite(S).all(), "non-finite Sigma_t"
    return S, info


# Hermite modes


def hermite_normalized(y: np.ndarray, nmax: int) -> np.ndarray:
    """``He_n(y)/sqrt(n!)`` for n = 0..nmax via the stable normalised recurrence."""
    out = np.empty((nmax + 1,) + y.shape)
    out[0] = 1.0
    if nmax >= 1:
        out[1] = y
    for n in range(1, nmax):
        out[n + 1] = (y * out[n] - np.sqrt(n) * out[n - 1]) / np.sqrt(n + 1)
    return out


@dataclass(frozen=True)
class HermiteModes:
    """Non-constant Hermite products sorted by eigenvalue (descending; ties by degree, a1)."""

    indices: np.ndarray  # (J, 2) int
    eigenvalues: np.ndarray  # (J,)
    max_degree: int

    @property
    def n(self) -> int:
        return int(self.eigenvalues.size)

    def rho_out(self, rho: np.ndarray) -> float:
        """Largest |eigenvalue| among modes of total degree > max_degree."""
        return float(np.abs(rho).max() ** (self.max_degree + 1))


def hermite_modes(rho: np.ndarray, max_degree: int) -> HermiteModes:
    idx = [(a, b) for a in range(max_degree + 1) for b in range(max_degree + 1 - a) if a + b > 0]
    ev = np.array([rho[0] ** a * rho[1] ** b for a, b in idx])
    key = sorted(range(len(idx)), key=lambda j: (-abs(ev[j]), sum(idx[j]), -idx[j][0]))
    return HermiteModes(
        indices=np.array([idx[j] for j in key], dtype=int),
        eigenvalues=ev[key],
        max_degree=int(max_degree),
    )


def mode_values(ar: GaussAR, X: np.ndarray, modes: HermiteModes) -> np.ndarray:
    """``psi_j(x)`` for all modes, shape (N, J)."""
    Y = ar.to_modes(X)
    h1 = hermite_normalized(Y[:, 0], modes.max_degree)
    h2 = hermite_normalized(Y[:, 1], modes.max_degree)
    return h1[modes.indices[:, 0]].T * h2[modes.indices[:, 1]].T


def gauss_hermite_grid(order: int):
    """Tensor probabilists' Gauss-Hermite rule for E over N(0, I_2): (nodes (N^2,2), weights)."""
    x, w = special.roots_hermitenorm(order)
    w = w / np.sqrt(2.0 * np.pi)
    Y = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    return Y, (w[:, None] * w[None, :]).ravel()


def embedded_modes(
    ar: GaussAR, nys: Nystrom, modes: HermiteModes, order: int, block: int = 8192
) -> dict:
    """``w_j = E_pi[f(X) psi_j(X)]`` (= the centred embedding since psi_j has mean zero).

    Also returns the quadrature feature mean ``E f`` for comparison with the closed form.
    """
    Y, wt = gauss_hermite_grid(order)
    W = np.zeros((nys.rank, modes.n))
    mean = np.zeros(nys.rank)
    h_deg = modes.max_degree
    for s in range(0, Y.shape[0], block):
        Yb, wb = Y[s : s + block], wt[s : s + block]
        Fb = nys.features(ar.from_modes(Yb))
        h1 = hermite_normalized(Yb[:, 0], h_deg)
        h2 = hermite_normalized(Yb[:, 1], h_deg)
        Psi = h1[modes.indices[:, 0]].T * h2[modes.indices[:, 1]].T
        W += Fb.T @ (wb[:, None] * Psi)
        mean += Fb.T @ wb
    assert np.isfinite(W).all(), "non-finite embedded modes"
    return {"W": W, "mean": mean, "order": int(order)}


def embedded_modes_converged(
    ar: GaussAR,
    nys: Nystrom,
    modes: HermiteModes,
    start_order: int,
    max_order: int,
    n_check: int,
    rel_tol: float,
) -> dict:
    """Double the quadrature order until the leading ``n_check`` embedded modes and the
    whole-spectrum Gram ``sum_j w_j w_j^T`` change by less than ``rel_tol`` (relative).

    Returns the last two evaluations (``W`` at the final order, ``W_prev`` at half of it) and
    the history. Raises if ``max_order`` is reached without meeting the tolerance.
    """
    hist, prev = [], None
    order = int(start_order)
    while order <= max_order:
        cur = embedded_modes(ar, nys, modes, order)
        if prev is not None:
            Wc, Wp = cur["W"][:, :n_check], prev["W"][:, :n_check]
            col = np.linalg.norm(Wc - Wp, axis=0) / np.linalg.norm(Wc, axis=0)
            G, Gp = cur["W"] @ cur["W"].T, prev["W"] @ prev["W"].T
            gram = float(np.linalg.norm(G - Gp, 2) / np.linalg.norm(G, 2))
            rec = {
                "order": order,
                "prev_order": prev["order"],
                "leading_col_rel_change_max": float(col.max()),
                "gram_rel_change": gram,
            }
            hist.append(rec)
            if col.max() < rel_tol and gram < rel_tol:
                return {
                    "W": cur["W"],
                    "W_prev": prev["W"],
                    "mean": cur["mean"],
                    "order": order,
                    "history": hist,
                }
        prev = cur
        order *= 2
    raise RuntimeError(
        f"embedded-mode quadrature did not converge to {rel_tol} by order {max_order}: {hist}"
    )


def sigma_t_modes(W: np.ndarray, eigenvalues: np.ndarray, t: int) -> np.ndarray:
    """Truncated Hermite expansion sum_j rho_j^(2t) w_j w_j^T."""
    L = W * eigenvalues**t
    return L @ L.T


def eigenfunction_identity_check(
    ar: GaussAR, modes: HermiteModes, X_test: np.ndarray, max_degree: int, noise_order: int
) -> dict:
    """Check ``E[psi_a(A x + xi)] = rho_a psi_a(x)`` by quadrature over the x-space noise.

    The noise is integrated with a tensor Gauss-Hermite rule mapped through the Cholesky factor
    of ``Q = C - A C A^T`` (computed from A and C, not from the mode coordinates). Exact for
    polynomial degree < 2 * noise_order.
    """
    if 2 * noise_order <= max_degree:
        raise ValueError("noise_order too small for exact polynomial quadrature")
    sel = modes.indices.sum(axis=1) <= max_degree
    sub = HermiteModes(modes.indices[sel], modes.eigenvalues[sel], modes.max_degree)
    Eta, wt = gauss_hermite_grid(noise_order)
    Xi = Eta @ linalg.cholesky(ar.Q, lower=True).T
    errs, scale = [], []
    for x in X_test:
        X1 = x[None, :] @ ar.A.T + Xi
        lhs = wt @ mode_values(ar, X1, sub)
        rhs = sub.eigenvalues * mode_values(ar, x[None, :], sub)[0]
        errs.append(np.abs(lhs - rhs).max())
        scale.append(np.abs(rhs).max())
    return {
        "max_abs_err": float(max(errs)),
        "max_abs_value": float(max(scale)),
        "n_modes_checked": int(sub.n),
        "n_points": int(len(X_test)),
    }


def orthonormality_check(ar: GaussAR, modes: HermiteModes, order: int) -> float:
    """``max |E_pi[psi_a psi_b] - delta_ab|`` under the order-``order`` tensor rule."""
    if order <= modes.max_degree:
        raise ValueError("quadrature order must exceed the maximal degree for exactness")
    Y, wt = gauss_hermite_grid(order)
    Psi = mode_values(ar, ar.from_modes(Y), modes)
    G = Psi.T @ (wt[:, None] * Psi)
    return float(np.abs(G - np.eye(modes.n)).max())


# baselines


def coordinate_monitors(
    ar: GaussAR, nys: Nystrom, Sigma0: np.ndarray, order: int, rel_cutoff: float
) -> dict:
    """Least-squares representation of the coordinate functions x_1, x_2 in the feature span.

    ``beta_i = Sigma0^+ b_i`` with ``b_i = E_pi[(f(X) - mu) x_i]`` (quadrature) and an explicit
    relative eigenvalue cutoff in the pseudo-inverse; the monitor ``beta_i^T (f(x) - mu)``
    approximates ``x_i`` and its diagnostic direction is ``beta_i``. Also returns the explained
    variance fraction of each coordinate.
    """
    Y, wt = gauss_hermite_grid(order)
    X = ar.from_modes(Y)
    F = nys.features(X)
    mu = F.T @ wt
    b = (F - mu).T @ (wt[:, None] * X)  # (r, 2); x has mean zero under pi
    lam, Q = eigh_desc(Sigma0, "Sigma0")
    keep = lam > rel_cutoff * lam[0]
    pinv = Q[:, keep] @ np.diag(1.0 / lam[keep]) @ Q[:, keep].T
    beta = pinv @ b
    var = np.diag(ar.C)
    explained = np.einsum("ri,ri->i", b, beta) / var
    return {"beta": beta, "b": b, "explained_fraction": explained, "pinv_rank": int(keep.sum())}


# extended precision


def _mp_orthonormal(mp, A):
    """Modified Gram-Schmidt (twice) on the columns of an mp matrix."""
    r, m = A.rows, A.cols
    Q = A.copy()
    for _ in range(2):
        for j in range(m):
            for i in range(j):
                c = sum(Q[k, i] * Q[k, j] for k in range(r))
                for k in range(r):
                    Q[k, j] -= c * Q[k, i]
            nrm = mp.sqrt(sum(Q[k, j] ** 2 for k in range(r)))
            if nrm == 0:
                raise ValueError("rank-deficient diagnostic span in extended precision")
            for k in range(r):
                Q[k, j] /= nrm
    return Q


def _mp_top_eig_small(mp, G):
    """Largest eigenvalue of a symmetric 1x1 or 2x2 mp matrix (closed form)."""
    if G.rows == 1:
        return G[0, 0]
    if G.rows != 2:
        raise ValueError("only m <= 2 supported in extended precision")
    a, b, c = G[0, 0], (G[0, 1] + G[1, 0]) / 2, G[1, 1]
    return (a + c) / 2 + mp.sqrt(((a - c) / 2) ** 2 + b**2)


def mp_population_at_t(
    W: np.ndarray,
    indices: np.ndarray,
    rho: tuple[float, float],
    t: int,
    spaces: dict,
    dps: int,
) -> dict:
    """Hermite expansion of Sigma_t in mpmath, treating float64 W and rho as exact inputs.

    Higher arithmetic precision does not remove quadrature, whitening, or
    input-rounding error. Modes below 10^-(dps+5) times the largest weight
    rho_j^(2t) ||w_j||^2 are dropped; their summed weight bounds the omitted
    operator norm for these inputs.

    ``spaces`` maps m -> {name: (r,m) basis}. Results include float64 summaries
    and decimal strings under ``mp`` for comparisons before float64 rounding.
    """
    import mpmath as mp

    with mp.workdps(dps):

        def put(rec, mrec, key, x):
            rec[key] = float(x)
            mrec[key] = mp.nstr(x, dps)

        r, J = W.shape
        rho_mp = [mp.mpf(float(x)) for x in rho]
        lam = [rho_mp[0] ** int(a) * rho_mp[1] ** int(b) for a, b in indices]
        col_sq = [mp.fsum(mp.mpf(float(x)) ** 2 for x in W[:, j]) for j in range(J)]
        weight = [lam[j] ** (2 * t) * col_sq[j] for j in range(J)]
        wmax = max(weight)
        thr = wmax * mp.mpf(10) ** (-(dps + 5))
        keep = [j for j in range(J) if weight[j] >= thr]
        dropped = mp.fsum(weight[j] for j in range(J) if weight[j] < thr)
        L = mp.matrix(r, len(keep))
        for c, j in enumerate(keep):
            sc = lam[j] ** t
            for i in range(r):
                L[i, c] = mp.mpf(float(W[i, j])) * sc
        S = L * L.T
        E, Q = mp.eigsy(S)
        order = sorted(range(r), key=lambda i: -E[i])
        eigs = [E[i] for i in order]
        out = {
            "t": int(t),
            "dps": int(dps),
            "n_modes_used": len(keep),
            "dropped_weight": float(dropped),
            "eigs": [float(x) for x in eigs[:6]],
            "m": {},
            "mp": {
                "dropped_weight": mp.nstr(dropped, dps),
                "eigs": [mp.nstr(x, dps) for x in eigs[:6]],
                "m": {},
            },
        }
        for m, sp in spaces.items():
            V = mp.matrix(r, m)
            for c in range(m):
                for i in range(r):
                    V[i, c] = Q[i, order[c]]
            rec, mrec = {}, {}
            put(rec, mrec, "lam_m", eigs[m - 1])
            put(rec, mrec, "lam_m1", eigs[m])
            put(rec, mrec, "gap", eigs[m - 1] - eigs[m])
            d = mp.sqrt(eigs[m]) if eigs[m] > 0 else mp.mpf(0)
            put(rec, mrec, "d_m", d)
            for name, Wsp in sp.items():
                Qw = _mp_orthonormal(mp, mp.matrix(Wsp.tolist()))
                P = V - Qw * (Qw.T * V)
                sin2 = _mp_top_eig_small(mp, P.T * P)
                sin = mp.sqrt(max(sin2, mp.mpf(0)))
                put(rec, mrec, f"theta_{name}", mp.asin(min(sin, mp.mpf(1))))
                SQ = S * Qw
                Gw = Qw.T * SQ
                Sc = S - SQ * mp.inverse(Gw) * SQ.T
                Sc = (Sc + Sc.T) / 2
                top = max(mp.eigsy(Sc, eigvals_only=True))
                width = mp.sqrt(top) if top > 0 else mp.mpf(0)
                put(rec, mrec, f"{name}_width", width)
                put(rec, mrec, f"{name}_width_sq", top)
                # first-order sensitivity ||(W^T S W)^{-1} W^T S P_perp|| (see constrained_width_cov)
                X = mp.inverse(Gw) * SQ.T
                XQ = X * Qw
                amp2 = _mp_top_eig_small(mp, X * X.T - XQ * XQ.T)
                put(rec, mrec, f"{name}_amplification", mp.sqrt(max(amp2, mp.mpf(0))))
                if d > 0:
                    put(rec, mrec, f"{name}_ratio", width / d)
                    put(rec, mrec, f"{name}_ratio_minus_one", width / d - 1)
            out["m"][int(m)] = rec
            out["mp"]["m"][int(m)] = mrec
        return out


def nystrom_metric(nys: Nystrom) -> dict:
    """Departure of the whitened Nystrom coordinates from RKHS-orthonormality.

    The feature map uses ``H`` from a float64 eigendecomposition of an ill-conditioned ``K_ZZ``, so
    ``G = H^T K_ZZ H`` is only approximately the identity. ``G`` is formed in extended precision
    (``np.longdouble``, ~19 digits) from the exact float64 landmarks. Returns ``E = G - I``,
    ``eta = ||E||_op`` and ``T = G^{1/2}``: mapping coordinate vectors ``v -> T v`` makes Euclidean
    inner products equal RKHS inner products of the represented functions (the "metric-corrected"
    coordinates used for the metric sensitivity estimate).
    """
    if np.finfo(np.longdouble).eps > 1e-18:
        raise RuntimeError("np.longdouble is not an extended-precision type on this platform")
    Zl = nys.Z.astype(np.longdouble)
    d = Zl[:, None, :] - Zl[None, :, :]
    K = np.exp(-(d * d).sum(-1) / (2 * np.longdouble(nys.ell) ** 2))
    Hl = nys.H.astype(np.longdouble)
    E = (Hl.T @ K @ Hl - np.eye(nys.rank, dtype=np.longdouble)).astype(np.float64)
    E = (E + E.T) / 2.0
    lam, Q = linalg.eigh(np.eye(nys.rank) + E)
    if lam[0] <= 0.0:
        raise FloatingPointError("Nystrom metric is not positive definite")
    return {"E": E, "eta": float(np.linalg.norm(E, 2)), "T": (Q * np.sqrt(lam)) @ Q.T}


__all__ = [
    "EPS",
    "GaussAR",
    "HermiteModes",
    "Nystrom",
    "coordinate_monitors",
    "eigenfunction_identity_check",
    "embedded_modes",
    "embedded_modes_converged",
    "gauss_hermite_grid",
    "hermite_modes",
    "hermite_normalized",
    "landmark_cross_moment",
    "landmark_lag_covariance",
    "landmark_mean",
    "mode_values",
    "mp_population_at_t",
    "nystrom_metric",
    "nystrom_whitening",
    "orthonormality_check",
    "rbf_kernel",
    "rotation",
    "sigma_t_landmark",
    "sigma_t_modes",
]
