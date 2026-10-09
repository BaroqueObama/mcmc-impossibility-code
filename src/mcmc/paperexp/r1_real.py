"""Projected discrepancies for Gaussian posteriors.

For a Gaussian target N(mu, C), diagonalize the kernel-whitened covariance
Lambda^{-1/2} C Lambda^{-1/2} = R diag(sig^2) R^T and write x = mu + T z,
T = Lambda^{1/2} R diag(sig). The RBF operator has product Mehler eigenpairs;
centering is diag(lambda) - v v^T, v_n = sqrt(lambda_n) E psi_n, and mixes only
the all-even modes. Enumeration stops at rel_floor * lambda_max.

Cell perturbations use two coordinates of z. The remaining coordinates
integrate analytically, giving a Kronecker-product cell Gram. Gibbs-law mean
embeddings are analytic; monitor readings use Gauss-Hermite quadrature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.special as sps

from mcmc.paperexp.r1_suite import composite_gl, mehler_1d, mehler_psi_1d, phi2, phi_interval


def _enumerate_modes(lams: list[np.ndarray], floor: float, max_modes: int) -> np.ndarray:
    """All multi-indices n with prod_j lams[j][n_j] >= floor (lams[j] decreasing)."""
    d = len(lams)
    tail_max = np.ones(d + 1)
    for j in range(d - 1, -1, -1):
        tail_max[j] = tail_max[j + 1] * lams[j][0]
    out: list[tuple] = []
    idx = [0] * d

    def rec(j: int, prod: float):
        if j == d:
            out.append(tuple(idx))
            if len(out) > max_modes:
                raise RuntimeError(f"more than {max_modes} modes above floor {floor}")
            return
        for n in range(lams[j].shape[0]):
            p = prod * lams[j][n]
            if p * tail_max[j + 1] < floor:
                break
            idx[j] = n
            rec(j + 1, p)
        idx[j] = 0

    rec(0, 1.0)
    return np.array(out, dtype=np.int64)


@dataclass
class ExactSpectrum:
    """Centered Gaussian/RBF eigenpairs from a truncated product Mehler expansion."""

    mu: np.ndarray
    C: np.ndarray
    scales: np.ndarray
    ell: float
    sig2: np.ndarray  # (d,) descending, kernel-whitened principal variances
    T: np.ndarray  # x = mu + T z
    modes: np.ndarray  # (M, d)
    lam_modes: np.ndarray  # (M,)
    m_modes: np.ndarray  # (M,) E psi_n
    eig: np.ndarray  # (n_out,) centered eigenvalues, descending
    W: np.ndarray  # (M, n_out): psi_i = sum_n W[n, i] (psi_n - m_n)
    lam1: list = field(default_factory=list)
    m1: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return self.mu.shape[0]


def exact_spectrum(
    mu, C, scales, ell: float, rel_floor: float = 1e-13, n_out: int = 64, max_modes: int = 400_000
) -> ExactSpectrum:
    mu = np.asarray(mu, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    d = mu.shape[0]
    lam_sd = scales * ell
    Cy = C / np.outer(lam_sd, lam_sd)
    sig2, R = np.linalg.eigh(0.5 * (Cy + Cy.T))
    order = np.argsort(sig2)[::-1]
    sig2, R = sig2[order], R[:, order]
    if (sig2 <= 0).any():
        raise np.linalg.LinAlgError(f"C_y not positive definite: {sig2}")
    T = lam_sd[:, None] * R * np.sqrt(sig2)[None, :]
    # per-axis 1-D Mehler problems, n_max from the floor
    lam1, m1 = [], []
    for s2 in sig2:
        l0, _, inf = mehler_1d(float(s2), 1)
        B = inf["B"]
        n_max = max(2, int(math.ceil(math.log(rel_floor) / math.log(B))) + 1)
        lam, m, _ = mehler_1d(float(s2), n_max)
        lam1.append(lam)
        m1.append(m)
    lam_max = float(np.prod([lam[0] for lam in lam1]))
    modes = _enumerate_modes(lam1, rel_floor * lam_max, max_modes)
    lam_modes = np.prod([lam1[j][modes[:, j]] for j in range(d)], axis=0)
    m_modes = np.prod([m1[j][modes[:, j]] for j in range(d)], axis=0)
    even = (modes % 2 == 0).all(1)
    # centered spectrum: odd-containing modes unchanged; even block diag - v v^T
    le = lam_modes[even]
    v = np.sqrt(le) * m_modes[even]
    Mb = np.diag(le) - np.outer(v, v)
    ev_e, U_e = np.linalg.eigh(0.5 * (Mb + Mb.T))
    odd_idx = np.nonzero(~even)[0]
    even_idx = np.nonzero(even)[0]
    cand = [(float(lam_modes[i]), "odd", int(i)) for i in odd_idx]
    cand += [(float(ev_e[k]), "even", int(k)) for k in range(ev_e.shape[0])]
    cand.sort(key=lambda c: -c[0])
    cand = cand[:n_out]
    W = np.zeros((modes.shape[0], len(cand)))
    eig = np.empty(len(cand))
    for i, (val, kind, k) in enumerate(cand):
        eig[i] = val
        if kind == "odd":
            W[k, i] = 1.0
        else:
            W[even_idx, i] = U_e[:, k] * np.sqrt(le / val)
    mu_norm2_exact = float(np.prod(1.0 / np.sqrt(1.0 + 2.0 * sig2)))
    info = {
        "d": d,
        "n_modes": int(modes.shape[0]),
        "n_even_modes": int(even.sum()),
        "rel_floor": rel_floor,
        "captured_trace": float(lam_modes.sum()),
        "omitted_uncentered_trace": float(1.0 - lam_modes.sum()),
        "mu_norm2_exact": mu_norm2_exact,
        "mu_norm2_from_modes": float((v * v).sum()),
        "sig2": sig2.tolist(),
        "n_max_per_axis": [int(lam.shape[0] - 1) for lam in lam1],
    }
    return ExactSpectrum(
        mu=mu, C=C, scales=scales, ell=float(ell), sig2=sig2, T=T, modes=modes,
        lam_modes=lam_modes, m_modes=m_modes, eig=eig, W=W, lam1=lam1, m1=m1, info=info,
    )  # fmt: skip


def interval_edges(n: int, cut: float) -> np.ndarray:
    """Equal-mass N(0,1) quantile edges, ends truncated at +-cut (finite)."""
    e = sps.ndtri(np.arange(n + 1) / n)
    e[0], e[-1] = -cut, cut
    return e


def interval_psi_integrals(
    sig2: float, n_max: int, edges: np.ndarray, h: float, q: int
) -> np.ndarray:
    """I[n, k] = int_{edges[k]}^{edges[k+1]} psi_n(sig z) phi(z) dz, composite GL (panels <= h)."""
    K = edges.shape[0] - 1
    out = np.zeros((n_max + 1, K))
    sig = math.sqrt(sig2)
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        z, w = composite_gl(lo, hi, max(1, int(math.ceil((hi - lo) / h))), q)
        w = w * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        out[:, k] = mehler_psi_1d(sig2, n_max, sig * z) @ w
    return out


def interval_gram_owen(sig2: float, edges: np.ndarray) -> np.ndarray:
    """g[k, k'] = int_{I_k} int_{I_k'} exp(-sig2 (z - z')^2/2) phi(z) phi(z') dz dz' (closed form).

    (z, z') with that weight is (1 + 2 sig2)^{-1/2} N2(0, S), S = [[1+s, s],[s, 1+s]]/(1+2s), s = sig2."""
    s = float(sig2)
    var = (1 + s) / (1 + 2 * s)
    rho = s / (1 + s)
    sq = math.sqrt(1 + 2 * s) / (1 + s)
    e = np.asarray(edges, dtype=np.float64) / math.sqrt(var)  # +-inf ends stay infinite
    F = phi2(e[:, None], e[None, :], rho, sq)
    R = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    return R / math.sqrt(1 + 2 * s)


def interval_gram_gl(sig2: float, edges: np.ndarray, h: float, q: int) -> np.ndarray:
    """Same g by composite GL in both variables (independent route; truncated at the given ends)."""
    K = edges.shape[0] - 1
    nodes, wts, cell = [], [], []
    for k in range(K):
        z, w = composite_gl(
            edges[k], edges[k + 1], max(1, int(math.ceil((edges[k + 1] - edges[k]) / h))), q
        )
        nodes.append(z)
        wts.append(w * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))
        cell.append(np.full(z.shape, k))
    z, w, c = np.concatenate(nodes), np.concatenate(wts), np.concatenate(cell)
    Kz = np.exp(-0.5 * sig2 * (z[:, None] - z[None, :]) ** 2) * w[:, None] * w[None, :]
    S = np.zeros((K, z.shape[0]))
    S[c, np.arange(z.shape[0])] = 1.0
    return S @ Kz @ S.T


def shifted_psi_means(sig2: float, n_max: int, shift: float, n_gh: int = 160) -> np.ndarray:
    """E_{z ~ N(0,1)} psi_n(sig (z + shift)), n = 0..n_max (probabilists' Gauss-Hermite)."""
    t, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    return mehler_psi_1d(sig2, n_max, math.sqrt(sig2) * (t + shift)) @ w


@dataclass
class PlaneSetup:
    spec: ExactSpectrum
    axes: tuple
    n: int
    edges: np.ndarray
    masses: np.ndarray  # (K,)
    G: np.ndarray  # (K, K)
    cell_psi: np.ndarray  # (n_out, K): int_cell psi_i dpi
    info: dict = field(default_factory=dict)


def plane_setup(
    spec: ExactSpectrum, axes: tuple, n: int, h: float, q: int, cut: float, gram: str
) -> PlaneSetup:
    """Cell integrals of every exact eigenfunction and the cell Gram for the plane ``axes``."""
    a, b = axes
    d = spec.d
    edges = interval_edges(n, cut)
    mass1 = phi_interval(sps.ndtri(np.arange(n) / n), sps.ndtri(np.arange(1, n + 1) / n))
    masses = np.outer(mass1, mass1).ravel()
    na, nb = spec.lam1[a].shape[0] - 1, spec.lam1[b].shape[0] - 1
    Ia = interval_psi_integrals(spec.sig2[a], na, edges, h, q)
    Ib = interval_psi_integrals(spec.sig2[b], nb, edges, h, q)
    off = [j for j in range(d) if j not in (a, b)]
    offfac = np.ones(spec.modes.shape[0])
    for j in off:
        offfac *= spec.m1[j][spec.modes[:, j]]
    n_out = spec.W.shape[1]
    cell_psi = np.zeros((n_out, n * n))
    keep = offfac != 0.0
    ma, mb = spec.modes[keep, a], spec.modes[keep, b]
    for i in range(n_out):
        Cf = np.zeros((na + 1, nb + 1))
        np.add.at(Cf, (ma, mb), spec.W[keep, i] * offfac[keep])
        c0 = float(spec.W[:, i] @ spec.m_modes)
        cell_psi[i] = (Ia.T @ Cf @ Ib).ravel() - c0 * masses
    if gram == "owen":
        full = sps.ndtri(np.arange(n + 1) / n)  # untruncated (+-inf) edges: exact
        ga, gb = interval_gram_owen(spec.sig2[a], full), interval_gram_owen(spec.sig2[b], full)
    elif gram == "gl":
        ga = interval_gram_gl(spec.sig2[a], edges, h, q)
        gb = interval_gram_gl(spec.sig2[b], edges, h, q)
    else:
        raise ValueError(f"unknown gram route {gram!r}")
    kfac = float(np.prod([1.0 / math.sqrt(1 + 2 * spec.sig2[j]) for j in off]))
    G = kfac * np.kron(ga, gb)
    info = {
        "axes": [int(a), int(b)],
        "sig2_plane": [float(spec.sig2[a]), float(spec.sig2[b])],
        "kfac": kfac,
        "gram_route": gram,
        "G_sum": float(G.sum()),
        "mu_norm2_exact": spec.info["mu_norm2_exact"],
        "quad": {"h": h, "q": q, "cut": cut},
    }
    return PlaneSetup(spec, (a, b), n, edges, masses, G, cell_psi, info)


def monitor_rows(ps: PlaneSetup, m: int) -> np.ndarray:
    """(m, K) rows int_{A_j} c_i dpi in reading units (<e_i, delta> = rows @ a)."""
    return np.sqrt(ps.spec.eig[:m])[:, None] * ps.cell_psi[:m]


def shifted_readings(spec: ExactSpectrum, shift: np.ndarray, m: int) -> np.ndarray:
    """<e_i, delta> for delta = mu_{N(mu + shift, C)} - mu_pi, i < m (exact up to mode truncation)."""
    sz = np.linalg.solve(spec.T, np.asarray(shift, dtype=np.float64))
    E = np.ones(spec.modes.shape[0])
    for j in range(spec.d):
        tab = shifted_psi_means(spec.sig2[j], spec.lam1[j].shape[0] - 1, float(sz[j]))
        E *= tab[spec.modes[:, j]]
    vals = (E - spec.m_modes) @ spec.W[:, :m]
    return np.sqrt(spec.eig[:m]) * vals


def shifted_mmd2(spec: ExactSpectrum, shift: np.ndarray) -> float:
    """MMD^2(N(mu + shift, C), N(mu, C)) = 2 ||mu_pi||^2 (1 - exp(-shift^T (Lam + 2C)^{-1} shift / 2))."""
    Lam = np.diag((spec.scales * spec.ell) ** 2)
    s = np.asarray(shift, dtype=np.float64)
    qf = float(s @ np.linalg.solve(Lam + 2 * spec.C, s))
    return 2.0 * spec.info["mu_norm2_exact"] * (-math.expm1(-0.5 * qf))


def eval_psi(spec: ExactSpectrum, X: np.ndarray, m: int) -> np.ndarray:
    """(n, m) values psi_i(x) of the centered eigenfunctions (for Monte Carlo witnesses)."""
    Zz = np.linalg.solve(spec.T, (np.asarray(X, dtype=np.float64) - spec.mu[None, :]).T).T
    P = np.ones((X.shape[0], spec.modes.shape[0]))
    for j in range(spec.d):
        tab = mehler_psi_1d(
            spec.sig2[j], spec.lam1[j].shape[0] - 1, math.sqrt(spec.sig2[j]) * Zz[:, j]
        )
        P *= tab[spec.modes[:, j]].T
    return (P - spec.m_modes[None, :]) @ spec.W[:, :m]


NUMERICAL_MARGIN = 1e-8  # additive margin on landmark eigenvalue bounds


def _landmark_operator(gl, kernel, basis, chain, t: int, margin: float):
    """Gaussian landmark operator with an absolute trace tolerance.

    A relative tolerance becomes unreliable when Sigma_t approaches roundoff.
    Allow tr_kept <= tr_full + margin; bounds use max(tau, 0) + margin.
    """
    Gam = chain.cross_cov(t)
    Mc = gl.branched_pair_centered_moment(kernel, basis.Z, chain.mu, chain.C, Gam)
    S = basis.W.T @ Mc @ basis.W
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    order = np.argsort(-w)
    tr_full = gl.trace_sigma_t(kernel, chain.C, Gam)
    tr_kept = float(np.trace(S))
    if tr_kept > tr_full + margin:
        raise FloatingPointError(
            f"t={t}: compressed trace {tr_kept} exceeds full {tr_full} + {margin}"
        )
    return gl.LandmarkOperator(
        t=t, S=S, eigvals=w[order], eigvecs=V[:, order], trace_full=tr_full, trace_kept=tr_kept
    )


def gibbs_projected_residuals(
    spec: ExactSpectrum,
    model,
    landmarks: np.ndarray,
    m_list,
    rtol: float,
) -> dict:
    """Projected discrepancies for the Gibbs laws N(mu + A^t a, C).

    Compare the Mehler eigenspace, a static landmark eigenspace and the time-t
    landmark eigenspace. The static envelope is R sqrt(lambda_{m+1}(Sigma));
    landmark bounds include the omitted trace tau_t and NUMERICAL_MARGIN.
    All inner products with the mean-embedding difference are analytic.
    """
    from mcmc.paperexp import gaussian_linear as gl

    kernel = gl.AnisoRBF(spec.scales, spec.ell)
    chain = gl.LinearGaussianChain(A=model.A, LQ=model.LQ, mu=spec.mu, C=spec.C)
    basis = gl.landmark_basis(kernel, landmarks, rtol=rtol)
    mmax = max(m_list)
    mu_pi_Z = gl.mean_embedding(kernel, spec.mu, spec.C, landmarks)
    op0 = _landmark_operator(gl, kernel, basis, chain, 0, NUMERICAL_MARGIN)
    coef_static = basis.W @ op0.top_basis(mmax)  # (M, mmax) landmark coefficients of V_m
    witness = {
        "landmark_rank": int(basis.rank),
        "landmark_orthonormality_error": float(basis.orthonormality_error),
        "tau_0": float(op0.tau),
        "lam_landmark_vs_exact_rel": (
            np.asarray([op0.lam(j) for j in range(1, mmax + 2)]) / spec.eig[: mmax + 1] - 1
        ).tolist(),
    }
    ops = {}
    out = []
    for st in model.settings:
        R = float(st["r_init"])
        a = np.asarray(st["shift"], dtype=np.float64)
        for t in model.checkpoints:
            t = int(t)
            if t not in ops:
                ops[t] = _landmark_operator(gl, kernel, basis, chain, t, NUMERICAL_MARGIN)
            opt = ops[t]
            s = np.linalg.matrix_power(model.A, t) @ a
            mmd2 = shifted_mmd2(spec, s)
            dmu = gl.mean_embedding(kernel, spec.mu + s, spec.C, landmarks) - mu_pi_Z
            r_exact = shifted_readings(spec, s, mmax)
            r_land = coef_static.T @ dmu
            r_dyn = (basis.W @ opt.top_basis(mmax)).T @ dmu
            row = {
                "setting": st["name"],
                "R": R,
                "t": t,
                "mmd": math.sqrt(max(mmd2, 0.0)),
                "mmd2": mmd2,
                "tau_t": float(opt.tau),
                "numerical_margin": NUMERICAL_MARGIN,
                "m": {},
            }
            for m in m_list:
                Vb = op0.top_basis(m)
                # V_m expressed in the time-t operator's basis: both use the same landmark basis
                comp = opt.compressed_top(Vb)
                p_ex = mmd2 - float(r_exact[:m] @ r_exact[:m])
                p_la = mmd2 - float(r_land[:m] @ r_land[:m])
                p_dy = mmd2 - float(r_dyn[:m] @ r_dyn[:m])
                row["m"][str(m)] = {
                    "proj2_exact": p_ex,
                    "proj2_landmark": p_la,
                    "proj2_dynamic": p_dy,
                    "env_static": R * math.sqrt(spec.eig[m]),
                    "bound_static_t_upper": R
                    * math.sqrt(max(comp, 0.0) + max(opt.tau, 0.0) + NUMERICAL_MARGIN),
                    "d_dyn_landmark": R * math.sqrt(max(opt.lam(m + 1), 0.0)),
                    "d_dyn_upper": R
                    * math.sqrt(max(opt.lam(m + 1), 0.0) + max(opt.tau, 0.0) + NUMERICAL_MARGIN),
                    "monitored2_exact": float(r_exact[:m] @ r_exact[:m]),
                }
            out.append(row)
    return {"witness": witness, "rows": out}
