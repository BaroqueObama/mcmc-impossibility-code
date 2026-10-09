"""Branched estimates of the dynamic covariance operator Sigma_t.

From independent X, X' ~ pi, run two independent branches from each start.
Writing D1 = phi(Y) - phi(Y') and D2 = phi(Z) - phi(Z'), the unbiased block
is (D1 tensor D2 + D2 tensor D1)/4. The kernel implementation diagonalizes
the average in the span of the endpoints.

For N blocks and k(x,x) <= kappa^2, the operator radius is
    e = (2 kappa^2/sqrt(N)) (1 + sqrt(2 log(1/(beta w_t)))).
The radii hold simultaneously when the checkpoint weights sum to at most one.
Callers add delta_trunc to account for Gram truncation or the enclosed
solver's numerical error. The finite-feature version controls only the
feature discrepancy unless an omitted-feature bound is supplied.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

F64 = torch.float64


@dataclass
class Span:
    """v_j = sum_a coefs[a, j] phi(points[a]), j < m."""

    points: torch.Tensor  # (n, d) float64
    coefs: torch.Tensor  # (n, m) float64
    label: str = ""

    def __post_init__(self):
        assert self.points.dtype == F64 and self.coefs.dtype == F64, "Span tensors must be float64"
        assert self.points.ndim == 2 and self.coefs.ndim == 2, "Span tensors must be 2-D"
        assert self.points.shape[0] == self.coefs.shape[0], (
            f"points {tuple(self.points.shape)} vs coefs {tuple(self.coefs.shape)}"
        )
        assert self.points.device == self.coefs.device, "Span tensors on different devices"

    @property
    def dim(self) -> int:
        return self.coefs.shape[1]

    def gram(self, kernel) -> torch.Tensor:
        K = kernel(self.points, self.points)
        G = self.coefs.T @ K @ self.coefs
        return 0.5 * (G + G.T)

    def orthonormality_error(self, kernel) -> float:
        G = self.gram(kernel)
        return float((G - torch.eye(self.dim, dtype=F64, device=G.device)).abs().max())

    def take(self, m: int) -> "Span":
        if m > self.dim:
            raise ValueError(f"span has dim {self.dim} < {m}")
        return Span(self.points, self.coefs[:, :m], self.label)


def orthonormalize(kernel, span: Span, rtol: float = 1e-10) -> Span:
    """Loewdin-orthonormalize a span (same subspace); raises if the vectors are nearly dependent."""
    G = span.gram(kernel)
    w, V = torch.linalg.eigh(G)
    if float(w.min()) <= rtol * float(w.max()):
        raise FloatingPointError(
            f"span vectors nearly dependent: Gram eigenvalues {w.min():.3e}..{w.max():.3e}"
        )
    Ginv_half = V @ torch.diag(w.rsqrt()) @ V.T
    return Span(span.points, span.coefs @ Ginv_half, span.label)


def require_orthonormal(kernel, span: Span, tol: float = 1e-8) -> float:
    err = span.orthonormality_error(kernel)
    if err > tol:
        raise FloatingPointError(
            f"span {span.label!r} not orthonormal (max Gram error {err:.2e} > {tol})"
        )
    return err


def operator_radius(kappa2: float, N: int, beta: float, weight: float) -> float:
    """e_{t,N} = (2 kappa^2/sqrt N)(1 + sqrt(2 log(1/(beta w_t))))."""
    if not (0 < beta < 1 and 0 < weight <= 1 and N >= 1 and kappa2 > 0):
        raise ValueError(f"bad radius inputs kappa2={kappa2} N={N} beta={beta} w={weight}")
    return 2.0 * kappa2 / math.sqrt(N) * (1.0 + math.sqrt(2.0 * math.log(1.0 / (beta * weight))))


def n_blocks_for_tolerance(
    eps: float, r_init: float, kappa2: float, beta: float, weight: float
) -> float:
    """Smallest N with r_init * sqrt(e_{t,N}) <= eps (the operator term alone; N^{-1/4} limitation)."""
    c = 2.0 * kappa2 * (1.0 + math.sqrt(2.0 * math.log(1.0 / (beta * weight))))
    return (c * r_init**2 / eps**2) ** 2


@dataclass
class KernelOperatorEstimate:
    endpoints: torch.Tensor  # (4N, d): Y, Y', Z, Z'
    N: int
    eigvals: torch.Tensor  # (r,) signed, descending (eigenvalues of the rank-r part O_r)
    coefs: torch.Tensor  # (4N, r) eigenvector coefficients over endpoints (orthonormal in H)
    gram_eigs: torch.Tensor  # (2N,) eigenvalues of G, descending
    rank: int
    rtol: float
    delta_trunc: float
    eigvec_orthonormality_error: float  # max |U^T U - I| over the leading block checked

    @property
    def n_pos(self) -> int:
        return int((self.eigvals > 0).sum())

    def lambda_plus(self, j: int) -> float:
        """j-th eigenvalue (1-indexed) of the positive part O_+ (0 beyond the positive spectrum)."""
        return float(self.eigvals[j - 1]) if j <= self.n_pos else 0.0

    def top_span(self, m: int, label: str = "") -> Span:
        if m > self.n_pos:
            raise ValueError(f"only {self.n_pos} positive eigenvalues, asked for {m} directions")
        return Span(self.endpoints, self.coefs[:, :m].contiguous(), label)

    def compressed_plus_norms(
        self, kernel, span: Span, ms, head: int = 1024
    ) -> list[tuple[float, dict]]:
        """``compressed_plus_norm`` for the nested spans span.take(m), m in ms (one kernel block)."""
        p = self.n_pos
        if p == 0:
            return [(0.0, {"exact": True, "slack_bound": 0.0}) for _ in ms]
        Kx = kernel(self.endpoints, span.points)
        Xfull = self.coefs[:, :p].T @ (Kx @ span.coefs)
        return [self._compressed_from_X(Xfull[:, :m], head) for m in ms]

    def _compressed_from_X(self, X: torch.Tensor, head: int) -> tuple[float, dict]:
        p = self.n_pos
        theta = self.eigvals[:p]
        Y = theta.sqrt()[:, None] * X
        h = min(p, head)
        Mhh = torch.diag(theta[:h]) - Y[:h] @ Y[:h].T
        a = float(torch.linalg.eigvalsh(0.5 * (Mhh + Mhh.T))[-1])
        if h == p:
            return a, {"exact": True, "slack_bound": 0.0}
        c = float(theta[h])
        b = float(torch.linalg.matrix_norm(Y[:h], ord=2) * torch.linalg.matrix_norm(Y[h:], ord=2))
        val = 0.5 * (a + c) + math.sqrt(0.25 * (a - c) ** 2 + b * b)
        # slack vs the exact compressed norm, which is >= a >= 0 (a is the exact compression of the head
        # block, a principal block): val - max(a, 0) is a valid upper bound on the slack. (For m < head,
        # interlacing gives a >= theta_{m+1} >= theta_{h+1} = c, so max(a, 0) = max(a, c).)
        return val, {"exact": False, "slack_bound": val - max(a, 0.0), "a": a, "b": b, "c": c}

    def compressed_plus_norm(self, kernel, span: Span, head: int = 1024) -> tuple[float, dict]:
        """Rigorous upper bound on ||P_{V-perp} O_+ P_{V-perp}||_op for an orthonormal span V.

        With O_+ = U Theta U^*, the compression has the nonzero spectrum of Theta - Y Y^T,
        Y = Theta^{1/2} U^* V. The leading ``head`` block is diagonalized exactly; the rest is
        bounded by the 2x2 block inequality lambda_max <= max eig [[a, b], [b, c]] with
        a = lambda_max(M_hh), c = theta_{h+1} >= lambda_max(M_tt), b = ||Y_h|| ||Y_t|| >= ||M_ht||.
        Exact (no slack) when the positive rank <= head.
        """
        return self.compressed_plus_norms(kernel, span, [span.dim], head)[0]


def _difference_gram(kernel, Y, Z, Yp, Zp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Endpoints E = [Y; Y'; Z; Z'], their Gram K (4N x 4N) and the difference Gram G = Psi^* Psi."""
    N, d = Y.shape
    for nm, T in (("Z", Z), ("Yp", Yp), ("Zp", Zp)):
        assert T.shape == (N, d), f"{nm} shape {tuple(T.shape)}, expected ({N},{d})"
        assert T.dtype == F64 and T.device == Y.device, f"{nm} must be float64 on {Y.device}"
    E = torch.cat([Y, Yp, Z, Zp], 0)
    K = kernel(E, E)
    assert K.shape == (4 * N, 4 * N)
    if not torch.isfinite(K).all():
        raise FloatingPointError("non-finite endpoint Gram")
    B = [[K[i * N : (i + 1) * N, j * N : (j + 1) * N] for j in range(4)] for i in range(4)]
    # index 0: Y, 1: Y', 2: Z, 3: Z'
    G11 = B[0][0] - B[0][1] - B[1][0] + B[1][1]
    G12 = B[0][2] - B[0][3] - B[1][2] + B[1][3]
    G22 = B[2][2] - B[2][3] - B[3][2] + B[3][3]
    del B
    G = torch.cat([torch.cat([G11, G12], 1), torch.cat([G12.T, G22], 1)], 0)
    G = 0.5 * (G + G.T)
    return E, K, G


ENCLOSED_METHODS = {"fp32-enclosed": torch.float32, "chol-fp64": F64}


def estimate_operator(
    kernel,
    Y: torch.Tensor,
    Z: torch.Tensor,
    Yp: torch.Tensor,
    Zp: torch.Tensor,
    rtol: float = 1e-11,
    check_top: int = 64,
    method: str = "fp64",
    n_refined: int = 128,
) -> KernelOperatorEstimate:
    """Kernel eigendecomposition of the N-block estimate; endpoints are (N, d) float64.

    For G = Psi^* Psi = V diag(g) V^T, retain g > rtol * g_max and set
    R = V_r sqrt(g_r). Diagonalize R^T C R, C = [[0,I],[I,0]]/(4N), and map
    vectors back through Psi V_r g_r^{-1/2}. The omitted Gram eigenvalues give
        delta_trunc = (2 sqrt(g_max g_cut) + g_cut)/(4N).
    At t=0, G has rank at most N because Y=Z and Y'=Z'.

    Methods "fp32-enclosed" and "chol-fp64" use estimate_operator_enclosed,
    which refines n_refined vectors and does not truncate the Gram matrix.
    """
    if method != "fp64":
        if method not in ENCLOSED_METHODS:
            raise ValueError(f"unknown method {method!r}; have fp64, {sorted(ENCLOSED_METHODS)}")
        return estimate_operator_enclosed(
            kernel, Y, Z, Yp, Zp, n_refined, check_top, ENCLOSED_METHODS[method], method
        )
    N = Y.shape[0]
    E, K, G = _difference_gram(kernel, Y, Z, Yp, Zp)
    g, V = torch.linalg.eigh(G)
    del G
    g, V = g.flip(0), V.flip(1)
    gmax = float(g[0])
    if not gmax > 0:
        raise FloatingPointError(f"difference Gram has no positive eigenvalue (g_max={gmax})")
    keep = g > rtol * gmax
    r = int(keep.sum())
    g_cut = float(g[~keep].abs().max()) if r < g.shape[0] else 0.0
    if float(g.min()) < -1e-8 * gmax:
        raise FloatingPointError(
            f"difference Gram markedly indefinite: min eig {float(g.min()):.3e}"
        )
    Vr, gr = V[:, :r], g[:r]
    del V
    R = Vr * gr.sqrt()[None, :]  # G ~ R R^T
    CR = torch.cat([R[N:], R[:N]], 0) / (4.0 * N)  # C R
    Wm = R.T @ CR
    Wm = 0.5 * (Wm + Wm.T)
    theta, w = torch.linalg.eigh(Wm)
    theta, w = theta.flip(0), w.flip(1)
    c_diff = (Vr * gr.rsqrt()[None, :]) @ w  # (2N, r)
    coefs = torch.cat([c_diff[:N], -c_diff[:N], c_diff[N:], -c_diff[N:]], 0)
    delta_trunc = (2.0 * math.sqrt(gmax * g_cut) + g_cut) / (4.0 * N)
    # orthonormality of the leading eigenvectors in H (u^T K_E u = I)
    k = min(check_top, r)
    Gu = coefs[:, :k].T @ K @ coefs[:, :k]
    del K
    orth_err = float((Gu - torch.eye(k, dtype=F64, device=Gu.device)).abs().max())
    if orth_err > 1e-6:
        raise FloatingPointError(
            f"leading operator eigenvectors not orthonormal (err {orth_err:.2e})"
        )
    return KernelOperatorEstimate(
        endpoints=E,
        N=N,
        eigvals=theta,
        coefs=coefs,
        gram_eigs=g,
        rank=r,
        rtol=rtol,
        delta_trunc=delta_trunc,
        eigvec_orthonormality_error=orth_err,
    )


# Shifted Cholesky and numerical enclosures
U64 = 2.0**-53  # unit roundoff of float64


def _gamma(k: int) -> float:
    """gamma_k = k u/(1 - k u): |fl(A B) - A B| <= gamma_k |A| |B| for inner dimension k (Higham 3.5)."""
    ku = k * U64
    assert ku < 1e-3, f"gamma_{k} outside the regime of the bound"
    return ku / (1.0 - ku)


TINY = 2.0**-1074  # smallest positive subnormal
MIN_NORMAL = 2.0**-1022
SAFE_MIN = 2.0**-969  # MIN_NORMAL * 2^53: certificate scalars must stay above this


def _ldexp(x, e: int):
    """x * 2^e in two exact power-of-two steps (each factor representable for |e| <= 2046)."""
    e1 = e // 2
    return (x * 2.0**e1) * 2.0 ** (e - e1)


def _fro(T: torch.Tensor) -> float:
    """Rigorous upper bound on the exact Frobenius norm, safe against under/overflow.

    For O(1)-scale data, v = fl(||T||_F); the computed sum of squares errs by
    <= gamma_n sum x^2 + n 2^-1075 (gradual underflow), so v (1 + 1e-6) is an upper bound whenever
    n u <= 5e-7 and v >= sqrt(1e6 n 2^-1074). Otherwise T is rescaled by the exact power of two of
    max |T| (entries <= 1, norm >= 1/2), bounded there, and scaled back (raising on overflow; a
    subnormal result is rounded up by 2^-1074). Non-finite entries raise.
    """
    n = T.numel()
    assert n * U64 <= 5e-7, f"too many entries ({n}) for the norm bound"
    v = float(torch.linalg.vector_norm(T))
    if math.isfinite(v) and v >= math.sqrt(1e6 * (n + 1) * TINY):
        return v * (1.0 + 1e-6)
    m = float(T.abs().max()) if n else 0.0
    if not math.isfinite(m):
        raise FloatingPointError("non-finite entries in a certificate norm")
    if m == 0.0:
        return 0.0
    e = math.frexp(m)[1]
    vs = float(torch.linalg.vector_norm(_ldexp(T, -e)))
    if not (math.isfinite(vs) and vs >= 0.5):
        raise FloatingPointError(f"scaled certificate norm failed ({vs})")
    r = _ldexp(vs * (1.0 + 1e-6), e)
    if not math.isfinite(r):
        raise FloatingPointError("certificate norm overflows float64")
    return r + TINY if r < MIN_NORMAL else r


def _uf(k: int, numel: int) -> float:
    """Spectral-norm bound of the gradual-underflow error of a matrix product with inner dimension k:
    each entry errs by <= k 2^-1075 beyond the gamma_k |A||B| term (Higham, model with underflow)."""
    return (k + 1) * math.sqrt(numel + 1) * TINY


class EigensolveNotConverged(FloatingPointError):
    """The enclosed solver's eigendecomposition did not converge."""


def _cholqr(Q: torch.Tensor) -> torch.Tensor:
    R = torch.linalg.cholesky(Q.T @ Q, upper=True)
    return torch.linalg.solve_triangular(R, Q, upper=True, left=False)


@dataclass
class EnclosedKernelOperatorEstimate(KernelOperatorEstimate):
    """Operator estimate with bounds from the shifted-Cholesky construction.

    eigvals[:refined] holds fp64 Ritz values and coefs holds their vectors.
    The remaining eigenvalues are diagnostic only; n_pos counts positive Ritz
    values. Callers add delta_trunc to lambda_plus or compressed_plus_norms
    to bound the operator on the corresponding orthogonal complement.
    """

    refined: int = 0
    n_orthonormal: int = 0  # leading vectors with max |U^T U - I| <= 1e-6 (usable as a span)
    omega: float = 0.0  # >= ||X'^T X' - I||_2
    rho: float = 0.0  # >= ||M X' - X' A||_2
    tail_bound: float = 0.0  # certified c >= lambda_max(M - X' A X'^T)
    eps_form: float = 0.0  # >= ||fl(M) - L^T C L||_2
    sigma: float = 0.0  # Cholesky shift
    method: str = ""
    scale_exp: int = 0  # the pipeline ran on G 2^-scale_exp (0 for O(1)-scale data)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_pos(self) -> int:
        return int((self.eigvals[: self.refined] > 0).sum())

    def top_span(self, m: int, label: str = "") -> Span:
        if m > self.n_orthonormal:
            raise ValueError(
                f"only {self.n_orthonormal} leading vectors are H-orthonormal to 1e-6, asked for {m}"
            )
        return super().top_span(m, label)

    def _amax(self) -> float:
        return float(self.eigvals[: self.refined].abs().max())

    def _scaled(self, x):
        """x 2^-scale_exp (exact; identity for O(1)-scale estimates)."""
        return _ldexp(x, -self.scale_exp) if self.scale_exp else x

    def _unscaled(self, x: float) -> float:
        if not self.scale_exp:
            return x
        r = _ldexp(x, self.scale_exp)
        if not math.isfinite(r):
            raise FloatingPointError("bound overflows float64 after scale-back")
        return r + TINY if r < MIN_NORMAL else r

    def lambda_plus(self, j: int) -> float:
        """max(alpha_j/(1 - omega), c, 0), alpha_j = a_j^+ + a_1^+ omega^2 + omega a_max + sqrt(1+omega) rho
        (evaluated in the scaled units of the pipeline, then scaled back exactly)."""
        c = max(self._scaled(self.tail_bound), 0.0)
        if j > self.refined:
            return self._unscaled(c)
        w = self.omega
        a = self._scaled(self.eigvals[: self.refined])
        alpha = (
            max(float(a[j - 1]), 0.0)
            + max(float(a[0]), 0.0) * w * w
            + w * float(a.abs().max())
            + math.sqrt(1.0 + w) * self._scaled(self.rho)
        )
        return self._unscaled(max(alpha / (1.0 - w) * (1.0 + 16.0 * U64), c))

    def compressed_plus_norms(
        self, kernel, span: Span, ms, head: int = 1024
    ) -> list[tuple[float, dict]]:
        """Certified bound on sup{<v, O_hat v> : v unit, v perp V_m} - delta_trunc for the supplied
        span (``head`` unused: exact over the q refined vectors). V_m need not be orthonormal.

        With D' = (A - c^+/(1+omega))_+ + E1 I (inflated by 8u), W = U^* V_m (exact), S = V_m^* V_m:
        sup_{v perp V_m} gamma^T D' gamma = lambda_max(D'^{1/2}(U^*U - W S^{-1} W^T) D'^{1/2}) and
        U^*U <= (1+omega) I, S^{-1} >= I/(1+omega_V) (omega_V >= ||S - I||_2, certified from the
        computed S and its rounding bound) give <= lambda_max(D' - Y Y^T/(1+omega_V)) + omega ||D'||,
        Y = D'^{1/2} W. The computed W_c has a certified entrywise rounding bound (certified_span_overlap),
        so Y Y^T >= Y_c Y_c^T - 2 ||D'|| ||W_c|| eps_W I. The small matrix is formed with a rounding
        margin and its lambda_max certified by a completed Cholesky of l I - M_q (Higham Thm 10.3).
        Raises if S is not certifiably within 1/2 of I (e.g. cancelling span coefficients).
        """
        q, w = self.refined, self.omega
        c = max(self._scaled(self.tail_bound), 0.0)
        a = self._scaled(self.eigvals[:q])
        amax = float(a.abs().max())
        E1 = (
            amax * ((1.0 - w) ** -2 - 1.0)
            + (w * amax + math.sqrt(1.0 + w) * self._scaled(self.rho)) / (1.0 - w) ** 2
        )
        Dp = ((a - c / (1.0 + w)).clamp_min(0.0) + E1) * (1.0 + 8.0 * U64)
        dmax = float(Dp.max())
        ov = certified_span_overlap(kernel, self.endpoints, self.coefs, span)
        out = []
        for m in ms:
            omega_V = span_gram_defect(ov["S"][:m, :m], ov["ES"][:m, :m])
            Wm = ov["W"][:, :m]
            eps_W, wn = _fro(ov["EW"][:, :m]), _fro(Wm)
            Yp = Dp.sqrt()[:, None] * Wm
            yf = _fro(Yp)
            Mq = torch.diag(Dp) - (Yp @ Yp.T) / (1.0 + omega_V)
            Mq = 0.5 * (Mq + Mq.T)
            form = _gamma(m + 12) * yf**2 + 4.0 * U64 * dmax + _uf(m + 12, q * q)
            lam = certified_lambda_max(Mq)
            pert = 2.0 * dmax * wn * eps_W
            val = (c + lam + form + pert + w * dmax) * (1.0 + 16.0 * U64)
            out.append(
                (
                    self._unscaled(val),
                    {
                        "exact": True,
                        "slack_bound": self._unscaled(val - c - lam),
                        "lam": self._unscaled(lam) if lam > 0 else lam,
                        "c": self._unscaled(c) if c > 0 else c,
                        "omega_V": omega_V,
                        "eps_W": eps_W,
                        "overlap_margin": self._unscaled(pert) if pert > 0 else pert,
                    },
                )
            )
        return out


def certified_span_overlap(kernel, endpoints, coefs, span: Span) -> dict:
    """W_c = fl(coefs^T K(E, P) s), S_c = fl(s^T K(P, P) s) with entrywise rounding bounds EW, ES.

    |fl(A B) - A B| <= gamma_k |A||B| (any summation order): for W_c = fl(coefs^T fl(K s)),
    |W_c - coefs^T K s| <= gamma_{|P|} |coefs|^T |K| |s| + gamma_{|E|} |coefs|^T |fl(K s)|, and likewise
    for S_c; the nonnegative bound products are inflated by (1 + 2 gamma) to cover their own rounding.
    Kernel values are taken as data (the default path's convention).
    """
    s = span.coefs
    nE, nP = endpoints.shape[0], span.points.shape[0]
    Kx = kernel(endpoints, span.points)
    Kp = kernel(span.points, span.points)
    if not (torch.isfinite(Kx).all() and torch.isfinite(Kp).all()):
        raise FloatingPointError("non-finite span kernel block")
    As, Ac = s.abs(), coefs.abs()
    infl = 1.0 + 2.0 * _gamma(max(nE, nP) + 2)
    T = Kx @ s
    W = coefs.T @ T
    EW = (_gamma(nP) * (Ac.T @ (Kx.abs() @ As)) + _gamma(nE) * (Ac.T @ T.abs())) * infl
    EW = EW + (nE + nP + 4) * TINY  # gradual-underflow floor of both products (no-op at O(1) scale)
    T2 = Kp @ s
    S = s.T @ T2
    ES = (_gamma(nP) * (As.T @ (Kp.abs() @ As)) + _gamma(nP) * (As.T @ T2.abs())) * infl
    ES = ES + (2 * nP + 4) * TINY
    return {"W": W, "EW": EW, "S": S, "ES": ES}


def span_gram_defect(S_c: torch.Tensor, ES: torch.Tensor) -> float:
    """Certified omega_V >= ||S - I||_2 from a computed Gram S_c with entrywise rounding bound ES;
    raises unless omega_V < 1/2 (the span cannot be certified near-orthonormal)."""
    m = S_c.shape[0]
    eye = torch.eye(m, dtype=S_c.dtype, device=S_c.device)
    omega_V = (_fro(S_c - eye) + _fro(ES)) * (1.0 + 4.0 * U64) + 4.0 * U64 * (1.0 + m)
    if not omega_V < 0.5:
        raise FloatingPointError(
            f"span Gram defect cannot be certified < 1/2 (bound {omega_V:.3e}): cancelling or "
            "non-orthonormal span coefficients; re-orthonormalize the span"
        )
    return omega_V


def certified_lambda_max(Mq: torch.Tensor, ladder=(1e-14, 1e-12, 1e-10, 1e-8, 1e-6)) -> float:
    """Upper bound on lambda_max of the (exactly known) symmetric fp64 matrix Mq: l = eigvalsh max,
    then a completed Cholesky of (l + tau s) I - Mq gives lambda_max <= l + tau s + gamma_{n+1}
    ||L_B||_F^2 + u (|l + tau s| + max |Mq_ii|) (diagonal-shift rounding)."""
    n = Mq.shape[0]
    lam = float(torch.linalg.eigvalsh(Mq)[-1])
    dg = float(Mq.diagonal().abs().max())
    scale = max(abs(lam), dg, float(torch.linalg.vector_norm(Mq)), 1e-300)
    for tau in ladder:
        c_try = lam + tau * scale
        B = -Mq
        B.diagonal().add_(c_try)
        LB, info = torch.linalg.cholesky_ex(B)
        if int(info) == 0 and torch.isfinite(LB).all():
            return (
                c_try
                + _gamma(n + 1) * _fro(LB) ** 2
                + U64 * (abs(c_try) + dg) * 1.01
                + _uf(n + 2, n * n)
            )
    raise FloatingPointError(f"lambda_max certificate failed up to {c_try:.3e}")


def default_path_margins(
    span_gram_err: float,
    span_dim: int,
    theta1: float,
    eig_gram_err: float = 0.0,
    eig_dim: int = 0,
    theta_absmax: float | None = None,
    overlap_eps: float | None = None,
    certified_inputs: bool = False,
) -> dict:
    """Gram-defect margins, conditional on certified input errors.

    These do not certify the fp64 eigensolver or its subsequent arithmetic.
    The formulas require omega_V >= ||V^*V-I||, omega_U >= ||U^*U-I|| and
    overlap_eps >= ||fl(U^*V)-U^*V|| over all participating vectors. Recorded
    Gram errors alone are insufficient: cancellation can hide overlap error.
    The status is CONDITIONAL only when these bounds are supplied, otherwise
    UNCERTIFIED. Use the enclosed solver for a certified bound.

    With theta1 the largest positive eigenvalue, the compression margin is
        theta1 (omega_U + omega_V (1+omega_U))
        + 2 theta1 sqrt((1+omega_U)(1+omega_V)) eps + theta1 eps^2.
    The lambda margin uses the block bound with c=0 and
    rho = sqrt(1+omega_U) theta_absmax omega_U.
    """
    tmax = theta1 if theta_absmax is None else theta_absmax
    t1 = max(theta1, 0.0)
    wV = span_dim * span_gram_err
    wU = eig_dim * eig_gram_err
    if not (wV < 0.5 and wU < 0.5):
        raise ValueError(f"Gram defects too large for the margin (omega_V={wV}, omega_U={wU})")
    e = 0.0 if overlap_eps is None else overlap_eps
    comp = (
        t1 * (wU + wV * (1.0 + wU)) + 2.0 * t1 * math.sqrt((1.0 + wU) * (1.0 + wV)) * e + t1 * e * e
    )
    rho = math.sqrt(1.0 + wU) * tmax * wU
    alpha_extra = t1 * wU + t1 * wU * wU + wU * tmax + math.sqrt(1.0 + wU) * rho
    lam = (t1 / (1.0 - wU) - t1) + alpha_extra / (1.0 - wU) + rho / math.sqrt(1.0 - wU)
    status = "CONDITIONAL" if (certified_inputs and overlap_eps is not None) else "UNCERTIFIED"
    return {
        "status": status,
        "omega_V": wV,
        "omega_U": wU,
        "compressed": comp * (1.0 + 1e-12),
        "lambda": lam * (1.0 + 1e-12),
    }


def estimate_operator_enclosed(
    kernel,
    Y: torch.Tensor,
    Z: torch.Tensor,
    Yp: torch.Tensor,
    Zp: torch.Tensor,
    n_refined: int = 128,
    check_top: int = 64,
    eig_dtype: torch.dtype = torch.float32,
    method: str = "fp32-enclosed",
    cert_ladder: tuple = (1e-6, 1e-5, 1e-4, 1e-3),
) -> EnclosedKernelOperatorEstimate:
    """O_hat with one low-precision eigensolve and an fp64 a-posteriori enclosure (no truncation).

    Notation: n = 2N, Psi = [D1_1..D1_N, D2_1..D2_N], G = Psi^* Psi (as computed: the data, same
    convention as the default path), C = [[0, I], [I, 0]]/(4N), O_hat = Psi C Psi^*, u = 2^-53.

    1. Shifted Cholesky (fp64). L L^T = G + sigma I + Delta with ||Delta|| <= u (max G_ii + sigma)
       + gamma_{n+1} ||L||_F^2 (Higham Thm 10.3; conventional blocked BLAS-3 assumed); sigma is
       raised until it dominates this bound, so Gamma := L L^T - G is PSD. Augmentation: with
       Psi_aug a = (Psi a, Gamma^{1/2} a) in H (+) R^n, Psi_aug^* Psi_aug = L L^T, so
       J = Psi_aug L^{-T} is an isometry, O_aug = Psi_aug C Psi_aug^* = J M J^* with
       M = L^T C L = (L1^T L2 + L2^T L1)/(4N), and <(v,0), O_aug (v,0)> = <v, O_hat v> for
       v in H (Psi_aug^* (v, 0) = Psi^* v). Hence <v, O_hat v> = x^T M x with x = J^*(v, 0),
       ||x|| <= ||v||, and x^T x'_k = <u_k, v>_H for x'_k := L^T Cd_k, u_k := Psi Cd_k (any Cd).
       fl(M) is within eps_form = (2 gamma_N ||L1||_F ||L2||_F + 2u ||L1^T L2 + L2^T L1||_F)/(4N).
    2. eigh(fl(M)) in eig_dtype, then fp64 refinement of the leading q = n_refined vectors
       by CholQR2 and Rayleigh-Ritz: Ritz pairs (a_k, X), Cd = L^{-T} X. A failed eigensolve
       raises EigensolveNotConverged.
    3. Enclosure, all fp64 with rounding margins (X'_c = fl(L^T Cd), ||X'_c - X'|| <= eta =
       gamma_n ||L||_F ||Cd||_F): omega >= ||X'^T X' - I||, rho >= ||fl(M) X' - X' A|| (A = diag a),
       and a certified c >= lambda_max(M_d), M_d = fl(M) - X' A X'^T, from a successful Cholesky of
       c I - fl(M_d) (B + Delta_B = L_B L_B^T, ||Delta_B|| <= gamma_{n+1} ||L_B||_F^2, plus the
       formation margins of fl(M_d)).
    The block bound needs no eigengap or exact orthonormality; these affect only tightness.
    P = projector onto span X', x = y + z, y = P x = X' beta, z = (I - P) x,
    s = ||y||, t = ||z||, s^2 + t^2 <= 1: since z^T M z = z^T M_d z <= c t^2, z^T M y = z^T R beta,
    X'^T M X' = (I + Omega) A + X'^T R and ||beta||^2 <= s^2/(1 - omega),
      (own)  v perp u_1..u_m <=> x perp x'_1..x'_m => beta_{<=m} = -(Omega beta)_{<=m}, so
             x^T M x <= [alpha s^2/(1-omega) + 2 rho s t/sqrt(1-omega) + c t^2]
                     <= max(alpha/(1-omega), c, 0) + rho/sqrt(1-omega),
             alpha = a_{m+1}^+ + a_1^+ omega^2 + omega a_max + sqrt(1+omega) rho   (c alone if m >= q);
      (span) for unit v perp an orthonormal V, gamma = U^* v = X'^T x, beta = (I+Omega)^{-1} gamma,
             c t^2 <= c^+ (1 - gamma^T (I+Omega)^{-1} gamma) gives
             x^T M x <= c^+ + gamma^T D' gamma + 2 rho sqrt(1+omega)/(1-omega),
             D' = (A - c^+/(1+omega))_+ + E1 I, E1 = a_max((1-omega)^-2 - 1)
             + (omega a_max + sqrt(1+omega) rho)/(1-omega)^2, and sup_{v perp V} gamma^T D' gamma
             <= lambda_max(D' - Y'Y'^T) + omega ||D'|| (U^* U = Cd^T G Cd <= X'^T X'), Y' = D'^{1/2} U^* V.
    Both add eps_form (fl(M) vs M); delta_trunc := eps_form + 2 rho sqrt(1+omega)/(1-omega) covers
    the rho terms of both, and lambda_plus / compressed_plus_norms return the rest. The tail c is
    charged only through max(., c) / the (1 - ||y||^2) factor, not additively.
    """
    t_start = time.perf_counter()
    N = Y.shape[0]
    n = 2 * N
    E, K, G = _difference_gram(kernel, Y, Z, Yp, Zp)
    dev = G.device
    diag = G.diagonal()
    trG, dmax = float(diag.sum()), float(diag.max())
    if not (trG > 0 and math.isfinite(trG)):
        raise FloatingPointError(f"difference Gram has zero or non-finite trace ({trG})")
    # At extreme scales, use G 4^-j so coefficients scale by the exact factor 2^-j.
    # Scale outputs back after the enclosure calculations.
    scale_exp = 0
    if not (2.0**-64 <= dmax <= 2.0**64):
        scale_exp = 2 * round(math.log2(dmax) / 2)
        Gs = _ldexp(G, -scale_exp)
        if not torch.equal(_ldexp(Gs, scale_exp), G):
            raise FloatingPointError(f"difference Gram cannot be rescaled exactly (2^{scale_exp})")
        G = Gs
        del Gs
        diag = G.diagonal()
        trG, dmax = float(diag.sum()), float(diag.max())
    # 1. shifted Cholesky with a backward-error certificate (Gamma = L L^T - G is PSD)
    g1 = _gamma(n + 1)
    sigma, tries = 4.0 * g1 * trG, 0
    while True:
        tries += 1
        if tries > 8:
            raise FloatingPointError(
                f"shifted Cholesky failed up to sigma={sigma:.3e} (tr G {trG:.3e})"
            )
        A = G.clone()
        A.diagonal().add_(sigma)
        L, info = torch.linalg.cholesky_ex(A)
        del A
        if int(info) != 0:
            sigma *= 100.0
            continue
        if not torch.isfinite(L).all():
            raise FloatingPointError("shifted Cholesky completed with non-finite entries")
        need = U64 * (dmax + sigma) + g1 * _fro(L) ** 2 + _uf(n + 2, n * n)
        if sigma >= need:
            break
        sigma = 2.0 * need
    del G
    L1, L2 = L[:N], L[N:]
    S = L1.T @ L2
    S = S + S.T
    M = S / (4.0 * N)
    eps_form = (
        2.0 * _gamma(N) * _fro(L1) * _fro(L2) + 2.0 * U64 * _fro(S) + 2.0 * _uf(N, n * n)
    ) / (4.0 * N)
    del S, L1, L2
    # 2. low-precision eigensolve, fp64 Rayleigh-Ritz on the leading q vectors
    try:
        w, Q = torch.linalg.eigh(M.to(eig_dtype))
    except torch.linalg.LinAlgError as exc:
        raise EigensolveNotConverged(
            f"{eig_dtype} eigh of M (n={n}) did not converge on {dev.type}: {exc}. "
            "Nearly singular RBF Gram matrices can cause this in float32. "
            'Use method="chol-fp64" for the same enclosure with an fp64 eigensolve, or "fp64".'
        ) from exc
    if not (torch.isfinite(w).all() and torch.isfinite(Q).all()):
        raise FloatingPointError(f"non-finite {eig_dtype} eigendecomposition of M")
    w, Q = w.flip(0).to(F64), Q.flip(1)
    q = min(n_refined, n)
    Q1 = _cholqr(_cholqr(Q[:, :q].to(F64)))
    del Q
    MQ = M @ Q1
    Hq = Q1.T @ MQ
    a, Zr = torch.linalg.eigh(0.5 * (Hq + Hq.T))
    a, Zr = a.flip(0), Zr.flip(1)
    X = Q1 @ Zr
    ritz_res = float(
        torch.linalg.vector_norm(MQ @ Zr - X * a[None, :])
    )  # informational (Frobenius)
    del MQ, Hq, Q1, Zr
    Cd = torch.linalg.solve_triangular(L.T, X, upper=True)
    del X
    # 3. enclosure
    eta = _gamma(n) * _fro(L) * _fro(Cd) + _uf(n, n * q)
    Xe = L.T @ Cd
    del L
    xf = _fro(Xe)
    eye_q = torch.eye(q, dtype=F64, device=dev)
    omega = (
        _fro(Xe.T @ Xe - eye_q) + _gamma(n + 1) * xf**2 + 2.0 * xf * eta + eta**2 + _uf(n, q * q)
    )
    if not omega < 0.5:
        raise FloatingPointError(f"refined vectors far from orthonormal (omega bound {omega:.2e})")
    amax = float(a.abs().max())
    XA = Xe * a[None, :]
    Rc = M @ Xe - XA
    mf = _fro(M)
    rho = (
        _fro(Rc) * (1.0 + 2.0 * U64)
        + _gamma(n) * mf * xf
        + U64 * amax * xf
        + (mf + amax) * eta
        + _uf(n + 1, n * q)
    )
    del Rc
    Md = M - XA @ Xe.T
    del XA
    Md = 0.5 * (Md + Md.T)
    m_md = (
        _gamma(q + 2) * amax * xf**2
        + 3.0 * U64 * _fro(Md)
        + amax * (2.0 * xf * eta + eta**2)
        + _uf(q + 2, n * n)
    )
    md_diag = float(Md.diagonal().abs().max())
    trace_M = float(M.diagonal().sum())  # = trace(O_hat) + trace(C Gamma) (diagnostic)
    m_scale = max(float(w.abs().max()), amax)
    base = max(float(w[q]), 0.0) if q < n else 0.0
    c_hat, c_try = None, None
    for rung, tau in enumerate(cert_ladder):
        c_try = base + tau * m_scale
        B = -Md
        B.diagonal().add_(c_try)
        LB, info = torch.linalg.cholesky_ex(B)
        del B
        if int(info) == 0:
            if not torch.isfinite(LB).all():
                raise FloatingPointError("certificate Cholesky completed with non-finite entries")
            c_hat = (
                c_try
                + U64 * (abs(c_try) + md_diag) * 1.01
                + _gamma(n + 1) * _fro(LB) ** 2
                + m_md
                + _uf(n + 2, n * n)
            )
            break
    if c_hat is None:
        raise FloatingPointError(
            f"tail certificate failed up to c={c_try:.3e} (fp eigenvalue w_q+1={base:.3e})"
        )
    del Md, LB, M
    if scale_exp:
        # exact power-of-two scale-back; certificate scalars must stay comfortably normal so that
        # the (1 + 16u) / (1 + 8u) inflations in lambda_plus / compressed_plus_norms also absorb the
        # rounding of subnormal Ritz values (<= 2^-1075 <= u rho).
        a, w = _ldexp(a, scale_exp), _ldexp(w, scale_exp)
        Cd = _ldexp(Cd, -scale_exp // 2)
        rho, eps_form, sigma = (_ldexp(x, scale_exp) for x in (rho, eps_form, sigma))
        c_hat, c_try, base, trace_M = (_ldexp(x, scale_exp) for x in (c_hat, c_try, base, trace_M))
        ritz_res = _ldexp(ritz_res, scale_exp)
        bad = [
            x
            for x in (rho, eps_form, abs(c_hat))
            if not (math.isfinite(x) and (x == 0.0 or x >= SAFE_MIN))
        ]
        if bad or not (rho >= SAFE_MIN and torch.isfinite(a).all() and torch.isfinite(Cd).all()):
            raise FloatingPointError(
                f"operator scale 2^{scale_exp} leaves the certifiable float64 range (rho={rho:.3e})"
            )
    coefs = torch.cat([Cd[:N], -Cd[:N], Cd[N:], -Cd[N:]], 0)
    # H-orthonormality of the leading vectors (span use): u_k = Psi Cd_k has H-norm deficit
    # Cd_k^T Gamma Cd_k ~ sigma ||Cd_k||^2, negligible unless a_k is tiny; top_span(m) is refused
    # beyond the longest prefix with max |U^T U - I| <= 1e-6 (the default path's tolerance).
    k = min(check_top, q)
    Gu = coefs[:, :k].T @ K @ coefs[:, :k]
    del K
    dev_ = torch.tril((Gu - torch.eye(k, dtype=F64, device=dev)).abs())
    prefix_err = torch.cummax(dev_.max(dim=1).values, 0).values
    n_orth = int((prefix_err <= 1e-6).sum())
    orth_err = float(prefix_err[max(n_orth, 1) - 1])
    return EnclosedKernelOperatorEstimate(
        endpoints=E,
        N=N,
        eigvals=torch.cat([a, w[q:]]),
        coefs=coefs,
        gram_eigs=torch.empty(0, dtype=F64, device=dev),
        rank=n,
        rtol=0.0,
        delta_trunc=eps_form + 2.0 * rho * math.sqrt(1.0 + omega) / (1.0 - omega),
        eigvec_orthonormality_error=orth_err,
        refined=q,
        n_orthonormal=n_orth,
        omega=omega,
        rho=rho,
        tail_bound=c_hat,
        eps_form=eps_form,
        sigma=sigma,
        method=method,
        scale_exp=scale_exp,
        diagnostics={
            "sigma_tries": tries,
            "cert_rung": rung,
            "cert_trial": c_try,
            "eig_w_q1": base,
            "trace": trace_M,
            "ritz_residual": ritz_res,
            "eta": eta,
            "m_md": m_md,
            "wall": time.perf_counter() - t_start,
        },
    )


# Finite-feature operator (does not bound the omitted RKHS component)
@dataclass
class FeatureOperatorEstimate:
    matrix: torch.Tensor  # (r, r) symmetric
    eigvals: torch.Tensor  # descending
    eigvecs: torch.Tensor  # (r, r)
    N: int
    label: str = "feature-space (Nystrom)"


def feature_operator(features, Y, Z, Yp, Zp) -> FeatureOperatorEstimate:
    """Same estimator in a finite feature map f: (n, d) -> (n, r): O = (D1^T D2 + D2^T D1)/(4N)."""
    N = Y.shape[0]
    D1 = features(Y) - features(Yp)
    D2 = features(Z) - features(Zp)
    assert D1.shape == D2.shape and D1.shape[0] == N
    Om = (D1.T @ D2 + D2.T @ D1) / (4.0 * N)
    Om = 0.5 * (Om + Om.T)
    w, V = torch.linalg.eigh(Om)
    return FeatureOperatorEstimate(matrix=Om, eigvals=w.flip(0), eigvecs=V.flip(1), N=N)
