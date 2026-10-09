"""Spectral enclosures checked against exact finite-feature operators."""

import math

import pytest
import torch

from mcmc.paperexp import operators as op

F64 = torch.float64
METHODS = ["fp32-enclosed", "chol-fp64"]


class RFFKernel:
    """k(x, y) = f(x).f(y), f(x) = sqrt(2/D) cos(W x + b): explicit D-dim features, kappa2 <= 2."""

    def __init__(self, d: int, D: int, seed: int, scale: float = 1.0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(D, d, generator=g, dtype=F64) / scale
        self.b = 2 * math.pi * torch.rand(D, generator=g, dtype=F64)
        self.D = D

    def features(self, X):
        return math.sqrt(2.0 / self.D) * torch.cos(X @ self.W.T + self.b)

    def __call__(self, X, Y):
        return self.features(X) @ self.features(Y).T


class LinearKernel:
    """k(x, y) = x.y (features = identity): O_hat is prescribed directly by the endpoints."""

    @staticmethod
    def features(X):
        return X

    def __call__(self, X, Y):
        return X @ Y.T


def _exact_operator(f, Y, Z, Yp, Zp):
    D1, D2 = f(Y) - f(Yp), f(Z) - f(Zp)
    Om = (D1.T @ D2 + D2.T @ D1) / (4.0 * Y.shape[0])
    return 0.5 * (Om + Om.T)


def _lam_max_perp(Om, V):
    """lambda_max of the compression of O onto span(V)^perp (V: feature-space columns)."""
    Qv, _ = torch.linalg.qr(V)
    P = torch.eye(Om.shape[0], dtype=F64) - Qv @ Qv.T
    return float(torch.linalg.eigvalsh(P @ Om @ P)[-1])


def _check_rigour(est, kernel, Om, extra_spans=()):
    """Check all reported bounds against the explicit operator, allowing float64 roundoff."""
    tol = 1e-12 * float(torch.linalg.matrix_norm(Om, ord=2)) + 1e-14
    lam = torch.linalg.eigvalsh(Om).flip(0)
    d = est.delta_trunc
    # Courant-Fischer: lambda_j(O_hat) <= sup_{v perp u_1..u_{j-1}} <v, Om v> <= lambda_plus(j) + d
    for j in range(1, min(est.refined + 4, lam.numel()) + 1):
        assert float(lam[j - 1]) <= est.lambda_plus(j) + d + tol, (j, float(lam[j - 1]))
    U = kernel.features(est.endpoints).T @ est.coefs  # feature vectors of u_k
    for m in sorted({1, 2, 5, est.n_pos // 2, est.n_pos} - {0}):
        exact = _lam_max_perp(Om, U[:, :m])
        assert exact <= est.lambda_plus(m + 1) + d + tol, (m, exact, est.lambda_plus(m + 1) + d)
    for span in extra_spans:
        Vf = kernel.features(span.points).T @ span.coefs
        ms = [1, span.dim // 2, span.dim]
        for m, (val, _) in zip(ms, est.compressed_plus_norms(kernel, span, ms)):
            exact = _lam_max_perp(Om, Vf[:, :m])
            assert exact <= val + d + tol, (m, exact, val + d)


def _random_span(kernel, d, n_pts, m, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(n_pts, d, generator=g, dtype=F64)
    return op.orthonormalize(kernel, op.Span(P, torch.randn(n_pts, m, generator=g, dtype=F64)))


def _rff_blocks(N, d, seed, same):
    g = torch.Generator().manual_seed(seed)
    Y, Yp = torch.randn(N, d, generator=g, dtype=F64), torch.randn(N, d, generator=g, dtype=F64)
    if same:
        return Y, Y.clone(), Yp, Yp.clone()
    Z = 0.8 * Y + 0.6 * torch.randn(N, d, generator=g, dtype=F64)
    Zp = 0.8 * Yp + 0.6 * torch.randn(N, d, generator=g, dtype=F64)
    return Y, Z, Yp, Zp


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("same", [False, True])
@pytest.mark.parametrize("q", [8, 48])
def test_enclosure_dominates_exact_operator_rff(method, same, q):
    k = RFFKernel(3, 120, seed=1)
    Y, Z, Yp, Zp = _rff_blocks(150, 3, 2, same)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method, n_refined=q)
    Om = _exact_operator(k.features, Y, Z, Yp, Zp)
    assert isinstance(est, op.EnclosedKernelOperatorEstimate) and est.refined == q
    _check_rigour(est, k, Om, [_random_span(k, 3, 9, 6, 3), est.top_span(min(6, est.n_pos))])
    # tight where the refined block resolves the spectrum (fp32: ~ sqrt(u_32) relative at most)
    lam = torch.linalg.eigvalsh(Om).flip(0)
    scale = float(lam.abs().max())
    for j in range(1, 4):
        assert est.lambda_plus(j) + est.delta_trunc - max(float(lam[j - 1]), 0.0) < 1e-5 * scale
    if same:  # t = 0: O_hat >= 0 and trace(O_hat) = mean ||D||^2 / 2
        assert float(est.eigvals[: est.refined].min()) > -1e-9 * scale


def _prescribed(N, spec, seed, pair_sign=None):
    """Linear-kernel endpoints with O_hat = diag-like prescribed spectrum `spec` (in an orthonormal
    basis): D1 = Q diag(s), D2 = D1 diag(sign) with s^2 = 2N |spec|."""
    D = spec.numel()
    g = torch.Generator().manual_seed(seed)
    Qn, _ = torch.linalg.qr(torch.randn(N, D, generator=g, dtype=F64))
    Rot, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=F64))
    s = (2.0 * N * spec.abs()).sqrt()
    sign = torch.sign(spec)
    D1 = Qn * s[None, :] @ Rot.T
    D2 = Qn * (s * sign)[None, :] @ Rot.T
    Yp = torch.randn(N, D, generator=g, dtype=F64)
    return D1 + Yp, D2 + Yp, Yp, Yp.clone()  # Y, Z, Y', Z'


ADVERSARIAL = {
    # exact multiplicities straddling the refined-block boundary q = 8
    "cluster_at_q": torch.tensor([1.0] * 3 + [0.25] * 10 + [1e-3] * 5 + [0.0] * 4, dtype=F64),
    # near-degenerate (1e-12 splits) cluster across q, then a tail just below it
    "near_degenerate": torch.cat(
        [
            torch.tensor([0.9, 0.9 - 1e-12, 0.9 - 2e-12], dtype=F64),
            0.4 + 1e-12 * torch.arange(12, dtype=F64),
            torch.tensor([0.4 - 1e-9, 1e-6, 1e-12], dtype=F64),
        ]
    ),
    # indefinite (t > 0): +/- pairs of equal magnitude, negative block larger than the positive one
    "indefinite_pairs": torch.tensor(
        [0.5, -0.5, 0.5, -0.5, 0.2, -0.2] + [0.05] * 6 + [-0.9] * 4 + [1e-8, -1e-8], dtype=F64
    ),
    # rank-deficient, one dominant eigenvalue (huge dynamic range: G condition ~ 1e14)
    "dynamic_range": torch.tensor([1.0, 1e-4, 1e-7, 1e-10, 1e-13] + [0.0] * 15, dtype=F64),
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_enclosure_adversarial_spectra(method, name):
    spec = ADVERSARIAL[name]
    k = LinearKernel()
    Y, Z, Yp, Zp = _prescribed(64, spec, seed=4)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method, n_refined=8)
    Om = _exact_operator(k.features, Y, Z, Yp, Zp)
    assert torch.allclose(
        torch.linalg.eigvalsh(Om), torch.sort(spec).values, atol=1e-10 * float(spec.abs().max())
    )
    g = torch.Generator().manual_seed(5)
    P = torch.randn(10, spec.numel(), generator=g, dtype=F64)
    span = op.orthonormalize(k, op.Span(P, torch.randn(10, 5, generator=g, dtype=F64)))
    _check_rigour(est, k, Om, [span])


def _frozen_default(kernel, Y, Z, Yp, Zp, rtol=1e-11):
    """Independent reference for the default float64 algorithm."""
    N = Y.shape[0]
    E = torch.cat([Y, Yp, Z, Zp], 0)
    K = kernel(E, E)
    B = [[K[i * N : (i + 1) * N, j * N : (j + 1) * N] for j in range(4)] for i in range(4)]
    G11 = B[0][0] - B[0][1] - B[1][0] + B[1][1]
    G12 = B[0][2] - B[0][3] - B[1][2] + B[1][3]
    G22 = B[2][2] - B[2][3] - B[3][2] + B[3][3]
    G = torch.cat([torch.cat([G11, G12], 1), torch.cat([G12.T, G22], 1)], 0)
    G = 0.5 * (G + G.T)
    g, V = torch.linalg.eigh(G)
    g, V = g.flip(0), V.flip(1)
    gmax = float(g[0])
    keep = g > rtol * gmax
    r = int(keep.sum())
    g_cut = float(g[~keep].abs().max()) if r < g.shape[0] else 0.0
    Vr, gr = V[:, :r], g[:r]
    R = Vr * gr.sqrt()[None, :]
    CR = torch.cat([R[N:], R[:N]], 0) / (4.0 * N)
    Wm = R.T @ CR
    Wm = 0.5 * (Wm + Wm.T)
    theta, w = torch.linalg.eigh(Wm)
    theta, w = theta.flip(0), w.flip(1)
    c_diff = (Vr * gr.rsqrt()[None, :]) @ w
    coefs = torch.cat([c_diff[:N], -c_diff[:N], c_diff[N:], -c_diff[N:]], 0)
    return theta, coefs, g, r, (2.0 * math.sqrt(gmax * g_cut) + g_cut) / (4.0 * N)


@pytest.mark.parametrize("same", [False, True])
def test_default_path_bit_identical(same):
    k = RFFKernel(3, 400, seed=7)
    Y, Z, Yp, Zp = _rff_blocks(120, 3, 8, same)
    theta, coefs, g, r, dt = _frozen_default(k, Y, Z, Yp, Zp)
    for est in (
        op.estimate_operator(k, Y, Z, Yp, Zp),
        op.estimate_operator(k, Y, Z, Yp, Zp, method="fp64"),
    ):
        assert type(est) is op.KernelOperatorEstimate
        assert torch.equal(est.eigvals, theta) and torch.equal(est.coefs, coefs)
        assert torch.equal(est.gram_eigs, g) and est.rank == r and est.delta_trunc == dt


def test_unknown_method_raises():
    k = LinearKernel()
    Y, Z, Yp, Zp = _prescribed(16, torch.tensor([1.0, 0.5], dtype=F64), seed=0)
    with pytest.raises(ValueError):
        op.estimate_operator(k, Y, Z, Yp, Zp, method="fp16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_enclosure_on_gpu():
    k = RFFKernel(3, 120, seed=1)
    Y, Z, Yp, Zp = _rff_blocks(150, 3, 2, False)
    kc = RFFKernel(3, 120, seed=1)
    kc.W, kc.b = k.W.cuda(), k.b.cuda()
    est = op.estimate_operator(
        kc, *(T.cuda() for T in (Y, Z, Yp, Zp)), method="fp32-enclosed", n_refined=16
    )
    est.endpoints, est.coefs, est.eigvals = est.endpoints.cpu(), est.coefs.cpu(), est.eigvals.cpu()
    _check_rigour(est, k, _exact_operator(k.features, Y, Z, Yp, Zp))


# Spectral enclosure regressions
def _diag_problem(diag):
    """Linear kernel, t = 0, Y' = Z' = 0: O_hat = Y^T Y/(2N) with Y = diag(diag), N = len(diag)."""
    Y = torch.diag(torch.tensor(diag, dtype=F64))
    Zero = torch.zeros_like(Y)
    return Y, Y.clone(), Zero, Zero.clone()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("q", [1, 128])
def test_cancelling_span_coefficients_are_rejected(method, q):
    """Cancellation among 2^54 coefficients hides a large Gram defect.

    The floating-point check reports an orthonormal span; the certified check
    must reject it before evaluating a bound."""
    k = LinearKernel()
    Y, Z, Yp, Zp = _diag_problem([2.0, 0.5])
    span = op.Span(
        torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]], dtype=F64),
        torch.tensor([-1.0, 2.0**54, 1.0, -(2.0**54), 1.0], dtype=F64)[:, None],
    )
    assert op.require_orthonormal(k, span) == 0.0  # the fp check is fooled
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method, n_refined=q)
    with pytest.raises(FloatingPointError, match="Gram defect"):
        est.compressed_plus_norm(k, span)


def _correlated_defect_span():
    Q = torch.tensor([[0.5, 0.5], [2**-0.5, -(2**-0.5)], [0.5, 0.5]], dtype=F64)
    V = Q @ (torch.eye(2, dtype=F64) + 4e-7 * torch.ones((2, 2), dtype=F64))
    return op.Span(torch.eye(3, dtype=F64), V)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("q", [1, 128])
def test_correlated_gram_defect_charged(method, q):
    """Span accepted at the entrywise 1e-6 tolerance (error 8e-7) whose complement is (1,0,-1)/sqrt2:
    true restricted eigenvalue 17/12; the uncorrected projector underbounded it by 6.7e-7."""
    k = LinearKernel()
    Y, Z, Yp, Zp = _diag_problem([4.0, 1.0, 1.0])
    span = _correlated_defect_span()
    assert op.require_orthonormal(k, span, tol=1e-6) > 7e-7
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method, n_refined=q)
    val, info = est.compressed_plus_norm(k, span)
    assert val + est.delta_trunc >= 17.0 / 12.0, (val + est.delta_trunc, info)
    assert val + est.delta_trunc - 17.0 / 12.0 < 1e-4  # charged, not vacuous


def test_default_path_post_hoc_margin_covers_correlated_defect():
    k = LinearKernel()
    Y, Z, Yp, Zp = _diag_problem([4.0, 1.0, 1.0])
    span = _correlated_defect_span()
    err = op.require_orthonormal(k, span, tol=1e-6)
    est = op.estimate_operator(k, Y, Z, Yp, Zp)
    val, _ = est.compressed_plus_norm(k, span)
    assert val + est.delta_trunc < 17.0 / 12.0  # the default path underbounds (reproduced)
    mg = op.default_path_margins(
        err, 2, float(est.eigvals[0]), est.eigvec_orthonormality_error, est.rank
    )
    assert mg["status"] == "UNCERTIFIED"  # recorded diagnostics are not certified inputs
    assert val + est.delta_trunc + mg["compressed"] >= 17.0 / 12.0  # numerically, in this example
    assert mg["compressed"] < 1e-4


def test_default_path_margins_uncertified_on_cancelling_span():
    """The recorded Gram error is zero despite cancellation in the span.

    Its additive margin misses the true value 1, so the status is UNCERTIFIED."""
    k = LinearKernel()
    Y, Z, Yp, Zp = _diag_problem([2.0, 0.5])
    span = op.Span(
        torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]], dtype=F64),
        torch.tensor([-1.0, 2.0**54, 1.0, -(2.0**54), 1.0], dtype=F64)[:, None],
    )
    est = op.estimate_operator(k, Y, Z, Yp, Zp)
    val, _ = est.compressed_plus_norm(k, span)
    mg = op.default_path_margins(
        span.orthonormality_error(k),
        1,
        float(est.eigvals[0]),
        est.eigvec_orthonormality_error,
        est.rank,
    )
    assert mg["status"] == "UNCERTIFIED"
    assert val + est.delta_trunc + mg["compressed"] < 1.0  # the margin is not a correction
    certified = op.default_path_margins(0.0, 1, 1.0, 0.0, 1, overlap_eps=0.0, certified_inputs=True)
    assert certified["status"] == "CONDITIONAL"


def test_certified_lambda_max_and_overlap_bounds():
    g = torch.Generator().manual_seed(11)
    A = torch.randn(40, 40, generator=g, dtype=F64)
    A = 0.5 * (A + A.T)
    lam = float(torch.linalg.eigvalsh(A)[-1])
    ub = op.certified_lambda_max(A)
    assert lam <= ub < lam + 1e-8 * float(torch.linalg.matrix_norm(A, ord=2))
    k = RFFKernel(3, 50, seed=2)
    E = torch.randn(30, 3, generator=g, dtype=F64)
    coefs = torch.randn(30, 4, generator=g, dtype=F64)
    span = op.Span(
        torch.randn(7, 3, generator=g, dtype=F64), torch.randn(7, 2, generator=g, dtype=F64)
    )
    ov = op.certified_span_overlap(k, E, coefs, span)
    # Independent float64 reference from explicit features.
    Fe, Fp = k.features(E), k.features(span.points)
    W = (Fe.T @ coefs).T @ (Fp.T @ span.coefs)
    assert ((ov["W"] - W).abs() <= ov["EW"] + 1e-14 * W.abs().max()).all()


# Extreme-scale underflow
def _scaled_operator_case(L, N, s):
    """Linear kernel, Y' = Z' = 0, Y = s L[:N], Z = s L[N:]: O_hat = s^2 A with
    A = (L1^T L2 + L2^T L1)/(4N), computed exactly at scale 1 (the comparison is normalized by s^2)."""
    Zr = torch.zeros(N, L.shape[1], dtype=F64)
    A = (L[:N].T @ L[N:] + L[N:].T @ L[:N]) / (4.0 * N)
    return (L[:N] * s, L[N:] * s, Zr, Zr.clone()), 0.5 * (A + A.T)


def _indefinite_blocks_4d():
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-2.0, 2.0**-8, 2.0**-10, 0.0],
            [2.0**-8, 2.0, 0.0, 2.0**-10],
        ],
        dtype=F64,
    )


def _indefinite_blocks_130d():
    n = 65
    B = -2.0 * torch.eye(n, dtype=F64)
    B[2, 2] = 2.0
    B[0, 2] = B[2, 0] = 2.0**-8
    top = torch.cat([torch.eye(n, dtype=F64), torch.zeros(n, n, dtype=F64)], 1)
    bot = torch.cat([B, 2.0**-10 * torch.eye(n, dtype=F64)], 1)
    return torch.cat([top, bot], 0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("s", [2.0**-300, 1.0, 2.0**300])
def test_extreme_scale_indefinite_4x4(method, s):
    """At scale 2^-300, a Frobenius norm can underflow and underestimate the bound.

    Compare after normalization, with no absolute tolerance.
    """
    k = LinearKernel()
    (Y, Z, Yp, Zp), A = _scaled_operator_case(_indefinite_blocks_4d(), 2, s)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method, n_refined=3)
    truth = float(torch.linalg.eigvalsh(A)[-1])
    assert (est.lambda_plus(1) + est.delta_trunc) / (s * s) >= truth * (1 - 1e-14)
    span = op.Span(torch.eye(4, dtype=F64) * s, torch.eye(4, dtype=F64)[:, 2:3] / s)
    keep = [0, 1, 3]
    truth_perp = float(torch.linalg.eigvalsh(A[keep][:, keep])[-1])
    v, _ = est.compressed_plus_norm(k, span)
    assert (v + est.delta_trunc) / (s * s) >= truth_perp * (1 - 1e-14)
    assert (v + est.delta_trunc) / (s * s) - truth_perp < 1e-4  # not vacuous


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("s", [2.0**-300, 1.0])
def test_extreme_scale_indefinite_n65_default_q(method, s):
    k = LinearKernel()
    (Y, Z, Yp, Zp), A = _scaled_operator_case(_indefinite_blocks_130d(), 65, s)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method=method)  # default n_refined=128
    lam = torch.linalg.eigvalsh(A).flip(0)
    for j in (1, 2, 3, 10):
        assert (est.lambda_plus(j) + est.delta_trunc) / (s * s) >= float(lam[j - 1]) * (1 - 1e-14)


def test_uncertifiable_scale_is_rejected():
    """G entries subnormal (s = 2^-530): the certificate scalars cannot stay normal -> raise."""
    k = LinearKernel()
    (Y, Z, Yp, Zp), _ = _scaled_operator_case(_indefinite_blocks_4d(), 2, 2.0**-530)
    with pytest.raises(FloatingPointError, match="certifiable|range|rescaled"):
        op.estimate_operator(k, Y, Z, Yp, Zp, method="fp32-enclosed", n_refined=3)


def test_fro_is_an_upper_bound_at_extreme_scales():
    g = torch.Generator().manual_seed(3)
    T = torch.randn(50, 7, generator=g, dtype=F64)
    exact = float(torch.linalg.vector_norm(T))
    for e in (-1070, -900, -600, -300, 0, 300, 600, 1000):
        Ts = op._ldexp(T, e)
        assert op._fro(Ts) >= op._ldexp(exact, e) * (1 - 1e-15) or e <= -1070
        if -1000 <= e <= 1000:
            assert op._fro(Ts) <= op._ldexp(exact, e) * (1 + 1e-5)
    with pytest.raises(FloatingPointError):
        op._fro(torch.tensor([float("inf"), 1.0], dtype=F64))


# cuSOLVER non-convergence on two-dimensional RBF inputs
def test_nonconvergence_raises_typed_error(monkeypatch):
    """A LinAlgError of the low-precision eigensolve becomes EigensolveNotConverged (no fallback)."""
    real = torch.linalg.eigh

    def failing(A, *a, **kw):
        if A.dtype == torch.float32:
            raise torch.linalg.LinAlgError(
                "linalg.eigh: The algorithm failed to converge (simulated)"
            )
        return real(A, *a, **kw)

    monkeypatch.setattr(torch.linalg, "eigh", failing)
    k = RFFKernel(3, 60, seed=1)
    Y, Z, Yp, Zp = _rff_blocks(40, 3, 2, True)
    with pytest.raises(op.EigensolveNotConverged, match="chol-fp64"):
        op.estimate_operator(k, Y, Z, Yp, Zp, method="fp32-enclosed")
    est = op.estimate_operator(k, Y, Z, Yp, Zp, method="chol-fp64")  # fp64 eigensolve unaffected
    _check_rigour(est, k, _exact_operator(k.features, Y, Z, Yp, Zp))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("tid", ["gauss-d2-c25", "funnel-d2-sy3"])
def test_d2_rbf_repro_gpu(tid):
    """The float32 solve on a wide 2D RBF kernel may fail to converge.

    It must raise EigensolveNotConverged; the float64 Cholesky solve succeeds."""
    from mcmc.paperexp.rng import seed_for, torch_gen
    from mcmc.paperexp.targets import load_frozen_suite

    suite = load_frozen_suite()
    tgt, k = suite.target(tid), suite.kernel(tid, kind="rbf", bw="ell")
    g = torch_gen(seed_for("fp32-eval-v1", f"d2probe-{tid}", 0), "cuda")
    X, Xp = tgt.sample(2048, g), tgt.sample(2048, g)
    try:
        est = op.estimate_operator(k, X, X.clone(), Xp, Xp.clone(), method="fp32-enclosed")
        assert est.lambda_plus(1) > 0
    except op.EigensolveNotConverged as exc:
        assert "chol-fp64" in str(exc)
    est = op.estimate_operator(k, X, X.clone(), Xp, Xp.clone(), method="chol-fp64")
    assert est.lambda_plus(1) > 0 and est.n_orthonormal >= 16
