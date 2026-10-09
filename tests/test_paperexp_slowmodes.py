"""Tests for TICA, subspace angles, and constrained widths."""

from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

from mcmc.paperexp import slowmodes as sm
from mcmc.paperexp.gauss_ar import GaussAR

REPO = Path(__file__).resolve().parents[1]


def random_orthogonal(rng, r):
    Q, R = np.linalg.qr(rng.standard_normal((r, r)))
    return Q * np.sign(np.diag(R))


def test_rejects_non_float64_and_nonfinite():
    with pytest.raises(TypeError):
        sm.psd_whitening(np.eye(3, dtype=np.float32), 1e-10)
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        sm.psd_whitening(bad, 1e-10)
    with pytest.raises(ValueError):
        sm.psd_whitening(np.array([[1.0, 0.5], [0.0, 1.0]]), 1e-10)  # asymmetric


def test_whitening_cutoff_is_explicit():
    lam = np.array([1.0, 1e-3, 1e-9, 1e-13, -1e-16])
    Q = random_orthogonal(np.random.default_rng(1), 5)
    C0 = Q @ np.diag(lam) @ Q.T
    wh = sm.psd_whitening(C0, rel_cutoff=1e-10)
    assert wh.rank == 3 and wh.n_dropped == 2
    np.testing.assert_allclose(wh.transform.T @ C0 @ wh.transform, np.eye(3), atol=1e-6)
    assert wh.most_negative <= 0.0


def test_zero_rank_and_empty_cutoff_raise_rank_deficient():
    """Empty retained spectra raise RankDeficientError for the caller to record."""
    with pytest.raises(sm.RankDeficientError):
        sm.psd_whitening(np.zeros((3, 3)), 1e-10)
    with pytest.raises(sm.RankDeficientError):
        sm.tica_from_path(np.ones((32, 3)), 1, 1e-10, n_modes=2)  # constant pilot path
    with pytest.raises(sm.RankDeficientError):
        sm.tica_symmetric(np.eye(2), np.eye(2), 1e-3, abs_floor=2.0)  # cutoff keeps nothing
    with pytest.raises(sm.RankDeficientError):
        sm.psd_whitening(np.diag([1.0, 0.5]), 1e-3, abs_floor=1.0)
    assert issubclass(sm.RankDeficientError, ValueError)  # backward compatible


def test_tica_population_known_answer():
    """Exact population covariances of Hermite features, mixed by an invertible matrix M.

    Features f = M psi with psi = (y1, y2, He2(y1)/sqrt2, y1 y2, He2(y2)/sqrt2) orthonormal and
    transition eigenvalues (r1, r2, r1^2, r1 r2, r2^2). Then C0 = M M^T, Clag = M diag(rho) M^T;
    TICA must return those eigenvalues, u_j^T f = +-psi_j and embedded w_j = C0 u_j = +-M e_j.
    """
    r1, r2 = 0.9, 0.5
    rho = np.array([r1, r2, r1**2, r1 * r2, r2**2])
    rng = np.random.default_rng(7)
    M = rng.standard_normal((5, 5)) + 3 * np.eye(5)
    C0, Clag = M @ M.T, M @ np.diag(rho) @ M.T
    res = sm.tica_symmetric(C0, Clag, rel_cutoff=1e-12)
    order = np.argsort(-rho)
    np.testing.assert_allclose(res.eigenvalues, rho[order], atol=1e-12)
    # u_j^T M = +- e_{order_j}^T
    P = res.coefficients.T @ M
    np.testing.assert_allclose(np.abs(P), np.eye(5)[order], atol=1e-10)
    np.testing.assert_allclose(np.abs(res.embedded), np.abs(M[:, order]), atol=1e-10)
    np.testing.assert_allclose(res.coefficients.T @ C0 @ res.coefficients, np.eye(5), atol=1e-10)


def test_tica_orders_by_absolute_value():
    C0 = np.eye(3)
    Clag = np.diag([0.2, -0.95, 0.6])
    res = sm.tica_symmetric(C0, Clag, 1e-12, order="abs")
    np.testing.assert_allclose(res.eigenvalues, [-0.95, 0.6, 0.2])
    res = sm.tica_symmetric(C0, Clag, 1e-12, order="signed")
    np.testing.assert_allclose(res.eigenvalues, [0.6, 0.2, -0.95])
    with pytest.raises(sm.RankDeficientError):
        sm.tica_symmetric(np.diag([1.0, 1e-14, 1e-15]), Clag, 1e-10, n_modes=2)


def test_tica_from_ar_path_and_pair_bound():
    ar = GaussAR((4.0, 1.0), 45.0, (0.9, 0.5))
    X = ar.path(np.random.default_rng(3), 40000)
    res = sm.tica_from_path(X, lag=1, rel_cutoff=1e-10)
    np.testing.assert_allclose(res.eigenvalues, [0.9, 0.5], atol=0.02)
    # pair-symmetrised estimator keeps whitened eigenvalues inside [-1, 1] even when overfitting
    F = np.random.default_rng(4).standard_normal((40, 30))
    res = sm.tica_from_path(F, lag=1, rel_cutoff=1e-12)
    assert res.max_abs_eigenvalue <= 1.0 + 1e-10


def test_dynamic_covariance_from_path_linear_features():
    """For f(x) = x on the reversible AR, Sigma_t = A^t C A^tT = A^{2t} C."""
    ar = GaussAR((4.0, 1.0), 45.0, (0.9, 0.5))
    X = ar.path(np.random.default_rng(5), 200000)
    for t in (0, 2, 4):
        est, _ = sm.dynamic_covariance_from_path(X, t)
        At = ar.A_power(t)
        np.testing.assert_allclose(est, At @ ar.C @ At.T, atol=0.25)


def test_principal_angles_known_and_tiny():
    rng = np.random.default_rng(11)
    r = 8
    Q = random_orthogonal(rng, r)
    # absolute accuracy ~1e-16 everywhere; relative accuracy is ~1e-11 at 1e-6 rad but only
    # ~1e-4 at 1e-12 rad (observed 7.6e-5: absolute roundoff / theta), so both are asserted separately
    for theta, rel in ((0.3, 1e-12), (1.2, 1e-12), (np.pi / 2, 1e-12), (1e-6, 1e-9), (1e-12, 2e-4)):
        A = Q[:, :2]
        B = np.column_stack([Q[:, 0], np.cos(theta) * Q[:, 1] + np.sin(theta) * Q[:, 2]])
        ang = sm.principal_angles(A, B)
        assert abs(ang[0]) < 1e-15
        assert abs(ang[1] - theta) < 1e-15 or abs(ang[1] - theta) <= 4 * np.finfo(float).eps * theta
        assert abs(ang[1] - theta) / theta < rel
        assert abs(sm.largest_principal_angle(A, B * 3.0) - theta) / theta < rel
    # cosine route loses tiny angles completely; sine route keeps them
    A = Q[:, :1]
    B = (np.cos(1e-12) * Q[:, 0] + np.sin(1e-12) * Q[:, 1])[:, None]
    assert sm.largest_angle_cosine_route(A, B) < 1e-7
    assert sm.largest_principal_angle(A, B) == pytest.approx(1e-12, rel=1e-6)
    with pytest.raises(sm.RankDeficientError):
        sm.orthonormal_basis(np.column_stack([Q[:, 0], Q[:, 0]]))


def test_constrained_width_closed_forms_and_agreement():
    rng = np.random.default_rng(13)
    r = 9
    Q = random_orthogonal(rng, r)
    lam = np.sort(rng.uniform(0.01, 1.0, r))[::-1]
    S = Q @ np.diag(lam) @ Q.T
    L, neg = sm.psd_factor(S)
    assert neg == 0.0 or abs(neg) < 1e-14
    for m in (1, 2, 3):
        # optimal monitors: leading eigenvectors give sqrt(lam_{m+1})
        w = sm.constrained_width_factor(L, Q[:, :m])
        assert w == pytest.approx(np.sqrt(lam[m]), rel=1e-12)
        assert sm.constrained_width_cov(S, Q[:, :m])["width"] == pytest.approx(
            np.sqrt(lam[m]), rel=1e-12
        )
        # arbitrary monitors: both forms agree and dominate the optimum
        W = rng.standard_normal((r, m))
        wf = sm.constrained_width_factor(L, W)
        wc = sm.constrained_width_cov(S, W)["width"]
        assert wf == pytest.approx(wc, rel=1e-10)
        assert wf >= np.sqrt(lam[m]) * (1 - 1e-12)
        # factor independence: any L rot gives the same width
        rot = random_orthogonal(rng, L.shape[1])
        assert sm.constrained_width_factor(L @ rot, W) == pytest.approx(wf, rel=1e-10)
        # secant bound of Corollary approximate-monitors
        theta = sm.largest_principal_angle(Q[:, :m], W)
        assert wf <= np.sqrt(lam[m]) / np.cos(theta) * (1 + 1e-10)


def test_constrained_width_brute_force():
    """Width = max ||L u|| over the unit ball intersected with ker(W^T L), by dense sampling."""
    rng = np.random.default_rng(17)
    L = rng.standard_normal((3, 4))
    W = rng.standard_normal((3, 1))
    N = linalg.null_space(W.T @ L)  # (4, 3)
    U = rng.standard_normal((200000, N.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    brute = np.linalg.norm(U @ N.T @ L.T, axis=1).max()
    exact = sm.constrained_width_factor(L, W)
    assert brute <= exact * (1 + 1e-12)
    assert brute >= exact * (1 - 1e-3)


def test_reproduces_saved_finite_example():
    """Constrained widths match the saved eight-state example."""
    setup = np.load(REPO / "tests/data/finite_setup.npz")
    b, rho = setup["B"].astype(np.float64), setup["rho"].astype(np.float64)
    static = np.linalg.eigh(b @ b.T)[1][:, -1:]
    j = int(np.argmax(np.abs(rho)))
    slow = b[:, j : j + 1]
    expected = {0: 1.08249, 1: 1.00977, 3: 1.000859, 10: 1.000000044}
    import json

    saved = json.loads(
        (
            REPO / "tests/data/diagnostics_time_and_estimation.json"
        ).read_text()
    )
    for t, val in expected.items():
        A = b @ np.diag(rho**t)
        s = linalg.svdvals(A)
        ratio = sm.constrained_width_factor(A, slow) / s[1]
        assert ratio == pytest.approx(val, abs=6e-6 if t == 0 else 6e-6 * val)
        row = next(
            r
            for r in saved["exact_time_comparison"]
            if r["time"] == t and r["method"] == "Embedded slow mode"
        )
        assert ratio == pytest.approx(row["ratio"], rel=1e-12)
        row = next(
            r
            for r in saved["exact_time_comparison"]
            if r["time"] == t and r["method"] == "Static covariance"
        )
        assert sm.constrained_width_factor(A, static) / s[1] == pytest.approx(
            row["ratio"], rel=1e-12
        )


def test_davis_kahan_helper():
    assert sm.davis_kahan_sin_bound(1e-3, 1e-4) == 1.0
    assert sm.davis_kahan_sin_bound(1e-6, 1.0) == pytest.approx(1e-6, rel=1e-5)


def test_tica_eigenvalue_noise_scale():
    """The lag-1 AR estimate is within five standard errors of rho."""
    rng = np.random.default_rng(23)
    rho, T = 0.9, 50000
    x = np.empty(T)
    x[0] = rng.standard_normal()
    e = rng.standard_normal(T) * np.sqrt(1 - rho**2)
    for s in range(T - 1):
        x[s + 1] = rho * x[s] + e[s + 1]
    res = sm.tica_from_path(x[:, None], 1, 1e-10)
    se = np.sqrt((1 - rho**2) / T)
    assert abs(res.eigenvalues[0] - rho) < 5 * se
