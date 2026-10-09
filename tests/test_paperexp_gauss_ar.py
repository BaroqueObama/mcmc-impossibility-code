"""Tests for reversible Gaussian AR chains and their Hermite spectra."""

import numpy as np
import pytest

from mcmc.paperexp import gauss_ar as g
from mcmc.paperexp import slowmodes as sm

CLEAR = (0.9, 0.5)
NARROW = (0.9, 0.809)


@pytest.fixture(scope="module")
def ar():
    return g.GaussAR((4.0, 1.0), 45.0, CLEAR)


@pytest.fixture(scope="module")
def nys(ar):
    Z = ar.sample_target(np.random.default_rng(20261005), 48)
    return g.nystrom_whitening(Z, 2.0, 1e-10)


def test_sampler_witnesses(ar):
    w = ar.witnesses()
    assert w["stationarity_resid"] < 1e-14
    assert w["noise_vs_closed_form"] < 1e-14
    assert w["noise_min_eig"] > 0.1
    assert w["reversibility_resid"] < 1e-14
    assert w["two_branch_vs_lag2t_max"] < 1e-14
    assert w["A_power_vs_matrix_power"] < 1e-14


def test_stationary_covariance_by_simulation(ar):
    rng = np.random.default_rng(1)
    X = ar.sample_target(rng, 200000)
    for _ in range(20):
        X = ar.step(rng, X)
    cov = X.T @ X / len(X)
    se = np.sqrt((ar.C**2 + np.outer(np.diag(ar.C), np.diag(ar.C))) / len(X))
    assert (np.abs(cov - ar.C) < 5 * se).all()


def test_mode_ranking():
    m = g.hermite_modes(np.array(CLEAR), 10)
    assert [tuple(i) for i in m.indices[:3]] == [(1, 0), (2, 0), (3, 0)]  # 2nd slowest is quadratic
    np.testing.assert_allclose(m.eigenvalues[:3], [0.9, 0.81, 0.729])
    m = g.hermite_modes(np.array(NARROW), 10)
    assert [tuple(i) for i in m.indices[:3]] == [(1, 0), (2, 0), (0, 1)]
    np.testing.assert_allclose(m.eigenvalues[:3], [0.9, 0.81, 0.809])
    assert m.rho_out(np.array(NARROW)) == pytest.approx(0.9**11)


def test_hermite_orthonormal_and_eigenfunctions(ar):
    modes = g.hermite_modes(ar.rho_arr, 12)
    assert g.orthonormality_check(ar, modes, 20) < 1e-11
    X = ar.sample_target(np.random.default_rng(2), 5)
    chk = g.eigenfunction_identity_check(ar, modes, X, max_degree=8, noise_order=12)
    assert chk["max_abs_err"] < 1e-10 * max(1.0, chk["max_abs_value"])


def test_landmark_mean_quadrature_and_mc(ar, nys):
    m_closed = g.landmark_mean(nys.Z, ar.C, nys.ell)
    Y, wt = g.gauss_hermite_grid(64)
    K = g.rbf_kernel(ar.from_modes(Y), nys.Z, nys.ell)
    np.testing.assert_allclose(wt @ K, m_closed, rtol=1e-12)
    X = ar.sample_target(np.random.default_rng(3), 100000)
    Kx = g.rbf_kernel(X, nys.Z, nys.ell)
    se = Kx.std(axis=0) / np.sqrt(len(X))
    assert (np.abs(Kx.mean(axis=0) - m_closed) < 5.5 * se).all()


def test_stable_vs_naive_lag_covariance(ar, nys):
    for s in (0, 1, 4, 16):
        D_s, info = g.landmark_lag_covariance(nys.Z, ar.C, ar.A_power(s), nys.ell)
        m = g.landmark_mean(nys.Z, ar.C, nys.ell)
        D_n = g.landmark_cross_moment(nys.Z, ar.C, ar.A_power(s), nys.ell) - np.outer(m, m)
        assert np.abs(D_s - D_n).max() < 1e-14
        assert info["P_rel_asym"] < 1e-12
        assert sm.relative_asymmetry(D_s) < 1e-12  # reversibility


def test_stable_lag_covariance_high_precision(ar):
    """At lag 400 the naive difference is pure cancellation; the stable form matches mpmath."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    Z = np.array([[0.7, -1.1], [2.0, 0.4]])
    s, ell = 400, 2.0
    D_s, _ = g.landmark_lag_covariance(Z, ar.C, ar.A_power(s), ell)
    # mpmath reference with A^s from the exact mode diagonalisation
    c = [mp.mpf(4), mp.mpf(1)]
    th = mp.pi / 4
    Ch = mp.matrix([[mp.sqrt(c[0]), 0], [0, mp.sqrt(c[1])]])
    Chi = mp.matrix([[1 / mp.sqrt(c[0]), 0], [0, 1 / mp.sqrt(c[1])]])
    U = mp.matrix([[mp.cos(th), -mp.sin(th)], [mp.sin(th), mp.cos(th)]])
    R = mp.matrix([[mp.mpf("0.9") ** s, 0], [0, mp.mpf("0.5") ** s]])
    As = Ch * U * R * U.T * Chi
    C = mp.matrix([[c[0], 0], [0, c[1]]])
    G = mp.matrix(4, 4)
    CAt, AC = C * As.T, As * C
    for i in range(2):
        for j in range(2):
            G[i, j], G[i + 2, j + 2] = C[i, j], C[i, j]
            G[i, j + 2], G[i + 2, j] = CAt[i, j], AC[i, j]
    l2 = mp.mpf(ell) ** 2
    Binv = (l2 * mp.eye(4) + G) ** -1
    det = mp.det(mp.eye(4) + G / l2) ** mp.mpf(-0.5)
    M1 = (l2 * mp.eye(2) + C) ** -1
    det1 = mp.det(mp.eye(2) + C / l2) ** mp.mpf(-0.5)
    for a in range(2):
        for b in range(2):
            z = mp.matrix([Z[a, 0], Z[a, 1], Z[b, 0], Z[b, 1]])
            za, zb = mp.matrix([Z[a, 0], Z[a, 1]]), mp.matrix([Z[b, 0], Z[b, 1]])
            Mab = det * mp.exp(-(z.T * Binv * z)[0] / 2)
            ma = det1 * mp.exp(-(za.T * M1 * za)[0] / 2)
            mb = det1 * mp.exp(-(zb.T * M1 * zb)[0] / 2)
            ref = float(Mab - ma * mb)
            assert D_s[a, b] == pytest.approx(ref, rel=1e-10)


def test_two_paths_agree(ar, nys):
    modes = g.hermite_modes(ar.rho_arr, 30)
    emb = g.embedded_modes(ar, nys, modes, 96)
    np.testing.assert_allclose(
        emb["mean"], nys.H.T @ g.landmark_mean(nys.Z, ar.C, nys.ell), atol=1e-9
    )
    S0, _ = g.sigma_t_landmark(ar, nys, 0)
    # Parseval: trace of Sigma_0 equals the sum of squared embedded-mode norms (up to tail)
    # (degree-30 truncation leaves a tail of relative size ~1e-6 at t = 0)
    assert np.trace(S0) == pytest.approx(np.sum(emb["W"] ** 2), rel=3e-6)
    assert np.trace(S0) >= np.sum(emb["W"] ** 2)
    for t in (1, 4, 16):
        SA, _ = g.sigma_t_landmark(ar, nys, t)
        SB = g.sigma_t_modes(emb["W"], modes.eigenvalues, t)
        assert np.linalg.norm(SA - SB, 2) < 1e-7 * np.linalg.norm(SA, 2)
        assert np.linalg.eigvalsh(SA)[0] > -1e-9  # PSD up to roundoff


def test_landmark_cross_moment_mc(ar, nys):
    """Closed-form lag-2t cross-moment versus explicit simulation (small version of the run check)."""
    rng = np.random.default_rng(9)
    t = 1
    X0 = ar.sample_target(rng, 200000)
    Xs = X0.copy()
    for _ in range(2 * t):
        Xs = ar.step(rng, Xs)
    K0, Ks = g.rbf_kernel(X0, nys.Z, nys.ell), g.rbf_kernel(Xs, nys.Z, nys.ell)
    prod_mean = K0.T @ Ks / len(X0)
    prod_sq = (K0**2).T @ (Ks**2) / len(X0)
    se = np.sqrt((prod_sq - prod_mean**2) / len(X0))
    M = g.landmark_cross_moment(nys.Z, ar.C, ar.A_power(2 * t), nys.ell)
    z = (prod_mean - M) / se
    assert np.abs(z).max() < 5.5


def test_coordinate_monitors_match_linear_modes(ar, nys):
    modes = g.hermite_modes(ar.rho_arr, 4)
    emb = g.embedded_modes(ar, nys, modes, 96)
    S0, _ = g.sigma_t_landmark(ar, nys, 0)
    coord = g.coordinate_monitors(ar, nys, S0, 96, 1e-10)
    j10 = [tuple(i) for i in modes.indices].index((1, 0))
    j01 = [tuple(i) for i in modes.indices].index((0, 1))
    w10, w01 = emb["W"][:, j10], emb["W"][:, j01]
    # x = C^{1/2} U y with U the 45-degree rotation: x1 = sqrt2 (y1 - y2), x2 = (y1 + y2)/sqrt2
    np.testing.assert_allclose(coord["b"][:, 0], np.sqrt(2) * (w10 - w01), atol=1e-10)
    np.testing.assert_allclose(coord["b"][:, 1], (w10 + w01) / np.sqrt(2), atol=1e-10)
    assert (coord["explained_fraction"] > 0.9).all() and (
        coord["explained_fraction"] <= 1 + 1e-8
    ).all()


def test_extended_precision_path_matches_float64_and_extends_it(ar):
    """Path Bmp agrees with float64 where float64 resolves, and keeps tiny angles beyond it."""
    pytest.importorskip("mpmath")
    Z = ar.sample_target(np.random.default_rng(5), 20)
    nys = g.nystrom_whitening(Z, 2.0, 1e-10)
    modes = g.hermite_modes(ar.rho_arr, 12)
    W = g.embedded_modes(ar, nys, modes, 48)["W"]
    S0 = g.sigma_t_modes(W, modes.eigenvalues, 0)
    _, Q0 = sm.eigh_desc(S0)
    spaces = {1: {"slow": W[:, :1], "static": Q0[:, :1]}}
    t = 4
    res = g.mp_population_at_t(W, modes.indices, ar.rho, t, spaces, dps=40)
    SB = g.sigma_t_modes(W, modes.eigenvalues, t)
    lam, Q = sm.eigh_desc(SB)
    np.testing.assert_allclose(res["eigs"][:3], lam[:3], rtol=1e-10)
    assert res["m"][1]["theta_slow"] == pytest.approx(
        sm.largest_principal_angle(Q[:, :1], W[:, :1]), rel=1e-8
    )
    ref = sm.constrained_width_cov(SB, Q0[:, :1])["width"]
    assert res["m"][1]["static_width"] == pytest.approx(ref, rel=1e-9)
    # far beyond float64 resolution: the angle obeys the computable bound sin(theta) <= e_t / l_t
    t = 200
    res = g.mp_population_at_t(W, modes.indices, ar.rho, t, spaces, dps=60)
    ev = modes.eigenvalues
    e_t = np.linalg.norm((W[:, 1:] * ev[1:] ** t) @ (W[:, 1:] * ev[1:] ** t).T, 2)
    ell_t = ev[0] ** (2 * t) * np.sum(W[:, 0] ** 2)
    theta = res["m"][1]["theta_slow"]
    assert 0 < theta and np.sin(theta) <= e_t / ell_t
    assert res["m"][1]["slow_ratio_minus_one"] >= -1e-30
