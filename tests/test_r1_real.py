"""Tests for Gaussian spectra and projected residuals."""

import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import r1_real as rr
from mcmc.paperexp import r1_suite as r1
from mcmc.paperexp.kernels import ScaledRBF
from mcmc.paperexp.rng import np_gen


def _gauss3():
    rng = np_gen(5)
    A = rng.normal(size=(3, 3))
    C = A @ A.T / 3 + 0.3 * np.eye(3)
    mu = np.array([0.3, -1.0, 2.0])
    scales = np.sqrt(np.diag(C))
    return mu, C, scales, 1.4


@pytest.fixture(scope="module")
def spec3():
    mu, C, s, ell = _gauss3()
    return rr.exact_spectrum(mu, C, s, ell, rel_floor=1e-15, n_out=24)


def test_exact_spectrum_2d_matches_existing_witness():
    C = np.array([[1.0, 0.6], [0.6, 2.0]])
    k = ScaledRBF(np.array([0.9, 1.3]), 1.5)
    sp = rr.exact_spectrum(np.zeros(2), C, k.scales, k.ell, rel_floor=1e-16, n_out=30)
    g = r1.gaussian_centered_spectrum(C, k)
    assert np.abs(sp.eig[:30] / g["eig"][:30] - 1).max() < 1e-10


def test_exact_spectrum_3d_matches_brute_force_quadrature(spec3):
    mu, C, s, ell = _gauss3()
    t, w = np.polynomial.hermite_e.hermegauss(17)
    w = w / w.sum()
    Z = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    W = np.prod(np.stack(np.meshgrid(w, w, w, indexing="ij"), -1).reshape(-1, 3), 1)
    X = mu[None, :] + Z @ np.linalg.cholesky(C).T
    k = gl.AnisoRBF(s, ell)
    K = k(torch.as_tensor(X), torch.as_tensor(X)).numpy()
    Kw = K @ W
    Kc = K - Kw[:, None] - Kw[None, :] + W @ Kw
    ev = np.sort(np.linalg.eigvalsh(np.sqrt(W)[:, None] * Kc * np.sqrt(W)[None, :]))[::-1]
    # the 17^3 Gauss-Hermite reference itself is accurate to ~1e-5 relative here
    assert np.allclose(spec3.eig[:12], ev[:12], rtol=1e-4)
    assert abs(spec3.info["mu_norm2_exact"] - gl.mean_embedding_norm2(k, C)) < 1e-14
    assert abs(spec3.info["mu_norm2_from_modes"] - spec3.info["mu_norm2_exact"]) < 1e-12


def test_eigenfunctions_orthonormal_and_centered(spec3):
    mu, C, s, ell = _gauss3()
    X = np_gen(9).multivariate_normal(mu, C, size=400_000)
    P = rr.eval_psi(spec3, X, 8)
    G = P.T @ P / X.shape[0]
    assert np.abs(G - np.eye(8)).max() < 0.02
    assert np.abs(P.mean(0)).max() < 0.01


def test_interval_gram_routes_agree():
    n = 12
    full = rr.sps.ndtri(np.arange(n + 1) / n)
    g1 = rr.interval_gram_owen(0.37, full)
    g2 = rr.interval_gram_gl(0.37, rr.interval_edges(n, 9.5), 0.2, 10)
    assert np.abs(g1 - g2).max() < 1e-13
    assert abs(g1.sum() - 1 / math.sqrt(1 + 2 * 0.37)) < 1e-14


def test_plane_cell_integrals_against_monte_carlo(spec3):
    mu, C, s, ell = _gauss3()
    n = 4
    ps = rr.plane_setup(spec3, (0, 2), n, 0.25, 8, 9.0, "owen")
    assert abs(ps.G.sum() - spec3.info["mu_norm2_exact"]) < 1e-13
    assert np.abs(ps.cell_psi.sum(1)).max() < 1e-12  # centered eigenfunctions
    X = np_gen(10).multivariate_normal(mu, C, size=1_000_000)
    from mcmc.experiments.exp_r1c_real import _plane_cell_index

    cell = _plane_cell_index(spec3, (0, 2), n, X)
    P = rr.eval_psi(spec3, X, 6)
    mc = (
        np.stack([np.bincount(cell, weights=P[:, i], minlength=n * n) for i in range(6)])
        / X.shape[0]
    )
    se = 0.004 / n
    assert np.abs(mc - ps.cell_psi[:6]).max() < 5 * se


def test_shifted_readings_and_mmd(spec3):
    mu, C, s, ell = _gauss3()
    shift = np.array([0.4, -0.2, 0.3])
    k = gl.AnisoRBF(s, ell)
    assert abs(rr.shifted_mmd2(spec3, shift) - gl.mmd2_shifted_gaussians(k, C, shift)) < 1e-15
    r = rr.shifted_readings(spec3, shift, 10)
    X = np_gen(11).multivariate_normal(mu + shift, C, size=1_000_000)
    mc = np.sqrt(spec3.eig[:10]) * rr.eval_psi(spec3, X, 10).mean(0)
    assert np.abs(mc - r).max() < 4e-3 * np.sqrt(spec3.eig[0])
    # Parseval: sum of all readings^2 <= MMD^2 (equality in the limit of all eigenfunctions)
    assert float(r @ r) <= rr.shifted_mmd2(spec3, shift) * (1 + 1e-12)
