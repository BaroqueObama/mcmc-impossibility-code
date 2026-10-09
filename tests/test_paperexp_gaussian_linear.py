"""Tests for linear Gaussian chains and closed-form RBF integrals."""

import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import gaussian_linear as gl


@pytest.fixture(scope="module")
def slow3():
    C = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    mu = np.array([0.3, -0.2, 0.1])
    return gl.gibbs_chain(np.linalg.inv(C), mu), gl.AnisoRBF(np.array([1.0, 0.8, 1.2]), 1.1)


def test_gibbs_matrix_form_is_pathwise_coordinatewise_sweep(slow3):
    ch, _ = slow3
    rng = np.random.default_rng(0)
    P = np.linalg.inv(ch.C)
    x, z = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    x_ref = gl.gibbs_sweep_coordinatewise(P, ch.mu, x, z)
    x_mat = ch.mu + (x - ch.mu) @ ch.A.T + z @ ch.LQ.T
    assert np.abs(x_ref - x_mat).max() < 1e-12
    assert ch.stationarity_error() < 1e-14
    assert ch.reversibility_error() > 1e-2  # systematic scan is NOT reversible here


def test_reversible_ar_is_reversible():
    C = np.array([[4.0, 0.0], [0.0, 1.0]])
    th = math.pi / 4
    U = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    ch = gl.reversible_ar(C, U, [0.9, 0.5])
    assert ch.reversibility_error() < 1e-12 and ch.stationarity_error() < 1e-12


def test_noise_cov_and_time_t_law(slow3):
    ch, _ = slow3
    for t in (1, 4, 9):
        At = ch.power(t)
        assert np.abs(ch.noise_cov(t) - (ch.C - At @ ch.C @ At.T)).max() < 1e-12
    g = torch.Generator().manual_seed(3)
    a = np.array([0.5, -0.4, 0.2])
    n, t = 400_000, 3
    X = ch.sweep(ch.sample_stationary(n, g, "cpu", shift=a), g, t).numpy()
    m_true = ch.mu + ch.power(t) @ a
    se = np.sqrt(np.diag(ch.C) / n)
    assert np.abs((X.mean(0) - m_true) / se).max() < 5
    assert np.abs(np.cov(X.T) - ch.C).max() < 0.01


def test_closed_forms_vs_monte_carlo(slow3):
    ch, k = slow3
    g = torch.Generator().manual_seed(4)
    n = 1_000_000
    X = ch.sample_stationary(n, g, "cpu")
    X2 = ch.sample_stationary(n, g, "cpu")
    zp = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 0.5, 0.0]]) + ch.mu
    vals = k(torch.as_tensor(zp), X)
    z = (vals.mean(1).numpy() - gl.mean_embedding(k, ch.mu, ch.C, zp)) / (
        vals.std(1).numpy() / math.sqrt(n)
    )
    assert np.abs(z).max() < 5
    v = torch.exp(-0.5 * (((X - X2) / torch.as_tensor(k.scales * k.ell)) ** 2).sum(1))
    assert abs(v.mean().item() - gl.mean_embedding_norm2(k, ch.C)) < 5 * v.std().item() / math.sqrt(
        n
    )
    b = np.array([0.2, -0.1, 0.3])
    assert (
        abs(
            gl.mmd2_shifted_gaussians(k, ch.C, b)
            - gl.mmd2_gaussians(k, ch.mu + b, ch.C, ch.mu, ch.C)
        )
        < 1e-13
    )
    # trace and branched-pair moments of Sigma_t
    for t in (0, 1, 3):
        Y, Z = ch.sweep(X, g, t), ch.sweep(X, g, t)
        kyz = torch.exp(-0.5 * (((Y - Z) / torch.as_tensor(k.scales * k.ell)) ** 2).sum(1))
        tr_mc = kyz.mean().item() - gl.mean_embedding_norm2(k, ch.C)
        se = max(kyz.std().item() / math.sqrt(n), 1e-15)
        assert abs(tr_mc - gl.trace_sigma_t(k, ch.C, ch.cross_cov(t))) < 5 * se + 1e-12
        cf = gl.mean_embedding(k, ch.mu, ch.C, zp)
        aY = k(torch.as_tensor(zp), Y) - torch.as_tensor(cf)[:, None]
        aZ = k(torch.as_tensor(zp), Z) - torch.as_tensor(cf)[:, None]
        prod = aY[:, None, :] * aZ[None, :, :]
        zs = (
            prod.mean(-1).numpy()
            - gl.branched_pair_centered_moment(k, zp, ch.mu, ch.C, ch.cross_cov(t))
        ) / (prod.std(-1).numpy() / math.sqrt(n) + 1e-15)
        assert np.abs(zs).max() < 5, (t, zs)


def test_trace_at_t0_is_one_minus_norm2(slow3):
    ch, k = slow3
    assert abs(gl.trace_sigma_t(k, ch.C, ch.C) - (1.0 - gl.mean_embedding_norm2(k, ch.C))) < 1e-14


def test_landmark_bound_contains_finer_compression(slow3):
    """lambda_j(S_big) <= lambda_j(Sigma_t) <= lambda_j(S_small) + tau_small (necessary consequence)."""
    ch, k = slow3
    g = torch.Generator().manual_seed(5)
    Zs = ch.sample_stationary(60, g, "cpu").numpy()
    Zb = np.vstack([Zs, ch.sample_stationary(500, g, "cpu").numpy()])
    bs, bb = gl.landmark_basis(k, Zs), gl.landmark_basis(k, Zb)
    assert bs.orthonormality_error < 1e-8 and bb.orthonormality_error < 1e-6
    for t in (0, 1, 4):
        Ls, Lb = gl.landmark_operator(k, bs, ch, t), gl.landmark_operator(k, bb, ch, t)
        assert Ls.tau >= -1e-12 and Lb.tau >= -1e-12 and Lb.tau <= Ls.tau + 1e-12
        for j in (1, 2, 3, 5, 9):
            assert Lb.lam(j) <= Ls.upper_lambda(j) + 1e-12
        # compressed bound on another span inside the small landmark span
        Vb = Ls.top_basis(2)
        assert Ls.compressed_top(Vb) <= Ls.lam(1) + 1e-12
        assert abs(Ls.compressed_top(Vb) - Ls.lam(3)) < 1e-10


def test_hs_norm_closed_form_vs_galerkin_and_landmarks(slow3):
    """||Sigma_t||_HS^2: 1-D Hermite-Galerkin agreement and landmark compression from below (3-D)."""
    from numpy.polynomial.hermite_e import hermegauss

    rho = 0.8
    ch1 = gl.LinearGaussianChain(
        A=np.array([[rho]]), LQ=np.array([[math.sqrt(1 - rho**2)]]), mu=np.zeros(1), C=np.eye(1)
    )
    k1 = gl.AnisoRBF(np.ones(1), 1.0)
    x, w = hermegauss(160)
    w = w / math.sqrt(2 * math.pi)
    nmax = 40
    H = np.zeros((nmax + 1, x.size))
    H[0], H[1] = 1.0, x
    for n in range(1, nmax):
        H[n + 1] = (x * H[n] - math.sqrt(n) * H[n - 1]) / math.sqrt(n + 1)
    Kn = ((H * w) @ np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2) @ (H * w).T)[1:, 1:]
    for t in (0, 1, 4):
        D = rho ** (t * np.arange(1, nmax + 1))
        G = D[:, None] * Kn * D[None, :]
        assert abs((G**2).sum() - gl.sigma_t_hs_norm2(k1, ch1.C, ch1.cross_cov(t))) < 1e-10
        # truncated Galerkin trace is a lower bound converging slowly at t = 0 (tail of the series)
        tr_closed = gl.trace_sigma_t(k1, ch1.C, ch1.cross_cov(t))
        assert np.trace(G) <= tr_closed + 1e-12 and tr_closed - np.trace(G) < 1e-7
    ch, k = slow3
    g = torch.Generator().manual_seed(5)
    b = gl.landmark_basis(k, ch.sample_stationary(800, g, "cpu").numpy(), rtol=1e-10)
    for t in (0, 1, 4):
        S = gl.landmark_operator(k, b, ch, t).S
        hs = gl.sigma_t_hs_norm2(k, ch.C, ch.cross_cov(t))
        assert (S**2).sum() <= hs + 1e-10 and (S**2).sum() >= hs * (1 - 1e-3)
        A, B = ch.sample_stationary(6, g, "cpu").numpy(), ch.sample_stationary(6, g, "cpu").numpy()
        full = gl.branched_pair_centered_moment(k, A, ch.mu, ch.C, ch.cross_cov(t), Z2=B)
        assert (
            np.abs(
                np.diag(full)
                - gl.branched_pair_centered_moment_pairs(k, A, B, ch.mu, ch.C, ch.cross_cov(t))
            ).max()
            < 1e-15
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_fused_rowsum_matches_dense_on_cuda():
    """The torch.compile fused row-sum fast path equals summing the dense kernel (fp64)."""
    k = gl.AnisoRBF(np.array([0.05, 0.02, 0.04]), 3.0)
    g = torch.Generator(device="cuda").manual_seed(0)
    X = 0.05 * torch.randn(3000, 3, generator=g, dtype=torch.float64, device="cuda")
    Y = 0.05 * torch.randn(1024, 3, generator=g, dtype=torch.float64, device="cuda") + 0.01
    fused, dense = k.rowsum(X, Y), k(X, Y).sum(1)
    assert float(((fused - dense).abs() / dense).max()) < 1e-13
    assert torch.equal(k.rowsum(X.cpu(), Y.cpu()), k(X.cpu(), Y.cpu()).sum(1))
