"""Kernel operator calculations checked against explicit feature maps."""

import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import operators as op

F64 = torch.float64


class PolyKernel:
    """k(x, y) = (1 + x.y)^2 with an explicit finite feature map (for exact comparisons)."""

    kappa2 = float("nan")

    def __call__(self, X, Y):
        return (1.0 + X @ Y.T) ** 2

    @staticmethod
    def features(X):
        n, d = X.shape
        cols = [torch.ones(n, dtype=F64, device=X.device)]
        cols += [math.sqrt(2.0) * X[:, i] for i in range(d)]
        cols += [X[:, i] ** 2 for i in range(d)]
        cols += [math.sqrt(2.0) * X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
        return torch.stack(cols, 1)


def _blocks(N, d, seed, same=False):
    g = torch.Generator().manual_seed(seed)
    Y, Yp = torch.randn(N, d, generator=g, dtype=F64), torch.randn(N, d, generator=g, dtype=F64)
    if same:
        return Y, Y.clone(), Yp, Yp.clone()
    Z, Zp = (
        Y + 0.5 * torch.randn(N, d, generator=g, dtype=F64),
        Yp + 0.5 * torch.randn(N, d, generator=g, dtype=F64),
    )
    return Y, Z, Yp, Zp


def _explicit_operator(f, Y, Z, Yp, Zp):
    D1, D2 = f(Y) - f(Yp), f(Z) - f(Zp)
    Om = (D1.T @ D2 + D2.T @ D1) / (4.0 * Y.shape[0])
    return 0.5 * (Om + Om.T)


@pytest.mark.parametrize("same", [False, True])
def test_kernel_trick_matches_explicit_features(same):
    k = PolyKernel()
    Y, Z, Yp, Zp = _blocks(40, 3, 0, same)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, rtol=1e-12)
    Om = _explicit_operator(k.features, Y, Z, Yp, Zp)
    w = torch.linalg.eigvalsh(Om).flip(0)
    nz = w[w.abs() > 1e-10]
    ours = est.eigvals[est.eigvals.abs() > 1e-10]
    assert torch.allclose(torch.sort(nz).values, torch.sort(ours).values, atol=1e-9)
    assert est.delta_trunc < 1e-6
    if same:  # t = 0: O = (1/2N) sum D D^T is PSD
        assert float(est.eigvals.min()) > -1e-9
    # eigenvectors: feature-space vector of u_k = sum_a c_ak f(E_a) is an eigenvector of O
    F = k.features(est.endpoints)
    U = F.T @ est.coefs[:, :3]
    assert torch.allclose(Om @ U, U * est.eigvals[:3][None, :], atol=1e-8)
    assert torch.allclose(U.T @ U, torch.eye(3, dtype=F64), atol=1e-8)


def test_compressed_plus_norm_exact_and_head_tail():
    k = PolyKernel()
    Y, Z, Yp, Zp = _blocks(60, 4, 1)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, rtol=1e-12)
    Om = _explicit_operator(k.features, Y, Z, Yp, Zp)
    w, V = torch.linalg.eigh(Om)
    Op = V @ torch.diag(w.clamp_min(0)) @ V.T
    g = torch.Generator().manual_seed(9)
    P = torch.randn(7, 4, generator=g, dtype=F64)
    span = op.orthonormalize(k, op.Span(P, torch.randn(7, 3, generator=g, dtype=F64)))
    assert span.orthonormality_error(k) < 1e-10
    Vf = k.features(P).T @ span.coefs  # orthonormal feature-space basis of V
    Pperp = torch.eye(Op.shape[0], dtype=F64) - Vf @ Vf.T
    exact = float(torch.linalg.eigvalsh(Pperp @ Op @ Pperp)[-1])
    val, info = est.compressed_plus_norm(k, span, head=10_000)
    assert info["exact"] and abs(val - exact) < 1e-9
    for head in (2, 4, 6):
        ub, inf2 = est.compressed_plus_norm(k, span, head=head)
        assert ub >= exact - 1e-10, (head, ub, exact)
    # own top span: compressed norm equals the next positive eigenvalue
    for m in (1, 3, 5):
        top = est.top_span(m)
        val, _ = est.compressed_plus_norm(k, top, head=10_000)
        assert abs(val - est.lambda_plus(m + 1)) < 1e-9


def test_compression_onto_landmarks_equals_feature_operator():
    C = np.array([[1.0, 0.6], [0.6, 2.0]])
    ch = gl.gibbs_chain(np.linalg.inv(C), np.zeros(2))
    k = gl.AnisoRBF(np.sqrt(np.diag(C)), 1.0)
    g = torch.Generator().manual_seed(2)
    basis = gl.landmark_basis(k, ch.sample_stationary(40, g, "cpu").numpy())
    Zl, W = torch.as_tensor(basis.Z), torch.as_tensor(basis.W)

    def feats(X):
        return k(X, Zl) @ W

    X, Xp = ch.sample_stationary(200, g, "cpu"), ch.sample_stationary(200, g, "cpu")
    Y, Z, Yp, Zp = ch.sweep(X, g, 2), ch.sweep(X, g, 2), ch.sweep(Xp, g, 2), ch.sweep(Xp, g, 2)
    est = op.estimate_operator(k, Y, Z, Yp, Zp)
    fo = op.feature_operator(feats, Y, Z, Yp, Zp)
    # <e_i, O_hat e_j> from the kernel-trick eigendecomposition
    p = est.eigvals.shape[0]
    X_ = (
        W.T @ k(Zl, est.endpoints) @ est.coefs[:, :p]
    )  # (r, p) coords of eigenvectors in landmark basis
    O_comp = X_ @ torch.diag(est.eigvals) @ X_.T
    assert torch.allclose(O_comp, fo.matrix, atol=1e-8 + 10 * est.delta_trunc)


def test_operator_radius_and_inverse():
    e = op.operator_radius(1.0, 2048, 0.0125, 1 / 16)
    assert abs(e - 2 / math.sqrt(2048) * (1 + math.sqrt(2 * math.log(16 / 0.0125)))) < 1e-15
    N = op.n_blocks_for_tolerance(0.05, 0.8, 1.0, 0.0125, 1 / 16)
    assert (
        abs(0.8 * math.sqrt(op.operator_radius(1.0, int(math.ceil(N)), 0.0125, 1 / 16)) - 0.05)
        < 1e-4
    )
    with pytest.raises(ValueError):
        op.operator_radius(1.0, 10, 1.5, 0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_cpu():
    C = np.array([[1.0, 0.6], [0.6, 2.0]])
    k = gl.AnisoRBF(np.sqrt(np.diag(C)), 1.0)
    Y, Z, Yp, Zp = _blocks(300, 2, 3)
    cpu = op.estimate_operator(k, Y, Z, Yp, Zp)
    gpu = op.estimate_operator(k, *(T.cuda() for T in (Y, Z, Yp, Zp)))
    n = min(cpu.rank, gpu.rank, 50)
    assert torch.allclose(cpu.eigvals[:n], gpu.eigvals[:n].cpu(), atol=1e-10)


def test_head_tail_slack_is_valid_including_m_ge_head():
    """The slack bound covers the head/tail approximation even when V spans the head.

    For V spanning both eigenvectors, the exact compressed norm is zero.
    """
    from types import SimpleNamespace

    z = SimpleNamespace(n_pos=2, eigvals=torch.tensor([2.0, 1.0], dtype=F64))
    val, info = op.KernelOperatorEstimate._compressed_from_X(z, torch.eye(2, dtype=F64), 1)
    assert val >= 0.0 and info["slack_bound"] >= val - 0.0 - 1e-12
    k = PolyKernel()
    Y, Z, Yp, Zp = _blocks(60, 4, 4)
    est = op.estimate_operator(k, Y, Z, Yp, Zp, rtol=1e-12)
    span = est.top_span(6)
    exact = [v for v, _ in est.compressed_plus_norms(k, span, [1, 3, 6], head=10_000)]
    for head in (2, 4, 6):
        for (v, info), ex, m in zip(
            est.compressed_plus_norms(k, span, [1, 3, 6], head=head), exact, [1, 3, 6]
        ):
            assert v >= ex - 1e-10, (head, m)
            assert info["slack_bound"] >= v - ex - 1e-10, (head, m, info, ex)
