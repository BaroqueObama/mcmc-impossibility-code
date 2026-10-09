"""Kernel calculations checked against direct pairwise evaluation."""

import math
import os

import numpy as np
import pytest
import torch

from mcmc.paperexp.kernels import (
    ScaledMatern32,
    ScaledRBF,
    kernel_from_dict,
    kernel_matvec,
    kernel_rowsum,
    kernel_sum,
    mmd2_from_embedding,
)
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.targets import make_target

F64 = torch.float64
CUDA = os.environ.get("MCMC_TEST_CUDA") == "1" and torch.cuda.is_available()


def _naive(kind, scales, ell, X, Y):
    s = torch.as_tensor(scales, dtype=F64)
    out = torch.empty(X.shape[0], Y.shape[0], dtype=F64)
    for i in range(X.shape[0]):
        r = torch.sqrt((((X[i] - Y) / s) ** 2).sum(1))
        if kind == "rbf":
            out[i] = torch.exp(-(r**2) / (2 * ell**2))
        else:
            a = math.sqrt(3) * r / ell
            out[i] = (1 + a) * torch.exp(-a)
    return out


@pytest.fixture(scope="module")
def data():
    t = make_target("funnel", 10, sigma_y=3.0)
    g = torch_gen(11)
    X = t.sample(300, g)
    Y = t.sample(170, g)
    scales = np.abs(np.random.default_rng(0).normal(size=10)) + 0.5
    return X, Y, scales


@pytest.mark.parametrize("cls,kind", [(ScaledRBF, "rbf"), (ScaledMatern32, "matern32")])
def test_kernel_matches_naive_pairwise(data, cls, kind):
    X, Y, scales = data
    k = cls(scales, 1.7)
    K = k(X, Y)
    assert K.dtype == F64 and K.shape == (300, 170)
    assert (K - _naive(kind, scales, 1.7, X, Y)).abs().max().item() < 1e-13
    assert k.kappa2 == 1.0
    # near pairs (incl. the diagonal) are recomputed exactly, so k(x, x) == 1 bit-for-bit
    assert torch.equal(k(X, X).diagonal(), torch.ones(300, dtype=F64))
    far = torch.cat([X, 1e3 * X[:5]])  # cancellation-prone heavy-tail points
    Kf = k(far, far)
    assert torch.equal(Kf.diagonal(), torch.ones(305, dtype=F64))
    assert (Kf - _naive(kind, scales, 1.7, far, far)).abs().max().item() < 1e-13
    assert torch.equal(k.diag(X), torch.ones(300, dtype=F64))
    assert kernel_from_dict(k.to_dict()).ell == k.ell


@pytest.mark.parametrize("cls", [ScaledRBF, ScaledMatern32])
def test_gram_is_psd_and_bounded(data, cls):
    X, _, scales = data
    K = cls(scales, 0.9)(X, X)
    ev = torch.linalg.eigvalsh(0.5 * (K + K.T))
    assert ev.min().item() > -1e-12 * X.shape[0]
    assert K.max().item() <= 1.0 + 1e-15 and K.min().item() >= 0.0


@pytest.mark.parametrize("cls", [ScaledRBF, ScaledMatern32])
@pytest.mark.parametrize("mem", [8 * 4 * 7, 8 * 4 * 1000, 1 << 30])
def test_blocked_reductions_match_dense(data, cls, mem):
    X, Y, scales = data
    k = cls(scales, 1.3)
    K = k(X, Y)
    V = torch.randn(170, 3, dtype=F64, generator=torch_gen(5))
    assert abs(kernel_sum(k, X, Y, mem).item() - K.sum().item()) < 1e-10
    assert (kernel_rowsum(k, X, Y, mem) - K.sum(1)).abs().max().item() < 1e-12
    assert (kernel_matvec(k, X, Y, V, mem) - K @ V).abs().max().item() < 1e-12
    assert (kernel_matvec(k, X, Y, V[:, 0], mem) - K @ V[:, 0]).abs().max().item() < 1e-12


def test_fp32_fast_path_discrepancy_is_small_and_recorded(data, record_property):
    X, Y, scales = data
    for cls in (ScaledRBF, ScaledMatern32):
        k = cls(scales, 1.3)
        K64 = k(X, Y)
        K32 = k(X, Y, compute_dtype=torch.float32).to(F64)
        err_entry = (K64 - K32).abs().max().item()
        s64 = kernel_sum(k, X, Y).item()
        s32 = kernel_sum(k, X, Y, compute_dtype=torch.float32).item()
        rel_sum = abs(s64 - s32) / s64
        record_property(f"{cls.kind}_fp32_max_entry_err", err_entry)
        record_property(f"{cls.kind}_fp32_rel_sum_err", rel_sum)
        print(f"{cls.kind}: fp32 max entry err {err_entry:.3e}, rel sum err {rel_sum:.3e}")
        assert err_entry < 1e-5 and rel_sum < 1e-6


def test_mmd2_from_embedding_against_explicit_formula(data):
    X, Y, scales = data
    k = ScaledRBF(scales, 1.3)
    mu = torch.rand(X.shape[0], dtype=F64, generator=torch_gen(1))
    K = k(X, X)
    J = X.shape[0]
    v = K.sum().item() / J**2 - 2 * mu.mean().item() + 0.37
    u = (K.sum().item() - J) / (J * (J - 1)) - 2 * mu.mean().item() + 0.37
    assert abs(mmd2_from_embedding(k, X, mu, 0.37) - v) < 1e-13
    assert abs(mmd2_from_embedding(k, X, mu, 0.37, unbiased=True) - u) < 1e-13


def test_kernel_rejects_bad_input(data):
    X, _, scales = data
    k = ScaledRBF(scales, 1.0)
    with pytest.raises(TypeError):
        k(X.float(), X.float())
    with pytest.raises(ValueError):
        k(X[:, :5], X)
    with pytest.raises(ValueError):
        ScaledRBF(-scales, 1.0)


@pytest.mark.skipif(not CUDA, reason="set MCMC_TEST_CUDA=1 on a CUDA machine")
def test_cuda_matches_cpu(data):
    X, Y, scales = data
    for cls in (ScaledRBF, ScaledMatern32):
        k = cls(scales, 1.3)
        Kc = k(X.cuda(), Y.cuda()).cpu()
        assert (Kc - k(X, Y)).abs().max().item() < 1e-13
        assert (
            abs(kernel_sum(k, X.cuda(), Y.cuda(), 8 * 4 * 1000).item() - kernel_sum(k, X, Y).item())
            < 1e-10
        )
