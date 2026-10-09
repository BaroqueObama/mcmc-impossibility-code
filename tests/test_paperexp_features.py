"""Tests for Nystrom feature maps and truncation errors."""

import numpy as np
import pytest
import torch

from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import ScaledMatern32, ScaledRBF
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.targets import make_target

F64 = torch.float64


@pytest.fixture(
    scope="module",
    params=[("rbf", "full"), ("matern32", "full"), ("rbf", "truncated"), ("matern32", "truncated")],
    ids=lambda p: f"{p[0]}-{p[1]}",
)
def setup(request):
    kind, regime = request.param
    cls = ScaledRBF if kind == "rbf" else ScaledMatern32
    if regime == "full":  # well-conditioned K_ZZ, full rank retained
        t = make_target("mixture", 10, sep=6.0, w=0.5)
        k = cls(t.coordinate_std(), 2.5)
        feats = NystromFeatures.from_target(k, t, 400, "mixture-d10-s6-w0.5", 0)
    else:  # wide kernel in d=2: fast eigendecay, the 1e-12 cutoff binds for RBF
        t = make_target("gauss", 2, cond=25.0)
        k = cls(t.coordinate_std(), 3.0)
        feats = NystromFeatures.from_target(k, t, 600, "gauss-d2-c25", 0)
        if kind == "rbf":
            assert feats.rank < 600
    return t, k, feats


def test_reproduces_kernel_on_landmark_span(setup):
    t, k, f = setup
    Z = f.Z
    FZ = f(Z)
    K = k(Z, Z)
    err = (FZ @ FZ.T - K).abs().max().item()
    # the error is bounded by the discarded eigen-mass (plus roundoff)
    assert err <= f.discarded_eig_sum + 1e-9, (err, f.discarded_eig_sum)
    # a point in the landmark span direction: k(x, .) for x = a landmark is reproduced exactly
    x = Z[:3]
    assert (f(x) @ FZ.T - k(x, Z)).abs().max().item() <= f.discarded_eig_sum + 1e-9


def test_whitened_basis_is_orthonormal_and_rank_recorded(setup):
    _, _, f = setup
    assert f.whitening_error() < 1e-4
    assert 1 <= f.rank <= f.n_landmarks
    kept = f.eigvals[: f.rank]
    assert (kept > f.rel_cutoff * f.eigvals[0]).all()
    if f.rank < f.n_landmarks:
        assert f.eigvals[f.rank] <= f.rel_cutoff * f.eigvals[0]


def test_feature_inner_products_match_nystrom_formula(setup):
    t, k, f = setup
    X = t.sample(50, torch_gen(9))
    Y = t.sample(40, torch_gen(10))
    lhs = f(X) @ f(Y).T
    Kxz, Kzy = k(X, f.Z), k(f.Z, Y)
    rhs = Kxz @ f.B @ f.B.T @ Kzy
    assert (lhs - rhs).abs().max().item() < 1e-10


def test_residual_energy_nonnegative_and_zero_on_landmarks(setup):
    t, k, f = setup
    X = t.sample(500, torch_gen(12))
    r = f.residual_diag(X)
    assert r.min().item() >= -1e-9 and r.max().item() <= 1.0 + 1e-12
    rz = f.residual_diag(f.Z)
    assert rz.abs().max().item() < 1e-6


def test_blocked_feature_evaluation_matches_unblocked(setup):
    t, _, f = setup
    X = t.sample(300, torch_gen(13))
    assert torch.allclose(f(X, mem_bytes=8 * 4 * 400 * 7), f(X), atol=1e-14, rtol=0)


def test_cutoff_validation():
    t = make_target("gauss", 2, cond=25.0)
    k = ScaledRBF(np.ones(2), 1.0)
    with pytest.raises(ValueError):
        NystromFeatures(k, t.sample(10, torch_gen(0)), rel_cutoff=0.0)
