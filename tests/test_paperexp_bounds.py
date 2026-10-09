"""Tests for confidence radii, budget allocation, and projected bounds."""

import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import bounds as bd
from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import operators as op

F64 = torch.float64


def test_radii_formulas():
    b = bd.replica_radius(1.0, 16384, 0.025, 1 / 24)
    assert abs(b - (1 / 128 + math.sqrt(2 * math.log(24 / 0.025) / 16384))) < 1e-15
    assert bd.target_radius(1.0, 4096, 0.01) == bd.replica_radius(1.0, 4096, 0.01, 1.0)
    with pytest.raises(ValueError):
        bd.replica_radius(1.0, 0, 0.1)


def test_budget_refuses_overflow_and_grids():
    with pytest.raises(ValueError):
        bd.BudgetAllocation(0.05, {"a": 0.03, "b": 0.03})
    bud = bd.BudgetAllocation(0.05, {"replicas": 0.025, "target": 0.0125, "operator": 0.0125})
    bud.declare_grid("replicas", {"t": [0, 1, 2, 4], "J": [1, 2, 3]})
    assert abs(bud.cell_beta("replicas") - 0.025 / 12) < 1e-18
    with pytest.raises(ValueError):
        bud.declare_grid("replicas", {"t": [0]})
    with pytest.raises(KeyError):
        bud.cell_beta("operator")


def test_prefix_means_and_vstat_match_dense():
    k = gl.AnisoRBF(np.array([1.0, 2.0]), 0.7)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 2, generator=g, dtype=F64)
    P = torch.randn(5, 2, generator=g, dtype=F64)
    pm = bd.kernel_prefix_means(k, P, X, [16, 32, 64], chunk=16)
    for q, J in enumerate([16, 32, 64]):
        assert torch.allclose(pm[q], k(P, X[:J]).mean(1), atol=1e-14)
    gm = bd.kernel_gram_prefix_means(k, X, [16, 64], chunk=16)
    assert abs(float(gm[1]) - float(k(X, X).mean())) < 1e-14
    assert abs(float(gm[0]) - float(k(X[:16], X[:16]).mean())) < 1e-14


def test_direct_norm_known_and_sampled():
    C = np.array([[1.0, 0.3], [0.3, 0.5]])
    k = gl.AnisoRBF(np.sqrt(np.diag(C)), 1.0)
    g = torch.Generator().manual_seed(1)
    L = torch.as_tensor(np.linalg.cholesky(C))
    X = torch.randn(256, 2, generator=g, dtype=F64) @ L.T + 0.4
    Xp = torch.randn(128, 2, generator=g, dtype=F64) @ L.T
    reps = bd.ReplicaSet(X, [128, 256], chunk=128)
    ref = bd.TargetReference.sampled(Xp, 1.0, 0.01)
    dn = bd.direct_norm(k, reps, ref)
    dense = k(X, X).mean() - 2 * k(X, Xp).mean() + k(Xp, Xp).mean()
    assert abs(float(dn[1]) ** 2 - float(dense)) < 1e-13
    mu_fn = lambda P: torch.as_tensor(gl.mean_embedding(k, np.zeros(2), C, P.cpu().numpy()))  # noqa: E731
    known = bd.TargetReference.known(mu_fn, gl.mean_embedding_norm2(k, C))
    dk = bd.direct_norm(k, reps, known)
    dense_k = k(X, X).mean() - 2 * mu_fn(X).mean() + gl.mean_embedding_norm2(k, C)
    assert abs(float(dk[1]) ** 2 - float(dense_k)) < 1e-13 and known.b_pi == 0.0


def test_projected_norm_equals_explicit_projection():
    from tests.test_paperexp_operators import PolyKernel  # explicit features

    k = PolyKernel()
    g = torch.Generator().manual_seed(2)
    X, Xp = (
        torch.randn(64, 3, generator=g, dtype=F64) + 0.3,
        torch.randn(64, 3, generator=g, dtype=F64),
    )
    span = op.orthonormalize(
        k,
        op.Span(
            torch.randn(6, 3, generator=g, dtype=F64), torch.randn(6, 2, generator=g, dtype=F64)
        ),
    )
    delta_at = k(span.points, X).mean(1) - k(span.points, Xp).mean(1)
    pn = bd.projected_norm(span.coefs, delta_at[None, :])
    F = PolyKernel.features
    delta = F(X).mean(0) - F(Xp).mean(0)
    Vf = F(span.points).T @ span.coefs
    assert abs(float(pn[0]) - float((Vf.T @ delta).norm())) < 1e-10


def test_replica_radius_coverage_by_simulation():
    """||mu_hat_t - mu_{pi_t}|| <= b_t holds (conservatively) across repetitions (exact norms)."""
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    k = gl.AnisoRBF(np.ones(2), 1.0)
    L = torch.as_tensor(np.linalg.cholesky(C))
    J, beta, reps = 256, 0.2, 300
    b = bd.replica_radius(1.0, J, beta)
    g = torch.Generator().manual_seed(7)
    n2 = gl.mean_embedding_norm2(k, C)
    hits, errs = 0, []
    for _ in range(reps):
        X = torch.randn(J, 2, generator=g, dtype=F64) @ L.T
        e2 = float(
            k(X, X).mean()
            - 2 * torch.as_tensor(gl.mean_embedding(k, np.zeros(2), C, X.numpy())).mean()
            + n2
        )
        errs.append(math.sqrt(max(e2, 0.0)))
        hits += errs[-1] <= b
    assert hits / reps >= 1 - beta
    assert max(errs) < b  # the bounded-differences radius is very conservative


def test_assembly():
    H = bd.H_bound(torch.tensor([0.01], dtype=F64), 0.02, 0.03)
    U = bd.U_bound(0.8, torch.tensor([0.0025], dtype=F64), e=0.0)
    B = bd.combined_bound(H, U)
    assert abs(float(B) - math.sqrt(0.06**2 + (0.8 * 0.05) ** 2)) < 1e-15
    with pytest.raises(ValueError):
        bd.U_bound(0.8, torch.tensor([-0.1], dtype=F64))
    with pytest.raises(TypeError):
        bd.H_bound(torch.tensor([0.01]), 0.02, 0.03)  # float32 refused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_fp32_direct_terms_within_rigorous_bound():
    """Float32 entries with float64 sums satisfy the stated error bounds."""
    torch.backends.cuda.matmul.allow_tf32 = False
    k = gl.AnisoRBF(np.array([0.05, 0.03, 0.04, 0.02]), 3.0)
    g = torch.Generator(device="cuda").manual_seed(1)
    sd = torch.tensor([0.05, 0.03, 0.04, 0.02], dtype=F64, device="cuda")
    X = torch.randn(4096, 4, generator=g, dtype=F64, device="cuda") * sd + 0.01
    Y = torch.randn(16384, 4, generator=g, dtype=F64, device="cuda") * sd
    rs32, err = k.rowsum32(X[:1024], Y[:8192])
    rs64 = k(X[:1024], Y[:8192]).sum(1)
    assert float(((rs32 - rs64).abs() / 8192).max()) <= err and err < 1e-4
    n2, e_n2, cross, e_c = bd.fp32_vstat_and_cross(k, X, [1024, 4096], Y, chunk=1024)
    reps = bd.ReplicaSet(X, [1024, 4096], chunk=1024)
    n2_64 = reps.norm2(k)
    cross_64 = reps.cross(k, bd.TargetReference.sampled(Y, 1.0, 0.01))
    assert float((n2 - n2_64).abs().max()) <= e_n2
    assert float((cross - cross_64).abs().max()) <= e_c
