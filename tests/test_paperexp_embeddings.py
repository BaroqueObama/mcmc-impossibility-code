"""Kernel means checked against independent quadrature and Monte Carlo."""

import math

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from mcmc.paperexp.embeddings import (
    QUAD_RTOL,
    default_probe_points,
    mc_mean_embedding,
    mc_mean_embedding_norm2,
    rbf_mean_embedding,
)
from mcmc.paperexp.kernels import ScaledMatern32, ScaledRBF
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.targets import (
    EVENT_VARIANT_TARGETS,
    PRIMARY_TARGETS,
    STRESS_TARGETS,
    make_target,
)

F64 = torch.float64
ALL = PRIMARY_TARGETS + STRESS_TARGETS + EVENT_VARIANT_TARGETS
IDS = [f"{f}-d{d}-{'-'.join(f'{k}{v}' for k, v in p.items())}" for f, d, p in ALL]


def _kernel(t, ell=1.5):
    scales = np.minimum(t.coordinate_std(), 5.0) * (1 + 0.1 * np.arange(t.d) / t.d)
    return ScaledRBF(scales, ell)


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_embedding_vs_monte_carlo(spec):
    f, d, p = spec
    t = make_target(f, d, **dict(p))
    k = _kernel(t)
    emb = rbf_mean_embedding(t, k)
    g = torch_gen(77)
    X = torch.cat([t.sample(6, g), t.sample_init(3, 0.8, g), torch.zeros(1, d, dtype=F64)])
    n = 400_000
    mc = mc_mean_embedding(k, t, X, n, g, chunk=1 << 18)
    z = ((emb(X) - mc["estimate"]) / mc["se"]).abs().max().item()
    mn = mc_mean_embedding_norm2(k, t, n, g, chunk=1 << 18)
    zn = abs(emb.norm2() - mn["estimate"]) / mn["se"]
    assert z < 4.5 and zn < 4.5, (z, zn)
    if emb.method == "quadrature":
        assert emb.info["rel_change_mu_last_doubling"] < QUAD_RTOL
        assert emb.info["rel_change_norm2_last_doubling"] < QUAD_RTOL


def test_gaussian_closed_form_vs_2d_numerical_integration():
    t = make_target("gauss", 2, cond=25.0)
    k = _kernel(t)
    emb = rbf_mean_embedding(t, k)
    C = t.covariance()
    rv = stats.multivariate_normal(np.zeros(2), C)
    lam = (k.scales * k.ell) ** 2
    for x in ([0.0, 0.0], [1.0, -2.0], [3.0, 0.5]):
        x = np.array(x)
        val, _ = integrate.dblquad(
            lambda b, a: (
                math.exp(-0.5 * ((x[0] - a) ** 2 / lam[0] + (x[1] - b) ** 2 / lam[1]))
                * rv.pdf([a, b])
            ),
            -30,
            30,
            -30,
            30,
            epsabs=1e-13,
            epsrel=1e-11,
        )
        assert abs(val - emb(torch.as_tensor(x[None], dtype=F64)).item()) < 1e-10


def test_mixture_closed_form_vs_2d_numerical_integration():
    t = make_target("mixture", 2, sep=6.0, w=0.5)
    k = _kernel(t)
    emb = rbf_mean_embedding(t, k)
    lam = (k.scales * k.ell) ** 2
    x = np.array([0.7, -1.1])

    def dens(b, a):
        return math.exp(t.log_prob(torch.tensor([[a, b]], dtype=F64)).item())

    val, _ = integrate.dblquad(
        lambda b, a: (
            math.exp(-0.5 * ((x[0] - a) ** 2 / lam[0] + (x[1] - b) ** 2 / lam[1])) * dens(b, a)
        ),
        -20,
        20,
        -20,
        20,
        epsabs=1e-13,
        epsrel=1e-11,
    )
    assert abs(val - emb(torch.as_tensor(x[None], dtype=F64)).item()) < 1e-10


@pytest.mark.parametrize("d,nu", [(10, 3.0), (2, 3.0), (10, 30.0)])
def test_student_t_trapezoid_vs_quadpack(d, nu):
    t = make_target("student_t", d, nu=nu)
    k = _kernel(t)
    emb = rbf_mean_embedding(t, k)
    lam = emb.lam
    for x in default_probe_points(t, seed=4)[:4]:

        def f(w):
            tau = nu / w
            return math.exp(
                -0.5 * np.log1p(tau / lam).sum() - 0.5 * (x**2 / (lam + tau)).sum()
            ) * stats.chi2.pdf(w, nu)

        ref, _ = integrate.quad(f, 0, np.inf, epsabs=0, epsrel=1e-13, limit=1000)
        ours = emb(torch.tensor(x[None], dtype=F64)).item()
        assert abs(ours - ref) / ref < 1e-10


def test_student_t_norm2_vs_nested_quadpack():
    t = make_target("student_t", 2, nu=3.0)
    k = _kernel(t)
    emb = rbf_mean_embedding(t, k)
    lam, nu = emb.lam, 3.0

    def g(w, wp):
        return (
            math.exp(-0.5 * np.log1p((nu / w + nu / wp) / lam).sum())
            * stats.chi2.pdf(w, nu)
            * stats.chi2.pdf(wp, nu)
        )

    ref, _ = integrate.quad(
        lambda w: integrate.quad(lambda wp: g(w, wp), 0, np.inf, epsabs=0, epsrel=1e-12, limit=500)[
            0
        ],
        0,
        np.inf,
        epsabs=0,
        epsrel=1e-11,
        limit=500,
    )
    assert abs(emb.norm2() - ref) / ref < 1e-9


@pytest.mark.parametrize("d,sy", [(10, 3.0), (2, 3.0), (10, 1.0)])
def test_funnel_trapezoid_vs_gauss_hermite(d, sy):
    t = make_target("funnel", d, sigma_y=sy)
    k = _kernel(t)
    a = rbf_mean_embedding(t, k, rule="trapezoid_y")
    b = rbf_mean_embedding(t, k, rule="gauss_hermite_y")
    X = torch.as_tensor(default_probe_points(t, seed=5))
    rel = ((a(X) - b(X)).abs() / b(X)).max().item()
    assert rel < 1e-11, rel
    assert abs(a.norm2() - b.norm2()) / b.norm2() < 1e-11


def test_matern_rejected_for_exact_embedding():
    t = make_target("gauss", 10, cond=25.0)
    with pytest.raises(TypeError):
        rbf_mean_embedding(t, ScaledMatern32(np.ones(10), 1.0))


def test_matern_monte_carlo_is_labelled_with_ci():
    t = make_target("gauss", 10, cond=25.0)
    k = ScaledMatern32(t.coordinate_std(), 2.0)
    out = mc_mean_embedding_norm2(k, t, 100_000, torch_gen(3))
    assert out["method"] == "monte_carlo"
    lo, hi = out["ci95_normal"]
    assert lo < out["estimate"] < hi and out["se"] > 0
