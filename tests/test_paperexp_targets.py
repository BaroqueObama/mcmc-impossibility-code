"""Tests for target densities, exact sampling, and random streams."""

import math

import mpmath
import numpy as np
import pytest
import torch
from scipy import integrate, stats

from mcmc.paperexp.rng import np_gen, seed_for, torch_gen
from mcmc.paperexp.targets import (
    EVENT_VARIANT_TARGETS,
    FROZEN_SUITE_PATH,
    PRIMARY_TARGETS,
    ROTATED_FAMILY_DIMS,
    STRESS_TARGETS,
    GaussTarget,
    haar_rotation,
    load_frozen_suite,
    load_rotation,
    make_target,
    rotation_seed,
    target_from_dict,
)

F64 = torch.float64
ALL_TARGETS = PRIMARY_TARGETS + STRESS_TARGETS + EVENT_VARIANT_TARGETS
IDS = [f"{f}-d{d}-{'-'.join(f'{k}{v}' for k, v in p.items())}" for f, d, p in ALL_TARGETS]


def _t(spec):
    f, d, p = spec
    return make_target(f, d, **dict(p))


# rng
def test_seed_for_deterministic_distinct_and_in_range():
    a = seed_for("gauss-d10-c25__rwm", "ens-replicas", 3)
    assert a == seed_for("gauss-d10-c25__rwm", "ens-replicas", 3)
    seeds = {seed_for(c, s, r) for c in ("a", "b") for s in ("x", "y") for r in range(50)}
    assert len(seeds) == 200
    assert all(0 <= s < 2**63 for s in seeds)
    with pytest.raises(ValueError):
        seed_for("a|b", "x", 0)
    with pytest.raises(ValueError):
        seed_for("a", "x", -1)
    g1, g2 = torch_gen(a), torch_gen(a)
    assert torch.equal(torch.randn(5, generator=g1), torch.randn(5, generator=g2))
    assert np.array_equal(np_gen(a).standard_normal(4), np_gen(a).standard_normal(4))


# construction
@pytest.mark.parametrize("family,d", ROTATED_FAMILY_DIMS)
def test_saved_rotation_matches_seeded_regeneration(family, d):
    saved = load_rotation(family, d)
    regen = haar_rotation(d, rotation_seed(family, d))
    assert np.abs(saved - regen).max() < 1e-12
    assert abs(np.linalg.det(saved) - 1.0) < 1e-12


@pytest.mark.parametrize("spec", ALL_TARGETS, ids=IDS)
def test_to_dict_roundtrip(spec):
    t = _t(spec)
    t2 = target_from_dict(t.to_dict())
    x = t.sample(64, torch_gen(1))
    assert torch.equal(t.log_prob(x), t2.log_prob(x))
    assert torch.equal(t.event_A(x), t2.event_A(x))


def test_gauss_spectrum_and_event_direction():
    t = make_target("gauss", 10, cond=25.0)
    lam = np.linalg.eigvalsh(t.covariance())
    assert abs(lam.max() / lam.min() - 25.0) < 1e-10
    assert abs(np.exp(np.log(lam).mean()) - 1.0) < 1e-12
    q = t.rotation[:, -1]
    assert abs(q @ t.covariance() @ q - lam.min()) < 1e-12


def test_event_thresholds_high_precision():
    t = make_target("mixture", 10, sep=6.0, w=0.1)
    with mpmath.workdps(40):
        c = mpmath.mpf(t.c_star)
        p = 0.1 * mpmath.ncdf(-(c + 3)) + 0.9 * mpmath.ncdf(-(c - 3))
        assert abs(float(p - mpmath.mpf("0.5"))) < 1e-14
    for d, nu in [(10, 3.0), (10, 5.0), (10, 30.0), (2, 3.0)]:
        t = make_target("student_t", d, nu=nu)
        assert abs(t.r_med**2 / d - stats.f.ppf(0.5, d, nu)) < 1e-10
        assert abs(stats.f.cdf(t.r_med**2 / d, d, nu) - 0.5) < 1e-13


# log densities
def _independent_logpdf(t, x: np.ndarray) -> np.ndarray:
    if t.family == "gauss":
        return stats.multivariate_normal(np.zeros(t.d), t.covariance()).logpdf(x)
    if t.family == "mixture":
        m = t.component_means()
        w = t.component_weights()
        comps = [stats.multivariate_normal(m[i], np.eye(t.d)).logpdf(x) for i in range(2)]
        return np.logaddexp(np.log(w[0]) + comps[0], np.log(w[1]) + comps[1])
    if t.family == "student_t":
        return stats.multivariate_t(np.zeros(t.d), np.eye(t.d), df=t.nu).logpdf(x)
    if t.family == "funnel":
        y = x[:, 0]
        out = stats.norm(0, t.sigma_y).logpdf(y)
        for i in range(1, t.d):
            out = out + stats.norm(0, np.exp(y / 2)).logpdf(x[:, i])
        return out
    raise AssertionError


@pytest.mark.parametrize("spec", ALL_TARGETS, ids=IDS)
def test_log_prob_matches_scipy(spec):
    t = _t(spec)
    x = t.sample(200, torch_gen(2))
    x = torch.cat([x, 1.5 * x[:50]])
    ours = t.log_prob(x).numpy()
    ref = _independent_logpdf(t, x.numpy())
    assert np.abs(ours - ref).max() < 1e-10 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize(
    "spec",
    [s for s in ALL_TARGETS if s[1] == 2],
    ids=[i for s, i in zip(ALL_TARGETS, IDS) if s[1] == 2],
)
def test_normalization_in_d2(spec):
    t = _t(spec)

    def dens(b, a):
        return math.exp(t.log_prob(torch.tensor([[a, b]], dtype=F64)).item())

    if t.family == "funnel":  # integrate x | y over a y-adapted window, y over the real line
        val, err = integrate.dblquad(
            dens,
            -40,
            40,
            lambda a: -60 * math.exp(a / 2),
            lambda a: 60 * math.exp(a / 2),
            epsabs=1e-12,
            epsrel=1e-10,
        )
    else:
        val, err = integrate.dblquad(
            dens, -np.inf, np.inf, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-10
        )
    assert abs(val - 1.0) < 1e-7, (val, err)


@pytest.mark.parametrize("spec", ALL_TARGETS, ids=IDS)
def test_score_matches_autograd(spec):
    t = _t(spec)
    x = t.sample(300, torch_gen(3))
    x = torch.cat([x, 0.3 * x[:50], 2.0 * x[:50]]).requires_grad_(True)
    (g_auto,) = torch.autograd.grad(t.log_prob(x).sum(), x)
    g = t.score(x.detach())
    scale = g_auto.abs().max().item()
    assert (g - g_auto).abs().max().item() < 1e-11 * max(1.0, scale)


def test_rejects_invalid_dtype_shape_and_radius():
    t = make_target("gauss", 10, cond=25.0)
    with pytest.raises(TypeError):
        t.log_prob(torch.zeros(3, 10, dtype=torch.float32))
    with pytest.raises(ValueError):
        t.log_prob(torch.zeros(3, 9, dtype=F64))
    with pytest.raises(ValueError):
        t.sample_init(10, 1.5, torch_gen(0))


# exact samplers
def _projection_cdfs(t):
    """List of (projection fn on numpy draws, exact 1-D CDF)."""
    out = []
    if t.family == "gauss":
        C = t.covariance()
        for v in (np.eye(t.d)[0], t.rotation[:, -1], np.ones(t.d) / math.sqrt(t.d)):
            sd = math.sqrt(v @ C @ v)
            out.append((lambda x, v=v: x @ v, stats.norm(0, sd).cdf))
    elif t.family == "mixture":
        m, w = t.component_means(), t.component_weights()
        for v in (t.rotation[:, 0], np.eye(t.d)[0], np.ones(t.d) / math.sqrt(t.d)):
            mu = m @ v
            out.append(
                (
                    lambda x, v=v: x @ v,
                    lambda z, mu=mu: (
                        w[0] * stats.norm.cdf(z - mu[0]) + w[1] * stats.norm.cdf(z - mu[1])
                    ),
                )
            )
    elif t.family == "student_t":
        v = np.ones(t.d) / math.sqrt(t.d)
        out.append((lambda x: x @ v, stats.t(t.nu).cdf))
        out.append((lambda x: (x * x).sum(1) / t.d, stats.f(t.d, t.nu).cdf))
    elif t.family == "funnel":
        out.append((lambda x: x[:, 0], stats.norm(0, t.sigma_y).cdf))
        for i in (1, t.d - 1):
            out.append((lambda x, i=i: x[:, i] * np.exp(-x[:, 0] / 2), stats.norm.cdf))
    return out


@pytest.mark.parametrize("spec", ALL_TARGETS, ids=IDS)
def test_exact_sampler_ks(spec):
    t = _t(spec)
    pvals = []
    for seed in range(3):
        x = t.sample(20000, torch_gen(seed_for(t.target_id, "test-ks", seed))).numpy()
        for proj, cdf in _projection_cdfs(t):
            pvals.append(stats.kstest(proj(x), cdf).pvalue)
    # Bonferroni at family level 1e-3 over all KS tests in this target
    assert min(pvals) > 1e-3 / len(pvals), pvals


@pytest.mark.parametrize("spec", ALL_TARGETS, ids=IDS)
def test_prob_A_is_half(spec):
    t = _t(spec)
    n = 400_000
    p = (
        t.event_A(t.sample(n, torch_gen(seed_for(t.target_id, "test-pA", 0))))
        .double()
        .mean()
        .item()
    )
    z = (p - 0.5) / math.sqrt(0.25 / n)
    assert abs(z) < 4.5, (p, z)


@pytest.mark.parametrize("eps", [0.8, 0.2])
@pytest.mark.parametrize(
    "spec",
    PRIMARY_TARGETS + [STRESS_TARGETS[6]] + EVENT_VARIANT_TARGETS,
    ids=IDS[:4] + [IDS[10]] + IDS[-len(EVENT_VARIANT_TARGETS) :],
)
def test_init_sampler_law(spec, eps):
    t = _t(spec)
    n = 200_000
    g = torch_gen(seed_for(t.target_id, f"test-init-{eps}", 0))
    x0 = t.sample_init(n, eps, g)
    assert x0.shape == (n, t.d)
    inA = t.event_A(x0)
    p = inA.double().mean().item()
    z = (p - (1 + eps) / 2) / math.sqrt(p * (1 - p) / n)
    assert abs(z) < 4.5, (p, z)
    # chi^2(pi0||pi) = E_pi0[h0] = eps (2 P_pi0(A) - 1) must equal eps^2
    chi2_hat = eps * (2 * p - 1)
    se = eps * 2 * math.sqrt(p * (1 - p) / n)
    assert abs(chi2_hat - eps**2) < 4.5 * se
    assert t.r_init(eps) == eps
    # conditional law on A and on A^c equals the target's conditional law (two-sample KS)
    xt = t.sample(n, g)
    inAt = t.event_A(xt)
    proj, _ = _projection_cdfs(t)[0]
    for a, b in ((x0[inA], xt[inAt]), (x0[~inA], xt[~inAt])):
        assert stats.ks_2samp(proj(a.numpy()), proj(b.numpy())).pvalue > 1e-4


def test_gauss_event_direction_option_is_backward_compatible():
    narrow = make_target("gauss", 10, cond=25.0)
    explicit = make_target("gauss", 10, cond=25.0, event_dir="narrow")
    wide = make_target("gauss", 10, cond=25.0, event_dir="wide")
    assert narrow.target_id == explicit.target_id == "gauss-d10-c25"
    assert wide.target_id == "gauss-d10-c25-ewide"
    x = narrow.sample(1000, torch_gen(4))
    q = narrow.rotation
    assert torch.equal(narrow.event_A(x), torch.as_tensor(x.numpy() @ q[:, -1] > 0))
    assert torch.equal(wide.event_A(x), torch.as_tensor(x.numpy() @ q[:, 0] > 0))
    # same distribution: identical log density, score and sampler stream
    assert torch.equal(narrow.log_prob(x), wide.log_prob(x))
    assert torch.equal(narrow.sample(5, torch_gen(9)), wide.sample(5, torch_gen(9)))
    lam = np.linalg.eigvalsh(wide.covariance())
    assert abs(q[:, 0] @ wide.covariance() @ q[:, 0] - lam.max()) < 1e-12
    assert target_from_dict(wide.to_dict()).target_id == "gauss-d10-c25-ewide"
    assert narrow.to_dict()["event_dir"] == "narrow"
    with pytest.raises(ValueError):
        make_target("gauss", 10, cond=25.0, event_dir="diagonal")
    with pytest.raises(TypeError):
        make_target("gauss", 10, cond=25.0, event_direction="wide")


def test_gauss_d1_supported_for_1d_witnesses():
    t = GaussTarget(1, 1.0, np.array([[1.0]]))
    x = torch.linspace(-3, 3, 7, dtype=F64)[:, None]
    assert torch.allclose(t.log_prob(x), torch.as_tensor(stats.norm.logpdf(x.numpy()[:, 0])))


# frozen suite
@pytest.mark.skipif(not FROZEN_SUITE_PATH.exists(), reason="frozen suite file is missing")
def test_frozen_suite_loads_every_config():
    suite = load_frozen_suite()
    assert len(suite.config_ids("primary")) == 8
    for cid in suite.config_ids():
        c = suite.config(cid)
        s = suite.sampler(cid)
        assert s.kind == c["sampler"] and s.h == c["h"]
        assert np.array_equal(s.scales, suite.scales(c["target_id"]))
        grid = suite.t_grid(cid)
        assert grid[0] == 0 and grid[-1] == c["t_max"] and grid == sorted(grid)
        assert abs(c["accept_verify"] - c["accept_target"]) <= 0.01
    for tid in suite.target_ids():
        t = suite.target(tid)
        assert t.target_id == tid
        for bw in ("ell", "ell_half", "ell_double"):
            suite.kernel(tid, "matern32", bw)
            if t.family in ("gauss", "mixture"):  # quadrature families are slower; checked in pilot
                suite.embedding(tid, bw)
