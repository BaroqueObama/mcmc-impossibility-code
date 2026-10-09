"""Tests for RWM and MALA transitions, stationarity, and detailed balance."""

import math
import os

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.samplers import branch, make_sampler
from mcmc.paperexp.targets import PRIMARY_TARGETS, GaussTarget, make_target

F64 = torch.float64
CUDA = os.environ.get("MCMC_TEST_CUDA") == "1" and torch.cuda.is_available()
KINDS = ["rwm", "mala"]
PIDS = [f"{f}-d{d}" for f, d, _ in PRIMARY_TARGETS]
H = {"gauss": 0.6, "mixture": 0.5, "student_t": 0.5, "funnel": 0.4}


def _t(spec):
    f, d, p = spec
    return make_target(f, d, **dict(p))


# Independent scalar densities and gradients.
def ref_logp_grad(t, x):
    x = np.asarray(x, dtype=np.float64)
    if t.family == "gauss":
        P = np.linalg.inv(t.covariance())
        return stats.multivariate_normal(np.zeros(t.d), t.covariance()).logpdf(x), -P @ x
    if t.family == "mixture":
        m, w = t.component_means(), t.component_weights()
        lps = np.array(
            [
                math.log(w[i]) + stats.multivariate_normal(m[i], np.eye(t.d)).logpdf(x)
                for i in range(2)
            ]
        )
        lp = np.logaddexp(lps[0], lps[1])
        r = np.exp(lps - lp)
        return lp, -(x - (r[0] * m[0] + r[1] * m[1]))
    if t.family == "student_t":
        lp = stats.multivariate_t(np.zeros(t.d), np.eye(t.d), df=t.nu).logpdf(x)
        return lp, -(t.nu + t.d) * x / (t.nu + x @ x)
    if t.family == "funnel":
        y, u = x[0], x[1:]
        lp = stats.norm(0, t.sigma_y).logpdf(y) + stats.norm(0, math.exp(y / 2)).logpdf(u).sum()
        gy = -y / t.sigma_y**2 - 0.5 * (t.d - 1) + 0.5 * math.exp(-y) * (u @ u)
        return lp, np.concatenate([[gy], -math.exp(-y) * u])
    raise AssertionError


def ref_step(kind, t, h, s, x, xi, u):
    lpx, gx = ref_logp_grad(t, x)
    if kind == "rwm":
        y = x + h * s * xi
        la = ref_logp_grad(t, y)[0] - lpx
    else:
        mx = x + 0.5 * h * h * s * s * gx
        y = mx + h * s * xi
        lpy, gy = ref_logp_grad(t, y)
        my = y + 0.5 * h * h * s * s * gy
        lq_xy = -0.5 * np.sum(((x - my) / (h * s)) ** 2)
        lq_yx = -0.5 * np.sum(((y - mx) / (h * s)) ** 2)
        la = lpy - lpx + lq_xy - lq_yx
    acc = math.log(u) < la
    return (y if acc else x), acc


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spec", PRIMARY_TARGETS, ids=PIDS)
def test_scalar_reference_reproduces_batched_step(spec, kind):
    t = _t(spec)
    s = t.coordinate_std() ** 0.5 if t.family == "funnel" else t.coordinate_std()
    smp = make_sampler(kind, t, H[t.family], s)
    g = torch_gen(21)
    x = t.sample(150, g)
    xi, u = smp.draw_noise(150, g)
    x_new, acc = smp.step_given(x, xi, u)
    n_acc = 0
    for i in range(150):
        y, a = ref_step(kind, t, smp.h, s, x[i].numpy(), xi[i].numpy(), u[i].item())
        assert a == bool(acc[i])
        assert np.abs(y - x_new[i].numpy()).max() < 1e-12 * max(1.0, np.abs(y).max())
        n_acc += a
    assert 0 < n_acc < 150  # both branches exercised


@pytest.mark.parametrize("kind", KINDS)
def test_run_equals_iterated_steps_and_counts(kind):
    t = _t(PRIMARY_TARGETS[3])
    smp = make_sampler(kind, t, 0.4, t.coordinate_std() ** 0.5)
    x0 = t.sample(64, torch_gen(1))
    out = smp.run(x0, [0, 1, 2, 4, 8], torch_gen(99))
    assert torch.equal(out["states"][0], x0)
    assert out["counters"]["logprob_evals"] == 64 * 9
    assert out["counters"]["grad_evals"] == (64 * 9 if kind == "mala" else 0)
    assert out["counters"]["chain_steps"] == 64 * 8
    g = torch_gen(99)
    x = x0.clone()
    for step in range(1, 9):
        x, _ = smp.step(x, g)
        if step in (1, 2, 4, 8):
            assert torch.equal(x, out["states"][step])
    assert set(out["accept_by_interval"]) == {1, 2, 4, 8}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spec", PRIMARY_TARGETS, ids=PIDS)
def test_detailed_balance_exact_density_random_pairs(spec, kind):
    t = _t(spec)
    smp = make_sampler(kind, t, H[t.family], t.coordinate_std() ** 0.5)
    g = torch_gen(4)
    x = t.sample(500, g)
    y = x + 0.3 * torch.randn(x.shape, generator=g, dtype=F64)
    fwd = t.log_prob(x) + smp.log_transition_density_offdiag(x, y)
    bwd = t.log_prob(y) + smp.log_transition_density_offdiag(y, x)
    assert ((fwd - bwd).abs() / fwd.abs().clamp_min(1.0)).max().item() < 1e-12


@pytest.mark.parametrize("kind", KINDS)
def test_detailed_balance_1d_gaussian_grid(kind):
    t = GaussTarget(1, 1.0, np.array([[1.0]]))
    smp = make_sampler(kind, t, 1.3, np.array([1.0]))
    grid = torch.linspace(-4, 4, 161, dtype=F64)
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    flux = (t.log_prob(x) + smp.log_transition_density_offdiag(x, y)).reshape(161, 161)
    off = ~torch.eye(161, dtype=torch.bool)
    assert (flux - flux.T)[off].abs().max().item() < 1e-12


@pytest.mark.parametrize("kind", KINDS)
def test_one_step_law_1d_matches_exact_transition(kind):
    """Empirical one-step law from x0 vs q(y|x0) alpha(x0,y) + r(x0) delta_x0 (chi^2 test)."""
    t = GaussTarget(1, 1.0, np.array([[1.0]]))
    h = 1.6
    smp = make_sampler(kind, t, h, np.array([1.0]))
    x0v = 0.7
    J = 400_000
    x0 = torch.full((J, 1), x0v, dtype=F64)
    x1, acc = smp.step(x0, torch_gen(8))

    def dens(yv):
        return math.exp(
            smp.log_transition_density_offdiag(
                torch.tensor([[x0v]], dtype=F64), torch.tensor([[yv]], dtype=F64)
            ).item()
        )

    p_move, _ = integrate.quad(dens, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12)
    p_hat = acc.double().mean().item()
    assert abs(p_hat - p_move) < 4.5 * math.sqrt(p_move * (1 - p_move) / J)
    edges = np.linspace(-4, 5, 31)
    ys = x1[acc, 0].numpy()
    counts, _ = np.histogram(ys, edges)
    expected = np.array(
        [integrate.quad(dens, a, b, epsabs=1e-13)[0] for a, b in zip(edges[:-1], edges[1:])]
    )
    keep = expected * J > 20
    chi2 = (((counts[keep] - J * expected[keep]) ** 2) / (J * expected[keep])).sum()
    pval = stats.chi2.sf(chi2, keep.sum())
    assert pval > 1e-4, pval


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cond", [25.0, 100.0])
def test_stationarity_gaussian_moments(kind, cond):
    t = make_target("gauss", 10, cond=cond)
    smp = make_sampler(kind, t, 0.5, t.coordinate_std())
    J = 200_000
    g = torch_gen(31)
    out = smp.run(t.sample(J, g), [10], g)
    x = out["states"][10]
    C = torch.as_tensor(t.covariance())
    sd = torch.sqrt(torch.diag(C))
    z_mean = (x.mean(0) / (sd / math.sqrt(J))).abs().max().item()
    emp = torch.cov(x.T)
    se = torch.sqrt((C**2 + torch.outer(torch.diag(C), torch.diag(C))) / J)
    z_cov = ((emp - C) / se).abs().max().item()
    assert z_mean < 4.5 and z_cov < 5.0, (z_mean, z_cov)


def test_branch_rows_are_independent_copies():
    x = torch.arange(6, dtype=F64).reshape(3, 2)
    b = branch(x, 4)
    assert b.shape == (12, 2) and torch.equal(b[4:8], x[1].expand(4, 2))


def test_bad_density_raises():
    class Broken(GaussTarget):
        def log_prob(self, x):
            lp = super().log_prob(x)
            return torch.where(x[:, 0] > 1.0, torch.full_like(lp, float("nan")), lp)

    t = Broken(1, 1.0, np.array([[1.0]]))
    smp = make_sampler("rwm", t, 2.0, np.array([1.0]))
    with pytest.raises(FloatingPointError):
        smp.run(torch.zeros(2000, 1, dtype=F64), [5], torch_gen(0))


@pytest.mark.skipif(not CUDA, reason="set MCMC_TEST_CUDA=1 on a CUDA machine")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spec", PRIMARY_TARGETS, ids=PIDS)
def test_cuda_matches_cpu_on_identical_noise(spec, kind):
    t = _t(spec)
    smp = make_sampler(kind, t, H[t.family], t.coordinate_std() ** 0.5)
    g = torch_gen(5)
    x = t.sample(4096, g)
    xs, xc = x.clone(), x.cuda()
    for _ in range(20):
        xi, u = smp.draw_noise(4096, g)
        xs, a_cpu = smp.step_given(xs, xi, u)
        xc, a_gpu = smp.step_given(xc, xi.cuda(), u.cuda())
        assert torch.equal(a_cpu, a_gpu.cpu())
    assert (xs - xc.cpu()).abs().max().item() < 1e-10 * max(1.0, xs.abs().max().item())
    out = smp.run(t.sample(1024, torch_gen(1, "cuda")), [0, 4], torch_gen(2, "cuda"))
    assert out["states"][4].device.type == "cuda"
