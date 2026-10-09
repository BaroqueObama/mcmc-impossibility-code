"""Tests for quadrature, constrained density construction, and residuals."""

import math

import mpmath
import numpy as np
import pytest
import torch
from scipy import integrate

from mcmc.paperexp import r1_suite as r1
from mcmc.paperexp.embeddings import rbf_mean_embedding
from mcmc.paperexp.kernels import ScaledMatern32, ScaledRBF
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.targets import make_target

F64 = torch.float64
SPEC = r1.QuadSpec(
    h_x=0.45, q_out=6, q_in=6, cap_out=0.5, cap_in=0.5, tail_mass=1e-15, r_max=40.0, q_min=4
)
LSPEC = r1.LandmarkSpec(h_L=0.4, eta_rel=1e-8, box_sd=9.0, r_box=30.0, x_cap=200.0)


def _gauss2():
    t = make_target("gauss", 2, cond=25.0)
    return t, ScaledRBF(t.coordinate_std() * np.array([1.0, 0.97]), 1.6)


# bivariate normal CDF
def _phi2_mp(h, k, rho):
    """P(X <= h, Y <= k) = int_{-inf}^h phi(x) Phi((k - rho x)/s) dx, with a breakpoint at the
    (possibly sharp) transition x = k/rho."""
    with mpmath.workdps(30):
        s = mpmath.sqrt(1 - mpmath.mpf(rho) ** 2)
        f = lambda x: mpmath.npdf(x) * mpmath.ncdf((k - rho * x) / s)  # noqa: E731
        pts = sorted({-40.0, h} | ({k / rho} if rho > 0 and k / rho < h else set()))
        return float(sum(mpmath.quad(f, [a, b]) for a, b in zip(pts[:-1], pts[1:])))


@pytest.mark.parametrize(
    "h,k,rho",
    [
        (0.3, -1.2, 0.5),
        (-2.0, -0.7, 0.9),
        (1.5, 2.5, 0.999),
        (0.0, 1.3, 0.7),
        (-0.4, 0.0, 0.2),
        (0.0, 0.0, 0.6),
        (-3.0, 4.0, 0.95),
        (2.0, -2.0, 0.9999),
        (0.7, 0.7, 0.0),
    ],
)
def test_phi2_matches_high_precision_quadrature(h, k, rho):
    sq = math.sqrt(1 - rho * rho)
    got = float(r1.phi2(h, k, rho, sq))
    assert abs(got - _phi2_mp(h, k, rho)) < 1e-14


def test_phi2_infinite_and_symmetry():
    rho = np.array(0.6)
    sq = np.sqrt(1 - rho**2)
    assert r1.phi2(-np.inf, 0.3, rho, sq) == 0.0
    assert r1.phi2(np.inf, np.inf, rho, sq) == 1.0
    assert abs(float(r1.phi2(np.inf, 0.3, rho, sq)) - 0.6179114221889527) < 1e-15
    h, k = np.array([0.4, -1.1]), np.array([-0.9, 0.2])
    assert np.allclose(r1.phi2(h, k, rho, sq), r1.phi2(k, h, rho, sq), atol=1e-15)


# Mehler closed form
def test_mehler_1d_against_brute_force_nystrom():
    sigma2 = 0.7
    lam, m, _ = r1.mehler_1d(sigma2, 30)
    assert abs(lam.sum() - 1.0) < 1e-13  # trace of the uncentered operator = E k(x, x)
    # ||mu||^2 = sum lam_n m_n^2 = E k(X, X') = 1/sqrt(1 + 2 sigma2)
    assert abs((lam * m * m).sum() - 1 / math.sqrt(1 + 2 * sigma2)) < 1e-13
    t, w = np.polynomial.hermite_e.hermegauss(200)
    x, w = math.sqrt(sigma2) * t, w / w.sum()
    Kw = np.sqrt(w)[:, None] * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2) * np.sqrt(w)[None, :]
    ev = np.sort(np.linalg.eigvalsh(Kw))[::-1]
    assert np.allclose(ev[:12], lam[:12], rtol=1e-10, atol=1e-15)
    psi = r1.mehler_psi_1d(sigma2, 12, x)
    G = (psi * w) @ psi.T
    assert np.abs(G - np.eye(13)).max() < 1e-10  # L2(pi)-orthonormal
    assert np.allclose(psi @ w, m[:13], atol=1e-12)


def test_gaussian_centered_spectrum_matches_finite_quadrature():
    t, k = _gauss2()
    g = r1.gaussian_centered_spectrum(t.covariance(), k)
    assert abs(g["mu_norm2_from_modes"] - rbf_mean_embedding(t, k).norm2()) < 1e-12
    # brute-force: weighted centered Gram on a large GH product rule in whitened coordinates
    A = t.rotation * np.sqrt(t.lam)[None, :]
    z, w = np.polynomial.hermite_e.hermegauss(70)
    w = w / w.sum()
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    X = np.stack([Z1.ravel(), Z2.ravel()], 1) @ A.T
    W = np.outer(w, w).ravel()
    K = k(torch.as_tensor(X), torch.as_tensor(X)).numpy()
    Kw = K @ W
    Kc = K - Kw[:, None] - Kw[None, :] + W @ Kw
    ev = np.sort(np.linalg.eigvalsh(np.sqrt(W)[:, None] * Kc * np.sqrt(W)[None, :]))[::-1]
    assert np.allclose(ev[:20], g["eig"][:20], rtol=1e-8)


# integrators
@pytest.mark.parametrize(
    "fam,params", [("gauss", {"cond": 25.0}), ("mixture", {"sep": 6.0, "w": 0.5})]
)
def test_latent_integrator_masses_and_mean(fam, params):
    t = make_target(fam, 2, **params)
    k = ScaledRBF(t.coordinate_std(), 1.5)
    integ = r1.make_integrator(t, k, 8, 6, SPEC)
    assert integ.info["quad_cell_mass_err_max"] < 1e-12
    assert abs(integ.masses.sum() - 1.0) < 1e-14
    assert integ.info["mass_exact_minus_uniform_max"] < 1e-13
    P = t.sample(20, torch_gen(3))
    exact = rbf_mean_embedding(t, k)(P)
    assert (integ.mean(P) - exact).abs().max() < 1e-11
    G = integ.cell_gram()
    assert abs(float(G.sum()) - rbf_mean_embedding(t, k).norm2()) < 1e-11


def test_student_t_integrator_against_core_embedding():
    t = make_target("student_t", 2, nu=3.0)
    k = ScaledRBF(np.array([1.14, 1.13]), 1.87)
    spec = r1.QuadSpec(
        h_x=0.5, q_out=5, q_in=5, cap_out=0.5, cap_in=0.5, tail_mass=1e-15, r_max=25.0, q_min=4
    )
    integ = r1.make_integrator(t, k, 6, 8, spec)
    # radial cells and sectors are exactly equal-mass; the outermost shell is truncated
    assert integ.info["mass_exact_minus_uniform_max"] < 1e-14
    assert integ.info["quad_cell_mass_err_max"] < 1e-10
    emb = rbf_mean_embedding(t, k)
    P = torch.cat([t.sample(20, torch_gen(4)), torch.tensor([[0.0, 0.0], [10.0, -3.0]], dtype=F64)])
    assert (integ.mean(P) - emb(P)).abs().max() < 1e-10
    # truncation at r_max = 25 drops int_{|x|>25} mu dpi ~ 137 r_max^{-8} ~ 9e-10 (kernel locality)
    deficit = emb.norm2() - float(integ.cell_gram().sum())
    assert 0.0 <= deficit < 3e-9, deficit


def _funnel(sig=1.0):
    t = make_target("funnel", 2, sigma_y=sig)
    return t, ScaledRBF(np.array([1.0, 0.8]), 1.7)


def test_funnel_integrator_mean_and_norm_vs_core():
    t, k = _funnel(3.0)
    integ = r1.make_integrator(t, k, 6, 6, SPEC)
    assert abs(integ.masses.sum() - 1.0) < 1e-12
    assert integ.info["mass_def_minus_uniform_max"] < 1e-12
    emb = rbf_mean_embedding(t, k)
    P = torch.cat([t.sample(30, torch_gen(5)), torch.tensor([[7.0, 30.0], [-5.0, 0.0]], dtype=F64)])
    assert (integ.mean(P) - emb(P)).abs().max() < 1e-12
    assert (integ.cell_integrals(P).sum(1) - emb(P)).abs().max() < 1e-12
    assert abs(float(integ.cell_gram().sum()) - emb.norm2()) < 1e-12


def test_funnel_gram_block_against_dblquad():
    """One G entry against adaptive 4-D-reduced quadrature (x1 inner integrals by quad)."""
    t, k = _funnel(1.0)
    integ = r1.make_integrator(t, k, 3, 3, SPEC)
    G = integ.cell_gram().numpy()
    by, b1 = integ.by, integ.b1
    j, jj = 1 * 3 + 2, 2 * 3 + 0  # (slab 1, x1-cell 2) x (slab 2, x1-cell 0)

    def cell_fn(j):
        k_, i_ = divmod(j, 3)
        return (
            integ.slab_edges[k_],
            integ.slab_edges[k_ + 1],
            integ.bounds[k_, i_],
            integ.bounds[k_, i_ + 1],
        )

    ya, yb, xa, xb = cell_fn(j)
    yc, yd, xc, xd = cell_fn(jj)

    def inner(y, yp):
        v, vp = math.exp(y), math.exp(yp)
        f = lambda x1, x2: (  # noqa: E731
            math.exp(-((x1 - x2) ** 2) / (2 * b1**2))
            * math.exp(-0.5 * x1 * x1 / v)
            / math.sqrt(2 * math.pi * v)
            * math.exp(-0.5 * x2 * x2 / vp)
            / math.sqrt(2 * math.pi * vp)
        )
        lo1, hi1 = max(xa, -12 * math.sqrt(v)), min(xb, 12 * math.sqrt(v))
        lo2, hi2 = max(xc, -12 * math.sqrt(vp)), min(xd, 12 * math.sqrt(vp))
        # dblquad integrates func(inner, outer): outer x1 in [lo1, hi1], inner x2 in [lo2, hi2]
        return integrate.dblquad(
            lambda x2, x1: f(x1, x2), lo1, hi1, lo2, hi2, epsabs=1e-13, epsrel=1e-11
        )[0]

    def outer(y, yp):
        p = math.exp(-0.5 * y * y) * math.exp(-0.5 * yp * yp) / (2 * math.pi)
        return p * math.exp(-((y - yp) ** 2) / (2 * by**2)) * inner(y, yp)

    ref = integrate.dblquad(
        lambda yp, y: outer(y, yp), ya, yb, max(yc, -8), min(yd, 8), epsabs=1e-12, epsrel=1e-8
    )[0]
    assert abs(G[j, jj] - ref) < 1e-9 * max(1.0, abs(ref)) + 1e-12, (G[j, jj], ref)


# spectrum vs Mehler, construction, sampling
@pytest.fixture(scope="module")
def gauss_setup():
    t, k = _gauss2()
    gi = r1.make_integrator(t, k, 1, 1, SPEC)
    Z, _ = r1.make_landmarks(t, k, gi, LSPEC)
    spec = r1.nystrom_spectrum(k, Z, gi, 1e-12)
    return t, k, spec


def test_quadrature_nystrom_matches_mehler(gauss_setup):
    t, k, spec = gauss_setup
    g = r1.gaussian_centered_spectrum(t.covariance(), k)
    rel = spec.lam[:33] / g["eig"][:33] - 1
    assert np.abs(rel).max() < 1e-7, rel
    assert (spec.lam[:33] <= g["eig"][:33] * (1 + 1e-10)).all()  # Rayleigh-Ritz lower bounds
    ang = r1.mehler_rkhs_angles(spec, g, k, [2, 4, 8, 16])
    for m, a in ang.items():
        assert a["max_angle_deg"] < 0.01, (m, a)


@pytest.mark.parametrize("m", [2, 8])
def test_construction_constraints_positivity_and_radius(gauss_setup, m):
    t, k, spec = gauss_setup
    integ = r1.make_integrator(t, k, 8, 8, SPEC)
    G = integ.cell_gram().numpy()
    rows = r1.moment_rows(spec, m, integ.cell_integrals(spec.Z))
    cs = r1.ConstructionSpec(R=0.5, n_dir=3, floor=0.01, n_combo=4)
    c = r1.construct(G, integ.masses, rows, cs)
    combo = c["combo"]
    assert combo["mmd2_construction"] >= c["mmd_construction"] ** 2 * (1 - 1e-12)
    assert np.abs(rows @ combo["a"]).max() < 1e-12 and combo["min_density"] >= cs.floor - 1e-14
    assert combo["R_realized"] <= cs.R * (1 + 1e-12)
    a = c["a"]
    scale = np.abs(rows).max() * np.abs(a).max() * a.shape[0]
    assert np.abs(rows @ a).max() < 1e-13 * scale  # exact moment equality in this representation
    assert abs(integ.masses @ a) < 1e-15 * a.shape[0]
    assert c["min_density"] >= cs.floor - 1e-14
    assert c["R_realized"] <= cs.R * (1 + 1e-12)
    # Piecewise-constant compression cannot exceed lambda_{m+1} beyond quadrature error.
    assert c["nu_top"][0] <= spec.lam[m] * (1 + 1e-6)
    # the chosen value is the Rayleigh quotient of the chosen vector
    assert abs(c["mmd_construction"] ** 2 - a @ G @ a) < 1e-15


def test_construct_default_is_full_eigensolve_subset_is_opt_in(gauss_setup):
    t, k, spec = gauss_setup
    integ = r1.make_integrator(t, k, 8, 8, SPEC)
    G = integ.cell_gram().numpy()
    rows = r1.moment_rows(spec, 4, integ.cell_integrals(spec.Z))
    cs = r1.ConstructionSpec(R=0.5, n_dir=3, floor=0.01)
    assert cs.eig_subset is False  # the synthetic suite as run (secondary search basis unchanged)
    full = r1.construct(G, integ.masses, rows, cs)
    sub = r1.construct(G, integ.masses, rows, r1.ConstructionSpec(0.5, 3, 0.01, eig_subset=True))
    assert np.allclose(full["nu_top"], sub["nu_top"], rtol=1e-10)
    assert abs(full["mmd_construction"] - sub["mmd_construction"]) < 1e-12


def test_sampling_from_construction_matches_cell_masses(gauss_setup):
    t, k, spec = gauss_setup
    integ = r1.make_integrator(t, k, 4, 4, SPEC)
    a = np.linspace(-0.9, 0.9, 16)
    a -= (integ.masses @ a) / integ.masses.sum()
    X = r1.sample_construction(t, integ, a, 400_000, torch_gen(11))
    freq = np.bincount(integ.cell_index(X.numpy()), minlength=16) / X.shape[0]
    p = integ.masses * (1 + a)
    z = (freq - p) / np.sqrt(p * (1 - p) / X.shape[0])
    assert np.abs(z).max() < 5.0


def test_matern_integrator_runs_and_mass(gauss_setup):
    t, _, _ = gauss_setup
    k = ScaledMatern32(t.coordinate_std(), 1.6)
    integ = r1.make_integrator(t, k, 4, 4, SPEC)
    G = integ.cell_gram().numpy()
    assert np.allclose(G, G.T)
    assert np.linalg.eigvalsh(G).min() > -1e-14


def test_lowrank_gram_within_certified_bound(gauss_setup):
    t, k, spec = gauss_setup
    integ = r1.make_integrator(t, k, 6, 6, SPEC)
    G = integ.cell_gram().numpy()
    Gl, s, info = integ.cell_gram_lowrank(spec.feats)
    Gl = Gl.numpy()
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=36) * 3
        diff = a @ G @ a - a @ Gl @ a
        assert -1e-14 <= diff <= (np.abs(a) @ s) ** 2 + 1e-14, (diff, (np.abs(a) @ s) ** 2)


# projected residuals
def test_projected_accumulator_matches_direct_u_statistics():
    t = make_target("gauss", 3, cond=4.0, rotation=np.eye(3))
    k = ScaledRBF(np.ones(3), 1.3)
    emb = rbf_mean_embedding(t, k)
    Y = t.sample(300, torch_gen(1))
    pca = r1.KernelPCA(k, Y, 6)
    assert pca.ortho_err < 1e-10 and pca.colsum_max < 1e-8
    Ec = emb(Y) @ pca.coef
    X = t.sample_init(240, 0.8, torch_gen(2))
    nb, M = 6, 5
    Xb = [X[i * 40 : (i + 1) * 40] for i in range(nb)]
    acc = r1.ProjectedAccumulator(nb, M)
    acc.add(k, Xb, [emb(x) for x in Xb], [pca.criteria(x, M) - Ec[None, :M] for x in Xb])
    est = acc.estimates(emb.norm2(), [2, 5])
    # direct: U-statistic of <zeta_i, zeta_j>, zeta = P(phi(X) - mu)
    J = X.shape[0]
    K = k(X, X)
    mu = emb(X)
    g = pca.criteria(X, M) - Ec[None, :M]
    for m in (2, 5):
        inner = K - mu[:, None] - mu[None, :] + emb.norm2() - g[:, :m] @ g[:, :m].T
        u = float((inner.sum() - torch.diagonal(inner).sum()) / (J * (J - 1)))
        assert abs(u - est[f"proj2_{m}"]["est"]) < 1e-13
        assert est[f"proj2_{m}"]["se"] > 0


def test_funnel_lattice_second_moment_matches_direct():
    t, k = _funnel(3.0)
    gi = r1.make_integrator(t, k, 1, 1, SPEC)
    Z, _ = r1.make_landmarks(t, k, gi, r1.LandmarkSpec(0.5, 1e-6, 5.0, 30.0, 60.0))
    Z = Z[::5]
    A, B = gi.second_moment(Z), gi.second_moment_direct(Z)
    assert float((A - B).abs().max()) < 1e-14


# Funnel x Matern-3/2 via the Gaussian scale mixture
def test_matern32_scale_mixture_identity_and_kernel():
    lam, w = r1.matern32_scale_mixture(0.35, -28.0, 0.3, 4.5)
    r = np.linspace(0.0, 30.0, 6001)
    assert np.abs(r1.mixture_kernel_values(lam, w, r) - r1.matern32(r)).max() < 1e-9
    km = ScaledMatern32(np.array([1.0, 0.8]), 1.7)
    kx = r1.ScaledGaussMixture(km.scales, km.ell, lam, w)
    X = torch.randn(200, 2, dtype=F64, generator=torch.Generator().manual_seed(3)) * 3
    assert float((kx(X, X[:50]) - km(X, X[:50])).abs().max()) < 1e-9
    # the taper at lam0 = 0.01 removes the narrow components: sup gap ~1.6e-3 at r = 0; rules
    # with different h agree spectrally
    lt, wt = r1.matern32_scale_mixture(0.05, math.log(0.01), 0.35, 5.0)
    gap = r1.matern32(r) - r1.mixture_kernel_values(lt, wt, r)
    assert 1.4e-3 < gap.max() < 1.7e-3 and gap.min() > -1e-9 and gap.argmax() == 0
    l4, w4 = r1.matern32_scale_mixture(0.4, math.log(0.01), 0.35, 4.5)
    l3, w3 = r1.matern32_scale_mixture(0.3, math.log(0.01), 0.35, 4.5, offset=0.15)
    ref = r1.mixture_kernel_values(lt, wt, r)
    assert np.abs(r1.mixture_kernel_values(l4, w4, r) - ref).max() < 5e-9
    assert np.abs(r1.mixture_kernel_values(l3, w3, r) - ref).max() < 5e-12


def test_funnel_mixture_single_component_equals_rbf_integrator():
    t, k = _funnel(3.0)
    mix = r1.ScaledGaussMixture(k.scales, k.ell, np.array([1.0]), np.array([1.0]))
    node = r1.NodeSpec(h_y=0.5, q_y=8, h_x=0.35, q_x=8, cap=0.4, margin_bw=14.0, s_cut=8.5)
    fm = r1.FunnelMixtureIntegrator(t, mix, 4, 4, SPEC, node)
    fi = r1.make_integrator(t, k, 4, 4, SPEC)
    assert np.array_equal(fm.masses, fi.masses)
    P = torch.cat([t.sample(20, torch_gen(5)), torch.tensor([[7.0, 30.0]], dtype=F64)])
    assert float((fm.cell_gram() - fi.cell_gram()).abs().max()) < 1e-15
    assert float((fm.cell_integrals(P) - fi.cell_integrals(P)).abs().max()) < 1e-15
    gi = r1.make_integrator(t, k, 1, 1, SPEC)
    Z, _ = r1.make_landmarks(t, k, gi, r1.LandmarkSpec(0.5, 1e-6, 5.0, 30.0, 60.0))
    Z = Z[::15]
    fm1 = r1.FunnelMixtureIntegrator(t, mix, 1, 1, SPEC, node)
    A, B = fm1.second_moment(Z), gi.second_moment(Z)
    assert float((A - B).abs().max()) < 1e-9 * float(B.abs().max())
    assert float((fm1.mean_nodes(Z) - gi.mean(Z)).abs().max()) < 1e-9


def test_funnel_mixture_pool_matches_serial():
    t, k = _funnel(3.0)
    lam, w = r1.matern32_scale_mixture(1.0, math.log(0.01), 0.35, 3.0)
    mix = r1.ScaledGaussMixture(k.scales, k.ell, lam, w)
    node = r1.NodeSpec(h_y=0.5, q_y=6, h_x=0.5, q_x=6, cap=0.5, margin_bw=14.0, s_cut=8.0)
    a = r1.FunnelMixtureIntegrator(t, mix, 3, 3, SPEC, node, workers=1).cell_gram()
    b = r1.FunnelMixtureIntegrator(t, mix, 3, 3, SPEC, node, workers=2).cell_gram()
    assert float((a - b).abs().max()) == 0.0


def test_tabulated_surrogate_matches_direct_mixture():
    from mcmc.paperexp.r1_surrogate import TabulatedRadialKernel

    lam, w = r1.matern32_scale_mixture(0.25, math.log(0.01), 0.35, 4.5)
    kt = TabulatedRadialKernel(np.array([1.0, 0.8]), 1.7, lam, w, 30.0, 300001)
    kx = r1.ScaledGaussMixture(kt.scales, kt.ell, lam, w)
    X = torch.randn(300, 2, dtype=F64, generator=torch.Generator().manual_seed(4)) * 4
    assert float((kt(X, X) - kx(X, X)).abs().max()) < 1e-14
    assert kt.max_abs_err_vs(200_000) < 1e-14
