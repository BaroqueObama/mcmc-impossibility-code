"""Tests for circle confidence bounds and the knapsack formulation."""

import math

import numpy as np
import pytest
from scipy import integrate, optimize, special

from mcmc.experiments import exp_r3_anytime_usefulness as E
from mcmc.paperexp import circle as C

P, SIGMA, KAPPA0 = 0.1, 0.5, 1.0


def _conf(**over):
    conf = {
        "kappa0": KAPPA0,
        "init_symmetry": 1,
        "p": P,
        "sigma": SIGMA,
        "checkpoints": [0, 1, 4, 16, 64],
        "m_values": [2, 8],
        "replica_counts": [64, 256],
        "beta": 0.05,
        "kernel_sup": 1.0,
        "radius_multipliers": [1.0, 2.0],
        "tolerances": [0.05, 0.1],
        "n_fourier": 20,
        "series_terms": 40,
        "knapsack": True,
        "bernstein": True,
        "truth_rounding_rel": 1e-9,
        "event_rounding_abs": 1e-9,
        "design_disclosure": "test",
    }
    conf.update(over)
    return conf


def test_numpy_von_mises_sampler_moments():
    rng = np.random.default_rng(1)
    x = C.wrap(rng.vonmises(0.0, KAPPA0, size=1 << 18))
    for n in range(1, 5):
        se = math.sqrt(0.5 / x.shape[0])
        assert abs(np.cos(n * x).mean() - C.bessel_ratio(n, KAPPA0)) < 5 * se
        assert abs(np.sin(n * x).mean()) < 5 * se


def test_replica_radii():
    delta = 0.05 / 44
    j = np.array([1024, 65536])
    b = E.replica_radius_mcdiarmid(j, delta)
    np.testing.assert_allclose(b, (1 + np.sqrt(2 * np.log(880))) / np.sqrt(j))
    bb = E.replica_radius_bernstein(j, delta)
    assert np.all(bb < b)
    # Bernstein radius solves s^2 = 2 L (J + 2 s / 3) with s = J * radius
    s = bb * j
    big_l = np.log(2 / delta)
    np.testing.assert_allclose(s**2, 2 * big_l * (j + 2 * s / 3), rtol=1e-12)
    with pytest.raises(ValueError):
        E.replica_radius_mcdiarmid(10, 1.5)


@pytest.mark.parametrize("symmetry", [1, 3])
def test_event_error_matches_dense_computation(symmetry):
    """||mu_hat - mu_t||^2 via sorting + Fourier inner product equals a dense quadrature value."""
    conf = _conf(checkpoints=[0, 3], replica_counts=[50, 200], init_symmetry=symmetry, n_fourier=40)
    out = E.simulate_ensemble((0, 77, conf))
    truth = E.truth_table(conf)
    err_sq = out["vnorm"] - 2 * out["inner"] + truth["mmd_sq_partial"][:, None]
    # dense reference: rerun the identical random stream and evaluate everything by quadrature
    rng = np.random.default_rng(np.random.SeedSequence([77, 0]))
    x = C.sample_symmetric_von_mises(rng, 200, KAPPA0, symmetry)
    grid = C.grid_nodes(1 << 14)
    h0 = np.exp(KAPPA0 * np.cos(symmetry * grid)) / special.i0(KAPPA0) - 1.0
    kd = C.kernel_closed_form(2 * np.pi * np.arange(grid.shape[0]) / grid.shape[0])
    previous = 0
    for ti, t in enumerate(conf["checkpoints"]):
        x = C.advance_checkpoint(x, t - previous, rng, P, SIGMA)
        previous = t
        ht = C.propagate_density_quadrature(h0, t, P, SIGMA)
        mu_grid = np.fft.irfft(np.fft.rfft(kd) * np.fft.rfft(ht), n=grid.shape[0]) / grid.shape[0]
        mu_norm_sq = C.mmd_sq_quadrature(ht)
        for ji, size in enumerate(conf["replica_counts"]):
            xs = x[:size]
            mu_at = np.interp(xs, np.append(grid, np.pi), np.append(mu_grid, mu_grid[0]))
            dense = C.empirical_mean_embedding_sq_dense(xs) - 2 * mu_at.mean() + mu_norm_sq
            assert abs(dense - err_sq[ti, ji]) < 1e-6


def test_knapsack_matches_linprog_and_slsqp():
    rng = np.random.default_rng(12)
    for trial in range(40):
        m = int(rng.choice([2, 4, 8]))
        t = int(rng.integers(0, 60))
        axes = E.dynamic_axes(m + 2, t, P, SIGMA) if trial % 2 else E.static_axes(m + 2)
        radius = float(rng.uniform(0.2, 1.0))
        center = rng.normal(scale=0.5 * radius * axes[:m])
        half = rng.uniform(0.0, 0.5, size=m) * radius * axes[:m]
        lo, hi = center - half, center + half
        low, high = E._interval_square_range(lo, hi)
        if np.sum(low / axes[:m] ** 2) > radius**2:
            with pytest.raises(E.InfeasibleSourceIntervals):
                E.knapsack_sq(lo, hi, axes, m, radius)
            continue
        u2 = E.knapsack_sq(lo, hi, axes, m, radius)
        weight = 1 - axes[m] ** 2 / axes[:m] ** 2
        res = optimize.linprog(
            -weight,
            A_ub=(1 / axes[:m] ** 2)[None],
            b_ub=[radius**2],
            bounds=list(zip(low, high)),
            method="highs",
        )
        assert res.status == 0
        lp = axes[m] ** 2 * radius**2 - res.fun
        assert abs(u2 - lp) < 1e-12 * max(1.0, lp)
        # the LP optimum maps back to a feasible x attaining it (exact reformulation)
        y = res.x
        x_star = np.where(hi >= np.sqrt(y), np.sqrt(y), -np.sqrt(y))
        assert np.all(x_star >= lo - 1e-9) and np.all(x_star <= hi + 1e-9)
        assert np.sum(x_star**2 / axes[:m] ** 2) <= radius**2 * (1 + 1e-9)
        assert abs(axes[m] ** 2 * radius**2 + np.sum(weight * x_star**2) - u2) < 1e-10
        # a generic constrained optimiser in x-space, and random feasible points, never exceed it
        for _ in range(5):
            x0 = np.clip(x_star + rng.normal(scale=0.1 * half + 1e-12), lo, hi)
            sol = optimize.minimize(
                lambda x: -np.sum(weight * x**2),
                x0,
                method="SLSQP",
                bounds=list(zip(lo, hi)),
                constraints=[
                    {"type": "ineq", "fun": lambda x: radius**2 - np.sum(x**2 / axes[:m] ** 2)}
                ],
            )
            if radius**2 - np.sum(sol.x**2 / axes[:m] ** 2) > -1e-12:
                assert axes[m] ** 2 * radius**2 - sol.fun <= u2 + 1e-10
        pts = rng.uniform(lo, hi, size=(2000, m))
        ok = np.sum(pts**2 / axes[:m] ** 2, axis=1) <= radius**2
        if ok.any():
            assert (
                np.max(axes[m] ** 2 * radius**2 + np.sum(weight * pts[ok] ** 2, axis=1))
                <= u2 + 1e-12
            )


def test_knapsack_empty_set_raises_and_batch_reports_nan():
    axes = E.static_axes(4)
    lo = np.array([2.0, -0.1])  # x_1 >= 2 is far outside the radius-0.1 ellipsoid
    hi = np.array([3.0, 0.1])
    with pytest.raises(E.InfeasibleSourceIntervals):
        E.knapsack_sq(lo, hi, axes, 2, 0.1)
    u2, bad = E.knapsack_sq_batch(lo[None], hi[None], axes[None], 2, 0.1)
    assert bad[0] and np.isnan(u2[0])


def test_knapsack_is_valid_and_between_tail_and_no_observation_bounds():
    rng = np.random.default_rng(5)
    radius = 0.65
    for t in (0, 10, 200):
        axes = E.dynamic_axes(40, t, P, SIGMA)
        for m in (2, 8, 32):
            for _ in range(50):
                coeff = rng.normal(size=40)
                coeff *= rng.uniform(0, 1) * radius / np.linalg.norm(coeff)  # ||h_0|| <= R
                z = axes * coeff  # RKHS coordinates of delta_t (truncated at j = 40)
                truth_sq = float(np.sum(z**2))
                half = rng.uniform(0, 0.05, size=m)
                shift = rng.uniform(-1, 1, size=m) * half
                lo, hi = z[:m] + shift - half, z[:m] + shift + half
                u2 = E.knapsack_sq(lo, hi, axes, m, radius)
                assert u2 >= truth_sq - 1e-15
                assert u2 >= axes[m] ** 2 * radius**2 - 1e-15
                assert u2 <= axes[0] ** 2 * radius**2 + 1e-15


def test_bounds_order_and_shapes():
    conf = _conf()
    data = {k: [] for k in ("vnorm", "coords", "inner")}
    for i in range(3):
        out = E.simulate_ensemble((i, 99, conf))
        for k in data:
            data[k].append(out[k])
    data = {k: np.stack(v) for k, v in data.items()}
    bnd = E.compute_bounds(data, conf)
    for rname in ("mcdiarmid", "bernstein"):
        b3, b4, bg = (bnd["bounds"][f"{x}_{rname}"] for x in ("B3", "B4", "Bglob"))
        assert b3.shape == (3, 5, 2, 2, 2)
        assert np.all(b4 <= b3 + 1e-15)
        assert np.all(b4 <= bg + 1e-15)  # rho_{m/2+1} <= rho_1
        assert np.all(bg <= b3 + 1e-15)
    radius = math.sqrt(C.von_mises_radius_sq(KAPPA0))
    np.testing.assert_allclose(bnd["no_obs_static"][:, 0], radius * math.sqrt(3) / math.pi)
    # m = 2 static tail floor is R sqrt(lambda_3) = R sqrt(3)/(2 pi)
    np.testing.assert_allclose(bnd["tails"]["static"][:, 0], radius * math.sqrt(3) / (2 * math.pi))


def test_simulated_coordinates_match_direct_evaluation():
    conf = _conf(checkpoints=[0], replica_counts=[32, 128])
    out = E.simulate_ensemble((3, 5, conf))
    rng = np.random.default_rng(np.random.SeedSequence([5, 3]))
    x = C.wrap(rng.vonmises(0.0, KAPPA0, size=128))
    for j in (1, 2, 5, 16):
        direct = math.sqrt(C.eigenvalue(j)) * C.fourier_basis(x[:32], j).mean()
        assert abs(out["coords"][0, 0, j - 1] - direct) < 1e-13
    assert abs(out["vnorm"][0, 1] - C.empirical_mean_embedding_sq_dense(x)) < 1e-12


def test_replica_radius_coverage_toy_known_mean():
    """Known-mean toy case: frequency of ||mu_hat - mu|| <= radius at a loose level."""
    rng = np.random.default_rng(31)
    size, trials, delta = 64, 4000, 0.2
    mu_sq, _ = C.von_mises_true_mmd_sq(KAPPA0, 0, P, SIGMA)
    c0 = C.von_mises_embedding_coefficients(KAPPA0, 0, P, SIGMA, 20)
    b_mc = E.replica_radius_mcdiarmid(size, delta)
    b_bn = E.replica_radius_bernstein(size, delta)
    errs = np.empty(trials)
    for i in range(trials):
        x = C.wrap(rng.vonmises(0.0, KAPPA0, size=size))
        fm = C.prefix_fourier_means(x, 20, [size])[0]
        errs[i] = math.sqrt(max(C.empirical_mean_embedding_sq(x) - 2 * fm.real @ c0 + mu_sq, 0.0))
    assert np.mean(errs > b_mc) <= delta
    assert np.mean(errs > b_bn) <= delta
    # the radii are conservative; record that the realised error is well inside them
    assert np.quantile(errs, 1 - delta) < b_bn < b_mc


def test_first_crossing_and_decisions():
    b = np.array([[[0.3], [0.2], [0.04], [0.06]], [[0.3], [0.3], [0.3], [0.3]]])  # (E=2, T=4, 1)
    fc = E.first_crossing(b, 0.05)
    assert fc[0, 0] == 2 and fc[1, 0] == -1
    truth = {
        "mmd_lo": np.array([0.1, 0.1, 0.05, 0.01]),
        "mmd_hi": np.array([0.1, 0.1, 0.05, 0.01]) + 1e-12,
    }
    dec = E.coverage_decisions(b, truth)
    assert dec[0, 2, 0] == -1 and dec[0, 0, 0] == 1
    nan_b = b.copy()
    nan_b[0, 1, 0] = np.nan
    assert E.coverage_decisions(nan_b, truth)[0, 1, 0] == -2


def test_clopper_pearson_matches_scipy():
    from scipy.stats import binomtest

    for k, n in ((0, 20), (19, 20), (1000, 1000), (950, 1000)):
        ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        lo, hi = E.clopper_pearson(k, n)
        assert abs(lo - ci.low) < 1e-12 and abs(hi - ci.high) < 1e-12


def test_true_mmd_series_matches_numerical_integration_t0():
    i0 = special.i0(KAPPA0)
    h = lambda x: np.exp(KAPPA0 * np.cos(x)) / i0 - 1.0  # noqa: E731
    val, _ = integrate.dblquad(
        lambda y, x: C.kernel_closed_form(np.array(x - y)) * h(x) * h(y) / (4 * np.pi**2),
        -np.pi,
        np.pi,
        -np.pi,
        np.pi,
        epsabs=1e-12,
    )
    series, tail = C.von_mises_true_mmd_sq(KAPPA0, 0, P, SIGMA)
    assert abs(val - series) < 1e-8


def test_symmetric_von_mises_sampler_and_series():
    rng = np.random.default_rng(17)
    x = C.sample_symmetric_von_mises(rng, 1 << 18, KAPPA0, 3)
    se = math.sqrt(0.5 / x.shape[0])
    for n in range(1, 8):
        expected = C.bessel_ratio(n // 3, KAPPA0) if n % 3 == 0 else 0.0
        assert abs(np.cos(n * x).mean() - expected) < 5 * se
        assert abs(np.sin(n * x).mean()) < 5 * se
    # symmetry 1 consumes exactly the von Mises stream (reproduces the frozen main runs)
    a = C.sample_symmetric_von_mises(np.random.default_rng(3), 1000, KAPPA0, 1)
    b = C.wrap(np.random.default_rng(3).vonmises(0.0, KAPPA0, size=1000))
    np.testing.assert_array_equal(a, b)
    # same L2 radius for every symmetry; truth series vs quadrature-propagated density
    grid = C.grid_nodes(1 << 13)
    h0 = np.exp(KAPPA0 * np.cos(3 * grid)) / special.i0(KAPPA0) - 1.0
    assert abs(np.mean(h0**2) - C.von_mises_radius_sq(KAPPA0)) < 1e-12
    for t in (0, 5, 40):
        ht = C.propagate_density_quadrature(h0, t, P, SIGMA)
        series, tail = C.von_mises_true_mmd_sq(KAPPA0, t, P, SIGMA, symmetry=3)
        assert tail < 1e-60
        assert abs(C.mmd_sq_quadrature(ht) / series - 1.0) < 1e-4
        kd = C.kernel_closed_form(2 * np.pi * np.arange(grid.shape[0]) / grid.shape[0])
        mu_quad = np.fft.irfft(np.fft.rfft(kd) * np.fft.rfft(ht), n=grid.shape[0]) / grid.shape[0]
        c = C.von_mises_embedding_coefficients(KAPPA0, t, P, SIGMA, 40, symmetry=3)
        assert np.all(c[np.arange(1, 41) % 3 != 0] == 0.0)
        mu_series = (c[:, None] * np.cos(np.outer(np.arange(1, 41), grid))).sum(axis=0)
        assert np.max(np.abs(mu_quad - mu_series)) < 1e-6
    assert C.von_mises_embedding_sup_tail(KAPPA0, 40, 3) < 1e-17


def _mp_true_mmd_sq(t, symmetry, digits=50, terms=60):
    import mpmath as mp

    mp.mp.dps = digits
    kappa, p, sigma = mp.mpf(KAPPA0), mp.mpf(P), mp.mpf(SIGMA)  # exact float64 parameter values
    i0 = mp.besseli(0, kappa)
    total = mp.mpf(0)
    for k in range(1, terms + 1):
        f = symmetry * k
        rho = 1 - p + p * mp.e ** (-(sigma**2) * f * f / 2)
        total += 6 / (mp.pi**2 * f * f) * (mp.besseli(k, kappa) / i0) ** 2 * rho ** (2 * t)
    return total


@pytest.mark.parametrize("symmetry", [1, 3])
def test_truth_enclosure_contains_high_precision_value(symmetry):
    """The outward-rounded truth enclosure contains a 50-digit evaluation at every checkpoint; the
    float64 evaluation error is far below the declared relative margin."""
    conf = _conf(checkpoints=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], init_symmetry=symmetry)
    truth = E.truth_table(conf)
    worst = 0.0
    for i, t in enumerate(conf["checkpoints"]):
        exact = _mp_true_mmd_sq(t, symmetry)
        worst = max(worst, float(abs(truth["mmd_sq_partial"][i] - exact) / exact))
        exact_mmd = float(exact.sqrt()) if hasattr(exact, "sqrt") else math.sqrt(float(exact))
        assert truth["mmd_lo"][i] <= exact_mmd <= truth["mmd_hi"][i]
    assert worst < 1e-11 < conf["truth_rounding_rel"]


def test_event_error_rounding_below_declared_margin():
    """float64 ||mu_hat - mu_t||^2 vs an extended-precision recomputation of every term."""
    conf = _conf(checkpoints=[0, 16, 512], replica_counts=[1024, 65536], n_fourier=20)
    truth = E.truth_table(conf)
    for ens in (0, 1):
        out = E.simulate_ensemble((ens, 2024, conf))
        rng = np.random.default_rng(np.random.SeedSequence([2024, ens]))
        x = C.sample_symmetric_von_mises(rng, 65536, KAPPA0, 1)
        previous = 0
        for ti, t in enumerate(conf["checkpoints"]):
            x = C.advance_checkpoint(x, t - previous, rng, P, SIGMA)
            previous = t
            for ji, size in enumerate(conf["replica_counts"]):
                xs = np.sort(x[:size]).astype(np.longdouble)
                n = size
                idx = np.arange(n, dtype=np.longdouble)
                pi = np.longdouble(np.pi)
                vnorm = (
                    1
                    - 3 * (2 * np.dot(2 * idx - n + 1, xs)) / (pi * n * n)
                    + 1.5 * (2 * n * np.dot(xs, xs) - 2 * np.sum(xs) ** 2) / (pi * pi * n * n)
                )
                c = C.von_mises_embedding_coefficients(KAPPA0, t, P, SIGMA, 20)
                inner = sum(
                    np.longdouble(c[k]) * np.mean(np.cos((k + 1) * x[:size].astype(np.longdouble)))
                    for k in range(20)
                )
                exact = float(vnorm - 2 * inner) + float(_mp_true_mmd_sq(t, 1))
                computed = (
                    out["vnorm"][ti, ji] - 2 * out["inner"][ti, ji] + truth["mmd_sq_partial"][ti]
                )
                assert abs(computed - exact) < 1e-10 < conf["event_rounding_abs"]
