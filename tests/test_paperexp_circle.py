"""Tests for circle kernels, Fourier expansions, and density constructions."""

import math

import numpy as np
import pytest
from scipy import integrate, special, stats

from mcmc.paperexp import circle as C

P, SIGMA = 0.1, 0.5


def test_closed_form_kernel_matches_fourier_series():
    d = np.linspace(-2 * np.pi, 2 * np.pi, 4001)
    series, tail = C.kernel_series(d, 4000)
    closed = C.kernel_closed_form(d)
    assert np.max(np.abs(closed - series)) <= tail
    assert tail < 2e-4
    assert C.kernel_closed_form(np.array([0.0]))[0] == 1.0
    # consistency with periodicity: k(d) = k(d - 2 pi) on [0, 2 pi]
    dd = np.linspace(0.0, 2 * np.pi, 777)
    np.testing.assert_allclose(
        C.kernel_closed_form(dd), C.kernel_closed_form(dd - 2 * np.pi), atol=1e-14
    )
    with pytest.raises(ValueError):
        C.kernel_closed_form(np.array([2 * np.pi + 1e-6]))


def test_eigenvalues_match_discretised_integral_operator():
    n_nodes = 2048
    x = C.grid_nodes(n_nodes)
    gram = C.kernel_closed_form(x[:, None] - x[None, :]) / n_nodes
    evals = np.sort(np.linalg.eigvalsh(gram))[::-1][:24]
    expected = C.eigenvalue(np.arange(1, 25))
    np.testing.assert_allclose(evals, expected, rtol=2e-3)
    # pairs are exact multiplicity-two
    assert np.all(expected[0::2] == expected[1::2])
    # target mean embedding is zero: the integral of k(x - y) over y vanishes; the trapezoid
    # value is the aliasing sum sum_{l != 0} w_{lN}/2 = 1/N^2 exactly (kinked kernel, O(N^-2)).
    assert np.max(np.abs(gram.sum(axis=1) - 1.0 / n_nodes**2)) < 1e-14


def test_fourier_basis_orthonormal():
    x = C.grid_nodes(4096)
    basis = np.stack([C.fourier_basis(x, j) for j in range(1, 13)])
    gram = basis @ basis.T / x.shape[0]
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-12)


def test_fft_mmd_matches_series_for_random_trig_polynomials():
    rng = np.random.default_rng(11)
    x = C.grid_nodes(1 << 14)
    for _ in range(5):
        a = rng.normal(scale=0.05, size=10)
        b = rng.normal(scale=0.05, size=10)
        n = np.arange(1, 11)
        h = (a[:, None] * np.cos(np.outer(n, x)) + b[:, None] * np.sin(np.outer(n, x))).sum(axis=0)
        series = float(np.sum(C.frequency_weight(n) * (a**2 + b**2) / 4.0))
        quad = C.mmd_sq_quadrature(h)
        assert abs(quad / series - 1.0) < 1e-4


def test_quadrature_propagation_matches_analytic_rho():
    x = C.grid_nodes(2048)
    for freq in (1, 3, 9):
        h0 = 0.3 * np.sqrt(2) * np.cos(freq * x)
        h10 = C.propagate_density_quadrature(h0, 10, P, SIGMA)
        coeff = np.mean(h10 * np.sqrt(2) * np.cos(freq * x))
        assert abs(coeff - 0.3 * C.lazy_rho(freq, P, SIGMA) ** 10) < 1e-12
        assert abs(C.integrate_periodic(h10)) < 1e-14


def test_wrapped_gaussian_density_normalised():
    x = C.grid_nodes(4096)
    g = C.wrapped_gaussian_density(x, SIGMA)
    assert abs(g.mean() * 2 * np.pi - 1.0) < 1e-12
    # Fourier coefficient of the wrapped Gaussian is exp(-sigma^2 n^2/2)
    for n in (1, 2, 5):
        assert abs(np.mean(g * np.cos(n * x)) * 2 * np.pi - np.exp(-0.5 * SIGMA**2 * n**2)) < 1e-12


def test_sorting_formula_matches_dense():
    rng = np.random.default_rng(3)
    for size in (2, 3, 17, 256, 1000):
        x = rng.uniform(-np.pi, np.pi, size)
        assert (
            abs(C.empirical_mean_embedding_sq(x) - C.empirical_mean_embedding_sq_dense(x)) < 1e-12
        )
    # ties and the boundary angle -pi
    x = np.array([-np.pi, -np.pi, 0.0, 0.0, 1.0, np.nextafter(np.pi, 0.0)])
    assert abs(C.empirical_mean_embedding_sq(x) - C.empirical_mean_embedding_sq_dense(x)) < 1e-12
    with pytest.raises(ValueError):
        C.empirical_mean_embedding_sq(np.array([0.0, np.pi]))


def test_sorting_formula_precision_large_sample():
    rng = np.random.default_rng(5)
    x = rng.uniform(-np.pi, np.pi, 1 << 18)
    fast = C.empirical_mean_embedding_sq(x)
    y = np.sort(x).astype(np.longdouble)
    n = y.shape[0]
    idx = np.arange(n, dtype=np.longdouble)
    pi = np.longdouble(np.pi)
    pair_abs = 2 * np.sum((2 * idx - n + 1) * y)
    pair_sq = 2 * n * np.sum(y * y) - 2 * np.sum(y) ** 2
    slow = 1 - 3 * pair_abs / (pi * n * n) + 1.5 * pair_sq / (pi * pi * n * n)
    assert abs(fast - float(slow)) < 1e-12


def test_unbiased_mmd_is_unbiased():
    rng = np.random.default_rng(8)
    amp, freq, size, reps = 0.6, 2, 400, 3000
    truth = C.frequency_weight(freq) * amp**2 / 4.0
    est = np.array(
        [C.unbiased_mmd_sq(C.sample_cosine_law(rng, size, freq, amp)) for _ in range(reps)]
    )
    z = (est.mean() - truth) / (est.std(ddof=1) / math.sqrt(reps))
    assert abs(z) < 4.0


def test_cosine_law_sampler_moments():
    rng = np.random.default_rng(9)
    x = C.sample_cosine_law(rng, 1 << 18, 3, 0.5)
    for n in (1, 2, 3, 4):
        expected = 0.25 if n == 3 else 0.0
        se = math.sqrt(0.5 / x.shape[0])
        assert abs(np.cos(n * x).mean() - expected) < 5 * se
        assert abs(np.sin(n * x).mean()) < 5 * se
    with pytest.raises(ValueError):
        C.sample_cosine_law(rng, 10, 1, 1.0)


def test_checkpoint_simulator_matches_explicit_steps():
    rng_a = np.random.default_rng(21)
    rng_b = np.random.default_rng(22)
    size, delta, freq, amp = 1 << 15, 25, 1, 0.8
    xa = C.advance_checkpoint(C.sample_cosine_law(rng_a, size, freq, amp), delta, rng_a, P, SIGMA)
    xb = C.simulate_explicit(C.sample_cosine_law(rng_b, size, freq, amp), delta, rng_b, P, SIGMA)
    assert stats.ks_2samp(xa, xb).pvalue > 1e-3
    expected = 0.5 * amp * C.lazy_rho(freq, P, SIGMA) ** delta
    se = math.sqrt(0.5 / size)
    for x in (xa, xb):
        assert abs(np.cos(freq * x).mean() - expected) < 5 * se
        assert abs(np.cos(2 * freq * x).mean()) < 5 * se
    # zero-gap checkpoint is the identity
    np.testing.assert_array_equal(C.advance_checkpoint(xa, 0, rng_a, P, SIGMA), xa)


def test_wrap_range():
    x = np.array([-np.pi, np.pi, 3 * np.pi, -3 * np.pi - 1e-17, -1e-300, 7.0])
    y = C.wrap(x)
    assert (y >= -np.pi).all() and (y < np.pi).all()
    np.testing.assert_allclose(np.cos(y), np.cos(x), atol=1e-12)


def test_prefix_fourier_means_match_direct():
    rng = np.random.default_rng(4)
    x = rng.uniform(-np.pi, np.pi, 4096)
    prefixes = [64, 256, 1024, 4096]
    fm = C.prefix_fourier_means(x, 17, prefixes)
    for i, pfx in enumerate(prefixes):
        for n in range(1, 18):
            assert abs(fm[i, n - 1] - np.exp(1j * n * x[:pfx]).mean()) < 1e-13
    coords = C.embedding_coordinates(fm)
    for j in (1, 2, 7, 34):
        direct = math.sqrt(C.eigenvalue(j)) * C.fourier_basis(x[:1024], j).mean()
        assert abs(coords[2, j - 1] - direct) < 1e-13


def test_von_mises_fourier_coefficients_match_integration():
    for kappa in (0.5, 1.0, 3.0):
        i0 = special.i0(kappa)
        for n in range(0, 8):
            val, err = integrate.quad(
                lambda x: np.exp(kappa * np.cos(x)) * np.cos(n * x),
                0,
                np.pi,
                epsabs=1e-13,
                limit=200,
            )
            assert abs(val / np.pi / i0 - C.bessel_ratio(n, kappa)) < 1e-12
        assert np.all(
            C.bessel_ratio_bound(np.arange(1, 30), kappa) >= C.bessel_ratio(np.arange(1, 30), kappa)
        )


def test_von_mises_radius_matches_integration():
    for kappa in (0.5, 1.0, 2.0):
        i0 = special.i0(kappa)
        val, _ = integrate.quad(
            lambda x: (np.exp(kappa * np.cos(x)) / i0 - 1.0) ** 2, -np.pi, np.pi, epsabs=1e-14
        )
        assert abs(val / (2 * np.pi) - C.von_mises_radius_sq(kappa)) < 1e-12


def test_von_mises_true_mmd_matches_propagated_quadrature():
    kappa = 1.0
    x = C.grid_nodes(1 << 13)
    h0 = np.exp(kappa * np.cos(x)) / special.i0(kappa) - 1.0
    for t in (0, 3, 10, 40):
        ht = C.propagate_density_quadrature(h0, t, P, SIGMA)
        quad = C.mmd_sq_quadrature(ht)
        series, tail = C.von_mises_true_mmd_sq(kappa, t, P, SIGMA)
        assert tail < 1e-60
        assert abs(quad / series - 1.0) < 1e-5


def test_von_mises_embedding_coefficients():
    kappa, t = 1.0, 7
    x = C.grid_nodes(1 << 12)
    h0 = np.exp(kappa * np.cos(x)) / special.i0(kappa) - 1.0
    ht = C.propagate_density_quadrature(h0, t, P, SIGMA)
    kd = C.kernel_closed_form(2 * np.pi * np.arange(x.shape[0]) / x.shape[0])
    mu_quad = np.fft.irfft(np.fft.rfft(kd) * np.fft.rfft(ht), n=x.shape[0]) / x.shape[0]
    c = C.von_mises_embedding_coefficients(kappa, t, P, SIGMA, 30)
    mu_series = (c[:, None] * np.cos(np.outer(np.arange(1, 31), x))).sum(axis=0)
    assert np.max(np.abs(mu_quad - mu_series)) < 1e-6
    assert C.von_mises_embedding_sup_tail(kappa, 30) < 1e-30


def test_e1_construction_is_positive_and_hits_envelope():
    from mcmc.experiments.exp_r1_residual_vs_diagnostics import construction

    x = C.grid_nodes(1 << 12)
    for r in (0.5, 1.0):
        for m in (2, 4, 8, 16, 32, 64):
            con = construction(m, r, 0.1, P, SIGMA, 10)
            assert con["init_amplitude"] < 0.41  # sqrt2 * 0.1 / 0.9**10 bounds every setting
            h10 = C.propagate_density_quadrature(
                con["init_coeff"] * C.fourier_basis(x, m + 1), 10, P, SIGMA
            )
            series_sq = C.eigenvalue(m + 1) * np.mean(h10 * C.fourier_basis(x, m + 1)) ** 2
            assert abs(math.sqrt(series_sq) / con["predicted_residual"] - 1.0) < 1e-10
    with pytest.raises(ValueError):
        construction(3, 0.5, 0.1, P, SIGMA, 10)


def test_extended_precision_vstat_matches_float64_and_dense():
    rng = np.random.default_rng(41)
    x = rng.uniform(-np.pi, np.pi, 3000)
    val, bound = C.empirical_mean_embedding_sq_extended(x)
    assert abs(val - C.empirical_mean_embedding_sq_dense(x)) < 1e-12
    assert abs(val - C.empirical_mean_embedding_sq(x)) < 1e-12
    assert 0.0 < bound < 1e-14


def test_valid_mmd_interval_covers_known_mmd():
    from mcmc.experiments.exp_r1_residual_vs_diagnostics import valid_mmd_interval

    rng = np.random.default_rng(43)
    amp, freq, size, delta, trials = 0.5, 2, 256, 0.2, 2000
    truth = math.sqrt(C.frequency_weight(freq) * amp**2 / 4.0)
    misses = 0
    for _ in range(trials):
        x = C.sample_cosine_law(rng, size, freq, amp)
        lo, hi, b = valid_mmd_interval(*C.empirical_mean_embedding_sq_extended(x), size, delta)
        misses += not (lo <= truth <= hi)
    assert misses / trials <= delta
    with pytest.raises(ValueError):
        valid_mmd_interval(0.1, 0.0, 10, 1.0)


def test_sampled_setting_reproduces_per_repetition_seeds():
    from mcmc.experiments.exp_r1_residual_vs_diagnostics import _sampled_setting, construction

    task = (4, 1.0, 0, 512, 3, 12345, 0.1, P, SIGMA, 10, 10)
    out = _sampled_setting(task)
    con = construction(4, 1.0, 0.1, P, SIGMA, 10)
    pooled = []
    for rep in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([12345, 4, 1000, 0, rep]))
        x0 = C.sample_cosine_law(rng, 512, con["freq"], con["init_amplitude"])
        xt = C.simulate_explicit(x0, 10, rng, P, SIGMA)
        assert out["u"][rep] == C.unbiased_mmd_sq(xt)
        pooled.append(xt)
    val, _ = C.empirical_mean_embedding_sq_extended(np.concatenate(pooled))
    assert out["pooled_vnorm"] == val
