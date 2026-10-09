"""Uniform-circle target, Fourier kernel, and lazy wrapped-Gaussian walk.

Angles lie in [-pi, pi). The kernel has Fourier weights 6/(pi^2 n^2),
diagonal one, and zero target mean embedding. Its L2 eigenfunctions are
sqrt(2) cos(nx), sqrt(2) sin(nx), each with eigenvalue 3/(pi^2 n^2);
the RKHS eigenvectors are e_j = sqrt(lambda_j) psi_j.

The walk moves by a wrapped N(0, sigma^2) increment with probability p.
Its frequency-n eigenvalue is 1 - p + p exp(-sigma^2 n^2 / 2).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

PI = math.pi
TWO_PI = 2.0 * math.pi
_DIFF_TOL = 1e-12


# kernel


def kernel_closed_form(d: np.ndarray) -> np.ndarray:
    """k(d) = 1 - 3|d|/pi + 3 d^2/(2 pi^2) for differences of represented angles, |d| <= 2 pi."""
    d = np.asarray(d, dtype=np.float64)
    assert np.isfinite(d).all(), "non-finite angle differences"
    a = np.abs(d)
    if a.size and a.max() > TWO_PI + _DIFF_TOL:
        raise ValueError(f"closed form needs |d| <= 2pi, got max |d| = {a.max()}")
    return 1.0 - 3.0 * a / PI + 1.5 * a * a / (PI * PI)


def kernel_series(d: np.ndarray, n_terms: int) -> tuple[np.ndarray, float]:
    """Partial Fourier sum of the kernel and a rigorous bound on the omitted tail (sup norm)."""
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    d = np.asarray(d, dtype=np.float64)
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    vals = (6.0 / (PI * PI * n * n))[None, :] * np.cos(np.outer(d.ravel(), n))
    # sum_{n>N} 1/n^2 <= 1/N
    tail = 6.0 / (PI * PI * n_terms)
    return vals.sum(axis=1).reshape(d.shape), tail


def frequency_of_index(j: np.ndarray | int) -> np.ndarray:
    """Frequency n = ceil(j/2) of the 1-indexed eigenfunction psi_j."""
    j = np.asarray(j)
    if (j < 1).any():
        raise ValueError("eigen-indices are 1-based")
    return (j + 1) // 2


def eigenvalue(j: np.ndarray | int) -> np.ndarray:
    """lambda_j = 3/(pi^2 n^2), n = ceil(j/2), from the closed-form eigen-analysis."""
    n = frequency_of_index(j).astype(np.float64)
    return 3.0 / (PI * PI * n * n)


def frequency_weight(n: np.ndarray | int) -> np.ndarray:
    """w_n = 6/(pi^2 n^2), the Fourier weight of cos(n d) in k (equals 2 lambda_{2n})."""
    n = np.asarray(n, dtype=np.float64)
    if (n < 1).any():
        raise ValueError("frequencies are >= 1")
    return 6.0 / (PI * PI * n * n)


def fourier_basis(x: np.ndarray, j: int) -> np.ndarray:
    """psi_j(x): sqrt2 cos(n x) for odd j, sqrt2 sin(n x) for even j, n = ceil(j/2)."""
    if j < 1:
        raise ValueError("eigen-indices are 1-based")
    n = (j + 1) // 2
    x = np.asarray(x, dtype=np.float64)
    return math.sqrt(2.0) * (np.cos(n * x) if j % 2 == 1 else np.sin(n * x))


# sampler


def wrap(x: np.ndarray) -> np.ndarray:
    """Map real angles to [-pi, pi)."""
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all(), "non-finite angles"
    y = np.mod(x + PI, TWO_PI) - PI
    # np.mod can return exactly 2pi for tiny negative inputs after rounding; keep [-pi, pi).
    y[y >= PI] -= TWO_PI
    assert (y >= -PI).all() and (y < PI).all()
    return y


def check_angles(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"angles must be 1-D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("non-finite angles")
    if x.size and (x.min() < -PI or x.max() >= PI):
        raise ValueError("angles must lie in [-pi, pi)")
    return x


def lazy_rho(n: np.ndarray | int, p: float, sigma: float) -> np.ndarray:
    """Frequency-n eigenvalue rho_n = 1 - p + p exp(-sigma^2 n^2/2) of the lazy wrapped walk."""
    if not (0.0 < p <= 1.0) or not sigma > 0.0:
        raise ValueError(f"need 0 < p <= 1 and sigma > 0, got p={p}, sigma={sigma}")
    n = np.asarray(n, dtype=np.float64)
    return 1.0 - p + p * np.exp(-0.5 * sigma * sigma * n * n)


def explicit_step(x: np.ndarray, rng: np.random.Generator, p: float, sigma: float) -> np.ndarray:
    """One explicit transition: each replica moves with probability p by N(0, sigma^2)."""
    move = rng.random(x.shape[0]) < p
    y = x.copy()
    y[move] += sigma * rng.standard_normal(int(move.sum()))
    return wrap(y)


def simulate_explicit(
    x0: np.ndarray, n_steps: int, rng: np.random.Generator, p: float, sigma: float
) -> np.ndarray:
    """n_steps explicit transitions for every replica."""
    x = check_angles(x0).copy()
    for _ in range(int(n_steps)):
        x = explicit_step(x, rng, p, sigma)
    return x


def advance_checkpoint(
    x: np.ndarray, delta: int, rng: np.random.Generator, p: float, sigma: float
) -> np.ndarray:
    """Exact law of delta transitions: N ~ Binomial(delta, p) jumps, then a wrapped N(0, N sigma^2).

    The sum of N iid N(0, sigma^2) increments is N(0, N sigma^2), and wrapping commutes with
    addition, so this reproduces the delta-step transition exactly (not each step individually).
    """
    if delta < 0:
        raise ValueError("delta must be >= 0")
    x = check_angles(x)
    if delta == 0:
        return x.copy()
    jumps = rng.binomial(int(delta), p, size=x.shape[0])
    return wrap(x + sigma * np.sqrt(jumps.astype(np.float64)) * rng.standard_normal(x.shape[0]))


def wrapped_gaussian_density(x: np.ndarray, sigma: float, n_wraps: int = 12) -> np.ndarray:
    """Lebesgue density of a wrapped N(0, sigma^2) on [-pi, pi), by direct summation of wraps.

    Omitted wraps |k| > n_wraps lie at distance >= (2 n_wraps - 1) pi; the caller should choose
    n_wraps so that is many standard deviations (the default is far beyond double precision
    for sigma <= 1).
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.arange(-n_wraps, n_wraps + 1, dtype=np.float64)
    z = (x[..., None] + TWO_PI * k) / sigma
    return np.exp(-0.5 * z * z).sum(axis=-1) / (sigma * math.sqrt(TWO_PI))


def propagate_density_quadrature(h: np.ndarray, n_steps: int, p: float, sigma: float) -> np.ndarray:
    """Propagate a relative density error h = dQ/dpi - 1 on the uniform grid by quadrature.

    One step: (1 + h') = (1 - p)(1 + h) + p * (g * (1 + h)), with g the wrapped Gaussian
    density sampled pointwise on the grid and * the periodic trapezoid convolution (computed
    by FFT of the sampled values). This path does NOT use the analytic eigenvalue rho_n.
    """
    h = np.asarray(h, dtype=np.float64)
    n_nodes = h.shape[0]
    grid = grid_nodes(n_nodes)
    # g evaluated at differences x_a - x_0 = 2 pi a / N, re-centred to [-pi, pi).
    g = wrapped_gaussian_density(wrap(grid - grid[0]), sigma)
    g_hat = np.fft.rfft(g) * (TWO_PI / n_nodes)
    dens = 1.0 + h
    for _ in range(int(n_steps)):
        conv = np.fft.irfft(np.fft.rfft(dens) * g_hat, n=n_nodes)
        dens = (1.0 - p) * dens + p * conv
    out = dens - 1.0
    assert np.isfinite(out).all()
    return out


# quadrature


def grid_nodes(n_nodes: int) -> np.ndarray:
    """Uniform periodic grid x_a = -pi + 2 pi a / N."""
    if n_nodes < 8:
        raise ValueError("need at least 8 nodes")
    return -PI + TWO_PI * np.arange(n_nodes, dtype=np.float64) / n_nodes


def integrate_periodic(values: np.ndarray) -> float:
    """Trapezoid rule for the pi-expectation (mean over the uniform grid)."""
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    return float(values.mean())


def mmd_sq_quadrature(h: np.ndarray) -> float:
    """MMD^2(Q, pi) = E_pi E_pi[k(X - Y) h(X) h(Y)] by the periodic trapezoid rule.

    Uses the closed-form kernel sampled on the grid (circular differences 2 pi c / N in
    [0, 2 pi)) and an FFT circular convolution; no Fourier eigenvalues are used.
    """
    h = np.asarray(h, dtype=np.float64)
    n_nodes = h.shape[0]
    kd = kernel_closed_form(TWO_PI * np.arange(n_nodes, dtype=np.float64) / n_nodes)
    conv = np.fft.irfft(np.fft.rfft(kd) * np.fft.rfft(h), n=n_nodes) / n_nodes
    val = float(np.mean(h * conv))
    assert math.isfinite(val)
    return val


# empirical MMD


def empirical_mean_embedding_sq(x: np.ndarray) -> float:
    """||mu_hat||^2 = J^{-2} sum_{i,j} k(x_i, x_j) in O(J log J) by sorting (target mean is 0).

    For sorted angles y_0 <= ... <= y_{J-1}:
        sum_{i,j} |y_i - y_j| = 2 sum_i (2i - J + 1) y_i,
        sum_{i,j} (y_i - y_j)^2 = 2 J sum_i y_i^2 - 2 (sum_i y_i)^2.
    """
    y = np.sort(check_angles(x))
    n = y.shape[0]
    if n < 2:
        raise ValueError("need at least two angles")
    idx = np.arange(n, dtype=np.float64)
    pair_abs = 2.0 * np.dot(2.0 * idx - n + 1.0, y)
    s1 = y.sum()
    pair_sq = 2.0 * n * np.dot(y, y) - 2.0 * s1 * s1
    val = 1.0 - 3.0 * pair_abs / (PI * n * n) + 1.5 * pair_sq / (PI * PI * n * n)
    assert math.isfinite(val)
    return float(val)


def empirical_mean_embedding_sq_dense(x: np.ndarray) -> float:
    """O(J^2) reference implementation of ||mu_hat||^2 with the closed-form kernel."""
    x = check_angles(x)
    return float(kernel_closed_form(x[:, None] - x[None, :]).mean())


def unbiased_mmd_sq(x: np.ndarray) -> float:
    """Unbiased U-statistic for MMD^2(Q, pi) when mu_pi = 0: (J ||mu_hat||^2 - 1)/(J - 1)."""
    n = np.asarray(x).shape[0]
    return (n * empirical_mean_embedding_sq(x) - 1.0) / (n - 1.0)


def empirical_mean_embedding_sq_extended(x: np.ndarray) -> tuple[float, float]:
    """||mu_hat||^2 by the sorting formula accumulated in extended precision, with a rounding bound.

    For very large samples (here up to 2^25 angles) the float64 sums can lose ~N u in absolute
    terms. The sums are accumulated in np.longdouble; a worst-case recursive-summation analysis of
    the three pair sums (each term bounded by pi times its index weight) bounds the absolute
    rounding error of the result by 12 N u_ld, which is returned alongside the value.
    """
    eps = float(np.finfo(np.longdouble).eps)
    if not eps < 1e-18:
        raise RuntimeError(f"np.longdouble is not extended precision here (eps={eps})")
    y = np.sort(check_angles(x)).astype(np.longdouble)
    n = y.shape[0]
    if n < 2:
        raise ValueError("need at least two angles")
    idx = np.arange(n, dtype=np.longdouble)
    pi = np.longdouble(PI)
    pair_abs = 2 * np.dot(2 * idx - n + 1, y)
    s1 = np.sum(y)
    pair_sq = 2 * n * np.dot(y, y) - 2 * s1 * s1
    val = 1 - 3 * pair_abs / (pi * n * n) + 1.5 * pair_sq / (pi * pi * n * n)
    bound = 12.0 * n * (eps / 2.0)
    val = float(val)
    assert math.isfinite(val)
    return val, bound


def prefix_fourier_means(x: np.ndarray, n_max: int, prefixes: list[int]) -> np.ndarray:
    """Complex means (1/P) sum_{i<P} exp(i n x_i), n = 1..n_max, for each nested prefix size P.

    Returns shape (len(prefixes), n_max). Powers of exp(i x) are built by repeated
    multiplication (relative error ~ n * eps).
    """
    x = check_angles(x)
    prefixes = [int(p) for p in prefixes]
    if sorted(prefixes) != prefixes or prefixes[0] < 1 or prefixes[-1] > x.shape[0]:
        raise ValueError(f"prefixes must be increasing within [1, {x.shape[0]}]: {prefixes}")
    base = np.exp(1j * x[: prefixes[-1]])
    power = np.ones_like(base)
    bounds = [0, *prefixes]
    out = np.empty((len(prefixes), n_max), dtype=np.complex128)
    for n in range(n_max):
        power = power * base
        block = np.array([power[bounds[b] : bounds[b + 1]].sum() for b in range(len(prefixes))])
        out[:, n] = np.cumsum(block) / np.asarray(prefixes, dtype=np.float64)
    assert np.isfinite(out.view(np.float64)).all()
    return out


def embedding_coordinates(fourier_means: np.ndarray) -> np.ndarray:
    """RKHS coordinates <e_j, mu_hat> = sqrt(lambda_j) E_hat psi_j for j = 1..2 n_max.

    psi_{2n-1} = sqrt2 cos, psi_{2n} = sqrt2 sin, so <e_{2n-1}, mu_hat> = sqrt(2 lambda) Re m_n and
    <e_{2n}, mu_hat> = sqrt(2 lambda) Im m_n with lambda = 3/(pi^2 n^2).
    """
    fm = np.asarray(fourier_means)
    n_max = fm.shape[-1]
    n = np.arange(1, n_max + 1, dtype=np.float64)
    scale = np.sqrt(2.0 * 3.0 / (PI * PI * n * n))
    out = np.empty(fm.shape[:-1] + (2 * n_max,), dtype=np.float64)
    out[..., 0::2] = scale * fm.real
    out[..., 1::2] = scale * fm.imag
    return out


# laws


def sample_cosine_law(
    rng: np.random.Generator, count: int, freq: int, amplitude: float
) -> np.ndarray:
    """Exact rejection sampling from dQ/dpi = 1 + amplitude * cos(freq * x) (|amplitude| < 1)."""
    if not abs(amplitude) < 1.0:
        raise ValueError(f"density must stay positive: |amplitude|={abs(amplitude)} >= 1")
    pieces = []
    left = int(count)
    bound = 1.0 + abs(amplitude)
    while left:
        prop = rng.uniform(-PI, PI, size=max(64, int(left * bound * 1.1)))
        accept = rng.random(prop.shape[0]) * bound < 1.0 + amplitude * np.cos(freq * prop)
        block = prop[accept][:left]
        pieces.append(block)
        left -= block.shape[0]
    return check_angles(np.concatenate(pieces))


def bessel_ratio(n: np.ndarray, kappa: float) -> np.ndarray:
    """I_n(kappa)/I_0(kappa) via exponentially scaled Bessel functions."""
    n = np.asarray(n, dtype=np.float64)
    val = special.ive(n, kappa) / special.ive(0, kappa)
    assert np.isfinite(val).all()
    return val


def bessel_ratio_bound(n: np.ndarray, kappa: float) -> np.ndarray:
    """Rigorous upper bound I_n(k)/I_0(k) <= (k/2)^n exp(k^2/4) / (n! I_0(k)).

    From I_n(k) = sum_s (k/2)^{2s+n}/(s!(s+n)!) and (s+n)! >= s! n!, s! >= 1.
    """
    n = np.asarray(n, dtype=np.float64)
    log_val = n * math.log(kappa / 2.0) + kappa * kappa / 4.0 - special.gammaln(n + 1.0)
    return np.exp(log_val) / special.i0(kappa)


def von_mises_radius_sq(kappa: float) -> float:
    """||dQ/dpi - 1||^2_{L^2(pi)} = I_0(2 kappa)/I_0(kappa)^2 - 1 for the von Mises(0, kappa) law."""
    return float(special.i0(2.0 * kappa) / special.i0(kappa) ** 2 - 1.0)


def sample_symmetric_von_mises(
    rng: np.random.Generator, count: int, kappa: float, symmetry: int = 1
) -> np.ndarray:
    """Exact draws from dQ/dpi = exp(kappa cos(s x)) / I_0(kappa), s = symmetry >= 1.

    s = 1 is von Mises(0, kappa) (one rng.vonmises call, nothing else drawn). For s > 1, draw
    y ~ von Mises(0, kappa) and an independent uniform branch u in {0..s-1}; x = (y + 2 pi u)/s
    (wrapped) has density proportional to exp(kappa cos(s x)) because s x = y mod 2 pi. The
    L2(pi) radius ||dQ/dpi - 1|| is the same for every s.
    """
    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")
    y = rng.vonmises(0.0, kappa, size=count)
    if symmetry == 1:
        return wrap(y)
    u = rng.integers(0, symmetry, size=count)
    return wrap((y + TWO_PI * u) / symmetry)


def von_mises_true_mmd_sq(
    kappa: float, t: int, p: float, sigma: float, n_terms: int = 40, symmetry: int = 1
) -> tuple[float, float]:
    """MMD^2(pi_t, pi) = sum_k w_{sk} (I_k/I_0)^2 rho_{sk}^{2t}; returns (partial sum, tail bound).

    s = symmetry (1 for the von Mises law). The tail bound uses bessel_ratio_bound, rho <= 1 and
    w_{sk} <= w_k, and sums the dominating geometric-factorial series to 200 extra terms plus a
    geometric remainder.
    """
    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")
    k = np.arange(1, n_terms + 1, dtype=np.float64)
    f = symmetry * k
    partial = float(
        np.sum(frequency_weight(f) * bessel_ratio(k, kappa) ** 2 * lazy_rho(f, p, sigma) ** (2 * t))
    )
    return partial, von_mises_series_tail(kappa, n_terms, power=2)


def von_mises_series_tail(kappa: float, n_terms: int, power: int) -> float:
    """Bound sum_{n>N} w_n (I_n/I_0)^power (rho_n <= 1 dropped)."""
    m = np.arange(n_terms + 1, n_terms + 201, dtype=np.float64)
    terms = frequency_weight(m) * bessel_ratio_bound(m, kappa) ** power
    # terms decrease super-geometrically once (kappa/2)^power/(n+1)^power < 1/2; bound the rest
    ratio = (kappa / 2.0) ** power / (m[-1] + 1.0) ** power
    if not ratio < 0.5:
        raise ValueError("tail bound requires (kappa/2)^power/(N+201)^power < 1/2")
    tail = float(terms.sum() + terms[-1] * ratio / (1.0 - ratio))
    assert math.isfinite(tail) and tail >= 0.0
    return tail


def von_mises_embedding_coefficients(
    kappa: float, t: int, p: float, sigma: float, n_terms: int, symmetry: int = 1
) -> np.ndarray:
    """c_n(t), n = 1..n_terms, with mu_{pi_t}(x) = sum_n c_n(t) cos(n x).

    c_n = w_n (I_{n/s}/I_0) rho_n^t when s divides n, else 0 (s = symmetry).
    """
    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")
    n = np.arange(1, n_terms + 1)
    out = np.zeros(n_terms, dtype=np.float64)
    hit = n % symmetry == 0
    nf = n[hit].astype(np.float64)
    out[hit] = (
        frequency_weight(nf) * bessel_ratio(nf / symmetry, kappa) * lazy_rho(nf, p, sigma) ** t
    )
    return out


def von_mises_embedding_sup_tail(kappa: float, n_terms: int, symmetry: int = 1) -> float:
    """sup_x |sum_{n>N} c_n(t) cos(n x)| <= sum_{k > floor(N/s)} w_k I_k/I_0 (any t; w_{sk} <= w_k)."""
    return von_mises_series_tail(kappa, n_terms // symmetry, power=1)
