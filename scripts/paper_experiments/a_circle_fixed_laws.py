"""Circle residual curves for two fixed population laws.

The starts are dQ0/dpi = 1 + R sign(cos(s x)), with s=1,3. Each curve uses the
same law at time ten; only the number of projected Fourier modes changes.
Analytic residuals are checked by quadrature and high-precision summation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mcmc.paperexp import circle as C  # noqa: E402


def exact_residual_squared(m: int, symmetry: int, R: float, p: float,
                           sigma: float, t: int) -> tuple[float, float]:
    """Analytic Fourier tail with a bounded omitted transition correction.

    Only odd harmonics k of s contribute. The cosine monitor at frequency s*k
    has index 2*s*k-1. Its squared embedding coefficient is
    24 R^2 rho_(s*k)^(2t) / (pi^4 s^2 k^4).
    The infinite constant-laziness tail is evaluated with Hurwitz zeta.
    """
    k0 = (m + 1) // (2 * symmetry) + 1
    if k0 % 2 == 0:
        k0 += 1
    odd_tail = float(special.zeta(4.0, k0 / 2.0) / 16.0)
    floor_power = (1.0 - p) ** (2 * t)
    last = max(63, k0 + 62)
    k = np.arange(k0, last + 1, 2, dtype=float)
    corrections = (C.lazy_rho(symmetry * k, p, sigma) ** (2 * t) - floor_power) / k**4
    factor = 24 * R**2 / (math.pi**4 * symmetry**2)
    value = factor * (floor_power * odd_tail + float(corrections.sum()))
    # x^q-y^q <= q(x-y), for 0 <= y <= x <= 1, bounds the remainder.
    next_k = last + 2
    # For decreasing x^-4, sum_{j>=0}(next_k+2j)^-4 is at most
    # next_k^-4 + (1/2) integral_{next_k}^infinity x^-4 dx.
    # Evaluate in logs and round upward; underflow is bounded by the smallest
    # positive float64 subnormal.
    log_terms = (
        math.log(factor), math.log(2 * t * p),
        -0.5 * sigma**2 * (symmetry * next_k)**2,
        math.log(next_k**-4 + 1.0 / (6.0 * next_k**3)),
    )
    log_upper = math.fsum(log_terms)
    log_upper += 64 * np.finfo(float).eps * math.fsum(abs(x) for x in log_terms)
    log_upper = float(np.nextafter(log_upper, math.inf))
    smallest_positive = float(np.nextafter(0.0, 1.0))
    omitted_upper = smallest_positive if log_upper <= math.log(smallest_positive) else float(
        np.nextafter(math.exp(log_upper), math.inf)
    )
    return value, omitted_upper


def quadrature(nodes: int, symmetry: int, R: float, p: float,
               sigma: float, t: int, max_m: int) -> dict:
    """Propagate midpoint-grid densities and integrate the closed-form kernel.

    No analytic density Fourier coefficients, rho_n, covariance eigenvalues,
    or residual formulas are substituted into this independent calculation.
    FFTs implement periodic trapezoidal convolution and its Parseval identity.
    """
    x = -math.pi + C.TWO_PI * (np.arange(nodes) + 0.5) / nodes
    h0 = R * np.sign(np.cos(symmetry * x))
    differences = C.TWO_PI * np.arange(nodes) / nodes
    transition = C.wrapped_gaussian_density(C.wrap(differences), sigma)
    propagator = ((1 - p) + p * np.fft.rfft(transition) * C.TWO_PI / nodes) ** t
    ht_fft = np.fft.rfft(h0) * propagator
    ht = np.fft.irfft(ht_fft, n=nodes)
    kernel_coefficients = np.fft.rfft(C.kernel_closed_form(differences)).real / nodes
    # For real vectors the conjugate positive/negative frequencies contribute twice.
    energy = 2 * np.abs(ht_fft / nodes)**2 * kernel_coefficients
    energy[0] /= 2
    energy[-1] /= 2
    tail = np.cumsum(energy[::-1])[::-1]
    residuals = []
    for m in range(1, max_m + 1):
        complete = m // 2
        if m % 2 == 0:
            val = float(tail[complete + 1])
        else:
            frequency = complete + 1
            # An odd m removes this cosine, while leaving the corresponding sine.
            moment = np.conj(ht_fft[frequency] / nodes) * np.exp(1j * frequency * x[0])
            val = float(tail[frequency + 1] + 2 * moment.imag**2 * kernel_coefficients[frequency])
        if val <= 0:
            raise RuntimeError("Nonpositive quadrature residual; do not clip")
        residuals.append(math.sqrt(val))

    # Independently confirm the FFT Parseval integration by real-space convolution.
    kernel_mean = np.fft.irfft(ht_fft * kernel_coefficients, n=nodes)
    full_real_space = float(np.mean(ht * kernel_mean))
    if not math.isclose(float(energy.sum()), full_real_space, rel_tol=1e-12, abs_tol=1e-18):
        raise RuntimeError("Kernel quadrature and Parseval identity disagree")
    return {
        "residuals": residuals,
        "source_norm_initial": float(np.sqrt(np.mean(h0**2))),
        "source_norm_time_t": float(np.sqrt(np.mean(ht**2))),
        "normalization_error": abs(float(ht.mean())),
        "min_density_initial": float((1 + h0).min()),
        "min_density_time_t": float((1 + ht).min()),
        "full_mmd": math.sqrt(full_real_space),
        "transition_mass_error": abs(float(transition.mean() * C.TWO_PI) - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-m", type=int, default=200)
    parser.add_argument("--coarse-nodes", type=int, default=131072)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/data/r1_dense")
    args = parser.parse_args()
    if args.max_m < 2 or args.coarse_nodes < 32 * args.max_m or args.coarse_nodes % 2:
        raise ValueError("Require m>=2 and an even grid with at least 32*m nodes")
    R, r, p, sigma, t = 0.1, 0.5, 0.1, 0.5, 10
    rows = []
    for label, symmetry in (("slow", 1), ("fast", 3)):
        coarse = quadrature(args.coarse_nodes, symmetry, R, p, sigma, t, args.max_m)
        fine = quadrature(2 * args.coarse_nodes, symmetry, R, p, sigma, t, args.max_m)
        full_sq, _ = exact_residual_squared(0, symmetry, R, p, sigma, t)
        for m in range(1, args.max_m + 1):
            squared, omitted_upper = exact_residual_squared(m, symmetry, R, p, sigma, t)
            residual = math.sqrt(squared)
            bound = R * math.sqrt(float(C.eigenvalue(m + 1)))
            qf, qc = fine["residuals"][m - 1], coarse["residuals"][m - 1]
            record = {
                "start": label, "symmetry": symmetry, "m": m, "effective_count": m + 2,
                "complete_pairs": m % 2 == 0, "R": R, "r": r, "inspection_time": t,
                "bound": bound, "residual": residual, "residual_squared": squared,
                "analytic_omitted_correction_upper_squared": omitted_upper,
                "quadrature_residual": qf, "quadrature_residual_coarse": qc,
                "relative_quadrature_error": qf / residual - 1,
                "relative_grid_change": qf / qc - 1,
                "full_mmd": math.sqrt(full_sq), "source_norm_initial": R,
                "source_norm_time_t_quadrature": fine["source_norm_time_t"],
                "min_initial_relative_density": 1 - R,
                "min_time_t_relative_density_quadrature": fine["min_density_time_t"],
            }
            if residual > bound:
                raise RuntimeError("Fixed-law residual exceeds its source-class bound")
            if abs(record["relative_quadrature_error"]) > 1e-4 or abs(record["relative_grid_change"]) > 1e-4:
                raise RuntimeError(f"Grid not sufficiently resolved: {record}")
            rows.append(record)
        selected = [x for x in rows if x["start"] == label]
        if np.any(np.diff([x["residual"] for x in selected]) > 1e-16):
            raise RuntimeError("Projected residual increased when adding a diagnostic")
        quadrature_error = max(abs(x["relative_quadrature_error"]) for x in selected)
        grid_change = max(abs(x["relative_grid_change"]) for x in selected)
        print(f"{label}: max relative quadrature error={quadrature_error:.3g}, "
              f"grid change={grid_change:.3g}")

    # Check the Fourier tail with mpmath at 60 digits.
    import mpmath as mp
    with mp.workdps(60):
        max_mp_difference = 0.0
        for s in (1, 3):
            for m in (0, 1, 2, 16, 100, args.max_m):
                k0 = (m + 1) // (2 * s) + 1
                if k0 % 2 == 0:
                    k0 += 1
                p_mp, R_mp, sd_mp = mp.mpf(str(p)), mp.mpf(str(R)), mp.mpf(str(sigma))
                base = (1 - p_mp) ** (2 * t)
                value = base * mp.zeta(4, mp.mpf(k0) / 2) / 16
                for k in range(k0, k0 + 200, 2):
                    rho = 1 - p_mp + p_mp * mp.exp(-sd_mp**2 * (s * k)**2 / 2)
                    value += (rho**(2*t) - base) / k**4
                value *= 24 * R_mp**2 / (mp.pi**4 * s**2)
                direct, _ = exact_residual_squared(m, s, R, p, sigma, t)
                max_mp_difference = max(max_mp_difference, abs(float(mp.mpf(direct) / value - 1)))
        if max_mp_difference > 1e-13:
            raise RuntimeError("High-precision analytic check failed")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "circle_fixed_laws.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}; "
          f"max relative high-precision difference={max_mp_difference:.3g}")


if __name__ == "__main__":
    main()
