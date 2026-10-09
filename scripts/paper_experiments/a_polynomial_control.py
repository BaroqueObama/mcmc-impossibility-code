"""Attainment examples for the circle's polynomial covariance spectrum.

Each row uses a law whose first m diagnostic expectations match the target.
Periodic quadrature propagates the law and evaluates its MMD. The spectrum is
lambda_(m+1) = 3 / (pi^2 ceil((m+1)/2)^2), so at even m the bound is
R [12/(pi^2 (m+2)^2)]^r: a log-log line of slope -2r against m+2.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mcmc.paperexp import circle as C  # noqa: E402


def quadrature_objects(nodes: int, p: float, sigma: float, t: int) -> dict:
    x = C.grid_nodes(nodes)
    differences = C.TWO_PI * np.arange(nodes, dtype=np.float64) / nodes
    transition_density = C.wrapped_gaussian_density(C.wrap(differences), sigma)
    transition_fft = np.fft.rfft(transition_density) * C.TWO_PI / nodes
    kernel_fft = np.fft.rfft(C.kernel_closed_form(differences)) / nodes
    return {
        "x": x,
        "nodes": nodes,
        "propagator": ((1.0 - p) + p * transition_fft) ** t,
        "kernel_fft": kernel_fft,
        "transition_mass_error": abs(float(transition_density.mean() * C.TWO_PI) - 1),
    }


def integrate_hidden_mode(m: int, rho_t: float, q: dict) -> dict:
    """Numerical propagation and integration without substituting the analytic MMD."""
    hidden = m + 1
    psi = C.fourier_basis(q["x"], hidden)
    h0 = psi / rho_t
    propagated_fft = np.fft.rfft(h0) * q["propagator"]
    ht = np.fft.irfft(propagated_fft, n=q["nodes"])
    kernel_mean = np.fft.irfft(propagated_fft * q["kernel_fft"], n=q["nodes"])
    mmd_sq = float(np.mean(ht * kernel_mean))
    if mmd_sq <= 0:
        raise RuntimeError(f"Nonpositive integrated MMD at m={m}")

    # FFT coefficients are the periodic quadrature of all monitored moments.
    # Grid nodes start at -pi, giving the frequency-dependent sign below.
    indices = np.arange(1, m + 1)
    frequencies = C.frequency_of_index(indices)
    fourier_means = propagated_fft[frequencies] / q["nodes"]
    shift = (-1.0) ** frequencies
    basis_means = math.sqrt(2) * shift * np.where(
        indices % 2 == 1, fourier_means.real, -fourier_means.imag
    )
    monitored = np.sqrt(C.eigenvalue(indices)) * basis_means
    return {
        "mmd": math.sqrt(mmd_sq),
        "min_h0": float(h0.min()),
        "min_ht": float(ht.min()),
        "normalization_error": abs(float(ht.mean())),
        "hidden_coefficient": float(np.mean(ht * psi)),
        "max_abs_monitored": float(np.abs(monitored).max()),
        "monitored_norm": float(np.linalg.norm(monitored)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-m", type=int, default=200)
    parser.add_argument("--coarse-nodes", type=int, default=131072)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/data/r1_dense")
    args = parser.parse_args()
    if args.max_m < 2 or args.coarse_nodes < 16 * args.max_m:
        raise ValueError("Require max-m >= 2 and coarse-nodes >= 16*max-m")

    R, p, sigma, t = 0.1, 0.1, 0.5, 10
    fractions = (0.25, 0.5, 1.0)
    smoothness = (0.5, 1.0)
    coarse = quadrature_objects(args.coarse_nodes, p, sigma, t)
    fine = quadrature_objects(2 * args.coarse_nodes, p, sigma, t)
    rows = []
    for m in range(1, args.max_m + 1):
        frequency = int(C.frequency_of_index(m + 1))
        lam = float(C.eigenvalue(m + 1))
        rho = float(C.lazy_rho(frequency, p, sigma))
        qc = integrate_hidden_mode(m, rho**t, coarse)
        qf = integrate_hidden_mode(m, rho**t, fine)
        for r in smoothness:
            bound = R * lam**r
            for fraction in fractions:
                coefficient = fraction * R * lam ** (r - 0.5)
                residual = coefficient * qf["mmd"]
                predicted = fraction * bound
                record = {
                    "m": m,
                    "r": r,
                    "R": R,
                    "fraction_of_radius": fraction,
                    "hidden_index": m + 1,
                    "frequency": frequency,
                    "effective_count": 2 * frequency,
                    "complete_pairs": m % 2 == 0,
                    "lambda_m1": lam,
                    "bound": bound,
                    "predicted_residual": predicted,
                    "quadrature_residual": residual,
                    "quadrature_residual_coarse": coefficient * qc["mmd"],
                    "relative_error_vs_exact": residual / predicted - 1.0,
                    "relative_grid_change": qf["mmd"] / qc["mmd"] - 1.0,
                    "min_density_initial": 1.0 + coefficient * qf["min_h0"],
                    "min_density_inspection": 1.0 + coefficient * qf["min_ht"],
                    "normalization_error": coefficient * qf["normalization_error"],
                    "max_abs_monitored_expectation": coefficient * qf["max_abs_monitored"],
                    "monitored_component_norm": coefficient * qf["monitored_norm"],
                    "source_norm_exact": fraction * R,
                    "source_norm_quadrature": fraction * R * qf["hidden_coefficient"],
                    "transition_eigenvalue": rho,
                    "inspection_time": t,
                    "coarse_nodes": coarse["nodes"],
                    "fine_nodes": fine["nodes"],
                }
                if record["min_density_initial"] <= 0 or record["min_density_inspection"] <= 0:
                    raise RuntimeError(f"Construction is not a probability law: {record}")
                if abs(record["relative_error_vs_exact"]) > 1e-6:
                    raise RuntimeError(f"Quadrature inaccurate: {record}")
                if abs(record["relative_grid_change"]) > 1e-6:
                    raise RuntimeError(f"Quadrature grid not resolved: {record}")
                if record["max_abs_monitored_expectation"] > 1e-12:
                    raise RuntimeError(f"A monitored expectation is not zero: {record}")
                if record["normalization_error"] > 1e-12:
                    raise RuntimeError(f"Density normalization inaccurate: {record}")
                rows.append(record)

    for r in smoothness:
        selected = [x for x in rows if x["r"] == r and x["fraction_of_radius"] == 1 and x["complete_pairs"]]
        slope = float(np.polyfit(np.log([x["m"] + 2 for x in selected]), np.log([x["bound"] for x in selected]), 1)[0])
        if abs(slope + 2*r) > 1e-12:
            raise RuntimeError("The declared polynomial slope is inconsistent with the spectrum")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "polynomial.csv"
    write_csv(csv_path, rows)
    quadrature_error = max(abs(x["relative_error_vs_exact"]) for x in rows)
    grid_change = max(abs(x["relative_grid_change"]) for x in rows)
    print(f"Wrote {len(rows)} rows to {csv_path}; "
          f"max relative quadrature error={quadrature_error:.3g}, "
          f"grid change={grid_change:.3g}")


if __name__ == "__main__":
    main()
