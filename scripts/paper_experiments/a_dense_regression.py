"""Analytic residual curves for the five Gaussian regression posteriors.

Each curve fixes a time-one law with R=0.2 and projects onto 1..200 covariance
eigenvectors. Two Mehler truncations and Gauss-Hermite orders check numerical
sensitivity. Residuals below the floating-point/refinement allowance are marked
unresolved. These allowances are not interval-arithmetic certificates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from mcmc.paperexp import r1_real as rr
from mcmc.paperexp import regression_data as rd

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("concrete", "energy", "airfoil", "wine_red", "kin8nm")
RADIUS = 0.2
CHECKPOINT = 1


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(name):
    """Load the posterior, bandwidth and saved initial shifts."""
    config_path = ROOT / "configs/experiment" / (
        "exp_r3_concrete.yaml" if name == "concrete" else "exp_r3_regression.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    if name == "concrete":
        ell = float(config["kernel"]["ell_frozen"])
        sigma2 = float(config["model"]["sigma2_declared"])
    else:
        ell = float(config["datasets"][name]["ell_frozen"])
        sigma2 = float(config["datasets"][name]["sigma2_declared"])
    X, y, _ = rd.load_dataset(name)
    model = rd.RegressionPosterior(name, X, y)
    assert model.sigma2 == sigma2
    # Preserve the recorded initial laws, including the phase of kin8nm's
    # complex Gibbs eigenvector (which is not invariant across LAPACK builds).
    saved_path = ROOT / "paper/figures/data" / (
        "r3_concrete/summary_main.json" if name == "concrete"
        else f"r3_regression/summary_{name}.json"
    )
    saved = json.loads(saved_path.read_text())
    assert saved["setup"]["r_init"] == [0.8, 0.2, 0.8, 0.2]
    shifts = {"slow": np.asarray(saved["setup"]["shifts"][1]),
              "fast": np.asarray(saved["setup"]["shifts"][3])}
    for shift in shifts.values():
        radius2 = math.expm1(float(shift @ model.P @ shift))
        assert abs(radius2 / RADIUS**2 - 1) < 1e-9
    return model, ell, shifts


def readings(spec, shift, count, order):
    """Evaluate covariance-eigenvector means at the chosen quadrature order."""
    sz = np.linalg.solve(spec.T, shift)
    means = np.ones(spec.modes.shape[0])
    for j in range(spec.d):
        tab = rr.shifted_psi_means(
            spec.sig2[j], spec.lam1[j].shape[0] - 1, float(sz[j]), n_gh=order
        )
        means *= tab[spec.modes[:, j]]
    return np.sqrt(spec.eig[:count]) * ((means - spec.m_modes) @ spec.W[:, :count])


def calculate(name, count):
    model, ell, initial_shifts = load_model(name)
    specs, values = [], []
    for floor, order in ((1e-13, 160), (1e-14, 240)):
        spec = rr.exact_spectrum(
            model.mu, model.C, model.coord_sd, ell, rel_floor=floor, n_out=count + 1
        )
        assert len(spec.eig) == count + 1 and np.all(spec.eig > 0)
        assert np.all(np.diff(spec.eig) <= 0)
        stage_values = {}
        for kind, initial in initial_shifts.items():
            shift = np.linalg.matrix_power(model.chain.A, CHECKPOINT) @ initial
            full2 = rr.shifted_mmd2(spec, shift)
            z = readings(spec, shift, count, order)
            # Compensated sums reduce cancellation error in small residuals.
            monitored2 = np.asarray([math.fsum(z[:m] ** 2) for m in range(1, count + 1)])
            stage_values[kind] = {"full2": full2, "monitored2": monitored2}
        values.append(stage_values)
        specs.append({"eig": spec.eig.copy()})
        print(f"{name}: floor={floor:g}, GH={order}, modes={len(spec.modes)}, "
              f"even={spec.info['n_even_modes']}", flush=True)
        del spec
    coarse, fine = specs
    eigen_relative = np.abs(fine["eig"] - coarse["eig"]) / fine["eig"]
    assert float(eigen_relative.max()) < 1e-8, (name, eigen_relative.max())
    rows, spectrum = [], []
    for j, (lam, old_lam) in enumerate(zip(fine["eig"], coarse["eig"]), 1):
        spectrum.append({"dataset": name, "j": j, "lambda": float(lam),
                         "lambda_coarse": float(old_lam),
                         "refinement_relative_difference": float(abs(lam-old_lam)/lam)})
    for kind in ("slow", "fast"):
        f, c = values[1][kind], values[0][kind]
        for m in range(1, count + 1):
            monitored2 = float(f["monitored2"][m-1])
            full2 = float(f["full2"])
            residual2 = full2 - monitored2
            coarse2 = float(c["full2"] - c["monitored2"][m-1])
            roundoff = 256 * math.ulp(max(full2, monitored2))
            refinement = abs(residual2 - coarse2)
            allowance = roundoff + refinement
            if residual2 < -allowance:
                raise FloatingPointError((name, kind, m, residual2, allowance))
            status = "RESOLVED" if residual2 > allowance else "BELOW_NUMERICAL_SENSITIVITY"
            residual = math.sqrt(residual2) if status == "RESOLVED" else None
            bound = RADIUS * math.sqrt(float(fine["eig"][m]))
            if residual is not None:
                assert residual <= bound, (name, kind, m)
            rows.append({"dataset": name, "start": kind, "setting": f"{kind}_R0.2",
                         "R": RADIUS, "r": 0.5, "t": CHECKPOINT, "m": m,
                         "bound": bound, "lambda_next": float(fine["eig"][m]),
                         "residual": residual, "residual_squared_raw": residual2,
                         "full_mmd": math.sqrt(full2), "full_mmd_squared": full2,
                         "monitored_squared": monitored2, "roundoff_allowance_squared": roundoff,
                         "refinement_difference_squared": refinement,
                         "numerical_allowance_squared": allowance,
                         "status": status,
                         "numerical_upper_level": math.sqrt(max(residual2 + allowance, 0.0))})
    resolved = sum(row["status"] == "RESOLVED" for row in rows)
    print(f"{name}: {resolved}/{len(rows)} residuals resolved; "
          f"max relative eigenvalue refinement={eigen_relative.max():.3g}", flush=True)
    return rows, spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--m-max", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures/data/r1_dense")
    args = parser.parse_args()
    assert args.m_max >= 32
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_spectrum = [], []
    for name in args.datasets:
        rows, spectrum = calculate(name, args.m_max)
        all_rows.extend(rows)
        all_spectrum.extend(spectrum)
        write_csv(args.out / f"regression_{name}.csv", rows)
    write_csv(args.out / "regression_curves.csv", all_rows)
    write_csv(args.out / "regression_spectrum.csv", all_spectrum)


if __name__ == "__main__":
    main()
