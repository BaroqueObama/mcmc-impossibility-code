"""Independent reference intervals for MMD(pi_t, pi).

The reference replicas and target draws are separate from those used in the
tested bound. With D_t = ||mu_hat_t - mu_hat_pi|| and reference radii b_t, b_pi,
L_t = max(0, D_t - b_t - b_pi) and U_t = D_t + b_t + b_pi hold simultaneously
with probability at least 1 - beta_ref.

A bound B < L_t is a violation, B >= U_t is covered, and intermediate values
are unresolved. Coverage summaries report both possible outcomes for unresolved
ensembles, with Clopper-Pearson intervals for each.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

VIOLATION, UNRESOLVED, COVERED = -1, 0, 1


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) binomial interval."""
    if not (0 <= k <= n and n >= 1):
        raise ValueError(f"bad counts k={k}, n={n}")
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def truth_intervals(
    D: np.ndarray, b_ref_t: np.ndarray, b_ref_pi: float
) -> tuple[np.ndarray, np.ndarray]:
    """L_t, U_t from reference distances D_t = ||mu_hat^ref_t - mu_hat^ref_pi|| and radii."""
    D = np.asarray(D, dtype=np.float64)
    b = np.asarray(b_ref_t, dtype=np.float64) + float(b_ref_pi)
    if D.shape != b.shape or (D < 0).any() or not np.isfinite(D).all():
        raise ValueError(f"bad reference inputs D={D}, b={b}")
    return np.maximum(0.0, D - b), D + b


def classify(B: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """(n_ens, T) codes in {VIOLATION, UNRESOLVED, COVERED}; B rows are ensembles, columns times."""
    B = np.asarray(B, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if B.ndim != 2 or L.shape != (B.shape[1],) or U.shape != L.shape:
        raise ValueError(f"shapes B {B.shape}, L {L.shape}, U {U.shape}")
    if not (np.isfinite(B).all() and np.isfinite(L).all() and np.isfinite(U).all()):
        raise ValueError("non-finite bound or interval")
    if (L > U).any():
        raise ValueError("L_t > U_t")
    codes = np.full(B.shape, UNRESOLVED, dtype=np.int8)
    codes[B < L[None, :]] = VIOLATION
    codes[B >= U[None, :]] = COVERED
    return codes


def classify_analytic(B: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Direct coverage check against exact truth: COVERED if B >= MMD_t, else VIOLATION."""
    B = np.asarray(B, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if B.ndim != 2 or truth.shape != (B.shape[1],):
        raise ValueError(f"shapes B {B.shape}, truth {truth.shape}")
    if not (np.isfinite(B).all() and np.isfinite(truth).all()):
        raise ValueError("non-finite bound or truth")
    return np.where(B >= truth[None, :], COVERED, VIOLATION).astype(np.int8)


def per_ensemble(codes: np.ndarray) -> dict:
    """Counts of covered-all / any-violation / unresolved ensembles over all inspected times."""
    codes = np.asarray(codes)
    covered_all = (codes == COVERED).all(1)
    any_violation = (codes == VIOLATION).any(1)
    unresolved = ~covered_all & ~any_violation
    return {
        "n": int(codes.shape[0]),
        "covered_all": int(covered_all.sum()),
        "any_violation": int(any_violation.sum()),
        "unresolved": int(unresolved.sum()),
        "unresolved_cells_fraction": float((codes == UNRESOLVED).mean()),
    }


def coverage_summary(codes: np.ndarray, alpha: float = 0.05, beta_ref: float | None = None) -> dict:
    """Empirical simultaneous coverage with Clopper-Pearson intervals; unresolved kept separate."""
    s = per_ensemble(codes)
    n, c, u = s["n"], s["covered_all"], s["unresolved"]
    s["coverage_if_unresolved_fail"] = c / n
    s["coverage_if_unresolved_succeed"] = (c + u) / n
    s["cp_if_unresolved_fail"] = clopper_pearson(c, n, alpha)
    s["cp_if_unresolved_succeed"] = clopper_pearson(c + u, n, alpha)
    s["cp_alpha"] = alpha
    s["beta_ref"] = beta_ref
    return s


def reference_radii(
    kappa: float,
    J_ref: int,
    N_ref: int | None,
    beta_ref_mu: float,
    beta_ref_pi: float,
    n_times: int,
):
    """App. E.2 radii for a reference ensemble with uniform weights over n_times checkpoints.

    N_ref=None means a known target embedding (b_pi = 0, beta_ref_pi unused).
    """
    w = 1.0 / n_times
    b_t = kappa / math.sqrt(J_ref) + kappa * math.sqrt(
        2.0 * math.log(1.0 / (beta_ref_mu * w)) / J_ref
    )
    if N_ref is None:
        b_pi = 0.0
    else:
        b_pi = kappa / math.sqrt(N_ref) + kappa * math.sqrt(
            2.0 * math.log(1.0 / beta_ref_pi) / N_ref
        )
    return b_t, b_pi
