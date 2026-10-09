"""Anytime MMD bounds for the lazy wrapped-Gaussian chain on the circle.

Replicas start from a von Mises law and continue through all checkpoints;
Binomial jump counts give exact transitions between checkpoints. The target
embedding is zero. The stress configuration uses exp(kappa0 cos(3x))/I_0(kappa0),
which has the same L2 radius but no error in the first two Fourier modes.

The shared event ||mu_hat_{t,J} - mu_t|| <= b_{t,J} is uniform over t and J.
B3 uses the static source class, B4 uses the time-dependent class, and Bglob
uses global contraction. Bernstein radii and the source-ellipsoid knapsack
provide additional comparisons. Population MMD and event errors use Bessel
series with tail bounds. Dynamic bounds assume the known sampler spectrum.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import resource
import time
from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy import stats

from mcmc.paperexp import circle as C
from mcmc.paperexp.runio import RunDir, utc_now


class InfeasibleSourceIntervals(ValueError):
    """The monitored intervals and the source ellipsoid have empty intersection."""


def replica_radius_mcdiarmid(J: np.ndarray | int, delta: float, kappa: float = 1.0) -> np.ndarray:
    """Appendix E.2 radius: E||mu_hat - mu|| <= kappa/sqrt(J) plus bounded differences 2 kappa/J."""
    J = np.asarray(J, dtype=np.float64)
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return kappa / np.sqrt(J) + kappa * np.sqrt(2.0 * math.log(1.0 / delta) / J)


def replica_radius_bernstein(J: np.ndarray | int, delta: float, kappa: float = 1.0) -> np.ndarray:
    """Candidate: Hilbert-space Bernstein (Pinelis 1994, D = 1) radius for the mean of J iid features.

    xi_i = phi(X_i) - mu has ||xi_i|| <= a = 2 kappa and E||xi_i||^2 = kappa_bar^2 - ||mu||^2 <= kappa^2.
    P(||sum xi_i|| >= s) <= 2 exp(-s^2 / (2 (J kappa^2 + a s / 3))); solving at level delta gives
    s = a L/3 + sqrt((a L/3)^2 + 2 L J kappa^2), L = log(2/delta); the radius is s/J.
    """
    J = np.asarray(J, dtype=np.float64)
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    big_l = math.log(2.0 / delta)
    a = 2.0 * kappa
    s = a * big_l / 3.0 + np.sqrt((a * big_l / 3.0) ** 2 + 2.0 * big_l * J * kappa * kappa)
    return s / J


def static_axes(j_max: int) -> np.ndarray:
    """a_j = sqrt(lambda_j), j = 1..j_max (Section 3 class with r = 1/2)."""
    return np.sqrt(C.eigenvalue(np.arange(1, j_max + 1)))


def dynamic_axes(j_max: int, t: int, p: float, sigma: float) -> np.ndarray:
    """a_j(t) = sqrt(lambda_j) rho_{ceil(j/2)}^t = sqrt(lambda_{j,t}) (Section 4 class)."""
    j = np.arange(1, j_max + 1)
    return np.sqrt(C.eigenvalue(j)) * C.lazy_rho(C.frequency_of_index(j), p, sigma) ** t


def _interval_square_range(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[min x^2, max x^2] over x in [lo, hi]."""
    if np.any(hi < lo):
        raise ValueError("interval with hi < lo")
    contains_zero = (lo <= 0.0) & (hi >= 0.0)
    low = np.where(contains_zero, 0.0, np.minimum(lo * lo, hi * hi))
    high = np.maximum(lo * lo, hi * hi)
    return low, high


def knapsack_sq_batch(
    lo: np.ndarray, hi: np.ndarray, axes: np.ndarray, m: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """U^2 = a_{m+1}^2 R^2 + max sum_{j<=m} (1 - a_{m+1}^2/a_j^2) x_j^2 over x_j in [lo_j, hi_j] with
    sum_{j<=m} x_j^2/a_j^2 <= R^2, for a batch of cells (leading axes ... x m).

    With y_j = x_j^2 in [l_j, u_j] this is a continuous knapsack: start from y = l (the cheapest
    feasible point) and spend the remaining source budget on larger a_j first (value per unit budget
    a_j^2 - a_{m+1}^2 is non-increasing in j because the axes are). Returns (U^2, infeasible);
    infeasible cells get NaN, never zero.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    if lo.shape != hi.shape or lo.shape[-1] != m:
        raise ValueError(f"interval arrays must end with m={m}: {lo.shape}, {hi.shape}")
    if axes.shape[-1] < m + 1:
        raise ValueError("need axes a_1..a_{m+1}")
    a_mon = axes[..., :m]
    a_next = axes[..., m]
    if np.any(np.diff(axes[..., : m + 1], axis=-1) > 0.0) or np.any(a_next <= 0.0):
        raise ValueError("axes must be positive and non-increasing")
    if np.any(a_next >= a_mon[..., -1]):
        raise ValueError("need a strict gap a_m > a_{m+1} (complete sine/cosine pairs)")
    low, high = _interval_square_range(lo, hi)
    inv_cost = a_mon * a_mon  # budget spent per unit y_j is 1/a_j^2
    weight = 1.0 - (a_next[..., None] ** 2) / inv_cost
    budget = radius * radius - np.sum(low / inv_cost, axis=-1)
    infeasible = budget < 0.0
    remaining = np.where(infeasible, 0.0, budget)
    y = low.copy()
    for j in range(m):
        add = np.minimum(high[..., j] - low[..., j], remaining * inv_cost[..., j])
        y[..., j] += add
        remaining = remaining - add / inv_cost[..., j]
    u2 = a_next**2 * radius * radius + np.sum(weight * y, axis=-1)
    u2 = np.where(infeasible, np.nan, u2)
    return u2, infeasible


def knapsack_sq(lo: np.ndarray, hi: np.ndarray, axes: np.ndarray, m: int, radius: float) -> float:
    """Scalar knapsack; raises InfeasibleSourceIntervals on an empty feasible set."""
    u2, infeasible = knapsack_sq_batch(
        np.asarray(lo)[None], np.asarray(hi)[None], np.asarray(axes)[None], m, radius
    )
    if infeasible[0]:
        raise InfeasibleSourceIntervals(
            "monitored intervals are incompatible with the source ellipsoid"
        )
    return float(u2[0])


def simulate_ensemble(task: tuple) -> dict:
    """One ensemble: J_max von Mises replicas through every checkpoint; nested-prefix summaries."""
    ens_idx, entropy, conf = task
    rng = np.random.default_rng(np.random.SeedSequence([entropy, ens_idx]))
    prefixes = list(conf["replica_counts"])
    j_max = prefixes[-1]
    checkpoints = list(conf["checkpoints"])
    n_f = int(conf["n_fourier"])
    p, sigma, kappa0 = conf["p"], conf["sigma"], conf["kappa0"]
    sym = conf["init_symmetry"]
    x = C.sample_symmetric_von_mises(rng, j_max, kappa0, sym)
    n_t, n_j = len(checkpoints), len(prefixes)
    vnorm = np.empty((n_t, n_j))
    coords = np.empty((n_t, n_j, 2 * n_f))
    inner = np.empty((n_t, n_j))
    previous = 0
    for ti, t in enumerate(checkpoints):
        x = C.advance_checkpoint(x, t - previous, rng, p, sigma)
        previous = t
        fm = C.prefix_fourier_means(x, n_f, prefixes)
        coords[ti] = C.embedding_coordinates(fm)
        c_t = C.von_mises_embedding_coefficients(kappa0, t, p, sigma, n_f, sym)
        inner[ti] = fm.real @ c_t  # <mu_hat, mu_t> = (1/J) sum_i mu_t(x_i)
        for ji, size in enumerate(prefixes):
            vnorm[ti, ji] = C.empirical_mean_embedding_sq(x[:size])
    assert np.isfinite(vnorm).all() and np.isfinite(coords).all() and np.isfinite(inner).all()
    return {"ens_idx": ens_idx, "vnorm": vnorm, "coords": coords, "inner": inner}


def _chunk_unit(task: tuple) -> str:
    chunk_idx, indices, entropy, conf, out_dir = task
    path = pathlib.Path(out_dir) / f"chunk_{chunk_idx:04d}.npz"
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    results = [simulate_ensemble((i, entropy, conf)) for i in indices]
    wall = time.perf_counter() - t0
    tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        ens_idx=np.array([r["ens_idx"] for r in results], dtype=np.int64),
        vnorm=np.stack([r["vnorm"] for r in results]),
        coords=np.stack([r["coords"] for r in results]),
        inner=np.stack([r["inner"] for r in results]),
        wall_seconds=np.array(wall),
        peak_rss_kib=np.array(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    os.replace(tmp, path)
    return str(path)


def truth_table(conf: dict) -> dict:
    """True MMD^2(pi_t, pi) by the Bessel series with tail bound, plus ||mu_t||^2 for the event."""
    partial, tails = [], []
    for t in conf["checkpoints"]:
        val, tail = C.von_mises_true_mmd_sq(
            conf["kappa0"], t, conf["p"], conf["sigma"], conf["series_terms"], conf["init_symmetry"]
        )
        partial.append(val)
        tails.append(tail)
    partial = np.array(partial)
    tails = np.array(tails)
    rel = conf["truth_rounding_rel"]
    if not (0.0 < rel < 1e-3):
        raise ValueError(f"truth_rounding_rel must be a small positive number, got {rel}")
    return {
        "mmd_sq_partial": partial,
        "mmd_sq_tail_bound": tails,
        "mmd_point": np.sqrt(partial),
        # outward-rounded enclosure: series tail plus a declared floating-point evaluation margin
        "mmd_lo": np.sqrt(partial * (1.0 - rel)),
        "mmd_hi": np.sqrt((partial + tails) * (1.0 + rel)),
        # <mu_hat, mu_t> uses n_fourier terms; |omitted| <= sup-norm tail of mu_t's series
        "inner_tail_bound": C.von_mises_embedding_sup_tail(
            conf["kappa0"], conf["n_fourier"], conf["init_symmetry"]
        ),
    }


def compute_bounds(data: dict, conf: dict) -> dict:
    """All bounds, shapes (E, T, M, J) unless noted; radius multiplier on the last axis where used."""
    checkpoints = np.array(conf["checkpoints"])
    prefixes = np.array(conf["replica_counts"], dtype=np.float64)
    m_values = list(conf["m_values"])
    n_t, n_j = checkpoints.shape[0], prefixes.shape[0]
    p, sigma, kappa = conf["p"], conf["sigma"], conf["kernel_sup"]
    delta_cell = conf["beta"] / (n_t * n_j)
    radius0 = math.sqrt(C.von_mises_radius_sq(conf["kappa0"]))
    mults = np.array(conf["radius_multipliers"], dtype=np.float64)
    coords = data["coords"]  # (E, T, J, 2 n_f)
    vnorm = data["vnorm"]  # (E, T, J)
    if 2 * conf["n_fourier"] < max(m_values) + 1:
        raise ValueError("n_fourier too small for the largest m")

    radii = {
        "mcdiarmid": np.broadcast_to(
            replica_radius_mcdiarmid(prefixes, delta_cell, kappa), (n_t, n_j)
        ),
        "bernstein": np.broadcast_to(
            replica_radius_bernstein(prefixes, delta_cell, kappa), (n_t, n_j)
        ),
    }
    rho1 = float(C.lazy_rho(1, p, sigma))
    lam1 = float(C.eigenvalue(1))
    out = {
        "radius": radius0,
        "radii": {k: np.array(v) for k, v in radii.items()},
        "delta_cell": delta_cell,
        "monitored_norm": np.empty(vnorm.shape[:2] + (len(m_values), n_j)),
        "direct": {k: np.sqrt(vnorm) + r[None] for k, r in radii.items()},
        "no_obs_static": radius0 * mults[None, :] * math.sqrt(lam1) * np.ones((n_t, 1)),
        "no_obs_dynamic": radius0 * mults[None, :] * math.sqrt(lam1) * (rho1**checkpoints)[:, None],
    }
    tails = {
        "static": np.empty((n_t, len(m_values))),
        "dynamic": np.empty((n_t, len(m_values))),
        "global": np.empty((n_t, len(m_values))),
    }
    for mi, m in enumerate(m_values):
        lam_next = float(C.eigenvalue(m + 1))
        rho_next = float(C.lazy_rho(m // 2 + 1, p, sigma))
        out["monitored_norm"][:, :, mi, :] = np.sqrt(np.sum(coords[..., :m] ** 2, axis=-1))
        tails["static"][:, mi] = radius0 * math.sqrt(lam_next)
        tails["dynamic"][:, mi] = radius0 * math.sqrt(lam_next) * rho_next**checkpoints
        tails["global"][:, mi] = radius0 * math.sqrt(lam_next) * rho1**checkpoints
    out["tails"] = tails
    # tail arrays (T, M) x radius multiplier -> (T, M, K)
    bounds = {}
    for rname, rad in radii.items():
        h = out["monitored_norm"] + rad[None, :, None, :]  # (E, T, M, J)
        bounds[f"H_{rname}"] = h
        for label, key in (("B3", "static"), ("B4", "dynamic"), ("Bglob", "global")):
            tail = tails[key][None, :, :, None, None] * mults[None, None, None, None, :]
            bounds[f"{label}_{rname}"] = np.sqrt(h[..., None] ** 2 + tail**2)  # (E, T, M, J, K)
    out["bounds"] = bounds

    # The Hilbert-norm event gives |zhat_j - z_j| <= b for each coordinate.
    knap = {}
    infeasible = {}
    if conf["knapsack"]:
        e_count = vnorm.shape[0]
        for rname, rad in radii.items():
            for cls in ("static", "dynamic"):
                u = np.full((e_count, n_t, len(m_values), n_j, mults.shape[0]), np.nan)
                bad = np.zeros_like(u, dtype=bool)
                for ti, t in enumerate(checkpoints):
                    axes_full = (
                        static_axes(2 * conf["n_fourier"])
                        if cls == "static"
                        else dynamic_axes(2 * conf["n_fourier"], int(t), p, sigma)
                    )
                    for mi, m in enumerate(m_values):
                        zhat = coords[:, ti, :, :m]  # (E, J, m)
                        half = rad[ti][None, :, None]
                        lo, hi = zhat - half, zhat + half
                        for ki, mult in enumerate(mults):
                            u2, inf = knapsack_sq_batch(
                                lo,
                                hi,
                                np.broadcast_to(axes_full[: m + 1], zhat.shape[:-1] + (m + 1,)),
                                m,
                                radius0 * mult,
                            )
                            u[:, ti, mi, :, ki] = np.sqrt(u2)
                            bad[:, ti, mi, :, ki] = inf
                knap[f"K{cls}_{rname}"] = u
                infeasible[f"K{cls}_{rname}"] = bad
    out["knapsack"] = knap
    out["knapsack_infeasible"] = infeasible
    return out


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def coverage_decisions(bound: np.ndarray, truth: dict) -> np.ndarray:
    """+1 confirmed coverage (B >= upper truth), -1 definite violation (B < lower truth), 0 unresolved.

    NaN bounds (an infeasible candidate cell) are -2: reported separately, never counted as covered.
    Truth arrays are indexed by checkpoint on axis 1 of ``bound``.
    """
    shape = [1] * bound.ndim
    shape[1] = -1
    hi = truth["mmd_hi"].reshape(shape)
    lo = truth["mmd_lo"].reshape(shape)
    dec = np.where(bound >= hi, 1, np.where(bound < lo, -1, 0))
    return np.where(np.isnan(bound), -2, dec)


def first_crossing(bound: np.ndarray, tol: float) -> np.ndarray:
    """Index of the first checkpoint (axis 1) with bound <= tol; -1 when censored (never)."""
    hit = bound <= tol
    any_hit = hit.any(axis=1)
    idx = np.argmax(hit, axis=1)
    return np.where(any_hit, idx, -1)


def _write_csv(path: pathlib.Path, rows: list[dict], design: str) -> None:
    """Include the initial-law design in each CSV row."""
    import csv

    rows = [{"design": design, **row} for row in rows]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(data: dict, conf: dict, out_dir: pathlib.Path) -> dict:
    truth = truth_table(conf)
    checkpoints = np.array(conf["checkpoints"])
    prefixes = list(conf["replica_counts"])
    m_values = list(conf["m_values"])
    mults = list(conf["radius_multipliers"])
    e_count = data["vnorm"].shape[0]
    bnd = compute_bounds(data, conf)

    # Enclose the Hilbert-norm error using the series tails and rounding margin.
    err_sq = data["vnorm"] - 2.0 * data["inner"] + truth["mmd_sq_partial"][None, :, None]
    trunc = 2.0 * truth["inner_tail_bound"] + float(truth["mmd_sq_tail_bound"].max())
    enclosure = (
        trunc
        + conf["event_rounding_abs"]
        + conf["truth_rounding_rel"] * float(truth["mmd_sq_partial"].max())
    )
    if err_sq.min() < -enclosure:
        raise RuntimeError(
            f"negative ||mu_hat - mu_t||^2 = {err_sq.min()} beyond the rounding enclosure"
        )
    err = np.sqrt(np.maximum(err_sq, 0.0))
    events = {}
    for rname, rad in bnd["radii"].items():
        confirmed = err_sq + enclosure <= rad[None] ** 2  # (E, T, J)
        violated = err_sq - enclosure > rad[None] ** 2
        margin = np.min(np.abs(err_sq - rad[None] ** 2))
        events[rname] = {
            "all_cells_per_ensemble": confirmed.all(axis=(1, 2)),
            "any_cell_violated_per_ensemble": violated.any(axis=(1, 2)),
            "unresolved_cells": int((~confirmed & ~violated).sum()),
            "cell_frequency": confirmed.mean(axis=0),
            "min_abs_margin_sq": float(margin),
        }

    rows_cov, rows_ratio, rows_paired, rows_cross, rows_pair_cross = [], [], [], [], []
    methods = {}
    for rname in ("mcdiarmid", "bernstein"):
        for label in ("B3", "B4", "Bglob"):
            methods[f"{label}_{rname}"] = bnd["bounds"][f"{label}_{rname}"]
        for key, val in bnd["knapsack"].items():
            if key.endswith(rname):
                methods[key] = val
        d = bnd["direct"][rname]  # (E, T, J)
        methods[f"direct_{rname}"] = np.broadcast_to(
            d[:, :, None, :, None],
            (e_count, len(checkpoints), len(m_values), len(prefixes), len(mults)),
        )
    primary_cells = {}
    min_cover_margin = float("inf")
    shape_t = [1, -1, 1, 1, 1]
    for name, b in methods.items():
        dec = coverage_decisions(b, truth)  # (E, T, M, J, K)
        primary_cells[name] = dec
        margins = b - truth["mmd_hi"].reshape(shape_t)
        if np.any(dec == 1):
            min_cover_margin = min(min_cover_margin, float(np.min(margins[dec == 1])))
        for ki, mult in enumerate(mults):
            for mi, m in enumerate(m_values):
                for ji, size in enumerate(prefixes):
                    d = dec[:, :, mi, ji, ki]
                    covered_all = (d == 1).all(axis=1)
                    violated = (d == -1).any(axis=1)
                    k_cov = int(covered_all.sum())
                    lo, hi = clopper_pearson(k_cov, e_count)
                    rows_cov.append(
                        {
                            "method": name,
                            "radius_mult": mult,
                            "m": m,
                            "J": size,
                            "n_ensembles": e_count,
                            "covered_all_times": k_cov,
                            "coverage": k_cov / e_count,
                            "cp95_lo": lo,
                            "cp95_hi": hi,
                            "definite_violation_any_time": int(violated.sum()),
                            "unresolved_cells": int((d == 0).sum()),
                            "infeasible_cells": int((d == -2).sum()),
                        }
                    )
            # simultaneous over every (t, m, J)
            d_all = dec[..., ki]
            k_cov = int((d_all == 1).all(axis=(1, 2, 3)).sum())
            lo, hi = clopper_pearson(k_cov, e_count)
            rows_cov.append(
                {
                    "method": name,
                    "radius_mult": mult,
                    "m": "all",
                    "J": "all",
                    "n_ensembles": e_count,
                    "covered_all_times": k_cov,
                    "coverage": k_cov / e_count,
                    "cp95_lo": lo,
                    "cp95_hi": hi,
                    "definite_violation_any_time": int((d_all == -1).any(axis=(1, 2, 3)).sum()),
                    "unresolved_cells": int((d_all == 0).sum()),
                    "infeasible_cells": int((d_all == -2).sum()),
                }
            )
    # joint coverage of B3 and B4 on the shared event (primary radius family)
    for rname in ("mcdiarmid", "bernstein"):
        for ki, mult in enumerate(mults):
            d3 = primary_cells[f"B3_{rname}"][..., ki]
            d4 = primary_cells[f"B4_{rname}"][..., ki]
            joint = ((d3 == 1) & (d4 == 1)).all(axis=(1, 2, 3))
            k_cov = int(joint.sum())
            lo, hi = clopper_pearson(k_cov, e_count)
            rows_cov.append(
                {
                    "method": f"B3&B4_{rname}",
                    "radius_mult": mult,
                    "m": "all",
                    "J": "all",
                    "n_ensembles": e_count,
                    "covered_all_times": k_cov,
                    "coverage": k_cov / e_count,
                    "cp95_lo": lo,
                    "cp95_hi": hi,
                    "definite_violation_any_time": int(
                        ((d3 == -1) | (d4 == -1)).any(axis=(1, 2, 3)).sum()
                    ),
                    "unresolved_cells": int(((d3 == 0) | (d4 == 0)).sum()),
                    "infeasible_cells": 0,
                }
            )
    for rname, ev in events.items():
        k_ev = int(ev["all_cells_per_ensemble"].sum())
        lo, hi = clopper_pearson(k_ev, e_count)
        rows_cov.append(
            {
                "method": f"EVENT_{rname}",
                "radius_mult": "n/a",
                "m": "all",
                "J": "all",
                "n_ensembles": e_count,
                "covered_all_times": k_ev,
                "coverage": k_ev / e_count,
                "cp95_lo": lo,
                "cp95_hi": hi,
                "definite_violation_any_time": e_count - k_ev,
                "unresolved_cells": 0,
                "infeasible_cells": 0,
            }
        )

    # bound / truth ratios, components separated (primary radius R, both concentration radii)
    truth_mid = truth["mmd_point"]
    for rname in ("mcdiarmid", "bernstein"):
        rad = bnd["radii"][rname]
        for ti, t in enumerate(checkpoints):
            for mi, m in enumerate(m_values):
                for ji, size in enumerate(prefixes):
                    comp = {
                        "B3": bnd["bounds"][f"B3_{rname}"][:, ti, mi, ji, 0],
                        "B4": bnd["bounds"][f"B4_{rname}"][:, ti, mi, ji, 0],
                        "Bglob": bnd["bounds"][f"Bglob_{rname}"][:, ti, mi, ji, 0],
                        "direct": bnd["direct"][rname][:, ti, ji],
                        "monitored_observed": bnd["monitored_norm"][:, ti, mi, ji],
                    }
                    if conf["knapsack"]:
                        comp["Kstatic"] = bnd["knapsack"][f"Kstatic_{rname}"][:, ti, mi, ji, 0]
                        comp["Kdynamic"] = bnd["knapsack"][f"Kdynamic_{rname}"][:, ti, mi, ji, 0]
                    row = {
                        "radius_family": rname,
                        "t": int(t),
                        "m": m,
                        "J": size,
                        "truth": float(truth_mid[ti]),
                        "uncertainty_b": float(rad[ti, ji]),
                        "static_tail": float(bnd["tails"]["static"][ti, mi]),
                        "dynamic_tail": float(bnd["tails"]["dynamic"][ti, mi]),
                        "global_tail": float(bnd["tails"]["global"][ti, mi]),
                        "no_obs_static": float(bnd["no_obs_static"][ti, 0]),
                        "no_obs_dynamic": float(bnd["no_obs_dynamic"][ti, 0]),
                    }
                    for key, val in comp.items():
                        q05, q50, q95 = np.nanquantile(val, [0.05, 0.5, 0.95])
                        row[f"{key}_median"] = float(q50)
                        row[f"{key}_q05"] = float(q05)
                        row[f"{key}_q95"] = float(q95)
                        row[f"{key}_over_truth_median"] = float(q50 / truth_mid[ti])
                    rows_ratio.append(row)
                    diff = comp["B3"] - comp["B4"]
                    n = diff.shape[0]
                    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
                    tq = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
                    rows_paired.append(
                        {
                            "radius_family": rname,
                            "t": int(t),
                            "m": m,
                            "J": size,
                            "mean_B3_minus_B4": float(diff.mean()),
                            "sd": sd,
                            "ci95_lo": float(diff.mean() - tq * sd / math.sqrt(n)),
                            "ci95_hi": float(diff.mean() + tq * sd / math.sqrt(n)),
                            "min": float(diff.min()),
                            "max": float(diff.max()),
                            "n_negative_beyond_roundoff": int((diff < -1e-12).sum()),
                        }
                    )

    # first crossing of each tolerance, cost J * t (equivalent transitions, not executed)
    no_obs = {"N3": bnd["no_obs_static"], "N4": bnd["no_obs_dynamic"]}
    for tol in conf["tolerances"]:
        for ki, mult in enumerate(mults):
            for name, b in methods.items():
                fc = first_crossing(b[..., ki], tol)  # (E, M, J)
                for mi, m in enumerate(m_values):
                    for ji, size in enumerate(prefixes):
                        f = fc[:, mi, ji]
                        reached = f >= 0
                        k = int(reached.sum())
                        lo, hi = clopper_pearson(k, e_count)
                        t_hit = checkpoints[f[reached]] if k else np.array([])
                        rows_cross.append(
                            {
                                "tolerance": tol,
                                "method": name,
                                "radius_mult": mult,
                                "m": m,
                                "J": size,
                                "n_ensembles": e_count,
                                "n_reached": k,
                                "frac_reached": k / e_count,
                                "cp95_lo": lo,
                                "cp95_hi": hi,
                                "n_censored": e_count - k,
                                "t_first_min": int(t_hit.min()) if k else -1,
                                "t_first_median": float(np.median(t_hit)) if k else float("nan"),
                                "t_first_max": int(t_hit.max()) if k else -1,
                                "equiv_transitions_median": float(np.median(t_hit) * size)
                                if k
                                else float("nan"),
                            }
                        )
            for name, b in no_obs.items():
                hit = np.nonzero(b[:, ki] <= tol)[0]
                rows_cross.append(
                    {
                        "tolerance": tol,
                        "method": name,
                        "radius_mult": mult,
                        "m": "n/a",
                        "J": 0,
                        "n_ensembles": e_count,
                        "n_reached": e_count if hit.size else 0,
                        "frac_reached": 1.0 if hit.size else 0.0,
                        "cp95_lo": float("nan"),
                        "cp95_hi": float("nan"),
                        "n_censored": 0 if hit.size else e_count,
                        "t_first_min": int(checkpoints[hit[0]]) if hit.size else -1,
                        "t_first_median": float(checkpoints[hit[0]]) if hit.size else float("nan"),
                        "t_first_max": int(checkpoints[hit[0]]) if hit.size else -1,
                        "equiv_transitions_median": 0.0 if hit.size else float("nan"),
                    }
                )
            # paired first-crossing comparisons on the same ensembles
            pairs = [("B3", "B4"), ("Bglob", "B4"), ("B3", "Bglob")]
            if conf["knapsack"]:
                pairs += [("B4", "Kdynamic"), ("B3", "Kstatic")]
            for rname in ("mcdiarmid", "bernstein"):
                for a_name, b_name in pairs:
                    fa = first_crossing(methods[f"{a_name}_{rname}"][..., ki], tol)
                    fb = first_crossing(methods[f"{b_name}_{rname}"][..., ki], tol)
                    for mi, m in enumerate(m_values):
                        for ji, size in enumerate(prefixes):
                            xa, xb = fa[:, mi, ji], fb[:, mi, ji]
                            both = (xa >= 0) & (xb >= 0)
                            gain = (
                                checkpoints[xa[both]] - checkpoints[xb[both]]
                                if both.any()
                                else np.array([])
                            )
                            rows_pair_cross.append(
                                {
                                    "tolerance": tol,
                                    "radius_family": rname,
                                    "radius_mult": mult,
                                    "first": a_name,
                                    "second": b_name,
                                    "m": m,
                                    "J": size,
                                    "both": int(both.sum()),
                                    "only_first": int(((xa >= 0) & (xb < 0)).sum()),
                                    "only_second": int(((xa < 0) & (xb >= 0)).sum()),
                                    "neither": int(((xa < 0) & (xb < 0)).sum()),
                                    "median_t_saving_when_both": float(np.median(gain))
                                    if gain.size
                                    else float("nan"),
                                    "median_equiv_transitions_saving_when_both": float(
                                        np.median(gain) * size
                                    )
                                    if gain.size
                                    else float("nan"),
                                }
                            )

    _write_csv(out_dir / "coverage.csv", rows_cov, conf["design_disclosure"])
    _write_csv(out_dir / "ratios.csv", rows_ratio, conf["design_disclosure"])
    _write_csv(out_dir / "paired_B3_minus_B4.csv", rows_paired, conf["design_disclosure"])
    _write_csv(out_dir / "crossing.csv", rows_cross, conf["design_disclosure"])
    _write_csv(out_dir / "paired_crossing.csv", rows_pair_cross, conf["design_disclosure"])
    b4_le_b3 = {
        rname: float(np.max(bnd["bounds"][f"B4_{rname}"] - bnd["bounds"][f"B3_{rname}"]))
        for rname in bnd["radii"]
    }
    infeasible_counts = {k: int(v.sum()) for k, v in bnd["knapsack_infeasible"].items()}
    truth_rows = [
        {
            "t": int(t),
            "mmd_sq_partial": float(truth["mmd_sq_partial"][i]),
            "mmd_sq_tail_bound": float(truth["mmd_sq_tail_bound"][i]),
            "mmd": float(truth["mmd_point"][i]),
            "mmd_enclosure_lo": float(truth["mmd_lo"][i]),
            "mmd_enclosure_hi": float(truth["mmd_hi"][i]),
        }
        for i, t in enumerate(checkpoints)
    ]
    _write_csv(out_dir / "truth.csv", truth_rows, conf["design_disclosure"])
    np.savez_compressed(
        out_dir / "bounds_per_ensemble.npz",
        checkpoints=checkpoints,
        m_values=np.array(m_values),
        replica_counts=np.array(prefixes),
        radius_multipliers=np.array(mults),
        truth=truth["mmd_point"],
        monitored_norm=bnd["monitored_norm"],
        event_error=err,
        **{f"radius_{k}": v for k, v in bnd["radii"].items()},
        **{f"direct_{k}": v for k, v in bnd["direct"].items()},
        **{k: v for k, v in bnd["bounds"].items() if not k.startswith("H_")},
        **bnd["knapsack"],
        no_obs_static=bnd["no_obs_static"],
        no_obs_dynamic=bnd["no_obs_dynamic"],
    )
    return {
        "n_ensembles": e_count,
        "radius": bnd["radius"],
        "delta_cell": bnd["delta_cell"],
        "replica_radius_mcdiarmid": {
            str(s): float(bnd["radii"]["mcdiarmid"][0, i]) for i, s in enumerate(prefixes)
        },
        "replica_radius_bernstein": {
            str(s): float(bnd["radii"]["bernstein"][0, i]) for i, s in enumerate(prefixes)
        },
        "event_frequency": {
            k: float(v["all_cells_per_ensemble"].mean()) for k, v in events.items()
        },
        "event_min_abs_margin_sq": {k: v["min_abs_margin_sq"] for k, v in events.items()},
        "truncation_error_bound": trunc,
        "event_enclosure_abs": enclosure,
        "event_unresolved_cells": {k: v["unresolved_cells"] for k, v in events.items()},
        "event_any_violation_ensembles": {
            k: int(v["any_cell_violated_per_ensemble"].sum()) for k, v in events.items()
        },
        "truth_rounding_rel": conf["truth_rounding_rel"],
        "event_rounding_abs": conf["event_rounding_abs"],
        "min_margin_covered_bound_minus_truth_hi": min_cover_margin,
        "design_disclosure": conf["design_disclosure"],
        "max_B4_minus_B3": b4_le_b3,
        "knapsack_infeasible_cells": infeasible_counts,
        "truth_mmd": {str(int(t)): float(truth["mmd_point"][i]) for i, t in enumerate(checkpoints)},
        "truth_tail_bound_max": float(truth["mmd_sq_tail_bound"].max()),
        "max_event_error_over_radius": {
            k: float(np.max(err / v[None])) for k, v in bnd["radii"].items()
        },
    }


def load_chunks(out_dir: pathlib.Path, n_chunks: int) -> dict:
    parts = {"ens_idx": [], "vnorm": [], "coords": [], "inner": [], "wall": [], "rss": []}
    for k in range(n_chunks):
        with np.load(out_dir / f"chunk_{k:04d}.npz") as z:
            parts["ens_idx"].append(z["ens_idx"])
            parts["vnorm"].append(z["vnorm"])
            parts["coords"].append(z["coords"])
            parts["inner"].append(z["inner"])
            parts["wall"].append(float(z["wall_seconds"]))
            parts["rss"].append(int(z["peak_rss_kib"]))
    data = {k: np.concatenate(parts[k]) for k in ("ens_idx", "vnorm", "coords", "inner")}
    order = np.argsort(data["ens_idx"])
    data = {k: v[order] for k, v in data.items()}
    if not np.array_equal(data["ens_idx"], np.arange(data["ens_idx"].shape[0])):
        raise RuntimeError("missing or duplicated ensembles in chunk files")
    data["chunk_wall_seconds"] = np.array(parts["wall"])
    data["chunk_peak_rss_kib"] = np.array(parts["rss"])
    return data


def flat_conf(cfg: DictConfig) -> dict:
    return {
        "kappa0": float(cfg.kappa0),
        "init_symmetry": int(cfg.init_symmetry),
        "p": float(cfg.sampler.p),
        "sigma": float(cfg.sampler.sigma),
        "checkpoints": [int(t) for t in cfg.checkpoints],
        "m_values": [int(m) for m in cfg.m_values],
        "replica_counts": [int(j) for j in cfg.replica_counts],
        "beta": float(cfg.beta),
        "kernel_sup": float(cfg.kernel_sup),
        "radius_multipliers": [float(x) for x in cfg.radius_multipliers],
        "tolerances": [float(x) for x in cfg.tolerances],
        "n_fourier": int(cfg.n_fourier),
        "series_terms": int(cfg.series_terms),
        "knapsack": bool(cfg.candidates.knapsack),
        "bernstein": bool(cfg.candidates.bernstein),
        "truth_rounding_rel": float(cfg.truth_rounding_rel),
        "event_rounding_abs": float(cfg.event_rounding_abs),
        "design_disclosure": str(cfg.design_disclosure),
    }


def run_study(cfg: DictConfig, resume: bool = False) -> RunDir:
    phase = str(cfg.phase)
    if phase not in cfg.phases:
        raise ValueError(f"unknown phase {phase!r}")
    ph = cfg.phases[phase]
    conf = flat_conf(cfg)
    if conf["checkpoints"] != sorted(set(conf["checkpoints"])) or conf["checkpoints"][0] != 0:
        raise ValueError("checkpoints must be strictly increasing from 0")
    if conf["replica_counts"] != sorted(set(conf["replica_counts"])):
        raise ValueError("replica counts must be strictly increasing (nested prefixes)")
    if not (conf["knapsack"] and conf["bernstein"]):
        raise ValueError("this runner reports both candidates; disable them in analysis, not here")
    run_id = cfg.run_id
    if run_id is None:
        if phase != "smoke":
            raise ValueError("run_id is required for pilot/main phases")
        run_id = "smoke_" + utc_now().replace(":", "").replace("-", "")
    n_ens = int(ph.n_ensembles)
    chunk = int(cfg.chunk_size)
    chunks = [list(range(s, min(s + chunk, n_ens))) for s in range(0, n_ens, chunk)]
    # Ignore the resume flag when comparing saved configurations.
    frozen = OmegaConf.to_container(cfg, resolve=True)
    frozen["resume"] = False
    config_text = OmegaConf.to_yaml(OmegaConf.create(frozen))
    if resume:
        run = RunDir.resume(str(cfg.study), phase, str(run_id))
        if (run.path / "config.yaml").read_text() != config_text:
            raise RuntimeError("resume with a different configuration")
    else:
        run = RunDir.create(
            str(cfg.study),
            phase,
            str(run_id),
            config_text,
            requested_units=len(chunks),
            allow_dirty=phase in ("smoke", "pilot"),
            extra_provenance={"numpy": np.__version__, "seed_entropy": int(ph.seed_entropy)},
        )
    t_start = time.perf_counter()
    cpu0 = os.times()
    try:
        tasks = [
            (k, idx, int(ph.seed_entropy), conf, str(run.path)) for k, idx in enumerate(chunks)
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=int(cfg.n_workers)) as pool:
            for _ in pool.map(_chunk_unit, tasks, chunksize=1):
                done += 1
                run.update(completed_units=done)
        t_sim = time.perf_counter() - t_start
        cpu1 = os.times()
        data = load_chunks(run.path, len(chunks))
        t_an0 = time.perf_counter()
        result = summarise(data, conf, run.path)
        t_an = time.perf_counter() - t_an0
        result.update(
            {
                "phase": phase,
                "run_id": str(run_id),
                "seed_entropy": int(ph.seed_entropy),
                "wall_seconds_simulation": t_sim,
                "wall_seconds_analysis": t_an,
                "cpu_seconds_children_simulation": (cpu1.children_user - cpu0.children_user)
                + (cpu1.children_system - cpu0.children_system),
                "chunk_wall_seconds_sum": float(data["chunk_wall_seconds"].sum()),
                "chunk_wall_seconds_per_ensemble": float(data["chunk_wall_seconds"].sum() / n_ens),
                "peak_rss_kib_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_rss_kib_largest_worker": int(data["chunk_peak_rss_kib"].max()),
                "replicas_per_ensemble": conf["replica_counts"][-1],
                "equivalent_transitions_per_ensemble_at_t_max": conf["replica_counts"][-1]
                * conf["checkpoints"][-1],
                "equivalent_transitions_total": n_ens
                * conf["replica_counts"][-1]
                * conf["checkpoints"][-1],
                "binomial_checkpoint_draws_executed_total": n_ens
                * conf["replica_counts"][-1]
                * (len(conf["checkpoints"]) - 1),
                "n_workers": int(cfg.n_workers),
            }
        )
        (run.path / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}")
        raise
    return run


SCIENTIFIC_KEYS = (
    "kappa0",
    "sampler",
    "checkpoints",
    "m_values",
    "replica_counts",
    "beta",
    "kernel_sup",
    "radius_multipliers",
    "tolerances",
    "n_fourier",
    "series_terms",
    "candidates",
    "chunk_size",
)


def run_reanalysis(cfg: DictConfig) -> RunDir:
    """Recompute summaries from a completed run without changing its saved chunks.

    Simulation settings must match. Missing init_symmetry means 1 in older runs;
    rounding margins may differ.
    """
    src = pathlib.Path(str(cfg.reanalyze_source))
    status = json.loads((src / "RUN-STATUS.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"source run {src} is {status['status']}, not COMPLETED")
    src_cfg = OmegaConf.load(src / "config.yaml")
    phase = str(cfg.phase)
    if str(src_cfg.phase) != phase:
        raise ValueError(f"phase {phase!r} differs from the source run's {src_cfg.phase!r}")
    for key in SCIENTIFIC_KEYS:
        a = (
            OmegaConf.to_container(src_cfg[key], resolve=True)
            if isinstance(src_cfg[key], DictConfig | ListConfig)
            else src_cfg[key]
        )
        b = (
            OmegaConf.to_container(cfg[key], resolve=True)
            if isinstance(cfg[key], DictConfig | ListConfig)
            else cfg[key]
        )
        if a != b:
            raise ValueError(f"config key {key!r} differs from the source run: {a!r} != {b!r}")
    src_sym = int(src_cfg.init_symmetry) if "init_symmetry" in src_cfg else 1
    if src_sym != int(cfg.init_symmetry):
        raise ValueError("init_symmetry differs from the source run")
    if OmegaConf.to_container(src_cfg.phases[phase]) != OmegaConf.to_container(cfg.phases[phase]):
        raise ValueError("phase block (ensemble count / seed entropy) differs from the source run")
    if cfg.run_id is None:
        raise ValueError("run_id is required for a reanalysis")
    conf = flat_conf(cfg)
    n_ens = int(cfg.phases[phase].n_ensembles)
    n_chunks = -(-n_ens // int(cfg.chunk_size))
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    run = RunDir.create(
        str(cfg.study),
        phase,
        str(cfg.run_id),
        config_text,
        requested_units=1,
        allow_dirty=False,
        extra_provenance={"reanalysis_of": str(src), "source_commit": status["commit"]},
    )
    try:
        t0 = time.perf_counter()
        data = load_chunks(src, n_chunks)
        result = summarise(data, conf, run.path)
        src_summary = json.loads((src / "summary.json").read_text())
        result.update(
            {
                "phase": phase,
                "run_id": str(cfg.run_id),
                "reanalysis_of": str(src),
                "source_commit": status["commit"],
                "seed_entropy": int(cfg.phases[phase].seed_entropy),
                "wall_seconds_reanalysis": time.perf_counter() - t0,
                "peak_rss_kib_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "source_simulation_cost": {
                    k: src_summary[k]
                    for k in (
                        "wall_seconds_simulation",
                        "cpu_seconds_children_simulation",
                        "chunk_wall_seconds_per_ensemble",
                        "peak_rss_kib_self",
                        "peak_rss_kib_largest_worker",
                        "equivalent_transitions_total",
                        "binomial_checkpoint_draws_executed_total",
                    )
                },
            }
        )
        (run.path / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        run.update(completed_units=1)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}")
        raise
    return run


@hydra.main(
    config_path="../../../configs/experiment",
    config_name="exp_r3_anytime_usefulness",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.reanalyze_source is not None:
        run = run_reanalysis(cfg)
    else:
        run = run_study(cfg, resume=bool(cfg.resume))
    print(f"run dir: {run.path}", flush=True)


if __name__ == "__main__":
    main()
