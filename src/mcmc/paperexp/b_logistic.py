"""TICA selection and learned-span alignment on logistic posteriors.

TICA gaps are finite-feature estimates, not certified Markov-operator gaps.
The reference-optimal spans use branched chains from NUTS reference states;
Davis-Kahan envelopes report empirical sensitivity only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mcmc.paperexp import slowmodes as sm

RULES = {
    "version": "b-logistic-select-v1",
    "lags": [1, 4, 16, 64, 256],
    "candidate_lags": [1, 4, 16, 64],
    "stability_lag_factor": 4,
    "m_candidates": list(range(1, 17)),
    "gap_factor": 5.0,
    "max_half_split_angle_deg": 10.0,
    "max_lag_stability_angle_deg": 10.0,
    "max_selected": 2,
    "sensitivity_unit": "max_{j in {m,m+1}} | |rho_j^A| - |rho_j^B| | / 2",
    "ordering": "TICA modes by |rho| descending (slowmodes.tica_symmetric order='abs')",
    "centering": "each chain set (full, half A, half B) centred by its own pooled mean",
    "estimator": "pair-symmetrised multi-chain lagged covariances (german_credit_r2.multichain_lagged)",
    "rule4": "UNAVAILABLE: no known transition spectrum; any gap is a finite-feature empirical gap",
}

# V2 takes the maximum over j=1..m+2; V1 uses only m and m+1.
RULES_V2 = {
    **RULES,
    "version": "b-logistic-select-v2",
    "sensitivity_indices": "1..m+2",
    "sensitivity_unit": "max_{j <= m+2} | |rho_j^A| - |rho_j^B| | / 2 (each half sorted by |rho|)",
    "amends": "b-logistic-select-v1 (sensitivity indices)",
    "rule2_addition": "strict |rho_m| > |rho_{m+1}| at the longer lag",
}


def half_split_sensitivity(rho_a: np.ndarray, rho_b: np.ndarray, m: int, indices: str) -> float:
    """Half the maximum difference of sorted |rho| over the selected 1-based indices."""
    a = np.sort(np.abs(rho_a))[::-1]
    b = np.sort(np.abs(rho_b))[::-1]
    if min(a.size, b.size) < m + 1:
        raise ValueError(f"need at least m+1={m + 1} eigenvalues in each half")
    if indices == "m,m+1":
        return float(max(abs(a[m - 1] - b[m - 1]), abs(a[m] - b[m])) / 2.0)
    if indices == "1..m+2":
        k = min(m + 2, a.size, b.size)
        return float(0.5 * np.max(np.abs(a[:k] - b[:k])))
    raise ValueError(f"unknown sensitivity indices {indices!r}")


# lagged moments of chain sets
@dataclass
class LagMoments:
    """Sufficient statistics per group of chains (all chains of equal length L), centre c.

    With y = f - c: ``n_chains[g]``; ``total[g] = sum y`` over all states; ``gram[g] = sum y y^T``
    over all states; per lag tau (index k): ``cross[k, g] = sum_s y_s y_{s+tau}^T``,
    ``head[k, g] = sum_{s < L - tau} y_s``, ``tail[k, g] = sum_{s >= tau} y_s``,
    ``edge_head[k, g] = sum_{s < tau} y y^T`` and ``edge_tail[k, g] = sum_{s >= L - tau} y y^T``.
    """

    L: int
    lags: tuple
    center: np.ndarray
    n_chains: np.ndarray
    total: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    head: np.ndarray
    tail: np.ndarray
    edge_head: np.ndarray
    edge_tail: np.ndarray

    @classmethod
    def empty(cls, n_groups: int, L: int, lags, center: np.ndarray) -> "LagMoments":
        lags = tuple(int(t) for t in lags)
        if any(not (1 <= t < L) for t in lags):
            raise ValueError(f"lags {lags} must satisfy 1 <= lag < L={L}")
        r = center.size
        k = len(lags)
        return cls(
            L=int(L),
            lags=lags,
            center=sm.as_f64("center", center, 1),
            n_chains=np.zeros(n_groups, dtype=np.int64),
            total=np.zeros((n_groups, r)),
            gram=np.zeros((n_groups, r, r)),
            cross=np.zeros((k, n_groups, r, r)),
            head=np.zeros((k, n_groups, r)),
            tail=np.zeros((k, n_groups, r)),
            edge_head=np.zeros((k, n_groups, r, r)),
            edge_tail=np.zeros((k, n_groups, r, r)),
        )

    def add_chain(self, g: int, F: np.ndarray) -> None:
        """Accumulate one chain's feature path F (L, r) into group g."""
        F = sm.as_f64("F", F, 2)
        if F.shape != (self.L, self.center.size):
            raise ValueError(f"chain path shape {F.shape}, expected ({self.L}, {self.center.size})")
        Y = F - self.center
        self.n_chains[g] += 1
        self.total[g] += Y.sum(0)
        self.gram[g] += Y.T @ Y
        L = self.L
        for k, tau in enumerate(self.lags):
            self.cross[k, g] += Y[: L - tau].T @ Y[tau:]
            self.head[k, g] += Y[: L - tau].sum(0)
            self.tail[k, g] += Y[tau:].sum(0)
            self.edge_head[k, g] += Y[:tau].T @ Y[:tau]
            self.edge_tail[k, g] += Y[L - tau :].T @ Y[L - tau :]

    def mean(self, groups) -> np.ndarray:
        groups = np.asarray(groups)
        n = int(self.n_chains[groups].sum()) * self.L
        return self.center + self.total[groups].sum(0) / n

    def covariances(self, groups, lag: int):
        """(C0, Clag, lag_asymmetry, n_pairs) of the union of ``groups``, centred by its own pooled
        mean; identical to ``german_credit_r2.multichain_lagged`` on those chains."""
        groups = np.asarray(groups)
        if (self.n_chains[groups] == 0).any():
            raise ValueError(f"empty group among {groups.tolist()}")
        k = self.lags.index(int(lag))
        C = int(self.n_chains[groups].sum())
        n = C * (self.L - lag)
        delta = self.total[groups].sum(0) / (C * self.L)  # set mean - centre
        a = self.head[k, groups].sum(0)
        b = self.tail[k, groups].sum(0)
        G = self.gram[groups].sum(0)
        dd = n * np.outer(delta, delta)
        cross = self.cross[k, groups].sum(0) - np.outer(a, delta) - np.outer(delta, b) + dd
        H0 = G - self.edge_tail[k, groups].sum(0) - np.outer(a, delta) - np.outer(delta, a) + dd
        H1 = G - self.edge_head[k, groups].sum(0) - np.outer(b, delta) - np.outer(delta, b) + dd
        C0 = (H0 + H1) / (2.0 * n)
        Clag = (cross + cross.T) / (2.0 * n)
        return 0.5 * (C0 + C0.T), Clag, sm.relative_asymmetry(cross), n


# TICA candidates and the selection rules
@dataclass
class TicaFit:
    """Top modes of one symmetric TICA fit (|rho| descending)."""

    rho: np.ndarray  # (k,) leading eigenvalues (k = min(n_keep, rank))
    embedded: np.ndarray  # (r, k) embedded slow directions C0 u_j
    rank: int
    n_dropped: int
    lag_asymmetry: float
    whitened_asymmetry: float
    max_abs_rho: float


def tica_fit(C0: np.ndarray, Clag: np.ndarray, rel_cutoff: float, n_keep: int, asym: float):
    modes = sm.tica_symmetric(C0, Clag, rel_cutoff, n_modes=None, order="abs", lag_asymmetry=asym)
    k = min(n_keep, modes.whitening.rank)
    return TicaFit(
        rho=modes.eigenvalues[:k].copy(),
        embedded=modes.embedded[:, :k].copy(),
        rank=modes.whitening.rank,
        n_dropped=modes.whitening.n_dropped,
        lag_asymmetry=modes.lag_asymmetry,
        whitened_asymmetry=modes.whitened_asymmetry,
        max_abs_rho=modes.max_abs_eigenvalue,
    )


def largest_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    return math.degrees(sm.largest_principal_angle(A, B))


def candidate_table(fits: dict, rules: dict = RULES) -> list[dict]:
    """Rows for every (candidate lag, m). ``fits[(set, lag)]`` = TicaFit, set in full / A / B."""
    rows = []
    fac = int(rules["stability_lag_factor"])
    lags = [int(t) for t in rules["lags"]]
    for tau in [int(t) for t in rules["candidate_lags"]]:
        full, A, B = fits[("full", tau)], fits[("A", tau)], fits[("B", tau)]
        nxt = tau * fac
        fn = fits.get(("full", nxt)) if nxt in lags else None
        for m in [int(v) for v in rules["m_candidates"]]:
            row = {"lag": tau, "m": m, "next_lag": nxt if fn is not None else None}
            kmax = min(full.rho.size, A.rho.size, B.rho.size)
            if m + 1 > kmax:
                row.update(eligible=False, reason=f"m+1={m + 1} exceeds retained TICA rank {kmax}")
                rows.append(row)
                continue
            af = np.abs(full.rho)
            g = float(af[m - 1] - af[m])
            s = half_split_sensitivity(
                A.rho, B.rho, m, str(rules.get("sensitivity_indices", "m,m+1"))
            )
            ang_ab = largest_angle_deg(A.embedded[:, :m], B.embedded[:, :m])
            row.update(
                rho_m=float(full.rho[m - 1]),
                rho_m1=float(full.rho[m]),
                rho_m_A=float(A.rho[m - 1]),
                rho_m1_A=float(A.rho[m]),
                rho_m_B=float(B.rho[m - 1]),
                rho_m1_B=float(B.rho[m]),
                gap=g,
                sensitivity=s,
                gap_ratio=(g / s) if s > 0 else (math.inf if g > 0 else math.nan),
                angle_half_deg=ang_ab,
            )
            rule1_gap = g > float(rules["gap_factor"]) * s
            rule1_ang = ang_ab <= float(rules["max_half_split_angle_deg"])
            reasons = []
            if not rule1_gap:
                reasons.append(f"gap {g:.3g} <= {rules['gap_factor']} x sensitivity {s:.3g}")
            if not rule1_ang:
                reasons.append(
                    f"half-split angle {ang_ab:.1f} deg > {rules['max_half_split_angle_deg']}"
                )
            if fn is None:
                rule2 = False
                reasons.append(f"no longer lag {nxt} for the stability check")
            elif m + 1 > fn.rho.size:
                rule2 = False
                reasons.append(f"m+1 exceeds retained rank at lag {nxt}")
            else:
                an = np.abs(fn.rho)
                ang_next = largest_angle_deg(full.embedded[:, :m], fn.embedded[:, :m])
                row.update(
                    rho_m_next=float(fn.rho[m - 1]),
                    rho_m1_next=float(fn.rho[m]),
                    gap_next=float(an[m - 1] - an[m]),
                    angle_next_deg=ang_next,
                )
                pos = an[m - 1] > an[m]
                ok = ang_next <= float(rules["max_lag_stability_angle_deg"])
                rule2 = bool(pos and ok)
                if not pos:
                    reasons.append(f"no gap at lag {nxt}")
                if not ok:
                    reasons.append(
                        f"lag-{nxt} span angle {ang_next:.1f} deg > {rules['max_lag_stability_angle_deg']}"
                    )
            row.update(
                rule1=bool(rule1_gap and rule1_ang),
                rule2=rule2,
                eligible=bool(rule1_gap and rule1_ang and rule2),
                reason="; ".join(reasons) if reasons else "eligible",
            )
            rows.append(row)
    return rows


def select_counts(rows: list[dict], rules: dict = RULES) -> dict:
    """Rule 3: smallest eligible lag, then up to ``max_selected`` m by gap ratio (ties: smaller m)."""
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        return {"status": "NONE_ELIGIBLE", "lag": None, "selected_m": [], "eligible": []}
    lag = min(r["lag"] for r in elig)
    at = sorted((r for r in elig if r["lag"] == lag), key=lambda r: (-r["gap_ratio"], r["m"]))
    chosen = at[: int(rules["max_selected"])]
    return {
        "status": "SELECTED",
        "lag": lag,
        "selected_m": [r["m"] for r in chosen],
        "selected_gap_ratio": [r["gap_ratio"] for r in chosen],
        "eligible": [(r["lag"], r["m"]) for r in elig],
    }


# reference-operator window projection (saved branched-pair Sigma_t)
def window_projection(
    ops: np.ndarray,
    ops_a: np.ndarray,
    ops_b: np.ndarray,
    grid,
    m: int,
    n_blocks_saved: int,
    n_blocks_fresh: int,
    n_batches_fresh: int,
    e_factor: float = 1.5,
) -> list[dict]:
    """Project the saved operator gap and half-split error to a larger block budget.

    Uses e_t = e_factor * ||S^A-S^B||_op and assumes error scales as N^-1/2.
    The Davis-Kahan term is arcsin(e_t/(Delta_t-e_t)) when Delta_t > 2e_t,
    otherwise 90 degrees. Half-budget results use half the fresh batches.
    """
    out = []
    half_saved = n_blocks_saved // 2
    half_fresh = n_batches_fresh * n_blocks_fresh // 2
    half_half = (n_batches_fresh // 2) * n_blocks_fresh // 2
    for i, t in enumerate(grid):
        lam = np.linalg.eigvalsh(0.5 * (ops[i] + ops[i].T))[::-1]
        dAB = float(np.linalg.norm(ops_a[i] - ops_b[i], 2))
        gap = float(lam[m - 1] - lam[m])
        row = {
            "t": int(t),
            "m": int(m),
            "lambda_m": float(lam[m - 1]),
            "lambda_m1": float(lam[m]),
            "gap": gap,
            "half_diff_op": dAB,
            "eps_saved": dAB / 2.0,
            "e_saved": e_factor * dAB,
        }
        for tag, nh in (("fresh", half_fresh), ("fresh_half_budget", half_half)):
            e = e_factor * dAB * math.sqrt(half_saved / nh)
            appl = gap > 2.0 * e
            row[f"e_{tag}"] = e
            row[f"snr_{tag}"] = gap / e
            row[f"dk_applicable_{tag}"] = bool(appl)
            row[f"dk_term_deg_{tag}"] = (
                math.degrees(math.asin(min(1.0, e / (gap - e)))) if appl else 90.0
            )
        row["snr_saved"] = gap / row["e_saved"]
        row["dk_applicable_saved"] = bool(gap > 2.0 * row["e_saved"])
        # Projected block counts are unreliable if the saved gap is unresolved.
        # Davis-Kahan applies at gap > 2e and gives <= 30 degrees at gap >= 3e.
        row["gap_resolved_saved"] = bool(gap > 3.0 * row["eps_saved"])
        for tag, k in (("applicable", 2.0), ("informative", 3.0)):
            row[f"n_blocks_needed_{tag}"] = (
                n_blocks_saved * (k * row["e_saved"] / gap) ** 2 if gap > 0 else math.inf
            )
        out.append(row)
    return out


# Logistic spans are learned; true-mode and theorem-bound columns are unavailable.
ANGLE_COLUMNS = (
    "model",
    "dataset",
    "sampler",
    "feature_rank",
    "m",
    "t",
    "span_kind",
    "fit",
    "pilot_length",
    "lag",
    "sin_theta",
    "sin_uncertainty",
    "angle_deg",
    "sin_angle_upper",
    "angle_upper",
    "theorem_bound_status",
    "mode_error_deg",
    "learned_sin_upper",
    "learned_angle_upper",
    "lambda_m",
    "lambda_m1",
    "eigengap",
    "rate_a",
    "rate_b",
    "c_m_squared",
    "kappa_squared",
    "numerical_allowance",
    "empirical_sensitivity",
    "estimate_status",
    "cutoff_reason",
)


def largest_sine(W: np.ndarray, V: np.ndarray) -> float:
    """sin of the largest principal angle between orthonormal W (r, m) and V (r, m), sine route."""
    R = W - V @ (V.T @ W)
    s = float(np.linalg.svd(R, compute_uv=False)[0])
    if s > 1.0 + 1e-12:
        raise FloatingPointError(f"sine {s} exceeds 1: bases not orthonormal")
    return min(s, 1.0)


def dk_term_deg(gap: float, e: float) -> float | None:
    """arcsin(e / (gap - e)) in degrees if Davis-Kahan applies (gap > 2 e), else None."""
    if not gap > 2.0 * e:
        return None
    return math.degrees(math.asin(min(1.0, e / (gap - e))))


def reference_at_t(
    batches: np.ndarray,
    m: int,
    halves=((0, 1), (2, 3)),
    half_budgets=((0, 2), (1, 3)),
    e_factor: float = 1.5,
) -> dict:
    """Leading m-dimensional span from independent operator batches (K,r,r).

    Error proxy e = e_factor * ||S^A-S^B||_op. Each half-budget estimate is
    consistent with the full estimate when both Davis-Kahan terms apply and
    their sum covers the angle between the estimated spans. This measures
    empirical sensitivity; it does not establish coverage.
    """
    ops = np.asarray(batches, dtype=np.float64)
    K = ops.shape[0]
    used = sorted(i for h in halves for i in h)
    if used != list(range(K)):
        raise ValueError(f"halves {halves} must partition the {K} batches")
    mean = lambda idx: np.mean(ops[list(idx)], axis=0)  # noqa: E731
    S = mean(range(K))
    lam, Q = sm.eigh_desc(0.5 * (S + S.T), "S_t", max_rel_asym=1e-12)
    V = Q[:, :m]
    e = e_factor * float(np.linalg.norm(mean(halves[0]) - mean(halves[1]), 2))
    gap = float(lam[m - 1] - lam[m])
    dk = dk_term_deg(gap, e)
    out = {
        "lambda_m": float(lam[m - 1]),
        "lambda_m1": float(lam[m]),
        "gap": gap,
        "e": e,
        "dk_deg": dk,
        "applicable": dk is not None,
        "most_negative": float(min(lam.min(), 0.0)),
        "V": V,
        "half_budget": [],
    }
    for a, b in half_budgets:
        Sh = mean((a, b))
        lh, Qh = sm.eigh_desc(0.5 * (Sh + Sh.T), "S_t half", max_rel_asym=1e-12)
        eh = e_factor * float(np.linalg.norm(ops[a] - ops[b], 2))
        gh = float(lh[m - 1] - lh[m])
        dkh = dk_term_deg(gh, eh)
        ang = math.degrees(math.asin(largest_sine(Qh[:, :m], V)))
        consistent = dk is not None and dkh is not None and ang <= dk + dkh
        out["half_budget"].append(
            {
                "batches": [a, b],
                "gap": gh,
                "e": eh,
                "dk_deg": dkh,
                "applicable": dkh is not None,
                "angle_to_full_deg": ang,
                "consistent": bool(consistent),
            }
        )
    out["stable"] = bool(all(h["consistent"] for h in out["half_budget"]))
    return out


def curve_cutoffs(grid, applicable: dict) -> dict:
    """Label checkpoints outside the first consecutive Davis-Kahan-applicable window.

    Returns t -> exclusion reason, with an empty string inside the window.
    """
    grid = sorted(grid)
    ok = [bool(applicable[t]) for t in grid]
    out = {t: "BEFORE_RESOLVED_WINDOW" for t in grid}
    if not any(ok):
        return {t: "NO_RESOLVED_CHECKPOINT" for t in grid}
    i0 = ok.index(True)
    fail = None
    for i in range(i0, len(grid)):
        t = grid[i]
        if fail is None and ok[i]:
            out[t] = ""
        else:
            if fail is None:
                fail = t
            out[t] = (
                f"BEYOND_FIRST_FAILED_CHECKPOINT(t={fail})"
                if t != fail
                else "FIRST_FAILED_CHECKPOINT"
            )
    return out


def envelope_row(base: dict, W: np.ndarray | None, ref: dict, cutoff: str) -> dict:
    """One angles_logistic row for a learned span W (orthonormal (r, m) or None = failed fit)."""
    row = {k: math.nan for k in ANGLE_COLUMNS}
    row.update(base)
    row.update(
        span_kind="learned",
        sin_angle_upper=math.nan,
        angle_upper=math.nan,
        theorem_bound_status="UNAVAILABLE",
        mode_error_deg=math.nan,
        lambda_m=ref["lambda_m"],
        lambda_m1=ref["lambda_m1"],
        eigengap=ref["gap"],
        empirical_sensitivity=ref["e"],
    )
    reasons = []
    if W is None:
        row.update(estimate_status="FAILED_FIT")
        reasons.append("learned TICA span unavailable for this fit")
    else:
        s = largest_sine(W, ref["V"])
        theta = math.degrees(math.asin(s))
        row.update(sin_theta=s, angle_deg=theta)
        if ref["applicable"]:
            up = min(90.0, theta + ref["dk_deg"])
            row.update(
                sin_uncertainty=math.sin(math.radians(ref["dk_deg"])),
                learned_angle_upper=up,
                learned_sin_upper=math.sin(math.radians(up)),
                estimate_status=(
                    "EMPIRICAL_SENSITIVITY_ONLY"
                    if up < 90.0
                    else "TRIVIAL(EMPIRICAL_SENSITIVITY_ONLY)"
                ),
            )
            if not ref["stable"]:
                reasons.append("DK_INCONSISTENT_AT_HALF_BUDGET")
        else:
            row.update(estimate_status="UNAVAILABLE")
            reasons.append("REFERENCE_GAP_UNRESOLVED: eigengap <= 2 e_t (DK not applicable)")
    if cutoff:
        reasons.insert(0, cutoff)
    row["cutoff_reason"] = "; ".join(reasons)
    return row
