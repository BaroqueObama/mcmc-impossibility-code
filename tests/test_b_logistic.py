"""Tests for logistic calibration and learned-subspace bounds."""

import math

import numpy as np
import pytest

from mcmc.paperexp import b_logistic as bl
from mcmc.paperexp import german_credit_r2 as r2


def _ar_paths(rho, C, L, seed, mix=None):
    """C independent stationary paths of independent AR(1) channels (optionally linearly mixed)."""
    rng = np.random.default_rng(seed)
    rho = np.asarray(rho, dtype=np.float64)
    F = np.empty((C, L, rho.size))
    F[:, 0] = rng.normal(size=(C, rho.size))
    for t in range(1, L):
        F[:, t] = rho * F[:, t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=(C, rho.size))
    F += 3.0  # a nonzero mean exercises the centring algebra
    return F if mix is None else F @ mix.T


def _moments(F, groups, lags, center):
    mom = bl.LagMoments.empty(max(groups) + 1, F.shape[1], lags, center)
    for c, g in enumerate(groups):
        mom.add_chain(g, F[c])
    return mom


@pytest.mark.parametrize("lag", [1, 3, 17])
def test_lag_moments_match_multichain_lagged_for_every_union(lag):
    rng = np.random.default_rng(0)
    F = rng.normal(size=(6, 120, 4)).cumsum(axis=1) * 0.2 + rng.normal(size=4) * 5
    groups = [0, 0, 1, 1, 2, 2]
    mom = _moments(F, groups, (1, 3, 17), F[0].mean(0) + 0.7)  # deliberately off-centre
    for sel, chains in (([0, 1, 2], slice(0, 6)), ([0], slice(0, 2)), ([1, 2], slice(2, 6))):
        C0, Cl, asym, n = mom.covariances(sel, lag)
        C0r, Clr, asymr = r2.multichain_lagged(F[chains], lag)
        np.testing.assert_allclose(C0, C0r, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(Cl, Clr, rtol=1e-11, atol=1e-12)
        assert abs(asym - asymr) < 1e-10
        assert n == (chains.stop - chains.start) * (120 - lag)
        np.testing.assert_allclose(mom.mean(sel), F[chains].reshape(-1, 4).mean(0), rtol=1e-12)


def test_lag_moments_reject_bad_input():
    mom = bl.LagMoments.empty(1, 10, (1, 2), np.zeros(3))
    with pytest.raises(ValueError):
        mom.add_chain(0, np.zeros((9, 3)))
    with pytest.raises(TypeError):
        mom.add_chain(0, np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        bl.LagMoments.empty(1, 10, (10,), np.zeros(3))
    with pytest.raises(ValueError):
        mom.covariances([0], 1)  # empty group


def test_selection_on_known_ar_spectrum():
    """At lag 1, the largest gap-to-sensitivity ratios occur at m = 3 and m = 2.

    At lag 64 the smaller eigenvalues nearly vanish, so their gaps are unresolved."""
    rho = np.array([0.995, 0.99, 0.9, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1, 0.05])
    r = rho.size
    rng = np.random.default_rng(3)
    mix = rng.normal(size=(r, r)) + 3 * np.eye(r)
    C, L = 16, 8000
    F = _ar_paths(rho, C, L, seed=4, mix=mix)
    lags = tuple(bl.RULES["lags"])
    mom = _moments(F, [0] * (C // 2) + [1] * (C // 2), lags, F.reshape(-1, r).mean(0))
    fits = {}
    for tau in lags:
        for name, sel in (("full", [0, 1]), ("A", [0]), ("B", [1])):
            C0, Cl, asym, _ = mom.covariances(sel, tau)
            fits[(name, tau)] = bl.tica_fit(C0, Cl, 1e-10, 17, asym)
    # recovered eigenvalues at lag 1 approximate rho (sorted)
    np.testing.assert_allclose(fits[("full", 1)].rho[:3], rho[:3], atol=0.01)
    rules = dict(bl.RULES, m_candidates=list(range(1, 8)))
    rows = bl.candidate_table(fits, rules)
    sel = bl.select_counts(rows, rules)
    assert sel["status"] == "SELECTED" and sel["lag"] == 1
    assert sel["selected_m"] == [3, 2], sel
    by = {(row["lag"], row["m"]): row for row in rows}
    np.testing.assert_allclose(fits[("full", 4)].rho[:3], rho[:3] ** 4, atol=0.02)
    # lag 64: rho_3^64 ~ 1e-3 and the rest ~ 0 -> no gap after m = 3, and mode 3 is lost
    assert not by[(64, 3)]["eligible"] and not by[(64, 3)]["rule1"]
    # lag 16, m = 3: mode 3 (0.9^16 = 0.19) is not represented at lag 64 -> rule 2 fails
    assert by[(16, 3)]["rule1"] and not by[(16, 3)]["rule2"]
    # lag-64 candidates are checked against lag 256; no lag-256 candidates exist
    assert all(row["lag"] in (1, 4, 16, 64) for row in rows)
    assert by[(64, 2)]["next_lag"] == 256


def _primary_study_unit(rho_a, rho_b, m):
    """Half the largest eigenvalue difference over the first m+2 modes."""
    k = min(m + 2, rho_a.size, rho_b.size)
    a = np.sort(np.abs(rho_a))[::-1][:k]
    b = np.sort(np.abs(rho_b))[::-1][:k]
    return float(0.5 * np.max(np.abs(a - b)))


def test_half_split_sensitivity_v1_v2():
    rng = np.random.default_rng(7)
    for _ in range(50):
        ra = -np.sort(-(rng.uniform(-1, 1, 20) ** 2)) * rng.choice([-1, 1], 20)
        ra = ra[np.argsort(-np.abs(ra), kind="stable")]
        rb = ra + rng.normal(scale=0.01, size=20)
        rb = rb[np.argsort(-np.abs(rb), kind="stable")]
        for m in range(1, 18):
            v2 = bl.half_split_sensitivity(ra, rb, m, "1..m+2")
            assert v2 == _primary_study_unit(ra, rb, m)
            v1 = bl.half_split_sensitivity(ra, rb, m, "m,m+1")
            aa, ab = np.abs(ra), np.abs(rb)
            assert v1 == max(abs(aa[m - 1] - ab[m - 1]), abs(aa[m] - ab[m])) / 2
            assert v2 >= v1  # superset of indices (m + 1 <= m + 2)
    with pytest.raises(ValueError):
        bl.half_split_sensitivity(np.ones(2), np.ones(2), 2, "1..m+2")
    with pytest.raises(ValueError):
        bl.half_split_sensitivity(np.ones(5), np.ones(5), 1, "bogus")


def test_v2_rules_differ_from_v1_only_in_the_sensitivity_unit():
    diff = {k for k in set(bl.RULES) | set(bl.RULES_V2) if bl.RULES.get(k) != bl.RULES_V2.get(k)}
    assert diff == {
        "version",
        "sensitivity_indices",
        "sensitivity_unit",
        "amends",
        "rule2_addition",
    }


def test_select_counts_tie_break_and_none():
    rows = [
        {"lag": 4, "m": 5, "eligible": True, "gap_ratio": 9.0},
        {"lag": 4, "m": 3, "eligible": True, "gap_ratio": 9.0},
        {"lag": 4, "m": 1, "eligible": True, "gap_ratio": 20.0},
        {"lag": 16, "m": 2, "eligible": True, "gap_ratio": 99.0},
        {"lag": 1, "m": 2, "eligible": False, "gap_ratio": 99.0},
    ]
    sel = bl.select_counts(rows)
    assert sel["lag"] == 4 and sel["selected_m"] == [1, 3]
    none = bl.select_counts([dict(r, eligible=False) for r in rows])
    assert none["status"] == "NONE_ELIGIBLE" and none["selected_m"] == []


def test_window_projection_scaling():
    """Diagonal operators: gap and half-split norms are exact; fresh-budget projection by sqrt."""
    lam = np.array([1.0, 0.6, 0.5, 0.1])
    op = np.diag(lam)[None]
    D = np.diag([0.02, -0.02, 0.0, 0.0])[None]
    rows = bl.window_projection(op, op + D / 2, op - D / 2, [3], 1, 8192, 8192, 4)
    row = rows[0]
    assert row["gap"] == pytest.approx(0.4)
    assert row["half_diff_op"] == pytest.approx(0.02)
    assert row["e_saved"] == pytest.approx(0.03)
    # halves: 4096 saved vs 16384 fresh blocks -> factor 1/2; half budget 8192 -> 1/sqrt(2)
    assert row["e_fresh"] == pytest.approx(0.015)
    assert row["e_fresh_half_budget"] == pytest.approx(0.03 / math.sqrt(2))
    assert row["dk_applicable_fresh"]
    # e scales as sqrt(8192 / N): gap = 3 e at N = 8192 (3 * 0.03 / 0.4)^2
    assert row["n_blocks_needed_informative"] == pytest.approx(8192 * (0.09 / 0.4) ** 2)
    assert row["gap_resolved_saved"]
    assert row["dk_term_deg_fresh"] == pytest.approx(math.degrees(math.asin(0.015 / 0.385)))
    small = bl.window_projection(
        np.diag([1.0, 0.99, 0.5, 0.1])[None], op + D / 2, op - D / 2, [3], 1, 8192, 8192, 4
    )
    assert not small[0]["dk_applicable_fresh"] and small[0]["dk_term_deg_fresh"] == 90.0


def _rot(r, seed):
    q, rr = np.linalg.qr(np.random.default_rng(seed).normal(size=(r, r)))
    return q * np.sign(np.diag(rr))


def test_reference_at_t_and_dk_envelope_contain_true_angle():
    """The Davis-Kahan envelope contains the true angle when ||S_hat - S|| <= e."""
    r, m = 12, 2
    U = _rot(r, 0)
    lam = np.array([1.0, 0.8, 0.3, 0.25] + [0.1] * (r - 4))
    S = U @ np.diag(lam) @ U.T
    Vtrue = U[:, :m]
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(40):
        noise = []
        for _ in range(4):
            E = rng.normal(scale=0.01, size=(r, r))
            noise.append(S + (E + E.T) / 2)
        ref = bl.reference_at_t(np.stack(noise), m)
        assert ref["applicable"] and ref["stable"]
        assert ref["e"] == pytest.approx(
            1.5 * np.linalg.norm((noise[0] + noise[1]) / 2 - (noise[2] + noise[3]) / 2, 2)
        )
        W, _ = np.linalg.qr(Vtrue + rng.normal(scale=0.2, size=(r, m)))
        row = bl.envelope_row({"m": m, "t": 1, "fit": trial}, W, ref, "")
        assert row["estimate_status"] == "EMPIRICAL_SENSITIVITY_ONLY"
        assert row["theorem_bound_status"] == "UNAVAILABLE"
        true_angle = math.degrees(math.asin(bl.largest_sine(W, Vtrue)))
        Shat = np.mean(noise, axis=0)
        if np.linalg.norm(Shat - S, 2) <= ref["e"]:
            hits += 1
            assert true_angle <= row["learned_angle_upper"] + 1e-9
        assert row["sin_theta"] == pytest.approx(math.sin(math.radians(row["angle_deg"])))
    assert hits >= 30  # the premise holds in most trials at this noise level


def test_reference_not_applicable_and_failed_fit_rows():
    r, m = 6, 1
    S = np.diag([1.0, 0.999, 0.5, 0.4, 0.3, 0.2])
    rng = np.random.default_rng(2)
    B = []
    for _ in range(4):
        E = rng.normal(scale=0.01, size=(r, r))
        B.append(S + (E + E.T) / 2)
    ref = bl.reference_at_t(np.stack(B), m)
    assert not ref["applicable"] and ref["dk_deg"] is None and not ref["stable"]
    W = np.eye(r)[:, :1]
    row = bl.envelope_row({"m": m}, W, ref, "BEFORE_RESOLVED_WINDOW")
    assert row["estimate_status"] == "UNAVAILABLE" and math.isnan(row["learned_sin_upper"])
    assert row["cutoff_reason"].startswith("BEFORE_RESOLVED_WINDOW; REFERENCE_GAP_UNRESOLVED")
    assert not math.isnan(row["sin_theta"])  # point estimate kept
    bad = bl.envelope_row({"m": m}, None, ref, "")
    assert bad["estimate_status"] == "FAILED_FIT" and math.isnan(bad["sin_theta"])
    assert set(bad) == set(bl.ANGLE_COLUMNS)
    with pytest.raises(ValueError):
        bl.reference_at_t(np.stack(B[:3]), m)


def test_trivial_envelope_status():
    r, m = 6, 1
    S = np.diag([1.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    ref = bl.reference_at_t(np.stack([S] * 4), m)
    assert ref["e"] == 0.0 and ref["dk_deg"] == 0.0 and ref["stable"]
    W = np.eye(r)[:, 1:2]  # orthogonal to V -> 90 deg
    row = bl.envelope_row({"m": m}, W, ref, "")
    assert row["angle_deg"] == pytest.approx(90.0)
    assert row["estimate_status"] == "TRIVIAL(EMPIRICAL_SENSITIVITY_ONLY)"


def test_curve_cutoffs():
    grid = [0, 1, 2, 4, 8, 16]
    app = {0: False, 1: True, 2: True, 4: False, 8: True, 16: False}
    c = bl.curve_cutoffs(grid, app)
    assert c == {
        0: "BEFORE_RESOLVED_WINDOW",
        1: "",
        2: "",
        4: "FIRST_FAILED_CHECKPOINT",
        8: "BEYOND_FIRST_FAILED_CHECKPOINT(t=4)",
        16: "BEYOND_FIRST_FAILED_CHECKPOINT(t=4)",
    }
    assert set(bl.curve_cutoffs(grid, dict.fromkeys(grid, False)).values()) == {
        "NO_RESOLVED_CHECKPOINT"
    }
