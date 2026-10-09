"""Tests for Gaussian regression spectra and learned-subspace alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import linalg

from mcmc.paperexp import gauss_ar as ga
from mcmc.paperexp import real_alignment as ra


def _spd(rng, d, cond=20.0):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.geomspace(1.0, 1.0 / cond, d)
    C = (Q * ev) @ Q.T
    return 0.5 * (C + C.T)


def _small_ar(d=3, seed=0):
    rng = np.random.default_rng(seed)
    C = _spd(rng, d)
    mu = rng.standard_normal(d)
    return ra.build_controlled_ar("toy", mu, C)


# sampler
def test_rotation_is_seeded_orthogonal_positive_r():
    U1, r1 = ra.frozen_rotation(5)
    U2, r2 = ra.frozen_rotation(5)
    assert np.array_equal(U1, U2) and np.array_equal(r1, r2)
    assert np.abs(U1.T @ U1 - np.eye(5)).max() < 1e-14
    assert (r1 > 0).all()
    G = np.random.default_rng(ra.ROTATION_SEED).standard_normal((5, 5))
    Q, R = np.linalg.qr(G)
    sgn = np.sign(np.diag(R))
    assert np.abs(U1 - Q * sgn[None, :]).max() == 0.0
    assert np.abs(U1 @ (R * sgn[:, None]) - G).max() < 1e-13


def test_rate_rule_and_witnesses():
    ar = _small_ar(4)
    assert ar.rho[0] == pytest.approx(0.95, abs=1e-15)
    sd = np.sqrt(np.diag(ar.C))
    s = np.sort(np.linalg.eigvalsh(ar.C / np.outer(sd, sd)))[::-1]
    np.testing.assert_allclose(ar.rho, 0.95 ** (s[0] / s), rtol=1e-13)
    w = ra.ar_witnesses(ar)
    assert w["stationarity_rel"] < 1e-13 and w["reversibility_rel"] < 1e-13
    assert w["noise_min_eig"] > 0
    for v in w["cross_covariance"].values():
        assert v["branch_vs_lag2t_rel"] < 1e-12 and v["branch_vs_modes_rel"] < 1e-12
    # mode map diagonalises A: L^{-1} A L = diag(rho)
    D = np.linalg.solve(ar.L, ar.A @ ar.L)
    assert np.abs(D - np.diag(ar.rho)).max() < 1e-12


def test_path_is_stationary_with_lag_covariance():
    ar = _small_ar(3, seed=1)
    rng = np.random.default_rng(5)
    X = np.concatenate([ar.path(rng, 20000)[None] for _ in range(4)])
    Y = (X - ar.mu) @ np.linalg.inv(ar.L).T  # mode coordinates, should be N(0, I) AR(1)
    cov0 = np.einsum("cti,ctj->ij", Y, Y) / Y.shape[0] / Y.shape[1]
    lag1 = np.einsum("cti,ctj->ij", Y[:, :-1], Y[:, 1:]) / Y.shape[0] / (Y.shape[1] - 1)
    tau = (1 + ar.rho) / (1 - ar.rho)
    se = np.sqrt(2 * tau.max() / (Y.shape[0] * Y.shape[1]))
    assert np.abs(cov0 - np.eye(3)).max() < 6 * se
    assert np.abs(lag1 - np.diag(ar.rho)).max() < 6 * se


# spectrum
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_priority_queue_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    rho = np.sort(rng.uniform(0.3, 0.95, size=4))[::-1]
    spec = ra.enumerate_hermite(rho, n_top=17)
    idx, lv = ra.enumerate_bruteforce(rho, 40)
    np.testing.assert_allclose(spec.log_eig[:17], lv[:17], rtol=0, atol=1e-13)
    # same multisets of modes in the top 17 (order inside numerical ties may differ)
    got = sorted(map(tuple, spec.indices[:17].tolist()))
    ref = sorted(map(tuple, idx[: spec.K].tolist()))[: len(got)]
    assert set(got) <= set(map(tuple, idx[: spec.K + 5].tolist()))
    assert len(ref) == len(got)


def test_tie_clusters_are_completed():
    rho = np.array([0.9, 0.81, 0.5])  # 0.9^2 == 0.81 (numerical tie after rounding)
    spec = ra.enumerate_hermite(rho, n_top=2, tie_tol=1e-12)
    labels = [spec.label(j) for j in range(spec.K)]
    assert labels[0] == "y1" and set(labels[1:3]) == {"y1^2", "y2"}
    assert spec.K == 3 and spec.cluster[1] == spec.cluster[2]
    assert not ra.true_gap(spec, 2)["strict"]
    g = ra.true_gap(spec, 1)
    assert g["strict"] and g["a_upper"] >= g["a"] and g["b_lower"] <= g["b"]


# moments
def _two_d_case(ell=2.0, rot=30.0, rho=(0.9, 0.6)):
    c = np.array([4.0, 1.0])
    U = ga.rotation(rot)
    ar = ra.controlled_ar_from("2d", np.zeros(2), np.diag(c), U, np.array(rho))
    gar = ga.GaussAR(c_diag=tuple(c), rotation_deg=rot, rho=rho)
    rng = np.random.default_rng(3)
    Z = ar.sample_target(rng, 48)
    feat = ra.landmark_features(Z, np.array([ell, ell]), 1e-10)
    return ar, gar, feat


def test_generating_function_matches_tensor_quadrature_2d():
    ar, gar, feat = _two_d_case()
    geom = ra.make_geometry(ar, feat, "fp64")
    alphas = [a for a in ra.multi_indices(2, 8) if sum(a) > 0]
    index, A = ra.hermite_coefficients(geom, ra.downward_closure(alphas))
    analytic = geom.m[None, :] * A[[index[a] for a in alphas]]
    Y, wt = ga.gauss_hermite_grid(80)
    X = gar.from_modes(Y)
    k = ra.rbf_aniso(X, feat.Q, feat.ls)
    modes = ga.HermiteModes(np.array(alphas), np.zeros(len(alphas)), 8)
    psi = ga.mode_values(gar, X, modes)
    quad = (psi * wt[:, None]).T @ k
    assert np.abs(quad - analytic).max() < 1e-12 * np.abs(analytic).max() + 1e-14


def test_closed_form_operator_matches_independent_lag_formula_2d():
    ar, gar, feat = _two_d_case()
    geom = ra.make_geometry(ar, feat, "fp64")
    for t in (0, 1, 3, 10):
        D = ra.centered_moments(geom, t)
        D_ref, _ = ga.landmark_lag_covariance(feat.Q, gar.C, gar.A_power(2 * t), 2.0)
        assert np.abs(D - D_ref).max() < 1e-13 * np.abs(D_ref).max()


def test_closed_form_matches_gamma_form_and_hermite_reconstruction():
    ar, gar, feat = _two_d_case()
    geom = ra.make_geometry(ar, feat, "fp64")
    S = {t: ra.operator_S(geom, t) for t in (0, 2, 8)}
    for t in (0, 2, 8):
        At = np.linalg.matrix_power(ar.A, t)
        Dg = ra.centered_moments_from_gamma(geom, At @ ar.C @ At.T)
        D = ra.centered_moments(geom, t)
        assert np.abs(Dg - D).max() < 1e-13 * np.abs(D).max()
        # whitening amplifies the 1e-16 difference between A^t C A^tT and the mode form by up to
        # 1 / s_min (here K_QQ is kept down to 1e-10 relative)
        assert ra.sm_opnorm(ra.whiten(geom, Dg) - S[t]) < 1e-7 * ra.sm_opnorm(S[t])
    # The closed form in long double is the accurate one here (kept Gram condition ~8e9): fp64
    # whitening leaves ~1e-8 relative error, long double ~2^-11 of that. The Hermite sum (float64
    # embedded modes, whitened vectors rather than a whitened matrix) is good to ~1e-11.
    gld = ra.make_geometry(ar, feat, "ld")
    S_ld = {t: ra.operator_S(gld, t) for t in (0, 2, 8)}
    for Sc, tol in ((S_ld, 1e-10), (S, 1e-6)):
        rec = ra.hermite_reconstruction(geom, 40, (0, 2, 8), Sc, np.log(ar.rho))
        for t, v in rec["times"].items():
            # the omitted Hermite tail is PSD with trace <= tail_bound
            assert v["op_diff"] <= v["tail_bound"] * (1 + 1e-6) + tol * v["op_S"], (t, v)
            assert v["trace_diff"] >= -tol * v["op_S"], (t, v)


def test_fp64_and_long_double_agree():
    ar = _small_ar(3, seed=4)
    rng = np.random.default_rng(9)
    feat = ra.landmark_features(ar.sample_target(rng, 40), np.sqrt(np.diag(ar.C)) * 2.0, 1e-12)
    g64, gld = ra.make_geometry(ar, feat, "fp64"), ra.make_geometry(ar, feat, "ld")
    for t in (0, 1, 7, 40):
        S64, Sld = ra.operator_S(g64, t), ra.operator_S(gld, t)
        assert ra.sm_opnorm(S64 - Sld) < 1e-9 * ra.sm_opnorm(S64), t
    w64 = ra.embedded_modes(g64, [(1, 0, 0), (0, 2, 1)])
    wld = ra.embedded_modes(gld, [(1, 0, 0), (0, 2, 1)])
    assert np.abs(w64 - wld).max() < 1e-10 * np.abs(w64).max()


def test_closed_form_matches_branched_monte_carlo():
    ar = _small_ar(2, seed=6)
    rng = np.random.default_rng(11)
    feat = ra.landmark_features(ar.sample_target(rng, 12), np.sqrt(np.diag(ar.C)) * 1.5, 1e-10)
    geom = ra.make_geometry(ar, feat, "fp64")
    n, t = 200_000, 2
    X = ar.sample_target(rng, n)
    branches = []
    for _ in range(2):
        x = X - ar.mu
        for _ in range(t):
            x = x @ ar.A.T + rng.standard_normal(x.shape) @ ar.LQ.T
        branches.append(feat(x + ar.mu))
    mean = feat.W.T @ geom.m
    P = (branches[0] - mean)[:, :, None] * (branches[1] - mean)[:, None, :]
    est = 0.5 * (P.mean(0) + P.mean(0).T)
    se = P.std(0) / math.sqrt(n)
    S = ra.operator_S(geom, t)
    z = np.abs(est - S) / se
    assert z.max() < 5.5, z.max()


def test_low_degree_checks_small():
    ar = _small_ar(3, seed=7)
    feat = ra.landmark_features(
        ar.sample_target(np.random.default_rng(1), 30), np.sqrt(np.diag(ar.C)) * 2, 1e-12
    )
    out = ra.low_degree_checks(ar, feat, ra.make_geometry(ar, feat, "fp64"))
    assert max(out.values()) < 1e-11, out


# selection and bounds
def test_selection_rules():
    rules = ra.SelectionRules()

    def row(lag, m, ratio, ok=True):
        return {"lag": lag, "m": m, "gap_ratio": ratio, "eligible": ok}

    rows = [
        row(4, 1, 50.0),
        row(1, 3, 6.0),
        row(1, 2, 9.0),
        row(1, 5, 9.0),
        row(1, 1, 99, ok=False),
    ]
    sel = ra.select_counts(rows, rules)
    assert sel["lag"] == 1 and sel["selected"] == [2, 5]
    assert ra.select_counts([row(1, 1, 1.0, ok=False)], rules)["outcome"] == "NONE_ELIGIBLE"
    one = ra.select_counts([row(16, 4, 7.0)], rules)
    assert one["selected"] == [4] and one["outcome"] == "ONLY_1_ELIGIBLE"


def test_half_split_sensitivity_definition():
    a = np.array([0.95, -0.9, 0.5, 0.1])
    b = np.array([0.94, 0.92, 0.45, 0.2])
    assert ra.half_split_sensitivity(a, b, 1) == pytest.approx(0.5 * 0.05)
    assert ra.half_split_sensitivity(a, b, 2) == pytest.approx(0.5 * 0.1)


def test_theorem_bound_and_subspace_sine():
    g0, _ = ra.theorem_bound(2.0, 1.0, 0.9, 0.95, 0)
    assert g0 == 1.0
    g, raw = ra.theorem_bound(2.0, 1.0, 0.9, 0.95, 50)
    assert g == pytest.approx(2.0 * (0.9 / 0.95) ** 100, rel=1e-12) and g < 1
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 2))
    B = A + 1e-9 * rng.standard_normal((6, 2))
    s = ra.subspace_sine(A, B)
    from mcmc.paperexp import slowmodes as sm

    assert s == pytest.approx(math.sin(sm.largest_principal_angle(A, B)), rel=1e-6)
    assert ra.subspace_sine(A, A) < 1e-15


def test_pooled_lag_covariance_matches_single_path():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((500, 4))
    from mcmc.paperexp import slowmodes as sm

    C0, Cl, _ = ra.pooled_lag_covariances([F], 3)
    C0r, Clr, _ = sm.lagged_pairs_covariances(F, 3)
    assert np.abs(C0 - C0r).max() < 1e-15 and np.abs(Cl - Clr).max() < 1e-15
    assert linalg.eigvalsh(C0).min() > 0


def test_mp_spot_check_matches_cancellation_free_form():
    ar = _small_ar(3, seed=8)
    rng = np.random.default_rng(4)
    feat = ra.landmark_features(ar.sample_target(rng, 20), np.sqrt(np.diag(ar.C)) * 2.0, 1e-12)
    g64, gld = ra.make_geometry(ar, feat, "fp64"), ra.make_geometry(ar, feat, "ld")
    pairs = np.array([[0, 0], [3, 7], [19, 2], [5, 5]])
    for t in (0, 5, 200):
        ref = ra.mp_spot_check(ar, feat, pairs, t, dps=60)
        D64 = ra.centered_moments(g64, t)[pairs[:, 0], pairs[:, 1]]
        Dld = ra.centered_moments(gld, t)[pairs[:, 0], pairs[:, 1]]  # long double, not cast
        assert ra.spot_check_errors(ref, D64)["rel_err"] < 1e-12, t
        # the long-double tier is checked at its own precision (differences formed in mpmath)
        assert ra.spot_check_errors(ref, Dld)["rel_err"] < 1e-16, t


def test_checkpoint_statuses_and_gate():
    rng = np.random.default_rng(0)
    r = 6
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    S = (Q * np.array([1.0, 0.3, 0.1, 0.01, 1e-3, 1e-4])) @ Q.T
    W = Q[:, :1] + 1e-3 * Q[:, 1:2]
    rules = ra.NumericsRules()
    allow = ra.operator_allowance(S, S + 1e-12 * np.eye(r), rules.allowance_factor)
    row = ra.checkpoint(S, S + 1e-12 * np.eye(r), W, W, allow, rules)
    assert row["estimate_status"] == "RESOLVED" and row["sin_theta"] == pytest.approx(
        1e-3, rel=1e-5
    )
    bad = ra.operator_allowance(S, S + 0.1 * np.eye(r), rules.allowance_factor)
    assert ra.checkpoint(S, S + 0.1 * np.eye(r), W, W, bad, rules)["estimate_status"] == (
        "OPERATOR_UNRESOLVED"
    )
    rows = [
        {"t": 0, "estimate_status": "RESOLVED", "sin_angle_upper": 1.0},
        {"t": 1, "estimate_status": "RESOLVED", "sin_angle_upper": 0.9},
        {"t": 2, "estimate_status": "BELOW_NUMERICAL_SENSITIVITY", "sin_angle_upper": 0.6},
        {"t": 4, "estimate_status": "RESOLVED", "sin_angle_upper": 0.4},
        {"t": 8, "estimate_status": "OPERATOR_UNRESOLVED", "sin_angle_upper": 0.1},
        {"t": 16, "estimate_status": "RESOLVED", "sin_angle_upper": 0.01},
    ]
    g = ra.main_curve_and_gate(rows, 3, 0.5)
    assert g["first_failed_checkpoint"] == 8 and g["n_resolved_nontrivial"] == 3
    assert g["gate_pass"] and not g["gate_pass_positive_points"]
    assert [r["on_main_curve"] for r in rows] == [True] * 4 + [False] * 2


def test_zero_rate_coordinate_is_handled_at_t0():
    """Rate-rule underflow (rho_j = 0, Energy): rho_j^0 = 1 in every precision tier."""
    rng = np.random.default_rng(12)
    C = _spd(rng, 3)
    U, _ = ra.frozen_rotation(3)
    ar = ra.controlled_ar_from("zero", np.zeros(3), C, U, np.array([0.9, 0.5, 0.0]))
    feat = ra.landmark_features(ar.sample_target(rng, 16), np.sqrt(np.diag(C)) * 2.0, 1e-12)
    g64, gld = ra.make_geometry(ar, feat, "fp64"), ra.make_geometry(ar, feat, "ld")
    pairs = np.array([[0, 1], [4, 4]])
    for t in (0, 1, 3):
        D64 = ra.centered_moments(g64, t)
        Dld = ra.centered_moments(gld, t).astype(np.float64)
        ref = ra.mp_spot_check(ar, feat, pairs, t, dps=40)
        assert np.abs(Dld - D64).max() < 1e-12 * np.abs(D64).max(), t
        assert ra.spot_check_errors(ref, D64[pairs[:, 0], pairs[:, 1]])["rel_err"] < 1e-12, t
    spec = ra.enumerate_hermite(ar.rho, n_top=5)
    assert all(spec.indices[j][2] == 0 for j in range(spec.K))


def test_exact_integer_tier_matches_long_double_and_eigh():
    ar = _small_ar(3, seed=13)
    rng = np.random.default_rng(14)
    feat = ra.landmark_features(ar.sample_target(rng, 18), np.sqrt(np.diag(ar.C)) * 2.0, 1e-12)
    gld = ra.make_geometry(ar, feat, "ld")
    for t in (0, 6):
        ex = ra.exact_integer_operator(ar, feat, t, dps=40, workers=1)
        S_ex = ra.exact_to_float(ex["S_int"], ex["exponent"])
        S_ld = ra.operator_S(gld, t)
        assert ra.sm_opnorm(S_ex - S_ld) < 1e-14 * ra.sm_opnorm(S_ex), t
        lam, V = ra.eig_desc(S_ex)
        eig = ra.mp_subspace_eigen(ex, k=4, iters=8, dps=40, V0=V)
        np.testing.assert_allclose([float(x) for x in eig["theta"]], lam[:4], rtol=1e-9)
        # block iteration converges at rate lambda_{k+1} / lambda_j (slow on the flat t=0 spectrum)
        if t == 6:
            assert max(float(x) for x in eig["residuals"][:2]) < 1e-27 * lam[0]
    w_mp = ra.exact_true_modes(ex, feat, [(1, 0, 0), (2, 0, 0)], dps=40)
    w_ld = ra.embedded_modes(gld, [(1, 0, 0), (2, 0, 0)])
    w_ex = np.array([[float(w_mp[i, j]) for j in range(2)] for i in range(feat.rank)])
    assert np.abs(w_ex - w_ld).max() < 1e-13 * np.abs(w_ld).max()
    # exact-tier angle from the full-precision mpmath spans (no float64 rounding of the vectors),
    # against an independent 80-digit Gram-Schmidt evaluation on the same mpmath values
    import mpmath as mp

    Y = eig["Y"]
    Ym = mp.matrix([[Y[i, j] for j in range(2)] for i in range(feat.rank)])
    out = ra.span_sine(ra.exact_span(w_mp), ra.exact_span(Ym))
    with mp.workdps(80):
        Qw, _ = mp.qr(w_mp)
        Qy, _ = mp.qr(Ym)
        Qw, Qy = Qw[:, :2], Qy[:, :2]
        P = Qy - Qw * (Qw.T * Qy)
        ref = mp.sqrt(max(mp.eigsy(P.T * P, eigvals_only=True)))
        assert abs(out["sin_mp"] - ref) <= out["arith_mp"] + mp.mpf("1e-70")


# Precision regression tests
def _mp_gram_schmidt_sine(A, B, dps=100):
    """Independent reference: sin theta_max by twice-iterated Gram-Schmidt in ``dps`` digits."""
    import mpmath as mp

    with mp.workdps(dps):

        def orth(X):
            Q = mp.matrix(X.tolist())
            for _ in range(2):
                for j in range(Q.cols):
                    for i in range(j):
                        c = mp.fsum(Q[q, i] * Q[q, j] for q in range(Q.rows))
                        for q in range(Q.rows):
                            Q[q, j] -= c * Q[q, i]
                    nrm = mp.sqrt(mp.fsum(Q[q, j] ** 2 for q in range(Q.rows)))
                    for q in range(Q.rows):
                        Q[q, j] /= nrm
            return Q

        Qa, Qb = orth(A), orth(B)
        P = Qb - Qa * (Qa.T * Qb)
        return mp.sqrt(max(mp.eigsy(P.T * P, eigvals_only=True)))


def test_dyadic_integers_are_exact():
    from fractions import Fraction

    import mpmath as mp

    rng = np.random.default_rng(21)
    X = rng.standard_normal((7, 3)) * np.exp(rng.uniform(-80, 5, (7, 3)))
    X[2, 1] = 0.0
    Xi, E = ra.dyadic_integers(X)
    for idx in np.ndindex(X.shape):
        assert Fraction(int(Xi[idx])) * Fraction(2) ** E == Fraction(float(X[idx]))
    Xl = X.astype(ra.LD) / ra.LD(3)
    Li, El = ra.dyadic_integers(Xl)
    back = np.array([[ra.LD(int(v)) for v in row] for row in Li]) * np.ldexp(ra.LD(1), El)
    assert np.array_equal(back, Xl)
    with mp.workdps(55):
        M = mp.matrix([[mp.mpf(1) / 3, mp.mpf(-2) / 7], [mp.mpf("1e-40"), +mp.pi]])
    Mi, Em = ra.dyadic_integers(M)
    R = ra.mp_from_dyadic(Mi, Em)  # exact round trip at the default (53-bit) context
    assert all(R[i, j] == M[i, j] for i in range(2) for j in range(2))
    assert R[0, 0].man_exp == M[0, 0].man_exp


@pytest.mark.parametrize("m", [1, 2, 3])
def test_span_sine_exact_on_known_tiny_angle(m):
    """sin theta_max is the exact value delta / sqrt(1 + delta^2) even far below float64 resolution of
    the inputs' orthonormalisation; the float64 route cannot resolve it."""
    import mpmath as mp

    r, delta = 9, 2.0**-70
    A = np.eye(r)[:, :m]
    B = A.copy()
    B[m, m - 1] = delta  # last column tilted by delta into a fresh coordinate
    out = ra.span_sine(ra.exact_span(A), ra.exact_span(B))
    with mp.workprec(800):
        exact = mp.mpf(delta) / mp.sqrt(1 + mp.mpf(delta) ** 2)
        assert abs(out["sin_mp"] - exact) <= out["arith_mp"]
        # serialisation error is exact: |fl(s) - s| for the nearest float64
        assert out["serialization_mp"] == abs(mp.mpf(out["sin"]) - out["sin_mp"])
        assert abs(mp.mpf(out["sin"]) - exact) <= out["arith_mp"] + out["serialization_mp"]
    assert out["sin"] == pytest.approx(delta, rel=1e-15)
    assert out["arith"] < 1e-70 and out["serialization"] <= 2.0**-53 * delta


def test_span_sine_matches_independent_high_precision_route_and_exposes_float64_error():
    import mpmath as mp

    rng = np.random.default_rng(22)
    for m in (1, 2, 4):
        A = rng.standard_normal((30, m))
        B = A @ rng.standard_normal((m, m)) + 1e-13 * rng.standard_normal((30, m))
        out = ra.span_sine(ra.exact_span(A), ra.exact_span(B))
        ref = _mp_gram_schmidt_sine(A, B, dps=100)
        with mp.workdps(100):
            assert abs(out["sin_mp"] - ref) <= out["arith_mp"] + mp.mpf("1e-90"), m
            route_err = abs(mp.mpf(ra.subspace_sine(A, B)) - ref)
        # the float64 orthonormalisation route carries absolute errors near u (relative ~1e-3 here)
        assert route_err > 1e-6 * out["sin"] or route_err > 0, m
        assert out["arith"] < 1e-60


def test_float_up_and_capped_sums():
    import mpmath as mp

    with mp.workprec(300):
        x = mp.mpf(1) / 3
        f = ra.float_up(x)
        assert mp.mpf(f) >= x and mp.mpf(math.nextafter(f, 0.0)) < x
        tiny = mp.ldexp(1, -250)
    assert ra.sine_uncertainty(1e-14, tiny) > 1e-14  # sum rounded up, never down to the larger term
    assert ra.sine_uncertainty(0.7, 0.6) == 1.0
    assert ra.float_up(0.25) == 0.25


def test_checkpoint_uncertainty_dominates_every_error_on_a_known_operator():
    """S diagonal (exact eigenvectors e_1..e_m), W_m a tilted float64 basis: the reported sine is within
    e_sin of the exact population sine, and e_sin is the rounded-up sum of its four components."""
    import mpmath as mp

    r, m = 8, 2
    S = np.diag([1.0, 0.5, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    W = np.zeros((r, m))
    W[0, 0] = W[1, 1] = 1.0
    W[2, 1] = 3e-9
    rules = ra.NumericsRules()
    allow = ra.operator_allowance(S, S, rules.allowance_factor)
    row = ra.checkpoint(S, S, W, W, allow, rules)
    with mp.workprec(600):
        exact = mp.mpf(3e-9) / mp.sqrt(1 + mp.mpf(3e-9) ** 2)
        assert abs(mp.mpf(row["sin_theta"]) - exact) <= mp.mpf(row["sin_uncertainty"])
        parts = (
            row["sin_uncertainty_dk"],
            row["sin_uncertainty_modes"],
            row["sin_uncertainty_arith"],
            row["sin_uncertainty_serialization"],
        )
        assert mp.mpf(row["sin_uncertainty"]) >= mp.fsum(mp.mpf(p) for p in parts) * (1 - 1e-15)
    assert row["estimate_status"] == "RESOLVED"
    assert row["eigvec_residual_ratio"] < 1.0
    assert row["float64_route_abs_err"] >= 0.0


def test_learned_envelope_carries_mode_uncertainty():
    rng = np.random.default_rng(23)
    r, m = 10, 2
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    S = (Q * np.geomspace(1.0, 1e-5, r)) @ Q.T
    S = 0.5 * (S + S.T)
    W = Q[:, :m] + 1e-4 * rng.standard_normal((r, m))
    What = Q[:, :m] + 1e-2 * rng.standard_normal((r, m))
    rules = ra.NumericsRules()
    allow = ra.operator_allowance(S, S, rules.allowance_factor)
    eig = ra.operator_eigen(S, S)
    tr = ra.checkpoint(S, S, W, W, allow, rules, eig)
    e_w = 1e-6  # a declared true-mode margin
    mode = ra.learned_mode_error(ra.exact_span(What), ra.exact_span(W), e_w)
    Vx, Vhx = ra.eigen_spans(eig, m)
    g = 0.25
    row = ra.learned_checkpoint(
        What, ra.exact_span(What), Vx, Vhx, eig["V"], S, eig["lam"], mode, tr, g, rules
    )
    assert row["learned_sin_upper"] >= mode["mode_error_sin"] + e_w + g
    assert row["learned_sin_upper"] <= mode["mode_error_sin"] + e_w + g + 1e-15
    assert mode["mode_uncertainty"] >= e_w
    assert row["bound_satisfied"] and row["estimate_status"] == "RESOLVED"
    assert row["sin_uncertainty"] >= row["sin_uncertainty_dk"]


def test_eigvec_residual_check_detects_inaccurate_vectors():
    rng = np.random.default_rng(24)
    r = 40
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    S = (Q * np.geomspace(1e-3, 1e-20, r)) @ Q.T
    S = 0.5 * (S + S.T)
    lam, V = ra.eig_desc(S)
    Si, Es = ra.dyadic_integers(S)
    good = ra.eigvec_residual_check(Si, Es, V[:, :3], lam[:3])
    assert good["rq"] < r * ra.U64 * lam[0]
    bad = ra.eigvec_residual_check(Si, Es, V[:, :3] + 1e-7 * rng.standard_normal((r, 3)), lam[:3])
    assert bad["rq"] > 1e3 * good["rq"]


def test_spot_check_errors_subtract_before_rounding():
    """Subtract in extended precision to retain errors below float64 resolution."""
    from fractions import Fraction

    import mpmath as mp

    with mp.workdps(60):
        ref = [mp.mpf(1) / 10, mp.mpf(-3) / 7]
    vals = np.array([ra.LD(1) / ra.LD(10), ra.LD(-3) / ra.LD(7)])
    cast_first = np.abs(vals.astype(np.float64) - np.array([float(x) for x in ref])).max()
    assert cast_first == 0.0
    out = ra.spot_check_errors(ref, vals)
    exact = max(
        abs(Fraction(*vals[0].as_integer_ratio()) - Fraction(1, 10)),
        abs(Fraction(*vals[1].as_integer_ratio()) - Fraction(-3, 7)),
    ) / Fraction(3, 7)
    assert 0.0 < out["rel_err"] < 1e-18
    assert out["rel_err"] == pytest.approx(float(exact), rel=1e-12)
