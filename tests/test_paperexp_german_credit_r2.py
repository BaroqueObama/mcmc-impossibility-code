"""Tests for logistic feature estimates, projections, and sampler adapters."""

import itertools
import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import german_credit as gc
from mcmc.paperexp import german_credit_r2 as r2
from mcmc.paperexp import slowmodes as sm
from mcmc.paperexp.kernels import ScaledRBF
from mcmc.paperexp.samplers import make_sampler

F64 = torch.float64


def _kernel(d=3, ell=1.3):
    return ScaledRBF(np.linspace(0.8, 1.2, d), ell)


def test_multichain_lagged_matches_single_chain_slowmodes():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(1, 200, 5)).cumsum(axis=1) * 0.1
    a = r2.multichain_lagged(F, 3)
    b = sm.lagged_pairs_covariances(F[0], 3)
    for x, y in zip(a[:2], b[:2]):
        np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-15)


def test_tica_on_known_ar_channels_recovers_slowest_direction():
    """Independent AR(1) channels: TICA eigenvalues ~ rho, embedded slow direction ~ e_slowest."""
    rng = np.random.default_rng(1)
    rho = np.array([0.5, 0.97, 0.2, 0.8, 0.1, 0.3, 0.6, 0.0, 0.4, 0.7])
    C, L, r = 4, 20000, rho.size
    F = np.empty((C, L, r))
    F[:, 0] = rng.normal(size=(C, r))
    for t in range(1, L):
        F[:, t] = rho * F[:, t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=(C, r))
    X = F[:, :, :3].copy()
    spans, info = r2.learn_spans(
        F, X, t_star=2, m_max=2, tica_cutoff=1e-10, coord_cutoff=1e-10, rng=np.random.default_rng(2)
    )
    np.testing.assert_allclose(sorted(info["tica_rho"], reverse=True), [0.97, 0.8], atol=0.02)
    w = spans["tica"][:, 0] / np.linalg.norm(spans["tica"][:, 0])
    assert abs(w[1]) > 0.999
    # lag-2t* covariance top direction is also the slowest channel (rho^4 largest)
    assert abs(spans["lag2t"][1, 0]) > 0.99
    # coordinates 0..2 are exactly linear in F: the projected coordinate functions fit perfectly
    assert min(info["coord_fixed_r2"]) > 1 - 1e-10


def test_checkpoint_stats_and_unbiased_mmd_against_brute_force():
    torch.manual_seed(0)
    k = _kernel()
    Z = torch.randn(12, 3, dtype=F64)
    from mcmc.paperexp.features import NystromFeatures

    feats = NystromFeatures(k, Z)
    ref_sets = {"s": np.random.default_rng(3).normal(size=(3, 15, 3)) + 0.3}
    ref = r2.build_ref_summary(k, feats, ref_sets, "cpu", 1 << 26, [0.5])
    x = torch.randn(16, 3, dtype=F64)
    G = 4
    st = r2.checkpoint_stats(x, k, feats, ref, G, 1 << 26, [0.5])
    K = k(x, x).numpy()
    S_bf = K.reshape(G, 4, G, 4).sum(axis=(1, 3))
    np.testing.assert_allclose(st["group_pair_ksum"], S_bf, rtol=1e-12)
    runs = np.arange(3)
    est = r2.mmd2_unbiased(
        st["group_pair_ksum"], st["group_run_cross"], ref.run_inner, 4, np.arange(G), runs
    )
    J = 16
    A = (K.sum() - J) / (J * (J - 1))
    R = [torch.as_tensor(ref_sets["s"][i]) for i in range(3)]
    B = np.mean([k(x, Rr).numpy().mean() for Rr in R])
    C = np.mean([k(R[a], R[b]).numpy().mean() for a, b in itertools.permutations(range(3), 2)])
    assert est == pytest.approx(A - 2 * B + C, rel=1e-12)
    np.testing.assert_allclose(
        st["group_feat"], feats(x).numpy().reshape(G, 4, -1).mean(1), rtol=1e-12
    )
    fm = np.stack([feats(Rr).numpy().mean(0) for Rr in R])
    np.testing.assert_allclose(ref.feat_run_means, fm, rtol=1e-12)


def test_projection_terms_against_explicit_formulas():
    rng = np.random.default_rng(4)
    r, k = 7, 5
    delta = rng.normal(size=r)
    E = rng.normal(size=(r, k)) * 0.3
    Qs = [sm.orthonormal_basis(rng.normal(size=(r, m))) for m in (1, 2, 2, 4)] + [None]
    out = r2.projection_terms(Qs, delta, E)
    for i, Q in enumerate(Qs[:-1]):
        v = Q.T @ delta
        A = Q.T @ E @ E.T @ Q
        assert out[i, 0] == pytest.approx(v @ v, rel=1e-12)
        assert out[i, 1] == pytest.approx(np.trace(A), rel=1e-12)
        assert out[i, 2] == pytest.approx(4 * v @ A @ v + 2 * np.trace(A @ A), rel=1e-12)
    assert np.isnan(out[-1]).all()


def test_residual_se_is_the_paired_jackknife_of_the_residual():
    """res_full2_se = delete-one-group + delete-one-run jackknife of MMD^2 - ||P_W delta||^2,
    recomputed here by brute force (re-running both estimators on every reduced sample)."""
    torch.manual_seed(6)
    k = _kernel()
    from mcmc.paperexp.features import NystromFeatures

    feats = NystromFeatures(k, torch.randn(12, 3, dtype=F64))
    ref = r2.build_ref_summary(
        k, feats, {"s": np.random.default_rng(7).normal(size=(5, 15, 3))}, "cpu", 1 << 26, [0.5]
    )
    G, n = 6, 3
    x = torch.randn(G * n, 3, dtype=F64) * 1.4 + 0.4
    st = r2.checkpoint_stats(x, k, feats, ref, G, 1 << 26, [0.5])
    st["n_per_group"] = n
    rng = np.random.default_rng(8)
    r = st["group_feat"].shape[1]
    Qs = [sm.orthonormal_basis(rng.normal(size=(r, m))) for m in (1, 2, 2, 3)] + [None]
    runs = np.arange(5)
    ev = r2.evaluate_spans(r2.law_reference_terms(st, ref, runs), Qs)

    def residual(groups, rr):
        sub = {
            "group_feat": st["group_feat"][groups],
            "group_pair_ksum": st["group_pair_ksum"][np.ix_(groups, groups)],
            "group_run_cross": st["group_run_cross"][groups],
            "n_per_group": n,
        }
        terms = r2.law_reference_terms(sub, ref, rr)
        pt = r2.projection_terms(Qs, terms["delta"], terms["E"])
        return terms["mmd2"] - (pt[:, 0] - pt[:, 1])

    jg = np.stack([residual(np.delete(np.arange(G), g), runs) for g in range(G)])
    jr = np.stack([residual(np.arange(G), np.delete(runs, i)) for i in range(runs.size)])
    se = np.sqrt(
        (G - 1) / G * ((jg - jg.mean(0)) ** 2).sum(0) + 4 / 5 * ((jr - jr.mean(0)) ** 2).sum(0)
    )
    np.testing.assert_allclose(ev["res_full2_se"][:-1], se[:-1], rtol=1e-9)
    np.testing.assert_allclose(ev["res_full2"][:-1], residual(np.arange(G), runs)[:-1], rtol=1e-12)
    assert np.isnan(ev["res_full2_se"][-1])


def test_debiased_projection_is_unbiased_in_simulation():
    """Monte Carlo: E[raw - bias] = ||Q^T delta||^2 when delta-hat = delta + E-scale noise."""
    rng = np.random.default_rng(5)
    r, G, R = 6, 32, 16
    delta = np.zeros(r)
    delta[0] = 0.05
    Q = sm.orthonormal_basis(rng.normal(size=(r, 2)))
    vals = []
    for _ in range(4000):
        gf = delta + rng.normal(size=(G, r)) * 0.2
        rf = rng.normal(size=(R, r)) * 0.1
        stats = {"group_feat": gf}
        f_ens, f_ref = gf.mean(0), rf.mean(0)
        E = np.concatenate(
            [(gf - f_ens).T / math.sqrt(G * (G - 1)), (rf - f_ref).T / math.sqrt(R * (R - 1))],
            axis=1,
        )
        out = r2.projection_terms([Q], f_ens - f_ref, E)[0]
        vals.append(out[0] - out[1])
        assert stats
    truth = float((Q.T @ delta) @ (Q.T @ delta))
    se = np.std(vals) / math.sqrt(len(vals))
    assert abs(np.mean(vals) - truth) < 4 * se


def test_branched_operator_at_t0_is_unbiased_static_covariance_form():
    rng = np.random.default_rng(6)
    N, r = 50, 4
    FX, FXp = rng.normal(size=(N, r)), rng.normal(size=(N, r))
    F = torch.as_tensor(np.concatenate([FX, FX, FXp, FXp]))
    out = r2.branched_operator_stats(F, N)
    D = FX - FXp
    np.testing.assert_allclose(out["O"], D.T @ D / (2 * N), rtol=1e-12)
    np.testing.assert_allclose(out["OA"], D[:25].T @ D[:25] / 50, rtol=1e-12)


def test_conventional_on_iid_trajectory():
    rng = np.random.default_rng(7)
    traj = rng.normal(size=(257, 8, 3))
    out = r2.conventional_on_traj(traj, [0, 1, 16, 64, 256], group_size=4)
    assert sorted(out) == [16, 64, 256]
    assert out[256]["rhat_max"].shape == (2,) and (out[256]["rhat_max"] < 1.05).all()
    assert (out[256]["ess_bulk_min"] > 250).all() and out[256]["window"] == 128


def test_start_laws_and_target_adapter_with_core_sampler():
    data = gc.load_german_credit()
    post = gc.make_posterior("hier", data)
    rng = np.random.default_rng(8)
    pool = rng.normal(size=(5000, post.d)) * 0.05
    pool[:, 0] += 0.9
    pool[:, 1] -= 1.3
    laws = r2.StartLaws(pool, np.full(post.d, 0.1), overdispersion=3.0, box_radius=2.0)
    gen = torch.Generator().manual_seed(0)
    x1 = laws.sample("I1", 4000, gen)
    np.testing.assert_allclose(x1.mean(0).numpy(), laws.mean, atol=0.02)
    x2 = laws.sample("I2", 1000, gen)
    assert x2.abs().max() <= 2.0
    x3 = laws.sample("I3", 1000, gen).numpy()
    assert ((x3 - laws.mean) / 0.1 @ laws.confine_dir > 0).all()
    target = r2.PosteriorTarget(post, pool)
    smp = make_sampler("mala", target, 0.5, np.full(post.d, 0.05))
    out = smp.run(target.sample(8, gen), [0, 5], gen)
    assert out["states"][5].shape == (8, post.d)
    assert smp.counters["grad_evals"] > 0


def test_widths_batch_matches_slowmodes_and_oracle_is_optimal():
    rng = np.random.default_rng(9)
    A = rng.normal(size=(12, 12)) * np.geomspace(1, 0.05, 12)
    cov = A @ A.T / 12
    noise = rng.normal(size=(12, 12)) * 1e-6
    op = r2.OperatorSummary.from_stats(cov, cov + noise + noise.T, cov - noise - noise.T)
    Ws = [sm.orthonormal_basis(rng.normal(size=(12, 3))) for _ in range(5)] + [op.top_span(3)]
    w, ok = r2.widths_batch(op.eigvals, op.eigvecs, np.stack(Ws))
    assert ok.all()
    for W, wi in zip(Ws, w):
        assert wi == pytest.approx(sm.constrained_width_cov(cov, W)["width"], rel=1e-9)
    ang = r2.largest_angles_batch(np.stack(Ws), op.top_span(3))
    for W, a in zip(Ws, ang):
        assert a == pytest.approx(
            math.degrees(sm.largest_principal_angle(W, op.top_span(3))), abs=1e-7
        )
    rows = r2.worst_case_batch(op, Ws + [None], 3)
    assert rows[5]["ratio"] == pytest.approx(1.0, rel=1e-9) and rows[5]["angle_deg"] < 1e-5
    assert all(r["ratio"] >= 1 - 1e-9 for r in rows[:5])
    assert rows[6]["ratio_status"] == "FAILED_SPAN"
    assert rows[0]["operator_status"] == "RESOLVED"
    assert rows[0]["ratio_status"] in ("RESOLVED", "UNRESOLVED")


def test_worst_case_requires_a_resolved_operator():
    """lambda_{m+1} below factor * eps: no excess may be called RESOLVED, whatever its size."""
    rng = np.random.default_rng(10)
    cov = np.diag([10.0, 5.0, 1e-4, 1e-5, 1e-6])
    noise = rng.normal(size=(5, 5)) * 1e-2
    op = r2.OperatorSummary.from_stats(cov, cov + noise + noise.T, cov - noise - noise.T)
    assert op.eigvals[1] > 3 * op.eps > op.eigvals[2]
    far = [sm.orthonormal_basis(np.eye(5)[:, [2, 3]])]  # misses both top directions: huge excess
    rows = r2.worst_case_batch(op, far, 2)
    assert rows[0]["ratio"] > 10 and rows[0]["operator_status"] == "UNRESOLVED"
    assert rows[0]["ratio_status"] == "OPERATOR_UNRESOLVED"
    assert rows[0]["op_resolution"] == pytest.approx(op.eigvals[2] / op.eps)
    one = r2.worst_case_batch(op, [sm.orthonormal_basis(np.eye(5)[:, [1]])], 1)
    assert one[0]["operator_status"] == "RESOLVED" and one[0]["ratio_status"] == "RESOLVED"


def test_learn_spans_marks_degenerate_pilot_as_failed():
    """A stuck pilot (constant path) gives failed (None) eigen-based and coordinate spans."""
    F = np.ones((4, 64, 10))
    X = np.ones((4, 64, 3))
    spans, info = r2.learn_spans(
        F, X, t_star=2, m_max=2, tica_cutoff=1e-10, coord_cutoff=1e-10, rng=np.random.default_rng(0)
    )
    assert spans["static"] is None and spans["lag2t"] is None and spans["tica"] is None
    assert "static_failed" in info and "tica_failed" in info
    assert r2.orthonormal_or_none(spans["coord_fixed"], 2) is None
