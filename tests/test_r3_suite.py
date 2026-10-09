"""Tests for synthetic anytime bounds, independent streams, and cost accounting."""

import math

import numpy as np
import pytest
import torch

from mcmc.paperexp import bounds as bd
from mcmc.paperexp import operators as op
from mcmc.paperexp import r3_suite as r3
from mcmc.paperexp.embeddings import rbf_mean_embedding
from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import ScaledRBF
from mcmc.paperexp.rng import torch_gen
from mcmc.paperexp.samplers import make_sampler
from mcmc.paperexp.targets import GaussTarget, haar_rotation, make_target

F64 = torch.float64


def _gauss(d=3, cond=9.0, seed=7):
    return GaussTarget(d, cond, haar_rotation(d, seed))


def test_schedule_rule():
    assert r3.r3_schedule(16) == [0, 1, 2, 4, 8, 16, 32, 64]
    assert r3.r3_schedule(4)[-1] == 64
    assert r3.r3_schedule(512)[-1] == 2048
    assert r3.r3_schedule(1024)[-1] == 4096
    assert r3.r3_schedule(8192)[-1] == 8192
    with pytest.raises(ValueError):
        r3.r3_schedule(12)


@pytest.mark.parametrize("kind", ["rwm", "mala"])
def test_grouped_chains_reproduce_sampler_run(kind):
    t = make_target("mixture", 3, sep=4.0, w=0.5, rotation=haar_rotation(3, 1))
    s = make_sampler(kind, t, 0.7, np.ones(3))
    x0 = t.sample(64, torch_gen(1, "cpu"))
    ref = s.run(x0, [0, 3, 7], torch_gen(2, "cpu"))
    ch = r3.GroupedChains(r3.MHStep(s), [("a", x0, torch_gen(2, "cpu"))])
    ch.advance_to(3)
    assert torch.equal(ch.get("a"), ref["states"][3])
    ch.advance_to(7)
    assert torch.equal(ch.get("a"), ref["states"][7])
    acc = ref["accept_count_per_chain"].sum().item() / (64 * 7)
    assert abs(ch.accept_rate("a") - acc) < 1e-15


def test_grouped_chains_streams_are_separate():
    t = _gauss()
    s = make_sampler("rwm", t, 0.8, np.ones(3))
    xa = t.sample(32, torch_gen(3, "cpu"))
    xb = t.sample(16, torch_gen(4, "cpu"))
    xb2 = t.sample(48, torch_gen(4, "cpu"))
    one = r3.GroupedChains(r3.MHStep(s), [("a", xa, torch_gen(5, "cpu"))])
    two = r3.GroupedChains(
        r3.MHStep(s), [("a", xa, torch_gen(5, "cpu")), ("b", xb, torch_gen(6, "cpu"))]
    )
    three = r3.GroupedChains(
        r3.MHStep(s), [("a", xa, torch_gen(5, "cpu")), ("b", xb2, torch_gen(6, "cpu"))]
    )
    for c in (one, two, three):
        c.advance_to(5)
    assert torch.equal(one.get("a"), two.get("a"))
    assert torch.equal(one.get("a"), three.get("a"))


def test_sample_init_counted_matches_core():
    t = _gauss(d=4)
    for eps in (0.0, 0.2, 0.8):
        a = t.sample_init(1000, eps, torch_gen(11, "cpu"))
        b, n_prop = r3.sample_init_counted(t, 1000, eps, torch_gen(11, "cpu"))
        assert torch.equal(a, b)
        assert n_prop >= 1000 * (1 + eps)


def _witness(d=3, eps=0.8, rho_slow=0.85, rho_fast=0.4):
    t = _gauss(d=d)
    Q = t.rotation
    rho = np.full(d, rho_fast)
    rho[-1] = rho_slow  # q_min direction (the event direction) is slow
    A, L = r3.ar_witness_chain(t.covariance(), rho, Q)
    k = ScaledRBF(np.ones(d), 1.3)
    w = r3.ARGaussWitness(t.covariance(), A, (k.scales * k.ell) ** 2, Q[:, -1], eps)
    return t, k, A, L, w


def test_ar_chain_is_stationary_and_reversible():
    t, _, A, L, _ = _witness()
    C = t.covariance()
    assert np.abs(A @ C @ A.T + L @ L.T - C).max() < 1e-13
    assert np.abs(A @ C - C @ A.T).max() < 1e-13


def test_ar_witness_mmd_and_mu_t_vs_monte_carlo():
    t, k, A, L, w = _witness()
    emb = rbf_mean_embedding(t, k)
    assert abs(emb.norm2() - w.mu_norm2) < 1e-14
    g = torch_gen(21, "cpu")
    n = 200_000
    X0 = t.sample_init(n, w.eps, g)
    step = r3.LinearGaussianStep(A, L)
    ch = r3.GroupedChains(step, [("x", X0, torch_gen(22, "cpu"))])
    Zp = t.sample(16, torch_gen(23, "cpu")) * 1.5
    for tt in (0, 1, 3, 8):
        ch.advance_to(tt)
        X = ch.get("x")
        # mu_{pi_t} at probe points: MC mean of k(z, X_t) with its standard error
        Kz = k(Zp, X)
        mc = Kz.mean(1)
        se = Kz.std(1) / math.sqrt(n)
        z = ((mc - w.mu_t(tt, Zp)) / se).abs().max().item()
        assert z < 5.0, f"t={tt}: mu_t z-score {z}"
        # MMD^2 U-statistic on a subsample, using the closed-form target mean.
        m2 = w.mmd2(tt)
        Xs = X[:20000]
        Kxx = k(Xs, Xs)
        J = Xs.shape[0]
        u_stat = (Kxx.sum() - J) / (J * (J - 1))
        mu_s = emb(Xs)
        est = float(u_stat - 2 * mu_s.mean() + emb.norm2())
        se = 2 * float((Kxx.mean(1) - mu_s).std()) / math.sqrt(J)  # Hoeffding projection
        assert abs(est - m2) < 4 * se, f"t={tt}: MC {est} vs analytic {m2} (se {se})"
        if tt in (0, 1):
            assert m2 > 1e-3
    # mmd^2 = <delta, delta> with delta = mu_t - mu_pi evaluated by the identity on mu_t:
    # ||mu_t||^2 = ||mu_pi||^2 + MMD^2 checked through the exact mean of mu_t under pi_t (MC)
    tt = 2
    ch2 = r3.GroupedChains(step, [("x", t.sample_init(n, w.eps, g), torch_gen(24, "cpu"))])
    ch2.advance_to(tt)
    v = w.mu_t(tt, ch2.get("x"))
    lhs = float(v.mean())
    rhs = w.mu_norm2 + w.mmd2(tt)
    assert abs(lhs - rhs) < 5 * float(v.std()) / math.sqrt(n)


def test_ar_witness_trace_vs_branches():
    t, k, A, L, w = _witness()
    step = r3.LinearGaussianStep(A, L)
    N = 100_000
    g = torch_gen(31, "cpu")
    X = t.sample(N, g)
    ch = r3.GroupedChains(step, [("b", torch.cat([X, X], 0), torch_gen(32, "cpu"))])
    for tt in (1, 4):
        ch.advance_to(tt)
        Y, Z = ch.get("b").split(N)
        kyz = r3.paired_kernel(k, Y, Z)
        est = float(kyz.mean()) - w.mu_norm2
        assert abs(est - w.trace(tt)) < 5 * float(kyz.std()) / math.sqrt(N)


def test_paired_kernel_is_diagonal():
    k = ScaledRBF(np.array([0.5, 2.0]), 0.9)
    X = torch.randn(50, 2, dtype=F64)
    Y = torch.randn(50, 2, dtype=F64)
    assert torch.allclose(r3.paired_kernel(k, X, Y), torch.diag(k(X, Y)), atol=1e-15)


def test_feature_operator_accumulate_matches_bounds():
    t = _gauss(d=3)
    k = ScaledRBF(np.ones(3), 1.1)
    feats = NystromFeatures(k, t.sample(64, torch_gen(41, "cpu")), rel_cutoff=1e-10)
    Y, Z, Yp, Zp = (t.sample(400, torch_gen(42 + i, "cpu")) for i in range(4))
    S, Sa, Sb, ftr = r3.feature_operator_accumulate(feats, Y, Z, Yp, Zp, chunk=96)
    ref = op.feature_operator(feats, Y, Z, Yp, Zp)
    assert torch.allclose(S, ref.matrix, atol=1e-14)
    assert torch.allclose(0.5 * (Sa + Sb), S, atol=1e-14)
    direct = float(
        (
            torch.diag(k(Y, Z))
            - torch.diag(k(Y, Zp))
            - torch.diag(k(Yp, Z))
            + torch.diag(k(Yp, Zp))
        ).mean()
        / 2
    )
    assert abs(ftr - direct) < 1e-14


def _toy_problem(**kw):
    t, k, A, L, w = _witness()
    args = dict(
        namespace="test-r3",
        config_id="toy",
        target=t,
        kernel=k,
        kernel_tag="rbf-test",
        step=r3.LinearGaussianStep(A, L),
        eps=0.8,
        t_grid=[0, 1, 2, 4, 8],
        J_grid=[512, 1024],
        N_grid=[128, 256],
        m_grid=[2, 4],
        n_dirs=4,
        N_pi=1024,
        budget=bd.BudgetAllocation(0.05, {"replicas": 0.02, "target": 0.01, "operator": 0.02}),
        embedding=rbf_mean_embedding(t, k),
        kernel_chunk=256,
        head=64,
    )
    args.update(kw)
    return r3.SuiteProblem(**args), w


def test_budget_sums_to_beta():
    p, _ = _toy_problem()
    b = p.budget_check()
    assert b["sum_equals_total"]
    assert abs(b["replicas_weight_sum"] - 1.0) < 1e-15
    assert abs(b["operator_weight_sum"] - 1.0) < 1e-15
    assert b["replicas_cells"] == 5 * 2 and b["operator_cells"] == 5 * 2
    assert p.b_t[0] == pytest.approx(
        1 / math.sqrt(512) + math.sqrt(2 * math.log(10 / 0.02) / 512), rel=1e-14
    )
    assert p.e_tN[0, 1] == pytest.approx(
        2 / math.sqrt(256) * (1 + math.sqrt(2 * math.log(10 / 0.02))), rel=1e-14
    )


def test_end_to_end_witness_cpu():
    p, w = _toy_problem()
    dev = torch.device("cpu")
    ref = r3.run_reference(
        p,
        J_ref=8192,
        N_refop=512,
        n_landmarks=96,
        beta_ref=0.01,
        device=dev,
        quad_margin=0.0,
        log=lambda *_: None,
    )
    truth = np.sqrt([w.mmd2(t) for t in p.t_grid])
    assert (ref["L"] <= truth).all() and (truth <= ref["U"]).all(), (ref["L"], truth, ref["U"])
    p.refop = r3.load_refop(ref, dev)
    outs = [r3.run_ensemble(p, rep, "smoke", dev, analytic=w) for rep in range(3)]
    for o in outs:
        assert (o["replica_event"] <= p.b_t[None, :]).all()
        assert o["op_own_check"].max() < 1e-10
        assert np.abs(o["op_trace"] - o["op_trace_paired"]).max() < 1e-8
    keys = [k_ for k_ in outs[0] if k_ != "timing"]
    data = {k_: np.stack([np.asarray(o[k_]) for o in outs]) for k_ in keys}
    tkeys = sorted(outs[0]["timing"])
    data["timing"] = np.array([[o["timing"][x] for x in tkeys] for o in outs])
    data["timing_keys"] = np.array(tkeys)
    setup = p.setup_arrays()
    asm = r3.assemble_bounds(setup, data, ref, quad_margin_known=0.0)
    raw = r3.assemble_bounds(setup, data, ref, quad_margin_known=0.0, gram_correction=False)
    R = p.r_init
    bias = p.b_t[None, None, :, None, None] + p.b_pi * np.array([1, 0])
    H = data["proj"][:, :, :, 0] + bias
    U = R * data["U3_unit"][:, None, 0, :]
    assert np.allclose(raw["B"]["B3_N128"], np.sqrt(H**2 + U[:, :, None, :, None] ** 2), rtol=1e-14)
    # Gram correction: V3 (t = 0 own span) inflated by 1/sqrt(1 - min(m, kd) e_0)
    om = r3.span_gram_omega(setup, data, ref)
    k0 = np.minimum(np.array(p.m_grid)[None, :], data["op_kdirs"][:, 0, 0, None])
    assert np.array_equal(om[:, 3, 0], k0 * data["op_orth_err"][:, 0, 0, None])
    Hc = data["proj"][:, :, :, 0] / np.sqrt(1.0 - om[:, :, None, 0, :, None]) + bias
    assert np.allclose(
        asm["B"]["B3_N128"], np.sqrt(Hc**2 + U[:, :, None, :, None] ** 2), rtol=1e-14
    )
    for name in asm["B"]:
        assert (asm["B"][name] >= raw["B"][name]).all(), name
    # quadrature margin: + q on the known cells (H and direct) only
    q = 1e-3
    aq = r3.assemble_bounds(setup, data, ref, quad_margin_known=q)
    assert np.allclose(
        aq["H"]["B4own_N256"][..., 1] - asm["H"]["B4own_N256"][..., 1], q, atol=1e-15
    )
    assert np.array_equal(aq["H"]["B4own_N256"][..., 0], asm["H"]["B4own_N256"][..., 0])
    assert np.allclose(aq["B"]["direct"][..., 1] - asm["B"]["direct"][..., 1], q, atol=1e-15)
    # every guaranteed bound covers the analytic truth here (B >= MMD)
    for name, B in asm["B"].items():
        if asm["guaranteed"][name]:
            assert (B >= truth[None, :, None, None, None]).all(), name
    design = {"N_pi": p.N_pi, "M": 96, "init_draws_per_replica": 1.2}
    summ = r3.summarize(setup, data, ref, ref, quad_margin_known=0.0, design=design)
    assert summ["joint_guaranteed_sampled"]["any_violation"] == 0
    co = summ["corrections"]
    assert co["omega_max"] == om.max() and co["methods"]["direct"]["max_change"] == 0.0
    pr = summ["paired"]["B3_N128-B4own_N128"]
    assert (
        pr["n_entries"] == 3 * len(p.t_grid) * 2 * 2 * 2
        and 0 <= pr["n_negative"] <= pr["n_entries"]
    )
    assert pr["n_negative_1e5"] <= pr["n_negative"]
    assert pr["n_negative_t_pos_1e5"] <= pr["n_negative_t_pos"]
    assert sum(pr["n_negative_by_t_1e5"].values()) == pr["n_negative_1e5"]
    assert {c["B4"] for c in summ["transition_matched"]} <= {"B4same_N128@J512", "B4own_N128@J512"}
    for cc in summ["cost_comparison"]:
        assert cc["modeled_seconds_B4"] > 0
    # applicability: only the direct bound applies (a conventional start at an assumed radius)
    sa = r3.summarize(setup, data, ref, ref, quad_margin_known=0.0, applicable=["direct"])
    assert sa["labels"]["direct"] == "guarantee"
    assert sa["labels"]["B4own_N256"] == "assumed radius (NOT a guarantee)"
    assert sa["labels"]["B4ref_own"] == "plug-in (NOT a guarantee)"
    assert sa["joint_guaranteed_sampled"]["methods"] == ["direct"]
    ja = sa["joint_assumed_radius_sampled"]
    assert "direct" not in ja["methods"] and "B3_N128" in ja["methods"]
    assert ja["radius_misspecification_exceedance"] == ja["any_violation"]
    assert (
        "joint_assumed_radius_sampled" not in summ
        and "methods" not in summ["joint_guaranteed_sampled"]
    )
    with pytest.raises(ValueError):
        r3.summarize(setup, data, ref, ref, quad_margin_known=0.0, applicable=["B4ref_own"])
    # NOref bounds are ensemble-invariant (one curve repeated): flagged as such
    assert summ["methods"]["NOref4"]["J512|m2|sampled"]["ensemble_invariant"]
    assert not summ["methods"]["B4own_N256"]["J512|m2|sampled"]["ensemble_invariant"]


def test_end_to_end_matern_sampled_reference_cpu():
    """Matern kernel: no exact mu_pi -> sampled reference target in both stages, refs == ['sampled']."""
    from mcmc.paperexp.kernels import ScaledMatern32

    t, _, A, L, w = _witness()
    k = ScaledMatern32(np.ones(3), 1.3)
    p, _ = _toy_problem(kernel=k, kernel_tag="matern32-test", embedding=None)
    assert p.refs == ["sampled"]
    dev = torch.device("cpu")
    ref = r3.run_reference(
        p,
        J_ref=4096,
        N_refop=512,
        n_landmarks=96,
        beta_ref=0.01,
        device=dev,
        quad_margin=0.0,
        N_ref_target=4096,
        log=lambda *_: None,
    )
    assert float(ref["beta_ref_replicas"]) + float(ref["beta_ref_target"]) == pytest.approx(0.01)
    assert float(ref["b_ref_pi"]) > 0
    p.refop = r3.load_refop(ref, dev)
    outs = [r3.run_ensemble(p, rep, "smoke", dev) for rep in range(2)]
    keys = [k_ for k_ in outs[0] if k_ != "timing"]
    data = {k_: np.stack([np.asarray(o[k_]) for o in outs]) for k_ in keys}
    tkeys = sorted(outs[0]["timing"])
    data["timing"] = np.array([[o["timing"][x] for x in tkeys] for o in outs])
    data["timing_keys"] = np.array(tkeys)
    assert data["proj"].shape[-1] == 1 and data["direct"].shape[-1] == 1
    summ = r3.summarize(p.setup_arrays(), data, ref, ref, quad_margin_known=0.0)
    assert "joint_guaranteed_known" not in summ
    js = summ["joint_guaranteed_sampled"]
    assert js["covered_all"] + js["any_violation"] + js["unresolved"] == 2


def _toy_ready(**kw):
    p, w = _toy_problem(**kw)
    ref = r3.run_reference(
        p,
        J_ref=2048,
        N_refop=256,
        n_landmarks=64,
        beta_ref=0.01,
        device=torch.device("cpu"),
        quad_margin=0.0,
        log=lambda *_: None,
    )
    p.refop = r3.load_refop(ref, torch.device("cpu"))
    return p, w


_SKIP = {"timing", "wall", "wall_cum", "op_time", "peak_gpu_bytes", "peak_rss_kb", "batch_size"}


def test_batched_ensembles_equal_solo_runs():
    p, _ = _toy_ready()
    dev = torch.device("cpu")
    solo = [r3.run_ensemble(p, r, "smoke", dev) for r in (3, 5)]
    batch = r3.run_ensembles(p, [3, 5], "smoke", dev)
    for a, b in zip(solo, batch):
        for key in a:
            if key in _SKIP:
                continue
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True), key


def test_enclosed_operator_upper_bounds_fp64_values():
    p64, _ = _toy_ready(N_grid=[256])
    pen, _ = _toy_ready(N_grid=[256], op_method="fp32-enclosed", n_refined=32)
    dev = torch.device("cpu")
    a = r3.run_ensemble(p64, 0, "smoke", dev)
    b = r3.run_ensemble(pen, 0, "smoke", dev)
    assert np.array_equal(a["proj"][:, :, 2:], b["proj"][:, :, 2:])  # plug-in spans unaffected
    assert np.array_equal(a["direct"], b["direct"])
    for key in ("U3_unit", "U4same_unit", "U4own_unit", "NO4_unit"):
        assert (b[key] >= a[key] - 1e-6).all(), key
        assert np.abs(b[key] / a[key] - 1).max() < 1e-2, key


def test_fused_rowsum_matches_dense_cpu_fallback():
    t, k, *_ = _witness()
    X, Y = t.sample(300, torch_gen(51, "cpu")), t.sample(200, torch_gen(52, "cpu"))
    ck = r3.CountingKernel(k, fused=True)
    assert torch.allclose(ck.rowsum(X, Y), k(X, Y).sum(1), rtol=1e-13, atol=0)
    assert r3._block_vstat(ck, X, 1 << 20) == pytest.approx(float(k(X, X).sum()), rel=1e-13)


def test_phase_overrides_on_struct_config():
    """Phase overrides merge into a structured Hydra config and are recorded."""
    import pathlib

    from omegaconf import OmegaConf

    from mcmc.experiments.exp_r3b_anytime_suite import apply_phase_overrides

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "configs/experiment/exp_r3b_anytime_suite.yaml"
    )
    for phase, mode in (("main", "ensembles"), ("stress", "ensembles"), ("smoke", "reference")):
        cfg = OmegaConf.load(path)
        cfg.phase, cfg.mode = phase, mode
        OmegaConf.set_struct(cfg, True)
        m = apply_phase_overrides(cfg)
        if phase in ("main", "stress"):
            assert list(m.N_grid) == [2048] and m.operator.method == "fp32-enclosed"
            assert m.batch_ensembles == 8 and m.chunk_size % m.batch_ensembles == 0
        assert m.applied_phase_overrides == ("reference" if mode == "reference" else phase)
    cfg = OmegaConf.load(path)
    cfg.phase = "screen"
    OmegaConf.set_struct(cfg, True)
    m = apply_phase_overrides(cfg)
    assert list(m.N_grid) == [1024, 2048] and m.operator.method == "fp64"


def test_config_overrides_after_phase_overrides():
    import pathlib

    from omegaconf import OmegaConf

    from mcmc.experiments.exp_r3b_anytime_suite import apply_phase_overrides

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "configs/experiment/exp_r3b_anytime_suite.yaml"
    )
    cfg = OmegaConf.load(path)
    cfg.phase, cfg.mode, cfg.config_id = "stress", "ensembles", "gauss-d2-c25__rwm"
    OmegaConf.set_struct(cfg, True)
    m = apply_phase_overrides(cfg)
    assert m.operator.method == "fp64" and m.operator.rtol == 1e-9 and list(m.N_grid) == [2048]
    assert m.applied_config_overrides == "gauss-d2-c25__rwm"
    cfg = OmegaConf.load(path)
    cfg.phase, cfg.mode, cfg.config_id = "stress", "ensembles", "gauss-d50-c25__rwm"
    OmegaConf.set_struct(cfg, True)
    m = apply_phase_overrides(cfg)
    assert m.operator.method == "fp32-enclosed" and m.applied_config_overrides is None


def test_fewer_positive_directions_than_requested():
    """n_pos < n_dirs: the own span uses all positive directions and U falls back to R sqrt(e)."""
    p, _ = _toy_ready(N_grid=[4], m_grid=[2, 8], n_dirs=8)
    o = r3.run_ensemble(p, 0, "smoke", torch.device("cpu"))
    kd = o["op_kdirs"][:, 0]
    assert (kd >= 1).all() and (kd <= 4).all()
    m8 = o["U4own_unit"][:, 0, 1]  # m = 8 > kd: lambda^+_{kd+1} = 0
    assert np.allclose(m8, np.sqrt(o["e_tot"][:, 0]), rtol=1e-12)
    assert np.isfinite(o["proj"]).all()


def test_declared_eigensolve_fallback_is_counted(monkeypatch):
    """A declared fallback retries the solve and records the event.

    Without a fallback, EigensolveNotConverged propagates to the caller."""
    p, _ = _toy_ready(N_grid=[256], op_method="fp32-enclosed", n_refined=32)
    dev = torch.device("cpu")
    ref_out = r3.run_ensemble(p, 0, "smoke", dev)
    orig = op.estimate_operator
    calls = {"n": 0}

    def flaky(kernel, *blocks, method="fp64", **kw):
        if method == "fp32-enclosed":
            calls["n"] += 1
            if calls["n"] == 3:  # the third checkpoint's solve
                raise op.EigensolveNotConverged("forced")
        return orig(kernel, *blocks, method=method, **kw)

    monkeypatch.setattr(op, "estimate_operator", flaky)
    with pytest.raises(op.EigensolveNotConverged):
        r3.run_ensemble(p, 0, "smoke", dev)
    calls["n"] = 0
    p.op_fallback = "chol-fp64"
    out = r3.run_ensemble(p, 0, "smoke", dev)
    assert out["op_fallback"][:, 0].tolist() == [0, 0, 1, 0, 0]
    assert ref_out["op_fallback"].sum() == 0
    same = [i for i in range(5) if i != 2]
    assert np.array_equal(out["U4own_unit"][same], ref_out["U4own_unit"][same])
    assert np.array_equal(out["direct"], ref_out["direct"])
    a, b = out["U4own_unit"][2, 0], ref_out["U4own_unit"][2, 0]
    assert (np.abs(a / b - 1) < 1e-3).all()


def test_init_laws():
    """Initialization formulas and random streams match their declared settings."""
    from omegaconf import OmegaConf

    from mcmc.experiments.exp_r3b_anytime_suite import namespace

    p, _ = _toy_problem()
    a, na = r3.draw_init(p, 500, torch_gen(9, "cpu"))
    b, nb = r3.sample_init_counted(p.target, 500, p.eps, torch_gen(9, "cpu"))
    assert torch.equal(a, b) and na == nb
    s = np.array([0.5, 2.0, 1.0])
    pp, _ = _toy_problem(init_kind="point", init_scale=3.0, init_scales=s)
    x, n = r3.draw_init(pp, 7, torch_gen(9, "cpu"))
    want = torch.as_tensor(p.target.mean(), dtype=F64).clone()
    want[1] += 6.0
    assert n == 0 and torch.equal(x, want[None].repeat(7, 1))
    pd, _ = _toy_problem(init_kind="dispersed", init_scale=3.0)
    x, n = r3.draw_init(pd, 50, torch_gen(9, "cpu"))
    m = torch.as_tensor(p.target.mean(), dtype=F64)
    assert n == 50 and torch.allclose(x, m + 3.0 * (p.target.sample(50, torch_gen(9, "cpu")) - m))
    cfg = OmegaConf.create(
        {
            "protocol_version": "v2",
            "kernel": {"kind": "rbf", "bw": "ell"},
            "eps": 0.8,
            "init": {"kind": "known_radius", "scale": 3.0},
        }
    )
    assert namespace(cfg, "c") == "r3b-v2:c:rbf-ell:eps0.8"
    cfg.init.kind = "point"
    assert namespace(cfg, "c") == "r3b-v2:c:rbf-ell:eps0.8:init-point-x3"


def test_sampled_reference_cross_term_blocked():
    """The memory-blocked cross term of a sampled-target reference equals the row-mean formula."""
    from mcmc.paperexp.kernels import ScaledMatern32, kernel_sum

    k = ScaledMatern32(np.ones(3), 1.3)
    X, Y = torch.randn(3000, 3, dtype=F64), torch.randn(4096, 3, dtype=F64)
    blocked = float(kernel_sum(k, X, Y, mem_bytes=1 << 20).item()) / (3000 * 4096)
    rows = float(bd.kernel_prefix_means(k, X, Y, [4096], chunk=1024)[0].mean().item())
    assert blocked == pytest.approx(rows, rel=1e-13)


def test_certified_span_gram_matches_core_and_bounds_the_exact_defect():
    """The blocked Gram calculation bounds the defect measured in long double."""
    k = ScaledRBF(np.ones(3), 1.1)
    g = torch.Generator().manual_seed(3)
    P = torch.randn(200, 3, generator=g, dtype=torch.float64)
    K = k(P, P)
    w, V = torch.linalg.eigh(K)
    C = V[:, -6:] / w[-6:].sqrt()  # H-orthonormal up to rounding
    C = C + 1e-9 * torch.randn(C.shape, generator=g, dtype=torch.float64)  # a small Gram defect
    S, ES = r3.certified_span_gram(k, P, C, block=64)
    ov = op.certified_span_overlap(k, P, C, op.Span(P, C))
    assert torch.allclose(S, ov["S"], rtol=0, atol=1e-13)
    assert torch.allclose(ES, ov["ES"], rtol=1e-12, atol=0)
    om = op.span_gram_defect(S, ES)
    Kl = K.numpy().astype(np.longdouble)
    Cl = C.numpy().astype(np.longdouble)
    Sl = Cl.T @ Kl @ Cl
    exact = float(np.linalg.norm((Sl - np.eye(6, dtype=np.longdouble)).astype(np.float64), 2))
    assert exact <= om
    lam_min = float(np.linalg.eigvalsh(Sl.astype(np.float64)).min())
    assert lam_min >= 1.0 - om


def test_certify_gram_flag_records_and_changes_nothing_else():
    p, w = _toy_ready()
    dev = torch.device("cpu")
    a = r3.run_ensemble(p, 0, "smoke", dev)
    p.certify_gram = True
    b = r3.run_ensemble(p, 0, "smoke", dev)
    extra = {"gram_cert_omega", "gram_cert_fro_dev", "gram_cert_fro_es"}
    assert set(b) == set(a) | extra
    skip = {"timing", "wall", "wall_cum", "op_time", "peak_rss_kb"}
    for key in a:
        if key not in skip:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True), key
    om = b["gram_cert_omega"]
    assert np.isfinite(om).all() and (om > 0).all() and (om < 0.5).all()


def test_init_chi2_rules():
    from mcmc.paperexp.targets import load_frozen_suite

    s = load_frozen_suite()
    st = s.target("studentt-d10-nu3")
    c = r3.init_chi2(st, "dispersed", 3.0, 0.8)
    assert c["finite"] and c["chi2"] == pytest.approx(7.947361614602057, rel=1e-10)
    assert c["R"] == pytest.approx(2.8191065277144207, rel=1e-10) and c["sup_ratio"] == 27.0
    for tid in ("gauss-d10-c25", "mixture-d10-s6-w0.5", "funnel-d10-sy3"):
        assert not r3.init_chi2(s.target(tid), "dispersed", 3.0, 0.8)["finite"], tid
    assert r3.init_chi2(st, "point", 3.0, 0.8)["chi2"] == math.inf
    assert r3.init_chi2(st, "known_radius", 3.0, 0.8)["R"] == 0.8


def test_method_cost_counts():
    setup = {"t_grid": np.array([0, 1, 2, 4])}
    design = {"N_pi": 100, "M": 10, "init_draws_per_replica": 1.5}
    c = r3.method_cost("B4own_N8", 3, 64, "sampled", setup, design)
    assert c["transitions"] == 64 * 4 + 4 * 8 * 4 and c["solves"] == 4
    assert c["kernel_proj"] == 4 * 4 * 8 * 64 and c["kernel_target"] == 4 * 4 * 8 * 100
    assert c["exact_draws"] == 64 * 1.5 + 100 + 16
    c3 = r3.method_cost("B3_N8", 3, 64, "sampled", setup, design)
    assert c3["transitions"] == 64 * 4 and c3["solves"] == 1
    assert r3.method_cost("NOref4", 3, 64, "sampled", setup, design) == {
        u: 0.0 for u in r3.COST_UNITS
    }
    d = r3.method_cost("direct", 1, 64, "known", setup, design)
    assert d["kernel_direct"] == 2 * (64 * 64 + 64) and d["exact_draws"] == 64 * 1.5
    ds = r3.method_cost("direct", 1, 64, "sampled", setup, design)
    assert ds["kernel_direct"] == 2 * (64 * 64 + 64 * 100) and ds["kernel_target"] == 100 * 100


def test_cost_units_counts_one_nested_operator_population():
    """Smaller N values reuse prefixes of one population of 4*N_max operator chains."""
    setup = {
        "J_grid": np.array([64, 128]),
        "N_grid": np.array([8, 16]),
        "t_grid": np.array([0, 1, 4]),
    }
    E = 3
    data = {
        "timing_keys": np.array(["direct", "projections", "sampling", "target"]),
        "timing": np.tile([1.0, 2.0, 3.0, 4.0], (E, 1)),
        "kernel_evals_cum": np.tile(np.arange(1.0, 6.0) * 10, (E, 3, 1)),
        "op_time": np.full((E, 3, 2), 0.5),
    }
    u = r3.cost_units(setup, data)
    assert u["transitions"] == pytest.approx(3.0 / ((128 + 4 * 16) * 4))
    assert u["kernel_direct"] == pytest.approx(1.0 / 40) and u["kernel_target"] == pytest.approx(
        4.0 / 10
    )


def test_paired_record_two_thresholds():
    Dd = np.zeros((2, 3, 1, 1, 1))
    Dd[0, 1] = -2e-5  # material reversal at t index 1
    Dd[1, 2] = -1e-9  # reversal beyond rounding only
    r = r3._paired_record(Dd, np.array([0, 1, 2]), [4], [2], ["sampled"])
    assert r["n_negative"] == 2 and r["n_negative_1e5"] == 1
    assert r["n_negative_by_t_1e5"] == {0: 0, 1: 1, 2: 0} and r["n_negative_last_t"] == 1
    assert r["n_negative_last_t_1e5"] == 0 and r["n_negative_t_pos_1e5"] == 1
