"""Tests for saved-state MMD estimates, operator widths, and first crossings."""

import importlib.util
import itertools
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from mcmc.paperexp import german_credit_r2 as r2
from mcmc.paperexp.bounds import replica_radius
from mcmc.paperexp.kernels import ScaledRBF

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "c_followup", REPO / "scripts" / "paper_experiments" / "c_realdata_budget_followup.py"
)
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


def test_floor_matches_replica_radius_and_closed_form():
    for n_t, J in itertools.product((14, 15, 16), (4, 16, 64)):
        w = 1.0 / (3 * n_t)
        b = c.replica_floor(J, n_t, 3)
        assert b == pytest.approx(replica_radius(1.0, J, 0.025, w), rel=1e-14)
        assert b == pytest.approx((1 + math.sqrt(2 * math.log(1 / (0.025 * w)))) / math.sqrt(J))
    assert c.replica_floor(64, 16, 3) == pytest.approx(0.611058139, abs=1e-9)
    assert c.replica_floor(4, 14, 3) > math.sqrt(2)  # vacuous at J = 4
    for eps in (0.10, 0.05):
        Jn = c.j_needed(eps, 16, 3)
        w = 1.0 / 48
        assert replica_radius(1.0, Jn, 0.025, w) <= eps < replica_radius(1.0, Jn - 1, 0.025, w)


def test_split_partition_nesting_and_offsets():
    info = c.check_split()
    assert info["max_overlap"] == 0
    off = c.group_offsets()
    assert off == {4: (0, 512), 16: (512, 128), 64: (640, 32)}
    assert c.N_GROUPS_TOTAL == 672
    g4, g64 = c.observation_groups(4), c.observation_groups(64)
    assert set(g64[3]) == set(g4[48:64].ravel())
    assert g64.max() == c.N_OBS - 1


def test_status_and_mmd_scale_keep_signed_estimates():
    est = np.array([-0.002, 0.0001, 0.02, 0.0105])
    se = np.array([0.001, 0.001, 0.001, 0.0001])
    st = c.status_code(est, se, 0.10)
    assert st.tolist() == [-1, -1, 1, 1]
    st5 = c.status_code(np.array([0.0025]), np.array([0.001]), 0.05)
    assert st5.tolist() == [0]
    p, lo, hi = c.mmd_scale_interval(est, se)
    assert np.isnan(p[0]) and lo[0] == 0 and hi[0] == 0  # interval [-0.004, 0]: unresolved from 0
    assert hi[1] == pytest.approx(math.sqrt(0.0021)) and lo[1] == 0
    with pytest.raises(ValueError):
        c.status_code(np.array([0.1]), np.array([-1.0]), 0.1)


def test_plugin_and_first_crossing():
    h2 = np.array([[0.04, 0.0], [0.0064, 0.0], [0.0001, 0.0]])  # (n_t, groups)
    w2 = np.array([0.0, 0.01, 0.001])
    P = c.plugin(h2, w2, 1.0)
    np.testing.assert_allclose(P[:, 0], [0.2, math.sqrt(0.0164), math.sqrt(0.0011)])
    assert c.first_crossing(P, 0.10).tolist() == [2, 0]
    assert c.first_crossing(P, 0.01).tolist() == [-1, 0]
    with pytest.raises(ValueError):
        c.plugin(-h2, w2, 1.0)


def test_operator_widths_known_spectrum():
    """Own-span width is lambda_{m+1}; a static span cannot improve on it."""
    rng = np.random.default_rng(3)
    r, n_t = 20, 3
    Q, _ = np.linalg.qr(rng.normal(size=(r, r)))
    Os, OA, OB = [], [], []
    for i in range(n_t):
        lam = np.sort(rng.uniform(0, 1, r) * 0.5**i)[::-1]
        lam[-2:] = [-1e-4, -2e-4]
        Qi = Q if i == 0 else np.linalg.qr(Q + 0.3 * rng.normal(size=(r, r)))[0]
        M = (Qi * lam) @ Qi.T
        E = 1e-3 * rng.normal(size=(r, r))
        E = 0.5 * (E + E.T)
        Os.append(M)
        OA.append(M + E)
        OB.append(M - E)
    sig = {"grid": np.arange(n_t), "O": np.stack(Os), "OA": np.stack(OA), "OB": np.stack(OB)}
    w = c.operator_widths(sig, list(range(n_t)))
    for i in range(n_t):
        lam = np.linalg.eigvalsh(Os[i])[::-1]
        for m in (4, 8):
            assert w[f"w2_S4own_m{m}"][i] == pytest.approx(max(lam[m], 0), abs=1e-12)
            assert w[f"w2_S4same_m{m}"][i] >= w[f"w2_S4own_m{m}"][i] - 1e-12
        assert w["eps"][i] == pytest.approx(np.linalg.norm(OA[i] - OB[i], 2) / 2, rel=1e-12)
    lam0 = np.linalg.eigvalsh(Os[0])[::-1]
    assert np.allclose(w["w2_S3_m8"], lam0[8])
    # at t = 0 the static span is the own span
    assert w["w2_S4same_m8"][0] == pytest.approx(w["w2_S4own_m8"][0], abs=1e-12)
    assert (w["most_negative"] < 0).all()


def _toy_eval_stats(seed=0, d=2, G=16, n=8, R=16, N=40):
    gen = np.random.default_rng(seed)
    k = ScaledRBF(np.ones(d), 1.0)
    X = torch.as_tensor(gen.normal(size=(G * n, d)) + 0.3)
    runs = [torch.as_tensor(gen.normal(size=(N, d))) for _ in range(R)]
    K = k(X, X).numpy()
    S = K.reshape(G, n, G, n).sum(axis=(1, 3))
    cross = np.stack([k(X, Y).numpy().sum(1) / N for Y in runs], axis=1).reshape(G, n, R).sum(1)
    inner = np.array([[k(A, B).numpy().mean() for B in runs] for A in runs])
    return k, X, runs, S, cross, inner


def test_mmd2_on_held_out_subset_matches_brute_force_u_statistic():
    k, X, runs, S, cross, inner = _toy_eval_stats()
    n = 8
    groups, held = np.arange(8, 16), np.arange(8, 16)
    est = r2.mmd2_unbiased(S, cross, inner, n, groups, held)
    Xe = X[8 * n :]
    J = Xe.shape[0]
    Kxx = k(Xe, Xe).numpy()
    a = (Kxx.sum() - np.trace(Kxx)) / (J * (J - 1))
    b = np.mean([k(Xe, runs[r]).numpy().mean() for r in held])
    cc = np.mean([k(runs[r], runs[s]).numpy().mean() for r in held for s in held if r != s])
    assert est == pytest.approx(a - 2 * b + cc, rel=1e-12)


def test_heldout_terms_wiring_on_toy_features():
    """Projected and residual feature energies add to the total."""
    rng = np.random.default_rng(5)
    r, G, R = 6, 16, 16
    gf = rng.normal(size=(G, r)) * 0.1 + 0.2
    rf = rng.normal(size=(R, r)) * 0.01
    ref = r2.RefSummary(
        [f"primary:{q}" for q in range(R)],
        [],
        rf,
        rng.normal(size=(R, R)),
        np.zeros((R, 2)),
        {},
        {},
        np.array([0.5]),
    )
    ref.run_inner = 0.5 * (ref.run_inner + ref.run_inner.T)
    stats = {
        "group_feat": gf,
        "group_pair_ksum": np.abs(rng.normal(size=(G, G))) * 100,
        "group_run_cross": np.abs(rng.normal(size=(G, R))) * 50,
        "n_per_group": 128,
    }
    stats["group_pair_ksum"] = 0.5 * (stats["group_pair_ksum"] + stats["group_pair_ksum"].T)
    Q = np.linalg.qr(rng.normal(size=(r, 3)))[0]
    h = c.heldout_terms(stats, ref, np.arange(8, 16), [Q, Q[:, :2]])
    assert h["mmd2_loo_run"].shape == (8,)
    np.testing.assert_allclose(h["res_feat2"] + h["proj2"], h["feat2"], rtol=1e-12)
    np.testing.assert_allclose(h["res_full2"] + h["proj2"], h["mmd2"], rtol=1e-12)
    assert h["omit2"] == pytest.approx(h["mmd2"] - h["feat2"])
    delta = gf.mean(0) - rf[8:].mean(0)
    raw = ((Q.T @ delta) ** 2).sum()
    assert h["proj2"][0] < raw  # debiased below the raw squared norm


def test_regression_settings_and_method_parsing():
    import json

    d = json.loads((REPO / "paper/figures/data/r3_concrete/summary_main.json").read_text())
    st = c.regression_settings(d)
    assert [(s["direction"], s["R_init"]) for s in st] == [
        ("slow", 0.8),
        ("slow", 0.2),
        ("fast", 0.8),
        ("fast", 0.2),
    ]
    assert c.parse_method("B4b_own_N2048") == ("B4b_own", 2048)
    assert c.parse_method("direct") == ("direct", None)
    assert set(c.REG_METHODS) >= {c.parse_method(k)[0] for k in d["methods"]}


_rspec = importlib.util.spec_from_file_location(
    "c_repl", REPO / "scripts" / "paper_experiments" / "c_realdata_replication.py"
)
crep = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(crep)


def test_replication_two_sample_mmd2_matches_brute_force():
    rng = np.random.default_rng(7)
    k = ScaledRBF(np.ones(3), 1.1)
    X, Y = rng.normal(size=(40, 3)), rng.normal(size=(48, 3)) + 0.2
    est, se = crep.two_sample_mmd2(k, X, Y, 5, 6)
    Kxx = k(torch.as_tensor(X), torch.as_tensor(X)).numpy()
    Kyy = k(torch.as_tensor(Y), torch.as_tensor(Y)).numpy()
    Kxy = k(torch.as_tensor(X), torch.as_tensor(Y)).numpy()
    a = (Kxx.sum() - np.trace(Kxx)) / (40 * 39)
    b = (Kyy.sum() - np.trace(Kyy)) / (48 * 47)
    assert est == pytest.approx(a + b - 2 * Kxy.mean(), rel=1e-12)
    assert se > 0


def test_replication_design_counts_and_seeds():
    des = crep.design()
    assert des["n_trajectories"] == 15360
    assert crep.GRID[0] == 0 and crep.GRID[-1] == 4096 and len(crep.GRID) == 14
    seeds = {
        crep.bundle_seed(m, s, law, b)
        for m in crep.MODELS
        for s in crep.SAMPLERS
        for law in crep.LAWS
        for b in range(crep.N_BUNDLES)
    }
    assert len(seeds) == 240


def test_negative_interval_is_flagged_not_zero():
    est = np.array([-1.4625e-4 + 7.183e-5, 0.02, -1e-5])
    se = np.array([3.5917e-5, 0.001, 1e-5])
    flags = c.interval_flag(est, se)
    assert flags.tolist() == ["NEGATIVE_INTERVAL", "POSITIVE", "CONTAINS_ZERO"]
    p, lo, hi = c.mmd_scale_interval(est, se)
    assert np.isnan(lo[0]) and np.isnan(hi[0]) and np.isnan(p[0])  # never [0, 0]
    assert lo[2] == 0 and hi[2] == pytest.approx(math.sqrt(1e-5))


def test_wrong_expected_hash_raises(tmp_path, monkeypatch):
    f = tmp_path / "a.npz"
    f.write_bytes(b"abc")
    good = c.sha256_file(f)
    assert c.verify_hashes({str(f): good}) == 1
    with pytest.raises(RuntimeError, match="differs from the manifest"):
        c.verify_hashes({str(f): "0" * 64})
    # verify_sources also checks the NUTS reference-set files of the unit
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    (ref_dir / "unit_000.npz").write_bytes(b"draws")
    monkeypatch.setattr(c, "unit_reference_ids", lambda model: ["set_a"])
    man = {
        "units": {"m__mala": {"sources": {str(f): {"sha256": good}}}},
        "reference_sets": {"set_a": {"path": str(ref_dir), "units": {"unit_000.npz": "0" * 64}}},
    }
    with pytest.raises(RuntimeError, match="differs from the manifest"):
        c.verify_sources(man, "m__mala")
    man["reference_sets"]["set_a"]["units"]["unit_000.npz"] = c.sha256_file(
        ref_dir / "unit_000.npz"
    )
    assert c.verify_sources(man, "m__mala") == {"unit_sources": 1, "reference_files": 1}


def test_regression_inputs_verified_against_manifest():
    import copy
    import json as _json

    man = _json.loads((c.OUT / "manifest.json").read_text())
    assert c.verify_regression_inputs(man) == 5
    bad = copy.deepcopy(man)
    k = next(iter(bad["regression_summaries"]["concrete"]))
    bad["regression_summaries"]["concrete"][k] = "0" * 64
    with pytest.raises(RuntimeError):
        c.verify_regression_inputs(bad)


def test_paired_swap_se_uses_shared_replicas():
    """Pairing removes shared group noise from the difference of two estimates."""
    k, X, runs, S, cross, inner = _toy_eval_stats(seed=3, G=16, n=8, R=16, N=40)
    G = 16
    old = c.R2_GROUP
    c.R2_GROUP = 8  # toy group size
    try:
        d, se, eH, eS = c.paired_swap_from_stats(S, cross, inner, np.arange(8, 16), np.arange(0, 8))
    finally:
        c.R2_GROUP = old
    _, seH = r2.mmd2_with_se(S, cross, inner, 8, np.arange(8, 16))
    _, seS = r2.mmd2_with_se(S, cross, inner, 8, np.arange(0, 8))
    assert d == pytest.approx(eH - eS)
    assert 0 < se < math.sqrt(seH**2 + seS**2)
    assert G == S.shape[0]


def test_floor_reproduces_saved_table_without_inventory(tmp_path, monkeypatch):
    expected = (c.OUT / "floor_table.csv").read_bytes()
    monkeypatch.setattr(c, "OUT", tmp_path)
    c.cmd_floor(None)
    assert (tmp_path / "floor_table.csv").read_bytes() == expected
    assert {p.name for p in tmp_path.iterdir()} == {"floor_table.csv"}


def test_floor_uses_configured_horizon_and_operator_budget(monkeypatch):
    monkeypatch.setattr(c, "load_yaml", lambda _: {
        "frozen": {"toy__mala": {"T": 8}},
        "horizon_min": 16,
        "sigma": {"N": 32768},
    })
    rows = c.floor_table()
    assert len(rows) == len(c.J_GRID)
    for row in rows:
        assert row["n_inspections"] == 6
        assert row["operator_N"] == 32768
        assert row["units"] == "toy__mala"
        assert row["floor_b_tJ"] == pytest.approx(c.replica_floor(row["J"], 6, 3))
        assert row["operator_floor_R1"] == pytest.approx(c.operator_floor(1.0, 32768, 6))


def test_verify_checks_inputs_without_writing_a_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c, "REPO", tmp_path)
    monkeypatch.setattr(c, "OUT", tmp_path / "export")
    monkeypatch.setattr(c, "unit_reference_ids", lambda _: ["reference"])
    c.OUT.mkdir()
    samples = tmp_path / "samples.npz"
    samples.write_bytes(b"chain samples")
    reference = tmp_path / "reference"
    reference.mkdir()
    draws = reference / "unit_000.npz"
    draws.write_bytes(b"independent target draws")
    summary = tmp_path / "paper/figures/data/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"value": 3.0}')
    manifest = {
        "units": {"toy__mala": {"sources": {str(samples): {"sha256": c.sha256_file(samples)}}}},
        "reference_sets": {"reference": {"path": str(reference), "units": {draws.name: c.sha256_file(draws)}}},
        "regression_summaries": {"toy": {summary.name: c.sha256_file(summary)}},
    }
    (c.OUT / "manifest.json").write_text(json.dumps(manifest))
    c.cmd_verify(None)
    result = json.loads(capsys.readouterr().out)
    assert result["all_match"]
    assert result["regression_inputs"] == 1
    assert result["distinct_reference_files"] == 1
    assert {p.name for p in c.OUT.iterdir()} == {"manifest.json"}
    draws.write_bytes(b"changed target draws")
    with pytest.raises(RuntimeError, match="differs from the manifest"):
        c.cmd_verify(None)
