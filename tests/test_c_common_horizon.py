"""Tests for exact checkpoint transitions and common-horizon bounds."""

import math
import pathlib

import mpmath as mp
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mcmc.experiments import exp_c_common_horizon as chz
from mcmc.experiments import exp_r3_concrete as r3
from mcmc.paperexp import checkpoint_chain as ckc
from mcmc.paperexp import gaussian_linear as gl

ROOT = pathlib.Path(__file__).resolve().parents[1]
F64 = torch.float64


@pytest.fixture(scope="module")
def slow3():
    C = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    mu = np.array([0.3, -0.2, 0.1])
    return gl.gibbs_chain(np.linalg.inv(C), mu), gl.AnisoRBF(np.array([1.0, 0.8, 1.2]), 1.1)


def test_gap_table_identities(slow3):
    ch, _ = slow3
    tab = ckc.gap_transitions(ch, [1, 2, 4, 8, 16, 32, 64, 128])
    for g, tr in tab.items():
        assert np.abs(tr.A - np.linalg.matrix_power(ch.A, g)).max() < 1e-13
        assert np.abs(tr.Q - ch.noise_cov(g)).max() < 1e-13
        assert np.abs(tr.A @ ch.C @ tr.A.T + tr.Q - ch.C).max() < 1e-13
        assert np.linalg.eigvalsh(tr.Q)[0] > 0
        assert np.abs(tr.L @ tr.L.T - tr.Q).max() < 1e-14
    assert np.array_equal(tab[1].L, ch.LQ) and np.array_equal(tab[1].A, ch.A)
    with pytest.raises(ValueError):
        ckc.gap_transitions(ch, [3])


def test_gap1_is_bitwise_the_one_step_sweep(slow3):
    ch, _ = slow3
    ex = ckc.CheckpointChain.from_chain(ch, [1, 2])
    x0 = ch.sample_stationary(1000, torch.Generator().manual_seed(1), "cpu", shift=np.ones(3))
    a = ch.sweep(x0, torch.Generator().manual_seed(7), 1, ch.torch_params("cpu"))
    b = ex.sweep(x0, torch.Generator().manual_seed(7), 1)
    assert torch.equal(a, b)
    assert ex.rows_transitioned == 1000 and ex.rows_sweeps_represented == 1000
    with pytest.raises(KeyError):
        ex.sweep(x0, torch.Generator().manual_seed(7), 4)


@pytest.mark.parametrize("g", [2, 4, 16])
def test_exact_gap_matches_repeated_one_step_in_law(slow3, g):
    """Conditional mean/covariance from a fixed state: exact one-draw transition vs g one-step sweeps."""
    ch, _ = slow3
    ex = ckc.CheckpointChain.from_chain(ch, [g])
    n = 400_000
    x0 = torch.as_tensor(ch.mu + np.array([1.5, -1.0, 0.5]), dtype=F64).expand(n, 3).contiguous()
    Xe = ex.sweep(x0, torch.Generator().manual_seed(11), g).numpy()
    Xs = ch.sweep(x0, torch.Generator().manual_seed(12), g).numpy()
    m = ch.mu + ch.power(g) @ (x0[0].numpy() - ch.mu)
    Qg = ch.noise_cov(g)
    se = np.sqrt(np.diag(Qg) / n)
    for X in (Xe, Xs):
        assert np.abs((X.mean(0) - m) / se).max() < 5
        se_cov = np.sqrt((np.outer(np.diag(Qg), np.diag(Qg)) + Qg**2) / n)
        assert np.abs((np.cov(X.T) - Qg) / se_cov).max() < 5
    assert np.abs((Xe.mean(0) - Xs.mean(0)) / (se * math.sqrt(2))).max() < 5


def test_joint_checkpoint_law_lag_covariance(slow3):
    """Exact checkpoint simulation keeps the temporal dependence: Cov(X_{s+g}, X_s) = A^g C."""
    ch, _ = slow3
    ck = [0, 1, 2, 4, 8, 16]
    ex = ckc.CheckpointChain.from_chain(ch, ckc.grid_gaps(ck))
    gen = torch.Generator().manual_seed(5)
    n = 300_000
    X = ch.sample_stationary(n, gen, "cpu")
    for g in ckc.grid_gaps(ck):
        Y = ex.sweep(X, gen, g)
        Xc, Yc = X.numpy() - ch.mu, Y.numpy() - ch.mu
        lag = Yc.T @ Xc / n
        target = ch.power(g) @ ch.C
        se = np.sqrt((np.outer(np.diag(ch.C), np.diag(ch.C)) + target**2) / n)
        assert np.abs((lag - target) / se).max() < 5
        X = Y


def test_log_mmd_path_matches_direct_formula_and_eigen_asymptotics(slow3):
    ch, k = slow3
    ck = [0] + [2**j for j in range(13)]  # to 4096
    a = 0.3 * np.array([1.0, -0.5, 0.2])
    ln = ckc.log_mmd_path(k, ch.C, ch.A, a, ck)
    for t, v in zip(ck, ln):
        direct = math.sqrt(gl.mmd2_shifted_gaussians(k, ch.C, ch.power(t) @ a))
        if direct * direct > ckc.TINY:
            assert abs(math.log(direct) - v) < 1e-9
            assert ckc.mmd_status(direct, v) == "RESOLVED"
        else:
            assert ckc.mmd_status(direct, v) in ("MMD2_UNDERFLOW", "BELOW_FLOAT64")
    # far tail: ln MMD(t) - ln MMD(t/2) -> (t/2) ln|lambda_1| (real dominant eigenvalue here)
    lam1 = max(np.abs(np.linalg.eigvals(ch.A)))
    assert abs((ln[-1] - ln[-2]) - 2048 * math.log(lam1)) < 1e-6 * 2048 * abs(math.log(lam1))
    # MMD^2 underflows while MMD is a normal float64 (recovered from the log value), and below float64
    assert ckc.mmd_status(0.0, math.log(7e-296)) == "MMD2_UNDERFLOW"
    assert abs(ckc.mmd_from_log(math.log(7e-296)) / 7e-296 - 1) < 1e-12
    assert ckc.mmd_status(1e-160, math.log(1e-160)) == "MMD2_UNDERFLOW"
    assert ckc.mmd_status(0.0, -1000.0) == "BELOW_FLOAT64" and ckc.mmd_from_log(-1000.0) is None
    with pytest.raises(FloatingPointError):
        ckc.mmd_status(0.2, math.log(0.3))


def _mp_moment(kernel, za, zb, mu, C, Gam, dps=60):
    """<phi(za), Sigma_t phi(zb)> from the definition in extended precision (exp(expo_t) - exp(expo_0))."""
    mp.mp.dps = dps
    d = len(mu)
    Lam = mp.diag([mp.mpf(float(s)) ** 2 for s in np.asarray(kernel.scales) * kernel.ell])
    Cm, Gm = mp.matrix(C.tolist()), mp.matrix(Gam.tolist())
    B = Lam + Cm
    S2 = mp.matrix(2 * d, 2 * d)
    for i in range(d):
        for j in range(d):
            S2[i, j], S2[d + i, d + j] = B[i, j], B[i, j]
            S2[i, d + j], S2[d + i, j] = Gm[i, j], Gm[j, i]
    w = mp.matrix([float(x) for x in np.r_[za - mu, zb - mu]])
    ua, ub = mp.matrix([float(x) for x in za - mu]), mp.matrix([float(x) for x in zb - mu])
    ldL = mp.log(mp.det(Lam))
    expo_t = -0.5 * (mp.log(mp.det(S2)) - 2 * ldL) - 0.5 * (w.T * mp.inverse(S2) * w)[0]
    Bi = mp.inverse(B)
    expo_0 = -(mp.log(mp.det(B)) - ldL) - 0.5 * ((ua.T * Bi * ua)[0] + (ub.T * Bi * ub)[0])
    return mp.exp(expo_t) - mp.exp(expo_0)


@pytest.mark.parametrize("t", [1, 3, 40, 200])
def test_stable_sigma_t_moment_vs_extended_precision(slow3, t):
    ch, k = slow3
    Gam = ch.cross_cov(t)
    Z = ch.mu + np.array([[0.4, -0.3, 0.9], [-1.2, 0.5, 0.1], [0.0, 0.7, -0.6]])
    Ms = gl.branched_pair_centered_moment_stable(k, Z, ch.mu, ch.C, Gam)
    Mo = gl.branched_pair_centered_moment(k, Z, ch.mu, ch.C, Gam)
    for a in range(3):
        for b in range(3):
            ref = float(_mp_moment(k, Z[a], Z[b], ch.mu, ch.C, Gam))
            assert abs(Ms[a, b] - ref) <= 1e-12 * abs(ref) + 1e-300
            if t <= 3:  # where the standard form is resolved, both agree
                assert abs(Mo[a, b] - ref) < 1e-12
    tr_s = gl.trace_sigma_t_stable(k, ch.C, Gam)
    mp.mp.dps = 60
    Lam = mp.diag([mp.mpf(float(s)) ** 2 for s in np.asarray(k.scales) * k.ell])
    E = Lam + 2 * mp.matrix(ch.C.tolist())
    E2 = E - 2 * mp.matrix(Gam.tolist())  # formed in extended precision (no fp64 cancellation)
    ref_tr = gl.mean_embedding_norm2(k, ch.C) * float(mp.expm1(0.5 * (mp.log(mp.det(E)) - mp.log(mp.det(E2)))))
    assert abs(tr_s - ref_tr) <= 1e-12 * abs(ref_tr)


def test_stable_delta_true_vs_standard_and_extended_precision(slow3):
    ch, k = slow3
    prob = type("P", (), {})()
    prob.chain, prob.kernel = ch, k
    P = torch.as_tensor(ch.mu + np.array([[0.4, -0.3, 0.9], [-1.2, 0.5, 0.1]]), dtype=F64)
    for scale in (0.3, 1e-9, 1e-40):
        b = scale * np.array([1.0, -0.5, 0.2])
        stable = r3.delta_shift_torch(prob, P, b).numpy()
        std = (r3.mu_pi_torch(prob, P, mean=ch.mu + b) - r3.mu_pi_torch(prob, P)).numpy()
        mp.mp.dps = 80
        Lam = np.diag((np.asarray(k.scales) * k.ell) ** 2)
        LS = mp.matrix((Lam + ch.C).tolist())
        c = mp.sqrt(mp.det(mp.matrix(Lam.tolist())) / mp.det(LS))
        Li = mp.inverse(LS)
        for i in range(2):
            u = mp.matrix([float(x) for x in P[i].numpy() - ch.mu])
            ub = u - mp.matrix([float(x) for x in b])
            ref = float(c * (mp.exp(-0.5 * (ub.T * Li * ub)[0]) - mp.exp(-0.5 * (u.T * Li * u)[0])))
            assert abs(stable[i] - ref) <= 1e-12 * abs(ref)
            if scale == 0.3:
                assert abs(std[i] - ref) < 1e-14


def test_tiny_pipeline_with_exact_transitions(slow3):
    """Exercise exact transitions, stable components and the checkpoint budget."""
    ch, k = slow3
    ck = [0, 1, 2, 4, 8]
    ex = ckc.CheckpointChain.from_chain(ch, ckc.grid_gaps(ck))
    a = 0.4 * np.array([1.0, -0.5, 0.2])
    cfg_like = {
        "checkpoints": ck, "budget_total": 0.05,
        "budget_components": {"replicas": 0.025, "target": 0.0125, "operator": 0.0125},
        "J_grid": [256], "N_grid": [64], "M": 64, "landmark_rtol": 1e-8, "m_grid": [2], "N_pi": 1024,
        "numerical_margin": 1e-8, "op_rtol": 1e-11, "head": 16, "n_dirs": 2,
        "landmark_closed_form": "stable",
    }  # fmt: skip
    prob = r3.build_linear_problem(ex, k, [r3.Setting("s", "slow", 0.8, a)], cfg_like, "test-chz")
    prob.direct_fp32, prob.stable_delta_true = False, True
    assert prob.budget.weight("replicas") == 1 / 5 and prob.budget.weight("operator") == 1 / 5
    out = r3.run_ensemble(prob, 0, "test-chz", torch.device("cpu"))
    assert ex.rows_transitioned == (256 + 4 * 64) * 4
    assert ex.rows_sweeps_represented == (256 + 4 * 64) * 8
    assert out["proj_true"].shape == (1, 5, 4, 1)
    assert (out["proj_true"][0, :, :, 0] <= prob.truth[0][:, None] * (1 + 1e-5)).all()


def test_config_is_the_specified_design():
    cfg = OmegaConf.load(ROOT / "configs/experiment/exp_c_common_horizon.yaml")
    assert [int(t) for t in cfg.checkpoints] == chz.GRID == [0] + [2**j for j in range(17)]
    assert len(chz.GRID) == 18 and chz.GRID[-1] == 65536
    assert list(cfg.J_grid) == [16384] and list(cfg.m_grid) == [8] and cfg.n_dirs == 8
    assert list(cfg.N_grid) == [2048] and cfg.N_pi == 65536 and cfg.landmarks.M == 2048
    assert dict(cfg.budget.components) == {"replicas": 0.025, "target": 0.0125, "operator": 0.0125}
    assert cfg.landmarks.rtol == 1e-8 and cfg.landmarks.numerical_margin == 1e-8
    assert cfg.operator.rtol == 1e-11 and cfg.operator.head == 256
    old_c = OmegaConf.load(ROOT / "configs/experiment/exp_r3_concrete.yaml")
    old_r = OmegaConf.load(ROOT / "configs/experiment/exp_r3_regression.yaml")
    assert cfg.datasets.concrete.ell_frozen == old_c.kernel.ell_frozen
    for ds in ("energy", "airfoil", "wine_red", "kin8nm"):
        assert cfg.datasets[ds].ell_frozen == old_r.datasets[ds].ell_frozen
        assert cfg.datasets[ds].sigma2_declared == old_r.datasets[ds].sigma2_declared
    ph = cfg.phases
    ranges = [(ph[p].rep_offset, ph[p].rep_offset + ph[p].n_ensembles) for p in ("smoke", "pilot", "main")]
    assert all(a[1] <= b[0] for a, b in zip(ranges[:-1], ranges[1:]))
    assert ph.smoke.n_ensembles == 2 and ph.pilot.n_ensembles == 10 and ph.main.n_ensembles == 500
    assert not str(cfg.seed_namespace).startswith("r3-")


def test_resume_requires_identical_config_and_setup(tmp_path, monkeypatch):
    """A resume with a changed design (e.g. J) or a different problem is refused before any chunk is run."""
    from mcmc.paperexp import runio

    monkeypatch.setenv("MCMC_PAPER_OUT", str(tmp_path))
    cfg = OmegaConf.load(ROOT / "configs/experiment/exp_c_common_horizon.yaml")
    cfg.run_id, cfg.phase = "energy_t", "smoke"
    text = chz._normalized_config_text(cfg)
    run = runio.RunDir.create(str(cfg.study), "smoke", "energy_t", text, requested_units=2, allow_dirty=True)
    cfg.resume = True
    chz.check_resume_config(run.path, cfg, "smoke")  # identical apart from the resume flag: accepted
    cfg.J_grid = [8192]
    with pytest.raises(ValueError, match="config"):
        chz.check_resume_config(run.path, cfg, "smoke")
    # setup arrays must reproduce the saved setup.npz bitwise
    ch = gl.gibbs_chain(np.eye(2) * 2.0, np.zeros(2))
    k = gl.AnisoRBF(np.ones(2), 1.0)
    cfg_like = {
        "checkpoints": [0, 1], "budget_total": 0.05,
        "budget_components": {"replicas": 0.025, "target": 0.0125, "operator": 0.0125},
        "J_grid": [64], "N_grid": [16], "M": 16, "landmark_rtol": 1e-8, "m_grid": [1], "N_pi": 64,
        "numerical_margin": 1e-8, "op_rtol": 1e-11, "head": 8, "n_dirs": 1,
    }  # fmt: skip
    prob = r3.build_linear_problem(ch, k, [r3.Setting("s", "slow", 0.8, np.array([0.3, 0.0]))], cfg_like, "t")
    meta = {"seed_namespace": "t", "rep_range": [0, 2], "ens_range": [0, 2], "blas_threads": {}}
    np.savez_compressed(run.path / "setup.npz", **r3.problem_arrays(prob))
    (run.path / "setup.json").write_text(__import__("json").dumps(meta))
    chz.check_resume_setup(run.path, prob, meta)
    with pytest.raises(ValueError, match="rep_range"):
        chz.check_resume_setup(run.path, prob, {**meta, "rep_range": [0, 4]})
    prob2 = r3.build_linear_problem(ch, k, [r3.Setting("s", "slow", 0.8, np.array([0.2, 0.0]))], cfg_like, "t")
    with pytest.raises(ValueError, match="setup.npz"):
        chz.check_resume_setup(run.path, prob2, meta)


def test_export_rejects_inconsistent_resume_records(tmp_path):
    """Input fingerprints control pooling; Git metadata does not."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("chz_export", ROOT / "scripts/paper_experiments/c_common_horizon_export.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ex.STUDY = tmp_path
    runs = []
    for index, commit in enumerate((None, "abc")):
        run = tmp_path / "main" / f"energy_main_e{index:03d}"
        run.mkdir(parents=True)
        (run / "RUN-STATUS.json").write_text(json.dumps({"status": "COMPLETED"}))
        (run / "PROVENANCE.json").write_text(json.dumps({"commit": commit, "dirty": True}))
        (run / "setup.json").write_text(json.dumps({"setup_cache": {"fingerprint": "fp"}}))
        resume = {"commit": "different", "dirty": True, "problem_fingerprint": "fp"}
        (run / "RESUMES.jsonl").write_text(json.dumps(resume) + "\n")
        runs.append(run)
    assert ex.runs_for("main", "energy", "main", None) == runs
    with pytest.raises(RuntimeError, match="expected commit"):
        ex.runs_for("main", "energy", "main", "abc")
    (runs[1] / "RESUMES.jsonl").write_text(json.dumps({"problem_fingerprint": "other"}) + "\n")
    with pytest.raises(RuntimeError, match="resume record"):
        ex.runs_for("main", "energy", "main", None)
    (runs[1] / "RESUMES.jsonl").unlink()
    (runs[1] / "setup.json").write_text(json.dumps({"setup_cache": {"fingerprint": "other"}}))
    with pytest.raises(RuntimeError, match="different problem fingerprints"):
        ex.runs_for("main", "energy", "main", None)
