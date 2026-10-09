"""Tests for German Credit data and hierarchical logistic posteriors."""

import os
import shutil

import numpy as np
import pytest
import torch

from mcmc.paperexp import german_credit as gc


@pytest.fixture(scope="module")
def data():
    return gc.load_german_credit()


@pytest.fixture(scope="module")
def jax_x64():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    return jax


# data


def test_pinned_hashes_and_shapes(data):
    assert gc.verify_raw_files() == gc.PINNED_SHA256
    assert data.X.shape == (gc.N_OBS, gc.N_PRED)
    assert data.y.shape == (gc.N_OBS,)
    assert set(np.unique(data.y)) == {0.0, 1.0}
    assert int(data.y.sum()) == 700
    np.testing.assert_array_equal(data.y, (data.raw_label == 1).astype(float))


def test_standardization_uses_population_std(data):
    np.testing.assert_allclose(data.X.mean(axis=0), 0.0, atol=1e-13)
    np.testing.assert_allclose(data.X.std(axis=0, ddof=0), 1.0, rtol=1e-13)
    np.testing.assert_allclose(data.col_std, data.raw_X.std(axis=0, ddof=0), rtol=0)


def test_numeric_encoding_matches_original(data):
    checks = gc.crosscheck_against_original(data)
    assert checks["label"] and checks["duration"] and checks["age"] and checks["checking_status"]
    assert checks["amount_div100_max_abs_err"] <= 0.5


def _copy_data_dir(tmp_path):
    dst = tmp_path / "gc"
    shutil.copytree(gc.DATA_DIR, dst)
    return dst


def test_loader_fails_on_tampered_file(tmp_path):
    dst = _copy_data_dir(tmp_path)
    path = dst / "german.data-numeric"
    raw = bytearray(path.read_bytes())
    raw[10] = ord("9") if raw[10] != ord("9") else ord("8")
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gc.load_german_credit(dst)


def test_loader_fails_on_missing_file(tmp_path):
    dst = _copy_data_dir(tmp_path)
    os.remove(dst / "german.doc")
    with pytest.raises(FileNotFoundError):
        gc.load_german_credit(dst)


def test_parser_rejects_malformed_rows(tmp_path):
    good = (gc.DATA_DIR / "german.data-numeric").read_text().splitlines()
    bad = tmp_path / "bad.txt"
    bad.write_text("\n".join(good[:-1] + [good[-1] + " 7"]) + "\n")
    with pytest.raises(ValueError, match="fields"):
        gc.parse_numeric_file(bad)
    bad.write_text("\n".join(good[:-1]) + "\n")
    with pytest.raises(ValueError, match="parsed shape"):
        gc.parse_numeric_file(bad)
    bad.write_text("\n".join(good[:-1] + [" ".join(good[-1].split()[:-1] + ["3"])]) + "\n")
    with pytest.raises(ValueError, match="labels"):
        gc.parse_numeric_file(bad)


# posteriors


@pytest.mark.parametrize("kind,d", [("hier", 26), ("horseshoe", 50)])
def test_dimensions_and_names(data, kind, d):
    post = gc.make_posterior(kind, data)
    assert post.d == d
    names = post.coordinate_names()
    assert len(names) == d and names[:2] == ["alpha", "log_sigma"]
    assert post.to_dict()["data_sha256"] == gc.PINNED_SHA256["german.data-numeric"]


def _points(d, n=24, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, d))


@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_log_prob_equals_numpyro_potential(data, jax_x64, kind):
    post = gc.make_posterior(kind, data)
    x = _points(post.d, seed=1)
    lp = post.log_prob(torch.from_numpy(x)).numpy()
    pe = np.array([gc.numpyro_potential_energy(kind, data, xi, centered=True) for xi in x])
    rel = np.abs(lp + pe) / np.abs(pe)
    assert rel.max() < 1e-10, rel.max()


@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_noncentered_potential_differs_by_jacobian(data, jax_x64, kind):
    post = gc.make_posterior(kind, data)
    x = _points(post.d, seed=2)
    p = gc.N_PRED
    lp = post.log_prob(torch.from_numpy(x)).numpy()
    pe = np.array([gc.numpyro_potential_energy(kind, data, xi, centered=False) for xi in x])
    log_jac = p * x[:, 1] if kind == "hier" else (x[:, 1:2] + x[:, 2 + p :]).sum(axis=1)
    rel = np.abs(lp + log_jac + pe) / np.abs(pe)
    assert rel.max() < 1e-10, rel.max()


@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_score_matches_autograd_and_jax(data, jax_x64, kind):
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import potential_energy

    post = gc.make_posterior(kind, data)
    x = _points(post.d, n=8, seed=3)
    g = post.score(torch.from_numpy(x))
    ga = post.score_autograd(torch.from_numpy(x))
    assert ((g - ga).abs().max() / ga.abs().max()).item() < 1e-12

    model = gc.numpyro_model(kind, centered=True)
    X, y = jnp.asarray(data.X), jnp.asarray(data.y)

    def neg_pe(flat):
        p = gc.N_PRED
        params = {"alpha": flat[0], "sigma": flat[1], "beta": flat[2 : 2 + p]}
        if kind == "horseshoe":
            params["tau"] = flat[2 + p :]
        return -potential_energy(model, (X, y), {}, params)

    grad = jax.grad(neg_pe)
    gj = np.stack([np.asarray(grad(jnp.asarray(xi))) for xi in x])
    rel = np.abs(g.numpy() - gj).max() / np.abs(gj).max()
    assert rel < 1e-10, rel


@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_score_matches_central_differences(data, kind):
    post = gc.make_posterior(kind, data)
    x = torch.from_numpy(_points(post.d, n=3, seed=4, scale=0.5))
    g = post.score(x)
    h = 1e-6
    fd = torch.empty_like(x)
    for j in range(post.d):
        e = torch.zeros(post.d, dtype=torch.float64)
        e[j] = h
        fd[:, j] = (post.log_prob(x + e) - post.log_prob(x - e)) / (2 * h)
    assert ((g - fd).abs().max() / g.abs().max()).item() < 1e-6


def test_log1pexp_is_exact_for_large_logits():
    z = torch.tensor([-800.0, -30.0, 0.0, 25.0, 40.0, 800.0], dtype=torch.float64)
    ref = torch.tensor([np.logaddexp(0.0, v) for v in z.tolist()], dtype=torch.float64)
    assert torch.allclose(gc._log1pexp(z), ref, rtol=1e-15, atol=0)


def test_declared_numpyro_round_trip(data):
    p = gc.N_PRED
    for kind in ("hier", "horseshoe"):
        post = gc.make_posterior(kind, data)
        x = _points(post.d, n=5, seed=5)
        for centered in (True, False):
            sites = [gc.numpyro_unconstrained_from_declared(kind, xi, centered) for xi in x]
            constrained = {
                "alpha": np.array([s["alpha"] for s in sites]),
                "sigma": np.exp([s["sigma"] for s in sites]),
            }
            if kind == "horseshoe":
                constrained["tau"] = np.exp(np.stack([s["tau"] for s in sites]))
            key = "beta" if centered else "z"
            constrained[key] = np.stack([s[key] for s in sites])
            back = gc.declared_from_numpyro(kind, constrained, centered)
            np.testing.assert_allclose(back, x, rtol=1e-13, atol=1e-13)
    assert p == 24


def test_input_validation(data):
    post = gc.make_posterior("hier", data)
    with pytest.raises(TypeError):
        post.log_prob(torch.zeros(2, 26, dtype=torch.float32))
    with pytest.raises(ValueError):
        post.log_prob(torch.zeros(2, 25, dtype=torch.float64))
    with pytest.raises(ValueError):
        post.score(torch.full((2, 26), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError):
        gc.make_posterior("probit", data)


@pytest.mark.skipif(
    os.environ.get("GC_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set GC_TEST_CUDA=1 on a CUDA machine",
)
@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_cuda_matches_cpu(data, kind):
    post = gc.make_posterior(kind, data)
    x = torch.from_numpy(_points(post.d, n=64, seed=6))
    lp_c, g_c = post.log_prob(x), post.score(x)
    xc = x.cuda()
    lp_g, g_g = post.log_prob(xc), post.score(xc)
    assert lp_g.device.type == "cuda" and g_g.dtype == torch.float64
    assert ((lp_g.cpu() - lp_c).abs() / lp_c.abs()).max().item() < 1e-12
    assert ((g_g.cpu() - g_c).abs().max() / g_c.abs().max()).item() < 1e-12


# references

from mcmc.paperexp import german_credit_reference as gref  # noqa: E402


def test_seed_for_is_deterministic_and_distinct():
    a = gref.seed_for("german_credit/x", "nuts-key", 0)
    assert a == gref.seed_for("german_credit/x", "nuts-key", 0)
    seeds = {gref.seed_for(c, s, r) for c in ("a", "b") for s in ("k", "i") for r in range(50)}
    assert len(seeds) == 200 and max(seeds) < 2**63


def test_parse_cores():
    assert gref.parse_cores("48-51,60") == [48, 49, 50, 51, 60]
    with pytest.raises(ValueError):
        gref.parse_cores("1-3,2")


def test_batch_gram_matches_brute_force():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(7 * 6, 3)) * 2.0
    G, min_d2 = gref.batch_gram(torch.from_numpy(U), batch_size=6, ell=1.3, block_batches=3)
    ref = gref.brute_force_batch_gram(U, 6, 1.3)
    np.testing.assert_allclose(G.numpy(), ref, rtol=1e-12, atol=0)
    assert min_d2 > -1e-9
    with pytest.raises(ValueError):
        gref.batch_gram(torch.from_numpy(U[:-1]), batch_size=6, ell=1.0, block_batches=2)


def _feature_gram(phi, L):
    nb = phi.shape[0] // L
    bsum = phi.reshape(nb, L, -1).sum(axis=1)
    return bsum @ bsum.T


def test_embedding_stats_against_explicit_features():
    rng = np.random.default_rng(1)
    R, nb, L, p = 5, 6, 4, 3
    phi = rng.normal(size=(R * nb * L, p)) + rng.normal(size=(R, 1, p)).repeat(nb * L, 1).reshape(
        -1, p
    )
    st = gref.embedding_stats(_feature_gram(phi, L), R, nb, L)
    runs = phi.reshape(R, nb * L, p)
    mu = runs.mean(axis=1)
    np.testing.assert_allclose(st["M"], mu @ mu.T, rtol=1e-12)
    mu_bar = mu.mean(axis=0)
    np.testing.assert_allclose(st["B"], ((mu - mu_bar) ** 2).sum() / (R - 1), rtol=1e-12)
    bm = runs.reshape(R, nb, L, p).mean(axis=2)
    v = ((bm - mu[:, None, :]) ** 2).sum(axis=(1, 2)) / (nb - 1) / nb
    np.testing.assert_allclose(st["v"], v, rtol=1e-12)
    d = runs[:, : nb * L // 2].mean(axis=1) - runs[:, nb * L // 2 :].mean(axis=1)
    np.testing.assert_allclose(st["D_half"], d @ d.T, rtol=1e-12, atol=1e-14)
    pair = ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(st["pairwise_sq"], pair, rtol=1e-10, atol=1e-13)


def test_exact_split_and_signflip_tests_against_brute_force():
    import itertools

    rng = np.random.default_rng(2)
    R, p = 6, 4
    mu = rng.normal(size=(R, p))
    M = mu @ mu.T
    group = np.array([1, 1, 1, 0, 0, 0], bool)
    res = gref.exact_split_test(M, group)
    stats = []
    for comb in itertools.combinations(range(R), R // 2):
        a = np.zeros(R, bool)
        a[list(comb)] = True
        stats.append(((mu[a].mean(0) - mu[~a].mean(0)) ** 2).sum())
    obs = ((mu[group].mean(0) - mu[~group].mean(0)) ** 2).sum()
    assert res["stat"] == pytest.approx(obs, rel=1e-12)
    assert res["p_value"] == pytest.approx(np.mean(np.array(stats) >= obs - 1e-12), abs=1e-12)

    d = rng.normal(size=(5, p))
    res = gref.exact_signflip_test(d @ d.T)
    flips = [
        ((np.array(s)[:, None] * d).mean(0) ** 2).sum()
        for s in itertools.product((1, -1), repeat=5)
    ]
    obs = (d.mean(0) ** 2).sum()
    assert res["stat"] == pytest.approx(obs, rel=1e-12)
    assert res["p_value"] == pytest.approx(np.mean(np.array(flips) >= obs - 1e-12), abs=1e-12)


def test_mc_cross_test_statistic_and_pvalue_range():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=(8, 3))
    mu[4:] += 5.0  # second group clearly shifted
    res = gref.mc_cross_test(mu @ mu.T, 4, 2000, seed=0)
    assert res["stat"] == pytest.approx(((mu[:4].mean(0) - mu[4:].mean(0)) ** 2).sum(), rel=1e-12)
    # observed split is the most extreme of C(8,4)=70 splits (with its complement): p ~ 2/70
    assert 0 < res["p_value"] < 0.1


def test_calibrate_kernel():
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 3)) * np.array([1.0, 2.0, 0.5])
    k = gref.calibrate_kernel(x, 100)
    s = x.std(axis=0, ddof=1)
    np.testing.assert_allclose(k["scales"], s, rtol=1e-14)
    idx = np.linspace(0, 999, 100).round().astype(int)
    assert k["ell"] == pytest.approx(np.median(pdist(x[idx] / s)), rel=1e-14)


def test_conventional_diagnostics_on_iid_draws():
    rng = np.random.default_rng(5)
    R, N, d = 4, 2000, 3
    ref = {
        "draws": rng.normal(size=(R, N, d)),
        "diverging": np.zeros((R, N), bool),
        "warm_diverging": np.zeros((R, 10), bool),
        "num_steps": np.full((R, N), 7),
        "warm_num_steps": np.full((R, 10), 7),
        "energy": rng.normal(size=(R, N)),
        "summaries": [{"step_size": 0.5}] * R,
    }
    c = gref.conventional_diagnostics(ref, ["a", "b", "c"], max_tree_depth=10)
    assert c["rhat_max"] < 1.01
    assert 0.8 * R * N < c["ess_bulk_min"] < 1.2 * R * N
    assert c["divergent_total"] == 0 and c["treedepth_hits_total"] == 0
    assert 1.5 < c["ebfmi_min"] < 2.5  # iid energies: E-BFMI ~ 2


@pytest.mark.parametrize("kind,param", [("hier", "centered"), ("horseshoe", "noncentered")])
def test_run_unit_smoke(tmp_path, jax_x64, kind, param):
    spec = gref.RefSpec(
        preset="t",
        model=kind,
        parameterization=param,
        target_accept_prob=0.9,
        max_tree_depth=6,
        dense_mass=False,
        n_runs=1,
        n_warmup=30,
        n_draws=40,
        thin=1,
        init_radius=2.0,
        role="reference",
        version="test",
        phase="smoke",
    )
    s = gref.run_unit(spec, 0, str(tmp_path))
    assert s["logprob_check_rel"] < gref.LOGPROB_CHECK_RTOL
    assert gref.unit_complete(tmp_path, 0, spec)
    with np.load(tmp_path / "unit_000.npz") as z:
        assert z["draws"].shape == (40, spec.d) and z["warm_num_steps"].shape == (30,)
        x0 = np.random.default_rng(gref.seed_for(spec.config_id, "nuts-init", 0)).uniform(
            -2, 2, spec.d
        )
        np.testing.assert_array_equal(z["x0"], x0)


@pytest.mark.parametrize("kind", ["hier", "horseshoe"])
def test_fused_log_prob_and_score(data, kind):
    post = gc.make_posterior(kind, data)
    x = torch.from_numpy(_points(post.d, n=16, seed=7))
    lp, g = post.log_prob_and_score(x)
    assert torch.equal(lp, post.log_prob(x)) and torch.equal(g, post.score(x))
    assert post.target_id == post.name


def test_calibrate_kernel_mad_rule():
    rng = np.random.default_rng(8)
    x = rng.standard_t(3, size=(4001, 3)) * np.array([1.0, 2.0, 0.5])
    k = gref.calibrate_kernel(x, 200, scale_rule="mad")
    med = np.median(x, axis=0)
    np.testing.assert_allclose(k["scales"], 1.4826 * np.median(np.abs(x - med), axis=0), rtol=1e-14)
    assert gref.kernel_key("hier", "mad") == "hier_mad" and gref.kernel_key("hier", "sd") == "hier"
    with pytest.raises(ValueError):
        gref.calibrate_kernel(x, 200, scale_rule="iqr")
