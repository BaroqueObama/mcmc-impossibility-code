"""Tests for logistic datasets, preprocessing, and posterior densities."""

import shutil

import numpy as np
import pytest
import torch

from mcmc.paperexp import german_credit as gc
from mcmc.paperexp import logistic_data as ld

DATASETS = ["pima", "australian", "ionosphere", "sonar"]
EXPECTED = {
    "pima": (768, 8, 268),
    "australian": (690, 14, 307),
    "ionosphere": (351, 33, 225),
    "sonar": (208, 60, 111),
}


@pytest.fixture(scope="module")
def jax_x64():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    return jax


@pytest.mark.parametrize("name", DATASETS)
def test_loader_shapes_hashes_standardization(name):
    d = ld.load_dataset(name)
    n, p, npos = EXPECTED[name]
    assert d.X.shape == (n, p) and d.y.shape == (n,) and int(d.y.sum()) == npos
    assert ld.verify_files(name) == ld.SPECS[name].files
    np.testing.assert_allclose(d.X.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(d.X.std(axis=0, ddof=0), 1.0, rtol=1e-12)
    assert gc.model_dim(f"hier_{name}") == p + 2


def test_ionosphere_drops_only_the_constant_column():
    d = ld.load_dataset("ionosphere")
    assert d.dropped == (1,) and np.ptp(d.raw_X[:, 1]) == 0.0
    np.testing.assert_allclose(
        d.X[:, 0], (d.raw_X[:, 0] - d.raw_X[:, 0].mean()) / d.raw_X[:, 0].std()
    )


@pytest.mark.parametrize("name", DATASETS)
def test_loader_fails_on_tampered_file(tmp_path, name):
    shutil.copytree(ld.DATA_ROOT / name, tmp_path / name)
    f = tmp_path / name / ld.SPECS[name].primary_file
    raw = bytearray(f.read_bytes())
    raw[-3] = ord("7") if raw[-3] != ord("7") else ord("6")
    f.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA256"):
        ld.load_dataset(name, tmp_path)


def test_loader_rejects_unknown_and_missing(tmp_path):
    with pytest.raises(ValueError):
        ld.load_dataset("iris")
    with pytest.raises(FileNotFoundError):
        ld.load_dataset("sonar", tmp_path)


@pytest.mark.parametrize("name", DATASETS)
def test_hier_log_prob_matches_numpyro_and_score_matches_autograd(jax_x64, name):
    kind = f"hier_{name}"
    post = gc.make_posterior(kind)
    assert post.name == f"{name}_hier_logistic" and post.d == gc.model_dim(kind)
    rng = np.random.default_rng(11)
    x = rng.uniform(-2, 2, size=(8, post.d))
    lp = post.log_prob(torch.from_numpy(x)).numpy()
    pe = np.array([gc.numpyro_potential_energy(kind, post.data, xi, centered=True) for xi in x])
    assert (np.abs(lp + pe) / np.abs(pe)).max() < 1e-10
    pe_nc = np.array([gc.numpyro_potential_energy(kind, post.data, xi, centered=False) for xi in x])
    log_jac = (post.d - 2) * x[:, 1]
    assert (np.abs(lp + log_jac + pe_nc) / np.abs(pe_nc)).max() < 1e-10
    g = post.score(torch.from_numpy(x))
    ga = post.score_autograd(torch.from_numpy(x))
    assert ((g - ga).abs().max() / ga.abs().max()).item() < 1e-12


def test_german_credit_names_and_dims_unchanged():
    h, s = gc.make_posterior("hier"), gc.make_posterior("horseshoe")
    assert (h.name, h.d, s.name, s.d) == (
        "german_credit_hier_logistic",
        26,
        "german_credit_horseshoe_logistic",
        50,
    )
    assert gc.split_kind("hier_sonar") == ("hier", "sonar")
    with pytest.raises(ValueError):
        gc.split_kind("horseshoe_sonar")
