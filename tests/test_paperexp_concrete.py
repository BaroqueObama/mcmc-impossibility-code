"""Tests for Concrete data and the Gaussian regression posterior."""

import math
import shutil

import numpy as np
import pytest

from mcmc.paperexp import concrete as cc


@pytest.fixture(scope="module")
def data():
    return cc.load_concrete()


def test_loader_shapes_and_provenance(data):
    X, y, prov = data
    assert X.shape == (1030, 8) and y.shape == (1030,)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert prov["xls_vs_csv_max_abs_gap"] <= 0.05 + 1e-9
    assert abs(y[0] - 79.98611076) < 1e-12  # full-precision first response (CSV prints 79.99)
    assert X[:, 7].min() == 1 and X[:, 7].max() == 365  # Age in days


def test_loader_fails_on_corruption_and_missing(tmp_path):
    for name in cc.SHA256:
        shutil.copy(cc.DATA_DIR / name, tmp_path / name)
    p = tmp_path / "concrete_data_uci.csv"
    b = bytearray(p.read_bytes())
    b[200] = ord("9") if b[200] != ord("9") else ord("8")
    p.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="SHA256"):
        cc.load_concrete(tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError):
        cc.load_concrete(tmp_path)


def test_posterior_and_gibbs(data):
    X, y, _ = data
    post = cc.ConcretePosterior(X, y)
    assert post.sigma2 == 0.39 and abs(post.ols_resid_var - 0.3875) < 1e-3
    assert np.abs(post.P @ post.C - np.eye(9)).max() < 1e-10
    assert post.chain.stationarity_error() < 1e-15
    assert post.chain.reversibility_error() > 1e-5
    assert abs(post.mu[0]) < 1e-12  # standardized response: intercept posterior mean 0
    dirs = post.directions()
    assert (
        abs(dirs["slow"]["eigval"] - 0.94030) < 1e-4
        and abs(dirs["fast"]["eigval"] - 0.16604) < 1e-4
    )
    for key in ("slow", "fast"):
        v = dirs[key]["vec"]
        Av = post.chain.A @ v
        assert np.abs(Av - dirs[key]["eigval"] * v).max() < 1e-12
        for R in (0.8, 0.2):
            a = post.shift(v, R)
            assert abs(math.expm1(a @ post.P @ a) - R**2) < 1e-12
