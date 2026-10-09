"""Tests for regression datasets, posterior parameters, and saved configurations."""

import hashlib
import math
import pathlib
import shutil

import numpy as np
import pytest

from mcmc.paperexp import regression_data as rd

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", rd.NAMES)
def test_load_and_model(name):
    X, y, prov = rd.load_dataset(name)
    assert X.shape == rd.SHAPES[name] and np.isfinite(X).all() and np.isfinite(y).all()
    m = rd.frozen_model(name)
    d = m.mu.shape[0]
    assert d == rd.SHAPES[name][1] + 1
    assert np.abs(m.A @ m.C @ m.A.T + m.Q - m.C).max() < 1e-12 * max(1.0, np.abs(m.C).max())
    assert np.abs(m.P @ m.C - np.eye(d)).max() < 1e-8
    for s in m.settings:
        assert abs(math.expm1(s["shift"] @ m.P @ s["shift"]) - s["r_init"] ** 2) < 1e-10
    for key in ("slow", "fast"):
        v = m.directions[key]["vec"]
        assert abs(np.linalg.norm(v) - 1) < 1e-12
    assert m.checkpoints[0] == 0 and m.seed_namespace


def test_registry_reproduces_concrete_directions():
    X, y, _ = rd.load_dataset("concrete")
    post = rd.RegressionPosterior("concrete", X, y)
    a, b = post.directions(), rd.registry_directions(post)
    for key in ("slow", "fast"):
        assert np.array_equal(a[key]["vec"], b[key]["vec"])


def test_declared_rules():
    assert (
        rd.frozen_model("wine_red").directions["fast"]["rule"].endswith("(tie -> larger modulus)")
    )
    assert (
        rd.frozen_model("kin8nm").directions["slow"]["rule"] == "real part of complex eigenvector"
    )


def test_loader_fails_on_corruption(tmp_path):
    (tmp_path / "airfoil").mkdir()
    src = rd.RAW / "airfoil" / "airfoil_self_noise.zip"
    b = bytearray(src.read_bytes())
    b[100] ^= 1
    (tmp_path / "airfoil" / "airfoil_self_noise.zip").write_bytes(bytes(b))
    with pytest.raises(ValueError, match="SHA256"):
        rd.load_dataset("airfoil", tmp_path)
    shutil.rmtree(tmp_path / "airfoil")
    with pytest.raises(FileNotFoundError):
        rd.load_dataset("airfoil", tmp_path)


def test_frozen_concrete_resolved_config_unchanged():
    """The resolved main configuration matches the one used for the saved Concrete results."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment"), version_base=None):
        cfg = compose(config_name="exp_r3_concrete", overrides=["phase=main", "run_id=main01"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert "wandb" not in resolved
    # The saved checksum includes this unused logging setting.
    resolved["wandb"] = {"mode": "disabled"}
    text = OmegaConf.to_yaml(OmegaConf.create(resolved))
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == "cbf43c704f6f5a9f1cc7d326209665aab045190c70655db0864ef85e0ec37a13"
    )
