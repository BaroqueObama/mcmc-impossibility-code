import hashlib
import json
import subprocess

import pytest

from mcmc.paperexp import runio


def test_run_from_source_archive(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(runio, "__file__", str(archive / "src/mcmc/paperexp/runio.py"))
    monkeypatch.setenv("MCMC_PAPER_OUT", str(tmp_path))

    run = runio.RunDir.create("example", "main", "archive", "seed: 42\n", requested_units=1)
    provenance = json.loads((run.path / "PROVENANCE.json").read_text())
    assert provenance["commit"] is None
    assert provenance["dirty"] is None
    assert run.status()["commit"] is None
    run.update(completed_units=1)
    run.finish("COMPLETED")
    assert run.status()["status"] == "COMPLETED"


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, ["git", "rev-parse"])]
)
def test_unavailable_git_is_optional(tmp_path, monkeypatch, error):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(runio, "__file__", str(tmp_path / "src/mcmc/paperexp/runio.py"))

    def fail(*args):
        raise error

    monkeypatch.setattr(runio, "_git", fail)
    assert runio.git_state() == {"commit": None, "dirty": None, "dirty_files": []}


def test_dirty_checkout_records_state_and_preserves_run_safeguards(tmp_path, monkeypatch):
    state = {"commit": "abc123", "dirty": True, "dirty_files": [" M config.yaml"]}
    monkeypatch.setattr(runio, "git_state", lambda: state.copy())
    monkeypatch.setenv("MCMC_PAPER_OUT", str(tmp_path))
    config = "seed: 7\n"
    run = runio.RunDir.create("example", "main", "edited", config, requested_units=2)

    provenance = json.loads((run.path / "PROVENANCE.json").read_text())
    assert all(provenance[k] == v for k, v in state.items())
    assert provenance["config_sha256"] == hashlib.sha256(config.encode()).hexdigest()
    assert (run.path / "config.yaml").read_text() == config

    with pytest.raises(RuntimeError, match="0/2"):
        run.finish("COMPLETED")
    run.update(completed_units=1)
    run.finish("INCOMPLETE")
    resumed = runio.RunDir.resume("example", "main", "edited")
    assert resumed.status()["completed_units"] == 1
    assert resumed.status()["status"] == "RUNNING"
    assert resumed.status()["config_sha256"] == provenance["config_sha256"]
    resumed.update(completed_units=2)
    resumed.finish("COMPLETED")

    with pytest.raises(FileExistsError, match="COMPLETED"):
        runio.RunDir.create("example", "main", "edited", config, requested_units=2)
    with pytest.raises(FileExistsError, match="COMPLETED"):
        runio.RunDir.resume("example", "main", "edited")
