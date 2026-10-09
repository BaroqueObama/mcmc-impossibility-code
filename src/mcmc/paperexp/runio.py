"""Experiment directories under $MCMC_PAPER_OUT/{study}/{phase}/{run_id}.

Runs save their config, provenance and completion status. Completed run IDs
cannot be reused.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys

STATUSES = ("RUNNING", "COMPLETED", "FAILED", "INCOMPLETE")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def output_root() -> pathlib.Path:
    root = os.environ.get("MCMC_PAPER_OUT")
    if not root:
        raise RuntimeError("Set MCMC_PAPER_OUT to an existing output directory")
    path = pathlib.Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"MCMC_PAPER_OUT={root} does not exist or is not a directory")
    return path


def resolve_output_path(value: str | pathlib.Path) -> pathlib.Path:
    """Expand MCMC_PAPER_OUT in saved paths; leave other paths unchanged."""
    text = str(value)
    for prefix in ("$MCMC_PAPER_OUT", "${MCMC_PAPER_OUT}"):
        if text == prefix or text.startswith(prefix + "/"):
            default = pathlib.Path(__file__).resolve().parents[3] / "output/paper_experiments"
            root = pathlib.Path(os.environ.get("MCMC_PAPER_OUT", default))
            suffix = text[len(prefix):].lstrip("/")
            return root / suffix
    return pathlib.Path(value)


def _git(*args: str) -> str:
    repo = pathlib.Path(__file__).resolve().parents[3]
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_state() -> dict:
    """Record Git state when available, including for a checkout with local edits."""
    unknown = {"commit": None, "dirty": None, "dirty_files": []}
    repo = pathlib.Path(__file__).resolve().parents[3]
    if not (repo / ".git").exists():
        return unknown
    try:
        commit = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return unknown
    dirty_files = [ln for ln in status.splitlines() if ln]
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files[:50]}


def config_sha256(config_text: str) -> str:
    return hashlib.sha256(config_text.encode()).hexdigest()


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


class RunDir:
    """Experiment directory with progress records and resumable status."""

    def __init__(self, path: pathlib.Path):
        self.path = path

    @classmethod
    def create(
        cls,
        study: str,
        phase: str,
        run_id: str,
        config_text: str,
        requested_units: int,
        allow_dirty: bool = False,  # retained for existing callers
        extra_provenance: dict | None = None,
    ) -> "RunDir":
        if phase not in ("smoke", "pilot", "screen", "main", "stress", "reference"):
            raise ValueError(f"unknown phase {phase!r}")
        state = git_state()
        path = output_root() / study / phase / run_id
        if (path / "RUN-STATUS.json").exists():
            prev = json.loads((path / "RUN-STATUS.json").read_text())
            if prev.get("status") == "COMPLETED":
                raise FileExistsError(f"{path} is COMPLETED; rerun under a new run id")
            raise FileExistsError(
                f"{path} already exists with status {prev.get('status')}; use RunDir.resume or a new id"
            )
        path.mkdir(parents=True, exist_ok=False)
        (path / "config.yaml").write_text(config_text)
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        prov = {
            **state,
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "host": socket.gethostname(),
            "cuda_visible_devices": devices,
            "config_sha256": config_sha256(config_text),
            "started_utc": utc_now(),
            "pid": os.getpid(),
        }
        if extra_provenance:
            prov.update(extra_provenance)
        _write_json_atomic(path / "PROVENANCE.json", prov)
        run = cls(path)
        _write_json_atomic(
            path / "RUN-STATUS.json",
            {
                "status": "RUNNING",
                "run_id": run_id,
                "study": study,
                "phase": phase,
                "commit": state["commit"],
                "config_sha256": prov["config_sha256"],
                "host": prov["host"],
                "cuda_visible_devices": devices,
                "started_utc": prov["started_utc"],
                "updated_utc": prov["started_utc"],
                "finished_utc": None,
                "requested_units": int(requested_units),
                "completed_units": 0,
                "failed_units": [],
                "note": "",
            },
        )
        return run

    @classmethod
    def resume(cls, study: str, phase: str, run_id: str) -> "RunDir":
        path = output_root() / study / phase / run_id
        status = json.loads((path / "RUN-STATUS.json").read_text())
        if status["status"] == "COMPLETED":
            raise FileExistsError(f"{path} is COMPLETED; nothing to resume")
        run = cls(path)
        run.update(status="RUNNING", note=f"resumed {utc_now()}")
        return run

    def status(self) -> dict:
        return json.loads((self.path / "RUN-STATUS.json").read_text())

    def update(self, **fields) -> None:
        status = self.status()
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"bad status {fields['status']!r}")
        status.update(fields)
        status["updated_utc"] = utc_now()
        _write_json_atomic(self.path / "RUN-STATUS.json", status)

    def finish(self, status: str, note: str = "") -> None:
        if status not in ("COMPLETED", "FAILED", "INCOMPLETE"):
            raise ValueError(f"finish() needs a terminal status, got {status!r}")
        current = self.status()
        if status == "COMPLETED" and current["completed_units"] != current["requested_units"]:
            raise RuntimeError(
                f"cannot mark COMPLETED with {current['completed_units']}/"
                f"{current['requested_units']} units; use INCOMPLETE"
            )
        self.update(status=status, finished_utc=utc_now(), note=note or current.get("note", ""))
