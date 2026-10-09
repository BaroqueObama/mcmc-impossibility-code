"""Tests for merging ensemble runs and classifying reference estimates."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_r3b", ROOT / "scripts/paper_experiments/analyze_r3b.py"
)
analyze_r3b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_r3b)


def _run(tmp, name, reps, b_t=(0.1, 0.05), status="COMPLETED", ref="abc"):
    d = tmp / name
    d.mkdir()
    (d / "RUN-STATUS.json").write_text(json.dumps({"status": status}))
    np.savez(d / "setup.npz", b_t=np.array(b_t), t_grid=np.arange(3))
    meta = {
        "reference_sha256": ref,
        "effective": {"N_grid": [2048]},
        "namespace": "ns",
        "stage_prefix": "main-ens-",
        "t_grid": [0, 1, 2],
    }
    (d / "setup.json").write_text(json.dumps(meta))
    np.savez(
        d / f"chunk_{reps[0]:05d}.npz",
        reps=np.array(reps),
        x=np.array(reps, float) * 2,
        timing_keys=np.array(["a"]),
    )
    return d


def test_union_of_main_and_extension(tmp_path):
    a = _run(tmp_path, "main", list(range(0, 4)))
    b = _run(tmp_path, "ext", list(range(4, 8)))
    setup, data, union = analyze_r3b.load_union([a, b], allow_partial=False)
    assert union["n"] == 8 and union["n_per_run"] == [4, 4]
    assert np.array_equal(np.sort(data["reps"]), np.arange(8))
    assert np.array_equal(data["x"], np.arange(8) * 2.0)


def test_union_rejects_overlap_and_mismatch(tmp_path):
    a = _run(tmp_path, "main", list(range(0, 4)))
    with pytest.raises(ValueError):
        analyze_r3b.load_union([a, _run(tmp_path, "ovl", [3, 4, 5, 6])], allow_partial=False)
    with pytest.raises(ValueError):
        analyze_r3b.load_union([a, _run(tmp_path, "bt", [4, 5], b_t=(0.2, 0.05))], False)
    with pytest.raises(ValueError):
        analyze_r3b.load_union([a, _run(tmp_path, "ref", [4, 5], ref="zzz")], False)
    with pytest.raises(RuntimeError):
        analyze_r3b.load_union([a, _run(tmp_path, "run", [4, 5], status="RUNNING")], False)


def _run2(tmp, name, reps, rep_range, status="COMPLETED", completed=None, fallback=None):
    d = _run(tmp, name, reps, status=status)
    me = json.loads((d / "setup.json").read_text())
    me["rep_range"] = rep_range
    if fallback is not None:
        me["effective"]["op_fallback"] = fallback
    (d / "setup.json").write_text(json.dumps(me))
    st = {"status": status, "completed_units": len(reps) if completed is None else completed}
    (d / "RUN-STATUS.json").write_text(json.dumps(st))
    if fallback is not None:  # newer runs carry op_fallback
        z = dict(np.load(d / f"chunk_{reps[0]:05d}.npz"))
        z["op_fallback"] = np.zeros((len(reps), 3, 1), dtype=np.int64)
        z["op_fallback"][-1, 1, 0] = 1
        np.savez(d / f"chunk_{reps[0]:05d}.npz", **z)
    return d


def test_union_accepts_failed_run_with_exact_continuation(tmp_path):
    a = _run2(tmp_path, "m", [0, 1, 2], [0, 6], status="FAILED")
    b = _run2(tmp_path, "mb", [3, 4, 5], [3, 6], fallback="chol-fp64")
    setup, data, union = analyze_r3b.load_union([a, b], allow_partial=False)
    assert union["n"] == 6 and data["op_fallback"].shape == (6, 3, 1)
    assert union["fallback_events"] == [{"rep": 5, "t": 1, "N_index": 0}]
    c = _run2(tmp_path, "x", [0, 1, 2], [0, 6], status="FAILED")
    d = _run2(tmp_path, "xb", [4, 5], [4, 6], fallback="chol-fp64")  # gap at rep 3
    with pytest.raises((RuntimeError, ValueError)):
        analyze_r3b.load_union([c, d], allow_partial=False)


def test_embedding_status_quadrature_vs_closed_form():
    ref = {"quad_margin": np.array(np.sqrt(3e-10))}
    q = analyze_r3b.embedding_status(
        {"embedding": {"method": "quadrature", "rule": "trapezoid_y", "order": 1024}}, ref
    )
    assert q["q_known"] == pytest.approx(np.sqrt(3e-10)) and "conditional" in q["conditional"]
    c = analyze_r3b.embedding_status({"embedding": {"method": "closed_form"}}, ref)
    assert c["q_known"] == 0.0 and c["conditional"] is None and "analytic" in c["kind"]
    n = analyze_r3b.embedding_status({"embedding": None}, ref)
    assert n["q_known"] == 0.0 and n["conditional"] is None
