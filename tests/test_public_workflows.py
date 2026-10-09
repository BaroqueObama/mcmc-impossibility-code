import csv
import hashlib
import io
import json

import pytest
from omegaconf import OmegaConf

from mcmc.experiments import exp_b_logistic as logistic
from mcmc.experiments import exp_b_real_alignment as alignment
from mcmc.experiments import exp_c_common_horizon as horizon
from mcmc.paperexp import runio


def _csv_bytes(rows, fields):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def _export_fixture(path, combined):
    row = dict.fromkeys(logistic.bl.ANGLE_COLUMNS, "")
    row.update(dataset="pima", sampler="mala", m="1", t="2", fit="0", pilot_length="4096")
    row["cutoff_reason"] = "DK_INCONSISTENT_AT_HALF_BUDGET"
    protocol = _csv_bytes([row], logistic.bl.ANGLE_COLUMNS)
    protocol_sha = hashlib.sha256(protocol).hexdigest()
    if combined:
        relabeled = dict(row, budget="protocol_32768")
        relabeled["cutoff_reason"] = (
            "HALF_BUDGET_UNRESOLVED (a half-budget eigengap <= 2 e_t; not an angular disagreement)"
        )
        raw = _csv_bytes(
            [relabeled, dict(row, budget="expanded_65536")],
            [*logistic.bl.ANGLE_COLUMNS, "budget"],
        )
    else:
        raw = protocol
    source = {
        "outputs_sha256": {"angles_logistic.csv": hashlib.sha256(raw).hexdigest()},
        "supersedes": {"angles_logistic.csv_sha256": protocol_sha},
    }
    (path / "angles_logistic.csv").write_bytes(raw)
    (path / "SOURCE_logistic.json").write_text(json.dumps(source))
    return row, source, protocol_sha


@pytest.mark.parametrize("combined", [False, True])
def test_protocol_export_loads_without_git_history(tmp_path, combined):
    expected, source, checksum = _export_fixture(tmp_path, combined)
    rows, loaded_source, loaded_checksum = logistic._load_protocol_export(tmp_path, "protocol_32768")
    assert rows == [expected]
    assert loaded_source == source
    assert loaded_checksum == checksum


def test_protocol_export_checks_file_hash(tmp_path):
    _export_fixture(tmp_path, True)
    path = tmp_path / "angles_logistic.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="does not match SOURCE"):
        logistic._load_protocol_export(tmp_path, "protocol_32768")


def test_protocol_export_checks_original_subset_hash(tmp_path):
    _, source, _ = _export_fixture(tmp_path, True)
    source["supersedes"]["angles_logistic.csv_sha256"] = "0" * 64
    (tmp_path / "SOURCE_logistic.json").write_text(json.dumps(source))
    with pytest.raises(RuntimeError, match="original export checksum"):
        logistic._load_protocol_export(tmp_path, "protocol_32768")


def test_protocol_export_rejects_missing_budget(tmp_path):
    _export_fixture(tmp_path, True)
    with pytest.raises(ValueError, match="no rows for"):
        logistic._load_protocol_export(tmp_path, "protocol_1")


def test_bundled_protocol_matches_original_checksum():
    path = logistic.REPO / "paper/figures/data/r2_real_alignment"
    rows, source, checksum = logistic._load_protocol_export(path, "protocol_32768")
    assert rows
    assert checksum == source["supersedes"]["angles_logistic.csv_sha256"]


def test_alignment_selection_loads_outside_git_and_checks_expected_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment, "REPO", tmp_path)
    raw = b'{"datasets": {"concrete": {"selected_m": [1]}}}\n'
    (tmp_path / "selection.json").write_bytes(raw)
    cfg = OmegaConf.create({"selection_file": "selection.json"})
    selection, checksum = alignment._load_selection(cfg)
    assert selection["datasets"]["concrete"]["selected_m"] == [1]
    assert checksum == hashlib.sha256(raw).hexdigest()
    cfg.selection_sha256 = checksum
    assert alignment._load_selection(cfg)[1] == checksum
    (tmp_path / "selection.json").write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="selection checksum"):
        alignment._load_selection(cfg)


def test_common_horizon_resume_checks_config_without_git(tmp_path, monkeypatch):
    def no_git():
        raise AssertionError("resuming from an archive must not require Git")

    monkeypatch.setattr(runio, "git_state", no_git)
    cfg = OmegaConf.create({"resume": True, "dataset": "concrete", "checkpoints": [0, 1, 2]})
    stored = horizon._normalized_config_text(cfg)
    (tmp_path / "config.yaml").write_text(stored)
    (tmp_path / "PROVENANCE.json").write_text(
        json.dumps({"config_sha256": runio.config_sha256(stored), "commit": "private-history"})
    )
    horizon.check_resume_config(tmp_path, cfg, "main")
    cfg.checkpoints = [0, 1, 4]
    with pytest.raises(ValueError, match="resolved config differs"):
        horizon.check_resume_config(tmp_path, cfg, "main")


def test_publish_combined_export_is_repeatable_without_old_commits(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    unit = "hier_pima__mala"
    row = dict.fromkeys(logistic.bl.ANGLE_COLUMNS, "")
    row.update(
        dataset="pima", sampler="mala", m="1", t="2", fit="0", pilot_length="4096",
        angle_deg="2.0", learned_angle_upper="3.0", estimate_status="EMPIRICAL_SENSITIVITY_ONLY",
    )
    reference = {
        "unit": unit, "m": "1", "t": "2", "cutoff": "", "applicable": "True", "stable": "True",
        "half1_dk_deg": "1.0", "half2_dk_deg": "1.0",
        "half1_consistent": "True", "half2_consistent": "True",
    }
    raw = _csv_bytes([row], logistic.bl.ANGLE_COLUMNS)
    checksum = hashlib.sha256(raw).hexdigest()
    (output / "angles_logistic.csv").write_bytes(raw)
    (output / "SOURCE_logistic.json").write_text(json.dumps({
        "outputs_sha256": {"angles_logistic.csv": checksum},
        "covariance_source": "independent branched batches", "generator_commit": "unavailable",
    }))
    (output / "lag2t_check_logistic_v2.csv").write_text("unit,m,t\n")
    (output / "windows.csv").write_text("unit,m,t\n")
    for tag in ("fresh_v1", "expanded_x1"):
        run = tmp_path / "runs" / "b_logistic" / "main" / f"{unit}_{tag}"
        run.mkdir(parents=True)
        (run / "RUN-STATUS.json").write_text('{"status": "COMPLETED"}')
        (run / "rows.csv").write_bytes(raw)
        (run / "reference.csv").write_bytes(_csv_bytes([reference], reference.keys()))
        for k in range(4):
            (run / f"batch_{k}.json").write_text(json.dumps({"seed": 100 + k, "wall_s": 1.0}))
    cfg = OmegaConf.create({
        "phase": "main", "study": "b_logistic", "calibration": {"out_dir": str(output)},
        "evaluation": {"tag": "v1", "pairs": {unit: {"m": [1]}}, "not_evaluated": {},
                       "none_eligible": [], "sigma": {"n_batches": 4, "n_blocks": 8192}},
        "expanded": {"tag": "x1", "design_sha256": "design"},
    })
    monkeypatch.setattr(logistic, "output_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(logistic, "check_frozen_pairs", lambda _: {"window_csv": {"path": "windows.csv"}})
    monkeypatch.setattr(logistic, "expanded_design", lambda _: {"units": {unit: {"N_total": 65536}}})
    logistic.run_publish2(cfg)
    first = (output / "angles_logistic.csv").read_bytes()
    logistic.run_publish2(cfg)
    assert (output / "angles_logistic.csv").read_bytes() == first
    source = json.loads((output / "SOURCE_logistic.json").read_text())
    assert source["supersedes"]["angles_logistic.csv_sha256"] == checksum
    assert source["covariance_source"]["protocol_32768"] == "independent branched batches"
    assert "generator_commit" not in source
    assert "pool_quality" not in source
    assert source["runs"][f"{unit}/protocol_32768"]["batches"] == [
        {"seed": 100 + k} for k in range(4)
    ]


@pytest.mark.parametrize("missing_input", [None, "sigma.npz"])
def test_logistic_selection_hashes_raw_inputs_without_cached_export(tmp_path, monkeypatch, missing_input):
    from types import SimpleNamespace

    unit = "hier_pima__mala"
    runs = tmp_path / "runs"
    calibration = runs / "b_logistic" / "pilot" / "calibrate_v1"
    calibration.mkdir(parents=True)
    (calibration / "RUN-STATUS.json").write_text('{"status": "COMPLETED"}')
    selection = {"status": "NONE_ELIGIBLE", "lag": None, "selected_m": [], "eligible": []}
    result = {"candidates": [], "selection": selection}
    (calibration / f"calib_{unit}.json").write_text(json.dumps({
        "results": {"primary": result, "sensitivity": result},
        "windows": {}, "meta": {"feature_rank": 8}, "fits": {},
    }))
    (calibration / f"calib_{unit}.npz").write_bytes(b"calibration arrays")
    inputs = tmp_path / "sampling"
    inputs.mkdir()
    expected = {}
    for name in ("pilotrun_stationary.npz", "setup.npz", "sigma.npz"):
        if name == missing_input:
            continue
        path = inputs / name
        raw = f"saved sampling data: {name}".encode()
        path.write_bytes(raw)
        expected[str(path)] = hashlib.sha256(raw).hexdigest()
    output = tmp_path / "selection"
    output.mkdir()
    cfg = OmegaConf.create({
        "study": "b_logistic", "tag": "v1", "units": [unit],
        "calibration": {"out_dir": str(output)}, "fresh": {},
    })
    monkeypatch.setattr(logistic, "output_root", lambda: runs)
    monkeypatch.setattr(logistic, "Unit", lambda *_: SimpleNamespace(
        v1=inputs, model="hier_pima", dataset="pima", sampler="mala", appendix=False,
    ))
    if missing_input:
        with pytest.raises(FileNotFoundError, match=missing_input):
            logistic.run_select(cfg)
        assert not (output / "selection_logistic_v1.json").exists()
    else:
        saved = json.loads(logistic.run_select(cfg).read_text())
        assert {key: saved["calibration_hashes"][key] for key in expected} == expected
        assert saved["units"][unit]["status"] == "NONE_ELIGIBLE"
        assert len(saved["calibration_hashes"]) == 5
