"""Local input checks for the logistic budget and replication scripts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / 'scripts/paper_experiments' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture(params=['budget', 'replication'])
def saved_design(request, tmp_path, monkeypatch):
    if request.param == 'budget':
        module = _load('c_realdata_budget_followup')
        monkeypatch.setattr(module, 'OUT', tmp_path)
        design_path = tmp_path / 'stage2_design.json'
        manifest_path = tmp_path / 'manifest.json'
        manifest = {}
    else:
        module = _load('c_realdata_replication')
        monkeypatch.setattr(module, 'OUT_R', tmp_path)
        design_path = tmp_path / 'design_replication.json'
        manifest_path = tmp_path / 'manifest_replication.json'
        seeds = {}
        for model in module.MODELS:
            for sampler in module.SAMPLERS:
                unit = {law: [module.bundle_seed(model, sampler, law, b)
                              for b in range(module.N_BUNDLES)] for law in module.LAWS}
                unit['bench'] = module.bench_seed(model, sampler)
                seeds[f'{model}__{sampler}'] = unit
        manifest = {'seeds': seeds, 'sources': {}}
    design = module.design()
    manifest['design_sha256'] = _digest(design)
    design_path.write_text(json.dumps(design))
    manifest_path.write_text(json.dumps(manifest))
    return module, design_path, manifest_path, request.param


def test_saved_design_works_without_git(saved_design):
    module, design_path, manifest_path, _ = saved_design
    design = json.loads(design_path.read_text())
    design['protocol'] = 'A different description with the same numerical design.'
    manifest = json.loads(manifest_path.read_text())
    manifest['design_sha256'] = _digest(design)
    design_path.write_text(json.dumps(design))
    manifest_path.write_text(json.dumps(manifest))
    assert module.verify_frozen()


def test_saved_design_rejects_changed_contents(saved_design):
    module, design_path, _, _ = saved_design
    design = json.loads(design_path.read_text())
    design['protocol'] = 'changed without updating the manifest'
    design_path.write_text(json.dumps(design))
    with pytest.raises(RuntimeError, match='design hash mismatch'):
        module.verify_frozen()


def test_saved_design_rejects_changed_parameters(saved_design):
    module, design_path, manifest_path, kind = saved_design
    design = json.loads(design_path.read_text())
    if kind == 'budget':
        design['replica_split']['evaluation_replicas'][0] += 1
    else:
        design['bundle_size'] += 1
    manifest = json.loads(manifest_path.read_text())
    manifest['design_sha256'] = _digest(design)
    design_path.write_text(json.dumps(design))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='parameters differ'):
        module.verify_frozen()


@pytest.mark.parametrize("saved_design", ["replication"], indirect=True)
def test_replication_rejects_changed_source(saved_design, tmp_path):
    module, _, manifest_path, kind = saved_design
    source = tmp_path / 'raw.npz'
    source.write_bytes(b'original input')
    manifest = json.loads(manifest_path.read_text())
    manifest['sources'] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    source.write_bytes(b'changed input')
    with pytest.raises(RuntimeError, match='sha256'):
        module.verify_frozen()


@pytest.mark.parametrize("saved_design", ["replication"], indirect=True)
def test_replication_rejects_changed_seeds(saved_design):
    module, _, manifest_path, kind = saved_design
    manifest = json.loads(manifest_path.read_text())
    unit = next(iter(manifest['seeds'].values()))
    unit['bench'] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='seeds differ'):
        module.verify_frozen()


def test_saved_output_paths_follow_environment(tmp_path, monkeypatch):
    from mcmc.paperexp.runio import resolve_output_path

    monkeypatch.setenv('MCMC_PAPER_OUT', str(tmp_path))
    assert resolve_output_path('$MCMC_PAPER_OUT/study/run') == tmp_path / 'study/run'
    assert resolve_output_path('${MCMC_PAPER_OUT}/study/run') == tmp_path / 'study/run'
    assert resolve_output_path('paper/results.csv') == Path('paper/results.csv')
    assert resolve_output_path('/elsewhere/raw.npz') == Path('/elsewhere/raw.npz')
    monkeypatch.delenv('MCMC_PAPER_OUT')
    assert resolve_output_path('$MCMC_PAPER_OUT/study/run') == ROOT / 'output/paper_experiments/study/run'


def test_source_hash_checks_expand_saved_output_paths(tmp_path, monkeypatch):
    module = _load('c_realdata_budget_followup')
    monkeypatch.setenv('MCMC_PAPER_OUT', str(tmp_path))
    path = tmp_path / 'raw.npz'
    path.write_bytes(b'raw input')
    expected = {'$MCMC_PAPER_OUT/raw.npz': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert module.verify_hashes(expected) == 1
    path.write_bytes(b'changed input')
    with pytest.raises(RuntimeError, match='sha256'):
        module.verify_hashes(expected)
