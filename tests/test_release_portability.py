"""Tests for relocated inputs and source archives."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mcmc.experiments import exp_b_real_alignment as alignment
from mcmc.experiments import exp_r1b_constructions_suite as residual
from mcmc.paperexp import german_credit_reference as references

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_relocated_setup_preserves_frozen_checksum(tmp_path, monkeypatch):
    setup = tmp_path / 'setup.npz'
    setup.write_bytes(b'a frozen setup fixture')
    entry = {'setup_run': '/unavailable-original-run',
             'setup_npz_sha256': hashlib.sha256(setup.read_bytes()).hexdigest()}
    cfg = OmegaConf.create({'setup_run': str(tmp_path)})
    monkeypatch.setattr(alignment, 'load_setup', lambda path: ('loaded', path))
    assert alignment._load_selected_setup(cfg, entry) == ('loaded', tmp_path)
    setup.write_bytes(b'a changed setup fixture')
    with pytest.raises(ValueError, match='differs from the frozen selection'):
        alignment._load_selected_setup(cfg, entry)


def test_frozen_suite_uses_supplied_file_without_git(monkeypatch):
    cfg = OmegaConf.load(ROOT / 'configs/experiment/exp_r1b_constructions_suite.yaml')
    assert cfg.frozen_suite_git_ref is None
    def no_git(*args, **kwargs):
        pytest.fail('Reading the frozen suite must not require private Git history')
    monkeypatch.setattr(residual.subprocess, 'run', no_git)
    doc, source = residual._frozen_doc(cfg)
    assert 'gauss-d10-c25__rwm' in doc['configs']
    assert source['source_sha256'] == hashlib.sha256(
        (ROOT / cfg.frozen_suite).read_bytes()).hexdigest()


def test_reference_auto_core_respects_restricted_affinity(monkeypatch):
    monkeypatch.setattr(references.os, 'sched_getaffinity', lambda pid: {3, 7, 19})
    assert references.parse_cores('auto') == [3]
    assert references.parse_cores('3,7,19') == [3, 7, 19]


def test_figure_collector_prefers_supplied_data_without_git(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('release_figure_data', ROOT / 'scripts/figures/main_figures_data.py')
    module = importlib.util.module_from_spec(spec)
    # dataclasses requires the defining module to be registered while loading it.
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / 'saved.csv').write_bytes(b'x,y\n1,2\n')
    def no_git(*args, **kwargs):
        pytest.fail('A supplied numerical export must not require Git history')
    import subprocess
    monkeypatch.setattr(subprocess, 'run', no_git)
    assert module.data_file('saved.csv') == b'x,y\n1,2\n'


def _paper_figures(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'portable_paper_figures', ROOT / 'scripts/figures/paper_main_figures.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('missing', ['both', 'r1a.csv', 'r1b.csv', 'force'])
def test_residual_plot_rebuilds_missing_or_forced_caches(tmp_path, monkeypatch, missing):
    import sys
    module = _paper_figures(monkeypatch)
    module.mfd.collect(tmp_path)
    expected = {name: (tmp_path / name).read_bytes() for name in ('r1a.csv', 'r1b.csv')}
    for name in expected:
        if missing in ('both', name):
            (tmp_path / name).unlink()
        elif missing == 'force':
            (tmp_path / name).write_text('stale cache\n')
    monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['paper_main_figures.py', '--only', 'r1']
                        + (['--collect'] if missing == 'force' else []))
    plotted = []
    def draw(*args):
        assert {name: (tmp_path / name).read_bytes() for name in expected} == expected
        plotted.append(True)
    monkeypatch.setattr(module, 'fig_r1', draw)
    module.main()
    assert plotted == [True]


def test_missing_residual_source_fails_before_writing_caches(tmp_path, monkeypatch):
    module = _paper_figures(monkeypatch)
    read = module.mfd.data_file
    monkeypatch.setattr(module.mfd, 'data_file',
                        lambda name: None if name.endswith('r1_suite/projected.csv') else read(name))
    with pytest.raises(FileNotFoundError, match='projected residuals'):
        module.mfd.collect(tmp_path)
    assert not list(tmp_path.glob('*.csv'))
