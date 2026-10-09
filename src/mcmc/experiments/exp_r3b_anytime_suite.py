"""Synthetic anytime bounds with estimated operators and target references.

mode=reference generates independent truth intervals and a Nystrom operator.
mode=ensembles runs repetitions using reference_run_id; mode=witness compares
the estimates with an analytic Gaussian AR model. See README.md for commands.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from mcmc.paperexp import bounds as bd
from mcmc.paperexp import r3_suite as r3
from mcmc.paperexp.runio import RunDir, output_root, utc_now
from mcmc.paperexp.targets import load_frozen_suite

log = logging.getLogger(__name__)
F64 = torch.float64


def kernel_tag(cfg: DictConfig) -> str:
    return f"{cfg.kernel.kind}-{cfg.kernel.bw}"


def namespace(cfg: DictConfig, config_id: str) -> str:
    """Separate random streams by model, kernel and initial law."""
    ns = f"r3b-{cfg.protocol_version}:{config_id}:{kernel_tag(cfg)}:eps{float(cfg.eps):g}"
    kind = str(cfg.init.kind)
    if kind != "known_radius":
        ns += f":init-{kind}-x{float(cfg.init.scale):g}"
    return ns


def _budget(cfg: DictConfig) -> bd.BudgetAllocation:
    return bd.BudgetAllocation(float(cfg.budget.total), dict(cfg.budget.components))


def _schedule(cfg: DictConfig, t_max: int) -> list[int]:
    s = cfg.schedule
    return r3.r3_schedule(int(t_max), int(s.factor), int(s.floor), int(s.cap))


def build_suite_problem(cfg: DictConfig, device) -> tuple[r3.SuiteProblem, dict]:
    suite = load_frozen_suite()
    cid = str(cfg.config_id)
    c = suite.config(cid)
    tid = c["target_id"]
    target = suite.target(tid)
    kernel = suite.kernel(tid, kind=str(cfg.kernel.kind), bw=str(cfg.kernel.bw))
    embedding = suite.embedding(tid, bw=str(cfg.kernel.bw)) if cfg.kernel.kind == "rbf" else None
    sampler = suite.sampler(cid)
    t_grid = _schedule(cfg, int(c["t_max"]))
    problem = r3.SuiteProblem(
        namespace=namespace(cfg, cid),
        config_id=cid,
        target=target,
        kernel=kernel,
        kernel_tag=kernel_tag(cfg),
        step=r3.MHStep(sampler),
        eps=float(cfg.eps),
        t_grid=t_grid,
        J_grid=[int(j) for j in cfg.J_grid],
        N_grid=[int(n) for n in cfg.N_grid],
        m_grid=[int(m) for m in cfg.m_grid],
        n_dirs=int(cfg.n_dirs),
        N_pi=int(cfg.N_pi),
        budget=_budget(cfg),
        embedding=embedding,
        op_rtol=float(cfg.operator.rtol),
        head=int(cfg.operator.head),
        kernel_chunk=int(cfg.kernel_chunk),
        op_method=str(cfg.operator.method),
        op_fallback=None if cfg.operator.fallback is None else str(cfg.operator.fallback),
        n_refined=int(cfg.operator.n_refined),
        fused_rowsum=bool(cfg.kernel_sums.fused) and str(cfg.kernel.kind) == "rbf",
        init_kind=str(cfg.init.kind),
        init_scale=float(cfg.init.scale),
        init_scales=suite.scales(tid),
    )
    meta = {
        "config_id": cid,
        "target": target.to_dict(),
        "kernel": kernel.to_dict(),
        "sampler": sampler.to_dict(),
        "frozen_suite": str(suite.path),
        "frozen_suite_sha256": hashlib.sha256(suite.path.read_bytes()).hexdigest(),
        "frozen_t_max": int(c["t_max"]),
        "t_grid": t_grid,
        "embedding": embedding.to_dict() if embedding is not None else None,
        "namespace": problem.namespace,
        "effective": {
            "N_grid": problem.N_grid,
            "op_method": problem.op_method,
            "op_fallback": problem.op_fallback,
            "n_refined": problem.n_refined,
            "fused_rowsum": problem.fused_rowsum,
            "batch_ensembles": int(cfg.batch_ensembles),
            "applied_phase_overrides": cfg.get("applied_phase_overrides"),
            "applied_config_overrides": cfg.get("applied_config_overrides"),
            "op_rtol": problem.op_rtol,
            "init": {
                "kind": problem.init_kind,
                "scale": problem.init_scale,
                "A3_holds": problem.init_kind == "known_radius",
                "assumed_R": problem.r_init,
            },
        },
    }
    return problem, meta


def build_witness_problem(
    cfg: DictConfig, device
) -> tuple[r3.SuiteProblem, r3.ARGaussWitness, dict]:
    w = cfg.witness
    suite = load_frozen_suite()
    tid = str(w.target_id)
    target = suite.target(tid)
    if target.family != "gauss":
        raise ValueError("the AR witness needs the Gaussian target")
    kernel = suite.kernel(tid, kind="rbf", bw=str(cfg.kernel.bw))
    embedding = suite.embedding(tid, bw=str(cfg.kernel.bw))
    C, Q = target.covariance(), target.rotation
    rho = np.full(target.d, float(w.rho_fast))
    rho[-1] = float(w.rho_slow)  # q_min = Q[:, -1] is the event direction
    A, L = r3.ar_witness_chain(C, rho, Q)
    analytic = r3.ARGaussWitness(C, A, (kernel.scales * kernel.ell) ** 2, Q[:, -1], float(cfg.eps))
    if abs(analytic.mu_norm2 - embedding.norm2()) > 1e-13:
        raise ValueError("witness ||mu_pi||^2 disagrees with the frozen embedding")
    # t_max rule of the suite applied to the analytic MMD: first power of two with MMD < threshold
    tm = 1
    while math.sqrt(analytic.mmd2(tm)) >= float(w.t_max_threshold):
        tm *= 2
        if tm > 8192:
            raise RuntimeError("witness MMD does not fall below the threshold by 8192")
    t_grid = _schedule(cfg, tm)
    cid = f"witness-ar-{tid}"
    problem = r3.SuiteProblem(
        namespace=namespace(cfg, cid),
        config_id=cid,
        target=target,
        kernel=kernel,
        kernel_tag=kernel_tag(cfg),
        step=r3.LinearGaussianStep(A, L),
        eps=float(cfg.eps),
        t_grid=t_grid,
        J_grid=[int(j) for j in cfg.J_grid],
        N_grid=[int(n) for n in cfg.N_grid],
        m_grid=[int(m) for m in cfg.m_grid],
        n_dirs=int(cfg.n_dirs),
        N_pi=int(cfg.N_pi),
        budget=_budget(cfg),
        embedding=embedding,
        op_rtol=float(cfg.operator.rtol),
        head=int(cfg.operator.head),
        kernel_chunk=int(cfg.kernel_chunk),
        op_method=str(cfg.operator.method),
        op_fallback=None if cfg.operator.fallback is None else str(cfg.operator.fallback),
        n_refined=int(cfg.operator.n_refined),
        fused_rowsum=bool(cfg.kernel_sums.fused) and str(cfg.kernel.kind) == "rbf",
    )
    meta = {
        "config_id": cid,
        "target": target.to_dict(),
        "kernel": kernel.to_dict(),
        "chain": {"kind": "AR(1) exact", "rho": rho.tolist(), "A": A.tolist(), "L": L.tolist()},
        "t_max_rule": {"threshold": float(w.t_max_threshold), "t_max": tm},
        "t_grid": t_grid,
        "analytic_mmd": [math.sqrt(analytic.mmd2(t)) for t in t_grid],
        "analytic_trace": [analytic.trace(t) for t in t_grid],
        "namespace": problem.namespace,
    }
    return problem, analytic, meta


def apply_phase_overrides(cfg: DictConfig) -> DictConfig:
    """Apply phase overrides, then per-model overrides, before saving the config."""
    key = "reference" if str(cfg.mode) == "reference" else str(cfg.phase)
    po = cfg.get("phase_overrides") or {}
    merged = OmegaConf.merge(cfg, po[key]) if key in po else cfg
    co = cfg.get("config_overrides") or {}
    cid = str(cfg.config_id)
    if cid in co:
        merged = OmegaConf.merge(merged, co[cid])
    with open_dict(merged):  # Hydra configs are in struct mode
        merged.applied_phase_overrides = key if key in po else None
        merged.applied_config_overrides = cid if cid in co else None
    return merged


def _save_npz(path: pathlib.Path, arrays: dict) -> None:
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_chunk(path: pathlib.Path, results: list, reps: list) -> None:
    keys = [k for k in results[0] if k != "timing"]
    arrs = {k: np.stack([np.asarray(r[k]) for r in results]) for k in keys}
    arrs["reps"] = np.array(reps)
    tk = sorted(results[0]["timing"])
    arrs["timing"] = np.array([[r["timing"].get(x, 0.0) for x in tk] for r in results])
    arrs["timing_keys"] = np.array(tk)
    _save_npz(path, arrs)


def load_chunks(run_path: pathlib.Path) -> dict:
    chunks = sorted(run_path.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {run_path}")
    parts = [dict(np.load(c)) for c in chunks]
    tk = [str(x) for x in parts[0]["timing_keys"]]
    for p in parts:
        if [str(x) for x in p["timing_keys"]] != tk:
            raise ValueError("timing keys differ across chunks")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "timing_keys"}
    data["timing_keys"] = parts[0]["timing_keys"]
    if len(np.unique(data["reps"])) != len(data["reps"]):
        raise ValueError("duplicate replicate ids across chunks")
    return data


def reference_path(ref_id: str, phase: str) -> pathlib.Path:
    """Locate a completed reference run; smoke runs may use a smoke reference."""
    p = output_root() / "r3b_suite" / ("smoke" if phase == "smoke" else "reference") / ref_id
    if not (p / "reference.npz").exists():
        raise FileNotFoundError(f"no reference.npz in {p}")
    status = json.loads((p / "RUN-STATUS.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"reference run {p} is {status['status']}, not COMPLETED")
    return p


def _device(cfg: DictConfig) -> torch.device:
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def _config_text(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))


def _reference_kwargs(cfg: DictConfig, problem: r3.SuiteProblem) -> dict:
    rc = cfg.reference
    return dict(
        J_ref=int(rc.J_ref),
        N_refop=int(rc.N_refop),
        n_landmarks=int(rc.n_landmarks),
        beta_ref=float(rc.beta_ref),
        quad_margin=math.sqrt(3.0 * float(rc.quad_abs_err)),
        N_ref_target=None if problem.embedding is not None else int(rc.N_ref_target),
        landmark_rtol=float(rc.landmark_rtol),
        feat_chunk=int(rc.feat_chunk),
        log=log.info,
    )


def run_reference_mode(cfg: DictConfig) -> RunDir:
    device = _device(cfg)
    phase = "smoke" if cfg.phase == "smoke" else "reference"
    run = RunDir.create(
        str(cfg.study), phase, str(cfg.run_id), _config_text(cfg), 1, allow_dirty=phase == "smoke"
    )
    try:
        problem, meta = build_suite_problem(cfg, device)
        ref = r3.run_reference(problem, device=device, **_reference_kwargs(cfg, problem))
        _save_npz(run.path / "reference.npz", ref)
        meta.update(
            {
                "budget_reference": {
                    "beta_ref": float(cfg.reference.beta_ref),
                    "replicas": float(ref["beta_ref_replicas"]),
                    "target": float(ref["beta_ref_target"]),
                },
                "quad_margin": float(ref["quad_margin"]),
                "wall": float(ref["wall"]),
                "timing": dict(zip([str(x) for x in ref["timing_keys"]], ref["timing"].tolist())),
                "peak_gpu_bytes": int(ref["peak_gpu_bytes"]),
                "finished_utc": utc_now(),
            }
        )
        (run.path / "reference.json").write_text(json.dumps(meta, indent=2, default=str))
        run.update(completed_units=1)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return run


def _run_ensembles(cfg, run, problem, meta, ref, ref_sha, device, n_ens, analytic=None):
    """Run replicate IDs [rep_offset, rep_offset + n_ens) in resumable chunks."""
    phase = str(cfg.phase)
    off = int(cfg.rep_offset)
    if off < 0:
        raise ValueError(f"rep_offset must be >= 0, got {off}")
    if not (run.path / "setup.npz").exists():
        _save_npz(run.path / "setup.npz", problem.setup_arrays())
        meta.update(
            {
                "budget": problem.budget.to_dict(),
                "budget_check": problem.budget_check(),
                "reference_sha256": ref_sha,
                "n_ensembles": n_ens,
                "rep_range": [off, off + n_ens],
                "stage_prefix": f"{phase}-ens-",
            }
        )
        (run.path / "setup.json").write_text(json.dumps(meta, indent=2, default=str))
    chunk, batch = int(cfg.chunk_size), int(cfg.batch_ensembles)
    if batch < 1 or chunk % batch:
        raise ValueError(
            f"chunk_size {chunk} must be a positive multiple of batch_ensembles {batch}"
        )
    for c0 in range(0, n_ens, chunk):
        c1 = min(n_ens, c0 + chunk)
        path = run.path / f"chunk_{off + c0:05d}.npz"
        if path.exists():
            run.update(completed_units=c1)
            continue
        results = []
        for b0 in range(c0, c1, batch):
            reps = list(range(off + b0, off + min(c1, b0 + batch)))
            t0 = time.perf_counter()
            results += r3.run_ensembles(problem, reps, phase, device, analytic=analytic)
            log.info(
                f"{problem.config_id} {phase} reps {reps[0]}..{reps[-1]}: batch wall "
                f"{time.perf_counter() - t0:.1f}s ({(time.perf_counter() - t0) / len(reps):.1f}"
                f" s/ensemble), peak GPU {int(results[-1]['peak_gpu_bytes']) / 2**20:.0f} MiB"
            )
        save_chunk(path, results, list(range(off + c0, off + c1)))
        run.update(completed_units=c1)


def run_ensembles_mode(cfg: DictConfig) -> RunDir:
    device = _device(cfg)
    phase = str(cfg.phase)
    n_ens = cfg.phases[phase].n_ensembles
    if n_ens is None:
        raise ValueError(f"phase {phase} has no frozen ensemble count")
    n_ens = int(n_ens)
    if cfg.reference_run_id is None:
        raise ValueError("ensembles need reference_run_id")
    ref_dir = reference_path(str(cfg.reference_run_id), phase)
    ref_sha = hashlib.sha256((ref_dir / "reference.npz").read_bytes()).hexdigest()
    if cfg.resume:
        run = RunDir.resume(str(cfg.study), phase, str(cfg.run_id))
    else:
        run = RunDir.create(
            str(cfg.study),
            phase,
            str(cfg.run_id),
            _config_text(cfg),
            n_ens,
            allow_dirty=phase in ("smoke", "pilot"),
            extra_provenance={"reference_dir": str(ref_dir), "reference_sha256": ref_sha},
        )
    try:
        problem, meta = build_suite_problem(cfg, device)
        ref_meta = json.loads((ref_dir / "reference.json").read_text())
        if ref_meta["namespace"] != problem.namespace or ref_meta["t_grid"] != problem.t_grid:
            raise ValueError(
                f"reference {ref_dir} was built for {ref_meta['namespace']} / {ref_meta['t_grid']}"
            )
        ref = dict(np.load(ref_dir / "reference.npz"))
        if ref["coefs_t"].shape[2] != problem.n_dirs or list(ref["t_grid"]) != problem.t_grid:
            raise ValueError("reference plug-in spans do not match n_dirs / schedule")
        problem.refop = r3.load_refop(ref, device)
        meta["reference_dir"] = str(ref_dir)
        _run_ensembles(cfg, run, problem, meta, ref, ref_sha, device, n_ens)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return run


def run_witness_mode(cfg: DictConfig) -> RunDir:
    device = _device(cfg)
    n_ens = int(cfg.witness.n_ensembles)
    phase = "smoke" if cfg.phase == "smoke" else "reference"
    if cfg.resume:
        run = RunDir.resume(str(cfg.study), phase, str(cfg.run_id))
    else:
        run = RunDir.create(
            str(cfg.study),
            phase,
            str(cfg.run_id),
            _config_text(cfg),
            n_ens,
            allow_dirty=phase == "smoke",
        )
    try:
        problem, analytic, meta = build_witness_problem(cfg, device)
        log.info(f"witness schedule {problem.t_grid}; analytic MMD {meta['analytic_mmd']}")
        ref_file = run.path / "reference.npz"
        if not ref_file.exists():
            ref = r3.run_reference(problem, device=device, **_reference_kwargs(cfg, problem))
            _save_npz(ref_file, ref)
        ref = dict(np.load(ref_file))
        ref_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()
        problem.refop = r3.load_refop(ref, device)
        cfg_phase = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_phase.phase = "witness"
        _run_ensembles(cfg_phase, run, problem, meta, ref, ref_sha, device, n_ens, analytic)
        data = load_chunks(run.path)
        setup = dict(np.load(run.path / "setup.npz"))
        mmd_true = np.array(meta["analytic_mmd"])
        checks = r3.witness_checks(setup, data, ref, mmd_true, problem.budget_check())
        checks["summary"] = r3.summarize(setup, data, ref, ref, quad_margin_known=0.0)
        (run.path / "witness.json").write_text(json.dumps(checks, indent=2, default=str))
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return run


@hydra.main(
    config_path="../../../configs/experiment",
    config_name="exp_r3b_anytime_suite",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.run_id is None:
        raise ValueError("run_id is required")
    cfg = apply_phase_overrides(cfg)
    t0 = time.perf_counter()
    mode = str(cfg.mode)
    if mode == "reference":
        run = run_reference_mode(cfg)
    elif mode == "ensembles":
        run = run_ensembles_mode(cfg)
    elif mode == "witness":
        run = run_witness_mode(cfg)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    log.info(f"{utc_now()} {mode} finished in {time.perf_counter() - t0:.1f}s: {run.path}")


if __name__ == "__main__":
    main()
