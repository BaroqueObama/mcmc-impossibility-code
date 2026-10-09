"""Anytime bounds at 18 common checkpoints, from 0 to 65,536 Gibbs sweeps.

Exact Gaussian transitions preserve the joint law at inspection times. Replica
and operator failure budgets are split over checkpoints; the target-reference
budget is spent once. Log-scale MMD avoids underflow at late checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mcmc.experiments import exp_r3_concrete as r3
from mcmc.experiments import exp_r3_regression as r3reg
from mcmc.paperexp import checkpoint_chain as ckc
from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import regression_data as rd
from mcmc.paperexp.runio import RunDir, utc_now

log = logging.getLogger(__name__)
REPO = pathlib.Path(__file__).resolve().parents[3]
GRID = [0] + [2**j for j in range(17)]  # 0, 1, 2, 4, ..., 65536 (18 checkpoints)
PROBLEM_KEYS = (
    "seed_namespace", "dataset", "model", "kernel", "setting", "checkpoints", "m_grid", "n_dirs",
    "J_grid", "N_grid", "N_pi", "landmarks", "operator", "budget", "transition",
)  # fmt: skip
BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def namespace(cfg: DictConfig) -> str:
    return f"{cfg.seed_namespace}-{cfg.dataset}"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(cfg: DictConfig) -> dict:
    """The original summary's slow_R0.8 shift (verbatim), r_init, meta and file hash."""
    dcfg = cfg.datasets[str(cfg.dataset)]
    path = REPO / str(dcfg.source_summary)
    summ = json.loads(path.read_text())
    i = int(cfg.setting.source_index)
    r_init = summ["setup"]["r_init"][i]
    if r_init != 0.8 or float(cfg.setting.r_init) != 0.8:
        raise ValueError(f"{cfg.dataset}: source r_init[{i}] = {r_init}, config {cfg.setting.r_init}")
    return {
        "path": str(dcfg.source_summary),
        "sha256": sha256_file(path),
        "index": i,
        "r_init": r_init,
        "shift": np.array(summ["setup"]["shifts"][i], dtype=np.float64),
        "old_checkpoints": summ["setup"]["checkpoints"],
        "old_truth": summ["setup"]["truth"][i],
        "meta": summ["meta"],
    }


def transition_accounting(checkpoints, rows_replicas: int, rows_branches: int, exact: bool) -> dict:
    """Represented Gibbs sweeps vs transitions actually simulated, cumulative per checkpoint."""
    ck = [int(t) for t in checkpoints]
    sim = list(range(len(ck))) if exact else ck  # exact: one transition per gap
    return {
        "checkpoints": ck,
        "per_chain_sweeps_represented": ck,
        "per_chain_transitions_simulated": sim,
        "rows_replicas": rows_replicas,
        "rows_operator_branches": rows_branches,
        "ensemble_sweeps_represented": (rows_replicas + rows_branches) * ck[-1],
        "ensemble_transitions_simulated": (rows_replicas + rows_branches) * sim[-1],
    }


def build_problem(cfg: DictConfig) -> tuple[r3.Problem, dict]:
    name = str(cfg.dataset)
    dcfg = cfg.datasets[name]
    checkpoints = [int(t) for t in cfg.checkpoints]
    if checkpoints != GRID:
        raise ValueError(f"checkpoints {checkpoints} != the fixed common grid {GRID}")
    if str(cfg.transition) not in ("exact", "stepwise"):
        raise ValueError(f"transition {cfg.transition!r} not in (exact, stepwise)")
    X, y, prov = rd.load_dataset(name)
    post = rd.RegressionPosterior(name, X, y, prior_var=float(cfg.model.prior_var))
    if post.sigma2 != float(dcfg.sigma2_declared):
        raise ValueError(f"{name}: declared sigma2 {dcfg.sigma2_declared} != OLS rounding {post.sigma2}")
    src = load_source(cfg)
    sm = src["meta"]
    diff_mu = float(np.abs(post.mu - np.array(sm["mu"])).max())
    diff_sd = float(np.abs(post.coord_sd - np.array(sm["coord_sd"])).max())
    if diff_mu > 1e-12 or diff_sd > 1e-12:
        raise ValueError(f"{name}: posterior differs from the source study (mu {diff_mu}, sd {diff_sd})")
    # Recompute the bandwidth using the original random stream.
    ell_median = r3reg.median_bandwidth(
        post.chain, post.coord_sd, str(dcfg.source_namespace), int(cfg.kernel.bandwidth_draws)
    )
    ell_frozen = float(dcfg.ell_frozen)
    if abs(ell_median - ell_frozen) > 1e-12 or float(sm["ell"]) != ell_frozen:
        raise ValueError(f"{name}: bandwidth {ell_median!r} / source {sm['ell']!r} != frozen {ell_frozen!r}")
    kernel = gl.AnisoRBF(post.coord_sd, ell_frozen * float(cfg.kernel.ell_factor))
    # Check the saved shift against its analytic construction.
    shift = src["shift"]
    chi2 = math.expm1(float(shift @ post.P @ shift))
    if abs(chi2 - src["r_init"] ** 2) > 1e-10:
        raise ValueError(f"{name}: saved shift has chi^2 {chi2}, expected {src['r_init'] ** 2}")
    dirs = rd.registry_directions(post)
    if str(cfg.setting.direction) != "slow":
        raise ValueError("this study uses the saved slow shift only")
    rederived = post.shift(dirs["slow"]["vec"], float(src["r_init"]))
    rederive_diff = float(np.abs(rederived - shift).max())
    if rederive_diff > 1e-10:
        raise ValueError(f"{name}: saved shift differs from its re-derivation by {rederive_diff:.3e}")
    gaps = ckc.grid_gaps(checkpoints)
    exact = str(cfg.transition) == "exact"
    chain = ckc.CheckpointChain.from_chain(post.chain, gaps) if exact else post.chain
    setting = r3.Setting(str(cfg.setting.name), "slow", float(src["r_init"]), shift)
    cfg_like = {
        "checkpoints": checkpoints,
        "budget_total": cfg.budget.total,
        "budget_components": dict(cfg.budget.components),
        "J_grid": list(cfg.J_grid),
        "N_grid": list(cfg.N_grid),
        "M": cfg.landmarks.M,
        "landmark_rtol": cfg.landmarks.rtol,
        "m_grid": list(cfg.m_grid),
        "N_pi": cfg.N_pi,
        "numerical_margin": cfg.landmarks.numerical_margin,
        "op_rtol": cfg.operator.rtol,
        "head": cfg.operator.head,
        "n_dirs": cfg.n_dirs,
        "landmark_closed_form": cfg.landmarks.closed_form,
    }
    problem = r3.build_linear_problem(chain, kernel, [setting], cfg_like, namespace(cfg))
    problem.direct_fp32 = False
    problem.stable_delta_true = True
    nT = len(checkpoints)
    if problem.budget.weight("replicas") != 1.0 / nT or problem.budget.weight("operator") != 1.0 / nT:
        raise ValueError("failure budget not split over the 18 checkpoints")
    ln_mmd = ckc.log_mmd_path(kernel, post.C, post.chain.A, shift, checkpoints)
    truth = problem.truth[0]
    horizon = [
        {
            "t": t,
            "mmd_fp64": float(truth[i]),
            "ln_mmd": ln_mmd[i],
            "log10_mmd": ln_mmd[i] / math.log(10.0),
            "mmd_from_log": ckc.mmd_from_log(ln_mmd[i]),
            "status": ckc.mmd_status(float(truth[i]), ln_mmd[i]),
        }
        for i, t in enumerate(checkpoints)
    ]
    old = dict(zip(src["old_checkpoints"], src["old_truth"]))
    old_diff = max(abs(old[t] - float(truth[i])) for i, t in enumerate(checkpoints) if t in old)
    if old_diff > 1e-12:
        raise ValueError(f"{name}: analytic MMD differs from the source study by {old_diff:.3e}")
    J, N = max(problem.J_grid), max(problem.N_grid)
    meta = {
        "dataset": name,
        "data_provenance": prov,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "sigma2": post.sigma2,
        "ols_resid_var": post.ols_resid_var,
        "mu": post.mu.tolist(),
        "coord_sd": post.coord_sd.tolist(),
        "gibbs_eigvals_real": dirs["eigvals_real"],
        "gibbs_eigvals_imag": dirs["eigvals_imag"],
        "slow_direction": {
            "eigval": str(dirs["slow"]["eigval"]),
            "modulus": dirs["slow"]["modulus"],
            "rule": dirs["slow"]["rule"],
        },
        "source": {
            "summary": src["path"],
            "summary_sha256": src["sha256"],
            "shift_index": src["index"],
            "r_init": src["r_init"],
            "shift": shift.tolist(),
            "shift_chi2": chi2,
            "shift_rederivation_max_abs_diff": rederive_diff,
            "posterior_mu_max_abs_diff": diff_mu,
            "posterior_coord_sd_max_abs_diff": diff_sd,
            "analytic_mmd_max_abs_diff_on_old_grid": old_diff,
            "namespace": str(dcfg.source_namespace),
        },
        "reversibility_error": post.chain.reversibility_error(),
        "stationarity_error": post.chain.stationarity_error(),
        "ell_median_reverified": ell_median,
        "ell": ell_frozen * float(cfg.kernel.ell_factor),
        "transition": str(cfg.transition),
        "gap_checks": (
            {str(g): tr.checks for g, tr in sorted(chain.gaps.items())} if exact else None
        ),
        "transitions": transition_accounting(checkpoints, J, 4 * N, exact),
        "analytic_horizon": horizon,
        "landmark_rank": problem.basis.rank,
        "landmark_orthonormality_error": problem.basis.orthonormality_error,
        "stable_delta_true": True,
        "landmark_closed_form": problem.landmark_closed_form,
        "direct_fp32": False,
    }
    return problem, meta


def fingerprint(cfg: DictConfig) -> str:
    sub = {
        k: OmegaConf.to_container(cfg[k], resolve=True) if OmegaConf.is_config(cfg[k]) else cfg[k]
        for k in PROBLEM_KEYS
    }
    sub["dataset_cfg"] = OmegaConf.to_container(cfg.datasets[str(cfg.dataset)], resolve=True)
    blas = {v: os.environ.get(v) for v in BLAS_VARS}
    return hashlib.sha256(
        json.dumps({"cfg": sub, "blas": blas}, sort_keys=True, default=str).encode()
    ).hexdigest()


def cached_problem(cfg: DictConfig) -> tuple[r3.Problem, dict]:
    """Load a matching setup cache or build the problem.

    A CPU run with +setup_only=true can prepare the cache before GPU sampling.
    """
    path = cfg.get("setup_cache", None)
    fp = fingerprint(cfg)
    if path is not None and pathlib.Path(str(path)).exists():
        blob = pickle.loads(pathlib.Path(str(path)).read_bytes())
        if blob["fingerprint"] != fp:
            raise ValueError(f"setup cache {path} fingerprint {blob['fingerprint'][:12]} != config {fp[:12]}")
        return blob["problem"], {**blob["meta"], "setup_cache": {"path": str(path), "fingerprint": fp}}
    t0 = time.perf_counter()
    problem, meta = build_problem(cfg)
    meta["setup_seconds"] = time.perf_counter() - t0
    if path is not None:
        tmp = pathlib.Path(str(path) + f".tmp{os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pickle.dumps({"fingerprint": fp, "problem": problem, "meta": meta}))
        os.replace(tmp, str(path))
        meta["setup_cache"] = {"path": str(path), "fingerprint": fp}
    return problem, meta


def _normalized_config_text(cfg: DictConfig) -> str:
    """Resolved config text as RunDir.create stores it, with the resume flag normalized to false."""
    d = OmegaConf.to_container(cfg, resolve=True)
    d["resume"] = False
    return OmegaConf.to_yaml(OmegaConf.create(d))


def check_resume_config(run_path: pathlib.Path, cfg: DictConfig, phase: str) -> None:
    """Require the resolved configuration to match the saved run."""
    from mcmc.paperexp.runio import config_sha256

    prov = json.loads((run_path / "PROVENANCE.json").read_text())
    stored = (run_path / "config.yaml").read_text()
    text = _normalized_config_text(cfg)
    if text != stored or config_sha256(text) != prov["config_sha256"]:
        raise ValueError(f"resume of {run_path}: resolved config differs from the run's config.yaml")


def check_resume_setup(run_path: pathlib.Path, problem: r3.Problem, meta: dict) -> None:
    """Check saved arrays, seeds and ensemble ranges before resuming."""
    saved = dict(np.load(run_path / "setup.npz"))
    arrays = r3.problem_arrays(problem)
    if set(saved) != set(arrays) or any(not np.array_equal(saved[k], arrays[k]) for k in saved):
        raise ValueError(f"resume of {run_path}: problem arrays differ from the saved setup.npz")
    old = json.loads((run_path / "setup.json").read_text())
    for key in ("seed_namespace", "rep_range", "ens_range", "blas_threads"):
        if old[key] != meta[key]:
            raise ValueError(f"resume of {run_path}: {key} {meta[key]} != saved {old[key]}")
    if old.get("setup_cache", {}).get("fingerprint") != meta.get("setup_cache", {}).get("fingerprint"):
        raise ValueError(f"resume of {run_path}: setup-cache fingerprint differs")


def run_one(problem: r3.Problem, rep: int, ns: str, device, meta: dict) -> dict:
    """Run one ensemble and verify its represented and simulated transition counts."""
    ch = problem.chain
    exact = isinstance(ch, ckc.CheckpointChain)
    if exact:
        ch.reset_counters()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res = r3.run_ensemble(problem, rep, ns, device)
    acc = meta["transitions"]
    if exact:
        if (
            ch.rows_transitioned != acc["ensemble_transitions_simulated"]
            or ch.rows_sweeps_represented != acc["ensemble_sweeps_represented"]
        ):
            raise RuntimeError(
                f"transition counts {ch.rows_transitioned}/{ch.rows_sweeps_represented} != "
                f"expected {acc['ensemble_transitions_simulated']}/{acc['ensemble_sweeps_represented']}"
            )
    res["transitions_simulated"] = acc["ensemble_transitions_simulated"]
    res["sweeps_represented"] = acc["ensemble_sweeps_represented"]
    res["peak_gpu_bytes_ensemble"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return res


def run_study(cfg: DictConfig) -> RunDir | None:
    name, phase, run_id = str(cfg.dataset), str(cfg.phase), cfg.run_id
    if name not in rd.NAMES:
        raise ValueError(f"unknown dataset {name!r}")
    if str(cfg.transition) == "stepwise" and phase != "smoke":
        raise ValueError("transition=stepwise is the smoke-phase implementation comparison only")
    if bool(cfg.get("setup_only", False)):
        if cfg.get("setup_cache", None) is None:
            raise ValueError("setup_only requires +setup_cache=<path>")
        _, meta = cached_problem(cfg)
        log.info(f"{name}: setup cached at {cfg.setup_cache} ({meta['setup_cache']['fingerprint'][:12]})")
        return None
    if run_id is None or not str(run_id).startswith(f"{name}_"):
        raise ValueError(f"run_id {run_id!r} must start with '{name}_'")
    pcfg = cfg.phases[phase]
    n_ens, rep0, chunk = int(pcfg.n_ensembles), int(pcfg.rep_offset), int(cfg.chunk_size)
    rng = cfg.get("ens_range", None)
    start, stop = (0, n_ens) if rng is None else (int(rng[0]), int(rng[1]))
    if not (0 <= start < stop <= n_ens) or start % chunk or (stop % chunk and stop != n_ens):
        raise ValueError(f"ens_range [{start},{stop}) invalid for n_ensembles={n_ens}, chunk={chunk}")
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    config_text = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    if bool(cfg.resume):
        from mcmc.paperexp.runio import output_root

        check_resume_config(output_root() / str(cfg.study) / phase / str(run_id), cfg, phase)
        run = RunDir.resume(str(cfg.study), phase, str(run_id))
        r3.record_resume(run, fingerprint(cfg))
    else:
        run = RunDir.create(
            str(cfg.study),
            phase,
            str(run_id),
            config_text,
            requested_units=stop - start,
            allow_dirty=phase in ("smoke", "pilot"),
        )
    try:
        problem, meta = cached_problem(cfg)
        meta.update(
            {
                "rep_range": [rep0, rep0 + n_ens],
                "ens_range": [start, stop],
                "budget": problem.budget.to_dict(),
                "blas_threads": {v: os.environ.get(v) for v in BLAS_VARS},
                "seed_namespace": namespace(cfg),
            }
        )
        if not (run.path / "setup.npz").exists():
            if bool(cfg.resume) and any(run.path.glob("chunk_*.npz")):
                raise FileNotFoundError(f"{run.path}: chunks without setup.npz")
            np.savez_compressed(run.path / "setup.npz", **r3.problem_arrays(problem))
            (run.path / "setup.json").write_text(json.dumps(meta, indent=2, default=str))
        elif bool(cfg.resume):
            check_resume_setup(run.path, problem, meta)
        log.info(f"{name}: setup ready; ell={meta['ell']:.6f}; transition={cfg.transition}")
        done = 0
        for c0 in range(start, stop, chunk):
            path = run.path / f"chunk_{c0:05d}.npz"
            c1 = min(stop, c0 + chunk)
            if path.exists():
                done += c1 - c0
                run.update(completed_units=done)
                continue
            results, reps = [], []
            for e in range(c0, c1):
                results.append(run_one(problem, rep0 + e, namespace(cfg), device, meta))
                reps.append(rep0 + e)
                log.info(f"ensemble {e} (rep {rep0 + e}) wall {results[-1]['wall']:.2f}s")
            r3._save_chunk(path, results, reps)
            done += c1 - c0
            run.update(completed_units=done)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return run


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_c_common_horizon", version_base=None
)
def main(cfg: DictConfig) -> None:
    # pickles must reference the importable module, not __main__
    from mcmc.experiments import exp_c_common_horizon as mod

    run = mod.run_study(cfg)
    if run is not None:
        log.info(f"{utc_now()} run finished: {run.path}")


if __name__ == "__main__":
    main()
