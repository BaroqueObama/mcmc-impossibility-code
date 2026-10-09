"""Anytime Gibbs bounds for Energy, Airfoil, Wine and kin8nm regression.

Uses exp_r3_concrete's model and bounds. Select a dataset with dataset=<name>;
results go to r3_<name>/<phase>/<run_id>.
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
from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import regression_data as rd
from mcmc.paperexp.runio import RunDir, utc_now

log = logging.getLogger(__name__)
PROBLEM_KEYS = (
    "seed_namespace", "dataset", "model", "kernel", "settings", "t_max_threshold", "t_max_cap",
    "m_grid", "n_dirs", "J_grid", "N_grid", "N_pi", "landmarks", "operator", "budget",
)  # fmt: skip


def namespace(cfg: DictConfig) -> str:
    return f"{cfg.seed_namespace}-{cfg.dataset}"


def median_bandwidth(chain, coord_sd: np.ndarray, ns: str, n_draws: int) -> float:
    """Median scaled pairwise distance of independent posterior draws."""
    g_bw = r3.torch_gen(r3.seed_for(ns, "pilot-bw", 0), "cpu")
    D = chain.sample_stationary(n_draws, g_bw, "cpu") / torch.as_tensor(coord_sd)
    dist = torch.cdist(D, D)
    iu = torch.triu_indices(D.shape[0], D.shape[0], 1)
    return float(dist[iu[0], iu[1]].median())


def t_max_checkpoints(
    kernel, chain, slow_shift: np.ndarray, threshold: float, cap: int
) -> tuple[list, bool]:
    """Checkpoint grid {0, 1, 2, 4, ..., t_max}: t_max = first power of two with analytic
    MMD(slow, R=0.8) < threshold, or the cap (returned flag True) if none is found up to it."""
    t = 1
    while t <= cap:
        mmd = math.sqrt(gl.mmd2_shifted_gaussians(kernel, chain.C, chain.power(t) @ slow_shift))
        if mmd < threshold:
            return [0] + [2**j for j in range(int(math.log2(t)) + 1)], False
        t *= 2
    return [0] + [2**j for j in range(int(math.log2(cap)) + 1)], True


def build_problem(cfg: DictConfig) -> tuple[r3.Problem, dict]:
    name = str(cfg.dataset)
    dcfg = cfg.datasets[name]
    X, y, prov = rd.load_dataset(name)
    post = rd.RegressionPosterior(name, X, y, prior_var=float(cfg.model.prior_var))
    if post.sigma2 != float(dcfg.sigma2_declared):
        raise ValueError(
            f"{name}: declared sigma2 {dcfg.sigma2_declared} != OLS rounding {post.sigma2}"
        )
    ch, ns = post.chain, namespace(cfg)
    ell_median = median_bandwidth(ch, post.coord_sd, ns, int(cfg.kernel.bandwidth_draws))
    if dcfg.ell_frozen is not None and abs(ell_median - float(dcfg.ell_frozen)) > 1e-12:
        raise ValueError(f"{name}: median bandwidth {ell_median!r} != frozen {dcfg.ell_frozen!r}")
    kernel = gl.AnisoRBF(post.coord_sd, ell_median * float(cfg.kernel.ell_factor))
    dirs = rd.registry_directions(post)
    settings = [
        r3.Setting(
            str(s.name),
            str(s.direction),
            float(s.r_init),
            post.shift(dirs[str(s.direction)]["vec"], float(s.r_init)),
        )
        for s in cfg.settings
    ]
    slow08 = post.shift(dirs["slow"]["vec"], 0.8)
    expected, cap_binds = t_max_checkpoints(
        kernel, ch, slow08, float(cfg.t_max_threshold), int(cfg.t_max_cap)
    )
    checkpoints = [int(t) for t in dcfg.checkpoints]
    if checkpoints != expected:
        raise ValueError(f"{name}: checkpoints {checkpoints} != t_max rule {expected}")
    cfg_like = {
        "checkpoints": checkpoints,
        "budget_total": cfg.budget.total,
        "budget_components": dict(cfg.budget.components),
        "J_grid": list(cfg.J_grid),
        "N_grid": list(
            dcfg.get("N_grid", cfg.N_grid)
        ),
        "M": cfg.landmarks.M,
        "landmark_rtol": cfg.landmarks.rtol,
        "m_grid": list(cfg.m_grid),
        "N_pi": cfg.N_pi,
        "numerical_margin": cfg.landmarks.numerical_margin,
        "op_rtol": cfg.operator.rtol,
        "head": cfg.operator.head,
        "n_dirs": cfg.n_dirs,
    }
    problem = r3.build_linear_problem(ch, kernel, settings, cfg_like, ns)
    problem.direct_fp32 = bool(
        dcfg.get("direct_fp32", False)
    )  # fp32 direct terms include the rounding-error bound
    meta = {
        "dataset": name,
        "data_provenance": prov,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "sigma2": post.sigma2,
        "ols_resid_var": post.ols_resid_var,
        "cond_C": float(np.linalg.cond(post.C)),
        "mu": post.mu.tolist(),
        "coord_sd": post.coord_sd.tolist(),
        "gibbs_eigvals_real": dirs["eigvals_real"],
        "gibbs_eigvals_imag": dirs["eigvals_imag"],
        "directions": {
            k: {
                "eigval": str(dirs[k]["eigval"]),
                "modulus": dirs[k]["modulus"],
                "rule": dirs[k]["rule"],
                "vec": dirs[k]["vec"].tolist(),
            }
            for k in ("slow", "fast")
        },
        "reversibility_error": ch.reversibility_error(),
        "stationarity_error": ch.stationarity_error(),
        "ell_median": ell_median,
        "ell": ell_median * float(cfg.kernel.ell_factor),
        "t_max_cap_binds": cap_binds,
        "N_grid": cfg_like["N_grid"],
        "direct_fp32": problem.direct_fp32,
        "landmark_rank": problem.basis.rank,
        "landmark_orthonormality_error": problem.basis.orthonormality_error,
    }
    return problem, meta


def fingerprint(cfg: DictConfig) -> str:
    sub = {
        k: OmegaConf.to_container(cfg[k], resolve=True) if OmegaConf.is_config(cfg[k]) else cfg[k]
        for k in PROBLEM_KEYS
    }
    sub["dataset_cfg"] = OmegaConf.to_container(cfg.datasets[str(cfg.dataset)], resolve=True)
    blas = {
        v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    return hashlib.sha256(
        json.dumps({"cfg": sub, "blas": blas}, sort_keys=True, default=str).encode()
    ).hexdigest()


def cached_problem(cfg: DictConfig) -> tuple[r3.Problem, dict]:
    path = cfg.get("setup_cache", None)
    fp = fingerprint(cfg)
    if path is not None and pathlib.Path(str(path)).exists():
        blob = pickle.loads(pathlib.Path(str(path)).read_bytes())
        if blob["fingerprint"] != fp:
            raise ValueError(
                f"setup cache {path} fingerprint {blob['fingerprint'][:12]} != config {fp[:12]}"
            )
        return blob["problem"], {
            **blob["meta"],
            "setup_cache": {"path": str(path), "fingerprint": fp},
        }
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


def run_study(cfg: DictConfig) -> RunDir | None:
    name, phase, run_id = str(cfg.dataset), str(cfg.phase), cfg.run_id
    if name not in rd.NAMES or name == "concrete":
        raise ValueError(f"dataset {name!r} not handled here (concrete: exp_r3_concrete)")
    if bool(cfg.get("setup_only", False)):
        if cfg.get("setup_cache", None) is None:
            raise ValueError("setup_only requires +setup_cache=<path>")
        _, meta = cached_problem(cfg)
        log.info(
            f"{name}: setup cached at {cfg.setup_cache} ({meta['setup_cache']['fingerprint'][:12]})"
        )
        return None
    if run_id is None:
        raise ValueError("run_id is required")
    pcfg = cfg.phases[phase]
    n_ens, rep0, chunk = int(pcfg.n_ensembles), int(pcfg.rep_offset), int(cfg.chunk_size)
    rng = cfg.get("ens_range", None)
    start, stop = (0, n_ens) if rng is None else (int(rng[0]), int(rng[1]))
    if not (0 <= start < stop <= n_ens) or start % chunk or (stop % chunk and stop != n_ens):
        raise ValueError(
            f"ens_range [{start},{stop}) invalid for n_ensembles={n_ens}, chunk={chunk}"
        )
    if phase in ("main", "stress") and cfg.datasets[name].ell_frozen is None:
        raise ValueError("main/stress phases require a frozen bandwidth")
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    config_text = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    study = f"r3_{name}"
    if bool(cfg.resume):
        run = RunDir.resume(study, phase, str(run_id))
        r3.record_resume(run, fingerprint(cfg))
    else:
        run = RunDir.create(
            study,
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
            }
        )
        meta["blas_threads"] = {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        if not (run.path / "setup.npz").exists():
            np.savez_compressed(run.path / "setup.npz", **r3.problem_arrays(problem))
            (run.path / "setup.json").write_text(json.dumps(meta, indent=2, default=str))
        log.info(
            f"{name}: setup ready; ell={meta['ell']:.6f}; checkpoints={problem.checkpoints}; truth[:,0]={problem.truth[:, 0]}"
        )
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
                results.append(r3.run_ensemble(problem, rep0 + e, namespace(cfg), device))
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
    config_path="../../../configs/experiment", config_name="exp_r3_regression", version_base=None
)
def main(cfg: DictConfig) -> None:
    # pickles must reference the importable module, not __main__
    from mcmc.experiments import exp_r3_regression as mod

    run = mod.run_study(cfg)
    if run is not None:
        log.info(f"{utc_now()} run finished: {run.path}")


if __name__ == "__main__":
    main()
