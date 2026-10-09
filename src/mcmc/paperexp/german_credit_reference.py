"""NUTS references, kernel calibration, and reference diagnostics.

CPU chains use float64 with independent warmup and dispersed initialization.
Separate runs calibrate the kernel. Permutation tests exchange whole runs
because draws within a run are autocorrelated; the resulting comparisons
remain empirical checks of the NUTS references.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import numpy as np
import yaml

from mcmc.paperexp import german_credit as gc
from mcmc.paperexp.runio import RunDir, output_root

log = logging.getLogger(__name__)

STUDY = "german_credit"
UNIT_FMT = "unit_{:03d}.npz"
LOGPROB_CHECK_RTOL = 1e-9
THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def seed_for(config_id: str, stage: str, rep: int) -> int:
    """sha256-derived 63-bit seed for (configuration, estimation stage, replicate)."""
    digest = hashlib.sha256(f"{config_id}|{stage}|{rep}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class RefSpec:
    preset: str
    model: str
    parameterization: str
    target_accept_prob: float
    max_tree_depth: int
    dense_mass: bool
    n_runs: int
    n_warmup: int
    n_draws: int
    thin: int
    init_radius: float
    role: str
    version: str
    phase: str

    def __post_init__(self):
        if self.model not in gc.model_keys():
            raise ValueError(f"unknown model {self.model!r}")
        if self.parameterization not in ("centered", "noncentered"):
            raise ValueError(f"unknown parameterization {self.parameterization!r}")
        if self.role not in ("reference", "kernel_calibration"):
            raise ValueError(f"unknown role {self.role!r}")
        if self.thin != 1:
            raise ValueError("thinning is not used for the references (thin must be 1)")
        if self.n_runs < 1 or self.n_draws < 1 or self.n_warmup < 1:
            raise ValueError("n_runs, n_draws and n_warmup must be positive")

    @property
    def run_id(self) -> str:
        return f"{self.preset}_{self.role}_{self.version}"

    @property
    def config_id(self) -> str:
        return f"{STUDY}/{self.run_id}"

    @property
    def d(self) -> int:
        return gc.model_dim(self.model)

    def to_yaml(self) -> str:
        header = "# NUTS reference parameters.\n"
        return header + yaml.safe_dump(asdict(self), sort_keys=True)


def make_spec(cfg, preset: str, role: str | None = None) -> RefSpec:
    role = cfg.role if role is None else role
    if preset not in cfg.presets:
        raise KeyError(f"preset {preset!r} not in config presets {list(cfg.presets)}")
    p = cfg.presets[preset]
    return RefSpec(
        preset=preset,
        model=str(p.model),
        parameterization=str(p.parameterization),
        target_accept_prob=float(p.target_accept_prob),
        max_tree_depth=int(p.max_tree_depth),
        dense_mass=bool(p.dense_mass),
        n_runs=int(cfg.n_runs if role == "reference" else cfg.n_runs_calibration),
        n_warmup=int(p.n_warmup) if "n_warmup" in p else int(cfg.n_warmup),
        n_draws=int(cfg.n_draws),
        thin=int(cfg.thin),
        init_radius=float(cfg.init_radius),
        role=role,
        version=str(cfg.version),
        phase=str(cfg.phase),
    )


def parse_cores(text: str) -> list[int]:
    if str(text) == "auto":
        return [min(os.sched_getaffinity(0))]
    cores: list[int] = []
    for part in str(text).split(","):
        lo, _, hi = part.partition("-")
        cores.extend(range(int(lo), int(hi or lo) + 1))
    if len(set(cores)) != len(cores) or not cores:
        raise ValueError(f"bad core list {text!r}")
    return cores


def reparam_log_jacobian(model: str, draws: np.ndarray, centered: bool) -> np.ndarray:
    """log |d beta / d z| for the noncentered parameterization (0 for centered)."""
    if centered:
        return np.zeros(draws.shape[0])
    base = gc.split_kind(model)[0]
    d = draws.shape[1]
    if base == "hier":
        return (d - 2) * draws[:, 1]
    p = (d - 2) // 2
    return (draws[:, 1:2] + draws[:, 2 + p :]).sum(axis=1)


# runs


def run_unit(spec: RefSpec, rep: int, out_dir: str) -> dict:
    """One independent NUTS reference run; writes ``unit_{rep}.npz`` atomically, returns a summary."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp
    import torch
    from numpyro.infer import MCMC, NUTS

    torch.set_num_threads(1)
    t_begin = time.time()
    data = gc.load_data(spec.model)
    post = gc.make_posterior(spec.model, data)
    centered = spec.parameterization == "centered"
    init_seed = seed_for(spec.config_id, "nuts-init", rep)
    key_seed = seed_for(spec.config_id, "nuts-key", rep)
    x0 = np.random.default_rng(init_seed).uniform(-spec.init_radius, spec.init_radius, size=post.d)
    init_params = {
        k: jnp.asarray(v, dtype=jnp.float64)
        for k, v in gc.numpyro_unconstrained_from_declared(spec.model, x0, centered).items()
    }
    kernel = NUTS(
        gc.numpyro_model(spec.model, centered),
        target_accept_prob=spec.target_accept_prob,
        max_tree_depth=spec.max_tree_depth,
        dense_mass=spec.dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=spec.n_warmup,
        num_samples=spec.n_draws,
        num_chains=1,
        thinning=spec.thin,
        progress_bar=False,
    )
    X = jnp.asarray(data.X, dtype=jnp.float64)
    y = jnp.asarray(data.y, dtype=jnp.float64)

    t0 = time.time()
    mcmc.warmup(
        jax.random.PRNGKey(key_seed),
        X,
        y,
        init_params=init_params,
        extra_fields=("diverging", "num_steps", "accept_prob"),
        collect_warmup=True,
    )
    warm = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    t1 = time.time()
    mcmc.run(
        mcmc.post_warmup_state.rng_key,
        X,
        y,
        extra_fields=("diverging", "num_steps", "accept_prob", "potential_energy", "energy"),
    )
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    fields = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    t2 = time.time()

    draws = gc.declared_from_numpyro(spec.model, samples, centered)
    assert draws.shape == (spec.n_draws, post.d), draws.shape
    for name, arr in fields.items():
        assert arr.shape[0] == spec.n_draws, (name, arr.shape)
    for name, arr in warm.items():
        assert arr.shape[0] == spec.n_warmup, (name, arr.shape)

    # Every saved draw: torch log density (+ reparameterization Jacobian) == -potential energy.
    lp = post.log_prob(torch.from_numpy(draws)).numpy()
    pe = fields["potential_energy"].astype(np.float64)
    resid = lp + reparam_log_jacobian(spec.model, draws, centered) + pe
    check_rel = float(np.abs(resid).max() / np.abs(pe).max())
    if not check_rel < LOGPROB_CHECK_RTOL:
        raise RuntimeError(f"{spec.run_id} unit {rep}: log_prob/potential mismatch {check_rel:.3e}")

    adapt = mcmc.last_state.adapt_state
    inv_mass = adapt.inverse_mass_matrix
    inv_mass_items = (
        sorted(inv_mass.items()) if isinstance(inv_mass, dict) else [(("all",), inv_mass)]
    )
    step_size = float(np.asarray(adapt.step_size))
    treedepth_cap = 2**spec.max_tree_depth - 1
    payload = {
        "draws": draws,
        "x0": x0,
        "init_seed": np.int64(init_seed),
        "key_seed": np.int64(key_seed),
        "warm_diverging": warm["diverging"].astype(bool),
        "warm_num_steps": warm["num_steps"].astype(np.int64),
        "warm_accept_prob": warm["accept_prob"].astype(np.float64),
        "diverging": fields["diverging"].astype(bool),
        "num_steps": fields["num_steps"].astype(np.int64),
        "accept_prob": fields["accept_prob"].astype(np.float64),
        "potential_energy": pe,
        "energy": fields["energy"].astype(np.float64),
        "step_size": np.float64(step_size),
        "logprob_check_rel": np.float64(check_rel),
        "t_warmup_s": np.float64(t1 - t0),
        "t_sample_s": np.float64(t2 - t1),
        "t_total_s": np.float64(time.time() - t_begin),
        "spec_yaml": np.array(spec.to_yaml()),
        "inv_mass_sites": np.array(json.dumps([list(k) for k, _ in inv_mass_items])),
    }
    for i, (_, mat) in enumerate(inv_mass_items):
        payload[f"inv_mass_{i}"] = np.asarray(mat, dtype=np.float64)

    out = pathlib.Path(out_dir)
    final = out / UNIT_FMT.format(rep)
    tmp = out / f".tmp_{os.getpid()}_{UNIT_FMT.format(rep)}"
    np.savez(tmp, **payload)
    os.replace(tmp, final)

    energy = payload["energy"]
    ebfmi = float(np.sum(np.diff(energy) ** 2) / np.sum((energy - energy.mean()) ** 2))
    summary = {
        "rep": rep,
        "divergent": int(payload["diverging"].sum()),
        "warm_divergent": int(payload["warm_diverging"].sum()),
        "treedepth_hits": int((payload["num_steps"] >= treedepth_cap).sum()),
        "warm_treedepth_hits": int((payload["warm_num_steps"] >= treedepth_cap).sum()),
        "mean_num_steps": float(payload["num_steps"].mean()),
        "grad_evals_warmup": int(payload["warm_num_steps"].sum()),
        "grad_evals_sample": int(payload["num_steps"].sum()),
        "mean_accept_prob": float(payload["accept_prob"].mean()),
        "step_size": step_size,
        "ebfmi": ebfmi,
        "logprob_check_rel": check_rel,
        "t_warmup_s": float(t1 - t0),
        "t_sample_s": float(t2 - t1),
        "t_total_s": float(payload["t_total_s"]),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "core": sorted(os.sched_getaffinity(0)),
        "mean_log_sigma": float(draws[:, 1].mean()),
    }
    (out / f"unit_{rep:03d}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def unit_complete(path: pathlib.Path, rep: int, spec: RefSpec) -> bool:
    f = path / UNIT_FMT.format(rep)
    if not f.exists():
        return False
    with np.load(f) as z:
        if str(z["spec_yaml"]) != spec.to_yaml():
            raise RuntimeError(f"{f} was produced by a different spec")
        if z["draws"].shape != (spec.n_draws, spec.d):
            raise RuntimeError(f"{f} has draws of shape {z['draws'].shape}")
    return (path / f"unit_{rep:03d}.summary.json").exists()


def _worker_init(core_queue) -> None:
    core = core_queue.get()
    os.sched_setaffinity(0, {core})


def _open_run(spec: RefSpec, driver_cfg_text: str) -> RunDir:
    path = output_root() / STUDY / spec.phase / spec.run_id
    if (path / "RUN-STATUS.json").exists():
        if (path / "config.yaml").read_text() != spec.to_yaml():
            raise RuntimeError(f"{path} exists with a different spec; use a new version")
        return RunDir.resume(STUDY, spec.phase, spec.run_id)
    return RunDir.create(
        STUDY,
        spec.phase,
        spec.run_id,
        spec.to_yaml(),
        requested_units=spec.n_runs,
        allow_dirty=False,
        extra_provenance={"driver_config": driver_cfg_text},
    )


def run_reference_sets(cfg, driver_cfg_text: str) -> list[RunDir]:
    specs = [make_spec(cfg, preset) for preset in cfg.sets]
    cores = parse_cores(cfg.cores)
    allowed = os.sched_getaffinity(0)
    if not set(cores) <= allowed:
        raise RuntimeError(f"cores {cores} not within this process's affinity {sorted(allowed)}")
    os.environ.update(THREAD_ENV)  # inherited by the spawned workers before they import jax/torch

    runs: dict[str, RunDir] = {}
    todo: list[tuple[RefSpec, int]] = []
    for spec in specs:
        run = _open_run(spec, driver_cfg_text)
        runs[spec.run_id] = run
        todo.extend((spec, r) for r in range(spec.n_runs) if not unit_complete(run.path, r, spec))
    rank = {p: int(cfg.presets[p].cost_rank) for p in cfg.presets}
    todo.sort(key=lambda item: (rank[item[0].preset], item[1]))
    log.info("reference units to run: %d on cores %s", len(todo), cores)

    ctx = mp.get_context("spawn")
    core_queue = ctx.Queue()
    for c in cores:
        core_queue.put(c)
    failures: dict[str, list] = {rid: [] for rid in runs}
    try:
        if todo:
            with ProcessPoolExecutor(
                max_workers=min(len(cores), len(todo)),
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(core_queue,),
            ) as pool:
                futures = {
                    pool.submit(run_unit, spec, rep, str(runs[spec.run_id].path)): (spec, rep)
                    for spec, rep in todo
                }
                for fut in as_completed(futures):
                    spec, rep = futures[fut]
                    exc = fut.exception()
                    run = runs[spec.run_id]
                    if exc is not None:
                        failures[spec.run_id].append({"rep": rep, "error": repr(exc)})
                        run.update(failed_units=failures[spec.run_id])
                        log.error("%s unit %d FAILED: %r", spec.run_id, rep, exc)
                        continue
                    s = fut.result()
                    done = sum(unit_complete(run.path, r, spec) for r in range(spec.n_runs))
                    run.update(completed_units=done)
                    log.info(
                        "%s unit %d: %.0fs, div=%d (warmup %d), treedepth=%d, steps=%.1f",
                        spec.run_id,
                        rep,
                        s["t_total_s"],
                        s["divergent"],
                        s["warm_divergent"],
                        s["treedepth_hits"],
                        s["mean_num_steps"],
                    )
    except BaseException as exc:  # record the failure in RUN-STATUS, then re-raise
        for spec in specs:
            runs[spec.run_id].finish("FAILED", note=f"driver aborted: {exc!r}")
        raise
    for spec in specs:
        run = runs[spec.run_id]
        run.update(
            completed_units=sum(unit_complete(run.path, r, spec) for r in range(spec.n_runs))
        )
        if failures[spec.run_id]:
            run.finish("FAILED", note=f"{len(failures[spec.run_id])} unit(s) failed")
        else:
            run.finish("COMPLETED")
    if any(failures.values()):
        raise RuntimeError(f"reference units failed: {failures}")
    return list(runs.values())


# loading


def set_path(phase: str, run_id: str) -> pathlib.Path:
    return output_root() / STUDY / phase / run_id


def load_set(path: pathlib.Path) -> dict:
    """Load a COMPLETED reference set: draws (R, N, d) and per-run sampler diagnostics."""
    status = json.loads((path / "RUN-STATUS.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"{path} is {status['status']}, not COMPLETED")
    spec = yaml.safe_load((path / "config.yaml").read_text())
    R = int(spec["n_runs"])
    keys = ("diverging", "num_steps", "accept_prob", "energy", "warm_diverging", "warm_num_steps")
    draws, diag, summaries = [], {k: [] for k in keys}, []
    for r in range(R):
        with np.load(path / UNIT_FMT.format(r)) as z:
            draws.append(z["draws"])
            for k in keys:
                diag[k].append(z[k])
        summaries.append(json.loads((path / f"unit_{r:03d}.summary.json").read_text()))
    out = {
        "spec": spec,
        "draws": np.stack(draws),
        "summaries": summaries,
        **{k: np.stack(v) for k, v in diag.items()},
    }
    assert out["draws"].shape == (R, int(spec["n_draws"]), gc.model_dim(spec["model"]))
    assert np.isfinite(out["draws"]).all()
    return out


# kernel calibration


SCALE_RULES = {
    "sd": "posterior sd (ddof=1) per coordinate",
    "mad": "1.4826 * MAD per coordinate",
}


def calibrate_kernel(draws: np.ndarray, n_points: int, scale_rule: str = "sd") -> dict:
    """Scales by ``scale_rule``; ell = median pairwise scaled distance of evenly spaced draws.

    ``sd`` uses sample standard deviation (ddof=1); ``mad`` uses 1.4826 times
    the median absolute deviation from the median.
    """
    from scipy.spatial.distance import pdist

    assert draws.ndim == 2
    if scale_rule == "sd":
        scales = draws.std(axis=0, ddof=1)
    elif scale_rule == "mad":
        med = np.median(draws, axis=0)
        scales = 1.4826 * np.median(np.abs(draws - med), axis=0)
    else:
        raise ValueError(f"unknown scale_rule {scale_rule!r}; choose from {sorted(SCALE_RULES)}")
    if not (scales > 0).all():
        raise ValueError(f"zero {scale_rule} scale in calibration draws")
    idx = np.unique(np.linspace(0, draws.shape[0] - 1, n_points).round().astype(np.int64))
    if idx.size != n_points:
        raise ValueError(
            f"cannot pick {n_points} distinct evenly spaced draws from {draws.shape[0]}"
        )
    ell = float(np.median(pdist(draws[idx] / scales)))
    return {
        "scales": scales.tolist(),
        "scale_rule": SCALE_RULES[scale_rule],
        "ell": ell,
        "n_points": int(n_points),
        "n_draws": int(draws.shape[0]),
    }


# kernel algebra


def batch_gram(U, batch_size: int, ell: float, block_batches: int):
    """G[a, b] = sum_{i in batch a, j in batch b} exp(-||u_i - u_j||^2 / (2 ell^2)).

    ``U`` is an (n, d) float64 torch tensor of SCALED points (x / s), n a multiple of
    ``batch_size``; batches are consecutive rows. Blocked over the upper triangle. Returns the
    (n_batches, n_batches) float64 matrix on U's device and the most negative squared distance
    produced by the Gram expansion (rounding; must be > -1e-9 and is clamped to 0 after the check).
    """
    import torch

    n, _ = U.shape
    if U.dtype != torch.float64:
        raise TypeError("batch_gram needs float64 points")
    if n % batch_size:
        raise ValueError(f"n={n} is not a multiple of batch_size={batch_size}")
    nb = n // batch_size
    G = torch.zeros(nb, nb, dtype=torch.float64, device=U.device)
    sq = (U * U).sum(dim=1)
    inv2l2 = 1.0 / (2.0 * ell * ell)
    min_d2 = 0.0
    for a0 in range(0, nb, block_batches):
        a1 = min(nb, a0 + block_batches)
        Ua, sqa = U[a0 * batch_size : a1 * batch_size], sq[a0 * batch_size : a1 * batch_size]
        for b0 in range(a0, nb, block_batches):
            b1 = min(nb, b0 + block_batches)
            Ub, sqb = U[b0 * batch_size : b1 * batch_size], sq[b0 * batch_size : b1 * batch_size]
            d2 = sqa[:, None] + sqb[None, :] - 2.0 * (Ua @ Ub.T)
            blk_min = float(d2.min())
            min_d2 = min(min_d2, blk_min)
            if blk_min < -1e-9:
                raise FloatingPointError(f"squared distance {blk_min} from cancellation")
            K = torch.exp(-d2.clamp_(min=0.0) * inv2l2)
            S = K.view(a1 - a0, batch_size, b1 - b0, batch_size).sum(dim=(1, 3))
            G[a0:a1, b0:b1] = S
            if b0 != a0:
                G[b0:b1, a0:a1] = S.T
    return G, min_d2


def brute_force_batch_gram(U: np.ndarray, batch_size: int, ell: float) -> np.ndarray:
    """Direct pairwise-difference evaluation of ``batch_gram`` (for tests and spot checks)."""
    diff = U[:, None, :] - U[None, :, :]
    K = np.exp(-(diff**2).sum(axis=2) / (2.0 * ell * ell))
    nb = U.shape[0] // batch_size
    return K.reshape(nb, batch_size, nb, batch_size).sum(axis=(1, 3))


def embedding_stats(G: np.ndarray, n_runs: int, n_batches: int, batch_size: int) -> dict:
    """Run-level statistics of one set whose batch Gram is G (runs are consecutive batch groups).

    M[r, s] = <mu_r, mu_s> for run mean embeddings; v_r = batch-means estimate of
    E||mu_r - E mu_r||^2; B = between-run variance; first/second half differences d_r and their
    Gram D[r, s] = <d_r, d_s>.
    """
    R, nb, L = n_runs, n_batches, batch_size
    assert G.shape == (R * nb, R * nb), G.shape
    if nb % 2:
        raise ValueError("an even number of batches per run is needed for half splits")
    N = nb * L
    G4 = G.reshape(R, nb, R, nb)
    M = G4.sum(axis=(1, 3)) / N**2
    v = np.empty(R)
    for r in range(R):
        Gb = G4[r, :, r, :] / L**2  # batch-mean inner products
        dev2 = np.diag(Gb) - 2.0 * Gb.mean(axis=1) + Gb.mean()
        v[r] = dev2.sum() / (nb - 1) / nb
    h = nb // 2
    first = G4[:, :h, :, :].sum(axis=1)  # (R, R, nb) partial sums, rows restricted to first half
    S11 = first[:, :, :h].sum(axis=2)
    S12 = first[:, :, h:].sum(axis=2)
    second = G4[:, h:, :, :].sum(axis=1)
    S21 = second[:, :, :h].sum(axis=2)
    S22 = second[:, :, h:].sum(axis=2)
    Nh = h * L
    D = (S11 - S12 - S21 + S22) / Nh**2  # <mu_r1 - mu_r2, mu_s1 - mu_s2>
    mu_bar_sq = M.mean()
    B = (np.trace(M) - R * mu_bar_sq) / (R - 1)
    pair = np.diag(M)[:, None] + np.diag(M)[None, :] - 2.0 * M
    return {
        "M": M,
        "v": v,
        "B": float(B),
        "D_half": D,
        "pairwise_sq": pair,
        "mu_bar_sq": float(mu_bar_sq),
    }


def exact_split_test(M: np.ndarray, group_a: np.ndarray) -> dict:
    """Exact run-label permutation test of ||mean_A mu - mean_B mu||^2 for an equal split."""
    R = M.shape[0]
    if R % 2 or group_a.sum() != R // 2:
        raise ValueError("equal split of an even number of runs required")
    combos = np.array(list(itertools.combinations(range(R), R // 2)))
    A = np.full((combos.shape[0], R), -2.0 / R)
    np.put_along_axis(A, combos, 2.0 / R, axis=1)
    stats = np.einsum("pr,rs,ps->p", A, M, A)
    a_obs = np.where(group_a, 2.0 / R, -2.0 / R)
    obs = float(a_obs @ M @ a_obs)
    tol = 1e-12 * max(1.0, abs(obs))
    return {
        "stat": obs,
        "p_value": float(np.mean(stats >= obs - tol)),
        "n_perm": int(stats.size),
        "null_q95": float(np.quantile(stats, 0.95)),
        "null_median": float(np.median(stats)),
    }


def exact_signflip_test(D: np.ndarray) -> dict:
    """Exact sign-flip test of ||R^-1 sum_r d_r||^2 given D[r, s] = <d_r, d_s>."""
    R = D.shape[0]
    if R > 20:
        raise ValueError("exact enumeration limited to R <= 20")
    signs = 1.0 - 2.0 * ((np.arange(2**R)[:, None] >> np.arange(R)[None, :]) & 1)
    stats = np.einsum("pr,rs,ps->p", signs, D, signs) / R**2
    obs = float(D.sum() / R**2)
    tol = 1e-12 * max(1.0, abs(obs))
    return {
        "stat": obs,
        "p_value": float(np.mean(stats >= obs - tol)),
        "n_perm": int(stats.size),
        "null_q95": float(np.quantile(stats, 0.95)),
        "null_median": float(np.median(stats)),
    }


def mc_cross_test(M_all: np.ndarray, n_a: int, n_perm: int, seed: int) -> dict:
    """||mean_{first n_a runs} mu - mean_{rest} mu||^2 with a Monte Carlo run-label permutation test.

    Exact only if the run-level mean embeddings of the two sets are exchangeable. Within one NUTS
    configuration (the 8v8 split) they are i.i.d.; ACROSS configurations (cross-parameterization
    checks) equal posterior means do not make the run-level distributions equal (autocorrelation,
    adaptation and MC error differ), so those p-values are exploratory approximations
    (Chung & Romano 2013, Ann. Statist. 41:484-507).
    """
    R = M_all.shape[0]
    n_b = R - n_a
    a_obs = np.r_[np.full(n_a, 1.0 / n_a), np.full(n_b, -1.0 / n_b)]
    obs = float(a_obs @ M_all @ a_obs)
    rng = np.random.default_rng(seed)
    count = 0
    null = np.empty(n_perm)
    chunk = 10000
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        perms = np.argsort(rng.random((m, R)), axis=1)
        A = np.where(perms < n_a, 1.0 / n_a, -1.0 / n_b)
        st = np.einsum("pr,rs,ps->p", A, M_all, A)
        null[start : start + m] = st
        count += int((st >= obs - 1e-12 * max(1.0, abs(obs))).sum())
    return {
        "stat": obs,
        "p_value": (1 + count) / (1 + n_perm),
        "n_perm": int(n_perm),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_median": float(np.median(null)),
    }


# conventional


def conventional_diagnostics(ref: dict, names: list[str], max_tree_depth: int) -> dict:
    import arviz as az

    x = ref["draws"]
    rhat = np.asarray(az.rhat(x, method="rank")).ravel()
    ess_bulk = np.asarray(az.ess(x, method="bulk")).ravel()
    ess_tail = np.asarray(
        az.ess(x, method="tail", prob=(0.05, 0.95))
    ).ravel()  # Vehtari et al. 2021
    mcse_mean = np.asarray(az.mcse(x, method="mean")).ravel()
    mcse_sd = np.asarray(az.mcse(x, method="sd")).ravel()
    assert rhat.shape == (x.shape[2],)
    cap = 2**max_tree_depth - 1
    energy = ref["energy"]
    ebfmi = np.sum(np.diff(energy, axis=1) ** 2, axis=1) / np.sum(
        (energy - energy.mean(axis=1, keepdims=True)) ** 2, axis=1
    )
    per_run_div = ref["diverging"].sum(axis=1)
    return {
        "rhat_max": float(rhat.max()),
        "rhat_argmax": names[int(rhat.argmax())],
        "n_rhat_gt_1.01": int((rhat > 1.01).sum()),
        "ess_bulk_min": float(ess_bulk.min()),
        "ess_bulk_argmin": names[int(ess_bulk.argmin())],
        "ess_tail_min": float(ess_tail.min()),
        "ess_tail_argmin": names[int(ess_tail.argmin())],
        "ess_bulk_median": float(np.median(ess_bulk)),
        "total_draws": int(x.shape[0] * x.shape[1]),
        "divergent_total": int(per_run_div.sum()),
        "divergent_per_run": per_run_div.tolist(),
        "divergent_rate": float(ref["diverging"].mean()),
        "warm_divergent_per_run": ref["warm_diverging"].sum(axis=1).tolist(),
        "treedepth_hits_total": int((ref["num_steps"] >= cap).sum()),
        "warm_treedepth_hits_total": int((ref["warm_num_steps"] >= cap).sum()),
        "mean_num_steps": float(ref["num_steps"].mean()),
        "ebfmi_min": float(ebfmi.min()),
        "step_size_range": [
            float(min(s["step_size"] for s in ref["summaries"])),
            float(max(s["step_size"] for s in ref["summaries"])),
        ],
        "per_coordinate": {
            "names": names,
            "mean": x.reshape(-1, x.shape[2]).mean(axis=0).tolist(),
            "sd": x.reshape(-1, x.shape[2]).std(axis=0, ddof=1).tolist(),
            "rhat": rhat.tolist(),
            "ess_bulk": ess_bulk.tolist(),
            "ess_tail": ess_tail.tolist(),
            "mcse_mean": mcse_mean.tolist(),
            "mcse_sd": mcse_sd.tolist(),
        },
    }


def coordinate_agreement(c1: dict, c2: dict) -> dict:
    """Per-coordinate mean and sd differences between two sets in MCSE units."""
    a, b = c1["per_coordinate"], c2["per_coordinate"]
    zm = (np.array(a["mean"]) - np.array(b["mean"])) / np.hypot(a["mcse_mean"], b["mcse_mean"])
    zs = (np.array(a["sd"]) - np.array(b["sd"])) / np.hypot(a["mcse_sd"], b["mcse_sd"])
    return {
        "max_abs_z_mean": float(np.abs(zm).max()),
        "argmax_z_mean": a["names"][int(np.abs(zm).argmax())],
        "n_abs_z_mean_gt_3": int((np.abs(zm) > 3).sum()),
        "max_abs_z_sd": float(np.abs(zs).max()),
        "argmax_z_sd": a["names"][int(np.abs(zs).argmax())],
        "n_abs_z_sd_gt_3": int((np.abs(zs) > 3).sum()),
        "z_mean": zm.tolist(),
        "z_sd": zs.tolist(),
    }


def cost_summary(ref: dict) -> dict:
    s = ref["summaries"]
    return {
        "wall_s_per_run_median": float(np.median([u["t_total_s"] for u in s])),
        "core_s_total": float(sum(u["t_total_s"] for u in s)),
        "warmup_s_total": float(sum(u["t_warmup_s"] for u in s)),
        "sample_s_total": float(sum(u["t_sample_s"] for u in s)),
        "grad_evals_warmup_total": int(sum(u["grad_evals_warmup"] for u in s)),
        "grad_evals_sample_total": int(sum(u["grad_evals_sample"] for u in s)),
        "maxrss_mb_max": float(max(u["maxrss_mb"] for u in s)),
        "logprob_check_rel_max": float(max(u["logprob_check_rel"] for u in s)),
    }


# analysis


def kernel_key(model: str, scale_rule: str) -> str:
    """Entry name in the kernel file: the sd-scaled kernel keeps the bare model name."""
    return model if scale_rule == "sd" else f"{model}_{scale_rule}"


def load_kernel_file(path: pathlib.Path, model: str) -> dict:
    spec = yaml.safe_load(path.read_text())
    if model not in spec:
        raise KeyError(f"kernel file {path} has no entry for model {model!r}")
    return spec[model]


def kernel_quality(sets: dict[str, dict], kernel: dict, acfg, perm_seed: int, device: str) -> dict:
    """Kernel reference-quality statistics for one model's reference sets (same frozen kernel)."""
    import torch

    names = list(sets)
    R = {k: sets[k]["draws"].shape[0] for k in names}
    N = {k: sets[k]["draws"].shape[1] for k in names}
    L = int(acfg.batch_size)
    if len(set(N.values())) != 1 or next(iter(N.values())) % L:
        raise ValueError(f"draw counts {N} must agree and be multiples of batch_size={L}")
    n_draws = next(iter(N.values()))
    nb = n_draws // L
    scales = np.asarray(kernel["scales"], dtype=np.float64)
    pts = np.concatenate([sets[k]["draws"].reshape(-1, scales.size) for k in names]) / scales
    U = torch.as_tensor(pts, dtype=torch.float64, device=device)

    # Spot-check entries of the blocked device Gram against direct differences on two batches.
    rng = np.random.default_rng(perm_seed)
    pick = np.sort(rng.choice(U.shape[0] // L, size=2, replace=False))
    spot = np.concatenate([pts[b * L : (b + 1) * L] for b in pick])

    out: dict = {"sets": names, "n_runs": R, "n_draws": n_draws, "batch_size": L, "bandwidths": {}}
    for mult in acfg.bandwidth_multipliers:
        ell = float(kernel["ell"]) * float(mult)
        t0 = time.time()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        Gt, min_d2 = batch_gram(U, L, ell, int(acfg.block_batches))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_gram = time.time() - t0
        G = Gt.cpu().numpy()
        spot_ref = brute_force_batch_gram(spot, L, ell)
        spot_err = float(np.abs(G[np.ix_(pick, pick)] - spot_ref).max() / np.abs(spot_ref).max())
        if spot_err > 1e-10:
            raise FloatingPointError(f"blocked Gram disagrees with direct evaluation: {spot_err}")

        res: dict = {
            "ell": ell,
            "t_gram_s": t_gram,
            "min_sq_dist_rounding": min_d2,
            "spot_check_rel_err": spot_err,
            "per_set": {},
        }
        offsets = np.cumsum([0] + [R[k] * nb for k in names])
        run_M_blocks = []
        for i, k in enumerate(names):
            sl = slice(offsets[i], offsets[i + 1])
            st = embedding_stats(G[sl, sl], R[k], nb, L)
            Rk = R[k]
            W = float(st["v"].mean())
            Vk = 1.0 - st["mu_bar_sq"]  # diag-one kernel: E k(x,x) - ||mu||^2 (pooled plug-in)
            half = np.r_[np.ones(Rk // 2, bool), np.zeros(Rk - Rk // 2, bool)]
            split = exact_split_test(st["M"], half)
            flip = exact_signflip_test(st["D_half"])
            pair = st["pairwise_sq"][np.triu_indices(Rk, 1)]
            vv = (st["v"][:, None] + st["v"][None, :])[np.triu_indices(Rk, 1)]
            M = st["M"]
            loo = []
            for r in range(Rk):
                rest = np.delete(np.arange(Rk), r)
                loo.append(M[r, r] - 2 * M[r, rest].mean() + M[np.ix_(rest, rest)].mean())
            res["per_set"][k] = {
                "between_run_var_B": st["B"],
                "within_run_batchmeans_var_W": W,
                "B_over_W": st["B"] / W,
                "V_k_plugin": Vk,
                "kernel_ess_per_run_batchmeans_median": float(np.median(Vk / st["v"])),
                "kernel_ess_per_run_batchmeans_min": float(np.min(Vk / st["v"])),
                "kernel_ess_per_run_between": Vk / st["B"],
                "pooled_reference_rms_mc_error": math.sqrt(st["B"] / Rk),
                "pairwise_mmd_sq_median": float(np.median(pair)),
                "pairwise_mmd_sq_max": float(pair.max()),
                "pairwise_over_batchmeans_median": float(np.median(pair / vv)),
                "pairwise_over_batchmeans_max": float((pair / vv).max()),
                "loo_mmd_sq_max": float(max(loo)),
                "loo_over_expected_max": float(
                    max(loo[r] / (st["v"][r] + W / (Rk - 1)) for r in range(Rk))
                ),
                "split_8v8": split,
                "half_runs_vs_full_mmd_sq": split["stat"] / 4.0,
                "half_draws_vs_full_mmd_sq": flip["stat"] / 4.0,
                "first_vs_second_half_signflip": flip,
                "v_per_run": st["v"].tolist(),
            }
            run_M_blocks.append((k, sl))
        # The first set is the primary reference; compare it with every other set.
        n_all = sum(R.values())
        M_all = G.reshape(n_all, nb, n_all, nb).sum(axis=(1, 3)) / n_draws**2
        run_off = np.cumsum([0] + [R[k] for k in names])
        res["cross_sets"] = []
        k1 = names[0]
        for i, k2 in enumerate(names[1:], start=1):
            idx = np.r_[run_off[0] : run_off[1], run_off[i] : run_off[i + 1]]
            cross = mc_cross_test(
                M_all[np.ix_(idx, idx)], R[k1], int(acfg.cross_permutations), perm_seed + i
            )
            B1 = res["per_set"][k1]["between_run_var_B"]
            B2 = res["per_set"][k2]["between_run_var_B"]
            cross["expected_under_equal_laws"] = B1 / R[k1] + B2 / R[k2]
            res["cross_sets"].append({"sets": [k1, k2], **cross})
        out["bandwidths"][str(mult)] = res
    return out


def analyze_model(cfg) -> RunDir:
    import torch
    from omegaconf import OmegaConf

    model = str(cfg.model)
    acfg = cfg.analysis
    prim = str(cfg.primary[model])
    crosses = [str(c) for c in cfg.crosscheck[model]]
    calib_spec = make_spec(cfg, str(cfg.calibration[model]), role="kernel_calibration")
    sets_spec = [make_spec(cfg, p, role="reference") for p in [prim, *crosses]]

    rule = str(acfg.scale_rule)
    kernel_file = pathlib.Path(__file__).resolve().parents[3] / str(acfg.kernel_file)
    kernel = load_kernel_file(kernel_file, kernel_key(model, rule))
    calib = load_set(set_path(calib_spec.phase, calib_spec.run_id))
    recomputed = calibrate_kernel(calib["draws"][0], int(acfg.n_bandwidth_points), rule)
    if not (
        np.allclose(recomputed["scales"], kernel["scales"], rtol=1e-12, atol=0)
        and math.isclose(recomputed["ell"], kernel["ell"], rel_tol=1e-12)
    ):
        raise RuntimeError(f"frozen kernel in {kernel_file} does not match calibration run")

    run = RunDir.create(
        STUDY,
        str(cfg.phase),
        f"quality_{model}_{acfg.tag}",
        yaml.safe_dump(
            {
                "model": model,
                "sets": [s.run_id for s in sets_spec],
                "calibration": calib_spec.run_id,
                "kernel": kernel,
                "analysis": OmegaConf.to_container(acfg, resolve=True),
            },
            sort_keys=True,
        ),
        requested_units=1,
    )
    try:
        sets = {s.run_id: load_set(set_path(s.phase, s.run_id)) for s in sets_spec}
        names = gc.make_posterior(model).coordinate_names()
        result: dict = {"model": model, "kernel": kernel, "kernel_file": str(acfg.kernel_file)}
        conv = {}
        for s in sets_spec:
            conv[s.run_id] = conventional_diagnostics(sets[s.run_id], names, s.max_tree_depth)
            conv[s.run_id]["cost"] = cost_summary(sets[s.run_id])
        result["conventional"] = conv
        result["coordinate_agreement"] = {
            s.run_id: coordinate_agreement(conv[sets_spec[0].run_id], conv[s.run_id])
            for s in sets_spec[1:]
        }
        perm_seed = seed_for(
            f"{STUDY}/quality_{model}_{acfg.tag}", str(acfg.permutation_seed_stage), 0
        )
        device = str(acfg.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("analysis.device=cuda but CUDA is unavailable; set analysis.device=cpu")
        t0 = time.time()
        result["kernel_quality"] = kernel_quality(sets, kernel, acfg, perm_seed, device)
        result["kernel_quality"]["t_total_s"] = time.time() - t0
        (run.path / "quality.json").write_text(json.dumps(result, indent=2, default=float))
        run.update(completed_units=1)
        run.finish("COMPLETED")
    except BaseException as exc:  # record, then re-raise
        run.finish("FAILED", note=repr(exc))
        raise
    return run


def calibrate(cfg) -> dict:
    """Add kernels from completed calibration runs to the configured YAML file.

    Existing entries must agree to relative tolerance 1e-12.
    """
    kernel_file = pathlib.Path(__file__).resolve().parents[3] / str(cfg.analysis.kernel_file)
    existing = yaml.safe_load(kernel_file.read_text()) if kernel_file.exists() else {}
    out = dict(existing)
    rule = str(cfg.analysis.scale_rule)
    for model, preset in cfg.calibration.items():
        spec = make_spec(cfg, str(preset), role="kernel_calibration")
        path = set_path(spec.phase, spec.run_id)
        if not (path / "RUN-STATUS.json").exists():
            continue
        calib = load_set(path)
        entry = calibrate_kernel(calib["draws"][0], int(cfg.analysis.n_bandwidth_points), rule)
        entry["calibration_run"] = f"{STUDY}/{spec.phase}/{spec.run_id}"
        entry["kernel"] = "diag-one scaled RBF exp(-||(x-y)/scales||^2 / (2 ell^2)); ell = median"
        key = kernel_key(str(model), rule)
        if key in existing:
            old = existing[key]
            if not (
                np.allclose(old["scales"], entry["scales"], rtol=1e-12, atol=0)
                and math.isclose(old["ell"], entry["ell"], rel_tol=1e-12)
            ):
                raise RuntimeError(f"{kernel_file} already freezes a different kernel for {key}")
            continue
        out[key] = entry
    header = (
        "# Logistic reference kernels from mode=calibrate.\n"
    )
    kernel_file.write_text(header + yaml.safe_dump(out, sort_keys=True))
    return out


def main_entry() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        config_path="../../../configs/experiment",
        config_name="exp_r2_german_credit_reference",
        version_base=None,
    )
    def _main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO)
        text = OmegaConf.to_yaml(cfg)
        if cfg.mode == "run":
            run_reference_sets(cfg, text)
        elif cfg.mode == "calibrate":
            print(yaml.safe_dump(calibrate(cfg), sort_keys=True))
        elif cfg.mode == "analyze":
            run = analyze_model(cfg)
            print(run.path)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

    _main()


if __name__ == "__main__":
    main_entry()
