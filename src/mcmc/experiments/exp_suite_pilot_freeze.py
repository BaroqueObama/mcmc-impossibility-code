"""Calibrate synthetic kernels, sampler step sizes and checkpoint schedules.

Calibration and verification use separate random streams. stage=freeze saves
completed pilot results to configs/paperexp/frozen_suite_v1.yaml.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
import resource
import time

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp.embeddings import (
    mc_mean_embedding,
    mc_mean_embedding_norm2,
    quadpack_mu,
    rbf_mean_embedding,
)
from mcmc.paperexp.kernels import ScaledRBF, kernel_sum, mmd2_from_embedding
from mcmc.paperexp.rng import seed_for, torch_gen
from mcmc.paperexp.runio import RunDir, output_root
from mcmc.paperexp.samplers import make_sampler
from mcmc.paperexp.targets import (
    PRIMARY_TARGETS,
    ROTATIONS_PATH,
    STRESS_TARGETS,
    Target,
    make_target,
    rotation_key,
)

log = logging.getLogger(__name__)
F64 = torch.float64
REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY = "suite_pilot"


def config_id(target_id: str, kind: str) -> str:
    return f"{target_id}__{kind}"


def tier_targets(tier: str) -> list[tuple[str, int, dict]]:
    if tier == "primary":
        return PRIMARY_TARGETS
    if tier == "stress":
        return STRESS_TARGETS
    raise ValueError(f"unknown tier {tier!r}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: pathlib.Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def stage_scale(t: Target, cfg, device) -> dict:
    tid = t.target_id
    x = t.sample(int(cfg.scale.n_draws), torch_gen(seed_for(tid, "pilot-scale", 0), device))
    med = x.median(0).values
    mad = (x - med).abs().median(0).values
    scales = (1.4826 * mad).cpu().numpy()
    assert np.isfinite(scales).all() and (scales > 0).all(), scales
    xb = t.sample(int(cfg.scale.n_bw), torch_gen(seed_for(tid, "pilot-bw", 0), device))
    u = xb / torch.as_tensor(scales, dtype=F64, device=device)
    iu = torch.triu_indices(u.shape[0], u.shape[0], offset=1, device=device)
    dist = torch.sqrt(((u[iu[0]] - u[iu[1]]) ** 2).sum(1))
    ell = float(dist.median().item())
    out = {
        "target": {k: v for k, v in t.to_dict().items() if k != "rotation"},
        "scales": scales.tolist(),
        "scale_rule": "1.4826 * MAD per coordinate",
        "scale_draws": int(cfg.scale.n_draws),
        "ell": ell,
        "ell_rule": "median pairwise ||(x_i - x_j)/s|| over i<j",
        "bw_draws": int(cfg.scale.n_bw),
        "analytic_coordinate_std": t.coordinate_std().tolist(),
        "rbf_embedding": {},
    }
    for name, mult in (("ell", 1.0), ("ell_half", 0.5), ("ell_double", 2.0)):
        emb = rbf_mean_embedding(t, ScaledRBF(scales, ell * mult))
        info = {k: v for k, v in emb.info.items() if k != "doubling_history"}
        out["rbf_embedding"][name] = {"ell": ell * mult, "norm2": emb.norm2(), **info}
    return out


def _acceptance(t, kind, h, scales, J, n_steps, seed, device):
    smp = make_sampler(kind, t, h, scales)
    g = torch_gen(seed, device)
    out = smp.run(t.sample(J, g), [n_steps], g, store=False)
    per_chain = out["accept_count_per_chain"].double() / n_steps
    return out["accept_rate"], float(per_chain.std().item() / math.sqrt(J))


def stage_tune(t: Target, kind: str, scales, cfg, device) -> dict:
    cid = config_id(t.target_id, kind)
    target_acc = float(cfg.tune.target_accept[kind])
    J, n_steps = int(cfg.tune.J), int(cfg.tune.n_steps)
    seed = seed_for(cid, "pilot-tune", 0)
    trace = []

    def acc(h):
        a, _ = _acceptance(t, kind, h, scales, J, n_steps, seed, device)
        trace.append({"h": h, "accept": a})
        return a

    h = 2.38 / math.sqrt(t.d) if kind == "rwm" else 1.65 / t.d ** (1 / 6)
    a = acc(h)
    lo = hi = None
    for _ in range(int(cfg.tune.max_expand)):
        if a > target_acc:
            lo = h
            if hi is not None:
                break
            h *= 2.0
        else:
            hi = h
            if lo is not None:
                break
            h /= 2.0
        a = acc(h)
    else:
        raise RuntimeError(f"{cid}: could not bracket acceptance {target_acc}; trace {trace}")
    if lo is None or hi is None:
        raise RuntimeError(f"{cid}: bracketing failed; trace {trace}")
    best = min(trace, key=lambda r: abs(r["accept"] - target_acc))
    for _ in range(int(cfg.tune.max_bisect)):
        if abs(best["accept"] - target_acc) < float(cfg.tune.inner_tol):
            break
        mid = math.sqrt(lo * hi)
        a = acc(mid)
        if a > target_acc:
            lo = mid
        else:
            hi = mid
        best = min(trace, key=lambda r: abs(r["accept"] - target_acc))
    if abs(best["accept"] - target_acc) >= float(cfg.tune.inner_tol):
        raise RuntimeError(f"{cid}: bisection did not reach inner_tol; trace {trace}")
    h_star = best["h"]
    v_acc, v_se = _acceptance(
        t, kind, h_star, scales, J, n_steps, seed_for(cid, "pilot-tune-verify", 0), device
    )
    if abs(v_acc - target_acc) > float(cfg.tune.tol):
        raise RuntimeError(f"{cid}: verification acceptance {v_acc} outside +-{cfg.tune.tol}")
    # coarse acceptance curve around h* (diagnostic, common random numbers)
    curve = []
    for mult in cfg.tune.curve_mults:
        a_c, _ = _acceptance(t, kind, h_star * mult, scales, J, n_steps, seed, device)
        curve.append({"h": h_star * mult, "mult": mult, "accept": a_c})
    return {
        "config_id": cid,
        "kind": kind,
        "h": h_star,
        "accept_target": target_acc,
        "accept_tune": best["accept"],
        "accept_verify": v_acc,
        "accept_verify_se": v_se,
        "tune_J": J,
        "tune_n_steps": n_steps,
        "start": "exact target draws",
        "bisection_trace": trace,
        "accept_curve": curve,
    }


def stage_tmax(
    t: Target, kind: str, scales, ell, h, emb, cfg, device, stage: str = "pilot-tmax"
) -> dict:
    cid = config_id(t.target_id, kind)
    J, eps, cap = int(cfg.tmax.J), float(cfg.tmax.eps), int(cfg.tmax.cap)
    k = ScaledRBF(scales, ell)
    g = torch_gen(seed_for(cid, stage, 0), device)
    x0 = t.sample_init(J, eps, g)
    grid = [0] + [2**i for i in range(int(math.log2(cap)) + 1)]
    smp = make_sampler(kind, t, h, scales)
    curve = {}

    def on_ckpt(tt, x):
        mu = emb(x)
        v = mmd2_from_embedding(k, x, mu, emb.norm2(), unbiased=False)
        u = mmd2_from_embedding(k, x, mu, emb.norm2(), unbiased=True)
        if v < -1e-12:
            raise FloatingPointError(f"negative V-statistic MMD^2 {v}")
        curve[tt] = {
            "mmd_v": math.sqrt(max(v, 0.0)),
            "mmd2_v": v,
            "mmd2_u": u,
            "p_A": float(t.event_A(x).double().mean().item()),
        }

    t0 = time.time()
    out = smp.run(x0, grid, g, store=False, on_checkpoint=on_ckpt)
    _sync(device)
    wall = time.time() - t0
    thr = float(cfg.tmax.threshold)
    below = [tt for tt in grid if tt > 0 and curve[tt]["mmd_v"] < thr]
    t_max = below[0] if below else cap
    floor = math.sqrt(max(1.0 - emb.norm2(), 0.0) / J)
    return {
        "config_id": cid,
        "J": J,
        "eps": eps,
        "rng_stage": stage,
        "threshold": thr,
        "cap": cap,
        "t_max": t_max,
        "cap_binds": not below,
        "t_grid": [0] + [2**i for i in range(int(math.log2(t_max)) + 1)],
        "mmd_curve": {str(tt): v for tt, v in curve.items()},
        "vstat_null_floor_approx": floor,
        "vstat_floor_note": "E[MMD_V^2] at pi_t = pi is (1 - ||mu||^2)/J; MMD_V floor ~ sqrt of it",
        "accept_rate": out["accept_rate"],
        "accept_by_interval": {str(k_): v for k_, v in out["accept_by_interval"].items()},
        "wall_s": wall,
        "counters": out["counters"],
    }


def perm_mmd_test(k, X, Y, n_perm, gen):
    """Two-sample MMD permutation test (equal sizes; V and U statistics give identical p)."""
    n = X.shape[0]
    assert Y.shape[0] == n
    Z = torch.cat([X, Y])
    K = k(Z, Z)
    s = torch.cat([torch.ones(n, dtype=F64), -torch.ones(n, dtype=F64)]).to(Z.device) / n
    t_obs = float(s @ K @ s)
    perms = torch.stack(
        [torch.randperm(2 * n, generator=gen, device=gen.device) for _ in range(n_perm)]
    )
    S = s[perms].T  # (2n, n_perm)
    t_perm = ((K @ S) * S).sum(0)
    p = (1 + int((t_perm >= t_obs).sum().item())) / (1 + n_perm)
    return {"mmd2_v": t_obs, "p_value": p, "n_per_side": n, "n_perm": n_perm}


def stage_witness_stationarity(t: Target, kind, scales, ell, h, cfg, device) -> dict:
    cid = config_id(t.target_id, kind)
    n = int(cfg.witness.n_mmd)
    k = ScaledRBF(scales, ell)
    times = [int(x) for x in cfg.witness.stationarity_times]
    g = torch_gen(seed_for(cid, "witness-stat-chain", 0), device)
    out = make_sampler(kind, t, h, scales).run(t.sample(n, g), times, g)
    gf = torch_gen(seed_for(cid, "witness-stat-fresh", 0), device)
    gp = torch_gen(seed_for(cid, "witness-stat-perm", 0), device)
    res = {}
    for tt in times:
        fresh = t.sample(n, gf)
        res[str(tt)] = perm_mmd_test(k, out["states"][tt], fresh, int(cfg.witness.n_perm), gp)
    return res


def stage_witness_embedding(t: Target, scales, ell, cfg, device) -> dict:
    """Compare embedding values and norms with Monte Carlo and quadrature.

    Report z-scores only where the Monte Carlo relative standard error is below
    the configured threshold. The probes share draws, so their z-scores are
    correlated. QUADPACK checks far points and probes unresolved by Monte Carlo.
    """
    tid = t.target_id
    n = int(cfg.witness.mc_draws)
    res = {}
    for name, mult in (("ell", 1.0), ("ell_half", 0.5), ("ell_double", 2.0)):
        k = ScaledRBF(scales, ell * mult)
        emb = rbf_mean_embedding(t, k)
        g = torch_gen(seed_for(tid, f"witness-emb-{name}", 0), device)
        draws = t.sample(8, g)
        probes = torch.cat(
            [
                draws,
                t.sample_init(4, 0.8, g),
                torch.zeros(1, t.d, dtype=F64, device=device),
                3.0 * draws[:4],
            ]
        )
        kinds = ["target"] * 8 + ["init"] * 4 + ["origin"] + ["far"] * 4
        mc = mc_mean_embedding(k, t, probes, n, g, chunk=1 << 20)
        exact = emb(probes).cpu().numpy()
        est, se = mc["estimate"].cpu().numpy(), mc["se"].cpu().numpy()
        resolved = (se > 0) & (se < float(cfg.witness.mc_resolved_rel_se) * np.abs(est))
        z = np.where(resolved, (exact - est) / np.where(se > 0, se, 1.0), np.nan)
        mn = mc_mean_embedding_norm2(k, t, n, g, chunk=1 << 21)
        row = {
            "ell": ell * mult,
            "method": emb.method,
            "n_mc_draws": n,
            "probe_kind": kinds,
            "mu_exact": exact,
            "mu_mc": est,
            "mu_mc_se": se,
            "mc_resolved": resolved.tolist(),
            "mu_z": [None if np.isnan(v) else float(v) for v in z],
            "n_resolved": int(resolved.sum()),
            "max_abs_z_mu_resolved": float(np.nanmax(np.abs(z))) if resolved.any() else None,
            "norm2_exact": emb.norm2(),
            "norm2_mc": mn["estimate"],
            "norm2_mc_se": mn["se"],
            "norm2_z": (emb.norm2() - mn["estimate"]) / mn["se"],
        }
        if emb.method == "quadrature":
            check = [i for i in range(len(kinds)) if kinds[i] == "far" or not resolved[i]]
            rel = {}
            for i in check:
                ref, _ = quadpack_mu(t, k, probes[i].cpu().numpy())
                rel[i] = abs(exact[i] - ref) / ref
            row["quadpack_rel_diff"] = {str(i): v for i, v in rel.items()}
            row["max_quadpack_rel_diff"] = float(max(rel.values())) if rel else None
        res[name] = row
    return res


def _time_steps(t, kind, h, scales, J, n_steps, device, seed, reps=3):
    """Median over reps of the per-step wall time of ``run`` (cached densities), after warm-up."""
    smp = make_sampler(kind, t, h, scales)
    g = torch_gen(seed, device)
    x0 = t.sample(J, g)
    smp.run(x0, [3], g, store=False)  # warm-up
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dts = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        smp.run(x0, [n_steps], g, store=False)
        _sync(device)
        dts.append((time.perf_counter() - t0) / n_steps)
    peak = torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None
    return float(np.median(dts)), dts, peak


def _time_kernel_sum(k, X, device, compute_dtype=F64, reps=2):
    kernel_sum(k, X[:256], X[:256], compute_dtype=compute_dtype)
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        s = kernel_sum(k, X, X, compute_dtype=compute_dtype).item()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return min(ts), s


def stage_throughput(t, kind, scales, ell, h, cfg, device, with_kernel_sum: bool) -> dict:
    cid = config_id(t.target_id, kind)
    res = {"sampler": [], "kernel_sum": []}
    devices = [device, "cpu"] if torch.device(device).type == "cuda" else ["cpu"]
    n_thr = int(cfg.cpu_threads)
    for J in cfg.throughput.J_list:
        for dev, threads in [(d_, n_thr) for d_ in devices] + [("cpu", 1)]:
            torch.set_num_threads(threads)
            n_steps = int(
                cfg.throughput.n_steps_gpu if dev != "cpu" else cfg.throughput.n_steps_cpu
            )
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            dt, dts, peak = _time_steps(
                t, kind, h, scales, int(J), n_steps, dev, seed_for(cid, "throughput", 0)
            )
            rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            res["sampler"].append(
                {
                    "J": int(J),
                    "device": "gpu" if dev != "cpu" else "cpu",
                    "ms_per_step": dt * 1e3,
                    "ms_per_step_reps": [x * 1e3 for x in dts],
                    "chain_steps_per_s": J / dt,
                    "n_steps_timed": n_steps,
                    "peak_gpu_mem_bytes": peak,
                    "process_maxrss_kb_before_after": [rss0, rss1],
                    "cpu_threads": threads if dev == "cpu" else None,
                }
            )
    torch.set_num_threads(n_thr)
    if with_kernel_sum:
        k = ScaledRBF(scales, ell)
        for J in cfg.throughput.J_list:
            for dev in devices:
                X = t.sample(int(J), torch_gen(seed_for(cid, "throughput-ks", int(J)), dev))
                dt64, s64 = _time_kernel_sum(k, X, dev)
                row = {"J": int(J), "device": "gpu" if dev != "cpu" else "cpu", "fp64_s": dt64}
                if dev != "cpu":
                    dt32, s32 = _time_kernel_sum(k, X, dev, compute_dtype=torch.float32)
                    row.update({"fp32_s": dt32, "fp32_rel_sum_err": abs(s32 - s64) / s64})
                res["kernel_sum"].append(row)
    return res


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_t(curve: dict, key: str) -> dict:
    """{int t: curve[t][key]} in increasing t (JSON keys are strings)."""
    return {int(t): curve[t][key] for t in sorted(curve, key=int)}


def _compact_throughput(thr: dict) -> dict:
    """Per-step sampler cost (ms, median of 3 timed repeats) and kernel-sum cost (s) by J."""
    out: dict = {"sampler_ms_per_step": {}, "peak_gpu_mem_MB": {}}
    for r in thr["sampler"]:
        key = "gpu" if r["device"] == "gpu" else f"cpu_{r['cpu_threads']}thr"
        out["sampler_ms_per_step"].setdefault(key, {})[int(r["J"])] = round(r["ms_per_step"], 4)
        if r["device"] == "gpu":
            out["peak_gpu_mem_MB"][int(r["J"])] = round(r["peak_gpu_mem_bytes"] / 2**20, 1)
    if thr["kernel_sum"]:
        ks: dict = {}
        for r in thr["kernel_sum"]:
            ks.setdefault(f"{r['device']}_fp64", {})[int(r["J"])] = round(r["fp64_s"], 5)
            if "fp32_s" in r:
                ks.setdefault("gpu_fp32", {})[int(r["J"])] = round(r["fp32_s"], 5)
                ks.setdefault("gpu_fp32_rel_sum_err", {})[int(r["J"])] = r["fp32_rel_sum_err"]
        out["kernel_sum_vstat_s"] = ks
    return out


def freeze(cfg) -> pathlib.Path:
    runs = [output_root() / STUDY / "pilot" / r for r in cfg.freeze_runs]
    targets, configs = {}, {}
    for run in runs:
        status = json.loads((run / "RUN-STATUS.json").read_text())
        if status["status"] != "COMPLETED":
            raise RuntimeError(f"{run} is {status['status']}, not COMPLETED")
        tier = yaml.safe_load((run / "config.yaml").read_text())["tier"]
        for tf in sorted((run / "targets").glob("*.json")):
            d = json.loads(tf.read_text())
            tid = d["target"]["target_id"]
            assert tid not in targets, f"duplicate target {tid}"
            tgt = d["target"]
            entry = {
                "tier": tier,
                "family": tgt["family"],
                "d": tgt["d"],
                "params": {
                    **{k: tgt[k] for k in ("cond", "sep", "w", "nu", "sigma_y") if k in tgt},
                    **(
                        {"event_dir": tgt["event_dir"]}
                        if tgt.get("event_dir", "narrow") != "narrow"
                        else {}
                    ),
                },
                "event": tgt["event"],
                "prob_A_abs_error": tgt["prob_A_abs_error"],
                "scales": d["scales"],
                "scale_rule": d["scale_rule"],
                "scale_draws": d["scale_draws"],
                "ell": d["ell"],
                "ell_half": 0.5 * d["ell"],
                "ell_double": 2.0 * d["ell"],
                "ell_rule": d["ell_rule"],
                "bw_draws": d["bw_draws"],
                "rbf_mu_norm2": {k: v["norm2"] for k, v in d["rbf_embedding"].items()},
                "rbf_embedding_method": {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if kk
                        in (
                            "method",
                            "rule",
                            "order",
                            "rel_change_mu_last_doubling",
                            "rel_change_norm2_last_doubling",
                            "abs_error",
                        )
                    }
                    for k, v in d["rbf_embedding"].items()
                },
            }
            for k in ("c_star", "r_med"):
                if k in tgt:
                    entry[k] = tgt[k]
            if tgt["family"] in ("gauss", "mixture"):
                entry["rotation_key"] = rotation_key(tgt["family"], tgt["d"])
                entry["rotation_seed"] = tgt["rotation_seed"]
            targets[tid] = entry
        for cf in sorted((run / "configs").glob("*.json")):
            d = json.loads(cf.read_text())
            cid = d["tune"]["config_id"]
            assert cid not in configs, f"duplicate config {cid}"
            tm = d["tmax"]
            configs[cid] = {
                "tier": tier,
                "target_id": cid.split("__")[0],
                "sampler": d["tune"]["kind"],
                "preconditioner": "S = diag(scales) of the target entry (frozen, non-adaptive)",
                "h": d["tune"]["h"],
                "accept_target": d["tune"]["accept_target"],
                "accept_tune": d["tune"]["accept_tune"],
                "accept_verify": d["tune"]["accept_verify"],
                "accept_verify_se": d["tune"]["accept_verify_se"],
                "tune_J": d["tune"]["tune_J"],
                "tune_n_steps": d["tune"]["tune_n_steps"],
                "t_max": tm["t_max"],
                "t_max_cap_binds": tm["cap_binds"],
                "t_grid": tm["t_grid"],
                "pilot_mmd_v": _by_t(tm["mmd_curve"], "mmd_v"),
                "pilot_mmd2_u": _by_t(tm["mmd_curve"], "mmd2_u"),
                "pilot_p_A": _by_t(tm["mmd_curve"], "p_A"),
                "pilot_vstat_floor": tm["vstat_null_floor_approx"],
                "pilot_J": tm["J"],
                "pilot_eps": tm["eps"],
                "throughput": _compact_throughput(d["throughput"]),
            }
            if "inherited_from" in d:
                configs[cid]["t_max_rng_stage"] = tm["rng_stage"]
    doc = {
        "version": "frozen_suite_v1",
        "rotations": {
            "path": str(ROTATIONS_PATH.relative_to(REPO)),
            "sha256": _sha256(ROTATIONS_PATH),
            "note": "Haar SO(d) from seed_for(f'{family}-d{d}', 'rotation', 0); load the saved array",
        },
        "conventions": {
            "kernel_rbf": "k(x,y) = exp(-||(x-y)/s||^2 / (2 ell^2)), s = scales, kappa2 = 1",
            "kernel_matern32": "k(x,y) = (1 + sqrt3 r/ell) exp(-sqrt3 r/ell), r = ||(x-y)/s||",
            "sampler_rwm": "y = x + h S xi",
            "sampler_mala": "y = x + (h^2/2) S^2 grad log pi(x) + h S xi, exact MH correction",
            "init_law": "d pi0/d pi = 1 + eps (2 1_A - 1), pi(A) = 1/2, R_init = eps",
            "t_max_rule": "first power of two with pilot V-statistic MMD < 0.01 (J=2^14, eps=0.8), cap 8192",
            "config_id": "<target_id>__<sampler>",
        },
        "targets": targets,
        "configs": configs,
    }
    path = REPO / "configs" / "paperexp" / "frozen_suite_v1.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    return path


def stage_event_witness(t: Target, cfg, device) -> dict:
    """pi(A) = 1/2 and the init law P0(A) = (1+eps)/2 by Monte Carlo (n = event_witness.n)."""
    n = int(cfg.event_variant.n_event_mc)
    g = torch_gen(seed_for(t.target_id, "witness-event", 0), device)
    pa = float(t.event_A(t.sample(n, g)).double().mean().item())
    out = {"n": n, "pi_A_mc": pa, "pi_A_se": math.sqrt(pa * (1 - pa) / n), "init": {}}
    for eps in (0.8, 0.2):
        p0 = float(t.event_A(t.sample_init(n, eps, g)).double().mean().item())
        se = math.sqrt(p0 * (1 - p0) / n)
        out["init"][str(eps)] = {
            "P0_A_mc": p0,
            "P0_A_exact": (1 + eps) / 2,
            "z": (p0 - (1 + eps) / 2) / se,
            "chi2_hat": eps * (2 * p0 - 1),
            "chi2_exact": eps**2,
        }
    return out


def run_event_variant(cfg) -> None:
    """Calibrate the Gaussian initialization along the widest eigendirection.

    The target and sampler are unchanged, so their calibration is reused.
    Recompute t_max with a separate random stream and check the initial law.
    """
    from mcmc.paperexp.targets import load_frozen_suite

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA is unavailable")
    torch.set_num_threads(int(cfg.cpu_threads))
    if cfg.tier != "stress":
        raise ValueError(f"the event variant is frozen with tier=stress; got tier={cfg.tier!r}")
    ev = cfg.event_variant
    base_tid = ev.base_target
    src = output_root() / STUDY / "pilot" / ev.source_run
    suite = load_frozen_suite()
    base_spec = suite.target_spec(base_tid)
    t = make_target(
        base_spec["family"], base_spec["d"], **dict(base_spec["params"]), event_dir="wide"
    )
    tid = t.target_id
    kinds = list(cfg.samplers)
    run = RunDir.create(
        STUDY,
        cfg.phase,
        cfg.run_id,
        OmegaConf.to_yaml(cfg),
        requested_units=1 + len(kinds),
        allow_dirty=True,
    )
    (run.path / "targets").mkdir()
    (run.path / "configs").mkdir()
    try:
        tinfo = json.loads((src / "targets" / f"{base_tid}.json").read_text())
        assert tinfo["scales"] == base_spec["scales"] and tinfo["ell"] == base_spec["ell"]
        scales, ell = np.array(tinfo["scales"]), tinfo["ell"]
        emb = rbf_mean_embedding(t, ScaledRBF(scales, ell))
        assert abs(emb.norm2() - base_spec["rbf_mu_norm2"]["ell"]) <= 1e-12 * emb.norm2()
        tinfo = dict(tinfo)
        tinfo["target"] = {k: v for k, v in t.to_dict().items() if k != "rotation"}
        tinfo["inherited_from"] = {
            "target_id": base_tid,
            "fields": ["scales", "ell", "rbf_embedding", "embedding_witness"],
            "reason": "same target distribution; only the event A differs",
        }
        tinfo["event_witness"] = stage_event_witness(t, cfg, device)
        write_json(run.path / "targets" / f"{tid}.json", tinfo)
        run.update(completed_units=1)
        log.info(f"[{tid}] event witness {tinfo['event_witness']}")
        for i, kind in enumerate(kinds):
            base_cid, cid = config_id(base_tid, kind), config_id(tid, kind)
            c = json.loads((src / "configs" / f"{base_cid}.json").read_text())
            h = c["tune"]["h"]
            assert h == suite.config(base_cid)["h"], (h, suite.config(base_cid)["h"])
            tune = {**c["tune"], "config_id": cid}
            tmax = stage_tmax(t, kind, scales, ell, h, emb, cfg, device, stage=ev.tmax_stage)
            log.info(f"[{cid}] t_max={tmax['t_max']} cap_binds={tmax['cap_binds']}")
            payload = {
                "tune": tune,
                "tmax": tmax,
                "witness_stationarity": c["witness_stationarity"],
                "throughput": c["throughput"],
                "inherited_from": {
                    "config_id": base_cid,
                    "fields": ["tune (h, acceptance)", "witness_stationarity", "throughput"],
                    "reason": "identical transition kernel (target, h, S); only the event A differs",
                },
            }
            write_json(run.path / "configs" / f"{cid}.json", payload)
            run.update(completed_units=2 + i)
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:2000])
        raise
    run.finish("COMPLETED")
    log.info(f"event variant COMPLETED at {run.path}")


def run_pilot(cfg) -> None:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA is unavailable")
    torch.set_num_threads(int(cfg.cpu_threads))
    specs = tier_targets(cfg.tier)
    if cfg.only_targets:
        specs = [
            s for s in specs if make_target(s[0], s[1], **dict(s[2])).target_id in cfg.only_targets
        ]
    kinds = list(cfg.samplers)
    units = len(specs) * (1 + len(kinds))
    config_text = OmegaConf.to_yaml(cfg)
    if cfg.phase not in ("smoke", "pilot"):
        raise ValueError(f"phase must be smoke or pilot, got {cfg.phase!r}")
    if cfg.resume:
        run = RunDir.resume(STUDY, cfg.phase, cfg.run_id)
    else:
        run = RunDir.create(
            STUDY, cfg.phase, cfg.run_id, config_text, requested_units=units, allow_dirty=True
        )
    (run.path / "targets").mkdir(exist_ok=True)
    (run.path / "configs").mkdir(exist_ok=True)
    done = 0
    t_all = time.time()
    try:
        for fam, d, params in specs:
            t = make_target(fam, d, **dict(params))
            tid = t.target_id
            tpath = run.path / "targets" / f"{tid}.json"
            if tpath.exists():
                tinfo = json.loads(tpath.read_text())
            else:
                t0 = time.time()
                tinfo = stage_scale(t, cfg, device)
                tinfo["embedding_witness"] = stage_witness_embedding(
                    t, np.array(tinfo["scales"]), tinfo["ell"], cfg, device
                )
                tinfo["wall_s"] = time.time() - t0
                write_json(tpath, tinfo)
                log.info(
                    f"[{tid}] scales/ell/embedding done: ell={tinfo['ell']:.4f} ({tinfo['wall_s']:.1f}s)"
                )
            done += 1
            run.update(completed_units=done)
            scales, ell = np.array(tinfo["scales"]), tinfo["ell"]
            emb = rbf_mean_embedding(t, ScaledRBF(scales, ell))
            for i, kind in enumerate(kinds):
                cid = config_id(tid, kind)
                cpath = run.path / "configs" / f"{cid}.json"
                if cpath.exists():
                    done += 1
                    run.update(completed_units=done)
                    continue
                t0 = time.time()
                tune = stage_tune(t, kind, scales, cfg, device)
                log.info(f"[{cid}] h={tune['h']:.5f} acc_verify={tune['accept_verify']:.4f}")
                tmax = stage_tmax(t, kind, scales, ell, tune["h"], emb, cfg, device)
                log.info(f"[{cid}] t_max={tmax['t_max']} cap_binds={tmax['cap_binds']}")
                wit = stage_witness_stationarity(t, kind, scales, ell, tune["h"], cfg, device)
                log.info(f"[{cid}] stationarity p={ {k: v['p_value'] for k, v in wit.items()} }")
                thr = stage_throughput(
                    t, kind, scales, ell, tune["h"], cfg, device, with_kernel_sum=(i == 0)
                )
                payload = {
                    "tune": tune,
                    "tmax": tmax,
                    "witness_stationarity": wit,
                    "throughput": thr,
                    "wall_s": time.time() - t0,
                }
                write_json(cpath, payload)
                done += 1
                run.update(completed_units=done)
                log.info(f"[{cid}] done in {payload['wall_s']:.1f}s")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}"[:2000])
        raise
    run.update(note=f"total wall {time.time() - t_all:.1f}s")
    run.finish("COMPLETED")
    log.info(f"pilot COMPLETED at {run.path}")


@hydra.main(
    config_path="../../../configs/experiment",
    config_name="exp_suite_pilot_freeze",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.stage == "pilot":
        if not cfg.run_id:
            raise ValueError("run_id is required")
        run_pilot(cfg)
    elif cfg.stage == "event_variant":
        if not cfg.run_id:
            raise ValueError("run_id is required")
        run_event_variant(cfg)
    elif cfg.stage == "freeze":
        path = freeze(cfg)
        log.info(f"frozen file written: {path}")
    else:
        raise ValueError(f"unknown stage {cfg.stage!r}")


if __name__ == "__main__":
    main()
