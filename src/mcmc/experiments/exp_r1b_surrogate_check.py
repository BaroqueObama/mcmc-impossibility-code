"""Check the Matern-3/2 scale-mixture approximation for the two-dimensional funnel.

Compare kernel errors and centered covariance spectra using the same landmarks
and quadrature for both kernels.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mcmc.experiments.exp_r1b_constructions_suite import (
    _lspec,
    _qspec,
    build_target,
    frozen_target_params,
    write_json,
)
from mcmc.paperexp import r1_suite as r1
from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import ScaledMatern32
from mcmc.paperexp.r1_surrogate import TabulatedRadialKernel
from mcmc.paperexp.rng import seed_for, torch_gen
from mcmc.paperexp.runio import RunDir, utc_now

log = logging.getLogger(__name__)
F64 = torch.float64
TARGET = "funnel-d2-sy3"


def _mixture(fm, rule, scales, ell, label):
    lam, w = r1.matern32_scale_mixture(
        float(rule.h), float(fm.taper.log_lam0), float(fm.taper.tau),
        float(rule.log_lam_max), float(rule.offset),
    )  # fmt: skip
    return r1.ScaledGaussMixture(scales, ell, lam, w, label=label)


def _moments(kernel, Z: torch.Tensor, X: torch.Tensor, w: torch.Tensor, block_bytes=1 << 29):
    """mu_a = sum_n w_n k(z_a, x_n), P_ab = sum_n w_n k(z_a, x_n) k(z_b, x_n)."""
    L = Z.shape[0]
    mu = torch.zeros(L, dtype=F64)
    P = torch.zeros((L, L), dtype=F64)
    cols = max(1, block_bytes // (8 * 4 * L))
    for j0 in range(0, X.shape[0], cols):
        Kb = kernel(Z, X[j0 : j0 + cols])
        wb = w[j0 : j0 + cols]
        mu += Kb @ wb
        P += (Kb * wb[None, :]) @ Kb.T
    return mu, 0.5 * (P + P.T)


def _spectrum(kernel, Z, mu, P, rel_cutoff):
    feats = NystromFeatures(kernel, Z, rel_cutoff=rel_cutoff)
    fbar = feats.B.T @ mu
    C = feats.B.T @ P @ feats.B - torch.outer(fbar, fbar)
    ev = torch.linalg.eigvalsh(0.5 * (C + C.T)).flip(0)
    return ev.numpy(), {"rank": int(feats.rank), "kzz_eig_min": feats.min_eig}


def run_kernel_error(cfg, out_dir: pathlib.Path) -> dict:
    fm = cfg.constructions.funnel_matern
    fp = frozen_target_params(TARGET, cfg)
    ks = _mixture(fm, fm.kernel_rule, np.asarray(fp["scales"]), fp["ell"], "kernel rule")
    r = np.concatenate([np.linspace(0.0, 0.5, 50001), np.linspace(0.5, 40.0, 39501)])
    km = r1.matern32(r)
    kr = r1.mixture_kernel_values(ks.lam, ks.w, r)
    gap = km - kr
    t = build_target(TARGET)
    sc = cfg.surrogate_check
    gen = torch_gen(seed_for(TARGET, f"r1b-surrogate-pairs-{cfg.seed_tag}", 0), torch.device("cpu"))
    n = int(sc.n_pair_mc)
    X, Y = t.sample(n, gen), t.sample(n, gen)
    rp = torch.sqrt((((X - Y) / torch.as_tensor(ks.scales * ks.ell)) ** 2).sum(1)).numpy()
    dp = np.abs(r1.matern32(rp) - r1.mixture_kernel_values(ks.lam, ks.w, rp))
    r_med = float(np.median(rp))
    k1m, k1s = (
        r1.matern32(np.array([1.0]))[0],
        r1.mixture_kernel_values(ks.lam, ks.w, np.array([1.0]))[0],
    )
    kmm = r1.matern32(np.array([r_med]))[0]
    kms = r1.mixture_kernel_values(ks.lam, ks.w, np.array([r_med]))[0]
    res = {
        "unit": "kernel_error",
        "label": "Matern-3/2 surrogate (tapered scale mixture) vs Matern-3/2",
        "kernel": ks.to_dict(),
        "sup_abs": float(np.abs(gap).max()),
        "r_at_sup": float(r[np.abs(gap).argmax()]),
        "gap_min": float(gap.min()),
        "r_where_abs_gap_below": {
            f"{e:g}": float(r[np.flatnonzero(np.abs(gap) >= e).max()]) for e in (1e-6, 1e-9, 1e-12)
        },
        "at_r1": {
            "matern": float(k1m),
            "surrogate": float(k1s),
            "abs": float(k1m - k1s),
            "rel": float((k1m - k1s) / k1m),
        },  # fmt: skip
        "pairs": {
            "n": n,
            "median_scaled_distance": r_med,
            "at_median": {
                "matern": float(kmm),
                "surrogate": float(kms),
                "rel": float((kmm - kms) / kmm),
            },
            "abs_gap_mean": float(dp.mean()),
            "abs_gap_q99": float(np.quantile(dp, 0.99)),
            "abs_gap_max": float(dp.max()),
            "frac_pairs_gap_gt_1e-6": float((dp > 1e-6).mean()),
        },  # fmt: skip
    }
    write_json(out_dir / "kernel_error.json", res)
    return res


def run_eig_level(lvl: str, cfg, out_dir: pathlib.Path) -> dict:
    cc, fm = cfg.constructions, cfg.constructions.funnel_matern
    t0 = time.time()
    t = build_target(TARGET)
    fp = frozen_target_params(TARGET, cfg)
    scales = np.asarray(fp["scales"])
    ks_mix = _mixture(fm, fm.kernel_rule, scales, fp["ell"], "kernel rule")
    sr = cfg.constructions.surrogate
    ks = TabulatedRadialKernel(scales, fp["ell"], ks_mix.lam, ks_mix.w, float(sr.table_r_max),
                               int(sr.table_n), label="kernel rule, tabulated")  # fmt: skip
    km = ScaledMatern32(scales, fp["ell"])
    mix = _mixture(fm, fm[f"rule_{lvl}"], scales, fp["ell"], f"{lvl} rule")
    qs = _qspec(cc[f"quad_{lvl}"])
    node = r1.NodeSpec(**{k: (int(v) if k.startswith("q_") else float(v))
                          for k, v in fm[f"nodes_{lvl}"].items()})  # fmt: skip
    gi = r1.FunnelMixtureIntegrator(t, mix, 1, 1, qs, node, workers=1)
    cut = float(fm.rel_cutoff)
    Z, linfo = r1.make_landmarks(t, ks, gi, _lspec(fm[f"landmarks_{lvl}"], cut))
    X, w, ninfo = gi.second_moment_nodes(Z)
    zsha = hashlib.sha256(Z.numpy().tobytes()).hexdigest()
    out = {"unit": f"eigs_{lvl}", "level": lvl, "landmarks": linfo, "landmarks_sha256": zsha,
           "surrogate_table_max_abs_err": ks.max_abs_err_vs(),
           "nodes": ninfo, "rel_cutoff": cut}  # fmt: skip
    lam = {}
    for name, k in (("surrogate", ks), ("matern32", km)):
        tt = time.time()
        mu, P = _moments(k, Z, X, w)
        ev, info = _spectrum(k, Z, mu, P, cut)
        lam[name] = ev
        out[name] = {"lam_top": ev[:48].tolist(), "trace": float(ev.sum()), **info,
                     "wall_s": time.time() - tt}  # fmt: skip
        log.info(
            "%s %s: lam_1=%.4e lam_33=%.4e (%.1fs)", lvl, name, ev[0], ev[32], time.time() - tt
        )
    M = int(cfg.surrogate_check.m_max)
    rel = lam["surrogate"][: M + 1] / lam["matern32"][: M + 1] - 1.0
    out["rel_err_surrogate_vs_matern"] = rel.tolist()  # index i = lambda_{i+1}
    out["max_abs_rel_err_lam1_to_lam33"] = float(np.abs(rel[:33]).max())
    for thr in cfg.surrogate_check.separation_thresholds:
        bad = np.flatnonzero(np.abs(rel) > float(thr))
        out[f"first_index_rel_err_gt_{thr}"] = int(bad[0] + 1) if bad.size else None
    out["wall_s"] = time.time() - t0
    out["finished_utc"] = utc_now()
    write_json(out_dir / f"eigs_{lvl}.json", out)
    return out


def run_schur_bound(cfg, out_dir: pathlib.Path) -> dict:
    """Estimate the Schur sensitivity of the kernel approximation.

    For nonnegative, positive semidefinite dk = k_Matern - k_surrogate,
    0 <= lambda_i(Matern) - lambda_i(surrogate) <= sup_x int dk(x, y) dpi(y).
    The finite search over x and Monte Carlo integral estimate this supremum;
    they do not certify the eigenvalue bound.
    """
    sc = cfg.surrogate_check
    fm, sr = cfg.constructions.funnel_matern, cfg.constructions.surrogate
    fp = frozen_target_params(TARGET, cfg)
    scales = np.asarray(fp["scales"])
    mix = _mixture(fm, fm.kernel_rule, scales, fp["ell"], "kernel rule")
    ks = TabulatedRadialKernel(scales, fp["ell"], mix.lam, mix.w, float(sr.table_r_max),
                               int(sr.table_n), label="kernel rule, tabulated")  # fmt: skip
    km = ScaledMatern32(scales, fp["ell"])
    t = build_target(TARGET)
    gen = torch_gen(seed_for(TARGET, f"r1b-surrogate-schur-{cfg.seed_tag}", 0), torch.device("cpu"))
    Y = t.sample(int(sc.schur_n_y), gen)
    Xs = t.sample(int(sc.schur_n_x), gen)
    yg = torch.linspace(-6.0 * 3.0, 6.0 * 3.0, int(sc.schur_n_axis), dtype=F64)
    Xa = torch.stack([yg, torch.zeros_like(yg)], 1)
    X = torch.cat([Xs, Xa])
    est = torch.zeros(X.shape[0], dtype=F64)
    cols = max(1, (1 << 28) // (8 * 6 * X.shape[0]))
    for j0 in range(0, Y.shape[0], cols):
        Yb = Y[j0 : j0 + cols]
        est += (km(X, Yb) - ks(X, Yb)).sum(1)
    est /= Y.shape[0]
    i = int(est.argmax())
    res = {
        "unit": "schur_bound",
        "label": "Schur sensitivity estimated over finitely many x using Monte Carlo integrals. "
        "The true supremum bounds the eigenvalue difference; this estimate does not certify it.",
        "n_y": int(Y.shape[0]),
        "n_x_sampled": int(Xs.shape[0]),
        "n_x_axis": int(Xa.shape[0]),
        "sup_estimate": float(est[i]),
        "argmax_x": X[i].tolist(),
        "min_estimate": float(est.min()),
        "mean_over_sampled_x": float(est[: Xs.shape[0]].mean()),
    }
    write_json(out_dir / "schur_bound.json", res)
    return res


@hydra.main(config_path="../../../configs/experiment", config_name="exp_r1b_surrogate_check",
            version_base=None)  # fmt: skip
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(int(cfg.threads))
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    unit_list = [u for u in cfg.surrogate_check.units if not cfg.only_units or u in cfg.only_units]
    try:
        run = RunDir.create(cfg.study, str(cfg.phase), str(cfg.run_id), config_text,
                            requested_units=len(unit_list), allow_dirty=cfg.phase == "smoke",
                            extra_provenance={"units": unit_list})  # fmt: skip
    except FileExistsError:
        run = RunDir.resume(cfg.study, str(cfg.phase), str(cfg.run_id))
        if run.status()["config_sha256"] != hashlib.sha256(config_text.encode()).hexdigest():
            raise
    out_dir = run.path / "units"
    out_dir.mkdir(exist_ok=True)
    done = 0
    for u in unit_list:
        if (out_dir / f"{u}.json").exists():
            done += 1
            continue
        log.info("unit %s start", u)
        try:
            if u == "kernel_error":
                run_kernel_error(cfg, out_dir)
            elif u == "schur_bound":
                run_schur_bound(cfg, out_dir)
            elif u.startswith("eigs_"):
                run_eig_level(u.removeprefix("eigs_"), cfg, out_dir)
            else:
                raise ValueError(f"unknown unit {u!r}")
        except BaseException as exc:
            run.update(failed_units=run.status()["failed_units"] + [u])
            run.finish("FAILED", note=f"unit {u}: {type(exc).__name__}: {exc}")
            raise
        done += 1
        run.update(completed_units=done)
        log.info("unit %s done (%d/%d)", u, done, len(unit_list))
    run.finish("COMPLETED")
    log.info("run %s finished at %s (pid %d)", run.path, utc_now(), os.getpid())


if __name__ == "__main__":
    main()
