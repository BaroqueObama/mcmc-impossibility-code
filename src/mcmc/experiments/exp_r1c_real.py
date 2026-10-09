"""Residual constructions and fixed-law comparisons for Gaussian regressions.

Cellwise constructions are fitted with Gauss-Legendre quadrature and evaluated
with Owen's T and a finer quadrature. Projected residuals use the exact Gibbs laws.

Results are written under $MCMC_PAPER_OUT/r1_real/{phase}/{run_id}/.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import pathlib
import time

import hydra
import numpy as np
import scipy.special as sps
import torch
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import gaussian_linear as gl
from mcmc.paperexp import r1_real as rr
from mcmc.paperexp import r1_suite as r1
from mcmc.paperexp import regression_data as rd
from mcmc.paperexp.rng import np_gen, seed_for
from mcmc.paperexp.runio import RunDir, utc_now

log = logging.getLogger(__name__)
F64 = torch.float64


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: pathlib.Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _construct_eval(pc, pe, m, R, cc) -> dict:
    rows_c = rr.monitor_rows(pc, m)
    rows_e = rr.monitor_rows(pe, m)
    cs = r1.ConstructionSpec(
        float(R),
        int(cc.n_dir),
        float(cc.floor),
        int(cc.n_combo),
        rank_tol=float(cc.rank_tol),
        eig_subset=True,
    )
    c = r1.construct(pc.G, pc.masses, rows_c, cs)
    ev = r1.evaluate(c["a"], pe.masses, pe.G, rows_e)
    env = R * math.sqrt(pc.spec.eig[m])
    combo = None
    if c["combo"] is not None:
        cev = r1.evaluate(c["combo"]["a"], pe.masses, pe.G, rows_e)
        combo = {
            **{k: v for k, v in c["combo"].items() if k != "a"},
            **{k: v for k, v in cev.items() if k != "readings_eval"},
        }
    return {
        **{k: v for k, v in c.items() if k not in ("a", "candidates", "combo")},
        **ev,
        "combo": combo,
        "lam_next": float(pc.spec.eig[m]),
        "envelope": env,
        "ratio_residual_envelope": ev["residual_eval"] / env,
        "ratio_width_envelope": c["width_unconstrained_construction"] / env,
        "_a": c["a"],
    }


def _plane_cell_index(spec, axes, n, X: np.ndarray) -> np.ndarray:
    Zz = np.linalg.solve(spec.T, (X - spec.mu[None, :]).T).T
    e = sps.ndtri(np.arange(n + 1) / n)
    ka = np.searchsorted(e, Zz[:, axes[0]], side="right") - 1
    kb = np.searchsorted(e, Zz[:, axes[1]], side="right") - 1
    if (ka < 0).any() or (ka >= n).any() or (kb < 0).any() or (kb >= n).any():
        raise ValueError("point outside the plane grid")
    return ka * n + kb


def _mc_check(spec, axes, n, a, n_samples, n_batches, seed) -> dict:
    """Exact draws from dQ = (1 + h) dpi (rejection) and the U-statistic MMD^2 (exact mu_pi)."""
    gen = np_gen(seed)
    top = 1.0 + float(a.max())
    out, got = [], 0
    while got < n_samples:
        mm = int(math.ceil((n_samples - got) * top * 1.2)) + 256
        X = spec.mu[None, :] + gen.standard_normal((mm, spec.d)) @ spec.T.T
        hx = a[_plane_cell_index(spec, axes, n, X)]
        acc = gen.random(mm) * top < 1.0 + hx
        out.append(X[acc])
        got += int(acc.sum())
    X = np.concatenate(out)[:n_samples]
    k = gl.AnisoRBF(spec.scales, spec.ell)
    mu_at = gl.mean_embedding(k, spec.mu, spec.C, X)
    mmd2, se = r1.mc_mmd2_unbiased(
        k, torch.as_tensor(X, dtype=F64), torch.as_tensor(mu_at, dtype=F64),
        spec.info["mu_norm2_exact"], n_batches,
    )  # fmt: skip
    return {"mmd2_mc": mmd2, "mmd2_mc_se": se, "n_samples": int(n_samples)}


def run_part_a_unit(name: str, cfg: DictConfig, out_dir: pathlib.Path) -> dict:
    ca = cfg.part_a
    t0 = time.time()
    model = rd.frozen_model(name)
    if model.ell is None:
        raise ValueError(f"{name}: frozen ell missing")
    spec = rr.exact_spectrum(
        model.mu, model.C, model.kernel_scales, model.ell,
        rel_floor=float(ca.rel_floor), n_out=int(ca.n_out),
    )  # fmt: skip
    m_values = [int(m) for m in ca.m_values]
    res = {
        "unit": name,
        "label": "Gaussian posterior N(mu, C); exact spectrum (Mehler + centering); constructions are "
        "feasible examples (not maximizers) on 2-D planes; projected residuals of the exact "
        "Gibbs laws (not satisfied criteria)",
        "model": {
            "name": name, "d": spec.d, "n": model.n, "sigma2": model.sigma2, "ell": model.ell,
            "kernel_scales": model.kernel_scales, "checkpoints": model.checkpoints,
            "gibbs_slow_modulus": model.directions["slow"]["modulus"],
            "settings": [{k: v for k, v in s.items()} for s in model.settings],
            "provenance": model.provenance,
        },
        "spectrum": {**spec.info, "eig": spec.eig.tolist()},
        "eigengap_rel": {m: float((spec.eig[m - 1] - spec.eig[m]) / spec.eig[m - 1]) for m in m_values},
    }  # fmt: skip
    arrays = {"eig": spec.eig}
    qc, qe = ca.quad_construction, ca.quad_eval

    # Projected residuals of the exact Gibbs laws.
    tt = time.time()
    Z = np_gen(seed_for(f"r1c-{name}", f"landmarks-{cfg.seed_tag}", 0)).multivariate_normal(
        model.mu, model.C, size=int(ca.landmarks.M)
    )
    proj = rr.gibbs_projected_residuals(spec, model, Z, m_values, float(ca.landmarks.rtol))
    for row in proj["rows"]:  # registry cross-check of the closed-form MMD
        st = [s for s in model.settings if s["name"] == row["setting"]][0]
        row["mmd_registry"] = rd.mmd_shift(model, row["t"], st["shift"])
    res["projected"] = proj
    res["projected_wall_s"] = time.time() - tt
    log.info(
        "%s projected done (%.1fs), tau0=%.2e", name, time.time() - tt, proj["witness"]["tau_0"]
    )

    # Select planes using the construction quadrature.
    tt = time.time()
    pairs = list(itertools.combinations(range(spec.d), 2))
    sel_tab = {}
    for axes in pairs:
        pc = rr.plane_setup(
            spec, axes, int(ca.select_n), float(qc.h), int(qc.q), float(qc.cut), "gl"
        )
        for m in m_values:
            cs = r1.ConstructionSpec(1.0, 1, 0.0, 0, rank_tol=float(ca.rank_tol), eig_subset=True)
            c = r1.construct(pc.G, pc.masses, rr.monitor_rows(pc, m), cs)
            sel_tab[(axes, m)] = c["nu_top"][0] / spec.eig[m]
    selected = {m: max(pairs, key=lambda p, m=m: sel_tab[(p, m)]) for m in m_values}
    res["plane_selection"] = {
        "rule": f"argmax over all axis pairs of nu_top/lambda_{{m+1}} at n={ca.select_n} "
        "(construction quadrature only; R-independent)",
        "selected": {m: list(p) for m, p in selected.items()},
        "table": {f"{p[0]}-{p[1]}": {m: sel_tab[(p, m)] for m in m_values} for p in pairs},
        "wall_s": time.time() - tt,
    }
    planes = sorted({(0, 1)} | set(selected.values()))
    log.info("%s planes %s (%.1fs)", name, planes, time.time() - tt)

    # constructions + independent evaluation
    cons = []
    for axes in planes:
        for n in [int(v) for v in ca.n_grid]:
            tt = time.time()
            pc = rr.plane_setup(spec, axes, n, float(qc.h), int(qc.q), float(qc.cut), str(qc.gram))
            pe = rr.plane_setup(spec, axes, n, float(qe.h), int(qe.q), float(qe.cut), str(qe.gram))
            gdiff = float(np.abs(pc.G - pe.G).max())
            for m in m_values:
                for R in [float(v) for v in ca.R_values]:
                    row = _construct_eval(pc, pe, m, R, ca)
                    a = row.pop("_a")
                    row.update(plane=list(axes), n=n, K=n * n, m=m, R=R,
                               selected_for_m=(selected[m] == axes), G_max_abs_diff=gdiff)  # fmt: skip
                    cons.append(row)
                    arrays[f"a_p{axes[0]}{axes[1]}_n{n}_m{m}_R{R:g}"] = a
            log.info("%s plane %s n=%d (%.1fs) res/env R=%.1f: %s", name, axes, n, time.time() - tt,
                     float(ca.R_values[0]),
                     [f"{r['ratio_residual_envelope']:.3f}" for r in cons
                      if r["plane"] == list(axes) and r["n"] == n and r["R"] == float(ca.R_values[0])])  # fmt: skip
    res["constructions"] = cons

    # Monte Carlo check (exact draws from Q in d dimensions)
    mc = ca.mc
    nmax = max(int(v) for v in ca.n_grid)
    mcs = []
    for m in [int(v) for v in mc.m_values]:
        key = f"a_p01_n{nmax}_m{m}_R{float(mc.R):g}"
        ref = [r for r in cons if r["plane"] == [0, 1] and r["n"] == nmax and r["m"] == m
               and r["R"] == float(mc.R)][0]  # fmt: skip
        out = _mc_check(spec, (0, 1), nmax, arrays[key], int(mc.n_samples), int(mc.n_batches),
                        seed_for(f"r1c-{name}", f"mc-{cfg.seed_tag}", m))  # fmt: skip
        out.update(
            m=m,
            mmd2_eval=ref["mmd2_eval"],
            z=(out["mmd2_mc"] - ref["mmd2_eval"]) / out["mmd2_mc_se"],
        )
        mcs.append(out)
    res["mc_check"] = mcs
    res["wall_s"] = time.time() - t0
    res["finished_utc"] = utc_now()
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    write_json(out_dir / f"{name}.json", res)
    return res


def units(cfg: DictConfig) -> list[str]:
    if cfg.part != "A":
        raise ValueError(f"unknown part {cfg.part!r}")
    u = list(cfg.part_a.datasets)
    if cfg.only_units:
        missing = set(cfg.only_units) - set(u)
        if missing:
            raise ValueError(f"unknown units {missing}")
        u = [x for x in u if x in set(cfg.only_units)]
    return u


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_r1c_real", version_base=None
)
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(int(cfg.threads))
    if cfg.run_id is None:
        if cfg.phase != "smoke":
            raise ValueError("run_id is required outside smoke")
        run_id = f"smoke_{cfg.part}_{time.strftime('%Y%m%dT%H%M%S')}"
    else:
        run_id = str(cfg.run_id)
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    unit_list = units(cfg)
    try:
        run = RunDir.create(
            cfg.study, str(cfg.phase), run_id, config_text, requested_units=len(unit_list),
            extra_provenance={"units": unit_list, "part": cfg.part},
        )  # fmt: skip
    except FileExistsError:
        run = RunDir.resume(cfg.study, str(cfg.phase), run_id)
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
            run_part_a_unit(u, cfg, out_dir)
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
