"""Residual constructions and fixed-chain comparisons for synthetic targets.

Cellwise density perturbations satisfy monitored constraints on one quadrature
and are evaluated on a finer rule. Fixed-chain residuals use replicas and an
independently fitted kernel-PCA span; these laws need not satisfy the diagnostics.
Estimated-eigenvalue curves are plug-in comparisons.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import subprocess
import time

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import r1_suite as r1
from mcmc.paperexp.embeddings import rbf_mean_embedding
from mcmc.paperexp.kernels import make_kernel
from mcmc.paperexp.r1_surrogate import TabulatedRadialKernel
from mcmc.paperexp.rng import seed_for, torch_gen
from mcmc.paperexp.runio import RunDir, utc_now
from mcmc.paperexp.samplers import make_sampler
from mcmc.paperexp.targets import FunnelTarget, GaussTarget, load_rotation, make_target

log = logging.getLogger(__name__)
F64 = torch.float64
REPO = pathlib.Path(__file__).resolve().parents[3]

TARGET_SPECS = {
    "gauss-d2-c25": ("gauss", 2, {"cond": 25.0}),
    "mixture-d2-s6-w0.5": ("mixture", 2, {"sep": 6.0, "w": 0.5}),
    "studentt-d2-nu3": ("student_t", 2, {"nu": 3.0}),
    "funnel-d2-sy3": ("funnel", 2, {"sigma_y": 3.0}),
    "gauss-d10-c25": ("gauss", 10, {"cond": 25.0}),
    "gauss-d50-c25": ("gauss", 50, {"cond": 25.0}),
    "mixture-d10-s6-w0.5": ("mixture", 10, {"sep": 6.0, "w": 0.5}),
    "studentt-d10-nu3": ("student_t", 10, {"nu": 3.0}),
    "funnel-d10-sy3": ("funnel", 10, {"sigma_y": 3.0}),
    # Vary one target parameter at a time.
    "gauss-d10-c1": ("gauss", 10, {"cond": 1.0}),
    "gauss-d10-c100": ("gauss", 10, {"cond": 100.0}),
    "mixture-d10-s2-w0.5": ("mixture", 10, {"sep": 2.0, "w": 0.5}),
    "mixture-d10-s10-w0.5": ("mixture", 10, {"sep": 10.0, "w": 0.5}),
    "mixture-d10-s6-w0.1": ("mixture", 10, {"sep": 6.0, "w": 0.1}),
    "studentt-d10-nu5": ("student_t", 10, {"nu": 5.0}),
    "studentt-d10-nu30": ("student_t", 10, {"nu": 30.0}),
    "funnel-d10-sy1": ("funnel", 10, {"sigma_y": 1.0}),
}


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _frozen_doc(cfg: DictConfig) -> tuple[dict, dict]:
    """Read the suite from the optional Git ref or local path; return (doc, source)."""
    if cfg.frozen_suite_git_ref:
        ref = str(cfg.frozen_suite_git_ref)
        text = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{ref}:{cfg.frozen_suite}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", ref], capture_output=True, text=True, check=True
        ).stdout.strip()
        src = {"source": f"git:{commit}:{cfg.frozen_suite}"}
    else:
        text = (REPO / cfg.frozen_suite).read_text()
        src = {"source": str(cfg.frozen_suite)}
    src["source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    doc = yaml.safe_load(text)
    if doc.get("version") != "frozen_suite_v1":
        raise ValueError(f"unexpected frozen-suite version {doc.get('version')!r}")
    return doc, src


def frozen_target_params(target_id: str, cfg: DictConfig) -> dict:
    doc, src = _frozen_doc(cfg)
    pjson = pathlib.Path(cfg.pilot_params_dir) / "targets" / f"{target_id}.json"
    out = {}
    if target_id in doc["targets"]:
        spec = doc["targets"][target_id]
        out = {"scales": [float(v) for v in spec["scales"]], "ell": float(spec["ell"]), **src}
    if pjson.exists():
        pj = json.loads(pjson.read_text())
        pilot = {"scales": [float(v) for v in pj["scales"]], "ell": float(pj["ell"])}
        if out:
            if pilot["scales"] != out["scales"] or pilot["ell"] != out["ell"]:
                raise ValueError(f"{target_id}: pilot params differ from frozen file")
            out["pilot_json_agrees"] = str(pjson)
        else:
            out = {**pilot, "source": str(pjson), "source_sha256": _sha(pjson)}
    if not out:
        raise FileNotFoundError(f"no frozen or pilot parameters for {target_id}")
    return out


def frozen_config_params(config_id: str, cfg: DictConfig) -> dict:
    doc, src = _frozen_doc(cfg)
    pjson = pathlib.Path(cfg.pilot_params_dir) / "configs" / f"{config_id}.json"
    out = {}
    if config_id in doc["configs"]:
        c = doc["configs"][config_id]
        out = {
            "sampler": c["sampler"],
            "h": float(c["h"]),
            "t_grid": [int(t) for t in c["t_grid"]],
            **src,
        }
    if pjson.exists():
        pj = json.loads(pjson.read_text())
        pilot = {
            "sampler": pj["tune"]["kind"],
            "h": float(pj["tune"]["h"]),
            "t_grid": [int(t) for t in pj["tmax"]["t_grid"]],
        }
        if out:
            for key in ("sampler", "h", "t_grid"):
                if pilot[key] != out[key]:
                    raise ValueError(f"{config_id}: pilot {key} differs from frozen file")
            out["pilot_json_agrees"] = str(pjson)
        else:
            out = {**pilot, "source": str(pjson), "source_sha256": _sha(pjson)}
    if not out:
        raise FileNotFoundError(f"no frozen or pilot parameters for {config_id}")
    return out


def build_target(target_id: str):
    fam, d, params = TARGET_SPECS[target_id]
    rot = load_rotation(fam, d) if fam in ("gauss", "mixture") else None
    t = make_target(fam, d, rotation=rot, **dict(params))
    if t.target_id != target_id:
        raise ValueError(f"built {t.target_id} != {target_id}")
    return t


_INT_FIELDS = ("q_out", "q_in", "q_min", "funnel_q_y")


def _qspec(c) -> r1.QuadSpec:
    return r1.QuadSpec(**{k: (int(v) if k in _INT_FIELDS else float(v)) for k, v in c.items()})


def _lspec(c, rel_cutoff) -> r1.LandmarkSpec:
    return r1.LandmarkSpec(**{k: float(v) for k, v in c.items()}, rel_cutoff=float(rel_cutoff))


def cell_gram(integ, spec: r1.Spectrum, gcfg) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Compute the cell Gram matrix, using Nystrom above the node-count limit.

    The low-rank route also returns a certified omission vector.
    """
    tt = time.time()
    if not hasattr(integ, "X"):
        G = integ.cell_gram().numpy()
        return G, None, {"method": "semi_analytic", "wall_s": time.time() - tt}
    n = integ.X.shape[0]
    method = gcfg.method
    if method == "auto":
        method = "exact" if n <= int(gcfg.auto_max_nodes) else "lowrank"
    if method == "exact":
        G = integ.cell_gram().numpy()
        return G, None, {"method": "exact", "n_nodes": n, "wall_s": time.time() - tt}
    if method != "lowrank":
        raise ValueError(f"unknown gram method {method!r}")
    G, s, info = integ.cell_gram_lowrank(spec.feats)
    return G.numpy(), s, {"method": "lowrank", "n_nodes": n, **info, "wall_s": time.time() - tt}


def run_construction_unit(target_id: str, kind: str, cfg: DictConfig, out_dir: pathlib.Path):
    cc = cfg.constructions
    t0 = time.time()
    t = build_target(target_id)
    fp = frozen_target_params(target_id, cfg)
    surrogate = kind == "matern32s"  # Matern-3/2 surrogate (tapered scale mixture), node route
    if surrogate:
        fm_, sr = cc.funnel_matern, cc.surrogate
        lam_, w_ = r1.matern32_scale_mixture(
            float(fm_.kernel_rule.h), float(fm_.taper.log_lam0), float(fm_.taper.tau),
            float(fm_.kernel_rule.log_lam_max), float(fm_.kernel_rule.offset),
        )  # fmt: skip
        kernel = TabulatedRadialKernel(
            np.asarray(fp["scales"]), fp["ell"], lam_, w_, float(sr.table_r_max),
            int(sr.table_n), label="Matern-3/2 surrogate (tapered scale mixture), tabulated",
        )  # fmt: skip
    else:
        kernel = make_kernel(kind, np.asarray(fp["scales"]), fp["ell"])
    semi = isinstance(t, FunnelTarget)
    cut = float(cc.rel_cutoff_semi_analytic if semi else cc.rel_cutoff)
    qc, qe = _qspec(cc.quad_construction), _qspec(cc.quad_eval)
    # For the funnel, approximate Matern-3/2 with a tapered Gaussian scale mixture
    # and integrate its RBF components semi-analytically.
    fm = cc.get("funnel_matern")
    mixture = semi and kind == "matern32"
    mixes, nodes = {}, {}
    if mixture:
        if fm is None:
            raise ValueError("funnel x matern32 needs constructions.funnel_matern")
        sc_ = np.asarray(fp["scales"])

        def _mix(rule, label):
            lam, w = r1.matern32_scale_mixture(
                float(rule.h), float(fm.taper.log_lam0), float(fm.taper.tau),
                float(rule.log_lam_max), float(rule.offset),
            )  # fmt: skip
            return r1.ScaledGaussMixture(sc_, fp["ell"], lam, w, label=label)

        kernel = _mix(fm.kernel_rule, "tapered Matern-3/2 scale mixture (surrogate, kernel rule)")
        mixes = {"construction": _mix(fm.rule_construction, "construction rule"),
                 "eval": _mix(fm.rule_eval, "evaluation rule")}  # fmt: skip
        nodes = {lvl: r1.NodeSpec(**{k: (int(v) if k.startswith("q_") else float(v))
                                     for k, v in fm[f"nodes_{lvl}"].items()})
                 for lvl in ("construction", "eval")}  # fmt: skip
        cut = float(fm.rel_cutoff)

    def mk(n: int, qs, lvl: str):
        if mixture:
            return r1.FunnelMixtureIntegrator(t, mixes[lvl], n, n, qs, nodes[lvl], int(fm.workers))
        return r1.make_integrator(t, kernel, n, n, qs)

    if mixture:
        lc = _lspec(fm.landmarks_construction, cut)
        le = _lspec(fm.landmarks_eval, cut)
    elif semi:
        lc = _lspec(cc.landmarks_construction_funnel, cut)
        le = _lspec(cc.landmarks_eval_funnel, cut)
    elif kind in ("matern32", "matern32s"):
        lc = _lspec(cc.landmarks_construction_matern, cut)
        le = _lspec(cc.landmarks_eval_matern, cut)
    else:
        lc, le = _lspec(cc.landmarks_construction, cut), _lspec(cc.landmarks_eval, cut)
    m_values = [int(m) for m in cc.m_values]
    mmax = max(m_values)
    res = {
        "unit": f"{target_id}__{kind}",
        "target": {k: v for k, v in t.to_dict().items() if k != "rotation"},
        "kernel": kernel.to_dict(),
        "kernel_params_source": fp,
        "quad_construction": qc.to_dict(),
        "quad_eval": qe.to_dict(),
        "semi_analytic": semi,
        "label": "feasible examples, not certified maximizers; r = 1/2 envelope",
    }
    if surrogate:
        res["label"] += "; kernel = Matern-3/2 surrogate (tapered scale mixture), not Matern-3/2"
        res["surrogate_table_max_abs_err"] = kernel.max_abs_err_vs()
    if mixture:
        res["label"] += (
            "; kernel = tapered Gaussian scale mixture of Matern-3/2 "
            "(components narrower than ~lam0 removed), not Matern-3/2 itself"
        )
        res["funnel_matern"] = OmegaConf.to_container(fm, resolve=True)
        res["mixture_rules"] = {lvl: mx.to_dict() for lvl, mx in mixes.items()}
    arrays = {}

    # spectra: construction level (defines the monitors) and evaluation level
    spectra = {}
    for lvl, qs, ls in (("construction", qc, lc), ("eval", qe, le)):
        tt = time.time()
        gi = mk(1, qs, lvl)
        Z, linfo = r1.make_landmarks(t, kernel, gi, ls)
        sp = r1.nystrom_spectrum(kernel, Z, gi, cut)
        if sp.lam.shape[0] < mmax + 2:
            raise RuntimeError(f"feature rank {sp.lam.shape[0]} < m_max + 2")
        spectra[lvl] = (sp, gi)
        res[f"spectrum_{lvl}"] = {
            **sp.info,
            "landmarks": linfo,
            "global_rule": gi.info,
            "lam_top": sp.lam[: mmax + 8].tolist(),
            "wall_s": time.time() - tt,
        }
        arrays[f"lam_{lvl}"] = sp.lam
        log.info(
            "%s %s spectrum %s: L=%d rank=%d lam_33=%.4e (%.1fs)",
            target_id,
            kind,
            lvl,
            Z.shape[0],
            sp.info["rank"],
            sp.lam[mmax],
            time.time() - tt,
        )
    sp_c, gi_c = spectra["construction"]
    sp_e, gi_e = spectra["eval"]
    res["eigengap_rel"] = {
        lvl: {m: float((sp.lam[m - 1] - sp.lam[m]) / sp.lam[m - 1]) for m in m_values}
        for lvl, (sp, _) in spectra.items()
    }
    res["lam_rel_diff_construction_vs_eval"] = np.abs(
        sp_c.lam[: mmax + 2] / sp_e.lam[: mmax + 2] - 1
    ).tolist()

    # witnesses
    wit = {}
    if kind == "rbf":
        emb = rbf_mean_embedding(t, kernel)
        for lvl, (sp, gi) in spectra.items():
            wit[f"mu_vs_core_max_abs_{lvl}"] = float((gi.mean(sp.Z) - emb(sp.Z)).abs().max())
        wit["core_mu_norm2"] = emb.norm2()
    if isinstance(t, GaussTarget) and kind == "rbf":
        g = r1.gaussian_centered_spectrum(t.covariance(), kernel)
        for lvl, (sp, _) in spectra.items():
            rel = sp.lam[: mmax + 2] / g["eig"][: mmax + 2] - 1
            wit[f"mehler_lam_rel_err_{lvl}"] = rel.tolist()
            wit[f"mehler_lam_rel_err_max_{lvl}"] = float(np.abs(rel).max())
        wit["mehler_angles_construction"] = r1.mehler_rkhs_angles(sp_c, g, kernel, m_values)
        wit["mehler_eig"] = g["eig"][: mmax + 8].tolist()
    if mixture:
        r = np.concatenate([np.linspace(0.0, 0.5, 5001), np.linspace(0.5, 40.0, 7901)])
        kref = r1.mixture_kernel_values(kernel.lam, kernel.w, r)
        gap = r1.matern32(r) - kref
        wit["surrogate_minus_matern32"] = {"sup_abs": float(np.abs(gap).max()),
                                           "r_at_sup": float(r[np.abs(gap).argmax()]),
                                           "at_r0": float(gap[0]), "min": float(gap.min())}  # fmt: skip
        for lvl, mx in mixes.items():
            kl = r1.mixture_kernel_values(mx.lam, mx.w, r)
            wit[f"level_rule_minus_kernel_rule_sup_{lvl}"] = float(np.abs(kl - kref).max())
        for lvl, (sp, gi) in spectra.items():
            wit[f"mean_semi_analytic_vs_nodes_max_abs_{lvl}"] = float(
                (gi.mean(sp.Z) - gi.mean_nodes(sp.Z)).abs().max()
            )
            wit[f"second_moment_nodes_{lvl}"] = gi.info.get("second_moment_nodes")
    res["witness"] = wit

    # compressed envelope for the actual monitors (eval-level Sigma)
    if cc.compressed_check:
        tt = time.time()
        comp = {}
        for m, (top, info) in r1.compressed_top_eigs(
            kernel, sp_c, m_values, sp_e.Z, gi_e, cut
        ).items():
            comp[m] = {"lam_compressed": top, **info}
        res["compressed"] = comp
        log.info("%s %s compressed check (%.1fs)", target_id, kind, time.time() - tt)

    # constructions over the cell grids
    cons = []
    for n in [int(v) for v in cc.n_grid]:
        tt = time.time()
        ic = mk(n, qc, "construction")
        ie = mk(n, qe, "eval")
        if not np.array_equal(ic.masses, ie.masses):
            raise ValueError("construction and evaluation integrators disagree on cell masses")
        Gc, sc, gic = cell_gram(ic, sp_c, cc.gram)
        Ge, se_, gie = cell_gram(ie, sp_e, cc.gram)
        Mc = ic.cell_integrals(sp_c.Z)
        Me = ie.cell_integrals(sp_c.Z)
        grid_info = {
            "n": n,
            "K": n * n,
            "integ_construction": ic.info,
            "integ_eval": ie.info,
            "G_sum_construction": float(Gc.sum()),
            "G_sum_eval": float(Ge.sum()),
            "G_max_abs_diff": float(np.abs(Gc - Ge).max()),
            "gram_construction": gic,
            "gram_eval": gie,
            "masses_min": float(ic.masses.min()),
            "masses_max": float(ic.masses.max()),
        }
        for m in m_values:
            if n * n <= m + 1:
                continue
            rows_c = r1.moment_rows(sp_c, m, Mc)
            rows_e = r1.moment_rows(sp_c, m, Me)
            for R in [float(v) for v in cc.R_values]:
                c = r1.construct(
                    Gc,
                    ic.masses,
                    rows_c,
                    r1.ConstructionSpec(R, int(cc.n_dir), float(cc.floor), int(cc.n_combo)),
                )
                ev = r1.evaluate(c["a"], ic.masses, Ge, rows_e, se_)
                combo = None
                if c["combo"] is not None:
                    cev = r1.evaluate(c["combo"]["a"], ic.masses, Ge, rows_e, se_)
                    combo = {
                        **{k: v for k, v in c["combo"].items() if k != "a"},
                        **{k: v for k, v in cev.items() if k != "readings_eval"},
                    }
                    arrays[f"acombo_n{n}_m{m}_R{R:g}"] = c["combo"]["a"]
                lam_next_e, lam_next_c = float(sp_e.lam[m]), float(sp_c.lam[m])
                env = R * math.sqrt(lam_next_e)
                row = {
                    "n": n,
                    "K": n * n,
                    "m": m,
                    "R": R,
                    **{k: v for k, v in c.items() if k not in ("a", "candidates", "combo")},
                    "combo": combo,
                    **ev,
                    "candidates": c["candidates"],
                    "lam_next_eval": lam_next_e,
                    "lam_next_construction": lam_next_c,
                    "envelope": env,
                    "envelope_realized_radius": c["R_realized"] * math.sqrt(lam_next_e),
                    "ratio_residual_envelope": ev["residual_eval"] / env,
                    # Width before positivity constraints, normalized by the evaluation envelope.
                    "ratio_width_envelope_eval": math.sqrt(max(c["nu_top"][0], 0.0) / lam_next_e),
                }
                if cc.compressed_check:
                    row["envelope_compressed_realized"] = c["R_realized"] * math.sqrt(
                        max(res["compressed"][m]["lam_compressed"], 0.0)
                    )
                cons.append(row)
                arrays[f"a_n{n}_m{m}_R{R:g}"] = c["a"]
            log.info(
                "%s %s n=%d m=%d: residual/env %s",
                target_id,
                kind,
                n,
                m,
                [
                    f"{r['ratio_residual_envelope']:.3f}"
                    for r in cons
                    if r["n"] == n and r["m"] == m
                ],
            )
        grid_info["wall_s"] = time.time() - tt
        res.setdefault("grids", []).append(grid_info)
        arrays[f"masses_n{n}"] = ic.masses
        if n == max(int(v) for v in cc.n_grid):
            finest = (ic, ie)
    res["constructions"] = cons

    # independent Monte Carlo check (exact draws from Q)
    mc = cc.mc_check
    if mc.enabled and kind in list(mc.kernels):
        emb = rbf_mean_embedding(t, kernel)
        ic, _ = finest
        n = max(int(v) for v in cc.n_grid)
        out = []
        for m in [int(v) for v in mc.m_values]:
            key = f"a_n{n}_m{m}_R{float(mc.R):g}"
            a = arrays[key]
            gen = torch_gen(seed_for(f"r1b-{target_id}", f"mc-check-{cfg.seed_tag}", m), "cpu")
            X = r1.sample_construction(t, ic, a, int(mc.n_samples), gen)
            mmd2, se = r1.mc_mmd2_unbiased(kernel, X, emb(X), emb.norm2(), int(mc.n_batches))
            cvals = sp_c.criteria(X, m)
            ref = [r for r in cons if r["n"] == n and r["m"] == m and r["R"] == float(mc.R)][0]
            out.append(
                {
                    "m": m,
                    "n_samples": int(mc.n_samples),
                    "mmd2_mc": mmd2,
                    "mmd2_mc_se": se,
                    "mmd2_eval": ref["mmd2_eval"],
                    "z_score": (mmd2 - ref["mmd2_eval"]) / se,
                    "criteria_mean_Q": cvals.mean(0).tolist(),
                    "criteria_se_Q": (cvals.std(0) / math.sqrt(X.shape[0])).tolist(),
                }
            )
        res["mc_check"] = out

    res["wall_s"] = time.time() - t0
    res["finished_utc"] = utc_now()
    np.savez_compressed(out_dir / f"{res['unit']}.npz", **arrays)
    write_json(out_dir / f"{res['unit']}.json", res)
    return res


def run_projected_unit(config_id: str, cfg: DictConfig, out_dir: pathlib.Path):
    pc = cfg.projected
    device = torch.device(cfg.device)
    t0 = time.time()
    target_id = config_id.split("__")[0]
    t = build_target(target_id)
    fp = frozen_target_params(target_id, cfg)
    cp = frozen_config_params(config_id, cfg)
    if cp["sampler"] not in ("rwm", "mala"):
        raise ValueError(f"projected residuals use a frozen RWM / MALA; got {cp['sampler']}")
    scales = np.asarray(fp["scales"])
    ell = fp["ell"] * {"ell": 1.0, "ell_half": 0.5, "ell_double": 2.0}[pc.bw]
    kernel = make_kernel(pc.kernel, scales, ell)
    emb = rbf_mean_embedding(t, kernel)
    sampler = make_sampler(cp["sampler"], t, cp["h"], scales)
    m_values = [int(m) for m in pc.m_values]
    M = int(pc.n_monitors)
    t_grid = cp["t_grid"] if pc.t_values == "all" else [int(v) for v in pc.t_values]
    res = {
        "unit": config_id,
        "label": "Projected residuals of a fixed chain law; monitored moments may be nonzero. "
        "Eigenvalue comparisons are plug-in estimates, not confidence bounds.",
        "kernel": kernel.to_dict(),
        "kernel_params_source": fp,
        "config_params_source": cp,
        "mu_norm2": emb.norm2(),
        "eps": float(pc.eps),
        "R_init": float(pc.eps),
        "t_grid": t_grid,
    }
    # estimated Sigma (independent stream) and held-out sample
    tt = time.time()
    Y = t.sample(
        int(pc.n_sigma), torch_gen(seed_for(config_id, f"r1b-sigma-{cfg.seed_tag}", 0), device)
    )
    pca = r1.KernelPCA(kernel, Y, M + 1)
    Yh = t.sample(
        int(pc.n_heldout),
        torch_gen(seed_for(config_id, f"r1b-sigma-heldout-{cfg.seed_tag}", 0), device),
    )
    lam_ho = {m: r1.heldout_compressed_top(pca, Yh, m) for m in m_values}
    Ec = (emb(Y) @ pca.coef).cpu()  # E_pi c_i exactly (given the estimated V)
    res["sigma_hat"] = {
        "n_sigma": int(pc.n_sigma),
        "lam_hat": pca.lam.tolist(),
        "ortho_err": pca.ortho_err,
        "colsum_max": pca.colsum_max,
        "lam_heldout_compressed": {str(m): v for m, v in lam_ho.items()},
        "wall_s": time.time() - tt,
    }
    # replicas of the fixed chain law, nested over checkpoints
    J, nb = int(pc.J), int(pc.n_batches)
    if J % nb:
        raise ValueError("J must be divisible by n_batches")
    gen = torch_gen(seed_for(config_id, f"r1b-replicas-{cfg.seed_tag}", 0), device)
    x0 = t.sample_init(J, float(pc.eps), gen)
    tt = time.time()
    run = sampler.run(x0, t_grid, gen, store=True, store_device="cpu")
    res["sampler"] = {
        "accept_rate": run["accept_rate"],
        "counters": run["counters"],
        "wall_s": time.time() - tt,
    }
    per_t = {}
    b = J // nb
    for tc in t_grid:
        X = run["states"][tc].to(device)
        Xb = [X[i * b : (i + 1) * b] for i in range(nb)]
        acc = r1.ProjectedAccumulator(nb, M)
        mu_at = [emb(x) for x in Xb]
        g_at = [pca.criteria(x, M).cpu() - Ec[None, :M] for x in Xb]
        acc.add(kernel, Xb, mu_at, g_at)
        est = acc.estimates(emb.norm2(), m_values)
        rows = {}
        for m in m_values:
            p2, mon2 = est[f"proj2_{m}"], est[f"mon2_{m}"]
            resolved = p2["est"] > 2.0 * p2["se"]
            rows[str(m)] = {
                "proj2": p2["est"],
                "proj2_se": p2["se"],
                "proj": math.sqrt(p2["est"]) if resolved else None,
                "proj_resolved": bool(resolved),
                "mon2": mon2["est"],
                "mon2_se": mon2["se"],
                "bound_plugin": float(pc.eps) * math.sqrt(pca.lam[m]),
                "bound_heldout_compressed": float(pc.eps) * math.sqrt(max(lam_ho[m], 0.0)),
            }
        per_t[str(tc)] = {"mmd2": est["mmd2"]["est"], "mmd2_se": est["mmd2"]["se"], "m": rows}
        log.info(
            "%s t=%d mmd2=%.3e proj2[m=2,32]=%.3e,%.3e",
            config_id,
            tc,
            est["mmd2"]["est"],
            rows[str(m_values[0])]["proj2"],
            rows[str(m_values[-1])]["proj2"],
        )
    res["per_t"] = per_t
    res["wall_s"] = time.time() - t0
    res["finished_utc"] = utc_now()
    write_json(out_dir / f"{config_id}.json", res)
    return res


def units(cfg: DictConfig) -> list[str]:
    if cfg.part == "constructions":
        u = [f"{tid}__{k}" for tid in cfg.constructions.targets for k in cfg.constructions.kernels]
        if cfg.constructions.get("funnel_matern") is None:  # funnel x matern: mixture route only
            u = [x for x in u if not (x.startswith("funnel") and not x.endswith("__rbf"))]
    elif cfg.part == "projected":
        u = list(cfg.projected.configs)
    else:
        raise ValueError(f"unknown part {cfg.part!r}")
    if cfg.only_units:
        missing = set(cfg.only_units) - set(u)
        if missing:
            raise ValueError(f"unknown units {missing}")
        u = [x for x in u if x in set(cfg.only_units)]
    return u


@hydra.main(
    config_path="../../../configs/experiment",
    config_name="exp_r1b_constructions_suite",
    version_base=None,
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
    study = f"{cfg.study}"
    try:
        run = RunDir.create(
            study,
            str(cfg.phase),
            run_id,
            config_text,
            requested_units=len(unit_list),
            allow_dirty=False,
            extra_provenance={"units": unit_list, "part": cfg.part},
        )
    except FileExistsError:
        run = RunDir.resume(study, str(cfg.phase), run_id)
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
            if cfg.part == "constructions":
                tid, kind = u.split("__")
                run_construction_unit(tid, kind, cfg, out_dir)
            else:
                run_projected_unit(u, cfg, out_dir)
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
