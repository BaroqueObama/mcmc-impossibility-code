"""Logistic slow-mode alignment using NUTS reference draws.

The reference posterior and transition operators are estimated. See README.md
for the calibration, sampling and export commands.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import resource
import time

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import b_logistic as bl
from mcmc.paperexp import german_credit as gc
from mcmc.paperexp import german_credit_r2 as r2
from mcmc.paperexp import slowmodes as sm
from mcmc.paperexp.german_credit_reference import load_set, set_path
from mcmc.paperexp.rng import np_gen, seed_for, torch_gen
from mcmc.paperexp.runio import RunDir, output_root
from mcmc.paperexp.samplers import make_sampler

log = logging.getLogger(__name__)
REPO = pathlib.Path(__file__).resolve().parents[3]
F64 = torch.float64


def sha256_file(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _json(path: pathlib.Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, default=_default, allow_nan=True))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_csv(rows: list[dict], dest: pathlib.Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class Unit:
    """Saved sampling and evaluation results for one model/sampler pair."""

    def __init__(self, cfg: DictConfig, name: str):
        self.name = name
        self.model, self.sampler = name.split("__")
        src = cfg.source
        base = output_root() / str(src.study) / str(src.phase)
        self.v1 = base / f"{name}_{src.tag}"
        self.ev = base / f"{name}_{src.tag}_{src.eval_tag}"
        for d in (self.v1, self.ev):
            st = json.loads((d / "RUN-STATUS.json").read_text())["status"]
            if st != "COMPLETED":
                raise RuntimeError(f"{d} is {st}, expected COMPLETED")
        prov = json.loads((self.ev / "PROVENANCE.json").read_text())
        if prov.get("source_run") != f"{src.study}/{src.phase}/{name}_{src.tag}":
            raise RuntimeError(f"{self.ev} is not a re-evaluation of {self.v1}")
        if (self.v1 / "config.yaml").read_bytes() != (self.ev / "config.yaml").read_bytes():
            raise RuntimeError(f"{name}: eval2 config differs from v1 config")
        self.cfg = OmegaConf.create((self.v1 / "config.yaml").read_text())
        if str(self.cfg.model) != self.model or str(self.cfg.sampler) != self.sampler:
            raise RuntimeError(f"{name}: stored config is {self.cfg.model}__{self.cfg.sampler}")
        fr = self.cfg.frozen[name]
        self.h, self.T, self.t_star = float(fr.h), int(fr.T), int(fr.t_star)
        self.horizon = max(self.T, int(self.cfg.horizon_min))
        self.dataset = str(cfg.datasets[self.model])
        self.appendix = self.model in list(cfg.appendix_models)


    def meta(self) -> dict:
        setup = json.loads((self.v1 / "setup.json").read_text())
        sig = json.loads((self.v1 / "sigma.json").read_text())
        return {
            "unit": self.name,
            "model": self.model,
            "dataset": self.dataset,
            "sampler": self.sampler,
            "appendix_stress": self.appendix,
            "d": int(np.load(self.v1 / "setup.npz")["Z"].shape[1]),
            "feature_rank": int(setup["features"]["rank"]),
            "n_landmarks": int(setup["features"]["n_landmarks"]),
            "nystrom_rel_cutoff": float(setup["features"]["rel_cutoff"]),
            "whitening_error": float(setup["whitening_error"]),
            "h": self.h,
            "T": self.T,
            "t_star": self.t_star,
            "horizon": self.horizon,
            "sigma_N_blocks": int(sig["N"]),
            "sigma_wall_s": float(sig["wall_s"]),
            "run_v1": str(self.v1),
            "run_eval2": str(self.ev),
        }


class SavedFeatures:
    """f(x) = k(x, Z) B using the basis in setup.npz."""

    def __init__(self, cfg: DictConfig, u: Unit):
        z = np.load(u.v1 / "setup.npz")
        setup = json.loads((u.v1 / "setup.json").read_text())
        entry = yaml.safe_load((REPO / str(cfg.kernel_file)).read_text())[
            str(u.cfg.references[u.model].kernel_key)
        ]
        if entry["scales"] != setup["kernel"]["scales"] or entry["ell"] != setup["kernel"]["ell"]:
            raise RuntimeError(f"{u.name}: kernel file entry differs from the saved setup kernel")
        self.kernel = r2.make_kernel(entry)
        self.Z = torch.as_tensor(z["Z"], dtype=F64)
        self.B = torch.as_tensor(z["B"], dtype=F64)
        G = self.B.T @ self.kernel(self.Z, self.Z) @ self.B
        self.whitening_error = float((G - torch.eye(G.shape[0], dtype=F64)).abs().max())
        if self.whitening_error > 1e-8:
            raise RuntimeError(f"{u.name}: saved basis whitening error {self.whitening_error:.2e}")
        self.rank = int(self.B.shape[1])

    def __call__(self, X: np.ndarray, rows: int = 8192) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=F64)
        out = torch.empty((Xt.shape[0], self.rank), dtype=F64)
        for i0 in range(0, Xt.shape[0], rows):
            out[i0 : i0 + rows] = self.kernel(Xt[i0 : i0 + rows], self.Z) @ self.B
        F = out.numpy()
        if not np.isfinite(F).all():
            raise FloatingPointError("non-finite features")
        return F


def calibrate_unit(cfg: DictConfig, u: Unit, out: pathlib.Path) -> dict:
    c = cfg.calibration
    t0 = time.time()
    feats = SavedFeatures(cfg, u)
    p = u.cfg.pilots
    n_fit, C = int(p.n_fit), int(p.n_chains)
    Lmax = max(int(v) for v in p.lengths)
    X = np.load(u.v1 / f"pilotrun_{c.pilot}.npz")["X"]
    d = X.shape[2]
    if X.shape != (n_fit * C, Lmax + 1, d):
        raise RuntimeError(f"{u.name}: pilot paths have shape {X.shape}")
    if n_fit % 2:
        raise ValueError("half split by fits needs an even n_fit")
    t_load = time.time() - t0
    half = (n_fit // 2) * C
    group = np.repeat([0, 1], half)  # chain -> half (A: fits 0..n/2-1, B: the rest)
    lags = tuple(int(t) for t in bl.RULES["lags"])
    # pass 1: pooled mean of the full set (states 1..Lmax, as R2's span learning)
    t1 = time.time()
    tot = np.zeros(feats.rank)
    for ch in range(X.shape[0]):
        tot += feats(np.ascontiguousarray(X[ch, 1:])).sum(0)
    center = tot / (X.shape[0] * Lmax)
    # pass 2: lagged moments per half
    mom = bl.LagMoments.empty(2, Lmax, lags, center)
    for ch in range(X.shape[0]):
        mom.add_chain(int(group[ch]), feats(np.ascontiguousarray(X[ch, 1:])))
    t_mom = time.time() - t1
    sets = {"full": [0, 1], "A": [0], "B": [1]}
    cutoffs = {"primary": float(c.rel_cutoff), "sensitivity": float(c.rel_cutoff_sensitivity)}
    n_keep, n_span = int(c.n_keep), int(c.n_span)
    arrays, fitinfo, results = {}, {}, {}
    for cname, cut in cutoffs.items():
        fits = {}
        for sname, sel in sets.items():
            for tau in lags:
                C0, Cl, asym, npairs = mom.covariances(sel, tau)
                if cname == "primary":
                    arrays[f"C0__{sname}__lag{tau}"] = C0
                    arrays[f"Clag__{sname}__lag{tau}"] = Cl
                f = bl.tica_fit(C0, Cl, cut, n_keep, asym)
                fits[(sname, tau)] = f
                arrays[f"rho__{cname}__{sname}__lag{tau}"] = f.rho
                arrays[f"span__{cname}__{sname}__lag{tau}"] = f.embedded[:, :n_span]
                fitinfo[f"{cname}/{sname}/lag{tau}"] = {
                    "rank": f.rank,
                    "n_dropped": f.n_dropped,
                    "lag_asymmetry": f.lag_asymmetry,
                    "whitened_asymmetry": f.whitened_asymmetry,
                    "max_abs_rho": f.max_abs_rho,
                    "n_pairs": npairs,
                }
        rows = bl.candidate_table(fits)
        results[cname] = {"candidates": rows, "selection": bl.select_counts(rows)}
    arrays["center"] = center
    arrays["mean_A"], arrays["mean_B"] = mom.mean([0]), mom.mean([1])
    np.savez(out / f"calib_{u.name}.npz", **arrays)
    # reference-operator window projection for the selected counts (primary cutoff)
    sig = np.load(u.v1 / "sigma.npz")
    grid = [int(t) for t in sig["grid"]]
    meta = u.meta()
    fr = cfg.fresh
    windows = {}
    for m in results["primary"]["selection"]["selected_m"]:
        rows = bl.window_projection(
            sig["O"],
            sig["OA"],
            sig["OB"],
            grid,
            int(m),
            meta["sigma_N_blocks"],
            int(fr.n_blocks),
            int(fr.n_batches),
            float(fr.e_factor),
        )
        informative = [
            r["t"]
            for r in rows
            if r["t"] >= 1 and r["dk_term_deg_fresh"] <= float(fr.dk_informative_deg)
        ]
        applicable = [r["t"] for r in rows if r["t"] >= 1 and r["dk_applicable_fresh"]]
        windows[str(m)] = {
            "rows": rows,
            "t_dk_applicable_fresh": applicable,
            "t_dk_informative_fresh": informative,
            "t_dk_applicable_saved": [
                r["t"] for r in rows if r["t"] >= 1 and r["dk_applicable_saved"]
            ],
        }
    rec = {
        "unit": u.name,
        "meta": meta,
        "status": "Empirical finite-feature gaps; the transition spectrum is unknown.",
        "pilot": {
            "file": str(u.v1 / f"pilotrun_{c.pilot}.npz"),
            "n_chains": int(X.shape[0]),
            "L": Lmax,
            "half_split": f"A = chains 0..{half - 1} (fits 0..{n_fit // 2 - 1}), "
            f"B = chains {half}..{X.shape[0] - 1}",
        },
        "saved_basis_whitening_error": feats.whitening_error,
        "fits": fitinfo,
        "results": results,
        "windows": windows,
        "wall_s": {"load": t_load, "moments": t_mom, "total": time.time() - t0},
    }
    _json(out / f"calib_{u.name}.json", rec)
    return rec


def run_calibrate(cfg: DictConfig) -> pathlib.Path:
    units = [Unit(cfg, u) for u in cfg.units]
    run_id = f"calibrate_{cfg.tag}"
    path = output_root() / str(cfg.study) / "pilot" / run_id
    if (path / "RUN-STATUS.json").exists():
        run = RunDir.resume(str(cfg.study), "pilot", run_id)
    else:
        run = RunDir.create(str(cfg.study), "pilot", run_id, OmegaConf.to_yaml(cfg), len(units))
    try:
        for u in units:
            if (run.path / f"calib_{u.name}.json").exists():
                continue
            rec = calibrate_unit(cfg, u, run.path)
            done = sum((run.path / f"calib_{v.name}.json").exists() for v in units)
            sel = rec["results"]["primary"]["selection"]
            run.update(
                completed_units=done,
                note=f"{u.name}: {sel['status']} lag={sel['lag']} m={sel['selected_m']} "
                f"{rec['wall_s']['total']:.0f}s rss "
                f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB",
            )
            log.info("%s: %s", u.name, sel)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=repr(exc)[:500])
        raise
    return run.path


CAND_FIELDS = (
    "lag",
    "m",
    "rho_m",
    "rho_m1",
    "rho_m_A",
    "rho_m1_A",
    "rho_m_B",
    "rho_m1_B",
    "gap",
    "sensitivity",
    "gap_ratio",
    "angle_half_deg",
    "next_lag",
    "rho_m_next",
    "rho_m1_next",
    "gap_next",
    "angle_next_deg",
    "rule1",
    "rule2",
    "eligible",
    "reason",
)


def run_select(cfg: DictConfig) -> pathlib.Path:
    """Save diagnostic-count selections and calibration tables."""
    cal = output_root() / str(cfg.study) / "pilot" / f"calibrate_{cfg.tag}"
    status = json.loads((cal / "RUN-STATUS.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"{cal} is {status['status']}; selection needs a COMPLETED calibration")
    out = REPO / str(cfg.calibration.out_dir)
    dest = out / f"selection_logistic_{cfg.tag}.json"
    if dest.exists():
        raise FileExistsError(f"{dest} exists; record amendments in a new file")
    cand_rows, win_rows, units = [], [], {}
    hashes = {}
    for name in cfg.units:
        u = Unit(cfg, str(name))
        rec = json.loads((cal / f"calib_{name}.json").read_text())
        for f in (f"calib_{name}.json", f"calib_{name}.npz"):
            hashes[str(cal / f)] = sha256_file(cal / f)
        for f in ("pilotrun_stationary.npz", "setup.npz", "sigma.npz"):
            p = u.v1 / f
            hashes[str(p)] = sha256_file(p)
        res = rec["results"]
        for cname in ("primary", "sensitivity"):
            for row in res[cname]["candidates"]:
                cand_rows.append(
                    {"unit": name, "cutoff": cname, **{k: row.get(k, "") for k in CAND_FIELDS}}
                )
        for m, w in rec["windows"].items():
            for row in w["rows"]:
                win_rows.append({"unit": name, **row})
        sel, sens = res["primary"]["selection"], res["sensitivity"]["selection"]
        chosen = {
            r["m"]: r
            for r in res["primary"]["candidates"]
            if sel["lag"] is not None and r["lag"] == sel["lag"] and r["m"] in sel["selected_m"]
        }
        units[str(name)] = {
            "model": u.model,
            "dataset": u.dataset,
            "sampler": u.sampler,
            "appendix_stress": u.appendix,
            "feature_rank": rec["meta"]["feature_rank"],
            "status": sel["status"],
            "lag": sel["lag"],
            "selected_m": sel["selected_m"],
            "selected": {
                str(m): {k: chosen[m].get(k) for k in CAND_FIELDS if k not in ("reason",)}
                for m in sel["selected_m"]
            },
            "eligible": sel["eligible"],
            "sensitivity_cutoff_selection": {k: sens[k] for k in ("status", "lag", "selected_m")},
            "sensitivity_cutoff_agrees": sens["lag"] == sel["lag"]
            and sens["selected_m"] == sel["selected_m"],
            "rule4": bl.RULES["rule4"],
            "gap_label": "finite-feature empirical gap (not a certified transition gap)",
            "reference_window": {
                m: {k: w[k] for k in w if k != "rows"} for m, w in rec["windows"].items()
            },
            "tica_fit_info": {
                k: v for k, v in rec["fits"].items() if k.startswith("primary/full/")
            },
        }
    _write_csv(cand_rows, out / f"calibration_candidates_logistic_{cfg.tag}.csv")
    _write_csv(win_rows, out / f"window_projection_logistic_{cfg.tag}.csv")
    payload = {
        "status": "Calibration-only selection from empirical TICA gaps. "
        "The logistic transition spectrum is unknown.",
        "algorithm": bl.RULES,
        "calibration_config": OmegaConf.to_container(cfg.calibration),
        "fresh_projection_config": OmegaConf.to_container(cfg.fresh),
        "calibration_hashes": hashes,
        "candidates_csv": {
            "path": f"calibration_candidates_logistic_{cfg.tag}.csv",
            "sha256": sha256_file(out / f"calibration_candidates_logistic_{cfg.tag}.csv"),
        },
        "window_csv": {
            "path": f"window_projection_logistic_{cfg.tag}.csv",
            "sha256": sha256_file(out / f"window_projection_logistic_{cfg.tag}.csv"),
        },
        "units": units,
        "candidates_primary": [r for r in cand_rows if r["cutoff"] == "primary"],
        "evaluation_design": "Calibration only; evaluation settings are stored separately.",
    }
    _json(dest, payload)
    return dest


RULESETS = {"b-logistic-select-v1": bl.RULES, "b-logistic-select-v2": bl.RULES_V2}


def saved_fits(cal: pathlib.Path, name: str, cname: str) -> dict:
    """TicaFit objects rebuilt from calib_<unit>.npz (top n_keep rho, top n_span embedded)."""
    z = np.load(cal / f"calib_{name}.npz")
    info = json.loads((cal / f"calib_{name}.json").read_text())["fits"]
    fits = {}
    for s in ("full", "A", "B"):
        for tau in bl.RULES["lags"]:
            fi = info[f"{cname}/{s}/lag{tau}"]
            fits[(s, int(tau))] = bl.TicaFit(
                rho=z[f"rho__{cname}__{s}__lag{tau}"],
                embedded=z[f"span__{cname}__{s}__lag{tau}"],
                rank=fi["rank"],
                n_dropped=fi["n_dropped"],
                lag_asymmetry=fi["lag_asymmetry"],
                whitened_asymmetry=fi["whitened_asymmetry"],
                max_abs_rho=fi["max_abs_rho"],
            )
    return fits


def _same_candidates(a: list, b: list) -> None:
    """Raise unless two candidate tables agree (eligibility exactly, numbers to 1e-12 relative)."""
    if len(a) != len(b):
        raise RuntimeError(f"candidate tables differ in length: {len(a)} vs {len(b)}")
    for ra, rb in zip(a, b):
        for k in CAND_FIELDS:
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                if not (math.isclose(va, vb, rel_tol=1e-12, abs_tol=1e-15) or va == vb):
                    raise RuntimeError(f"candidate {ra['lag']},{ra['m']} {k}: {va} vs {vb}")
            elif va != vb:
                raise RuntimeError(f"candidate {ra['lag']},{ra['m']} {k}: {va!r} vs {vb!r}")


def run_amend_select(cfg: DictConfig) -> pathlib.Path:
    """Apply revised selection rules to the saved calibration arrays.

    Check that the arrays reproduce the previous candidate table before selecting
    again. The previous selection file is retained.
    """
    am = cfg.amendment
    rules = RULESETS[str(am.rules)]
    cal = output_root() / str(cfg.study) / "pilot" / f"calibrate_{cfg.tag}"
    status = json.loads((cal / "RUN-STATUS.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"{cal} is {status['status']}; selection needs a COMPLETED calibration")
    out = REPO / str(cfg.calibration.out_dir)
    dest = out / str(am.out_name)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; record further amendments in a new file")
    amended_path = out / str(am.amends_file)
    amended = json.loads(amended_path.read_text())
    old_rules = RULESETS[amended["algorithm"]["version"]]
    if amended["algorithm"] != old_rules:
        raise RuntimeError("the amended file's algorithm differs from the registered rule set")
    tag = str(am.csv_tag)
    fr = cfg.fresh
    cand_rows, win_rows, units, hashes = [], [], {}, {}
    old_cands = {}
    for row in amended["candidates_primary"]:
        old_cands.setdefault(row["unit"], []).append(row)
    for name in cfg.units:
        name = str(name)
        u = Unit(cfg, name)
        rec = json.loads((cal / f"calib_{name}.json").read_text())
        for f in (f"calib_{name}.json", f"calib_{name}.npz"):
            hashes[str(cal / f)] = sha256_file(cal / f)
        hashes[str(u.v1 / "sigma.npz")] = amended["calibration_hashes"][str(u.v1 / "sigma.npz")]
        res = {}
        for cname in ("primary", "sensitivity"):
            fits = saved_fits(cal, name, cname)
            # Check the previous candidate table before applying new selection rules.
            _same_candidates(
                bl.candidate_table(fits, old_rules), rec["results"][cname]["candidates"]
            )
            rows = bl.candidate_table(fits, rules)
            res[cname] = {"candidates": rows, "selection": bl.select_counts(rows, rules)}
            for row in rows:
                cand_rows.append(
                    {"unit": name, "cutoff": cname, **{k: row.get(k, "") for k in CAND_FIELDS}}
                )
        if [
            {k: r.get(k, "") for k in CAND_FIELDS} for r in rec["results"]["primary"]["candidates"]
        ] != [{k: r[k] for k in CAND_FIELDS} for r in old_cands[name]]:
            raise RuntimeError(f"{name}: amended selection file differs from calib json")
        sel, sens = res["primary"]["selection"], res["sensitivity"]["selection"]
        sig = np.load(u.v1 / "sigma.npz")
        grid = [int(t) for t in sig["grid"]]
        meta = u.meta()
        windows = {}
        for m in sel["selected_m"]:
            rows = bl.window_projection(
                sig["O"],
                sig["OA"],
                sig["OB"],
                grid,
                int(m),
                meta["sigma_N_blocks"],
                int(fr.n_blocks),
                int(fr.n_batches),
                float(fr.e_factor),
            )
            for row in rows:
                win_rows.append({"unit": name, **row})
            windows[str(m)] = {
                "t_dk_applicable_fresh": [
                    r["t"] for r in rows if r["t"] >= 1 and r["dk_applicable_fresh"]
                ],
                "t_dk_informative_fresh": [
                    r["t"]
                    for r in rows
                    if r["t"] >= 1 and r["dk_term_deg_fresh"] <= float(fr.dk_informative_deg)
                ],
                "t_dk_applicable_saved": [
                    r["t"] for r in rows if r["t"] >= 1 and r["dk_applicable_saved"]
                ],
            }
        chosen = {
            r["m"]: r
            for r in res["primary"]["candidates"]
            if sel["lag"] is not None and r["lag"] == sel["lag"] and r["m"] in sel["selected_m"]
        }
        units[name] = {
            "model": u.model,
            "dataset": u.dataset,
            "sampler": u.sampler,
            "appendix_stress": u.appendix,
            "feature_rank": meta["feature_rank"],
            "status": sel["status"],
            "lag": sel["lag"],
            "selected_m": sel["selected_m"],
            "selected": {
                str(m): {k: chosen[m].get(k) for k in CAND_FIELDS if k != "reason"}
                for m in sel["selected_m"]
            },
            "eligible": sel["eligible"],
            "sensitivity_cutoff_selection": {k: sens[k] for k in ("status", "lag", "selected_m")},
            "sensitivity_cutoff_agrees": sens["lag"] == sel["lag"]
            and sens["selected_m"] == sel["selected_m"],
            "rule4": rules["rule4"],
            "gap_label": "finite-feature empirical gap (not a certified transition gap)",
            "reference_window": windows,
            "tica_fit_info": {
                k: v for k, v in rec["fits"].items() if k.startswith("primary/full/")
            },
        }
    _write_csv(cand_rows, out / f"calibration_candidates_logistic_{tag}.csv")
    _write_csv(win_rows, out / f"window_projection_logistic_{tag}.csv")
    payload = {
        "status": f"Revised calibration-only selection ({am.id}) from empirical TICA gaps. "
        "The logistic transition spectrum is unknown.",
        "algorithm": rules,
        "amendment": {
            "id": str(am.id),
            "amends": {
                "file": str(am.amends_file),
                "sha256": sha256_file(amended_path),
                "algorithm_version": amended["algorithm"]["version"],
            },
            "reason": str(am.reason),
            "before_any_evaluation": True,
        },
        "calibration_config": OmegaConf.to_container(cfg.calibration),
        "fresh_projection_config": OmegaConf.to_container(cfg.fresh),
        "calibration_hashes": hashes,
        "candidates_csv": {
            "path": f"calibration_candidates_logistic_{tag}.csv",
            "sha256": sha256_file(out / f"calibration_candidates_logistic_{tag}.csv"),
        },
        "window_csv": {
            "path": f"window_projection_logistic_{tag}.csv",
            "sha256": sha256_file(out / f"window_projection_logistic_{tag}.csv"),
        },
        "units": units,
        "candidates_primary": [r for r in cand_rows if r["cutoff"] == "primary"],
        "evaluation_design": "Calibration only; evaluation settings are stored separately.",
    }
    _json(dest, payload)
    return dest


def _grid_to(t_max: int) -> list[int]:
    if t_max < 1 or t_max & (t_max - 1):
        raise ValueError(f"t_max must be a power of two, got {t_max}")
    return [0] + [2**i for i in range(int(math.log2(t_max)) + 1)]


def check_frozen_pairs(cfg: DictConfig) -> dict:
    """Check evaluation pairs and time limits against the saved selection."""
    ev = cfg.evaluation
    out = REPO / str(cfg.calibration.out_dir)
    sel_path = out / str(ev.selection_file)
    if sha256_file(sel_path) != str(ev.selection_sha256):
        raise RuntimeError(f"{sel_path} sha256 differs from the frozen value")
    sel = json.loads(sel_path.read_text())
    win_csv = out / sel["window_csv"]["path"]
    if sha256_file(win_csv) != sel["window_csv"]["sha256"]:
        raise RuntimeError(f"{win_csv} differs from the selection's recorded sha256")
    grids: dict = {}
    with open(win_csv) as f:
        for row in csv.DictReader(f):
            grids.setdefault(row["unit"], set()).add(int(row["t"]))
    derived, not_eval, none = {}, {}, []
    for unit, v in sel["units"].items():
        if v["status"] != "SELECTED":
            none.append(unit)
            continue
        ms = [m for m in v["selected_m"] if v["reference_window"][str(m)]["t_dk_applicable_fresh"]]
        rest = [m for m in v["selected_m"] if m not in ms]
        if rest:
            not_eval[unit] = rest
        if ms:
            last = max(max(v["reference_window"][str(m)]["t_dk_applicable_fresh"]) for m in ms)
            later = sorted(t for t in grids[unit] if t > last)
            derived[unit] = {"m": ms, "t_max": later[0] if later else last, "lag": v["lag"]}
    frozen = {u: {"m": list(p.m), "t_max": int(p.t_max)} for u, p in ev.pairs.items()}
    if {u: {"m": d["m"], "t_max": d["t_max"]} for u, d in derived.items()} != frozen:
        raise RuntimeError(f"frozen pairs {frozen} differ from the selection-derived {derived}")
    if not_eval != {u: list(v) for u, v in ev.not_evaluated.items()}:
        raise RuntimeError(f"not_evaluated {dict(ev.not_evaluated)} differs from {not_eval}")
    if sorted(none) != sorted(ev.none_eligible):
        raise RuntimeError(f"none_eligible differs: {none}")
    for u, d in derived.items():
        if d["lag"] != int(ev.pilots.lag):
            raise RuntimeError(f"{u}: selected lag {d['lag']} != frozen pilot lag {ev.pilots.lag}")
    return sel


class DeviceFeatures:
    """f(x) = k(x, Z) B with the saved R2 basis, on any device (blocked rows)."""

    def __init__(self, sf: "SavedFeatures", device: str, rows: int = 8192):
        self.kernel = sf.kernel
        self.Z = sf.Z.to(device)
        self.B = sf.B.to(device)
        self.rank = sf.rank
        self.rows = rows

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.empty((X.shape[0], self.rank), dtype=F64, device=X.device)
        for i0 in range(0, X.shape[0], self.rows):
            out[i0 : i0 + self.rows] = self.kernel(X[i0 : i0 + self.rows], self.Z) @ self.B
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite features")
        return out


def _fresh_run(cfg: DictConfig, unit: str, n_units: int) -> RunDir:
    rid = f"{unit}_fresh_{cfg.evaluation.tag}"
    phase = str(cfg.phase)
    if phase not in ("smoke", "main"):
        raise ValueError(f"phase must be smoke or main, got {phase!r}")
    path = output_root() / str(cfg.study) / phase / rid
    if (path / "RUN-STATUS.json").exists():
        stored = OmegaConf.create((path / "config.yaml").read_text())
        if OmegaConf.to_yaml(stored.evaluation) != OmegaConf.to_yaml(cfg.evaluation):
            raise RuntimeError(f"{path} exists with a different evaluation config")
        return RunDir.resume(str(cfg.study), phase, rid)
    return RunDir.create(str(cfg.study), phase, rid, OmegaConf.to_yaml(cfg), n_units)


def _fresh_parts(cfg) -> list[str]:
    return [f"batch_{k}.npz" for k in range(int(cfg.evaluation.sigma.n_batches))] + [
        "pilotrun.npz",
        "rows.csv",
    ]


def _done(path: pathlib.Path, cfg) -> int:
    return sum((path / f).exists() for f in _fresh_parts(cfg))


def run_fresh_gpu(cfg: DictConfig) -> pathlib.Path:
    """Fresh branched batches (sigma) and fresh pilot chains for one unit (GPU)."""
    check_frozen_pairs(cfg)
    unit, dev = str(cfg.unit), str(cfg.device)
    ev = cfg.evaluation
    pair = ev.pairs[unit]
    u = Unit(cfg, unit)
    run = _fresh_run(cfg, unit, len(_fresh_parts(cfg)))
    path = run.path

    try:
        cid = f"gc-{unit}"
        post = gc.make_posterior(u.model)
        setup = json.loads((u.v1 / "setup.json").read_text())
        scales = np.asarray(setup["kernel"]["scales"], dtype=np.float64)
        pool = load_set(set_path("reference", str(u.cfg.references[u.model].pool)))["draws"]
        R, _, d = pool.shape
        target = r2.PosteriorTarget(post, pool.reshape(-1, d))
        feats = DeviceFeatures(SavedFeatures(cfg, u), dev)
        grid = _grid_to(int(pair.t_max))
        sg = ev.sigma
        N = int(sg.n_blocks)
        for k in range(int(sg.n_batches)):
            f = path / f"batch_{k}.npz"
            if f.exists():
                continue
            xr, xpr = (list(map(int, v)) for v in sg.pool_runs[k])
            if max(xr + xpr) >= R or set(xr) & set(xpr):
                raise ValueError(f"bad pool-run subsets {xr} / {xpr} for R={R}")
            rng = np_gen(seed_for(cid, str(sg.start_seed_stage), k))
            P, Pp = pool[xr].reshape(-1, d), pool[xpr].reshape(-1, d)
            X, Xp = P[rng.integers(0, P.shape[0], N)], Pp[rng.integers(0, Pp.shape[0], N)]
            x0 = torch.as_tensor(np.concatenate([X, X, Xp, Xp]), dtype=F64, device=dev)
            ops = {}

            def cb(t, x):
                ops[t] = r2.branched_operator_stats(feats(x), N)["O"]

            smp = make_sampler(u.sampler, target, u.h, scales)
            t0 = time.time()
            out = smp.run(
                x0,
                grid,
                torch_gen(seed_for(cid, str(sg.seed_stage), k), dev),
                store=False,
                on_checkpoint=cb,
            )
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            wall = time.time() - t0
            tmp = path / f".tmp_{f.name}"  # ends in .npz, so np.savez keeps the name
            np.savez(tmp, grid=np.asarray(grid), O=np.stack([ops[t] for t in grid]))
            os.replace(tmp, f)
            _json(
                path / f"batch_{k}.json",
                {
                    "batch": k,
                    "N_blocks": N,
                    "x_pool_runs": xr,
                    "xprime_pool_runs": xpr,
                    "seed": seed_for(cid, str(sg.seed_stage), k),
                    "start_seed": seed_for(cid, str(sg.start_seed_stage), k),
                    "accept_rate": out["accept_rate"],
                    "counters": out["counters"],
                    "wall_s": wall,
                    "grid": grid,
                },
            )
            run.update(completed_units=_done(path, cfg), note=f"batch {k} {wall:.0f}s")
        f = path / "pilotrun.npz"
        if not f.exists():
            p = ev.pilots
            n_fit, C = int(p.n_fit), int(p.n_chains)
            Lmax = max(int(v) for v in p.lengths)
            laws = r2.StartLaws(pool.reshape(-1, d), scales)
            x0 = laws.sample(
                str(p.start_law),
                n_fit * C,
                torch_gen(seed_for(cid, str(p.start_seed_stage), 0), dev),
            )
            traj = torch.empty((Lmax + 1, n_fit * C, d), dtype=F64, device=dev)

            def cbp(t, x):
                traj[t] = x

            smp = make_sampler(u.sampler, target, u.h, scales)
            t0 = time.time()
            out = smp.run(
                x0,
                range(Lmax + 1),
                torch_gen(seed_for(cid, str(p.seed_stage), 0), dev),
                store=False,
                on_checkpoint=cbp,
            )
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            wall = time.time() - t0
            Xall = traj.transpose(0, 1).contiguous().cpu().numpy()
            tmp = path / ".tmp_pilotrun.npz"
            np.savez(tmp, X=Xall)
            os.replace(tmp, f)
            _json(
                path / "pilotrun.json",
                {
                    "n_fit": n_fit,
                    "n_chains": C,
                    "Lmax": Lmax,
                    "start_law": str(p.start_law),
                    "seed": seed_for(cid, str(p.seed_stage), 0),
                    "start_seed": seed_for(cid, str(p.start_seed_stage), 0),
                    "accept_rate": out["accept_rate"],
                    "counters": out["counters"],
                    "wall_s": wall,
                    "never_accepted_chains": int((out["accept_count_per_chain"] == 0).sum().item()),
                },
            )
            peak = (
                torch.cuda.max_memory_allocated() / 2**20
                if dev.startswith("cuda")
                else float("nan")
            )
            run.update(
                completed_units=_done(path, cfg),
                note=f"pilots {wall:.0f}s; gpu peak {peak:.0f} MiB",
            )
        run.finish("INCOMPLETE", note="gpu stage done; cpu stage pending")
    except BaseException as exc:
        run.finish("FAILED", note=f"[gpu] {exc!r}"[:500])
        raise
    return path


def learned_spans(cfg: DictConfig, sf: "SavedFeatures", X: np.ndarray, ms: list) -> tuple:
    """Fresh learned TICA spans per fit and nested pilot length (frozen lag and cutoff).

    Returns ({(fit, L, m): orthonormal (r, m) or None}, [per-fit info]); failures are kept.
    """
    p = cfg.evaluation.pilots
    n_fit, C = int(p.n_fit), int(p.n_chains)
    lengths = sorted(int(v) for v in p.lengths)
    Lmax, lag, m_max = lengths[-1], int(p.lag), max(ms)
    d = X.shape[2]
    if X.shape != (n_fit * C, Lmax + 1, d):
        raise RuntimeError(f"pilot paths have shape {X.shape}")
    spans, info = {}, []
    for i in range(n_fit):
        Xi = np.ascontiguousarray(X[i * C : (i + 1) * C, 1:])
        F = sf(Xi.reshape(-1, d)).reshape(C, Lmax, sf.rank)
        for L in lengths:
            C0, C1, asym = r2.multichain_lagged(F[:, :L], lag)
            rec = {"fit": i, "L": L}
            try:
                modes = sm.tica_symmetric(
                    C0,
                    C1,
                    float(p.tica_rel_cutoff),
                    n_modes=m_max,
                    order="abs",
                    lag_asymmetry=asym,
                )
            except sm.RankDeficientError as exc:
                rec["failed"] = repr(exc)
                for m in ms:
                    spans[(i, L, m)] = None
                info.append(rec)
                continue
            rec.update(
                rho=modes.eigenvalues.tolist(), rank=modes.whitening.rank, lag_asymmetry=asym
            )
            for m in ms:
                try:
                    spans[(i, L, m)] = sm.orthonormal_basis(
                        np.ascontiguousarray(modes.embedded[:, :m])
                    )
                except sm.RankDeficientError as exc:
                    spans[(i, L, m)] = None
                    rec[f"m{m}_failed"] = repr(exc)
            info.append(rec)
    return spans, info


REF_FIELDS = (
    "unit",
    "m",
    "t",
    "lambda_m",
    "lambda_m1",
    "gap",
    "e",
    "dk_deg",
    "applicable",
    "stable",
    "most_negative",
    "half1_batches",
    "half1_gap",
    "half1_e",
    "half1_dk_deg",
    "half1_angle_to_full_deg",
    "half1_consistent",
    "half2_batches",
    "half2_gap",
    "half2_e",
    "half2_dk_deg",
    "half2_angle_to_full_deg",
    "half2_consistent",
    "projected_e_fresh",
    "projected_dk_deg_fresh",
    "witness_diff_op",
    "witness_eps_fresh",
    "witness_eps_r2",
    "witness_ratio",
    "cutoff",
)


def run_fresh_cpu(cfg: DictConfig) -> pathlib.Path:
    """Learned spans, reference optimum, DK envelope rows and the stability check (CPU)."""
    sel = check_frozen_pairs(cfg)
    unit = str(cfg.unit)
    if str(cfg.device) != "cpu":
        raise ValueError("stage=fresh_cpu requires device=cpu")
    ev = cfg.evaluation
    pair = ev.pairs[unit]
    ms = [int(m) for m in pair.m]
    u = Unit(cfg, unit)
    run = _fresh_run(cfg, unit, len(_fresh_parts(cfg)))
    path = run.path
    try:
        if _done(path, cfg) < len(_fresh_parts(cfg)) - 1:
            raise RuntimeError(f"{path}: GPU parts missing")
        meta = u.meta()
        grid = _grid_to(int(pair.t_max))
        sg = ev.sigma
        K = int(sg.n_batches)
        B = []
        for k in range(K):
            z = np.load(path / f"batch_{k}.npz")
            if list(z["grid"]) != grid:
                raise RuntimeError(f"batch {k} grid differs from the frozen grid")
            B.append(z["O"])
        B = np.stack(B)  # (K, n_t, r, r)
        r2sig = np.load(u.v1 / "sigma.npz")
        r2grid = [int(t) for t in r2sig["grid"]]
        proj = {}
        with open(REPO / str(cfg.calibration.out_dir) / sel["window_csv"]["path"]) as f:
            for row in csv.DictReader(f):
                if row["unit"] == unit:
                    proj[(int(row["m"]), int(row["t"]))] = row
        halves = [tuple(map(int, h)) for h in sg.halves]
        hb = [tuple(map(int, h)) for h in sg.half_budgets]
        refs, ref_rows = {}, []
        for m in ms:
            for i, t in enumerate(grid):
                refs[(m, t)] = bl.reference_at_t(B[:, i], m, halves, hb, float(sg.e_factor))
            cut = bl.curve_cutoffs(grid, {t: refs[(m, t)]["applicable"] for t in grid})
            for i, t in enumerate(grid):
                ref = refs[(m, t)]
                S = B[:, i].mean(0)
                eps_f = (
                    float(
                        np.linalg.norm(
                            B[list(halves[0]), i].mean(0) - B[list(halves[1]), i].mean(0), 2
                        )
                    )
                    / 2
                )
                j = r2grid.index(t)
                eps_r2 = float(np.linalg.norm(r2sig["OA"][j] - r2sig["OB"][j], 2)) / 2
                diff = float(np.linalg.norm(S - r2sig["O"][j], 2))
                pr = proj[(m, t)]
                row = {
                    "unit": unit,
                    "m": m,
                    "t": t,
                    **{
                        k: ref[k]
                        for k in (
                            "lambda_m",
                            "lambda_m1",
                            "gap",
                            "e",
                            "dk_deg",
                            "applicable",
                            "stable",
                            "most_negative",
                        )
                    },
                    "projected_e_fresh": float(pr["e_fresh"]),
                    "projected_dk_deg_fresh": float(pr["dk_term_deg_fresh"]),
                    "witness_diff_op": diff,
                    "witness_eps_fresh": eps_f,
                    "witness_eps_r2": eps_r2,
                    "witness_ratio": diff / math.hypot(eps_f, eps_r2),
                    "cutoff": cut[t],
                }
                for q, h in enumerate(ref["half_budget"], start=1):
                    row.update(
                        {
                            f"half{q}_batches": "+".join(map(str, h["batches"])),
                            f"half{q}_gap": h["gap"],
                            f"half{q}_e": h["e"],
                            f"half{q}_dk_deg": h["dk_deg"],
                            f"half{q}_angle_to_full_deg": h["angle_to_full_deg"],
                            f"half{q}_consistent": h["consistent"],
                        }
                    )
                ref_rows.append({k: row.get(k) for k in REF_FIELDS})
        t0 = time.time()
        sf = SavedFeatures(cfg, u)
        X = np.load(path / "pilotrun.npz")["X"]
        spans, info = learned_spans(cfg, sf, X, ms)
        t_learn = time.time() - t0
        rows = []
        lengths = sorted(int(v) for v in ev.pilots.lengths)
        for m in ms:
            cut = bl.curve_cutoffs(grid, {t: refs[(m, t)]["applicable"] for t in grid})
            for t in grid:
                for L in lengths:
                    for i in range(int(ev.pilots.n_fit)):
                        base = {
                            "model": u.model,
                            "dataset": u.dataset,
                            "sampler": u.sampler,
                            "feature_rank": meta["feature_rank"],
                            "m": m,
                            "t": t,
                            "fit": i,
                            "pilot_length": L,
                            "lag": int(ev.pilots.lag),
                        }
                        rows.append(bl.envelope_row(base, spans[(i, L, m)], refs[(m, t)], cut[t]))
        _write_csv(ref_rows, path / "reference.csv")
        _json(path / "learned.json", {"wall_learn_s": t_learn, "fits": info})
        _write_csv(rows, path / "rows.csv")
        run.update(
            completed_units=_done(path, cfg),
            note=f"cpu stage: {len(rows)} rows, learn {t_learn:.0f}s",
        )
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"[cpu] {exc!r}"[:500])
        raise
    return path


def _q(vals, q):
    v = [x for x in vals if not math.isnan(x)]
    return float(np.quantile(v, q)) if v else math.nan


def _batch_settings(batch: dict) -> dict:
    """Keep the sampling design and random seeds in the exported data."""
    fields = ("batch", "N_blocks", "grid", "seed", "seed_stage", "start_seed",
              "x_pool_runs", "xprime_pool_runs", "starts", "n_sub_batches", "sub_batch_blocks")
    return {key: batch[key] for key in fields if key in batch}


def run_publish(cfg: DictConfig) -> pathlib.Path:
    """Export completed runs to CSV without overwriting existing results."""
    sel = check_frozen_pairs(cfg)
    ev = cfg.evaluation
    out = REPO / str(cfg.calibration.out_dir)
    names = (
        "angles_logistic.csv",
        "reference_logistic.csv",
        "status_logistic.csv",
        "summary_logistic.csv",
        "SOURCE_logistic.json",
    )
    for n in names:
        if (out / n).exists():
            raise FileExistsError(f"{out / n} exists; publish into new names")
    rows, refs, status, summ, runs = [], [], [], [], {}
    for unit, pair in ev.pairs.items():
        if str(cfg.phase) != "main":
            raise ValueError("publish reads phase=main runs only")
        path = output_root() / str(cfg.study) / "main" / f"{unit}_fresh_{ev.tag}"
        stt = json.loads((path / "RUN-STATUS.json").read_text())
        if stt["status"] != "COMPLETED":
            raise RuntimeError(f"{path} is {stt['status']}")
        runs[unit] = {
            "batches": [
                _batch_settings(json.loads((path / f"batch_{k}.json").read_text()))
                for k in range(int(ev.sigma.n_batches))
            ],
        }
        with open(path / "rows.csv") as f:
            urows = list(csv.DictReader(f))
        with open(path / "reference.csv") as f:
            urefs = list(csv.DictReader(f))
        rows += urows
        refs += urefs
        for m in [int(v) for v in pair.m]:
            mr = [r for r in urefs if int(r["m"]) == m]
            shown = [int(r["t"]) for r in mr if r["cutoff"] == ""]
            status.append(
                {
                    "unit": unit,
                    "m": m,
                    "status": "EVALUATED",
                    "displayed_t_first": min(shown) if shown else "",
                    "displayed_t_last": max(shown) if shown else "",
                    "n_displayed": len(shown),
                    "n_checkpoints": len(mr),
                    "n_dk_applicable": sum(r["applicable"] == "True" for r in mr),
                    "n_half_budget_inconsistent_displayed": sum(
                        r["stable"] != "True" for r in mr if r["cutoff"] == ""
                    ),
                    "note": "reference-based empirical sensitivity envelope; fresh 4 x 8192 branched batches",
                }
            )
            for t in sorted({int(r["t"]) for r in urows if int(r["m"]) == m}):
                for L in sorted({int(r["pilot_length"]) for r in urows}):
                    g = [
                        r
                        for r in urows
                        if int(r["m"]) == m and int(r["t"]) == t and int(r["pilot_length"]) == L
                    ]
                    ang = [float(r["angle_deg"]) for r in g]
                    up = [float(r["learned_angle_upper"]) for r in g]
                    summ.append(
                        {
                            "unit": unit,
                            "m": m,
                            "t": t,
                            "pilot_length": L,
                            "n_fits": len(g),
                            "n_failed": sum(r["estimate_status"] == "FAILED_FIT" for r in g),
                            "angle_median": _q(ang, 0.5),
                            "angle_q25": _q(ang, 0.25),
                            "angle_q75": _q(ang, 0.75),
                            "upper_median": _q(up, 0.5),
                            "upper_q25": _q(up, 0.25),
                            "upper_q75": _q(up, 0.75),
                            "estimate_status": ";".join(sorted({r["estimate_status"] for r in g})),
                            "cutoff": next(
                                rr["cutoff"]
                                for rr in urefs
                                if int(rr["m"]) == m and int(rr["t"]) == t
                            ),
                            "note": "median / IQR over fits; not confidence intervals",
                        }
                    )
    lag2t = {}
    with open(out / "lag2t_check_logistic_v2.csv") as f:
        for r in csv.DictReader(f):
            lag2t[(r["unit"], int(r["m"]), int(r["t"]))] = r
    win = {}
    with open(out / sel["window_csv"]["path"]) as f:
        for r in csv.DictReader(f):
            win.setdefault((r["unit"], int(r["m"])), []).append(r)
    for unit, mlist in ev.not_evaluated.items():
        for m in [int(v) for v in mlist]:
            w = win[(unit, m)]
            best = max((float(r["snr_fresh"]), int(r["t"])) for r in w if int(r["t"]) >= 1)
            l2 = [lag2t[(unit, m, t)] for t in (2, 8) if (unit, m, t) in lag2t]
            status.append(
                {
                    "unit": unit,
                    "m": m,
                    "status": "NOT_EVALUATED_NO_PROJECTED_WINDOW",
                    "note": f"projected best gap/e at 4x8192 = {best[0]:.2f} (t={best[1]}) < 2; "
                    + "; ".join(
                        f"lag-2t Sigma_t gap/eps {float(r['gap_lag2t']) / float(r['eps_lag2t']):.1f} vs "
                        f"branched {float(r['gap_branched']) / float(r['eps_branched']):.1f} at t={r['t']}"
                        for r in l2
                    )
                    + " (calibration only: the gap is real; the branched reference noise is the limit)",
                }
            )
    for unit in ev.none_eligible:
        status.append(
            {
                "unit": unit,
                "m": "",
                "status": "NONE_ELIGIBLE",
                "note": "no (lag, m) passes rules 1-2 (half-split embedded spans 80-90 deg apart at every m and lag)",
            }
        )
    rows.sort(
        key=lambda r: (
            r["dataset"],
            r["sampler"],
            int(r["m"]),
            int(r["t"]),
            int(r["pilot_length"]),
            int(r["fit"]),
        )
    )
    with open(out / "angles_logistic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bl.ANGLE_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    _write_csv(refs, out / "reference_logistic.csv")
    _write_csv(status, out / "status_logistic.csv")
    _write_csv(summ, out / "summary_logistic.csv")
    source = {
        "status": "Learned lag-1 TICA spans compared with the estimated optimum V_m(S_t-hat). "
        "The Davis-Kahan envelope uses empirical sensitivity e_t = 3 ||S^A - S^B||_op / 2; "
        "it is not a certified bound or confidence interval. The theorem bound is unavailable "
        "because the logistic transition spectrum is unknown.",
        "selection": {
            "file": str(ev.selection_file),
            "sha256": str(ev.selection_sha256),
        },
        "evaluation_config": OmegaConf.to_container(ev),
        "covariance_source": "branched-pair Sigma_t-hat (german_credit_r2.branched_operator_stats): per "
        "batch N blocks, starts X / X' drawn with replacement from disjoint NUTS pool-run subsets "
        "(cross-check set <dataset>_nc_diag_reference_v1); halves = batches {0,1} vs {2,3}; shared "
        "NUTS pool -> reference bias unquantified",
        "samplers": {
            u: {
                "kind": Unit(cfg, u).sampler,
                "h": Unit(cfg, u).h,
                "preconditioner": "diag(frozen MAD kernel scales)",
            }
            for u in ev.pairs
        },
        "features": "512 Nystrom landmarks (evenly spaced calibration draws), MAD-scaled RBF, saved "
        "R2 basis (german_credit_r2/main/<unit>_v1/setup.npz); fixed finite feature geometry, not "
        "full-RKHS angles",
        "numerical_method": "float64 throughout (GPU sampling and branched sums, CPU eigh); "
        "angles by the sine route sigma_max((I - V V^T) W)",
        "margins": "no numerical margin (empirical_sensitivity = e_t); budget-doubling check: half "
        "budgets {0,2} and {1,3} must be DK-applicable with angle(V_full, V_half) <= dk_full + dk_half",
        "columns_not_applicable": [
            "sin_angle_upper",
            "angle_upper",
            "mode_error_deg",
            "rate_a",
            "rate_b",
            "c_m_squared",
            "kappa_squared",
            "numerical_allowance",
        ],
        "runs": runs,
        "outputs_sha256": {n: sha256_file(out / n) for n in names[:-1]},
    }
    _json(out / "SOURCE_logistic.json", source)
    return out / "SOURCE_logistic.json"


def expanded_design(cfg: DictConfig) -> dict:
    ex = cfg.expanded
    p = REPO / str(cfg.calibration.out_dir) / str(ex.design_file)
    if sha256_file(p) != str(ex.design_sha256):
        raise RuntimeError(f"{p} sha256 differs from the frozen value")
    return json.loads(p.read_text())


def _expanded_run(cfg: DictConfig, unit: str, n_units: int) -> RunDir:
    rid = f"{unit}_expanded_{cfg.expanded.tag}"
    phase = str(cfg.phase)
    if phase not in ("smoke", "main"):
        raise ValueError(f"phase must be smoke or main, got {phase!r}")
    path = output_root() / str(cfg.study) / phase / rid
    if (path / "RUN-STATUS.json").exists():
        stored = OmegaConf.create((path / "config.yaml").read_text())
        for key in ("evaluation", "expanded"):
            if OmegaConf.to_yaml(stored[key]) != OmegaConf.to_yaml(cfg[key]):
                raise RuntimeError(f"{path} exists with a different {key} config")
        return RunDir.resume(str(cfg.study), phase, rid)
    return RunDir.create(str(cfg.study), phase, rid, OmegaConf.to_yaml(cfg), n_units)


def protocol_pilot_file(cfg: DictConfig, unit: str) -> pathlib.Path | None:
    """Return the completed protocol run's pilot file, or None if no run exists."""
    d = output_root() / str(cfg.study) / "main" / f"{unit}_fresh_{cfg.evaluation.tag}"
    if not (d / "RUN-STATUS.json").exists():
        return None
    if json.loads((d / "RUN-STATUS.json").read_text())["status"] != "COMPLETED":
        raise RuntimeError(f"{d} exists but is not COMPLETED")
    return d / "pilotrun.npz"


def _expanded_parts(cfg: DictConfig, unit: str) -> list[str]:
    parts = [f"batch_{k}.npz" for k in range(int(cfg.expanded.n_batches))]
    if protocol_pilot_file(cfg, unit) is None:
        parts.append("pilotrun.npz")
    return parts + ["rows.csv"]


def expanded_starts(cfg: DictConfig, u: "Unit", plan: dict, k: int, d: int):
    """(X, X', provenance) for batch k: new-pool quarters without replacement, or the R2 pool."""
    ex = cfg.expanded
    cid, nb = f"gc-{u.name}", int(plan["N_batch"])
    if plan["new_pool"]:
        npool = ex.new_pools[u.model]
        draws = load_set(set_path("reference", str(npool.set)))["draws"]
        R = draws.shape[0]
        if R != int(npool.n_runs) or R % 4:
            raise RuntimeError(f"new pool {npool.set} has {R} runs, expected {npool.n_runs}")
        q = R // 4
        h, i = divmod(k, 2)  # half h (batches {0,1} / {2,3}), position i within the half
        SX = draws[h * q : (h + 1) * q].reshape(-1, d)
        SXp = draws[(2 + h) * q : (3 + h) * q].reshape(-1, d)
        if 2 * nb > min(SX.shape[0], SXp.shape[0]):
            raise RuntimeError(f"pool quarter holds fewer than {2 * nb} states")
        pX = np_gen(seed_for(cid, f"{ex.perm_seed_stage}-X", h)).permutation(SX.shape[0])
        pXp = np_gen(seed_for(cid, f"{ex.perm_seed_stage}-Xp", h)).permutation(SXp.shape[0])
        X, Xp = SX[pX[i * nb : (i + 1) * nb]], SXp[pXp[i * nb : (i + 1) * nb]]
        prov = {
            "pool": str(npool.set),
            "x_runs": list(range(h * q, (h + 1) * q)),
            "xprime_runs": list(range((2 + h) * q, (3 + h) * q)),
            "sampling": "without replacement within the quarter (disjoint across the half's batches)",
        }
    else:
        pool = load_set(set_path("reference", str(u.cfg.references[u.model].pool)))["draws"]
        xr, xpr = (list(map(int, v)) for v in ex.pool_runs[k])
        if max(xr + xpr) >= pool.shape[0] or set(xr) & set(xpr):
            raise ValueError(f"bad pool-run subsets {xr} / {xpr}")
        rng = np_gen(seed_for(cid, str(ex.start_seed_stage), k))
        P, Pp = pool[xr].reshape(-1, d), pool[xpr].reshape(-1, d)
        X, Xp = P[rng.integers(0, P.shape[0], nb)], Pp[rng.integers(0, Pp.shape[0], nb)]
        prov = {
            "pool": str(u.cfg.references[u.model].pool),
            "x_runs": xr,
            "xprime_runs": xpr,
            "sampling": "with replacement (R2 pool, as the protocol tier)",
        }
    return X, Xp, prov


def run_expanded_gpu(cfg: DictConfig) -> pathlib.Path:
    """Sample larger branched batches, reusing existing pilot paths where available."""
    check_frozen_pairs(cfg)
    design = expanded_design(cfg)
    unit, dev = str(cfg.unit), str(cfg.device)
    plan = design["units"][unit]
    u = Unit(cfg, unit)
    parts = _expanded_parts(cfg, unit)
    run = _expanded_run(cfg, unit, len(parts))
    path = run.path

    done = lambda: sum((path / f).exists() for f in parts)  # noqa: E731
    try:
        cid = f"gc-{unit}"
        ex = cfg.expanded
        post = gc.make_posterior(u.model)
        setup = json.loads((u.v1 / "setup.json").read_text())
        scales = np.asarray(setup["kernel"]["scales"], dtype=np.float64)
        pool = load_set(set_path("reference", str(u.cfg.references[u.model].pool)))["draws"]
        d = pool.shape[2]
        target = r2.PosteriorTarget(post, pool.reshape(-1, d))
        feats = DeviceFeatures(SavedFeatures(cfg, u), dev)
        grid = _grid_to(int(plan["t_max"]))
        nb, sub = int(plan["N_batch"]), int(design["sub_batch_blocks"])
        for k in range(int(ex.n_batches)):
            f = path / f"batch_{k}.npz"
            if f.exists():
                continue
            X, Xp, prov = expanded_starts(cfg, u, plan, k, d)
            acc = {t: np.zeros((feats.rank, feats.rank)) for t in grid}
            t0, counters, acc_rate = time.time(), [], []
            for j, s0 in enumerate(range(0, nb, sub)):
                n = min(sub, nb - s0)
                x0 = torch.as_tensor(
                    np.concatenate(
                        [X[s0 : s0 + n], X[s0 : s0 + n], Xp[s0 : s0 + n], Xp[s0 : s0 + n]]
                    ),
                    dtype=F64,
                    device=dev,
                )
                ops = {}

                def cb(t, x, n=n, ops=ops):
                    ops[t] = r2.branched_operator_stats(feats(x), n)["O"]

                smp = make_sampler(u.sampler, target, u.h, scales)
                out = smp.run(
                    x0,
                    grid,
                    torch_gen(seed_for(cid, f"{ex.seed_stage}-b{k}", j), dev),
                    store=False,
                    on_checkpoint=cb,
                )
                for t in grid:
                    acc[t] += n * ops[t]
                counters.append(out["counters"])
                acc_rate.append(out["accept_rate"])
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            wall = time.time() - t0
            tmp = path / f".tmp_{f.name}"
            np.savez(tmp, grid=np.asarray(grid), O=np.stack([acc[t] / nb for t in grid]))
            os.replace(tmp, f)
            _json(
                path / f"batch_{k}.json",
                {
                    "batch": k,
                    "N_blocks": nb,
                    "sub_batch_blocks": sub,
                    "n_sub_batches": len(counters),
                    "starts": prov,
                    "seed_stage": f"{ex.seed_stage}-b{k}",
                    "accept_rate_mean": float(np.mean(acc_rate)),
                    "chain_steps": int(sum(c["chain_steps"] for c in counters)),
                    "wall_s": wall,
                    "grid": grid,
                },
            )
            run.update(completed_units=done(), note=f"batch {k} {wall:.0f}s")
        if "pilotrun.npz" in parts and not (path / "pilotrun.npz").exists():
            p = cfg.evaluation.pilots
            n_fit, C = int(p.n_fit), int(p.n_chains)
            Lmax = max(int(v) for v in p.lengths)
            laws = r2.StartLaws(pool.reshape(-1, d), scales)
            x0 = laws.sample(
                str(p.start_law),
                n_fit * C,
                torch_gen(seed_for(cid, str(p.start_seed_stage), 0), dev),
            )
            traj = torch.empty((Lmax + 1, n_fit * C, d), dtype=F64, device=dev)

            def cbp(t, x):
                traj[t] = x

            smp = make_sampler(u.sampler, target, u.h, scales)
            t0 = time.time()
            out = smp.run(
                x0,
                range(Lmax + 1),
                torch_gen(seed_for(cid, str(p.seed_stage), 0), dev),
                store=False,
                on_checkpoint=cbp,
            )
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            wall = time.time() - t0
            tmp = path / ".tmp_pilotrun.npz"
            np.savez(tmp, X=traj.transpose(0, 1).contiguous().cpu().numpy())
            os.replace(tmp, path / "pilotrun.npz")
            _json(
                path / "pilotrun.json",
                {
                    "n_fit": n_fit,
                    "n_chains": C,
                    "Lmax": Lmax,
                    "start_law": str(p.start_law),
                    "seed": seed_for(cid, str(p.seed_stage), 0),
                    "start_seed": seed_for(cid, str(p.start_seed_stage), 0),
                    "accept_rate": out["accept_rate"],
                    "counters": out["counters"],
                    "wall_s": wall,
                    "never_accepted_chains": int((out["accept_count_per_chain"] == 0).sum().item()),
                },
            )
            run.update(completed_units=done(), note=f"pilots {wall:.0f}s")
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev.startswith("cuda") else float("nan")
        run.finish("INCOMPLETE", note=f"gpu stage done (peak {peak:.0f} MiB); cpu stage pending")
    except BaseException as exc:
        run.finish("FAILED", note=f"[gpu] {exc!r}"[:500])
        raise
    return path


def run_expanded_cpu(cfg: DictConfig) -> pathlib.Path:
    """Expanded tier: the same learned spans and envelope rules against the larger reference."""
    sel = check_frozen_pairs(cfg)
    design = expanded_design(cfg)
    unit = str(cfg.unit)
    if str(cfg.device) != "cpu":
        raise ValueError("stage=expanded_cpu requires device=cpu")
    plan = design["units"][unit]
    ev, ex = cfg.evaluation, cfg.expanded
    ms = [int(m) for m in plan["m"]]
    nb = int(plan["N_batch"])
    budget = f"expanded_{4 * nb}"
    u = Unit(cfg, unit)
    parts = _expanded_parts(cfg, unit)
    run = _expanded_run(cfg, unit, len(parts))
    path = run.path
    try:
        if sum((path / f).exists() for f in parts[:-1]) != len(parts) - 1:
            raise RuntimeError(f"{path}: GPU parts missing")
        meta = u.meta()
        grid = _grid_to(int(plan["t_max"]))
        K = int(ex.n_batches)
        B = []
        for k in range(K):
            z = np.load(path / f"batch_{k}.npz")
            if list(z["grid"]) != grid:
                raise RuntimeError(f"batch {k} grid differs from the frozen grid")
            B.append(z["O"])
        B = np.stack(B)
        r2sig = np.load(u.v1 / "sigma.npz")
        r2grid = [int(t) for t in r2sig["grid"]]
        esaved = {}
        with open(REPO / str(cfg.calibration.out_dir) / sel["window_csv"]["path"]) as f:
            for row in csv.DictReader(f):
                if row["unit"] == unit:
                    esaved[(int(row["m"]), int(row["t"]))] = float(row["e_saved"])
        sg = ev.sigma
        halves = [tuple(map(int, h)) for h in sg.halves]
        hb = [tuple(map(int, h)) for h in sg.half_budgets]
        refs, ref_rows = {}, []
        for m in ms:
            for i, t in enumerate(grid):
                refs[(m, t)] = bl.reference_at_t(B[:, i], m, halves, hb, float(sg.e_factor))
            cut = bl.curve_cutoffs(grid, {t: refs[(m, t)]["applicable"] for t in grid})
            for i, t in enumerate(grid):
                ref = refs[(m, t)]
                S = B[:, i].mean(0)
                eps_f = 0.5 * float(
                    np.linalg.norm(B[list(halves[0]), i].mean(0) - B[list(halves[1]), i].mean(0), 2)
                )
                j = r2grid.index(t)
                eps_r2 = float(np.linalg.norm(r2sig["OA"][j] - r2sig["OB"][j], 2)) / 2
                diff = float(np.linalg.norm(S - r2sig["O"][j], 2))
                e_proj = esaved[(m, t)] * math.sqrt(8192 / (4 * nb))
                row = {
                    "budget": budget,
                    "unit": unit,
                    "m": m,
                    "t": t,
                    **{
                        k: ref[k]
                        for k in (
                            "lambda_m",
                            "lambda_m1",
                            "gap",
                            "e",
                            "dk_deg",
                            "applicable",
                            "stable",
                            "most_negative",
                        )
                    },
                    "projected_e_fresh": e_proj,
                    "projected_dk_deg_fresh": bl.dk_term_deg(ref["gap"], e_proj),
                    "witness_diff_op": diff,
                    "witness_eps_fresh": eps_f,
                    "witness_eps_r2": eps_r2,
                    "witness_ratio": diff / math.hypot(eps_f, eps_r2),
                    "cutoff": cut[t],
                }
                for q, h in enumerate(ref["half_budget"], start=1):
                    row.update(
                        {
                            f"half{q}_batches": "+".join(map(str, h["batches"])),
                            f"half{q}_gap": h["gap"],
                            f"half{q}_e": h["e"],
                            f"half{q}_dk_deg": h["dk_deg"],
                            f"half{q}_angle_to_full_deg": h["angle_to_full_deg"],
                            f"half{q}_consistent": h["consistent"],
                        }
                    )
                ref_rows.append({"budget": budget, **{k: row.get(k) for k in REF_FIELDS}})
        t0 = time.time()
        sf = SavedFeatures(cfg, u)
        pfile = protocol_pilot_file(cfg, unit) or path / "pilotrun.npz"
        X = np.load(pfile)["X"]
        spans, info = learned_spans(cfg, sf, X, ms)
        t_learn = time.time() - t0
        rows = []
        lengths = sorted(int(v) for v in ev.pilots.lengths)
        for m in ms:
            cut = bl.curve_cutoffs(grid, {t: refs[(m, t)]["applicable"] for t in grid})
            for t in grid:
                for L in lengths:
                    for i in range(int(ev.pilots.n_fit)):
                        base = {
                            "model": u.model,
                            "dataset": u.dataset,
                            "sampler": u.sampler,
                            "feature_rank": meta["feature_rank"],
                            "m": m,
                            "t": t,
                            "fit": i,
                            "pilot_length": L,
                            "lag": int(ev.pilots.lag),
                        }
                        r = bl.envelope_row(base, spans[(i, L, m)], refs[(m, t)], cut[t])
                        rows.append({**r, "budget": budget})
        _write_csv(ref_rows, path / "reference.csv")
        _json(
            path / "learned.json",
            {
                "wall_learn_s": t_learn,
                "pilot_file": str(pfile),
                "pilot_sha256": sha256_file(pfile),
                "fits": info,
            },
        )
        _write_csv(rows, path / "rows.csv")
        run.update(
            completed_units=len(parts), note=f"cpu stage: {len(rows)} rows, learn {t_learn:.0f}s"
        )
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"[cpu] {exc!r}"[:500])
        raise
    return path


def _load_protocol_export(out: pathlib.Path, budget: str) -> tuple[list[dict], dict, str]:
    """Recover the protocol table from either a single-budget or combined export."""
    source = json.loads((out / "SOURCE_logistic.json").read_text())
    raw = (out / "angles_logistic.csv").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != source["outputs_sha256"]["angles_logistic.csv"]:
        raise RuntimeError("angles_logistic.csv does not match SOURCE_logistic.json")
    reader = csv.DictReader(io.StringIO(raw.decode()))
    combined = "budget" in reader.fieldnames
    rows = [
        {k: row[k] for k in bl.ANGLE_COLUMNS}
        for row in reader
        if not combined or row["budget"] == budget
    ]
    if not rows:
        raise ValueError(f"no rows for {budget} in angles_logistic.csv")
    if combined:
        # The combined export distinguishes two cases of the original half-budget flag.
        for row in rows:
            row["cutoff_reason"] = row["cutoff_reason"].replace(
                "HALF_BUDGET_UNRESOLVED (a half-budget eigengap <= 2 e_t; not an angular disagreement)",
                "DK_INCONSISTENT_AT_HALF_BUDGET",
            ).replace("HALF_BUDGET_ANGULAR_DISAGREEMENT", "DK_INCONSISTENT_AT_HALF_BUDGET")
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(bl.ANGLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        checksum = hashlib.sha256(buf.getvalue().encode()).hexdigest()
        if checksum != source["supersedes"]["angles_logistic.csv_sha256"]:
            raise RuntimeError("protocol rows do not match the original export checksum")
    return rows, source, checksum


def _half_status(r: dict, q: int) -> str:
    """Classify half q, separating unresolved estimates from angular disagreements."""
    if r["applicable"] != "True":
        return "FULL_BUDGET_UNRESOLVED"
    if r[f"half{q}_dk_deg"] in ("", "None", "nan"):
        if r[f"half{q}_consistent"] != "False":
            raise RuntimeError(f"half {q} without DK marked consistent: {r}")
        return "HALF_BUDGET_UNRESOLVED"
    return "CONSISTENT" if r[f"half{q}_consistent"] == "True" else "ANGULAR_DISAGREEMENT"


def _doubling_status(r: dict) -> str:
    st = {_half_status(r, 1), _half_status(r, 2)}
    for s in (
        "FULL_BUDGET_UNRESOLVED",
        "ANGULAR_DISAGREEMENT",
        "HALF_BUDGET_UNRESOLVED",
        "CONSISTENT",
    ):
        if s in st:
            return s
    raise AssertionError(st)


def run_publish2(cfg: DictConfig) -> pathlib.Path:
    """Combine both reference budgets after verifying the original protocol rows."""
    sel = check_frozen_pairs(cfg)
    design = expanded_design(cfg)
    ev, ex = cfg.evaluation, cfg.expanded
    if str(cfg.phase) != "main":
        raise ValueError("publish2 reads phase=main runs only")
    out = REPO / str(cfg.calibration.out_dir)
    proto_budget = f"protocol_{int(ev.sigma.n_batches) * int(ev.sigma.n_blocks)}"
    old_rows, old_source, protocol_sha = _load_protocol_export(out, proto_budget)
    tiers = [
        (u, output_root() / str(cfg.study) / "main" / f"{u}_fresh_{ev.tag}", proto_budget)
        for u in ev.pairs
    ]
    tiers += [
        (
            u,
            output_root() / str(cfg.study) / "main" / f"{u}_expanded_{ex.tag}",
            f"expanded_{p['N_total']}",
        )
        for u, p in design["units"].items()
    ]
    rows, refs, status, summ, runs = [], [], [], [], {}
    for unit, path, budget in tiers:
        stt = json.loads((path / "RUN-STATUS.json").read_text())
        if stt["status"] != "COMPLETED":
            raise RuntimeError(f"{path} is {stt['status']}")
        runs[f"{unit}/{budget}"] = {
            "batches": [_batch_settings(json.loads((path / f"batch_{k}.json").read_text())) for k in range(4)],
        }
        with open(path / "rows.csv") as f:
            urows = [{**r, "budget": budget, "_unit": unit} for r in csv.DictReader(f)]
        with open(path / "reference.csv") as f:
            urefs = [{**r, "budget": budget} for r in csv.DictReader(f)]
        rows += urows
        refs += urefs
        for m in sorted({int(r["m"]) for r in urefs}):
            mr = [r for r in urefs if int(r["m"]) == m]
            shown = [int(r["t"]) for r in mr if r["cutoff"] == ""]
            status.append(
                {
                    "budget": budget,
                    "unit": unit,
                    "m": m,
                    "status": "EVALUATED",
                    "displayed_t_first": min(shown) if shown else "",
                    "displayed_t_last": max(shown) if shown else "",
                    "n_displayed": len(shown),
                    "n_checkpoints": len(mr),
                    "n_dk_applicable": sum(r["applicable"] == "True" for r in mr),
                    "n_half_budget_inconsistent_displayed": sum(
                        r["stable"] != "True" for r in mr if r["cutoff"] == ""
                    ),
                    "note": "reference-based empirical sensitivity envelope",
                }
            )
            for t in sorted({int(r["t"]) for r in mr}):
                for L in sorted({int(r["pilot_length"]) for r in urows}):
                    g = [
                        r
                        for r in urows
                        if int(r["m"]) == m and int(r["t"]) == t and int(r["pilot_length"]) == L
                    ]
                    ang = [float(r["angle_deg"]) for r in g]
                    up = [float(r["learned_angle_upper"]) for r in g]
                    summ.append(
                        {
                            "budget": budget,
                            "unit": unit,
                            "m": m,
                            "t": t,
                            "pilot_length": L,
                            "n_fits": len(g),
                            "n_failed": sum(r["estimate_status"] == "FAILED_FIT" for r in g),
                            "angle_median": _q(ang, 0.5),
                            "angle_q25": _q(ang, 0.25),
                            "angle_q75": _q(ang, 0.75),
                            "upper_median": _q(up, 0.5),
                            "upper_q25": _q(up, 0.25),
                            "upper_q75": _q(up, 0.75),
                            "estimate_status": ";".join(sorted({r["estimate_status"] for r in g})),
                            "cutoff": next(rr["cutoff"] for rr in mr if int(rr["t"]) == t),
                            "note": "median / IQR over fits; not confidence intervals",
                        }
                    )
    proto = [{k: r[k] for k in bl.ANGLE_COLUMNS} for r in rows if r["budget"] == proto_budget]

    def key(r):
        return (
            r["dataset"],
            r["sampler"],
            int(r["m"]),
            int(r["t"]),
            int(r["pilot_length"]),
            int(r["fit"]),
        )

    if sorted(proto, key=key) != old_rows:
        raise RuntimeError("protocol-tier rows do not reproduce the saved angles_logistic.csv")
    # Compare full and half reference budgets.
    dbl = {}
    for r in refs:
        r["half1_status"] = _half_status(r, 1)
        r["half2_status"] = _half_status(r, 2)
        r["doubling_status"] = _doubling_status(r)
        if (r["doubling_status"] == "CONSISTENT") != (r["stable"] == "True"):
            raise RuntimeError(f"doubling status disagrees with stable flag: {r}")
        dbl[(r["budget"], r["unit"], int(r["m"]), int(r["t"]))] = r["doubling_status"]
    old_reason = "DK_INCONSISTENT_AT_HALF_BUDGET"
    for r in rows:
        if old_reason not in r["cutoff_reason"]:
            continue
        d = dbl[(r["budget"], r["_unit"], int(r["m"]), int(r["t"]))]
        if d not in ("HALF_BUDGET_UNRESOLVED", "ANGULAR_DISAGREEMENT"):
            raise RuntimeError(f"row flagged {old_reason} but doubling status {d}: {r}")
        new = (
            "HALF_BUDGET_UNRESOLVED (a half-budget eigengap <= 2 e_t; not an angular disagreement)"
            if d == "HALF_BUDGET_UNRESOLVED"
            else "HALF_BUDGET_ANGULAR_DISAGREEMENT"
        )
        r["cutoff_reason"] = r["cutoff_reason"].replace(old_reason, new)
    for s_ in status:
        if s_["status"] != "EVALUATED":
            continue
        shown = [
            r
            for r in refs
            if r["budget"] == s_["budget"]
            and r["unit"] == s_["unit"]
            and int(r["m"]) == int(s_["m"])
            and r["cutoff"] == ""
        ]
        s_["n_half_budget_unresolved_displayed"] = sum(
            r["doubling_status"] == "HALF_BUDGET_UNRESOLVED" for r in shown
        )
        s_["n_half_budget_disagreement_displayed"] = sum(
            r["doubling_status"] == "ANGULAR_DISAGREEMENT" for r in shown
        )
        if (
            s_["n_half_budget_unresolved_displayed"] + s_["n_half_budget_disagreement_displayed"]
            != s_["n_half_budget_inconsistent_displayed"]
        ):
            raise RuntimeError(f"doubling counts do not add up: {s_}")
    with open(out / "lag2t_check_logistic_v2.csv") as f:
        lag2t = {(r["unit"], int(r["m"]), int(r["t"])): r for r in csv.DictReader(f)}
    win = {}
    with open(out / sel["window_csv"]["path"]) as f:
        for r in csv.DictReader(f):
            win.setdefault((r["unit"], int(r["m"])), []).append(r)
    for unit, mlist in ev.not_evaluated.items():
        for m in [int(v) for v in mlist]:
            best = max(
                (float(r["snr_fresh"]), int(r["t"])) for r in win[(unit, m)] if int(r["t"]) >= 1
            )
            l2 = [lag2t[(unit, m, t)] for t in (2, 8) if (unit, m, t) in lag2t]
            status.append(
                {
                    "budget": proto_budget,
                    "unit": unit,
                    "m": m,
                    "status": "NOT_EVALUATED_NO_PROJECTED_WINDOW",
                    "note": f"projected best gap/e at the protocol budget = {best[0]:.2f} (t={best[1]}) < 2; "
                    + "; ".join(
                        f"lag-2t Sigma_t gap/eps {float(r['gap_lag2t']) / float(r['eps_lag2t']):.1f} vs "
                        f"branched {float(r['gap_branched']) / float(r['eps_branched']):.1f} at t={r['t']}"
                        for r in l2
                    )
                    + " (calibration only); evaluated in the expanded tier",
                }
            )
    for unit in ev.none_eligible:
        status.append(
            {
                "budget": "",
                "unit": unit,
                "m": "",
                "status": "NONE_ELIGIBLE",
                "note": "no (lag, m) passes rules 1-2 (half-split embedded spans 80-90 deg apart at every m and lag)",
            }
        )
    rows.sort(key=lambda r: (r["budget"] != proto_budget, *key(r)))
    cols = list(bl.ANGLE_COLUMNS) + ["budget"]
    with open(out / "angles_logistic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in rows)
    _write_csv(refs, out / "reference_logistic.csv")
    _write_csv(status, out / "status_logistic.csv")
    _write_csv(summ, out / "summary_logistic.csv")
    names = (
        "angles_logistic.csv",
        "reference_logistic.csv",
        "status_logistic.csv",
        "summary_logistic.csv",
    )
    source = {
        k: v for k, v in old_source.items()
        if k in ("status", "selection", "evaluation_config", "covariance_source", "samplers",
                 "features", "numerical_method", "margins", "columns_not_applicable")
    }
    source.update(
        {
            "tiers": {
                proto_budget: "protocol budget: 4 batches x 8192 branched blocks, R2 pool halves",
                "expanded_<N>": "expanded budget: per-unit block counts for DK "
                "applicability x1.5 over the calibration window (expanded_design_logistic_x1.json); "
                "new NUTS pools (bpool1) for hier and australian",
            },
            "covariance_source": {
                proto_budget: (
                    old_source["covariance_source"][proto_budget]
                    if isinstance(old_source["covariance_source"], dict)
                    else old_source["covariance_source"]
                ),
                "expanded_<N>": "branched-pair Sigma_t-hat as the protocol tier; starts from the R2 "
                "pool with replacement (units without a new pool) or from the new bpool1 NUTS pools "
                "without replacement (German Credit hier, Australian; four run quarters give the X / X' "
                "subsets of the two halves); see expanded_config",
            },
            "smoke_replay": "Seeds depend on unit, stage and batch, but not phase. Smoke and main runs "
            "replay the same streams and are not independent replications.",
            "expanded_config": OmegaConf.to_container(ex),
            "expanded_design_sha256": str(ex.design_sha256),
            "runs": runs,
            "supersedes": {
                "angles_logistic.csv_sha256": protocol_sha,
                "note": "protocol-tier-only export; its rows are reproduced exactly as the "
                f"budget={proto_budget} subset (checked)",
            },
            "outputs_sha256": {n: sha256_file(out / n) for n in names},
            "column_notes": {
                "budget": f"{proto_budget} = protocol tier (4 x 8192 blocks); expanded_<N> = "
                "expanded tier with N total blocks (4 batches)",
                "sin_theta / angle_deg": "largest principal angle between the learned span and "
                "V_m(S_t-hat) (point estimate against the reference-estimated optimum)",
                "sin_uncertainty": "sin of the DK term arcsin(e_t / (gap - e_t)) when DK applies "
                "(an empirical sensitivity of the angle to the reference optimum), else nan",
                "learned_angle_upper / learned_sin_upper": "min(90, angle + DK term): empirical "
                "sensitivity envelope for the angle to V_m(S_t), nan where DK does not apply",
                "estimate_status": "EMPIRICAL_SENSITIVITY_ONLY | TRIVIAL(EMPIRICAL_SENSITIVITY_ONLY) "
                "(envelope reaches 90 deg) | UNAVAILABLE (gap <= 2 e_t) | FAILED_FIT (no learned span)",
                "cutoff_reason": "display rule (BEFORE_RESOLVED_WINDOW, FIRST_FAILED_CHECKPOINT, "
                "BEYOND_FIRST_FAILED_CHECKPOINT(t=..), NO_RESOLVED_CHECKPOINT) and per-row reasons "
                "(REFERENCE_GAP_UNRESOLVED; HALF_BUDGET_UNRESOLVED = a half budget is not "
                "DK-applicable, not an angular disagreement; HALF_BUDGET_ANGULAR_DISAGREEMENT = "
                "angle(V_full, V_half) > dk_full + dk_half)",
                "half{1,2}_status / doubling_status (reference_logistic.csv)": "CONSISTENT | "
                "HALF_BUDGET_UNRESOLVED | ANGULAR_DISAGREEMENT | FULL_BUDGET_UNRESOLVED",
                "n_half_budget_inconsistent_displayed (status_logistic.csv)": "displayed checkpoints "
                "whose budget-doubling check is not CONSISTENT = n_half_budget_unresolved_displayed + "
                "n_half_budget_disagreement_displayed",
                "empirical_sensitivity": "e_t = 3 ||S^A - S^B||_op / 2",
            },
        }
    )
    _json(out / "SOURCE_logistic.json", source)
    return out / "SOURCE_logistic.json"


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_b_logistic", version_base=None
)
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    stage = str(cfg.stage)
    if stage == "calibrate":
        print(run_calibrate(cfg))
    elif stage == "select":
        print(run_select(cfg))
    elif stage == "amend_select":
        print(run_amend_select(cfg))
    elif stage == "fresh_gpu":
        print(run_fresh_gpu(cfg))
    elif stage == "fresh_cpu":
        print(run_fresh_cpu(cfg))
    elif stage == "publish":
        print(run_publish(cfg))
    elif stage == "expanded_gpu":
        print(run_expanded_gpu(cfg))
    elif stage == "expanded_cpu":
        print(run_expanded_cpu(cfg))
    elif stage == "publish2":
        print(run_publish2(cfg))
    else:
        raise ValueError(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()
