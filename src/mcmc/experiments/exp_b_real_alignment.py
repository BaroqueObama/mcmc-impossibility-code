"""Slow-mode alignment for Gaussian regression posteriors (Figures 2B and 4).

The reversible Gaussian AR sampler has a known Hermite spectrum. Population
operators use closed-form moments; learned spans use independent TICA fits.
Exact-integer Gram calculations refine angles at small eigengaps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import pathlib
import resource
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import real_alignment as ra
from mcmc.paperexp import regression_data as rd
from mcmc.paperexp.rng import np_gen, seed_for
from mcmc.paperexp.runio import RunDir, output_root, resolve_output_path

log = logging.getLogger(__name__)
REPO = pathlib.Path(__file__).resolve().parents[3]
CONFIG_FILE = REPO / "configs/experiment/exp_b_real_alignment.yaml"
PARTS = ("setup", "calibrate", "freeze", "population", "learned", "precision", "learned_refine")


def jdump(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stream(ds: str) -> str:
    return f"b-real-{ds}"


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.t[key] = timer.t.get(key, 0.0) + time.perf_counter() - self.t0

        return _Ctx()


def build_model(cfg: DictConfig, setup_npz=None) -> dict:
    """Build the posterior, sampler and landmark features.

    If setup_npz is supplied, reuse its feature bases after checking the landmarks
    and sampler against their deterministic reconstruction.
    """
    ds = str(cfg.dataset)
    if ds not in rd.NAMES:
        raise ValueError(f"unknown dataset {ds!r}")
    if (
        float(cfg.sampler.base_rate) != ra.BASE_RATE
        or int(cfg.sampler.rotation_seed) != ra.ROTATION_SEED
    ):
        raise ValueError("config sampler rule differs from the library's frozen rule")
    fm = rd.frozen_model(ds)
    if fm.ell is None:
        raise ValueError(f"{ds}: frozen bandwidth missing in the R3 config")
    ar = ra.build_controlled_ar(ds, fm.mu, fm.C)
    ls = fm.kernel_scales * fm.ell
    src_cfg = REPO / (
        "configs/experiment/exp_r3_concrete.yaml"
        if ds == "concrete"
        else "configs/experiment/exp_r3_regression.yaml"
    )
    Z = ar.sample_target(
        np_gen(seed_for(stream(ds), str(cfg.features.landmark_stream), 0)),
        int(cfg.features.n_landmarks_sensitivity),
    )
    M = int(cfg.features.n_landmarks)
    if M > Z.shape[0]:
        raise ValueError("primary landmark count exceeds the sensitivity draw")
    cut = float(cfg.features.gram_rel_cutoff)
    if setup_npz is None:
        feat = ra.landmark_features(Z[:M], ls, cut)
        feat_s = ra.landmark_features(Z, ls, cut)
    else:
        for key, ref in (("Z", Z), ("U", ar.U), ("A", ar.A), ("rho", ar.rho)):
            if not np.array_equal(setup_npz[key], ref):
                raise ValueError(f"deterministic rebuild differs from the setup run in {key}")
        feat = ra.LandmarkFeatures(
            Z[:M].copy(), ls.copy(), cut, setup_npz["W"], setup_npz["gram_eigs"]
        )
        feat_s = ra.LandmarkFeatures(
            Z.copy(), ls.copy(), cut, setup_npz["W_sens"], setup_npz["gram_eigs_sens"]
        )
    return {
        "dataset": ds,
        "fm": fm,
        "ar": ar,
        "ls": ls,
        "Z": Z,
        "feat": feat,
        "feat_sens": feat_s,
        "provenance": {
            "posterior": "exact N(mu, C) of the R3 regression model (regression_data.frozen_model)",
            "n": fm.n,
            "p": fm.p,
            "sigma2": fm.sigma2,
            "ell": fm.ell,
            "kernel_scales": fm.kernel_scales.tolist(),
            "data_provenance": fm.provenance,
            "source_config": str(src_cfg.relative_to(REPO)),
            "source_config_sha256": sha256_file(src_cfg),
            "mu_sha256": ra.sha256_array(fm.mu),
            "C_sha256": ra.sha256_array(fm.C),
            "landmark_seed": seed_for(stream(ds), str(cfg.features.landmark_stream), 0),
        },
    }


def load_setup(path) -> tuple[dict, dict, str]:
    path = resolve_output_path(path)
    st = json.loads((path / "RUN-STATUS.json").read_text())
    if st["status"] != "COMPLETED":
        raise RuntimeError(f"setup run {path} is {st['status']}")
    meta = json.loads((path / "setup.json").read_text())
    return meta, dict(np.load(path / "setup.npz")), str(path)


def true_modes(model: dict, spec: ra.HermiteSpectrum, which: str = "feat") -> dict:
    feat = model[which]
    out = {}
    for prec in ("fp64", "ld"):
        g = ra.make_geometry(model["ar"], feat, prec)
        out[prec] = ra.embedded_modes(g, spec.indices)
        out[f"geom_{prec}"] = g
    return out


def run_setup(cfg: DictConfig, run: RunDir) -> None:
    tm = Timer()
    with tm("model"):
        model = build_model(cfg)
    ds, ar = model["dataset"], model["ar"]
    with tm("witnesses"):
        wit = ra.ar_witnesses(ar)
    for key in (
        "stationarity_rel",
        "reversibility_rel",
        "noise_vs_C_minus_ACAt_rel",
        "LLt_vs_C_rel",
    ):
        if wit[key] > 1e-12:
            raise FloatingPointError(f"{ds}: sampler witness {key} = {wit[key]:.3e}")
    if wit["noise_min_eig"] <= 0:
        raise FloatingPointError(f"{ds}: noise covariance not positive definite")
    with tm("spectrum"):
        spec = ra.enumerate_hermite(
            ar.rho, int(cfg.spectrum.n_top), float(cfg.spectrum.tie_tol_log)
        )
    gaps = {
        m: ra.true_gap(spec, m) for m in range(1, min(spec.K - 1, int(cfg.calibration.m_max)) + 1)
    }
    feats = {}
    with tm("features"):
        for key in ("feat", "feat_sens"):
            f = model[key]
            feats[key] = {**f.to_dict(), **ra.metric_defect(f)}
    with tm("moment_checks"):
        g64 = ra.make_geometry(ar, model["feat"], "fp64")
        low = ra.low_degree_checks(ar, model["feat"], g64)
        if max(low.values()) > 1e-10:
            raise FloatingPointError(f"{ds}: low-degree moment check failed {low}")
        mc = ra.mc_moment_check(
            ar,
            model["feat"],
            g64,
            np_gen(seed_for(stream(ds), str(cfg.checks.mc_stream), 0)),
            int(cfg.checks.mc_n),
            int(cfg.checks.mc_landmarks),
            int(cfg.checks.mc_max_degree),
        )
        if mc["max_abs_z"] > mc["bonferroni_z_0.001"]:
            raise FloatingPointError(f"{ds}: Monte Carlo moment check failed {mc}")
    with tm("true_modes"):
        tw = {key: true_modes(model, spec, key) for key in ("feat", "feat_sens")}
    rank = {}
    for key in ("feat", "feat_sens"):
        rank[key] = {}
        for m in gaps:
            W64, Wld = tw[key]["fp64"][:, :m], tw[key]["ld"][:, :m]
            sv = np.linalg.svd(W64, compute_uv=False)
            rank[key][m] = {
                "sigma_min": float(sv[-1]),
                "sigma_max": float(sv[0]),
                "fp64_vs_ld_op": float(np.linalg.svd(W64 - Wld, compute_uv=False)[0]),
                "rank_ok": bool(sv[-1] > 1e-8 * sv[0]),
            }
    np.savez(
        run.path / "setup.npz",
        Z=model["Z"],
        W=model["feat"].W,
        W_sens=model["feat_sens"].W,
        gram_eigs=model["feat"].gram_eigs,
        gram_eigs_sens=model["feat_sens"].gram_eigs,
        U=ar.U,
        rho=ar.rho,
        A=ar.A,
        spec_indices=spec.indices,
        spec_log_eig=spec.log_eig,
        w_true_fp64=tw["feat"]["fp64"],
        w_true_ld=tw["feat"]["ld"],
        w_true_sens_fp64=tw["feat_sens"]["fp64"],
        w_true_sens_ld=tw["feat_sens"]["ld"],
    )
    jdump(
        run.path / "setup.json",
        {
            "dataset": ds,
            "sampler": ar.to_dict(),
            "sampler_label": "controlled reversible Gaussian AR sampler (declared; not RWM/MALA/Gibbs)",
            "witnesses": wit,
            "spectrum": spec.to_dict(),
            "true_gaps": gaps,
            "features": feats,
            "moment_checks": {"low_degree": low, "monte_carlo": mc},
            "embedded_rank": rank,
            "provenance": model["provenance"],
            "timings_s": tm.t,
            "peak_rss_mib": peak_rss_mib(),
        },
    )
    run.update(completed_units=1)
    log.info("%s setup done: rank %d, timings %s", ds, model["feat"].rank, tm.t)


def run_calibrate(cfg: DictConfig, run: RunDir) -> None:
    tm = Timer()
    meta, npz, setup_path = load_setup(cfg.setup_run)
    model = build_model(cfg, npz)
    if meta["dataset"] != model["dataset"]:
        raise ValueError(f"setup run is for {meta['dataset']}")
    setup = {"meta": meta, "npz": npz, "path": setup_path}
    ds, ar, feat = model["dataset"], model["ar"], model["feat"]
    cc = cfg.calibration
    paths, F = [], []
    seeds = [seed_for(stream(ds), str(cc.stream), c) for c in range(int(cc.n_chains))]
    with tm("simulate"):
        for s in seeds:
            paths.append(ar.path(np_gen(s), int(cc.n_transitions)))
    with tm("features"):
        for X in paths:
            F.append(feat(X))
    halves = [list(h) for h in cc.halves]
    lags = [int(x) for x in cc.lags]
    cand_lags = [int(x) for x in cc.candidate_lags]
    partner = {lag: lags[lags.index(lag) + 1] for lag in cand_lags}
    rules = ra.SelectionRules(
        gap_units=float(cc.rules.gap_units),
        max_half_angle_deg=float(cc.rules.max_half_angle_deg),
        max_stability_angle_deg=float(cc.rules.max_stability_angle_deg),
        max_counts=int(cc.rules.max_counts),
        m_max=int(cc.m_max),
    )
    results, eig_store = {}, {}
    for tag, cut in (
        ("primary", float(cc.whitening_cutoff)),
        ("sensitivity", float(cc.whitening_cutoff_sensitivity)),
    ):
        fits = {}
        with tm(f"tica_{tag}"):
            for lag in lags:
                fits[lag] = {
                    "pooled": ra.tica_fit(F, lag, cut),
                    "A": ra.tica_fit([F[i] for i in halves[0]], lag, cut),
                    "B": ra.tica_fit([F[i] for i in halves[1]], lag, cut),
                }
                for k, v in fits[lag].items():
                    eig_store[f"{tag}_lag{lag}_{k}"] = v.eigenvalues[:64]
        rows = ra.calibration_candidates(fits, cand_lags, partner, rules)
        sel = ra.select_counts(rows, rules)
        whit = {
            f"lag{lag}_{k}": {
                "rank": v.whitening.rank,
                "threshold": v.whitening.threshold,
                "lag_asymmetry": v.lag_asymmetry,
                "whitened_asymmetry": v.whitened_asymmetry,
            }
            for lag in lags
            for k, v in fits[lag].items()
        }
        results[tag] = {
            "cutoff": cut,
            "candidates": rows,
            "selection": sel,
            "whitening": whit,
            "fits": fits,
        }
    # Validate the selected counts against the known Hermite spectrum.
    spec = ra.enumerate_hermite(ar.rho, int(cfg.spectrum.n_top), float(cfg.spectrum.tie_tol_log))
    w64 = setup["npz"]["w_true_fp64"]
    sel = results["primary"]["selection"]
    validation = []
    for m in sel["selected"]:
        gap = ra.true_gap(spec, m)
        rk = setup["meta"]["embedded_rank"]["feat"][str(m)]
        P = results["primary"]["fits"][sel["lag"]]["pooled"]
        cluster_ok = gap["strict"]
        validation.append(
            {
                "m": m,
                "lag": sel["lag"],
                "strict_true_gap": cluster_ok,
                "true_b": gap["b"],
                "true_a": gap["a"],
                "true_rel_gap": gap["rel_gap"],
                "true_modes": [spec.label(j) for j in range(m)],
                "next_true_mode": spec.label(m),
                "embedded_rank_ok": rk["rank_ok"],
                "embedded_sigma_min": rk["sigma_min"],
                "pilot_vs_true_span_angle_deg": ra.angle_deg(P.embedded[:, :m], w64[:, :m]),
                "pilot_per_step_rates": [
                    float(abs(x) ** (1.0 / sel["lag"])) for x in P.eigenvalues[: m + 1]
                ],
                "true_rates": [float(x) for x in spec.eig[: m + 1]],
                "status": "VALIDATED"
                if (cluster_ok and rk["rank_ok"])
                else "FAILED_TRUE_SPECTRUM_CHECK",
            }
        )
    hashes = {f"chain{c}": ra.sha256_array(X) for c, X in enumerate(paths)}
    np.savez(
        run.path / "calibration.npz",
        paths=np.stack(paths),
        seeds=np.array(seeds, dtype=np.int64),
        **eig_store,
    )
    out = {
        "dataset": ds,
        "selection_version": ra.SELECTION_VERSION,
        "setup_run": setup["path"],
        "calibration": {
            "n_chains": int(cc.n_chains),
            "n_transitions": int(cc.n_transitions),
            "start": "exact target draw (stationary)",
            "seeds": seeds,
            "seed_stream": f"seed_for('{stream(ds)}', '{cc.stream}', chain)",
            "halves": halves,
            "lags": lags,
            "candidate_lags": cand_lags,
            "stability_partner": partner,
            "path_sha256": hashes,
            "features": "512-landmark whitened Nystrom (setup run); centring by the pooled empirical mean",
        },
        "rules": {
            "version": ra.SELECTION_VERSION,
            "rule1": f"|rho_m|-|rho_m+1| (pooled, lag-tau eigenvalues) > {rules.gap_units} half-split units "
            "(max_{j<=m+2} ||rho_j^A|-|rho_j^B||/2) and half-split embedded-span angle <= "
            f"{rules.max_half_angle_deg} deg",
            "rule2": f"pooled top-m embedded span at the next longer lag within {rules.max_stability_angle_deg} deg",
            "rule3": "smallest eligible lag; up to two m with largest gap ratio (ties -> smaller m)",
            "rule4": "strict true Hermite gap after the complete selected cluster + embedded rank m (recorded, "
            "never replaced)",
            "m_range": [1, rules.m_max],
        },
        "primary": {k: v for k, v in results["primary"].items() if k != "fits"},
        "sensitivity_cutoff": {k: v for k, v in results["sensitivity"].items() if k != "fits"},
        "validation": validation,
        "timings_s": tm.t,
        "peak_rss_mib": peak_rss_mib(),
    }
    jdump(run.path / f"selection_{ds}.json", out)
    run.update(completed_units=1)
    log.info(
        "%s calibration: selection %s (sensitivity %s); timings %s",
        ds,
        sel,
        results["sensitivity"]["selection"],
        tm.t,
    )


def run_freeze(cfg: DictConfig) -> None:
    target = REPO / str(cfg.selection_file)
    runs = dict(cfg.get("calibration_runs", {}))
    if target.exists():
        doc = json.loads(target.read_text())
    else:
        doc = {
            "description": "Diagnostic-count selections for Gaussian regression posteriors with a "
            "reversible Gaussian AR sampler.",
            "datasets": {},
            "numerical_refinements": [],
        }
    for ds, path in runs.items():
        if frozen_entry(doc, ds, missing_ok=True) is not None:
            raise FileExistsError(f"{ds} already selected in {target}; use a new selection_file")
        path = _completed(path)
        sel = json.loads((path / f"selection_{ds}.json").read_text())
        setup_path = resolve_output_path(sel["setup_run"])
        setup = json.loads((setup_path / "setup.json").read_text())
        if sel["dataset"] != ds or setup["dataset"] != ds:
            raise ValueError(f"calibration or setup run is for a different dataset than {ds}")
        try:
            setup_run = "$MCMC_PAPER_OUT/" + str(
                setup_path.resolve().relative_to(resolve_output_path("$MCMC_PAPER_OUT").resolve())
            )
        except ValueError:
            setup_run = str(setup_path)
        doc["datasets"][ds] = {
            "setup_run": setup_run,
            "setup_npz_sha256": sha256_file(setup_path / "setup.npz"),
            "sampler": {
                k: v for k, v in setup["sampler"].items()
                if not k.endswith("_sha256") or k in ("U_sha256", "A_sha256")
            },
            "feature_basis": {
                k: v for k, v in setup["features"]["feat"].items()
                if not k.endswith("_sha256") or k == "W_sha256"
            },
            "feature_basis_sensitivity": {
                k: v for k, v in setup["features"]["feat_sens"].items()
                if not k.endswith("_sha256")
            },
            "posterior_provenance": {
                k: v for k, v in setup["provenance"].items()
                if k in ("posterior", "n", "p", "sigma2", "ell", "kernel_scales", "landmark_seed")
            },
            "selected_m": sel["primary"]["selection"]["selected"],
            "selected_lag": sel["primary"]["selection"]["lag"],
            "selection_outcome": sel["primary"]["selection"]["outcome"],
            "eligible_at_lag": sel["primary"]["selection"].get("eligible_at_lag", []),
            "validation": sel["validation"],
            "candidates": sel["primary"]["candidates"],
            "sensitivity_cutoff_1e-8": {
                "selection": sel["sensitivity_cutoff"]["selection"],
                "candidates": sel["sensitivity_cutoff"]["candidates"],
            },
            "calibration": {k: v for k, v in sel["calibration"].items() if k != "path_sha256"},
            "rules": sel["rules"],
            "time_grid": list(cfg.population.t_grid),
            "time_extension_rule": f"geometric to at most {cfg.population.t_extension_max} only if the "
            "evaluated bound predicts an informative window beyond 512 (resolved positive point with g_t < 1 "
            "at t = 512)",
            "learned_budget": OmegaConf.to_container(cfg.learned, resolve=True),
        }
    for value in cfg.get("numerical_refinements", []):
        refinement = {
            "dataset": str(value.dataset),
            "m": int(value.m),
            "times": [int(t) for t in value.times],
            "method": str(value.method),
        }
        entry = frozen_entry(doc, refinement["dataset"])
        if refinement["m"] not in entry["selected_m"]:
            raise ValueError(f"refinement count is not selected: {refinement}")
        if refinement not in doc["numerical_refinements"]:
            doc["numerical_refinements"].append(refinement)
    target.parent.mkdir(parents=True, exist_ok=True)
    jdump(target, doc)
    log.info("wrote %s (datasets %s)", target, sorted(doc["datasets"]))


def frozen_entry(doc: dict, ds: str, missing_ok: bool = False):
    """Selected count and calibration results for a dataset."""
    if ds in doc["datasets"]:
        return doc["datasets"][ds]
    if missing_ok:
        return None
    raise KeyError(f"{ds} not selected in the selection file")


def _load_selection(cfg: DictConfig) -> tuple[dict, str]:
    path = REPO / str(cfg.selection_file)
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    expected = cfg.get("selection_sha256")
    if expected is not None and checksum != str(expected):
        raise ValueError(f"{path}: selection checksum differs from configuration")
    return json.loads(raw), checksum


def _load_selected_setup(cfg: DictConfig, entry: dict):
    """Use an explicit rerun location without changing the frozen setup or selection."""
    path = resolve_output_path(cfg.setup_run or entry["setup_run"])
    if sha256_file(path / "setup.npz") != entry["setup_npz_sha256"]:
        raise ValueError("setup.npz differs from the frozen selection")
    return load_setup(path)


def run_population(cfg: DictConfig, run: RunDir) -> None:
    tm = Timer()
    doc, sel_sha = _load_selection(cfg)
    ds = str(cfg.dataset)
    entry = frozen_entry(doc, ds)
    meta, npz, _ = _load_selected_setup(cfg, entry)
    model = build_model(cfg, npz)
    ar = model["ar"]
    for key, ref in (("U_sha256", ra.sha256_array(ar.U)), ("A_sha256", ra.sha256_array(ar.A))):
        if entry["sampler"][key] != ref:
            raise ValueError(f"sampler {key} differs from the frozen selection")
    if entry["feature_basis"]["W_sha256"] != ra.sha256_array(model["feat"].W):
        raise ValueError("feature basis differs from the frozen selection")
    ms = [int(m) for m in entry["selected_m"]]
    rules = ra.NumericsRules(
        allowance_factor=float(cfg.population.allowance_factor),
        op_factor=float(cfg.population.op_factor),
        sin_factor=float(cfg.population.sin_factor),
        max_precision_change_deg=float(cfg.population.max_precision_change_deg),
    )
    spec = ra.enumerate_hermite(ar.rho, int(cfg.spectrum.n_top), float(cfg.spectrum.tie_tol_log))
    t_grid = [int(t) for t in entry["time_grid"]]
    summary = {
        "dataset": ds,
        "selection_sha256": sel_sha,
        "selected_m": ms,
        "lag": entry["selected_lag"],
        "per_landmarks": {},
    }
    all_rows = []
    if not ms:
        summary["outcome"] = "NO_SELECTED_COUNT"
        jdump(run.path / "population.json", summary)
        run.update(completed_units=run.status()["requested_units"])
        return
    for key in ("feat", "feat_sens"):
        feat = model[key]
        with tm(f"geometry_{key}"):
            g64 = ra.make_geometry(ar, feat, "fp64")
            gld = ra.make_geometry(ar, feat, "ld")
            mmax = max(ms)
            w64 = ra.embedded_modes(g64, spec.indices[:mmax])
            wld = ra.embedded_modes(gld, spec.indices[:mmax])
        S64, Sld, allow, eigs, eig = {}, {}, {}, {}, {}

        def evaluate(t):
            with tm(f"S_fp64_{key}"):
                S64[t], asym64 = ra.operator_S_info(g64, t)
            with tm(f"S_ld_{key}"):
                Sld[t], asym_ld = ra.operator_S_info(gld, t)
            allow[t] = ra.operator_allowance(S64[t], Sld[t], rules.allowance_factor)
            allow[t]["whitening_asymmetry_fp64"] = asym64
            allow[t]["whitening_asymmetry_ld"] = asym_ld
            with tm(f"eigen_{key}"):
                eig[t] = ra.operator_eigen(S64[t], Sld[t])
            eigs[t] = {"fp64": eig[t]["lam"][:24].tolist(), "ld": eig[t]["lam_hi"][:24].tolist()}

        for t in t_grid:
            evaluate(t)
        binp = {
            m: ra.bound_inputs(
                S64[0], allow[0]["e_t"], w64[:, :m], wld[:, :m], ra.true_gap(spec, m), rules
            )
            for m in ms
        }
        rows_by_m = {m: [] for m in ms}

        def add_rows(t):
            for m in ms:
                with tm(f"angles_{key}"):
                    row = ra.checkpoint(
                        S64[t], Sld[t], w64[:, :m], wld[:, :m], allow[t], rules, eig[t]
                    )
                g, gstat, graw = ra.bound_value(binp[m], t)
                row.update(
                    {
                        "dataset": ds,
                        "landmarks": feat.M,
                        "feature_rank": feat.rank,
                        "m": m,
                        "t": t,
                        "sin_angle_upper": g,
                        "sin_angle_upper_raw": graw,
                        "angle_upper": math.degrees(math.asin(g)),
                        "theorem_bound_status": gstat,
                        "rate_a": binp[m]["a"],
                        "rate_b": binp[m]["b"],
                        "c_m_squared": binp[m]["c_m_squared_lower"],
                        "kappa_squared": binp[m]["kappa2_upper"],
                        "bound_satisfied": row["sin_theta"] <= g + row["sin_uncertainty"],
                    }
                )
                rows_by_m[m].append(row)

        for t in t_grid:
            add_rows(t)
        # Extend geometrically while the angle and its nontrivial bound remain resolved.
        ext = []
        last = t_grid[-1]
        while last < int(cfg.population.t_extension_max) and any(
            r["t"] == last
            and r["estimate_status"] == "RESOLVED"
            and r["sin_angle_upper"] < 1.0
            and all(
                rr["estimate_status"] in ("RESOLVED", "BELOW_NUMERICAL_SENSITIVITY")
                for rr in rows_by_m[m]
                if rr["t"] <= last
            )
            for m in ms
            for r in rows_by_m[m]
        ):
            last *= 2
            ext.append(last)
            evaluate(last)
            add_rows(last)
        gates = {
            m: ra.main_curve_and_gate(
                rows_by_m[m],
                int(cfg.population.gate.min_checkpoints),
                float(cfg.population.gate.max_sine_bound),
            )
            for m in ms
        }
        times = sorted(S64)
        with tm(f"hermite_recon_{key}"):
            with np.errstate(divide="ignore"):
                lr = np.log(ar.rho)
            lr = np.where(np.isfinite(lr), lr, -1e300)
            recon = ra.hermite_reconstruction(
                g64, int(cfg.checks.hermite_recon_degree[ds]), times, Sld, lr
            )
        spot = {}
        if key == "feat":
            with tm("mp_spot"):
                pairs = np_gen(seed_for(stream(ds), "mp-spot", 0)).integers(
                    0, feat.M, size=(int(cfg.checks.mp_spot_pairs), 2)
                )
                for t in [int(x) for x in cfg.checks.mp_spot_times]:
                    ref = ra.mp_spot_check(ar, feat, pairs, t)  # mpmath values (60 digits)
                    d64 = ra.centered_moments(g64, t)[pairs[:, 0], pairs[:, 1]]
                    dld = ra.centered_moments(gld, t)[pairs[:, 0], pairs[:, 1]]  # long double
                    e64 = ra.spot_check_errors(ref, d64)
                    eld = ra.spot_check_errors(ref, dld)
                    spot[t] = {
                        "max_abs_ref": e64["max_abs_ref"],
                        "fp64_rel_err": e64["rel_err"],
                        "ld_rel_err": eld["rel_err"],
                        "method": "differences formed in mpmath from the exact stored float64 / long "
                        "double values and the 60-digit reference; only the final ratio is rounded (up)",
                    }
            # inputs of every reported angle (exact spans are re-formed from these float64 arrays)
            k = min(S64[times[0]].shape[0], max(ms) + 8)
            np.savez(
                run.path / "operators.npz",
                times=np.array(times, dtype=np.int64),
                S64=np.stack([S64[t] for t in times]),
                V64=np.stack([eig[t]["V"][:, :k] for t in times]),
                Vld=np.stack([eig[t]["V_hi"][:, :k] for t in times]),
                lam64=np.stack([eig[t]["lam"] for t in times]),
                lamld=np.stack([eig[t]["lam_hi"] for t in times]),
                w64=w64,
                wld=wld,
            )
        summary["per_landmarks"][key] = {
            "landmarks": feat.M,
            "feature_rank": feat.rank,
            "bound_inputs": binp,
            "allowance": {t: allow[t] for t in times},
            "eigenvalues": {t: eigs[t] for t in times},
            "gates": gates,
            "extension_times": ext,
            "hermite_reconstruction": recon,
            "mp_spot_check": spot,
        }
        for m in ms:
            for r in rows_by_m[m]:
                r["landmark_set"] = "primary" if key == "feat" else "sensitivity_1024"
                all_rows.append(r)
        run.update(completed_units=run.status()["completed_units"] + 1)
    summary["gate"] = {m: summary["per_landmarks"]["feat"]["gates"][m] for m in ms}
    summary["gate_pass_any"] = any(g["gate_pass"] for g in summary["gate"].values())
    summary["timings_s"] = tm.t
    summary["peak_rss_mib"] = peak_rss_mib()
    jdump(run.path / "population.json", summary)
    write_csv(run.path / "population_rows.csv", all_rows)
    log.info(
        "%s population: gate %s; timings %s",
        ds,
        {m: (g["gate_pass"], g["n_resolved_nontrivial"]) for m, g in summary["gate"].items()},
        tm.t,
    )


def _quantiles(vals: list) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": int(v.size), "median": float(med), "q25": float(q25), "q75": float(q75)}


def _completed(path) -> pathlib.Path:
    path = resolve_output_path(path)
    st = json.loads((path / "RUN-STATUS.json").read_text())
    if st["status"] != "COMPLETED":
        raise RuntimeError(f"{path} is {st['status']}")
    return path


def _float(x: str) -> float:
    """CSV float field (repr round-trips exactly); empty fields are an error."""
    if x == "":
        raise ValueError("empty numeric field")
    return float(x)


def load_population(path, ds: str, ms: list, t_grid: list) -> dict:
    """Load population operators and angles, checking the dataset and (m, t) grid."""
    path = _completed(path)
    pj = json.loads((path / "population.json").read_text())
    if pj["dataset"] != ds or [int(m) for m in pj["selected_m"]] != ms:
        raise ValueError(f"population run {path} is for {pj['dataset']} m={pj['selected_m']}")
    ops = dict(np.load(path / "operators.npz"))
    if [int(t) for t in ops["times"]] != t_grid:
        raise ValueError(f"population operators at {ops['times']} differ from the frozen grid")
    rows = {}
    for r in csv.DictReader((path / "population_rows.csv").open()):
        if r["landmark_set"] == "primary":
            rows[(int(r["m"]), int(r["t"]))] = r
    if set(rows) != {(m, t) for m in ms for t in t_grid}:
        raise ValueError("population rows do not cover the frozen (m, t) grid")
    return {"path": path, "json": pj, "ops": ops, "rows": rows}


def run_learned(cfg: DictConfig, run: RunDir) -> None:
    """Fit TICA spans and compare them with the analytic optimum and true slow modes.

    The population run supplies operators and numerical allowances. Save fitted
    spans for later precision refinement.
    """
    tm = Timer()
    doc, sel_sha = _load_selection(cfg)
    ds = str(cfg.dataset)
    entry = frozen_entry(doc, ds)
    ms = [int(m) for m in entry["selected_m"]]
    out = {
        "dataset": ds,
        "selection_sha256": sel_sha,
        "selected_m": ms,
        "lag": entry["selected_lag"],
    }
    if not ms:
        out["outcome"] = "NO_SELECTED_COUNT"
        jdump(run.path / "learned.json", out)
        run.update(completed_units=run.status()["requested_units"])
        return
    lag = int(entry["selected_lag"])
    _, npz, _ = _load_selected_setup(cfg, entry)
    model = build_model(cfg, npz)
    ar, feat = model["ar"], model["feat"]
    lc = cfg.learned
    budget = OmegaConf.to_container(lc, resolve=True)
    if budget != entry["learned_budget"] and str(cfg.phase) != "smoke":
        raise ValueError(
            f"learned budget {budget} differs from the frozen {entry['learned_budget']}"
        )
    rules = ra.NumericsRules(
        allowance_factor=float(cfg.population.allowance_factor),
        op_factor=float(cfg.population.op_factor),
        sin_factor=float(cfg.population.sin_factor),
        max_precision_change_deg=float(cfg.population.max_precision_change_deg),
    )
    t_grid = [int(t) for t in entry["time_grid"]]
    mmax = max(ms)
    with tm("inputs"):
        pop = load_population(cfg.population_run, ds, ms, t_grid)
        ops = pop["ops"]
        if ops["w64"].shape != (feat.rank, mmax) or ops["S64"].shape[1] != feat.rank:
            raise ValueError("population operators do not match the frozen feature basis")
        ti = {t: i for i, t in enumerate(t_grid)}
        Vx, Wx, e_w = {}, {}, {}
        for m in ms:
            Wx[m] = ra.exact_span(ops["w64"][:, :m])
            modes = {_float(pop["rows"][(m, t)]["sin_uncertainty_modes"]) for t in t_grid}
            if len(modes) != 1:
                raise ValueError(f"true-mode term differs across times for m={m}: {modes}")
            e_w[m] = modes.pop()
            for t in t_grid:
                Vx[(m, t)] = (
                    ra.exact_span(ops["V64"][ti[t]][:, :m]),
                    ra.exact_span(ops["Vld"][ti[t]][:, :m]),
                )
        true_rows = {}
        for (m, t), r in pop["rows"].items():
            true_rows[(m, t)] = {
                "estimate_status": r["estimate_status"],
                "cutoff_reason": r["cutoff_reason"],
                "numerical_allowance": _float(r["numerical_allowance"]),
                "eigengap": _float(r["eigengap"]),
                "lambda_m": _float(r["lambda_m"]),
                "lambda_m1": _float(r["lambda_m1"]),
                "g": _float(r["sin_angle_upper"]),
                "g_status": r["theorem_bound_status"],
            }
    lengths = sorted(int(L) for L in lc.lengths)
    cut = float(cfg.calibration.whitening_cutoff)
    rows = []
    n_chains = int(lc.n_chains)
    spans = np.full((int(lc.n_fits), len(lengths), feat.rank, mmax), np.nan)
    whit_rank = np.zeros((int(lc.n_fits), len(lengths)), dtype=np.int64)
    for f in range(int(lc.n_fits)):
        seeds = [seed_for(stream(ds), str(lc.stream), f * n_chains + c) for c in range(n_chains)]
        with tm("simulate"):
            paths = [ar.path(np_gen(sd), lengths[-1]) for sd in seeds]
        with tm("features"):
            F = [feat(X) for X in paths]
        for iL, L in enumerate(lengths):
            base = {
                "dataset": ds,
                "fit": f,
                "pilot_length": L,
                "lag": lag,
                "landmarks": feat.M,
                "feature_rank": feat.rank,
                "seeds": " ".join(str(x) for x in seeds),
            }
            try:
                with tm("tica"):
                    fit = ra.tica_fit([Fc[: L + 1] for Fc in F], lag, cut)
            except (ra.sm.RankDeficientError, FloatingPointError) as exc:
                for m in ms:
                    for t in t_grid:
                        rows.append(
                            {
                                **base,
                                "m": m,
                                "t": t,
                                "estimate_status": "FIT_FAILED",
                                "cutoff_reason": f"{type(exc).__name__}: {str(exc)[:160]}",
                            }
                        )
                continue
            whit_rank[f, iL] = fit.whitening.rank
            k = min(mmax, fit.embedded.shape[1])
            spans[f, iL, :, :k] = fit.embedded[:, :k]
            for m in ms:
                if fit.embedded.shape[1] < m + 1:
                    for t in t_grid:
                        rows.append(
                            {
                                **base,
                                "m": m,
                                "t": t,
                                "estimate_status": "FIT_FAILED",
                                "cutoff_reason": f"whitening rank {fit.whitening.rank} < m + 1",
                            }
                        )
                    continue
                What = fit.embedded[:, :m]
                rates = [float(abs(x) ** (1.0 / lag)) for x in fit.eigenvalues[: m + 1]]
                with tm("angles"):
                    What_x = ra.exact_span(What)
                    mode = ra.learned_mode_error(What_x, Wx[m], e_w[m])
                for t in t_grid:
                    tr = true_rows[(m, t)]
                    i = ti[t]
                    with tm("evaluate"):
                        r = ra.learned_checkpoint(
                            What, What_x, *Vx[(m, t)], ops["V64"][i], ops["S64"][i],
                            ops["lam64"][i], mode, tr, tr["g"], rules,
                        )  # fmt: skip
                    r.update(
                        {
                            **base,
                            "m": m,
                            "t": t,
                            "sin_angle_upper": tr["g"],
                            "angle_upper": math.degrees(math.asin(tr["g"])),
                            "theorem_bound_status": tr["g_status"],
                            "tica_rate_m": rates[m - 1],
                            "tica_rate_m1": rates[m],
                            "whitening_rank": fit.whitening.rank,
                            "lambda_m": tr["lambda_m"],
                            "lambda_m1": tr["lambda_m1"],
                            "eigengap": tr["eigengap"],
                            "numerical_allowance": tr["numerical_allowance"],
                        }
                    )
                    rows.append(r)
        run.update(completed_units=f + 1)
        log.info("%s fit %d done (%s)", ds, f, {k: round(v, 1) for k, v in tm.t.items()})
    np.savez(
        run.path / "spans.npz",
        spans=spans,
        lengths=np.array(lengths, dtype=np.int64),
        whitening_rank=whit_rank,
        selected_m=np.array(ms, dtype=np.int64),
    )
    summary = {}
    for m in ms:
        for L in lengths:
            for t in t_grid:
                sel = [r for r in rows if r["m"] == m and r["pilot_length"] == L and r["t"] == t]
                statuses = sorted({r["estimate_status"] for r in sel})
                summary[f"m{m}_L{L}_t{t}"] = {
                    "n_fits": len(sel),
                    "status_counts": {
                        s: sum(r["estimate_status"] == s for r in sel) for s in statuses
                    },
                    "sin_theta_resolved": _quantiles(
                        [r["sin_theta"] for r in sel if r["estimate_status"] == "RESOLVED"]
                    ),
                    "mode_error_deg": _quantiles(
                        [r.get("mode_error_deg", float("nan")) for r in sel]
                    ),
                    "learned_sin_upper": _quantiles(
                        [r.get("learned_sin_upper", float("nan")) for r in sel]
                    ),
                    "residual_ratio_resolved": _quantiles(
                        [
                            r["residual_ratio"]
                            for r in sel
                            if r.get("residual_ratio_status") == "RESOLVED"
                        ]
                    ),
                    "bound_violations": sum(
                        1
                        for r in sel
                        if r["estimate_status"] == "RESOLVED" and r.get("bound_satisfied") is False
                    ),
                    "note": "median and interquartile range over fits; not confidence intervals",
                }
    out.update(
        {
            "population_run": str(pop["path"]),
            "summary": summary,
            "bound_inputs": pop["json"]["per_landmarks"]["feat"]["bound_inputs"],
            "true_rows_status": {
                f"m{m}_t{t}": true_rows[(m, t)]["estimate_status"] for m in ms for t in t_grid
            },
            "timings_s": tm.t,
            "peak_rss_mib": peak_rss_mib(),
        }
    )
    jdump(run.path / "learned.json", out)
    write_csv(run.path / "learned_rows.csv", rows)
    log.info("%s learned: %d rows; timings %s", ds, len(rows), tm.t)


def _exact_diff_opnorm(a: dict, b: dict) -> float:
    """||S_a - S_b||_op for two exact rational operators S = S_int 2^-exponent."""
    ea, eb = a["exponent"], b["exponent"]
    e = max(ea, eb)
    diff = a["S_int"] * (1 << (e - ea)) - b["S_int"] * (1 << (e - eb))
    return ra.sm_opnorm(ra.exact_to_float(diff, e))


def run_precision(cfg: DictConfig, run: RunDir) -> None:
    import mpmath as mp

    tm = Timer()
    doc, sel_sha = _load_selection(cfg)
    ds = str(cfg.dataset)
    entry = frozen_entry(doc, ds)
    m = int(cfg.precision.m)
    times = [int(t) for t in cfg.precision.times]
    refinements = [
        r
        for r in doc["numerical_refinements"]
        if r["dataset"] == ds and int(r["m"]) == m and set(times) <= set(int(t) for t in r["times"])
    ]
    if not refinements:
        raise ValueError(f"no numerical refinement configured for {ds} m={m} t={times}")
    if m not in [int(x) for x in entry["selected_m"]]:
        raise ValueError(f"m={m} is not a frozen selected count for {ds}")
    _, npz, _ = _load_selected_setup(cfg, entry)
    model = build_model(cfg, npz)
    ar, feat = model["ar"], model["feat"]
    rules = ra.NumericsRules(
        allowance_factor=float(cfg.population.allowance_factor),
        op_factor=float(cfg.population.op_factor),
        sin_factor=float(cfg.population.sin_factor),
        max_precision_change_deg=float(cfg.population.max_precision_change_deg),
    )
    spec = ra.enumerate_hermite(ar.rho, int(cfg.spectrum.n_top), float(cfg.spectrum.tie_tol_log))
    alphas = [tuple(int(v) for v in spec.indices[j]) for j in range(m)]
    with tm("bound_inputs"):
        g64 = ra.make_geometry(ar, feat, "fp64")
        gld = ra.make_geometry(ar, feat, "ld")
        w64 = ra.embedded_modes(g64, spec.indices[:m])
        wld = ra.embedded_modes(gld, spec.indices[:m])
        S0 = ra.operator_S(g64, 0)
        binp = ra.bound_inputs(
            S0,
            ra.operator_allowance(S0, ra.operator_S(gld, 0), rules.allowance_factor)["e_t"],
            w64,
            wld,
            ra.true_gap(spec, m),
            rules,
        )
    dps_list = [int(x) for x in cfg.precision.dps]
    s_min_W = float(feat.gram_eigs[feat.rank - 1])
    results, vectors = {}, {}
    for t in times:
        tiers = {}
        for dps in dps_list:
            with tm(f"operator_t{t}_dps{dps}"):
                ex = ra.exact_integer_operator(ar, feat, t, dps, workers=int(cfg.precision.workers))
            S_f = ra.exact_to_float(ex["S_int"], ex["exponent"])
            lam, V = ra.eig_desc(S_f)
            with tm(f"eigen_t{t}_dps{dps}"):
                eig = ra.mp_subspace_eigen(
                    ex,
                    k=m + int(cfg.precision.extra_vectors),
                    iters=int(cfg.precision.iters),
                    dps=dps,
                    V0=V,
                )
            with tm(f"modes_t{t}_dps{dps}"):
                Wm = ra.exact_true_modes(ex, feat, alphas, dps)
            with mp.workdps(dps):
                Y = eig["Y"]
                Ym = mp.matrix([[Y[i, j] for j in range(m)] for i in range(feat.rank)])
                sv = [mp.sqrt(x) for x in mp.eigsy(Wm.T * Wm, eigvals_only=True)]
                # Gram-Schmidt gives an independent angle calculation at the same precision.
                Qw = _mp_orth(mp, Wm)
                Pm = Ym - Qw * (Qw.T * Ym)
                sin_gs = mp.sqrt(max(max(mp.eigsy(Pm.T * Pm, eigvals_only=True)), mp.mpf(0)))
            with tm(f"angle_t{t}_dps{dps}"):
                # reported: exact sine between the stored mpmath spans (exact integer Gram matrices)
                sn = ra.span_sine(ra.exact_span(Wm), ra.exact_span(Ym))
            tiers[dps] = {
                "ex": ex,
                "theta": [float(x) for x in eig["theta"]],
                "theta_str": [mp.nstr(x, 25) for x in eig["theta"]],
                "residuals": [float(x) for x in eig["residuals"]],
                "residual_history": eig["residual_history"],
                "sin": sn["sin"],
                "sin_mp": sn["sin_mp"],
                "sin_str": mp.nstr(sn["sin_mp"], 40),
                "sin_arith": sn["arith"],
                "sin_arith_mp": sn["arith_mp"],
                "sin_serialization_mp": sn["serialization_mp"],
                "sin_serialization": sn["serialization"],
                "sin_gs_route_str": mp.nstr(sin_gs, 25),
                "Wm_mp": Wm,
                "Y_mp": Y,
                "sigma_min_Wm": float(min(sv)),
                "fp64_rounded_eigs": lam[: m + 3].tolist(),
                "D_round_abs": ex["D_round_abs"],
                "D_max_abs": ex["D_max_abs"],
            }
        lo, hi = dps_list[0], dps_list[-1]
        A, B = tiers[lo], tiers[hi]
        cross = _exact_diff_opnorm(A["ex"], B["ex"])
        # a-priori bound on the operator effect of rounding D to fixed point (entrywise <= D_round):
        # ||W^T dD W||_op <= ||W||_2^2 ||dD||_F <= (1 / s_min) M D_round
        round_bound = feat.M * B["D_round_abs"] / s_min_W
        res = max(B["residuals"][: m + 1])
        e_t = rules.allowance_factor * cross + round_bound + res
        lam_m, lam_m1 = B["theta"][m - 1], B["theta"][m]
        gap = lam_m - lam_m1
        # cross-tier difference of the embedded true modes formed in mpmath (float64 rounding would hide it)
        with mp.workdps(hi):
            dWm = A["Wm_mp"] - B["Wm_mp"]
            dW = rules.allowance_factor * ra.float_up(
                mp.sqrt(max(mp.eigsy(dWm.T * dWm, eigvals_only=True)))
            )
        e_w = ra.mode_term(B["sigma_min_Wm"], dW)
        e_dk = ra.dk_sine_allowance(e_t, gap)
        e_sin = ra.sine_uncertainty(e_dk, e_w, B["sin_arith_mp"], B["sin_serialization_mp"])
        ang = {k: math.degrees(math.asin(v["sin"])) for k, v in tiers.items()}
        change = ra.asin_deg_change(A["sin_mp"], B["sin_mp"])
        g, gstat, graw = ra.bound_value(binp, t)
        row = {
            "dataset": ds,
            "m": m,
            "t": t,
            "tier": f"exact_integer_whitening_dps{hi}",
            "lambda_m": lam_m,
            "lambda_m1": lam_m1,
            "eigengap": gap,
            "numerical_allowance": e_t,
            "cross_precision_op": cross,
            "D_rounding_bound": round_bound,
            "ritz_residual_max": res,
            "sin_theta": B["sin"],
            "sin_theta_lo": A["sin"],
            "angle_deg": ang[hi],
            "precision_change_deg": change,
            "sin_uncertainty": e_sin,
            "sin_uncertainty_dk": ra.float_up(e_dk),
            "sin_uncertainty_modes": ra.float_up(e_w),
            "sin_uncertainty_arith": B["sin_arith"],
            "sin_uncertainty_serialization": B["sin_serialization"],
            "sin_theta_str": B["sin_str"],
            "lambda_ok": lam_m > rules.op_factor * e_t,
            "gap_ok": gap > rules.op_factor * e_t,
            "precision_ok": change < rules.max_precision_change_deg,
            "sin_angle_upper": g,
            "sin_angle_upper_raw": graw,
            "theorem_bound_status": gstat,
        }
        if not (row["lambda_ok"] and row["gap_ok"]):
            row["estimate_status"] = "OPERATOR_UNRESOLVED"
            why = []
            if not row["lambda_ok"]:
                why.append(f"lambda_m={lam_m:.3e} <= {rules.op_factor:g} e_t={e_t:.3e}")
            if not row["gap_ok"]:
                why.append(f"gap={gap:.3e} <= {rules.op_factor:g} e_t={e_t:.3e}")
            row["cutoff_reason"] = "; ".join(why)
        elif not row["precision_ok"]:
            row["estimate_status"] = "PRECISION_UNSTABLE"
            row["cutoff_reason"] = (
                f"angle change {change:.3g} deg >= {rules.max_precision_change_deg:g}"
            )
        elif row["sin_theta"] > rules.sin_factor * e_sin:
            row["estimate_status"] = "RESOLVED"
            row["cutoff_reason"] = ""
        else:
            row["estimate_status"] = "BELOW_NUMERICAL_SENSITIVITY"
            row["cutoff_reason"] = (
                f"s_t={row['sin_theta']:.3e} <= {rules.sin_factor:g} e_sin={e_sin:.3e}"
            )
        for k, v in tiers.items():
            # Ritz vectors and embedded true modes at full precision (exact integers + exponent)
            for name, X in (("Y", v["Y_mp"]), ("Wm", v["Wm_mp"])):
                Xi, E = ra.dyadic_integers(X)
                vectors[f"t{t}_{name}_dps{k}"] = {
                    "shape": list(Xi.shape),
                    "exponent": int(E),
                    "ints_hex": [[hex(int(x)) for x in row] for row in Xi.tolist()],
                }
        drop = ("ex", "Wm_mp", "Y_mp", "sin_mp", "sin_arith_mp", "sin_serialization_mp")
        results[t] = {
            "row": row,
            "tiers": {
                k: {kk: vv for kk, vv in v.items() if kk not in drop} for k, v in tiers.items()
            },
        }
        log.info(
            "%s m=%d t=%d exact tier: %s",
            ds,
            m,
            t,
            {
                k: row[k]
                for k in (
                    "lambda_m",
                    "lambda_m1",
                    "numerical_allowance",
                    "sin_theta",
                    "precision_change_deg",
                    "estimate_status",
                )
            },
        )
        run.update(completed_units=run.status()["completed_units"] + 1)
    jdump(
        run.path / "precision.json",
        {
            "dataset": ds,
            "m": m,
            "selection_sha256": sel_sha,
            "numerical_refinements": refinements,
            "bound_inputs": binp,
            "results": results,
            "timings_s": tm.t,
            "peak_rss_mib": peak_rss_mib(),
        },
    )
    jdump(run.path / "precision_vectors_exact.json", vectors)


def _mp_cols(X, k: int):
    """First k columns of an mpmath matrix (entries copied, not re-rounded)."""
    import mpmath as mp

    if not 1 <= k <= X.cols:
        raise ValueError(f"k={k} for a matrix with {X.cols} columns")
    return mp.matrix([[X[i, j] for j in range(k)] for i in range(X.rows)])


def _mp_orth(mp, A):
    """Orthonormal columns (modified Gram-Schmidt, twice) of an mpmath matrix."""
    Q = A.copy()
    r, k = Q.rows, Q.cols
    for _ in range(2):
        for j in range(k):
            for i in range(j):
                c = mp.fsum(Q[q, i] * Q[q, j] for q in range(r))
                for q in range(r):
                    Q[q, j] -= c * Q[q, i]
            nrm = mp.sqrt(mp.fsum(Q[q, j] ** 2 for q in range(r)))
            for q in range(r):
                Q[q, j] /= nrm
    return Q


def load_exact_vectors(path: pathlib.Path) -> dict:
    """mpmath matrices from precision_vectors_exact.json (exact: integers times a power of two)."""
    raw = json.loads(path.read_text())
    out = {}
    for key, v in raw.items():
        Xi = np.array([[int(x, 16) for x in row] for row in v["ints_hex"]], dtype=object)
        if list(Xi.shape) != v["shape"]:
            raise ValueError(f"{key}: shape {Xi.shape} != {v['shape']}")
        out[key] = ra.mp_from_dyadic(Xi, int(v["exponent"]))
    return out


def run_learned_refine(cfg: DictConfig, run: RunDir) -> None:
    """Recompute learned-span angles against the high-precision Ritz vectors.

    Reuse the stored TICA spans and their mode-estimation bound. Evaluate both
    precision levels, including Davis-Kahan, arithmetic and serialization allowances.
    """
    import mpmath as mp

    tm = Timer()
    doc, sel_sha = _load_selection(cfg)
    ds = str(cfg.dataset)
    entry = frozen_entry(doc, ds)
    prec_dir = _completed(cfg.precision_run)
    learned_dir = _completed(cfg.learned_run)
    pj = json.loads((prec_dir / "precision.json").read_text())
    lj = json.loads((learned_dir / "learned.json").read_text())
    if pj["dataset"] != ds or lj["dataset"] != ds:
        raise ValueError("precision / learned run is for another dataset")
    m = int(pj["m"])
    if m not in [int(x) for x in entry["selected_m"]]:
        raise ValueError(f"m={m} is not a frozen selected count for {ds}")
    times = sorted(int(t) for t in pj["results"])
    with tm("inputs"):
        vec = load_exact_vectors(prec_dir / "precision_vectors_exact.json")
        sp = np.load(learned_dir / "spans.npz")
        spans, lengths = sp["spans"], [int(L) for L in sp["lengths"]]
        stored = {}
        for r in csv.DictReader((learned_dir / "learned_rows.csv").open()):
            if int(r["m"]) == m and int(r["t"]) in times:
                stored[(int(r["fit"]), int(r["pilot_length"]), int(r["t"]))] = r
    rules = ra.NumericsRules(
        allowance_factor=float(cfg.population.allowance_factor),
        op_factor=float(cfg.population.op_factor),
        sin_factor=float(cfg.population.sin_factor),
        max_precision_change_deg=float(cfg.population.max_precision_change_deg),
    )
    lag = int(entry["selected_lag"])
    dps_lo, dps_hi = sorted(int(x) for x in pj["results"][str(times[0])]["tiers"])
    tier_spans = {}
    for t in times:
        Y = {d: vec[f"t{t}_Y_dps{d}"] for d in (dps_lo, dps_hi)}
        tier_spans[t] = {d: ra.exact_span(_mp_cols(Y[d], m)) for d in (dps_lo, dps_hi)}
        tier_spans[t]["float"] = np.array(
            [[ra.float_nearest(Y[dps_hi][i, j]) for j in range(m)] for i in range(Y[dps_hi].rows)]
        )
    rows = []
    for f in range(spans.shape[0]):
        for iL, L in enumerate(lengths):
            What = spans[f, iL][:, :m]
            if not np.isfinite(What).all():
                for t in times:
                    if stored[(f, L, t)]["estimate_status"] != "FIT_FAILED":
                        raise ValueError(f"missing span for a fit that did not fail: {f}, {L}")
                continue
            with tm("angles"):
                What_x = ra.exact_span(What)
            for t in times:
                tr = pj["results"][str(t)]["row"]
                lr = stored[(f, L, t)]
                g = tr["sin_angle_upper"]
                if abs(float(lr["sin_angle_upper"]) - g) > 1e-12 * g:
                    raise ValueError(
                        f"g_t mismatch between the learned and precision runs at t={t}"
                    )
                with tm("angles"):
                    sn = ra.span_sine(What_x, tier_spans[t][dps_hi])
                    sl = ra.span_sine(What_x, tier_spans[t][dps_lo])
                e_dk = ra.dk_sine_allowance(tr["numerical_allowance"], tr["eigengap"])
                e_sin = ra.sine_uncertainty(e_dk, sn["arith_mp"], sn["serialization_mp"])
                s = sn["sin"]
                upper = float(lr["learned_sin_upper"])
                route = ra.subspace_sine(What, tier_spans[t]["float"])  # earlier float64 route
                row = {
                    "dataset": ds, "fit": f, "pilot_length": L, "m": m, "t": t, "lag": lag,
                    "landmarks": int(What.shape[0]), "feature_rank": int(What.shape[0]), "tier": tr["tier"],
                    "sin_theta": s, "sin_theta_lo": sl["sin"], "angle_deg": math.degrees(math.asin(s)),
                    "precision_change_deg": ra.asin_deg_change(sn["sin_mp"], sl["sin_mp"]),
                    "sin_uncertainty": e_sin, "sin_uncertainty_dk": ra.float_up(e_dk),
                    "sin_uncertainty_arith": sn["arith"], "sin_uncertainty_serialization": sn["serialization"],
                    "sin_theta_float64_route": route,
                    "float64_route_abs_err": ra.float_up(abs(mp.mpf(route) - sn["sin_mp"])),
                    "mode_error_sin": float(lr["mode_error_sin"]), "mode_error_deg": float(lr["mode_error_deg"]),
                    "mode_uncertainty": float(lr["mode_uncertainty"]),
                    "learned_sin_upper": upper, "learned_angle_upper": float(lr["learned_angle_upper"]),
                    "sin_angle_upper": g, "angle_upper": math.degrees(math.asin(g)),
                    "theorem_bound_status": tr["theorem_bound_status"],
                    "lambda_m": tr["lambda_m"], "lambda_m1": tr["lambda_m1"], "eigengap": tr["eigengap"],
                    "numerical_allowance": tr["numerical_allowance"],
                    "bound_satisfied": s <= upper + e_sin,
                }  # fmt: skip
                unresolved = (
                    tr["cutoff_reason"] if tr["estimate_status"] == "OPERATOR_UNRESOLVED" else None
                )
                ra._status(row, s, e_sin, rules, unresolved)
                rows.append(row)
        run.update(completed_units=f + 1)
    jdump(
        run.path / "learned_refine.json",
        {
            "dataset": ds, "m": m, "times": times, "selection_sha256": sel_sha,
            "precision_run": str(prec_dir), "learned_run": str(learned_dir),
            "spans": "learned_run/spans.npz (stored float64 spans; no regeneration)",
            "timings_s": tm.t, "peak_rss_mib": peak_rss_mib(),
        },
    )  # fmt: skip
    write_csv(run.path / "learned_refined_rows.csv", rows)
    log.info("%s m=%d learned refine: %d rows", ds, m, len(rows))


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_b_real_alignment", version_base=None
)
def main(cfg: DictConfig) -> None:
    part = str(cfg.part)
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    if part == "freeze":
        run_freeze(cfg)
        return
    if not cfg.get("run_id"):
        raise ValueError("run_id is required; never reuse a COMPLETED id")
    cfg_text = OmegaConf.to_yaml(cfg)
    output_root()
    extra = {
        "part": part,
        "dataset": str(cfg.dataset),
        "config_file_sha256": sha256_file(CONFIG_FILE),
        "library_sha256": sha256_file(pathlib.Path(ra.__file__)),
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    n_units = {
        "population": 2,
        "learned": int(cfg.learned.n_fits),
        "learned_refine": int(cfg.learned.n_fits),
        "precision": len(cfg.precision.times) if cfg.get("precision") else 1,
    }.get(part, 1)
    run = RunDir.create(
        str(cfg.study),
        str(cfg.phase),
        str(cfg.run_id),
        cfg_text,
        requested_units=n_units,
        allow_dirty=str(cfg.phase) == "smoke",
        extra_provenance=extra,
    )
    log.info("run dir %s (part=%s, dataset=%s)", run.path, part, cfg.dataset)
    t0 = time.perf_counter()
    try:
        {
            "setup": run_setup,
            "calibrate": run_calibrate,
            "population": run_population,
            "learned": run_learned,
            "precision": run_precision,
            "learned_refine": run_learned_refine,
        }[part](cfg, run)
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {str(exc)[:500]}")
        raise
    jdump(
        run.path / "cost.json",
        {"wall_seconds": time.perf_counter() - t0, "peak_rss_mib": peak_rss_mib()},
    )
    run.finish("COMPLETED")
    log.info("COMPLETED %s in %.1fs", run.path, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
