"""Circle attainment examples for the residual bound R lambda_{m+1}^r.

For even m, dQ_m/dpi = 1 + R lambda_{m+1}^{r-1/2} psi_{m+1} satisfies the
first m Fourier diagnostics. Dividing the initial coefficient by rho_n^t
reaches each law at the same inspection time; changing m changes the initial law.

Periodic quadrature evaluates population MMD. Sampled MMD uses exact initial
draws and explicit transitions; negative estimates are retained.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from mcmc.paperexp import circle as C
from mcmc.paperexp.runio import RunDir, utc_now


def construction(m: int, r: float, R: float, p: float, sigma: float, t: int) -> dict:
    """Coefficients of the E1 law; predicted residual uses the eigenvalue formula only here."""
    if m < 2 or m % 2:
        raise ValueError(f"m must be a positive even number of complete sine/cosine pairs, got {m}")
    n = m // 2 + 1
    lam = float(C.eigenvalue(m + 1))
    target_coeff = R * lam ** (r - 0.5)  # <h_t, psi_{m+1}>_{L2(pi)}
    rho = float(C.lazy_rho(n, p, sigma))
    init_coeff = target_coeff / rho**t
    amplitude0 = math.sqrt(2.0) * init_coeff  # dQ_0/dpi = 1 + amplitude0 cos(n x)
    if not amplitude0 < 1.0:
        raise ValueError(f"initial density not positive: amplitude {amplitude0}")
    return {
        "m": m,
        "r": r,
        "freq": n,
        "lambda_m1": lam,
        "rho_freq": rho,
        "target_coeff": target_coeff,
        "init_coeff": init_coeff,
        "init_amplitude": amplitude0,
        "predicted_residual": R * lam**r,
        "predicted_residual_sq": (R * lam**r) ** 2,
    }


def population_setting(m: int, r: float, cfg: DictConfig) -> dict:
    p, sigma, t = float(cfg.sampler.p), float(cfg.sampler.sigma), int(cfg.t_inspect)
    con = construction(m, r, float(cfg.R), p, sigma, t)
    q = cfg.quadrature
    nodes = int(q.start_nodes)
    history = []
    while True:
        if nodes > int(q.max_nodes):
            raise RuntimeError(
                f"quadrature did not resolve m={m}, r={r} within {q.max_nodes} nodes"
            )
        x = C.grid_nodes(nodes)
        h0 = con["init_coeff"] * C.fourier_basis(x, m + 1)
        ht = C.propagate_density_quadrature(h0, t, p, sigma)
        mmd_sq = C.mmd_sq_quadrature(ht)
        if not mmd_sq > 0.0:
            raise RuntimeError(f"non-positive quadrature MMD^2 {mmd_sq} at m={m}, r={r}, N={nodes}")
        history.append((nodes, mmd_sq))
        if len(history) >= 2:
            rel_change = abs(math.sqrt(mmd_sq) / math.sqrt(history[-2][1]) - 1.0)
            if rel_change < float(q.rel_tol):
                break
        nodes *= 2

    # Independent fine-grid checks of the laws themselves.
    xf = C.grid_nodes(int(q.check_nodes))
    h0 = con["init_coeff"] * C.fourier_basis(xf, m + 1)
    ht = C.propagate_density_quadrature(h0, t, p, sigma)
    moments = {}
    for label, h in (("t0", h0), ("t", ht)):
        mon = np.array(
            [
                math.sqrt(C.eigenvalue(j)) * C.integrate_periodic(C.fourier_basis(xf, j) * h)
                for j in range(1, m + 1)
            ]
        )
        moments[label] = {
            "min_density": float((1.0 + h).min()),
            "normalisation_error": C.integrate_periodic(h),
            "max_abs_monitored_moment": float(np.abs(mon).max()),
            "monitored_component_norm": float(np.sqrt(np.sum(mon**2))),
            "coeff_psi_m1": C.integrate_periodic(C.fourier_basis(xf, m + 1) * h),
        }
    if moments["t0"]["min_density"] <= 0.0 or moments["t"]["min_density"] <= 0.0:
        raise RuntimeError(f"non-positive density for m={m}, r={r}: {moments}")
    measured = math.sqrt(history[-1][1])
    return {
        **con,
        "measured_residual": measured,
        "measured_residual_sq": history[-1][1],
        "relative_difference": measured / con["predicted_residual"] - 1.0,
        "nodes_used": history[-1][0],
        "resolution_error": abs(measured / math.sqrt(history[-2][1]) - 1.0),
        "mmd_by_nodes": [[n_, math.sqrt(v)] for n_, v in history],
        "check_nodes": int(q.check_nodes),
        "min_density_t0": moments["t0"]["min_density"],
        "min_density_t": moments["t"]["min_density"],
        "normalisation_error_t0": moments["t0"]["normalisation_error"],
        "normalisation_error_t": moments["t"]["normalisation_error"],
        "max_abs_monitored_moment_t0": moments["t0"]["max_abs_monitored_moment"],
        "max_abs_monitored_moment_t": moments["t"]["max_abs_monitored_moment"],
        "monitored_component_norm_t": moments["t"]["monitored_component_norm"],
        "coeff_psi_m1_t_numerical": moments["t"]["coeff_psi_m1"],
    }


def valid_mmd_interval(
    vnorm: float, rounding: float, n_total: int, delta: float
) -> tuple[float, float, float]:
    """Distribution-free interval for MMD(Q, pi) from N iid replicas (Appendix E.2 radius).

    With mu_pi = 0 and k(x, x) = 1, P(| ||mu_hat_N|| - MMD | > b) <= P(||mu_hat_N - mu_Q|| > b) <= delta
    for b = 1/sqrt(N) + sqrt(2 log(1/delta)/N) (expected norm <= 1/sqrt(N), bounded differences 2/N).
    The computed ||mu_hat_N||^2 carries an absolute rounding bound, applied outward.
    Returns (lo, hi, b) with lo clipped at 0 only as a reported endpoint (MMD >= 0).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    b = 1.0 / math.sqrt(n_total) + math.sqrt(2.0 * math.log(1.0 / delta) / n_total)
    lo = math.sqrt(max(vnorm - rounding, 0.0)) - b
    hi = math.sqrt(vnorm + rounding) + b
    return max(lo, 0.0), hi, b


def _sampled_setting(task: tuple) -> dict:
    """Run independent repetitions of one (m, r, J) setting and pool their replicas."""
    m, r, j_idx, J, reps, entropy, R, p, sigma, t, steps = task
    con = construction(m, r, R, p, sigma, t)
    pooled = np.empty(reps * J, dtype=np.float64)
    u = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        seed = np.random.SeedSequence([entropy, m, int(round(r * 1000)), j_idx, rep])
        rng = np.random.default_rng(seed)
        x0 = C.sample_cosine_law(rng, J, con["freq"], con["init_amplitude"])
        xt = C.simulate_explicit(x0, steps, rng, p, sigma)
        u[rep] = C.unbiased_mmd_sq(xt)
        pooled[rep * J : (rep + 1) * J] = xt
    assert np.isfinite(u).all()
    vnorm, rounding = C.empirical_mean_embedding_sq_extended(pooled)
    return {
        "m": m,
        "r": r,
        "j_idx": j_idx,
        "J": J,
        "u": u,
        "pooled_vnorm": vnorm,
        "pooled_rounding": rounding,
    }


def summarise_sampled(results: list[dict], cfg: DictConfig) -> list[dict]:
    """Compute nominal t-intervals and distribution-free pooled intervals.

    Near zero, skewed U-statistics can give the t-interval less than 95% coverage.
    The pooled intervals use failure budgets beta per setting and beta/n_settings
    for simultaneous coverage.
    """
    rows = []
    beta = float(cfg.valid_interval.beta)
    n_settings = len(results)
    for res in sorted(results, key=lambda d: (d["r"], d["m"], d["J"])):
        m, r, J, u = res["m"], res["r"], res["J"], res["u"]
        n = u.shape[0]
        if n < 2:
            raise ValueError("need at least two repetitions for the nominal t-interval")
        con = construction(
            m, r, float(cfg.R), float(cfg.sampler.p), float(cfg.sampler.sigma), int(cfg.t_inspect)
        )
        mean = float(u.mean())
        sd = float(u.std(ddof=1))
        se = sd / math.sqrt(n)
        tq = float(stats.t.ppf(0.975, n - 1))
        lo, hi = mean - tq * se, mean + tq * se
        resolved_t = bool(lo > 0.0)
        n_total = n * J
        v_lo, v_hi, v_b = valid_mmd_interval(
            res["pooled_vnorm"], res["pooled_rounding"], n_total, beta
        )
        s_lo, s_hi, s_b = valid_mmd_interval(
            res["pooled_vnorm"], res["pooled_rounding"], n_total, beta / n_settings
        )
        rows.append(
            {
                "m": m,
                "r": r,
                "J": J,
                "repetitions": n,
                "mean_unbiased_mmd_sq": mean,
                "sd": sd,
                "se": se,
                "t_nominal_ci95_lo": lo,
                "t_nominal_ci95_hi": hi,
                "n_negative": int((u < 0).sum()),
                "predicted_residual_sq": con["predicted_residual_sq"],
                "z_vs_predicted": (mean - con["predicted_residual_sq"]) / se,
                "resolved_t_nominal": resolved_t,
                # square roots only where the nominal interval's lower end is positive
                "sqrt_mean": math.sqrt(mean) if resolved_t else float("nan"),
                "sqrt_t_nominal_ci95_lo": math.sqrt(lo) if resolved_t else float("nan"),
                "sqrt_t_nominal_ci95_hi": math.sqrt(hi) if resolved_t else float("nan"),
                "predicted_residual": con["predicted_residual"],
                "valid_pooled_replicas": n_total,
                "valid_pooled_vstat_sq": res["pooled_vnorm"],
                "valid_pooled_rounding_bound": res["pooled_rounding"],
                "valid_delta_pointwise": beta,
                "valid_radius_pointwise": v_b,
                "valid_mmd_lo_pointwise": v_lo,
                "valid_mmd_hi_pointwise": v_hi,
                "resolved_valid_pointwise": bool(v_lo > 0.0),
                "valid_delta_simultaneous": beta / n_settings,
                "valid_radius_simultaneous": s_b,
                "valid_mmd_lo_simultaneous": s_lo,
                "valid_mmd_hi_simultaneous": s_hi,
                "resolved_valid_simultaneous": bool(s_lo > 0.0),
                "explicit_transitions_executed": int(J) * int(cfg.sampled.explicit_steps) * n,
            }
        )
    return rows


def _write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
            )


def run_study(cfg: DictConfig) -> RunDir:
    phase = str(cfg.phase)
    if phase not in cfg.phases:
        raise ValueError(f"unknown phase {phase!r}")
    ph = cfg.phases[phase]
    if int(cfg.sampled.explicit_steps) != int(cfg.t_inspect):
        raise ValueError("sampled.explicit_steps must equal t_inspect (the construction's time)")
    run_id = cfg.run_id
    if run_id is None:
        if phase != "smoke":
            raise ValueError("run_id is required for pilot/main phases")
        run_id = "smoke_" + utc_now().replace(":", "").replace("-", "")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    pop_settings = [(int(m), float(r)) for r in cfg.r_values for m in cfg.m_values]
    tasks = []
    for r in cfg.r_values:
        for m in cfg.sampled.m_values:
            for j_idx, J in enumerate(ph.replica_counts):
                tasks.append(
                    (
                        int(m),
                        float(r),
                        j_idx,
                        int(J),
                        int(ph.repetitions),
                        int(ph.seed_entropy),
                        float(cfg.R),
                        float(cfg.sampler.p),
                        float(cfg.sampler.sigma),
                        int(cfg.t_inspect),
                        int(cfg.sampled.explicit_steps),
                    )
                )
    run = RunDir.create(
        str(cfg.study),
        phase,
        str(run_id),
        config_text,
        requested_units=len(pop_settings) + len(tasks),
        allow_dirty=phase in ("smoke", "pilot"),
        extra_provenance={"numpy": np.__version__, "seed_entropy": int(ph.seed_entropy)},
    )
    t_start = time.perf_counter()
    cpu0 = os.times()
    try:
        pop_rows = []
        for m, r in pop_settings:
            pop_rows.append(population_setting(m, r, cfg))
            run.update(completed_units=len(pop_rows))
        t_pop = time.perf_counter() - t_start
        _write_csv(run.path / "population.csv", pop_rows)
        (run.path / "population.json").write_text(json.dumps(pop_rows, indent=2))

        results = []
        with ProcessPoolExecutor(max_workers=int(cfg.n_workers)) as pool:
            for k, out in enumerate(pool.map(_sampled_setting, tasks, chunksize=1)):
                results.append(out)
                run.update(completed_units=len(pop_rows) + k + 1)
        raw_arr = np.array(
            [
                (d["m"], d["r"], d["j_idx"], rep, d["J"], d["u"][rep])
                for d in results
                for rep in range(d["u"].shape[0])
            ],
            dtype=np.float64,
        )
        np.savez(
            run.path / "sampled_raw.npz",
            table=raw_arr,
            columns=np.array(["m", "r", "J_index", "repetition", "J", "unbiased_mmd_sq"]),
            pooled=np.array(
                [(d["m"], d["r"], d["J"], d["pooled_vnorm"], d["pooled_rounding"]) for d in results]
            ),
            pooled_columns=np.array(["m", "r", "J", "pooled_vstat_sq", "pooled_rounding_bound"]),
        )
        samp_rows = summarise_sampled(results, cfg)
        _write_csv(run.path / "sampled_summary.csv", samp_rows)
        wall = time.perf_counter() - t_start
        cpu1 = os.times()
        summary = {
            "phase": phase,
            "run_id": run_id,
            "seed_entropy": int(ph.seed_entropy),
            "wall_seconds_total": wall,
            "wall_seconds_population": t_pop,
            "cpu_seconds_self": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
            "cpu_seconds_children": (cpu1.children_user - cpu0.children_user)
            + (cpu1.children_system - cpu0.children_system),
            "peak_rss_kib_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_kib_largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "max_abs_relative_difference": max(abs(row["relative_difference"]) for row in pop_rows),
            "max_resolution_error": max(row["resolution_error"] for row in pop_rows),
            "min_density_overall": min(
                min(row["min_density_t0"], row["min_density_t"]) for row in pop_rows
            ),
            "max_abs_monitored_moment_overall": max(
                max(row["max_abs_monitored_moment_t0"], row["max_abs_monitored_moment_t"])
                for row in pop_rows
            ),
            "max_abs_normalisation_error": max(
                max(abs(row["normalisation_error_t0"]), abs(row["normalisation_error_t"]))
                for row in pop_rows
            ),
            "sampled_settings": len(samp_rows),
            "sampled_resolved_t_nominal": sum(row["resolved_t_nominal"] for row in samp_rows),
            "sampled_resolved_valid_pointwise": sum(
                row["resolved_valid_pointwise"] for row in samp_rows
            ),
            "sampled_resolved_valid_simultaneous": sum(
                row["resolved_valid_simultaneous"] for row in samp_rows
            ),
        }
        (run.path / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=f"{type(exc).__name__}: {exc}")
        raise
    return run


@hydra.main(
    config_path="../../../configs/experiment",
    config_name="exp_r1_residual_vs_diagnostics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if str(cfg.task) != "main":
        raise ValueError(f"unknown task {cfg.task!r}")
    run = run_study(cfg)
    print(f"run dir: {run.path}", flush=True)


if __name__ == "__main__":
    main()
