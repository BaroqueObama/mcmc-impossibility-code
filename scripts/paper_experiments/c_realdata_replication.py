"""Replicate logistic stopping comparisons with independent chain bundles.

German Credit, Pima and Australian Credit use RWM/MALA with starts I1/I2:
20 bundles of 64 chains, prefixes J=4,16,64 and checkpoints through 4096.
Kernels, operators, NUTS references and starting pools are shared with the
original experiment, so the comparison is conditional on those inputs.

Run freeze, run --model MODEL --sampler SAMPLER, then analyze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import resource
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c_realdata_budget_followup as c  # noqa: E402

OUT_R = c.OUT / "replication"
MODELS = ("hier", "hier_pima", "hier_australian")
SAMPLERS = ("mala", "rwm")
LAWS = ("I1", "I2")
N_BUNDLES = 20
BUNDLE = 64
PREFIXES = (4, 16, 64)
GRID = [0] + [2**k for k in range(13)]  # 0, 1, 2, ..., 4096
STAGE2_RUN = ("main", "c_v1")
RUN_PREFIX = "repl_v1"


def cid(model: str, sampler: str) -> str:
    return f"c-repl-{model}__{sampler}"


def bundle_seed(model: str, sampler: str, law: str, b: int) -> int:
    from mcmc.paperexp.rng import seed_for

    return seed_for(cid(model, sampler), f"bundle-{law}", b)


def bench_seed(model: str, sampler: str) -> int:
    from mcmc.paperexp.rng import seed_for

    return seed_for(cid(model, sampler), "bench-I1", 0)


def design() -> dict:
    return {
        "study": "c_realdata_followup / replication",
        "version": "v1",
        "combinations": [f"{m}__{s}__{law}" for m in MODELS for s in SAMPLERS for law in LAWS],
        "n_bundles": N_BUNDLES,
        "bundle_size": BUNDLE,
        "prefixes": list(PREFIXES),
        "grid": GRID,
        "n_trajectories": len(MODELS) * len(SAMPLERS) * len(LAWS) * N_BUNDLES * BUNDLE,
        "seeds": "seed_for('c-repl-<model>__<sampler>', 'bundle-<law>', b), b = 0..19 (torch CUDA "
        "generator, one per bundle); benchmark bundle: stage 'bench-I1', rep 0 (not used in results)",
        "frozen_from_R2": "per unit: config.yaml frozen h and start laws (pool set, overdispersion 3, "
        "box radius 2), kernel/Nystrom basis (setup.npz), operators O_t (sigma.npz), NUTS reference "
        "sets; the R2 code path (Context, StartLaws.sample, MHSampler.run)",
        "readings": "identical to Stage 2: prefix-group feature means minus the mean feature embedding of "
        "NUTS runs 0-7 of the evaluation set; spans S3 / S4own / S4same, m = 8 (4), margins none / "
        "halfsplit, R in {0.5, 1, 2, 4}, eps in {0.10, 0.05}",
        "heldout": "the Stage-2 held-out estimate (run main/c_v1, split 'heldout': saved-ensemble "
        "evaluation half, replicas 2048..4095, vs NUTS runs 8-15) at each t <= 4096; the bundles are new "
        "draws of the same chain law (same frozen sampler and start law), checked by a two-sample MMD^2 "
        "between all 1280 bundle chains and that evaluation half at every t (law-identity check)",
        "comparisons": "bundle-to-bundle variation (quantiles over 20 bundles) vs the Stage-2 saved-state "
        "groups (512 x 4, 128 x 16, 32 x 64) at the same t; crossings restricted to t <= 4096 for both",
        "statuses": "plug-ins: reference-based plug-in at an assumed radius (feature space); held-out: "
        "empirical estimate; no confidence statement beyond the Stage-1 floor table",
        "conditioning": "all bundles share the NUTS references (runs 0-7 / 8-15), the pool (start laws), the "
        "calibration run (features), the operators O_t, the frozen h and the R2 grid choice (horizon "
        "pilot used all 16 runs); bundles are independent of each other and of the saved ensembles "
        "only given these; this is an empirical practicality replication, not a 95% coverage study",
        "no_binomial_intervals": True,
    }


def build_manifest() -> dict:
    from mcmc.paperexp.rng import seed_for

    seeds, known = {}, set()
    for m in MODELS:
        for s in SAMPLERS:
            gcid = f"gc-{m}__{s}"
            for off in (0, 1000, 2000):
                for st in (
                    ["r2-sigma", "r2-sigma-starts", "r2-tmax", "pilot-tune"]
                    + [f"r2-ens-{law}" for law in ("I0", "I1", "I2", "I3")]
                    + [f"r2-nuts-{law}" for law in ("I1", "I2")]
                ):
                    known.add(seed_for(gcid, st, off))
            seeds[f"{m}__{s}"] = {
                law: [bundle_seed(m, s, law, b) for b in range(N_BUNDLES)] for law in LAWS
            }
            seeds[f"{m}__{s}"]["bench"] = bench_seed(m, s)
    flat = [v for d in seeds.values() for k, x in d.items() for v in (x if k != "bench" else [x])]
    if len(set(flat)) != len(flat) or set(flat) & known:
        raise RuntimeError("replication seeds collide with each other or with R2 streams")
    root = c.out_root()
    st2 = root / c.STUDY_C / STAGE2_RUN[0] / STAGE2_RUN[1]
    src = {}
    for m in MODELS:
        for s in SAMPLERS:
            r2 = c.unit_dirs(m, s)["r2"]
            for f in ("config.yaml", "setup.npz", "sigma.npz"):
                src[str(r2 / f)] = c.sha256_file(r2 / f)
            src[str(st2 / f"{m}__{s}.npz")] = c.sha256_file(st2 / f"{m}__{s}.npz")
    return {
        "design_sha256": hashlib.sha256(json.dumps(design(), sort_keys=True).encode()).hexdigest(),
        "seeds": seeds,
        "sources": src,
    }


def cmd_freeze(args) -> None:
    c.write_json(OUT_R / "design_replication.json", design())
    c.write_json(OUT_R / "manifest_replication.json", build_manifest())
    print(OUT_R)


def verify_frozen() -> dict:
    """Check the saved replication design, seeds and source hashes."""
    des_f, man_f = OUT_R / "design_replication.json", OUT_R / "manifest_replication.json"
    des, man = json.loads(des_f.read_text()), json.loads(man_f.read_text())
    expected = design()
    fields = ("study", "version", "combinations", "n_bundles", "bundle_size",
              "prefixes", "grid", "n_trajectories", "no_binomial_intervals")
    if any(des[k] != expected[k] for k in fields):
        raise RuntimeError("design_replication.json parameters differ from the code")
    digest = hashlib.sha256(json.dumps(des, sort_keys=True).encode()).hexdigest()
    if man["design_sha256"] != digest:
        raise RuntimeError("replication manifest design hash mismatch")
    for model in MODELS:
        for sampler in SAMPLERS:
            seeds = man["seeds"][f"{model}__{sampler}"]
            expected_seeds = {law: [bundle_seed(model, sampler, law, b)
                                    for b in range(N_BUNDLES)] for law in LAWS}
            expected_seeds["bench"] = bench_seed(model, sampler)
            if seeds != expected_seeds:
                raise RuntimeError(f"{model}/{sampler}: replication seeds differ from the code")
    c.verify_hashes(man["sources"])
    return {des_f.name: c.sha256_file(des_f), man_f.name: c.sha256_file(man_f)}


# Chain generation
def make_context(model: str, sampler: str, device: str):
    from omegaconf import OmegaConf

    from mcmc.experiments import exp_r2_german_credit as r2x

    d = c.unit_dirs(model, sampler)
    cfg = OmegaConf.merge(
        OmegaConf.create((d["r2"] / "config.yaml").read_text()), {"device": device}
    )
    ctx = r2x.Context(cfg, types.SimpleNamespace(path=d["r2"]))  # read-only; adopts the saved basis
    ctx.load_frozen()
    if ctx.grid[: len(GRID)] != GRID:
        raise RuntimeError(f"replication grid is not a prefix of the R2 grid {ctx.grid}")
    return ctx


def generate_bundle(ctx, law: str, seed: int, device: str) -> dict:
    import torch

    from mcmc.paperexp.rng import torch_gen

    gen = torch_gen(seed, device)
    x0 = ctx.laws.sample(law, BUNDLE, gen)
    states = {}

    def cb(t, x):
        states[t] = x.detach().cpu().numpy().copy()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = ctx.sampler().run(x0, GRID, gen, store=False, on_checkpoint=cb)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    arr = np.stack([states[t] for t in GRID])
    if arr.shape != (len(GRID), BUNDLE, ctx.post.d) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad bundle states {arr.shape}")
    return {
        "states": arr,
        "accept_per_chain": out["accept_count_per_chain"].cpu().numpy(),
        "info": {
            "seed": seed,
            "wall_s": wall,
            "accept_rate": out["accept_rate"],
            "counters": out["counters"],
            "gpu_peak_bytes": int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0,
            "rss_max_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }


def cmd_bench(args) -> None:
    import torch

    from mcmc.paperexp.runio import RunDir

    torch.set_num_threads(args.threads)
    hashes = verify_frozen()
    run = RunDir.create(
        c.STUDY_C,
        "pilot",
        f"{RUN_PREFIX}_bench",
        json.dumps(design()),
        requested_units=len(MODELS) * len(SAMPLERS),
        extra_provenance=hashes,
    )
    res = []
    try:
        for m in MODELS:
            for s in SAMPLERS:
                ctx = make_context(m, s, args.device)
                a = generate_bundle(ctx, "I1", bench_seed(m, s), args.device)
                b = generate_bundle(ctx, "I1", bench_seed(m, s), args.device)  # determinism witness
                same = bool(np.array_equal(a["states"], b["states"]))
                if not same:
                    raise RuntimeError(f"{m}__{s}: bundle generation is not deterministic")
                res.append(
                    {
                        "unit": f"{m}__{s}",
                        "law": "I1",
                        "chains": BUNDLE,
                        "steps": GRID[-1],
                        **a["info"],
                        "rerun_wall_s": b["info"]["wall_s"],
                        "rerun_identical": same,
                    }
                )
                print(json.dumps(res[-1]), flush=True)
                run.update(completed_units=len(res))
        tot = sum(r["rerun_wall_s"] for r in res) / len(res)  # warm wall per bundle (2nd run)
        proj = {
            "per_bundle_warm_wall_s_mean": tot,
            "bundles_total": len(MODELS) * len(SAMPLERS) * len(LAWS) * N_BUNDLES,
            "projected_gpu_h_total": sum(r["rerun_wall_s"] for r in res)
            * len(LAWS)
            * N_BUNDLES
            / 3600,
        }
        c.write_json(run.path / "bench.json", {"bundles": res, "projection": proj})
        run.finish("COMPLETED")
        print(json.dumps(proj))
    except BaseException as exc:
        run.finish("FAILED", note=repr(exc)[:500])
        raise


def cmd_run(args) -> None:
    import torch

    from mcmc.paperexp.runio import RunDir

    torch.set_num_threads(args.threads)
    hashes = verify_frozen()
    m, s = args.model, args.sampler
    run = RunDir.create(
        c.STUDY_C,
        "main",
        f"{RUN_PREFIX}_{m}__{s}",
        json.dumps(design()),
        requested_units=len(LAWS) * N_BUNDLES,
        extra_provenance=hashes,
    )
    try:
        ctx = make_context(m, s, args.device)
        k = 0
        for law in LAWS:
            for b in range(N_BUNDLES):
                g = generate_bundle(ctx, law, bundle_seed(m, s, law, b), args.device)
                np.savez(
                    run.path / f"{law}_b{b:02d}.npz",
                    grid=np.asarray(GRID),
                    states=g["states"],
                    accept_per_chain=g["accept_per_chain"],
                )
                c.write_json(
                    run.path / f"{law}_b{b:02d}.json", {"law": law, "bundle": b, **g["info"]}
                )
                k += 1
                run.update(completed_units=k)
                print(f"[run] {m}__{s} {law} b{b:02d} {g['info']['wall_s']:.1f} s", flush=True)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=repr(exc)[:500])
        raise


# Compare new bundles with the saved held-out estimates
def two_sample_mmd2(k, X, Y, gx: int, gy: int) -> tuple[float, float]:
    """U-statistic MMD^2 between two iid samples split into gx / gy groups, with a delete-one-group
    jackknife SE over both (groups are independent blocks of chains)."""
    import torch

    nx, ny = X.shape[0], Y.shape[0]
    if nx % gx or ny % gy:
        raise ValueError("group counts must divide the sample sizes")
    Z = torch.as_tensor(np.concatenate([X, Y]))
    K = k(Z, Z).numpy()
    K = 0.5 * (K + K.T)
    sx, sy = nx // gx, ny // gy
    Sxx = K[:nx, :nx].reshape(gx, sx, gx, sx).sum(axis=(1, 3))
    Syy = K[nx:, nx:].reshape(gy, sy, gy, sy).sum(axis=(1, 3))
    Sxy = K[:nx, nx:].reshape(gx, sx, gy, sy).sum(axis=(1, 3))

    def est(ix, iy):
        n, p = ix.size * (nx // gx), iy.size * (ny // gy)
        a = (Sxx[np.ix_(ix, ix)].sum() - n) / (n * (n - 1))
        b = (Syy[np.ix_(iy, iy)].sum() - p) / (p * (p - 1))
        cc = Sxy[np.ix_(ix, iy)].sum() / (n * p)
        return a + b - 2 * cc

    ax, ay = np.arange(gx), np.arange(gy)
    full = est(ax, ay)
    jx = np.array([est(np.delete(ax, i), ay) for i in range(gx)])
    jy = np.array([est(ax, np.delete(ay, i)) for i in range(gy)])
    var = (gx - 1) / gx * ((jx - jx.mean()) ** 2).sum() + (gy - 1) / gy * (
        (jy - jy.mean()) ** 2
    ).sum()
    return float(full), float(math.sqrt(var))


def cmd_analyze(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    t_all = time.time()
    root = c.out_root() / c.STUDY_C / "main"
    st2 = c.out_root() / c.STUDY_C / STAGE2_RUN[0] / STAGE2_RUN[1]
    nt = len(GRID)
    off = c.group_offsets()
    rows_b, rows_q, rows_x, rows_cmp, rows_law, rows_bund = [], [], [], [], [], []
    for m in MODELS:
        for s in SAMPLERS:
            rdir = root / f"{RUN_PREFIX}_{m}__{s}"
            if json.loads((rdir / "RUN-STATUS.json").read_text())["status"] != "COMPLETED":
                raise RuntimeError(f"{rdir} not COMPLETED")
            ctx = make_context(m, s, "cpu")
            ctx.ensure_ref()
            z2 = dict(np.load(st2 / f"{m}__{s}.npz", allow_pickle=False))
            if list(z2["grid"][:nt]) != GRID or str(z2["sets"][0]) != "primary":
                raise RuntimeError("Stage-2 grid / set mismatch")
            labels = ctx.ref.run_labels
            read_idx = np.array([labels.index(f"primary:{q}") for q in c.READ_RUNS])
            fref = ctx.ref.feat_run_means[read_idx].mean(0)
            V0 = z2["op__V0"]
            Vt = z2["op__Vt"][:nt]
            base = {"model": m, "dataset": c.DISPLAY[m], "sampler": s}
            for law in LAWS:
                pre = f"{law}__"
                H = {
                    k: z2[pre + "held__" + k][0, 0, :nt]
                    for k in ("mmd2", "mmd2_se", "feat2", "feat2_se")
                }
                st_full = {e: c.status_code(H["mmd2"], H["mmd2_se"], e) for e in c.EPS}
                _, feat_lo, _ = c.mmd_scale_interval(H["feat2"], H["feat2_se"])
                # bundles: features, prefix readings
                cV0 = np.empty((N_BUNDLES, nt, len(PREFIXES), 8))
                cVt = np.empty_like(cV0)
                stuck = np.zeros((N_BUNDLES, len(PREFIXES)), dtype=bool)
                pooled = []
                for b in range(N_BUNDLES):
                    zb = np.load(rdir / f"{law}_b{b:02d}.npz")
                    jb = json.loads((rdir / f"{law}_b{b:02d}.json").read_text())
                    if jb["seed"] != bundle_seed(m, s, law, b) or list(zb["grid"]) != GRID:
                        raise RuntimeError(f"{rdir}: bundle {law} {b} seed/grid mismatch")
                    X = zb["states"]
                    pooled.append(X)
                    acc = zb["accept_per_chain"]
                    for pj, J in enumerate(PREFIXES):
                        stuck[b, pj] = bool((acc[:J] == 0).any())
                    for i in range(nt):
                        F = ctx.feats(torch.as_tensor(X[i])).numpy()
                        for pj, J in enumerate(PREFIXES):
                            rg = F[:J].mean(0) - fref
                            cV0[b, i, pj] = rg @ V0[:, :8]
                            cVt[b, i, pj] = rg @ Vt[i][:, :8]
                    rows_bund.append(
                        {
                            **base,
                            "law": law,
                            "bundle": b,
                            "seed": jb["seed"],
                            "wall_s": jb["wall_s"],
                            "accept_rate": jb["accept_rate"],
                            "logprob_evals": jb["counters"]["logprob_evals"],
                            "grad_evals": jb["counters"]["grad_evals"],
                            "gpu_peak_bytes": jb["gpu_peak_bytes"],
                            **{
                                f"never_accepted_in_prefix{J}": int((acc[:J] == 0).sum())
                                for J in PREFIXES
                            },
                        }
                    )
                # law-identity check: all 1280 bundle chains vs the saved evaluation half
                states_saved = (
                    np.load(c.unit_dirs(m, s)["r2"] / f"ens_{law}_states.npz")["states"]
                    if (c.unit_dirs(m, s)["r2"] / f"ens_{law}_states.npz").exists()
                    else np.load(c.unit_dirs(m, s)["r1_states"] / f"states_{law}.npz")["states"]
                )
                P = np.stack(pooled, axis=1)  # (nt, 20, 64, d)
                for i, t in enumerate(GRID):
                    mm, se = two_sample_mmd2(
                        ctx.kernel,
                        P[i].reshape(-1, P.shape[-1]),
                        states_saved[i, c.N_OBS :],
                        N_BUNDLES,
                        16,
                    )
                    rows_law.append(
                        {
                            **base,
                            "law": law,
                            "t": t,
                            "mmd2_bundles_vs_saved_eval": mm,
                            "se": se,
                            "z": mm / se if se > 0 else np.nan,
                            "label": "law-identity check (two-sample U-statistic, group jackknife SE)",
                        }
                    )
                # saved-state groups (Stage 2, split heldout) restricted to the replication grid
                sV0 = z2[pre + "coordV0"][0, 0, :nt]
                sVt = z2[pre + "coordVt"][0, 0, :nt]
                never = z2[pre + "never_accepted"]
                for fam in c.FAMILIES:
                    for mm_ in c.M_LIST:
                        src_b = cV0 if fam in ("S3", "S4same") else cVt
                        src_s = sV0 if fam in ("S3", "S4same") else sVt
                        hb = (src_b[..., :mm_] ** 2).sum(-1)  # (20, nt, 3)
                        hs = (src_s[..., :mm_] ** 2).sum(-1)  # (nt, 672)
                        for margin in c.MARGINS:
                            w2 = c.width2(z2, fam, mm_, margin)[:nt]
                            for R in c.R_ASSUMED:
                                for pj, J in enumerate(PREFIXES):
                                    Pb = c.plugin(hb[:, :, pj].T, w2, R)  # (nt, 20)
                                    o, n = off[J]
                                    Ps = c.plugin(hs[:, o : o + n], w2, R)  # (nt, n)
                                    qb = np.quantile(Pb, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
                                    qs = np.quantile(Ps, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
                                    for i, t in enumerate(GRID):
                                        rows_q.append(
                                            {
                                                **base,
                                                "law": law,
                                                "J": J,
                                                "family": fam,
                                                "m": mm_,
                                                "margin": margin,
                                                "R": R,
                                                "t": t,
                                                **{
                                                    f"bundles_q{q}": qb[k, i]
                                                    for k, q in enumerate((5, 25, 50, 75, 95))
                                                },
                                                "bundles_min": Pb[i].min(),
                                                "bundles_max": Pb[i].max(),
                                                **{
                                                    f"saved_q{q}": qs[k, i]
                                                    for k, q in enumerate((5, 25, 50, 75, 95))
                                                },
                                                "bundles_frac_le_0.1": float(
                                                    (Pb[i] <= 0.10).mean()
                                                ),
                                                "saved_frac_le_0.1": float((Ps[i] <= 0.10).mean()),
                                                "bundles_frac_below_heldout_feat_lo": float(
                                                    (Pb[i] < feat_lo[i]).mean()
                                                ),
                                                "label": "reference-based plug-in at an assumed radius (feature space)",
                                            }
                                        )
                                    if (
                                        fam == c.PRIMARY["family"]
                                        and mm_ == c.PRIMARY["m"]
                                        and margin == c.PRIMARY["margin"]
                                        and R == c.PRIMARY["R"]
                                    ):
                                        from scipy.stats import ks_2samp

                                        for i, t in enumerate(GRID):
                                            ks = ks_2samp(Pb[i], Ps[i])
                                            rows_cmp.append(
                                                {
                                                    **base,
                                                    "law": law,
                                                    "J": J,
                                                    "t": t,
                                                    "bundle_median": np.median(Pb[i]),
                                                    "saved_median": np.median(Ps[i]),
                                                    "ks_stat": ks.statistic,
                                                    "ks_p_conditional": ks.pvalue,
                                                    "n_bundles": Pb.shape[1],
                                                    "n_saved_groups": n,
                                                }
                                            )
                                            for b in range(N_BUNDLES):
                                                rows_b.append(
                                                    {
                                                        **base,
                                                        "law": law,
                                                        "J": J,
                                                        "t": t,
                                                        "bundle": b,
                                                        "plugin": Pb[i, b],
                                                        "stuck_in_prefix": bool(stuck[b, pj]),
                                                    }
                                                )
                                    for e in c.EPS:
                                        for who, PP, stk in (
                                            ("bundles", Pb, stuck[:, pj]),
                                            ("saved", Ps, c.stuck_groups(never, J)),
                                        ):
                                            fc = c.first_crossing(PP, e)
                                            cr = fc >= 0
                                            ts = np.asarray(GRID)[fc[cr]]
                                            sf = st_full[e][fc[cr]]
                                            rows_x.append(
                                                {
                                                    **base,
                                                    "law": law,
                                                    "who": who,
                                                    "J": J,
                                                    "family": fam,
                                                    "m": mm_,
                                                    "margin": margin,
                                                    "R": R,
                                                    "eps": e,
                                                    "n": PP.shape[1],
                                                    "n_crossed": int(cr.sum()),
                                                    "frac_crossed": float(cr.mean()),
                                                    "t_first_median": float(np.median(ts))
                                                    if ts.size
                                                    else np.nan,
                                                    "stop_mmd_ABOVE": int((sf == 1).sum()),
                                                    "stop_mmd_UNRESOLVED": int((sf == 0).sum()),
                                                    "stop_mmd_BELOW": int((sf == -1).sum()),
                                                    "n_with_stuck": int(stk.sum()),
                                                    "n_crossed_with_stuck": int((cr & stk).sum()),
                                                    "horizon": GRID[-1],
                                                }
                                            )
                print(f"[analyze] {m}__{s} {law}", flush=True)
    OUT_R.mkdir(parents=True, exist_ok=True)
    c.write_df(rows_bund, OUT_R / "bundles.csv")
    c.write_df(rows_law, OUT_R / "law_identity_check.csv")
    c.write_df(rows_b, OUT_R / "plugin_bundles_primary.csv.gz")
    c.write_df(rows_q, OUT_R / "plugin_bundle_vs_saved_quantiles.csv.gz")
    c.write_df(rows_x, OUT_R / "crossings_bundles_vs_saved.csv.gz")
    c.write_df(rows_cmp, OUT_R / "compare_saved_primary.csv")
    print(f"[analyze] done in {time.time() - t_all:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    for name in ("bench", "run"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=2)
        if name == "run":
            p.add_argument("--model", required=True, choices=MODELS)
            p.add_argument("--sampler", required=True, choices=SAMPLERS)
    p = sub.add_parser("analyze")
    p.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"freeze": cmd_freeze, "bench": cmd_bench, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
