"""Learn diagnostic spans and estimate discrepancies for logistic posteriors.

NUTS draws provide the reference posterior. Sampling and fitting run in separate
GPU and CPU stages; stage=reeval recomputes evaluations from saved samples.
"""

from __future__ import annotations

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
from omegaconf import DictConfig, OmegaConf

from mcmc.paperexp import german_credit as gc
from mcmc.paperexp import german_credit_r2 as r2
from mcmc.paperexp.german_credit_reference import load_kernel_file, load_set, set_path
from mcmc.paperexp.rng import np_gen, seed_for, torch_gen
from mcmc.paperexp.runio import RunDir, output_root
from mcmc.paperexp.samplers import make_sampler

log = logging.getLogger(__name__)
STUDY = "german_credit_r2"
REPO = pathlib.Path(__file__).resolve().parents[3]
F64 = torch.float64


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _json(path: pathlib.Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=_default))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _savez(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def grid_from_T(T: int) -> list[int]:
    if T < 1 or T & (T - 1):
        raise ValueError(f"T must be a power of two, got {T}")
    return [0] + [2**i for i in range(int(math.log2(T)) + 1)]


def t_star_from_T(T: int) -> int:
    return 2 ** int(round(math.log2(T) / 2))


class Context:
    """Posterior, kernel, reference draws and starting laws for one run."""

    def __init__(self, cfg: DictConfig, run: RunDir):
        self.cfg = cfg
        self.run = run
        self.path = run.path
        self.device = str(cfg.device)
        self.model = str(cfg.model)
        self.kind = str(cfg.sampler)
        self.cid = f"gc-{self.model}__{self.kind}"
        self.off = int(cfg.rep_offset[str(cfg.phase)])
        self.mem = int(cfg.mem_bytes)
        refs = cfg.references[self.model]
        self.refs = refs
        self.post = gc.make_posterior(self.model)
        entry = load_kernel_file(REPO / str(cfg.kernel_file), str(refs.kernel_key))
        self.kernel_entry = entry
        self.kernel = r2.make_kernel(entry)
        self.scales = np.asarray(entry["scales"], dtype=np.float64)
        calib = load_set(set_path("reference", str(refs.calibration)))["draws"][0]
        rep = cfg.representation
        self.feats = r2.make_features(
            self.kernel, calib, int(rep.n_landmarks), float(rep.nystrom_rel_cutoff), self.device
        )
        pool = load_set(set_path("reference", str(refs.pool)))["draws"]
        self.pool_runs = pool  # (R, N, d)
        ens = cfg.ensembles
        self.laws = r2.StartLaws(
            pool.reshape(-1, pool.shape[2]),
            self.scales,
            float(ens.overdispersion),
            float(ens.box_radius),
        )
        self.target = r2.PosteriorTarget(self.post, pool.reshape(-1, pool.shape[2]))
        self.eval_sets = {
            name: load_set(set_path("reference", str(rid)))["draws"]
            for name, rid in refs.evaluation.items()
        }
        self.ref: r2.RefSummary | None = None
        if (self.path / "setup.npz").exists():
            self.adopt_saved_basis(np.load(self.path / "setup.npz", allow_pickle=False))
        self.h: float | None = None
        self.T: int | None = None
        self.t_star: int | None = None

    def load_frozen(self) -> None:
        key = f"{self.model}__{self.kind}"
        if str(self.cfg.phase) == "main":
            fr = self.cfg.frozen.get(key)
            if fr is None:
                raise RuntimeError(f"main phase needs frozen values for {key}")
            self.h, self.T, self.t_star = float(fr.h), int(fr.T), int(fr.t_star)
            if self.t_star != t_star_from_T(self.T):
                raise RuntimeError("frozen t* does not follow the prespecified rule")
        else:
            tune = json.loads((self.path / "tune.json").read_text())
            tmax = json.loads((self.path / "tmax.json").read_text())
            self.h, self.T, self.t_star = float(tune["h"]), int(tmax["T"]), int(tmax["t_star"])
        self.horizon = max(self.T, int(self.cfg.horizon_min))
        self.grid = grid_from_T(self.horizon)

    def sampler(self):
        return make_sampler(self.kind, self.target, self.h, self.scales)

    def gen(self, stage: str, i: int = 0) -> torch.Generator:
        return torch_gen(seed_for(self.cid, stage, self.off + i), self.device)

    def adopt_saved_basis(self, z) -> None:
        """Reuse the saved Nystrom basis across devices.

        Recomputing can flip eigenvector signs or change the rank cutoff. Check the
        landmarks, projector (when ranks agree), and B^T K_ZZ B before adopting B.
        """
        if not np.array_equal(z["Z"], self.feats.Z.cpu().numpy()):
            raise RuntimeError("landmarks differ from the saved setup")
        B = z["B"]
        Bt = torch.as_tensor(B, dtype=F64, device=self.device)
        if B.shape == tuple(self.feats.B.shape):
            Pr = self.feats.B.cpu().numpy() @ self.feats.B.cpu().numpy().T
            Ps = B @ B.T
            dev = np.abs(Pr - Ps).max() / np.abs(Ps).max()
            if dev > 1e-8:
                raise RuntimeError(f"recomputed Nystrom projector differs from saved by {dev:.2e}")
        self.feats.B = Bt
        self.feats.rank = B.shape[1]
        self.feats.Z = self.feats.Z.to(self.device)
        err = self.feats.whitening_error()
        if err > 1e-8:
            raise RuntimeError(f"saved Nystrom basis whitening error {err:.2e}")

    def ensure_ref(self) -> None:
        if self.ref is not None:
            return
        f = self.path / "setup.npz"
        q = list(self.cfg.ensembles.quantiles)
        if f.exists():
            z = np.load(f, allow_pickle=False)
            self.adopt_saved_basis(z)
            Xs, labels = [], []
            for name, draws in self.eval_sets.items():
                for r in range(draws.shape[0]):
                    Xs.append(torch.as_tensor(draws[r], dtype=F64, device=self.device))
                    labels.append(f"{name}:{r}")
            if labels != json.loads(str(z["run_labels"])):
                raise RuntimeError("evaluation runs differ from the saved setup")
            cq = {k: z[f"coord_quantiles__{k}"] for k in self.eval_sets}
            cs = {k: z[f"coord_sd__{k}"] for k in self.eval_sets}
            self.ref = r2.RefSummary(
                labels,
                Xs,
                z["feat_run_means"],
                z["run_inner"],
                z["coord_run_means"],
                cq,
                cs,
                np.asarray(q),
            )
            return
        t0 = time.time()
        self.ref = r2.build_ref_summary(
            self.kernel, self.feats, self.eval_sets, self.device, self.mem, q
        )
        _sync(self.device)
        extra = {f"coord_quantiles__{k}": v for k, v in self.ref.coord_quantiles.items()}
        extra.update({f"coord_sd__{k}": v for k, v in self.ref.coord_sd.items()})
        _savez(
            f,
            B=self.feats.B.cpu().numpy(),
            Z=self.feats.Z.cpu().numpy(),
            eigvals=self.feats.eigvals.cpu().numpy(),
            feat_run_means=self.ref.feat_run_means,
            run_inner=self.ref.run_inner,
            coord_run_means=self.ref.coord_run_means,
            run_labels=np.array(json.dumps(self.ref.run_labels)),
            **extra,
        )
        _json(
            self.path / "setup.json",
            {
                "features": self.feats.to_dict(),
                "whitening_error": self.feats.whitening_error(),
                "kernel": self.kernel_entry,
                "start_laws": self.laws.to_dict(),
                "pool": str(self.refs.pool),
                "evaluation": OmegaConf.to_container(self.refs.evaluation),
                "choices": OmegaConf.to_container(self.refs.choices),
                "wall_s": time.time() - t0,
            },
        )

    def choice_runs(self) -> dict[str, np.ndarray]:
        out = {}
        labels = self.ref.run_labels
        for name, (setname, a, b) in self.refs.choices.items():
            idx = [labels.index(f"{setname}:{r}") for r in range(int(a), int(b) + 1)]
            out[str(name)] = np.asarray(idx)
        return out


def part_tune(ctx: Context) -> None:
    from mcmc.experiments.exp_suite_pilot_freeze import stage_tune

    t0 = time.time()
    res = stage_tune(ctx.target, ctx.kind, ctx.scales, ctx.cfg, ctx.device)
    _sync(ctx.device)
    res["wall_s"] = time.time() - t0
    res["start"] = "pool states (approximately stationary), not exact posterior draws"
    _json(ctx.path / "tune.json", res)
    log.info("tune %s: h=%.4g accept=%.4f", ctx.cid, res["h"], res["accept_verify"])


def part_tmax(ctx: Context) -> None:
    ctx.ensure_ref()
    tune = json.loads((ctx.path / "tune.json").read_text())
    cap = int(ctx.cfg.tmax.cap)
    J, G = int(ctx.cfg.tmax.J), int(ctx.cfg.ensembles.G)
    grid = grid_from_T(cap)
    smp = make_sampler(ctx.kind, ctx.target, float(tune["h"]), ctx.scales)
    gen = ctx.gen("r2-tmax")
    x0 = ctx.laws.sample("I1", J, gen)
    runs = ctx.choice_runs()[next(iter(ctx.refs.choices))]
    curve = {}

    def cb(t, x):
        st = r2.checkpoint_stats(x, ctx.kernel, ctx.feats, ctx.ref, G, ctx.mem, [0.5])
        mmd2, se = r2.mmd2_with_se(
            st["group_pair_ksum"], st["group_run_cross"], ctx.ref.run_inner, J // G, runs
        )
        curve[t] = {"mmd2_u": mmd2, "mmd2_se": se, "mmd": math.sqrt(max(mmd2, 0.0))}

    t0 = time.time()
    out = smp.run(x0, grid, gen, store=False, on_checkpoint=cb)
    _sync(ctx.device)
    thr = float(ctx.cfg.tmax.threshold)
    below = [t for t in grid if t > 0 and curve[t]["mmd2_u"] < thr**2]
    T = below[0] if below else cap
    res = {
        "T": T,
        "t_star": t_star_from_T(T),
        "cap_binds": not below,
        "threshold": thr,
        "reference_choice": next(iter(ctx.refs.choices)),
        "J": J,
        "curve": curve,
        "accept_rate": out["accept_rate"],
        "counters": out["counters"],
        "wall_s": time.time() - t0,
    }
    _json(ctx.path / "tmax.json", res)
    log.info("tmax %s: T=%d t*=%d cap_binds=%s", ctx.cid, T, res["t_star"], res["cap_binds"])


def part_setup(ctx: Context) -> None:
    ctx.ensure_ref()


def part_sigma(ctx: Context) -> None:
    ctx.ensure_ref()
    N = int(ctx.cfg.sigma.N)
    R = ctx.pool_runs.shape[0]
    half = R // 2
    rng = np_gen(seed_for(ctx.cid, "r2-sigma-starts", ctx.off))
    first = ctx.pool_runs[:half].reshape(-1, ctx.post.d)
    second = ctx.pool_runs[half:].reshape(-1, ctx.post.d)
    X = first[rng.integers(0, first.shape[0], N)]
    Xp = second[rng.integers(0, second.shape[0], N)]
    x0 = torch.as_tensor(np.concatenate([X, X, Xp, Xp]), dtype=F64, device=ctx.device)
    ops = {}

    def cb(t, x):
        ops[t] = r2.branched_operator_stats(ctx.feats(x), N)

    smp = ctx.sampler()
    t0 = time.time()
    out = smp.run(x0, ctx.grid, ctx.gen("r2-sigma"), store=False, on_checkpoint=cb)
    _sync(ctx.device)
    _savez(
        ctx.path / "sigma.npz",
        grid=np.asarray(ctx.grid),
        O=np.stack([ops[t]["O"] for t in ctx.grid]),
        OA=np.stack([ops[t]["OA"] for t in ctx.grid]),
        OB=np.stack([ops[t]["OB"] for t in ctx.grid]),
    )
    _json(
        ctx.path / "sigma.json",
        {
            "N": N,
            "accept_rate": out["accept_rate"],
            "counters": out["counters"],
            "wall_s": time.time() - t0,
            "starts": "X from pool runs 0..R/2-1, X' from R/2..R-1",
        },
    )


def part_ensemble(ctx: Context, law: str) -> None:
    ctx.ensure_ref()
    e = ctx.cfg.ensembles
    J, G, K = int(e.J), int(e.G), int(e.K_conv)
    gen = ctx.gen(f"r2-ens-{law}")
    x0 = ctx.laws.sample(law, J, gen)
    T = ctx.horizon
    traj = torch.empty((T + 1, K, ctx.post.d), dtype=F64, device=ctx.device)
    gridset = set(ctx.grid)
    stats = {}
    states = torch.empty((len(ctx.grid), J, ctx.post.d), dtype=F64, device="cpu")
    t_stats = [0.0]

    def cb(t, x):
        traj[t] = x[:K]
        if t in gridset:
            states[ctx.grid.index(t)] = x.cpu()  # replica states for R1 (kept in memory)
            ts = time.time()
            stats[t] = r2.checkpoint_stats(
                x, ctx.kernel, ctx.feats, ctx.ref, G, ctx.mem, list(e.quantiles)
            )
            t_stats[0] += time.time() - ts

    smp = ctx.sampler()
    t0 = time.time()
    out = smp.run(x0, range(T + 1), gen, store=False, on_checkpoint=cb)
    _sync(ctx.device)
    wall = time.time() - t0
    _savez(ctx.path / f"ens_{law}_conv_traj.npz", traj=traj.cpu().numpy())
    _savez(ctx.path / f"ens_{law}_states.npz", grid=np.asarray(ctx.grid), states=states.numpy())
    keys = (
        "group_feat",
        "group_pair_ksum",
        "group_run_cross",
        "group_coord_mean",
        "coord_quantiles",
    )
    _savez(
        ctx.path / f"ens_{law}.npz",
        grid=np.asarray(ctx.grid),
        **{k: np.stack([stats[t][k] for t in ctx.grid]) for k in keys},
        accept_per_chain=out["accept_count_per_chain"].cpu().numpy(),
    )
    acc_by = {str(t): out["accept_by_interval"][t] for t in ctx.grid if t > 0}
    _json(
        ctx.path / f"ens_{law}.json",
        {
            "law": law,
            "J": J,
            "G": G,
            "K_conv": K,
            "accept_rate": out["accept_rate"],
            "accept_by_grid_interval_end": acc_by,
            "counters": out["counters"],
            "wall_s": wall,
            "stats_s": t_stats[0],
        },
    )


def part_conventional(ctx: Context, law: str) -> None:
    """CPU stage: rank R-hat, bulk/tail ESS on the ensemble's first K_conv chains (saved paths)."""
    e = ctx.cfg.ensembles
    legacy = ctx.path / f"ens_{law}.npz"
    if not (ctx.path / f"ens_{law}_conv_traj.npz").exists():
        z = np.load(legacy)
        if "conv_rhat_max" not in z:
            raise RuntimeError(f"no recorded chains and no conventional diagnostics for {law}")
        # Older saved runs store these diagnostics with the ensemble statistics.
        _savez(
            ctx.path / f"conv_{law}.npz",
            conv_t=z["conv_t"],
            conv_rhat_max=z["conv_rhat_max"],
            conv_ess_bulk_min=z["conv_ess_bulk_min"],
            conv_ess_tail_min=z["conv_ess_tail_min"],
            window=np.asarray([t - t // 2 for t in z["conv_t"]]),
        )
        _json(
            ctx.path / f"conv_{law}.json", {"law": law, "source": "computed in GPU stage (pilot)"}
        )
        return
    traj = np.load(ctx.path / f"ens_{law}_conv_traj.npz")["traj"]
    t0 = time.time()
    conv = r2.conventional_on_traj(traj, ctx.grid, int(e.conv_group))
    tc = sorted(conv)
    _savez(
        ctx.path / f"conv_{law}.npz",
        conv_t=np.asarray(tc),
        conv_rhat_max=np.stack([conv[t]["rhat_max"] for t in tc]),
        conv_ess_bulk_min=np.stack([conv[t]["ess_bulk_min"] for t in tc]),
        conv_ess_tail_min=np.stack([conv[t]["ess_tail_min"] for t in tc]),
        window=np.asarray([conv[t]["window"] for t in tc]),
    )
    _json(
        ctx.path / f"conv_{law}.json",
        {
            "law": law,
            "K_conv": int(traj.shape[1]),
            "group_size": int(e.conv_group),
            "wall_s": time.time() - t0,
        },
    )


def part_pilot_run(ctx: Context, ptype: str) -> None:
    """GPU stage: run n_fit x n_chains pilot chains and save their full paths."""
    p = ctx.cfg.pilots
    n_fit, C = int(p.n_fit), int(p.n_chains)
    Lmax = max(int(v) for v in p.lengths)
    law = {"stationary": "I0", "confined": "I3"}[ptype]
    x0 = ctx.laws.sample(law, n_fit * C, ctx.gen(f"r2-pilot-{ptype}-starts"))
    traj = torch.empty((Lmax + 1, n_fit * C, ctx.post.d), dtype=F64, device=ctx.device)

    def cb(t, x):
        traj[t] = x

    smp = ctx.sampler()
    t0 = time.time()
    out = smp.run(x0, range(Lmax + 1), ctx.gen(f"r2-pilot-{ptype}"), store=False, on_checkpoint=cb)
    _sync(ctx.device)
    t_run = time.time() - t0
    X = traj.transpose(0, 1).contiguous().cpu().numpy()  # (n_fit * C, Lmax + 1, d)
    _savez(ctx.path / f"pilotrun_{ptype}.npz", X=X)
    _json(
        ctx.path / f"pilotrun_{ptype}.json",
        {
            "ptype": ptype,
            "start_law": law,
            "n_fit": n_fit,
            "n_chains": C,
            "Lmax": Lmax,
            "accept_rate": out["accept_rate"],
            "counters": out["counters"],
            "wall_run_s": t_run,
        },
    )


def part_pilots(ctx: Context, ptype: str) -> None:
    """CPU stage: learn every span family from each pilot (nested prefixes)."""
    p = ctx.cfg.pilots
    n_fit, C = int(p.n_fit), int(p.n_chains)
    lengths = sorted(int(v) for v in p.lengths)
    Lmax = lengths[-1]
    m_max = max(int(m) for m in ctx.cfg.m_grid)
    law = {"stationary": "I0", "confined": "I3"}[ptype]
    runinfo = json.loads((ctx.path / f"pilotrun_{ptype}.json").read_text())
    out = {"accept_rate": runinfo["accept_rate"], "counters": runinfo["counters"]}
    t_run = runinfo["wall_run_s"]
    Xall = np.load(ctx.path / f"pilotrun_{ptype}.npz")["X"]
    if Xall.shape != (n_fit * C, Lmax + 1, ctx.post.d):
        raise RuntimeError(f"pilot paths have shape {Xall.shape}")
    r = ctx.feats.rank
    spans = np.full((n_fit, len(lengths), len(r2.FAMILIES), r, m_max), np.nan)
    infos = []
    t1 = time.time()
    for i in range(n_fit):
        Xn = np.ascontiguousarray(Xall[i * C : (i + 1) * C, 1:])  # (C, Lmax, d)
        Xi = torch.as_tensor(Xn, dtype=F64, device=ctx.device)
        Fi = ctx.feats(Xi.reshape(-1, ctx.post.d)).reshape(C, Lmax, r).cpu().numpy()
        rand = np_gen(seed_for(ctx.cid, "r2-random", ctx.off + i)).standard_normal((r, m_max))
        for li, L in enumerate(lengths):
            sp, info = r2.learn_spans(
                Fi[:, :L],
                Xn[:, :L],
                ctx.t_star,
                m_max,
                float(p.tica_rel_cutoff),
                float(p.coord_rel_cutoff),
                np_gen(0),
            )
            sp["random"] = rand
            for fi, fam in enumerate(r2.FAMILIES):
                if sp[fam] is not None:
                    spans[i, li, fi] = sp[fam]
            infos.append({"fit": i, "L": L, **info})
    t_learn = time.time() - t1
    per_step = out["counters"]
    _savez(
        ctx.path / f"pilots_{ptype}.npz",
        spans=spans,
        lengths=np.asarray(lengths),
        families=np.asarray(r2.FAMILIES),
    )
    _json(
        ctx.path / f"pilots_{ptype}.json",
        {
            "ptype": ptype,
            "start_law": law,
            "n_fit": n_fit,
            "n_chains": C,
            "lengths": lengths,
            "t_star": ctx.t_star,
            "accept_rate": out["accept_rate"],
            "counters": per_step,
            "pilot_cost_per_fit_target_evals": {str(L): C * L for L in lengths},
            "pilot_cost_per_fit_grad_evals": {
                str(L): (C * L if ctx.kind == "mala" else 0) for L in lengths
            },
            "wall_run_s": t_run,
            "wall_learn_s": t_learn,
            "info": infos,
        },
    )


def evaluate_laws(ctx: Context, files: dict, grid: list, Qs: list, extra_spans, nS: int, J: int):
    """Debiased projections/residuals for every (law, t, reference choice, span).

    ``files`` maps law -> npz of checkpoint statistics on ``grid``; ``extra_spans(t)`` returns the
    time-dependent spans appended after ``Qs`` (None entries give NaN rows).
    """
    G = int(ctx.cfg.ensembles.G)
    choices = ctx.choice_runs()
    shape = (len(files), len(grid), len(choices), nS)
    keys = (
        "proj2",
        "proj2_se",
        "res_full2",
        "res_full2_se",
        "res_feat2",
        "captured_feat",
        "captured_full",
    )
    A = {k: np.full(shape, np.nan) for k in keys}
    L3 = {k: np.full(shape[:3], np.nan) for k in ("mmd2", "mmd2_se", "feat2", "feat2_se")}
    # ordinary coordinates (a separate comparison from the kernel-approximated coordinate spans):
    # mean difference in reference posterior-sd units and its z (group means vs run means)
    coord = {k: np.full(shape[:3], np.nan) for k in ("max_mean_sd", "max_quantile_sd")}
    d = ctx.post.d
    coord.update({k: np.full(shape[:3] + (d,), np.nan) for k in ("coord_mean_sd", "coord_mean_z")})
    stat_keys = (
        "group_feat",
        "group_pair_ksum",
        "group_run_cross",
        "group_coord_mean",
        "coord_quantiles",
    )
    for a, (lw, f) in enumerate(files.items()):
        z = np.load(f)
        if list(z["grid"]) != list(grid):
            raise RuntimeError(f"{f}: grid differs")
        for b, t in enumerate(grid):
            st = {k: z[k][b] for k in stat_keys}
            st["n_per_group"] = J // G
            Qt = Qs + list(extra_spans(t))
            if len(Qt) != nS:
                raise RuntimeError(f"span count {len(Qt)} != table {nS}")
            for c, (cname, runs) in enumerate(choices.items()):
                terms = r2.law_reference_terms(st, ctx.ref, runs)
                ev = r2.evaluate_spans(terms, Qt)
                for k in A:
                    A[k][a, b, c] = ev[k]
                L3["mmd2"][a, b, c], L3["mmd2_se"][a, b, c] = terms["mmd2"], terms["mmd2_se"]
                L3["feat2"][a, b, c] = terms["feat_raw"] - terms["feat_bias"]
                L3["feat2_se"][a, b, c] = math.sqrt(terms["feat_var"])
                setname = ctx.refs.choices[cname][0]
                sd = ctx.ref.coord_sd[setname]
                gm, rm = st["group_coord_mean"], ctx.ref.coord_run_means[runs]
                diff = gm.mean(0) - rm.mean(0)
                se = np.sqrt(gm.var(0, ddof=1) / gm.shape[0] + rm.var(0, ddof=1) / rm.shape[0])
                coord["coord_mean_sd"][a, b, c] = diff / sd
                with np.errstate(divide="ignore", invalid="ignore"):  # all chains identical
                    coord["coord_mean_z"][a, b, c] = diff / se
                coord["max_mean_sd"][a, b, c] = np.max(np.abs(diff) / sd)
                coord["max_quantile_sd"][a, b, c] = np.max(
                    np.abs(st["coord_quantiles"] - ctx.ref.coord_quantiles[setname]) / sd
                )
    return A, L3, coord


def load_operators(ctx: Context) -> tuple[list, dict]:
    sig = np.load(ctx.path / "sigma.npz")
    grid = list(sig["grid"])
    if grid != ctx.grid:
        raise RuntimeError("sigma grid differs from the frozen grid")
    ops = {
        t: r2.OperatorSummary.from_stats(sig["O"][i], sig["OA"][i], sig["OB"][i])
        for i, t in enumerate(grid)
    }
    return grid, ops


def span_table(ctx: Context, ops: dict) -> tuple[list, list, int]:
    """Rows + orthonormal bases: learned spans, static oracle V*_m(0), time-t* oracle V*_m(t*),
    then (rows only) the time-matched oracle V*_m(t) whose bases are supplied per t."""
    m_grid = [int(m) for m in ctx.cfg.m_grid]
    ptypes = [pt for pt in ("stationary", "confined") if (ctx.path / f"pilots_{pt}.npz").exists()]
    table, Qs = [], []
    for pt in ptypes:
        z = np.load(ctx.path / f"pilots_{pt}.npz")
        S, lengths = z["spans"], list(z["lengths"])
        for i in range(S.shape[0]):
            for li, L in enumerate(lengths):
                for fi, fam in enumerate(r2.FAMILIES):
                    W = S[i, li, fi]
                    W = None if np.isnan(W).any() else W
                    for m in m_grid:
                        table.append(
                            {
                                "kind": "learned",
                                "family": fam,
                                "ptype": pt,
                                "fit": i,
                                "L": int(L),
                                "m": m,
                            }
                        )
                        Qs.append(r2.orthonormal_or_none(W, m))
    n_learned = len(Qs)
    ref_row = {"ptype": "reference", "fit": -1, "L": -1}
    for m in m_grid:
        table.append({"kind": "static_oracle", "family": "static_oracle", **ref_row, "m": m})
        Qs.append(ops[0].top_span(m))
        table.append({"kind": "oracle_tstar", "family": "oracle_tstar", **ref_row, "m": m})
        Qs.append(ops[ctx.t_star].top_span(m))
    for m in m_grid:
        table.append({"kind": "oracle_t", "family": "oracle_t", **ref_row, "m": m})
    return table, Qs, n_learned


def worst_case_for(op, Qs: list, ms: list, idx, fac: float) -> dict:
    """Batched worst-case rows for the spans ``idx`` (grouped by dimension)."""
    idx = list(idx)
    out = {}
    for m in sorted({ms[k] for k in idx}):
        ks = [k for k in idx if ms[k] == m]
        out.update(zip(ks, r2.worst_case_batch(op, [Qs[k] for k in ks], m, fac)))
    return out


def part_evaluate(ctx: Context, out: pathlib.Path | None = None) -> None:
    """Reads inputs from ctx.path; writes eval outputs to ``out`` (default ctx.path)."""
    out = ctx.path if out is None else out
    ctx.ensure_ref()
    m_grid = [int(m) for m in ctx.cfg.m_grid]
    fac = float(ctx.cfg.resolution_factor)
    grid, ops = load_operators(ctx)
    ts = ctx.t_star
    table, Qs, n_learned = span_table(ctx, ops)

    # worst case under Sigma_t-hat
    t0 = time.time()
    ms = [row["m"] for row in table[: len(Qs)]]
    wc = worst_case_for(ops[ts], Qs, ms, range(len(Qs)), fac)
    wc_tstar = [wc[k] for k in range(len(Qs))]
    wc_t = {}
    L_max = max((r["L"] for r in table[:n_learned]), default=-1)
    sel = [
        k
        for k, row in enumerate(table[: len(Qs)])
        if row["kind"] != "learned" or (row["ptype"] == "stationary" and row["L"] == L_max)
    ]
    for t in grid:
        if t == 0:
            continue  # Sigma_0: the static oracle is optimal by definition; keep t >= 1
        wc_t[t] = worst_case_for(ops[t], Qs, ms, sel, fac)
    t_wc = time.time() - t0

    # actual laws
    laws = [lw for lw in ("I0", "I1", "I2", "I3") if (ctx.path / f"ens_{lw}.npz").exists()]
    choices = ctx.choice_runs()
    nS = len(table)
    conv_summary = {}
    t0 = time.time()
    files = {lw: ctx.path / f"ens_{lw}.npz" for lw in laws}
    A, L3, coord = evaluate_laws(
        ctx,
        files,
        grid,
        Qs,
        lambda t: [ops[t].top_span(m) for m in m_grid],
        nS,
        int(ctx.cfg.ensembles.J),
    )
    for lw in laws:
        z = np.load(ctx.path / f"conv_{lw}.npz")
        conv_summary[lw] = {
            "t": z["conv_t"].tolist(),
            "frac_rhat_lt_1.01": (z["conv_rhat_max"] < 1.01).mean(1).tolist(),
            "frac_ess_ge_400": ((z["conv_ess_bulk_min"] >= 400) & (z["conv_ess_tail_min"] >= 400))
            .mean(1)
            .tolist(),
            "frac_pass_both": (
                (z["conv_rhat_max"] < 1.01)
                & (z["conv_ess_bulk_min"] >= 400)
                & (z["conv_ess_tail_min"] >= 400)
            )
            .mean(1)
            .tolist(),
            "median_rhat_max": np.median(z["conv_rhat_max"], 1).tolist(),
        }
    t_law = time.time() - t0

    _savez(
        out / "eval.npz",
        grid=np.asarray(grid),
        laws=np.asarray(laws),
        choices=np.asarray(list(choices)),
        **A,
        **L3,
        **coord,
        op_eigvals=np.stack([ops[t].eigvals[:16] for t in grid]),
        op_eps=np.asarray([ops[t].eps for t in grid]),
    )
    import pandas as pd

    pd.DataFrame(table).to_csv(out / "spans_index.csv", index_label="span")
    wc_rows = [{"span": k, **table[k], **w} for k, w in enumerate(wc_tstar)]
    pd.DataFrame(wc_rows).to_csv(out / "worst_case_tstar.csv", index=False)
    rows = [{"t": t, "span": k, **table[k], **w} for t in wc_t for k, w in wc_t[t].items()]
    pd.DataFrame(rows).to_csv(out / "worst_case_all_t.csv", index=False)
    _json(
        out / "eval.json",
        {
            "t_star": ts,
            "grid": grid,
            "laws": laws,
            "choices": list(choices),
            "n_spans": nS,
            "n_learned": n_learned,
            "conventional": conv_summary,
            "operator": {
                str(t): {
                    "eig_top10": ops[t].eigvals[:10].tolist(),
                    "eps": ops[t].eps,
                    "most_negative": ops[t].most_negative,
                }
                for t in grid
            },
            "wall_worst_case_s": t_wc,
            "wall_laws_s": t_law,
        },
    )


def part_nuts(ctx: Context, law: str) -> None:
    """Compare independently adapted NUTS chains, including warmup checkpoints.

    Each chain has its own adaptation, so these chain laws do not share a single
    transition operator. Record mean cumulative gradient evaluations as the cost.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    ctx.ensure_ref()
    n = ctx.cfg.nuts
    st_cfg = n.settings[ctx.model]
    J, W, S = int(n.J[ctx.model]), int(n.n_warmup), int(n.n_sample)
    G = int(ctx.cfg.ensembles.G)
    centered = str(st_cfg.parameterization) == "centered"
    x0 = ctx.laws.sample(law, J, ctx.gen(f"r2-nuts-{law}-starts")).cpu().numpy()
    sites = [gc.numpyro_unconstrained_from_declared(ctx.model, xi, centered) for xi in x0]
    init = {k: jnp.asarray(np.stack([si[k] for si in sites])) for k in sites[0]}
    kernel = NUTS(
        gc.numpyro_model(ctx.model, centered),
        target_accept_prob=float(st_cfg.target_accept_prob),
        max_tree_depth=int(st_cfg.max_tree_depth),
        dense_mass=bool(st_cfg.dense_mass),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=W,
        num_samples=S,
        num_chains=J,
        chain_method="vectorized",
        progress_bar=False,
    )
    data = ctx.post.data
    X, y = jnp.asarray(data.X), jnp.asarray(data.y)
    key = jax.random.PRNGKey(seed_for(ctx.cid, f"r2-nuts-{law}", ctx.off))
    t0 = time.time()
    mcmc.warmup(
        key, X, y, init_params=init, extra_fields=("num_steps", "diverging"), collect_warmup=True
    )
    ws = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    wf = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    mcmc.run(mcmc.post_warmup_state.rng_key, X, y, extra_fields=("num_steps", "diverging"))
    ss = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    sf = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    wall = time.time() - t0
    traj = np.concatenate(
        [
            x0[:, None, :],
            gc.declared_from_numpyro(ctx.model, ws, centered),
            gc.declared_from_numpyro(ctx.model, ss, centered),
        ],
        axis=1,
    )
    assert traj.shape == (J, W + S + 1, ctx.post.d), traj.shape
    steps = np.concatenate([wf["num_steps"], sf["num_steps"]], axis=1)  # (J, W + S)
    div = np.concatenate([wf["diverging"], sf["diverging"]], axis=1)
    T = W + S
    grid = grid_from_T(T)
    cum_grad = np.concatenate([[0.0], np.cumsum(steps.mean(0))])
    q = list(ctx.cfg.ensembles.quantiles)
    stats = {}
    for t in grid:
        xt = torch.as_tensor(traj[:, t], dtype=F64, device=ctx.device)
        stats[t] = r2.checkpoint_stats(xt, ctx.kernel, ctx.feats, ctx.ref, G, ctx.mem, q)
    keys = (
        "group_feat",
        "group_pair_ksum",
        "group_run_cross",
        "group_coord_mean",
        "coord_quantiles",
    )
    _savez(
        ctx.path / f"nuts_{law}.npz",
        grid=np.asarray(grid),
        **{k: np.stack([stats[t][k] for t in grid]) for k in keys},
        cum_grad_mean=cum_grad[grid],
        divergent_cum=div.cumsum(1).sum(0)[np.asarray(grid[1:]) - 1],
    )
    _json(
        ctx.path / f"nuts_{law}.json",
        {
            "law": law,
            "J": J,
            "n_warmup": W,
            "n_sample": S,
            "settings": OmegaConf.to_container(st_cfg),
            "grid": grid,
            "cum_grad_mean": cum_grad[grid].tolist(),
            "divergent_total": int(div.sum()),
            "divergent_warmup": int(wf["diverging"].sum()),
            "max_num_steps": int(steps.max()),
            "wall_s": wall,
            "note": "separately adapted chains; practical comparison only (not one operator)",
        },
    )


def part_evaluate_nuts(ctx: Context, out: pathlib.Path | None = None) -> None:
    """Reads inputs from ctx.path; writes eval_nuts outputs to ``out`` (default ctx.path)."""
    out = ctx.path if out is None else out
    ctx.ensure_ref()
    _, ops = load_operators(ctx)
    table, Qs, n_learned = span_table(ctx, ops)
    m_grid = [int(m) for m in ctx.cfg.m_grid]
    laws = [str(lw) for lw in ctx.cfg.nuts.laws]
    files = {lw: ctx.path / f"nuts_{lw}.npz" for lw in laws}
    grids = {lw: list(np.load(f)["grid"]) for lw, f in files.items()}
    grid = grids[laws[0]]
    if any(g != grid for g in grids.values()):
        raise RuntimeError("NUTS grids differ between laws")
    J = int(ctx.cfg.nuts.J[ctx.model])
    t0 = time.time()
    A, L3, coord = evaluate_laws(
        ctx, files, grid, Qs, lambda t: [None] * len(m_grid), len(table), J
    )
    _savez(
        out / "eval_nuts.npz",
        grid=np.asarray(grid),
        laws=np.asarray(laws),
        choices=np.asarray(list(ctx.choice_runs())),
        cum_grad_mean=np.stack([np.load(files[lw])["cum_grad_mean"] for lw in laws]),
        **A,
        **L3,
        **coord,
    )
    _json(
        out / "eval_nuts.json",
        {
            "grid": grid,
            "laws": laws,
            "n_spans": len(table),
            "n_learned": n_learned,
            "wall_s": time.time() - t0,
            "note": "oracle_t rows are NaN for NUTS (no operator)",
        },
    )


PART_FILES = {
    "tune": "tune.json",
    "tmax": "tmax.json",
    "setup": "setup.npz",
    "sigma": "sigma.npz",
    "evaluate": "eval.json",
    "evaluate_nuts": "eval_nuts.json",
}


def part_file(part: str) -> str:
    if part.startswith("ens_"):
        return f"{part}.json"
    if part.startswith(("pilots_", "pilotrun_", "nuts_", "conv_")):
        return f"{part}.json"
    return PART_FILES[part]


# These execution settings may differ between stages of the same run.
VOLATILE_KEYS = ("stage", "device", "mem_bytes", "parts_gpu", "parts_cpu", "reeval_tag")


def stage_parts(cfg: DictConfig, stage: str) -> list[str]:
    parts = [str(p) for p in cfg[f"parts_{stage}"]]
    if stage == "gpu" and str(cfg.phase) in ("smoke", "pilot"):
        parts = ["tune", "tmax"] + parts
    if str(cfg.sampler) != str(cfg.nuts.in_sampler):  # NUTS comparison runs once per model
        parts = [p for p in parts if not (p.startswith("nuts_") or p == "evaluate_nuts")]
    return parts


def config_identity(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=False)
    for k in VOLATILE_KEYS:
        c.pop(k, None)
    c.pop("hydra", None)
    return OmegaConf.to_yaml(OmegaConf.create(c))


def run_parts(cfg: DictConfig) -> pathlib.Path:
    """Run sampling/kernel statistics on GPU or span fitting and NUTS on CPU."""
    phase, stage = str(cfg.phase), str(cfg.stage)
    if stage not in ("gpu", "cpu"):
        raise ValueError(f"stage must be gpu or cpu, got {stage!r}")
    if stage == "cpu" and str(cfg.device) != "cpu":
        raise ValueError("stage=cpu requires device=cpu")
    all_parts = stage_parts(cfg, "gpu") + stage_parts(cfg, "cpu")
    parts = stage_parts(cfg, stage)
    run_id = f"{cfg.model}__{cfg.sampler}_{cfg.tag}"
    ident = config_identity(cfg)
    path = output_root() / STUDY / phase / run_id
    if (path / "RUN-STATUS.json").exists():
        stored = config_identity(OmegaConf.create((path / "config.yaml").read_text()))
        if stored != ident:
            raise RuntimeError(f"{path} exists with a different config; use a new tag")
        run = RunDir.resume(STUDY, phase, run_id)
        prev = run.status()["requested_units"]
        if prev != len(all_parts):  # the declared parts list changed (execution structure only)
            run.update(
                requested_units=len(all_parts),
                note=f"requested_units {prev} -> {len(all_parts)}: parts list changed",
            )
    else:
        run = RunDir.create(
            STUDY,
            phase,
            run_id,
            OmegaConf.to_yaml(cfg),
            requested_units=len(all_parts),
            allow_dirty=False,
        )
        (path / "config_identity.yaml").write_text(ident)
    from mcmc.paperexp.runio import git_state, utc_now

    _json(
        path / f"stage_{stage}_provenance_{utc_now().replace(':', '')}.json",
        {
            **git_state(),
            "stage": stage,
            "device": str(cfg.device),
            "parts": parts,
            "command": " ".join(os.sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_slot": os.environ.get("MCMC_GPU_SLOT"),
        },
    )
    try:
        ctx = Context(cfg, run)
        for p in parts:
            if (run.path / part_file(p)).exists():
                continue
            if p not in ("tune", "tmax", "setup") and ctx.T is None:
                ctx.load_frozen()
            t0 = time.time()
            log.info("%s [%s]: part %s", run_id, stage, p)
            if p == "tune":
                part_tune(ctx)
            elif p == "tmax":
                part_tmax(ctx)
            elif p == "setup":
                part_setup(ctx)
            elif p == "sigma":
                part_sigma(ctx)
            elif p.startswith("ens_"):
                part_ensemble(ctx, p[4:])
            elif p.startswith("conv_"):
                part_conventional(ctx, p[5:])
            elif p.startswith("pilotrun_"):
                part_pilot_run(ctx, p[9:])
            elif p.startswith("pilots_"):
                part_pilots(ctx, p[7:])
            elif p == "evaluate":
                part_evaluate(ctx)
            elif p.startswith("nuts_"):
                part_nuts(ctx, p[5:])
            elif p == "evaluate_nuts":
                part_evaluate_nuts(ctx)
            else:
                raise ValueError(f"unknown part {p!r}")
            done = sum((run.path / part_file(q)).exists() for q in all_parts)
            peak = (
                torch.cuda.max_memory_allocated() / 2**20
                if str(cfg.device).startswith("cuda")
                else float("nan")
            )
            run.update(
                completed_units=done,
                note=f"[{stage}] {p} {time.time() - t0:.0f}s; gpu peak {peak:.0f} MiB; "
                f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB",
            )
        done = sum((run.path / part_file(q)).exists() for q in all_parts)
        run.update(completed_units=done)
        if done == len(all_parts):
            run.finish("COMPLETED")
        else:
            run.finish(
                "INCOMPLETE", note=f"{stage} stage done; {len(all_parts) - done} part(s) pending"
            )
    except BaseException as exc:
        run.finish("FAILED", note=f"[{stage}] {exc!r}"[:500])
        raise
    return run.path


REEVAL_PARTS = ("evaluate", "evaluate_nuts")


def run_reevaluate(cfg: DictConfig) -> pathlib.Path:
    """Recompute evaluations from a completed run without resampling.

    Use its saved configuration, with device and memory settings supplied here.
    Write results to a new run named <model>__<sampler>_<tag>_<reeval_tag>.
    """
    import types

    if str(cfg.device) != "cpu":
        raise ValueError("stage=reeval requires device=cpu")
    phase, src_id = str(cfg.phase), f"{cfg.model}__{cfg.sampler}_{cfg.tag}"
    src = output_root() / STUDY / phase / src_id
    status = json.loads((src / "RUN-STATUS.json").read_text())["status"]
    if status != "COMPLETED":
        raise RuntimeError(f"{src} is {status}; stage=reeval needs a COMPLETED source run")
    text = (src / "config.yaml").read_text()
    scfg = OmegaConf.create(text)
    for k in ("device", "mem_bytes"):
        scfg[k] = cfg[k]
    parts = [p for p in REEVAL_PARTS if (src / part_file(p)).exists()]  # what the source evaluated
    run_id = f"{src_id}_{cfg.reeval_tag}"
    if (output_root() / STUDY / phase / run_id / "RUN-STATUS.json").exists():
        run = RunDir.resume(STUDY, phase, run_id)  # refuses a COMPLETED re-evaluation
    else:
        run = RunDir.create(
            STUDY,
            phase,
            run_id,
            text,
            requested_units=len(parts),
            allow_dirty=False,
            extra_provenance={"source_run": f"{STUDY}/{phase}/{src_id}", "parts": parts},
        )
    try:
        ctx = Context(scfg, types.SimpleNamespace(path=src))  # read-only source
        ctx.load_frozen()
        for p in parts:
            if (run.path / part_file(p)).exists():
                continue
            t0 = time.time()
            log.info("%s [reeval]: part %s", run_id, p)
            (part_evaluate if p == "evaluate" else part_evaluate_nuts)(ctx, out=run.path)
            done = sum((run.path / part_file(q)).exists() for q in parts)
            run.update(
                completed_units=done,
                note=f"[reeval] {p} {time.time() - t0:.0f}s; rss "
                f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB",
            )
        run.finish("COMPLETED", note=f"re-evaluation of {STUDY}/{phase}/{src_id}")
    except BaseException as exc:
        run.finish("FAILED", note=f"[reeval] {exc!r}"[:500])
        raise
    return run.path


@hydra.main(
    config_path="../../../configs/experiment", config_name="exp_r2_german_credit", version_base=None
)
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    print(run_reevaluate(cfg) if str(cfg.stage) == "reeval" else run_parts(cfg))


if __name__ == "__main__":
    main()
