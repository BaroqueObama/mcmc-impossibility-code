"""Stopping budgets and plug-in comparisons from saved logistic chains.

Run floor and freeze before analyze; summarize writes the tables.
regression exports the Gaussian regression comparison. Raw arrays live under
MCMC_PAPER_OUT/c_realdata_followup.

The confidence-radius floor is deterministic. NUTS-reference discrepancies
use jackknife errors; assumed-radius plug-ins are not confidence bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import time
import types

import numpy as np
from mcmc.paperexp.runio import resolve_output_path

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures" / "data" / "c_realdata_followup"
STUDY_R2 = "german_credit_r2"
MODELS = ("hier", "hier_pima", "hier_australian", "hier_ionosphere", "hier_sonar", "horseshoe")
PRIMARY_MODELS = MODELS[
    :5
]  # horseshoe: appendix sensitivity only (its reference failed a cross-check)
DISPLAY = {
    "hier": "German Credit",
    "hier_pima": "Pima",
    "hier_australian": "Australian Credit",
    "hier_ionosphere": "Ionosphere",
    "hier_sonar": "Sonar",
    "horseshoe": "German Credit horseshoe",
}
SAMPLERS = ("mala", "rwm")
LAWS = ("I0", "I1", "I2", "I3")
J_GRID = (4, 16, 64)
# Same failure budget as the regression comparisons.
BETA = {"replicas": 0.025, "target": 0.0125, "operator": 0.0125}
R_ASSUMED = (0.5, 1.0, 2.0, 4.0)
EPS = (0.10, 0.05)
KAPPA = 1.0  # diagonal-one scaled RBF: k(x, x) = 1, 0 < k <= 1, so MMD <= sqrt(2) for every pair of laws


def out_root() -> pathlib.Path:
    root = os.environ.get("MCMC_PAPER_OUT")
    if not root:
        raise RuntimeError("MCMC_PAPER_OUT is not set")
    return pathlib.Path(root)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def grid_from_T(T: int) -> list[int]:
    if T < 1 or T & (T - 1):
        raise ValueError(f"T must be a power of two, got {T}")
    return [0] + [2**i for i in range(int(math.log2(T)) + 1)]


def unit_dirs(model: str, sampler: str) -> dict:
    root = out_root()
    return {
        "r2": root / STUDY_R2 / "main" / f"{model}__{sampler}_v1",
        "eval2": root / STUDY_R2 / "main" / f"{model}__{sampler}_v1_eval2",
        "r1_states": root / "realdata_r1_export" / "main" / f"{model}__{sampler}_states_v1",
    }


def load_yaml(path: pathlib.Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


# Confidence-radius floors
def replica_floor(
    J: int, n_inspections: int, n_budgets: int, beta: float = BETA["replicas"]
) -> float:
    """b_{t,J} = (1 + sqrt(2 log(1/(beta w))))/sqrt(J), w = 1/(n_inspections n_budgets), kappa = 1.

    Same formula as ``bounds.replica_radius``; uniform weights over the declared inspection x J grid.
    """
    from mcmc.paperexp.bounds import replica_radius

    w = 1.0 / (n_inspections * n_budgets)
    b = replica_radius(KAPPA, J, beta, w)
    alt = (1.0 + math.sqrt(2.0 * math.log(1.0 / (beta * w)))) / math.sqrt(J)
    assert abs(b - alt) < 1e-12, (b, alt)
    return b


def j_needed(eps: float, n_inspections: int, n_budgets: int, beta: float = BETA["replicas"]) -> int:
    """Smallest J with b_{t,J} <= eps at the same weight (the weight held fixed)."""
    w = 1.0 / (n_inspections * n_budgets)
    return math.ceil(((1.0 + math.sqrt(2.0 * math.log(1.0 / (beta * w)))) / eps) ** 2)


def operator_floor(R: float, N: int, n_inspections: int, beta: float = BETA["operator"]) -> float:
    """R sqrt(e_{t,N}) (App. D): the operator term of U_t at an ASSUMED radius R, other terms zero."""
    from mcmc.paperexp.operators import operator_radius

    return R * math.sqrt(operator_radius(KAPPA**2, N, beta, 1.0 / n_inspections))


def floor_table() -> list[dict]:
    cfg = load_yaml(REPO / "configs" / "experiment" / "exp_r2_german_credit.yaml")
    grids: dict[int, list[str]] = {}
    for key, fr in cfg["frozen"].items():
        n_t = len(grid_from_T(max(int(fr["T"]), int(cfg["horizon_min"]))))
        grids.setdefault(n_t, []).append(key)
    N = int(cfg["sigma"]["N"])
    rows = []
    for n_t in sorted(grids):
        for J in J_GRID:
            row = {
                "n_inspections": n_t,
                "units": " ".join(sorted(grids[n_t])),
                "J": J,
                "beta_H": BETA["replicas"],
                "weight_primary": 1.0 / (n_t * len(J_GRID)),
                # primary: uniform over inspections x J in {4, 16, 64}
                "floor_b_tJ": replica_floor(J, n_t, len(J_GRID)),
                # one J only (weights over inspections only)
                "floor_b_tJ_single_budget": replica_floor(J, n_t, 1),
                # one inspection, one J (w = 1): the smallest value any allocation of beta_H can give
                "floor_b_single_look": replica_floor(J, 1, 1),
                "trivial_mmd_max": math.sqrt(2.0),
                "J_needed_eps0.10": j_needed(0.10, n_t, len(J_GRID)),
                "J_needed_eps0.05": j_needed(0.05, n_t, len(J_GRID)),
                "operator_N": N,
                "beta_Sigma": BETA["operator"],
            }
            for R in (0.5, 0.8, 1.0, 2.0, 4.0):
                row[f"operator_floor_R{R:g}"] = operator_floor(R, N, n_t)
            row["vacuous_primary"] = row["floor_b_tJ"] >= math.sqrt(2.0)
            row["reaches_0.10"] = row["floor_b_tJ"] <= 0.10
            row["reaches_0.05"] = row["floor_b_tJ"] <= 0.05
            rows.append(row)
    return rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows:
        if list(r) != keys:
            raise ValueError("rows have different columns")
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(tmp, path)


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=_default))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# Saved input files
def _npz_shapes(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: list(z[k].shape) for k in z.files}


def reference_set_info(run_id: str, load: bool) -> dict:
    """RUN-STATUS / config of one NUTS reference set; optionally load and check its draws."""
    from mcmc.paperexp.german_credit_reference import load_set, set_path

    path = set_path("reference", run_id)
    st = json.loads((path / "RUN-STATUS.json").read_text())
    spec = load_yaml(path / "config.yaml")
    units = sorted(p.name for p in path.glob("unit_*.npz"))
    info = {
        "path": str(path),
        "status": st["status"],
        "n_runs": int(spec["n_runs"]),
        "n_draws": int(spec["n_draws"]),
        "unit_files": len(units),
    }
    if st["status"] != "COMPLETED" or len(units) != info["n_runs"]:
        raise RuntimeError(f"reference set {run_id} unusable: {info}")
    if load:
        draws = load_set(path)["draws"]
        info["draws_shape"] = list(draws.shape)
        info["unit_sha256"] = {u: sha256_file(path / u) for u in units}
    return info


def inventory_unit(model: str, sampler: str, frozen: dict) -> dict:
    d = unit_dirs(model, sampler)
    r2 = d["r2"]
    st = json.loads((r2 / "RUN-STATUS.json").read_text())
    cfg = load_yaml(r2 / "config.yaml")
    key = f"{model}__{sampler}"
    fr = cfg["frozen"][key]
    if {k: fr[k] for k in ("h", "T", "t_star")} != {
        k: frozen[key][k] for k in ("h", "T", "t_star")
    }:
        raise RuntimeError(f"{key}: run config frozen values differ from the frozen config")
    grid = grid_from_T(max(int(fr["T"]), int(cfg["horizon_min"])))
    J, G, K = (
        int(cfg["ensembles"]["J"]),
        int(cfg["ensembles"]["G"]),
        int(cfg["ensembles"]["K_conv"]),
    )
    refs = cfg["references"][model]
    u = {
        "model": model,
        "sampler": sampler,
        "r2_dir": str(r2),
        "r2_status": st["status"],
        "frozen": {k: fr[k] for k in ("h", "T", "t_star", "cap_binds")},
        "grid": grid,
        "J": J,
        "G": G,
        "K_conv": K,
        "sigma_N": int(cfg["sigma"]["N"]),
        "references": refs,
        "files": {},
        "laws": {},
        "checks": {},
    }
    if st["status"] != "COMPLETED":
        raise RuntimeError(f"{r2} is {st['status']}")
    # setup / sigma / eval2
    setup = np.load(r2 / "setup.npz", allow_pickle=False)
    labels = json.loads(str(setup["run_labels"]))
    u["files"]["setup.npz"] = {
        "sha256": sha256_file(r2 / "setup.npz"),
        "shapes": _npz_shapes(r2 / "setup.npz"),
    }
    u["run_labels"] = labels
    r = setup["B"].shape[1]
    u["feature_rank"] = int(r)
    if (
        setup["feat_run_means"].shape != (len(labels), r)
        or setup["run_inner"].shape != (len(labels),) * 2
    ):
        raise RuntimeError(f"{key}: setup.npz shapes inconsistent")
    sig = np.load(r2 / "sigma.npz", allow_pickle=False)
    if list(sig["grid"]) != grid or sig["O"].shape != (len(grid), r, r):
        raise RuntimeError(
            f"{key}: sigma.npz grid/shape {sig['O'].shape} vs grid {len(grid)}, r {r}"
        )
    for k in ("O", "OA", "OB"):
        if not np.isfinite(sig[k]).all():
            raise RuntimeError(f"{key}: non-finite {k} in sigma.npz")
    u["files"]["sigma.npz"] = {
        "sha256": sha256_file(r2 / "sigma.npz"),
        "shapes": _npz_shapes(r2 / "sigma.npz"),
    }
    ev = d["eval2"]
    est = json.loads((ev / "RUN-STATUS.json").read_text())
    u["eval2"] = {
        "dir": str(ev),
        "status": est["status"],
        "sha256_eval": sha256_file(ev / "eval.npz"),
    }
    for f in (
        "pilotrun_stationary.npz",
        "pilots_stationary.npz",
        "pilotrun_confined.npz",
        "pilots_confined.npz",
    ):
        u["files"][f] = {
            "exists": (r2 / f).exists(),
            "bytes": (r2 / f).stat().st_size if (r2 / f).exists() else 0,
        }
    for f in ("nuts_I1.npz", "nuts_I2.npz"):
        u["files"][f] = {"exists": (r2 / f).exists()}
    # laws: summaries, states, conventional paths
    for law in LAWS:
        ens = np.load(r2 / f"ens_{law}.npz", allow_pickle=False)
        if list(ens["grid"]) != grid:
            raise RuntimeError(f"{key}/{law}: ens grid differs")
        exp = {
            "group_feat": (len(grid), G, r),
            "group_pair_ksum": (len(grid), G, G),
            "group_run_cross": (len(grid), G, len(labels)),
            "accept_per_chain": (J,),
        }
        for k, shp in exp.items():
            if ens[k].shape != shp or not np.isfinite(ens[k]).all():
                raise RuntimeError(f"{key}/{law}: {k} shape {ens[k].shape} != {shp} or non-finite")
        f_r2 = r2 / f"ens_{law}_states.npz"
        f_r1 = d["r1_states"] / f"states_{law}.npz"
        src = f_r2 if f_r2.exists() else f_r1
        z = np.load(src, allow_pickle=False)
        states = z["states"]
        if list(z["grid"]) != grid or states.shape != (len(grid), J, states.shape[2]):
            raise RuntimeError(f"{key}/{law}: states grid/shape {states.shape}")
        if not np.isfinite(states).all():
            raise RuntimeError(f"{key}/{law}: non-finite states")
        dim = states.shape[2]
        law_info = {
            "states_source": str(src),
            "states_source_kind": "R2 run (ens_<law>_states.npz)"
            if src == f_r2
            else "R1 export (deterministic re-run of the R2 stream, verified against R2 group_feat)",
            "states_sha256": sha256_file(src),
            "states_shape": list(states.shape),
            "summary_sha256": sha256_file(r2 / f"ens_{law}.npz"),
            "never_accepted": int((ens["accept_per_chain"] == 0).sum()),
        }
        if src == f_r2 and f_r1.exists():  # the R1 export copy must hold the same array
            z1 = np.load(f_r1, allow_pickle=False)
            law_info["r1_copy_identical"] = bool(np.array_equal(z1["states"], states))
            if not law_info["r1_copy_identical"]:
                raise RuntimeError(f"{key}/{law}: R1 export copy differs from the R2 states")
        r1meta = d["r1_states"] / "meta.json"
        if r1meta.exists():
            m = json.loads(r1meta.read_text())["laws"][law]
            law_info["r1_verify_group_feat_max_rel"] = m["verify_group_feat_max_rel"]
            law_info["r1_how"] = m["how"]
        # conventional full paths of the first K replicas: identity with the checkpoint states
        traj = np.load(r2 / f"ens_{law}_conv_traj.npz", allow_pickle=False)["traj"]
        if traj.shape != (grid[-1] + 1, K, dim):
            raise RuntimeError(f"{key}/{law}: conv_traj shape {traj.shape}")
        same = np.array_equal(traj[np.asarray(grid)], states[:, :K])
        law_info["conv_traj_shape"] = list(traj.shape)
        law_info["conv_traj_equals_states_first_K"] = bool(same)
        if not same:
            raise RuntimeError(
                f"{key}/{law}: saved paths differ from checkpoint states (first {K})"
            )
        u["laws"][law] = law_info
        u["d"] = int(dim)
    return u


def reproduce_checkpoint(
    model: str, sampler: str, law: str, t: int, device: str, threads: int
) -> dict:
    """Recompute one checkpoint summary (all 32 groups) from raw states with the R2 code path."""
    import torch
    from omegaconf import OmegaConf

    from mcmc.experiments import exp_r2_german_credit as r2x
    from mcmc.paperexp import german_credit_r2 as r2

    torch.set_num_threads(threads)
    d = unit_dirs(model, sampler)
    cfg = OmegaConf.merge(
        OmegaConf.create((d["r2"] / "config.yaml").read_text()), {"device": device}
    )
    t0 = time.time()
    ctx = r2x.Context(cfg, types.SimpleNamespace(path=d["r2"]))  # read-only; adopts the saved basis
    ctx.load_frozen()
    ctx.ensure_ref()  # saved setup.npz (reference run means and inner products)
    t_ctx = time.time() - t0
    i = ctx.grid.index(t)
    f_r2 = d["r2"] / f"ens_{law}_states.npz"
    src = f_r2 if f_r2.exists() else d["r1_states"] / f"states_{law}.npz"
    x = torch.as_tensor(np.load(src)["states"][i], dtype=torch.float64, device=device)
    e = cfg.ensembles
    t1 = time.time()
    st = r2.checkpoint_stats(
        x, ctx.kernel, ctx.feats, ctx.ref, int(e.G), ctx.mem, list(e.quantiles)
    )
    t_stats = time.time() - t1
    saved = np.load(d["r2"] / f"ens_{law}.npz", allow_pickle=False)
    out = {
        "unit": f"{model}__{sampler}",
        "law": law,
        "t": t,
        "grid_index": i,
        "states_source": str(src),
    }
    for k in (
        "group_feat",
        "group_pair_ksum",
        "group_run_cross",
        "group_coord_mean",
        "coord_quantiles",
    ):
        a, b = st[k], saved[k][i]
        out[f"{k}_max_rel"] = float(np.abs(a - b).max() / np.abs(b).max())
    G = int(e.G)
    runs = ctx.choice_runs()
    ev = np.load(d["eval2"] / "eval.npz", allow_pickle=False)
    li = list(ev["laws"]).index(law)
    for ci, ch in enumerate(ev["choices"]):
        mmd2, se = r2.mmd2_with_se(
            st["group_pair_ksum"],
            st["group_run_cross"],
            ctx.ref.run_inner,
            int(e.J) // G,
            runs[str(ch)],
        )
        out[f"mmd2_{ch}_recomputed"] = mmd2
        out[f"mmd2_{ch}_eval2"] = float(ev["mmd2"][li, i, ci])
        out[f"mmd2_se_{ch}_recomputed"] = se
        out[f"mmd2_se_{ch}_eval2"] = float(ev["mmd2_se"][li, i, ci])
    out["kernel_evals"] = int(x.shape[0]) * (
        int(x.shape[0]) + sum(X.shape[0] for X in ctx.ref.X) + ctx.feats.n_landmarks
    )
    out["wall_context_s"] = t_ctx
    out["wall_stats_s"] = t_stats
    out["device"] = device
    out["threads"] = threads
    tol = 1e-10
    bad = [k for k, v in out.items() if k.endswith("_max_rel") and not v < tol]
    for ch in ev["choices"]:
        if abs(out[f"mmd2_{ch}_recomputed"] - out[f"mmd2_{ch}_eval2"]) > 1e-12 + 1e-9 * abs(
            out[f"mmd2_{ch}_eval2"]
        ):
            bad.append(f"mmd2_{ch}")
    out["reproduced"] = not bad
    if bad:
        raise RuntimeError(f"checkpoint not reproduced: {bad}: {out}")
    return out


def bench_cross(
    model: str, sampler: str, n_rows: int, n_runs: int, device: str, threads: int
) -> dict:
    """Wall time of the heaviest optional Stage-2 term: n_rows replicas x n_runs whole reference runs."""
    import torch

    from mcmc.paperexp import german_credit_r2 as r2
    from mcmc.paperexp.german_credit_reference import load_kernel_file, load_set, set_path
    from mcmc.paperexp.kernels import kernel_rowsum

    torch.set_num_threads(threads)
    d = unit_dirs(model, sampler)
    cfg = load_yaml(d["r2"] / "config.yaml")
    refs = cfg["references"][model]
    entry = load_kernel_file(REPO / cfg["kernel_file"], refs["kernel_key"])
    k = r2.make_kernel(entry)
    draws = load_set(set_path("reference", next(iter(refs["evaluation"].values()))))["draws"][
        :n_runs
    ]
    f_r2 = d["r2"] / "ens_I1_states.npz"
    src = f_r2 if f_r2.exists() else d["r1_states"] / "states_I1.npz"
    x = torch.as_tensor(np.load(src)["states"][-1][:n_rows], dtype=torch.float64, device=device)
    Y = torch.as_tensor(draws.reshape(-1, draws.shape[2]), dtype=torch.float64, device=device)
    t0 = time.time()
    s = kernel_rowsum(k, x, Y, int(cfg["mem_bytes"]))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    assert torch.isfinite(s).all()
    return {
        "unit": f"{model}__{sampler}",
        "d": int(x.shape[1]),
        "rows": n_rows,
        "cols": int(Y.shape[0]),
        "kernel_evals": n_rows * int(Y.shape[0]),
        "wall_s": wall,
        "device": device,
        "threads": threads,
        "evals_per_s": n_rows * int(Y.shape[0]) / wall,
    }


def cmd_inventory(args) -> None:
    frozen = load_yaml(REPO / "configs" / "experiment" / "exp_r2_german_credit.yaml")["frozen"]
    inv = {
        "units": {},
        "reference_sets": {},
    }
    for model in MODELS:
        for sampler in SAMPLERS:
            u = inventory_unit(model, sampler, frozen)
            inv["units"][f"{model}__{sampler}"] = u
            print(
                f"[inventory] {model}__{sampler}: OK grid {len(u['grid'])} d {u['d']}", flush=True
            )
            for role in ("pool", "calibration"):
                rid = u["references"][role]
                if rid not in inv["reference_sets"]:
                    inv["reference_sets"][rid] = {
                        "role": role,
                        **reference_set_info(rid, load=False),
                    }
            for name, rid in u["references"]["evaluation"].items():
                if rid not in inv["reference_sets"]:
                    inv["reference_sets"][rid] = {
                        "role": f"evaluation:{name}",
                        **reference_set_info(rid, load=True),
                    }
    if args.reproduce:
        model, sampler, law, t = args.reproduce.split(":")
        inv["reproduction"] = reproduce_checkpoint(
            model, sampler, law, int(t), args.device, args.threads
        )
        print(f"[reproduce] {json.dumps(inv['reproduction'], default=_default)}", flush=True)
    if args.bench:
        inv["bench"] = [
            bench_cross(m, s, 2048, 8, args.device, args.threads)
            for m, s in (("hier", "mala"), ("hier_sonar", "mala"), ("horseshoe", "mala"))
        ]
        print(f"[bench] {json.dumps(inv['bench'])}", flush=True)
    print(f"Checked {len(inv['units'])} units and {len(inv['reference_sets'])} reference sets.")


def cmd_floor(args) -> None:
    rows = floor_table()
    write_csv(OUT / "floor_table.csv", rows)
    for r in rows:
        print(
            f"n_t={r['n_inspections']:2d} J={r['J']:3d} floor={r['floor_b_tJ']:.3f} "
            f"(one J {r['floor_b_tJ_single_budget']:.3f}, one look {r['floor_b_single_look']:.3f}) "
            f"J_needed 0.10/0.05 = {r['J_needed_eps0.10']}/{r['J_needed_eps0.05']}; "
            f"operator R=0.5/1/2/4: {r['operator_floor_R0.5']:.3f}/{r['operator_floor_R1']:.3f}/"
            f"{r['operator_floor_R2']:.3f}/{r['operator_floor_R4']:.3f}"
        )
    print(OUT / "floor_table.csv")


def cmd_bench(args) -> None:
    res = [
        bench_cross(m, s, 2048, 8, args.device, args.threads)
        for m, s in (
            ("hier", "mala"),
            ("hier", "mala"),
            ("hier_sonar", "mala"),
            ("horseshoe", "mala"),
        )
    ]  # the first call includes device warm-up
    for r in res:
        print(json.dumps(r), flush=True)


# Fix splits and analysis choices before evaluating held-out results.
STUDY_C = "c_realdata_followup"
N_REPLICAS = 4096
N_OBS = 2048  # observation replicas 0..2047; evaluation replicas 2048..4095
R2_GROUP = 128  # the R2 summaries hold 32 groups of 128 consecutive replicas
EVAL_GROUPS = np.arange(16, 32)  # = replicas 2048..4095
N_RUNS_SET = 16
READ_RUNS = np.arange(0, 8)  # within each evaluation NUTS set: runs supplying the readings' target
HELD_RUNS = np.arange(8, 16)  # runs supplying the held-out evaluation discrepancy
SPLITS = ("heldout", "swap")  # swap: readings use runs 8-15, held-out evaluation runs 0-7
FAMILIES = ("S3", "S4own", "S4same")
M_LIST = (8, 4)  # m = 8 primary, m = 4 sensitivity; V_4 = first 4 columns of V_8 (nested)
MARGINS = ("none", "halfsplit")
PRIMARY = {"family": "S4own", "m": 8, "margin": "none", "R": 1.0, "split": "heldout"}
Z_HELD = 2.0  # pointwise +-2 jackknife SE interval for held-out estimates (not simultaneous)
CHECK_RTOL = 1e-10
SUMMARY_KEYS = ("group_feat", "group_pair_ksum", "group_run_cross", "group_coord_mean")
CONV_RULE = {"rhat_lt": 1.01, "ess_bulk_ge": 400, "ess_tail_ge": 400}


def group_offsets() -> dict:
    """Observation groups are stored concatenated [J=4 (512) | J=16 (128) | J=64 (32)]."""
    out, off = {}, 0
    for J in J_GRID:
        out[J] = (off, N_OBS // J)
        off += N_OBS // J
    return out


N_GROUPS_TOTAL = sum(N_OBS // J for J in J_GRID)


def observation_groups(J: int) -> np.ndarray:
    """(N_OBS / J, J) replica indices: contiguous blocks, identical membership at every t."""
    if N_OBS % J:
        raise ValueError(f"J={J} does not divide {N_OBS}")
    return np.arange(N_OBS).reshape(-1, J)


def check_split() -> dict:
    """Check that replica and reference-run splits are disjoint and complete."""
    obs = np.arange(N_OBS)
    ev = (EVAL_GROUPS[:, None] * R2_GROUP + np.arange(R2_GROUP)).ravel()
    if not np.array_equal(ev, np.arange(N_OBS, N_REPLICAS)):
        raise AssertionError("evaluation groups 16..31 are not replicas 2048..4095")
    if np.intersect1d(obs, ev).size or obs.size + ev.size != N_REPLICAS:
        raise AssertionError("observation and evaluation replicas overlap or miss replicas")
    for Js, Jb in zip(J_GRID[:-1], J_GRID[1:]):
        small, big = observation_groups(Js), observation_groups(Jb)
        if not np.array_equal(small.reshape(big.shape[0], -1), big):
            raise AssertionError(f"groups of {Js} are not nested in groups of {Jb}")
    if np.intersect1d(READ_RUNS, HELD_RUNS).size or READ_RUNS.size + HELD_RUNS.size != N_RUNS_SET:
        raise AssertionError("reference run split is not a partition")
    return {
        "observation_replicas": [0, N_OBS - 1],
        "evaluation_replicas": [N_OBS, N_REPLICAS - 1],
        "evaluation_replicas_as_r2_groups": EVAL_GROUPS.tolist(),
        "observation_groups": {
            str(J): {"n_groups": N_OBS // J, "group_g": f"replicas [{J}g, {J}g+{J})"}
            for J in J_GRID
        },
        "nested": "each 64-group is four consecutive 16-groups, each 16-group four consecutive 4-groups",
        "reading_runs": READ_RUNS.tolist(),
        "heldout_runs": HELD_RUNS.tolist(),
        "max_overlap": 0,
    }


def design() -> dict:
    return {
        "study": STUDY_C,
        "version": "v1",
        "units": {
            "primary": [f"{m}__{s}" for m in PRIMARY_MODELS for s in SAMPLERS],
            "appendix_sensitivity": [f"horseshoe__{s}" for s in SAMPLERS],
            "horseshoe_reference_sets": "v1 and v2, each split by whole run as below; never selected",
        },
        "laws": {"primary_practical": ["I1", "I2"], "controls": ["I0", "I3"]},
        "frozen_from_R2": "kernels, Nystrom basis (setup.npz), samplers, inspection grids, start laws, "
        "branched-pair operators (sigma.npz); nothing re-tuned",
        "replica_split": {
            **check_split(),
            "rule": "contiguous halves of the 4096 saved replicas",
            "iid_justification": (
                "the 4096 replicas of a law are independent and identically distributed by "
                "construction: initial states are iid draws (torch.randint from the pool, or iid "
                "uniform for I2) and each step uses independent per-chain proposals and uniforms, so "
                "any outcome-independent index split gives iid halves; contiguous halves align with "
                "the 32 saved R2 groups of 128, so the evaluation half (groups 16..31) is checked "
                "against the saved full-ensemble summaries"
            ),
        },
        "spans": {
            "S3": "V*_m(0): top-m eigenvectors of the static estimate O_0 (sigma.npz, t = 0); "
            "width^2 = lambda_{m+1}(O_{0,+}) (Section 3 baseline, r = 1/2)",
            "S4own": "V*_m(t): top-m eigenvectors of O_t; width^2 = lambda_{m+1}(O_{t,+}) (Section 4)",
            "S4same": "V*_m(0) at time t; width^2 = lambda_max(P_perp O_{t,+} P_perp), the compression "
            "of O_{t,+} onto the complement of V*_m(0) (Section 4 on the static span)",
            "m": list(M_LIST),
            "no_observation": "R sqrt(lambda_1(O_{t,+})) reported per t (no readings)",
        },
        "operator_margin": {
            "none": "primary plug-in (eigenvalues of the estimate)",
            "halfsplit": "sensitivity plug-in: + ||O_A - O_B||/2 (eps_0 for S3, eps_t otherwise); "
            "a half-split proxy, not a bound",
        },
        "R_assumed": list(R_ASSUMED),
        "eps": list(EPS),
        "readings": "empirical feature mean of each observation group minus the mean feature embedding "
        "of the reading NUTS runs; reading h = ||V^T (fbar_g - fbar_ref)|| in the 512-feature Nystrom "
        "representation (an RKHS-orthonormal basis of the landmark span)",
        "plugin": "P(R) = sqrt(h^2 + R^2 width^2): REFERENCE-BASED PLUG-IN AT AN ASSUMED RADIUS, FEATURE "
        "SPACE; never a confidence bound or a certified stopping rule",
        "heldout": "evaluation replicas (16 groups of 128) vs held-out runs: U-statistic MMD^2 "
        "(german_credit_r2.mmd2_jackknife), debiased feature ||delta||^2 and ||P_V delta||^2 "
        "(projection_jackknife), omitted-feature part MMD^2 - feature^2, unmonitored residuals; SEs by "
        "delete-one-group + delete-one-run jackknife; all signed, never clipped",
        "heldout_status": f"ABOVE eps if est - {Z_HELD} SE > eps^2, BELOW if est + {Z_HELD} SE < eps^2, "
        "else UNRESOLVED (pointwise, not simultaneous over t; empirical, not a guarantee)",
        "floor": "floor_table.csv: deterministic confidence-radius floors",
        "budget": BETA,
        "primary_display": PRIMARY,
        "conventional_join": {
            "groups": "the 16 observation groups of 4 with saved full paths (replicas 0..63) = R2 conv groups",
            "rule": CONV_RULE,
            "window": "second half (t/2, t] of each path (R2 conv_<law>.npz)",
            "status": "appendix, descriptive",
        },
        "no_binomial_intervals": "groups share one reference, operator and feature map: fractions "
        "over groups are descriptive only",
        "shared_randomness": "within a unit every group shares the Nystrom landmarks (calibration run), "
        "the operator estimates O_t (pool starts, stream r2-sigma), the reading reference runs and the "
        "held-out evaluation half and runs; groups of 4/16/64 are nested, paired and dependent; the "
        "observation groups are independent of each other only given these shared inputs",
        "feature_space": "readings, widths and plug-ins live in the 512-feature Nystrom space and address "
        "the feature discrepancy ||P_Z delta||; the full MMD adds an omitted-feature part for which no "
        "bound is supplied (reported separately as MMD^2 - feature^2)",
        "checks": f"recomputed {', '.join(SUMMARY_KEYS)} for all 32 groups at every t vs the saved R2 "
        f"summaries (max rel {CHECK_RTOL:g}); full-ensemble MMD^2 (all runs) vs eval2",
    }


def unit_sources(model: str, sampler: str) -> dict:
    """Every file the analysis reads for one unit (path -> role)."""
    d = unit_dirs(model, sampler)
    r2 = d["r2"]
    src = {
        str(r2 / "config.yaml"): "R2 resolved config (frozen h, T, t*, grid, references)",
        str(r2 / "setup.npz"): "Nystrom basis, reference run feature means and inner products",
        str(r2 / "setup.json"): "kernel, start laws",
        str(r2 / "sigma.npz"): "branched-pair operators O_t, O_A, O_B",
        str(r2 / "sigma.json"): "operator provenance and cost",
        str(d["eval2"] / "eval.npz"): "full-ensemble MMD^2 (check)",
    }
    for law in LAWS:
        src[str(r2 / f"ens_{law}.npz")] = "saved checkpoint summaries"
        src[str(r2 / f"ens_{law}.json")] = "trajectory cost"
        src[str(r2 / f"conv_{law}.npz")] = (
            "conventional diagnostics of the 16 saved four-chain groups"
        )
        f2 = r2 / f"ens_{law}_states.npz"
        src[str(f2 if f2.exists() else d["r1_states"] / f"states_{law}.npz")] = "replica states"
    return src


def independence_record(model: str, sampler: str) -> dict:
    """Confirm the operators, monitors and features use no evaluation replica and no held-out run."""
    d = unit_dirs(model, sampler)
    cfg = load_yaml(d["r2"] / "config.yaml")
    refs = cfg["references"][model]
    eval_ids = set(refs["evaluation"].values())
    sig = json.loads((d["r2"] / "sigma.json").read_text())
    if refs["pool"] in eval_ids or refs["calibration"] in eval_ids:
        raise RuntimeError(f"{model}: pool/calibration set coincides with an evaluation set")
    if not sig["starts"].startswith("X from pool runs"):
        raise RuntimeError(
            f"{model}__{sampler}: operator starts not from the pool: {sig['starts']}"
        )
    tmax = out_root() / STUDY_R2 / "pilot"
    pil = sorted(tmax.glob(f"{model}__{sampler}_v*/tmax.json"))
    pilot = [
        {k: json.loads(p.read_text())[k] for k in ("T", "reference_choice")}
        for p in pil
    ]
    return {
        "operators": f"sigma.npz: branched pairs started from the pool set {refs['pool']} ({sig['starts']}),"
        " simulated with their own stream 'r2-sigma'; they use no ensemble replica and no evaluation "
        "NUTS run (neither runs 0-7 nor 8-15)",
        "monitors": "V*_m(0), V*_m(t) are eigenvectors of the saved O_0, O_t only",
        "features": f"landmarks = evenly spaced draws of the calibration run {refs['calibration']}",
        "start_laws": f"I0, I1, I3 from the pool set {refs['pool']}; I2 uniform box",
        "evaluation_sets": refs["evaluation"],
        "uses_evaluation_replicas_or_heldout_runs": False,
        "disclosed_conditionality": "the inspection horizon T was fixed in R2 by an independent pilot "
        "(rep offset 1000) whose I1 MMD curve used ALL 16 runs of the evaluation set (reference_choice "
        "below), including the held-out runs 8-15; the grid is reused unchanged",
        "pilot_tmax": pilot,
    }


def build_manifest() -> dict:
    des = design()
    man = {
        "design_sha256": hashlib.sha256(json.dumps(des, sort_keys=True).encode()).hexdigest(),
        "units": {},
        "reference_sets": {},
        "regression_summaries": {},
    }
    from mcmc.paperexp.german_credit_reference import set_path

    for model in MODELS:
        for sampler in SAMPLERS:
            src = unit_sources(model, sampler)
            man["units"][f"{model}__{sampler}"] = {
                "sources": {
                    p: {"role": role, "sha256": sha256_file(pathlib.Path(p))}
                    for p, role in src.items()
                },
                "independence": independence_record(model, sampler),
            }
            cfg = load_yaml(unit_dirs(model, sampler)["r2"] / "config.yaml")
            refs = cfg["references"][model]
            roles = {refs["pool"]: "pool", refs["calibration"]: "calibration"}
            roles.update({rid: f"evaluation:{n}" for n, rid in refs["evaluation"].items()})
            for rid, role in roles.items():
                if rid in man["reference_sets"]:
                    continue
                path = set_path("reference", rid)
                units = sorted(path.glob("unit_*.npz"))
                man["reference_sets"][rid] = {
                    "role": role,
                    "path": str(path),
                    "units": {u.name: sha256_file(u) for u in units},
                }
                if role.startswith("evaluation") and len(units) != N_RUNS_SET:
                    raise RuntimeError(f"{rid}: {len(units)} runs, expected {N_RUNS_SET}")
            print(f"[manifest] {model}__{sampler}", flush=True)
    for ds, summ in REG_FILES.items():
        man["regression_summaries"][ds] = {
            summ: sha256_file(REPO / "paper" / "figures" / "data" / summ)
        }
    return man


def cmd_freeze(args) -> None:
    des = design()
    man = build_manifest()
    write_json(OUT / "stage2_design.json", des)
    write_json(OUT / "manifest.json", man)
    print(OUT / "stage2_design.json", OUT / "manifest.json")


def design_parameters(record: dict) -> dict:
    """Numerical choices and sample splits, excluding descriptive metadata."""
    split_keys = ("observation_replicas", "evaluation_replicas",
                  "evaluation_replicas_as_r2_groups", "observation_groups",
                  "reading_runs", "heldout_runs", "max_overlap")
    return {
        **{k: record[k] for k in ("study", "version", "laws", "R_assumed", "eps",
                                  "budget", "primary_display")},
        "units": {k: record["units"][k] for k in ("primary", "appendix_sensitivity")},
        "replica_split": {k: record["replica_split"][k] for k in split_keys},
        "m": record["spans"]["m"],
        "operator_margins": sorted(record["operator_margin"]),
        "conventional_rule": record["conventional_join"]["rule"],
    }


def verify_frozen() -> tuple[dict, dict, dict]:
    """Check the saved design, sample splits and manifest hash."""
    des_f, man_f = OUT / "stage2_design.json", OUT / "manifest.json"
    des, man = json.loads(des_f.read_text()), json.loads(man_f.read_text())
    if design_parameters(des) != design_parameters(design()):
        raise RuntimeError("stage2_design.json parameters differ from the code")
    digest = hashlib.sha256(json.dumps(des, sort_keys=True).encode()).hexdigest()
    if man["design_sha256"] != digest:
        raise RuntimeError("manifest design hash mismatch")
    hashes = {"design_sha256": sha256_file(des_f), "manifest_sha256": sha256_file(man_f)}
    return des, man, hashes


def verify_hashes(expected: dict) -> int:
    """Raise unless every file hashes to its expected sha256 (path -> hex digest)."""
    for p, h in expected.items():
        got = sha256_file(resolve_output_path(p))
        if got != h:
            raise RuntimeError(f"{p}: sha256 {got} differs from the manifest {h}")
    return len(expected)


def unit_reference_ids(model: str) -> list[str]:
    """NUTS reference sets one unit reads: evaluation set(s), pool and calibration run."""
    refs = load_yaml(unit_dirs(model, "mala")["r2"] / "config.yaml")["references"][model]
    return list(refs["evaluation"].values()) + [refs["pool"], refs["calibration"]]


def verify_sources(man: dict, unit: str) -> dict:
    """Unit source files AND every NUTS reference-set file the unit reads (raise on mismatch)."""
    n_src = verify_hashes({p: rec["sha256"] for p, rec in man["units"][unit]["sources"].items()})
    n_ref = 0
    for rid in unit_reference_ids(unit.split("__")[0]):
        rs = man["reference_sets"][rid]
        n_ref += verify_hashes(
            {str(pathlib.Path(rs["path"]) / u): h for u, h in rs["units"].items()}
        )
    return {"unit_sources": n_src, "reference_files": n_ref}


def verify_regression_inputs(man: dict) -> int:
    root = REPO / "paper" / "figures" / "data"
    return sum(
        verify_hashes({str(root / p): h for p, h in files.items()})
        for files in man["regression_summaries"].values()
    )


def cmd_verify(args) -> None:
    """Re-verify every manifest hash (unit sources, NUTS sets, regression inputs)."""
    man = json.loads((OUT / "manifest.json").read_text())
    res = {u: verify_sources(man, u) for u in man["units"]}
    files_ref = {
        str(pathlib.Path(rs["path"]) / u)
        for rs in man["reference_sets"].values()
        for u in rs["units"]
    }
    out = {
        "units": res,
        "distinct_unit_source_files": sum(len(u["sources"]) for u in man["units"].values()),
        "distinct_reference_files": len(files_ref),
        "regression_inputs": verify_regression_inputs(man),
        "all_match": True,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "units"}))


# Readings and held-out estimates from raw states
def operator_widths(sig, grid: list[int]) -> dict:
    """Eigen-structure and plug-in widths of the saved feature-space operators (float64, CPU)."""
    from mcmc.paperexp import german_credit_r2 as r2

    if list(sig["grid"]) != list(grid):
        raise RuntimeError("sigma.npz grid differs from the unit grid")
    n_t, r = sig["O"].shape[0], sig["O"].shape[1]
    mmax = max(M_LIST)
    lam = np.empty((n_t, r))
    Vt = np.empty((n_t, r, mmax))
    eps = np.empty(n_t)
    most_neg = np.empty(n_t)
    pos = []
    for i in range(n_t):
        op = r2.OperatorSummary.from_stats(sig["O"][i], sig["OA"][i], sig["OB"][i])
        lam[i], Vt[i], eps[i], most_neg[i] = (
            op.eigvals,
            op.eigvecs[:, :mmax],
            op.eps,
            op.most_negative,
        )
        lp = np.clip(op.eigvals, 0.0, None)  # positive part O_{t,+}
        pos.append((op.eigvecs * lp) @ op.eigvecs.T)
    V0 = Vt[0].copy()
    lamp = np.clip(lam, 0.0, None)
    out = {
        "lam": lam,
        "eps": eps,
        "most_negative": most_neg,
        "n_pos": (lam > 0).sum(1),
        "V0": V0,
        "Vt": Vt,
        "lam1_pos": lamp[:, 0],
        "orth_err": np.array([np.abs(Vt[i].T @ Vt[i] - np.eye(mmax)).max() for i in range(n_t)]),
    }
    for m in M_LIST:
        out[f"w2_S3_m{m}"] = np.full(n_t, lamp[0, m])
        out[f"w2_S4own_m{m}"] = lamp[:, m].copy()
        P = np.eye(r) - V0[:, :m] @ V0[:, :m].T
        same = np.empty(n_t)
        for i in range(n_t):
            M = P @ pos[i] @ P
            same[i] = np.linalg.eigvalsh(0.5 * (M + M.T))[-1]
            # witness: the compression onto the complement of the OWN span is lambda_{m+1}(O_{t,+})
            Pt = np.eye(r) - Vt[i][:, :m] @ Vt[i][:, :m].T
            Mt = Pt @ pos[i] @ Pt
            own = np.linalg.eigvalsh(0.5 * (Mt + Mt.T))[-1]
            if abs(own - lamp[i, m]) > 1e-12 + 1e-9 * lamp[i, 0]:
                raise RuntimeError(
                    f"compression witness failed at t index {i}: {own} vs {lamp[i, m]}"
                )
        out[f"w2_S4same_m{m}"] = same
        if (same < out[f"w2_S4own_m{m}"] - 1e-12).any():
            raise RuntimeError("static-span compression below the own-span width (impossible)")
    return out


def heldout_terms(stats: dict, ref, runs: np.ndarray, spans: list) -> dict:
    """Held-out empirical estimates for the evaluation replicas against the given NUTS runs."""
    from mcmc.paperexp import german_credit_r2 as r2

    terms = r2.law_reference_terms(stats, ref, runs)
    ev = r2.evaluate_spans(terms, spans)
    r = terms["delta"].size
    full, jg, jr = r2.projection_jackknife(
        list(spans) + [np.eye(r)], terms["group_feat"], terms["run_feat"]
    )
    feat2, fjg, fjr = full[-1], jg[:, -1], jr[:, -1]
    pjg, pjr = jg[:, :-1], jr[:, :-1]
    if not np.allclose(full[:-1], ev["proj2"], rtol=1e-9, atol=1e-14):
        raise RuntimeError("projection estimates disagree")
    if abs(feat2 - (terms["feat_raw"] - terms["feat_bias"])) > 1e-12 + 1e-9 * terms["feat_raw"]:
        raise RuntimeError("feature discrepancy estimates disagree")
    mg, mr = terms["mmd2_jack_g"], terms["mmd2_jack_r"]
    jse = r2.jackknife_se
    return {
        "mmd2": terms["mmd2"],
        "mmd2_se": terms["mmd2_se"],
        "mmd2_loo_run": mr,
        "feat2": feat2,
        "feat2_se": float(jse(fjg, fjr)),
        "feat2_loo_run": fjr,
        "omit2": terms["mmd2"] - feat2,
        "omit2_se": float(jse(mg - fjg, mr - fjr)),
        "proj2": ev["proj2"],
        "proj2_se": jse(pjg, pjr),
        "res_feat2": feat2 - ev["proj2"],
        "res_feat2_se": jse(fjg[:, None] - pjg, fjr[:, None] - pjr),
        "res_full2": ev["res_full2"],
        "res_full2_se": ev["res_full2_se"],
    }


HELD_SCALARS = ("mmd2", "mmd2_se", "feat2", "feat2_se", "omit2", "omit2_se")
HELD_SPANS = ("proj2", "proj2_se", "res_feat2", "res_feat2_se", "res_full2", "res_full2_se")
SPAN_NAMES = ("V0_m8", "Vt_m8", "V0_m4", "Vt_m4")


def analyze_unit(model: str, sampler: str, device: str) -> tuple[dict, dict]:
    import torch
    from omegaconf import OmegaConf

    from mcmc.experiments import exp_r2_german_credit as r2x
    from mcmc.paperexp import german_credit_r2 as r2

    t_unit = time.time()
    d = unit_dirs(model, sampler)
    cfg = OmegaConf.merge(
        OmegaConf.create((d["r2"] / "config.yaml").read_text()), {"device": device}
    )
    ctx = r2x.Context(cfg, types.SimpleNamespace(path=d["r2"]))  # read-only; adopts saved basis
    ctx.load_frozen()
    ctx.ensure_ref()
    grid = ctx.grid
    n_t = len(grid)
    G, J = int(cfg.ensembles.G), int(cfg.ensembles.J)
    if G * R2_GROUP != J or J != N_REPLICAS:
        raise RuntimeError(f"unexpected ensemble layout G={G}, J={J}")
    labels = ctx.ref.run_labels
    sets = [str(s) for s in cfg.references[model].evaluation.keys()]
    run_idx = {s: np.array([labels.index(f"{s}:{q}") for q in range(N_RUNS_SET)]) for s in sets}
    if sorted(np.concatenate(list(run_idx.values())).tolist()) != list(range(len(labels))):
        raise RuntimeError("evaluation run labels do not partition into the declared sets")
    ops = operator_widths(np.load(d["r2"] / "sigma.npz"), grid)
    V0 = ops["V0"]
    spans_t = [[V0[:, :8], ops["Vt"][i][:, :8], V0[:, :4], ops["Vt"][i][:, :4]] for i in range(n_t)]
    r = ctx.feats.rank
    res = {"grid": np.asarray(grid), "sets": np.array(sets)}
    res.update({f"op__{k}": v for k, v in ops.items()})
    info = {
        "unit": f"{model}__{sampler}",
        "sets": sets,
        "run_idx": {s: v.tolist() for s, v in run_idx.items()},
        "feature_rank": r,
        "laws": {},
    }
    q = list(cfg.ensembles.quantiles)
    ns = len(sets)
    for law in LAWS:
        t_law = time.time()
        saved = np.load(d["r2"] / f"ens_{law}.npz", allow_pickle=False)
        f2 = d["r2"] / f"ens_{law}_states.npz"
        states = np.load(f2 if f2.exists() else d["r1_states"] / f"states_{law}.npz")["states"]
        if states.shape[:2] != (n_t, J):
            raise RuntimeError(f"{law}: states shape {states.shape}")
        coordV0 = np.empty((ns, 2, n_t, N_GROUPS_TOTAL, 8))
        coordVt = np.empty_like(coordV0)
        fn2 = np.empty((ns, 2, n_t, N_GROUPS_TOTAL))
        held = {k: np.empty((ns, 2, n_t)) for k in HELD_SCALARS}
        held.update({k: np.empty((ns, 2, n_t, len(SPAN_NAMES))) for k in HELD_SPANS})
        held["mmd2_loo_run"] = np.empty((ns, 2, n_t, HELD_RUNS.size))
        held["feat2_loo_run"] = np.empty((ns, 2, n_t, HELD_RUNS.size))
        mmd2_all_runs = np.empty((ns, n_t))
        check = {k: np.empty(n_t) for k in SUMMARY_KEYS}
        for i in range(n_t):
            x = torch.as_tensor(states[i], dtype=torch.float64, device=device)
            st = r2.checkpoint_stats(x, ctx.kernel, ctx.feats, ctx.ref, G, ctx.mem, q)
            for k in SUMMARY_KEYS:
                a, b = st[k], saved[k][i]
                check[k][i] = float(np.abs(a - b).max() / np.abs(b).max())
                if not check[k][i] <= CHECK_RTOL:
                    raise RuntimeError(
                        f"{law} t={grid[i]}: recomputed {k} differs (rel {check[k][i]:.2e})"
                    )
            F = ctx.feats(x).cpu().numpy()
            if np.abs(F.reshape(G, R2_GROUP, r).mean(1) - st["group_feat"]).max() > 1e-12:
                raise RuntimeError("replica features inconsistent with group features")
            Fo = F[:N_OBS]
            gm = np.concatenate([Fo.reshape(-1, Jg, r).mean(1) for Jg in J_GRID])
            stats_eval = {
                "group_feat": st["group_feat"][EVAL_GROUPS],
                "group_pair_ksum": st["group_pair_ksum"][np.ix_(EVAL_GROUPS, EVAL_GROUPS)],
                "group_run_cross": st["group_run_cross"][EVAL_GROUPS],
                "n_per_group": R2_GROUP,
            }
            for si, s in enumerate(sets):
                all_runs = run_idx[s]
                mmd2_all_runs[si, i] = r2.mmd2_unbiased(
                    st["group_pair_ksum"],
                    st["group_run_cross"],
                    ctx.ref.run_inner,
                    R2_GROUP,
                    np.arange(G),
                    all_runs,
                )
                for pi, split in enumerate(SPLITS):
                    rd, hd = (
                        (READ_RUNS, HELD_RUNS) if split == "heldout" else (HELD_RUNS, READ_RUNS)
                    )
                    read_idx, held_idx = all_runs[rd], all_runs[hd]
                    if np.intersect1d(read_idx, held_idx).size:
                        raise AssertionError("reading and held-out runs overlap")
                    rg = gm - ctx.ref.feat_run_means[read_idx].mean(0)
                    coordV0[si, pi, i] = rg @ V0[:, :8]
                    coordVt[si, pi, i] = rg @ ops["Vt"][i][:, :8]
                    fn2[si, pi, i] = (rg * rg).sum(1)
                    h = heldout_terms(stats_eval, ctx.ref, held_idx, spans_t[i])
                    for k in HELD_SCALARS:
                        held[k][si, pi, i] = h[k]
                    for k in HELD_SPANS:
                        held[k][si, pi, i] = h[k]
                    held["mmd2_loo_run"][si, pi, i] = h["mmd2_loo_run"]
                    held["feat2_loo_run"][si, pi, i] = h["feat2_loo_run"]
        # the saved eval2 full-ensemble MMD^2 (all 32 groups, all runs of the set) must be reproduced
        ev = np.load(d["eval2"] / "eval.npz", allow_pickle=False)
        li = list(ev["laws"]).index(law)
        choices = [str(c) for c in ev["choices"]]
        for si, s in enumerate(sets):
            ch = "all" if s == "primary" else f"{s}_all"
            ref_v = ev["mmd2"][li, :, choices.index(ch)]
            if np.abs(mmd2_all_runs[si] - ref_v).max() > 1e-12 + 1e-9 * np.abs(ref_v).max():
                raise RuntimeError(f"{law}/{s}: full-ensemble MMD^2 differs from eval2")
        acc = saved["accept_per_chain"]
        pre = f"{law}__"
        res.update(
            {
                pre + "coordV0": coordV0,
                pre + "coordVt": coordVt,
                pre + "featnorm2": fn2,
                pre + "mmd2_full_ensemble": mmd2_all_runs,
                pre + "never_accepted": acc == 0,
            }
        )
        res.update({pre + "held__" + k: v for k, v in held.items()})
        res.update({pre + "check__" + k: v for k, v in check.items()})
        info["laws"][law] = {
            "check_max_rel": {k: float(v.max()) for k, v in check.items()},
            "never_accepted_obs": int((acc[:N_OBS] == 0).sum()),
            "never_accepted_eval": int((acc[N_OBS:] == 0).sum()),
            "wall_s": time.time() - t_law,
            "kernel_evals": int(
                n_t * J * (J + sum(X.shape[0] for X in ctx.ref.X) + ctx.feats.n_landmarks)
            ),
        }
        print(
            f"[analyze] {model}__{sampler} {law}: max rel "
            f"{max(info['laws'][law]['check_max_rel'].values()):.1e}, {info['laws'][law]['wall_s']:.0f} s",
            flush=True,
        )
    info["wall_s"] = time.time() - t_unit
    info["device"] = device
    return res, info


def cmd_analyze(args) -> None:
    import torch

    from mcmc.paperexp.runio import RunDir

    torch.set_num_threads(args.threads)
    des, man, hashes = verify_frozen()
    units = [u for u in man["units"] if not args.units or u in args.units.split(",")]
    run = RunDir.create(
        STUDY_C,
        args.phase,
        args.run_id,
        json.dumps(des, indent=2),
        requested_units=len(units),
        allow_dirty=args.phase in ("smoke", "pilot"),
        extra_provenance=hashes,
    )
    try:
        for k, unit in enumerate(units):
            verify_sources(man, unit)
            model, sampler = unit.split("__")
            res, info = analyze_unit(model, sampler, args.device)
            np.savez_compressed(run.path / f"{unit}.npz", **res)
            write_json(run.path / f"{unit}.json", info)
            run.update(completed_units=k + 1)
        run.finish("COMPLETED")
    except BaseException as exc:
        run.finish("FAILED", note=repr(exc)[:500])
        raise
    print(run.path)


# Plug-in curves, crossings and conventional diagnostics
STATUS = {1: "ABOVE", 0: "UNRESOLVED", -1: "BELOW"}


def status_code(est, se, eps: float) -> np.ndarray:
    """+1 resolved above eps^2, -1 resolved below, 0 unresolved (pointwise +-Z_HELD SE)."""
    est, se = np.asarray(est), np.asarray(se)
    if (se < 0).any() or not np.isfinite(est).all() or not np.isfinite(se).all():
        raise ValueError("bad held-out estimate or SE")
    return np.where(est - Z_HELD * se > eps**2, 1, np.where(est + Z_HELD * se < eps**2, -1, 0))


def interval_flag(est, se) -> np.ndarray:
    """POSITIVE (lo > 0), CONTAINS_ZERO (lo <= 0 <= hi) or NEGATIVE_INTERVAL (hi < 0) for the
    signed +-Z SE interval of a squared estimate."""
    est, se = np.asarray(est), np.asarray(se)
    lo, hi = est - Z_HELD * se, est + Z_HELD * se
    return np.where(lo > 0, "POSITIVE", np.where(hi < 0, "NEGATIVE_INTERVAL", "CONTAINS_ZERO"))


def mmd_scale_interval(est, se) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a squared estimate and its +-Z SE interval to the MMD scale.

    Nonpositive point estimates become NaN. Intervals wholly below zero have
    NaN endpoints; intervals containing zero have lower endpoint zero.
    """
    est, se = np.asarray(est), np.asarray(se)
    lo, hi = est - Z_HELD * se, est + Z_HELD * se
    with np.errstate(invalid="ignore"):
        point = np.where(est > 0, np.sqrt(np.clip(est, 0, None)), np.nan)
        neg = hi < 0
        lo_m = np.where(neg, np.nan, np.sqrt(np.clip(lo, 0, None)))
        hi_m = np.where(neg, np.nan, np.sqrt(np.clip(hi, 0, None)))
    return point, lo_m, hi_m


def readings_h2(z, law: str, si: int, pi: int) -> dict:
    """Squared readings ||V^T r_g||^2, (n_t, N_GROUPS_TOTAL), per (family, m)."""
    cV0 = z[f"{law}__coordV0"][si, pi]
    cVt = z[f"{law}__coordVt"][si, pi]
    out = {}
    for m in M_LIST:
        h0 = (cV0[..., :m] ** 2).sum(-1)
        out[("S3", m)] = h0
        out[("S4same", m)] = h0
        out[("S4own", m)] = (cVt[..., :m] ** 2).sum(-1)
    return out


def width2(z, family: str, m: int, margin: str) -> np.ndarray:
    w = z[f"op__w2_{family}_m{m}"].copy()
    if margin == "halfsplit":
        eps = z["op__eps"]
        w = w + (eps[0] if family == "S3" else eps)
    elif margin != "none":
        raise ValueError(margin)
    return w


def plugin(h2: np.ndarray, w2: np.ndarray, R: float) -> np.ndarray:
    """Reference-based plug-in at an assumed radius R (feature space): sqrt(h^2 + R^2 w^2)."""
    if (h2 < 0).any() or (w2 < 0).any():
        raise ValueError("negative squared reading or width")
    return np.sqrt(h2 + R**2 * w2[:, None])


def first_crossing(P: np.ndarray, eps: float) -> np.ndarray:
    """Index of the first inspection with P <= eps for each group (column); -1 if never."""
    le = P <= eps
    return np.where(le.any(0), le.argmax(0), -1)


def stuck_groups(never: np.ndarray, J: int) -> np.ndarray:
    return never[:N_OBS].reshape(-1, J).any(1)


def unit_meta(model: str, sampler: str) -> dict:
    d = unit_dirs(model, sampler)
    cfg = load_yaml(d["r2"] / "config.yaml")
    fr = cfg["frozen"][f"{model}__{sampler}"]
    if int(cfg["sigma"]["N"]) != 8192:
        raise RuntimeError("operator block count differs from the declared N = 8192")
    return {
        "T": int(fr["T"]),
        "t_star": int(fr["t_star"]),
        "cap_binds": bool(fr["cap_binds"]),
        "h": float(fr["h"]),
    }


_SWAP_CACHE: dict = {}


def _jk_var(x: np.ndarray) -> float:
    n = x.shape[0]
    return float((n - 1) / n * ((x - x.mean()) ** 2).sum())


def paired_swap_from_stats(S, cross, inner, runs_h, runs_s) -> tuple[float, float, float, float]:
    """(d_H - d_S, paired jackknife SE, d_H, d_S) for two U-statistic MMD^2 estimates sharing the
    same replica groups (S, cross) but using disjoint reference runs."""
    from mcmc.paperexp import german_credit_r2 as r2

    eH, gH, rH = r2.mmd2_jackknife(S, cross, inner, R2_GROUP, runs_h)
    eS, gS, rS = r2.mmd2_jackknife(S, cross, inner, R2_GROUP, runs_s)
    if np.intersect1d(runs_h, runs_s).size:
        raise ValueError("the two run sets must be disjoint")
    return eH - eS, math.sqrt(_jk_var(gH - gS) + _jk_var(rH) + _jk_var(rS)), eH, eS


def swap_paired(z, model: str, sampler: str, s: str, si: int, law: str) -> dict:
    """Compare disjoint NUTS reference halves using the same evaluation replicas.

    The paired jackknife variance combines delete-one-group differences with
    the separate delete-one-run variances. Check against the saved estimates.
    """
    key = (model, sampler, s, law)
    if key in _SWAP_CACHE:
        return _SWAP_CACHE[key]
    d = unit_dirs(model, sampler)
    ens = np.load(d["r2"] / f"ens_{law}.npz", allow_pickle=False)
    setup = np.load(d["r2"] / "setup.npz", allow_pickle=False)
    labels = json.loads(str(setup["run_labels"]))
    runs = np.array([labels.index(f"{s}:{q}") for q in range(N_RUNS_SET)])
    inner = setup["run_inner"]
    n_t = len(z["grid"])
    diff, se = np.empty(n_t), np.empty(n_t)
    for i in range(n_t):
        S = ens["group_pair_ksum"][i][np.ix_(EVAL_GROUPS, EVAL_GROUPS)]
        cross = ens["group_run_cross"][i][EVAL_GROUPS]
        diff[i], se[i], eH, eS = paired_swap_from_stats(
            S, cross, inner, runs[HELD_RUNS], runs[READ_RUNS]
        )
        for e, pi in ((eH, 0), (eS, 1)):
            stored = z[f"{law}__held__mmd2"][si, pi, i]
            if abs(e - stored) > 1e-12 + 1e-9 * abs(stored):
                raise RuntimeError(
                    f"swap recomputation differs from the stored estimate ({e} vs {stored})"
                )
    _SWAP_CACHE[key] = {"diff": diff, "se": se}
    return _SWAP_CACHE[key]


def summarize_unit(z, info: dict, model: str, sampler: str) -> dict[str, list]:
    grid = z["grid"].tolist()
    n_t = len(grid)
    sets = [str(s) for s in z["sets"]]
    meta = unit_meta(model, sampler)
    off = group_offsets()
    rows = {"plugin": [], "cross": [], "held": [], "ops": [], "conv": [], "results": []}
    base_u = {"model": model, "dataset": DISPLAY[model], "sampler": sampler}
    e_tN = operator_radius_unit(n_t)
    for i, t in enumerate(grid):
        row = {
            **base_u,
            "t": t,
            "eps_halfsplit": z["op__eps"][i],
            "e_tN_protocol": e_tN,
            "lam1_pos": z["op__lam1_pos"][i],
            "most_negative": z["op__most_negative"][i],
            "n_pos": int(z["op__n_pos"][i]),
            "basis_orth_err": z["op__orth_err"][i],
        }
        for m in M_LIST:
            row[f"lam_{m + 1}"] = z["op__lam"][i, m]
            row[f"op_resolution_m{m}"] = z["op__lam"][i, m] / z["op__eps"][i]
            for fam in FAMILIES:
                row[f"w2_{fam}_m{m}"] = z[f"op__w2_{fam}_m{m}"][i]
        for R in R_ASSUMED:
            row[f"no_obs_plugin_R{R:g}"] = R * math.sqrt(z["op__lam1_pos"][i])
        rows["ops"].append(row)
    d = unit_dirs(model, sampler)
    sig_cost = json.loads((d["r2"] / "sigma.json").read_text())
    swaps = {
        (s, law): swap_paired(z, model, sampler, s, si, law)
        for si, s in enumerate(sets)
        for law in LAWS
    }
    for si, s in enumerate(sets):
        for law in LAWS:
            pre = f"{law}__"
            never = z[pre + "never_accepted"]
            # held-out table
            for pi, split in enumerate(SPLITS):
                H = {k: z[pre + "held__" + k][si, pi] for k in HELD_SCALARS + HELD_SPANS}
                loo = z[pre + "held__mmd2_loo_run"][si, pi]
                for i, t in enumerate(grid):
                    row = {
                        **base_u,
                        "reference_set": s,
                        "split": split,
                        "law": law,
                        "t": t,
                        "status": "empirical estimate (reference-based, held-out)",
                    }
                    for k in HELD_SCALARS:
                        row[k] = H[k][i]
                    row["mmd2_z"] = H["mmd2"][i] / H["mmd2_se"][i]
                    row["mmd2_loo_run_min"], row["mmd2_loo_run_max"] = loo[i].min(), loo[i].max()
                    row["mmd2_full_ensemble_all_runs"] = z[pre + "mmd2_full_ensemble"][si, i]
                    p, lo, hi = mmd_scale_interval(H["mmd2"][i], H["mmd2_se"][i])
                    row["mmd_point"], row["mmd_lo"], row["mmd_hi"] = float(p), float(lo), float(hi)
                    row["mmd_interval_flag"] = str(interval_flag(H["mmd2"][i], H["mmd2_se"][i]))
                    p, lo, hi = mmd_scale_interval(H["feat2"][i], H["feat2_se"][i])
                    row["feat_point"], row["feat_lo"], row["feat_hi"] = (
                        float(p),
                        float(lo),
                        float(hi),
                    )
                    row["feat_interval_flag"] = str(interval_flag(H["feat2"][i], H["feat2_se"][i]))
                    sw = swaps[(s, law)]
                    row["mmd2_swap_diff_heldout_minus_swap"] = sw["diff"][i]
                    row["mmd2_swap_diff_paired_se"] = sw["se"][i]
                    for eps in EPS:
                        row[f"status_mmd_eps{eps:g}"] = STATUS[
                            int(status_code(H["mmd2"][i], H["mmd2_se"][i], eps))
                        ]
                        row[f"status_feat_eps{eps:g}"] = STATUS[
                            int(status_code(H["feat2"][i], H["feat2_se"][i], eps))
                        ]
                    for j, sp in enumerate(SPAN_NAMES):
                        for k in HELD_SPANS:
                            row[f"{k}__{sp}"] = H[k][i, j]
                    rows["held"].append(row)
            # plug-ins and crossings
            for pi, split in enumerate(SPLITS):
                H = {
                    k: z[pre + "held__" + k][si, pi]
                    for k in ("mmd2", "mmd2_se", "feat2", "feat2_se")
                }
                st_full = {eps: status_code(H["mmd2"], H["mmd2_se"], eps) for eps in EPS}
                st_feat = {eps: status_code(H["feat2"], H["feat2_se"], eps) for eps in EPS}
                _, feat_lo, _ = mmd_scale_interval(H["feat2"], H["feat2_se"])
                _, full_lo, _ = mmd_scale_interval(H["mmd2"], H["mmd2_se"])
                h2 = readings_h2(z, law, si, pi)
                for (fam, m), hh in h2.items():
                    for margin in MARGINS:
                        w2 = width2(z, fam, m, margin)
                        for R in R_ASSUMED:
                            P_all = plugin(hh, w2, R)
                            for J in J_GRID:
                                o, n = off[J]
                                P = P_all[:, o : o + n]
                                stuck = stuck_groups(never, J)
                                if split == "heldout":
                                    qs = np.quantile(P, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
                                    for i, t in enumerate(grid):
                                        rows["plugin"].append(
                                            {
                                                **base_u,
                                                "reference_set": s,
                                                "law": law,
                                                "J": J,
                                                "family": fam,
                                                "m": m,
                                                "margin": margin,
                                                "R": R,
                                                "t": t,
                                                "q05": qs[0, i],
                                                "q25": qs[1, i],
                                                "q50": qs[2, i],
                                                "q75": qs[3, i],
                                                "q95": qs[4, i],
                                                "frac_le_0.1": float((P[i] <= 0.10).mean()),
                                                "frac_le_0.05": float((P[i] <= 0.05).mean()),
                                                "frac_below_heldout_feat_lo": float(
                                                    (P[i] < feat_lo[i]).mean()
                                                ),
                                                "frac_below_heldout_mmd_lo": float(
                                                    (P[i] < full_lo[i]).mean()
                                                ),
                                                "label": "reference-based plug-in at an assumed radius (feature space)",
                                            }
                                        )
                                for eps in EPS:
                                    fc = first_crossing(P, eps)
                                    crossed = fc >= 0
                                    ts = np.asarray(grid)[fc[crossed]]
                                    sf = st_full[eps][fc[crossed]]
                                    se_ = st_feat[eps][fc[crossed]]
                                    counts = {
                                        str(t): int((ts == t).sum())
                                        for t in grid
                                        if (ts == t).any()
                                    }
                                    rows["cross"].append(
                                        {
                                            **base_u,
                                            "reference_set": s,
                                            "split": split,
                                            "law": law,
                                            "J": J,
                                            "family": fam,
                                            "m": m,
                                            "margin": margin,
                                            "R": R,
                                            "eps": eps,
                                            "n_groups": n,
                                            "n_crossed": int(crossed.sum()),
                                            "frac_crossed": float(crossed.mean()),
                                            "t_first_median": float(np.median(ts))
                                            if ts.size
                                            else np.nan,
                                            "t_first_min": int(ts.min()) if ts.size else -1,
                                            "first_t_counts": json.dumps(counts),
                                            "stop_mmd_ABOVE": int((sf == 1).sum()),
                                            "stop_mmd_UNRESOLVED": int((sf == 0).sum()),
                                            "stop_mmd_BELOW": int((sf == -1).sum()),
                                            "stop_feat_ABOVE": int((se_ == 1).sum()),
                                            "stop_feat_UNRESOLVED": int((se_ == 0).sum()),
                                            "stop_feat_BELOW": int((se_ == -1).sum()),
                                            "n_groups_with_stuck": int(stuck.sum()),
                                            "n_crossed_with_stuck": int((crossed & stuck).sum()),
                                            "label": "reference-based plug-in at an assumed radius (feature space)",
                                        }
                                    )
            # matched-information conventional join (appendix)
            conv = np.load(d["r2"] / f"conv_{law}.npz", allow_pickle=False)
            ct = conv["conv_t"].tolist()
            ok = (
                (conv["conv_rhat_max"] < CONV_RULE["rhat_lt"])
                & (conv["conv_ess_bulk_min"] >= CONV_RULE["ess_bulk_ge"])
                & (conv["conv_ess_tail_min"] >= CONV_RULE["ess_tail_ge"])
            )
            if ok.shape[1] != 16:
                raise RuntimeError("expected 16 conventional four-chain groups")
            hp = readings_h2(z, law, si, 0)[(PRIMARY["family"], PRIMARY["m"])][:, :16]
            Pp = plugin(
                hp, width2(z, PRIMARY["family"], PRIMARY["m"], PRIMARY["margin"]), PRIMARY["R"]
            )
            stuck4 = stuck_groups(never, 4)[:16]
            Hh = {k: z[pre + "held__" + k][si, 0] for k in ("mmd2", "mmd2_se", "feat2", "feat2_se")}
            for c, t in enumerate(ct):
                i = grid.index(t)
                pl = Pp[i] <= 0.10
                cv = ok[c]
                rows["conv"].append(
                    {
                        **base_u,
                        "reference_set": s,
                        "law": law,
                        "t": t,
                        "n_groups": 16,
                        "n_conv_pass": int(cv.sum()),
                        "n_plugin_le_0.1": int(pl.sum()),
                        "n_both": int((cv & pl).sum()),
                        "n_conv_only": int((cv & ~pl).sum()),
                        "n_plugin_only": int((~cv & pl).sum()),
                        "n_neither": int((~cv & ~pl).sum()),
                        "n_stuck_groups": int(stuck4.sum()),
                        "n_conv_pass_stuck": int((cv & stuck4).sum()),
                        "n_plugin_le_stuck": int((pl & stuck4).sum()),
                        "heldout_mmd_status_0.1": STATUS[
                            int(status_code(Hh["mmd2"][i], Hh["mmd2_se"][i], 0.10))
                        ],
                        "heldout_feat_status_0.1": STATUS[
                            int(status_code(Hh["feat2"][i], Hh["feat2_se"][i], 0.10))
                        ],
                        "plugin": f"{PRIMARY['family']} m={PRIMARY['m']} R={PRIMARY['R']:g} margin={PRIMARY['margin']}",
                        "status": "appendix, descriptive (same 4 chains; conventional uses the path window (t/2, t])",
                    }
                )
            # results-table row
            rows["results"].append(
                results_row(z, info, model, sampler, s, si, law, meta, sig_cost, rows)
            )
    return rows


def operator_radius_unit(n_t: int) -> float:
    from mcmc.paperexp.operators import operator_radius

    return operator_radius(KAPPA**2, 8192, BETA["operator"], 1.0 / n_t)


def results_row(z, info, model, sampler, s, si, law, meta, sig_cost, rows) -> dict:
    grid = z["grid"].tolist()
    n_t = len(grid)
    pre = f"{law}__"
    H = {k: z[pre + "held__" + k][si, 0] for k in HELD_SCALARS}
    Hs = {k: z[pre + "held__" + k][si, 1] for k in ("mmd2", "mmd2_se")}
    never = z[pre + "never_accepted"]
    role = (
        "appendix sensitivity"
        if model == "horseshoe"
        else "primary practical start"
        if law in ("I1", "I2")
        else "control"
    )
    row = {
        "model": model,
        "dataset": DISPLAY[model],
        "sampler": sampler,
        "reference_set": s,
        "law": law,
        "role": role,
        "n_inspections": n_t,
        "horizon": grid[-1],
        "t_star": meta["t_star"],
        "T_cap_binds": meta["cap_binds"],
        "never_accepted_frac_obs": float(never[:N_OBS].mean()),
        "never_accepted_frac_eval": float(never[N_OBS:].mean()),
    }
    for eps in EPS:
        stm = status_code(H["mmd2"], H["mmd2_se"], eps)
        stf = status_code(H["feat2"], H["feat2_se"], eps)
        below = np.flatnonzero(stm == -1)
        row[f"heldout_first_t_mmd_BELOW_{eps:g}"] = grid[below[0]] if below.size else "never"
        row[f"heldout_n_t_mmd_ABOVE_{eps:g}"] = int((stm == 1).sum())
        row[f"heldout_n_t_mmd_UNRESOLVED_{eps:g}"] = int((stm == 0).sum())
        row[f"heldout_n_t_mmd_BELOW_{eps:g}"] = int((stm == -1).sum())
        row[f"heldout_last_status_mmd_{eps:g}"] = STATUS[int(stm[-1])]
        row[f"heldout_last_status_feat_{eps:g}"] = STATUS[int(stf[-1])]
    row["heldout_mmd2_t0"], row["heldout_mmd2_se_t0"] = H["mmd2"][0], H["mmd2_se"][0]
    row["heldout_mmd2_last"], row["heldout_mmd2_se_last"] = H["mmd2"][-1], H["mmd2_se"][-1]
    row["heldout_feat2_last"], row["heldout_omit2_last"] = H["feat2"][-1], H["omit2"][-1]
    loo = z[pre + "held__mmd2_loo_run"][si, 0][-1]
    row["heldout_mmd2_loo_run_range_last"] = float(loo.max() - loo.min())
    sw = swap_paired(z, model, sampler, s, si, law)
    dz = np.abs(sw["diff"]) / sw["se"]
    row["swap_max_abs_diff_in_paired_se"] = float(dz.max())
    row["swap_n_t_above_2_paired_se"] = int((dz > 2.0).sum())
    row["swap_status_agree_0.1"] = int(
        (
            status_code(H["mmd2"], H["mmd2_se"], 0.1) == status_code(Hs["mmd2"], Hs["mmd2_se"], 0.1)
        ).sum()
    )
    for J in J_GRID:
        row[f"floor_b_J{J}"] = replica_floor(J, n_t, len(J_GRID))
    unit_rows = [
        r
        for r in rows["cross"]
        if r["reference_set"] == s
        and r["law"] == law
        and r["split"] == "heldout"
        and r["family"] == PRIMARY["family"]
        and r["m"] == PRIMARY["m"]
        and r["margin"] == PRIMARY["margin"]
    ]
    for J in J_GRID:
        for R in (1.0, 4.0) if J == 64 else (1.0,):
            for eps in (0.10,):
                (c,) = [r for r in unit_rows if r["J"] == J and r["R"] == R and r["eps"] == eps]
                tag = f"plugin_R{R:g}_J{J}_eps{eps:g}"
                row[f"{tag}_frac_crossed"] = c["frac_crossed"]
                row[f"{tag}_t_first_median"] = c["t_first_median"]
                row[f"{tag}_stop_mmd_ABOVE"] = c["stop_mmd_ABOVE"]
                row[f"{tag}_stop_mmd_UNRESOLVED"] = c["stop_mmd_UNRESOLVED"]
                row[f"{tag}_n_crossed_with_stuck"] = c["n_crossed_with_stuck"]
    ts = grid.index(meta["t_star"])
    for m in M_LIST:
        row[f"op_resolution_m{m}_tstar"] = float(z["op__lam"][ts, m] / z["op__eps"][ts])
    lawinfo = info["laws"][law]
    ens = json.loads((unit_dirs(model, sampler)["r2"] / f"ens_{law}.json").read_text())
    row["cost_original_trajectory_wall_s"] = ens["wall_s"]
    row["cost_original_trajectory_logprob_evals"] = ens["counters"]["logprob_evals"]
    row["cost_original_trajectory_grad_evals"] = ens["counters"]["grad_evals"]
    row["cost_original_operator_wall_s_shared"] = sig_cost["wall_s"]
    row["cost_original_operator_logprob_evals_shared"] = sig_cost["counters"]["logprob_evals"]
    row["cost_original_operator_grad_evals_shared"] = sig_cost["counters"]["grad_evals"]
    row["cost_original_reference_core_s_shared"] = REF_COST[(model, s)]["core_s"]
    row["cost_original_reference_grad_evals_shared"] = REF_COST[(model, s)]["grad_evals"]
    row["cost_incremental_analysis_wall_s"] = lawinfo["wall_s"]
    row["cost_incremental_kernel_evals"] = lawinfo["kernel_evals"]
    row["check_max_rel"] = max(lawinfo["check_max_rel"].values())
    row["guarantee"] = "none: confidence floor b_J >= 0.607 > 0.10 at every J (floor_table.csv)"
    return row


REF_COST: dict = {}


def reference_costs() -> None:
    from mcmc.paperexp.german_credit_reference import set_path

    for model in MODELS:
        cfg = load_yaml(unit_dirs(model, "mala")["r2"] / "config.yaml")
        for s, rid in cfg["references"][model]["evaluation"].items():
            path = set_path("reference", rid)
            summ = [json.loads(p.read_text()) for p in sorted(path.glob("unit_*.summary.json"))]
            REF_COST[(model, str(s))] = {
                "core_s": float(sum(u["t_total_s"] for u in summ)),
                "grad_evals": int(
                    sum(u["grad_evals_warmup"] + u["grad_evals_sample"] for u in summ)
                ),
            }


def write_df(rows: list[dict], path: pathlib.Path) -> None:
    import pandas as pd

    if not rows:
        raise RuntimeError(f"no rows for {path}")
    df = pd.DataFrame(rows)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    if path.suffix == ".gz":
        import gzip

        text = df.to_csv(index=False, float_format="%.10g").encode()
        with (
            open(tmp, "wb") as fh,
            gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz,
        ):
            gz.write(text)  # byte-deterministic: no file name, no time stamp in the header
    else:
        df.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)


def compact_results(rows: list[dict]) -> list[dict]:
    """One line per model/sampler/start/reference set: the columns a paper table needs."""
    out = []
    for r in rows:
        m2, se = r["heldout_mmd2_last"], r["heldout_mmd2_se_last"]
        c = {
            "dataset": r["dataset"],
            "sampler": r["sampler"],
            "law": r["law"],
            "reference_set": r["reference_set"],
            "role": r["role"],
            "never_accepted_frac": round(r["never_accepted_frac_eval"], 4),
            "horizon": r["horizon"],
            "heldout_mmd2_last": m2,
            "heldout_mmd2_se_last": se,
            "heldout_mmd_last": math.sqrt(m2) if m2 > 0 else "<=0 (unresolved from 0)",
            "heldout_status_last_0.1": r["heldout_last_status_mmd_0.1"],
            "heldout_first_t_below_0.1": r["heldout_first_t_mmd_BELOW_0.1"],
            "heldout_first_t_below_0.05": r["heldout_first_t_mmd_BELOW_0.05"],
            "confidence_floor_J4_J16_J64": "/".join(f"{r[f'floor_b_J{J}']:.2f}" for J in J_GRID),
        }
        for J in J_GRID:
            c[f"plugin_R1_J{J}_frac_crossed_0.1"] = r[f"plugin_R1_J{J}_eps0.1_frac_crossed"]
        for k in ("stop_mmd_ABOVE", "stop_mmd_UNRESOLVED", "n_crossed_with_stuck"):
            c[f"plugin_R1_J64_{k}"] = r[f"plugin_R1_J64_eps0.1_{k}"]
        c["plugin_R4_J64_frac_crossed_0.1"] = r["plugin_R4_J64_eps0.1_frac_crossed"]
        c["operator_resolution_m8_tstar"] = round(r["op_resolution_m8_tstar"], 3)
        c["swap_max_abs_diff_in_paired_se"] = round(r["swap_max_abs_diff_in_paired_se"], 3)
        c["swap_n_t_above_2_paired_se"] = r["swap_n_t_above_2_paired_se"]
        c["cost_original_trajectory_wall_s"] = round(r["cost_original_trajectory_wall_s"], 1)
        c["cost_original_operator_wall_s_shared"] = round(
            r["cost_original_operator_wall_s_shared"], 1
        )
        c["cost_original_reference_core_s_shared"] = round(
            r["cost_original_reference_core_s_shared"], 1
        )
        c["cost_incremental_analysis_wall_s"] = round(r["cost_incremental_analysis_wall_s"], 1)
        c["guarantee"] = "none (floor table)"
        c["plugin_label"] = "reference-based plug-in at an assumed radius, feature space"
        out.append(c)
    return out


def cmd_summarize(args) -> None:
    t0 = time.time()
    run_path = out_root() / STUDY_C / args.phase / args.run_id
    st = json.loads((run_path / "RUN-STATUS.json").read_text())
    if st["status"] != "COMPLETED":
        raise RuntimeError(f"{run_path} is {st['status']}")
    reference_costs()
    allrows: dict[str, list] = {}
    for f in sorted(run_path.glob("*__*.npz")):
        unit = f.stem
        model, sampler = unit.split("__")
        z = dict(np.load(f, allow_pickle=False))
        info = json.loads((run_path / f"{unit}.json").read_text())
        rows = summarize_unit(z, info, model, sampler)
        for k, v in rows.items():
            allrows.setdefault(k, []).extend(v)
        print(f"[summarize] {unit}", flush=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_df(allrows["held"], out / "heldout.csv.gz")
    write_df(allrows["ops"], out / "operators.csv")
    write_df(allrows["plugin"], out / "plugin_summary.csv.gz")
    write_df(allrows["cross"], out / "crossings.csv.gz")
    write_df(allrows["conv"], out / "conventional_join.csv")
    write_df(allrows["results"], out / "results_table.csv")
    write_df(compact_results(allrows["results"]), out / "results_compact.csv")
    print(f"[summarize] done in {time.time() - t0:.0f} s")


# Regression comparisons from the saved summaries
REG_FILES = {
    "concrete": "r3_concrete/summary_main.json",
    "kin8nm": "r3_regression/summary_kin8nm.json",
    "airfoil": "r3_regression/summary_airfoil.json",
    "wine_red": "r3_regression/summary_wine_red.json",
    "energy": "r3_regression/summary_energy.json",
}
REG_METHODS = {
    # family: (description, operator, uses observations)
    "B3b": ("Section 3 bound, static span", "estimated", True),
    "B4b_same": ("Section 4 bound, Section 3's span", "estimated", True),
    "B4b_own": ("Section 4 bound, own span", "estimated", True),
    "NO3b": ("no-observation bound R sqrt(lambda_1(Sigma))", "estimated", False),
    "NO4b": ("no-observation bound R sqrt(lambda_1(Sigma_t))", "estimated", False),
    "B3a": ("Section 3 bound, static span", "analytic", True),
    "B4a_same": ("Section 4 bound, Section 3's span", "analytic", True),
    "B4a_own": ("Section 4 bound, own span", "analytic", True),
    "NO3a": ("no-observation bound R sqrt(lambda_1(Sigma))", "analytic", False),
    "NO4a": ("no-observation bound R sqrt(lambda_1(Sigma_t))", "analytic", False),
    "direct": ("direct bound ||delta_hat|| + eta (no operator, no radius)", "none", True),
}


def parse_method(key: str) -> tuple[str, int | None]:
    if "_N" in key:
        fam, n = key.rsplit("_N", 1)
        return fam, int(n)
    return key, None


def regression_settings(d: dict) -> list[dict]:
    """Setting index -> direction (slow/fast, from the shift vs the declared eigenvectors) and R_init."""
    dirs = d["meta"]["directions"]
    out = []
    for s, (a, R) in enumerate(zip(d["setup"]["shifts"], d["setup"]["r_init"])):
        a = np.asarray(a)
        cos = {
            k: abs(a @ np.asarray(v["vec"])) / (np.linalg.norm(a) * np.linalg.norm(v["vec"]))
            for k, v in dirs.items()
        }
        (name,) = [k for k, c in cos.items() if c > 0.999]
        out.append(
            {
                "setting": s,
                "direction": name,
                "R_init": float(R),
                "label": f"{name} shift, R_init={R:g}",
            }
        )
    return out


def cmd_regression(args) -> None:
    n_checked = verify_regression_inputs(json.loads((OUT / "manifest.json").read_text()))
    print(f"[regression] {n_checked} input summaries match the manifest")
    root = REPO / "paper" / "figures" / "data"
    est_rows, appx_rows, cross_rows = [], [], []
    for ds, summ in REG_FILES.items():
        d = json.loads((root / summ).read_text())
        setup = d["setup"]
        cps = setup["checkpoints"]
        sets = regression_settings(d)
        n_pi = int(d["design"]["N_pi"])
        for key, cells in d["methods"].items():
            fam, N = parse_method(key)
            desc, operator, obs = REG_METHODS[fam]
            for ck, c in cells.items():
                s, J, m, ref = ck.split("|")
                s, J, m = int(s), int(J), int(m)
                st = sets[s]
                base = {
                    "dataset": ds,
                    "summary_key": f"{key}|{ck}",
                    "method": key,
                    "family": fam,
                    "description": desc,
                    "operator": operator if N is None else f"{operator}, N={N} blocks",
                    "N_blocks": N if N is not None else 0,
                    "setting": st["label"],
                    "direction": st["direction"],
                    "R_init_known": st["R_init"],
                    "J_chains": J if obs else 0,
                    "m": m,
                    "target_reference": ref,
                    "N_pi_exact_target_draws": n_pi if ref == "sampled" else 0,
                    "n_ensembles": int(d["n_ensembles"]),
                }
                for i, t in enumerate(cps):
                    row = {
                        **base,
                        "t": t,
                        "B_median": c["B_median"][i],
                        "B_q05": c["B_q05"][i],
                        "B_q95": c["B_q95"][i],
                        "H_median": c["H_median"][i] if c.get("H_median") else np.nan,
                        "U_median": c["U_median"][i] if c.get("U_median") else np.nan,
                        "exact_mmd": setup["truth"][s][i],
                        "covered_all_t": c["covered_all_t"],
                        "cp_lo": c["cp"][0],
                        "cp_hi": c["cp"][1],
                        "min_margin": c["min_margin"],
                        "status": "confidence bound under supplied R_init and exact target draws"
                        if operator != "none"
                        else "confidence bound under exact target draws",
                    }
                    (est_rows if operator == "estimated" else appx_rows).append(row)
                for eps in EPS:
                    x = c[f"cross_{eps:g}"]
                    cross_rows.append(
                        {
                            **base,
                            "eps": eps,
                            "frac_crossed": x["fraction"],
                            "n_censored": x["n_censored"],
                            "t_median_crossed": x["t_median_crossed"],
                            "first_t_counts": json.dumps(x["t_values"]),
                        }
                    )
    write_df(est_rows, OUT / "regression_estimated.csv.gz")
    write_df(appx_rows, OUT / "regression_appendix_analytic_direct.csv.gz")
    write_df(cross_rows, OUT / "regression_crossings.csv.gz")
    print(len(est_rows), len(appx_rows), len(cross_rows))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--reproduce", default="hier:mala:I1:8", help="model:sampler:law:t ('' to skip)")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    sub.add_parser("floor")
    p = sub.add_parser("bench")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    sub.add_parser("freeze")
    p = sub.add_parser("analyze")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--phase", default="main")
    p.add_argument("--run-id", default="c_v1")
    p.add_argument("--units", default="", help="comma-separated subset (smoke only)")
    p = sub.add_parser("summarize")
    p.add_argument("--phase", default="main")
    p.add_argument("--run-id", default="c_v1")
    p.add_argument("--out", default=str(OUT))
    sub.add_parser("regression")
    sub.add_parser("verify")
    args = ap.parse_args()
    {
        "inventory": cmd_inventory,
        "floor": cmd_floor,
        "bench": cmd_bench,
        "freeze": cmd_freeze,
        "analyze": cmd_analyze,
        "summarize": cmd_summarize,
        "regression": cmd_regression,
        "verify": cmd_verify,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
