"""Coverage, tightness and cost tables from saved synthetic ensembles.

Classify coverage against independent reference intervals, keeping unresolved
cases separate. Output goes to $MCMC_PAPER_OUT/r3b_suite/analysis/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

from mcmc.experiments.exp_r3b_anytime_suite import load_chunks
from mcmc.paperexp import r3_suite as r3
from mcmc.paperexp.targets import load_frozen_suite

TABLE_METHODS = [
    ("direct", "direct full-MMD"),
    ("B3_N{N}", "Sec.3 (est. Sigma)"),
    ("B4same_N{N}", "Sec.4 same space (est.)"),
    ("B4own_N{N}", "Sec.4 own space (est.)"),
    ("NO3_N{N}", "no-obs static (est.)"),
    ("NO4_N{N}", "no-obs dynamic (est.)"),
    ("B3ref", "Sec.3 plug-in ref-op"),
    ("B4ref_same", "Sec.4 same plug-in"),
    ("B4ref_own", "Sec.4 own plug-in"),
    ("B4ref_own_tr", "Sec.4 own plug-in+tau"),
]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


ZERO_FILL = ("op_fallback",)  # Earlier runs failed instead of using a fallback.
EFFECTIVE_MAY_DIFFER = ("op_fallback",)  # Fallback for cuSOLVER non-convergence.


def _continued(d: pathlib.Path, run_dirs: list[pathlib.Path]) -> bool:
    """A FAILED run is usable iff a continuation run <id>b starts exactly at its completed count."""
    cont = d.with_name(d.name + "b")
    if cont not in run_dirs:
        return False
    st = json.loads((d / "RUN-STATUS.json").read_text())
    me_d = json.loads((d / "setup.json").read_text())
    me_c = json.loads((cont / "setup.json").read_text())
    return me_c["rep_range"][0] == me_d["rep_range"][0] + int(st["completed_units"]) and (
        me_c["rep_range"][1] == me_d["rep_range"][1]
    )


def load_union(run_dirs: list[pathlib.Path], allow_partial: bool) -> tuple[dict, dict, dict]:
    """Combine runs with matching setups, references and numerical settings.

    Replicate IDs must be disjoint and cover 0..N-1. A failed run is accepted only
    with a continuation from its completed count, unless allow_partial is set.
    """
    setups, metas, parts, statuses, accepted_failed = [], [], [], {}, []
    for d in run_dirs:
        st = json.loads((d / "RUN-STATUS.json").read_text())["status"]
        if st == "FAILED" and _continued(d, run_dirs):
            accepted_failed.append(str(d))
        if (
            st != "COMPLETED"
            and not allow_partial
            and not (st == "FAILED" and _continued(d, run_dirs))
        ):
            raise RuntimeError(f"{d} is {st} (use --allow-partial to include incomplete runs)")
        statuses[str(d)] = st
        setups.append(dict(np.load(d / "setup.npz")))
        metas.append(json.loads((d / "setup.json").read_text()))
        parts.append(load_chunks(d))
    for d, su, me in zip(run_dirs[1:], setups[1:], metas[1:]):
        if set(su) != set(setups[0]) or any(not np.array_equal(su[k], setups[0][k]) for k in su):
            raise ValueError(f"{d}: setup arrays differ from {run_dirs[0]}")
        for key in ("reference_sha256", "namespace", "stage_prefix", "t_grid"):
            if me.get(key) != metas[0].get(key):
                raise ValueError(f"{d}: {key} differs from {run_dirs[0]}")
        e0 = {
            k: v
            for k, v in (metas[0].get("effective") or {}).items()
            if k not in EFFECTIVE_MAY_DIFFER
        }
        e1 = {k: v for k, v in (me.get("effective") or {}).items() if k not in EFFECTIVE_MAY_DIFFER}
        if e0 != e1:
            raise ValueError(f"{d}: effective settings differ from {run_dirs[0]}: {e1} vs {e0}")
    reps = np.concatenate([pp["reps"] for pp in parts])
    if len(np.unique(reps)) != len(reps):
        raise ValueError(f"overlapping reps across {run_dirs}")
    if not allow_partial and not np.array_equal(np.sort(reps), np.arange(len(reps))):
        raise ValueError(f"union of reps is not 0..{len(reps) - 1}")
    for k in ZERO_FILL:
        have = [pp for pp in parts if k in pp]
        for pp in parts:
            if k not in pp and have:
                shape = (len(pp["reps"]),) + have[0][k].shape[1:]
                pp[k] = np.zeros(shape, dtype=have[0][k].dtype)
    keys = [k for k in parts[0] if k != "timing_keys"]
    for pp in parts[1:]:
        if set(k for k in pp if k != "timing_keys") != set(keys):
            raise ValueError(f"output keys differ across runs: {sorted(set(pp) ^ set(parts[0]))}")
    data = {k: np.concatenate([pp[k] for pp in parts]) for k in keys}
    data["timing_keys"] = parts[0]["timing_keys"]
    union = {
        "runs": [str(d) for d in run_dirs],
        "statuses": statuses,
        "n_per_run": [int(len(pp["reps"])) for pp in parts],
        "rep_min": int(reps.min()),
        "rep_max": int(reps.max()),
        "n": int(len(reps)),
        "effective_per_run": [me.get("effective") for me in metas],
        "accepted_failed": accepted_failed,  # FAILED runs continued exactly at their completed count
        "runs_detail": [],
    }
    for d, me, pp in zip(run_dirs, metas, parts):
        prov = (
            json.loads((d / "PROVENANCE.json").read_text())
            if (d / "PROVENANCE.json").exists()
            else {}
        )
        fb = (me.get("effective") or {}).get("op_fallback")
        capable = fb is not None
        n_ev = int(np.asarray(pp["op_fallback"]).sum()) if "op_fallback" in pp else 0
        union["runs_detail"].append(
            {
                "run": d.name,
                "status": statuses[str(d)],
                "commit": prov.get("commit"),
                "rep_range": me.get("rep_range"),
                "reps_present": [int(pp["reps"].min()), int(pp["reps"].max())],
                "n_present": int(len(pp["reps"])),
                "fallback_capable": capable,
                "fallback_method": fb,
                "fallback_events": n_ev,
                "events_note": "counted"
                if capable
                else "0: this run had no fallback; solver failure would have stopped it",
            }
        )
    if "op_fallback" in data:
        ev = np.argwhere(data["op_fallback"] > 0)  # (ensemble index, t index, N index)
        t_grid = np.asarray(setups[0]["t_grid"])
        union["fallback_events"] = [
            {"rep": int(data["reps"][e]), "t": int(t_grid[ti]), "N_index": int(ni)}
            for e, ti, ni in ev
        ]
    return setups[0], data, union


def analyze_run(
    run_dir: pathlib.Path, ref_dir: pathlib.Path, allow_partial: bool, extra_runs=()
) -> dict:
    runs = [run_dir] + [d for d in extra_runs if (d / "RUN-STATUS.json").exists()]
    setup, data, union = load_union(runs, allow_partial)
    ok = all(
        v == "COMPLETED" or k in union["accepted_failed"] for k, v in union["statuses"].items()
    )  # a FAILED run continued exactly at its completed count is part of a complete sample
    status = {"status": "COMPLETED" if ok else "PARTIAL"}
    meta = json.loads((run_dir / "setup.json").read_text())
    sha = hashlib.sha256((ref_dir / "reference.npz").read_bytes()).hexdigest()
    if sha != meta["reference_sha256"]:
        raise ValueError(
            f"{ref_dir}/reference.npz sha256 {sha} != recorded {meta['reference_sha256']}"
        )
    ref = dict(np.load(ref_dir / "reference.npz"))
    emb = embedding_status(meta, ref)
    design = {
        "N_pi": int(setup["N_pi"]),
        "M": int(np.asarray(ref["landmarks"]).shape[0]),
        "init_draws_per_replica": float(np.median(data["init_proposals"]))
        / float(setup["J_grid"][-1]),
    }
    init = (meta.get("effective") or {}).get("init") or {}
    conventional = bool(init) and not init.get("A3_holds", True)
    chi = None
    if conventional:
        target = load_frozen_suite().target(meta["target"]["target_id"])
        chi = r3.init_chi2(
            target, str(init["kind"]), float(init["scale"]), float(init["assumed_R"])
        )
    # For conventional starts, the assumed R is invalid; only the direct bound is R-free.
    applicable = ["direct"] if conventional else None
    summ = r3.summarize(
        setup,
        data,
        ref,
        ref,
        quad_margin_known=emb["q_known"],
        design=design,
        applicable=applicable,
    )
    summ["embedding_status"] = emb
    summ["design"] = design
    summ["applicability"] = {
        "applicable_guarantees": "all guaranteed-by-construction methods"
        if applicable is None
        else applicable,
        "reason": "known radius: (A3) holds with R = eps"
        if applicable is None
        else "conventional start: only the direct bound is R-free; every other bound uses the "
        "assumed radius, which is invalid for this initial law",
    }
    if conventional:
        summ["init_chi2"] = chi
        R_a = float(init["assumed_R"])
        why = (
            f"(A3) fails: chi^2 = infinity ({chi['argument']})"
            if not chi["finite"]
            else f"R = {R_a} understates the true R = sqrt(chi^2) = {chi['R']:.4f}"
        )
        summ["labels"] = {
            k: v if k == "direct" else f"{v}; assumed R = {R_a}; {why}"
            for k, v in summ["labels"].items()
        }
        if chi["finite"]:
            summ["interpretation"] = (
                f"Initial law {init['kind']}: chi^2(pi0 || pi) = {chi['chi2']:.6g} "
                f"({chi['argument']}). Assumption (A3) holds with R = {chi['R']:.6g}, but "
                f"the assumed R = {R_a} is too small. Bounds evaluated at R = {R_a} therefore "
                "lack coverage guarantees; joint_assumed_radius records their exceedances. "
                f"The at_true_radius results use R = {chi['R']:.6g} for a valid coverage check. "
                "The direct bound is independent of R."
            )
            tr = r3.summarize(
                setup,
                data,
                ref,
                ref,
                quad_margin_known=emb["q_known"],
                r_init=chi["R"],
            )
            summ["at_true_radius"] = reduced_coverage(tr, chi["R"])
        else:
            summ["interpretation"] = (
                f"Initial law {init['kind']}: chi^2(pi0 || pi) = infinity ({chi['argument']}), "
                f"so (A3) fails for the assumed R = {R_a}. Only the direct bound has a guarantee. "
                "For the other methods, joint_assumed_radius counts B_t < L_t as a radius "
                "misspecification exceedance; 'covered' means the bound exceeds the reference "
                "interval, without a coverage guarantee. joint_guaranteed includes the R-free "
                "direct bound."
            )
    summ["init"] = init
    summ["run_status"] = status["status"]
    summ["run_dir"] = str(run_dir)
    summ["union"] = union
    summ["reference_dir"] = str(ref_dir)
    summ["setup"] = {k: setup[k].tolist() for k in setup}
    summ["truth"] = {
        k: ref[k].tolist()
        for k in ("D", "L", "U", "mmd2_u", "pA", "b_ref_t", "b_ref_pi", "quad_margin", "beta_ref")
    }
    summ["reference_cost"] = {
        "wall": float(ref["wall"]),
        "timing": dict(zip([str(x) for x in ref["timing_keys"]], ref["timing"].tolist())),
        "peak_gpu_bytes": int(ref["peak_gpu_bytes"]),
        "J_ref": int(ref["J_ref"]),
        "refop_N": int(ref["refop_N"]),
        "refop_rank": int(ref["refop_rank"]),
    }
    summ["refop"] = {
        "own": ref["own"].tolist(),
        "own_half_a": ref["own_half_a"].tolist(),
        "own_half_b": ref["own_half_b"].tolist(),
        "same": ref["same"].tolist(),
        "lam1": ref["lam1"].tolist(),
        "tau": ref["tau"].tolist(),
    }
    summ["transitions"] = {
        "replica_per_ensemble_Jmax": float(setup["J_grid"][-1] * setup["t_grid"][-1]),
        "operator_per_ensemble_Nmax": float(4 * setup["N_grid"][-1] * setup["t_grid"][-1]),
        "grad_per_eval": int(data["grad_per_eval"][0]),
    }
    return summ


def embedding_status(meta: dict, ref: dict) -> dict:
    """Target embedding and numerical margin for the known-target bounds.

    Gaussian and mixture embeddings use closed forms (q=0). Student-t and funnel
    embeddings use q=sqrt(3 eps_quad); their reference intervals also include q.
    Coverage therefore depends on the declared quadrature tolerance.
    """
    emb = meta.get("embedding")
    if emb is None:
        return {"kind": "none (sampled reference only)", "q_known": 0.0, "conditional": None}
    if emb.get("method") == "quadrature":
        q = float(ref["quad_margin"])
        return {
            "kind": f"quadrature ({emb.get('rule')}, order {emb.get('order')})",
            "q_known": q,
            "conditional": (
                "conditional on the numerical quadrature tolerance (declared |mu~ - mu_pi| <= "
                f"{q * q / 3:.0e}; q = {q:.3e} added to the known cells and to every truth interval)"
            ),
        }
    return {"kind": "closed form (analytic; roundoff only)", "q_known": 0.0, "conditional": None}


def reduced_coverage(summ: dict, R: float) -> dict:
    """Coverage/crossing counts per method and cell (for a re-evaluation at another radius)."""
    keep = (
        "n",
        "covered_all",
        "any_violation",
        "unresolved",
        "cp_if_unresolved_fail",
        "cp_if_unresolved_succeed",
    )
    out = {"R": R, "labels": summ["labels"], "methods": {}}
    for name, md in summ["methods"].items():
        out["methods"][name] = {
            cell: {
                **{k: rec[k] for k in keep},
                "cross_0.1": rec["cross_0.1"]["n_crossed"],
                "cross_0.05": rec["cross_0.05"]["n_crossed"],
                "B_median_last": rec["B_median"][-1],
            }
            for cell, rec in md.items()
        }
    for k in [k for k in summ if k.startswith("joint_guaranteed_")]:
        out[k] = summ[k]
    return out


def fmt_cp(rec) -> str:
    n = rec["n"]
    c, u, v = rec["covered_all"], rec["unresolved"], rec["any_violation"]
    lo = rec["cp_if_unresolved_fail"][0]
    hi = rec["cp_if_unresolved_succeed"][1]
    return f"{c}/{n} (u{u} v{v}) [{lo:.3f},{hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--proto", default="v2")
    ap.add_argument("--kernel", default="rbf-ell")
    ap.add_argument("--eps", default="0.8")
    ap.add_argument("--init", choices=["point", "disp"], help="conventional-start runs (no A3)")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--J", type=int, default=32768)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--m", type=int, default=8)
    a = ap.parse_args()
    root = pathlib.Path(os.environ["MCMC_PAPER_OUT"]) / "r3b_suite"
    configs = a.configs or load_frozen_suite().config_ids("primary")
    itag = f"__init-{a.init}" if a.init else ""
    out_dir = root / "analysis" / f"{a.phase}-{a.proto}__{a.kernel}__eps{a.eps}{itag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cov_lines, cost_lines = [], [], []
    cov_lines.append(
        f"# Synthetic coverage: {a.phase} ({a.proto}, {a.kernel}, eps={a.eps})\n\n"
        f"J={a.J}, N={a.N}, m={a.m}, sampled target reference. Coverage is simultaneous over "
        "the checkpoint grid. Each cell reports covered_all/n, unresolved (u), violations (v), "
        "and a Clopper-Pearson interval treating unresolved cases as failures/successes. "
        "Crossings report the count and median time at tolerances 0.10 and 0.05.\n\n"
        "Bounds include Gram inflation 1/sqrt(1 - omega) and the quadrature margin q in "
        "known-target cells. Plug-in estimates have no coverage guarantee.\n"
    )
    cost_lines.append(f"# R3b {a.phase} cost per ensemble ({a.proto}, {a.kernel}, eps={a.eps})\n")
    cost_lines.append(
        "| config | n | T3 | wall med (s) | wall max (s) | peak GPU MiB | op s/ckpt (N grid) | "
        "kernel evals (target/repl/op/direct) | replica transitions | operator transitions | "
        "init proposals | accept rep | reference wall (s) |"
    )
    cost_lines.append("|" + "---|" * 13)
    for cid in configs:
        tag = f"{cid}__{a.kernel}__eps{a.eps}{itag}"
        run_dir = root / a.phase / f"{tag}__{a.phase}-{a.proto}"
        if not (run_dir / "RUN-STATUS.json").exists():
            print(f"[skip] {run_dir} missing", file=sys.stderr)
            continue
        # Load the recorded reference; analyze_run checks its hash.
        ref_dir = pathlib.Path(json.loads((run_dir / "setup.json").read_text())["reference_dir"])
        names = [f"{tag}__{a.phase}-{a.proto}b"]  # continuation after a FAILED run
        if a.phase == "main":
            names += [f"{tag}__{a.phase}-ext-{a.proto}", f"{tag}__{a.phase}-ext-{a.proto}b"]
        ext = [root / a.phase / nm for nm in names]
        summ = analyze_run(run_dir, ref_dir, a.allow_partial, ext)
        (out_dir / f"{run_dir.name}.summary.json").write_text(
            json.dumps(summ, indent=1, default=_json_default)
        )
        t = summ["setup"]["t_grid"]
        mid = len(t) // 2
        D = summ["truth"]["D"]
        cov_lines.append(
            f"\n## {cid} (n={summ['n_ensembles']}, status {summ['run_status']})\n\n"
            f"truth D_t (t=0/{t[mid]}/{t[-1]}): {D[0]:.4f}/{D[mid]:.4f}/{D[-1]:.4f}; interval "
            f"half-width {summ['truth']['U'][0] - D[0]:.4f}; joint guaranteed (sampled): "
            f"{summ['joint_guaranteed_sampled']}\n"
        )
        if "joint_assumed_radius_sampled" in summ:
            ja = summ["joint_assumed_radius_sampled"]
            cov_lines.append(
                "Methods with applicable guarantees: "
                f"{summ['applicability']['applicable_guarantees']}. Assumed-radius bounds "
                f"({len(ja['methods'])} methods, no coverage guarantee), jointly with sampled references: "
                f"covered {ja['covered_all']}, radius-misspecification exceedance (B < L) "
                f"{ja['radius_misspecification_exceedance']}, unresolved {ja['unresolved']}\n"
            )
        es = summ["embedding_status"]
        cov_lines.append(
            f"Target embedding: {es['kind']}"
            + (f"; {es['conditional']}" if es["conditional"] else "")
            + "\n"
        )
        co = summ["corrections"]
        gm = [v for v in co["methods"].values() if v["guaranteed"]]
        cov_lines.append(
            f"Corrections vs uncorrected assembly: omega max {co['omega_max']:.3e} (inflation "
            f"{co['inflation_max'] - 1:.2e}); guaranteed methods: max change "
            f"{max(v['max_change'] for v in gm):.3e} (Gram part {max(v['max_change_gram'] for v in gm):.3e}), "
            f"changed coverage codes {sum(v['cells_code_changed'] for v in gm)}, changed crossing "
            "counts "
            + str(
                {
                    n: v["crossings_changed"]
                    for n, v in co["methods"].items()
                    if v["guaranteed"] and v["crossings_changed"]
                }
                or "none"
            )
            + "; plug-ins: changed codes "
            f"{sum(v['cells_code_changed'] for v in co['methods'].values() if not v['guaranteed'])}, "
            "changed crossing counts "
            + str(
                {
                    n: v["crossings_changed"]
                    for n, v in co["methods"].items()
                    if not v["guaranteed"] and v["crossings_changed"]
                }
                or "none"
            )
            + "\n"
        )
        if "interpretation" in summ:
            cov_lines.append(f"Initial law: {summ['interpretation']}\n")
        if "at_true_radius" in summ:
            tr = summ["at_true_radius"]
            cell = f"J{a.J}|m{a.m}|sampled"
            cov_lines.append(
                f"At the true radius R = {tr['R']:.4f}, (A3) holds for all non-plug-in methods. "
                "Joint coverage with sampled references: "
                f"{tr['joint_guaranteed_sampled']}; "
                + "; ".join(
                    f"{pat.format(N=a.N)} {fmt_cp(tr['methods'][pat.format(N=a.N)][cell])} "
                    f"(B last {tr['methods'][pat.format(N=a.N)][cell]['B_median_last']:.4f}, "
                    f"cross 0.10 {tr['methods'][pat.format(N=a.N)][cell]['cross_0.1']})"
                    for pat, _ in TABLE_METHODS
                    if tr["labels"][pat.format(N=a.N)] == "guarantee" or pat == "B4ref_own"
                )
                + "\n"
            )
        un = summ.get("union", {})
        if un:
            parts_txt = "; ".join(
                f"{r['run']} [{r['status']}, {str(r['commit'])[:8]}, reps {r['reps_present'][0]}.."
                f"{r['reps_present'][1]} (n={r['n_present']}), "
                f"fallback {'capable' if r['fallback_capable'] else 'not capable'}: "
                f"{r['fallback_events']} event(s)]"
                for r in un.get("runs_detail", [])
            )
            ev = un.get("fallback_events", [])
            cov_lines.append(
                f"Sample: n={un['n']} = {parts_txt}. Eigensolve fallback events (rep, t, N_index): "
                f"{len(ev)} {[(e['rep'], e['t'], e['N_index']) for e in ev]}\n"
            )
        cov_lines.append(
            "| method | label | coverage | B t=0 | B mid | B last | B/D t=0 | B/D mid | "
            "cross 0.10 | cross 0.05 |"
        )
        cov_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for pat, label in TABLE_METHODS:
            name = pat.format(N=a.N)
            rec = summ["methods"][name][f"J{a.J}|m{a.m}|sampled"]
            Bm = rec["B_median"]
            c10, c05 = rec["cross_0.1"], rec["cross_0.05"]
            cov_lines.append(
                f"| {name} | {label} ({summ['labels'][name]}) | {fmt_cp(rec)} | {Bm[0]:.4f} | "
                f"{Bm[mid]:.4f} | {Bm[-1]:.4f} | {rec['ratio_to_D_median'][0]:.2f} | "
                f"{rec['ratio_to_D_median'][mid]:.2f} | "
                f"{c10['n_crossed']} @t~{c10['t_median_crossed']} | "
                f"{c05['n_crossed']} @t~{c05['t_median_crossed']} |"
            )
            rows.append(
                {
                    "config": cid,
                    "method": name,
                    "label": summ["labels"][name],
                    "n": rec["n"],
                    "covered_all": rec["covered_all"],
                    "unresolved": rec["unresolved"],
                    "any_violation": rec["any_violation"],
                    "B_t0": Bm[0],
                    "B_last": Bm[-1],
                    "D_t0": D[0],
                    "D_last": D[-1],
                    "cross010": c10["n_crossed"],
                    "cross010_t": c10["t_median_crossed"],
                    "cross005": c05["n_crossed"],
                    "cross005_t": c05["t_median_crossed"],
                    "false_stops_010": c10["false_stops_definite"],
                    "false_stops_005": c05["false_stops_definite"],
                }
            )
        # worst case over every (J, m, reference) cell of each method
        cov_lines.append(
            "\nAll cells (every J, m, reference kind): method: min covered_all, max any_violation, "
            "max unresolved over cells"
        )
        for name, md in summ["methods"].items():
            vals = list(md.values())
            line = (
                f"- {name} [{summ['labels'][name]}]: min covered {min(v['covered_all'] for v in vals)}"
                f"/{vals[0]['n']}, max violation {max(v['any_violation'] for v in vals)}, "
                f"max unresolved {max(v['unresolved'] for v in vals)}"
            )
            if vals[0]["ensemble_invariant"] and max(v["unresolved"] for v in vals):
                tu = sorted({t_ for v in vals for t_ in v["t_unresolved"]})
                line += (
                    f"; one curve repeated across {vals[0]['n']} ensembles, so these are not "
                    "independent coverage checks. Unresolved checkpoints: "
                    f"{tu}; any one makes the trajectory unresolved."
                )
            cov_lines.append(line)
        # paired B3 - B4 at the primary cell, last checkpoint (mean +- se over ensembles)
        J_grid, m_grid = summ["setup"]["J_grid"], summ["setup"]["m_grid"]
        ji, mi = J_grid.index(a.J), m_grid.index(a.m)
        cov_lines.append(
            f"\nPaired B3 - B4 at J={a.J}, m={a.m}, sampled reference (mean +- se). "
            "Negative counts cover every ensemble, t, J, m and reference: differences below "
            "-1e-12, with counts below -1e-5 in brackets."
        )
        for key, pr in summ["paired"].items():
            mean = np.array(pr["mean"])[:, ji, mi, 0]
            se = np.array(pr["se"])[:, ji, mi, 0]
            am = pr["argmin"]
            cov_lines.append(
                f"- {key}: t=0 {mean[0]:+.4f}+-{se[0]:.4f}; t={t[mid]} {mean[mid]:+.4f}+-{se[mid]:.4f}; "
                f"t={t[-1]} {mean[-1]:+.4f}+-{se[-1]:.4f}; negative {pr['n_negative']}/"
                f"{pr['n_entries']} ({100 * pr['n_negative'] / pr['n_entries']:.4f}%) "
                f"[{pr['n_negative_1e5']}], at t>0 {pr['n_negative_t_pos']}/{pr['n_entries_t_pos']} "
                f"[{pr['n_negative_t_pos_1e5']}], at the last t {pr['n_negative_last_t']} "
                f"[{pr['n_negative_last_t_1e5']}]; t with material reversals "
                f"{ {t_: v for t_, v in pr['n_negative_by_t_1e5'].items() if v} }; min "
                f"{pr['min']:+.3e} (ensemble {am['ensemble_index']}, "
                f"t={am['t']}, J={am['J']}, m={am['m']}, {am['ref']})"
            )
        cov_lines.append(
            "\nMatched transition counts: (J + 4N)t for B3 with J + 4N replicas and B4 with "
            "J replicas plus N blocks. Total computation may differ. m="
            f"{a.m}, sampled, last checkpoint mean B3 - B4:"
        )
        for cm in summ["transition_matched"]:
            mean = np.array(cm["mean_B3_minus_B4"])[:, mi, 0]
            cov_lines.append(f"- {cm['B3']} vs {cm['B4']}: {mean[-1]:+.4f} (t=0 {mean[0]:+.4f})")
        if "cost_comparison" in summ:
            u = summ["cost_units"]
            cov_lines.append(
                "\nCost model: each method pays its full operation count. Seconds per operation "
                "use the median stage time measured on shared GPUs: "
                f"transition {u['transitions']:.3e} s, kernel eval (projection/direct/"
                f"target) {u['kernel_proj']:.3e}/{u['kernel_direct']:.3e}/{u['kernel_target']:.3e} s, "
                f"operator estimate {u['solves']:.3f} s. Modeled seconds to the last checkpoint, "
                f"sampled, m={a.m}:"
            )
            for nm_ in ("direct", f"B3_N{a.N}", f"B4same_N{a.N}", f"B4own_N{a.N}", "B4ref_own"):
                cl = [summ["methods"][nm_][f"J{J}|m{a.m}|sampled"]["cost_last"] for J in J_grid]
                cov_lines.append(
                    f"- {nm_}: "
                    + ", ".join(
                        f"J={J}: {c_['modeled_seconds']:.2f} s" for J, c_ in zip(J_grid, cl)
                    )
                )
            cov_lines.append(
                "Compare B4 with the largest grid J for which B3's modeled cost is no larger. "
                "The finite grid can leave B3 below the equal-cost budget. Also report the "
                "fraction with R U3 > B4 at the last checkpoint: since B3 >= R U3 for every J, "
                "a fraction of 1 rules out matching B4 with more replicas at this N, m and "
                "operator realization. A smaller fraction does not establish that more "
                "replicas close the gap:"
            )
            for cc in summ["cost_comparison"]:
                if cc["ref"] != "sampled":
                    continue
                d_ = cc.get("mean_B3_minus_B4_last_by_m", {})
                cov_lines.append(
                    f"- {cc['B4']} ({cc['modeled_seconds_B4']:.2f} s): "
                    + (
                        f"{cc['B3_at_most_B4_cost']} mean B3 - B4 {d_[str(a.m)] if str(a.m) in d_ else d_.get(a.m):+.4f}"
                        f" (B4/B3 modeled cost {cc['cost_ratio_B4_over_B3']:.2f})"
                        if "B3_at_most_B4_cost" in cc
                        else "no grid B3 within its cost"
                    )
                    + f"; R U3 > B4 in {cc['B3_floor_RU3_exceeds_B4_frac_by_m'].get(a.m, cc['B3_floor_RU3_exceeds_B4_frac_by_m'].get(str(a.m))):.3f}"
                    " of ensembles"
                )
        c = summ["cost"]
        ke = c["kernel_evals_median_total"]
        op_t = np.array(c["op_time_median_per_checkpoint"]).mean(0)
        cost_lines.append(
            f"| {cid} | {summ['n_ensembles']} | {t[-1]} | {c['wall_per_ensemble_median']:.1f} | "
            f"{c['wall_per_ensemble_max']:.1f} | {c['peak_gpu_bytes_max'] / 2**20:.0f} | "
            f"{'/'.join(f'{x:.2f}' for x in op_t)} | "
            f"{'/'.join(f'{x:.2e}' for x in ke[:4])} | "
            f"{summ['transitions']['replica_per_ensemble_Jmax']:.3g} | "
            f"{summ['transitions']['operator_per_ensemble_Nmax']:.3g} | "
            f"{c['init_proposals_median']:.0f} | {c['accept_rep_final_median']:.3f} | "
            f"{summ['reference_cost']['wall']:.0f} |"
        )
    (out_dir / "table_coverage.md").write_text("\n".join(cov_lines) + "\n")
    (out_dir / "table_cost.md").write_text("\n".join(cost_lines) + "\n")
    if rows:
        with open(out_dir / "table_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(cov_lines))
    print("\n".join(cost_lines))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
