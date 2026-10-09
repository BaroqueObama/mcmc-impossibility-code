"""Check span-Gram defects by reproducing selected synthetic ensembles.

Numerical arrays must match the saved run exactly; timing arrays are excluded.
Save bounds on ||V*V-I|| to $MCMC_PAPER_OUT/r3b_suite/diag/gram_certify/.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from mcmc.experiments.exp_r3b_anytime_suite import (
    apply_phase_overrides,
    build_suite_problem,
    reference_path,
)
from mcmc.paperexp import r3_suite as r3
from mcmc.paperexp.targets import load_frozen_suite

SKIP = {"timing", "wall", "wall_cum", "op_time", "peak_gpu_bytes", "peak_rss_kb", "batch_size"}
CERT = ("gram_cert_omega", "gram_cert_fro_dev", "gram_cert_fro_es")


def _range(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def _saved_chunk(root: pathlib.Path, tag: str, reps: list[int]) -> tuple[str, dict]:
    """The saved main chunk that holds exactly these reps (main-v2, -v2b, -ext-v2)."""
    for name in ("main-v2", "main-v2b", "main-ext-v2", "main-ext-v2b"):
        f = root / "main" / f"{tag}__{name}" / f"chunk_{reps[0]:05d}.npz"
        if f.exists():
            z = dict(np.load(f))
            if [int(r) for r in z["reps"]] != reps:
                raise ValueError(f"{f} holds reps {z['reps']}, expected {reps}")
            return f"{tag}__{name}", z
    raise FileNotFoundError(f"no saved main chunk starting at rep {reps[0]} for {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--reps", nargs="+", default=["0-7", "500-507"])
    ap.add_argument("--extra", nargs="*", default=[], help="cid:a-b extra rep batches")
    ap.add_argument("--kernel", default="rbf-ell")
    ap.add_argument("--eps", default="0.8")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    out_root = pathlib.Path(os.environ["MCMC_PAPER_OUT"]) / "r3b_suite"
    out_dir = out_root / "diag" / "gram_certify"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = a.configs or load_frozen_suite().config_ids("primary")
    extra: dict[str, list[list[int]]] = {}
    for e in a.extra:
        cid, rr = e.split(":")
        extra.setdefault(cid, []).append(_range(rr))
    cfg_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "configs/experiment/exp_r3b_anytime_suite.yaml"
    )
    for cid in configs:
        tag = f"{cid}__{a.kernel}__eps{a.eps}"
        cfg = OmegaConf.load(cfg_path)
        cfg.config_id, cfg.phase, cfg.mode = cid, "main", "ensembles"
        cfg = apply_phase_overrides(cfg)
        problem, meta = build_suite_problem(cfg, dev)
        ref = dict(np.load(reference_path(f"{tag}__ref-v2", "main") / "reference.npz"))
        problem.refop = r3.load_refop(ref, dev)
        problem.certify_gram = True
        rec = {"config": cid, "batches": [], "effective": meta["effective"]}
        for reps in [_range(r) for r in a.reps] + extra.get(cid, []):
            run_name, saved = _saved_chunk(out_root, tag, reps)
            setup_eff = json.loads((out_root / "main" / run_name / "setup.json").read_text())[
                "effective"
            ]
            for key in ("N_grid", "op_method", "n_refined", "fused_rowsum", "op_rtol", "init"):
                saved_v, now_v = setup_eff.get(key), meta["effective"].get(key)
                if key == "init" and saved_v is None:  # runs before the init key: known radius
                    if now_v["kind"] != "known_radius":
                        raise ValueError(f"{run_name}: init {now_v} vs a known-radius run")
                    continue
                if key not in setup_eff:
                    raise ValueError(f"{run_name}: effective setting {key} not recorded")
                if saved_v != now_v:
                    raise ValueError(f"{run_name}: effective {key} {saved_v} differs from {now_v}")
            t0 = time.perf_counter()
            outs = r3.run_ensembles(problem, reps, "main", dev)
            wall = time.perf_counter() - t0
            mismatch = []
            for i, o in enumerate(outs):
                for key in saved:
                    if key in SKIP or key in ("reps", "timing_keys"):
                        continue
                    if not np.array_equal(np.asarray(o[key]), saved[key][i], equal_nan=True):
                        mismatch.append([reps[i], key])
            cert = np.stack([o["gram_cert_omega"] for o in outs])  # (8, T, nN)
            kd = saved["op_kdirs"].astype(np.float64)
            rec_om = kd * saved["op_orth_err"]
            batch = {
                "run": run_name,
                "reps": reps,
                "bit_identical": not mismatch,
                "mismatches": mismatch[:20],
                "n_mismatch": len(mismatch),
                "fallback_events": int(sum(np.asarray(o["op_fallback"]).sum() for o in outs)),
                "wall_s": wall,
                "cert_omega": cert.tolist(),
                "recorded_omega": rec_om.tolist(),
                "cert_fro_dev": np.stack([o["gram_cert_fro_dev"] for o in outs]).tolist(),
                "cert_fro_es": np.stack([o["gram_cert_fro_es"] for o in outs]).tolist(),
                "max_cert_omega": float(cert.max()),
                "max_recorded_omega": float(rec_om.max()),
                "max_ratio_cert_over_recorded": float((cert / rec_om).max()),
                "min_ratio_cert_over_recorded": float((cert / rec_om).min()),
            }
            rec["batches"].append(batch)
            print(
                f"{cid} reps {reps[0]}..{reps[-1]} ({run_name}): bit-identical {not mismatch} "
                f"({len(mismatch)} mismatches), fallback {batch['fallback_events']}, cert omega max "
                f"{batch['max_cert_omega']:.3e}, recorded k e max {batch['max_recorded_omega']:.3e}, "
                f"cert/recorded {batch['min_ratio_cert_over_recorded']:.3f}.."
                f"{batch['max_ratio_cert_over_recorded']:.3f} ({wall:.0f} s)",
                flush=True,
            )
        b = rec["batches"]
        rec["summary"] = {
            "n_ensembles": sum(len(x["reps"]) for x in b),
            "all_bit_identical": all(x["bit_identical"] for x in b),
            "max_cert_omega": max(x["max_cert_omega"] for x in b),
            "max_recorded_omega": max(x["max_recorded_omega"] for x in b),
            "max_ratio_cert_over_recorded": max(x["max_ratio_cert_over_recorded"] for x in b),
        }
        (out_dir / f"{cid}.json").write_text(json.dumps(rec, indent=1))
        del problem
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
