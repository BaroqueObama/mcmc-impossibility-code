"""Generate the four seed-zero MALA chains and diagnostics for Figure 1."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import arviz
import numpy as np
import torch
from omegaconf import OmegaConf

from mcmc.diagnostics.standard import compute_ess, compute_rhat
from mcmc.paperexp.intro_mixture import (
    mixture_gold_with_labels,
    run_mixture_mala,
)
from mcmc.targets.mixture import GaussianMixtureTarget
from mcmc.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, help="destination for draws and diagnostics")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/experiment/exp_intro_mixture.yaml"
    cfg = OmegaConf.load(config_path)
    seed = 0
    mc = cfg.mixture
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    set_seed(seed)
    target = GaussianMixtureTarget.rotated_bimodal(int(mc.d), float(mc.separation))
    gold, labels = mixture_gold_with_labels(target, int(mc.n_gold), seed + 100)
    print(f"Sampling seed {seed}: {mc.n_chains} chains, {mc.n_draws} draws each.", flush=True)
    chains = run_mixture_mala(target, mc, seed)
    assert chains.shape == (int(mc.n_chains), int(mc.n_draws), int(mc.d))
    assert np.isfinite(chains).all()
    u = (torch.ones(int(mc.d)) / math.sqrt(int(mc.d))).numpy()
    projections = chains @ u
    output = args.out_dir or root / "paper/figures/data"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "intro_mixture_seed0_draws.npz",
        chains=chains,
        projections=projections,
        gold=gold.numpy(),
        gold_projections=gold.numpy() @ u,
        gold_labels=labels.numpy(),
        direction=u,
    )
    prefixes = np.array([200, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000])
    rhat = np.stack([compute_rhat(chains[:, :n]) for n in prefixes])
    ess = np.stack([compute_ess(chains[:, :n]) for n in prefixes])
    assert np.isfinite(rhat).all() and np.isfinite(ess).all()
    np.savez_compressed(
        output / "intro_mixture_seed0_diagnostics.npz",
        prefixes=prefixes,
        rhat_by_coordinate=rhat,
        ess_by_coordinate=ess,
        rhat_max=rhat.max(axis=1),
        ess_min=ess.min(axis=1),
    )
    source_paths = [
        "src/mcmc/paperexp/intro_mixture.py",
        "src/mcmc/chains/mala.py",
        "src/mcmc/targets/mixture.py",
        "src/mcmc/diagnostics/standard.py",
        "configs/experiment/exp_intro_mixture.yaml",
    ]
    metadata = {
        "description": "Four MALA chains on the rotated bimodal Gaussian target, seed 0.",
        "seed": seed,
        "config": OmegaConf.to_container(mc, resolve=True),
        "warmup_draws_discarded": 0,
        "warmup_note": "All sampled draws are retained; n_warmup is unused.",
        "diagnostic_definition": "Rank-normalized split Rhat and bulk ESS on the ten original coordinates, using all four chains; max/min over coordinates.",
        "final_rhat_max": float(rhat[-1].max()),
        "final_ess_min": float(ess[-1].min()),
        "draws_in_negative_halfspace": int((projections < 0).sum()),
        "draws_total": int(projections.size),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "arviz": arviz.__version__},
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths},
    }
    (output / "intro_mixture_seed0.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ["final_rhat_max", "final_ess_min", "draws_in_negative_halfspace", "draws_total"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
