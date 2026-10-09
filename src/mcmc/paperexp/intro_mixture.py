"""Sampling for the introductory trapped-mixture example; no warmup is discarded."""

import numpy as np
import torch

from mcmc.targets.mixture import GaussianMixtureTarget


def mixture_gold_with_labels(
    mixture: GaussianMixtureTarget, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw mixture samples (n,d) and component labels (n,) using a local generator."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.multinomial(mixture.weights, n, replacement=True, generator=gen)
    X = torch.empty(n, mixture.d)
    for k in range(mixture.K):
        mask = idx == k
        nk = int(mask.sum())
        if nk > 0:
            L = torch.linalg.cholesky(mixture.sigmas[k])
            X[mask] = torch.randn(nk, mixture.d, generator=gen) @ L.T + mixture.mus[k]
    return X, idx


def run_mixture_mala(mixture: GaussianMixtureTarget, mc, seed: int) -> np.ndarray:
    """Fixed-step MALA (PyTorch), returning [C, n_draws, d]."""
    from mcmc.chains.mala import mala

    mode0 = mixture.mus[mc.init_basin]
    step = float(mc.mala_step_size)
    chains = np.empty((mc.n_chains, mc.n_draws, mixture.d), dtype=np.float32)
    for c in range(mc.n_chains):
        gen = torch.Generator().manual_seed(seed + c)
        x0 = mode0 + 0.1 * torch.randn(mixture.d, generator=gen)
        ch, _ = mala(mixture.log_prob, x0, mc.n_draws, step_size=step, seed=seed + 1000 + c)
        chains[c] = ch.numpy()
    return chains
