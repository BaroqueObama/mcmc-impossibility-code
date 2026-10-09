"""Standard MCMC diagnostics via ArviZ."""

import numpy as np


def compute_rhat(chains: np.ndarray) -> np.ndarray:
    """Rank-normalized R-hat per parameter: [n_chains, n_steps, d] -> [d]."""
    import arviz as az

    assert chains.ndim == 3, f"Expected 3D array [n_chains, n_steps, d], got {chains.ndim}D"
    n_chains, n_steps, d = chains.shape

    rhat_values = np.empty(d)
    for i in range(d):
        data = az.convert_to_dataset(chains[:, :, i : i + 1])
        rhat_values[i] = az.rhat(data).to_array().values.item()

    return rhat_values


def compute_ess(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS per parameter: [n_chains, n_steps, d] -> [d]."""
    import arviz as az

    assert chains.ndim == 3, f"Expected 3D array [n_chains, n_steps, d], got {chains.ndim}D"
    n_chains, n_steps, d = chains.shape

    ess_values = np.empty(d)
    for i in range(d):
        data = az.convert_to_dataset(chains[:, :, i : i + 1])
        ess_values[i] = az.ess(data).to_array().values.item()

    return ess_values
