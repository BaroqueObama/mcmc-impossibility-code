"""Gaussian mixture target distribution."""

import torch
import math


class GaussianMixtureTarget:
    """Mixture of Gaussians: pi(x) = sum_k w_k * N(x; mu_k, Sigma_k)."""

    def __init__(
        self,
        mus: list[torch.Tensor],
        sigmas: list[torch.Tensor],
        weights: torch.Tensor | None = None,
    ):
        """
        Args:
            mus: List of mean vectors, each [d].
            sigmas: List of covariance matrices, each [d, d].
            weights: Mixture weights [K] (default: uniform).
        """
        self.K = len(mus)
        assert len(sigmas) == self.K
        self.d = mus[0].shape[0]

        self.mus = torch.stack(mus)  # [K, d]
        self.sigmas = torch.stack(sigmas)  # [K, d, d]

        if weights is None:
            self.weights = torch.ones(self.K) / self.K
        else:
            assert weights.shape == (self.K,)
            assert torch.isclose(weights.sum(), torch.tensor(1.0))
            self.weights = weights

        self._precisions = torch.linalg.inv(self.sigmas)  # [K, d, d]
        self._log_dets = torch.linalg.slogdet(self.sigmas).logabsdet  # [K]
        self._log_consts = -0.5 * (self.d * math.log(2 * math.pi) + self._log_dets)  # [K]

    @staticmethod
    def symmetric_bimodal(d: int, separation: float) -> "GaussianMixtureTarget":
        """Two-component mixture with modes at +/- separation/2 along first axis."""
        mu1 = torch.zeros(d)
        mu1[0] = separation / 2
        mu2 = torch.zeros(d)
        mu2[0] = -separation / 2
        sigma = torch.eye(d)
        return GaussianMixtureTarget([mu1, mu2], [sigma, sigma])

    @staticmethod
    def rotated_bimodal(d: int, separation: float, direction: torch.Tensor | None = None) -> "GaussianMixtureTarget":
        """Two-component mixture with modes separated along a given direction.

        If direction is None, uses the all-ones vector (1/sqrt(d), ..., 1/sqrt(d)).
        """
        if direction is None:
            direction = torch.ones(d) / math.sqrt(d)
        else:
            direction = direction / direction.norm()

        mu1 = (separation / 2) * direction
        mu2 = -(separation / 2) * direction
        sigma = torch.eye(d)
        return GaussianMixtureTarget([mu1, mu2], [sigma, sigma])

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density via logsumexp over components."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = x.shape[0]
        # [n, K]
        log_components = torch.empty(n, self.K, device=x.device)
        for k in range(self.K):
            diff = x - self.mus[k]  # [n, d]
            mahal = (diff @ self._precisions[k] * diff).sum(-1)  # [n]
            log_components[:, k] = self._log_consts[k] - 0.5 * mahal + torch.log(self.weights[k])

        result = torch.logsumexp(log_components, dim=1)
        if squeeze:
            return result.squeeze(0)
        return result

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score function grad log pi(x). Maps [n, d] -> [n, d]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = x.shape[0]
        log_components = torch.empty(n, self.K, device=x.device)
        grads = torch.zeros(n, self.d, device=x.device)

        for k in range(self.K):
            diff = x - self.mus[k]
            mahal = (diff @ self._precisions[k] * diff).sum(-1)
            log_components[:, k] = self._log_consts[k] - 0.5 * mahal + torch.log(self.weights[k])

        responsibilities = torch.softmax(log_components, dim=1)  # [n, K]

        for k in range(self.K):
            diff = x - self.mus[k]
            grad_k = -diff @ self._precisions[k].T  # [n, d]
            grads += responsibilities[:, k : k + 1] * grad_k

        if squeeze:
            return grads.squeeze(0)
        return grads

    def sample(self, n: int, *, seed: int | None = None) -> torch.Tensor:
        """Sample from the mixture."""
        if seed is not None:
            torch.manual_seed(seed)

        indices = torch.multinomial(self.weights, n, replacement=True)

        samples = torch.empty(n, self.d)
        for k in range(self.K):
            mask = indices == k
            n_k = mask.sum().item()
            if n_k > 0:
                L = torch.linalg.cholesky(self.sigmas[k])
                samples[mask] = torch.randn(n_k, self.d) @ L.T + self.mus[k]

        return samples
