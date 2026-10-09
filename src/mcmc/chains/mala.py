"""Metropolis-Adjusted Langevin Algorithm (MALA)."""

import torch
from typing import Callable


def mala(
    log_prob: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int,
    step_size: float = 0.1,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Run MALA from x0 [d]; return samples [n_steps, d] and acceptance rate.

    log_prob maps [d] to a scalar and must support autograd.
    """
    if seed is not None:
        torch.manual_seed(seed)

    assert x0.dim() == 1, f"x0 must be 1D, got {x0.dim()}D"
    d = x0.shape[0]
    eps = step_size
    eps_sq = eps**2

    chains = torch.empty(n_steps, d, device=x0.device, dtype=x0.dtype)
    x = x0.clone().requires_grad_(True)
    lp = log_prob(x)
    grad_x = torch.autograd.grad(lp, x)[0].detach()
    x = x.detach()

    n_accept = 0
    for t in range(n_steps):
        # Proposal: y = x + (eps^2/2)*grad_log_p(x) + eps*z
        z = torch.randn_like(x)
        y = x + (eps_sq / 2) * grad_x + eps * z

        y_req = y.requires_grad_(True)
        lp_y = log_prob(y_req)
        grad_y = torch.autograd.grad(lp_y, y_req)[0].detach()
        y = y.detach()
        lp_y = lp_y.detach()

        # q(y|x) = N(y; x + eps^2/2 * grad_x, eps^2 I)
        # q(x|y) = N(x; y + eps^2/2 * grad_y, eps^2 I)
        log_q_yx = -0.5 * ((y - x - (eps_sq / 2) * grad_x) ** 2).sum() / eps_sq
        log_q_xy = -0.5 * ((x - y - (eps_sq / 2) * grad_y) ** 2).sum() / eps_sq

        log_alpha = lp_y - lp + log_q_xy - log_q_yx
        if torch.log(torch.rand(1, device=x.device)) < log_alpha:
            x = y
            grad_x = grad_y
            lp = lp_y
            n_accept += 1

        chains[t] = x

    acceptance_rate = n_accept / n_steps
    return chains, acceptance_rate
