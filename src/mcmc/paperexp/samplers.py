"""Batched random-walk Metropolis and MALA on (J, d) float64 tensors.

With fixed S = diag(scales), the proposals are
    RWM:  y = x + h S xi,
    MALA: y = x + (h^2/2) S^2 grad log pi(x) + h S xi.
Each row is an independent chain. Each step draws xi ~ N(0, I) before the
acceptance uniforms; this order is part of seed reproducibility. Zero-density
proposals are rejected; NaN or +inf densities and non-finite MALA ratios raise.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from mcmc.paperexp.rng import canonical_device

F64 = torch.float64
LOG_2PI = math.log(2.0 * math.pi)


class MHSampler:
    def __init__(self, kind: str, target, h: float, scales):
        if kind not in ("rwm", "mala"):
            raise ValueError(f"kind must be 'rwm' or 'mala', got {kind!r}")
        if not (math.isfinite(h) and h > 0):
            raise ValueError(f"step size h must be finite and > 0, got {h}")
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (target.d,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(f"scales must be finite positive of shape ({target.d},), got {scales}")
        self.kind = kind
        self.target = target
        self.h = float(h)
        self.scales = scales
        self.d = target.d
        self._s_cache: dict[str, torch.Tensor] = {}
        self.reset_counters()

    def reset_counters(self) -> None:
        self.counters = {
            "logprob_evals": 0,
            "grad_evals": 0,
            "chain_steps": 0,
            "accepted": 0,
        }

    @property
    def acceptance_rate(self) -> float:
        c = self.counters
        if c["chain_steps"] == 0:
            raise ValueError("no steps taken yet")
        return c["accepted"] / c["chain_steps"]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "h": self.h,
            "scales": self.scales.tolist(),
            "target_id": self.target.target_id,
        }

    def _S(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._s_cache:
            self._s_cache[key] = torch.as_tensor(self.scales, dtype=F64, device=device)
        return self._s_cache[key]

    def _eval(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = x.shape[0]
        self.counters["logprob_evals"] += n
        if self.kind == "mala":
            self.counters["grad_evals"] += n
            lp, g = self.target.log_prob_and_score(x)
            return lp, g
        return self.target.log_prob(x), None

    def _drift_mean(self, x: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "rwm":
            return x
        S = self._S(x.device)
        return x + 0.5 * self.h**2 * (S * S) * g

    def _log_q(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """log N(y; mean, h^2 S^2) including the normalizing constant."""
        S = self._S(y.device)
        z = (y - mean) / (self.h * S)
        return (
            -0.5 * (z * z).sum(1)
            - self.d * math.log(self.h)
            - float(np.log(self.scales).sum())
            - 0.5 * self.d * LOG_2PI
        )

    def _transition(self, x, lp_x, g_x, xi, u):
        """Returns (x_new, lp_new, g_new, accepted, bad) where bad flags NaN/+inf densities."""
        S = self._S(x.device)
        mean_x = self._drift_mean(x, g_x)
        y = mean_x + self.h * S * xi
        lp_y, g_y = self._eval(y)
        zero_density = lp_y == -math.inf
        bad = torch.isnan(lp_y) | (lp_y == math.inf)
        if self.kind == "rwm":
            log_alpha = lp_y - lp_x
        else:
            log_alpha = (
                lp_y - lp_x + self._log_q(x, self._drift_mean(y, g_y)) - self._log_q(y, mean_x)
            )
        valid = ~zero_density & ~bad
        bad = bad | (valid & ~torch.isfinite(log_alpha))
        log_alpha = torch.where(valid, log_alpha, torch.full_like(log_alpha, -math.inf))
        acc = valid & (torch.log(u) < log_alpha)
        x_new = torch.where(acc[:, None], y, x)
        lp_new = torch.where(acc, lp_y, lp_x)
        g_new = None if g_x is None else torch.where(acc[:, None], g_y, g_x)
        return x_new, lp_new, g_new, acc, bad

    def _check_x(self, x: torch.Tensor) -> None:
        self.target._check(x)

    def step_given(self, x: torch.Tensor, xi: torch.Tensor, u: torch.Tensor):
        """One MH step with supplied N(0, I) noise xi (J, d) and uniforms u (J,)."""
        self._check_x(x)
        J = x.shape[0]
        if xi.shape != (J, self.d) or u.shape != (J,) or xi.dtype != F64 or u.dtype != F64:
            raise ValueError(f"xi {tuple(xi.shape)} {xi.dtype}, u {tuple(u.shape)} {u.dtype}")
        lp_x, g_x = self._eval(x)
        if not torch.isfinite(lp_x).all():
            raise FloatingPointError("current state has non-finite log density")
        x_new, _, _, acc, bad = self._transition(x, lp_x, g_x, xi, u)
        if bad.any():
            raise FloatingPointError(f"{int(bad.sum())} proposals with NaN/+inf density or ratio")
        self.counters["chain_steps"] += J
        self.counters["accepted"] += int(acc.sum().item())
        return x_new, acc

    def draw_noise(self, J: int, gen: torch.Generator):
        xi = torch.randn((J, self.d), generator=gen, dtype=F64, device=gen.device)
        u = torch.rand((J,), generator=gen, dtype=F64, device=gen.device)
        return xi, u

    def step(self, x: torch.Tensor, gen: torch.Generator):
        """One MH step for every row of x, noise drawn from gen (xi first, then u)."""
        if canonical_device(x.device) != canonical_device(gen.device):
            raise ValueError(f"x on {x.device}, generator on {gen.device}")
        xi, u = self.draw_noise(x.shape[0], gen)
        return self.step_given(x, xi, u)

    def run(
        self,
        x0: torch.Tensor,
        t_grid,
        gen: torch.Generator,
        store: bool = True,
        store_device: str | torch.device | None = None,
        on_checkpoint: Callable[[int, torch.Tensor], None] | None = None,
    ) -> dict:
        """Advance all rows to max(t_grid); record states at every t in t_grid (0 = x0).

        Returns {"states": {t: (J, d)} (if store), "accept_rate": overall, "accept_by_interval":
        {t: acceptance over steps (t_prev, t]}, "accept_count_per_chain": (J,) int64 accepted
        moves per chain, "counters": evaluation counts for this run, "final": state at max t}.
        """
        self._check_x(x0)
        if canonical_device(x0.device) != canonical_device(gen.device):
            raise ValueError(f"x0 on {x0.device}, generator on {gen.device}")
        grid = sorted({int(t) for t in t_grid})
        if not grid or grid[0] < 0:
            raise ValueError(f"bad t_grid {t_grid}")
        J = x0.shape[0]
        before = dict(self.counters)
        x = x0.clone()
        lp, g = self._eval(x)
        if not torch.isfinite(lp).all():
            raise FloatingPointError("initial states have non-finite log density")
        bad_total = torch.zeros((), dtype=torch.int64, device=x.device)
        acc_total = torch.zeros((), dtype=torch.int64, device=x.device)
        acc_per_chain = torch.zeros(J, dtype=torch.int64, device=x.device)
        states, acc_by = {}, {}
        t, t_prev, acc_prev = 0, 0, 0

        def record(tt):
            nonlocal t_prev, acc_prev
            nbad = int(bad_total.item())
            if nbad:
                raise FloatingPointError(
                    f"{nbad} proposals with NaN/+inf density or ratio by t={tt}"
                )
            a = int(acc_total.item())
            if tt > t_prev:
                acc_by[tt] = (a - acc_prev) / (J * (tt - t_prev))
            t_prev, acc_prev = tt, a
            if store:
                states[tt] = x.clone() if store_device is None else x.to(store_device, copy=True)
            if on_checkpoint is not None:
                on_checkpoint(tt, x)

        if grid[0] == 0:
            record(0)
        for target_t in grid:
            while t < target_t:
                xi, u = self.draw_noise(J, gen)
                x, lp, g, acc, bad = self._transition(x, lp, g, xi, u)
                acc_total += acc.sum()
                acc_per_chain += acc
                bad_total += bad.sum()
                t += 1
            if target_t > 0:
                record(target_t)
        n_acc = int(acc_total.item())
        self.counters["chain_steps"] += J * t
        self.counters["accepted"] += n_acc
        run_counters = {k: self.counters[k] - before[k] for k in self.counters}
        return {
            "states": states,
            "accept_rate": n_acc / (J * t) if t else float("nan"),
            "accept_count_per_chain": acc_per_chain,
            "accept_by_interval": acc_by,
            "counters": run_counters,
            "final": x,
        }

    def log_transition_density_offdiag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log of the absolutely continuous part q(y|x) min(1, alpha(x, y)) for y != x."""
        self._check_x(x)
        self._check_x(y)
        lp_x, g_x = self._eval(x)
        lp_y, g_y = self._eval(y)
        mean_x = self._drift_mean(x, g_x)
        log_qyx = self._log_q(y, mean_x)
        log_qxy = self._log_q(x, self._drift_mean(y, g_y))
        log_alpha = lp_y - lp_x + log_qxy - log_qyx
        return log_qyx + torch.clamp(log_alpha, max=0.0)


def make_sampler(kind: str, target, h: float, scales) -> MHSampler:
    return MHSampler(kind, target, h, scales)


def branch(x: torch.Tensor, n_branches: int) -> torch.Tensor:
    """(J*B, d) with rows [x_0]*B, [x_1]*B, ...; evolving them gives B independent branches."""
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches}")
    return x.repeat_interleave(n_branches, dim=0)
