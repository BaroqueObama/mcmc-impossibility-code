"""Exact checkpoint transitions for linear-Gaussian chains and log-scale MMD.

Binary doubling gives A_{2g} = A_g A_g and Q_{2g} = Q_g + A_g Q_g A_g^T,
avoiding cancellation in C - A_g C A_g^T. One Gaussian draw per checkpoint
preserves the joint law, including temporal dependence; random draws differ
from stepwise simulation. Gap 1 uses the original noise factor.

Log-scale mean propagation keeps late-time MMD values distinguishable from
zero when direct float64 evaluation underflows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mcmc.paperexp import gaussian_linear as gl

F64 = torch.float64
TINY = float(np.finfo(np.float64).tiny)  # smallest normal float64
LN_TINY = math.log(TINY)


# Gap transitions by binary doubling
@dataclass
class GapTransition:
    """Exact g-step transition: X' = mu + A (X - mu) + L z with L L^T = Q."""

    gap: int
    A: np.ndarray  # A^g (binary doubling; A itself for g = 1)
    Q: np.ndarray  # Q_g (binary doubling; LQ LQ^T for g = 1)
    L: np.ndarray  # LQ for g = 1, else the Cholesky factor of Q_g
    checks: dict


def _is_pow2(g: int) -> bool:
    return g >= 1 and (g & (g - 1)) == 0


def gap_transitions(chain: gl.LinearGaussianChain, gaps, stat_tol: float = 1e-10) -> dict:
    """Return {g: GapTransition} for power-of-two gaps.

    Check stationarity, positive noise covariance, and agreement with
    stepwise covariance accumulation. Residuals are stored in ``checks``.
    """
    gaps = sorted({int(g) for g in gaps})
    if not gaps or any(not _is_pow2(g) for g in gaps):
        raise ValueError(f"gaps must be powers of two, got {gaps}")
    C = chain.C
    cmax = float(np.abs(C).max())
    table, A_g, Q_g, g = {}, chain.A.copy(), chain.Q, 1
    while g <= gaps[-1]:
        if g > 1:
            Q_g = Q_g + A_g @ Q_g @ A_g.T  # Q_{2g} from (A_g, Q_g) at the previous g
            A_g = A_g @ A_g
        Q_g = 0.5 * (Q_g + Q_g.T)
        if g in gaps:
            stat = float(np.abs(A_g @ C @ A_g.T + Q_g - C).max()) / cmax
            if not stat <= stat_tol:
                raise FloatingPointError(f"gap {g}: stationary residual {stat:.3e} > {stat_tol}")
            eig = np.linalg.eigvalsh(Q_g)
            if not eig[0] > 0:
                raise FloatingPointError(f"gap {g}: Q_g not positive definite (min eig {eig[0]:.3e})")
            L = chain.LQ.copy() if g == 1 else np.linalg.cholesky(Q_g)
            A_ref = np.linalg.matrix_power(chain.A, g)
            Q_ref = chain.noise_cov(g)
            qmax = float(np.abs(Q_g).max())
            checks = {
                "stationary_residual_rel": stat,
                "Q_min_eig": float(eig[0]),
                "Q_max_eig": float(eig[-1]),
                "Q_cond": float(eig[-1] / eig[0]),
                "factor_residual_rel": float(np.abs(L @ L.T - Q_g).max()) / qmax,
                "A_vs_matrix_power_max_abs": float(np.abs(A_g - A_ref).max()),
                "A_max_abs": float(np.abs(A_ref).max()),
                "Q_vs_stepwise_max_abs_rel": float(np.abs(Q_g - Q_ref).max()) / qmax,
            }
            if checks["Q_vs_stepwise_max_abs_rel"] > 1e-9 or checks["factor_residual_rel"] > 1e-12:
                raise FloatingPointError(f"gap {g}: transition checks failed {checks}")
            table[g] = GapTransition(gap=g, A=A_g.copy(), Q=Q_g.copy(), L=L, checks=checks)
        g *= 2
    return table


class CheckpointChain(gl.LinearGaussianChain):
    """A LinearGaussianChain whose ``sweep(x, gen, g)`` draws the exact g-step transition at once.

    Only gaps in the precomputed table are allowed (raises otherwise). Counters record the rows
    actually transitioned (simulated transitions) and the sweeps they represent.
    """

    @classmethod
    def from_chain(cls, chain: gl.LinearGaussianChain, gaps) -> "CheckpointChain":
        new = cls(A=chain.A, LQ=chain.LQ, mu=chain.mu, C=chain.C, name=f"{chain.name}+exact_gaps")
        new.gaps = gap_transitions(chain, gaps)
        new._dev = {}
        new.reset_counters()
        return new

    def reset_counters(self) -> None:
        self.rows_transitioned = 0  # sum over calls of rows (one Gaussian draw per row per call)
        self.rows_sweeps_represented = 0  # sum over calls of rows * gap

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_dev"] = {}  # never pickle device tensors
        return state

    def _params(self, g: int, device) -> tuple:
        key = (str(device), g)
        if key not in self._dev:
            tr = self.gaps[g]
            self._dev[key] = (
                torch.as_tensor(self.mu, dtype=F64, device=device),
                torch.as_tensor(tr.A, dtype=F64, device=device),
                torch.as_tensor(tr.L, dtype=F64, device=device),
            )
        return self._dev[key]

    def sweep(self, x: torch.Tensor, gen: torch.Generator, n_steps: int = 1, params=None):
        g = int(n_steps)
        if g not in self.gaps:
            raise KeyError(f"no exact transition for gap {g}; have {sorted(self.gaps)}")
        mu, Ag, Lg = self._params(g, x.device)
        z = torch.randn(x.shape, generator=gen, dtype=F64, device=x.device)
        self.rows_transitioned += x.shape[0]
        self.rows_sweeps_represented += x.shape[0] * g
        return mu + (x - mu) @ Ag.T + z @ Lg.T


def grid_gaps(checkpoints) -> list:
    ck = [int(t) for t in checkpoints]
    if ck != sorted(ck) or ck[0] != 0 or len(set(ck)) != len(ck):
        raise ValueError(f"checkpoints must be strictly increasing from 0: {ck}")
    return [b - a for a, b in zip(ck[:-1], ck[1:])]


# Log-scale analytic MMD along the checkpoint grid
def log_mean_shift_path(A: np.ndarray, shift: np.ndarray, checkpoints) -> list:
    """[(ln ||A^t a||, A^t a / ||A^t a||) for t in checkpoints], without underflow.

    Gap powers are doubled with renormalization (A_g = s_g * Ahat_g, max|Ahat_g| = 1, ln s_g tracked).
    """
    gaps = grid_gaps(checkpoints)
    need = sorted({g for g in gaps if g > 0})
    if any(not _is_pow2(g) for g in need):
        raise ValueError(f"gaps must be powers of two, got {need}")
    scaled, Ah, ls, g = {}, A / np.abs(A).max(), math.log(np.abs(A).max()), 1
    while need and g <= need[-1]:
        if g > 1:
            M = Ah @ Ah
            n = float(np.abs(M).max())
            if not n > 0:
                raise FloatingPointError(f"scaled A^{g} vanished (nilpotent part?)")
            Ah, ls = M / n, 2.0 * ls + math.log(n)
        scaled[g] = (Ah.copy(), ls)
        g *= 2
    nrm = float(np.linalg.norm(shift))
    if not nrm > 0:
        raise ValueError("zero shift")
    v, ell = np.asarray(shift, dtype=np.float64) / nrm, math.log(nrm)
    out = [(ell, v.copy())]
    for gap in gaps:
        Ah_g, ls_g = scaled[gap]
        w = Ah_g @ v
        nw = float(np.linalg.norm(w))
        if not nw > 0:
            raise FloatingPointError(f"shift annihilated at gap {gap}")
        ell, v = ell + ls_g + math.log(nw), w / nw
        out.append((ell, v.copy()))
    return out


def log_mmd_path(kernel, C: np.ndarray, A: np.ndarray, shift: np.ndarray, checkpoints) -> list:
    """ln MMD(N(mu + A^t a, C), N(mu, C)) for each checkpoint t (natural log, finite everywhere).

    MMD^2 = 2 ||mu_pi||^2 (-expm1(-q/2)), q = b^T (Lambda + 2C)^{-1} b, b = A^t a. With b = e^ell v,
    ln q = 2 ell + ln(v^T (Lambda + 2C)^{-1} v); for ln q < -600, ln(-expm1(-q/2)) = ln(q/2) to
    float64 accuracy (relative correction q/4).
    """
    Lam = gl.lengthscale_matrix(kernel)
    P2 = np.linalg.inv(Lam + 2.0 * C)
    P2 = 0.5 * (P2 + P2.T)
    ln_n2 = math.log(gl.mean_embedding_norm2(kernel, C))
    res = []
    for ell, v in log_mean_shift_path(A, shift, checkpoints):
        ln_q = 2.0 * ell + math.log(float(v @ P2 @ v))
        if ln_q > -600.0:
            ln_x = math.log(-math.expm1(-0.5 * math.exp(ln_q)))
        else:
            ln_x = ln_q - math.log(2.0)
        res.append(0.5 * (math.log(2.0) + ln_n2 + ln_x))
    return res


def mmd_status(mmd_fp64: float, ln_mmd: float) -> str:
    """Numerical status of the analytic MMD at one checkpoint.

    The direct float64 evaluation goes through MMD^2 = 2 ||mu_pi||^2 (-expm1(-q/2)), so it loses precision
    once MMD^2 (not MMD) leaves the normal float64 range, i.e. for MMD below about 1.5e-154.
      RESOLVED:        MMD^2 is a normal float64; the direct value agrees with exp(ln MMD) (rel. 1e-6).
      MMD2_UNDERFLOW:  MMD^2 is below the normal range (the direct value is 0 or imprecise), but MMD itself
                       is a normal float64, recovered from the log-scale value (``mmd_from_log``).
      BELOW_FLOAT64:   MMD itself is below the normal float64 range; only ln MMD is reported.
    Neither of the last two is a claim of exact convergence: ln MMD is finite.
    """
    if not math.isfinite(ln_mmd):
        raise FloatingPointError(f"non-finite log-scale MMD {ln_mmd}")
    if mmd_fp64 * mmd_fp64 >= TINY:
        if abs(math.log(mmd_fp64) - ln_mmd) > 1e-6:
            raise FloatingPointError(f"float64 MMD {mmd_fp64!r} disagrees with log-scale exp({ln_mmd})")
        return "RESOLVED"
    return "MMD2_UNDERFLOW" if ln_mmd >= LN_TINY else "BELOW_FLOAT64"


def mmd_from_log(ln_mmd: float) -> float | None:
    """exp(ln MMD) when it is a normal float64, else None (only the log value is meaningful)."""
    return math.exp(ln_mmd) if ln_mmd >= LN_TINY else None
