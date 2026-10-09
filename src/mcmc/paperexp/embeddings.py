"""RBF target mean embeddings, in closed form or by quadrature.

Gaussian mixtures use analytic Gaussian integrals. Student-t targets integrate
over W ~ chi^2_nu with conditional variance nu/W; funnels integrate over the
Gaussian scale coordinate. Quadrature order doubles until relative changes
in the embedding norm and probe values are below QUAD_RTOL. Probe points
must cover the region where the embedding will be evaluated.

Other kernels use Monte Carlo estimates with standard errors.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special as sps
import torch

from mcmc.paperexp.kernels import DEFAULT_MEM_BYTES, ScaledRBF
from mcmc.paperexp.targets import (
    FunnelTarget,
    GaussTarget,
    MixtureTarget,
    StudentTTarget,
    Target,
)

F64 = torch.float64
QUAD_RTOL = 1e-12
_MU_FLOOR = 1e-100
_LOG_DENSITY_CUT = math.log(1e-35)


class MeanEmbedding:
    """mu_pi for one (target, ScaledRBF) pair. ``emb(X)`` -> (n,) float64; ``emb.norm2()``."""

    method: str = ""

    def __init__(self, target: Target, kernel: ScaledRBF):
        if not isinstance(kernel, ScaledRBF):
            raise TypeError(
                f"exact embeddings exist only for ScaledRBF, got {type(kernel).__name__}"
            )
        if kernel.d != target.d:
            raise ValueError(f"kernel d={kernel.d} vs target d={target.d}")
        self.target = target
        self.kernel = kernel
        self.lam = (kernel.scales * kernel.ell) ** 2
        self.info: dict = {"method": self.method}

    def __call__(self, X: torch.Tensor, mem_bytes: int = DEFAULT_MEM_BYTES) -> torch.Tensor:
        self.target._check(X)
        out = torch.empty(X.shape[0], dtype=F64, device=X.device)
        rows = max(1, mem_bytes // (8 * 4 * max(1, self._width())))
        for i0 in range(0, X.shape[0], rows):
            out[i0 : i0 + rows] = self._eval(X[i0 : i0 + rows])
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite mean embedding values")
        return out

    def norm2(self) -> float:
        return self._norm2

    def to_dict(self) -> dict:
        return {
            "target_id": self.target.target_id,
            "kernel": self.kernel.to_dict(),
            "norm2": self._norm2,
            **self.info,
        }

    def _width(self) -> int:
        return 1

    def _eval(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


# closed form: finite Gaussian mixtures (gauss, mixture)
class GaussianMixtureEmbedding(MeanEmbedding):
    method = "closed_form"

    def __init__(self, target, kernel, weights, means, covs):
        super().__init__(target, kernel)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covs = np.asarray(covs, dtype=np.float64)
        d = target.d
        assert self.means.shape == (len(self.weights), d)
        assert self.covs.shape == (len(self.weights), d, d)
        assert abs(self.weights.sum() - 1.0) < 1e-15
        lam = self.lam
        isq = 1.0 / np.sqrt(lam)
        self._prec, self._logpref = [], []
        for C in self.covs:
            B = np.eye(d) + isq[:, None] * C * isq[None, :]  # Lam^{-1/2} (Lam + C) Lam^{-1/2}
            sign, logdet = np.linalg.slogdet(B)
            assert sign > 0
            self._logpref.append(-0.5 * logdet)
            self._prec.append(np.linalg.inv(np.diag(lam) + C))
        n2 = 0.0
        for a in range(len(self.weights)):
            for b in range(len(self.weights)):
                n2 += (
                    self.weights[a]
                    * self.weights[b]
                    * _gauss_pair(lam, self.means[a] - self.means[b], self.covs[a] + self.covs[b])
                )
        self._norm2 = float(n2)
        self._dev: dict[str, tuple] = {}
        self.info.update({"abs_error": "roundoff only (closed form)"})

    def _width(self) -> int:
        return len(self.weights) * self.target.d

    def _eval(self, X):
        key = str(X.device)
        if key not in self._dev:
            self._dev[key] = (
                torch.as_tensor(self.means, dtype=F64, device=X.device),
                torch.as_tensor(np.stack(self._prec), dtype=F64, device=X.device),
            )
        means, precs = self._dev[key]
        out = torch.zeros(X.shape[0], dtype=F64, device=X.device)
        for c in range(len(self.weights)):
            r = X - means[c]
            q = ((r @ precs[c]) * r).sum(1)
            out += self.weights[c] * torch.exp(self._logpref[c] - 0.5 * q)
        return out


def _gauss_pair(lam: np.ndarray, delta: np.ndarray, Csum: np.ndarray) -> float:
    d = lam.shape[0]
    isq = 1.0 / np.sqrt(lam)
    B = np.eye(d) + isq[:, None] * Csum * isq[None, :]
    sign, logdet = np.linalg.slogdet(B)
    assert sign > 0
    quad = float(delta @ np.linalg.solve(np.diag(lam) + Csum, delta))
    return math.exp(-0.5 * logdet - 0.5 * quad)


# Quadrature
class _QuadratureEmbedding(MeanEmbedding):
    method = "quadrature"
    rule_name = ""
    _dev: dict

    def __init__(self, target, kernel, probe_X: np.ndarray, n0: int, n_max: int):
        super().__init__(target, kernel)
        probe_X = np.asarray(probe_X, dtype=np.float64)
        if probe_X.ndim != 2 or probe_X.shape[1] != target.d:
            raise ValueError(f"probe_X shape {probe_X.shape}")
        Xp = torch.as_tensor(probe_X, dtype=F64)
        n = n0
        self._set_rule(n)
        prev_mu, prev_n2 = self._eval(Xp).numpy(), self._norm2_rule()
        history = []
        while True:
            n *= 2
            if n > n_max:
                raise RuntimeError(
                    f"{self.rule_name} quadrature for {target.target_id} did not converge to "
                    f"{QUAD_RTOL} by order {n_max}; history {history}"
                )
            self._set_rule(n)
            mu, n2 = self._eval(Xp).numpy(), self._norm2_rule()
            keep = np.abs(mu) > _MU_FLOOR
            if not keep.any():
                raise ValueError("no probe point has mu above the floor; probe set is useless")
            rel_mu = float(np.max(np.abs(mu[keep] - prev_mu[keep]) / np.abs(mu[keep])))
            rel_n2 = abs(n2 - prev_n2) / abs(n2)
            history.append({"order": n, "rel_change_mu": rel_mu, "rel_change_norm2": rel_n2})
            prev_mu, prev_n2 = mu, n2
            if rel_mu < QUAD_RTOL and rel_n2 < QUAD_RTOL:
                break
        self._norm2 = float(n2)
        self.info.update(
            {
                "rule": self.rule_name,
                "order": n,
                "rel_change_mu_last_doubling": rel_mu,
                "rel_change_norm2_last_doubling": rel_n2,
                "doubling_history": history,
                "n_probe": int(probe_X.shape[0]),
                "n_probe_above_floor": int(keep.sum()),
                "probe_mu_floor": _MU_FLOOR,
            }
        )

    def _width(self) -> int:
        return self._nodes.shape[0]

    def _dev_rule(self, device):
        """Device copies of the current rule (reset by ``_set_rule``)."""
        key = str(device)
        if key not in self._dev:
            self._dev[key] = self._device_payload(device)
        return self._dev[key]


def _chunked_pair_sum(w: np.ndarray, pair_log: callable, rows: int = 256) -> float:
    """sum_ab w_a w_b exp(pair_log(a_slice, :)) over row chunks (bounded memory)."""
    total = 0.0
    for a0 in range(0, w.shape[0], rows):
        sl = slice(a0, a0 + rows)
        total += float((w[sl, None] * w[None, :] * np.exp(pair_log(sl))).sum())
    return total


class StudentTEmbedding(_QuadratureEmbedding):
    """Gaussian scale mixture over W ~ chi^2_nu, tau = nu/W.

    Trapezoid integration in log W stops where density is below 1e-35 of
    its peak; bounded integrands give truncation error below 1e-33.
    """

    def __init__(
        self, target: StudentTTarget, kernel, probe_X, rule="trapezoid_logW", n0=64, n_max=1 << 14
    ):
        if rule != "trapezoid_logW":
            raise ValueError(f"unknown rule {rule!r}")
        self.rule_name = rule
        nu = target.nu
        self._nu = nu
        logc = -(nu / 2) * math.log(2.0) - math.lgamma(nu / 2)

        def logp(v):
            return (nu / 2) * v - 0.5 * np.exp(v) + logc

        v_mode = math.log(nu)
        top = logp(v_mode)
        v_lo = v_mode
        while logp(v_lo) - top > _LOG_DENSITY_CUT:
            v_lo -= 0.25
        v_hi = v_mode
        while logp(v_hi) - top > _LOG_DENSITY_CUT:
            v_hi += 0.25
        self._logp, self._v_lo, self._v_hi = logp, v_lo, v_hi
        super().__init__(target, kernel, probe_X, n0, n_max)
        self.info.update({"v_range": [v_lo, v_hi], "density_cut_rel": 1e-35})

    def _set_rule(self, n):
        v = np.linspace(self._v_lo, self._v_hi, n + 1)
        h = v[1] - v[0]
        w = h * np.exp(self._logp(v))
        w[0] *= 0.5
        w[-1] *= 0.5
        tau = self._nu * np.exp(-v)
        keep = w > 0
        self._nodes, self._weights = tau[keep], w[keep]
        self._dev = {}

    def _device_payload(self, device):
        tau = self._nodes
        lam = self.lam
        logpref = -0.5 * np.log1p(tau[None, :] / lam[:, None]).sum(0)  # (K,)
        inv = 1.0 / (lam[:, None] + tau[None, :])  # (d, K)
        return (
            torch.as_tensor(inv, dtype=F64, device=device),
            torch.as_tensor(logpref + np.log(self._weights), dtype=F64, device=device),
        )

    def _eval(self, X):
        inv, logw = self._dev_rule(X.device)
        expo = -0.5 * (X * X) @ inv + logw  # (n, K)
        return torch.exp(expo).sum(1)

    def _norm2_rule(self):
        tau, lam = self._nodes, self.lam

        def pair_log(sl):
            s = tau[sl, None] + tau[None, :]
            return -0.5 * np.log1p(s[..., None] / lam).sum(-1)

        return _chunked_pair_sum(self._weights, pair_log)


class FunnelEmbedding(_QuadratureEmbedding):
    """Quadrature over y ~ N(0, sigma_y^2); x | y Gaussian handled in closed form.

    rule "trapezoid_y" (default): trapezoid on |y| <= 12.7 sigma_y (density > 1e-35 of its max);
    the integrand is analytic in the strip |Im y| < pi, so the rule converges geometrically.
    rule "gauss_hermite_y": probabilists' Gauss-Hermite, y = sigma_y t (independent cross-check;
    needs ~4096 nodes for 1e-12 because of the singularities at Im y = +-pi).
    """

    def __init__(
        self, target: FunnelTarget, kernel, probe_X, rule="trapezoid_y", n0=32, n_max=1 << 13
    ):
        if rule not in ("trapezoid_y", "gauss_hermite_y"):
            raise ValueError(f"unknown rule {rule!r}")
        self.rule_name = rule
        self._sig = target.sigma_y
        self._L = self._sig * math.sqrt(-2.0 * _LOG_DENSITY_CUT)
        super().__init__(target, kernel, probe_X, n0, n_max)
        if rule == "trapezoid_y":
            self.info.update({"y_range": [-self._L, self._L], "density_cut_rel": 1e-35})

    def _set_rule(self, n):
        if self.rule_name == "trapezoid_y":
            y = np.linspace(-self._L, self._L, n + 1)
            h = y[1] - y[0]
            w = h * np.exp(-0.5 * (y / self._sig) ** 2) / (self._sig * math.sqrt(2 * math.pi))
            w[0] *= 0.5
            w[-1] *= 0.5
        else:
            t, w = sps.roots_hermitenorm(n)
            y = self._sig * t
            w = w / math.sqrt(2 * math.pi)
        assert abs(w.sum() - 1.0) < 1e-12, w.sum()
        keep = w > 0
        self._nodes, self._weights = y[keep], w[keep]
        self._dev = {}

    def _device_payload(self, device):
        y, lam = self._nodes, self.lam
        lx = lam[1:]
        ey = np.exp(y)
        logpref = -0.5 * np.log1p(ey[None, :] / lx[:, None]).sum(0)
        inv = 1.0 / (lx[:, None] + ey[None, :])  # (d-1, K)
        return (
            torch.as_tensor(inv, dtype=F64, device=device),
            torch.as_tensor(
                np.stack([logpref + np.log(self._weights), y]), dtype=F64, device=device
            ),
        )

    def _eval(self, X):
        inv, aux = self._dev_rule(X.device)
        logw, y = aux[0], aux[1]
        a0 = X[:, :1]
        ax = X[:, 1:]
        expo = -0.5 * (a0 - y[None, :]) ** 2 / self.lam[0] - 0.5 * (ax * ax) @ inv + logw
        return torch.exp(expo).sum(1)

    def _norm2_rule(self):
        y, lam = self._nodes, self.lam
        ey = np.exp(y)

        def pair_log(sl):
            s = ey[sl, None] + ey[None, :]
            return -0.5 * (y[sl, None] - y[None, :]) ** 2 / lam[0] - 0.5 * np.log1p(
                s[..., None] / lam[1:]
            ).sum(-1)

        return _chunked_pair_sum(self._weights, pair_log)


# factory
def default_probe_points(target: Target, n: int = 64, seed: int = 12345) -> np.ndarray:
    """Probe set for quadrature convergence: target draws, the origin, and 2x, 4x scaled draws."""
    from mcmc.paperexp.rng import torch_gen

    g = torch_gen(seed, "cpu")
    x = target.sample(n, g)
    pts = torch.cat([x, torch.zeros(1, target.d, dtype=F64), 2 * x[: n // 4], 4 * x[: n // 8]])
    return pts.numpy()


def rbf_mean_embedding(
    target: Target, kernel: ScaledRBF, probe_X: np.ndarray | None = None, rule: str | None = None
) -> MeanEmbedding:
    """Target mean embedding for ScaledRBF, using closed forms or quadrature."""
    if isinstance(target, GaussTarget):
        return GaussianMixtureEmbedding(
            target, kernel, [1.0], np.zeros((1, target.d)), target.covariance()[None]
        )
    if isinstance(target, MixtureTarget):
        eye = np.eye(target.d)
        return GaussianMixtureEmbedding(
            target,
            kernel,
            target.component_weights(),
            target.component_means(),
            np.stack([eye, eye]),
        )
    probe = default_probe_points(target) if probe_X is None else probe_X
    if isinstance(target, StudentTTarget):
        return StudentTEmbedding(target, kernel, probe, rule=rule or "trapezoid_logW")
    if isinstance(target, FunnelTarget):
        return FunnelEmbedding(target, kernel, probe, rule=rule or "trapezoid_y")
    raise TypeError(f"no embedding for {type(target).__name__}")


# Monte Carlo (any kernel; labelled estimates with standard errors)
def mc_mean_embedding(
    kernel,
    target: Target,
    X: torch.Tensor,
    n_draws: int,
    gen: torch.Generator,
    chunk: int = 1 << 20,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict:
    """Monte Carlo mu_pi(X) with per-point standard errors (IID target draws on gen.device)."""
    n = X.shape[0]
    s1 = torch.zeros(n, dtype=F64, device=X.device)
    s2 = torch.zeros(n, dtype=F64, device=X.device)
    done = 0
    rows = max(1, mem_bytes // (8 * 4 * chunk))
    while done < n_draws:
        m = min(chunk, n_draws - done)
        Y = target.sample(m, gen).to(X.device)
        for i0 in range(0, n, rows):
            K = kernel(X[i0 : i0 + rows], Y)
            s1[i0 : i0 + rows] += K.sum(1)
            s2[i0 : i0 + rows] += (K * K).sum(1)
        done += m
    mean = s1 / n_draws
    var = (s2 / n_draws - mean**2) * n_draws / (n_draws - 1)
    se = torch.sqrt(var.clamp_min(0.0) / n_draws)
    return {"method": "monte_carlo", "estimate": mean, "se": se, "n_draws": int(n_draws)}


def mc_mean_embedding_norm2(
    kernel, target: Target, n_pairs: int, gen: torch.Generator, chunk: int = 1 << 20
) -> dict:
    """Monte Carlo E k(Y, Y') from independent pairs (IID terms in [0, kappa2]).

    Returns estimate, se, a 95% normal CI and a 95% Hoeffding CI (terms bounded in [0, 1]).
    """
    s1 = s2 = 0.0
    done = 0
    while done < n_pairs:
        m = min(chunk, n_pairs - done)
        Y = target.sample(m, gen)
        Z = target.sample(m, gen)
        inv = kernel._inv(Y.device)
        d2 = (((Y - Z) * inv) ** 2).sum(1)
        kv = kernel.from_sqdist(d2)
        s1 += kv.sum().item()
        s2 += (kv * kv).sum().item()
        done += m
    mean = s1 / n_pairs
    var = (s2 / n_pairs - mean**2) * n_pairs / (n_pairs - 1)
    se = math.sqrt(max(var, 0.0) / n_pairs)
    hoeff = math.sqrt(math.log(2 / 0.05) / (2 * n_pairs))
    return {
        "method": "monte_carlo",
        "estimate": mean,
        "se": se,
        "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
        "ci95_hoeffding": [mean - hoeff, mean + hoeff],
        "n_pairs": int(n_pairs),
    }


# Independent adaptive-quadrature comparison.
def quadpack_mu(target: Target, kernel: ScaledRBF, x: np.ndarray) -> tuple[float, float]:
    """mu_pi(x) by adaptive Gauss-Kronrod (scipy.integrate.quad) for student_t / funnel.

    Independent of the trapezoid rules above (different algorithm and variable): student_t
    integrates over W ~ chi^2_nu directly; funnel integrates over y on |y| <= 12.7 sigma_y with
    a breakpoint at y = x_0. Returns (value, quad's absolute error estimate).
    """
    from scipy import integrate, stats

    lam = (kernel.scales * kernel.ell) ** 2
    x = np.asarray(x, dtype=np.float64)
    if isinstance(target, StudentTTarget):
        nu = target.nu

        def f(w):
            tau = nu / w
            return math.exp(
                -0.5 * np.log1p(tau / lam).sum() - 0.5 * (x**2 / (lam + tau)).sum()
            ) * stats.chi2.pdf(w, nu)

        val, err = integrate.quad(f, 0, np.inf, epsabs=0, epsrel=1e-13, limit=2000)
        return val, err
    if isinstance(target, FunnelTarget):
        sig = target.sigma_y
        L = sig * math.sqrt(-2.0 * _LOG_DENSITY_CUT)

        def f(y):
            ey = math.exp(y)
            return math.exp(
                -0.5 * (y / sig) ** 2
                - 0.5 * (x[0] - y) ** 2 / lam[0]
                - 0.5 * np.log1p(ey / lam[1:]).sum()
                - 0.5 * (x[1:] ** 2 / (lam[1:] + ey)).sum()
            ) / (sig * math.sqrt(2 * math.pi))

        pts = [float(np.clip(x[0], -L + 1e-9, L - 1e-9))]
        val, err = integrate.quad(f, -L, L, points=pts, epsabs=0, epsrel=1e-13, limit=2000)
        return val, err
    raise TypeError(f"quadpack_mu is for student_t / funnel, got {type(target).__name__}")
