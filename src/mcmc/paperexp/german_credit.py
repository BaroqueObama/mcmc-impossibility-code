"""German Credit data and hierarchical logistic posteriors.

The numeric encoding has 1000 rows and 24 predictors. Predictors are
standardized with population standard deviations (ddof=0); y=1 means good
credit (file label 1). The full dataset conditions the posterior.

Both models use alpha ~ N(0,10^2) and sigma ~ HalfCauchy(1). The hierarchical
model has beta_j | sigma ~ N(0,sigma^2). The horseshoe model adds independent
tau_j ~ HalfCauchy(1) with beta_j | sigma,tau_j ~ N(0,sigma^2 tau_j^2).
Coordinates are (alpha, log sigma, beta[, log tau]). Densities include the
log-Jacobian and all normalizing constants except the evidence; inputs are
(n,d) float64 tensors. The analytic score and NumPyro model use the same law.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import torch

UCI_DATASET_ID = 144
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw" / "german_credit"

# Downloaded 2026-10-05 from UCI_BASE_URL.
PINNED_SHA256 = {
    "german.data-numeric": "2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8",
    "german.data": "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
    "german.doc": "69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49",
}
N_OBS = 1000
N_PRED = 24
N_GOOD = 700

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2_OVER_PI = math.log(2.0 / math.pi)
INTERCEPT_PRIOR_SD = 10.0


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_raw_files(data_dir: pathlib.Path = DATA_DIR) -> dict[str, str]:
    """Check that every pinned file exists with its pinned SHA256; raise otherwise."""
    data_dir = pathlib.Path(data_dir)
    for name, expected in PINNED_SHA256.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"German Credit file {path} is missing; run download_raw_files() explicitly"
            )
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"SHA256 mismatch for {path}: got {got}, pinned {expected}")
    return dict(PINNED_SHA256)


def download_raw_files(data_dir: pathlib.Path = DATA_DIR, timeout_s: float = 60.0) -> None:
    """Download the pinned files from UCI and verify them. Never called implicitly by the loader."""
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    urls = {name: UCI_BASE_URL + name for name in PINNED_SHA256}
    for name, url in urls.items():
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            payload = resp.read()
        got = hashlib.sha256(payload).hexdigest()
        if got != PINNED_SHA256[name]:
            raise ValueError(f"downloaded {url} has SHA256 {got}, pinned {PINNED_SHA256[name]}")
        tmp = data_dir / f".{name}.part"
        tmp.write_bytes(payload)
        tmp.replace(data_dir / name)
    verify_raw_files(data_dir)


def parse_numeric_file(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Strictly parse ``german.data-numeric``: N_OBS rows of N_PRED + 1 integers."""
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != N_PRED + 1:
            raise ValueError(f"{path}:{lineno}: {len(tokens)} fields, expected {N_PRED + 1}")
        rows.append([int(tok) for tok in tokens])  # int() raises on anything non-integer
    data = np.asarray(rows, dtype=np.int64)
    if data.shape != (N_OBS, N_PRED + 1):
        raise ValueError(f"{path}: parsed shape {data.shape}, expected {(N_OBS, N_PRED + 1)}")
    label = data[:, -1]
    if not np.isin(label, (1, 2)).all():
        raise ValueError(f"{path}: labels outside {{1, 2}}: {np.unique(label)}")
    return data[:, :-1].astype(np.float64), label


@dataclass(frozen=True)
class GermanCreditData:
    X: np.ndarray  # (N_OBS, N_PRED) standardized predictors, float64
    y: np.ndarray  # (N_OBS,) float64 in {0, 1}; 1 = good credit (file label 1)
    raw_X: np.ndarray  # (N_OBS, N_PRED) unstandardized integer-valued predictors
    raw_label: np.ndarray  # (N_OBS,) file labels in {1, 2}
    col_mean: np.ndarray
    col_std: np.ndarray  # population standard deviations (ddof=0)
    sha256: dict[str, str] = field(default_factory=dict)


def load_german_credit(data_dir: pathlib.Path = DATA_DIR) -> GermanCreditData:
    """Verify and standardize the German Credit numeric data."""
    sha = verify_raw_files(data_dir)
    raw_X, raw_label = parse_numeric_file(pathlib.Path(data_dir) / "german.data-numeric")
    col_mean = raw_X.mean(axis=0)
    col_std = raw_X.std(axis=0, ddof=0)
    if not (col_std > 0).all():
        raise ValueError(f"constant predictor columns: {np.flatnonzero(col_std <= 0)}")
    X = (raw_X - col_mean) / col_std
    y = (raw_label == 1).astype(np.float64)
    assert X.shape == (N_OBS, N_PRED) and y.shape == (N_OBS,)
    assert np.isfinite(X).all()
    if int(y.sum()) != N_GOOD:
        raise ValueError(f"expected {N_GOOD} good credits, found {int(y.sum())}")
    return GermanCreditData(X, y, raw_X, raw_label, col_mean, col_std, sha)


def crosscheck_against_original(data: GermanCreditData, data_dir: pathlib.Path = DATA_DIR) -> dict:
    """Compare columns of the numeric encoding with the symbolic original ``german.data``.

    Checked: label, duration (numeric col 2 = attribute 2), age (col 10 = attribute 13),
    credit amount (col 4 = attribute 5 / 100, rounded) and checking-account status
    (col 1 = attribute 1, A1k -> k). Returns the per-check agreement; raises on disagreement.
    """
    rows = [ln.split() for ln in (pathlib.Path(data_dir) / "german.data").read_text().splitlines()]
    rows = [r for r in rows if r]
    if len(rows) != N_OBS or {len(r) for r in rows} != {21}:
        raise ValueError("german.data is not 1000 rows of 21 fields")
    orig = np.asarray(rows)
    checks = {
        "label": np.array_equal(orig[:, 20].astype(np.int64), data.raw_label),
        "duration": np.array_equal(orig[:, 1].astype(np.float64), data.raw_X[:, 1]),
        "age": np.array_equal(orig[:, 12].astype(np.float64), data.raw_X[:, 9]),
        "checking_status": np.array_equal(
            np.array([int(s[2:]) for s in orig[:, 0]], dtype=np.float64), data.raw_X[:, 0]
        ),
    }
    amount_err = float(np.max(np.abs(orig[:, 4].astype(np.float64) / 100.0 - data.raw_X[:, 3])))
    checks["amount_div100_max_abs_err"] = amount_err
    failed = [k for k, v in checks.items() if v is False]
    if failed or amount_err > 0.5:
        raise ValueError(f"numeric/original encodings disagree: {checks}")
    return checks


def _log1pexp(z: torch.Tensor) -> torch.Tensor:
    """Exact softplus log(1 + e^z) (torch's softplus switches to the identity above 20)."""
    return torch.clamp(z, min=0) + torch.log1p(torch.exp(-torch.abs(z)))


class _LogisticPosterior:
    """Shared logistic likelihood, with dimension set by the dataset's retained predictors."""

    name: str = ""
    d: int = 0
    n_pred: int = N_PRED

    def __init__(self, data=None, dataset: str = "german"):
        self.dataset = dataset
        self.data = load_data_for(dataset) if data is None else data
        n, p = self.data.X.shape
        if dataset == "german" and (n, p) != (N_OBS, N_PRED):
            raise ValueError(f"German Credit data X has shape {self.data.X.shape}")
        if self.data.y.shape != (n,):
            raise ValueError(f"y has shape {self.data.y.shape}, expected ({n},)")
        self.n_pred = p
        self.d = self._dim(p)
        if dataset != "german":
            self.name = f"{dataset}_{self.name.removeprefix('german_credit_')}"
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _dim(p: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def _xy(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (
                torch.as_tensor(self.data.X, dtype=torch.float64, device=device),
                torch.as_tensor(self.data.y, dtype=torch.float64, device=device),
            )
        return self._cache[key]

    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 2 or x.shape[1] != self.d:
            raise ValueError(f"{self.name}: expected (n, {self.d}) input, got {tuple(x.shape)}")
        if x.dtype != torch.float64:
            raise TypeError(f"{self.name}: expected float64 input, got {x.dtype}")
        if not torch.isfinite(x).all():
            raise ValueError(f"{self.name}: non-finite input")

    def _loglik_and_resid(
        self, alpha: torch.Tensor, beta: torch.Tensor, need_resid: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        X, y = self._xy(alpha.device)
        eta = alpha[:, None] + beta @ X.T  # (n, N_OBS)
        loglik = (y[None, :] * eta - _log1pexp(eta)).sum(dim=1)
        resid = (y[None, :] - torch.sigmoid(eta)) if need_resid else None
        return loglik, resid

    @staticmethod
    def _intercept_logprior(alpha: torch.Tensor) -> torch.Tensor:
        sd = INTERCEPT_PRIOR_SD
        return -0.5 * (alpha / sd) ** 2 - math.log(sd) - 0.5 * _LOG_2PI

    @staticmethod
    def _halfcauchy_logprior_logscale(s: torch.Tensor) -> torch.Tensor:
        """log HalfCauchy(1) density of exp(s) plus the log-Jacobian s."""
        return _LOG_2_OVER_PI - _log1pexp(2.0 * s) + s

    def coordinate_names(self) -> list[str]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "d": self.d,
            "coordinates": self.coordinate_names(),
            "dataset": self.dataset,
            "data_sha256": self.data.sha256[
                getattr(self.data, "primary_file", "") or "german.data-numeric"
            ],
            "intercept_prior_sd": INTERCEPT_PRIOR_SD,
        }

    @property
    def target_id(self) -> str:
        return self.name

    def _eval(self, x: torch.Tensor, want_grad: bool):  # pragma: no cover - abstract
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._eval(x, want_grad=False)[0]

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self._eval(x, want_grad=True)[1]

    def log_prob_and_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both from one logit evaluation (used by the batched MALA sampler)."""
        return self._eval(x, want_grad=True)

    def score_autograd(self, x: torch.Tensor) -> torch.Tensor:
        """Reference gradient via autograd (used to test the analytic ``score``)."""
        xr = x.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.log_prob(xr).sum(), xr)
        return g.detach()


class HierLogisticPosterior(_LogisticPosterior):
    """Centered hierarchical logistic regression, x = (alpha, log sigma, beta)."""

    name = "german_credit_hier_logistic"
    d = 2 + N_PRED

    @staticmethod
    def _dim(p: int) -> int:
        return 2 + p

    def coordinate_names(self) -> list[str]:
        return ["alpha", "log_sigma"] + [f"beta_{j + 1}" for j in range(self.n_pred)]

    def _eval(self, x: torch.Tensor, want_grad: bool):
        self._check(x)
        alpha, s, beta = x[:, 0], x[:, 1], x[:, 2:]
        p = self.n_pred
        inv_var = torch.exp(-2.0 * s)
        lp = self._intercept_logprior(alpha) + self._halfcauchy_logprior_logscale(s)
        lp = lp - 0.5 * inv_var * (beta**2).sum(dim=1) - p * s - 0.5 * p * _LOG_2PI
        loglik, resid = self._loglik_and_resid(alpha, beta, need_resid=want_grad)
        lp = lp + loglik
        assert lp.shape == (x.shape[0],)
        if not want_grad:
            return lp, None
        X, _ = self._xy(x.device)
        g = torch.empty_like(x)
        g[:, 0] = -alpha / INTERCEPT_PRIOR_SD**2 + resid.sum(dim=1)
        g[:, 1] = -2.0 * torch.sigmoid(2.0 * s) + 1.0 - p + inv_var * (beta**2).sum(dim=1)
        g[:, 2:] = -beta * inv_var[:, None] + resid @ X
        return lp, g


class HorseshoeLogisticPosterior(_LogisticPosterior):
    """Centered horseshoe logistic regression, x = (alpha, log sigma, beta, log tau)."""

    name = "german_credit_horseshoe_logistic"
    d = 2 + 2 * N_PRED

    @staticmethod
    def _dim(p: int) -> int:
        return 2 + 2 * p

    def coordinate_names(self) -> list[str]:
        p = self.n_pred
        return (
            ["alpha", "log_sigma"]
            + [f"beta_{j + 1}" for j in range(p)]
            + [f"log_tau_{j + 1}" for j in range(p)]
        )

    def _split(self, x: torch.Tensor):
        p = self.n_pred
        return x[:, 0], x[:, 1], x[:, 2 : 2 + p], x[:, 2 + p :]

    def _eval(self, x: torch.Tensor, want_grad: bool):
        self._check(x)
        alpha, s, beta, lt = self._split(x)
        p = self.n_pred
        log_scale = s[:, None] + lt  # log(sigma * tau_j)
        inv_var = torch.exp(-2.0 * log_scale)  # 1 / (sigma tau_j)^2
        r2 = beta**2 * inv_var  # (beta_j / (sigma tau_j))^2
        lp = self._intercept_logprior(alpha) + self._halfcauchy_logprior_logscale(s)
        lp = lp + self._halfcauchy_logprior_logscale(lt).sum(dim=1)
        lp = lp + (-0.5 * r2 - log_scale).sum(dim=1) - 0.5 * p * _LOG_2PI
        loglik, resid = self._loglik_and_resid(alpha, beta, need_resid=want_grad)
        lp = lp + loglik
        assert lp.shape == (x.shape[0],)
        if not want_grad:
            return lp, None
        X, _ = self._xy(x.device)
        g = torch.empty_like(x)
        g[:, 0] = -alpha / INTERCEPT_PRIOR_SD**2 + resid.sum(dim=1)
        g[:, 1] = -2.0 * torch.sigmoid(2.0 * s) + 1.0 - p + r2.sum(dim=1)
        g[:, 2 : 2 + p] = -beta * inv_var + resid @ X
        g[:, 2 + p :] = -2.0 * torch.sigmoid(2.0 * lt) + r2
        return lp, g


MODELS = {"hier": HierLogisticPosterior, "horseshoe": HorseshoeLogisticPosterior}
# Additional datasets use the hierarchical model: "hier_<dataset>".
EXTRA_DATASETS = ("pima", "australian", "ionosphere", "sonar")


def split_kind(kind: str) -> tuple[str, str]:
    """Model key -> (base model, data set): "hier"/"horseshoe" are German Credit."""
    if kind in MODELS:
        return kind, "german"
    base, _, ds = kind.partition("_")
    if base == "hier" and ds in EXTRA_DATASETS:
        return "hier", ds
    raise ValueError(f"unknown model {kind!r}; choose from {sorted(model_keys())}")


def model_keys() -> list[str]:
    return list(MODELS) + [f"hier_{ds}" for ds in EXTRA_DATASETS]


def load_data_for(dataset: str):
    if dataset == "german":
        return load_german_credit()
    from mcmc.paperexp import logistic_data

    return logistic_data.load_dataset(dataset)


def load_data(kind: str):
    return load_data_for(split_kind(kind)[1])


def model_dim(kind: str) -> int:
    base, ds = split_kind(kind)
    if ds == "german":
        p = N_PRED
    else:
        from mcmc.paperexp import logistic_data

        p = logistic_data.SPECS[ds].p
    return MODELS[base]._dim(p)


def make_posterior(kind: str, data=None) -> _LogisticPosterior:
    base, ds = split_kind(kind)
    return MODELS[base](data, ds)


# numpyro definitions (for NUTS references and for validating the torch log densities).
# "centered" uses exactly the sites of the torch model; "noncentered" replaces beta by
# z ~ N(0, I) with beta = sigma * z (hier) or beta = sigma * tau * z (horseshoe). Both define
# the same posterior law on the declared coordinates; see ``declared_from_numpyro``.


def numpyro_model(kind: str, centered: bool):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def hier(X, y):
        p = X.shape[1]
        alpha = numpyro.sample("alpha", dist.Normal(0.0, INTERCEPT_PRIOR_SD))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        if centered:
            beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), sigma).to_event(1))
        else:
            z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0).to_event(1))
            beta = sigma * z
        numpyro.sample("y", dist.Bernoulli(logits=alpha + X @ beta).to_event(1), obs=y)

    def horseshoe(X, y):
        p = X.shape[1]
        alpha = numpyro.sample("alpha", dist.Normal(0.0, INTERCEPT_PRIOR_SD))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(p)).to_event(1))
        if centered:
            beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), sigma * tau).to_event(1))
        else:
            z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0).to_event(1))
            beta = sigma * tau * z
        numpyro.sample("y", dist.Bernoulli(logits=alpha + X @ beta).to_event(1), obs=y)

    base = split_kind(kind)[0]
    return hier if base == "hier" else horseshoe


def numpyro_unconstrained_from_declared(kind: str, x: np.ndarray, centered: bool) -> dict:
    """Map one declared vector x (d,) to numpyro's unconstrained site dict for that model."""
    x = np.asarray(x, dtype=np.float64)
    base = split_kind(kind)[0]
    if x.ndim != 1 or x.shape[0] != model_dim(kind):
        raise ValueError(f"{kind}: expected a ({model_dim(kind)},) vector, got {x.shape}")
    p = x.shape[0] - 2 if base == "hier" else (x.shape[0] - 2) // 2
    alpha, s, beta = x[0], x[1], x[2 : 2 + p]
    out = {"alpha": alpha, "sigma": s}
    if base == "hier":
        scale = np.exp(s)
    else:
        lt = x[2 + p :]
        out["tau"] = lt
        scale = np.exp(s + lt)
    if centered:
        out["beta"] = beta
    else:
        out["z"] = beta / scale
    return out


def declared_from_numpyro(kind: str, samples: dict, centered: bool) -> np.ndarray:
    """Map numpyro constrained samples (leading batch shape B) to declared coordinates (B..., d)."""
    alpha = np.asarray(samples["alpha"], dtype=np.float64)
    sigma = np.asarray(samples["sigma"], dtype=np.float64)
    if not (np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("non-finite or non-positive sigma draws")
    s = np.log(sigma)
    parts = [alpha[..., None], s[..., None]]
    kind = split_kind(kind)[0]
    if kind == "horseshoe":
        tau = np.asarray(samples["tau"], dtype=np.float64)
        if not (np.isfinite(tau).all() and (tau > 0).all()):
            raise ValueError("non-finite or non-positive tau draws")
        scale = sigma[..., None] * tau
    elif kind == "hier":
        tau = None
        scale = sigma[..., None]
    else:
        raise ValueError(f"unknown logistic model {kind!r}")
    beta = np.asarray(samples["beta"]) if centered else scale * np.asarray(samples["z"])
    parts.append(beta.astype(np.float64))
    if tau is not None:
        parts.append(np.log(tau))
    out = np.concatenate(parts, axis=-1)
    if not np.isfinite(out).all():
        raise ValueError("non-finite declared coordinates")
    return out


def numpyro_potential_energy(kind: str, data: GermanCreditData, x: np.ndarray, centered: bool):
    """numpyro potential energy (-log density in numpyro's unconstrained sites) at declared x.

    For ``centered=True`` this equals ``-log_prob(x)``. For ``centered=False`` the sites are
    (alpha, log sigma, z[, log tau]) and the energy differs from ``-log_prob(x)`` by the
    log-Jacobian of beta = scale * z, i.e. sum_j log(scale_j).
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import potential_energy

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("enable float64 in JAX (jax.config.update('jax_enable_x64', True))")
    params = {
        k: jnp.asarray(v, dtype=jnp.float64)
        for k, v in numpyro_unconstrained_from_declared(kind, x, centered).items()
    }
    X = jnp.asarray(data.X, dtype=jnp.float64)
    y = jnp.asarray(data.y, dtype=jnp.float64)
    return float(potential_energy(numpyro_model(kind, centered), (X, y), {}, params))
