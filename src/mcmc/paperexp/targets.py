"""Gaussian, Gaussian-mixture, Student-t and funnel targets.

Densities, scores and samplers use (n, d) float64 tensors. Each target defines
an event A with pi(A) = 1/2, giving the initial law d pi_0/d pi =
1 + eps (2 1_A - 1) and radius sqrt(chi^2(pi_0 || pi)) = eps.
Rotations are shared within each family and dimension, and are saved in
configs/paperexp/rotations_v1.npz.
"""

from __future__ import annotations

import math
import pathlib

import mpmath
import numpy as np
import torch

from mcmc.paperexp.rng import np_gen, seed_for

F64 = torch.float64
LOG_2PI = math.log(2.0 * math.pi)
_MP_DPS = 50


def haar_rotation(d: int, seed: int) -> np.ndarray:
    """Haar-distributed rotation in SO(d) (QR of a Gaussian matrix with the Mezzadri sign fix)."""
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    g = np_gen(seed).standard_normal((d, d))
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    err = np.abs(q.T @ q - np.eye(d)).max()
    assert err < 1e-12, f"rotation not orthogonal: {err}"
    assert np.linalg.det(q) > 0
    return q


def rotation_seed(family: str, d: int) -> int:
    return seed_for(f"{family}-d{d}", "rotation", 0)


def _fmt(v: float) -> str:
    return f"{v:g}"


class Target:
    """Shared parameter caching and rejection sampling of the initial law."""

    family: str = ""
    prob_A: float = 0.5

    def __init__(self, d: int):
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self._np_params: dict[str, np.ndarray] = {}
        self._dev_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _register(self, name: str, value) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite parameter {name}")
        self._np_params[name] = arr

    def _p(self, name: str, device: torch.device) -> torch.Tensor:
        key = (name, str(device))
        if key not in self._dev_cache:
            self._dev_cache[key] = torch.as_tensor(self._np_params[name], dtype=F64, device=device)
        return self._dev_cache[key]

    def _check(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(x)}")
        if x.dtype != F64:
            raise TypeError(f"expected float64 input, got {x.dtype}")
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"expected shape (n, {self.d}), got {tuple(x.shape)}")

    @property
    def name(self) -> str:
        return self.target_id

    @property
    def config_id(self) -> str:
        return self.target_id

    @property
    def target_id(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def event_A(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def mean(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def covariance(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def coordinate_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance()))

    def to_dict(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def log_prob_and_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_prob(x), self.score(x)

    def _randn(self, n: int, k: int, gen: torch.Generator) -> torch.Tensor:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return torch.randn((n, k), generator=gen, dtype=F64, device=gen.device)

    def r_init(self, eps: float) -> float:
        """chi^2(pi0 || pi)^{1/2} for d pi0/d pi = 1 + eps (2 1_A - 1) with pi(A) = 1/2."""
        _check_eps(eps)
        return float(eps)

    def init_density_ratio(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """d pi0 / d pi at x."""
        _check_eps(eps)
        sign = 2.0 * self.event_A(x).to(F64) - 1.0
        return 1.0 + eps * sign

    def sample_init(self, n: int, eps: float, gen: torch.Generator) -> torch.Tensor:
        """Exact draws from pi0 by rejection from pi: accept w.p. (1 + eps(2 1_A - 1))/(1 + eps).

        Accepted proposals are kept in proposal order and the first n are returned, which is an
        IID sample from pi0 (overall acceptance 1/(1+eps)).
        """
        _check_eps(eps)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        chunks: list[torch.Tensor] = []
        got = 0
        while got < n:
            m = int(math.ceil((n - got) * (1.0 + eps) * 1.15)) + 64
            prop = self.sample(m, gen)
            u = torch.rand((m,), generator=gen, dtype=F64, device=gen.device)
            acc = u * (1.0 + eps) < self.init_density_ratio(prop, eps)
            chunks.append(prop[acc])
            got += int(acc.sum().item())
        out = torch.cat(chunks, dim=0)[:n]
        assert out.shape == (n, self.d)
        return out


def _check_eps(eps: float) -> None:
    if not (0.0 <= float(eps) <= 1.0):
        raise ValueError(f"eps must be in [0, 1], got {eps}")


class GaussTarget(Target):
    """N(0, Q diag(lam) Q^T), lam = geomspace(sqrt(cond), 1/sqrt(cond), d).

    A is the positive half-space along the narrowest or widest principal axis.
    """
    family = "gauss"

    def __init__(
        self,
        d: int,
        cond: float,
        rotation: np.ndarray,
        rot_seed: int | None = None,
        event_dir: str = "narrow",
    ):
        super().__init__(d)
        if cond < 1.0:
            raise ValueError(f"condition number must be >= 1, got {cond}")
        if event_dir not in GAUSS_EVENT_DIRS:
            raise ValueError(f"event_dir must be one of {GAUSS_EVENT_DIRS}, got {event_dir!r}")
        rotation = np.asarray(rotation, dtype=np.float64)
        _check_rotation(rotation, d)
        self.cond = float(cond)
        self.rot_seed = rot_seed
        self.event_dir = event_dir
        lam = np.geomspace(self.cond**0.5, self.cond**-0.5, d)
        self._register("lam", lam)
        self._register("Q", rotation)
        # narrow: eigvec of the smallest eigenvalue (last column); wide: of the largest (first)
        self._register("q_event", rotation[:, -1] if event_dir == "narrow" else rotation[:, 0])

    @property
    def target_id(self) -> str:
        suffix = "" if self.event_dir == "narrow" else f"-e{self.event_dir}"
        return f"gauss-d{self.d}-c{_fmt(self.cond)}{suffix}"

    @property
    def lam(self) -> np.ndarray:
        return self._np_params["lam"]

    @property
    def rotation(self) -> np.ndarray:
        return self._np_params["Q"]

    def log_prob(self, x):
        self._check(x)
        lam = self._p("lam", x.device)
        z = x @ self._p("Q", x.device)
        return (
            -0.5 * (z * z / lam).sum(1)
            - 0.5 * float(np.log(self.lam).sum())
            - 0.5 * self.d * LOG_2PI
        )

    def score(self, x):
        self._check(x)
        q = self._p("Q", x.device)
        return -((x @ q) / self._p("lam", x.device)) @ q.T

    def sample(self, n, gen):
        xi = self._randn(n, self.d, gen)
        return (xi * torch.sqrt(self._p("lam", gen.device))) @ self._p("Q", gen.device).T

    def event_A(self, x):
        self._check(x)
        return (x @ self._p("q_event", x.device)) > 0

    def mean(self):
        return np.zeros(self.d)

    def covariance(self):
        q = self.rotation
        return (q * self.lam[None, :]) @ q.T

    def to_dict(self):
        return {
            "family": self.family,
            "target_id": self.target_id,
            "d": self.d,
            "cond": self.cond,
            "lam": self.lam.tolist(),
            "rotation_seed": self.rot_seed,
            "rotation": self.rotation.tolist(),
            "event_dir": self.event_dir,
            "event": (
                "q_min^T x > 0, q_min = rotation[:, -1] (eigvec of smallest eigenvalue)"
                if self.event_dir == "narrow"
                else "q_max^T x > 0, q_max = rotation[:, 0] (eigvec of largest eigenvalue)"
            ),
            "prob_A": 0.5,
            "prob_A_abs_error": 0.0,
        }


GAUSS_EVENT_DIRS = ("narrow", "wide")


def mixture_event_threshold(sep: float, w: float) -> tuple[float, float]:
    """c* with w Phi_c(c + s/2) + (1-w) Phi_c(c - s/2) = 1/2 and |pi(A) - 1/2| at the double c*."""
    if w == 0.5:
        return 0.0, 0.0
    with mpmath.workdps(_MP_DPS):
        s, ww = mpmath.mpf(sep), mpmath.mpf(w)

        def p_above(c):
            return ww * mpmath.ncdf(-(c + s / 2)) + (1 - ww) * mpmath.ncdf(-(c - s / 2))

        root = mpmath.findroot(
            lambda c: p_above(c) - mpmath.mpf("0.5"),
            (-s / 2 - 40, s / 2 + 40),
            solver="illinois",
        )
        c_double = float(root)
        err = abs(float(p_above(mpmath.mpf(c_double)) - mpmath.mpf("0.5")))
    if err > 1e-14:
        raise RuntimeError(f"mixture event threshold inaccurate: |pi(A)-1/2| = {err}")
    return c_double, err


class MixtureTarget(Target):
    """Q z with z ~ w N(-(sep/2)e1, I) + (1-w) N((sep/2)e1, I).

    A = {z_1 > c_star}, where c_star is the mixture median.
    """
    family = "mixture"

    def __init__(
        self, d: int, sep: float, w: float, rotation: np.ndarray, rot_seed: int | None = None
    ):
        super().__init__(d)
        if not 0.0 < w < 1.0:
            raise ValueError(f"weight must be in (0,1), got {w}")
        if sep < 0:
            raise ValueError(f"separation must be >= 0, got {sep}")
        rotation = np.asarray(rotation, dtype=np.float64)
        _check_rotation(rotation, d)
        self.sep = float(sep)
        self.w = float(w)
        self.rot_seed = rot_seed
        self.c_star, self.prob_A_abs_error = mixture_event_threshold(self.sep, self.w)
        self._register("Q", rotation)
        self._register("q1", rotation[:, 0])

    @property
    def target_id(self) -> str:
        return f"mixture-d{self.d}-s{_fmt(self.sep)}-w{_fmt(self.w)}"

    @property
    def rotation(self) -> np.ndarray:
        return self._np_params["Q"]

    def component_means(self) -> np.ndarray:
        """(2, d): row 0 = -(s/2) q1 (weight w), row 1 = +(s/2) q1 (weight 1-w)."""
        q1 = self.rotation[:, 0]
        return np.stack([-(self.sep / 2) * q1, (self.sep / 2) * q1])

    def component_weights(self) -> np.ndarray:
        return np.array([self.w, 1.0 - self.w])

    def _log_terms(self, z1: torch.Tensor) -> torch.Tensor:
        h = self.sep / 2
        return torch.stack(
            [math.log(self.w) - 0.5 * (z1 + h) ** 2, math.log1p(-self.w) - 0.5 * (z1 - h) ** 2],
            dim=1,
        )

    def log_prob(self, x):
        self._check(x)
        z = x @ self._p("Q", x.device)
        rest = (z[:, 1:] ** 2).sum(1)
        return (
            torch.logsumexp(self._log_terms(z[:, 0]), dim=1) - 0.5 * rest - 0.5 * self.d * LOG_2PI
        )

    def score(self, x):
        self._check(x)
        q = self._p("Q", x.device)
        z = x @ q
        resp = torch.softmax(self._log_terms(z[:, 0]), dim=1)  # (n, 2): (minus, plus)
        g = -z
        g[:, 0] = g[:, 0] + (resp[:, 1] - resp[:, 0]) * (self.sep / 2)
        return g @ q.T

    def sample(self, n, gen):
        z = self._randn(n, self.d, gen)
        u = torch.rand((n,), generator=gen, dtype=F64, device=gen.device)
        sign = torch.where(u < self.w, -1.0, 1.0).to(F64)
        z[:, 0] = z[:, 0] + sign * (self.sep / 2)
        return z @ self._p("Q", gen.device).T

    def event_A(self, x):
        self._check(x)
        return (x @ self._p("q1", x.device)) > self.c_star

    def mean(self):
        return self.component_weights() @ self.component_means()

    def covariance(self):
        q1 = self.rotation[:, 0]
        return np.eye(self.d) + self.w * (1 - self.w) * self.sep**2 * np.outer(q1, q1)

    def to_dict(self):
        return {
            "family": self.family,
            "target_id": self.target_id,
            "d": self.d,
            "sep": self.sep,
            "w": self.w,
            "weight_convention": "w on N(-(s/2) e1, I), 1-w on N(+(s/2) e1, I), then x = Q z",
            "rotation_seed": self.rot_seed,
            "rotation": self.rotation.tolist(),
            "event": "(Q^T x)_1 > c_star",
            "c_star": self.c_star,
            "prob_A": 0.5,
            "prob_A_abs_error": self.prob_A_abs_error,
        }


def student_t_median_radius(d: int, nu: float) -> tuple[float, float]:
    """r_med with P(||X|| <= r_med) = 1/2 for X ~ t_nu(0, I_d), and |pi(A)-1/2| at the double.

    ||X||^2 / d ~ F(d, nu); P(||X||^2 <= r^2) = I_{r^2/(r^2+nu)}(d/2, nu/2) (regularized beta).
    """
    with mpmath.workdps(_MP_DPS):
        a, b = mpmath.mpf(d) / 2, mpmath.mpf(nu) / 2

        def cdf_r2(r2):
            return mpmath.betainc(a, b, 0, r2 / (r2 + nu), regularized=True)

        u = mpmath.findroot(
            lambda uu: mpmath.betainc(a, b, 0, uu, regularized=True) - mpmath.mpf("0.5"),
            (mpmath.mpf(0), mpmath.mpf(1)),
            solver="illinois",
        )
        r2 = nu * u / (1 - u)
        r_double = float(mpmath.sqrt(r2))
        err = abs(float(cdf_r2(mpmath.mpf(r_double) ** 2) - mpmath.mpf("0.5")))
    if err > 1e-14:
        raise RuntimeError(f"Student-t median radius inaccurate: |pi(A)-1/2| = {err}")
    return r_double, err


class StudentTTarget(Target):
    """Multivariate t_nu(0, I), with A the ball of median probability.

    The direct sampler requires integer nu and uses a chi-squared denominator.
    """
    family = "student_t"

    def __init__(self, d: int, nu: float):
        super().__init__(d)
        if nu <= 2:
            raise ValueError(f"nu must be > 2 (finite covariance), got {nu}")
        self.nu = float(nu)
        self.r_med, self.prob_A_abs_error = student_t_median_radius(d, self.nu)
        self._log_norm = (
            math.lgamma((self.nu + d) / 2)
            - math.lgamma(self.nu / 2)
            - 0.5 * d * math.log(self.nu * math.pi)
        )

    @property
    def target_id(self) -> str:
        return f"studentt-d{self.d}-nu{_fmt(self.nu)}"

    def log_prob(self, x):
        self._check(x)
        r2 = (x * x).sum(1)
        return self._log_norm - 0.5 * (self.nu + self.d) * torch.log1p(r2 / self.nu)

    def score(self, x):
        self._check(x)
        r2 = (x * x).sum(1, keepdim=True)
        return -(self.nu + self.d) * x / (self.nu + r2)

    def sample(self, n, gen):
        if self.nu != round(self.nu):
            raise NotImplementedError(
                f"exact sampler needs integer nu (sum of squares), got {self.nu}"
            )
        z = self._randn(n, self.d, gen)
        w = (self._randn(n, int(self.nu), gen) ** 2).sum(1, keepdim=True)  # chi^2_nu
        return z * torch.sqrt(self.nu / w)

    def event_A(self, x):
        self._check(x)
        return (x * x).sum(1) <= self.r_med**2

    def mean(self):
        return np.zeros(self.d)

    def covariance(self):
        return np.eye(self.d) * self.nu / (self.nu - 2)

    def to_dict(self):
        return {
            "family": self.family,
            "target_id": self.target_id,
            "d": self.d,
            "nu": self.nu,
            "event": "||x|| <= r_med, r_med^2 = d F^{-1}_{d,nu}(1/2)",
            "r_med": self.r_med,
            "prob_A": 0.5,
            "prob_A_abs_error": self.prob_A_abs_error,
        }


class FunnelTarget(Target):
    """y ~ N(0, sigma_y^2), x | y ~ N(0, exp(y) I), with A = {y > 0}."""
    family = "funnel"

    def __init__(self, d: int, sigma_y: float):
        super().__init__(d)
        if d < 2:
            raise ValueError(f"funnel needs d >= 2, got {d}")
        if sigma_y <= 0:
            raise ValueError(f"sigma_y must be > 0, got {sigma_y}")
        self.sigma_y = float(sigma_y)

    @property
    def target_id(self) -> str:
        return f"funnel-d{self.d}-sy{_fmt(self.sigma_y)}"

    def log_prob(self, x):
        self._check(x)
        y, u = x[:, 0], x[:, 1:]
        s2 = self.sigma_y**2
        return (
            -0.5 * y * y / s2
            - 0.5 * math.log(2 * math.pi * s2)
            - 0.5 * (self.d - 1) * (LOG_2PI + y)
            - 0.5 * torch.exp(-y) * (u * u).sum(1)
        )

    def score(self, x):
        self._check(x)
        y, u = x[:, 0], x[:, 1:]
        e = torch.exp(-y)
        gy = -y / self.sigma_y**2 - 0.5 * (self.d - 1) + 0.5 * e * (u * u).sum(1)
        return torch.cat([gy[:, None], -e[:, None] * u], dim=1)

    def sample(self, n, gen):
        xi = self._randn(n, self.d, gen)
        y = self.sigma_y * xi[:, :1]
        return torch.cat([y, torch.exp(0.5 * y) * xi[:, 1:]], dim=1)

    def event_A(self, x):
        self._check(x)
        return x[:, 0] > 0

    def mean(self):
        return np.zeros(self.d)

    def covariance(self):
        return np.diag([self.sigma_y**2] + [math.exp(self.sigma_y**2 / 2)] * (self.d - 1))

    def to_dict(self):
        return {
            "family": self.family,
            "target_id": self.target_id,
            "d": self.d,
            "sigma_y": self.sigma_y,
            "coordinate_order": "(y, x_1, ..., x_{d-1})",
            "event": "y > 0",
            "prob_A": 0.5,
            "prob_A_abs_error": 0.0,
        }


def _check_rotation(q: np.ndarray, d: int) -> None:
    if q.shape != (d, d):
        raise ValueError(f"rotation shape {q.shape}, expected ({d}, {d})")
    err = np.abs(q.T @ q - np.eye(d)).max()
    if err > 1e-12:
        raise ValueError(f"rotation is not orthogonal (max |Q^T Q - I| = {err})")


ROTATIONS_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "configs" / "paperexp" / "rotations_v1.npz"
)


def rotation_key(family: str, d: int) -> str:
    return f"{family}-d{d}"


def load_rotation(family: str, d: int, path: pathlib.Path | None = None) -> np.ndarray:
    """Load the saved rotation for (family, d) and check orthogonality."""
    path = ROTATIONS_PATH if path is None else pathlib.Path(path)
    with np.load(path) as store:
        key = rotation_key(family, d)
        if key not in store.files:
            raise KeyError(f"rotation {key} not in {path}")
        q = np.array(store[key], dtype=np.float64)
    _check_rotation(q, d)
    return q


def make_target(family: str, d: int, rotation: np.ndarray | None = None, **params) -> Target:
    """Build a target. Rotated families use ``rotation`` if given, otherwise the saved one.

    params: gauss(cond, event_dir="narrow"|"wide"), mixture(sep, w), student_t(nu), funnel(sigma_y).
    Unknown parameters raise TypeError.
    """
    params = dict(params)
    if family == "gauss":
        q = load_rotation(family, d) if rotation is None else rotation
        t = GaussTarget(
            d,
            cond=params.pop("cond"),
            rotation=q,
            rot_seed=rotation_seed(family, d),
            event_dir=params.pop("event_dir", "narrow"),
        )
    elif family == "mixture":
        q = load_rotation(family, d) if rotation is None else rotation
        t = MixtureTarget(
            d,
            sep=params.pop("sep"),
            w=params.pop("w"),
            rotation=q,
            rot_seed=rotation_seed(family, d),
        )
    elif family == "student_t":
        t = StudentTTarget(d, nu=params.pop("nu"))
    elif family == "funnel":
        t = FunnelTarget(d, sigma_y=params.pop("sigma_y"))
    else:
        raise ValueError(f"unknown family {family!r}")
    if params:
        raise TypeError(f"unknown parameters for {family}: {sorted(params)}")
    return t


def target_from_dict(spec: dict) -> Target:
    """Rebuild a target from ``Target.to_dict()`` (rotation taken from the dict)."""
    fam, d = spec["family"], int(spec["d"])
    if fam == "gauss":
        t = GaussTarget(
            d,
            spec["cond"],
            np.asarray(spec["rotation"]),
            spec.get("rotation_seed"),
            event_dir=spec.get("event_dir", "narrow"),
        )
    elif fam == "mixture":
        t = MixtureTarget(
            d, spec["sep"], spec["w"], np.asarray(spec["rotation"]), spec.get("rotation_seed")
        )
    elif fam == "student_t":
        t = StudentTTarget(d, spec["nu"])
    elif fam == "funnel":
        t = FunnelTarget(d, spec["sigma_y"])
    else:
        raise ValueError(f"unknown family {fam!r}")
    if t.target_id != spec["target_id"]:
        raise ValueError(f"target_id mismatch: rebuilt {t.target_id}, saved {spec['target_id']}")
    return t


# The broad-suite target list (primary first). Each entry: (family, d, params).
PRIMARY_TARGETS: list[tuple[str, int, dict]] = [
    ("gauss", 10, {"cond": 25.0}),
    ("mixture", 10, {"sep": 6.0, "w": 0.5}),
    ("student_t", 10, {"nu": 3.0}),
    ("funnel", 10, {"sigma_y": 3.0}),
]

STRESS_TARGETS: list[tuple[str, int, dict]] = [
    ("gauss", 2, {"cond": 25.0}),
    ("gauss", 50, {"cond": 25.0}),
    ("gauss", 10, {"cond": 1.0}),
    ("gauss", 10, {"cond": 100.0}),
    ("mixture", 10, {"sep": 2.0, "w": 0.5}),
    ("mixture", 10, {"sep": 10.0, "w": 0.5}),
    ("mixture", 10, {"sep": 6.0, "w": 0.1}),
    ("mixture", 2, {"sep": 6.0, "w": 0.5}),
    ("student_t", 10, {"nu": 5.0}),
    ("student_t", 10, {"nu": 30.0}),
    ("student_t", 2, {"nu": 3.0}),
    ("funnel", 10, {"sigma_y": 1.0}),
    ("funnel", 2, {"sigma_y": 3.0}),
]

# Same gauss-d10-c25 transition and kernel, with initialization along the widest axis.
EVENT_VARIANT_TARGETS: list[tuple[str, int, dict]] = [
    ("gauss", 10, {"cond": 25.0, "event_dir": "wide"}),
]

ROTATED_FAMILY_DIMS: list[tuple[str, int]] = sorted(
    {(f, d) for f, d, _ in PRIMARY_TARGETS + STRESS_TARGETS if f in ("gauss", "mixture")}
)


FROZEN_SUITE_PATH = ROTATIONS_PATH.parent / "frozen_suite_v1.yaml"


class FrozenSuite:
    """Saved target, sampler and kernel parameters from frozen_suite_v1.yaml.

    Rotations are loaded from the associated NPZ file and checked by SHA-256.
    """

    def __init__(self, path: pathlib.Path | None = None):
        import hashlib

        import yaml

        self.path = FROZEN_SUITE_PATH if path is None else pathlib.Path(path)
        self.doc = yaml.safe_load(self.path.read_text())
        if self.doc.get("version") != "frozen_suite_v1":
            raise ValueError(f"unexpected frozen-suite version {self.doc.get('version')!r}")
        rot = self.doc["rotations"]
        rot_path = pathlib.Path(__file__).resolve().parents[3] / rot["path"]
        sha = hashlib.sha256(rot_path.read_bytes()).hexdigest()
        if sha != rot["sha256"]:
            raise ValueError(f"rotation file {rot_path} sha256 {sha} != frozen {rot['sha256']}")
        self.rotations_path = rot_path
        self._targets: dict[str, Target] = {}

    def target_ids(self, tier: str | None = None) -> list[str]:
        return [k for k, v in self.doc["targets"].items() if tier is None or v["tier"] == tier]

    def config_ids(self, tier: str | None = None) -> list[str]:
        return [k for k, v in self.doc["configs"].items() if tier is None or v["tier"] == tier]

    def target_spec(self, target_id: str) -> dict:
        return self.doc["targets"][target_id]

    def config(self, config_id: str) -> dict:
        return self.doc["configs"][config_id]

    def target(self, target_id: str) -> Target:
        if target_id not in self._targets:
            spec = self.target_spec(target_id)
            fam, d = spec["family"], int(spec["d"])
            rot = (
                load_rotation(fam, d, self.rotations_path) if fam in ("gauss", "mixture") else None
            )
            t = make_target(fam, d, rotation=rot, **dict(spec["params"]))
            if t.target_id != target_id:
                raise ValueError(f"rebuilt {t.target_id} != frozen {target_id}")
            for key in ("c_star", "r_med"):
                if key in spec and getattr(t, key) != spec[key]:
                    raise ValueError(f"{target_id}: {key} {getattr(t, key)} != frozen {spec[key]}")
            self._targets[target_id] = t
        return self._targets[target_id]

    def scales(self, target_id: str) -> np.ndarray:
        return np.asarray(self.target_spec(target_id)["scales"], dtype=np.float64)

    def kernel(self, target_id: str, kind: str = "rbf", bw: str = "ell"):
        from mcmc.paperexp.kernels import make_kernel

        if bw not in ("ell", "ell_half", "ell_double"):
            raise ValueError(f"bw must be ell | ell_half | ell_double, got {bw!r}")
        return make_kernel(kind, self.scales(target_id), float(self.target_spec(target_id)[bw]))

    def embedding(self, target_id: str, bw: str = "ell"):
        """Exact RBF mean embedding; ||mu||^2 is checked against the frozen value (rel 1e-10)."""
        from mcmc.paperexp.embeddings import rbf_mean_embedding

        emb = rbf_mean_embedding(self.target(target_id), self.kernel(target_id, "rbf", bw))
        frozen = float(self.target_spec(target_id)["rbf_mu_norm2"][bw])
        if abs(emb.norm2() - frozen) > 1e-10 * frozen:
            raise ValueError(f"{target_id}/{bw}: ||mu||^2 {emb.norm2()} != frozen {frozen}")
        return emb

    def sampler(self, config_id: str):
        from mcmc.paperexp.samplers import make_sampler

        c = self.config(config_id)
        return make_sampler(
            c["sampler"], self.target(c["target_id"]), float(c["h"]), self.scales(c["target_id"])
        )

    def t_grid(self, config_id: str) -> list[int]:
        return [int(t) for t in self.config(config_id)["t_grid"]]


def load_frozen_suite(path: pathlib.Path | None = None) -> FrozenSuite:
    return FrozenSuite(path)
