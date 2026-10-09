"""Bayesian linear regression on Concrete, Energy, Airfoil, Wine and kin8nm.

Predictors and response are standardized with ddof=1; all predictor columns
are numeric, including Energy's category codes. The prior is N(0, I), and the
fixed noise variance is the OLS residual variance rounded to two significant
digits. The posterior is Gaussian, sampled by systematic-scan Gibbs.

frozen_model loads the paper's bandwidth, checkpoints and initial mean shifts.
Each shift a satisfies chi^2(pi_0 || pi) = exp(a^T C^{-1} a) - 1 = R_init^2.
"""

from __future__ import annotations

import hashlib
import io
import math
import pathlib
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree

import numpy as np

from mcmc.paperexp import concrete as _concrete
from mcmc.paperexp.concrete import ConcretePosterior

RAW = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
CONFIGS = pathlib.Path(__file__).resolve().parents[3] / "configs" / "experiment"

SHA256 = {
    "energy": {
        "energy_efficiency.zip": "499441eee27929a4b00417f58fd8c63c9cc14b8a71520cd0dd27fcb626738351",
        "energy_data_uci.csv": "db44dbe453acd464b5cf65be2fb01a28aa9c5b2630300e65fbe28cde35f5d96f",
    },
    "airfoil": {
        "airfoil_self_noise.zip": "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
    },
    "wine_red": {
        "wine_quality.zip": "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4"
    },
    "kin8nm": {"kin8nm.arff": "698fb5297abde73c001b07f8fb2fe31626366cf179678aed0990737cc0689c79"},
}
SOURCES = {
    "energy": {
        "energy_efficiency.zip": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
        "energy_data_uci.csv": "https://archive.ics.uci.edu/static/public/242/data.csv",
    },
    "airfoil": {
        "airfoil_self_noise.zip": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
    },
    "wine_red": {
        "wine_quality.zip": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    },
    "kin8nm": {
        "kin8nm.arff": "https://openml.org/data/v1/download/3626/kin8nm.arff (OpenML md5 b36414070701ab8bc985104c3ebd9b77)"
    },
}
SHAPES = {
    "concrete": (1030, 8),
    "energy": (768, 8),
    "airfoil": (1503, 5),
    "wine_red": (1599, 11),
    "kin8nm": (8192, 8),
}
NAMES = tuple(SHAPES)


def _read(name: str, fname: str, data_root: pathlib.Path) -> bytes:
    path = data_root / name / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing; download from {SOURCES[name][fname]}"
        )
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256[name][fname]:
        raise ValueError(f"SHA256 mismatch for {path}: {got} != {SHA256[name][fname]}")
    return data


def _finite(D: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(D).all():
        raise ValueError(f"{name}: non-finite values")
    return D


def _xlsx_numbers(xlsx: bytes, sheet: str = "xl/worksheets/sheet1.xml") -> tuple[dict, dict]:
    """Minimal XLSX reader: {(row, col): float} numeric cells and {(row, col): str} shared-string cells."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    zz = zipfile.ZipFile(io.BytesIO(xlsx))
    sst = [
        si.findtext("m:t", default="", namespaces=ns)
        for si in ElementTree.fromstring(zz.read("xl/sharedStrings.xml")).findall("m:si", ns)
    ]
    root = ElementTree.fromstring(zz.read(sheet))
    nums, strs = {}, {}
    for c in root.iter("{%s}c" % ns["m"]):
        ref = c.get("r")
        col = 0
        i = 0
        while ref[i].isalpha():
            col = col * 26 + (ord(ref[i]) - 64)
            i += 1
        row = int(ref[i:])
        v = c.findtext("m:v", default=None, namespaces=ns)
        if v is None:
            continue
        if c.get("t") == "s":
            strs[(row, col - 1)] = sst[int(v)]
        elif c.get("t") in (None, "n"):
            nums[(row, col - 1)] = float(v)
        else:
            raise ValueError(f"unexpected cell type {c.get('t')} at {ref}")
    return nums, strs


def _load_energy(root: pathlib.Path):
    z = zipfile.ZipFile(io.BytesIO(_read("energy", "energy_efficiency.zip", root)))
    nums, strs = _xlsx_numbers(z.read("ENB2012_data.xlsx"))
    header = [strs.get((1, c)) for c in range(10)]
    if header != ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]:
        raise ValueError(f"Energy header {header}")
    if any(r > 769 or c > 9 for r, c in nums):
        raise ValueError("Energy: numeric cells outside A2:J769")
    D = np.array([[nums[(r, c)] for c in range(10)] for r in range(2, 770)])
    lines = _read("energy", "energy_data_uci.csv", root).decode().strip().splitlines()
    Dc = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    if lines[0].split(",") != header or Dc.shape != D.shape:
        raise ValueError("Energy CSV header/shape mismatch")
    gap = float(np.abs(D - Dc).max())
    if gap > 0.005 + 1e-9:
        raise ValueError(f"Energy XLSX vs CSV gap {gap}")
    D = _finite(D, "energy")
    return (
        D[:, :8],
        D[:, 8],
        {
            "source": "ENB2012_data.xlsx in UCI zip (XLSX parsed)",
            "response": "Y1 heating load",
            "xlsx_vs_csv_max_abs_gap": gap,
        },
    )


def _load_airfoil(root: pathlib.Path):
    z = zipfile.ZipFile(io.BytesIO(_read("airfoil", "airfoil_self_noise.zip", root)))
    rows = [ln.split("\t") for ln in z.read("airfoil_self_noise.dat").decode().strip().splitlines()]
    if any(len(r) != 6 for r in rows):
        raise ValueError("Airfoil: row with != 6 fields")
    D = _finite(np.array([[float(v) for v in r] for r in rows]), "airfoil")
    return (
        D[:, :5],
        D[:, 5],
        {
            "source": "airfoil_self_noise.dat in UCI zip",
            "response": "scaled sound pressure level (dB)",
        },
    )


def _load_wine_red(root: pathlib.Path):
    z = zipfile.ZipFile(io.BytesIO(_read("wine_red", "wine_quality.zip", root)))
    lines = z.read("winequality-red.csv").decode().strip().splitlines()
    header = [h.strip('"') for h in lines[0].split(";")]
    if len(header) != 12 or header[-1] != "quality":
        raise ValueError(f"Wine header {header}")
    rows = [ln.split(";") for ln in lines[1:]]
    if any(len(r) != 12 for r in rows):
        raise ValueError("Wine: row with != 12 fields")
    D = _finite(np.array([[float(v) for v in r] for r in rows]), "wine_red")
    return (
        D[:, :11],
        D[:, 11],
        {
            "source": "winequality-red.csv in UCI zip",
            "response": "quality (integer score, treated as continuous)",
        },
    )


def _load_kin8nm(root: pathlib.Path):
    text = _read("kin8nm", "kin8nm.arff", root).decode()
    attrs, data, in_data = [], [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            parts = s.split()
            if parts[-1].lower() not in ("numeric", "real"):
                raise ValueError(f"kin8nm non-numeric attribute {s}")
            attrs.append(parts[1].strip("'"))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            data.append([float(v) for v in s.split(",")])
    if attrs[-1] != "y" or len(attrs) != 9:
        raise ValueError(f"kin8nm attributes {attrs}")
    D = _finite(np.array(data), "kin8nm")
    return D[:, :8], D[:, 8], {"source": "kin8nm.arff (OpenML 189)", "response": "y"}


def load_dataset(name: str, data_root: pathlib.Path | str | None = None):
    """(X (n, p), y (n,), provenance) with SHA256 verification and shape checks."""
    root = pathlib.Path(data_root) if data_root is not None else RAW
    if name == "concrete":
        X, y, prov = _concrete.load_concrete(root / "concrete")
    else:
        X, y, prov = {
            "energy": _load_energy,
            "airfoil": _load_airfoil,
            "wine_red": _load_wine_red,
            "kin8nm": _load_kin8nm,
        }[name](root)
        prov = {**prov, "sha256": SHA256[name]}
    if X.shape != SHAPES[name] or y.shape != (SHAPES[name][0],):
        raise ValueError(f"{name}: shapes {X.shape}, {y.shape} != {SHAPES[name]}")
    if (X.std(0, ddof=1) <= 0).any():
        raise ValueError(f"{name}: constant predictor column")
    return X, y, prov


class RegressionPosterior(ConcretePosterior):
    """The Concrete model applied to any registered dataset (same standardization, prior, sigma^2 rule)."""

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        prior_var: float = 1.0,
        sigma2: float | None = None,
    ):
        super().__init__(X, y, prior_var=prior_var, sigma2=sigma2)
        self.name = name
        self.chain.name = f"{name}_gibbs_systematic"


def registry_directions(post: ConcretePosterior, zero_tol: float = 1e-12) -> dict:
    """Initial mean-shift directions from the Gibbs map's eigenvectors.

    The slow direction has largest eigenvalue modulus; for a complex pair, use
    its real part. The fast direction is real, excludes the slow pair, and has
    modulus nearest the median of the nonzero spectrum (ties favor the larger
    modulus). If none is real, use the real part of the nearest complex mode.
    Directions have unit norm and their largest-magnitude entry is positive.
    """
    w, V = post.chain.eig_modes()
    mods = np.abs(w)
    nonzero = mods > zero_tol
    med = float(np.median(mods[nonzero]))

    def unit(v):
        v = np.real(v).astype(np.float64)
        n = np.linalg.norm(v)
        if not n > 0:
            raise ValueError("zero real part of eigenvector")
        v = v / n
        return v * np.sign(v[np.argmax(np.abs(v))])

    slow_i = 0
    if mods[0] - mods[1] < 1e-8 and w[0].imag == 0:
        raise ValueError(f"largest-modulus real Gibbs eigenvalue not simple: {w[:3]}")
    pair_of_slow = {0, 1} if w[0].imag != 0 else {0}
    cands = [i for i in range(len(w)) if nonzero[i] and w[i].imag == 0 and i not in pair_of_slow]
    if cands:
        dist = np.array([abs(mods[i] - med) for i in cands])
        best = np.flatnonzero(np.abs(dist - dist.min()) <= 1e-12)
        fast_i = max((cands[j] for j in best), key=lambda i: mods[i])
        fast_rule = "real, closest to median modulus" + (
            " (tie -> larger modulus)" if len(best) > 1 else ""
        )
    else:
        cc = [i for i in range(len(w)) if nonzero[i] and i not in pair_of_slow]
        fast_i = min(cc, key=lambda i: (abs(mods[i] - med), -mods[i]))
        fast_rule = "no real candidate: real part of complex eigenvector closest to median modulus"
    return {
        "eigvals_real": w.real.tolist(),
        "eigvals_imag": w.imag.tolist(),
        "median_nonzero_modulus": med,
        "slow": {
            "index": slow_i,
            "eigval": complex(w[slow_i]),
            "modulus": float(mods[slow_i]),
            "vec": unit(V[:, slow_i]),
            "rule": "real eigenvector"
            if w[slow_i].imag == 0
            else "real part of complex eigenvector",
        },
        "fast": {
            "index": fast_i,
            "eigval": complex(w[fast_i]),
            "modulus": float(mods[fast_i]),
            "vec": unit(V[:, fast_i]),
            "rule": fast_rule,
        },
    }


@dataclass
class FrozenModel:
    """Posterior, Gibbs transition, kernel and initial shifts for one dataset."""

    name: str
    n: int
    p: int
    sigma2: float
    ols_resid_var: float
    mu: np.ndarray
    C: np.ndarray
    P: np.ndarray
    A: np.ndarray
    LQ: np.ndarray
    kernel_scales: np.ndarray
    ell: float | None
    gibbs_eigvals: np.ndarray
    directions: dict
    settings: list = field(default_factory=list)  # dicts: name, direction, r_init, shift
    provenance: dict = field(default_factory=dict)
    checkpoints: list = field(default_factory=list)  # R3 inspection grid (t_max rule)
    seed_namespace: str = ""

    @property
    def Q(self) -> np.ndarray:
        return self.LQ @ self.LQ.T


def frozen_config(name: str) -> dict:
    """Read the bandwidth, checkpoints, noise variance and seed namespace from the configs."""
    from omegaconf import OmegaConf

    if name == "concrete":
        c = OmegaConf.load(CONFIGS / "exp_r3_concrete.yaml")
        return {
            "ell": c.kernel.ell_frozen,
            "checkpoints": list(c.checkpoints),
            "sigma2_declared": float(c.model.sigma2_declared),
            "namespace": str(c.seed_namespace),
        }
    c = OmegaConf.load(CONFIGS / "exp_r3_regression.yaml")
    d = c.datasets[name]
    return {
        "ell": d.ell_frozen,
        "checkpoints": list(d.checkpoints),
        "sigma2_declared": float(d.sigma2_declared),
        "namespace": f"{c.seed_namespace}-{name}",
    }


def frozen_model(
    name: str,
    ell: float | str | None = "config",
    settings=((0.8, "slow"), (0.2, "slow"), (0.8, "fast"), (0.2, "fast")),
) -> FrozenModel:
    """Posterior, Gibbs transition and initial mean shifts for a regression dataset.

    ell="config" reads the saved bandwidth; a float overrides it. The declared
    noise variance must agree with the rounded OLS estimate.
    """
    fc = frozen_config(name)
    if ell == "config":
        ell = None if fc["ell"] is None else float(fc["ell"])
    X, y, prov = load_dataset(name)
    post = RegressionPosterior(name, X, y)
    if post.sigma2 != fc["sigma2_declared"]:
        raise ValueError(f"{name}: OLS rounding {post.sigma2} != declared {fc['sigma2_declared']}")
    dirs = registry_directions(post)
    sets = []
    for R, d in settings:
        a = post.shift(dirs[d]["vec"], R)
        sets.append({"name": f"{d}_R{R}", "direction": d, "r_init": R, "shift": a})
    ch = post.chain
    return FrozenModel(
        name=name,
        n=X.shape[0],
        p=X.shape[1],
        sigma2=post.sigma2,
        ols_resid_var=post.ols_resid_var,
        mu=post.mu,
        C=post.C,
        P=post.P,
        A=ch.A,
        LQ=ch.LQ,
        kernel_scales=post.coord_sd,
        ell=ell,
        gibbs_eigvals=np.array(dirs["eigvals_real"]) + 1j * np.array(dirs["eigvals_imag"]),
        directions={
            k: {kk: dirs[k][kk] for kk in ("eigval", "modulus", "vec", "rule")}
            for k in ("slow", "fast")
        },
        settings=sets,
        provenance=prov,
        checkpoints=fc["checkpoints"],
        seed_namespace=fc["namespace"],
    )


def mmd_shift(model: FrozenModel, t: int, shift: np.ndarray) -> float:
    """Analytic MMD(pi_t, pi) for pi_0 = N(mu + shift, C) under the frozen RBF kernel (needs ell)."""
    from mcmc.paperexp import gaussian_linear as gl

    if model.ell is None:
        raise ValueError("frozen bandwidth ell not set")
    k = gl.AnisoRBF(model.kernel_scales, model.ell)
    return math.sqrt(
        gl.mmd2_shifted_gaussians(k, model.C, np.linalg.matrix_power(model.A, t) @ shift)
    )
