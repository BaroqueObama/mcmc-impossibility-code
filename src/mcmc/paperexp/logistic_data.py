"""Binary-classification datasets for hierarchical logistic regression.

Files are read from data/raw and checked against SHA256 hashes. Predictors
use population standard deviations (ddof=0), with no train/test split.
Pima's zero-coded missing values are kept without imputation. Australian
Credit uses the provider's numeric encoding and imputation. Ionosphere's
constant second column is dropped. Positive labels are tested_positive
(Pima), 1 (Australian), g (Ionosphere), and M (Sonar).
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DATA_ROOT = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_pima(d: pathlib.Path):
    text = (d / "dataset_37_diabetes.arff").read_text()
    head, sep, body = text.partition("@data")
    if not sep:
        raise ValueError("pima ARFF has no @data section")
    attrs = [
        ln.split()[1].strip("'") for ln in head.splitlines() if ln.lower().startswith("@attribute")
    ]
    if attrs != ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]:
        raise ValueError(f"unexpected pima attributes {attrs}")
    rows = [ln.split(",") for ln in body.split() if ln.strip()]
    if any(len(r) != 9 for r in rows) or any("?" in r for r in rows):
        raise ValueError("pima rows must have 9 fields and no missing '?' values")
    X = np.array([[float(v) for v in r[:8]] for r in rows])
    lab = np.array([r[8] for r in rows])
    if not set(lab) <= {"tested_positive", "tested_negative"}:
        raise ValueError(f"unexpected pima labels {set(lab)}")
    return X, (lab == "tested_positive").astype(np.float64), lab


def _parse_australian(d: pathlib.Path):
    rows = [ln.split() for ln in (d / "australian.dat").read_text().splitlines() if ln.strip()]
    if any(len(r) != 15 for r in rows):
        raise ValueError("australian rows must have 15 fields")
    a = np.array([[float(v) for v in r] for r in rows])
    lab = a[:, 14]
    if not set(np.unique(lab)) <= {0.0, 1.0}:
        raise ValueError(f"unexpected australian labels {np.unique(lab)}")
    return a[:, :14], lab.copy(), lab


def _parse_csv_label(fname: str, n_feat: int, positive: str, labels: set):
    def parse(d: pathlib.Path):
        rows = [ln.split(",") for ln in (d / fname).read_text().splitlines() if ln.strip()]
        if any(len(r) != n_feat + 1 for r in rows):
            raise ValueError(f"{fname} rows must have {n_feat + 1} fields")
        X = np.array([[float(v) for v in r[:n_feat]] for r in rows])
        lab = np.array([r[n_feat].strip() for r in rows])
        if set(lab) != labels:
            raise ValueError(f"unexpected labels in {fname}: {set(lab)}")
        return X, (lab == positive).astype(np.float64), lab

    return parse


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    files: dict  # file name -> sha256
    parse: Callable
    n: int
    p_raw: int
    n_pos: int
    drop: tuple = ()
    primary_file: str = ""
    source: str = ""
    label_rule: str = ""

    @property
    def p(self) -> int:
        return self.p_raw - len(self.drop)


SPECS = {
    "pima": DatasetSpec(
        name="pima",
        files={
            "dataset_37_diabetes.arff": "4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40",
            "openml_37_description.json": "9c5737c190ab2ed1cfcc84bf1c0a281ece11acd3dce065706217d1b0f5e3d2e7",
        },
        parse=_parse_pima,
        n=768,
        p_raw=8,
        n_pos=268,
        primary_file="dataset_37_diabetes.arff",
        source="https://api.openml.org/data/v1/download/37 (OpenML file 37; md5 3cbaa3e5... matches "
        "OpenML's description), downloaded 2026-10-05",
        label_rule="y = 1{class == tested_positive}",
    ),
    "australian": DatasetSpec(
        name="australian",
        files={
            "australian.dat": "3abe5af151afa50b999a9ba21cbc884e80d80a4050b716cf88b34a2e6ecb731c",
            "australian.doc": "baa316f419e75abb50ea24ef1f1268c22e49c3a903efd6ae77cb6fb87d7f186f",
        },
        parse=_parse_australian,
        n=690,
        p_raw=14,
        n_pos=307,
        primary_file="australian.dat",
        source="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/ (UCI 143)",
        label_rule="y = last column (1 = '+', credit approved)",
    ),
    "ionosphere": DatasetSpec(
        name="ionosphere",
        files={
            "ionosphere.data": "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd",
            "ionosphere.names": "a368e4fd9b910a3ac97fa306add17446c13105367e17f5b691455db2ea8411cf",
        },
        parse=_parse_csv_label("ionosphere.data", 34, "g", {"g", "b"}),
        n=351,
        p_raw=34,
        n_pos=225,
        drop=(1,),
        primary_file="ionosphere.data",
        source="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ (UCI 52)",
        label_rule="y = 1{class == 'g'}; attribute 2 (identically 0) dropped",
    ),
    "sonar": DatasetSpec(
        name="sonar",
        files={
            "sonar.all-data": "e90434cdbf00fcf93ffa911fe447ae25606979658e60f1d32e155c3b5240234d",
            "sonar.names": "f3d650ae9772b1a5021d977fe9f8194259f3dddec1922128c8104cf1c4e7250f",
        },
        parse=_parse_csv_label("sonar.all-data", 60, "M", {"M", "R"}),
        n=208,
        p_raw=60,
        n_pos=111,
        primary_file="sonar.all-data",
        source="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/"
        "connectionist-bench/sonar/ (UCI 151)",
        label_rule="y = 1{class == 'M'} (mine)",
    ),
}


@dataclass(frozen=True)
class BinaryData:
    name: str
    X: np.ndarray  # (n, p) standardized float64
    y: np.ndarray  # (n,) float64 in {0, 1}
    raw_X: np.ndarray
    raw_label: np.ndarray
    col_mean: np.ndarray
    col_std: np.ndarray
    dropped: tuple
    sha256: dict = field(default_factory=dict)
    primary_file: str = ""


def verify_files(name: str, data_root: pathlib.Path = DATA_ROOT) -> dict:
    spec = SPECS[name]
    d = pathlib.Path(data_root) / name
    for fname, expected in spec.files.items():
        path = d / fname
        if not path.is_file():
            raise FileNotFoundError(f"{name}: pinned file {path} is missing")
        got = _sha256(path)
        if got != expected:
            raise ValueError(f"{name}: SHA256 mismatch for {path}: got {got}, pinned {expected}")
    return dict(spec.files)


def load_dataset(name: str, data_root: pathlib.Path = DATA_ROOT) -> BinaryData:
    if name not in SPECS:
        raise ValueError(f"unknown data set {name!r}; choose from {sorted(SPECS)}")
    spec = SPECS[name]
    sha = verify_files(name, data_root)
    raw_X, y, raw_label = spec.parse(pathlib.Path(data_root) / name)
    if raw_X.shape != (spec.n, spec.p_raw) or y.shape != (spec.n,):
        raise ValueError(f"{name}: parsed shapes {raw_X.shape}, {y.shape}")
    if not np.isfinite(raw_X).all():
        raise ValueError(f"{name}: non-finite predictor values")
    if int(y.sum()) != spec.n_pos:
        raise ValueError(f"{name}: expected {spec.n_pos} positives, found {int(y.sum())}")
    for j in spec.drop:
        if np.ptp(raw_X[:, j]) != 0.0:
            raise ValueError(f"{name}: column {j} declared constant but varies")
    keep = [j for j in range(spec.p_raw) if j not in spec.drop]
    Xk = raw_X[:, keep]
    col_mean, col_std = Xk.mean(axis=0), Xk.std(axis=0, ddof=0)
    if not (col_std > 0).all():
        raise ValueError(f"{name}: undeclared constant columns {np.flatnonzero(col_std <= 0)}")
    X = (Xk - col_mean) / col_std
    assert X.shape == (spec.n, spec.p) and np.isfinite(X).all()
    return BinaryData(
        name, X, y, raw_X, raw_label, col_mean, col_std, tuple(spec.drop), sha, spec.primary_file
    )
