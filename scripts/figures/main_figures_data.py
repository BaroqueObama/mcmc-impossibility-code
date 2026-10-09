"""Collect the saved residual experiments into the appendix plotting tables."""

from __future__ import annotations
import csv
import dataclasses
import io
import math
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]

REGRESSION = ["concrete", "energy", "airfoil", "wine_red", "kin8nm"]

FAMILY_LABEL = {"gauss": "Gaussian", "mixture": "mixture", "studentt": "Student-t", "funnel": "funnel"}

REG_LABEL = {
    "concrete": "Concrete", "energy": "Energy", "airfoil": "Airfoil", "wine_red": "Wine (red)",
    "kin8nm": "Kin8nm",
}  # fmt: skip






def data_file(path: str) -> bytes | None:
    """Read a saved local input, if present."""
    local = ROOT / path
    return local.read_bytes() if local.is_file() else None


def read_csv_bytes(b: bytes) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(b.decode())))
    if not rows:
        raise ValueError("empty CSV")
    return rows


def fnum(x) -> float:
    if x is None or x in ("", "None", "nan", "NaN"):
        return math.nan
    return float(x)


@dataclasses.dataclass
class Part:
    """Rows extracted for one panel input, with a reason if data are missing."""

    name: str
    rows: list[dict]
    pending: list[dict]


def pending(what: str, reason: str) -> dict:
    return {"what": what, "reason": reason}


def frozen_tmax() -> dict:
    got = data_file("configs/paperexp/frozen_suite_v1.yaml")
    if got is None:
        raise FileNotFoundError("Missing configs/paperexp/frozen_suite_v1.yaml")
    data = got
    cfg = yaml.safe_load(data)["configs"]
    out = {cid: (int(c["t_max"]), bool(c["t_max_cap_binds"])) for cid, c in cfg.items()}
    TIER.update({cid: str(c["tier"]) for cid, c in cfg.items()})
    return out

TIER: dict[str, str] = {}  # config id -> primary | stress (filled by frozen_tmax)

R1B_MAIN_EXTRA = {"gauss-d50-c25__rwm"}


def r1a_circle() -> Part:
    got = data_file("paper/figures/data/r1_circle/population.csv")
    if got is None:
        return Part("r1a_circle", [], [pending("circle E1 population", "file not found")])
    data = got
    rows = []
    for r in read_csv_bytes(data):
        meas, pred = fnum(r["measured_residual"]), fnum(r["predicted_residual"])
        assert meas > 0 and pred > 0, r
        rows.append({
            "group": "analytic", "dataset": "circle", "label": "circle (Fourier, exact)",
            "R": 0.1, "r": fnum(r["r"]), "m": int(r["m"]), "ratio": meas / pred,
            "residual": meas, "envelope": pred, "kind": "exact law (quadrature)",
            "resolved": True,
        })  # fmt: skip
    return Part("r1a_circle", rows, [])


def r1a_suite() -> Part:
    got = data_file("paper/figures/data/r1_suite/constructions.csv")
    if got is None:
        return Part("r1a_suite", [], [pending("2-D synthetic constructions", "r1_suite data not found")])
    data = got
    rows = []
    allrows = [r for r in read_csv_bytes(data) if r["kernel"] == "rbf"]
    for tid in sorted({r["target_id"] for r in allrows}):
        rr = [r for r in allrows if r["target_id"] == tid]
        nmax = max(int(r["n"]) for r in rr)
        fam = tid.split("-")[0]
        for r in rr:
            if int(r["n"]) != nmax:
                continue
            res, env = fnum(r["residual_eval"]), fnum(r["envelope"])
            ratio = fnum(r["ratio_residual_envelope"])
            assert abs(ratio - res / env) <= 1e-9 * max(1.0, ratio), (tid, r["m"], ratio, res / env)
            rows.append({
                "group": "synthetic", "dataset": tid, "label": f"{FAMILY_LABEL[fam]} (2-D)",
                "R": fnum(r["R"]), "r": 0.5, "m": int(r["m"]), "ratio": ratio,
                "residual": res, "envelope": env,
                "kind": f"feasible positive law, K={nmax}^2 cells (independent quadrature)",
                "resolved": True,
            })  # fmt: skip
    return Part("r1a_suite", rows, [])

R1_REAL = "paper/figures/data/r1_real"


def _r1_real_export(name: str) -> list[dict] | None:
    data = data_file(f"{R1_REAL}/{name}")
    return read_csv_bytes(data) if data is not None else None




def r1a_regression() -> Part:
    got = _r1_real_export("constructions_a.csv")
    if got is None:
        return Part("r1a_regression", [], [pending("Gaussian-regression constructions (r1_real part A)",
                                                       "r1_real export not found")])  # fmt: skip
    data = got
    rows = []
    for ds in sorted({r["unit"] for r in data}):
        sel = [r for r in data if r["unit"] == ds and r["selected"] == "True"]
        nmax = max(int(r["n"]) for r in sel)
        for r in sel:
            if int(r["n"]) != nmax:
                continue
            res, env = fnum(r["residual"]), fnum(r["envelope"])
            rows.append({
                "group": "regression", "dataset": ds, "label": REG_LABEL.get(ds, ds), "R": fnum(r["R"]), "r": 0.5,
                "m": int(r["m"]), "ratio": res / env, "residual": res, "envelope": env,
                "kind": f"feasible positive law on a 2-D plane, K={nmax}^2 (exact posterior)", "resolved": True,
                "phase": "main",
            })  # fmt: skip
    keys = [(r["dataset"], r["R"], r["m"]) for r in rows]
    assert len(keys) == len(set(keys)), "more than one selected plane per (dataset, R, m)"
    pend = [pending(f"r1_real part A: {d}", "not in the export") for d in REGRESSION if d not in {r["dataset"] for r in rows}]
    return Part("r1a_regression", rows, pend)


def r1b_suite() -> Part:
    got = data_file("paper/figures/data/r1_suite/projected.csv")
    if got is None:
        return Part("r1b_suite", [], [pending("projected residuals d=10/50", "not found")])
    data = got
    rows = []
    for r in read_csv_bytes(data):
        cid = r["config_id"]
        fam, d = cid.split("-")[0], cid.split("-")[1]
        p2, se = fnum(r["proj2"]), fnum(r["proj2_se"])
        bound = fnum(r["bound_plugin"])
        resolved = r["proj_resolved"] == "True"
        assert resolved == (p2 > 2 * se), (cid, r["t"], r["m"], p2, se)
        rows.append({
            "group": "synthetic", "dataset": cid, "label": f"{FAMILY_LABEL[fam]} ({d.replace('d', 'd=')})",
            "setting": "eps=0.8", "t": int(r["t"]), "m": int(r["m"]),
            "ratio": math.sqrt(p2) / bound if resolved else math.nan,
            "ratio_hi": math.sqrt(max(p2 + 2 * se, 0.0)) / bound,
            "resolved": resolved, "bound": bound,
            "bound_kind": "plug-in R_init*lambda_hat_{m+1}^{1/2} (estimated eigenvalue, not a confidence bound)",
            "truth_kind": "replica estimate (jackknife +-2 SE)", "law_family": "",
            "ratio_lo": math.sqrt(max(p2 - 2 * se, 0.0)) / bound,
            "tier": TIER[cid], "sampler": cid.split("__")[1],
            "in_main": TIER[cid] == "primary" or cid in R1B_MAIN_EXTRA,
        })  # fmt: skip
    return Part("r1b_suite", rows, [])


def r1b_regression() -> Part:
    got = _r1_real_export("projected_a.csv")
    if got is None:
        return Part("r1b_regression", [], [pending("exact projected residuals of the Gibbs laws (r1_real part A)",
                                                       "r1_real export not found")])  # fmt: skip
    data = got
    rows = []
    for r in data:
        p2 = fnum(r["proj2_exact"])
        assert p2 >= -1e-15, (r["unit"], r["setting"], r["m"], p2)
        env = fnum(r["env_static"])  # proj2_exact pairs with the static envelope only
        rows.append({
            "group": "regression", "dataset": r["unit"], "label": REG_LABEL.get(r["unit"], r["unit"]),
            "setting": r["setting"], "t": int(r["t"]), "m": int(r["m"]),
            "ratio": math.sqrt(max(p2, 0.0)) / env, "ratio_hi": math.nan, "ratio_lo": math.nan,
            "resolved": True, "bound": env, "bound_kind": "exact envelope R*lambda_{m+1}^{1/2}",
            "truth_kind": "exact (closed form)", "phase": "main", "law_family": "",
        })  # fmt: skip
    pend = [pending(f"r1_real part A projected: {d}", "not in the export") for d in REGRESSION
            if d not in {r["dataset"] for r in rows}]  # fmt: skip
    return Part("r1b_regression", rows, pend)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def collect(out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_tmax()
    panels = {
        "r1a": [r1a_circle(), r1a_suite(), r1a_regression()],
        "r1b": [r1b_suite(), r1b_regression()],
    }
    missing = [item for parts in panels.values() for part in parts for item in part.pending]
    if missing:
        details = "; ".join(f"{item['what']}: {item['reason']}" for item in missing)
        raise FileNotFoundError(f"Cannot collect residual tables: {details}")
    summary = {"panels": {}}
    for name, parts in panels.items():
        rows = [row for part in parts for row in part.rows]
        path = out_dir / f"{name}.csv"
        if rows:
            _write_csv(path, rows)
        elif path.exists():
            path.unlink()
        summary["panels"][name] = {
            "n_rows": len(rows),
            "inputs": {p.name: {"n_rows": len(p.rows), "pending": p.pending} for p in parts},
        }
    return summary
