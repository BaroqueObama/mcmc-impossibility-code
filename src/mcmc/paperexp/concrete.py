"""Concrete compressive-strength data and Gaussian regression posterior.

The full-precision XLS is parsed with an OLE2/BIFF8 numeric-cell reader and
checked against the rounded UCI CSV (1 decimal for predictors, 2 for response).
File hashes are checked before parsing.

Predictors and response use sample standard deviations (ddof=1). The design
is [1, X_std], with beta ~ N(0, I_9) and fixed noise variance equal to the OLS
residual variance RSS/(n-9), rounded to two significant digits. The posterior
is Gaussian; systematic-scan Gibbs is generally nonreversible.
"""

from __future__ import annotations

import hashlib
import io
import math
import pathlib
import struct
import zipfile

import numpy as np

from mcmc.paperexp.gaussian_linear import LinearGaussianChain, gibbs_chain

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw" / "concrete"
SHA256 = {
    "concrete_compressive_strength.zip": "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7",
    "concrete_data_uci.csv": "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af",
}
XLS_SHA256 = "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4"
COLUMNS = (
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
    "Concrete compressive strength",
)
N_OBS = 1030
SOURCE_URLS = {
    "concrete_compressive_strength.zip": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    "concrete_data_uci.csv": "https://archive.ics.uci.edu/static/public/165/data.csv",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified(path: pathlib.Path, expected: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing; download from {SOURCE_URLS.get(path.name)}"
        )
    data = path.read_bytes()
    got = _sha256(data)
    if got != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {got} != {expected}")
    return data


# Minimal OLE2 compound-file + BIFF8 numeric-cell reader
_ENDOFCHAIN = 0xFFFFFFFE
_FREESECT = 0xFFFFFFFF


def _ole_stream(data: bytes, name: str) -> bytes:
    if data[:8] != bytes.fromhex("D0CF11E0A1B11AE1"):
        raise ValueError("not an OLE2 compound file")
    sector_size = 1 << struct.unpack_from("<H", data, 30)[0]
    mini_size = 1 << struct.unpack_from("<H", data, 32)[0]
    n_fat = struct.unpack_from("<I", data, 44)[0]
    dir_start = struct.unpack_from("<I", data, 48)[0]
    mini_cutoff = struct.unpack_from("<I", data, 56)[0]
    minifat_start = struct.unpack_from("<I", data, 60)[0]
    difat_start, n_difat = struct.unpack_from("<II", data, 68)
    difat = list(struct.unpack_from("<109I", data, 76))
    sec = difat_start
    for _ in range(n_difat):
        off = (sec + 1) * sector_size
        entries = struct.unpack_from(f"<{sector_size // 4}I", data, off)
        difat.extend(entries[:-1])
        sec = entries[-1]
    fat_sectors = [s for s in difat if s != _FREESECT][:n_fat]
    if len(fat_sectors) != n_fat:
        raise ValueError("FAT sector list inconsistent")
    fat = []
    for s in fat_sectors:
        fat.extend(struct.unpack_from(f"<{sector_size // 4}I", data, (s + 1) * sector_size))

    def chain(start: int) -> list[int]:
        out, cur = [], start
        while cur != _ENDOFCHAIN:
            if cur >= len(fat) or len(out) > len(fat):
                raise ValueError("corrupt FAT chain")
            out.append(cur)
            cur = fat[cur]
        return out

    def read_chain(start: int) -> bytes:
        return b"".join(data[(s + 1) * sector_size : (s + 2) * sector_size] for s in chain(start))

    directory = read_chain(dir_start)
    entries = {}
    for i in range(len(directory) // 128):
        e = directory[i * 128 : (i + 1) * 128]
        nlen = struct.unpack_from("<H", e, 64)[0]
        if nlen == 0:
            continue
        ename = e[: nlen - 2].decode("utf-16-le")
        etype = e[66]
        start = struct.unpack_from("<I", e, 116)[0]
        size = struct.unpack_from("<I", e, 120)[0]
        entries[ename] = (etype, start, size)
    if name not in entries:
        raise KeyError(f"stream {name!r} not in compound file (have {sorted(entries)})")
    etype, start, size = entries[name]
    if etype != 2:
        raise ValueError(f"{name!r} is not a stream (type {etype})")
    if size >= mini_cutoff:
        stream = read_chain(start)
    else:
        root = entries["Root Entry"]
        mini_stream = read_chain(root[1])
        minifat = []
        for s in chain(minifat_start):
            minifat.extend(struct.unpack_from(f"<{sector_size // 4}I", data, (s + 1) * sector_size))
        out, cur = [], start
        while cur != _ENDOFCHAIN:
            out.append(mini_stream[cur * mini_size : (cur + 1) * mini_size])
            cur = minifat[cur]
        stream = b"".join(out)
    if len(stream) < size:
        raise ValueError(f"stream {name!r} truncated ({len(stream)} < {size})")
    return stream[:size]


def _rk_value(rk: int) -> float:
    if rk & 0x02:
        v = float(rk >> 2 if rk < 0x80000000 else (rk >> 2) - (1 << 30))
    else:
        v = struct.unpack("<d", struct.pack("<Q", (rk & 0xFFFFFFFC) << 32))[0]
    return v / 100.0 if rk & 0x01 else v


def _biff8_first_sheet_numbers(wb: bytes) -> dict[tuple[int, int], float]:
    """Numeric cells {(row, col): value} of the first worksheet substream."""
    pos, depth, sheet_index, cells = 0, 0, -1, {}
    in_sheet = False
    while pos + 4 <= len(wb):
        rtype, rlen = struct.unpack_from("<HH", wb, pos)
        body = wb[pos + 4 : pos + 4 + rlen]
        pos += 4 + rlen
        if rtype == 0x0809:  # BOF
            depth += 1
            (substream,) = struct.unpack_from("<H", body, 2)
            if substream == 0x0010:
                sheet_index += 1
                in_sheet = sheet_index == 0
        elif rtype == 0x000A:  # EOF
            depth -= 1
            if in_sheet:
                break
        elif not in_sheet:
            continue
        elif rtype == 0x0203:  # NUMBER
            r, c, _ = struct.unpack_from("<HHH", body, 0)
            cells[(r, c)] = struct.unpack_from("<d", body, 6)[0]
        elif rtype == 0x027E:  # RK
            r, c, _, rk = struct.unpack_from("<HHHI", body, 0)
            cells[(r, c)] = _rk_value(rk)
        elif rtype == 0x00BD:  # MULRK
            r, c0 = struct.unpack_from("<HH", body, 0)
            (c1,) = struct.unpack_from("<H", body, rlen - 2)
            for k in range(c1 - c0 + 1):
                _, rk = struct.unpack_from("<HI", body, 4 + 6 * k)
                cells[(r, c0 + k)] = _rk_value(rk)
        elif rtype == 0x0006:  # FORMULA (cached numeric result) -- unexpected in this file
            raise ValueError("formula cell encountered; refusing to interpret cached results")
    if not cells:
        raise ValueError("no numeric cells parsed from first worksheet")
    return cells


def load_concrete(
    data_dir: pathlib.Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X (1030, 8), y (1030,), provenance) from the verified XLS, cross-checked against the CSV."""
    data_dir = pathlib.Path(data_dir) if data_dir is not None else DATA_DIR
    zbytes = _read_verified(
        data_dir / "concrete_compressive_strength.zip", SHA256["concrete_compressive_strength.zip"]
    )
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        xls = zf.read("Concrete_Data.xls")
    if _sha256(xls) != XLS_SHA256:
        raise ValueError("Concrete_Data.xls inside the zip has an unexpected SHA256")
    cells = _biff8_first_sheet_numbers(_ole_stream(xls, "Workbook"))
    rows = sorted({r for r, _ in cells})
    if rows != list(range(1, N_OBS + 1)):
        raise ValueError(f"unexpected numeric rows: {rows[:3]}...{rows[-3:]} ({len(rows)} rows)")
    if any(c not in range(9) for _, c in cells):
        raise ValueError("numeric cell outside the 9 expected columns")
    if len(cells) != N_OBS * 9:
        raise ValueError(f"expected {N_OBS * 9} numeric cells, got {len(cells)}")
    D = np.array([[cells[(r, c)] for c in range(9)] for r in rows], dtype=np.float64)
    if not np.isfinite(D).all():
        raise ValueError("non-finite values in Concrete_Data.xls")

    csv_bytes = _read_verified(data_dir / "concrete_data_uci.csv", SHA256["concrete_data_uci.csv"])
    lines = csv_bytes.decode("utf-8").strip().splitlines()
    header = tuple(h.strip() for h in lines[0].split(","))
    if header != COLUMNS:
        raise ValueError(f"CSV header {header} != {COLUMNS}")
    Dc = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=np.float64)
    if Dc.shape != (N_OBS, 9):
        raise ValueError(f"CSV shape {Dc.shape}")
    # The CSV export prints predictors to 1 decimal and the response to 2 decimals.
    gap = np.abs(D - Dc)
    tol = np.array([0.05] * 8 + [0.005]) + 1e-9
    max_gap = float(gap.max())
    if (gap > tol[None, :]).any():
        bad = np.unravel_index(np.argmax(gap / tol[None, :]), gap.shape)
        raise ValueError(
            f"XLS and CSV disagree beyond the CSV's rounding at {bad}: {D[bad]} vs {Dc[bad]}"
        )
    prov = {
        "source": "Concrete_Data.xls in UCI zip (OLE2/BIFF8 parsed)",
        "zip_sha256": SHA256["concrete_compressive_strength.zip"],
        "xls_sha256": XLS_SHA256,
        "csv_sha256": SHA256["concrete_data_uci.csv"],
        "xls_vs_csv_max_abs_gap": max_gap,
        "n": N_OBS,
        "columns": list(COLUMNS),
    }
    return D[:, :8], D[:, 8], prov


# Posterior and Gibbs sampler
def round_sig(x: float, sig: int = 2) -> float:
    if x <= 0 or not math.isfinite(x):
        raise ValueError(f"cannot round {x}")
    return float(f"{x:.{sig}g}")


class ConcretePosterior:
    """Gaussian regression posterior and its systematic-scan Gibbs chain."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, prior_var: float = 1.0, sigma2: float | None = None
    ):
        n, p = X.shape
        assert y.shape == (n,), f"y shape {y.shape}"
        self.x_mean, self.x_sd = X.mean(0), X.std(0, ddof=1)
        self.y_mean, self.y_sd = float(y.mean()), float(y.std(ddof=1))
        Xs = (X - self.x_mean) / self.x_sd
        ys = (y - self.y_mean) / self.y_sd
        Xd = np.hstack([np.ones((n, 1)), Xs])
        self.d = p + 1
        beta_ols, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
        self.ols_resid_var = float(((ys - Xd @ beta_ols) ** 2).sum() / (n - self.d))
        self.sigma2 = round_sig(self.ols_resid_var, 2) if sigma2 is None else float(sigma2)
        self.prior_var = float(prior_var)
        self.P = Xd.T @ Xd / self.sigma2 + np.eye(self.d) / self.prior_var
        self.P = 0.5 * (self.P + self.P.T)
        C = np.linalg.inv(self.P)
        self.C = 0.5 * (C + C.T)
        self.mu = self.C @ (Xd.T @ ys) / self.sigma2
        self.chain: LinearGaussianChain = gibbs_chain(
            self.P, self.mu, name="concrete_gibbs_systematic"
        )
        self.coord_sd = np.sqrt(np.diag(self.C))

    def directions(self, zero_tol: float = 1e-12) -> dict:
        """Prespecified real shift directions from the eigenvalues of the Gibbs map A.

        slow: real eigenvector of the largest-modulus eigenvalue (must be real and simple).
        fast: real eigenvector of the REAL nonzero eigenvalue whose modulus is closest to the median
              modulus of the nonzero spectrum (complex pairs have no real eigenvector and are skipped).
        Sign: the largest-|entry| component is made positive.
        """
        w, V = self.chain.eig_modes()
        mods = np.abs(w)
        if mods[0] - mods[1] < 1e-8 or abs(w[0].imag) > 0:
            raise ValueError(f"largest-modulus Gibbs eigenvalue not real/simple: {w[:3]}")
        nonzero = mods > zero_tol
        med = float(np.median(mods[nonzero]))
        real_idx = [i for i in range(len(w)) if nonzero[i] and w[i].imag == 0 and i != 0]
        dist = np.array([abs(mods[i] - med) for i in real_idx])
        if (np.sort(dist)[1] - np.sort(dist)[0]) < 1e-10:
            raise ValueError("fast-direction choice is tied")
        fast_i = real_idx[int(np.argmin(dist))]

        def real_unit(v):
            v = np.real_if_close(v, tol=1)
            if np.iscomplexobj(v):
                raise ValueError("eigenvector not real")
            v = np.asarray(v, dtype=np.float64)
            v = v / np.linalg.norm(v)
            return v * np.sign(v[np.argmax(np.abs(v))])

        return {
            "eigvals_real": w.real.tolist(),
            "eigvals_imag": w.imag.tolist(),
            "median_nonzero_modulus": med,
            "slow": {"index": 0, "eigval": float(w[0].real), "vec": real_unit(V[:, 0])},
            "fast": {
                "index": fast_i,
                "eigval": float(w[fast_i].real),
                "vec": real_unit(V[:, fast_i]),
            },
        }

    def shift(self, v: np.ndarray, r_init: float) -> np.ndarray:
        """a = c v with chi^2(N(mu+a, C) || N(mu, C)) = exp(a^T C^{-1} a) - 1 = r_init^2."""
        q = float(v @ self.P @ v)
        a = v * math.sqrt(math.log1p(r_init**2) / q)
        assert abs(math.expm1(float(a @ self.P @ a)) - r_init**2) < 1e-10
        return a
