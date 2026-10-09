"""Learned diagnostic spans and discrepancies on logistic posteriors.

The target embedding is estimated from independent NUTS runs. These are
reference-based comparisons, not coverage certificates. Nystrom coordinates
represent an orthonormal RKHS basis, so projections within the feature span
are full-RKHS quantities. Full residuals use exact kernel sums:
||P_W-perp delta||^2 = MMD^2(Q,ref) - ||P_W delta||^2.

Replica groups and independent reference runs supply bias corrections and
jackknife errors; within-run autocorrelation is retained. Diagnostic spans
are fitted on separate pilot chains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from mcmc.paperexp import slowmodes as sm
from mcmc.paperexp.features import NystromFeatures
from mcmc.paperexp.kernels import ScaledRBF, kernel_rowsum, kernel_sum

F64 = torch.float64
FAMILIES = ("static", "lag2t", "tica", "coord_slow", "coord_fixed", "random")


# targets and starting laws
class PosteriorTarget:
    """Posterior target whose samples come from an independent reference pool.

    Sampling is uniform with replacement, giving approximately stationary
    starts rather than exact posterior draws.
    """

    def __init__(self, posterior, pool: np.ndarray):
        if pool.ndim != 2 or pool.shape[1] != posterior.d or not np.isfinite(pool).all():
            raise ValueError(f"pool must be finite (n, {posterior.d}), got {pool.shape}")
        self.post = posterior
        self.d = posterior.d
        self.pool = pool
        self._pool_dev: dict[str, torch.Tensor] = {}

    @property
    def target_id(self) -> str:
        return self.post.target_id

    @property
    def name(self) -> str:
        return self.post.name

    def _check(self, x: torch.Tensor) -> None:
        self.post._check(x)

    def log_prob(self, x):
        return self.post.log_prob(x)

    def score(self, x):
        return self.post.score(x)

    def log_prob_and_score(self, x):
        return self.post.log_prob_and_score(x)

    def pool_on(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._pool_dev:
            self._pool_dev[key] = torch.as_tensor(self.pool, dtype=F64, device=device)
        return self._pool_dev[key]

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        P = self.pool_on(gen.device)
        idx = torch.randint(0, P.shape[0], (n,), generator=gen, device=gen.device)
        return P[idx]

    def to_dict(self) -> dict:
        return {**self.post.to_dict(), "pool_size": int(self.pool.shape[0])}


@dataclass
class StartLaws:
    """Initial laws I0-I3, defined from an independent reference pool (declared coordinates).

    I0 stationary control: pool states.
    I1 overdispersed: m + c (x_pool - m), with m the pool mean and c = ``overdispersion``.
    I2 box: U([-r, r]^d), the NUTS initialization law.
    I3 confined: pool states with u^T (x - m)/s > 0, where u is the leading eigenvector of the
       pool covariance of x/s and s are the frozen kernel scales.
    """

    pool: np.ndarray
    scales: np.ndarray
    overdispersion: float = 3.0
    box_radius: float = 2.0
    mean: np.ndarray = field(init=False)
    confine_dir: np.ndarray = field(init=False)
    confined_idx: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mean = self.pool.mean(axis=0)
        u = (self.pool - self.mean) / self.scales
        lam, Q = np.linalg.eigh(np.cov(u, rowvar=False))
        self.confine_dir = Q[:, -1] * np.sign(Q[np.abs(Q[:, -1]).argmax(), -1])
        self.confined_idx = np.flatnonzero(u @ self.confine_dir > 0)
        if self.confined_idx.size < 1000:
            raise ValueError("confined half-space holds too few pool states")

    def sample(self, law: str, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = gen.device
        d = self.pool.shape[1]
        if law in ("I0", "I1"):
            idx = torch.randint(0, self.pool.shape[0], (n,), generator=gen, device=dev)
            x = torch.as_tensor(self.pool, dtype=F64, device=dev)[idx]
            if law == "I1":
                m = torch.as_tensor(self.mean, dtype=F64, device=dev)
                x = m + self.overdispersion * (x - m)
            return x
        if law == "I2":
            u = torch.rand((n, d), generator=gen, dtype=F64, device=dev)
            return (2.0 * u - 1.0) * self.box_radius
        if law == "I3":
            k = torch.randint(0, self.confined_idx.size, (n,), generator=gen, device=dev)
            idx = torch.as_tensor(self.confined_idx, device=dev)[k]
            return torch.as_tensor(self.pool, dtype=F64, device=dev)[idx]
        raise ValueError(f"unknown start law {law!r}")

    def to_dict(self) -> dict:
        return {
            "overdispersion": self.overdispersion,
            "box_radius": self.box_radius,
            "pool_mean": self.mean.tolist(),
            "confine_dir_scaled": self.confine_dir.tolist(),
            "confined_fraction": float(self.confined_idx.size / self.pool.shape[0]),
        }


# representation and reference summaries
def make_kernel(entry: dict) -> ScaledRBF:
    return ScaledRBF(np.asarray(entry["scales"], dtype=np.float64), float(entry["ell"]))


def evenly_spaced(n_total: int, n: int) -> np.ndarray:
    idx = np.unique(np.linspace(0, n_total - 1, n).round().astype(np.int64))
    if idx.size != n:
        raise ValueError(f"cannot pick {n} distinct evenly spaced indices from {n_total}")
    return idx


def make_features(kernel, calib_draws: np.ndarray, n_landmarks: int, rel_cutoff: float, device):
    Z = torch.as_tensor(
        calib_draws[evenly_spaced(calib_draws.shape[0], n_landmarks)], device=device
    )
    return NystromFeatures(kernel, Z.to(F64), rel_cutoff)


def feature_means(feats, X: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    tot = torch.zeros(feats.rank, dtype=F64, device=X.device)
    for i0 in range(0, X.shape[0], chunk):
        tot += feats(X[i0 : i0 + chunk]).sum(0)
    return tot / X.shape[0]


@dataclass
class RefSummary:
    """Evaluation reference: R_tot independent runs (possibly from several sets)."""

    run_labels: list[str]
    X: list[torch.Tensor]  # per-run (N, d) draws on device
    feat_run_means: np.ndarray  # (R_tot, r)
    run_inner: np.ndarray  # (R_tot, R_tot) <mu_r, mu_s>_H
    coord_run_means: np.ndarray  # (R_tot, d)
    coord_quantiles: dict  # set name -> (nq, d) pooled quantiles
    coord_sd: dict  # set name -> (d,) pooled sd
    quantile_levels: np.ndarray

    @property
    def n_runs(self) -> int:
        return len(self.run_labels)

    def mu_runs_at(self, kernel, X: torch.Tensor, mem_bytes: int) -> torch.Tensor:
        """(J, R_tot): mu_r(x_j) = mean_i k(x_j, ref_{r,i}) for every run r."""
        cols = [kernel_rowsum(kernel, X, Xr, mem_bytes) / Xr.shape[0] for Xr in self.X]
        return torch.stack(cols, dim=1)


def build_ref_summary(
    kernel, feats, sets: dict, device, mem_bytes: int, quantile_levels
) -> RefSummary:
    """sets: name -> (R, N, d) draws. Run inner products by blocked exact kernel sums (fp64)."""
    labels, Xs, fm, cm = [], [], [], []
    qs, sds = {}, {}
    q = np.asarray(quantile_levels, dtype=np.float64)
    for name, draws in sets.items():
        R = draws.shape[0]
        qs[name] = np.quantile(draws.reshape(-1, draws.shape[2]), q, axis=0)
        sds[name] = draws.reshape(-1, draws.shape[2]).std(axis=0, ddof=1)
        for r in range(R):
            Xr = torch.as_tensor(draws[r], dtype=F64, device=device)
            labels.append(f"{name}:{r}")
            Xs.append(Xr)
            fm.append(feature_means(feats, Xr).cpu().numpy())
            cm.append(draws[r].mean(axis=0))
    n = len(Xs)
    inner = np.empty((n, n))
    for a in range(n):
        for b in range(a, n):
            s = kernel_sum(kernel, Xs[a], Xs[b], mem_bytes).item()
            inner[a, b] = inner[b, a] = s / (Xs[a].shape[0] * Xs[b].shape[0])
    return RefSummary(labels, Xs, np.stack(fm), inner, np.stack(cm), qs, sds, q)


# ensemble checkpoints
def checkpoint_stats(x: torch.Tensor, kernel, feats, ref: RefSummary, G: int, mem_bytes: int, q):
    """Sufficient statistics of one ensemble checkpoint (J replicas in G consecutive groups)."""
    J = x.shape[0]
    if J % G:
        raise ValueError(f"J={J} not divisible by G={G}")
    n = J // G
    F = feats(x)
    group_feat = F.view(G, n, -1).mean(1)
    xg = x.view(G, n, -1)
    S = torch.zeros((G, G), dtype=F64, device=x.device)
    rows = max(n, (mem_bytes // (8 * 4 * J)) // n * n)  # whole groups per block
    for i0 in range(0, J, rows):
        Kb = kernel(x[i0 : i0 + rows], x)
        gi = Kb.shape[0] // n
        S[i0 // n : i0 // n + gi] = Kb.view(gi, n, G, n).sum(dim=(1, 3))
    S = 0.5 * (S + S.T)
    mu = ref.mu_runs_at(kernel, x, mem_bytes)  # (J, R)
    cross = mu.view(G, n, -1).sum(1)  # (G, R)
    qq = torch.as_tensor(np.asarray(q), dtype=F64, device=x.device)
    return {
        "group_feat": group_feat.cpu().numpy(),
        "group_pair_ksum": S.cpu().numpy(),
        "group_run_cross": cross.cpu().numpy(),
        "group_coord_mean": xg.mean(1).cpu().numpy(),
        "coord_quantiles": torch.quantile(x, qq, dim=0).cpu().numpy(),
    }


def mmd2_unbiased(
    S: np.ndarray,
    cross: np.ndarray,
    inner: np.ndarray,
    n_per_group: int,
    groups: np.ndarray,
    runs: np.ndarray,
) -> float:
    """U-statistic MMD^2 between the ensemble (selected groups) and reference (selected runs).

    ||mu_Q||^2: off-diagonal replica pairs (k(x, x) = 1 removed). <mu_Q, mu_ref>: mean of
    mu_r(x_j). ||mu_ref||^2: mean over DISTINCT run pairs (unbiased because runs are independent).
    """
    J = groups.size * n_per_group
    A = (S[np.ix_(groups, groups)].sum() - J) / (J * (J - 1))
    B = cross[np.ix_(groups, runs)].sum() / (J * runs.size)
    M = inner[np.ix_(runs, runs)]
    C = (M.sum() - np.trace(M)) / (runs.size * (runs.size - 1))
    return float(A - 2.0 * B + C)


def mmd2_jackknife(S, cross, inner, n_per_group, runs) -> tuple[float, np.ndarray, np.ndarray]:
    """Point estimate plus its delete-one-group (G,) and delete-one-run (R,) replicates."""
    G = S.shape[0]
    g_all = np.arange(G)
    est = mmd2_unbiased(S, cross, inner, n_per_group, g_all, runs)
    jg = np.array(
        [mmd2_unbiased(S, cross, inner, n_per_group, np.delete(g_all, g), runs) for g in range(G)]
    )
    jr = np.array(
        [
            mmd2_unbiased(S, cross, inner, n_per_group, g_all, np.delete(runs, i))
            for i in range(runs.size)
        ]
    )
    return est, jg, jr


def jackknife_se(jg: np.ndarray, jr: np.ndarray) -> np.ndarray:
    """Delete-one-group plus delete-one-run jackknife SE (replicates along axis 0)."""
    G, R = jg.shape[0], jr.shape[0]
    var_g = (G - 1) / G * ((jg - jg.mean(0)) ** 2).sum(0)
    var_r = (R - 1) / R * ((jr - jr.mean(0)) ** 2).sum(0)
    return np.sqrt(var_g + var_r)


def mmd2_with_se(S, cross, inner, n_per_group, runs) -> tuple[float, float]:
    """Point estimate plus delete-one-group and delete-one-run jackknife standard error."""
    est, jg, jr = mmd2_jackknife(S, cross, inner, n_per_group, runs)
    return est, float(jackknife_se(jg, jr))


# conventional diagnostics on recorded chains
def conventional_on_traj(traj: np.ndarray, eval_grid, group_size: int = 4, min_t: int = 16) -> dict:
    """Rank R-hat, bulk and tail ESS over groups of ``group_size`` chains.

    The window at each t is the second half (t//2, t], i.e. the first half is discarded as
    warmup. ``traj`` has shape (T+1, K, d).
    """
    import arviz as az

    T1, K, d = traj.shape
    if K % group_size:
        raise ValueError(f"K={K} recorded chains not divisible by group size {group_size}")
    ng = K // group_size
    out = {}
    for t in eval_grid:
        if t < min_t:
            continue
        w = traj[t // 2 + 1 : t + 1]  # (n, K, d)
        rh, eb, et = np.empty(ng), np.empty(ng), np.empty(ng)
        for g in range(ng):
            a = np.ascontiguousarray(w[:, g * group_size : (g + 1) * group_size].transpose(1, 0, 2))
            rh[g] = np.nanmax(np.asarray(az.rhat(a, method="rank")))
            eb[g] = np.nanmin(np.asarray(az.ess(a, method="bulk")))
            et[g] = np.nanmin(np.asarray(az.ess(a, method="tail", prob=(0.05, 0.95))))
        out[int(t)] = {
            "rhat_max": rh,
            "ess_bulk_min": eb,
            "ess_tail_min": et,
            "window": int(w.shape[0]),
        }
    return out


# branched-pair Sigma_t (feature space)
def branched_operator_stats(F: torch.Tensor, N: int) -> dict:
    """F = features of [Y (N), Z (N), Y' (N), Z' (N)]; O = (D1^T D2 + D2^T D1)/(4N), plus halves."""
    if F.shape[0] != 4 * N:
        raise ValueError(f"expected {4 * N} rows, got {F.shape[0]}")
    FY, FZ, FYp, FZp = F[:N], F[N : 2 * N], F[2 * N : 3 * N], F[3 * N :]
    D1, D2 = FY - FYp, FZ - FZp
    h = N // 2

    def op(sl):
        a, b = D1[sl], D2[sl]
        n = a.shape[0]
        op_ = (a.T @ b + b.T @ a) / (4.0 * n)
        return (0.5 * (op_ + op_.T)).cpu().numpy()

    return {"O": op(slice(0, N)), "OA": op(slice(0, h)), "OB": op(slice(h, 2 * h))}


# learned spans from pilot chains
def multichain_lagged(F: np.ndarray, lag: int):
    """Pair-symmetrised (C0, Clag, asym) pooled over chains: F has shape (C, L, r).

    Centred by the pooled mean; equals ``slowmodes.lagged_pairs_covariances`` for C = 1.
    """
    F = sm.as_f64("F", F, 3)
    C, L, r = F.shape
    if not (1 <= lag < L):
        raise ValueError(f"lag {lag} must satisfy 1 <= lag < L={L}")
    Y = F - F.reshape(-1, r).mean(axis=0)
    Y0 = Y[:, : L - lag].reshape(-1, r)
    Y1 = Y[:, lag:].reshape(-1, r)
    n = Y0.shape[0]
    cross = Y0.T @ Y1
    C0 = (Y0.T @ Y0 + Y1.T @ Y1) / (2.0 * n)
    Clag = (cross + cross.T) / (2.0 * n)
    return C0, Clag, sm.relative_asymmetry(cross)


def coordinate_directions(F: np.ndarray, X: np.ndarray, coords, rel_cutoff: float):
    """Fit centered coordinates by least squares in the pilot's feature span.

    These kernel approximations need not match x_j outside the pilot's
    sampled region. Ordinary coordinate-mean discrepancies are evaluated
    separately. Returns coefficients (r,len(coords)) and in-sample R^2.
    """
    Fc = F - F.mean(axis=0)
    Xc = X[:, coords] - X[:, coords].mean(axis=0)
    coef, *_ = np.linalg.lstsq(Fc, Xc, rcond=rel_cutoff)
    resid = Xc - Fc @ coef
    with np.errstate(invalid="ignore", divide="ignore"):  # constant coordinate: R^2 = NaN
        r2 = 1.0 - (resid**2).sum(0) / (Xc**2).sum(0)
    return coef, r2


def learn_spans(
    F: np.ndarray,
    X: np.ndarray,
    t_star: int,
    m_max: int,
    tica_cutoff: float,
    coord_cutoff: float,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """All families from one pilot: F (C, L, r) features, X (C, L, d) states.

    Returns ({family: (r, m_max) basis or None}, info). A family that cannot be formed (lag too
    long, rank deficiency) is None, with the reason in info.
    """
    import arviz as az

    C, L, r = F.shape
    d = X.shape[2]
    Ff, Xf = F.reshape(-1, r), X.reshape(-1, d)
    spans, info = {}, {}
    lam, V = sm.leading_eigenspace(sm.static_covariance(Ff), m_max)
    info["static_eigs"] = lam[: m_max + 1].tolist()
    # A degenerate pilot (e.g. stuck chains) must not yield arbitrary eigenvectors of a ~zero matrix.
    if lam[0] > 0 and lam[m_max - 1] > 1e-12 * lam[0]:
        spans["static"] = V
    else:
        spans["static"] = None
        info["static_failed"] = "degenerate pilot covariance (top-m eigenvalues not positive)"
    if 2 * t_star < L:
        _, Clag2, asym = multichain_lagged(F, 2 * t_star)
        lam2, V2 = sm.leading_eigenspace(Clag2, m_max)
        if lam2[0] > 0 and lam2[m_max - 1] > 1e-12 * lam2[0]:
            spans["lag2t"] = V2
        else:
            spans["lag2t"] = None
            info["lag2t_failed"] = "degenerate lag-2t* covariance (top-m eigenvalues not positive)"
        info["lag2t_eigs"] = lam2[: m_max + 1].tolist()
        info["lag2t_asym"] = asym
    else:
        spans["lag2t"] = None
        info["lag2t_unavailable"] = f"2 t*={2 * t_star} >= pilot length {L}"
    C0, C1, asym1 = multichain_lagged(F, 1)
    try:
        modes = sm.tica_symmetric(C0, C1, tica_cutoff, n_modes=m_max, lag_asymmetry=asym1)
        spans["tica"] = modes.embedded
        info["tica_rho"] = modes.eigenvalues.tolist()
        info["tica_rank"] = modes.whitening.rank
    except sm.RankDeficientError as exc:
        spans["tica"] = None
        info["tica_failed"] = repr(exc)
    ess = np.asarray(az.ess(np.ascontiguousarray(X), method="bulk")).ravel()
    slow = np.argsort(ess, kind="stable")[:m_max]
    spans["coord_slow"], r2s = coordinate_directions(Ff, Xf, slow, coord_cutoff)
    info["coord_slow_idx"] = slow.tolist()
    info["coord_slow_ess"] = ess[slow].tolist()
    info["coord_slow_r2"] = r2s.tolist()
    spans["coord_fixed"], r2f = coordinate_directions(Ff, Xf, np.arange(m_max), coord_cutoff)
    info["coord_fixed_r2"] = r2f.tolist()
    spans["random"] = rng.standard_normal((r, m_max))
    return spans, info


# evaluation
def orthonormal_or_none(W: np.ndarray | None, m: int):
    if W is None:
        return None
    try:
        return sm.orthonormal_basis(np.ascontiguousarray(W[:, :m]))
    except sm.RankDeficientError:
        return None


def projection_terms(Qs: list, delta: np.ndarray, E: np.ndarray) -> np.ndarray:
    """For orthonormal bases Q (r, m) (None -> NaN): columns [raw ||Q^T d||^2, debias, variance].

    E (r, k) holds scaled deviations with E E^T = S_G/G + S_R/R (the covariance of delta-hat).
    """
    out = np.full((len(Qs), 3), np.nan)
    by_m: dict[int, list[int]] = {}
    for i, Q in enumerate(Qs):
        if Q is not None:
            by_m.setdefault(Q.shape[1], []).append(i)
    for m, idx in by_m.items():
        Qb = np.stack([Qs[i] for i in idx])  # (S, r, m)
        v = np.einsum("srm,r->sm", Qb, delta)
        P = np.einsum("srm,rk->smk", Qb, E)  # Q^T E
        A = np.einsum("smk,snk->smn", P, P)  # Q^T Sigma_e Q
        raw = (v**2).sum(1)
        bias = np.einsum("smm->s", A)
        var = 4.0 * np.einsum("sm,smn,sn->s", v, A, v) + 2.0 * (A * A).sum((1, 2))
        out[idx, 0], out[idx, 1], out[idx, 2] = raw, bias, var
    return out


def _leave_one_out(U: np.ndarray):
    """For replicate projections U (n, S, m): mean (S, m), sum of squared deviations (S,), and their
    leave-one-out versions (n, S, m) and (n, S)."""
    n = U.shape[0]
    mean = U.mean(0)
    dev2 = ((U - mean) ** 2).sum(2)
    ss = dev2.sum(0)
    return mean, ss, (n * mean - U) / (n - 1), ss - n / (n - 1) * dev2


def projection_jackknife(Qs: list, gf: np.ndarray, rf: np.ndarray):
    """Debiased ||Q^T delta||^2 (S,) with its delete-one-group (G, S) and delete-one-run (R, S)
    replicates; None -> NaN.

    With U_g = Q^T f_g (group feature means) and V_r = Q^T f_r (reference run means) the estimate
    is ||U-bar - V-bar||^2 - sum_g ||U_g - U-bar||^2/(G(G-1)) - sum_r ||V_r - V-bar||^2/(R(R-1)),
    i.e. raw - debias of ``projection_terms``. Each replicate recomputes it without one group or
    one run, in the same order as ``mmd2_jackknife``, so residual replicates can be paired.
    """
    G, R = gf.shape[0], rf.shape[0]
    if G < 3 or R < 3:
        raise ValueError(f"jackknife needs >= 3 groups and runs, got G={G}, R={R}")
    full = np.full(len(Qs), np.nan)
    jg, jr = np.full((G, len(Qs)), np.nan), np.full((R, len(Qs)), np.nan)
    by_m: dict[int, list[int]] = {}
    for i, Q in enumerate(Qs):
        if Q is not None:
            by_m.setdefault(Q.shape[1], []).append(i)
    for m, idx in by_m.items():
        Qb = np.stack([Qs[i] for i in idx])  # (S, r, m)
        Um, Uss, Uloo, Ussloo = _leave_one_out(np.einsum("srm,gr->gsm", Qb, gf))
        Vm, Vss, Vloo, Vssloo = _leave_one_out(np.einsum("srm,kr->ksm", Qb, rf))
        bias_g, bias_r = Uss / (G * (G - 1)), Vss / (R * (R - 1))
        full[idx] = ((Um - Vm) ** 2).sum(1) - bias_g - bias_r
        jg[:, idx] = ((Uloo - Vm) ** 2).sum(2) - Ussloo / ((G - 1) * (G - 2)) - bias_r
        jr[:, idx] = ((Um - Vloo) ** 2).sum(2) - bias_g - Vssloo / ((R - 1) * (R - 2))
    return full, jg, jr


def law_reference_terms(stats: dict, ref: RefSummary, runs: np.ndarray) -> dict:
    """delta-hat in features, its error factor E, the full feature norm and MMD^2 (+ SEs)."""
    gf = stats["group_feat"]
    G = gf.shape[0]
    rf = ref.feat_run_means[runs]
    R = runs.size
    f_ens, f_ref = gf.mean(0), rf.mean(0)
    delta = f_ens - f_ref
    E = np.concatenate(
        [(gf - f_ens).T / math.sqrt(G * (G - 1)), (rf - f_ref).T / math.sqrt(R * (R - 1))], axis=1
    )
    full = projection_terms([np.eye(delta.size)], delta, E)[0]
    n_per = stats["n_per_group"]
    mmd2, mmd2_jg, mmd2_jr = mmd2_jackknife(
        stats["group_pair_ksum"], stats["group_run_cross"], ref.run_inner, n_per, runs
    )
    return {
        "delta": delta,
        "E": E,
        "group_feat": gf,
        "run_feat": rf,
        "feat_raw": full[0],
        "feat_bias": full[1],
        "feat_var": full[2],
        "mmd2": mmd2,
        "mmd2_se": float(jackknife_se(mmd2_jg, mmd2_jr)),
        "mmd2_jack_g": mmd2_jg,
        "mmd2_jack_r": mmd2_jr,
    }


def evaluate_spans(terms: dict, Qs: list) -> dict:
    """Debiased ||P_W delta||^2 with variance; feature and full-RKHS residuals for every span.

    res_full2_se jackknifes the residual MMD^2 - ||P_W delta||^2 itself (paired delete-one-group /
    delete-one-run replicates), so it includes -2 Cov(MMD^2-hat, projection-hat).
    """
    pt = projection_terms(Qs, terms["delta"], terms["E"])
    proj2 = pt[:, 0] - pt[:, 1]
    full, jg, jr = projection_jackknife(Qs, terms["group_feat"], terms["run_feat"])
    scale = max(terms["feat_raw"], np.finfo(float).tiny)
    if not np.allclose(full, proj2, rtol=1e-9, atol=1e-12 * scale, equal_nan=True):
        raise RuntimeError("jackknife projection estimate differs from projection_terms")
    res_se = jackknife_se(terms["mmd2_jack_g"][:, None] - jg, terms["mmd2_jack_r"][:, None] - jr)
    feat2 = terms["feat_raw"] - terms["feat_bias"]
    return {
        "proj2": proj2,
        "proj2_se": np.sqrt(pt[:, 2]),
        "res_feat2": feat2 - proj2,
        "res_full2": terms["mmd2"] - proj2,
        "res_full2_se": res_se,
        "captured_feat": proj2 / feat2 if feat2 > 0 else np.full_like(proj2, np.nan),
        "captured_full": proj2 / terms["mmd2"]
        if terms["mmd2"] > 0
        else np.full_like(proj2, np.nan),
    }


@dataclass
class OperatorSummary:
    """Branched-pair Sigma_t estimate: eigendecompositions of the full and half-split estimates.

    ``eps = ||O_A - O_B||_op / 2`` is the half-split error proxy of the full estimate. Widths use
    the positive part (eigenvalues > ``rel_cutoff * lambda_max``); ``most_negative`` records the
    PSD violation of the symmetric estimate.
    """

    eigvals: np.ndarray  # descending, full estimate
    eigvecs: np.ndarray
    halves: tuple  # ((lam_A, Q_A), (lam_B, Q_B)), descending
    eps: float
    most_negative: float

    @classmethod
    def from_stats(cls, est: np.ndarray, OA: np.ndarray, OB: np.ndarray) -> "OperatorSummary":
        lam, Q = sm.eigh_desc(est, "O_t", max_rel_asym=1e-10)
        hA = sm.eigh_desc(OA, "O_t^A", max_rel_asym=1e-10)
        hB = sm.eigh_desc(OB, "O_t^B", max_rel_asym=1e-10)
        eps = float(np.linalg.norm(OA - OB, 2) / 2.0)
        return cls(lam, Q, (hA, hB), eps, float(min(lam.min(), 0.0)))

    def top_span(self, m: int) -> np.ndarray:
        return self.eigvecs[:, :m]

    def d_m(self, m: int) -> float:
        return math.sqrt(max(self.eigvals[m], 0.0))


def _positive_part(lam: np.ndarray, Q: np.ndarray, rel_cutoff: float):
    keep = lam > max(rel_cutoff * lam[0], 0.0)
    return lam[keep], Q[:, keep]


def widths_batch(
    lam: np.ndarray, Q: np.ndarray, W: np.ndarray, rel_cutoff: float = 1e-14, chunk: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained widths of orthonormal spans W (S, r, m) for the PSD part of Q diag(lam) Q^T.

    width^2 = lambda_max(S - S W (W^T S W)^{-1} W^T S) = lambda_max((I - P_C) Lambda (I - P_C)),
    where C = Lambda^{1/2} Q^T W (k x m) and P_C projects onto range(C) in R^k. This is the same
    quantity as ``slowmodes.constrained_width_cov`` (radius 1), batched over spans. Returns
    (width (S,), ok (S,)). ok is False if C is rank deficient (W^T S W singular).
    """
    lk, Qk = _positive_part(lam, Q, rel_cutoff)
    S = W.shape[0]
    width = np.full(S, np.nan)
    ok = np.zeros(S, dtype=bool)
    sq = np.sqrt(lk)
    for a in range(0, S, chunk):
        Wb = W[a : a + chunk]
        C = sq[None, :, None] * np.einsum("rk,srm->skm", Qk, Wb)
        U, R = np.linalg.qr(C)
        rd = np.abs(np.diagonal(R, axis1=1, axis2=2)).min(1)
        good = rd > 1e-10 * np.linalg.norm(C, axis=(1, 2))
        LU = lk[None, :, None] * U
        G = np.einsum("skm,skn->smn", U, LU)
        M = (
            np.diag(lk)[None]
            - LU @ U.transpose(0, 2, 1)
            - U @ LU.transpose(0, 2, 1)
            + U @ G @ U.transpose(0, 2, 1)
        )
        M = 0.5 * (M + M.transpose(0, 2, 1))
        top = np.linalg.eigvalsh(M)[:, -1]
        width[a : a + chunk] = np.where(good, np.sqrt(np.clip(top, 0.0, None)), np.nan)
        ok[a : a + chunk] = good
    return width, ok


def largest_angles_batch(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Largest principal angle (degrees) between orthonormal W (S, r, m) and orthonormal V (r, m),
    by the sine route sigma_max((I - V V^T) W) (accurate for small angles)."""
    R = W - np.einsum("rm,smn->srn", V, np.einsum("rm,srn->smn", V, W))
    sv = np.linalg.svd(R, compute_uv=False)[:, 0]
    return np.degrees(np.arcsin(np.clip(sv, 0.0, 1.0)))


def worst_case_batch(op: OperatorSummary, Qs: list, m: int, factor: float = 3.0) -> list[dict]:
    """w_t(W)/d_m(t) and theta_max(W, V*_m(t)) for spans of dimension m (None -> FAILED_SPAN).

    ratio_A / ratio_B use the half-split estimates (their own d_m). The ratio needs a resolved
    operator: operator_status is RESOLVED iff lambda_{m+1} >= factor * eps (the denominator
    d_m = sqrt(lambda_{m+1}) is above the half-split noise); otherwise ratio_status is
    OPERATOR_UNRESOLVED whatever the excess. With a resolved operator the excess ratio - 1 is
    RESOLVED if it exceeds factor * |ratio_A - ratio_B| / 2. The angle is RESOLVED if
    gap_m > factor * eps and lambda_m > factor * eps (else the eigenspace is not identified).
    """
    lam = op.eigvals
    gap = float(lam[m - 1] - lam[m])
    operator_ok = bool(lam[m] >= factor * op.eps)
    base = {
        "lambda_m": float(lam[m - 1]),
        "lambda_m1": float(lam[m]),
        "gap": gap,
        "eps": op.eps,
        "d_m": op.d_m(m),
        "op_resolution": float(lam[m]) / op.eps if op.eps > 0 else math.inf,
        "operator_status": "RESOLVED" if operator_ok else "UNRESOLVED",
    }
    idx = [i for i, Q in enumerate(Qs) if Q is not None]
    rows = [
        dict(
            base,
            ratio=np.nan,
            ratio_A=np.nan,
            ratio_B=np.nan,
            angle_deg=np.nan,
            ratio_status="FAILED_SPAN",
            angle_status="FAILED_SPAN",
        )
        for _ in Qs
    ]
    if not idx:
        return rows
    W = np.stack([Qs[i] for i in idx])
    if W.shape[2] != m:
        raise ValueError(f"spans of dimension {W.shape[2]} passed for m={m}")
    w, ok = widths_batch(lam, op.eigvecs, W)
    ratios = {"": w / op.d_m(m)}
    for tag, (lh, Qh) in zip(("A", "B"), op.halves):
        wh, _ = widths_batch(lh, Qh, W)
        ratios[tag] = wh / math.sqrt(max(lh[m], 0.0))
    ang = largest_angles_batch(W, op.top_span(m))
    angle_ok = gap > factor * op.eps and lam[m - 1] > factor * op.eps
    for j, i in enumerate(idx):
        r, ra, rb = ratios[""][j], ratios["A"][j], ratios["B"][j]
        err = abs(ra - rb) / 2.0
        rows[i].update(
            ratio=r,
            ratio_A=ra,
            ratio_B=rb,
            excess_err=err,
            width=w[j],
            angle_deg=ang[j],
            ratio_status=(
                "RANK_DEFICIENT"
                if not ok[j]
                else "OPERATOR_UNRESOLVED"
                if not operator_ok
                else "RESOLVED"
                if r - 1.0 > factor * err
                else "UNRESOLVED"
            ),
            angle_status="RESOLVED" if angle_ok else "UNRESOLVED",
        )
    return rows
