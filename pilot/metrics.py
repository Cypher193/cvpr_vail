"""Scoring for show and know readouts. Run `python -m pilot.metrics` for the self-check."""
import numpy as np
from scipy import stats

# Symmetric MRA thresholds: 5%..50% off, then up to 5x off so weak models do not all tie at 0.
SYM_TAUS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
# QuantiPhy's original thresholds (appendix set), reported in one column for comparability.
ORIG_C = np.arange(0.50, 0.951, 0.05)
ZERO_EPS = 1e-3   # absolute tolerance used when the true value is 0


def sym_mra(pred, true, taus=SYM_TAUS, zero_eps=ZERO_EPS):
    """Per-item symmetric MRA: share of thresholds with |ln(pred/true)| < ln(1+tau).
    Wrong sign or non-positive prediction scores 0. True value 0 uses |pred| < zero_eps."""
    pred, true = np.broadcast_arrays(np.asarray(pred, float), np.asarray(true, float))
    out = np.zeros(pred.shape)
    zero = np.abs(true) < 1e-12
    out[zero] = (np.abs(pred[zero]) < zero_eps).astype(float)
    ok = ~zero & (np.sign(pred) == np.sign(true)) & (pred != 0)
    r = np.abs(np.log(pred[ok] / true[ok]))
    out[ok] = (r[:, None] < np.log1p(taus)[None, :]).mean(1)
    return out


def orig_mra(pred, true, C=ORIG_C):
    """QuantiPhy MRA per item: share of theta in C with |pred-true|/|true| < 1-theta."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / np.abs(true)
    return (rel[..., None] < (1 - C)).mean(-1)


def log_ratio(pred, true):
    return np.log(np.asarray(pred, float) / np.asarray(true, float))


def signed_log_bias(pred, true):
    """Median ln(pred/true): < 0 means values come out too low."""
    return float(np.median(log_ratio(pred, true)))


def abs_log_err(pred, true):
    return float(np.median(np.abs(log_ratio(pred, true))))


def silog(pred, true):
    """Scale-invariant log error: spread of ln(pred/true) after removing a global scale (the clock)."""
    r = log_ratio(pred, true)
    return float(np.sqrt(max(np.mean(r ** 2) - np.mean(r) ** 2, 0.0)))


def gamma_fit(pred, true, n_boot=2000, seed=0, groups=None):
    """Slope of ln(pred) on ln(true). 1 = follows the input, 0 = prior wins.
    A global clock or scale error only moves the intercept. Bootstrap resamples groups (scenes) if given."""
    x, y = np.log(np.asarray(true, float)), np.log(np.asarray(pred, float))
    slope = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)
    g = np.arange(len(x)) if groups is None else np.asarray(groups)
    ug = np.unique(g)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([np.flatnonzero(g == k) for k in pick])
        if np.ptp(x[idx]) > 0:
            boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(slope), (float(lo), float(hi))


def spearman(pred, true):
    return float(stats.spearmanr(pred, true).statistic)


def accel_residual(x, dt):
    """Constant-acceleration residual: RMS of the change in second differences, normalised by motion size.
    Near 0 = motion follows one constant-acceleration law; large = no consistent law."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    a = np.diff(x, 2, axis=0) / dt ** 2
    jerk = np.diff(a, axis=0)
    scale = np.linalg.norm(np.ptp(x, axis=0)) / (dt * len(x)) ** 2 + 1e-12
    return float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))) / scale)


def crps_ensemble(samples, y, log=True):
    """CRPS of an ensemble (e.g. seeds) against the true value; on log scale by default so it is scale-free."""
    s, y = np.asarray(samples, float), float(y)
    if log:
        s, y = np.log(s), np.log(y)
    return float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])))


def coverage90(samples, y):
    lo, hi = np.percentile(samples, [5, 95])
    return float(lo <= y <= hi)


def seed_condition_ratio(values):
    """values[condition, seed]. Mean within-condition variance / variance of condition means.
    >= 1 means the seed matters as much as the condition."""
    v = np.log(np.asarray(values, float))
    return float(v.var(axis=1, ddof=1).mean() / v.mean(axis=1).var(ddof=1))


def effective_score(scores, measurable):
    """Mean score with unmeasurable clips counted as 0, so dropping failures cannot inflate results."""
    s = np.where(np.asarray(measurable, bool), np.nan_to_num(np.asarray(scores, float)), 0.0)
    return float(s.mean())


def paired_bootstrap(a, b, groups, n_boot=10000, seed=0):
    """Mean of (a - b) with a 95% CI, resampling groups (scenes). Returns (mean, (lo, hi))."""
    a, b, g = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    ug = np.unique(g)
    per = np.array([np.mean(a[g == k] - b[g == k]) for k in ug])
    rng = np.random.default_rng(seed)
    boots = per[rng.integers(0, len(ug), (n_boot, len(ug)))].mean(1)
    return float(per.mean()), tuple(float(v) for v in np.percentile(boots, [2.5, 97.5]))


def holm(pvals):
    """Holm-adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


if __name__ == "__main__":
    # symmetric MRA: exact = 1, half and double score the same, 6x off = 0, wrong sign = 0
    assert sym_mra(9.81, 9.81) == 1.0
    assert np.isclose(sym_mra(4.905, 9.81), sym_mra(19.62, 9.81))
    assert sym_mra(9.81 * 6, 9.81) == 0.0 and sym_mra(-9.81, 9.81) == 0.0
    assert sym_mra(0.0, 0.0) == 1.0 and sym_mra(0.5, 0.0) == 0.0
    # original MRA is lopsided: with the wide set, half scores 0.4 and double 0
    wide = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    assert np.isclose(orig_mra(9.81 / 2, 9.81, wide), 0.4) and orig_mra(19.62, 9.81, wide) == 0.0
    assert orig_mra(9.81, 9.81) == 1.0
    # gamma: pure scale (clock) error keeps gamma = 1; ignoring the input gives 0
    t = np.array([4.9, 9.81, 19.6] * 5)
    assert np.isclose(gamma_fit(0.2 * t, t)[0], 1.0)
    assert np.isclose(gamma_fit(np.full_like(t, 9.81), t)[0], 0.0)
    # SILog removes a global scale
    assert silog(0.3 * t, t) < 1e-6
    # residual: parabola ~ 0, noise large
    tt = np.linspace(0, 2, 49)
    assert accel_residual(0.5 * 9.81 * tt ** 2, tt[1] - tt[0]) < 1e-6
    assert accel_residual(np.random.default_rng(0).normal(size=49), tt[1] - tt[0]) > 1
    # bias sign, effective score, seed ratio, CRPS, Holm
    assert signed_log_bias(0.5 * t, t) < 0
    assert effective_score([1.0, 1.0], [True, False]) == 0.5
    assert seed_condition_ratio([[1, 1.01, 0.99], [4, 4.02, 3.98]]) < 0.01
    assert crps_ensemble([9.81] * 3, 9.81) == 0.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    m, (lo, hi) = paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3])
    assert m == 1 and lo == 1 and hi == 1
    print("metrics self-check passed")
