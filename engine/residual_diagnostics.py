"""
residual_diagnostics.py – Is a base model's residual white noise, or structured?

This is the empirical heart of the "*when* does residual learning help" question
(plan R6). The residual hybrid (`P̂ = base + r̂es`) can only beat the base if the
base's residuals contain **learnable structure** — i.e. autocorrelation the
residual learner can exploit. If the residuals are white noise, there is nothing
to learn and the hybrid reduces to the base.

So we diagnose the residual series ``res_t = y_true_t − base_pred_t``:

* **ACF** — sample autocorrelation at each lag. |ACF(1)| is a quick structure
  scalar.
* **Ljung–Box Q-test** — the headline: jointly tests "all autocorrelations up to
  lag h are zero" (white noise). A small p ⇒ the residual is autocorrelated
  (structured); a large p ⇒ indistinguishable from white noise.
* **Runs test** — non-parametric check for non-randomness in the sign sequence.
* **Variance ratio** — Lo–MacKinlay VR(q); ≈ 1 under a random walk, ≠ 1 under
  mean reversion / momentum.

Everything is pure numpy/stdlib. The only non-trivial dependency is a chi-square
survival function for the Ljung–Box p-value: we use ``scipy.stats.chi2`` when
available (the ``[forecast]`` extra) and fall back to a numpy regularised upper
incomplete gamma otherwise — so the module works with or without scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ----------------------------------------------------------------------
# chi-square survival (scipy if present, else numpy upper-incomplete-gamma)
# ----------------------------------------------------------------------


def _chi2_sf(x: float, dof: int) -> float:
    """P(χ²_dof > x). scipy if available, else regularised upper incomplete gamma."""
    if x <= 0:
        return 1.0
    if _SCIPY_AVAILABLE:
        return float(_scipy_stats.chi2.sf(x, dof))
    # Q(a, x) = Γ(a, x)/Γ(a), with a = dof/2, x' = x/2.
    return _gammaincc(dof / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a,x) (Numerical Recipes gcf/gser)."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Series for the lower P(a,x); Q = 1 − P.
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    # Continued fraction for Q(a,x) directly (Lentz).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# ----------------------------------------------------------------------
# Core diagnostics
# ----------------------------------------------------------------------


def _clean(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def acf(x: ArrayLike, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation at lags 0..nlags (acf[0] == 1 by construction)."""
    arr = _clean(x)
    n = arr.size
    if n < 2:
        return np.full(nlags + 1, np.nan)
    arr = arr - arr.mean()
    denom = float(np.dot(arr, arr))
    if denom <= 0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out
    out = np.empty(min(nlags, n - 1) + 1)
    for k in range(len(out)):
        out[k] = float(np.dot(arr[: n - k], arr[k:])) / denom
    if len(out) < nlags + 1:  # pad short series
        out = np.concatenate([out, np.full(nlags + 1 - len(out), np.nan)])
    return out


@dataclass(frozen=True)
class LjungBoxResult:
    lags: int
    stat: float
    p_value: float
    n: int


def ljung_box(x: ArrayLike, lags: int = 10) -> LjungBoxResult:
    """Ljung–Box Q-test for autocorrelation up to ``lags``.

    H0: the series is white noise (all ρ_k = 0, k=1..lags). A small p_value
    rejects H0 → the residual is autocorrelated (structured). Q ~ χ²_lags.
    """
    arr = _clean(x)
    n = arr.size
    if n <= lags + 1:
        return LjungBoxResult(lags=lags, stat=float("nan"), p_value=float("nan"), n=n)
    r = acf(arr, nlags=lags)
    q = 0.0
    for k in range(1, lags + 1):
        rk = r[k]
        if not np.isfinite(rk):
            continue
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    p = _chi2_sf(q, lags)
    return LjungBoxResult(lags=lags, stat=float(q), p_value=float(p), n=n)


@dataclass(frozen=True)
class RunsTestResult:
    n: int
    runs: int
    z: float
    p_value: float


def runs_test(x: ArrayLike) -> RunsTestResult:
    """Wald–Wolfowitz runs test on the sign sequence (randomness of signs).

    Counts runs of consecutive same-sign deviations from the mean and compares
    to the expected number under randomness via a normal approximation. Small p
    ⇒ the signs are non-random (structure).
    """
    arr = _clean(x)
    signs = np.sign(arr - arr.mean())
    signs = signs[signs != 0]
    n = signs.size
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    if n_pos == 0 or n_neg == 0 or n < 2:
        return RunsTestResult(n=n, runs=0, z=float("nan"), p_value=float("nan"))
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mean = 1.0 + 2.0 * n_pos * n_neg / n
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1))
    if var <= 0:
        return RunsTestResult(n=n, runs=runs, z=float("nan"), p_value=float("nan"))
    z = (runs - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return RunsTestResult(n=n, runs=runs, z=float(z), p_value=float(min(1.0, p)))


def variance_ratio(x: ArrayLike, q: int = 2) -> float:
    """Lo–MacKinlay variance ratio VR(q) on the series of increments.

    VR ≈ 1 ⇒ random-walk-like; VR > 1 ⇒ positive autocorrelation (momentum);
    VR < 1 ⇒ mean reversion. Computed on the differences of ``x`` (so pass the
    *residual* series; its first-difference structure is what matters).
    """
    arr = _clean(x)
    if arr.size < 2 * q + 1:
        return float("nan")
    diffs = np.diff(arr)
    n = diffs.size
    mu = diffs.mean()
    var1 = float(np.mean((diffs - mu) ** 2))
    if var1 <= 0:
        return float("nan")
    # q-period sums.
    sums = np.array([diffs[i : i + q].sum() for i in range(n - q + 1)])
    varq = float(np.mean((sums - q * mu) ** 2)) / q
    return varq / var1


# ----------------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualDiagnostics:
    """All structure diagnostics for one residual series."""

    n: int
    acf1: float  # autocorrelation at lag 1 (quick |structure| scalar)
    ljung_box_stat: float
    ljung_box_p: float
    ljung_box_lags: int
    runs_z: float
    runs_p: float
    variance_ratio: float
    structured: bool  # Ljung–Box rejects white noise at 5%

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class StructureGainRow:
    """One ticker's residual-structure vs hybrid-skill-gain cell — the paper's
    central cross-tab (plan R6.2)."""

    ticker: str
    n: int
    acf1: float
    ljung_box_stat: float
    ljung_box_p: float
    structured: bool
    u2_base: float
    u2_hybrid: float
    gain: float  # ΔU2 = U2_base − U2_hybrid; > 0 ⇒ the hybrid improved on the base


def structure_vs_gain(
    cases: list[tuple[str, ArrayLike, float, float]],
    *,
    lags: int = 10,
) -> list[StructureGainRow]:
    """Build the residual-structure vs hybrid-gain cross-tab.

    ``cases`` is ``(ticker, base_residuals, u2_base, u2_hybrid)``. For each, we
    diagnose the base residual's autocorrelation (Ljung–Box / ACF1) and pair it
    with the hybrid's skill gain ``ΔU2 = U2_base − U2_hybrid``.

    The paper's thesis: residual learning helps (gain > 0) **iff** the base
    residual is structured. A monotone relationship — more structure → more gain
    — confirms it; structure absent everywhere with gain ≈ 0 is the clean
    negative ("residuals are white noise, the hybrid can't beat the base").
    """
    rows: list[StructureGainRow] = []
    for ticker, resid, u2_base, u2_hybrid in cases:
        d = diagnose(resid, lags=lags)
        rows.append(
            StructureGainRow(
                ticker=ticker,
                n=d.n,
                acf1=d.acf1,
                ljung_box_stat=d.ljung_box_stat,
                ljung_box_p=d.ljung_box_p,
                structured=d.structured,
                u2_base=float(u2_base),
                u2_hybrid=float(u2_hybrid),
                gain=float(u2_base) - float(u2_hybrid),
            )
        )
    return rows


def diagnose(residuals: ArrayLike, *, lags: int = 10, vr_q: int = 2) -> ResidualDiagnostics:
    """Run every diagnostic on a residual series and bundle the result.

    ``structured`` is True when Ljung–Box rejects white noise (p < 0.05) — i.e.
    there is autocorrelation a residual learner could exploit.
    """
    arr = _clean(residuals)
    r = acf(arr, nlags=max(1, lags))
    lb = ljung_box(arr, lags=lags)
    rt = runs_test(arr)
    vr = variance_ratio(arr, q=vr_q)
    acf1 = float(r[1]) if r.size > 1 else float("nan")
    structured = bool(np.isfinite(lb.p_value) and lb.p_value < 0.05)
    return ResidualDiagnostics(
        n=int(arr.size),
        acf1=acf1,
        ljung_box_stat=lb.stat,
        ljung_box_p=lb.p_value,
        ljung_box_lags=lb.lags,
        runs_z=rt.z,
        runs_p=rt.p_value,
        variance_ratio=vr,
        structured=structured,
    )
