"""
significance.py – Statistical-significance tests for backtests.

Why this module exists
----------------------

The backtester reports accuracy, return, Sharpe, Sortino — point estimates.
On ~40–300 trading days those point estimates are *noisy*: an accuracy of
0.54 over 50 days is entirely consistent with a coin flip. Before we read
any meaning into "model X beat 0.5", we need:

* a **p-value** for "is directional accuracy different from chance?"
  (exact **binomial test**, H0: p = 0.5),
* an **honest interval** on that accuracy (**Wilson score CI** — far better
  than the naive ``p ± z·√(p(1-p)/n)`` near 0 / 1 and for small n),
* an interval on the strategy **return** that respects fat tails
  (**bootstrap CI** — resample the daily P&L series), and
* a non-parametric check against a shuffled-direction null
  (**permutation test**).

And the critical caveat: across hundreds of model×period×ticker
combos, raw p-values are selection-inflated. So we also
provide **Benjamini-Hochberg FDR** correction to apply across the set of
models being compared in one report.

Everything here is **pure** (numpy + stdlib ``math`` only — no scipy, which
is only a transitive dependency here and may be absent). Console rendering
lives in ``backtest_helpers``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------
# Binomial test (exact two-sided, H0: p = 0.5)
# ----------------------------------------------------------------------


def binomial_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for ``k`` successes in ``n`` trials.

    Under H0 each trial succeeds with probability ``p``. The two-sided
    p-value is the total probability of every outcome *no more likely than*
    the observed one (the standard "method of small p" used by
    ``scipy.stats.binomtest``). For ``p = 0.5`` this reduces to the
    symmetric two-tailed sum.

    Returns 1.0 for ``n == 0``. Always in [0, 1].
    """
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))

    # Probability mass of the observed outcome.
    def pmf(i: int) -> float:
        return math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))

    observed = pmf(k)
    # Sum the mass of all outcomes at most as probable as the observed one.
    # A tiny epsilon guards against floating-point equality misses.
    eps = observed * 1e-9
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + eps)
    return float(min(1.0, total))


# ----------------------------------------------------------------------
# Wilson score confidence interval on a proportion
# ----------------------------------------------------------------------


# z critical values for the common two-sided confidence levels, so we don't
# need scipy's ppf. (0.90 → 1.6449, 0.95 → 1.9600, 0.99 → 2.5758.)
_Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def _z_for(confidence: float) -> float:
    return _Z.get(round(confidence, 2), 1.9600)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a lower/upper confidence bound."""

    point: float
    lo: float
    hi: float
    confidence: float


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion ``k/n``.

    Symmetric, well-behaved near 0/1 and for small ``n`` — unlike the normal
    approximation, it never produces bounds outside [0, 1]. Returns a
    degenerate ``[0, 1]`` interval around 0.5 for ``n == 0``.
    """
    if n <= 0:
        return Interval(point=0.0, lo=0.0, hi=1.0, confidence=confidence)
    z = _z_for(confidence)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return Interval(
        point=round(phat, 6),
        lo=round(max(0.0, center - margin), 6),
        hi=round(min(1.0, center + margin), 6),
        confidence=confidence,
    )


# ----------------------------------------------------------------------
# Bootstrap CI on a statistic of the daily P&L series
# ----------------------------------------------------------------------


def _sharpe(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(arr) / std * math.sqrt(252))


def bootstrap_ci(
    pnls: list[float],
    statistic: str = "sum",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap CI for a statistic of the daily P&L series.

    Resamples ``pnls`` with replacement ``n_boot`` times and takes the
    empirical percentile interval. ``statistic`` is one of:

      * ``"sum"`` — total return (Σ daily net P&L), the headline metric.
      * ``"mean"`` — mean daily return.
      * ``"sharpe"`` — annualized Sharpe of the resampled series.

    Returns a degenerate interval for an empty series. Seeded so reports
    reproduce exactly.
    """
    arr = np.asarray(pnls, dtype=float)
    n = arr.size
    if n == 0:
        return Interval(point=0.0, lo=0.0, hi=0.0, confidence=confidence)

    def stat(x: np.ndarray) -> float:
        if statistic == "sum":
            return float(np.sum(x))
        if statistic == "mean":
            return float(np.mean(x))
        if statistic == "sharpe":
            return _sharpe(x)
        raise ValueError(f"unknown statistic {statistic!r}")

    point = stat(arr)
    if n == 1:
        return Interval(
            point=round(point, 8), lo=round(point, 8), hi=round(point, 8), confidence=confidence
        )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    if statistic == "sum":
        stats = samples.sum(axis=1)
    elif statistic == "mean":
        stats = samples.mean(axis=1)
    else:  # sharpe
        means = samples.mean(axis=1)
        stds = samples.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            stats = np.where(stds < 1e-12, 0.0, means / stds * math.sqrt(252))

    alpha = 1.0 - confidence
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return Interval(
        point=round(point, 8),
        lo=round(lo, 8),
        hi=round(hi, 8),
        confidence=confidence,
    )


# ----------------------------------------------------------------------
# Permutation test vs a shuffled-direction null
# ----------------------------------------------------------------------


def permutation_test_accuracy(
    predicted: list[str],
    actual: list[str],
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """Permutation p-value for directional accuracy beating a random null.

    The null shuffles the *predicted* directions, breaking any relationship
    with ``actual`` while preserving the marginal UP/DOWN mix of the
    predictions. The p-value is the fraction of shuffles whose accuracy is
    ≥ the observed accuracy (one-sided: "is the model better than randomly
    permuted predictions?"). A (+1, +1) Laplace-style correction keeps the
    p-value strictly positive. Returns 1.0 for empty input.
    """
    n = len(predicted)
    if n == 0 or n != len(actual):
        return 1.0
    pred = np.array(predicted)
    act = np.array(actual)
    observed = float(np.mean(pred == act))

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pred)
        if float(np.mean(shuffled == act)) >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


# ----------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ----------------------------------------------------------------------


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control across a family of p-values.

    Returns a boolean list (aligned with the input order) of which
    hypotheses are rejected at false-discovery-rate ``alpha``. This is the
    plan's antidote to p-hacking the model×period×ticker grid: apply it
    across every model shown in one report rather than reading a single
    raw p-value.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    # Largest rank k with p_(k) <= (k/m)·alpha; reject all up to it.
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * alpha:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            rejected[idx] = True
    return rejected


# ----------------------------------------------------------------------
# Convenience: full significance bundle for one model's days
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceReport:
    """All significance tests for one model over one backtest."""

    n: int
    correct: int
    accuracy: float
    binomial_p: float
    wilson: Interval
    return_ci: Interval
    permutation_p: float


def significance_for_days(
    predicted: list[str],
    actual: list[str],
    pnls: list[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
) -> SignificanceReport:
    """Compute the full significance bundle from one model's per-day arrays.

    ``predicted``/``actual`` are the UP/DOWN strings of *traded* days;
    ``pnls`` is the matching net daily P&L. The caller is responsible for
    having already excluded gated/sat-out days.
    """
    n = len(predicted)
    correct = int(sum(1 for p, a in zip(predicted, actual, strict=False) if p == a))
    acc = correct / n if n else 0.0
    return SignificanceReport(
        n=n,
        correct=correct,
        accuracy=round(acc, 6),
        binomial_p=round(binomial_test_two_sided(correct, n), 6),
        wilson=wilson_interval(correct, n, confidence=confidence),
        return_ci=bootstrap_ci(pnls, "sum", confidence=confidence, n_boot=n_boot, seed=seed),
        permutation_p=round(
            permutation_test_accuracy(predicted, actual, n_perm=n_perm, seed=seed), 6
        ),
    )
