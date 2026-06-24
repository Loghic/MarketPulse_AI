"""
forecast_significance.py – Statistical comparison of point forecasts.

``significance.py`` answers the *directional* question (is accuracy ≠ 50%?).
This module answers the *forecast* question: **is model A's error genuinely
smaller than model B's, or is the gap noise?** A 0.998-vs-1.000 Theil U2 looks
like a win until you ask whether it's significant — usually it isn't, and that's
the publishable claim.

Two paired tests on the per-step errors ``e_t = y_true_t − y_pred_t``:

* **Diebold–Mariano (DM)** — the standard forecast-comparison test. On the loss
  differential ``d_t = g(e¹_t) − g(e²_t)`` (``g`` = squared or absolute), the DM
  statistic is ``d̄ / sqrt(HAC-var(d̄))`` with a Newey–West long-run variance
  (``h−1`` lags for an ``h``-step forecast). We apply the **Harvey–Leybourne–
  Newbold (HLN)** small-sample correction and compare to a Student-``t_{T−1}``.
  Pure numpy; uses ``scipy.stats.t`` for the p-value when available, else a
  numpy Student-t CDF fallback.
* **Wilcoxon signed-rank** — a non-parametric companion on the paired losses
  (no normality assumption). Uses ``scipy.stats.wilcoxon`` when installed
  (``[forecast]`` extra), else a numpy normal approximation with tie correction.

Across a model × ticker × horizon grid, raw p-values are multiple-comparison
inflated, so reuse ``significance.benjamini_hochberg`` (the same FDR rule the
directional track uses) — never read a single raw DM p off a 200-cell grid.

Sign convention: errors are ``model_1`` vs ``model_2``. ``DM < 0`` ⇒ model_1 has
the smaller loss (it's better); ``DM > 0`` ⇒ model_2 is better. So to test "does
the hybrid beat the random walk", pass ``(hybrid_errors, rw_errors)`` and a
significant **negative** statistic means the hybrid genuinely wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

try:
    # scipy ships a partial py.typed marker, so mypy flags import-untyped even
    # with ignore_missing_imports; scipy-stubs isn't a required dev dep here.
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ----------------------------------------------------------------------
# Student-t two-sided p-value (scipy if present, else numpy fallback)
# ----------------------------------------------------------------------


def _student_t_sf(t_abs: float, dof: float) -> float:
    """One-sided survival P(T > t_abs) for Student-t with ``dof`` d.o.f.

    Uses scipy when available; otherwise the regularised incomplete beta
    identity ``P(T>t) = 0.5 · I_{x}(dof/2, 1/2)`` with ``x = dof/(dof+t²)``,
    evaluated via a numpy continued-fraction ``betainc`` (no scipy needed).
    """
    if dof <= 0:
        return float("nan")
    if _SCIPY_AVAILABLE:
        return float(_scipy_stats.t.sf(t_abs, dof))
    x = dof / (dof + t_abs * t_abs)
    return 0.5 * _betainc(dof / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) via Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    import math

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    # Lentz's algorithm for the continued fraction.
    f, c, d = 1.0, 1.0, 0.0
    tiny = 1e-30
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        cd = c * d
        f *= cd
        if abs(1.0 - cd) < 1e-12:
            break
    # I_x(a,b) using the symmetry that keeps the CF convergent for our x range.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * (f - 1.0)
    return 1.0 - front * (f - 1.0)


# ----------------------------------------------------------------------
# Loss differential
# ----------------------------------------------------------------------


def _loss(errors: np.ndarray, kind: str) -> np.ndarray:
    if kind == "squared":
        return errors**2
    if kind == "absolute":
        return np.abs(errors)
    raise ValueError(f"loss must be 'squared' or 'absolute', got {kind!r}")


def _clean_errors(e1: ArrayLike, e2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(e1, dtype=float).ravel()
    b = np.asarray(e2, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"error arrays differ in length: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


# ----------------------------------------------------------------------
# Diebold–Mariano
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    """Diebold–Mariano comparison of model_1 vs model_2.

    ``stat`` < 0 ⇒ model_1 has the lower loss (better). ``mean_diff`` is the
    mean loss differential (g(e1) − g(e2)); negative ⇒ model_1 better.
    """

    n: int
    stat: float
    p_value: float
    mean_diff: float
    loss: str
    horizon: int


def dm_test(
    errors1: ArrayLike,
    errors2: ArrayLike,
    *,
    horizon: int = 1,
    loss: str = "squared",
) -> DMResult:
    """Harvey–Leybourne–Newbold-corrected Diebold–Mariano test.

    Args:
        errors1, errors2: per-step forecast errors ``y_true − y_pred`` for the
            two models, aligned step-for-step.
        horizon: forecast horizon ``h``; the HAC variance uses ``h−1`` lags.
        loss: ``"squared"`` (default) or ``"absolute"``.

    Returns a ``DMResult``. With < 2 paired points the statistic/p are NaN.
    """
    e1, e2 = _clean_errors(errors1, errors2)
    n = e1.size
    d = _loss(e1, loss) - _loss(e2, loss)
    if n < 2:
        return DMResult(
            n=n,
            stat=float("nan"),
            p_value=float("nan"),
            mean_diff=float(d.mean()) if n else float("nan"),
            loss=loss,
            horizon=horizon,
        )

    d_bar = float(d.mean())
    # Newey–West long-run variance of the mean with (h-1) lags.
    lags = max(0, horizon - 1)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    lrv = gamma0
    for k in range(1, lags + 1):
        if k >= n:
            break
        cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        weight = 1.0 - k / (lags + 1.0)  # Bartlett
        lrv += 2.0 * weight * cov
    if lrv <= 0:
        # Degenerate (e.g. identical forecasts → d≡0): no detectable difference.
        return DMResult(n=n, stat=0.0, p_value=1.0, mean_diff=d_bar, loss=loss, horizon=horizon)

    dm = d_bar / np.sqrt(lrv / n)
    # Harvey–Leybourne–Newbold small-sample correction.
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * hln

    dof = n - 1
    p = 2.0 * _student_t_sf(abs(dm_corrected), dof)
    p = float(min(1.0, max(0.0, p)))
    return DMResult(
        n=n, stat=float(dm_corrected), p_value=p, mean_diff=d_bar, loss=loss, horizon=horizon
    )


# ----------------------------------------------------------------------
# Wilcoxon signed-rank
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    n: int
    stat: float
    p_value: float
    loss: str


def wilcoxon_loss_test(
    errors1: ArrayLike,
    errors2: ArrayLike,
    *,
    loss: str = "squared",
) -> WilcoxonResult:
    """Wilcoxon signed-rank on the paired per-step losses (non-parametric DM companion).

    Uses ``scipy.stats.wilcoxon`` when available; otherwise a numpy normal
    approximation with tie correction. A small p with model_1's median loss
    below model_2's means model_1 is significantly better.
    """
    e1, e2 = _clean_errors(errors1, errors2)
    d = _loss(e1, loss) - _loss(e2, loss)
    d = d[d != 0.0]  # drop zero differences (Wilcoxon convention)
    n = d.size
    if n < 1:
        return WilcoxonResult(n=n, stat=float("nan"), p_value=1.0, loss=loss)

    if _SCIPY_AVAILABLE:
        try:
            res = _scipy_stats.wilcoxon(d, zero_method="wilcox", correction=False)
            return WilcoxonResult(
                n=n, stat=float(res.statistic), p_value=float(res.pvalue), loss=loss
            )
        except ValueError:
            return WilcoxonResult(n=n, stat=float("nan"), p_value=1.0, loss=loss)

    # numpy normal approximation with tie correction.
    ranks = _rankdata(np.abs(d))
    signs = np.sign(d)
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    # Tie correction term.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return WilcoxonResult(n=n, stat=w, p_value=1.0, loss=loss)
    z = (w - mean_w) / np.sqrt(var_w)
    # Two-sided normal p.
    import math

    p = math.erfc(abs(z) / math.sqrt(2.0))
    return WilcoxonResult(n=n, stat=float(w), p_value=float(min(1.0, p)), loss=loss)


# ----------------------------------------------------------------------
# Grid comparison vs a reference, with FDR control
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """One model-vs-reference comparison cell in the grid."""

    label: str  # e.g. "AAPL / Prophet+LSTM"
    n: int
    dm_stat: float
    dm_p: float
    wilcoxon_p: float
    mean_diff: float  # g(model) − g(reference); < 0 ⇒ model beats reference
    dm_significant: bool = False  # filled in after FDR across the family


def compare_to_reference(
    cases: list[tuple[str, ArrayLike, ArrayLike]],
    *,
    horizon: int = 1,
    loss: str = "squared",
    alpha: float = 0.05,
) -> list[ComparisonRow]:
    """DM + Wilcoxon for each (model errors vs reference errors), FDR-corrected.

    ``cases`` is a list of ``(label, model_errors, reference_errors)``. Each is
    tested with DM (and Wilcoxon); then Benjamini–Hochberg FDR is applied to the
    DM p-values **across the whole list** so a single cell can't be cherry-picked
    off a big grid (the same anti-p-hacking rule as the directional track). A row
    is ``dm_significant`` only if it survives FDR *and* its mean loss differential
    is negative (the model actually beats the reference, not just differs).
    """
    from engine.significance import benjamini_hochberg

    rows: list[ComparisonRow] = []
    for label, m_err, ref_err in cases:
        dm = dm_test(m_err, ref_err, horizon=horizon, loss=loss)
        wil = wilcoxon_loss_test(m_err, ref_err, loss=loss)
        rows.append(
            ComparisonRow(
                label=label,
                n=dm.n,
                dm_stat=dm.stat,
                dm_p=dm.p_value,
                wilcoxon_p=wil.p_value,
                mean_diff=dm.mean_diff,
            )
        )

    pvals = [r.dm_p if r.dm_p == r.dm_p else 1.0 for r in rows]  # NaN -> 1.0
    rejected = benjamini_hochberg(pvals, alpha=alpha)
    return [
        ComparisonRow(
            label=r.label,
            n=r.n,
            dm_stat=r.dm_stat,
            dm_p=r.dm_p,
            wilcoxon_p=r.wilcoxon_p,
            mean_diff=r.mean_diff,
            dm_significant=bool(rej and r.mean_diff < 0),
        )
        for r, rej in zip(rows, rejected, strict=True)
    ]


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank. (scipy-free)"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks
