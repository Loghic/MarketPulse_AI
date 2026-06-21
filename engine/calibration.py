"""
calibration.py – Confidence calibration & gating metrics.

The question this module answers
--------------------------------

Every model emits a per-day ``confidence`` alongside its UP/DOWN call.
``confidence`` is the model's stated probability *of the direction it
chose* (so it is always ≥ 0.5). Before we let confidence drive anything
(e.g. sit-out-low-confidence-days gating), we have to know whether it is
**calibrated**: when a model says 70%, is it actually right ~70% of the
time? If the curve is flat — high-confidence days are no more accurate
than low-confidence days — then gating only shrinks exposure, it cannot
add edge.

The functions here are all **pure** — they take a list of ``DayResult``
(or plain ``(confidence, correct)`` pairs) and return numbers/dataclasses.
No I/O, no globals. The console rendering lives in ``backtest_helpers``;
the gate itself lives in ``backtester.py``.

Metrics
-------

* **Reliability bins** — group predictions into confidence buckets and
  compare *mean confidence* vs *observed accuracy* in each bucket. This
  is the data behind a reliability diagram.
* **Brier score** — mean squared error between confidence and the 0/1
  correctness outcome. Lower is better; 0.25 is the "always say 0.5"
  baseline for a binary outcome.
* **Expected Calibration Error (ECE)** — the bin-size-weighted average
  gap between confidence and accuracy across the reliability bins. 0 =
  perfectly calibrated.
* **Gating metrics** — given a threshold θ, what is the accuracy on the
  *traded* days (confidence ≥ θ), the coverage (% of days traded), the
  total return of the gated strategy, and the fees saved by sitting out.
"""

from __future__ import annotations

from dataclasses import dataclass

# We only depend on the *shape* of DayResult (``.confidence``, ``.correct``,
# ``.trade_pnl_net``, ``.trade_pnl``, ``.fee``-free), so we keep the import
# inside TYPE_CHECKING to avoid a hard engine.backtester import cycle.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from engine.backtester import DayResult


# ----------------------------------------------------------------------
# Reliability bins / diagram data
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of a reliability diagram.

    ``lo``/``hi`` are the bin edges; ``count`` predictions fell in it;
    ``mean_confidence`` is their average stated confidence and
    ``accuracy`` is the fraction that were actually correct. A
    well-calibrated model has ``mean_confidence ≈ accuracy`` in every bin.
    """

    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap (confidence − accuracy) for this bin."""
        return self.mean_confidence - self.accuracy


def reliability_bins(
    pairs: list[tuple[float, bool]],
    n_bins: int = 5,
    lo: float = 0.5,
    hi: float = 1.0,
) -> list[ReliabilityBin]:
    """Bucket ``(confidence, correct)`` pairs into equal-width bins.

    Confidence for a directional call lives in [0.5, 1.0], so the default
    range is [0.5, 1.0]. A prediction sitting exactly on the top edge is
    folded into the last bin. Empty bins are returned with count 0 so the
    diagram keeps a stable x-axis.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    width = (hi - lo) / n_bins
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        if conf < lo:
            idx = 0
        elif conf >= hi:
            idx = n_bins - 1
        else:
            idx = int((conf - lo) / width)
            idx = min(idx, n_bins - 1)
        buckets[idx].append((conf, correct))

    out: list[ReliabilityBin] = []
    for i, bucket in enumerate(buckets):
        edge_lo = lo + i * width
        edge_hi = lo + (i + 1) * width
        if bucket:
            mc = sum(c for c, _ in bucket) / len(bucket)
            acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        else:
            mc = 0.0
            acc = 0.0
        out.append(
            ReliabilityBin(
                lo=round(edge_lo, 6),
                hi=round(edge_hi, 6),
                count=len(bucket),
                mean_confidence=round(mc, 6),
                accuracy=round(acc, 6),
            )
        )
    return out


def brier_score(pairs: list[tuple[float, bool]]) -> float:
    """Mean squared error between confidence and the 0/1 correctness outcome.

    Because ``confidence`` is the probability of the *chosen* direction,
    the target is simply ``1.0`` when the call was correct and ``0.0``
    when it was wrong. Lower is better. Returns ``0.0`` for an empty input.
    """
    if not pairs:
        return 0.0
    total = sum((conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in pairs)
    return round(total / len(pairs), 6)


def expected_calibration_error(
    pairs: list[tuple[float, bool]],
    n_bins: int = 5,
    lo: float = 0.5,
    hi: float = 1.0,
) -> float:
    """Bin-size-weighted mean |confidence − accuracy| across reliability bins.

    ECE = Σ_b (n_b / N) · |conf_b − acc_b|. 0 = perfectly calibrated.
    Returns ``0.0`` for an empty input.
    """
    bins = reliability_bins(pairs, n_bins=n_bins, lo=lo, hi=hi)
    n = sum(b.count for b in bins)
    if n == 0:
        return 0.0
    ece = sum(b.count / n * abs(b.gap) for b in bins if b.count)
    return round(ece, 6)


# ----------------------------------------------------------------------
# Confidence gating
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GatingResult:
    """Outcome of applying a min-confidence gate to one model's days.

    ``threshold`` — the θ applied.
    ``traded`` / ``total`` — days that cleared the gate vs all days.
    ``coverage`` — traded / total.
    ``traded_accuracy`` — directional accuracy *on traded days only*
        (the headline "is the gated subset actually predictive?" number).
    ``gated_return`` — summed net P&L counting only traded days (sat-out
        days contribute 0 and pay no fee).
    ``ungated_return`` — summed net P&L of every day (θ = 0 baseline) for
        a direct comparison.
    ``fees_saved`` — round-trip fees not paid on the sat-out days.
    """

    threshold: float
    traded: int
    total: int
    coverage: float
    traded_accuracy: float
    gated_return: float
    ungated_return: float
    fees_saved: float


def gating_metrics(
    days: list[DayResult],
    threshold: float,
    fee_pct: float,
) -> GatingResult:
    """Compute gated-strategy metrics for one model at one threshold.

    A day is *traded* iff ``confidence >= threshold``. Sat-out days add
    nothing to the return and pay no fee. The fee saved per sat-out day is
    the round-trip fee ``2 * fee_pct / 100`` (matching
    ``Backtester._apply_fees``). This mirrors what the in-engine gate
    (``Backtester(min_confidence=θ)``) produces, but is computed
    post-hoc so a single un-gated backtest can be swept over many θ.
    """
    total = len(days)
    traded_days = [d for d in days if d.confidence >= threshold]
    traded = len(traded_days)
    sat_out = total - traded

    coverage = traded / total if total else 0.0
    correct = sum(1 for d in traded_days if d.correct)
    traded_accuracy = correct / traded if traded else 0.0
    gated_return = sum(d.trade_pnl_net for d in traded_days)
    ungated_return = sum(d.trade_pnl_net for d in days)
    fees_saved = sat_out * (2 * fee_pct / 100.0)

    return GatingResult(
        threshold=round(threshold, 6),
        traded=traded,
        total=total,
        coverage=round(coverage, 6),
        traded_accuracy=round(traded_accuracy, 6),
        gated_return=round(gated_return, 8),
        ungated_return=round(ungated_return, 8),
        fees_saved=round(fees_saved, 8),
    )


def gating_sweep(
    days: list[DayResult],
    thresholds: list[float],
    fee_pct: float,
) -> list[GatingResult]:
    """Run :func:`gating_metrics` for each θ in ``thresholds`` (sorted)."""
    return [gating_metrics(days, theta, fee_pct) for theta in sorted(set(thresholds))]


# ----------------------------------------------------------------------
# Convenience: pull (confidence, correct) pairs off a result
# ----------------------------------------------------------------------


def pairs_from_days(days: list[DayResult]) -> list[tuple[float, bool]]:
    """Extract ``(confidence, correct)`` from a list of DayResult."""
    return [(float(d.confidence), bool(d.correct)) for d in days]
