"""
baseline_models.py – Trivial "predictors" the real models must clear.

Before claiming any ML edge, beat the dumbest sensible baselines, not
just buy-and-hold. Each baseline implements the same
``predict(df, use_time_weights, sentiment_score) -> (direction, confidence)``
contract the trained models use, so they slot straight into the variants
list in ``backtest_helpers.run_single_backtest`` and share every metric.

Price-only baselines (ignore news):

* ``AlwaysLongBaseline``         — predict UP every day (round-trips daily)
* ``HoldLongBaseline``           — buy once, hold to the end (no churn, one fee);
                                   emits the ``"HOLD"`` signal, always booked as
                                   a single buy-and-hold by the backtester
                                   regardless of turnover/position flags.
* ``AlwaysShortBaseline``        — predict DOWN every day (mirror of Always-Long;
                                   a check on how much the bull-market
                                   assumption is doing the work)
* ``PreviousDayBaseline``        — copy yesterday's realised direction
* ``MomentumBaseline(n)``        — UP iff ``close[t-1] > close[t-1-n]``
                                   (n=5 by default = "5-Day Momentum";
                                    instantiate with n=20 / 50 for the
                                    "Sign-Only Momentum" longer windows)
* ``RandomBaseline(seed)``       — deterministic coin flip seeded by the
                                   (training-window-last-date, seed)
                                   pair so re-running gives identical
                                   sequences

News-aware baselines (use the look-ahead-safe per-day ``sentiment_score``):

* ``NewsAwarePreviousDayBaseline`` — copy yesterday's direction, but flip to
                                   match the news when today's |sentiment|
                                   clears ``threshold`` ("assume continuation
                                   unless the news clearly says otherwise").
* ``NewsInformedBaseline``       — predict the news sign when |sentiment| ≥
                                   ``threshold``, else fall back to
                                   previous-day (the "person who only acts on
                                   clear news" baseline).
* ``NewsAwareMomentumBaseline(n)`` — n-day momentum, flipped to match the news
                                   when |sentiment| clears ``threshold``.

The news-aware ones remain **stateless** — they *react* to today's sentiment
with a fixed rule but never *learn* from past sentiment→outcome pairs. That
keeps them valid baselines (a fixed floor), not models. (A predictor that
learns the sentiment→direction relationship belongs as a real model, e.g. a
sentiment-conditioned k-NN, not here.) The price-only baselines accept
``sentiment_score`` for interface parity and ignore it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from config import BASELINE_NEWS_THRESHOLD


def _news_override(sentiment_score: float, threshold: float) -> str | None:
    """Direction the news implies, or None when it isn't strong enough.

    UP when sentiment ≥ +threshold, DOWN when ≤ −threshold, else None
    (too weak to act on). Pure function of today's look-ahead-safe score.
    """
    if sentiment_score >= threshold:
        return "UP"
    if sentiment_score <= -threshold:
        return "DOWN"
    return None


# Confidence values per baseline — used by confidence gating.
# AlwaysLong returns max confidence (it never "doubts"); the directional
# baselines sit at a token 0.55 (clearly above 0.5 so gating with θ=0.5
# admits them, but below 0.6 so a θ=0.6 gate excludes them); Random
# truthfully reports 0.5 since it IS a coin flip.
_CONF_ALWAYS = 1.0
_CONF_DIRECTIONAL = 0.55
_CONF_RANDOM = 0.5


def _last_date(df: pd.DataFrame) -> str:
    """Return the date string of the last row, or empty string if absent.

    Used to deterministically seed ``RandomBaseline`` per backtest day.
    """
    if "date" in df.columns and len(df) > 0:
        return str(df["date"].iloc[-1])
    return ""


@dataclass
class AlwaysLongBaseline:
    """Predict UP every day. The "is the model better than blind
    optimism in a bull market?" floor."""

    name: str = "Baseline Always Long"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002 - interface parity
        sentiment_score: float = 0.0,  # noqa: ARG002 - baselines ignore news
    ) -> tuple[str, float]:
        return "UP", _CONF_ALWAYS


@dataclass
class HoldLongBaseline:
    """Buy on day one, hold to the end — no daily churn, one round-trip fee.

    Returns the ``"HOLD"`` signal, which the backtester always books as a single
    compounded buy-and-hold trade (one fee, rides through intraday stops),
    *regardless* of --turnover-fees / --position-mode. This is the "just hold
    the ticker" reference: distinct from AlwaysLong, which predicts UP every day
    and (without position mode) round-trips daily and pays a fee each day. The
    honest 'did active trading add anything over passively holding?' floor."""

    name: str = "Baseline Hold Long"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002 - interface parity
        sentiment_score: float = 0.0,  # noqa: ARG002 - price-only
    ) -> tuple[str, float]:
        return "HOLD", _CONF_ALWAYS


@dataclass
class AlwaysShortBaseline:
    """Predict DOWN every day — the mirror of Always-Long.

    Useful as a control: if Always-Long looks good only because the market
    rose, Always-Short makes that explicit (it should look terrible in a
    bull market, mirroring Always-Long's success)."""

    name: str = "Baseline Always Short"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002 - interface parity
        sentiment_score: float = 0.0,  # noqa: ARG002 - baselines ignore news
    ) -> tuple[str, float]:
        return "DOWN", _CONF_ALWAYS


@dataclass
class PreviousDayBaseline:
    """Predict today = yesterday's realised direction.

    Yesterday's direction = ``sign(close[-1] - close[-2])``. If the data
    window has fewer than two rows we default to UP (mirrors the
    permissive behaviour of the real models on insufficient data).
    """

    name: str = "Baseline Previous Day"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,  # noqa: ARG002
    ) -> tuple[str, float]:
        if len(df) < 2 or "close" not in df.columns:
            return "UP", _CONF_DIRECTIONAL
        close = df["close"].astype(float)
        last, prev = close.iloc[-1], close.iloc[-2]
        direction = "UP" if last > prev else "DOWN"
        return direction, _CONF_DIRECTIONAL


@dataclass
class MomentumBaseline:
    """Predict UP iff ``close[-1] > close[-1-n]``.

    Equivalently: positive cumulative return over the last ``n`` days.
    ``n=5`` is the "5-Day Momentum" baseline; ``n=20`` / ``n=50`` are
    the "Sign-Only Momentum" longer windows.
    """

    n: int = 5

    @property
    def name(self) -> str:
        return f"Baseline {self.n}-Day Momentum"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,  # noqa: ARG002
    ) -> tuple[str, float]:
        if len(df) < self.n + 1 or "close" not in df.columns:
            return "UP", _CONF_DIRECTIONAL
        close = df["close"].astype(float)
        last, lookback = close.iloc[-1], close.iloc[-1 - self.n]
        direction = "UP" if last > lookback else "DOWN"
        return direction, _CONF_DIRECTIONAL


@dataclass
class RandomBaseline:
    """Seeded coin flip — the null hypothesis we want every real model
    to beat.

    Deterministic: the per-day flip is seeded by
    ``hash(last_date_in_train_df, seed)`` so the same backtest run
    gives the same predictions every time. That keeps results
    reproducible across CI runs without making the baseline trivially
    "lucky" via a single fixed seed.
    """

    seed: int = 42

    @property
    def name(self) -> str:
        return f"Baseline Random (seed={self.seed})"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,  # noqa: ARG002
    ) -> tuple[str, float]:
        date_str = _last_date(df)
        # First 8 bytes of SHA-256 give us a 64-bit space we can flip
        # a single bit out of; the leading bit decides UP/DOWN.
        digest = hashlib.sha256(f"{date_str}:{self.seed}".encode()).digest()
        bit = digest[0] & 0x01
        return ("UP" if bit else "DOWN"), _CONF_RANDOM


# ----------------------------------------------------------------------
# News-aware baselines — react to today's look-ahead-safe sentiment with a
# fixed rule (stateless; they never learn from past sentiment→outcome pairs).
# ----------------------------------------------------------------------


@dataclass
class NewsAwarePreviousDayBaseline:
    """Previous-day continuation, but flip to the news when it's strong.

    Default to yesterday's realised direction (the "things keep going the way
    they were" prior). If today's |sentiment| clears ``threshold``, override
    with the news direction — modelling someone who assumes continuation
    unless the headlines clearly say otherwise.
    """

    threshold: float = BASELINE_NEWS_THRESHOLD
    name: str = "Baseline News Previous Day"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,
    ) -> tuple[str, float]:
        # Base call = yesterday's direction (UP on insufficient data).
        if len(df) < 2 or "close" not in df.columns:
            base = "UP"
        else:
            close = df["close"].astype(float)
            base = "UP" if close.iloc[-1] > close.iloc[-2] else "DOWN"
        news = _news_override(sentiment_score, self.threshold)
        if news is not None and news != base:
            # News strong enough to override the price rule → higher conviction.
            return news, _CONF_DIRECTIONAL
        return base, _CONF_DIRECTIONAL


@dataclass
class NewsInformedBaseline:
    """Trade only on a clear headline; otherwise sit out (FLAT).

    When today's |sentiment| ≥ ``threshold``, predict the news sign. When the
    news is weak/neutral, return ``"FLAT"`` — a deliberate no-trade day (the
    backtester sits it out: 0 P&L, no fee, excluded from accuracy). This is the
    "only act on a clear signal, otherwise stay in cash" baseline, and it's what
    makes this distinct from ``NewsAwarePreviousDayBaseline`` (which always
    trades, defaulting to yesterday's direction). Expect low coverage — it only
    trades on the minority of days with strong news.
    """

    threshold: float = BASELINE_NEWS_THRESHOLD
    name: str = "Baseline News Informed"

    def predict(
        self,
        df: pd.DataFrame,  # noqa: ARG002 - decision is news-only
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,
    ) -> tuple[str, float]:
        news = _news_override(sentiment_score, self.threshold)
        if news is not None:
            return news, _CONF_DIRECTIONAL
        # Weak / no news → sit out today (no trade).
        return "FLAT", _CONF_RANDOM


@dataclass
class NewsAwareMomentumBaseline:
    """n-day momentum, flipped to the news when sentiment is strong.

    Same momentum rule as ``MomentumBaseline`` (UP iff
    ``close[-1] > close[-1-n]``), but if today's |sentiment| clears
    ``threshold`` the news direction overrides it.
    """

    n: int = 5
    threshold: float = BASELINE_NEWS_THRESHOLD

    @property
    def name(self) -> str:
        return f"Baseline News {self.n}-Day Momentum"

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,  # noqa: ARG002
        sentiment_score: float = 0.0,
    ) -> tuple[str, float]:
        if len(df) < self.n + 1 or "close" not in df.columns:
            base = "UP"
        else:
            close = df["close"].astype(float)
            base = "UP" if close.iloc[-1] > close.iloc[-1 - self.n] else "DOWN"
        news = _news_override(sentiment_score, self.threshold)
        if news is not None and news != base:
            return news, _CONF_DIRECTIONAL
        return base, _CONF_DIRECTIONAL


# ----------------------------------------------------------------------
# Factory used by backtest_helpers.run_single_backtest
# ----------------------------------------------------------------------


def default_baseline_variants() -> list[tuple[object, str, bool]]:
    """Return ``[(model_instance, label, uses_news), ...]`` for the default set.

    ``uses_news`` tells ``run_single_backtest`` whether to feed the baseline the
    look-ahead-safe per-day sentiment provider. Price-only baselines get
    ``False`` (no sentiment); the news-aware ones get ``True`` so they receive
    the same per-day sentiment the "+ News" model variants use.

    Centralising the choice of N values + the baseline set here means there is
    one place to edit if we ever want to add 50-day momentum or a second seed.
    """
    return [
        # Price-only.
        (AlwaysLongBaseline(), AlwaysLongBaseline().name, False),
        (HoldLongBaseline(), HoldLongBaseline().name, False),
        (AlwaysShortBaseline(), AlwaysShortBaseline().name, False),
        (PreviousDayBaseline(), PreviousDayBaseline().name, False),
        (MomentumBaseline(n=5), MomentumBaseline(n=5).name, False),
        (MomentumBaseline(n=20), MomentumBaseline(n=20).name, False),
        (RandomBaseline(seed=42), RandomBaseline(seed=42).name, False),
        # News-aware (stateless).
        (NewsAwarePreviousDayBaseline(), NewsAwarePreviousDayBaseline().name, True),
        (NewsInformedBaseline(), NewsInformedBaseline().name, True),
        (NewsAwareMomentumBaseline(n=5), NewsAwareMomentumBaseline(n=5).name, True),
    ]
