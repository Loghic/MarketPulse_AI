"""
baseline_models.py – Trivial "predictors" the real models must clear.

Before claiming any ML edge, beat the dumbest sensible baselines, not
just buy-and-hold. Each baseline implements the same
``predict(df, use_time_weights, sentiment_score) -> (direction, confidence)``
contract the trained models use, so they slot straight into the variants
list in ``backtest_helpers.run_single_backtest`` and share every metric.

The five baselines:

* ``AlwaysLongBaseline``         — predict UP every day
* ``PreviousDayBaseline``        — copy yesterday's realised direction
* ``MomentumBaseline(n)``        — UP iff ``close[t-1] > close[t-1-n]``
                                   (n=5 by default = "5-Day Momentum";
                                    instantiate with n=20 / 50 for the
                                    "Sign-Only Momentum" longer windows)
* ``RandomBaseline(seed)``       — deterministic coin flip seeded by the
                                   (training-window-last-date, seed)
                                   pair so re-running gives identical
                                   sequences

The ``sentiment_score`` argument is accepted for interface parity and
is deliberately ignored — baselines exist precisely to test what happens
without any model signal. There is no "+ News" variant for baselines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

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
# Factory used by backtest_helpers.run_single_backtest
# ----------------------------------------------------------------------


def default_baseline_variants() -> list[tuple[object, str]]:
    """Return ``[(model_instance, label), ...]`` for the default set.

    Centralising the choice of N values here means there is one place
    to edit if we ever want to add 50-day momentum or a second random
    seed.
    """
    return [
        (AlwaysLongBaseline(), AlwaysLongBaseline().name),
        (PreviousDayBaseline(), PreviousDayBaseline().name),
        (MomentumBaseline(n=5), MomentumBaseline(n=5).name),
        (MomentumBaseline(n=20), MomentumBaseline(n=20).name),
        (RandomBaseline(seed=42), RandomBaseline(seed=42).name),
    ]
