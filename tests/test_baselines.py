"""test_baselines.py — Naive-strategy baselines.

Each baseline implements the same ``predict(df, use_time_weights,
sentiment_score) -> (direction, confidence)`` contract as the real
models. These tests pin the behaviour of every baseline on fixed
inputs and verify the interface guarantees the variants list relies
on.
"""

from __future__ import annotations

import pandas as pd

from config import BASELINE_NEWS_THRESHOLD
from engine.baseline_models import (
    AlwaysLongBaseline,
    AlwaysShortBaseline,
    HoldLongBaseline,
    MomentumBaseline,
    NewsAwareMomentumBaseline,
    NewsAwarePreviousDayBaseline,
    NewsInformedBaseline,
    PreviousDayBaseline,
    RandomBaseline,
    default_baseline_variants,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _df(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Build a minimal price DataFrame the baselines accept."""
    dates = pd.date_range(start=start, periods=len(closes), freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "close": closes})


# ----------------------------------------------------------------------
# AlwaysLong
# ----------------------------------------------------------------------


class TestAlwaysLong:
    def test_always_predicts_up(self):
        b = AlwaysLongBaseline()
        for closes in ([100], [100, 99], [100, 99, 98, 97, 96]):
            direction, conf = b.predict(_df(closes))
            assert direction == "UP"
            assert conf == 1.0

    def test_ignores_sentiment(self):
        b = AlwaysLongBaseline()
        direction, _ = b.predict(_df([100, 99]), sentiment_score=-1.0)
        assert direction == "UP"


# ----------------------------------------------------------------------
# PreviousDay
# ----------------------------------------------------------------------


class TestPreviousDay:
    def test_up_when_last_close_higher(self):
        b = PreviousDayBaseline()
        direction, conf = b.predict(_df([100, 102]))
        assert direction == "UP"
        assert 0.0 < conf < 1.0

    def test_down_when_last_close_lower(self):
        b = PreviousDayBaseline()
        direction, _ = b.predict(_df([100, 99]))
        assert direction == "DOWN"

    def test_insufficient_data_defaults_up(self):
        # Single bar: nothing to compare against → permissive default
        b = PreviousDayBaseline()
        direction, _ = b.predict(_df([100]))
        assert direction == "UP"

    def test_ignores_sentiment(self):
        b = PreviousDayBaseline()
        # Strong negative sentiment must NOT flip a clear UP signal.
        direction, _ = b.predict(_df([100, 105]), sentiment_score=-0.9)
        assert direction == "UP"


# ----------------------------------------------------------------------
# Momentum
# ----------------------------------------------------------------------


class TestMomentum:
    def test_up_when_close_above_lookback(self):
        b = MomentumBaseline(n=5)
        # 6 bars: close[-1]=110 vs close[-6]=100 → UP
        direction, _ = b.predict(_df([100, 101, 102, 103, 104, 110]))
        assert direction == "UP"

    def test_down_when_close_below_lookback(self):
        b = MomentumBaseline(n=5)
        direction, _ = b.predict(_df([100, 99, 98, 97, 96, 95]))
        assert direction == "DOWN"

    def test_n_changes_decision(self):
        # Same series, different lookback => different call.
        # closes[0..6] -> compare close[-1]=105 vs close[-1-N]
        closes = [100, 98, 96, 94, 102, 108, 105]
        b1 = MomentumBaseline(n=1)  # 105 vs 108 → DOWN
        b5 = MomentumBaseline(n=5)  # 105 vs 98  → UP
        d1, _ = b1.predict(_df(closes))
        d5, _ = b5.predict(_df(closes))
        assert d1 == "DOWN"
        assert d5 == "UP"

    def test_insufficient_data_defaults_up(self):
        b = MomentumBaseline(n=5)
        direction, _ = b.predict(_df([100, 101]))  # not enough rows
        assert direction == "UP"

    def test_label_includes_n(self):
        assert MomentumBaseline(n=5).name == "Baseline 5-Day Momentum"
        assert MomentumBaseline(n=20).name == "Baseline 20-Day Momentum"


# ----------------------------------------------------------------------
# Random
# ----------------------------------------------------------------------


class TestRandom:
    def test_deterministic_given_same_last_date(self):
        b = RandomBaseline(seed=42)
        df = _df([100, 101, 102])
        d1, _ = b.predict(df)
        d2, _ = b.predict(df)
        d3, _ = b.predict(df)
        assert d1 == d2 == d3

    def test_different_seeds_can_disagree(self):
        # We're not guaranteed disagreement on a single date, but across
        # 20 different dates two different seeds should produce different
        # sequences at least once.
        b1 = RandomBaseline(seed=1)
        b2 = RandomBaseline(seed=999)
        diffs = 0
        for i in range(20):
            df = _df([100 + j for j in range(i + 1)])
            if b1.predict(df)[0] != b2.predict(df)[0]:
                diffs += 1
        assert diffs > 0, "Two different seeds never disagreed on 20 dates"

    def test_confidence_is_half(self):
        b = RandomBaseline(seed=42)
        _, conf = b.predict(_df([100, 101]))
        assert conf == 0.5

    def test_ignores_sentiment(self):
        b = RandomBaseline(seed=42)
        df = _df([100, 101, 102])
        with_news, _ = b.predict(df, sentiment_score=0.9)
        without_news, _ = b.predict(df, sentiment_score=0.0)
        assert with_news == without_news


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


class TestFactory:
    def test_default_variants_cover_expected_set(self):
        variants = default_baseline_variants()
        labels = [label for _, label, _ in variants]
        for needle in (
            "Always Long",
            "Hold Long",
            "Always Short",
            "Previous Day",
            "5-Day Momentum",
            "20-Day Momentum",
            "Random",
            "News Previous Day",
            "News Informed",
            "News 5-Day Momentum",
        ):
            assert any(needle in lbl for lbl in labels), f"missing baseline: {needle}"

    def test_every_default_label_starts_with_baseline(self):
        # The variants filter resolves family via name.startswith("Baseline"),
        # so this prefix must hold for every default baseline.
        for _, label, _ in default_baseline_variants():
            assert label.startswith("Baseline"), (
                f"Baseline label {label!r} must start with 'Baseline' "
                "for --models filtering to recognise it"
            )

    def test_uses_news_flag(self):
        # Price-only baselines flagged False; the three news-aware ones True.
        by_label = {label: uses for _, label, uses in default_baseline_variants()}
        assert by_label["Baseline Always Long"] is False
        assert by_label["Baseline Always Short"] is False
        assert by_label["Baseline 5-Day Momentum"] is False
        assert by_label["Baseline News Previous Day"] is True
        assert by_label["Baseline News Informed"] is True
        assert by_label["Baseline News 5-Day Momentum"] is True

    def test_predict_signature_accepts_kwargs(self):
        # All variants must accept the same keyword set the backtester
        # invokes them with — silent TypeError there is hard to debug.
        df = _df([100, 101, 102, 103, 104, 105, 106])
        for model, _, _ in default_baseline_variants():
            direction, conf = model.predict(df, use_time_weights=True, sentiment_score=0.3)
            # UP/DOWN are calls; HOLD = buy-hold; FLAT = sit out (all valid).
            assert direction in ("UP", "DOWN", "HOLD", "FLAT")
            assert 0.0 <= conf <= 1.0


# ----------------------------------------------------------------------
# News-aware baselines
# ----------------------------------------------------------------------

# A clearly-strong and a clearly-weak sentiment relative to the threshold.
_STRONG = BASELINE_NEWS_THRESHOLD + 0.3
_WEAK = BASELINE_NEWS_THRESHOLD / 2


class TestHoldLong:
    def test_emits_hold_signal(self):
        b = HoldLongBaseline()
        for closes in ([100], [100, 99], [100, 101, 102]):
            direction, conf = b.predict(_df(closes))
            assert direction == "HOLD"
            assert conf == 1.0


class TestAlwaysShort:
    def test_always_predicts_down(self):
        b = AlwaysShortBaseline()
        for closes in ([100], [100, 101], [100, 99, 98]):
            direction, conf = b.predict(_df(closes))
            assert direction == "DOWN"
            assert conf == 1.0


class TestNewsAwarePreviousDay:
    def test_no_news_copies_previous_day(self):
        b = NewsAwarePreviousDayBaseline()
        assert b.predict(_df([100, 99]), sentiment_score=0.0)[0] == "DOWN"  # fell yesterday
        assert b.predict(_df([100, 101]), sentiment_score=0.0)[0] == "UP"  # rose yesterday

    def test_strong_news_overrides(self):
        b = NewsAwarePreviousDayBaseline()
        # Yesterday DOWN, but strong positive news flips to UP.
        assert b.predict(_df([100, 99]), sentiment_score=_STRONG)[0] == "UP"
        # Yesterday UP, but strong negative news flips to DOWN.
        assert b.predict(_df([100, 101]), sentiment_score=-_STRONG)[0] == "DOWN"

    def test_weak_news_does_not_override(self):
        b = NewsAwarePreviousDayBaseline()
        assert b.predict(_df([100, 99]), sentiment_score=_WEAK)[0] == "DOWN"


class TestNewsInformed:
    def test_strong_news_drives_prediction(self):
        b = NewsInformedBaseline()
        assert b.predict(_df([100, 101]), sentiment_score=_STRONG)[0] == "UP"
        assert b.predict(_df([100, 101]), sentiment_score=-_STRONG)[0] == "DOWN"

    def test_weak_news_sits_out_flat(self):
        b = NewsInformedBaseline()
        # Weak / no news → FLAT (no trade), regardless of yesterday's move.
        assert b.predict(_df([100, 101]), sentiment_score=_WEAK)[0] == "FLAT"
        assert b.predict(_df([100, 99]), sentiment_score=0.0)[0] == "FLAT"


class TestNewsAwareMomentum:
    def test_momentum_then_news_override(self):
        b = NewsAwareMomentumBaseline(n=5)
        rising = _df(list(range(100, 110)))  # clear up-momentum
        assert b.predict(rising, sentiment_score=0.0)[0] == "UP"
        assert b.predict(rising, sentiment_score=-_STRONG)[0] == "DOWN"  # news flips it

    def test_weak_news_keeps_momentum(self):
        b = NewsAwareMomentumBaseline(n=5)
        rising = _df(list(range(100, 110)))
        assert b.predict(rising, sentiment_score=-_WEAK)[0] == "UP"

    def test_name_includes_n(self):
        assert "5-Day" in NewsAwareMomentumBaseline(n=5).name
