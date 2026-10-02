"""
sentiment_data.py – Per-ticker daily positive/negative sentiment panel (R4.3).

Mirrors ``engine.macro_data``'s leakage contract but the construction differs:
macro series are persistent levels (ffill + shift(1) is enough), news is a
sparse flow, so each row is built directly as a half-life-weighted average of
same-sign headline scores published strictly *before* that date. That "< d"
cutoff already gives the same guarantee ``align_macro(lag=1)`` gives for macro
(row d holds only information known before d) — no further lag is applied.
"""

from __future__ import annotations

import math

import pandas as pd

from engine.logger import get_logger

log = get_logger(__name__)


def daily_sentiment_panel(
    ticker_dates: pd.Index | list[str],
    news_df: pd.DataFrame,
    *,
    lookback_days: int = 7,
    half_life_days: float = 3.0,
) -> pd.DataFrame:
    """Build a (sentiment_pos, sentiment_neg) panel, one row per ticker date.

    ``news_df`` — as returned by ``db_manager.get_news(ticker)`` — needs
    ``sentiment_score`` and ``published_at`` (falls back to ``date``) columns.
    Positive/negative are separate half-life-weighted averages of same-sign
    headline scores (negative reported as a positive magnitude, i.e. both
    columns are >= 0); a day with no qualifying news gets 0.0 in both.
    """
    dates = pd.to_datetime(pd.Index(ticker_dates)).sort_values()
    out_index = dates.strftime("%Y-%m-%d")
    empty = pd.DataFrame({"sentiment_pos": 0.0, "sentiment_neg": 0.0}, index=out_index)
    if news_df is None or news_df.empty:
        return empty

    df = news_df.copy()
    eff = df["published_at"].fillna(df.get("date")) if "published_at" in df.columns else df["date"]
    df["effective_date"] = pd.to_datetime(eff, errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["effective_date", "sentiment_score"]).sort_values("effective_date")
    if df.empty:
        return empty

    rows = []
    for d in dates:
        mask = df["effective_date"] < d
        if lookback_days > 0:
            mask &= df["effective_date"] >= (d - pd.Timedelta(days=lookback_days))
        sub = df[mask]
        if sub.empty:
            rows.append((0.0, 0.0))
            continue
        ages = (d - sub["effective_date"]).dt.days.to_numpy()
        rows.append(_weighted_pos_neg(sub["sentiment_score"].to_numpy(), ages, half_life_days))

    return pd.DataFrame(rows, columns=["sentiment_pos", "sentiment_neg"], index=out_index)


def _weighted_pos_neg(scores, ages, half_life_days: float) -> tuple[float, float]:
    pos_num = pos_den = neg_num = neg_den = 0.0
    for s, age in zip(scores, ages):
        w = math.pow(0.5, age / half_life_days) if half_life_days > 0 else 1.0
        if s > 0:
            pos_num += s * w
            pos_den += w
        elif s < 0:
            neg_num += -s * w
            neg_den += w
    pos = pos_num / pos_den if pos_den > 0 else 0.0
    neg = neg_num / neg_den if neg_den > 0 else 0.0
    return pos, neg


def fetch_sentiment_panel(
    api,
    ticker: str,
    df: pd.DataFrame,
    *,
    method: str = "vader",
    source: str = "gdelt",
    lookback_days: int = 180,
) -> pd.DataFrame | None:
    """Build ``ticker``'s leakage-safe daily sentiment panel for one scorer.

    Shared by ``scripts/forecast_harness.py`` and
    ``scripts/train_hybrid_residual.py`` so both the scoring and the
    pretraining path fetch sentiment identically. Uses whatever news is
    already in the DB, filtered to rows scored with ``method`` (this
    pipeline has no arbitrary historical-date backfill — GDELT/Yahoo only
    return recent news relative to "now", so old cached rows are the only
    real historical coverage available; see docs/forecasting-regression.md
    and plan.md R4.3). If nothing is cached for this method, one
    best-effort live top-up scores fresh headlines with it (never blocks
    the run on failure). Coverage may be partial (only the dates real news
    exists for) — days with no qualifying news get 0.0, not a fabricated
    value; report the coverage window honestly.
    """
    try:
        news_df = api.db.get_news(ticker)
        if news_df is not None and not news_df.empty and "method" in news_df.columns:
            news_df = news_df[news_df["method"] == method]
        if news_df is None or news_df.empty:
            try:
                api._process_news_with_db(
                    ticker,
                    method=method,
                    source=source,
                    lookback_days=lookback_days,
                    force_refresh=True,
                )
                news_df = api.db.get_news(ticker)
                if news_df is not None and not news_df.empty and "method" in news_df.columns:
                    news_df = news_df[news_df["method"] == method]
            except Exception as e:  # noqa: BLE001 — network/rate-limit; proceed with no news
                log.debug(
                    "%s: live news top-up (%s) failed (%s); using cache only.", ticker, method, e
                )
        dates = df["date"].astype(str) if "date" in df.columns else [str(i) for i in range(len(df))]
        return daily_sentiment_panel(dates, news_df)
    except Exception as e:  # noqa: BLE001 — one ticker's news failure shouldn't kill the run
        log.warning("%s: sentiment (%s) fetch/panel failed (%s); skipping.", ticker, method, e)
        return None


def demo() -> None:
    dates = pd.date_range("2026-01-01", periods=5).strftime("%Y-%m-%d")
    news = pd.DataFrame(
        {
            "published_at": ["2025-12-30", "2025-12-31", "2026-01-02"],
            "sentiment_score": [0.8, -0.4, 0.6],
        }
    )
    panel = daily_sentiment_panel(dates, news, lookback_days=7, half_life_days=3.0)
    assert list(panel.columns) == ["sentiment_pos", "sentiment_neg"]
    # 2026-01-01: sees only the two Dec news items (strictly before 01-01).
    assert panel.loc["2026-01-01", "sentiment_pos"] > 0
    assert panel.loc["2026-01-01", "sentiment_neg"] > 0
    # 2026-01-02: the 01-02 item itself must NOT be visible (< d, not <=).
    assert panel.loc["2026-01-02", "sentiment_pos"] == panel.loc["2026-01-01", "sentiment_pos"]
    # 2026-01-03: now the 01-02 item is visible → pos average changes.
    assert panel.loc["2026-01-03", "sentiment_pos"] != panel.loc["2026-01-02", "sentiment_pos"]
    print("sentiment_data.demo: OK")


if __name__ == "__main__":
    demo()
