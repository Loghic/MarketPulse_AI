"""
backtest_helpers.py – Shared helpers for backtest.py and run_all.py.

Contains: period filtering, direction accuracy, export row builders,
and display/print functions.
"""

import csv
import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from config import (
    DEFAULT_NEWS_HALF_LIFE_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    FORECAST_MODELS,
    MODEL_FAMILY_LABELS,
    get_benchmarks,
)
from engine.backtester import BacktestResult
from engine.logger import get_logger
from engine.news_scraper import NewsScraper
from engine.utils import period_to_start_date

_FAMILY_ORDER = ["k-NN", "LinReg", "LSTM", "Prophet", "Chronos-2", "TiRex", "Kronos", "Other"]
_FAMILY_KEYS = {label: key for key, label in MODEL_FAMILY_LABELS.items()}

# (model, model_name, use_time_weights, sentiment_provider_or_None)
ModelVariant = tuple[Any, str, bool, Callable[[str], float] | None]

log = get_logger("helpers")


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Trim a price DataFrame to only include rows within the given period."""
    if period == "max":
        return df
    start = period_to_start_date(period)
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"]).dt.date
    filtered = df[df["_date"] >= start].drop(columns=["_date"])
    return filtered


def direction_accuracy(result: BacktestResult):
    """Return (up_correct, up_total, down_correct, down_total)."""
    up_c = sum(1 for d in result.days if d.predicted == "UP" and d.correct)
    up_t = sum(1 for d in result.days if d.predicted == "UP")
    dn_c = sum(1 for d in result.days if d.predicted == "DOWN" and d.correct)
    dn_t = sum(1 for d in result.days if d.predicted == "DOWN")
    return up_c, up_t, dn_c, dn_t


def compute_benchmarks(api, ticker: str, day_results) -> dict[str, float]:
    """
    Compute buy-and-hold return for benchmark indices over the same
    period as the backtest results.

    Reads directly from DB cache (no re-download). Benchmark data
    should already be present from the upfront refresh step.
    """
    if not day_results:
        return {}

    benchmarks = get_benchmarks(ticker)
    if not benchmarks:
        return {}

    start_date = day_results[0].date
    end_date = day_results[-1].date

    results = {}
    for bench in benchmarks:
        try:
            # Read from DB only — no freshness check, no re-download
            df = api.db.get_prices(bench)
            if df.empty:
                log.warning(f"Benchmark {bench} not in DB. Run refresh first.")
                continue

            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            period_df = df[mask]

            if len(period_df) < 2:
                continue

            entry = float(period_df["close"].iloc[0])
            exit_ = float(period_df["close"].iloc[-1])
            if entry > 0:
                results[bench] = round((exit_ - entry) / entry, 8)
        except Exception:
            continue

    return results


# ------------------------------------------------------------------
# Export row builders
# ------------------------------------------------------------------


def result_to_summary_row(
    r: BacktestResult, ticker: str, period: str, benchmarks: dict[str, float] | None = None
) -> dict:
    """Convert a BacktestResult to a summary export row."""
    up_c, up_t, dn_c, dn_t = direction_accuracy(r)
    row = {
        "ticker": ticker,
        "period": period,
        "model": r.model_name,
        "accuracy": round(r.accuracy, 4),
        "correct": r.correct,
        "total": r.test_days,
        "total_return": round(r.total_return, 8),
        "buy_hold_return": round(r.buy_hold_return, 8),
        "profit_factor": round(r.profit_factor, 4),
        "max_drawdown": round(r.max_drawdown, 8),
        "buy_hold_max_drawdown": round(r.buy_hold_max_drawdown, 8),
        "sharpe_ratio": round(r.sharpe_ratio, 4),
        "sortino_ratio": round(r.sortino_ratio, 4),
        "fee_pct": r.fee_pct,
        "stop_loss_pct": r.stop_loss_pct,
        "stopped_out": r.stopped_out_count,
        "turnover_fees": r.turnover_fees,
        "hold_days": r.hold_days,
        "turnover_count": r.turnover_count,
        "fees_paid": round(r.fees_paid, 8),
        "win_trades": r.win_trades,
        "loss_trades": r.loss_trades,
        "avg_win": round(r.avg_win, 8),
        "avg_loss": round(r.avg_loss, 8),
        "best_day": round(r.best_day, 8),
        "worst_day": round(r.worst_day, 8),
        "longest_win_streak": r.longest_win_streak,
        "longest_loss_streak": r.longest_loss_streak,
        "avg_win_streak": round(r.avg_win_streak, 1),
        "avg_loss_streak": round(r.avg_loss_streak, 1),
        "up_accuracy": round(up_c / up_t, 4) if up_t > 0 else None,
        "up_predictions": up_t,
        "down_accuracy": round(dn_c / dn_t, 4) if dn_t > 0 else None,
        "down_predictions": dn_t,
    }
    # Add benchmark returns as separate columns
    if benchmarks:
        for bench, ret in benchmarks.items():
            row[f"bench_{bench}"] = ret
    return row


def result_to_daily_rows(r: BacktestResult, ticker: str, period: str) -> list[dict]:
    """Convert a BacktestResult to per-day export rows (for --full)."""
    rows = []
    for d in r.days:
        rows.append(
            {
                "ticker": ticker,
                "period": period,
                "model": r.model_name,
                "date": d.date,
                "predicted": d.predicted,
                "actual": d.actual,
                "correct": d.correct,
                "confidence": round(d.confidence, 4),
                "trade_pnl": round(d.trade_pnl, 8),
                "trade_pnl_net": round(d.trade_pnl_net, 8),
                "fee_pct": r.fee_pct,
                "stopped_out": d.stopped_out,
                "exit_price": round(d.exit_price, 4),
                "close_before": round(d.close_before, 4),
                "close_actual": round(d.close_actual, 4),
            }
        )
    return rows


def export_rows(rows: list, output: str):
    """Export a list of dicts to CSV or JSON."""
    if not rows:
        print("  No data to export.")
        return
    if output.endswith(".json"):
        with open(output, "w") as f:
            json.dump(rows, f, indent=2)
    else:
        if not output.endswith(".csv"):
            output += ".csv"
        with open(output, "w", newline="") as f:
            all_keys: dict[str, None] = {}
            for row in rows:
                for k in row:
                    all_keys[k] = None
            writer = csv.DictWriter(f, fieldnames=list(all_keys.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults exported to {output}")


# ------------------------------------------------------------------
# Run models for a single ticker × period
# ------------------------------------------------------------------


def run_single_backtest(
    api,
    backtester,
    ticker,
    df,
    period,
    n_days,
    full,
    news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    news_half_life_days: float = DEFAULT_NEWS_HALF_LIFE_DAYS,
    sentiment_method: str | None = None,
    models: list[str] | None = None,
    include_baselines: bool = True,
    sl_levels: list[float] | None = None,
):
    """
    Run backtest for one ticker × one period. Returns list of BacktestResult.

    News handling (no look-ahead):
        Variants whose name contains "+ News" use a per-day sentiment
        provider that looks up news strictly published BEFORE the
        prediction date, within the ``news_lookback_days`` window, and
        weighted by exponential decay with half-life
        ``news_half_life_days`` (0 = no decay).

        Variants without "+ News" pass sentiment_score=0.0 (no news).

    Args:
        news_lookback_days: How many days of news to consider. 0 = unbounded.
        news_half_life_days: Decay half-life for the per-day weighted score.
            0 disables decay (uniform within window).
        sentiment_method: Restrict to news scored with this method
            ("vader" | "finbert" | "naive"). None = whatever's in the DB.

    Stop-loss:
        ``sl_levels`` (e.g. ``[0, 5, 10, 15]``) runs each model once per
        level — a stop-loss sweep. ``0`` is the no-SL baseline; non-zero
        levels get an ``SL{n}%`` name suffix. When ``sl_levels`` is None we
        fall back to the legacy behaviour driven by the passed backtester's
        ``stop_loss_pct``: if it's > 0, each model runs twice (no-SL baseline
        + SL); if 0, once. Turnover/hold settings are inherited from the
        passed backtester across every level.
    """
    filtered = filter_by_period(df, period)
    if len(filtered) < n_days + 20:
        return []

    # Refresh today's news so the DB has at least *something* to query for
    # the most-recent backtest days. Historical days still need historical
    # news to have been ingested separately (e.g. via GDELT refresh).
    # Tolerate transient DB hiccups (we'll fall through to a no-news backtest).
    try:
        _, headlines_today = api._process_news_with_db(ticker, method=sentiment_method)
    except Exception as e:  # noqa: BLE001
        log.warning(f"{ticker}: news refresh skipped ({e}); proceeding without news.")
        headlines_today = []

    # Pre-fetch the FULL news history for this ticker ONCE and filter in
    # memory in the per-day closure. The old per-call ``get_news_before``
    # path opened thousands of SQLite connections (one per backtest day ×
    # per "+ News" variant) which exhausted file descriptors on macOS during
    # long ``run_all.py`` jobs. The in-memory version is also ~100× faster.
    try:
        all_news_df: pd.DataFrame = api.db.get_news(ticker)
    except Exception as e:  # noqa: BLE001
        log.warning(f"{ticker}: could not load news history ({e}); proceeding without news.")
        all_news_df = pd.DataFrame()

    if not all_news_df.empty:
        # Restrict to the requested scoring method (NULL rows from the
        # pre-2026 schema are kept for backward compatibility).
        if sentiment_method is not None and "method" in all_news_df.columns:
            mask = (all_news_df["method"] == sentiment_method) | all_news_df["method"].isna()
            all_news_df = all_news_df[mask]
        # Compute the effective publication date once.
        if "published_at" in all_news_df.columns:
            all_news_df = all_news_df.assign(
                effective_date=all_news_df["published_at"].fillna(all_news_df["date"])
            )
        else:
            all_news_df = all_news_df.assign(effective_date=all_news_df["date"])

    has_news = bool(headlines_today) or not all_news_df.empty

    # Build a per-day sentiment provider that uses only news from BEFORE
    # the prediction date. This is the look-ahead-safe path — same
    # semantics as ``db.get_news_before`` but evaluated in memory.
    def per_day_sentiment(prediction_date: str) -> float:
        if all_news_df.empty:
            return 0.0
        mask = all_news_df["effective_date"] < prediction_date
        if news_lookback_days and news_lookback_days > 0:
            cutoff = (
                pd.to_datetime(prediction_date) - pd.Timedelta(days=news_lookback_days)
            ).strftime("%Y-%m-%d")
            mask &= all_news_df["effective_date"] >= cutoff
        subset = all_news_df[mask]
        if subset.empty:
            return 0.0
        return NewsScraper.weighted_score(
            subset,
            asof_date=prediction_date,
            half_life_days=news_half_life_days,
        )

    # (model, name, use_time_weights, sentiment_provider_or_None)
    # When the fourth tuple element is None, no news is used (constant 0).
    # Explicit type annotation so mypy doesn't narrow the slot to ``None``
    # (we extend with callable-bearing tuples below for the "+ News" variants).
    variants: list[ModelVariant] = [
        (api.knn, "k-NN", False, None),
        (api.knn, "k-NN Time-Weighted", True, None),
        (api.knn_enhanced, "k-NN Enhanced", False, None),
        (api.knn_enhanced, "k-NN Enh. TW", True, None),
        (api.linreg, "LinReg", False, None),
        (api.linreg, "LinReg Time-Weighted", True, None),
        (api.linreg_enhanced, "LinReg Enhanced", False, None),
        (api.linreg_enhanced, "LinReg Enh. TW", True, None),
    ]

    if has_news:
        variants.extend(
            [
                (api.knn, "k-NN TW + News", True, per_day_sentiment),
                (api.knn_enhanced, "k-NN Enh. TW + News", True, per_day_sentiment),
                (api.linreg, "LinReg TW + News", True, per_day_sentiment),
                (api.linreg_enhanced, "LinReg Enh. TW + News", True, per_day_sentiment),
            ]
        )

    # Add LSTM if available
    if api.lstm_available:
        lstm_model = api._load_lstm(ticker, period)
        if lstm_model:
            variants.append((lstm_model, "LSTM", False, None))
            if has_news:
                variants.append((lstm_model, "LSTM + News", False, per_day_sentiment))
        else:
            log.info(
                f"No trained LSTM for {ticker} (period={period}). "
                f"Train: uv run python train.py --ticker {ticker} --period {period}"
            )

    # Naive baselines (Phase 1.2): the bar real models must clear.
    # They ignore sentiment_score so we only emit a plain (non-News)
    # variant for each — no "+ News" sibling.
    if include_baselines:
        from engine.baseline_models import default_baseline_variants

        for baseline_model, label in default_baseline_variants():
            variants.append((baseline_model, label, False, None))

    # Forecasting models (Prophet, Chronos-2, …): one plain variant each,
    # plus "+ News" when news exists. Flows through the same SL-duplication
    # and Backtester.run loop below, unchanged.
    for mt, label in FORECAST_MODELS:
        if not api.forecast_available(mt):
            continue
        try:
            fmodel = api._get_model(mt, ticker, period)
        except Exception as e:  # noqa: BLE001
            log.info(f"{label} unavailable for {ticker}: {e}")
            continue
        variants.append((fmodel, label, False, None))
        if has_news:
            variants.append((fmodel, f"{label} + News", False, per_day_sentiment))

    # --- Keep only requested model families (--models). None/empty = all. ---
    if models:
        allowed = set(models)
        variants = [v for v in variants if _family_key(v[1]) in allowed]

    from engine.backtester import Backtester as BT

    # Resolve the stop-loss levels to run each model at.
    #   sl_levels given  → run once per level (the sweep). 0 = no-SL baseline.
    #   sl_levels None   → legacy: if the passed backtester has SL on, run the
    #                      no-SL baseline + the SL level; otherwise a single run.
    if sl_levels is not None:
        # De-dupe + sort, keep 0 first so the baseline reads naturally.
        levels = sorted(set(round(float(s), 6) for s in sl_levels))
    elif backtester.stop_loss_pct > 0:
        levels = [0.0, float(backtester.stop_loss_pct)]
    else:
        levels = [0.0]

    # One reusable backtester per level, inheriting fee / gate / turnover /
    # hold settings from the passed instance so only the SL knob varies.
    def _bt_for(sl: float) -> BT:
        if sl == backtester.stop_loss_pct:
            return backtester  # reuse the caller's instance
        return BT(
            n_days=backtester.n_days,
            fee_pct=backtester.fee_pct,
            stop_loss_pct=sl,
            min_confidence=backtester.min_confidence,
            turnover_fees=backtester.turnover_fees,
            hold_days=backtester.hold_days,
        )

    backtesters = [(sl, _bt_for(sl)) for sl in levels]

    results = []
    for model, name, tw, sent_provider in variants:
        for sl, bt in backtesters:
            label = name if sl == 0 else f"{name} SL{sl:g}%"
            results.append(
                bt.run(
                    model=model,
                    model_name=label,
                    df=filtered,
                    ticker=ticker,
                    use_time_weights=tw,
                    sentiment_provider=sent_provider,
                )
            )

    return results


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------


def pf_str(pf: float) -> str:
    """Format profit factor for display."""
    return f"{pf:.2f}" if pf < 100 else "∞"


def _family_key(name: str) -> str:
    """Lowercase family key for --models filtering (e.g. 'k-NN Enh. TW' -> 'knn')."""
    return _FAMILY_KEYS.get(_model_family(name), "other")


def _model_family(name: str) -> str:
    for fam in (
        "k-NN",
        "LinReg",
        "LSTM",
        "Prophet",
        "Chronos-2",
        "Chronos",
        "TiRex",
        "Kronos",
        "Baseline",
    ):
        if name.startswith(fam):
            return "Chronos-2" if fam == "Chronos" else fam
    return "Other"


def print_summary_table(results, show_buy_hold=False, benchmarks=None, group=True):
    """Summary table grouped by model family and ranked by return within group."""
    if not results:
        print("  No results.")
        return
    has_sl = any(r.stop_loss_pct > 0 for r in results)
    best_ret = max(r.total_return for r in results)

    header = (
        f"  {'MODEL':<25}| {'ACC.':<6}| {'RETURN':<11}| "
        f"{'PF':<6}| {'MAX DD':<10}| {'SHARPE':<7}| {'W/L':<7}"
    )
    if has_sl:
        header += f"| {'SL':<4}"
    if show_buy_hold:
        header += f"| {'B&H':<11}"
    if benchmarks:
        for b in benchmarks:
            header += f"| {b:<10}"
    width = len(header)
    print(header)
    print(f"  {'─' * (width - 2)}")

    def fmt_row(r):
        star = "★" if r.total_return == best_ret else " "
        wl = f"{r.win_trades}/{r.loss_trades}"
        line = (
            f"  {r.model_name:<23}{star} | {r.accuracy:<5.0%}| "
            f"{r.total_return:<+11.4%}| {pf_str(r.profit_factor):<6}| "
            f"{r.max_drawdown:<+10.4%}| {r.sharpe_ratio:<7.2f}| {wl:<7}"
        )
        if has_sl:
            line += f"| {r.stopped_out_count:<4}"
        if show_buy_hold:
            line += f"| {r.buy_hold_return:<+11.4%}"
        if benchmarks:
            for _b, ret in benchmarks.items():
                line += f"| {ret:<+10.4%}"
        return line

    if group:
        groups: dict[str, list] = {}
        for r in results:
            groups.setdefault(_model_family(r.model_name), []).append(r)
        ordered = [f for f in _FAMILY_ORDER if f in groups] + [
            f for f in groups if f not in _FAMILY_ORDER
        ]
        for fam in ordered:
            rows = sorted(groups[fam], key=lambda r: r.total_return, reverse=True)
            label = f"  ── {fam} "
            print(label + "─" * max(0, width - len(label) - 1))
            for r in rows:
                print(fmt_row(r))
    else:
        for r in sorted(results, key=lambda r: r.total_return, reverse=True):
            print(fmt_row(r))

    print(f"  {'─' * (width - 2)}")
    winner = max(results, key=lambda r: r.total_return)
    print(
        f"  ★ Best return: {winner.model_name}  "
        f"({winner.total_return:+.2%}, acc {winner.accuracy:.0%}, Sharpe {winner.sharpe_ratio:.2f})"
    )
    if show_buy_hold:
        bh = results[0].buy_hold_return
        beat = sum(1 for r in results if r.total_return > bh)
        print(f"    Buy & Hold: {bh:+.2%}   |   models beating B&H: {beat}/{len(results)}")


def print_timing_table(results, top=None):
    """Per-model compute time, slowest first. Toggled by backtest.py --timing."""
    times: dict[str, float] = {}
    runs: dict[str, int] = {}
    for r in results:
        secs = getattr(r, "elapsed_seconds", 0.0)
        times[r.model_name] = times.get(r.model_name, 0.0) + secs
        runs[r.model_name] = runs.get(r.model_name, 0) + 1
    total = sum(times.values())
    if not times or total == 0:
        print("\n  TIMING: no data — is engine/backtester.py recording elapsed_seconds?")
        return
    multi = any(c > 1 for c in runs.values())
    rows = sorted(times.items(), key=lambda kv: kv[1], reverse=True)
    if top:
        rows = rows[:top]

    header = f"  {'MODEL':<25}| {'TIME':>9}| {'SHARE':>6}"
    if multi:
        header += f"| {'RUNS':>5}| {'PER-RUN':>9}"
    print("\n  TIMING BY MODEL  (compute time, slowest first)")
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for name, secs in rows:
        line = f"  {name:<25}| {secs:>8.2f}s| {secs / total:>5.0%}"
        if multi:
            n = runs[name]
            line += f"| {n:>5}| {secs / n:>8.2f}s"
        print(line)
    print(f"  {'─' * (len(header) - 2)}")
    print(f"  {'TOTAL':<25}| {total:>8.2f}s")


def print_consensus(all_results, n_days):
    """Day-by-day consensus."""
    if not all_results or not all_results[0].days:
        return

    dates = [d.date for d in all_results[0].days]
    print(f"\n  {'DAY-BY-DAY CONSENSUS':=^66}")
    header = f"  {'DATE':<12} |"
    for r in all_results:
        short = r.model_name.replace("Time-Weighted", "TW").replace(" + News", "+N")
        header += f" {short:<8} |"
    header += f" {'ACTUAL':<8} | {'AGREE'}"
    print(header)
    print(f"  {'-' * (14 + len(all_results) * 11 + 20)}")

    for i, date_str in enumerate(dates):
        row = f"  {date_str:<12} |"
        predictions = []
        actual = None
        for r in all_results:
            if i < len(r.days):
                d = r.days[i]
                mark = "✓" if d.correct else "✗"
                row += f" {d.predicted:<5}{mark}  |"
                predictions.append(d.predicted)
                actual = d.actual
        if actual:
            up_count = predictions.count("UP")
            down_count = predictions.count("DOWN")
            total = len(predictions)
            majority = "UP" if up_count >= down_count else "DOWN"
            agree_pct = max(up_count, down_count) / total if total > 0 else 0
            consensus_correct = "✓" if majority == actual else "✗"
            row += f" {actual:<8} | {agree_pct:.0%} {consensus_correct}"
            if agree_pct == 1.0:
                row += " ◄ unanimous"
        print(row)


def print_direction_accuracy(all_results):
    """Per-direction accuracy breakdown."""
    print(f"\n  {'DIRECTION ACCURACY':=^66}")
    print(f"  {'MODEL':<25} | {'UP acc.':<12} | {'DOWN acc.':<12}")
    print(f"  {'-' * 55}")
    for r in all_results:
        up_c, up_t, dn_c, dn_t = direction_accuracy(r)
        up_s = f"{up_c}/{up_t} ({up_c / up_t:.0%})" if up_t > 0 else "n/a"
        dn_s = f"{dn_c}/{dn_t} ({dn_c / dn_t:.0%})" if dn_t > 0 else "n/a"
        print(f"  {r.model_name:<25} | {up_s:<12} | {dn_s:<12}")


def print_confidence_calibration(all_results):
    """Are high-confidence predictions actually more accurate?

    First the original high/low split, then per-model Brier score + ECE — the
    summary numbers that say whether confidence is calibrated at all. If ECE
    is small and high-confidence accuracy clearly beats low, gating can add
    edge; if the curve is flat, gating only shrinks exposure.
    """
    from engine.calibration import brier_score, expected_calibration_error, pairs_from_days

    print(f"\n  {'CONFIDENCE CALIBRATION':=^66}")
    print(f"  {'MODEL':<25} | {'High (>65%)':<16} | {'Low (≤65%)':<16}")
    print(f"  {'-' * 62}")
    for r in all_results:
        high = [d for d in r.days if d.confidence > 0.65]
        low = [d for d in r.days if d.confidence <= 0.65]
        ha = sum(1 for d in high if d.correct) / len(high) if high else 0
        la = sum(1 for d in low if d.correct) / len(low) if low else 0
        hs = f"{ha:.0%} ({len(high)} pred)" if high else "n/a"
        ls = f"{la:.0%} ({len(low)} pred)" if low else "n/a"
        print(f"  {r.model_name:<25} | {hs:<16} | {ls:<16}")

    # Brier + ECE (lower = better calibrated). 0.25 Brier = "always 0.5".
    print(f"\n  {'CALIBRATION SCORES (lower = better)':=^66}")
    print(f"  {'MODEL':<25} | {'BRIER':<10} | {'ECE':<10} | {'N':<6}")
    print(f"  {'-' * 58}")
    for r in all_results:
        pairs = pairs_from_days(r.days)
        bs = brier_score(pairs)
        ece = expected_calibration_error(pairs)
        print(f"  {r.model_name:<25} | {bs:<10.4f} | {ece:<10.4f} | {len(pairs):<6}")


def print_confidence_sweep(all_results, fee_pct: float, thresholds: list[float] | None = None):
    """θ-sweep table: coverage / traded-day accuracy / return /
    fees saved at each confidence gate, computed post-hoc from one ungated run.

    Pass bar: traded-day accuracy materially > 0.5 AND return improves vs θ=0.
    """
    from config import CONFIDENCE_SWEEP
    from engine.calibration import gating_sweep

    thresholds = thresholds or list(CONFIDENCE_SWEEP)
    print(f"\n  {'CONFIDENCE GATING SWEEP':=^66}")
    print(
        f"  {'MODEL':<22} | {'θ':<6} | {'COVERAGE':<14} | "
        f"{'TRADED ACC':<11} | {'RETURN':<11} | {'FEES SAVED':<10}"
    )
    print(f"  {'-' * 86}")
    for r in all_results:
        rows = gating_sweep(r.days, thresholds, fee_pct)
        for i, g in enumerate(rows):
            name = r.model_name if i == 0 else ""
            cov = f"{g.traded}/{g.total} ({g.coverage:.0%})"
            acc = f"{g.traded_accuracy:.1%}" if g.traded else "n/a"
            print(
                f"  {name:<22} | {g.threshold:<6.2f} | {cov:<14} | "
                f"{acc:<11} | {g.gated_return:<+11.4%} | {g.fees_saved:<+10.4%}"
            )
        print(f"  {'-' * 86}")


def print_significance(all_results, confidence: float = 0.95):
    """Statistical-significance tests: binomial p + Wilson CI on
    accuracy, bootstrap CI on return, permutation p vs shuffled directions,
    with Benjamini-Hochberg FDR across the models shown.

    Only *traded* days (confidence ≥ gate) feed the tests, matching what the
    reported accuracy/return describe.
    """
    from engine.significance import benjamini_hochberg, significance_for_days

    print(f"\n  {'STATISTICAL SIGNIFICANCE':=^66}")
    reports = []
    for r in all_results:
        traded = [d for d in r.days if d.traded]
        rep = significance_for_days(
            [d.predicted for d in traded],
            [d.actual for d in traded],
            [d.trade_pnl_net for d in traded],
            confidence=confidence,
        )
        reports.append(rep)

    # FDR across the binomial p-values of every model in this report.
    rejected = benjamini_hochberg([rep.binomial_p for rep in reports], alpha=1 - confidence)

    print(
        f"  {'MODEL':<22} | {'ACC':<7} | {'ACC ' + str(int(confidence * 100)) + '% CI':<16} | "
        f"{'BINOM p':<9} | {'PERM p':<8} | {'RET CI':<22} | FDR✓"
    )
    print(f"  {'-' * 104}")
    for r, rep, rej in zip(all_results, reports, rejected, strict=False):
        ci = f"[{rep.wilson.lo:.2f}, {rep.wilson.hi:.2f}]"
        ret_ci = f"[{rep.return_ci.lo:+.2%}, {rep.return_ci.hi:+.2%}]"
        mark = "✓" if rej else ""
        print(
            f"  {r.model_name:<22} | {rep.accuracy:<7.1%} | {ci:<16} | "
            f"{rep.binomial_p:<9.4f} | {rep.permutation_p:<8.4f} | {ret_ci:<22} | {mark}"
        )
    print(
        f"  {'-' * 104}\n"
        f"  FDR✓ = accuracy ≠ 0.5 survives Benjamini-Hochberg at "
        f"α={1 - confidence:.2f} across the {len(reports)} models above.\n"
        f"  Binomial H0: accuracy = 0.5. Return CI = {int(confidence * 100)}% "
        f"bootstrap on daily net P&L (a CI spanning 0 = return indistinguishable from flat)."
    )


def print_profit_analysis(
    all_results, show_buy_hold: bool = False, benchmarks: dict[str, float] | None = None
):
    """Detailed profit metrics + streaks."""
    print(f"\n  {'PROFIT ANALYSIS':=^66}")
    header = (
        f"  {'MODEL':<25} | {'RETURN':<10} | {'P.FACTOR':<10} | {'AVG WIN':<10} | {'AVG LOSS':<10}"
    )
    if show_buy_hold:
        header += f" | {'B&H':<10}"
    print(header)
    print(f"  {'-' * (70 + (13 if show_buy_hold else 0))}")

    for r in all_results:
        avg_w = f"{r.avg_win:+.4%}" if r.win_trades > 0 else "n/a"
        avg_l = f"{r.avg_loss:+.4%}" if r.loss_trades > 0 else "n/a"
        line = (
            f"  {r.model_name:<25} | {r.total_return:<+10.4%} | "
            f"{pf_str(r.profit_factor):<10} | {avg_w:<10} | {avg_l:<10}"
        )
        if show_buy_hold:
            line += f" | {r.buy_hold_return:<+10.4%}"
        print(line)

    # Streaks
    print(f"\n  {'STREAKS':=^66}")
    print(
        f"  {'MODEL':<25} | {'MAX WIN':<10} | {'MAX LOSS':<10} | {'AVG WIN':<10} | {'AVG LOSS':<10}"
    )
    print(f"  {'-' * 70}")
    for r in all_results:
        print(
            f"  {r.model_name:<25} | {r.longest_win_streak:<10} | "
            f"{r.longest_loss_streak:<10} | {r.avg_win_streak:<10.1f} | "
            f"{r.avg_loss_streak:<10.1f}"
        )

    # Risk metrics
    print(f"\n  {'RISK METRICS':=^66}")
    print(f"  {'MODEL':<25} | {'MAX DD':<10} | {'SHARPE':<10} | {'SORTINO':<10} | {'W/L':<8}")
    print(f"  {'-' * 70}")
    for r in all_results:
        sortino_s = f"{r.sortino_ratio:.2f}" if r.sortino_ratio < 100 else "∞"
        print(
            f"  {r.model_name:<25} | {r.max_drawdown:<+10.4%} | "
            f"{r.sharpe_ratio:<10.2f} | {sortino_s:<10} | "
            f"{r.win_trades}/{r.loss_trades}"
        )

    if all_results:
        best = max(all_results, key=lambda r: r.total_return)
        worst = min(all_results, key=lambda r: r.total_return)
        print(f"\n  Most profitable:      {best.model_name} ({best.total_return:+.4%})")
        print(f"  Least profitable:     {worst.model_name} ({worst.total_return:+.4%})")
        bp = max(all_results, key=lambda r: r.profit_factor)
        print(f"  Best profit factor:   {bp.model_name} ({pf_str(bp.profit_factor)})")
        best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio)
        print(f"  Best Sharpe ratio:    {best_sharpe.model_name} ({best_sharpe.sharpe_ratio:.2f})")
        least_dd = max(all_results, key=lambda r: r.max_drawdown)  # closest to 0
        print(f"  Smallest max DD:      {least_dd.model_name} ({least_dd.max_drawdown:+.4%})")
        bws = max(all_results, key=lambda r: r.longest_win_streak)
        wls = max(all_results, key=lambda r: r.longest_loss_streak)
        print(f"  Longest win streak:   {bws.model_name} ({bws.longest_win_streak} days)")
        print(f"  Longest loss streak:  {wls.model_name} ({wls.longest_loss_streak} days)")
        if show_buy_hold and all_results[0].buy_hold_return != 0:
            bh = all_results[0].buy_hold_return
            bh_dd = all_results[0].buy_hold_max_drawdown
            beat = sum(1 for r in all_results if r.total_return > bh)
            print(f"\n  Buy & Hold return:    {bh:+.4%}  (max DD: {bh_dd:+.4%})")
            print(f"  Models beating B&H:   {beat}/{len(all_results)}")

        if benchmarks:
            print(f"\n  {'BENCHMARK COMPARISON':=^66}")
            best_return = max(r.total_return for r in all_results)
            for bench, ret in benchmarks.items():
                beat = sum(1 for r in all_results if r.total_return > ret)
                marker = "✓" if best_return > ret else "✗"
                print(
                    f"  {bench:<10} return: {ret:+.4%}  "
                    f"| Models beating: {beat}/{len(all_results)} {marker}"
                )

        if any(r.stop_loss_pct > 0 for r in all_results):
            total_stopped = sum(r.stopped_out_count for r in all_results)
            total_trades = sum(r.test_days for r in all_results)
            print(
                f"\n  Stop-loss triggers:   {total_stopped}/{total_trades} trades "
                f"({total_stopped / total_trades:.0%})"
            )

    # Yearly performance (only shown if data spans multiple years)
    has_yearly = any(r.yearly_performance for r in all_results)
    if has_yearly:
        print_yearly_performance(all_results)


def print_yearly_performance(all_results):
    """Show performance breakdown by calendar year for the best model."""
    # Pick the model with most yearly data
    best = max(all_results, key=lambda r: len(r.yearly_performance))
    if not best.yearly_performance:
        return

    print(f"\n  {'YEARLY PERFORMANCE':=^66}")
    print(f"  Model: {best.model_name}\n")
    print(
        f"  {'YEAR':<8} | {'TRADES':<8} | {'ACC.':<8} | {'RETURN':<10} | {'PF':<8} | {'MAX DD':<10}"
    )
    print(f"  {'-' * 60}")

    for yp in best.yearly_performance:
        pf_s = f"{yp.profit_factor:.2f}" if yp.profit_factor < 100 else "∞"
        print(
            f"  {yp.year:<8} | {yp.trades:<8} | {yp.accuracy:<8.0%} | "
            f"{yp.total_return:<+10.4%} | {pf_s:<8} | {yp.max_drawdown:<+10.4%}"
        )

    # Also show top 3 models' yearly summary if multiple have yearly data
    models_with_yearly = [r for r in all_results if r.yearly_performance]
    if len(models_with_yearly) > 1:
        years = sorted(set(yp.year for r in models_with_yearly for yp in r.yearly_performance))
        if len(years) >= 2:
            print(f"\n  {'YEARLY RETURN BY MODEL':=^66}")
            header = f"  {'MODEL':<25} |"
            for y in years:
                header += f" {y:^8} |"
            print(header)
            print(f"  {'-' * (28 + len(years) * 11)}")

            for r in models_with_yearly[:8]:  # limit to top 8 for readability
                row = f"  {r.model_name:<25} |"
                yp_map = {yp.year: yp for yp in r.yearly_performance}
                for y in years:
                    if y in yp_map:
                        row += f" {yp_map[y].total_return:>+7.2%} |"
                    else:
                        row += f" {'n/a':^8} |"
                print(row)


def print_next_day_forecast(all_results):
    """What would each model predict for the most recent holdout day?"""
    if not all_results or not all_results[0].days:
        return
    print(f"\n  {'NEXT-DAY SIGNAL (most recent holdout day)':=^66}")
    last_day = all_results[0].days[-1]
    print(f"  Date: {last_day.date}")
    print(f"  Price: {last_day.close_before:.2f} → {last_day.close_actual:.2f}\n")
    up_votes = down_votes = 0
    for r in all_results:
        if not r.days:
            continue
        d = r.days[-1]
        mark = "✓" if d.correct else "✗"
        print(f"  {r.model_name:<25}  {d.predicted:<6} (conf: {d.confidence:.1%})  {mark}")
        if d.predicted == "UP":
            up_votes += 1
        else:
            down_votes += 1
    total = up_votes + down_votes
    if total > 0:
        signal = "UP" if up_votes > down_votes else "DOWN"
        strength = max(up_votes, down_votes) / total
        print(f"\n  Consensus: {signal} ({strength:.0%} of models agree)")
