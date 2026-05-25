"""
news_impact.py – Compare news vs no-news model variants from a run_all.py result tree.

The walk-forward backtester emits, per (ticker × period), a price-only variant
and a matching "+ News" variant for each model family that supports the
news-aware path. This script pairs them and quantifies the news effect.

Outputs three CSVs per run directory:

  1. ``_news_vs_no_news_{TICKER}.csv`` – one row per (model_family, period)
     pair with base / news / delta columns for accuracy, return, profit
     factor, max drawdown, Sharpe and Sortino, plus a boolean
     ``return_news_wins`` for quick filtering.

  2. ``_news_vs_no_news_summary.csv`` – one row per (ticker, model_family)
     aggregated across the periods that have both sides, including
     win-counts and median deltas.

  3. ``_news_vs_no_news_overall.csv`` – a single overall row plus a few
     "leaderboard" lines for the paper/poster.

Usage::

    uv run python scripts/news_impact.py results/stocks_50d_fee003_bh
    uv run python scripts/news_impact.py results/stocks_50d_fee003_bh \\
                                          results/crypto_50d_fee015_bh

Each positional argument is a run directory produced by ``run_all.py``.
Pass several to process them all in one invocation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

# Map "+ News" variant name → its price-only baseline. Both rows must exist
# in the same per-ticker CSV (same period) to form a comparable pair.
PAIRS: dict[str, str] = {
    "k-NN TW + News": "k-NN Time-Weighted",
    "k-NN Enh. TW + News": "k-NN Enh. TW",
    "LinReg TW + News": "LinReg Time-Weighted",
    "LinReg Enh. TW + News": "LinReg Enh. TW",
    "LSTM + News": "LSTM",
}

# Metrics to diff. Each tuple is (column_name, higher_is_better).
METRICS: list[tuple[str, bool]] = [
    ("accuracy", True),
    ("total_return", True),
    ("profit_factor", True),
    ("max_drawdown", True),  # closer to 0 is better → larger value is better
    ("sharpe_ratio", True),
    ("sortino_ratio", True),
]


# ----------------------------------------------------------------------
# Pure helpers (no I/O — easy to unit-test)
# ----------------------------------------------------------------------


def safe_float(value) -> float | None:
    """Parse a numeric cell; return None for missing / empty / non-numeric."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pair_rows(rows: list[dict]) -> list[dict]:
    """
    Given the raw CSV rows for one ticker, return one comparison row per
    (period, model_family) where both the base and the "+ News" variant
    are present.
    """
    by_period: dict[str, dict[str, dict]] = {}
    for r in rows:
        period = r.get("period", "")
        model = r.get("model", "")
        if not period or not model:
            continue
        by_period.setdefault(period, {})[model] = r

    out: list[dict] = []
    for period, models in by_period.items():
        for news_name, base_name in PAIRS.items():
            news_row = models.get(news_name)
            base_row = models.get(base_name)
            if news_row is None or base_row is None:
                continue

            comparison: dict = {
                "ticker": news_row.get("ticker") or base_row.get("ticker"),
                "period": period,
                "model_family": base_name,
            }
            # Keep typed local refs for the three headline metrics so the
            # type checker can see they're ``float | None`` when we feed
            # them into ``_bool_win`` (the comparison dict is a wide union
            # of str | float | None | bool, so dict-lookup loses precision).
            base_vals: dict[str, float | None] = {}
            news_vals: dict[str, float | None] = {}
            for col, higher_better in METRICS:
                b = safe_float(base_row.get(col))
                n = safe_float(news_row.get(col))
                base_vals[col] = b
                news_vals[col] = n
                comparison[f"{col}_base"] = b
                comparison[f"{col}_news"] = n
                if b is None or n is None:
                    comparison[f"{col}_delta"] = None
                else:
                    comparison[f"{col}_delta"] = n - b if higher_better else b - n

            # Quick boolean: did news help the headline metrics?
            comparison["accuracy_news_wins"] = _bool_win(
                base_vals["accuracy"], news_vals["accuracy"]
            )
            comparison["return_news_wins"] = _bool_win(
                base_vals["total_return"], news_vals["total_return"]
            )
            comparison["sharpe_news_wins"] = _bool_win(
                base_vals["sharpe_ratio"], news_vals["sharpe_ratio"]
            )
            # B&H is the same for both rows; keep it for context.
            bh = safe_float(base_row.get("buy_hold_return"))
            comparison["buy_hold_return"] = bh
            out.append(comparison)
    return out


def _bool_win(base: float | None, news: float | None) -> bool | None:
    """True if news strictly beats base; False if base beats news; None if tied or missing."""
    if base is None or news is None:
        return None
    if news > base:
        return True
    if base > news:
        return False
    return None


def summarize_per_ticker_model(comparisons: list[dict]) -> list[dict]:
    """
    Aggregate across periods → one row per (ticker, model_family) with
    median deltas and win-counts.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for c in comparisons:
        key = (c["ticker"], c["model_family"])
        grouped.setdefault(key, []).append(c)

    summary: list[dict] = []
    for (ticker, model), rows in grouped.items():
        deltas_return = [
            r["total_return_delta"] for r in rows if r["total_return_delta"] is not None
        ]
        deltas_acc = [r["accuracy_delta"] for r in rows if r["accuracy_delta"] is not None]
        deltas_sharpe = [
            r["sharpe_ratio_delta"] for r in rows if r["sharpe_ratio_delta"] is not None
        ]

        wins_return = sum(1 for r in rows if r["return_news_wins"] is True)
        wins_acc = sum(1 for r in rows if r["accuracy_news_wins"] is True)
        wins_sharpe = sum(1 for r in rows if r["sharpe_news_wins"] is True)

        # Best and worst period for news, by return delta
        ranked = [r for r in rows if r["total_return_delta"] is not None]
        ranked.sort(key=lambda r: r["total_return_delta"], reverse=True)
        best_period = ranked[0]["period"] if ranked else None
        worst_period = ranked[-1]["period"] if ranked else None

        summary.append(
            {
                "ticker": ticker,
                "model_family": model,
                "periods_compared": len(rows),
                "median_return_delta": _median(deltas_return),
                "mean_return_delta": _mean(deltas_return),
                "median_accuracy_delta": _median(deltas_acc),
                "mean_accuracy_delta": _mean(deltas_acc),
                "median_sharpe_delta": _median(deltas_sharpe),
                "news_wins_return": wins_return,
                "news_wins_accuracy": wins_acc,
                "news_wins_sharpe": wins_sharpe,
                "best_period_for_news": best_period,
                "worst_period_for_news": worst_period,
            }
        )

    summary.sort(key=lambda r: (r["ticker"], r["model_family"]))
    return summary


def overall_stats(comparisons: list[dict]) -> dict:
    """
    Headline numbers across every (ticker, model_family, period) pair —
    what you'd quote in the abstract.
    """
    n = len(comparisons)
    if n == 0:
        return {"pairs": 0}

    def _frac_win(key: str) -> tuple[int, int]:
        wins = sum(1 for c in comparisons if c[key] is True)
        defined = sum(1 for c in comparisons if c[key] is not None)
        return wins, defined

    rw, rn = _frac_win("return_news_wins")
    aw, an = _frac_win("accuracy_news_wins")
    sw, sn = _frac_win("sharpe_news_wins")

    deltas_return = [
        c["total_return_delta"] for c in comparisons if c["total_return_delta"] is not None
    ]
    deltas_acc = [c["accuracy_delta"] for c in comparisons if c["accuracy_delta"] is not None]

    return {
        "pairs": n,
        "return_news_wins": rw,
        "return_pairs_defined": rn,
        "return_news_win_rate": (rw / rn) if rn else None,
        "accuracy_news_wins": aw,
        "accuracy_pairs_defined": an,
        "accuracy_news_win_rate": (aw / an) if an else None,
        "sharpe_news_wins": sw,
        "sharpe_pairs_defined": sn,
        "sharpe_news_win_rate": (sw / sn) if sn else None,
        "median_return_delta": _median(deltas_return),
        "mean_return_delta": _mean(deltas_return),
        "median_accuracy_delta": _median(deltas_acc),
        "mean_accuracy_delta": _mean(deltas_acc),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def read_ticker_csv(path: Path) -> list[dict]:
    """Read one per-ticker CSV produced by run_all.py into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def discover_ticker_csvs(run_dir: Path) -> list[Path]:
    """All *.csv in run_dir except files starting with '_'."""
    return sorted(p for p in run_dir.glob("*.csv") if not p.name.startswith("_"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: dict[str, None] = {}
    for r in rows:
        for k in r:
            fieldnames[k] = None
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Run-directory driver
# ----------------------------------------------------------------------


def process_run_dir(run_dir: Path) -> dict:
    """
    Walk the per-ticker CSVs in ``run_dir`` and emit the three derived
    files alongside them. Returns a dict with the overall stats so the
    caller can print a short digest.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory")

    ticker_csvs = discover_ticker_csvs(run_dir)
    if not ticker_csvs:
        raise FileNotFoundError(f"No per-ticker CSVs found under {run_dir}")

    all_comparisons: list[dict] = []
    per_ticker_counts: list[tuple[str, int]] = []

    for csv_path in ticker_csvs:
        ticker = csv_path.stem
        rows = read_ticker_csv(csv_path)
        comps = pair_rows(rows)
        if not comps:
            per_ticker_counts.append((ticker, 0))
            continue

        out_path = run_dir / f"_news_vs_no_news_{ticker}.csv"
        write_csv(out_path, comps)

        all_comparisons.extend(comps)
        per_ticker_counts.append((ticker, len(comps)))

    summary = summarize_per_ticker_model(all_comparisons)
    write_csv(run_dir / "_news_vs_no_news_summary.csv", summary)

    overall = overall_stats(all_comparisons)
    # Write a single-row overall CSV too — handy for joining across runs.
    if overall.get("pairs"):
        write_csv(run_dir / "_news_vs_no_news_overall.csv", [overall])

    # Leaderboards — best and worst comparisons by return delta.
    ranked = [c for c in all_comparisons if c["total_return_delta"] is not None]
    ranked.sort(key=lambda c: c["total_return_delta"], reverse=True)
    top = ranked[:5]
    bottom = ranked[-5:][::-1]

    return {
        "run_dir": run_dir,
        "per_ticker_counts": per_ticker_counts,
        "overall": overall,
        "top": top,
        "bottom": bottom,
    }


# ----------------------------------------------------------------------
# Console output
# ----------------------------------------------------------------------


def print_digest(report: dict, stream=sys.stdout) -> None:
    o = report["overall"]
    rd = report["run_dir"]

    def pct(rate):
        return "n/a" if rate is None else f"{rate:.0%}"

    def fmt(v, fmt_spec=":+.4%"):
        return "n/a" if v is None else format(v, fmt_spec.lstrip(":"))

    print(f"\n{'=' * 78}", file=stream)
    print(f" News-vs-no-news report — {rd}", file=stream)
    print(f"{'=' * 78}", file=stream)

    print(f"\n  Pairs compared:  {o.get('pairs', 0)}", file=stream)

    if not o.get("pairs"):
        print("  No (base, +News) pairs found in this directory.", file=stream)
        print(
            "  (Pre-populate news with `refresh.py --news-source gdelt ...` before backtesting.)",
            file=stream,
        )
        return

    print(
        "  News beats no-news on:",
        file=stream,
    )
    print(
        f"      Accuracy → "
        f"{o['accuracy_news_wins']}/{o['accuracy_pairs_defined']}  "
        f"({pct(o['accuracy_news_win_rate'])})",
        file=stream,
    )
    print(
        f"      Return   → "
        f"{o['return_news_wins']}/{o['return_pairs_defined']}  "
        f"({pct(o['return_news_win_rate'])})",
        file=stream,
    )
    print(
        f"      Sharpe   → "
        f"{o['sharpe_news_wins']}/{o['sharpe_pairs_defined']}  "
        f"({pct(o['sharpe_news_win_rate'])})",
        file=stream,
    )

    print(f"\n  Median return delta:    {fmt(o['median_return_delta'])}", file=stream)
    print(f"  Mean return delta:      {fmt(o['mean_return_delta'])}", file=stream)
    print(
        f"  Median accuracy delta:  {fmt(o['median_accuracy_delta'], ':+.4f')}",
        file=stream,
    )

    print("\n  Top 5 (ticker, model, period) where news helped most:", file=stream)
    for c in report["top"]:
        print(
            f"      {c['ticker']:<10} {c['model_family']:<22} {c['period']:<6}  "
            f"Δreturn={c['total_return_delta']:+.4%}  "
            f"Δacc={(c['accuracy_delta'] or 0.0):+.4f}",
            file=stream,
        )
    print("\n  Bottom 5 (news hurt most):", file=stream)
    for c in report["bottom"]:
        print(
            f"      {c['ticker']:<10} {c['model_family']:<22} {c['period']:<6}  "
            f"Δreturn={c['total_return_delta']:+.4%}  "
            f"Δacc={(c['accuracy_delta'] or 0.0):+.4f}",
            file=stream,
        )

    print("\n  Wrote:", file=stream)
    for t, n in report["per_ticker_counts"]:
        if n > 0:
            print(f"    {rd}/_news_vs_no_news_{t}.csv  ({n} comparisons)", file=stream)
    print(f"    {rd}/_news_vs_no_news_summary.csv", file=stream)
    print(f"    {rd}/_news_vs_no_news_overall.csv", file=stream)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare news vs no-news model variants from a run_all.py result tree.",
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="One or more directories produced by run_all.py (e.g. results/stocks_50d_fee003_bh).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the console digest; only write CSVs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    for run_dir in args.run_dirs:
        try:
            report = process_run_dir(run_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            print_digest(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
