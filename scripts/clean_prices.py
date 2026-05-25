"""
clean_prices.py – One-off cleanup for bad price rows in the SQLite DB.

If a backtest ever showed a 100×+ single-day return (best_day > 0.5), the
underlying ``stock_prices`` table probably has at least one row with
``close <= 0`` or NULL — most often a holiday / weekend gap that yfinance
filled in with zero, or a very-recent partial bar that hadn't settled
yet. Subsequent fetches will overwrite the bad row with INSERT OR REPLACE,
but until then the backtester sees a price jump of 10,000:1 and produces
the absurd metrics you saw in the CSV.

This script:

  1. Reports how many rows per ticker have ``close <= 0`` or NULL.
  2. Optionally deletes them (``--apply``).
  3. Optionally flags any remaining adjacent-day moves larger than
     50% so you can spot one-off data spikes that survived (rare, but
     they exist for sub-penny tickers).

Usage::

    uv run python scripts/clean_prices.py                # report only
    uv run python scripts/clean_prices.py --apply        # delete bad rows
    uv run python scripts/clean_prices.py --threshold 1  # also flag >100% moves

After cleaning, re-download:

    uv run python refresh.py --all
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("data") / "market_data.db"


def report(conn: sqlite3.Connection) -> dict[str, int]:
    """Count bad rows per ticker."""
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, COUNT(*) FROM stock_prices "
        "WHERE close IS NULL OR close <= 0 "
        "GROUP BY ticker ORDER BY 2 DESC"
    )
    return dict(cur.fetchall())


def delete(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM stock_prices WHERE close IS NULL OR close <= 0")
    conn.commit()
    return cur.rowcount


def flag_jumps(
    conn: sqlite3.Connection, threshold: float
) -> list[tuple[str, str, float, float, float]]:
    """
    Return adjacent-day pairs with |move| > threshold, per ticker.
    This catches rows that aren't NULL/zero but are still implausible
    (e.g. a single-day BTC value of $5 from a yfinance glitch).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, date, close FROM stock_prices "
        "WHERE close IS NOT NULL AND close > 0 ORDER BY ticker, date"
    )
    rows = cur.fetchall()
    out = []
    prev_ticker = None
    prev_close = None
    prev_date = None
    for ticker, date, close in rows:
        if ticker != prev_ticker:
            prev_ticker = ticker
            prev_close = close
            prev_date = date
            continue
        if prev_close and prev_close > 0:
            move = abs(close - prev_close) / prev_close
            if move > threshold:
                out.append((ticker, f"{prev_date} → {date}", prev_close, close, move))
        prev_close = close
        prev_date = date
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually delete bad rows (default: report only)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Also flag adjacent-day moves bigger than THRESHOLD as suspicious (default 0.5 = 50%%)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to SQLite DB (default: {DB_PATH})",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found at {args.db}", file=sys.stderr)
        return 1

    with sqlite3.connect(args.db) as conn:
        bad = report(conn)
        total = sum(bad.values())
        print(f"Bad price rows (close NULL or ≤ 0): {total}")
        for ticker, n in bad.items():
            print(f"  {ticker:<10}  {n}")

        if args.apply and total > 0:
            removed = delete(conn)
            print(f"\nDeleted {removed} bad rows.")
        elif total > 0:
            print("\nRe-run with --apply to delete them.")

        if args.threshold > 0:
            jumps = flag_jumps(conn, args.threshold)
            if jumps:
                print(
                    f"\nSuspicious adjacent-day moves > {args.threshold:.0%} "
                    f"(may indicate stale or glitched rows):"
                )
                for ticker, dates, p0, p1, move in jumps[:30]:
                    print(f"  {ticker:<10}  {dates}  {p0:.4f} → {p1:.4f}  ({move:+.1%})")
                if len(jumps) > 30:
                    print(f"  … {len(jumps) - 30} more")
            else:
                print(f"\nNo suspicious moves > {args.threshold:.0%} found. ✓")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
