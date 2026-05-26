"""
clean_test_contamination.py – Remove synthetic test fixture data that
leaked into the production prices DB.

The bug (root-caused in tests/conftest.py):
    The ``api`` pytest fixture patches yfinance to return a fixed 400-day
    ``_make_prices(seed=42)`` series, then instantiates ``StockAppAPI()``.
    ``DatabaseManager()`` defaults to ``data/market_data.db`` — the **real**
    production DB. Any test that subsequently called
    ``api.get_data("AAPL")``, ``api.refresh_tickers(...)``, etc. on a real
    ticker symbol would persist the synthetic fixture under that real
    ticker name. The contamination spreads through the upsert because
    every test re-write produces identical rows under the same (ticker,
    date) PK.

How we detect it:
    Synthetic fixture rows have a per-day ``close`` value that is *bit-
    identical* across every contaminated ticker because they all come
    from the same numpy seed. The ``TEST`` ticker — which is legitimately
    populated from the same fixture and was never genuine yfinance data —
    is our reference: any (ticker, date) row whose ``close`` matches
    TEST's ``close`` on the same date is contamination.

Usage:
    uv run python scripts/clean_test_contamination.py            # dry-run
    uv run python scripts/clean_test_contamination.py --apply   # delete

After cleanup, repopulate with real yfinance data:
    uv run python refresh.py --all
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "market_data.db"

# Synthetic test fixture tickers — leave these alone.
TEST_FIXTURE_TICKERS = ("TEST", "TEST1", "TEST2", "TINY")


def find_contamination(conn) -> dict[str, int]:
    """Return {ticker: count_of_contaminated_rows} for real tickers."""
    cur = conn.execute(
        f"""
        SELECT a.ticker, COUNT(*)
        FROM stock_prices a
        JOIN stock_prices ref ON a.date = ref.date AND ref.ticker = 'TEST'
        WHERE a.ticker NOT IN ({",".join("?" * len(TEST_FIXTURE_TICKERS))})
          AND a.close = ref.close
        GROUP BY a.ticker
        ORDER BY a.ticker
        """,
        TEST_FIXTURE_TICKERS,
    )
    return {ticker: count for ticker, count in cur.fetchall()}


def find_date_range(conn) -> tuple[str | None, str | None]:
    cur = conn.execute(
        f"""
        SELECT MIN(a.date), MAX(a.date)
        FROM stock_prices a
        JOIN stock_prices ref ON a.date = ref.date AND ref.ticker = 'TEST'
        WHERE a.ticker NOT IN ({",".join("?" * len(TEST_FIXTURE_TICKERS))})
          AND a.close = ref.close
        """,
        TEST_FIXTURE_TICKERS,
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete contaminated rows (default: dry-run).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to the DB (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)
    try:
        contamination = find_contamination(conn)
        if not contamination:
            print("No contamination detected. DB looks clean.")
            return

        first, last = find_date_range(conn)
        total = sum(contamination.values())

        print("Contamination summary")
        print("─" * 50)
        print(f"  Date range:    {first} → {last}")
        print(f"  Tickers hit:   {len(contamination)}")
        print(f"  Total rows:    {total}")
        print()
        print(f"  {'TICKER':<12} ROWS")
        print(f"  {'-' * 22}")
        for ticker, count in sorted(contamination.items()):
            print(f"  {ticker:<12} {count}")
        print()

        if not args.apply:
            print("This was a DRY RUN. Re-run with --apply to delete these rows.")
            print(
                "After deletion, repopulate genuine prices with:\n"
                "    uv run python refresh.py --all"
            )
            return

        # Delete contaminated rows in a single transaction.
        cur = conn.execute(
            f"""
            DELETE FROM stock_prices
            WHERE (ticker, date) IN (
                SELECT a.ticker, a.date
                FROM stock_prices a
                JOIN stock_prices ref ON a.date = ref.date AND ref.ticker = 'TEST'
                WHERE a.ticker NOT IN ({",".join("?" * len(TEST_FIXTURE_TICKERS))})
                  AND a.close = ref.close
            )
            """,
            TEST_FIXTURE_TICKERS,
        )
        conn.commit()
        print(f"✓ Deleted {cur.rowcount} contaminated rows.")
        print("Now run:  uv run python refresh.py --all")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
