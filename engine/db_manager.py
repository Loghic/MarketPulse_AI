"""
db_manager.py – SQLite správce pro cenová data a news sentiment.

News table schema (current):
    news_sentiment(
        ticker          TEXT,
        date            TEXT,   -- "bucket" date (legacy; today's date for old rows)
        headline        TEXT,
        sentiment_score REAL,
        published_at    TEXT,   -- ISO date of actual publication (added 2026)
        source          TEXT,   -- "yahoo", "gdelt", ...
        method          TEXT,   -- scoring method ("vader", "finbert", "naive")
        PRIMARY KEY (ticker, date, headline)
    )

The ``published_at`` column is what enables look-ahead-safe backtests:
sentiment for a given prediction date is computed only from rows whose
``published_at`` strictly precedes that date.

Old rows pre-dating the migration have ``published_at`` NULL — queries
fall back to the ``date`` column via COALESCE.
"""

import sqlite3
from pathlib import Path

import pandas as pd


class DatabaseManager:
    def __init__(self, db_name: str = "market_data.db"):
        self.db_path = Path("data") / db_name
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
        self._migrate_news_columns()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    ticker TEXT,
                    asset_type TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    ticker TEXT,
                    date TEXT,
                    headline TEXT,
                    sentiment_score REAL,
                    published_at TEXT,
                    source TEXT,
                    method TEXT,
                    PRIMARY KEY (ticker, date, headline)
                )
            """)
            conn.commit()

    def _migrate_news_columns(self):
        """Add new columns to existing DBs created before the migration."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(news_sentiment)")
            existing = {row[1] for row in cur.fetchall()}
            for col in ("published_at", "source", "method"):
                if col not in existing:
                    cur.execute(f"ALTER TABLE news_sentiment ADD COLUMN {col} TEXT")
            conn.commit()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def save_prices(self, ticker: str, df: pd.DataFrame, asset_type: str):
        """Uloží DataFrame do DB (upsert přes INSERT OR REPLACE).

        Defensively drops rows with missing or non-positive ``close`` —
        these otherwise produce nonsensical day-over-day ratios in the
        backtester (a single zero-price row yields a 10,000×+ trade PnL).
        """
        if df.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            df = df.copy().reset_index()

            # Drop garbage rows up front so they never make it into the DB.
            # yfinance occasionally returns a zero/NaN close for very-recent
            # rows or holidays; both are useless for the backtester.
            close_col_candidates = ("close", "Close")
            for cname in close_col_candidates:
                if cname in df.columns:
                    before = len(df)
                    df = df[df[cname].notna() & (df[cname] > 0)]
                    dropped = before - len(df)
                    if dropped:
                        print(
                            f"  DB: dropped {dropped} {ticker} rows with missing/non-positive close"
                        )
                    break
            if df.empty:
                return

            # Normalizace sloupců
            df.columns = [c.lower() for c in df.columns]

            # yfinance používá 'date' nebo 'datetime' jako index name
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)

            # Ošetření timezone → naive string
            if "date" in df.columns:
                df["date"] = (
                    pd.to_datetime(df["date"], utc=True)
                    .dt.tz_localize(None)
                    .dt.strftime("%Y-%m-%d")
                )

            df["ticker"] = ticker
            df["asset_type"] = asset_type

            valid_columns = [
                "ticker",
                "asset_type",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            # Ponecháme jen sloupce, které existují
            existing = [c for c in valid_columns if c in df.columns]
            df = df[existing]

            df.to_sql("_temp_prices", conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR REPLACE INTO stock_prices
                SELECT * FROM _temp_prices
            """)
            conn.execute("DROP TABLE IF EXISTS _temp_prices")
            conn.commit()
            print(f"  DB: Synced {len(df)} rows for {ticker} ({asset_type}).")

    def get_prices(self, ticker: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM stock_prices WHERE ticker = ? ORDER BY date"
            return pd.read_sql_query(query, conn, params=[ticker])

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def save_news(self, ticker: str, news_df: pd.DataFrame):
        """
        Insert news rows with INSERT OR IGNORE for deduplication.

        Accepts dataframes containing any subset of:
            ticker, date, headline, sentiment_score, published_at, source, method

        Missing columns are filled with sensible defaults so old call sites
        keep working without modification.
        """
        if news_df.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            news_df = news_df.copy()

            # Sjednocení column name: API může posílat 'title' i 'headline'
            if "title" in news_df.columns:
                news_df.rename(columns={"title": "headline"}, inplace=True)

            news_df["ticker"] = ticker
            for col, default in [
                ("date", None),
                ("sentiment_score", 0.0),
                ("published_at", None),
                ("source", "unknown"),
                ("method", None),
            ]:
                if col not in news_df.columns:
                    news_df[col] = default

            # If date is missing but published_at is present, use it as the bucket
            news_df["date"] = news_df["date"].fillna(news_df["published_at"])

            cols = [
                "ticker",
                "date",
                "headline",
                "sentiment_score",
                "published_at",
                "source",
                "method",
            ]
            news_df = news_df[cols]

            news_df.to_sql("_temp_news", conn, if_exists="replace", index=False)
            conn.execute(
                """
                INSERT OR IGNORE INTO news_sentiment
                    (ticker, date, headline, sentiment_score, published_at, source, method)
                SELECT ticker, date, headline, sentiment_score, published_at, source, method
                FROM _temp_news
                """
            )
            conn.execute("DROP TABLE IF EXISTS _temp_news")
            conn.commit()

    def get_news(self, ticker: str, date: str | None = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            if date:
                query = "SELECT * FROM news_sentiment WHERE ticker = ? AND date = ?"
                return pd.read_sql_query(query, conn, params=[ticker, date])
            else:
                query = "SELECT * FROM news_sentiment WHERE ticker = ? ORDER BY date DESC"
                return pd.read_sql_query(query, conn, params=[ticker])

    def get_news_before(
        self,
        ticker: str,
        asof_date: str,
        lookback_days: int | None = None,
        method: str | None = None,
    ) -> pd.DataFrame:
        """
        Return news strictly published BEFORE ``asof_date`` (the prediction
        date) — never on or after it. This is the look-ahead-safe query
        used by the backtester.

        ``COALESCE(published_at, date)`` lets old rows (NULL published_at)
        still be queryable; they just may not have a faithful publication
        timestamp.

        Args:
            ticker: Asset symbol.
            asof_date: "YYYY-MM-DD". News must be strictly older than this.
            lookback_days: Window. If set, news must also be newer than
                ``asof_date - lookback_days``. None = no lower bound.
            method: If set, restrict to rows scored by this method. Useful
                when comparing VADER vs FinBERT in the same DB.

        Returns:
            DataFrame ordered by effective publication date ASC.
        """
        params: list = [ticker, asof_date]
        clauses = [
            "ticker = ?",
            "COALESCE(published_at, date) < ?",
        ]

        if lookback_days is not None and lookback_days > 0:
            cutoff = (pd.to_datetime(asof_date) - pd.Timedelta(days=lookback_days)).strftime(
                "%Y-%m-%d"
            )
            clauses.append("COALESCE(published_at, date) >= ?")
            params.append(cutoff)

        if method is not None:
            clauses.append("(method = ? OR method IS NULL)")
            params.append(method)

        query = (
            "SELECT *, COALESCE(published_at, date) AS effective_date "
            "FROM news_sentiment WHERE " + " AND ".join(clauses) + " ORDER BY effective_date ASC"
        )
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
