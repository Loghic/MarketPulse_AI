"""
db_manager.py – SQLite správce pro cenová data a news sentiment.
"""

import sqlite3
import pandas as pd
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_name: str = "market_data.db"):
        self.db_path = Path("data") / db_name
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

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
                    PRIMARY KEY (ticker, date, headline)
                )
            """)
            conn.commit()

    def save_prices(self, ticker: str, df: pd.DataFrame, asset_type: str):
        """Uloží DataFrame do DB (upsert přes INSERT OR REPLACE)."""
        if df.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            df = df.copy().reset_index()

            # Normalizace sloupců
            df.columns = [c.lower() for c in df.columns]

            # yfinance používá 'date' nebo 'datetime' jako index name
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)

            # Ošetření timezone → naive string
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

            df["ticker"] = ticker
            df["asset_type"] = asset_type

            valid_columns = ["ticker", "asset_type", "date", "open", "high", "low", "close", "volume"]
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

    def save_news(self, ticker: str, news_df: pd.DataFrame):
        """Uloží zprávy (INSERT OR IGNORE pro deduplikaci)."""
        if news_df.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            news_df = news_df.copy()

            # Sjednocení column name: API může posílat 'title' i 'headline'
            if "title" in news_df.columns:
                news_df.rename(columns={"title": "headline"}, inplace=True)

            news_df["ticker"] = ticker
            cols = ["ticker", "date", "headline", "sentiment_score"]
            news_df = news_df[[c for c in cols if c in news_df.columns]]

            news_df.to_sql("_temp_news", conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR IGNORE INTO news_sentiment (ticker, date, headline, sentiment_score)
                SELECT ticker, date, headline, sentiment_score FROM _temp_news
            """)
            conn.execute("DROP TABLE IF EXISTS _temp_news")
            conn.commit()

    def get_news(self, ticker: str, date: str = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            if date:
                query = "SELECT * FROM news_sentiment WHERE ticker = ? AND date = ?"
                return pd.read_sql_query(query, conn, params=[ticker, date])
            else:
                query = "SELECT * FROM news_sentiment WHERE ticker = ? ORDER BY date DESC"
                return pd.read_sql_query(query, conn, params=[ticker])
