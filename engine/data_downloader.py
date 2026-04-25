"""
data_downloader.py – Download historical market data via yfinance.
Compatible with yfinance >= 1.3.0 (MultiIndex columns, new API format).
"""

import yfinance as yf
import pandas as pd


def get_historical_data(ticker_symbol: str = "AAPL", period: str = "1y", debug: bool = False) -> pd.DataFrame:
    """
    Download historical price data for a given ticker from Yahoo Finance.

    Args:
        ticker_symbol: Ticker symbol (e.g. AAPL, BTC-USD).
        period: Time range – 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        debug: Print extended information when True.

    Returns:
        DataFrame with open, high, low, close, volume (empty on failure).
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)

        if df.empty:
            print(f"WARNING: No data returned for {ticker_symbol} (period={period})")
            return pd.DataFrame()

        # yfinance >= 0.2.31 may return MultiIndex columns (ticker, field).
        # Flatten to simple column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only the columns we need (yfinance sometimes adds dividends, stock splits)
        keep = ["open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        if debug:
            print(f"Downloaded {len(df)} rows for {ticker_symbol}")
            print(df.head())

        return df

    except Exception as e:
        print(f"ERROR downloading {ticker_symbol}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    hd = get_historical_data(ticker_symbol="MSFT", period="1mo", debug=True)
    print(hd)
