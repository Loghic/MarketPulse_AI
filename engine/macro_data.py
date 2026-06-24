"""
macro_data.py – Macro / exogenous features (VIX, DXY, Gold, SP500, 1Y rate).

The forecasting paper (plan R4) wants exogenous macro inputs alongside the
ticker's own history. This module fetches them, transforms them sensibly
(equity-like series → **log-returns**; the Treasury rate stays a level), caches
them to SQLite, and — the correctness-critical part — aligns them onto a
ticker's trading calendar with a **strict 1-day lag** so only information
available at ``t`` is ever used to predict ``t+1`` (the R0.2 leakage rule).

Two clearly separated pieces:

* ``align_macro(...)`` — **pure** pandas, no network. Reindex → forward-fill →
  lag. This is the part that must be right, and it's the part the tests pin.
* ``fetch_macro(...)`` — I/O. yfinance for VIX/DXY/Gold/SP500, the public FRED
  CSV (no API key) for the 1-Year Treasury (DGS1). Every series is fetched
  defensively: a failure drops that one column and logs, never raising.

Sources:
  VIX     ^VIX        (CBOE volatility index)         → log-return
  DXY     DX-Y.NYB    (US dollar index; UUP fallback)  → log-return
  Gold    GC=F        (gold futures; GLD fallback)     → log-return
  SP500   ^GSPC       (S&P 500 index)                  → log-return
  DGS1    FRED:DGS1   (1-Year Treasury, % yield)       → level (forward-filled)
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.logger import get_logger

log = get_logger(__name__)

# (column name, primary yfinance symbol, fallback symbol or None)
_YF_MACRO: list[tuple[str, str, str | None]] = [
    ("vix", "^VIX", None),
    ("dxy", "DX-Y.NYB", "UUP"),
    ("gold", "GC=F", "GLD"),
    ("sp500", "^GSPC", "SPY"),
]

# FRED series fetched as a level (yield in percent), forward-filled.
_FRED_SERIES = {"dgs1": "DGS1"}

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


# ----------------------------------------------------------------------
# R4.2 — alignment (pure, leakage-safe)
# ----------------------------------------------------------------------


def align_macro(
    ticker_dates: pd.Index | pd.Series | list[str],
    macro_df: pd.DataFrame,
    *,
    lag: int = 1,
) -> pd.DataFrame:
    """Align a macro DataFrame onto a ticker's trading calendar, lagged.

    Steps (in order — order matters for the no-lookahead guarantee):
      1. **Reindex** the macro series onto ``ticker_dates`` (union first so we
         can forward-fill across the macro calendar, then restrict).
      2. **Forward-fill** — macro series have their own holidays/missing days
         (FRED is business-day with gaps); the most-recent known value carries
         forward.
      3. **Lag by ``lag`` days** — shift forward so the value attached to date
         ``t`` is the one known as of ``t − lag``. With ``lag=1`` (default), the
         feature row for predicting ``t+1`` uses macro information available at
         ``t``, never ``t+1``.

    Args:
        ticker_dates: the ticker's dates (str ``YYYY-MM-DD`` or datetime).
        macro_df:     date-indexed macro columns (from ``fetch_macro``).
        lag:          days to lag (>= 0). 0 = same-day (use only if you've
                      already lagged upstream).

    Returns a DataFrame indexed by ``ticker_dates`` (as strings), one column per
    macro series, with the leading ``lag`` rows NaN (no prior value yet).
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    dates = pd.to_datetime(pd.Index(ticker_dates))
    if macro_df is None or macro_df.empty:
        return pd.DataFrame(index=dates.strftime("%Y-%m-%d"))

    m = macro_df.copy()
    m.index = pd.to_datetime(m.index)
    m = m.sort_index()

    # 1) Reindex onto the union so forward-fill can bridge ticker dates that the
    #    macro calendar is missing, then 2) forward-fill.
    union = m.index.union(dates).sort_values()
    m = m.reindex(union).ffill()

    # 3) Lag, THEN restrict to the ticker's dates. Lagging on the union index
    #    (which includes every ticker date) guarantees the shift is by calendar
    #    position on the aligned grid.
    if lag > 0:
        m = m.shift(lag)
    out = m.reindex(dates)
    out.index = dates.strftime("%Y-%m-%d")
    return out


# ----------------------------------------------------------------------
# R4.1 — fetch (I/O)
# ----------------------------------------------------------------------


def _normalize_index(s: pd.Series) -> pd.Series:
    """Coerce a series' index to tz-naive, day-resolution datetimes.

    yfinance returns tz-aware (often UTC) DatetimeIndexes while FRED parses
    tz-naive; mixing the two in one DataFrame raises "Cannot join tz-naive with
    tz-aware". Stripping the tz and flooring to the date makes every macro series
    share one comparable daily index.
    """
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx.normalize()
    return s


def _to_log_returns(close: pd.Series) -> pd.Series:
    """Log-returns of a positive price series (NaN-safe, first row dropped)."""
    c = pd.to_numeric(close, errors="coerce")
    c = c[c > 0]
    return np.log(c).diff()


def _fetch_yf_logret(symbol: str) -> pd.Series | None:
    """Daily log-returns for a yfinance symbol, date-indexed, or None on failure."""
    try:
        from engine.data_downloader import get_historical_data

        df = get_historical_data(symbol, period="max")
        if df is None or df.empty or "close" not in df.columns:
            return None
        s = _to_log_returns(df["close"])
        return _normalize_index(s).dropna()
    except Exception as e:  # noqa: BLE001 — a bad macro series must not crash a run
        log.warning("macro: yfinance fetch failed for %s (%s).", symbol, e)
        return None


def _fetch_fred_level(series_id: str) -> pd.Series | None:
    """Fetch a FRED series as a level from the public CSV endpoint (no API key)."""
    try:
        import urllib.request

        url = _FRED_CSV_URL.format(series=series_id)
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed FRED host
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        # FRED CSV: first col is the date, second is the series (named by id).
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # "." → NaN
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col]))
        return _normalize_index(s).dropna()
    except Exception as e:  # noqa: BLE001
        log.warning("macro: FRED fetch failed for %s (%s).", series_id, e)
        return None


def fetch_macro() -> pd.DataFrame:
    """Fetch all macro series into one tidy date-indexed DataFrame.

    Equity-like series are log-returns; the FRED rate is a level. Any series
    that fails to download is simply omitted (logged), so the result always
    contains whatever succeeded — callers should not assume every column exists.
    """
    cols: dict[str, pd.Series] = {}

    for name, primary, fallback in _YF_MACRO:
        s = _fetch_yf_logret(primary)
        if (s is None or s.empty) and fallback is not None:
            log.info("macro: %s primary %s empty; trying fallback %s.", name, primary, fallback)
            s = _fetch_yf_logret(fallback)
        if s is not None and not s.empty:
            cols[name] = s

    for name, fred_id in _FRED_SERIES.items():
        s = _fetch_fred_level(fred_id)
        if s is not None and not s.empty:
            cols[name] = s

    if not cols:
        log.warning("macro: no series could be fetched.")
        return pd.DataFrame()

    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    return out


# ----------------------------------------------------------------------
# SQLite cache
# ----------------------------------------------------------------------


@dataclass
class MacroCache:
    """Tiny SQLite cache for the macro panel (one row per date, columns = series)."""

    db_path: str = "data/market_data.db"
    table: str = "macro_series"

    def save(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        import sqlite3

        long = df.reset_index().melt(id_vars="date", var_name="series", value_name="value")
        long = long.dropna(subset=["value"])
        long["date"] = long["date"].astype(str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "date TEXT, series TEXT, value REAL, PRIMARY KEY (date, series))"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {self.table} (date, series, value) VALUES (?, ?, ?)",
                long[["date", "series", "value"]].itertuples(index=False, name=None),
            )
            conn.commit()

    def load(self) -> pd.DataFrame:
        import sqlite3

        try:
            with sqlite3.connect(self.db_path) as conn:
                long = pd.read_sql(f"SELECT date, series, value FROM {self.table}", conn)
        except Exception:  # noqa: BLE001 — no cache yet
            return pd.DataFrame()
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="series", values="value").sort_index()
        wide.columns.name = None
        return wide
