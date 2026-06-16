"""
news_sources.py – Pluggable news providers (Yahoo Finance, GDELT, ...).

Each provider implements `fetch(ticker, lookback_days) → list[NewsItem]` and
returns headlines tagged with their actual publication date (not "today").
This is what allows look-ahead-safe backtesting downstream.

Currently implemented:

* ``YahooNewsProvider`` – Yahoo Finance via yfinance. Returns ~10 most-recent
  headlines, mostly with real publication timestamps when yfinance provides
  them (recent yfinance versions include `pubDate`/`providerPublishTime`).
  Very limited history (days, not weeks).

* ``GDELTNewsProvider`` – GDELT 2.0 Doc API. Free, no key needed,
  ~24 h after publish before it appears in the index, but provides
  multi-year historical coverage. Query is a free-text search built from
  the ticker symbol and the company name.

All providers return ``NewsItem`` objects, normalized to UTC date strings.
Failures (network, rate limit) return an empty list rather than raising —
the caller decides whether missing news is a problem.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

# Ticker → GDELT search query lives in config (the single asset registry), so
# adding a ticker there automatically gives it a sensible news query here.
from config import TICKER_NAMES

# Imported at module level so tests can patch ``engine.news_sources.yf``.
try:
    import yfinance as yf  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - yfinance is a required dep
    yf = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Data class
# ----------------------------------------------------------------------


@dataclass
class NewsItem:
    """A single news article with its real publication date."""

    ticker: str
    published_at: str  # ISO date "YYYY-MM-DD" (UTC)
    headline: str
    source: str = "unknown"  # "yahoo", "gdelt", ...
    url: str = ""

    def __post_init__(self) -> None:
        # Normalize headline whitespace
        self.headline = " ".join(self.headline.split())


# ----------------------------------------------------------------------
# Provider protocol
# ----------------------------------------------------------------------


class NewsProvider(Protocol):
    name: str

    def fetch(self, ticker: str, lookback_days: int = 7) -> list[NewsItem]: ...


# ----------------------------------------------------------------------
# Yahoo
# ----------------------------------------------------------------------


class YahooNewsProvider:
    """
    Yahoo Finance via yfinance. Limited history (hours-days) but no rate
    limits in practice and no key required.

    Returns headlines tagged with their reported publication time when
    yfinance provides one; otherwise the current UTC date is used (this
    is the historical bug that prevented look-ahead-safe backtests).
    """

    name = "yahoo"

    def fetch(self, ticker: str, lookback_days: int = 7) -> list[NewsItem]:
        if yf is None:
            return []
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception:
            return []

        items: list[NewsItem] = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        for entry in raw:
            content = entry.get("content") if isinstance(entry, dict) else None
            if isinstance(content, dict):
                title = content.get("title") or ""
                pub = content.get("pubDate") or content.get("displayTime") or ""
                link = (
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else ""
                )
            else:
                title = entry.get("title") or entry.get("headline") or ""
                pub = entry.get("providerPublishTime") or entry.get("pubDate") or ""
                link = entry.get("link", "")

            if not title:
                continue

            published_dt = _parse_pub(pub) or datetime.now(UTC)
            if published_dt < cutoff:
                continue

            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published_dt.strftime("%Y-%m-%d"),
                    headline=title,
                    source="yahoo",
                    url=link or "",
                )
            )

        return items


# ----------------------------------------------------------------------
# GDELT
# ----------------------------------------------------------------------


class GDELTNewsProvider:
    """
    GDELT 2.0 Doc API.

    Free, no auth, covers global news in 100+ languages. The Doc API returns
    individual articles matching a free-text query and a date range. We use
    the English-language filter and the ticker/company name as the query.

    GDELT indexes articles ~15-60 minutes after publication, so 'today' may
    have partial coverage. Historical depth: from 2017 onwards.

    Rate limits (informal): a few QPS per IP. We cap maxrecords at 250
    (the API's documented hard ceiling).

    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    """

    name = "gdelt"
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    MAX_RECORDS = 250
    TIMEOUT = 10

    def __init__(self, language: str = "english", max_records: int = 100) -> None:
        self.language = language
        self.max_records = min(max_records, self.MAX_RECORDS)

    def fetch(self, ticker: str, lookback_days: int = 30) -> list[NewsItem]:
        query = self._build_query(ticker)
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=lookback_days)

        params = {
            "query": f"{query} sourcelang:{self.language}",
            "mode": "ArtList",
            "format": "JSON",
            "maxrecords": str(self.max_records),
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
            "sort": "DateDesc",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=self.TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body) if body.strip().startswith("{") else {"articles": []}
        except Exception:
            return []

        items: list[NewsItem] = []
        for art in data.get("articles", []):
            title = art.get("title") or ""
            seendate = art.get("seendate") or ""  # "YYYYMMDDTHHMMSSZ"
            link = art.get("url") or ""
            if not title:
                continue

            pub = _parse_gdelt_date(seendate) or datetime.now(UTC)
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=pub.strftime("%Y-%m-%d"),
                    headline=title,
                    source="gdelt",
                    url=link,
                )
            )
        return items

    @staticmethod
    def _build_query(ticker: str) -> str:
        """Build a GDELT free-text query for a ticker."""
        name = TICKER_NAMES.get(ticker.upper(), "")
        if not name:
            # Drop the "-USD" suffix for crypto tickers (BTC-USD → BTC)
            symbol = ticker.replace("-USD", "")
            return f'"{symbol}"'
        # Quoted phrase ensures GDELT matches the literal company name
        return f'"{name}"'


# ----------------------------------------------------------------------
# MultiProvider
# ----------------------------------------------------------------------


@dataclass
class MultiProvider:
    """Combine multiple providers; dedupe by (date, headline)."""

    providers: list[NewsProvider] = field(default_factory=list)
    name: str = "multi"

    def fetch(self, ticker: str, lookback_days: int = 30) -> list[NewsItem]:
        seen: set[tuple[str, str]] = set()
        out: list[NewsItem] = []
        for p in self.providers:
            try:
                for item in p.fetch(ticker, lookback_days=lookback_days):
                    key = (item.published_at, item.headline.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
            except Exception:
                continue
        return out


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


_PROVIDER_REGISTRY: dict[str, type] = {
    "yahoo": YahooNewsProvider,
    "gdelt": GDELTNewsProvider,
}


def get_provider(name: str | list[str] = "yahoo") -> NewsProvider:
    """
    Return a provider (or MultiProvider) for the requested source(s).

    Examples::

        get_provider("yahoo")              # single provider
        get_provider(["yahoo", "gdelt"])   # combined, dedup
    """
    if isinstance(name, str):
        cls = _PROVIDER_REGISTRY.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown news source '{name}'. Known: {list(_PROVIDER_REGISTRY)}")
        return cls()

    providers = [get_provider(n) for n in name]
    return MultiProvider(providers=providers)


# ----------------------------------------------------------------------
# Date parsing helpers
# ----------------------------------------------------------------------


def _parse_pub(value: object) -> datetime | None:
    """Parse a yfinance publication-date field. Returns UTC-aware datetime or None."""
    if not value:
        return None
    # Unix epoch (int / numeric string)
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(int(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    # ISO-ish "2025-04-12T15:30:00Z"
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_gdelt_date(value: str) -> datetime | None:
    """Parse GDELT's compact 'YYYYMMDDTHHMMSSZ' format."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        # Some endpoints return just "YYYYMMDD"
        try:
            return datetime.strptime(value[:8], "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            return None
