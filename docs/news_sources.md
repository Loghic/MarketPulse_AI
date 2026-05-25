# News Sources

MarketPulse AI pulls headlines from pluggable providers in
`engine/news_sources.py`. Each provider implements:

```python
class MyProvider:
    name: str
    def fetch(self, ticker: str, lookback_days: int) -> list[NewsItem]: ...
```

`NewsItem` has `ticker`, `published_at` (ISO date, UTC), `headline`,
`source`, and `url`. Crucially, `published_at` reflects the *real*
publication date — this is what makes downstream backtests look-ahead-safe.

## Built-in providers

### Yahoo Finance (`"yahoo"`)

* Same source the project has always used: `yf.Ticker(ticker).news`.
* No API key. No rate limits in practice.
* **History:** very shallow (hours to a few days). Don't expect to find
  last month's news here.
* yfinance ≥ 1.3 includes `pubDate` on most articles, so headlines are
  tagged with their actual publication date when available.

### GDELT 2.0 Doc API (`"gdelt"`)

* The [GDELT Project](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
  indexes global news in 100+ languages.
* **Free, no auth required.**
* **History:** from 2017 onward.
* **Latency:** ~15-60 minutes after publication.
* **Cap:** 250 articles per request (the API's documented hard ceiling);
  default 100.
* Query is built from the ticker symbol and a curated `TICKER_NAMES`
  lookup. For "AAPL" the query becomes `"Apple"`; for "BTC-USD" it
  becomes `"Bitcoin"`. If a ticker isn't in the lookup, the bare symbol
  is used (less precise — extend `TICKER_NAMES` in `news_sources.py`).

```python
from engine.news_sources import GDELTNewsProvider
items = GDELTNewsProvider().fetch("AAPL", lookback_days=30)
for it in items:
    print(it.published_at, it.headline)
```

### MultiProvider (combine sources)

`get_provider(["yahoo", "gdelt"])` returns a `MultiProvider` that calls
each provider in order and deduplicates by `(published_at, headline.lower())`.
This is the recommended setting for building a historical corpus — Yahoo
for the very latest, GDELT for everything older.

```python
from engine.news_sources import get_provider
provider = get_provider(["yahoo", "gdelt"])
items = provider.fetch("AAPL", lookback_days=60)
```

## Why historical news matters

The original Yahoo-only pipeline stored every headline under "today's
date", regardless of when the article was actually published. That made
the cache useless for backtests: every prediction day saw the same
"current" sentiment, which leaks information from after the prediction
into the past (look-ahead bias).

With `published_at` populated by either provider, the look-ahead-safe
DB query `db.get_news_before(ticker, asof_date)` does the right thing:
sentiment for day N comes only from news with `published_at < N`.

## Configuration

`config.py` exposes:

```python
DEFAULT_NEWS_SOURCES      = ["yahoo"]   # ["yahoo", "gdelt"] for combined
DEFAULT_NEWS_LOOKBACK_DAYS = 7          # 0 = unbounded
DEFAULT_NEWS_HALF_LIFE_DAYS = 3.0       # 0 = no decay
DEFAULT_SENTIMENT_METHOD  = "vader"     # "vader" | "finbert" | "naive"
```

CLI overrides (on `backtest.py`):

```
--sentiment-method {vader,finbert,naive}
--news-lookback   <int days>     # 0 = unbounded
--news-half-life  <float days>   # 0 = no decay
```

## Adding a new provider

1. Implement a class in `engine/news_sources.py` with `name` and `fetch`.
2. Register it in `_PROVIDER_REGISTRY`.
3. Return `NewsItem` objects with an honest `published_at`.

Adapter sketch for any "news + date" REST API:

```python
class MyProvider:
    name = "mysource"

    def fetch(self, ticker, lookback_days=7):
        items = []
        for entry in _call_api(ticker, lookback_days):
            items.append(NewsItem(
                ticker=ticker,
                published_at=entry["published_at"][:10],
                headline=entry["title"],
                source="mysource",
                url=entry.get("url", ""),
            ))
        return items
```

Failures should return `[]`, not raise — the upstream caller treats
"no news" as "neutral sentiment" rather than crashing.
