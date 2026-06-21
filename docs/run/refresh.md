# `refresh.py` — populate the DB

This is the script you use when you want to **pre-load historical
news** (for the article-style backtests) without running any models.

## Daily refresh (same as before — no flags needed)

```bash
uv run python refresh.py              # all tickers, Yahoo, VADER
uv run python refresh.py --stocks
uv run python refresh.py --crypto
uv run python refresh.py --tickers AAPL NVDA BTC-USD
```

This pulls prices for each ticker, fetches a handful of recent
headlines from Yahoo, scores them with VADER, and stores each row with
its real `published_at` date.

## Historical bulk fetch (new in 2026)

The shape is always:

```bash
uv run python refresh.py <SCOPE> \
    --news-source <SOURCE...> \
    --news-history-days <N> \
    --sentiment-method <SCORER> \
    --force-news
```

`<SCOPE>` is one of `--stocks`, `--crypto`, `--all`, or
`--tickers AAPL NVDA …`.

### Single ticker — 6 months of GDELT, scored with FinBERT

```bash
uv run python refresh.py --tickers AAPL \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --force-news
```

### All stocks — 1 year of GDELT, FinBERT

```bash
uv run python refresh.py --stocks \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

### All crypto — 1 year of GDELT, FinBERT

```bash
uv run python refresh.py --crypto \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

Crypto tickers (`BTC-USD`, `ETH-USD`, `SOL-USD`) are handled the same
way as stocks. The `-USD` suffix is automatically stripped when
building the GDELT search query, so `BTC-USD` becomes `"Bitcoin"`,
`ETH-USD` → `"Ethereum"`, `SOL-USD` → `"Solana"` (the GDELT query map
lives in `config.py`'s asset registry as `TICKER_NAMES`, re-exported
by `engine/news_sources`).

### Stocks + crypto in one shot

```bash
# All 19 tickers (every asset class), Yahoo + GDELT combined and deduped
uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

### Explicit ticker list (mix of stocks and crypto)

```bash
uv run python refresh.py --tickers AAPL NVDA TSLA BTC-USD ETH-USD \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --force-news
```

### Score the same news a second time with a different scorer

VADER and FinBERT rows can live side-by-side in the DB (distinguished
by the `method` column). Re-run with `--sentiment-method vader` and
`--force-news` to add a second set of scores, which lets you A/B both
scorers on the same historical headlines:

```bash
uv run python refresh.py --all \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method vader --force-news
```

* `--news-history-days 365` tells the provider to pull a year of
  headlines. Yahoo silently caps at ~3-7 days; GDELT honours larger
  values up to 250 articles per call.
* `--force-news` bypasses the same-day cache, which would otherwise
  short-circuit a re-fetch.

## Verify what landed

```bash
uv run python -c "
from engine.db_manager import DatabaseManager
db = DatabaseManager()
df = db.get_news('AAPL')
print(df[['published_at','source','method','sentiment_score','headline']].head(10))
print(f'Total rows for AAPL: {len(df)}')
print(f'Date range: {df[\"published_at\"].min()} → {df[\"published_at\"].max()}')
"
```
