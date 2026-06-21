# `main.py` — predictions

```bash
# Default: a curated subset of tickers, all models, current prices
uv run python main.py

# Pick scope
uv run python main.py --stocks
uv run python main.py --crypto
uv run python main.py --tickers AAPL NVDA BTC-USD

# Pick scorer / news source
uv run python main.py --stocks --sentiment-method finbert
```

Prints a per-ticker table with each model's UP/DOWN call, confidence,
and the sentiment-adjusted version. No DB writes other than the
news/price caches.

For the common flags (`--no-refresh`, `--sentiment-method`,
`--news-source`, etc.) see the [common flags table in
README.md](README.md#common-flags-across-scripts).

When you want a deeper analysis than today's call alone, jump to
[backtest.md](backtest.md). For the next-day prediction served over
HTTP / the Web GUI, see [web-gui.md](web-gui.md).
