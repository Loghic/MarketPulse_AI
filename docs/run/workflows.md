# Suggested workflows

## Daily (just want a prediction)

```bash
uv run python main.py --stocks
```

That auto-refreshes data and prints today's prediction table. Done.

## Daily (want to backtest yesterday)

```bash
uv run python backtest.py --stocks --days 1 --buy-hold
```

## Researcher mode (deep historical analysis for the article)

```bash
# ----------------------------------------------------------------------
# 1. One-off: populate DB with 1 year of GDELT for EVERYTHING
#    (stocks + crypto). Score the same headlines with both FinBERT and
#    VADER so the comparison can be done offline afterwards.
# ----------------------------------------------------------------------
uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news

uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method vader --force-news

# ----------------------------------------------------------------------
# 2. Headline runs — stocks and crypto each get their own CSV.
#    `--no-refresh` keeps everything offline (the DB now has all the news).
# ----------------------------------------------------------------------
uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/stocks_finbert_20d.csv

uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/crypto_finbert_20d.csv

uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
    --sentiment-method vader --output results/stocks_vader_20d.csv

uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method vader --output results/crypto_vader_20d.csv

# ----------------------------------------------------------------------
# 3. Sensitivity sweep — same FinBERT scorer, different lookback windows.
#    One CSV per setting so you can chart "accuracy vs window length".
# ----------------------------------------------------------------------
for L in 3 7 30; do
    uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
        --sentiment-method finbert --news-lookback $L \
        --output results/stocks_finbert_l${L}.csv
    uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
        --sentiment-method finbert --news-lookback $L \
        --output results/crypto_finbert_l${L}.csv
done

# ----------------------------------------------------------------------
# 4. Single-ticker spotlight (useful when you want a clean focused chart
#    of one asset for the poster — e.g. the asset with the biggest news lift)
# ----------------------------------------------------------------------
uv run python backtest.py --tickers AAPL --days 30 --full --buy-hold --no-refresh \
    --sentiment-method finbert --output results/aapl_spotlight.csv

uv run python backtest.py --tickers BTC-USD --days 30 --full --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/btc_spotlight.csv

# ----------------------------------------------------------------------
# 5. Out-of-sample honesty check (see research.md for details)
# ----------------------------------------------------------------------
uv run python scripts/oos_harness.py --stocks --days 50 --fees 0.03 \
    --buy-hold --no-refresh --sentiment-method finbert
uv run python scripts/oos_harness.py --crypto --days 50 --fees 0.15 \
    --buy-hold --no-refresh --sentiment-method finbert

# ----------------------------------------------------------------------
# 6. Measurement rigor — is confidence calibrated, and is any reported
#    edge distinguishable from chance? (see research.md)
# ----------------------------------------------------------------------
# Sweep the confidence gate + print Brier/ECE per model.
uv run python backtest.py --stocks --days 100 --no-refresh \
    --sentiment-method finbert --confidence-sweep

# Significance: binomial p + Wilson CI + bootstrap CI + BH-FDR.
uv run python backtest.py --stocks --days 100 --no-refresh \
    --sentiment-method finbert --significance --buy-hold
```

Each CSV is self-describing — the `ticker`, `period`, `model`,
`accuracy`, `total_return`, `profit_factor`, `max_drawdown`,
`sharpe_ratio`, `buy_hold_return` and `bench_*` columns are enough to
slice the results later in pandas or a spreadsheet.
