# Models & baselines

Every model looks at a ticker's past prices (and optionally recent news) and
outputs a direction for tomorrow — **UP** or **DOWN** — plus a **confidence**
between 50% and 100%. Confidence is the model's stated probability *of the
direction it picked*.

## Model families

- **k-NN (k-Nearest Neighbours)** — finds days in history that looked most like
  today and assumes tomorrow rhymes with what followed them. "Enhanced" adds
  technical indicators (RSI, MACD, volume, volatility); "(TW)" = time-weighted,
  so recent history counts more.
- **LinReg (Linear Regression)** — fits a straight-line relationship between the
  recent features and the next move. Same Enhanced / TW variants as k-NN.
- **LSTM** — a small neural network that reads the recent sequence of days. It
  must be trained first (Training tab) and needs the optional ML dependency.
- **Prophet** — a classical time-series forecaster (trend + seasonality). It
  predicts a price level, which we turn into a direction.
- **Chronos-2** — a pre-trained "foundation" forecasting model; no training
  needed, it forecasts the next values zero-shot.
- **Kronos** — a forecasting model that reads full candlesticks (open/high/low/
  close/volume), not just the closing price.

Forecasting models (Prophet, Chronos-2, Kronos) only appear when their optional
software is installed; if you don't see one, that's why.

## Time-weighting and "Enhanced"

- **Enhanced** = the model also uses technical indicators, not just past
  returns.
- **(TW) / Time-Weighted** = recent days are weighted more heavily than old
  ones, so the model adapts faster to changing conditions.

## News / sentiment

Optionally, models can read recent **news headlines** scored as positive or
negative ("sentiment") and nudge their prediction. You choose the scorer:

- **VADER** — fast, general-purpose, rule-based.
- **FinBERT** — a finance-tuned AI model; more accurate on financial text but
  slower and a larger download.
- **Naive** — a simple keyword baseline.

A "+ News" variant is the same model with sentiment switched on. News is
look-ahead-safe in backtests — only headlines published *before* each day are
used, so the model never peeks at the future.

## Baselines

Baselines are deliberately dumb "predictors". They exist so a real model has to
clear a **real bar**, not just beat a coin. If a sophisticated model can't beat
*Always-Long*, it isn't adding anything.

- **Always-Long** — predict UP every single day (blind optimism; in a rising
  market this is surprisingly hard to beat).
- **Always-Short** — predict DOWN every day, the mirror of Always-Long. A
  control: if Always-Long only looks good because the market rose, Always-Short
  makes that explicit (it should look terrible in the same bull market).
- **Previous-Day** — predict that tomorrow repeats today's realised direction.
- **5-Day / 20-Day Momentum** — predict UP if the price is higher than it was
  5 (or 20) days ago.
- **Random** — a seeded coin flip. The floor, and the reference for
  significance tests.

### News-aware baselines

These still aren't *models* — they're fixed rules — but they're allowed to
**react** to today's news sentiment (the same look-ahead-safe score the "+ News"
models use). They never *learn* from past news outcomes; that would make them
models, not baselines. News overrides the price rule only when the sentiment is
clearly strong (above a small threshold); weak news is ignored.

- **News Previous-Day** — assume tomorrow repeats today's direction, *unless*
  the news is clearly positive/negative, in which case follow the news. ("Things
  keep going the way they were, unless the headlines say otherwise.")
- **News-Informed** — when the news is clearly positive/negative, predict that;
  otherwise fall back to previous-day. (The "person who only acts on a clear
  headline, and otherwise expects more of the same.")
- **News 5-Day Momentum** — 5-day momentum, but flipped to match strong news.

The rule of thumb: a model is only worth attention if it beats **Previous-Day
and Always-Long** (and now the news-aware baselines too), not merely 50% or
buy-and-hold.
