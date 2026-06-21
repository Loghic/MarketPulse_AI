# Getting started

Welcome. This app predicts whether a stock or crypto will go **UP or DOWN**
the next day, and lets you test how well those predictions would have done if
you had actually traded on them. You don't need any trading or machine-learning
background — this Help section explains every term you'll see.

## What the app does

There are three things you'll mostly use:

- **Predict** — for a ticker you pick, every model says UP or DOWN for the next
  day, with a confidence. A *consensus* combines them.
- **Backtest** — "if I had followed this model every day for the last N days,
  how would I have done?" You get accuracy, profit, risk, and a comparison to
  just buying and holding.
- **Out-of-Sample (OOS)** — the *honest* version of backtesting. It picks the
  best setup on one stretch of history and then scores it on a **different**
  stretch it never saw, so good-looking results can't just be luck. See
  [Out-of-sample testing](oos#out-of-sample-testing).

## The most important caveat

Predicting the market is genuinely hard. Across a lot of testing, these models
land close to a **coin flip (~50% accuracy)**, and a daily buy/sell strategy
usually trails simply buying and holding once trading fees are counted. That's
not a bug — it's the finding. The tools here (especially OOS, calibration, and
significance tests) exist to tell you *honestly* whether something works, rather
than to flatter the numbers. Treat results as research, not investment advice.

## How to read a result, in one line

A model is only interesting if it beats the **naive baselines**
([see Baselines](models#baselines)) and **buy-and-hold**
([see Buy-and-hold](strategy#buy-and-hold)) — not just if its accuracy is above
50%. The rest of this Help section explains each piece.

## Where to go next

- [Models & baselines](models) — what each predictor is.
- [Strategy & fees](strategy) — stop-loss, fees, the confidence gate, holding.
- [Backtest metrics](metrics) — how to read the results table.
- [Out-of-sample & honesty](oos) — the trustworthy read.
