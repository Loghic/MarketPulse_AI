# Reading the results

After a backtest, each row is one model on one ticker over one period. Here's
what every column means and which direction is "good".

## Accuracy

The share of days the model got the direction right. **50% is a coin flip.**
Above 50% is necessary but not sufficient — fees and the size of the moves on
right-vs-wrong days decide whether accuracy actually makes money.

## Return

Total simulated profit/loss over the test window, **after fees**. Green is a
gain, red a loss. Compare it to the **B&H** column (buy-and-hold) — beating B&H
is the bar, not just being positive.

## Buy-and-hold (B&H) and "Beat?"

The return you'd get doing nothing but holding. The **Beat?** column is ✓ when
the strategy's return exceeds B&H, ✗ otherwise. In a bull market, ✗ is common
and expected.

## Profit factor (PF)

Gross profit ÷ gross loss. Above 1 means winning days outweigh losing days; "∞"
means there were no losing days in the window. Higher is better.

## Sharpe & Sortino

Risk-adjusted return — how much return you got per unit of "bumpiness".

- **Sharpe** divides average return by its overall volatility.
- **Sortino** is the same but only counts *downside* volatility, so it doesn't
  penalise a strategy for big *up* days.

Higher is better; both are annualised. They let you compare a calm small gain
against a wild large one fairly.

## Max drawdown (Max DD / DD)

The worst peak-to-trough drop in the equity curve over the window — your worst
losing streak, as a percentage. It's always ≤ 0; **closer to 0 is better**. A
high return with a brutal drawdown may be untradeable in practice.

## Win / Loss (W/L)

Count of winning vs losing days.

## Coverage

The percentage of days you actually traded; the rest were sat out. Days get
sat out either by the [confidence gate](strategy#confidence-gate-min-confidence)
(confidence below θ) or because the model itself chose not to trade — the
News-Informed baseline, for instance, sits out whenever the news isn't clear.
Low coverage with high accuracy can be noise — a great score on 3 traded days
means little.

## Turnover

Only shown with [turnover fees](strategy#turnover-fees) or
[hold days](strategy#hold-days). It's the number of times the position changed
(and therefore paid a fee). Lower turnover = lower fee drag.

## Sentiment column

The average news sentiment the model saw across the window (only non-zero for
"+ News" variants). Positive = mostly upbeat headlines, negative = mostly grim.

## Calibration

Calibration asks: **when a model says 70%, is it actually right ~70% of the
time?** If yes, confidence is meaningful and the
[confidence gate](strategy#confidence-gate-min-confidence) can help. Two
scores summarise it (lower is better for both):

- **Brier score** — average squared error between confidence and outcome.
  0.25 is the "always say 50%" baseline.
- **ECE (Expected Calibration Error)** — average gap between stated confidence
  and actual accuracy.

If these don't improve with confidence, the gate only shrinks your trading; it
can't find an edge.
