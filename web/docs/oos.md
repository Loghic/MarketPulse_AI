# Out-of-sample & honesty

This is the most important section for trusting any result.

## The trap: picking the best of many

A normal backtest lets you try many models, periods, and settings and keep the
best-looking one. But across hundreds of combinations, **some will look great by
pure luck** — like flipping enough coins until one lands heads ten times. If you
report that winner, you're fooling yourself. This is called *selection
inflation*.

## Out-of-sample testing

The **OOS (Out-of-Sample) harness** fixes this. For each ticker it:

1. Takes a **selection window** of history and picks the best model+period
   there.
2. Then scores that winner on the **next, completely separate window** it never
   got to choose on.

Because the scoring data is disjoint from the selection data, a lucky winner has
nowhere to hide — it has to perform on fresh data. The OOS result is the
*honest* number.

## Selection-inflation gap

The headline diagnostic. It's the median of *(in-sample return − out-of-sample
return)* across tickers. A **large positive gap** means the in-sample winners
were mostly overfit luck that vanished on fresh data. A small gap means the
edge (if any) was real. This single number tells you how much to distrust a
normal "best model" leaderboard.

## OOS beat-B&H rate

Of the tickers tested, the fraction whose OOS winner actually beat buy-and-hold
on the fresh window. This is the trustworthy version of "how often does this
work?" Historically it's been low (and it *decays* over longer horizons as fees
compound) — which is the honest finding.

## Significance: is it different from a coin flip?

Even an out-of-sample number can be noise on a short window. Significance tests
put error bars on it:

- **Binomial p-value** — the probability of seeing this accuracy (or better) if
  the model were truly a 50/50 coin. Small p (e.g. < 0.05) = unlikely to be
  chance.
- **Wilson confidence interval** — an honest range for the true accuracy. If the
  interval **includes 50%**, you can't claim an edge.
- **Bootstrap interval on return** — a range for the true return. If it
  **includes 0**, the profit is indistinguishable from break-even.

A result is only worth believing when the accuracy interval excludes 50%, the
return interval excludes 0, **and** the p-value survives correction for having
tested many models at once.

## Comparing runs (OOS Compare)

Run the harness once per setting — e.g. with and without the confidence gate, or
with turnover fees on vs off — then use the **OOS Compare** tab to put two runs
side by side. It shows which setting won on each headline metric and on each
ticker's out-of-sample return, so you can see whether a change actually helped
*honestly*, not just in-sample.

## Bottom line

If a setup shows no out-of-sample edge, "tuning" it further just fits noise.
OOS is the gate to pass before taking any result seriously.
