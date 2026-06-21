# Strategy, fees & risk knobs

These settings (mostly on the Backtest and OOS tabs) control how a prediction is
turned into a simulated trade and what it costs. They don't change the model's
accuracy — they change the **money** outcome and the **risk**.

## Trading fees

Every simulated trade pays a fee. **Fee %** is charged *per side* — you pay it
when you buy and again when you sell, so a round-trip costs twice the number you
enter. Real-world fees are roughly 0.05–0.20% per side for stocks/crypto. Over a
long daily-trading backtest these fees pile up and are often the single biggest
drag on returns.

## Turnover fees

By default the backtest assumes you close and re-open a position **every day**,
paying the round-trip fee daily. That's unrealistic if the signal says the same
thing two days running. Turn on **Turnover fees** to charge the fee **only on
days the position actually changes** (a new direction or a flip). Same-direction
days are held for free. This is the realistic "trade only when the signal
changes" cost, and it usually improves returns by cutting fee churn.

## Hold days

**Hold days = N** means once you open a position you hold it for N trading days
before re-reading the signal, ignoring day-to-day flip-flops in between. The
model still makes a prediction every day (so accuracy is unchanged), but your
*position* — and therefore your profit/loss — follows the held trade. Most
useful together with turnover fees, which then skips fees on the held days.

## Position mode

By default the backtest is **daily mark-to-market**: it books each day's
close-to-close move as its own little trade. That's fine for measuring
direction accuracy, but it isn't how you'd actually trade — you wouldn't sell
and re-buy every single day.

**Position mode** models real holding instead. While the signal keeps saying
the same direction, you hold **one** position; you only cash out when the
signal flips (or you get stopped out, or the test ends). That held run's profit
is the **compounded** price change from where you entered to where you exited,
and you pay the round-trip fee **once** for the whole run — not once per day.

Example: you go long at \$100, it rises to \$110, you keep holding, then it
falls to \$90 and the signal flips. Position mode books that as a single trade:
\$100 → \$90 = −10%, minus one round-trip fee. (Daily mode would instead book
+10% on day one and −18% on day two, with a fee each day.)

Use it together with [turnover fees](strategy#turnover-fees) thinking in mind —
position mode already charges fees per held run, so it's the most realistic
"hold while the signal agrees" cost. It changes the return, drawdown, and fee
numbers; accuracy (a per-day, per-prediction measure) is unaffected.

## Stop-loss

A **stop-loss** automatically closes a losing position once it falls a set
percentage against you intraday, capping that day's loss. **SL %** is that
threshold (e.g. 5 = exit if the trade is down 5% during the day). `0` disables
it. It's a *risk* control, not an edge: it can reduce big losing days but won't
create profit that isn't there. It matters most on volatile names.

## Stop-loss sweep

Instead of one stop-loss level, **SL sweep** runs the backtest at several levels
at once (by default 0, 5, 10, 15%). `0` is the no-stop baseline. You then
compare how each level changed returns and risk side by side — handy because the
best stop-loss is rarely obvious up front.

## Confidence gate (min confidence)

Models output a confidence with each call. The **confidence gate** sits out any
day whose confidence is below your threshold θ — you simply don't trade those
days (no profit, no loss, no fee). The idea: maybe a model is only worth
following when it's *sure*.

Whether this helps depends on **calibration** — see
[Calibration](metrics#calibration). If high-confidence days really are more
accurate, gating isolates the good trades; if confidence is meaningless, gating
just trades less. The result table shows **Coverage** (the % of days you still
traded) so you can see the trade-off.

## Buy-and-hold

**Buy-and-hold (B&H)** is the do-nothing benchmark: buy on day one, hold to the
end, no trading. Beating B&H *after fees* is the real test — in a rising market,
constantly trading in and out usually loses to simply holding. Every backtest
shows the B&H return and a ✓/✗ for whether the strategy beat it.
