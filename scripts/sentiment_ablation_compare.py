"""One-off: compare a --sentiment run against the baseline (no-news) run.

Throwaway research script (mirrors paper_aggregate.py's style), not part of
the package. For each sentiment-bearing model in the new run, matches the
same ticker/date rows against its no-suffix counterpart in the baseline run
and reports U2 (both runs) + a DM test on the paired errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from engine.forecast_significance import dm_test
from engine.regression_metrics import theil_u2

# sentiment-variant model name -> its baseline (no-news) counterpart
_PAIRS = {
    "XGBoost + sentiment": "XGBoost",
    "Prophet + sentiment": "Prophet",
    "XGBoost + macro + sentiment": "XGBoost + macro",
    "Prophet + macro + sentiment": "Prophet + macro",
}


def load_run(run_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(run_dir.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        frames.append(pd.read_csv(f, parse_dates=["date"]))
    return pd.concat(frames, ignore_index=True)


def compare(baseline: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sent_model, base_model in _PAIRS.items():
        sent_sub = sentiment[sentiment.model == sent_model]
        base_sub = baseline[baseline.model == base_model]
        if sent_sub.empty or base_sub.empty:
            continue
        for ticker, g in sent_sub.groupby("ticker"):
            b = base_sub[base_sub.ticker == ticker]
            m = g.merge(b[["date", "y_true", "y_pred"]], on="date", suffixes=("_sent", "_base"))
            if m.empty:
                continue
            u2_sent = theil_u2(m.y_true_sent, m.y_pred_sent, m.y_naive)
            u2_base = theil_u2(m.y_true_base, m.y_pred_base, m.y_naive)
            e_sent = m.y_true_sent - m.y_pred_sent
            e_base = m.y_true_base - m.y_pred_base
            dm = dm_test(e_sent.to_numpy(), e_base.to_numpy())
            rows.append(
                dict(
                    model=base_model,
                    ticker=ticker,
                    n=len(m),
                    u2_no_news=u2_base,
                    u2_with_news=u2_sent,
                    delta_u2=u2_base - u2_sent,
                    dm_stat=dm.stat,
                    dm_p=dm.p_value,
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("sentiment_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    baseline = load_run(args.baseline_dir)
    sentiment = load_run(args.sentiment_dir)
    out = compare(baseline, sentiment)
    out_path = args.out or (args.sentiment_dir / "_sentiment_ablation.csv")
    out.to_csv(out_path, index=False)
    print(f"{len(out)} rows -> {out_path}")
    if not out.empty:
        print(out.groupby("model")[["u2_no_news", "u2_with_news", "delta_u2"]].median())
        sig = out[out.dm_p < 0.05]
        print(f"DM-significant (p<0.05) rows: {len(sig)}/{len(out)}")


if __name__ == "__main__":
    main()
