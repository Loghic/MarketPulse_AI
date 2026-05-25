"""
test_news_impact.py – Tests for scripts/news_impact.py.

We build a tiny synthetic run-directory in tmp_path that mimics the shape
of ``run_all.py`` output (one CSV per ticker, each with rows per
(model, period)), run the impact script on it, and assert the derived
CSVs come out with the right shape and correct deltas / win-counts.
"""

from __future__ import annotations

import csv

# Add the project root to sys.path so we can import scripts.news_impact
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.news_impact import (  # noqa: E402
    PAIRS,
    overall_stats,
    pair_rows,
    process_run_dir,
    safe_float,
    summarize_per_ticker_model,
)

# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


class TestSafeFloat:
    def test_numeric_str(self):
        assert safe_float("0.42") == 0.42

    def test_blank(self):
        assert safe_float("") is None
        assert safe_float(None) is None
        assert safe_float("nan") is None

    def test_int(self):
        assert safe_float(3) == 3.0


class TestPairRows:
    def _row(self, ticker, period, model, **kw):
        base = {
            "ticker": ticker,
            "period": period,
            "model": model,
            "accuracy": "0.5",
            "total_return": "0.0",
            "profit_factor": "1.0",
            "max_drawdown": "0.0",
            "sharpe_ratio": "0.0",
            "sortino_ratio": "0.0",
            "buy_hold_return": "0.0",
        }
        base.update({k: str(v) for k, v in kw.items()})
        return base

    def test_basic_pair(self):
        rows = [
            self._row("AAPL", "1y", "k-NN Enh. TW", accuracy=0.55, total_return=0.05),
            self._row("AAPL", "1y", "k-NN Enh. TW + News", accuracy=0.60, total_return=0.07),
        ]
        out = pair_rows(rows)
        assert len(out) == 1
        c = out[0]
        assert c["ticker"] == "AAPL"
        assert c["period"] == "1y"
        assert c["model_family"] == "k-NN Enh. TW"
        assert abs(c["accuracy_base"] - 0.55) < 1e-9
        assert abs(c["accuracy_news"] - 0.60) < 1e-9
        assert abs(c["accuracy_delta"] - 0.05) < 1e-9
        assert c["accuracy_news_wins"] is True
        assert c["return_news_wins"] is True

    def test_unpaired_models_dropped(self):
        """A model with no '+ News' sibling should produce no comparison row."""
        rows = [self._row("AAPL", "1y", "k-NN Enhanced", accuracy=0.5)]  # no TW + News sibling
        assert pair_rows(rows) == []

    def test_per_period_pairing(self):
        rows = [
            self._row("AAPL", "1y", "k-NN Enh. TW", total_return=0.05),
            self._row("AAPL", "1y", "k-NN Enh. TW + News", total_return=0.06),
            self._row("AAPL", "2y", "k-NN Enh. TW", total_return=0.08),
            self._row("AAPL", "2y", "k-NN Enh. TW + News", total_return=0.04),  # news loses
            # Unpaired (no base for 5y)
            self._row("AAPL", "5y", "k-NN Enh. TW + News", total_return=0.10),
        ]
        out = pair_rows(rows)
        periods = sorted(c["period"] for c in out)
        assert periods == ["1y", "2y"]
        wins = {c["period"]: c["return_news_wins"] for c in out}
        assert wins["1y"] is True
        assert wins["2y"] is False

    def test_max_drawdown_higher_is_better(self):
        """MAX DD closer to 0 is better → delta = base - news (we flip sign internally)."""
        rows = [
            self._row("AAPL", "1y", "k-NN Enh. TW", max_drawdown=-0.10),
            self._row("AAPL", "1y", "k-NN Enh. TW + News", max_drawdown=-0.05),
        ]
        out = pair_rows(rows)
        # News has smaller drawdown (-0.05 vs -0.10) → news is better → delta > 0
        assert out[0]["max_drawdown_delta"] > 0


class TestAggregations:
    def _comp(self, ticker, model, period, return_base, return_news):
        delta = return_news - return_base
        return {
            "ticker": ticker,
            "model_family": model,
            "period": period,
            "accuracy_base": 0.5,
            "accuracy_news": 0.55 if delta > 0 else 0.45,
            "accuracy_delta": 0.05 if delta > 0 else -0.05,
            "accuracy_news_wins": delta > 0,
            "total_return_base": return_base,
            "total_return_news": return_news,
            "total_return_delta": delta,
            "return_news_wins": delta > 0,
            "sharpe_ratio_base": 0.0,
            "sharpe_ratio_news": 0.0,
            "sharpe_ratio_delta": 0.0,
            "sharpe_news_wins": None,
            "profit_factor_base": None,
            "profit_factor_news": None,
            "profit_factor_delta": None,
            "max_drawdown_base": None,
            "max_drawdown_news": None,
            "max_drawdown_delta": None,
            "sortino_ratio_base": None,
            "sortino_ratio_news": None,
            "sortino_ratio_delta": None,
            "buy_hold_return": None,
        }

    def test_summary_groups_by_ticker_and_model(self):
        comps = [
            self._comp("AAPL", "k-NN Enh. TW", "1y", 0.05, 0.07),
            self._comp("AAPL", "k-NN Enh. TW", "2y", 0.04, 0.02),  # news loses
            self._comp("AAPL", "LSTM", "1y", 0.10, 0.11),
        ]
        s = summarize_per_ticker_model(comps)
        by_key = {(r["ticker"], r["model_family"]): r for r in s}
        knn = by_key[("AAPL", "k-NN Enh. TW")]
        assert knn["periods_compared"] == 2
        assert knn["news_wins_return"] == 1  # only 1y wins
        # Median return delta of [+0.02, -0.02] = 0
        assert abs(knn["median_return_delta"]) < 1e-9
        # Best/worst period
        assert knn["best_period_for_news"] == "1y"
        assert knn["worst_period_for_news"] == "2y"

    def test_overall(self):
        comps = [
            self._comp("AAPL", "k-NN Enh. TW", "1y", 0.05, 0.07),
            self._comp("AAPL", "k-NN Enh. TW", "2y", 0.04, 0.02),
            self._comp("AAPL", "LSTM", "1y", 0.10, 0.11),
        ]
        o = overall_stats(comps)
        assert o["pairs"] == 3
        assert o["return_news_wins"] == 2
        assert o["return_pairs_defined"] == 3
        # 2/3 win rate
        assert abs(o["return_news_win_rate"] - (2 / 3)) < 1e-9


# ----------------------------------------------------------------------
# End-to-end (writes files into a tmp run dir)
# ----------------------------------------------------------------------


class TestProcessRunDir:
    def _write_ticker_csv(self, run_dir: Path, ticker: str, rows: list[dict]) -> None:
        cols = sorted({k for r in rows for k in r})
        with open(run_dir / f"{ticker}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    def _make_pair(self, ticker, period, model_family, return_base, return_news):
        news_name = next(n for n, b in PAIRS.items() if b == model_family)
        return [
            {
                "ticker": ticker,
                "period": period,
                "model": model_family,
                "accuracy": 0.50,
                "total_return": return_base,
                "profit_factor": 1.0,
                "max_drawdown": -0.05,
                "sharpe_ratio": 0.1,
                "sortino_ratio": 0.15,
                "buy_hold_return": 0.08,
            },
            {
                "ticker": ticker,
                "period": period,
                "model": news_name,
                "accuracy": 0.55 if return_news > return_base else 0.45,
                "total_return": return_news,
                "profit_factor": 1.2,
                "max_drawdown": -0.03,
                "sharpe_ratio": 0.2,
                "sortino_ratio": 0.25,
                "buy_hold_return": 0.08,
            },
        ]

    def test_full_pipeline(self, tmp_path):
        run_dir = tmp_path / "stocks_50d_fee003_bh"
        run_dir.mkdir()

        # Two tickers, two periods each, two model families each → 8 pairs total
        aapl_rows: list[dict] = []
        aapl_rows += self._make_pair("AAPL", "1y", "k-NN Enh. TW", 0.05, 0.07)
        aapl_rows += self._make_pair("AAPL", "2y", "k-NN Enh. TW", 0.04, 0.02)
        aapl_rows += self._make_pair("AAPL", "1y", "LSTM", 0.10, 0.11)
        aapl_rows += self._make_pair("AAPL", "2y", "LSTM", 0.08, 0.09)

        msft_rows: list[dict] = []
        msft_rows += self._make_pair("MSFT", "1y", "k-NN Enh. TW", 0.06, 0.05)
        msft_rows += self._make_pair("MSFT", "2y", "k-NN Enh. TW", 0.03, 0.04)
        msft_rows += self._make_pair("MSFT", "1y", "LSTM", 0.12, 0.13)
        msft_rows += self._make_pair("MSFT", "2y", "LSTM", 0.09, 0.07)

        self._write_ticker_csv(run_dir, "AAPL", aapl_rows)
        self._write_ticker_csv(run_dir, "MSFT", msft_rows)
        # Touch a stray _summary.csv so we confirm the underscore-skip works
        (run_dir / "_summary.csv").write_text("ticker,model\nAAPL,foo\n")

        report = process_run_dir(run_dir)

        # Per-ticker output files exist with the right number of rows
        aapl_out = run_dir / "_news_vs_no_news_AAPL.csv"
        msft_out = run_dir / "_news_vs_no_news_MSFT.csv"
        summary_out = run_dir / "_news_vs_no_news_summary.csv"
        overall_out = run_dir / "_news_vs_no_news_overall.csv"
        for p in (aapl_out, msft_out, summary_out, overall_out):
            assert p.exists(), f"missing {p}"

        with open(aapl_out) as f:
            aapl_comp = list(csv.DictReader(f))
        # 4 pairs for AAPL
        assert len(aapl_comp) == 4

        with open(summary_out) as f:
            summary = list(csv.DictReader(f))
        # 4 (ticker, model_family) rows total
        assert len(summary) == 4

        # Overall report from the return value
        o = report["overall"]
        assert o["pairs"] == 8
        # AAPL: 1y k-NN wins, 2y k-NN loses, 1y LSTM wins, 2y LSTM wins → 3
        # MSFT: 1y k-NN loses, 2y k-NN wins, 1y LSTM wins, 2y LSTM loses → 2
        # Total: 5/8 news wins on return
        assert o["return_news_wins"] == 5
        assert o["return_pairs_defined"] == 8

        # Top-5 leaderboard preserves the actual top performer
        assert report["top"], "expected non-empty top leaderboard"

    def test_empty_dir_raises(self, tmp_path):
        import pytest

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            process_run_dir(empty)
