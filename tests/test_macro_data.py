"""test_macro_data.py — Macro features: leakage-safe alignment + cache.

The network fetch (yfinance/FRED) isn't unit-tested here; the correctness-
critical, pure piece is the calendar alignment + 1-day lag, which guarantees no
look-ahead. Those tests run everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.macro_data import MacroCache, _to_log_returns, align_macro


def _macro(dates: list[str], **cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=pd.to_datetime(dates))


class TestAlignNoLookahead:
    def test_lag1_uses_prior_day(self):
        macro = _macro(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"],
            vix=[10.0, 11.0, 12.0, 13.0],
        )
        out = align_macro(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"], macro, lag=1)
        # First row has no prior macro → NaN; each later row uses the PREVIOUS
        # trading day's value (never its own / the future).
        assert np.isnan(out["vix"].iloc[0])
        assert out["vix"].iloc[1] == 10.0
        assert out["vix"].iloc[2] == 11.0
        assert out["vix"].iloc[3] == 12.0

    def test_lag0_uses_same_day(self):
        macro = _macro(["2026-01-01", "2026-01-02"], vix=[10.0, 11.0])
        out = align_macro(["2026-01-01", "2026-01-02"], macro, lag=0)
        assert out["vix"].iloc[0] == 10.0
        assert out["vix"].iloc[1] == 11.0

    def test_forward_fills_macro_gaps_before_lag(self):
        # Macro is missing 2026-01-02; ticker trades it.
        macro = _macro(["2026-01-01", "2026-01-03"], vix=[10.0, 12.0])
        out = align_macro(["2026-01-01", "2026-01-02", "2026-01-03"], macro, lag=1)
        assert np.isnan(out["vix"].iloc[0])
        assert out["vix"].iloc[1] == 10.0  # prior day = 01-01
        # 01-03's lag-1 input is 01-02, which forward-fills to 01-01's 10.0.
        assert out["vix"].iloc[2] == 10.0

    def test_index_is_ticker_dates(self):
        macro = _macro(["2026-01-01", "2026-01-02"], vix=[1.0, 2.0])
        out = align_macro(["2026-01-01", "2026-01-02"], macro, lag=1)
        assert list(out.index) == ["2026-01-01", "2026-01-02"]

    def test_empty_macro_returns_empty_frame(self):
        out = align_macro(["2026-01-01", "2026-01-02"], pd.DataFrame(), lag=1)
        assert out.empty
        assert list(out.index) == ["2026-01-01", "2026-01-02"]

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            align_macro(["2026-01-01"], _macro(["2026-01-01"], vix=[1.0]), lag=-1)

    def test_multiple_series_aligned_together(self):
        macro = _macro(["2026-01-01", "2026-01-02"], vix=[10.0, 11.0], dgs1=[4.5, 4.6])
        out = align_macro(["2026-01-01", "2026-01-02"], macro, lag=1)
        assert set(out.columns) == {"vix", "dgs1"}
        assert out["dgs1"].iloc[1] == 4.5


class TestLogReturns:
    def test_log_returns(self):
        lr = _to_log_returns(pd.Series([100.0, 110.0, 121.0]))
        # ln(110/100) == ln(121/110) == ln(1.1).
        assert lr.iloc[1] == pytest.approx(np.log(1.1))
        assert lr.iloc[2] == pytest.approx(np.log(1.1))

    def test_drops_nonpositive(self):
        lr = _to_log_returns(pd.Series([100.0, 0.0, 110.0]))
        assert np.isfinite(lr.dropna()).all()


class TestMacroCache:
    def test_save_load_roundtrip(self, tmp_path):
        db = str(tmp_path / "m.db")
        df = pd.DataFrame(
            {"vix": [0.01, -0.02], "dgs1": [4.5, 4.6]},
            index=pd.Index(["2026-01-01", "2026-01-02"], name="date"),
        )
        cache = MacroCache(db_path=db)
        cache.save(df)
        loaded = cache.load()
        assert set(loaded.columns) == {"vix", "dgs1"}
        assert loaded.loc["2026-01-02", "dgs1"] == pytest.approx(4.6)

    def test_load_missing_is_empty(self, tmp_path):
        assert MacroCache(db_path=str(tmp_path / "none.db")).load().empty
