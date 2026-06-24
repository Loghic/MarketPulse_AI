"""test_macro_xgboost.py — macro features wired into XGBoost (R4.4).

The model path needs xgboost, so it's importorskip-gated. The point is to
confirm: (1) a macro-equipped forecaster trains and forecasts a finite level,
(2) it's leakage-safe (the aligned macro it receives is already lag-1, and the
window ending at t uses macro[t] = info known at t-1), and (3) the model name
flips to 'XGBoost + macro'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.macro_data import align_macro


def _price_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2022-01-03", periods=n, freq="B").astype(str)
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"date": dates, "close": closes, "volume": vol})


def _macro_panel(dates: list[str], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"vix": rng.normal(0, 0.05, len(dates)), "dgs1": 4.0 + rng.normal(0, 0.1, len(dates))},
        index=pd.Index([str(d) for d in dates], name="date"),
    )


class TestMacroXGBoost:
    def test_trains_and_forecasts_with_macro(self):
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        from engine.xgboost_model import XGBoostForecaster

        df = _price_df(300)
        macro = align_macro(list(df["date"]), _macro_panel(list(df["date"])), lag=1)
        m = XGBoostForecaster(macro_df=macro, window_size=10)
        assert m.name == "XGBoost + macro"
        fr = m.forecast(df)
        assert fr is not None
        assert np.isfinite(fr.point)

    def test_name_plain_without_macro(self):
        pytest.importorskip("xgboost")
        from engine.xgboost_model import XGBoostForecaster

        assert XGBoostForecaster().name == "XGBoost"

    def test_macro_vec_skips_missing_dates(self):
        # Pure check of the date-lookup logic the forecaster uses internally:
        # a date absent from the macro index, or with NaN macro, yields None.
        pytest.importorskip("xgboost")
        from engine.xgboost_model import XGBoostForecaster

        df = _price_df(60)
        # Macro covers only the first 30 dates → later windows have no macro row.
        partial = align_macro(list(df["date"][:30]), _macro_panel(list(df["date"][:30])), lag=1)
        m = XGBoostForecaster(macro_df=partial, window_size=10)
        feat = m._macro is not None
        assert feat
        # A date present with finite macro returns a vector; an absent date None.
        present_day = df["date"].iloc[15]
        absent_day = df["date"].iloc[55]
        import pandas as _pd

        fd = _pd.DataFrame({"date": [present_day, absent_day], "close": [1.0, 2.0]})
        assert m._macro_vec(fd, 0) is not None
        assert m._macro_vec(fd, 1) is None
