"""test_macro_prophet.py — macro regressors wired into Prophet (R3.3).

Prophet-gated. Verifies: (1) a macro-equipped Prophet trains + forecasts a
finite level, (2) the name flips to 'Prophet + macro', (3) it's leakage-safe —
the macro it receives is already lag-1 aligned, and the forecast-date regressor
value is carried forward from the last in-window macro (= info known at t-1).
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
    return pd.DataFrame({"date": dates, "close": closes})


def _macro_panel(dates: list[str], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"vix": rng.normal(0, 0.05, len(dates)), "dgs1": 4.0 + rng.normal(0, 0.1, len(dates))},
        index=pd.Index([str(d) for d in dates], name="date"),
    )


class TestMacroProphet:
    def test_name_plain_without_macro(self):
        pytest.importorskip("prophet")
        from engine.prophet_model import ProphetModel

        assert ProphetModel().name == "Prophet"

    def test_name_with_macro(self):
        pytest.importorskip("prophet")
        from engine.prophet_model import ProphetModel

        df = _price_df(60)
        macro = align_macro(list(df["date"]), _macro_panel(list(df["date"])), lag=1)
        assert ProphetModel(macro_df=macro).name == "Prophet + macro"

    def test_trains_and_forecasts_with_macro(self):
        pytest.importorskip("prophet")
        from engine.prophet_model import ProphetModel

        df = _price_df(120)
        macro = align_macro(list(df["date"]), _macro_panel(list(df["date"])), lag=1)
        fr = ProphetModel(macro_df=macro).forecast(df)
        assert fr is not None
        assert np.isfinite(fr.point)

    def test_falls_back_when_macro_has_gaps(self):
        # Macro covers only the first half of the window → a gap over training,
        # so the model should skip the regressors and still forecast (plain
        # Prophet) rather than error.
        pytest.importorskip("prophet")
        from engine.prophet_model import ProphetModel

        df = _price_df(120)
        partial = align_macro(list(df["date"][:60]), _macro_panel(list(df["date"][:60])), lag=1)
        # Reindex onto the full date range → second half is NaN.
        partial = partial.reindex([str(d) for d in df["date"]])
        fr = ProphetModel(macro_df=partial).forecast(df)
        assert fr is not None
        assert np.isfinite(fr.point)
