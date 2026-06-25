"""test_forecast_model_selection.py — --models selector (keys + groups).

Pins the pure key/group resolution the forecast harness uses. The module
imports StockAppAPI at load (yfinance), so these tests need the full env; they
exercise only resolve_model_keys, which is pure.
"""

from __future__ import annotations

import pytest

# Skip the whole module if the harness can't import (e.g. yfinance missing).
fh = pytest.importorskip("scripts.forecast_harness")


class TestResolveModelKeys:
    def test_default_is_paper(self):
        # None / empty → the 'paper' group (study + benchmarks, no foundation).
        assert fh.resolve_model_keys(None) == [
            "rw",
            "rwdrift",
            "seasonal",
            "arima",
            "xgboost",
            "prophet",
        ]
        assert fh.resolve_model_keys("  ") == fh.resolve_model_keys(None)

    def test_all_includes_foundation(self):
        keys = fh.resolve_model_keys("all")
        assert "chronos" in keys and "kronos" in keys
        assert keys[0] == "rw"  # stable order

    def test_foundation_group(self):
        assert fh.resolve_model_keys("foundation") == ["chronos", "kronos"]

    def test_benchmarks_group_excludes_prophet(self):
        keys = fh.resolve_model_keys("benchmarks")
        assert "prophet" not in keys
        assert "xgboost" in keys

    def test_individual_keys_comma_and_space(self):
        assert fh.resolve_model_keys("prophet,chronos") == ["prophet", "chronos"]
        assert fh.resolve_model_keys("chronos prophet") == ["prophet", "chronos"]  # reordered

    def test_group_plus_key_dedupes(self):
        # benchmarks already has xgboost; adding it again must not duplicate.
        keys = fh.resolve_model_keys("benchmarks xgboost kronos")
        assert keys.count("xgboost") == 1
        assert "kronos" in keys

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError):
            fh.resolve_model_keys("prophet,nope")

    def test_stable_order_regardless_of_input_order(self):
        a = fh.resolve_model_keys("kronos,rw,prophet")
        b = fh.resolve_model_keys("prophet,kronos,rw")
        assert a == b
