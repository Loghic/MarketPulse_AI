"""test_cli_helpers.py — shared CLI argument groups + resolvers.

These guard the cli_helpers refactor: the groups reused by backtest.py /
run_all.py / oos_harness.py must register the right flags with the right
shapes, and resolve_sl_levels must reproduce the dispatch logic the CLIs
relied on inline.
"""

from __future__ import annotations

import argparse

import pytest

from cli_helpers import (
    add_common_run_args,
    add_model_filter_args,
    add_news_args,
    add_strategy_args,
    resolve_sl_levels,
)
from config import SL_SWEEP


def _strategy_parser(**kwargs) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_strategy_args(p, **kwargs)
    return p


# ----------------------------------------------------------------------
# resolve_sl_levels — the three dispatch paths
# ----------------------------------------------------------------------


class TestResolveSlLevels:
    def test_sl_sweep_uses_config_set(self):
        args = _strategy_parser().parse_args(["--sl-sweep"])
        sl_levels, legacy = resolve_sl_levels(args)
        assert sl_levels == list(SL_SWEEP)
        assert legacy == 0.0

    def test_sl_sweep_overrides_explicit_stop_loss(self):
        args = _strategy_parser().parse_args(["--sl-sweep", "--stop-loss", "2"])
        sl_levels, legacy = resolve_sl_levels(args)
        assert sl_levels == list(SL_SWEEP)
        assert legacy == 0.0

    def test_multi_value_stop_loss_is_a_sweep(self):
        args = _strategy_parser().parse_args(["--stop-loss", "0", "5", "10", "15"])
        sl_levels, legacy = resolve_sl_levels(args)
        assert sl_levels == [0.0, 5.0, 10.0, 15.0]
        assert legacy == 0.0

    def test_single_stop_loss_is_legacy_path(self):
        args = _strategy_parser().parse_args(["--stop-loss", "2"])
        sl_levels, legacy = resolve_sl_levels(args)
        assert sl_levels is None
        assert legacy == 2.0

    def test_default_is_no_stop_loss(self):
        args = _strategy_parser().parse_args([])
        sl_levels, legacy = resolve_sl_levels(args)
        assert sl_levels is None
        assert legacy == 0.0


# ----------------------------------------------------------------------
# add_strategy_args — sweep vs single-SL modes
# ----------------------------------------------------------------------


class TestStrategyArgs:
    def test_sweep_mode_has_list_stop_loss_and_sl_sweep(self):
        args = _strategy_parser().parse_args(["--stop-loss", "5", "10"])
        assert isinstance(args.stop_loss, list)
        assert hasattr(args, "sl_sweep")

    def test_single_mode_has_scalar_stop_loss_and_no_sl_sweep(self):
        args = _strategy_parser(sl_sweep=False).parse_args(["--stop-loss", "5"])
        assert isinstance(args.stop_loss, float)
        assert not hasattr(args, "sl_sweep")

    def test_strategy_flags_present(self):
        args = _strategy_parser().parse_args(
            [
                "--fees",
                "0.03",
                "--turnover-fees",
                "--hold-days",
                "5",
                "--min-confidence",
                "0.6",
                "--buy-hold",
            ]
        )
        assert args.fees == 0.03
        assert args.turnover_fees is True
        assert args.hold_days == 5
        assert args.min_confidence == 0.6
        assert args.buy_hold is True

    def test_min_confidence_help_override(self, capsys):
        p = _strategy_parser(min_confidence_help="SENTINEL-HELP-TEXT")
        with pytest.raises(SystemExit):
            p.parse_args(["--help"])
        assert "SENTINEL-HELP-TEXT" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Other groups
# ----------------------------------------------------------------------


class TestOtherGroups:
    def test_model_filter(self):
        p = argparse.ArgumentParser()
        add_model_filter_args(p)
        args = p.parse_args(["--models", "knn", "lstm", "--no-baselines"])
        assert args.models == ["knn", "lstm"]
        assert args.no_baselines is True
        # default: all families, baselines kept
        d = p.parse_args([])
        assert d.models is None
        assert d.no_baselines is False

    def test_news_group(self):
        p = argparse.ArgumentParser()
        add_news_args(p)
        args = p.parse_args(
            [
                "--sentiment-method",
                "finbert",
                "--news-lookback-days",
                "14",
                "--news-half-life-days",
                "5",
            ]
        )
        assert args.sentiment_method == "finbert"
        assert args.news_lookback_days == 14
        assert args.news_half_life_days == 5.0

    def test_common_run_days_default_and_periods(self):
        p = argparse.ArgumentParser()
        add_common_run_args(p, days_default=42)
        args = p.parse_args([])
        assert args.days == 42
        assert args.no_refresh is False
        assert args.periods  # populated by default
        # with_periods=False omits --periods (single-period backtest mode)
        p2 = argparse.ArgumentParser()
        add_common_run_args(p2, days_default=5, with_periods=False)
        a2 = p2.parse_args([])
        assert not hasattr(a2, "periods")
