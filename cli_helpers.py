"""
cli_helpers.py – Shared CLI argument helpers.

One definition of the asset-scope selectors (--stocks / --crypto /
--commodities / --indices / --fx / --all / --tickers), driven by
``config.ASSET_CLASSES``. Every CLI (main.py, backtest.py, run_all.py,
refresh.py) calls ``add_scope_args(parser)`` to register them and
``resolve_scope(args)`` to turn them into a ticker list, so adding an asset
class in config automatically gives it a flag everywhere — no per-script edits.
"""

from __future__ import annotations

import argparse
import sys

from config import (
    ALL_PERIODS,
    ALL_TICKERS,
    ASSET_CLASSES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEWS_HALF_LIFE_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_SENTIMENT_METHOD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TRADING_FEE_PCT,
    MODEL_FAMILIES,
    SL_SWEEP,
    tickers_for_scope,
)


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    """Register the ticker-scope selectors on an argparse parser.

    The per-class flags (--stocks, --crypto, --commodities, --indices, --fx)
    can be combined — ``resolve_scope`` unions their tickers (e.g. ``--commodities
    --fx`` -> GLD + FXE). --all selects every class; --tickers takes explicit
    symbols. ``resolve_scope`` applies precedence: tickers > all > class flags.
    """
    group = parser.add_argument_group("asset scope")
    for ac in ASSET_CLASSES:
        group.add_argument(
            f"--{ac.cli_flag}",
            action="store_true",
            help=f"Include {ac.label.lower()}: {', '.join(ac.tickers)}",
        )
    group.add_argument(
        "--all",
        action="store_true",
        help="Every configured ticker (all asset classes)",
    )
    group.add_argument(
        "--tickers",
        nargs="+",
        metavar="SYM",
        help="Explicit ticker symbols, e.g. --tickers AAPL NVDA GLD",
    )


def resolve_scope(args: argparse.Namespace, default: list[str] | None = None) -> list[str]:
    """Resolve parsed scope flags to a ticker list.

    Precedence: --tickers > --all > per-class flags > ``default``. If nothing
    is selected and no ``default`` is given, falls back to every ticker.
    Explicit --tickers are upper-cased (symbols are canonical upper-case).
    """
    if getattr(args, "tickers", None):
        # Guard the common mix-up: `--tickers commodities fx` (class names) vs
        # the `--commodities --fx` flags. Warn, but still honour literal symbols.
        flag_names = {ac.cli_flag for ac in ASSET_CLASSES} | {"all"}
        looks_like_flags = [t for t in args.tickers if t.lower() in flag_names]
        if looks_like_flags:
            hint = " ".join(f"--{t.lower()}" for t in looks_like_flags)
            print(
                f"warning: {looks_like_flags} look like asset-class flags, not symbols — "
                f"did you mean {hint}? Treating them as literal tickers.",
                file=sys.stderr,
            )
        return [t.upper() for t in args.tickers]
    if getattr(args, "all", False):
        return list(ALL_TICKERS)
    keys = [ac.key for ac in ASSET_CLASSES if getattr(args, ac.cli_flag, False)]
    if keys:
        return tickers_for_scope(*keys)
    return list(default) if default is not None else list(ALL_TICKERS)


def scope_label(args: argparse.Namespace, default: str = "all") -> str:
    """Short label for the chosen scope (used in run_all.py output-dir names).

    --tickers -> "custom", --all -> "all", one or more class flags -> their flag
    stems joined by "-" (e.g. "commodities-fx"); otherwise ``default``.
    """
    if getattr(args, "tickers", None):
        return "custom"
    if getattr(args, "all", False):
        return "all"
    flags = [ac.cli_flag for ac in ASSET_CLASSES if getattr(args, ac.cli_flag, False)]
    if flags:
        return "-".join(flags)
    return default


def scope_label_from_tickers(tickers: list[str], default: str = "custom") -> str:
    """Label a *ticker list* by which asset class(es) it exactly matches.

    Used by the web layer, which works with resolved ticker lists rather than
    argparse flags. Returns "all" for the full universe, a single class's
    cli_flag (or "-"-joined combo) when the list is exactly those classes'
    tickers, else ``default`` (typically "custom").
    """
    wanted = set(tickers)
    if not wanted:
        return default
    if wanted == set(ALL_TICKERS):
        return "all"
    matched = [ac.cli_flag for ac in ASSET_CLASSES if wanted >= set(ac.tickers)]
    covered = {t for ac in ASSET_CLASSES if ac.cli_flag in matched for t in ac.tickers}
    if matched and covered == wanted:
        return "-".join(matched)
    return default


# ----------------------------------------------------------------------
# Shared flag groups — reused by backtest.py / run_all.py / oos_harness.py
# so a flag's spelling, default and help text live in exactly one place.
# Each ``add_*_args`` registers a coherent group; the matching resolver (if
# any) turns the parsed namespace into the values the run functions expect.
# ----------------------------------------------------------------------


def add_strategy_args(
    parser: argparse.ArgumentParser,
    *,
    sl_sweep: bool = True,
    min_confidence_help: str | None = None,
) -> None:
    """Register the trading-strategy / fee / risk flags.

    --fees, --stop-loss, --turnover-fees, --hold-days, --min-confidence,
    --buy-hold.

    ``sl_sweep=True`` (default) makes --stop-loss multi-valued and adds
    --sl-sweep — pair with ``resolve_sl_levels`` to get ``(sl_levels,
    legacy_sl)``. ``sl_sweep=False`` (e.g. the OOS harness, which must not
    sweep SL) makes --stop-loss a single float and omits --sl-sweep; read
    ``args.stop_loss`` directly as the level.

    ``min_confidence_help`` overrides the default --min-confidence help text
    (the OOS harness explains its both-windows / not-swept semantics).
    """
    group = parser.add_argument_group("strategy / fees / risk")
    group.add_argument(
        "--fees",
        type=float,
        default=DEFAULT_TRADING_FEE_PCT,
        help=f"Trading fee %% per side (default: {DEFAULT_TRADING_FEE_PCT})",
    )
    if sl_sweep:
        group.add_argument(
            "--stop-loss",
            type=float,
            nargs="+",
            default=[DEFAULT_STOP_LOSS_PCT],
            metavar="PCT",
            help=(
                "Stop-loss %% (0=disabled). Exit if position drops by this %%. "
                "Pass several values to sweep, e.g. --stop-loss 0 5 10 15 "
                "(each model runs once per level; 0 = no-SL baseline)."
            ),
        )
        group.add_argument(
            "--sl-sweep",
            action="store_true",
            help=(
                f"Sweep the default stop-loss set {SL_SWEEP} (overrides --stop-loss). "
                "Each model runs once per level; 0 is the no-SL baseline."
            ),
        )
    else:
        group.add_argument(
            "--stop-loss",
            type=float,
            default=DEFAULT_STOP_LOSS_PCT,
            metavar="PCT",
            help="Stop-loss %% (0 = disabled).",
        )
    group.add_argument(
        "--turnover-fees",
        action="store_true",
        help=(
            "Charge the round-trip fee only on days the position changes "
            "(open / flip), not on every same-direction day — the realistic "
            "'trade only on signal changes' cost. Default: fee every traded day."
        ),
    )
    group.add_argument(
        "--hold-days",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Hold an opened position N days before re-reading the signal "
            "(default 1 = re-evaluate daily). Most meaningful with "
            "--turnover-fees, which then skips fees on the held days."
        ),
    )
    group.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        metavar="THETA",
        help=min_confidence_help
        or (
            "Confidence gate: sit out days whose model confidence "
            "is below THETA (0..1). Sat-out days are flat (0 P&L, no fee) and "
            "excluded from accuracy. 0 = trade every day (default)."
        ),
    )
    group.add_argument(
        "--buy-hold",
        action="store_true",
        help="Compare with buy-and-hold benchmark",
    )


def resolve_sl_levels(args: argparse.Namespace) -> tuple[list[float] | None, float]:
    """Resolve --stop-loss / --sl-sweep into ``(sl_levels, legacy_sl)``.

    * ``--sl-sweep``            -> (SL_SWEEP, 0.0)
    * multi-valued --stop-loss  -> (those levels, 0.0)
    * single --stop-loss X      -> (None, X)  — legacy single-run path, where
      run_single_backtest pairs the no-SL baseline with the SL run itself.

    Returning ``sl_levels=None`` for the single-value case preserves the exact
    legacy behaviour; ``legacy_sl`` is what callers pass as the backtester's
    ``stop_loss_pct``.
    """
    if getattr(args, "sl_sweep", False):
        return list(SL_SWEEP), 0.0
    stop_loss = getattr(args, "stop_loss", [DEFAULT_STOP_LOSS_PCT])
    if len(stop_loss) > 1:
        return [float(s) for s in stop_loss], 0.0
    return None, float(stop_loss[0])


def add_model_filter_args(parser: argparse.ArgumentParser) -> None:
    """Register --models (family filter) and --no-baselines."""
    group = parser.add_argument_group("model selection")
    group.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_FAMILIES,
        default=None,
        help="Only run these model families (default: all). e.g. --models knn lstm chronos",
    )
    group.add_argument(
        "--no-baselines",
        action="store_true",
        help=(
            "Skip the naive baselines (AlwaysLong, PreviousDay, 5/20-Day "
            "Momentum, Random). Default: baselines included."
        ),
    )


def add_news_args(parser: argparse.ArgumentParser) -> None:
    """Register the per-day sentiment knobs shared by run_all.py / oos_harness.py.

    --sentiment-method, --news-lookback-days, --news-half-life-days. Bulk-fetch
    knobs (--news-source / --news-history-days / --force-news) are run_all-only
    and stay there.
    """
    group = parser.add_argument_group("news / sentiment")
    group.add_argument(
        "--sentiment-method",
        choices=["vader", "finbert", "naive"],
        default=DEFAULT_SENTIMENT_METHOD,
        help=f"Sentiment scorer (default: {DEFAULT_SENTIMENT_METHOD}).",
    )
    group.add_argument(
        "--news-lookback-days",
        type=int,
        default=DEFAULT_NEWS_LOOKBACK_DAYS,
        help=(
            "Per-day window: only count news from the last N days "
            f"(default: {DEFAULT_NEWS_LOOKBACK_DAYS}, 0 = unbounded)."
        ),
    )
    group.add_argument(
        "--news-half-life-days",
        type=float,
        default=DEFAULT_NEWS_HALF_LIFE_DAYS,
        help=(
            "Exponential decay half-life for per-day sentiment weighting "
            f"(default: {DEFAULT_NEWS_HALF_LIFE_DAYS}, 0 = no decay)."
        ),
    )


def add_common_run_args(
    parser: argparse.ArgumentParser, *, days_default: int, with_periods: bool = True
) -> None:
    """Register the run-wide flags: --days, --no-refresh, and (optionally) --periods.

    ``days_default`` varies per CLI (backtest 5, run_all 20, oos 50), so it's a
    required keyword. ``with_periods`` lets the single-period backtest mode omit
    --periods (it owns a separate --period flag).
    """
    parser.add_argument(
        "--days",
        type=int,
        default=days_default,
        help=f"Number of walk-forward days to evaluate (default: {days_default}).",
    )
    if with_periods:
        parser.add_argument(
            "--periods",
            nargs="+",
            choices=ALL_PERIODS,
            default=list(ALL_PERIODS),
            help="Restrict the period set (default: all of ALL_PERIODS). e.g. --periods 1y 2y",
        )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip data download, use only cached data from DB (offline mode).",
    )
