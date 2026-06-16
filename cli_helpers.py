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

from config import ALL_TICKERS, ASSET_CLASSES, tickers_for_scope


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
