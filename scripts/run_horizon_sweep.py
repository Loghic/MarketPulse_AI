"""
run_horizon_sweep.py – Drive the full train + score pipeline across multiple
(--days, --horizon) pairs.

LSTM-reg / hybrid-residual weights are horizon-specific (``close[t+h]-close[t]``
is a different target per horizon), so each ``(days, horizon)`` pair needs its
own ``train_lstm_regressor.py`` + ``train_hybrid_residual.py --macro`` pass
before ``forecast_harness.py --hybrid --hybrid-fit pretrained`` can use it.
Doing that by hand for N pairs means N manual 3-command sequences; this script
is that loop, with the same flags plan.md's Phase-R sweep procedure specifies
for every pair, and continues past a failing pair instead of aborting the
whole sweep.

Example (the h=5/10/20 rerun at --days 50/100, plan.md item 9):
    uv run python scripts/run_horizon_sweep.py --tickers AAPL MSFT NVDA GOOGL \\
        META TSLA BTC-USD ETH-USD GLD SLV VOO QQQM FXE FXY \\
        --days 50 100 --horizons 5 10 20 --no-refresh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_helpers import add_scope_args, resolve_scope  # noqa: E402
from config import ALL_TICKERS  # noqa: E402
from engine.logger import get_logger  # noqa: E402

log = get_logger("run_horizon_sweep")

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(args: list[str], log_path: Path) -> bool:
    """Run one ``uv run python ...`` step, tee its output to ``log_path``.

    Returns True on success (exit 0). Never raises — a failing pair should not
    kill the rest of the sweep.
    """
    log.info("$ %s", " ".join(args))
    with open(log_path, "w") as fh:
        proc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log.warning("FAILED (exit %d): %s — see %s", proc.returncode, " ".join(args), log_path)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train (LSTM-reg + hybrid-residual --macro) and score "
            "(forecast_harness, full model set) every (--days, --horizon) pair."
        )
    )
    add_scope_args(parser)
    parser.add_argument(
        "--days", type=int, nargs="+", required=True, help="--days values to sweep."
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", required=True, help="--horizon values to sweep."
    )
    parser.add_argument("--preset", choices=["quick", "standard", "cluster"], default="standard")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument(
        "--log-dir", type=str, default="results/sweep_logs", help="Where per-step logs land."
    )
    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(d, h) for d in args.days for h in args.horizons]
    log.info("Sweeping %d (days, horizon) pairs across %d tickers.", len(pairs), len(tickers))

    results: list[tuple[int, int, str, bool]] = []
    t0 = time.monotonic()
    for days, horizon in pairs:
        tag = f"d{days}_h{horizon}"
        log.info("=== %s ===", tag)
        common = ["--tickers", *tickers, "--days", str(days), "--horizon", str(horizon)]
        if args.no_refresh:
            common.append("--no-refresh")

        ok = _run(
            [
                "uv",
                "run",
                "python",
                str(SCRIPTS_DIR / "train_lstm_regressor.py"),
                *common,
                "--preset",
                args.preset,
            ],
            log_dir / f"{tag}_train_lstm.log",
        )
        results.append((days, horizon, "train_lstm_regressor", ok))

        ok = _run(
            [
                "uv",
                "run",
                "python",
                str(SCRIPTS_DIR / "train_hybrid_residual.py"),
                *common,
                "--macro",
                "--preset",
                args.preset,
            ],
            log_dir / f"{tag}_train_hybrid.log",
        )
        results.append((days, horizon, "train_hybrid_residual", ok))

        ok = _run(
            [
                "uv",
                "run",
                "python",
                str(SCRIPTS_DIR / "forecast_harness.py"),
                *common,
                "--macro",
                "--hybrid",
                "--hybrid-fit",
                "pretrained",
                "--sentiment",
            ],
            log_dir / f"{tag}_harness.log",
        )
        results.append((days, horizon, "forecast_harness", ok))

    elapsed = time.monotonic() - t0
    print(f"\nSweep done in {elapsed / 60:.1f} min. {len(pairs)} pairs, {len(results)} steps.")
    failed = [r for r in results if not r[3]]
    if failed:
        print(f"{len(failed)} step(s) failed — see {log_dir}/:")
        for days, horizon, step, _ in failed:
            print(f"  d{days}_h{horizon}: {step}")
        return 1
    print("All steps succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
