"""
train.py – CLI for training LSTM models.

Trains and saves models to the models/ directory. Saved models are loaded
automatically by main.py and backtest.py — no retraining needed.

Usage:
    uv run python train.py --ticker AAPL --period 1y --preset quick
    uv run python train.py --ticker BTC-USD --period max --preset standard
    uv run python train.py --stocks --preset quick            # train all stocks
    uv run python train.py --all --preset cluster             # train everything (cluster)
    uv run python train.py --list                             # show saved models
"""

import argparse
from pathlib import Path

from interface.api import StockAppAPI
from engine.ai_model import AIModel, TRAINING_PRESETS, MODELS_DIR
from config import ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS, DEFAULT_PERIOD


def train_model(api: StockAppAPI, ticker: str, period: str, preset: str):
    """Train and save a single model."""
    print(f"\n{'=' * 60}")
    print(f" TRAINING: {ticker} (period={period}, preset={preset})")
    print(f"{'=' * 60}")

    # Fetch data
    df = api.get_data(ticker, period="max")
    if df.empty:
        print(f"  ERROR: No data available for {ticker}")
        return False

    # Filter to period
    from backtest import filter_by_period
    filtered = filter_by_period(df, period)

    if len(filtered) < 60:
        print(f"  ERROR: Not enough data for {ticker} period={period} "
              f"({len(filtered)} rows, need 60+)")
        return False

    # Train
    model = AIModel(preset=preset)

    try:
        info = model.train(filtered, verbose=True)
    except Exception as e:
        print(f"  ERROR: Training failed: {e}")
        return False

    # Save
    path = AIModel.model_path(ticker, period, preset)
    model.save(path)

    print(f"  Accuracy: {info['final_val_accuracy']:.1%}")
    print(f"  Duration: {info['duration_seconds']:.1f}s")
    print(f"  Saved to: {path}")

    return True


def list_models():
    """Show all saved models."""
    if not MODELS_DIR.exists():
        print("No models directory found. Train a model first.")
        return

    models = list(MODELS_DIR.glob("*.pt"))
    if not models:
        print("No saved models found. Train a model first.")
        return

    print(f"\nSaved models ({len(models)}):\n")
    print(f"  {'FILE':<45} | {'SIZE':<10}")
    print(f"  {'-' * 60}")

    for m in sorted(models):
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  {m.name:<45} | {size_mb:.1f} MB")

    print(f"\n  Directory: {MODELS_DIR.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Train LSTM Models")
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Single ticker to train (e.g. AAPL, BTC-USD)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Multiple tickers to train"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stocks", action="store_true",
        help=f"Train all stocks: {STOCKS}"
    )
    group.add_argument(
        "--crypto", action="store_true",
        help=f"Train all crypto: {CRYPTO}"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Train all tickers"
    )
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, choices=ALL_PERIODS,
        help=f"Training data period (default: {DEFAULT_PERIOD})"
    )
    parser.add_argument(
        "--periods", nargs="+", default=None, choices=ALL_PERIODS,
        help="Train on multiple periods"
    )
    parser.add_argument(
        "--preset", default="quick", choices=list(TRAINING_PRESETS.keys()),
        help="Training preset: quick (~5min), standard (~30min), cluster (hours)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all saved models"
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.stocks:
        tickers = STOCKS
    elif args.crypto:
        tickers = CRYPTO
    elif args.all:
        tickers = ALL_TICKERS
    else:
        parser.error("Specify --ticker, --tickers, --stocks, --crypto, or --all")

    # Determine periods
    periods = args.periods or [args.period]

    # Show plan
    total = len(tickers) * len(periods)
    preset_info = TRAINING_PRESETS[args.preset]
    print(f"Training plan: {total} model(s)")
    print(f"  Tickers: {tickers}")
    print(f"  Periods: {periods}")
    print(f"  Preset:  {args.preset} — {preset_info['description']}")

    # Train
    api = StockAppAPI()
    success = 0
    failed = 0

    for ticker in tickers:
        for period in periods:
            if train_model(api, ticker, period, args.preset):
                success += 1
            else:
                failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f" TRAINING COMPLETE: {success} succeeded, {failed} failed")
    print(f"{'=' * 60}")

    if success > 0:
        print(f"\nUse trained models:")
        print(f"  uv run python main.py --tickers {tickers[0]}")
        print(f"  uv run python backtest.py --tickers {tickers[0]} --days 20")
        print(f"\nList all models:")
        print(f"  uv run python train.py --list")


if __name__ == "__main__":
    main()
