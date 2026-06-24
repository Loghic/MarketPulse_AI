"""
train_lstm_regressor.py – Train per-ticker LSTM *regressors* for the forecast harness.

Produces ``models/{ticker}_reg.pt`` weights that ``LSTMRegressorForecaster``
loads. These are **separate** from the directional classifiers
(``{ticker}_{period}_{preset}.pt``) — a regression head predicting the next-close
Δ, not UP/DOWN.

Leakage discipline (the whole point of a *pre-trained* forecaster being valid):
the regressor must never be fit on the days it will later be scored on. This
script trims the **last ``--days + --horizon`` rows** off each ticker before
training, so the harness's evaluation window (the most-recent ``--days`` steps)
is entirely unseen. Run this with the **same** ``--days``/``--horizon`` you will
pass to ``scripts/forecast_harness.py``.

Example:
    uv run python scripts/train_lstm_regressor.py --stocks --days 100 --horizon 1
    uv run python scripts/forecast_harness.py     --stocks --days 100 --horizon 1 --no-refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_helpers import add_scope_args, resolve_scope  # noqa: E402
from config import ALL_TICKERS  # noqa: E402
from engine.logger import get_logger, progress_bar  # noqa: E402
from engine.lstm_regressor import (  # noqa: E402
    _MODELS_DIR,
    _TORCH_AVAILABLE,
    regressor_path,
    train_regressor,
)
from interface.api import StockAppAPI  # noqa: E402

log = get_logger("train_lstm_regressor")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-ticker LSTM regressors for the forecast harness. Trims the "
            "evaluation window before fitting so the saved weights are OOS-valid."
        )
    )
    add_scope_args(parser)
    parser.add_argument(
        "--days", type=int, default=100, help="Eval window to EXCLUDE from training."
    )
    parser.add_argument(
        "--horizon", type=int, default=1, help="Forecast horizon (target = close[t+h]−close[t])."
    )
    parser.add_argument("--window", type=int, default=20, help="LSTM sequence length.")
    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "cluster"],
        default="standard",
        help="Training-effort tier (capacity + budget). Same idea as the classifier presets.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the preset's epochs.")
    parser.add_argument(
        "--hidden", type=int, default=None, help="Override the preset's hidden size."
    )
    parser.add_argument(
        "--layers", type=int, default=None, help="Override the preset's num layers."
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Cap training to the most-recent N rows BEFORE the eval window (0 = all history).",
    )
    parser.add_argument("--no-refresh", action="store_true", help="Use cached prices.")
    parser.add_argument("--models-dir", type=str, default=str(_MODELS_DIR))
    args = parser.parse_args()

    if not _TORCH_AVAILABLE:
        log.error("PyTorch is not installed. Install with: uv pip install -e '.[ai]'")
        return 1

    tickers = resolve_scope(args, default=ALL_TICKERS)
    models_dir = Path(args.models_dir)

    api = StockAppAPI()
    if not args.no_refresh:
        api.refresh_tickers(list(tickers), verbose=False)

    trained, skipped = 0, 0
    for ticker in progress_bar(tickers, desc="Train LSTM-reg"):
        df = api.get_data(ticker, period="max")
        if df is None or df.empty:
            log.info("%s: no data, skipping.", ticker)
            skipped += 1
            continue

        # Trim the eval window (+horizon so the last target isn't peeked) so the
        # saved weights never see the days the harness will score.
        cutoff = len(df) - args.days - args.horizon
        if cutoff <= 0:
            log.info("%s: too short to leave an eval window, skipping.", ticker)
            skipped += 1
            continue
        train_df = df.iloc[:cutoff]
        if args.max_train > 0:
            train_df = train_df.iloc[-args.max_train :]

        ckpt = train_regressor(
            train_df,
            preset=args.preset,
            window_size=args.window,
            horizon=args.horizon,
            hidden_size=args.hidden,  # None → preset default
            num_layers=args.layers,
            epochs=args.epochs,
        )
        if ckpt is None:
            log.info("%s: not enough training data, skipping.", ticker)
            skipped += 1
            continue

        import torch

        out = regressor_path(ticker, models_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        log.info(
            "%s: trained on %d rows (val_loss=%.4f) → %s",
            ticker,
            ckpt["n_train"],
            ckpt["val_loss"],
            out,
        )
        trained += 1

    print(f"\nDone. Trained {trained}, skipped {skipped}.")
    print("Now run the harness with the SAME --days/--horizon to score them OOS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
