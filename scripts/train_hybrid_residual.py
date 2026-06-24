"""
train_hybrid_residual.py – Pretrain the residual learner for the Prophet+LSTM hybrid.

The residual hybrid (``engine/residual_hybrid.py``) is slow when its LSTM
residual learner refits on every walk-forward step. This script trains that
learner **once per ticker** on the base model's in-sample residuals from the
pre-evaluation window, and saves the weights to ``models/{ticker}_hybrid_res.pt``.
The harness can then run the hybrid in ``--hybrid-fit pretrained`` mode: frozen
weights, predict-only, ~N× faster.

Leakage discipline (same as ``train_lstm_regressor.py``): trim the **last
``--days + --horizon`` rows** before fitting the base and computing residuals,
so the harness's evaluation window is never seen. Run with the **same**
``--days``/``--horizon`` you'll score with.

Base model: Prophet by default (the paper's hybrid). Needs Prophet + torch.

Example:
    uv run python scripts/train_hybrid_residual.py --stocks --days 100 --horizon 1
    uv run python scripts/forecast_harness.py --stocks --days 100 --horizon 1 \
        --hybrid --hybrid-fit pretrained --no-refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from cli_helpers import add_scope_args, resolve_scope  # noqa: E402
from config import ALL_TICKERS  # noqa: E402
from engine.logger import get_logger, progress_bar  # noqa: E402
from engine.lstm_regressor import REG_TRAINING_PRESETS  # noqa: E402
from engine.residual_learners import (  # noqa: E402
    _TORCH_AVAILABLE,
    LSTMResidualLearner,
    hybrid_residual_path,
)
from interface.api import StockAppAPI  # noqa: E402

log = get_logger("train_hybrid_residual")


def _make_base(kind: str):
    if kind == "prophet":
        from engine.prophet_model import _PROPHET_AVAILABLE, ProphetModel

        if not _PROPHET_AVAILABLE:
            raise RuntimeError(
                "Prophet not installed. Install with: uv pip install -e '.[forecast]'"
            )
        return ProphetModel()
    if kind == "arima":
        from engine.arima_model import ARIMAForecaster

        return ARIMAForecaster()
    if kind == "rw":
        from engine.naive_forecasters import RandomWalkForecaster

        return RandomWalkForecaster()
    raise ValueError(f"unknown base '{kind}'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain the residual learner for the residual hybrid (leakage-safe)."
    )
    add_scope_args(parser)
    parser.add_argument(
        "--days", type=int, default=100, help="Eval window to EXCLUDE from training."
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--base", choices=["prophet", "arima", "rw"], default="prophet")
    parser.add_argument("--window", type=int, default=20, help="Residual-learner sequence length.")
    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "cluster"],
        default="standard",
        help="Residual-learner effort tier (same tiers as the LSTM regressor).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the preset's epochs.")
    parser.add_argument(
        "--hidden", type=int, default=None, help="Override the preset's hidden size."
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Cap residual history to the most-recent N rows before eval (0 = all).",
    )
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    if not _TORCH_AVAILABLE:
        log.error("PyTorch not installed. Install with: uv pip install -e '.[ai]'")
        return 1

    tickers = resolve_scope(args, default=ALL_TICKERS)
    models_dir = Path(args.models_dir)

    api = StockAppAPI()
    if not args.no_refresh:
        api.refresh_tickers(list(tickers), verbose=False)

    trained, skipped = 0, 0
    for ticker in progress_bar(tickers, desc="Train hybrid-res"):
        df = api.get_data(ticker, period="max")
        if df is None or df.empty:
            skipped += 1
            continue

        cutoff = len(df) - args.days - args.horizon
        if cutoff <= 0:
            log.info("%s: too short to leave an eval window, skipping.", ticker)
            skipped += 1
            continue
        train_df = df.iloc[:cutoff]
        if args.max_train > 0:
            train_df = train_df.iloc[-args.max_train :]

        # Base in-sample residuals on the (eval-excluded) training window.
        try:
            base = _make_base(args.base)
            fitted = np.asarray(base.fit_in_sample(train_df), dtype=float).ravel()
            closes = np.asarray(train_df["close"], dtype=float).ravel()
        except Exception as e:  # noqa: BLE001
            log.warning("%s: base fit failed (%s); skipping.", ticker, e)
            skipped += 1
            continue
        if fitted.shape != closes.shape:
            log.info("%s: base fit misaligned; skipping.", ticker)
            skipped += 1
            continue
        residuals = closes - fitted
        residuals = residuals[np.isfinite(residuals)]

        # Map the preset tier onto the residual learner (same tiers as the
        # LSTM regressor); explicit --hidden/--epochs override.
        cfg = REG_TRAINING_PRESETS[args.preset]
        learner = LSTMResidualLearner(
            window=args.window,
            hidden_size=args.hidden if args.hidden is not None else cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            epochs=args.epochs if args.epochs is not None else cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            patience=cfg["patience"],
        )
        learner.fit(residuals)
        if not learner.is_trained:
            log.info("%s: residual learner couldn't train (too few residuals); skipping.", ticker)
            skipped += 1
            continue

        out = hybrid_residual_path(ticker, models_dir)
        learner.save(out)
        log.info("%s: trained hybrid residual learner (%s base) → %s", ticker, args.base, out)
        trained += 1

    print(f"\nDone. Trained {trained}, skipped {skipped}.")
    print("Run the harness with: --hybrid --hybrid-fit pretrained (same --days/--horizon).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
