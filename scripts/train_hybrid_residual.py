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
        default=None,
        help=(
            "Cap residual history to the most-recent N rows before eval "
            "(0 = all). Default: 504 (~2 trading years) when --macro is set "
            "(macro series don't reach back further, so an uncapped window "
            "silently trains on pre-macro-era history and skips every "
            "ticker's macro variant); 0 otherwise."
        ),
    )
    parser.add_argument(
        "--macro",
        action="store_true",
        help=(
            "Also train a macro-aware residual learner (lag-1-aligned VIX/DXY/"
            "Gold/SP500/DGS1 alongside the residual window), saved to a "
            "separate '_hybrid_res_macro.pt' path. Needs xgboost-free macro "
            "fetch (engine.macro_data); skipped per-ticker on any gap."
        ),
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help=(
            "Also train a sentiment-aware residual learner (leakage-safe "
            "per-ticker pos/neg news panel alongside the residual window), "
            "saved to a separate '_hybrid_res_sentiment.pt' path (or "
            "'_hybrid_res_macro_sentiment.pt' when combined with --macro). "
            "Single fixed scorer (--sentiment-method) — unlike the harness's "
            "XGBoost/Prophet sentiment variants (fit fresh per call, one per "
            "method), this network is pretrained once; scoring it with a "
            "different method's panel than it trained on would feed "
            "out-of-distribution exog, so only one method is supported here."
        ),
    )
    parser.add_argument(
        "--sentiment-method",
        choices=["vader", "finbert", "naive"],
        default="vader",
        help="Sentiment scorer for --sentiment (default: vader).",
    )
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    if args.max_train is None:
        args.max_train = 504 if args.macro else 0
        if args.macro:
            log.warning(
                "--macro with no --max-train: defaulting to 504 (~2 trading "
                "years) so the macro variant isn't silently skipped for every "
                "ticker. Pass --max-train 0 explicitly to train on full "
                "uncapped history anyway (expect most/all macro variants to "
                "be skipped)."
            )
    elif args.macro and args.max_train == 0:
        log.warning(
            "--macro --max-train 0: training on full uncapped history. Macro "
            "series (VIX/DXY/Gold/SP500/DGS1) don't reach back that far, so "
            "most/all tickers' macro variant will likely be skipped — watch "
            "for 'skipping macro variant' messages below."
        )

    if not _TORCH_AVAILABLE:
        log.error("PyTorch not installed. Install with: uv pip install -e '.[ai]'")
        return 1

    tickers = resolve_scope(args, default=ALL_TICKERS)
    models_dir = Path(args.models_dir)

    api = StockAppAPI()
    if not args.no_refresh:
        api.refresh_tickers(list(tickers), verbose=False)

    macro_panel = None
    if args.macro:
        from engine.macro_data import MacroCache, fetch_macro

        cache = MacroCache()
        macro_panel = cache.load() if args.no_refresh else None
        if macro_panel is None or macro_panel.empty:
            macro_panel = fetch_macro()
            cache.save(macro_panel)
        if macro_panel is None or macro_panel.empty:
            log.warning("--macro: no macro series available; skipping the macro variant.")
            macro_panel = None

    trained, skipped = 0, 0
    trained_macro, skipped_macro = 0, 0
    trained_sentiment, skipped_sentiment = 0, 0
    trained_macro_sentiment, skipped_macro_sentiment = 0, 0
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
        residuals_raw = closes - fitted
        finite_mask = np.isfinite(residuals_raw)
        residuals = residuals_raw[finite_mask]

        # Map the preset tier onto the residual learner (same tiers as the
        # LSTM regressor); explicit --hidden/--epochs override.
        cfg = REG_TRAINING_PRESETS[args.preset]

        def _new_learner(cfg=cfg):
            return LSTMResidualLearner(
                window=args.window,
                hidden_size=args.hidden if args.hidden is not None else cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                epochs=args.epochs if args.epochs is not None else cfg["epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                patience=cfg["patience"],
            )

        learner = _new_learner()
        learner.fit(residuals)
        if not learner.is_trained:
            log.info("%s: residual learner couldn't train (too few residuals); skipping.", ticker)
            skipped += 1
        else:
            out = hybrid_residual_path(ticker, models_dir)
            learner.save(out, horizon=args.horizon)
            log.info("%s: trained hybrid residual learner (%s base) → %s", ticker, args.base, out)
            trained += 1

        dates = train_df["date"].astype(str).to_numpy() if "date" in train_df.columns else None
        macro_exog = None
        if macro_panel is not None:
            from engine.macro_data import align_macro

            if dates is not None:
                aligned = align_macro(list(dates), macro_panel, lag=1)
                rows = aligned.reindex(dates[finite_mask])
                if not rows.isna().any().any():
                    macro_exog = rows.to_numpy(dtype=float)
            if macro_exog is None:
                log.info("%s: macro unavailable/misaligned; skipping macro variant.", ticker)
                skipped_macro += 1
            else:
                learner_m = _new_learner()
                learner_m.fit(residuals, macro_exog)
                if not learner_m.is_trained:
                    log.info("%s: macro residual learner couldn't train; skipping.", ticker)
                    skipped_macro += 1
                else:
                    out_m = hybrid_residual_path(ticker, models_dir, macro=True)
                    learner_m.save(out_m, horizon=args.horizon)
                    log.info("%s: trained macro hybrid residual learner → %s", ticker, out_m)
                    trained_macro += 1

        if args.sentiment:
            from engine.sentiment_data import fetch_sentiment_panel

            sent_exog = None
            if dates is not None:
                panel = fetch_sentiment_panel(api, ticker, train_df, method=args.sentiment_method)
                if panel is not None:
                    rows = panel.reindex(dates[finite_mask])
                    if not rows.isna().any().any():
                        sent_exog = rows.to_numpy(dtype=float)
            if sent_exog is None:
                log.info(
                    "%s: sentiment unavailable/misaligned; skipping sentiment variant.", ticker
                )
                skipped_sentiment += 1
            else:
                learner_s = _new_learner()
                learner_s.fit(residuals, sent_exog)
                if not learner_s.is_trained:
                    log.info("%s: sentiment residual learner couldn't train; skipping.", ticker)
                    skipped_sentiment += 1
                else:
                    out_s = hybrid_residual_path(ticker, models_dir, sentiment=True)
                    learner_s.save(out_s, horizon=args.horizon)
                    log.info("%s: trained sentiment hybrid residual learner → %s", ticker, out_s)
                    trained_sentiment += 1

            if macro_exog is not None:
                combined = (
                    np.concatenate([macro_exog, sent_exog], axis=1)
                    if sent_exog is not None
                    else None
                )
                if combined is None:
                    log.info("%s: sentiment unavailable; skipping macro+sentiment variant.", ticker)
                    skipped_macro_sentiment += 1
                else:
                    learner_ms = _new_learner()
                    learner_ms.fit(residuals, combined)
                    if not learner_ms.is_trained:
                        log.info(
                            "%s: macro+sentiment residual learner couldn't train; skipping.",
                            ticker,
                        )
                        skipped_macro_sentiment += 1
                    else:
                        out_ms = hybrid_residual_path(
                            ticker, models_dir, macro=True, sentiment=True
                        )
                        learner_ms.save(out_ms, horizon=args.horizon)
                        log.info(
                            "%s: trained macro+sentiment hybrid residual learner → %s",
                            ticker,
                            out_ms,
                        )
                        trained_macro_sentiment += 1

    print(f"\nDone. Trained {trained}, skipped {skipped}.")
    if args.macro:
        print(f"Macro variant: trained {trained_macro}, skipped {skipped_macro}.")
        if trained_macro == 0:
            log.warning(
                "Macro variant: 0 tickers trained — every ticker was skipped "
                "(see 'skipping macro variant' messages above). The harness's "
                "'+ macro' hybrid run will silently fall back to the plain "
                "hybrid for all of them. Check --max-train and macro data "
                "availability before spending time on the real run."
            )
        elif skipped_macro > 0:
            log.warning(
                "Macro variant: %d/%d tickers skipped — those will silently "
                "fall back to the plain hybrid in the harness run.",
                skipped_macro,
                trained_macro + skipped_macro,
            )
    if args.sentiment:
        print(
            f"Sentiment variant ({args.sentiment_method}): trained {trained_sentiment}, "
            f"skipped {skipped_sentiment}."
        )
        if trained_sentiment == 0:
            log.warning(
                "Sentiment variant: 0 tickers trained — the harness's '+ "
                "sentiment' hybrid run will silently fall back to the plain "
                "hybrid for all of them. Check news coverage before spending "
                "time on the real run."
            )
        if args.macro:
            print(
                f"Macro+sentiment variant: trained {trained_macro_sentiment}, "
                f"skipped {skipped_macro_sentiment}."
            )
    print("Run the harness with: --hybrid --hybrid-fit pretrained (same --days/--horizon).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
