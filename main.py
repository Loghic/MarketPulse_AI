"""
main.py – CLI entry point for MarketPulse AI.

Tickers and periods are configured in config.py.
"""

import argparse

# imports — drop STOCKS, CRYPTO; add the helper
from cli_helpers import add_scope_args, resolve_scope
from config import ALL_PERIODS, ALL_TICKERS
from engine.logger import get_logger, progress_bar
from interface.api import PredictionConfig, StockAppAPI

log = get_logger("main")


def run_prediction(
    api: StockAppAPI, ticker: str, period: str, model: str, time_weighted: bool, news: bool
):
    """Run a single prediction and return a formatted table row (or error)."""
    cfg = PredictionConfig(
        ticker=ticker,
        period=period,
        model_type=model,
        use_time_weights=time_weighted,
        include_news=news,
    )
    try:
        r = api.get_prediction(cfg)
        return (f"{r.prediction:<8} | {r.confidence:<10} | {r.data_points:<8}"), r
    except RuntimeError as e:
        return f"{'ERROR':<8} | {str(e)[:24]:<10} |", None


def print_report(api: StockAppAPI, ticker: str):
    """Print a strategic prediction report for a single ticker."""

    print(f"\n{'=' * 90}")
    print(f" STRATEGIC REPORT: {ticker}")
    print(f"{'=' * 90}")
    print(f"{'PERIOD':<10} | {'MODEL':<22} | {'PRED.':<8} | {'CONF.':<10} | {'SAMPLES':<8}")
    print("-" * 90)

    last_res = None

    for p in ALL_PERIODS:
        # --- k-NN naive ---
        row, _ = run_prediction(api, ticker, p, "knn", False, False)
        print(f"{p:<10} | {'k-NN':<22} | {row}")

        row, _ = run_prediction(api, ticker, p, "knn", True, False)
        print(f"{'':<10} | {'k-NN Time-Weighted':<22} | {row}")

        # --- k-NN enhanced ---
        row, _ = run_prediction(api, ticker, p, "knn_enhanced", False, False)
        print(f"{'':<10} | {'k-NN Enhanced':<22} | {row}")

        row, _ = run_prediction(api, ticker, p, "knn_enhanced", True, False)
        print(f"{'':<10} | {'k-NN Enh. TW':<22} | {row}")

        # --- LinReg naive ---
        row, _ = run_prediction(api, ticker, p, "linreg", False, False)
        print(f"{'':<10} | {'LinReg':<22} | {row}")

        row, _ = run_prediction(api, ticker, p, "linreg", True, False)
        print(f"{'':<10} | {'LinReg Time-Weighted':<22} | {row}")

        # --- LinReg enhanced ---
        row, _ = run_prediction(api, ticker, p, "linreg_enhanced", False, False)
        print(f"{'':<10} | {'LinReg Enhanced':<22} | {row}")

        row, _ = run_prediction(api, ticker, p, "linreg_enhanced", True, False)
        print(f"{'':<10} | {'LinReg Enh. TW':<22} | {row}")

        # --- News-enhanced variants (only for 1mo) ---
        if p == "1mo":
            row, res = run_prediction(api, ticker, p, "knn", True, True)
            print(f"{'':<10} | {'k-NN TW + News':<22} | {row}")

            row, res_enh = run_prediction(api, ticker, p, "knn_enhanced", True, True)
            print(f"{'':<10} | {'k-NN Enh. TW + News':<22} | {row}")

            row, res_lr = run_prediction(api, ticker, p, "linreg", True, True)
            print(f"{'':<10} | {'LinReg TW + News':<22} | {row}")

            row, res_lr_enh = run_prediction(api, ticker, p, "linreg_enhanced", True, True)
            print(f"{'':<10} | {'LinReg Enh. TW + News':<22} | {row}")

            last_res = res_lr_enh or res_lr or res_enh or res

        print("-" * 90)

        # Fallback for summary block
        if last_res is None:
            _, last_res = run_prediction(api, ticker, p, "knn", False, False)

    # --- Summary block ---
    if last_res:
        print(f"  Current Market Price: {last_res.last_price:.2f} USD")
        if last_res.headlines:
            print(
                f"  Market Sentiment:    {last_res.sentiment} (Score: {last_res.sentiment_score})"
            )
            print("  Latest News:")
            for title in last_res.headlines[:3]:
                print(f"    > {title}")
    print("*" * 90)


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Predictions")
    add_scope_args(parser)
    parser.add_argument(
        "--no-refresh", action="store_true", help="Skip data download, use only cached data from DB"
    )
    args = parser.parse_args()

    # Determine tickers
    tickers = resolve_scope(args, default=ALL_TICKERS[:3])

    log.info(f"Tickers: {tickers}")

    api = StockAppAPI()

    # Refresh all data upfront (prices + news)
    if not args.no_refresh:
        api.refresh_tickers(tickers)

    for ticker in progress_bar(tickers, desc="Predicting"):
        print_report(api, ticker)


if __name__ == "__main__":
    main()
