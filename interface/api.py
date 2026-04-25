"""
api.py – Facade (StockAppAPI) bridging the UI layer with the engine logic.

All model calls, data fetching, caching, and sentiment analysis are routed
through StockAppAPI so that swapping the CLI for a web or desktop UI requires
no changes to the engine layer.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime, timedelta, date

import pandas as pd

from engine.db_manager import DatabaseManager
from engine.data_downloader import get_historical_data
from engine.knn_model import KNNModel
from engine.lin_reg_model import LinearRegressionModel
from engine.news_scraper import NewsScraper


@dataclass
class PredictionConfig:
    """Configuration object for a single prediction request."""
    ticker: str
    period: str = "1y"
    model_type: str = "knn"         # "knn" or "linreg"
    use_time_weights: bool = False
    include_news: bool = True


@dataclass
class PredictionResult:
    """Standardized output returned by get_prediction()."""
    ticker: str
    prediction: str
    confidence: str
    last_price: float
    analysis_period: str
    model_type: str = "knn"
    sentiment: str = "N/A"
    sentiment_score: float = 0.0
    headlines: List[str] = field(default_factory=list)
    data_points: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


class StockAppAPI:
    """
    Main system facade — single entry point for CLI and future web/desktop UI.
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.knn = KNNModel(k=5)
        self.linreg = LinearRegressionModel()
        self.news_scraper = NewsScraper()
        print("MarketPulse AI: System initialized (k-NN + LinReg + News Engine)")

    def set_knn_k(self, k: int):
        """Dynamically change the number of neighbors for k-NN."""
        self.knn = KNNModel(k=k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_required_start_date(period: str) -> date:
        """Map a period string to the earliest date we need from the DB."""
        today = datetime.now().date()
        mapping = {
            "1mo": today - timedelta(days=30),
            "1y":  today - timedelta(days=365),
            "2y":  today - timedelta(days=730),
            "5y":  today - timedelta(days=1825),
            "max": date(1900, 1, 1),
        }
        return mapping.get(period, today - timedelta(days=365))

    def _get_model(self, model_type: str):
        """Return the model instance for the given type string."""
        models = {
            "knn": self.knn,
            "linreg": self.linreg,
        }
        model = models.get(model_type)
        if model is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {', '.join(models.keys())}"
            )
        return model

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    def get_data(self, ticker: str, period: str = "max") -> pd.DataFrame:
        """
        Return price data for *ticker*. Fetches from the DB first; if the data
        is missing or stale it downloads fresh data via yfinance and updates
        the DB before returning.
        """
        ticker = ticker.upper()
        asset_type = "crypto" if "-USD" in ticker else "stock"
        df = self.db.get_prices(ticker)

        if df.empty:
            return self._refresh_data(ticker, period, asset_type)

        # Staleness check
        last_date = pd.to_datetime(df["date"].max()).date()
        today = datetime.now().date()
        diff_days = (today - last_date).days

        needs_update = False
        if asset_type == "crypto" and diff_days >= 1:
            needs_update = True
        elif asset_type == "stock" and diff_days >= 1:
            # On a weekday, or if data is older than a weekend gap
            if today.weekday() < 5 or diff_days >= 3:
                needs_update = True

        if needs_update:
            return self._refresh_data(ticker, period, asset_type)

        return df

    def _refresh_data(self, ticker: str, period: str, asset_type: str) -> pd.DataFrame:
        """Download fresh data and persist it to the DB."""
        print(f"  Refreshing data for {ticker} (period={period})...")
        new_data = get_historical_data(ticker, period=period)
        if not new_data.empty:
            self.db.save_prices(ticker, new_data, asset_type)
            return self.db.get_prices(ticker)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # News / sentiment
    # ------------------------------------------------------------------

    def _process_news_with_db(self, ticker: str) -> Tuple[float, List[str]]:
        """
        Fetch news headlines and their sentiment score for *ticker*.
        Results are cached in the DB for one calendar day.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")

        # 1. Try the DB cache
        existing_news = self.db.get_news(ticker, date=today_str)

        if not existing_news.empty:
            headlines = existing_news["headline"].tolist()
            if headlines:
                score = existing_news["sentiment_score"].mean()
                return float(score), headlines

        # 2. Download fresh headlines
        score, headlines = self.news_scraper.get_sentiment(ticker)

        if headlines:
            news_df = pd.DataFrame({
                "ticker": [ticker] * len(headlines),
                "date": [today_str] * len(headlines),
                "headline": headlines,
                "sentiment_score": [score] * len(headlines),
            })
            self.db.save_news(ticker, news_df)

        return score, headlines

    # ------------------------------------------------------------------
    # Prediction (main entry point)
    # ------------------------------------------------------------------

    def get_prediction(self, config: PredictionConfig) -> PredictionResult:
        """
        Run a full prediction pipeline for the given configuration:
            1. Load / refresh price data.
            2. (Optional) Fetch news sentiment.
            3. Run the selected model with optional sentiment adjustment.
            4. Return a PredictionResult.
        """
        ticker = config.ticker.upper()

        # 1. Price data
        data = self.get_data(ticker, period=config.period)
        if data.empty:
            raise RuntimeError(f"No data available for {ticker}")

        # Filter to the requested analysis period
        required_start = self._get_required_start_date(config.period)
        data["_date"] = pd.to_datetime(data["date"]).dt.date
        filtered = data[data["_date"] >= required_start].copy().drop(columns=["_date"])

        if filtered.empty or len(filtered) < 10:
            raise RuntimeError(f"Insufficient data for {ticker} (period={config.period})")

        # 2. News sentiment (fetched BEFORE the model so we can feed it in)
        sentiment_label = "NEUTRAL"
        sentiment_score = 0.0
        headlines = []

        if config.include_news:
            sentiment_score, headlines = self._process_news_with_db(ticker)
            if sentiment_score > 0.15:
                sentiment_label = "POSITIVE"
            elif sentiment_score < -0.15:
                sentiment_label = "NEGATIVE"

        # 3. Model prediction — both models share the same .predict() interface
        model = self._get_model(config.model_type)
        label, prob = model.predict(
            filtered,
            use_time_weights=config.use_time_weights,
            sentiment_score=sentiment_score,
        )

        return PredictionResult(
            ticker=ticker,
            prediction=label,
            confidence=f"{prob * 100:.1f}%",
            last_price=float(filtered["close"].iloc[-1]),
            analysis_period=config.period,
            model_type=config.model_type,
            sentiment=sentiment_label,
            sentiment_score=round(sentiment_score, 2),
            headlines=headlines[:5],
            data_points=len(filtered),
        )
