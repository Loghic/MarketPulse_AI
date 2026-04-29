"""
api.py – Facade (StockAppAPI) bridging the UI layer with the engine logic.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from engine.data_downloader import get_historical_data
from engine.db_manager import DatabaseManager
from engine.features import ALL_FEATURES
from engine.knn_model import KNNModel
from engine.lin_reg_model import LinearRegressionModel
from engine.logger import get_logger, progress_bar
from engine.news_scraper import NewsScraper
from engine.utils import period_to_start_date

try:
    from engine.ai_model import TORCH_AVAILABLE, AIModel
except ImportError:
    TORCH_AVAILABLE = False

log = get_logger(__name__)


@dataclass
class PredictionConfig:
    ticker: str
    period: str = "1y"
    model_type: str = "knn"
    use_time_weights: bool = False
    include_news: bool = True


@dataclass
class PredictionResult:
    ticker: str
    prediction: str
    confidence: str
    last_price: float
    analysis_period: str
    model_type: str = "knn"
    sentiment: str = "N/A"
    sentiment_score: float = 0.0
    headlines: list[str] = field(default_factory=list)
    data_points: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


class StockAppAPI:
    def __init__(self):
        self.db = DatabaseManager()
        self.knn = KNNModel(k=5, features=["returns"])
        self.knn_enhanced = KNNModel(k=5, features=ALL_FEATURES)
        self.linreg = LinearRegressionModel(features=["returns"])
        self.linreg_enhanced = LinearRegressionModel(features=ALL_FEATURES)
        self.news_scraper = NewsScraper()
        self._lstm_cache = {}
        self.lstm_available = TORCH_AVAILABLE

        models = "k-NN + k-NN Enh. + LinReg + LinReg Enh."
        if self.lstm_available:
            models += " + LSTM"
        log.info(f"System initialized ({models} + News Engine)")

    def set_knn_k(self, k: int):
        features_naive = self.knn.features
        features_enhanced = self.knn_enhanced.features
        self.knn = KNNModel(k=k, features=features_naive)
        self.knn_enhanced = KNNModel(k=k, features=features_enhanced)

    def _load_lstm(self, ticker: str, period: str):
        if not self.lstm_available:
            return None
        cache_key = (ticker, period)
        if cache_key in self._lstm_cache:
            return self._lstm_cache[cache_key]
        for preset in ["cluster", "standard", "quick"]:
            path = AIModel.model_path(ticker, period, preset)
            model = AIModel(preset=preset)
            if model.load(path):
                log.info(f"Loaded LSTM model: {path.name}")
                self._lstm_cache[cache_key] = model
                return model
        return None

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh_tickers(self, tickers: list[str], verbose: bool = True):
        """Pre-fetch prices and news for all tickers into SQLite."""
        log.info(f"Refreshing {len(tickers)} tickers...")

        for ticker in progress_bar(tickers, desc="Refreshing data"):
            ticker = ticker.upper()
            try:
                df = self.get_data(ticker, period="max")
                rows = len(df)
                last = df["date"].iloc[-1] if not df.empty else "n/a"
            except Exception as e:
                log.error(f"{ticker}: price download failed: {e}")
                continue

            try:
                self._process_news_with_db(ticker)
            except Exception as e:
                log.warning(f"{ticker}: news fetch failed: {e}")

            log.debug(f"{ticker}: {rows} rows (→ {last})")

        log.info("Refresh complete.")

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    def get_data(self, ticker: str, period: str = "max") -> pd.DataFrame:
        ticker = ticker.upper()
        asset_type = "crypto" if "-USD" in ticker else "stock"
        df = self.db.get_prices(ticker)

        if df.empty:
            return self._refresh_data(ticker, period, asset_type)

        last_date = pd.to_datetime(df["date"].max()).date()
        today = datetime.now().date()
        diff_days = (today - last_date).days

        needs_update = False
        if asset_type == "crypto" and diff_days >= 1:
            needs_update = True
        elif asset_type == "stock" and diff_days >= 1:
            if today.weekday() < 5 or diff_days >= 3:
                needs_update = True

        if needs_update:
            return self._refresh_data(ticker, period, asset_type)
        return df

    def _refresh_data(self, ticker: str, period: str, asset_type: str) -> pd.DataFrame:
        log.debug(f"Downloading {ticker} (period={period})...")
        new_data = get_historical_data(ticker, period=period)
        if not new_data.empty:
            self.db.save_prices(ticker, new_data, asset_type)
            return self.db.get_prices(ticker)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model(self, model_type: str, ticker: str = "", period: str = ""):
        models = {
            "knn": self.knn,
            "knn_enhanced": self.knn_enhanced,
            "linreg": self.linreg,
            "linreg_enhanced": self.linreg_enhanced,
        }
        if model_type == "lstm":
            if not self.lstm_available:
                raise RuntimeError("LSTM requires PyTorch. Install: uv pip install torch")
            model = self._load_lstm(ticker, period)
            if model is None:
                raise RuntimeError(
                    f"No trained LSTM for {ticker} (period={period}). "
                    f"Train: uv run python train.py --ticker {ticker} --period {period}"
                )
            return model
        model = models.get(model_type)
        if model is None:
            available = list(models.keys())
            if self.lstm_available:
                available.append("lstm")
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")
        return model

    # ------------------------------------------------------------------
    # News / sentiment
    # ------------------------------------------------------------------

    def _process_news_with_db(self, ticker: str):
        today_str = datetime.now().strftime("%Y-%m-%d")
        existing = self.db.get_news(ticker, date=today_str)
        if not existing.empty:
            headlines = existing["headline"].tolist()
            if headlines:
                return float(existing["sentiment_score"].mean()), headlines

        score, headlines = self.news_scraper.get_sentiment(ticker)
        if headlines:
            news_df = pd.DataFrame(
                {
                    "ticker": [ticker] * len(headlines),
                    "date": [today_str] * len(headlines),
                    "headline": headlines,
                    "sentiment_score": [score] * len(headlines),
                }
            )
            self.db.save_news(ticker, news_df)
        return score, headlines

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def get_prediction(self, config: PredictionConfig) -> PredictionResult:
        ticker = config.ticker.upper()
        data = self.get_data(ticker, period=config.period)
        if data.empty:
            raise RuntimeError(f"No data available for {ticker}")

        required_start = period_to_start_date(config.period)
        data["_date"] = pd.to_datetime(data["date"]).dt.date
        filtered = data[data["_date"] >= required_start].copy().drop(columns=["_date"])

        if filtered.empty or len(filtered) < 10:
            raise RuntimeError(f"Insufficient data for {ticker} (period={config.period})")

        sentiment_label = "NEUTRAL"
        sentiment_score = 0.0
        headlines = []

        if config.include_news:
            sentiment_score, headlines = self._process_news_with_db(ticker)
            if sentiment_score > 0.15:
                sentiment_label = "POSITIVE"
            elif sentiment_score < -0.15:
                sentiment_label = "NEGATIVE"

        model = self._get_model(config.model_type, ticker, config.period)
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
