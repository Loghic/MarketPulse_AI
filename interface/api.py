"""
api.py – Facade (StockAppAPI) bridging the UI layer with the engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config import (
    DEFAULT_NEWS_HALF_LIFE_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_NEWS_SOURCES,
    DEFAULT_SENTIMENT_METHOD,
)
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

from config import CHRONOS_CONTEXT, CHRONOS_MODEL_ID, FORECAST_DEVICE  # add to existing import

try:
    from engine.prophet_model import _PROPHET_AVAILABLE
except ImportError:
    _PROPHET_AVAILABLE = False
try:
    from engine.chronos_model import _CHRONOS_AVAILABLE
except ImportError:
    _CHRONOS_AVAILABLE = False

log = get_logger(__name__)


@dataclass
class PredictionConfig:
    ticker: str
    period: str = "1y"
    model_type: str = "knn"
    use_time_weights: bool = False
    include_news: bool = True
    sentiment_method: str = DEFAULT_SENTIMENT_METHOD  # "vader" | "finbert" | "naive"
    news_sources: list[str] | None = None  # falls back to config default
    news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS
    news_half_life_days: float = DEFAULT_NEWS_HALF_LIFE_DAYS


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
    def __init__(
        self,
        sentiment_method: str = DEFAULT_SENTIMENT_METHOD,
        news_sources: list[str] | str | None = None,
    ):
        self.db = DatabaseManager()
        self.knn = KNNModel(k=5, features=["returns"])
        self.knn_enhanced = KNNModel(k=5, features=ALL_FEATURES)
        self.linreg = LinearRegressionModel(features=["returns"])
        self.linreg_enhanced = LinearRegressionModel(features=ALL_FEATURES)
        self.news_scraper = NewsScraper(
            default_method=sentiment_method,
            default_source=news_sources or DEFAULT_NEWS_SOURCES,
        )
        self._lstm_cache: dict = {}
        self.lstm_available = TORCH_AVAILABLE
        self.prophet_available = _PROPHET_AVAILABLE
        self.chronos_available = _CHRONOS_AVAILABLE

        try:
            from engine.kronos_model import _KRONOS_AVAILABLE
        except ImportError:
            _KRONOS_AVAILABLE = False

        self.kronos_available = _KRONOS_AVAILABLE
        self._forecast_cache: dict = {}

        models = "k-NN + k-NN Enh. + LinReg + LinReg Enh."
        if self.lstm_available:
            models += " + LSTM"
        log.info(f"System initialized ({models} + News Engine [{sentiment_method}])")

    def set_knn_k(self, k: int):
        features_naive = self.knn.features
        features_enhanced = self.knn_enhanced.features
        self.knn = KNNModel(k=k, features=features_naive)
        self.knn_enhanced = KNNModel(k=k, features=features_enhanced)

    def forecast_available(self, model_type: str) -> bool:
        return {
            "prophet": self.prophet_available,
            "chronos": self.chronos_available,
            "kronos": self.kronos_available,
        }.get(model_type, False)

    def _load_forecast_model(self, model_type: str):
        """Load + cache a forecasting model once (they're ticker-agnostic)."""
        from engine.forecast_base import ForecastModel

        if model_type in self._forecast_cache:
            return self._forecast_cache[model_type]
        model: ForecastModel
        if model_type == "prophet":
            from engine.prophet_model import ProphetModel

            model = ProphetModel()
        elif model_type == "chronos":
            from engine.chronos_model import Chronos2Model

            model = Chronos2Model(
                model_id=CHRONOS_MODEL_ID, device=FORECAST_DEVICE, context_length=CHRONOS_CONTEXT
            )
        elif model_type == "kronos":
            from engine.kronos_model import KronosModel

            model = KronosModel()
        else:
            return None
        self._forecast_cache[model_type] = model
        return model

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

    def refresh_tickers(
        self,
        tickers: list[str],
        verbose: bool = True,
        news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
        sentiment_method: str | None = None,
        news_sources: str | list[str] | None = None,
        force_news_refresh: bool = False,
    ):
        """
        Pre-fetch prices and news for all tickers into SQLite.

        Args:
            tickers: Symbols to refresh.
            news_lookback_days: How many days of news to pull from each
                source. Yahoo will ignore values larger than its ~7-day
                window; GDELT honours up to multi-year ranges (capped at
                250 articles per call).
            sentiment_method: Override the scorer for this run only
                (None → use the scraper's default).
            news_sources: Override the source(s) for this run only
                (None → use the scraper's default).
            force_news_refresh: If True, bypass the "already-fetched-today"
                cache and re-pull news. Use this for bulk historical
                fetches when populating the DB for backtests.
        """
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
                self._process_news_with_db(
                    ticker,
                    method=sentiment_method,
                    source=news_sources,
                    lookback_days=news_lookback_days,
                    force_refresh=force_news_refresh,
                )
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

        if model_type in ("prophet", "chronos", "kronos"):
            if not self.forecast_available(model_type):
                raise RuntimeError(
                    f"{model_type} not installed. Install: uv pip install -e '.[forecast]'"
                )
            return self._load_forecast_model(model_type)

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

    def _process_news_with_db(
        self,
        ticker: str,
        method: str | None = None,
        source: str | list[str] | None = None,
        lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
        force_refresh: bool = False,
    ) -> tuple[float, list[str]]:
        """
        Refresh news from the configured provider, store with real dates,
        and return (mean_score_of_freshly_fetched_items, headlines).

        Old call sites that used this only for the "current sentiment"
        keep working — we still return today's average. The big change
        is that each headline is now persisted with its **actual**
        publication date in the ``published_at`` column, making the
        DB suitable for look-ahead-safe backtests.

        ``force_refresh=True`` skips the same-day cache check, which is
        what you want for a bulk historical fetch (e.g. enabling GDELT
        for the first time on a DB that already has a Yahoo entry from
        earlier today).
        """
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Quick cache: if we've already pulled today, don't hit the network again.
        # Bypassed when force_refresh=True or when caller wants deep history.
        if not force_refresh and lookback_days <= DEFAULT_NEWS_LOOKBACK_DAYS:
            try:
                cached_today = self.db.get_news(ticker, date=today_str)
            except Exception as e:  # noqa: BLE001
                # Transient SQLite issues (e.g. macOS Spotlight indexing a
                # WAL file) shouldn't kill a long-running batch. Skip the
                # cache check and try the live fetch instead.
                log.warning(f"{ticker}: news cache lookup failed ({e}); attempting live fetch.")
                cached_today = pd.DataFrame()
            if not cached_today.empty:
                headlines = cached_today["headline"].tolist()
                if headlines:
                    return float(cached_today["sentiment_score"].mean()), headlines

        items = self.news_scraper.fetch_and_score(
            ticker,
            lookback_days=lookback_days,
            method=method,
            source=source,
        )

        if not items:
            return 0.0, []

        rows = []
        for it in items:
            rows.append(
                {
                    "ticker": ticker,
                    "date": today_str,  # bucket date = today (when fetched)
                    "headline": it.headline,
                    "sentiment_score": it.sentiment_score,
                    "published_at": it.published_at,  # ★ real publication date
                    "source": it.source,
                    "method": it.method,
                }
            )
        news_df = pd.DataFrame(rows)
        self.db.save_news(ticker, news_df)

        score = sum(i.sentiment_score for i in items) / len(items)
        return float(score), [i.headline for i in items]

    def get_sentiment_asof(
        self,
        ticker: str,
        asof_date: str,
        lookback_days: int | None = None,
        half_life_days: float | None = None,
        method: str | None = None,
    ) -> tuple[float, list[str]]:
        """
        Compute sentiment **as if today were ``asof_date``** — using only
        news whose ``published_at`` is strictly older than that date.

        This is the function the backtester calls per prediction day to
        guarantee no look-ahead leakage. Combines:

          * lookback window (drop news older than N days)
          * exponential half-life decay (recent news weighted higher)

        Args:
            ticker: Asset symbol.
            asof_date: "YYYY-MM-DD" – the prediction date. News on or
                after this date is dropped.
            lookback_days: How far back to search. None → use config default.
                0 disables the window (use any history available).
            half_life_days: Decay half-life. None → use config default.
                0 disables decay (all news weighted equally).
            method: Restrict to news scored by this method. None → use
                whatever's stored. Useful for VADER vs FinBERT comparisons.

        Returns:
            (weighted_score in [-1, 1], headlines that contributed).
        """
        if lookback_days is None:
            lookback_days = DEFAULT_NEWS_LOOKBACK_DAYS
        if half_life_days is None:
            half_life_days = DEFAULT_NEWS_HALF_LIFE_DAYS

        df = self.db.get_news_before(
            ticker,
            asof_date=asof_date,
            lookback_days=lookback_days if lookback_days > 0 else None,
            method=method,
        )
        if df.empty:
            return 0.0, []

        score = NewsScraper.weighted_score(df, asof_date=asof_date, half_life_days=half_life_days)
        return score, df["headline"].tolist()

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
        headlines: list[str] = []

        if config.include_news:
            sentiment_score, headlines = self._process_news_with_db(
                ticker,
                method=config.sentiment_method,
                source=config.news_sources,
                lookback_days=config.news_lookback_days,
            )
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
