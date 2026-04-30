"""
schemas.py – Pydantic models for API request/response.

Maps to existing engine dataclasses but adds validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ------------------------------------------------------------------
# Ticker / Data
# ------------------------------------------------------------------

class TickerInfo(BaseModel):
    ticker: str
    asset_type: str  # "stock" or "crypto"
    rows: int
    last_date: Optional[str] = None


class OHLCVRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TickerDataResponse(BaseModel):
    ticker: str
    rows: int
    data: list[OHLCVRow]


class RefreshRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, description="Empty = all tickers")


class RefreshStatus(BaseModel):
    ticker: str
    rows: int
    last_date: str
    news_count: int


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, description="Empty = all")
    models: list[str] = Field(
        default=["all"],
        description="Model types: knn, knn_enhanced, linreg, linreg_enhanced, lstm, all"
    )
    period: str = "1y"
    include_news: bool = True
    refresh_data: bool = True


class PredictionRow(BaseModel):
    ticker: str
    model: str
    period: str
    prediction: str  # "UP" or "DOWN"
    confidence: float
    last_price: float
    sentiment: str
    sentiment_score: float
    headlines: list[str]
    timestamp: str


class PredictResponse(BaseModel):
    predictions: list[PredictionRow]
    cached: bool = False


# ------------------------------------------------------------------
# Backtest
# ------------------------------------------------------------------

class BacktestRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    days: int = 20
    period: str = "max"
    fee_pct: float = 0.05
    stop_loss_pct: float = 0.0
    compare_periods: bool = False
    buy_hold: bool = True
    refresh_data: bool = True


class BacktestDayRow(BaseModel):
    date: str
    predicted: str
    actual: str
    correct: bool
    confidence: float
    trade_pnl: float
    trade_pnl_net: float
    stopped_out: bool


class BacktestModelResult(BaseModel):
    model: str
    ticker: str
    period: str
    accuracy: float
    total_return: float
    buy_hold_return: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    buy_hold_max_drawdown: float
    fee_pct: float
    stop_loss_pct: float
    stopped_out_count: int
    win_trades: int
    loss_trades: int
    avg_win: float
    avg_loss: float
    best_day: float
    worst_day: float
    longest_win_streak: int
    longest_loss_streak: int
    benchmarks: dict[str, float] = Field(default_factory=dict)
    days: list[BacktestDayRow] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    results: list[BacktestModelResult]
    best_by_return: Optional[BacktestModelResult] = None
    best_by_sharpe: Optional[BacktestModelResult] = None


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

class TrainRequest(BaseModel):
    ticker: str
    period: str = "1y"
    preset: str = "quick"  # quick, standard, cluster


class TrainStatus(BaseModel):
    ticker: str
    period: str
    preset: str
    status: str  # "running", "complete", "error"
    epoch: int = 0
    total_epochs: int = 0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    message: str = ""


class ModelInventoryItem(BaseModel):
    ticker: str
    period: str
    preset: str
    filename: str
    size_kb: float
    modified: str  # ISO datetime


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

class UserSettings(BaseModel):
    # Global defaults
    default_period: str = "1y"
    default_fee_pct: float = 0.05
    default_stop_loss_pct: float = 0.0
    default_backtest_days: int = 20

    # Per-model overrides
    knn_k: int = 5
    knn_enhanced_k: int = 5

    # LSTM
    lstm_preset: str = "standard"

    # Display
    log_mode: str = "gui"


# ------------------------------------------------------------------
# Analysis (News vs No-News comparison)
# ------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    days: int = 50
    period: str = "max"
    fee_pct: float = 0.05


class NewsComparisonRow(BaseModel):
    ticker: str
    model: str
    period: str
    return_no_news: float
    return_with_news: float
    diff: float
    sharpe_no_news: float
    sharpe_with_news: float
    accuracy_no_news: float
    accuracy_with_news: float
