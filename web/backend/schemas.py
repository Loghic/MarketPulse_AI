"""
schemas.py – Pydantic models for API request/response.

Maps to existing engine dataclasses but adds validation and serialization.
"""

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Ticker / Data
# ------------------------------------------------------------------


class TickerInfo(BaseModel):
    ticker: str
    asset_type: str  # registry key: stock / crypto / commodity / index / fx
    rows: int
    last_date: str | None = None


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


class RefreshNewsRequest(BaseModel):
    """Dedicated news refresh — matches ``refresh.py``'s news-only flags."""

    tickers: list[str] = Field(default_factory=list, description="Empty = all tickers")
    sentiment_method: str | None = Field(
        default=None, description="vader | finbert | naive (None = config default)"
    )
    news_source: list[str] | None = Field(
        default=None,
        description="One or more of yahoo / gdelt. Combined with deduplication when multiple.",
    )
    news_history_days: int = Field(
        default=7, description="How many days of news to pull per ticker"
    )
    force_news: bool = Field(
        default=True, description="Bypass the same-day cache check (recommended for bulk fetch)"
    )


class RefreshNewsStatus(BaseModel):
    ticker: str
    headlines_pulled: int
    mean_sentiment: float
    method: str
    sources: list[str]
    error: str | None = None


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


class PredictRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, description="Empty = all")
    models: list[str] = Field(
        default=["all"],
        description="Model types: knn, knn_enhanced, linreg, linreg_enhanced, lstm, all",
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
    periods: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of periods to backtest (overrides period + "
            "compare_periods). e.g. ['1y', '2y']."
        ),
    )
    fee_pct: float = 0.05
    stop_loss_pct: float = 0.0
    compare_periods: bool = False
    buy_hold: bool = True
    refresh_data: bool = True
    # News / sentiment knobs (default to config values when omitted)
    sentiment_method: str | None = None
    news_lookback_days: int | None = None
    news_half_life_days: float | None = None
    # Model-family filter + baselines toggle (None / True = all + baselines).
    models: list[str] | None = None
    include_baselines: bool = True
    # Confidence gate (Plan 1.3): days below this are sat out.
    min_confidence: float = 0.0
    # Turnover / fee realism (2.1).
    turnover_fees: bool = False
    hold_days: int = 1
    # Position mode: compound consecutive same-direction days into one held
    # trade (one round-trip fee per run) instead of daily mark-to-market.
    position_mode: bool = False
    # Stop-loss sweep (2.2): explicit levels, or sl_sweep to use config.SL_SWEEP.
    # When either is set, each model runs once per level (overrides stop_loss_pct).
    sl_levels: list[float] | None = None
    sl_sweep: bool = False


class BacktestDayRow(BaseModel):
    date: str
    predicted: str
    actual: str
    correct: bool
    confidence: float
    trade_pnl: float
    trade_pnl_net: float
    stopped_out: bool
    # Sentiment that drove this day's prediction (0.0 for price-only rows).
    # Lets the Backtest tab chart "how news contributed each day".
    sentiment_score: float = 0.0


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
    # Confidence gate + turnover (Plan 1.3 / 2.1). Defaults keep older
    # persisted runs loadable.
    min_confidence: float = 0.0
    sat_out_count: int = 0
    coverage: float = 1.0
    turnover_fees: bool = False
    hold_days: int = 1
    turnover_count: int = 0
    fees_paid: float = 0.0
    position_mode: bool = False
    benchmarks: dict[str, float] = Field(default_factory=dict)
    days: list[BacktestDayRow] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    results: list[BacktestModelResult]
    best_by_return: BacktestModelResult | None = None
    best_by_sharpe: BacktestModelResult | None = None


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


# ------------------------------------------------------------------
# Meta — config-driven options for the frontend (single source of truth)
# ------------------------------------------------------------------


class ModelFamily(BaseModel):
    """A model family the frontend can offer, with availability gating."""

    key: str  # MODEL_FAMILIES key, e.g. "knn", "lstm", "chronos", "baseline"
    label: str  # display label, e.g. "Chronos-2"
    available: bool  # False when the optional dep / clone isn't installed
    predict: bool  # True if usable in the per-ticker Predict flow (not just backtests)
    note: str = ""  # short reason when unavailable / backtest-only
    # UI grouping tier: "educational" (simple, illustrative — k-NN, LinReg),
    # "forecast" (the main predictive models — LSTM, Prophet, Chronos-2, Kronos),
    # or "baseline" (reference floors). Lets the frontend de-emphasise the
    # simple ones without dropping them.
    tier: str = "forecast"
    # Rough compute cost hint for the UI ("fast" / "slow"). LSTM is fast once
    # trained; Prophet/Chronos-2/Kronos are slow (fit/inference per call).
    slow: bool = False


class AssetClassInfo(BaseModel):
    """One asset class from the registry (drives the scope pickers)."""

    key: str  # "stock", "crypto", "commodity", "index", "fx"
    label: str  # "Stocks", "Crypto", …
    cli_flag: str  # "stocks", "crypto", …
    tickers: list[str]
    benchmarks: list[str]


class MetaResponse(BaseModel):
    """Everything the frontend needs to render pickers without hardcoding."""

    model_families: list[ModelFamily]
    asset_classes: list[AssetClassInfo]
    benchmarks: list[str]  # every benchmark symbol available
    periods: list[str]
    sentiment_methods: list[str]
    sl_sweep: list[float]  # default stop-loss sweep set
    confidence_sweep: list[float]
    defaults: dict[str, float | int | str | bool]


# ------------------------------------------------------------------
# Out-of-sample harness (select on window N → evaluate on disjoint N+1)
# ------------------------------------------------------------------


class OOSRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    days: int = 50
    periods: list[str] | None = None  # None = all periods as selection candidates
    fee_pct: float = 0.05
    stop_loss_pct: float = 0.0
    buy_hold: bool = True
    refresh_data: bool = True
    models: list[str] | None = None
    include_baselines: bool = True
    # Same gate / turnover knobs as backtest, applied to BOTH windows.
    min_confidence: float = 0.0
    turnover_fees: bool = False
    hold_days: int = 1
    position_mode: bool = False
    # News / sentiment
    sentiment_method: str | None = None
    news_lookback_days: int | None = None
    news_half_life_days: float | None = None


class OOSTickerRow(BaseModel):
    """One ticker's selection winner + its out-of-sample evaluation."""

    ticker: str
    winner_model: str
    winner_period: str
    winner_family: str
    in_sample_return: float
    in_sample_accuracy: float
    in_sample_buy_hold: float
    oos_return: float
    oos_accuracy: float
    oos_buy_hold: float
    oos_sharpe: float
    beats_bh_oos: int
    stable: int
    # Gate / calibration (only meaningful when min_confidence > 0)
    min_confidence: float = 0.0
    oos_coverage: float = 1.0
    oos_traded_days: int = 0
    oos_sat_out: int = 0
    oos_brier: float = 0.0
    oos_ece: float = 0.0
    oos_binomial_p: float = 1.0
    oos_acc_ci_lo: float = 0.0
    oos_acc_ci_hi: float = 1.0


class OOSSummary(BaseModel):
    tickers: int
    oos_beat_bh_rate: float
    median_oos_return: float
    mean_oos_return: float
    median_in_sample_return: float
    in_sample_minus_oos_median: float
    median_oos_accuracy: float
    min_confidence: float = 0.0
    median_oos_coverage: float = 1.0
    median_oos_brier: float = 0.0
    median_oos_ece: float = 0.0
    tickers_significant_p05: int = 0


class OOSResponse(BaseModel):
    rows: list[OOSTickerRow]
    summary: OOSSummary
    results_dir: str | None = None
    run_id: str | None = None
