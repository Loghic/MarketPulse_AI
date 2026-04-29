"""test_logger.py – Logger modes and progress bar tests."""


class TestLogger:
    """Logging configuration."""

    def test_get_logger(self):
        from engine.logger import get_logger

        log = get_logger("test_module")
        assert log.name == "marketpulse.test_module"

    def test_logger_has_handler(self):
        import logging

        from engine.logger import get_logger

        get_logger("test")
        root = logging.getLogger("marketpulse")
        assert len(root.handlers) > 0


class TestProgressBar:
    """Progress bar with and without tqdm."""

    def test_progress_bar_iterable(self):
        from engine.logger import progress_bar

        items = list(range(5))
        result = list(progress_bar(items, desc="test"))
        assert result == items

    def test_epoch_progress(self):
        from engine.logger import epoch_progress

        pbar = epoch_progress(10, desc="test")
        for i in range(10):
            pbar.update(1)
        pbar.close()
        assert pbar.n == 10  # should track count

    def test_epoch_progress_partial(self):
        """Simulates early stopping — only 3/10 epochs."""
        from engine.logger import epoch_progress

        pbar = epoch_progress(10, desc="test")
        for i in range(3):
            pbar.update(1)
            pbar.set_postfix_str(f"loss={0.5 - i * 0.1:.1f}")
        pbar.close()
        assert pbar.n == 3


class TestConfig:
    """Configuration sanity checks."""

    def test_all_tickers_defined(self):
        from config import ALL_TICKERS, CRYPTO, STOCKS

        assert len(STOCKS) > 0
        assert len(CRYPTO) > 0
        assert ALL_TICKERS == STOCKS + CRYPTO

    def test_periods_defined(self):
        from config import ALL_PERIODS

        assert "1mo" in ALL_PERIODS
        assert "max" in ALL_PERIODS

    def test_defaults_reasonable(self):
        from config import DEFAULT_STOP_LOSS_PCT, DEFAULT_TRADING_FEE_PCT

        assert 0 <= DEFAULT_TRADING_FEE_PCT <= 1.0
        assert DEFAULT_STOP_LOSS_PCT >= 0

    def test_benchmarks_defined(self):
        from config import CRYPTO_BENCHMARKS, STOCK_BENCHMARKS, get_benchmarks

        assert "SPY" in STOCK_BENCHMARKS
        assert "BTC-USD" in CRYPTO_BENCHMARKS
        assert len(get_benchmarks("AAPL")) > 0
        assert len(get_benchmarks("BTC-USD")) == 0  # BTC is its own benchmark

    def test_log_mode_valid(self):
        from config import LOG_MODE

        assert LOG_MODE in ("cli", "gui")
