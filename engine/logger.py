"""
logger.py – Centralized logging and progress bar support.

Two modes (set in config.py → LOG_MODE):
    "cli" — INFO level, progress bars, detailed output for terminal
    "gui" — WARNING level, no progress bars, minimal output for UI

Usage:
    from engine.logger import get_logger, progress_bar

    log = get_logger(__name__)
    log.info("Starting prediction...")
    log.warning("Data might be stale")
    log.error("Download failed")

    for ticker in progress_bar(tickers, desc="Predicting"):
        ...
"""

import logging
import sys
from typing import Iterable, Any

from config import LOG_MODE, LOG_LEVEL

# Try to import tqdm for nice progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------

_initialized = False


def _setup_logging():
    """Configure root logger once."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Determine level
    if LOG_LEVEL:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    elif LOG_MODE == "gui":
        level = logging.WARNING
    else:
        level = logging.INFO

    # Format: cli gets timestamps, gui is minimal
    if LOG_MODE == "gui":
        fmt = "%(levelname)s: %(message)s"
    else:
        fmt = "%(asctime)s %(levelname)-5s %(name)-20s │ %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger("marketpulse")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Usage: log = get_logger(__name__)
    All loggers are children of 'marketpulse' root logger.
    """
    _setup_logging()
    # Strip 'engine.' prefix for cleaner names
    short = name.replace("engine.", "").replace("interface.", "")
    return logging.getLogger(f"marketpulse.{short}")


# ------------------------------------------------------------------
# Progress bar
# ------------------------------------------------------------------

class _SimpleProgress:
    """Fallback progress bar when tqdm is not available."""

    def __init__(self, iterable, desc="", total=None, **kwargs):
        self.iterable = iterable
        self.desc = desc
        self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
        self.n = 0

    def __iter__(self):
        for item in self.iterable:
            self.n += 1
            if self.total and LOG_MODE == "cli":
                pct = self.n / self.total * 100
                print(f"\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)",
                      end="", flush=True)
            yield item
        if self.total and LOG_MODE == "cli":
            print()  # newline after progress

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _SimpleManualProgress:
    """Fallback for manual update (epoch tracking)."""

    def __init__(self, total, desc="", **kwargs):
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n=1):
        self.n += n
        if LOG_MODE == "cli":
            pct = self.n / self.total * 100
            print(f"\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)",
                  end="", flush=True)

    def set_postfix_str(self, s):
        if LOG_MODE == "cli":
            pct = self.n / self.total * 100
            print(f"\r  {self.desc}: {self.n}/{self.total} ({pct:.0f}%) {s}",
                  end="", flush=True)

    def close(self):
        if LOG_MODE == "cli":
            print()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def progress_bar(iterable: Iterable, desc: str = "", total: int | None = None,
                 **kwargs) -> Any:
    """
    Wrap an iterable with a progress bar.

    In CLI mode: shows tqdm progress bar (or simple fallback).
    In GUI mode: silent passthrough (no output).
    """
    if LOG_MODE == "gui":
        return iterable

    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=f"  {desc}", total=total,
                    bar_format="{l_bar}{bar:30}{r_bar}", **kwargs)
    return _SimpleProgress(iterable, desc=desc, total=total)


def epoch_progress(total: int, desc: str = "Training") -> Any:
    """
    Create a manual-update progress bar for training epochs.

    Usage:
        pbar = epoch_progress(100, desc="LSTM Training")
        for epoch in range(100):
            ...
            pbar.update(1)
            pbar.set_postfix_str(f"loss={loss:.4f}")
        pbar.close()
    """
    if LOG_MODE == "gui":
        return _SimpleManualProgress(total, desc)  # silent but tracks .n

    if TQDM_AVAILABLE:
        return tqdm(total=total, desc=f"  {desc}",
                    bar_format="{l_bar}{bar:30}{r_bar}")
    return _SimpleManualProgress(total, desc=desc)

