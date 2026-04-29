"""
features.py – Shared feature engineering for all models.

Computes technical indicators (RSI, MACD, volatility, volume change)
and builds feature vectors from a sliding window. Used by both
KNNModel and LinearRegressionModel to avoid code duplication.
"""

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ["returns"]
ALL_FEATURES = ["returns", "volume", "rsi", "volatility", "macd"]


def validate_features(features: list[str]):
    """Raise ValueError if any feature name is unknown."""
    for f in features:
        if f not in ALL_FEATURES:
            raise ValueError(f"Unknown feature '{f}'. Available: {ALL_FEATURES}")


def feature_label(features: list[str]) -> str:
    """Human-readable label describing the active feature set."""
    if features == DEFAULT_FEATURES:
        return "naive"
    extras = [f for f in features if f != "returns"]
    return "returns+" + "+".join(extras)


def min_rows_needed(features: list[str], window_size: int) -> int:
    """Minimum DataFrame rows needed for the given feature set."""
    n = window_size + 2
    if "rsi" in features:
        n = max(n, 16 + window_size)  # RSI needs 14 warmup
    if "macd" in features:
        n = max(n, 35 + window_size)  # MACD needs ~34 warmup
    return n


# ------------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------------


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute RSI (Relative Strength Index).
    Returns array of same length as input, with NaN for first `period` values.
    Normalized to [0, 1] instead of [0, 100] for better scaling.
    """
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full_like(close, np.nan, dtype=float)

    if len(close) <= period:
        return rsi

    avg_gain = np.mean(gains[1 : period + 1])
    avg_loss = np.mean(losses[1 : period + 1])

    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = rs / (1.0 + rs)

    return rsi


def compute_macd_histogram(close: np.ndarray) -> np.ndarray:
    """
    Compute MACD histogram (MACD line - signal line).
    Uses standard 12/26/9 EMA parameters.
    Returns array of same length, with NaN for warmup period.
    """

    def ema(data, span):
        alpha = 2.0 / (span + 1)
        result = np.full_like(data, np.nan, dtype=float)
        result[span - 1] = np.mean(data[:span])
        for i in range(span, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26

    valid_start = 25
    signal = np.full_like(close, np.nan, dtype=float)

    if len(close) > valid_start + 9:
        macd_valid = macd_line[valid_start:]
        signal_valid = ema(macd_valid, 9)
        signal[valid_start:] = signal_valid

    return macd_line - signal


# ------------------------------------------------------------------
# Feature column computation
# ------------------------------------------------------------------


def compute_feature_columns(
    df: pd.DataFrame, features: list[str], window_size: int
) -> pd.DataFrame:
    """
    Add all requested indicator columns to the DataFrame.
    Column names are prefixed with '_' to avoid collisions.
    """
    df = df.copy()

    df["_ret"] = df["close"].pct_change()

    if "volume" in features:
        if "volume" in df.columns:
            df["_vol_chg"] = df["volume"].pct_change().clip(-5, 5)
        else:
            df["_vol_chg"] = 0.0

    if "rsi" in features:
        df["_rsi"] = compute_rsi(df["close"].values)

    if "volatility" in features:
        df["_volat"] = df["_ret"].rolling(window=window_size).std()

    if "macd" in features:
        df["_macd"] = compute_macd_histogram(df["close"].values)

    return df


def build_feature_vector(
    df: pd.DataFrame, idx: int, features: list[str], window_size: int
) -> np.ndarray | None:
    """
    Build a single feature vector for the window ending at `idx`.
    Returns None if any values are NaN.
    """
    parts = []

    # Returns window (always included)
    ret_window = df["_ret"].iloc[idx : idx + window_size].values
    if np.isnan(ret_window).any():
        return None
    parts.append(ret_window)

    if "volume" in features:
        vol_window = df["_vol_chg"].iloc[idx : idx + window_size].values
        if np.isnan(vol_window).any():
            return None
        parts.append(vol_window)

    if "rsi" in features:
        rsi_val = df["_rsi"].iloc[idx + window_size - 1]
        if np.isnan(rsi_val):
            return None
        parts.append(np.array([rsi_val]))

    if "volatility" in features:
        vol_val = df["_volat"].iloc[idx + window_size - 1]
        if np.isnan(vol_val):
            return None
        parts.append(np.array([vol_val]))

    if "macd" in features:
        macd_val = df["_macd"].iloc[idx + window_size - 1]
        if np.isnan(macd_val):
            return None
        parts.append(np.array([macd_val]))

    return np.concatenate(parts)


# ------------------------------------------------------------------
# Full feature matrix builder
# ------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    target_type: str = "binary",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Build feature matrix X and target vector y from a price DataFrame.

    Args:
        df:          DataFrame with 'close' (and optionally 'volume') column.
        features:    List of feature names to include.
        window_size: Sliding window size for returns/volume.
        target_type: "binary" (1=UP, 0=DOWN) for classification,
                     "continuous" (next day's return) for regression.

    Returns:
        (X, y) numpy arrays, or (None, None) if data is insufficient.
    """
    min_n = min_rows_needed(features, window_size)
    if len(df) < min_n:
        return None, None

    df = compute_feature_columns(df, features, window_size)

    # Compute target
    if target_type == "binary":
        df["_target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    else:
        df["_target"] = df["close"].pct_change().shift(-1)

    df = df.dropna(subset=["_ret", "_target"])

    X, y = [], []
    for i in range(len(df) - window_size):
        vec = build_feature_vector(df, i, features, window_size)
        if vec is None:
            continue
        y.append(df["_target"].iloc[i + window_size - 1])
        X.append(vec)

    if len(X) == 0:
        return None, None

    return np.array(X), np.array(y)
