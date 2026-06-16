"""
chronos_model.py – Amazon's Chronos-2 as a next-day direction model.

Chronos-2 is a 120M-parameter, zero-shot time-series foundation model. Nothing
to train: weights download from the Hugging Face Hub on first use and the
pipeline is then reused for every ticker (it is ticker-agnostic), so we load it
once and cache it.

We feed it the recent close series and ask for a 1-step quantile forecast, then
derive P(up) from the predicted quantiles (handled by ForecastModel).

    pip install chronos-forecasting        # >= 2.0 for Chronos-2

Verified against chronos-forecasting 2.2.2. Uses ``Chronos2Pipeline.predict_quantiles``
with a list of univariate series as the positional ``inputs`` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

log = get_logger(__name__)

try:
    import torch
    from chronos import Chronos2Pipeline

    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False

# Quantile grid used to estimate the direction probability. Wider than needed so
# the CDF inversion in the base class has enough resolution near the median.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

MIN_ROWS = 20


class Chronos2Model(ForecastModel):
    """Chronos-2 zero-shot next-day forecaster. Loads the pipeline lazily."""

    name = "Chronos-2"

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str | None = None,
        context_length: int = 512,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
    ):
        if not _CHRONOS_AVAILABLE:
            raise RuntimeError(
                "chronos-forecasting is not installed. "
                "Install with: uv pip install chronos-forecasting\n"
                "Or: uv pip install -e '.[forecast]'"
            )
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        self.quantile_levels = list(quantile_levels)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        log.info(f"Loading Chronos-2 pipeline: {self.model_id} (device={self.device})")
        # On CPU, let it default to float32 (and stay quiet). On GPU, request
        # bfloat16 via the modern `dtype=` kwarg ('torch_dtype' is deprecated in
        # recent transformers and emits a warning).
        kwargs: dict = {"device_map": self.device}
        if self.device != "cpu":
            kwargs["dtype"] = torch.bfloat16
        try:
            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            # Older transformers that only accept the legacy kwarg name.
            kwargs.pop("dtype", None)
            if self.device != "cpu":
                kwargs["torch_dtype"] = torch.bfloat16
            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, **kwargs)
        return self._pipe

    def _forecast_quantiles(self, closes: np.ndarray) -> tuple[float, dict[float, float]]:
        """Return (point, {level: value}) for the 1-step-ahead forecast.

        Verified against chronos-forecasting 2.2.2: ``predict_quantiles`` takes a
        positional ``inputs`` (a list of series — a 1-D tensor per univariate
        series), and returns (quantiles, mean) as *lists* of tensors, one entry
        per input series. Each quantiles entry has shape
        (n_variates, prediction_length, n_quantile_levels); each mean entry
        (n_variates, prediction_length).
        """
        pipe = self._load()
        series = torch.tensor(np.asarray(closes, dtype=np.float32))
        quantiles, mean = pipe.predict_quantiles(
            [series],  # one univariate series
            prediction_length=1,
            quantile_levels=self.quantile_levels,
        )
        q = np.asarray(quantiles[0], dtype=float)  # (n_variates, pred_len, n_levels)
        m = np.asarray(mean[0], dtype=float)  # (n_variates, pred_len)
        qrow = q[0, 0, :]  # variate 0, step 0
        point = float(m[0, 0])
        return point, {lvl: float(v) for lvl, v in zip(self.quantile_levels, qrow)}

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if len(df) < MIN_ROWS or "close" not in df.columns:
            return None
        closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(closes) < MIN_ROWS:
            return None
        closes = closes[-self.context_length :]
        last_close = float(closes[-1])

        point, quantiles = self._forecast_quantiles(closes)
        if not quantiles:
            return None
        return ForecastResult(
            last_close=last_close,
            point=point,
            horizon=horizon,
            quantiles=quantiles,
        )
