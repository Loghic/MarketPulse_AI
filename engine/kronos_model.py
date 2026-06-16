"""
kronos_model.py – Kronos (financial K-line foundation model) as a next-day
direction model.

Kronos is a zero-shot, decoder-only foundation model trained on OHLCV
candlesticks. It is **not** a pip package — it's imported as `from model import …`
against the repo root. We expect it cloned as a *sibling* of this repo:

    <parent>/
        <this-repo>/        (engine/, interface/, …)
        Kronos/             git clone https://github.com/shiyu-coder/Kronos.git

Override the location with `config.KRONOS_PATH` or the `KRONOS_PATH` env var.
After cloning, install its deps:  `pip install -r Kronos/requirements.txt`.

We feed it the recent OHLCV window and read the predicted close for direction.
`KronosPredictor.predict()` averages its internal samples and returns a single
path, so by default we derive direction/confidence from that point estimate.
Set `KRONOS_PROB_SAMPLES > 1` to instead run that many single-sample passes and
build an empirical P(up) (closer to Kronos's own "probability up" metric, but
N× slower per day).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

log = get_logger(__name__)

MIN_ROWS = 20
_PRICE_COLS = ["open", "high", "low", "close"]

# engine/kronos_model.py -> repo root is one level up
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _kronos_repo_path() -> Path:
    """Resolve the Kronos clone location: env var > config > sibling default."""
    override = os.environ.get("KRONOS_PATH") or getattr(config, "KRONOS_PATH", None)
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT.parent / "Kronos").resolve()


# Try to make Kronos importable. We APPEND to sys.path (not insert) so the
# Kronos repo root can never shadow this project's own top-level modules
# (e.g. config.py / utils.py) — only its unique `model` package is picked up.
_KRONOS_REPO = _kronos_repo_path()
try:
    import torch  # noqa: F401  (needed for device selection)

    if _KRONOS_REPO.is_dir() and str(_KRONOS_REPO) not in sys.path:
        sys.path.append(str(_KRONOS_REPO))
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    _KRONOS_AVAILABLE = True
except Exception:
    _KRONOS_AVAILABLE = False


def _resolve_device() -> str:
    """Kronos defaults to cuda:0; pick a sane device for this machine."""
    dev = getattr(config, "FORECAST_DEVICE", None)
    if dev:
        return dev
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class KronosModel(ForecastModel):
    """Kronos zero-shot next-day forecaster. Loads the predictor lazily."""

    name = "Kronos"

    def __init__(
        self,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        device: str | None = None,
        max_context: int | None = None,
        sample_count: int | None = None,
        prob_samples: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        if not _KRONOS_AVAILABLE:
            raise RuntimeError(
                f"Kronos not found (looked in {_KRONOS_REPO}). Clone it next to this repo:\n"
                "  cd <parent of this repo> && git clone https://github.com/shiyu-coder/Kronos.git\n"
                "  pip install -r Kronos/requirements.txt\n"
                "Or set KRONOS_PATH (config.py or env var) to the clone location."
            )
        self.model_id = model_id or getattr(config, "KRONOS_MODEL_ID", "NeoQuasar/Kronos-small")
        self.tokenizer_id = tokenizer_id or getattr(
            config, "KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base"
        )
        self.device = device or _resolve_device()
        self.max_context = max_context or getattr(config, "KRONOS_MAX_CONTEXT", 512)
        self.sample_count = sample_count or getattr(config, "KRONOS_SAMPLE_COUNT", 5)
        self.prob_samples = prob_samples or getattr(config, "KRONOS_PROB_SAMPLES", 1)
        self.temperature = temperature or getattr(config, "KRONOS_T", 1.0)
        self.top_p = top_p or getattr(config, "KRONOS_TOP_P", 0.9)
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        log.info(f"Loading Kronos: {self.model_id} + {self.tokenizer_id} (device={self.device})")
        try:
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
            model = Kronos.from_pretrained(self.model_id)
            self._predictor = KronosPredictor(
                model, tokenizer, device=self.device, max_context=self.max_context
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"Kronos failed to load ({e}); model will be skipped.")
            raise
        return self._predictor

    def _build_inputs(self, df: pd.DataFrame, horizon: int):
        """Return (x_df, x_timestamp, y_timestamp, last_close) or None.

        Built positionally from numpy arrays so a sliced/reindexed input df
        (as the backtester passes) can't misalign columns against timestamps.
        """
        if "date" in df.columns:
            ts = pd.to_datetime(df["date"], errors="coerce")
        else:
            ts = pd.Series(pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="D"))

        data: dict[str, np.ndarray] = {"_dt": np.asarray(ts)}
        if all(c in df.columns for c in _PRICE_COLS):
            for c in _PRICE_COLS:
                data[c] = pd.to_numeric(np.asarray(df[c]), errors="coerce")
        else:
            close = pd.to_numeric(np.asarray(df["close"]), errors="coerce")
            data["open"] = data["high"] = data["low"] = data["close"] = close
        if "volume" in df.columns:
            data["volume"] = pd.to_numeric(np.asarray(df["volume"]), errors="coerce")

        work = pd.DataFrame(data)  # clean RangeIndex
        if "volume" in work.columns:
            work["volume"] = work["volume"].fillna(0.0)
        work = work.dropna().tail(self.max_context).reset_index(drop=True)
        if len(work) < MIN_ROWS:
            return None

        feature_cols = _PRICE_COLS + (["volume"] if "volume" in work.columns else [])
        x_df = work[feature_cols].astype(float).reset_index(drop=True)
        x_timestamp = pd.to_datetime(work["_dt"]).reset_index(drop=True)
        last_ts = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(
            pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=horizon, freq="D")
        )
        last_close = float(x_df["close"].iloc[-1])
        return x_df, x_timestamp, y_timestamp, last_close

    def _predict_close(self, x_df, x_ts, y_ts, horizon, sample_count) -> float:
        predictor = self._load()
        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=sample_count,
            verbose=False,
        )
        return float(pred["close"].iloc[-1])

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if len(df) < MIN_ROWS or "close" not in df.columns:
            return None
        built = self._build_inputs(df, horizon)
        if built is None:
            return None
        x_df, x_ts, y_ts, last_close = built

        if self.prob_samples and self.prob_samples > 1:
            # Empirical distribution: several single-sample stochastic paths.
            closes = np.array(
                [
                    self._predict_close(x_df, x_ts, y_ts, horizon, 1)
                    for _ in range(self.prob_samples)
                ]
            )
            return ForecastResult(
                last_close=last_close,
                point=float(closes.mean()),
                horizon=horizon,
                samples=closes,
            )

        # Fast path: one averaged forecast → point-based direction.
        point = self._predict_close(x_df, x_ts, y_ts, horizon, self.sample_count)
        return ForecastResult(last_close=last_close, point=point, horizon=horizon)
