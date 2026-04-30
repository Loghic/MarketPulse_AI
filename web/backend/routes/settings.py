"""
routes/train.py – LSTM training and model inventory endpoints.
"""

import os
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks

from config import ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS
from web.backend.schemas import TrainRequest, TrainStatus, ModelInventoryItem
from web.backend.routes.data import get_api

router = APIRouter(prefix="/api/train", tags=["training"])

MODELS_DIR = Path("models")

# Track running training jobs
_training_status: dict[str, TrainStatus] = {}


@router.get("/models", response_model=list[ModelInventoryItem])
def list_models():
    """List all saved LSTM model files."""
    if not MODELS_DIR.exists():
        return []

    items = []
    for f in sorted(MODELS_DIR.glob("*.pt")):
        # Parse filename: {ticker}_{period}_{preset}.pt
        parts = f.stem.rsplit("_", 2)
        if len(parts) == 3:
            ticker, period, preset = parts
        else:
            ticker, period, preset = f.stem, "?", "?"

        stat = f.stat()
        items.append(ModelInventoryItem(
            ticker=ticker,
            period=period,
            preset=preset,
            filename=f.name,
            size_kb=round(stat.st_size / 1024, 1),
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))

    return items


@router.post("/start", response_model=TrainStatus)
def start_training(req: TrainRequest, bg: BackgroundTasks):
    """Start LSTM training in background."""
    key = f"{req.ticker}_{req.period}_{req.preset}"

    # Check if already running
    if key in _training_status and _training_status[key].status == "running":
        return _training_status[key]

    status = TrainStatus(
        ticker=req.ticker.upper(),
        period=req.period,
        preset=req.preset,
        status="running",
        message="Training started...",
    )
    _training_status[key] = status

    bg.add_task(_run_training, req, key)
    return status


def _run_training(req: TrainRequest, key: str):
    """Background training task."""
    try:
        from engine.ai_model import AIModel, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            _training_status[key].status = "error"
            _training_status[key].message = "PyTorch not installed"
            return

        api = get_api()
        ticker = req.ticker.upper()

        df = api.get_data(ticker, period=req.period)
        if df.empty:
            _training_status[key].status = "error"
            _training_status[key].message = f"No data for {ticker}"
            return

        model = AIModel(preset=req.preset)
        _training_status[key].total_epochs = model.config["epochs"]

        info = model.train(df, verbose=True)

        save_path = AIModel.model_path(ticker, req.period, req.preset)
        model.save(save_path)

        _training_status[key].status = "complete"
        _training_status[key].epoch = info["epochs_run"]
        _training_status[key].val_loss = info.get("best_val_loss", 0)
        _training_status[key].val_accuracy = info.get("final_val_accuracy", 0)
        _training_status[key].message = (
            f"Complete: {info['epochs_run']} epochs, "
            f"val_acc={info.get('final_val_accuracy', 0):.1%}"
        )

    except Exception as e:
        _training_status[key].status = "error"
        _training_status[key].message = str(e)


@router.get("/status/{key}", response_model=TrainStatus)
def get_training_status(key: str):
    """Check status of a training job."""
    if key not in _training_status:
        return TrainStatus(
            ticker="?", period="?", preset="?",
            status="not_found", message=f"No job: {key}"
        )
    return _training_status[key]
