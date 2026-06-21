"""test_web_api.py – FastAPI backend endpoint tests.

Tests all routes with mocked StockAppAPI (no network, no real DB).
"""

from datetime import datetime
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_prices(days=400, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    n = len(dates)
    prices = 150.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.015, n))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.015, n))),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        },
        index=dates,
    ).rename_axis("Date")


def _make_news():
    return [
        {"content": {"title": "Stock rally continues with strong growth"}},
        {"content": {"title": "Market risk warning: debt levels rising"}},
    ]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create TestClient with mocked yfinance **and isolated DB**.

    The web backend lazily instantiates ``StockAppAPI()``, whose
    ``DatabaseManager`` writes to ``Path("data") / "market_data.db"`` —
    a path relative to CWD. Without chdir-to-tmp here, every
    ``POST /api/data/refresh`` (especially ``test_refresh_all`` with an
    empty ticker list) would persist the mocked synthetic fixture into
    the developer's real ``data/market_data.db`` under real ticker
    names. See scripts/clean_test_contamination.py for the cleanup of
    that historical mess.
    """
    tmp = tmp_path_factory.mktemp("web_api_db")
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)
    try:
        with (
            patch("engine.data_downloader.yf") as mock_dl,
            patch("engine.news_sources.yf") as mock_ns,
        ):
            mock_dl.Ticker.return_value.history.return_value = _make_prices()
            type(mock_ns.Ticker.return_value).news = PropertyMock(return_value=_make_news())

            # Reset shared API instance
            from web.backend.routes import data as data_module

            data_module._api = None

            from web.backend.app import app

            with TestClient(app) as c:
                yield c
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def refreshed_client(client):
    """Client with data pre-loaded via refresh."""
    client.post("/api/data/refresh", json={"tickers": ["TEST"]})
    return client


# ------------------------------------------------------------------
# Health & Root
# ------------------------------------------------------------------


class TestRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "MarketPulse AI"
        assert "endpoints" in data

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ------------------------------------------------------------------
# Data endpoints
# ------------------------------------------------------------------


class TestData:
    def test_list_tickers(self, client):
        r = client.get("/api/data/tickers")
        assert r.status_code == 200
        tickers = r.json()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "ticker" in tickers[0]
        assert "asset_type" in tickers[0]

    def test_refresh(self, client):
        r = client.post("/api/data/refresh", json={"tickers": ["TEST"]})
        assert r.status_code == 200
        results = r.json()
        assert isinstance(results, list)
        assert len(results) > 0
        # At least TEST should be refreshed
        test_result = next((x for x in results if x["ticker"] == "TEST"), None)
        assert test_result is not None
        assert test_result["rows"] > 0

    def test_refresh_all(self, client):
        r = client.post("/api/data/refresh", json={"tickers": []})
        assert r.status_code == 200
        results = r.json()
        assert len(results) > 0

    def test_refresh_news_dedicated_endpoint(self, client):
        """POST /api/data/refresh-news with explicit source/scorer/history knobs."""
        r = client.post(
            "/api/data/refresh-news",
            json={
                "tickers": ["TEST"],
                "sentiment_method": "naive",
                "news_source": ["yahoo"],
                "news_history_days": 7,
                "force_news": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 1
        row = data[0]
        assert row["ticker"] == "TEST"
        assert "headlines_pulled" in row
        assert "mean_sentiment" in row
        assert row["method"] == "naive"
        assert row["sources"] == ["yahoo"]

    def test_get_ticker_data(self, refreshed_client):
        r = refreshed_client.get("/api/data/ticker/TEST?period=1y&limit=100")
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "TEST"
        assert data["rows"] > 0
        assert len(data["data"]) > 0
        row = data["data"][0]
        assert "date" in row
        assert "open" in row
        assert "close" in row
        assert "volume" in row

    def test_get_ticker_data_periods(self, refreshed_client):
        """Different periods should return different row counts."""
        r1 = refreshed_client.get("/api/data/ticker/TEST?period=1y")
        r2 = refreshed_client.get("/api/data/ticker/TEST?period=max")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # max should have >= 1y rows
        assert r2.json()["rows"] >= r1.json()["rows"]

    def test_get_ticker_not_found(self, client):
        r = client.get("/api/data/ticker/NONEXISTENT")
        assert r.status_code == 404

    def test_ticker_data_sorted_descending(self, refreshed_client):
        r = refreshed_client.get("/api/data/ticker/TEST?period=1y&limit=10")
        data = r.json()["data"]
        dates = [row["date"] for row in data]
        assert dates == sorted(dates, reverse=True)


# ------------------------------------------------------------------
# Predict endpoints
# ------------------------------------------------------------------


class TestPredict:
    def test_predict_single(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [{"model": "k-NN", "period": "1y", "news": False}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert "consensus" in data
        preds = data["predictions"]
        assert len(preds) >= 1
        p = preds[0]
        assert p["ticker"] == "TEST"
        assert p["prediction"] in ("UP", "DOWN", "N/A")

    def test_predict_multiple_models(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": "k-NN", "period": "1y", "news": False},
                    {"model": "k-NN (TW)", "period": "1y", "news": False},
                ],
            },
        )
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) >= 1
        labels = {p["model"] for p in preds}
        assert any("k-NN" in label for label in labels)

    def test_predict_all_models(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": m, "period": "1y", "news": False}
                    for m in ["k-NN", "k-NN (TW)", "LinReg", "LSTM"]
                ],
            },
        )
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) >= 2

    def test_predict_with_news(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [{"model": "k-NN", "period": "1y", "news": True}],
            },
        )
        assert r.status_code == 200

    def test_consensus_in_response(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": "k-NN", "period": "1y", "news": False},
                    {"model": "k-NN (TW)", "period": "1y", "news": False},
                ],
            },
        )
        assert r.status_code == 200
        cons = r.json()["consensus"]
        assert cons["direction"] in ("UP", "DOWN", "SPLIT")
        assert cons["up"] + cons["down"] == cons["total"]
        assert 0 <= cons["agreement"] <= 1


# ------------------------------------------------------------------
# Backtest endpoints
# ------------------------------------------------------------------


class TestBacktest:
    """
    The 2026 backend exposes a single endpoint, ``POST /api/backtest``,
    that takes one shared ``fee_pct`` / ``stop_loss_pct`` per request and
    fans out across every model variant produced by
    ``engine.backtest_helpers.run_single_backtest``. The response uses
    ``BacktestResponse`` (``results`` list + ``best_by_return`` and
    ``best_by_sharpe``); there is no per-item config and no ``summary``
    key — the "summary" concept is the two best-of pointers.
    """

    @pytest.fixture(autouse=True)
    def _redirect_persistence(self, tmp_path, monkeypatch):
        """
        Every test in this class triggers POST /api/backtest, which writes
        cache JSON + results CSVs to disk. Redirect both directories to
        tmp_path so the developer's real ``backtests/`` and ``results/``
        trees aren't polluted by test runs (those artifacts would then
        show up in the Analysis tab picker).
        """
        from pathlib import Path

        from web.backend.routes import backtest as _bt

        monkeypatch.setattr(_bt, "CACHE_DIR", Path(str(tmp_path / "backtests")))
        monkeypatch.setattr(_bt, "RESULTS_DIR", Path(str(tmp_path / "results")))
        yield

    @staticmethod
    def _body(**overrides) -> dict:
        """Default backtest request body — override per-test."""
        body = {
            "tickers": ["TEST"],
            "days": 5,
            "period": "1y",
            "fee_pct": 0.0,
            "stop_loss_pct": 0.0,
            "compare_periods": False,
            "buy_hold": False,
            "refresh_data": False,
        }
        body.update(overrides)
        return body

    def test_backtest_basic(self, refreshed_client):
        r = refreshed_client.post("/api/backtest", json=self._body(fee_pct=0.03))
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "best_by_return" in data
        assert "best_by_sharpe" in data
        results = data["results"]
        assert len(results) > 0
        result = results[0]
        assert result["ticker"] == "TEST"
        assert "accuracy" in result
        assert "total_return" in result

    def test_backtest_has_daily_data(self, refreshed_client):
        r = refreshed_client.post("/api/backtest", json=self._body())
        result = r.json()["results"][0]
        assert len(result.get("days", [])) == 5
        # Each day must expose the sentiment_score field (the per-day
        # value the engine fed into model.predict()). For the default
        # request body the model variants this row covers are all
        # price-only, so sentiment is 0.0 everywhere.
        for d in result["days"]:
            assert "sentiment_score" in d
            assert isinstance(d["sentiment_score"], int | float)

    def test_backtest_with_stop_loss(self, refreshed_client):
        """With SL>0, run_single_backtest runs each model twice (baseline + SL variant)."""
        r = refreshed_client.post("/api/backtest", json=self._body(stop_loss_pct=2.0))
        assert r.status_code == 200
        results = r.json()["results"]
        names = [item["model"] for item in results]
        # Baseline rows (no SL suffix) AND SL rows must both appear.
        assert any("SL" in n for n in names)
        assert any("SL" not in n for n in names)

    def test_backtest_with_fees(self, refreshed_client):
        """Higher fees must reduce realised return for at least one matching model."""
        r_no = refreshed_client.post("/api/backtest", json=self._body(fee_pct=0.0))
        r_fee = refreshed_client.post("/api/backtest", json=self._body(fee_pct=0.5))
        no_results = {row["model"]: row["total_return"] for row in r_no.json()["results"]}
        fee_results = {row["model"]: row["total_return"] for row in r_fee.json()["results"]}
        # Every shared model should have a lower (or equal) return under fees.
        # Assert it's strictly lower for at least one variant — sanity check
        # that the fee parameter is actually being honoured by the endpoint.
        shared = set(no_results) & set(fee_results)
        assert shared, "expected overlap between no-fee and with-fee runs"
        assert any(fee_results[m] < no_results[m] for m in shared)

    def test_backtest_best_by_return_and_sharpe(self, refreshed_client):
        """The 'summary' concept is now best_by_return and best_by_sharpe."""
        r = refreshed_client.post("/api/backtest", json=self._body())
        data = r.json()
        assert "best_by_return" in data
        assert "best_by_sharpe" in data
        best_ret = data["best_by_return"]
        best_sharpe = data["best_by_sharpe"]
        # Both must be populated when there are results.
        assert best_ret is not None
        assert best_sharpe is not None
        assert "total_return" in best_ret
        assert "sharpe_ratio" in best_sharpe
        # Sanity: best_by_return really IS the row with the highest return.
        all_returns = [row["total_return"] for row in data["results"]]
        assert best_ret["total_return"] == max(all_returns)

    def test_backtest_compare_periods(self, refreshed_client):
        """compare_periods=True fans out across every period in ALL_PERIODS."""
        r = refreshed_client.post("/api/backtest", json=self._body(compare_periods=True))
        assert r.status_code == 200
        results = r.json()["results"]
        # Each period × N model variants. With 400 days of mock data, at least
        # the short periods (1mo, 1y) should yield rows.
        periods = {row["period"] for row in results}
        assert len(periods) >= 2, f"expected results from multiple periods, got {periods}"
        assert len(results) >= 4  # at minimum a few model variants per period

    def test_backtest_progress_endpoint(self, client):
        """``GET /api/backtest/progress`` returns the running flag even between runs."""
        r = client.get("/api/backtest/progress")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data

    def test_backtest_persists_to_disk(self, refreshed_client, tmp_path):
        """A successful backtest writes JSON cache + results/ CSVs.

        The autouse ``_redirect_persistence`` fixture already pointed
        ``CACHE_DIR`` + ``RESULTS_DIR`` at ``tmp_path``, so this just
        exercises that the write actually happens.
        """
        r = refreshed_client.post("/api/backtest", json=self._body(fee_pct=0.03))
        assert r.status_code == 200

        # JSON cache should exist with at least one file
        cache_files = list((tmp_path / "backtests").glob("*.json"))
        assert len(cache_files) >= 1, "expected a backtest JSON cache file"

        # CSV export directory should exist with at least one ticker CSV
        results_subdirs = (
            list((tmp_path / "results").iterdir()) if (tmp_path / "results").exists() else []
        )
        assert len(results_subdirs) == 1, "expected exactly one timestamped results subdir"
        sub = results_subdirs[0]
        # Subdir name encodes scope_days_fee + timestamp
        assert "20d" in sub.name or sub.name.startswith("custom")
        assert any(f.suffix == ".csv" for f in sub.iterdir()), "no CSV in results subdir"
        # _summary.csv should be present
        assert (sub / "_summary.csv").exists()

    def test_backtest_runs_listing(self, refreshed_client):
        """After a backtest, ``GET /api/backtest/runs`` shows it; loading round-trips."""
        refreshed_client.post("/api/backtest", json=self._body())

        r = refreshed_client.get("/api/backtest/runs")
        assert r.status_code == 200
        runs = r.json()
        assert len(runs) >= 1
        first = runs[0]
        assert "run_id" in first
        assert "saved_at" in first
        assert "results_dir" in first
        assert first["result_count"] > 0

        # Load that run back
        r2 = refreshed_client.get(f"/api/backtest/runs/{first['run_id']}")
        assert r2.status_code == 200
        loaded = r2.json()
        assert loaded["run_id"] == first["run_id"]
        assert loaded["response"]["results"]

        # Path traversal blocked — the run_id contains `..` but isn't a literal
        # parent path (httpx would normalize that away), so the handler sees it.
        r3 = refreshed_client.get("/api/backtest/runs/..%2Fetc")
        assert r3.status_code in (400, 404)
        r4 = refreshed_client.get("/api/backtest/runs/foo..bar")
        assert r4.status_code == 400


# ------------------------------------------------------------------
# Training endpoints
# ------------------------------------------------------------------


class TestTraining:
    def test_list_models_empty(self, client):
        r = client.get("/api/train/models")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_training_status_not_found(self, client):
        r = client.get("/api/train/status/nonexistent_key")
        assert r.status_code == 200
        assert r.json()["status"] == "not_found"


# ------------------------------------------------------------------
# Settings endpoints
# ------------------------------------------------------------------


class TestSettings:
    def test_get_defaults(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        s = r.json()
        assert "default_period" in s
        assert "knn_k" in s
        assert "default_fee_pct" in s
        assert s["knn_k"] == 5

    def test_patch_settings(self, client):
        r = client.patch("/api/settings", json={"knn_k": 8})
        assert r.status_code == 200
        assert r.json()["knn_k"] == 8

        # Verify persisted
        r2 = client.get("/api/settings")
        assert r2.json()["knn_k"] == 8

        # Reset
        client.patch("/api/settings", json={"knn_k": 5})

    def test_put_settings(self, client):
        r = client.get("/api/settings")
        original = r.json()

        # Full replace
        modified = {**original, "default_fee_pct": 0.1}
        r2 = client.put("/api/settings", json=modified)
        assert r2.status_code == 200
        assert r2.json()["default_fee_pct"] == 0.1

        # Restore
        client.put("/api/settings", json=original)


# ------------------------------------------------------------------
# Analysis endpoints
# ------------------------------------------------------------------


class TestAnalysis:
    def test_news_comparison(self, refreshed_client):
        r = refreshed_client.post(
            "/api/analysis/news-comparison",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "fee_pct": 0.03,
            },
        )
        assert r.status_code == 200
        results = r.json()
        assert isinstance(results, list)
        if len(results) > 0:
            row = results[0]
            assert "ticker" in row
            assert "model" in row
            assert "return_no_news" in row
            assert "return_with_news" in row
            assert "diff" in row
            assert "sharpe_no_news" in row
            assert "accuracy_no_news" in row

    def test_results_dirs_empty_ok(self, client):
        """The directory browser must never crash when results/ is missing."""
        r = client.get("/api/analysis/results-dirs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_result_csv_404_on_missing(self, client):
        """Bad dir / file should return 404, not 500."""
        r = client.get("/api/analysis/result-csv?dir=does_not_exist&file=AAPL")
        assert r.status_code == 404

    def test_result_csv_path_traversal_blocked(self, client):
        """Path traversal attempts must be rejected with 400."""
        r = client.get("/api/analysis/result-csv?dir=..&file=etc")
        assert r.status_code == 400

    def test_result_csv_roundtrip(self, client, tmp_path, monkeypatch):
        """Write a fake CSV under results/ and confirm the endpoint serves it."""
        import csv as _csv
        from pathlib import Path

        from web.backend.routes import analysis as _analysis_module

        # Redirect RESULTS_DIR to tmp_path/results for this test
        results = tmp_path / "results"
        sub = results / "stocks_5d_fee003_bh"
        sub.mkdir(parents=True)
        with open(sub / "AAPL.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["ticker", "period", "model", "total_return"])
            w.writeheader()
            w.writerow({"ticker": "AAPL", "period": "1y", "model": "k-NN", "total_return": "0.05"})

        monkeypatch.setattr(_analysis_module, "RESULTS_DIR", Path(str(results)))

        r = client.get("/api/analysis/results-dirs")
        assert r.status_code == 200
        names = [d["name"] for d in r.json()]
        assert "stocks_5d_fee003_bh" in names

        r = client.get("/api/analysis/result-csv?dir=stocks_5d_fee003_bh&file=AAPL")
        assert r.status_code == 200
        data = r.json()
        assert data["row_count"] == 1
        assert data["rows"][0]["ticker"] == "AAPL"
        assert data["rows"][0]["total_return"] == "0.05"


# ------------------------------------------------------------------
# Per-ticker cached prediction endpoint
# ------------------------------------------------------------------


class TestPredictCacheByTicker:
    def test_cached_empty(self, client):
        """No cache yet → empty payload, never 404."""
        r = client.get("/api/predict/cached/NEVER_FETCHED")
        assert r.status_code == 200
        data = r.json()
        assert data["predictions"] == []
        assert data["consensus"] is None

    def test_cached_after_run(self, refreshed_client):
        """After a /predict/run the per-ticker cache returns the rows + timestamp."""
        refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [{"model": "k-NN", "period": "1y", "news": False}],
                "refresh_data": False,
            },
        )
        r = refreshed_client.get("/api/predict/cached/TEST")
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "TEST"
        assert len(data["predictions"]) > 0
        assert "cached_at" in data
        # ISO-ish timestamp with seconds
        assert "T" in data["cached_at"]


# ------------------------------------------------------------------
# Meta — config-driven options for the frontend
# ------------------------------------------------------------------


class TestMeta:
    def test_meta_shape(self, client):
        r = client.get("/api/meta")
        assert r.status_code == 200
        data = r.json()
        for key in (
            "model_families",
            "asset_classes",
            "benchmarks",
            "periods",
            "sentiment_methods",
            "sl_sweep",
            "confidence_sweep",
            "defaults",
        ):
            assert key in data

    def test_model_families_include_all_and_gating(self, client):
        data = client.get("/api/meta").json()
        keys = {f["key"] for f in data["model_families"]}
        # Every config family is reported.
        assert {"knn", "linreg", "lstm", "prophet", "chronos", "kronos", "baseline"} <= keys
        for f in data["model_families"]:
            assert isinstance(f["available"], bool)
            assert isinstance(f["predict"], bool)
        # k-NN / LinReg are always available + predict-capable.
        by_key = {f["key"]: f for f in data["model_families"]}
        assert by_key["knn"]["available"] and by_key["knn"]["predict"]
        # Baselines are backtest-only.
        assert by_key["baseline"]["predict"] is False

    def test_asset_classes_include_new_classes(self, client):
        data = client.get("/api/meta").json()
        keys = {ac["key"] for ac in data["asset_classes"]}
        # The new (post-2026) classes must surface, not just stock/crypto.
        assert {"stock", "crypto", "commodity", "index", "fx"} <= keys

    def test_sl_and_confidence_sweeps(self, client):
        data = client.get("/api/meta").json()
        assert data["sl_sweep"] == [0.0, 5.0, 10.0, 15.0]
        assert data["confidence_sweep"][0] == 0.0


# ------------------------------------------------------------------
# Tickers carry registry asset_type (commodities/indices/fx)
# ------------------------------------------------------------------


class TestTickerAssetTypes:
    def test_tickers_use_registry_classes(self, client):
        rows = client.get("/api/data/tickers").json()
        types = {row["asset_type"] for row in rows}
        # The "-USD"→crypto heuristic is gone; registry classes appear.
        assert "stock" in types
        # At least one of the new ETF-proxy classes should be tagged.
        assert types & {"commodity", "index", "fx"}


# ------------------------------------------------------------------
# Backtest — new Phase 1.3 / 2.1 / 2.2 args
# ------------------------------------------------------------------


class TestBacktestNewArgs:
    @pytest.fixture(autouse=True)
    def _redirect_persistence(self, tmp_path, monkeypatch):
        from pathlib import Path

        from web.backend.routes import backtest as _bt

        monkeypatch.setattr(_bt, "CACHE_DIR", Path(str(tmp_path / "backtests")))
        monkeypatch.setattr(_bt, "RESULTS_DIR", Path(str(tmp_path / "results")))
        yield

    @staticmethod
    def _body(**overrides) -> dict:
        body = {
            "tickers": ["TEST"],
            "days": 5,
            "period": "1y",
            "fee_pct": 0.0,
            "buy_hold": False,
            "refresh_data": False,
            "include_baselines": True,
        }
        body.update(overrides)
        return body

    def test_result_carries_new_fields(self, refreshed_client):
        r = refreshed_client.post("/api/backtest", json=self._body())
        assert r.status_code == 200
        result = r.json()["results"][0]
        for key in (
            "min_confidence",
            "coverage",
            "turnover_fees",
            "hold_days",
            "turnover_count",
            "fees_paid",
        ):
            assert key in result

    def test_min_confidence_gate(self, refreshed_client):
        r = refreshed_client.post("/api/backtest", json=self._body(min_confidence=0.6))
        assert r.status_code == 200
        for res in r.json()["results"]:
            assert res["min_confidence"] == 0.6
            assert 0.0 <= res["coverage"] <= 1.0

    def test_turnover_fees_flag(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest", json=self._body(fee_pct=0.05, turnover_fees=True, hold_days=3)
        )
        assert r.status_code == 200
        res = r.json()["results"][0]
        assert res["turnover_fees"] is True
        assert res["hold_days"] == 3

    def test_sl_sweep_produces_multiple_levels(self, refreshed_client):
        r = refreshed_client.post("/api/backtest", json=self._body(sl_sweep=True))
        assert r.status_code == 200
        levels = {res["stop_loss_pct"] for res in r.json()["results"]}
        # The sweep includes the no-SL baseline (0) plus non-zero levels.
        assert 0.0 in levels
        assert len(levels) > 1

    def test_models_filter(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest", json=self._body(models=["knn"], include_baselines=False)
        )
        assert r.status_code == 200
        names = [res["model"] for res in r.json()["results"]]
        assert names  # got something
        assert all(n.startswith("k-NN") for n in names)


# ------------------------------------------------------------------
# OOS harness endpoints
# ------------------------------------------------------------------


class TestOOS:
    @pytest.fixture(autouse=True)
    def _redirect_persistence(self, tmp_path, monkeypatch):
        from pathlib import Path

        from web.backend.routes import oos as _oos

        monkeypatch.setattr(_oos, "CACHE_DIR", Path(str(tmp_path / "oos_runs")))
        monkeypatch.setattr(_oos, "RESULTS_DIR", Path(str(tmp_path / "results")))
        yield

    @staticmethod
    def _body(**overrides) -> dict:
        # baseline-only candidates keep the run instant (no LSTM/forecast load)
        body = {
            "tickers": ["AAPL"],
            "days": 50,
            "periods": ["max"],
            "fee_pct": 0.0,
            "buy_hold": True,
            "refresh_data": False,
            "models": ["baseline"],
            "include_baselines": True,
        }
        body.update(overrides)
        return body

    def test_oos_run_shape(self, refreshed_client):
        # AAPL resolves to the mocked 400-row series → enough for 2 disjoint windows.
        r = refreshed_client.post("/api/oos", json=self._body())
        assert r.status_code == 200
        data = r.json()
        assert "rows" in data and "summary" in data
        assert data["run_id"]
        assert data["summary"]["tickers"] == len(data["rows"])
        if data["rows"]:
            row = data["rows"][0]
            for key in ("winner_model", "oos_return", "beats_bh_oos", "oos_coverage"):
                assert key in row

    def test_oos_persisted_and_listed(self, refreshed_client):
        run = refreshed_client.post("/api/oos", json=self._body()).json()
        run_id = run["run_id"]
        runs = refreshed_client.get("/api/oos/runs").json()
        assert any(x["run_id"] == run_id for x in runs)
        loaded = refreshed_client.get(f"/api/oos/runs/{run_id}").json()
        assert loaded["run_id"] == run_id
        assert loaded["response"]["summary"]["tickers"] == run["summary"]["tickers"]

    def test_oos_progress_endpoint(self, refreshed_client):
        r = refreshed_client.get("/api/oos/progress")
        assert r.status_code == 200
        assert "running" in r.json()

    def test_oos_gate_columns(self, refreshed_client):
        r = refreshed_client.post("/api/oos", json=self._body(min_confidence=0.55))
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["min_confidence"] == 0.55


# ------------------------------------------------------------------
# Concept docs (Help tab)
# ------------------------------------------------------------------


class TestDocs:
    def test_list_docs(self, client):
        r = client.get("/api/docs")
        assert r.status_code == 200
        docs = r.json()
        slugs = {d["slug"] for d in docs}
        # The core glossary pages must be present and titled.
        assert {"getting-started", "models", "strategy", "metrics", "oos"} <= slugs
        for d in docs:
            assert d["title"]

    def test_get_doc_markdown(self, client):
        r = client.get("/api/docs/strategy")
        assert r.status_code == 200
        data = r.json()
        assert data["slug"] == "strategy"
        assert "# Strategy" in data["markdown"]
        # A section the Backtest tab deep-links into must exist.
        assert "## Stop-loss" in data["markdown"]

    def test_unknown_doc_404(self, client):
        r = client.get("/api/docs/does-not-exist")
        assert r.status_code == 404

    def test_no_path_traversal(self, client):
        # A traversal-style slug must not escape the docs dir.
        r = client.get("/api/docs/..%2f..%2fconfig")
        assert r.status_code in (404, 400)
