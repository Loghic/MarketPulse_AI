"""One-off aggregator for the residual-hybrid paper: DM/Wilcoxon grid, residual
structure-vs-gain cross-tab, and bull/bear + vol-tercile regime slicing.

Consumes a forecast_harness.py output dir (results/fc_<scope>_<d>d_h<h>_<ts>/)
and prints/saves the tables the paper's Results section needs. Throwaway
research script, not part of the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.forecast_significance import compare_to_reference
from engine.macro_data import MacroCache, fetch_macro
from engine.regression_metrics import theil_u2
from engine.residual_diagnostics import structure_vs_gain


def load_run(run_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(run_dir.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dm_wilcoxon_grid(df: pd.DataFrame, ref_model: str = "Random Walk") -> pd.DataFrame:
    cases = []
    for (ticker, model), g in df.groupby(["ticker", "model"]):
        if model == ref_model:
            continue
        ref = df[(df.ticker == ticker) & (df.model == ref_model)]
        m = g.merge(ref[["date", "y_true", "y_pred"]], on="date", suffixes=("", "_ref"))
        if m.empty:
            continue
        e1 = m.y_true - m.y_pred
        e2 = m.y_true - m.y_pred_ref
        cases.append((f"{ticker}/{model}", e1.to_numpy(), e2.to_numpy()))
    if not cases:
        return pd.DataFrame()
    rows = compare_to_reference(cases)
    return pd.DataFrame(
        [
            dict(
                label=r.label,
                n=r.n,
                dm_stat=r.dm_stat,
                dm_p=r.dm_p,
                wilcoxon_p=r.wilcoxon_p,
                mean_diff=r.mean_diff,
                dm_significant=r.dm_significant,
            )
            for r in rows
        ]
    )


def residual_structure_cross_tab(
    df: pd.DataFrame, base_model: str, hybrid_model: str
) -> pd.DataFrame:
    cases = []
    for ticker, g in df[df.model == base_model].groupby("ticker"):
        base = g.sort_values("date")
        hyb = df[(df.ticker == ticker) & (df.model == hybrid_model)].sort_values("date")
        if base.empty or hyb.empty:
            continue
        resid = (base.y_true - base.y_pred).to_numpy()
        u2_base = theil_u2(base.y_true, base.y_pred, base.y_naive)
        u2_hybrid = theil_u2(hyb.y_true, hyb.y_pred, hyb.y_naive)
        cases.append((ticker, resid, u2_base, u2_hybrid))
    rows = structure_vs_gain(cases)
    return pd.DataFrame(
        [
            dict(
                ticker=r.ticker,
                n=r.n,
                acf1=r.acf1,
                ljung_box_stat=r.ljung_box_stat,
                ljung_box_p=r.ljung_box_p,
                structured=r.structured,
                u2_base=r.u2_base,
                u2_hybrid=r.u2_hybrid,
                gain=r.gain,
            )
            for r in rows
        ]
    )


def confidence_gating_table(
    df: pd.DataFrame, thresholds: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
) -> pd.DataFrame:
    """Coverage/accuracy at increasing confidence thresholds (plan.md #5).

    Only models with an ``interval_width`` (Prophet, ARIMA, the hybrid —
    XGBoost/LSTM-reg have no native interval and are silently absent, not
    faked). At each threshold, keeps the ``threshold``-fraction of steps with
    the *narrowest* interval (pooled across tickers) and reports Theil U2 on
    that acted-on subset — coverage=1.0 reproduces the ungated number.
    """
    rows = []
    if "interval_width" not in df.columns:
        return pd.DataFrame()
    for model, g in df.groupby("model"):
        g = g.dropna(subset=["interval_width"])
        if g.empty:
            continue
        g = g.sort_values("interval_width")
        n_total = len(g)
        for thr in thresholds:
            k = max(1, int(round(n_total * thr)))
            sub = g.iloc[:k]
            rows.append(
                dict(
                    model=model,
                    coverage=thr,
                    n=k,
                    n_total=n_total,
                    mean_interval_width=float(sub["interval_width"].mean()),
                    theil_u2=theil_u2(sub.y_true, sub.y_pred, sub.y_naive),
                )
            )
    return pd.DataFrame(rows)


def regime_labels() -> pd.DataFrame:
    """SPY 50/200d trend (bull/bear) + VIX tercile (high/low-vol) per date."""
    macro = MacroCache().load()
    if macro is None or macro.empty:
        macro = fetch_macro()
        MacroCache().save(macro)
    macro.index = pd.to_datetime(macro.index)
    import yfinance as yf

    spy = yf.download("SPY", period="10y", progress=False, auto_adjust=True)["Close"]
    spy = spy.squeeze()
    ma50 = spy.rolling(50).mean()
    ma200 = spy.rolling(200).mean()
    trend = pd.Series(np.where(ma50 > ma200, "bull", "bear"), index=spy.index)

    vix = macro["vix"] if "vix" in macro.columns else None
    if vix is not None:
        terciles = pd.qcut(vix.dropna(), 3, labels=["low-vol", "mid-vol", "high-vol"])
    else:
        terciles = pd.Series(dtype=object)

    out = pd.DataFrame({"trend": trend})
    out["vol"] = terciles.reindex(out.index)
    out.index.name = "date"
    return out.reset_index()


def per_asset_trend(
    tickers: list[str], min_ma_rows: int = 200, fallback_window: int = 20
) -> pd.DataFrame:
    """Each ticker's OWN 50/200-day MA-cross bull/bear label (full history via
    yfinance) — distinct from ``regime_labels()``'s single market-wide SPY
    label. Lets an idiosyncratic decline (e.g. one stock down while SPY is up)
    count as a real bear-regime observation instead of being masked by
    whatever the index is doing. Falls back to a rolling ``fallback_window``
    -day return-sign heuristic when a ticker has fewer than ``min_ma_rows`` of
    history to support a real 200-day MA (positive trailing return -> bull).
    """
    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            px = yf.download(ticker, period="10y", progress=False, auto_adjust=True)["Close"]
            px = px.squeeze()
        except Exception as e:  # noqa: BLE001 — one ticker's fetch shouldn't kill the rest
            print(f"per_asset_trend: {ticker} fetch failed ({e}); skipping.")
            continue
        if px.empty:
            continue
        if len(px) >= min_ma_rows:
            ma50 = px.rolling(50).mean()
            ma200 = px.rolling(200).mean()
            trend = pd.Series(np.where(ma50 > ma200, "bull", "bear"), index=px.index)
        else:
            roll_ret = px.pct_change(fallback_window)
            trend = pd.Series(np.where(roll_ret > 0, "bull", "bear"), index=px.index)
        for date, label in trend.dropna().items():
            rows.append((ticker, date, label))
    out = pd.DataFrame(rows, columns=["ticker", "date", "trend_asset"])
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    return out


def regime_table(
    df: pd.DataFrame, regimes: pd.DataFrame, asset_trend: pd.DataFrame | None = None
) -> pd.DataFrame:
    m = df.merge(regimes, on="date", how="left")
    if asset_trend is not None and not asset_trend.empty:
        m = m.merge(asset_trend, on=["ticker", "date"], how="left")
    rows = []
    regime_cols = [
        ("trend", ["bull", "bear"], "spy-wide"),
        ("vol", ["low-vol", "high-vol"], "spy-wide"),
    ]
    if asset_trend is not None and not asset_trend.empty:
        regime_cols.append(("trend_asset", ["bull", "bear"], "per-asset"))
    for regime_col, label_set, source in regime_cols:
        for label in label_set:
            sub = m[m[regime_col] == label]
            for model, g in sub.groupby("model"):
                if g.empty:
                    continue
                u2 = theil_u2(g.y_true, g.y_pred, g.y_naive)
                rows.append(dict(source=source, regime=label, model=model, n=len(g), theil_u2=u2))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--base-model", default="Prophet")
    ap.add_argument("--hybrid-model", default="Prophet+LSTM (hybrid)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = load_run(args.run_dir)
    out_dir = args.out or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = dm_wilcoxon_grid(df)
    dm.to_csv(out_dir / "_dm_wilcoxon.csv", index=False)
    print(f"DM/Wilcoxon grid: {len(dm)} rows -> {out_dir / '_dm_wilcoxon.csv'}")

    models = sorted(df.model.unique())
    print("Models present:", models)

    if args.base_model in models and args.hybrid_model in models:
        cross = residual_structure_cross_tab(df, args.base_model, args.hybrid_model)
        cross.to_csv(out_dir / "_residual_structure_gain.csv", index=False)
        print(
            f"Residual structure-vs-gain: {len(cross)} rows -> {out_dir / '_residual_structure_gain.csv'}"
        )
    else:
        print(
            f"Skipping structure-vs-gain: need {args.base_model!r} and {args.hybrid_model!r} in {models}"
        )

    gate = confidence_gating_table(df)
    if not gate.empty:
        gate.to_csv(out_dir / "_confidence_gating.csv", index=False)
        print(f"Confidence gating: {len(gate)} rows -> {out_dir / '_confidence_gating.csv'}")
    else:
        print("Confidence gating: no model reported an interval_width; skipped.")

    try:
        regimes = regime_labels()
        try:
            asset_trend = per_asset_trend(sorted(df.ticker.unique().tolist()))
        except Exception as e:  # noqa: BLE001 — per-asset trend is best-effort
            print(f"per_asset_trend failed ({e}); regime table will use spy-wide only.")
            asset_trend = None
        rt = regime_table(df, regimes, asset_trend)
        rt.to_csv(out_dir / "_regime_table.csv", index=False)
        print(f"Regime table: {len(rt)} rows -> {out_dir / '_regime_table.csv'}")
    except Exception as e:  # pragma: no cover - network/data best-effort
        print(f"Regime analysis failed: {e}")


if __name__ == "__main__":
    main()
