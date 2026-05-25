import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import PriceChart from "../components/PriceChart";
import { s, Panel, Btn, pct } from "../components/ui";

const PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const CHART_PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const ALL_MODELS = [
  "k-NN", "k-NN (TW)", "k-NN Enhanced", "k-NN Enhanced (TW)",
  "LinReg", "LinReg (TW)", "LinReg Enhanced", "LinReg Enhanced (TW)",
  "LSTM",
];
const METRICS = [
  { key: "total_return", label: "Return", fmt: (v: number) => pct(v * 100), higher: true },
  { key: "accuracy", label: "Accuracy", fmt: (v: number) => pct(v * 100, false), higher: true },
  { key: "profit_factor", label: "PF", fmt: (v: number) => v >= 100 ? "∞" : v.toFixed(2), higher: true },
  { key: "sharpe_ratio", label: "Sharpe", fmt: (v: number) => v.toFixed(2), higher: true },
  { key: "sortino_ratio", label: "Sortino", fmt: (v: number) => v >= 100 ? "∞" : v.toFixed(2), higher: true },
  { key: "max_drawdown", label: "Max DD", fmt: (v: number) => pct(v * 100), higher: true }, // higher=closer to 0
  { key: "buy_hold_return", label: "B&H", fmt: (v: number) => pct(v * 100), higher: true },
];

const FILTER_GROUPS = [
  { label: "Family", tags: ["k-NN", "LinReg", "LSTM"] },
  { label: "Type", tags: ["Basic", "Enhanced"] },
  { label: "Variant", tags: ["TW", "No TW"] },
  { label: "Sentiment", tags: ["News", "No News"] },
  { label: "Period", tags: PERIODS.map((p) => p.toUpperCase()) },
];

function getTags(model: string, period: string): string[] {
  const t: string[] = [];
  if (model.startsWith("k-NN")) t.push("k-NN");
  if (model.startsWith("LinReg")) t.push("LinReg");
  if (model.startsWith("LSTM")) t.push("LSTM");
  if (model.includes("Enhanced")) t.push("Enhanced"); else t.push("Basic");
  if (model.includes("(TW)")) t.push("TW"); else t.push("No TW");
  if (model.includes("News")) t.push("News"); else t.push("No News");
  t.push(period.toUpperCase());
  return t;
}

interface BItem { model: string; period: string; news: boolean; fee: number; sl: number }
function bKey(r: BItem) { return `${r.model}|${r.period}|${r.news}|${r.fee}|${r.sl}`; }
type Res = Record<string, unknown>;

export default function Backtest() {
  const [selTickers, setSelTickers] = useState<Set<string>>(new Set(["AAPL"]));
  const [chartTicker, setChartTicker] = useState("AAPL");
  const [chartPeriod, setChartPeriod] = useState("1y");
  const [showChart, setShowChart] = useState(true);

  const [selModels, setSelModels] = useState<Set<string>>(new Set(ALL_MODELS));
  const [selPeriods, setSelPeriods] = useState<Set<string>>(new Set(["1y"]));
  const [selNews, setSelNews] = useState<"no" | "yes" | "both">("no");
  const [globalFee, setGlobalFee] = useState("0.05");
  const [globalSL, setGlobalSL] = useState("0");
  const [days, setDays] = useState("20");
  const [buyHold, setBuyHold] = useState(true);
  const [refreshData, setRefreshData] = useState(false);

  const [items, setItems] = useState<BItem[]>([]);
  const [buildSel, setBuildSel] = useState<Set<string>>(new Set());

  // Summary
  const [sumMetrics, setSumMetrics] = useState<Set<string>>(new Set(["total_return", "sharpe_ratio", "accuracy"]));
  const [sumTickers, setSumTickers] = useState<Set<string>>(new Set()); // empty = all

  // Results filter
  const [resSort, setResSort] = useState<{ col: string; asc: boolean }>({ col: "total_return", asc: false });
  const [resFilters, setResFilters] = useState<Set<string>>(new Set());
  const [resSearch, setResSearch] = useState("");
  const [newsTicker, setNewsTicker] = useState("all");

  const { data: tickers } = useQuery({ queryKey: ["tickers"], queryFn: api.getTickers });
  const { data: tickerData } = useQuery({
    queryKey: ["tickerData", chartTicker, chartPeriod],
    queryFn: () => api.getTickerData(chartTicker, chartPeriod),
    enabled: showChart && !!chartTicker,
  });
  const chartRows = useMemo(() => [...(tickerData?.data ?? [])].reverse(), [tickerData]);

  const stocks = tickers?.filter((t) => t.asset_type === "stock") ?? [];
  const crypto = tickers?.filter((t) => t.asset_type === "crypto") ?? [];
  const stockTickers = stocks.map((t) => t.ticker);
  const cryptoTickers = crypto.map((t) => t.ticker);

  const runMut = useMutation({
    mutationFn: () =>
      fetch("/api/backtest/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: [...selTickers], items: items.map((r) => ({ model: r.model, period: r.period, news: r.news, fee_pct: r.fee, stop_loss_pct: r.sl })),
          days: parseInt(days) || 20, buy_hold: buyHold, refresh_data: refreshData,
        }),
      }).then((r) => r.json()),
    onSuccess: () => setSumTickers(new Set()),
  });

  const results: Res[] = runMut.data?.results ?? [];
  const newsData: Record<string, string[]> = runMut.data?.news ?? {};

  // Ticker toggle helpers
  const toggleTicker = (t: string) => {
    const n = new Set(selTickers); if (n.has(t)) { if (n.size > 1) n.delete(t); } else n.add(t);
    setSelTickers(n); if (!n.has(chartTicker)) setChartTicker([...n][0]);
  };
  const setAllStocks = () => { const n = new Set(stockTickers); setSelTickers(n); setChartTicker(stockTickers[0]); };
  const setAllCrypto = () => { const n = new Set(cryptoTickers); setSelTickers(n); setChartTicker(cryptoTickers[0]); };
  const setAllTickers = () => { const n = new Set([...stockTickers, ...cryptoTickers]); setSelTickers(n); };

  // Builder
  const addSelected = () => {
    const nv: boolean[] = selNews === "both" ? [false, true] : selNews === "yes" ? [true] : [false];
    const fee = parseFloat(globalFee) || 0; const sl = parseFloat(globalSL) || 0;
    const existing = new Set(items.map(bKey)); const toAdd: BItem[] = [];
    for (const m of selModels) for (const p of selPeriods) for (const n of nv) {
      const row: BItem = { model: m, period: p, news: n, fee, sl };
      if (!existing.has(bKey(row))) { toAdd.push(row); existing.add(bKey(row)); }
    }
    if (toAdd.length > 0) setItems([...items, ...toAdd]);
  };
  const previewCount = useMemo(() => {
    const nv = selNews === "both" ? [false, true] : selNews === "yes" ? [true] : [false];
    const fee = parseFloat(globalFee) || 0; const sl = parseFloat(globalSL) || 0;
    const existing = new Set(items.map(bKey)); let c = 0;
    for (const m of selModels) for (const p of selPeriods) for (const n of nv)
      if (!existing.has(bKey({ model: m, period: p, news: n, fee, sl }))) c++;
    return c;
  }, [selModels, selPeriods, selNews, globalFee, globalSL, items]);

  const allResultTickers = useMemo(() => [...new Set(results.map((r) => String(r.ticker)))], [results]);

  // Filtered results (OR within group, AND between groups)
  const filteredResults = useMemo(() => {
    const activeByGroup: Map<string, string[]> = new Map();
    for (const g of FILTER_GROUPS) {
      const active = g.tags.filter((t) => resFilters.has(t));
      if (active.length > 0) activeByGroup.set(g.label, active);
    }
    // Ticker filters (not in FILTER_GROUPS)
    const tickerFilters = allResultTickers.filter((t) => resFilters.has(t.toUpperCase()));
    if (tickerFilters.length > 0) activeByGroup.set("Ticker", tickerFilters.map((t) => t.toUpperCase()));

    return results.filter((r) => {
      const tags = getTags(String(r.model ?? ""), String(r.period ?? ""));
      tags.push(String(r.ticker ?? "").toUpperCase());
      const matchFilters = activeByGroup.size === 0 || [...activeByGroup.values()].every((gt) => gt.some((t) => tags.includes(t)));
      const matchSearch = !resSearch || String(r.model ?? "").toLowerCase().includes(resSearch.toLowerCase()) ||
        String(r.ticker ?? "").toLowerCase().includes(resSearch.toLowerCase());
      return matchFilters && matchSearch;
    });
  }, [results, resFilters, resSearch, allResultTickers]);

  const sortedResults = useMemo(() => {
    return [...filteredResults].sort((a, b) => {
      const va = (a[resSort.col] as number) ?? 0; const vb = (b[resSort.col] as number) ?? 0;
      return resSort.asc ? va - vb : vb - va;
    });
  }, [filteredResults, resSort]);

  const handleResSort = (col: string) => setResSort((p) => p.col === col ? { col, asc: !p.asc } : { col, asc: false });
  const toggleResFilter = (f: string) => { const n = new Set(resFilters); if (n.has(f)) n.delete(f); else n.add(f); setResFilters(n); };

  // Summary with ties + ticker filter
  const summaryData = useMemo(() => {
    const pool = results.filter((r) => {
      if (r.error) return false;
      if (sumTickers.size === 0) return true;
      return sumTickers.has(String(r.ticker ?? ""));
    });
    if (pool.length === 0) return {};

    const out: Record<string, { models: { model: string; ticker: string; period: string; value: number }[] }> = {};
    for (const m of METRICS) {
      if (!sumMetrics.has(m.key)) continue;
      const vals = pool.map((r) => ({ model: String(r.model), ticker: String(r.ticker), period: String(r.period), value: (r[m.key] as number) ?? (m.key === "max_drawdown" ? -999 : -999) }));
      const bestVal = vals.reduce((best, v) => v.value > best ? v.value : best, -Infinity);
      const epsilon = Math.abs(bestVal) * 0.001 || 0.0001;
      const ties = vals.filter((v) => Math.abs(v.value - bestVal) < epsilon);
      out[m.key] = { models: ties };
    }
    return out;
  }, [results, sumMetrics, sumTickers]);

  const filteredNews = useMemo(() => {
    if (newsTicker === "all") return newsData;
    return newsTicker in newsData ? { [newsTicker]: newsData[newsTicker] } : {};
  }, [newsData, newsTicker]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>Backtest</h2>

      {/* Tickers */}
      <Panel title="Tickers">
        <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: s.muted, minWidth: 50 }}>Stocks</span>
            <Chip label="All Stocks" active={stockTickers.every((t) => selTickers.has(t))} onClick={setAllStocks} accent />
            {stocks.map((t) => <Chip key={t.ticker} label={t.ticker} active={selTickers.has(t.ticker)} onClick={() => toggleTicker(t.ticker)} />)}
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: s.muted, minWidth: 50 }}>Crypto</span>
            <Chip label="All Crypto" active={cryptoTickers.every((t) => selTickers.has(t))} onClick={setAllCrypto} accent />
            {crypto.map((t) => <Chip key={t.ticker} label={t.ticker} active={selTickers.has(t.ticker)} onClick={() => toggleTicker(t.ticker)} />)}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Chip label="All" active={selTickers.size === stockTickers.length + cryptoTickers.length} onClick={setAllTickers} accent />
            <span style={{ fontSize: 11, color: s.muted }}>{selTickers.size} selected</span>
          </div>
        </div>
      </Panel>

      {/* Chart */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Chk label="Chart" checked={showChart} onChange={setShowChart} />
        {showChart && (<>
          <select value={chartTicker} onChange={(e) => setChartTicker(e.target.value)} style={selSt}>
            {[...selTickers].map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <Pills values={CHART_PERIODS} selected={chartPeriod} onSelect={setChartPeriod} />
        </>)}
      </div>
      {showChart && chartRows.length > 0 && (
        <Panel title={`${chartTicker} — ${chartPeriod.toUpperCase()}`}>
          <PriceChart rows={chartRows} height={260} />
        </Panel>
      )}

      {/* Builder */}
      <Panel title="Backtest Builder">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <SelRow label="Models">
            <Chip label="All" active={selModels.size === ALL_MODELS.length} onClick={() => setSelModels(selModels.size === ALL_MODELS.length ? new Set() : new Set(ALL_MODELS))} accent />
            {ALL_MODELS.map((m) => <Chip key={m} label={m} active={selModels.has(m)} onClick={() => { const n = new Set(selModels); if (n.has(m)) n.delete(m); else n.add(m); setSelModels(n); }} />)}
          </SelRow>
          <SelRow label="Periods">
            <Chip label="All" active={selPeriods.size === PERIODS.length} onClick={() => setSelPeriods(selPeriods.size === PERIODS.length ? new Set() : new Set(PERIODS))} accent />
            {PERIODS.map((p) => <Chip key={p} label={p.toUpperCase()} active={selPeriods.has(p)} onClick={() => { const n = new Set(selPeriods); if (n.has(p)) n.delete(p); else n.add(p); setSelPeriods(n); }} />)}
          </SelRow>
          <SelRow label="News">
            {(["no", "yes", "both"] as const).map((v) => <Chip key={v} label={v === "no" ? "Without" : v === "yes" ? "With" : "Both"} active={selNews === v} onClick={() => setSelNews(v)} accent={selNews === v} />)}
          </SelRow>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <NumField label="Fee %" value={globalFee} onChange={setGlobalFee} width={60} />
            <NumField label="SL %" value={globalSL} onChange={setGlobalSL} width={60} />
            <NumField label="Days" value={days} onChange={setDays} width={50} />
            <Chk label="B&H benchmark" checked={buyHold} onChange={setBuyHold} />
            <Chk label="Update data" checked={refreshData} onChange={setRefreshData} />
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn onClick={addSelected} label={previewCount > 0 ? `Add ${previewCount}` : "All added"} secondary={previewCount === 0} />
            <span style={{ fontSize: 11, color: s.muted }}>{selModels.size}×{selPeriods.size}×{selNews === "both" ? 2 : 1}</span>
          </div>
          <div style={{ borderTop: `1px solid ${s.border}` }} />

          {items.length > 0 ? (<>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <Chk label={`All (${items.length})`} checked={buildSel.size === items.length} onChange={() => setBuildSel(buildSel.size === items.length ? new Set() : new Set(items.map(bKey)))} />
              {buildSel.size > 0 && <SmBtn label={`Remove (${buildSel.size})`} color={s.red} onClick={() => { setItems(items.filter((r) => !buildSel.has(bKey(r)))); setBuildSel(new Set()); }} />}
              <SmBtn label="Clear" color={s.muted} onClick={() => { setItems([]); setBuildSel(new Set()); }} />
              <span style={{ marginLeft: "auto", fontSize: 12, color: s.muted }}>{items.length} × {selTickers.size} = {items.length * selTickers.size} backtests</span>
            </div>
            <div style={{ maxHeight: 200, overflowY: "auto", border: `1px solid ${s.border}`, borderRadius: 6 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={thSt}></th><th style={{ ...thSt, textAlign: "left" }}>Model</th><th style={thSt}>Period</th><th style={thSt}>News</th><th style={thSt}>Fee</th><th style={thSt}>SL</th>
                </tr></thead>
                <tbody>{items.map((r, i) => { const k = bKey(r); return (
                  <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }} onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                    <td style={{ padding: "3px 6px", width: 24 }}><input type="checkbox" checked={buildSel.has(k)} onChange={() => { const n = new Set(buildSel); if (n.has(k)) n.delete(k); else n.add(k); setBuildSel(n); }} style={{ accentColor: s.accent }} /></td>
                    <td style={{ padding: "3px 6px", color: s.text }}>{r.model}</td>
                    <td style={{ padding: "3px 6px", color: s.muted, textAlign: "center" }}>{r.period.toUpperCase()}</td>
                    <td style={{ padding: "3px 6px", textAlign: "center", color: r.news ? s.green : s.muted }}>{r.news ? "✓" : "—"}</td>
                    <td style={{ padding: "3px 6px", textAlign: "center", color: s.muted }}>{r.fee}%</td>
                    <td style={{ padding: "3px 6px", textAlign: "center", color: r.sl > 0 ? s.text : s.muted }}>{r.sl > 0 ? `${r.sl}%` : "—"}</td>
                  </tr>);})}</tbody>
              </table>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <Btn onClick={() => runMut.mutate()} loading={runMut.isPending} label={runMut.isPending ? "Running..." : `Run ${items.length * selTickers.size}`} />
            </div>
          </>) : (<div style={{ textAlign: "center", padding: 16, color: s.muted, fontSize: 13 }}>Select models → Add → Run</div>)}
        </div>
      </Panel>

      {runMut.isPending && <div style={{ textAlign: "center", padding: 32, color: s.muted }}>Running backtests...</div>}

      {/* Summary */}
      {results.length > 0 && (
        <Panel title="Summary — Best Models">
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: s.muted }}>Metrics:</span>
              {METRICS.map((m) => <Chip key={m.key} label={m.label} active={sumMetrics.has(m.key)} onClick={() => { const n = new Set(sumMetrics); if (n.has(m.key)) n.delete(m.key); else n.add(m.key); setSumMetrics(n); }} />)}
            </div>
            {allResultTickers.length > 1 && (
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontSize: 11, color: s.muted }}>For tickers:</span>
                <Chip label="All" active={sumTickers.size === 0} onClick={() => setSumTickers(new Set())} accent />
                {allResultTickers.map((t) => <Chip key={t} label={t} active={sumTickers.has(t)} onClick={() => { const n = new Set(sumTickers); if (n.has(t)) n.delete(t); else n.add(t); setSumTickers(n); }} />)}
              </div>
            )}
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: s.mono, minWidth: 500 }}>
                <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={{ ...rTh, textAlign: "left" }}>Metric</th><th style={{ ...rTh, textAlign: "left" }}>Best Model(s)</th><th style={rTh}>Ticker</th><th style={rTh}>Period</th><th style={rTh}>Value</th>
                </tr></thead>
                <tbody>{METRICS.filter((m) => sumMetrics.has(m.key)).map((m) => {
                  const d = summaryData[m.key];
                  if (!d || d.models.length === 0) return null;
                  return d.models.map((best, j) => (
                    <tr key={`${m.key}-${j}`} style={{ borderBottom: `1px solid ${s.border}` }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                      {j === 0 && <td rowSpan={d.models.length} style={{ padding: "6px 10px", color: s.accent, fontWeight: 600, verticalAlign: "top" }}>{m.label}{d.models.length > 1 ? ` (${d.models.length} tied)` : ""}</td>}
                      <td style={{ padding: "6px 10px", color: s.text }}>{best.model}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: s.muted }}>{best.ticker}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: s.muted }}>{best.period}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", fontWeight: 700, color: m.key === "max_drawdown" ? s.red : best.value >= 0 ? s.green : s.red }}>{m.fmt(best.value)}</td>
                    </tr>
                  ));
                })}</tbody>
              </table>
            </div>
          </div>
        </Panel>
      )}

      {/* Results table */}
      {results.length > 0 && (
        <Panel title={`Results (${results.length})`}>
          <div style={{ padding: "8px 12px", borderBottom: `1px solid ${s.border}`, display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <input placeholder="Search..." value={resSearch} onChange={(e) => setResSearch(e.target.value)}
                style={{ padding: "4px 8px", borderRadius: 4, fontSize: 11, background: "transparent", border: `1px solid ${s.border}`, color: s.text, width: 140, outline: "none" }} />
              {resFilters.size > 0 && <SmBtn label="Clear filters" color={s.accent} onClick={() => setResFilters(new Set())} />}
              <span style={{ fontSize: 11, color: s.muted }}>{sortedResults.length}/{results.length}</span>
              <SmBtn label="Export CSV" color={s.muted} onClick={() => {
                const cols = ["model", "ticker", "period", "total_return", "accuracy", "profit_factor", "sharpe_ratio", "max_drawdown", "buy_hold_return", "fee_pct", "stop_loss_pct"];
                const h = cols.join(",") + "\n"; const csv = sortedResults.map((r) => cols.map((c) => r[c] ?? "").join(",")).join("\n");
                const b = new Blob([h + csv], { type: "text/csv" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `backtest_${days}d.csv`; a.click();
              }} />
            </div>
            {FILTER_GROUPS.map((g) => (
              <div key={g.label} style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontSize: 10, color: s.muted, minWidth: 55, textAlign: "right", paddingRight: 4 }}>{g.label}</span>
                {g.tags.map((t) => <Chip key={t} label={t} active={resFilters.has(t)} onClick={() => toggleResFilter(t)} />)}
              </div>
            ))}
            {allResultTickers.length > 1 && (
              <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontSize: 10, color: s.muted, minWidth: 55, textAlign: "right", paddingRight: 4 }}>Ticker</span>
                {allResultTickers.map((t) => <Chip key={t} label={t} active={resFilters.has(t.toUpperCase())} onClick={() => toggleResFilter(t.toUpperCase())} />)}
              </div>
            )}
          </div>
          <div style={{ maxHeight: 500, overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 11, minWidth: 900 }}>
              <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                <STh col="model" l="Model" al="left" sort={resSort} onSort={handleResSort} />
                <STh col="ticker" l="Ticker" sort={resSort} onSort={handleResSort} />
                <STh col="period" l="Per." sort={resSort} onSort={handleResSort} />
                <STh col="total_return" l="Return" sort={resSort} onSort={handleResSort} />
                <STh col="accuracy" l="Acc." sort={resSort} onSort={handleResSort} />
                <STh col="profit_factor" l="PF" sort={resSort} onSort={handleResSort} />
                <STh col="sharpe_ratio" l="Sharpe" sort={resSort} onSort={handleResSort} />
                <STh col="max_drawdown" l="DD" sort={resSort} onSort={handleResSort} />
                <STh col="buy_hold_return" l="B&H" sort={resSort} onSort={handleResSort} />
                <STh col="win_trades" l="W/L" sort={resSort} onSort={handleResSort} />
                <th style={rTh}>Beat?</th>
              </tr></thead>
              <tbody>{sortedResults.map((r, i) => {
                if (r.error) return <tr key={i} style={{ borderBottom: `1px solid ${s.border}`, opacity: 0.4 }}><td style={{ padding: "4px 8px" }} colSpan={11}>{String(r.model)} — {String(r.error)}</td></tr>;
                const ret = (r.total_return as number) ?? 0; const bh = (r.buy_hold_return as number) ?? 0; const beat = ret > bh;
                return (
                  <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }} onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                    <td style={{ padding: "4px 8px", color: s.text, whiteSpace: "nowrap" }}>{String(r.model)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{String(r.ticker)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{String(r.period)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: ret >= 0 ? s.green : s.red }}>{pct(ret * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center" }}>{pct((r.accuracy as number) * 100, false)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center" }}>{(r.profit_factor as number) >= 100 ? "∞" : (r.profit_factor as number).toFixed(2)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center" }}>{(r.sharpe_ratio as number).toFixed(2)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: s.red }}>{pct((r.max_drawdown as number) * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: bh >= 0 ? s.green : s.red }}>{pct(bh * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center" }}>{String(r.win_trades)}/{String(r.loss_trades)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: beat ? s.green : s.red }}>{beat ? "✓" : "✗"}</td>
                  </tr>);
              })}</tbody>
            </table>
          </div>
        </Panel>
      )}

      {/* News */}
      {Object.keys(newsData).length > 0 && (
        <Panel title="News Headlines">
          <div style={{ padding: "8px 12px", borderBottom: `1px solid ${s.border}`, display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 11, color: s.muted }}>Ticker:</span>
            <select value={newsTicker} onChange={(e) => setNewsTicker(e.target.value)} style={{ ...selSt, fontSize: 11 }}>
              <option value="all">All</option>
              {Object.keys(newsData).map((t) => <option key={t} value={t}>{t} ({newsData[t].length})</option>)}
            </select>
          </div>
          <div style={{ padding: 12, maxHeight: 200, overflowY: "auto" }}>
            {Object.entries(filteredNews).map(([tk, hl]) => (
              <div key={tk}>
                {Object.keys(filteredNews).length > 1 && <div style={{ fontSize: 11, fontWeight: 700, color: s.accent, marginTop: 4, marginBottom: 2 }}>{tk}</div>}
                {hl.map((h, i) => <div key={i} style={{ fontSize: 11, color: s.text, padding: "3px 0", borderBottom: `1px solid ${s.border}` }}>{h}</div>)}
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ---- Helpers ---- */
function SelRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (<div style={{ display: "flex", gap: 8, alignItems: "flex-start", flexWrap: "wrap" }}>
    <span style={{ fontSize: 12, color: s.muted, minWidth: 55, paddingTop: 4 }}>{label}</span>
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: 1 }}>{children}</div>
  </div>);
}
function Pills({ values, selected, onSelect }: { values: string[]; selected: string; onSelect: (v: string) => void }) {
  return (<div style={{ display: "flex", gap: 3 }}>{values.map((v) => (
    <button key={v} onClick={() => onSelect(v)} style={{ padding: "4px 10px", borderRadius: 5, fontSize: 11, fontWeight: 600, cursor: "pointer", border: `1px solid ${s.border}`, background: selected === v ? s.accent : "transparent", color: selected === v ? "#fff" : s.muted }}>{v.toUpperCase()}</button>
  ))}</div>);
}
function Chip({ label, active, onClick, accent }: { label: string; active: boolean; onClick: () => void; accent?: boolean }) {
  return (<button onClick={onClick} style={{ padding: "4px 10px", borderRadius: 5, fontSize: 11, fontWeight: 600, cursor: "pointer",
    border: `1px solid ${active ? (accent ? s.accent : "#475569") : s.border}`,
    background: active ? (accent ? "rgba(59,130,246,0.2)" : "rgba(71,85,105,0.2)") : "transparent",
    color: active ? (accent ? s.accent : s.text) : s.muted }}>{label}</button>);
}
function Chk({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (<label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", fontSize: 12, color: s.muted }}>
    <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ accentColor: s.accent }} />{label}</label>);
}
function SmBtn({ label, color, onClick }: { label: string; color: string; onClick: () => void }) {
  return (<button onClick={onClick} style={{ padding: "3px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, cursor: "pointer",
    border: `1px solid ${color}30`, background: `${color}10`, color }}>{label}</button>);
}
function NumField({ label, value, onChange, width }: { label: string; value: string; onChange: (v: string) => void; width: number }) {
  return (<div style={{ display: "flex", alignItems: "center", gap: 4 }}>
    <span style={{ fontSize: 11, color: s.muted }}>{label}</span>
    <input type="text" inputMode="decimal" value={value} onChange={(e) => onChange(e.target.value)}
      style={{ width, padding: "4px 6px", borderRadius: 4, fontSize: 12, fontFamily: s.mono, fontWeight: 600,
        background: "transparent", border: `1px solid ${s.border}`, color: s.accent, textAlign: "right", outline: "none" }} />
  </div>);
}
function STh({ col, l, al, sort, onSort }: { col: string; l: string; al?: string; sort: { col: string; asc: boolean }; onSort: (c: string) => void }) {
  const a = sort.col === col;
  return (<th onClick={() => onSort(col)} style={{ ...rTh, textAlign: (al ?? "center") as "left" | "center", cursor: "pointer", userSelect: "none", color: a ? s.text : s.muted }}>
    {l}{a && <span style={{ marginLeft: 3, opacity: 0.6 }}>{sort.asc ? "↑" : "↓"}</span>}</th>);
}
const selSt: React.CSSProperties = { padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600, background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer" };
const thSt: React.CSSProperties = { padding: "5px 6px", fontSize: 10, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
const rTh: React.CSSProperties = { padding: "5px 8px", fontSize: 10, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
