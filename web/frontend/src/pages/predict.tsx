import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import PriceChart from "../components/PriceChart";
import { s, Panel, Btn, pct } from "../components/ui";

const CHART_PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const PERIODS = ["1mo", "1y", "2y", "5y", "max"];

const ALL_MODELS = [
  "k-NN", "k-NN (TW)", "k-NN Enhanced", "k-NN Enhanced (TW)",
  "LinReg", "LinReg (TW)", "LinReg Enhanced", "LinReg Enhanced (TW)",
  "LSTM",
];

const FILTER_GROUPS = [
  { label: "Family", tags: ["k-NN", "LinReg", "LSTM"] },
  { label: "Type", tags: ["Basic", "Enhanced"] },
  { label: "Variant", tags: ["TW", "No TW"] },
  { label: "Sentiment", tags: ["News", "No News"] },
  { label: "Period", tags: PERIODS.map((p) => p.toUpperCase()) },
];

interface BuilderRow { model: string; period: string; news: boolean }
function bKey(r: BuilderRow): string { return `${r.model}|${r.period}|${r.news}`; }

interface Pred {
  model: string; prediction: string; confidence: number;
  period: string; sentiment_score: number; headlines: string[];
  [key: string]: unknown;
}

// Tags for filtering
function getTags(m: string): string[] {
  const tags: string[] = [];
  if (m.startsWith("k-NN")) tags.push("k-NN");
  if (m.startsWith("LinReg")) tags.push("LinReg");
  if (m.startsWith("LSTM")) tags.push("LSTM");
  if (m.includes("Enhanced")) tags.push("Enhanced");
  if (m.includes("(TW)")) tags.push("TW");
  if (m.includes("News")) tags.push("News");
  if (!m.includes("Enhanced")) tags.push("Basic");
  if (!m.includes("(TW)")) tags.push("No TW");
  if (!m.includes("News")) tags.push("No News");
  return tags;
}

export default function Predict() {
  const [ticker, setTicker] = useState("AAPL");
  const [chartPeriod, setChartPeriod] = useState("1y");
  const [showChart, setShowChart] = useState(false);
  const [refreshData, setRefreshData] = useState(true);
  const [histDate, setHistDate] = useState("");
  const [histPeriod, setHistPeriod] = useState("1y");

  // Builder
  const [items, setItems] = useState<BuilderRow[]>([]);
  const [buildSel, setBuildSel] = useState<Set<string>>(new Set());
  const [selModels, setSelModels] = useState<Set<string>>(new Set(ALL_MODELS));
  const [selPeriods, setSelPeriods] = useState<Set<string>>(new Set(["1y"]));
  const [selNews, setSelNews] = useState<"no" | "yes" | "both">("no");

  // Consensus
  const [consChecked, setConsChecked] = useState<Set<number>>(new Set());
  const [consFilters, setConsFilters] = useState<Set<string>>(new Set());
  const [consSearch, setConsSearch] = useState("");
  const [resSort, setResSort] = useState<{ col: string; asc: boolean }>({ col: "confidence", asc: false });

  const queryClient = useQueryClient();

  const { data: tickers } = useQuery({ queryKey: ["tickers"], queryFn: api.getTickers });
  const { data: predictInfo } = useQuery({
    queryKey: ["predictInfo"],
    queryFn: () => fetch("/api/predict/info").then((r) => r.json()),
  });
  const { data: tickerData } = useQuery({
    queryKey: ["tickerData", ticker, chartPeriod],
    queryFn: () => api.getTickerData(ticker, chartPeriod),
    enabled: !!ticker && showChart,
  });

  // Per-ticker cached prediction loader.
  // The backend writes predictions/{ticker}/{date}.json after every run; this
  // query reads back the most recent file so switching tabs or reloading the
  // page redisplays the latest results instead of forcing a re-run. Each
  // ticker gets its own cache, so AAPL can be 10 days old while MSFT is
  // 5 minutes old.
  const { data: cachedPred } = useQuery({
    queryKey: ["cachedPrediction", ticker],
    queryFn: () => api.cachedForTicker(ticker),
    enabled: !!ticker,
    staleTime: 5_000, // refetch on tab return after a short delay
  });
  const chartRows = useMemo(() => [...(tickerData?.data ?? [])].reverse(), [tickerData]);
  const nextDay = predictInfo?.next_trading_day ?? "—";
  const isCrypto = ticker.includes("-USD");

  const runMut = useMutation({
    mutationFn: () =>
      fetch("/api/predict/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, items: items.map((r) => ({ model: r.model, period: r.period, news: r.news })), refresh_data: refreshData }),
      }).then((r) => r.json()),
    onSuccess: (data) => {
      const valid = new Set<number>();
      (data.predictions as Pred[]).forEach((p, i) => { if (p.prediction === "UP" || p.prediction === "DOWN") valid.add(i); });
      setConsChecked(valid);
      setConsFilters(new Set());
      setConsSearch("");
      // Invalidate the per-ticker cache so the badge picks up the new timestamp
      queryClient.invalidateQueries({ queryKey: ["cachedPrediction", ticker] });
    },
  });

  // When ticker changes, reset the in-memory mutation result so the next
  // page render shows the freshly-loaded cache for this ticker instead of
  // the previous ticker's run.
  useEffect(() => {
    runMut.reset();
    setConsChecked(new Set());
    setConsFilters(new Set());
    setConsSearch("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  // When the cached prediction set arrives (or refreshes), pre-select all
  // valid rows for consensus — same behaviour as after a fresh run, so the
  // consensus widget works on tab return without the user having to tick
  // every box again.
  useEffect(() => {
    if (runMut.data) return; // a fresh run already populated consChecked
    if (!cachedPred?.predictions?.length) return;
    const valid = new Set<number>();
    cachedPred.predictions.forEach((p, i) => {
      if (p.prediction === "UP" || p.prediction === "DOWN") valid.add(i);
    });
    setConsChecked(valid);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cachedPred]);

  const histMut = useMutation({
    mutationFn: () =>
      fetch(`/api/predict/historical?ticker=${ticker}&date=${histDate}&period=${histPeriod}`, { method: "POST" }).then((r) => r.json()),
  });

  // Show fresh-run results first; otherwise fall back to whatever the
  // backend has cached for this ticker.
  const predictions: Pred[] =
    (runMut.data?.predictions as Pred[] | undefined) ??
    (cachedPred?.predictions as unknown as Pred[] | undefined) ??
    [];
  const fromCache = !runMut.data && (cachedPred?.predictions?.length ?? 0) > 0;
  const cachedAt = cachedPred?.cached_at;
  const cachedDate = cachedPred?.date;

  // Live consensus
  const liveConsensus = useMemo(() => {
    const sel = predictions.filter((_, i) => consChecked.has(i));
    const valid = sel.filter((p) => p.prediction === "UP" || p.prediction === "DOWN");
    const up = valid.filter((p) => p.prediction === "UP").length;
    const down = valid.length - up;
    const total = up + down;
    return { direction: up > down ? "UP" : down > up ? "DOWN" : "SPLIT", up, down, total, agreement: total > 0 ? Math.max(up, down) / total : 0, count: consChecked.size };
  }, [predictions, consChecked]);

  // Filtered predictions (OR within group, AND between groups)
  const filteredPredIdx = useMemo(() => {
    // Group active filters by their group
    const activeByGroup: Map<string, string[]> = new Map();
    for (const g of FILTER_GROUPS) {
      const active = g.tags.filter((t) => consFilters.has(t));
      if (active.length > 0) activeByGroup.set(g.label, active);
    }

    return predictions.map((p, i) => {
      const tags = getTags(p.model);
      tags.push(p.period.toUpperCase());

      // For each group with active filters, model must match at least ONE (OR)
      // Between groups: ALL must pass (AND)
      const matchesFilters = activeByGroup.size === 0 ||
        [...activeByGroup.values()].every((groupTags) =>
          groupTags.some((t) => tags.includes(t))
        );

      const matchesSearch = !consSearch ||
        p.model.toLowerCase().includes(consSearch.toLowerCase()) ||
        p.period.toLowerCase().includes(consSearch.toLowerCase());

      return matchesFilters && matchesSearch ? i : -1;
    }).filter((i) => i >= 0);
  }, [predictions, consFilters, consSearch]);

  // Sort filtered results
  const sortedFilteredIdx = useMemo(() => {
    const { col, asc } = resSort;
    return [...filteredPredIdx].sort((a, b) => {
      const pa = predictions[a];
      const pb = predictions[b];
      let va: string | number;
      let vb: string | number;
      switch (col) {
        case "model": va = pa.model; vb = pb.model; break;
        case "prediction": va = pa.prediction; vb = pb.prediction; break;
        case "confidence": va = pa.confidence; vb = pb.confidence; break;
        case "period": va = pa.period; vb = pb.period; break;
        case "sentiment_score": va = pa.sentiment_score; vb = pb.sentiment_score; break;
        default: va = pa.confidence; vb = pb.confidence;
      }
      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return asc ? cmp : -cmp;
    });
  }, [filteredPredIdx, predictions, resSort]);

  const handleResSort = (col: string) => {
    setResSort((prev) => prev.col === col ? { col, asc: !prev.asc } : { col, asc: col === "model" || col === "prediction" });
  };

  // Builder
  const addSelected = () => {
    const nv: boolean[] = selNews === "both" ? [false, true] : selNews === "yes" ? [true] : [false];
    const existing = new Set(items.map(bKey));
    const toAdd: BuilderRow[] = [];
    for (const m of selModels) for (const p of selPeriods) for (const n of nv) {
      const row = { model: m, period: p, news: n };
      if (!existing.has(bKey(row))) { toAdd.push(row); existing.add(bKey(row)); }
    }
    if (toAdd.length > 0) setItems([...items, ...toAdd]);
  };

  const previewCount = useMemo(() => {
    const nv = selNews === "both" ? [false, true] : selNews === "yes" ? [true] : [false];
    const existing = new Set(items.map(bKey));
    let c = 0;
    for (const m of selModels) for (const p of selPeriods) for (const n of nv)
      if (!existing.has(bKey({ model: m, period: p, news: n }))) c++;
    return c;
  }, [selModels, selPeriods, selNews, items]);

  // Consensus filter toggle
  const toggleConsFilter = (f: string) => {
    const n = new Set(consFilters);
    if (n.has(f)) n.delete(f); else n.add(f);
    setConsFilters(n);
  };

  // Select/deselect filtered rows
  const selectFiltered = () => {
    const n = new Set(consChecked);
    filteredPredIdx.forEach((i) => {
      const p = predictions[i];
      if (p.prediction === "UP" || p.prediction === "DOWN") n.add(i);
    });
    setConsChecked(n);
  };
  const deselectFiltered = () => {
    const n = new Set(consChecked);
    filteredPredIdx.forEach((i) => n.delete(i));
    setConsChecked(n);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap" }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Predictions</h2>
        <div style={{ fontSize: 13, color: s.muted }}>Target: <strong style={{ color: s.accent }}>{isCrypto ? "tomorrow (24/7)" : nextDay}</strong></div>
      </div>

      {/* Ticker + chart */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: s.muted }}>Ticker</span>
        <select value={ticker} onChange={(e) => setTicker(e.target.value)} style={selSt}>
          {tickers?.filter((t) => t.asset_type === "stock").map((t) => <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}
          {tickers?.filter((t) => t.asset_type === "crypto").map((t) => <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}
        </select>
        <Chk label="Chart" checked={showChart} onChange={setShowChart} />
        {showChart && <><span style={{ fontSize: 12, color: s.muted }}>Period:</span><Pills values={CHART_PERIODS} selected={chartPeriod} onSelect={setChartPeriod} /></>}
        <Chk label="Update data" checked={refreshData} onChange={setRefreshData} />
      </div>

      {showChart && chartRows.length > 0 && (
        <Panel title={`${ticker} — ${chartPeriod.toUpperCase()}`} extra={liveConsensus.total > 0 ? <ConsBadge c={liveConsensus} /> : undefined}>
          <PriceChart rows={chartRows} height={280} />
        </Panel>
      )}

      {/* ======== BUILDER ======== */}
      <Panel title="Prediction Builder">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 14 }}>
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

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn onClick={addSelected} label={previewCount > 0 ? `Add ${previewCount}` : "All added"} secondary={previewCount === 0} />
            <span style={{ fontSize: 11, color: s.muted }}>{selModels.size}×{selPeriods.size}×{selNews === "both" ? 2 : 1}{previewCount < selModels.size * selPeriods.size * (selNews === "both" ? 2 : 1) ? ` · ${selModels.size * selPeriods.size * (selNews === "both" ? 2 : 1) - previewCount} dupes` : ""}</span>
          </div>

          <div style={{ borderTop: `1px solid ${s.border}` }} />

          {items.length > 0 ? (<>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <Chk label={`All (${items.length})`} checked={buildSel.size === items.length} onChange={() => setBuildSel(buildSel.size === items.length ? new Set() : new Set(items.map(bKey)))} />
              {buildSel.size > 0 && <SmBtn label={`Remove (${buildSel.size})`} color={s.red} onClick={() => { setItems(items.filter((r) => !buildSel.has(bKey(r)))); setBuildSel(new Set()); }} />}
              <SmBtn label="Clear" color={s.muted} onClick={() => { setItems([]); setBuildSel(new Set()); }} />
              <span style={{ marginLeft: "auto", fontSize: 12, color: s.muted }}>{items.length} queued</span>
            </div>
            <div style={{ maxHeight: 220, overflowY: "auto", border: `1px solid ${s.border}`, borderRadius: 6 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={thSt}></th><th style={{ ...thSt, textAlign: "left" }}>Model</th><th style={thSt}>Period</th><th style={thSt}>News</th>
                </tr></thead>
                <tbody>{items.map((r, i) => { const k = bKey(r); return (
                  <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }} onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                    <td style={{ padding: "3px 8px", width: 28 }}><input type="checkbox" checked={buildSel.has(k)} onChange={() => { const n = new Set(buildSel); if (n.has(k)) n.delete(k); else n.add(k); setBuildSel(n); }} style={{ accentColor: s.accent }} /></td>
                    <td style={{ padding: "3px 8px", color: s.text }}>{r.model}</td>
                    <td style={{ padding: "3px 8px", color: s.muted, textAlign: "center" }}>{r.period.toUpperCase()}</td>
                    <td style={{ padding: "3px 8px", textAlign: "center", color: r.news ? s.green : s.muted }}>{r.news ? "✓" : "—"}</td>
                  </tr>);})}</tbody>
              </table>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <Btn onClick={() => runMut.mutate()} loading={runMut.isPending} label={runMut.isPending ? "Running..." : `Run ${items.length}`} />
            </div>
          </>) : (
            <div style={{ textAlign: "center", padding: 16, color: s.muted, fontSize: 13 }}>Select models above → Add → Run</div>
          )}
        </div>
      </Panel>

      {runMut.isPending && <Ld text="Running models..." />}

      {/* ======== RESULTS + CONSENSUS ======== */}
      {predictions.length > 0 && (
        <Panel
          title={`Results & Consensus`}
          extra={fromCache ? <CachedBadge at={cachedAt} date={cachedDate} onRerun={() => items.length > 0 && runMut.mutate()} disabled={items.length === 0} /> : undefined}
        >
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>

            {/* Live consensus bar */}
            {liveConsensus.total > 0 && (
              <div style={{ padding: 12, borderRadius: 6, border: `1px solid ${s.border}`, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                <ConsBar up={liveConsensus.up} down={liveConsensus.down} />
                <span style={{ fontFamily: s.mono, fontSize: 22, fontWeight: 700, color: liveConsensus.direction === "UP" ? s.green : liveConsensus.direction === "DOWN" ? s.red : s.muted }}>{liveConsensus.direction}</span>
                <span style={{ fontSize: 12, color: s.muted }}>{liveConsensus.up}↑ / {liveConsensus.down}↓ · {pct(liveConsensus.agreement * 100, false)} · {liveConsensus.count}/{predictions.length} selected</span>
              </div>
            )}

            {/* Consensus filter chips (multi-select, AND within group) */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontSize: 11, color: s.muted, minWidth: 55 }}>Filter:</span>
                <input placeholder="Search models..." value={consSearch} onChange={(e) => setConsSearch(e.target.value)}
                  style={{ padding: "4px 8px", borderRadius: 4, fontSize: 11, background: "transparent", border: `1px solid ${s.border}`, color: s.text, width: 140, outline: "none" }} />
                {consFilters.size > 0 && <SmBtn label="Clear filters" color={s.accent} onClick={() => setConsFilters(new Set())} />}
              </div>
              {FILTER_GROUPS.map((g) => (
                <div key={g.label} style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                  <span style={{ fontSize: 10, color: s.muted, minWidth: 55, textAlign: "right", paddingRight: 4 }}>{g.label}</span>
                  {g.tags.map((t) => <Chip key={t} label={t} active={consFilters.has(t)} onClick={() => toggleConsFilter(t)} />)}
                </div>
              ))}
            </div>

            {/* Select/deselect filtered */}
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
              <SmBtn label={`Select filtered (${filteredPredIdx.length})`} color={s.green} onClick={selectFiltered} />
              <SmBtn label="Deselect filtered" color={s.red} onClick={deselectFiltered} />
              <SmBtn label="Select all" color={s.accent} onClick={() => {
                const n = new Set<number>(); predictions.forEach((p, i) => { if (p.prediction === "UP" || p.prediction === "DOWN") n.add(i); }); setConsChecked(n);
              }} />
              <SmBtn label="Select none" color={s.muted} onClick={() => setConsChecked(new Set())} />
              {(consFilters.size > 0 || consSearch) && (
                <span style={{ fontSize: 11, color: s.muted }}>{filteredPredIdx.length} of {predictions.length} shown</span>
              )}
            </div>

            {/* Results table */}
            <div style={{ maxHeight: 450, overflowY: "auto", border: `1px solid ${s.border}`, borderRadius: 6 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 12, minWidth: 550 }}>
                <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={{ ...rTh, width: 28 }}><input type="checkbox"
                    checked={filteredPredIdx.every((i) => consChecked.has(i)) && filteredPredIdx.length > 0}
                    onChange={() => { const allChecked = filteredPredIdx.every((i) => consChecked.has(i)); if (allChecked) deselectFiltered(); else selectFiltered(); }}
                    style={{ accentColor: s.accent }} /></th>
                  <SortTh col="model" label="Model" align="left" sort={resSort} onSort={handleResSort} />
                  <SortTh col="prediction" label="Dir." sort={resSort} onSort={handleResSort} />
                  <SortTh col="confidence" label="Conf." sort={resSort} onSort={handleResSort} />
                  <SortTh col="period" label="Period" sort={resSort} onSort={handleResSort} />
                  <SortTh col="sentiment_score" label="Sent." sort={resSort} onSort={handleResSort} />
                </tr></thead>
                <tbody>
                  {sortedFilteredIdx.map((i) => {
                    const p = predictions[i];
                    const valid = p.prediction === "UP" || p.prediction === "DOWN";
                    const checked = consChecked.has(i);
                    return (
                      <tr key={i} style={{ borderBottom: `1px solid ${s.border}`, opacity: valid ? 1 : 0.35 }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                        <td style={{ padding: "4px 8px" }}>
                          {valid && <input type="checkbox" checked={checked}
                            onChange={() => { const n = new Set(consChecked); if (n.has(i)) n.delete(i); else n.add(i); setConsChecked(n); }}
                            style={{ accentColor: s.accent }} />}
                        </td>
                        <td style={{ padding: "4px 8px", color: checked ? s.text : s.muted, fontFamily: "inherit" }}>{p.model}</td>
                        <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700,
                          color: p.prediction === "UP" ? s.green : p.prediction === "DOWN" ? s.red : s.muted }}>{p.prediction}</td>
                        <td style={{ padding: "4px 8px", textAlign: "center" }}>{p.confidence > 0 ? pct(p.confidence * 100, false) : "—"}</td>
                        <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{p.period}</td>
                        <td style={{ padding: "4px 8px", textAlign: "center",
                          color: p.sentiment_score > 0 ? s.green : p.sentiment_score < 0 ? s.red : s.muted }}>
                          {p.sentiment_score === 0 ? "—" : (p.sentiment_score > 0 ? "+" : "") + p.sentiment_score.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <SmBtn label="Export CSV" color={s.muted} onClick={() => {
                const h = "Model,Prediction,Confidence,Period,Sentiment\n";
                const csv = predictions.map((p) => `${p.model},${p.prediction},${p.confidence},${p.period},${p.sentiment_score}`).join("\n");
                const b = new Blob([h + csv], { type: "text/csv" }); const a = document.createElement("a");
                a.href = URL.createObjectURL(b); a.download = `${ticker}_predictions.csv`; a.click();
              }} />
            </div>
          </div>
        </Panel>
      )}

      {/* News */}
      {predictions.some((p) => (p.headlines?.length ?? 0) > 0) && (
        <Panel title="News Headlines">
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {[...new Set(predictions.flatMap((p) => p.headlines ?? []))].map((h, i) => (
              <div key={i} style={{ fontSize: 12, color: s.text, padding: "4px 0", borderBottom: `1px solid ${s.border}` }}>{h}</div>
            ))}
          </div>
        </Panel>
      )}

      {/* Historical */}
      <Panel title="Historical Prediction">
        <div style={{ padding: 16, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: s.muted }}>Date:</span>
          <input type="date" value={histDate} onChange={(e) => setHistDate(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 6, fontSize: 13, background: "transparent", border: `1px solid ${s.border}`, color: s.text, colorScheme: "dark" }} />
          <span style={{ fontSize: 12, color: s.muted }}>Period:</span>
          <select value={histPeriod} onChange={(e) => setHistPeriod(e.target.value)} style={{ ...selSt, fontSize: 12 }}>
            {PERIODS.map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
          <Btn onClick={() => histMut.mutate()} loading={histMut.isPending} label="Run" secondary={!histDate} />
          {histMut.data?.error && <span style={{ fontSize: 12, color: s.red }}>{histMut.data.error}</span>}
        </div>
        {(histMut.data?.predictions?.length ?? 0) > 0 && (
          <div style={{ padding: "0 16px 16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 6 }}>
              {histMut.data.predictions.map((p: Record<string, unknown>, i: number) => (
                <div key={i} style={{ padding: "6px 10px", borderRadius: 5, border: `1px solid ${s.border}`, display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, color: s.text }}>{String(p.model)}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, fontFamily: s.mono, color: p.prediction === "UP" ? s.green : s.red }}>
                    {String(p.prediction)} {typeof p.confidence === "number" && p.confidence > 0 ? pct(p.confidence * 100, false) : ""}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Panel>
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
function ConsBadge({ c }: { c: { direction: string; up: number; down: number } }) {
  const cl = c.direction === "UP" ? s.green : c.direction === "DOWN" ? s.red : s.muted;
  return (<div style={{ display: "flex", gap: 6, alignItems: "center" }}>
    <span style={{ fontSize: 11, color: s.muted }}>{c.up}↑ {c.down}↓</span>
    <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700, background: cl + "20", color: cl }}>{c.direction}</span>
  </div>);
}
function ConsBar({ up, down }: { up: number; down: number }) {
  const t = up + down || 1;
  return (<div style={{ width: 160, height: 10, borderRadius: 5, overflow: "hidden", display: "flex" }}>
    <div style={{ width: `${(up / t) * 100}%`, background: s.green, transition: "width 0.3s" }} />
    <div style={{ flex: 1, background: s.red }} /></div>);
}
function Ld({ text }: { text: string }) { return <div style={{ textAlign: "center", padding: 32, color: s.muted }}>{text}</div>; }

function CachedBadge({ at, date, onRerun, disabled }: { at?: string; date?: string; onRerun: () => void; disabled?: boolean }) {
  // Show a human-readable "X minutes ago" when possible
  const rel = at ? relTime(at) : null;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: s.muted }}>
      <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(148,163,184,0.15)", color: s.muted, fontWeight: 600 }}>CACHED</span>
      <span>
        {date ? `${date}` : ""}
        {rel ? `  ·  ${rel}` : at ? `  ·  ${at}` : ""}
      </span>
      <button
        onClick={onRerun}
        disabled={disabled}
        title={disabled ? "Build at least one item first" : "Re-run the same models"}
        style={{
          padding: "3px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer",
          border: `1px solid ${s.accent}30`, background: `${s.accent}10`, color: s.accent,
          opacity: disabled ? 0.5 : 1,
        }}
      >
        Re-run
      </button>
    </div>
  );
}

function relTime(iso: string): string {
  const d = new Date(iso);
  const ms = Date.now() - d.getTime();
  if (isNaN(ms)) return iso;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
  return `${days}d ago`;
}

function SortTh({ col, label, align, sort, onSort }: {
  col: string; label: string; align?: string;
  sort: { col: string; asc: boolean }; onSort: (col: string) => void;
}) {
  const active = sort.col === col;
  return (
    <th onClick={() => onSort(col)} style={{
      ...rTh,
      textAlign: (align ?? "center") as "left" | "center" | "right",
      cursor: "pointer",
      userSelect: "none",
      color: active ? s.text : s.muted,
    }}>
      {label}
      {active && <span style={{ marginLeft: 3, opacity: 0.6 }}>{sort.asc ? "↑" : "↓"}</span>}
    </th>
  );
}

const selSt: React.CSSProperties = { padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600, background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer" };
const thSt: React.CSSProperties = { padding: "5px 8px", fontSize: 11, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
const rTh: React.CSSProperties = { padding: "5px 8px", fontSize: 11, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
