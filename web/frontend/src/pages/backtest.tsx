import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import PriceChart from "../components/priceChart";
import { s, Panel, Btn, pct, HelpLink } from "../components/ui";

// Fallback period set used until /api/meta loads (kept in sync with config).
const PERIODS_FALLBACK = ["1mo", "1y", "2y", "5y", "max"];
const CHART_PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const METRICS = [
  { key: "total_return", label: "Return", fmt: (v: number) => pct(v * 100), higher: true },
  { key: "accuracy", label: "Accuracy", fmt: (v: number) => pct(v * 100, false), higher: true },
  { key: "profit_factor", label: "PF", fmt: (v: number) => (v >= 100 ? "∞" : v.toFixed(2)), higher: true },
  { key: "sharpe_ratio", label: "Sharpe", fmt: (v: number) => v.toFixed(2), higher: true },
  { key: "sortino_ratio", label: "Sortino", fmt: (v: number) => (v >= 100 ? "∞" : v.toFixed(2)), higher: true },
  { key: "max_drawdown", label: "Max DD", fmt: (v: number) => pct(v * 100), higher: true },
  { key: "buy_hold_return", label: "B&H", fmt: (v: number) => pct(v * 100), higher: true },
];

// Family display labels used for the result-table "Family" filter + tagging.
const FAMILY_LABELS = ["k-NN", "LinReg", "LSTM", "Prophet", "Chronos-2", "Kronos", "Baseline"];

function getTags(model: string, period: string): string[] {
  const t: string[] = [];
  const name = String(model);
  for (const fam of FAMILY_LABELS) {
    if (name.startsWith(fam)) { t.push(fam); break; }
  }
  if (name.includes("Enhanced")) t.push("Enhanced"); else t.push("Basic");
  if (name.includes("(TW)") || name.includes("Time-Weighted") || name.includes("TW")) t.push("TW");
  else t.push("No TW");
  if (name.includes("News")) t.push("News"); else t.push("No News");
  if (/SL\d/.test(name)) t.push("SL");
  t.push(period.toUpperCase());
  return t;
}

type Res = Record<string, unknown>;

export default function Backtest() {
  const [selTickers, setSelTickers] = useState<Set<string>>(new Set(["AAPL"]));
  const [chartTicker, setChartTicker] = useState("AAPL");
  const [chartPeriod, setChartPeriod] = useState("1y");
  const [showChart, setShowChart] = useState(false);

  // Model FAMILIES (keys from /api/meta), sent to the backend as `models`.
  // Empty = all families. Baselines toggled separately.
  const [selFamilies, setSelFamilies] = useState<Set<string>>(new Set());
  const [includeBaselines, setIncludeBaselines] = useState(true);
  const [selPeriods, setSelPeriods] = useState<Set<string>>(new Set(["1y"]));
  const [selNews, setSelNews] = useState<"no" | "yes" | "both">("no");
  const [globalFee, setGlobalFee] = useState("0.05");
  const [globalSL, setGlobalSL] = useState("0");
  const [slSweep, setSlSweep] = useState(false);
  const [days, setDays] = useState("20");
  const [buyHold, setBuyHold] = useState(true);
  const [refreshData, setRefreshData] = useState(false);
  // Phase 1.3 / 2.1 knobs.
  const [minConfidence, setMinConfidence] = useState("0");
  const [turnoverFees, setTurnoverFees] = useState(false);
  const [holdDays, setHoldDays] = useState("1");

  // News / sentiment knobs — match the CLI defaults (config.py).
  const [sentimentMethod, setSentimentMethod] = useState<"vader" | "finbert" | "naive">("vader");
  const [newsLookback, setNewsLookback] = useState("7");
  const [newsHalfLife, setNewsHalfLife] = useState("3");

  // Summary
  const [sumMetrics, setSumMetrics] = useState<Set<string>>(new Set(["total_return", "sharpe_ratio", "accuracy"]));
  const [sumTickers, setSumTickers] = useState<Set<string>>(new Set());

  // Results filter
  const [resSort, setResSort] = useState<{ col: string; asc: boolean }>({ col: "total_return", asc: false });
  const [resFilters, setResFilters] = useState<Set<string>>(new Set());
  const [resSearch, setResSearch] = useState("");

  const queryClient = useQueryClient();
  const { data: tickers } = useQuery({ queryKey: ["tickers"], queryFn: api.getTickers });
  const { data: meta } = useQuery({ queryKey: ["meta"], queryFn: api.getMeta, staleTime: 60_000 });
  const { data: tickerData } = useQuery({
    queryKey: ["tickerData", chartTicker, chartPeriod],
    queryFn: () => api.getTickerData(chartTicker, chartPeriod),
    enabled: showChart && !!chartTicker,
  });

  // Persisted runs (every POST /api/backtest writes a JSON + CSVs).
  const { data: persistedRuns } = useQuery({
    queryKey: ["backtestRuns"],
    queryFn: api.listBacktestRuns,
    staleTime: 5_000,
  });
  const [loadedRunId, setLoadedRunId] = useState<string | null>(null);
  useEffect(() => {
    if (loadedRunId !== null) return;
    if (persistedRuns && persistedRuns.length > 0) setLoadedRunId(persistedRuns[0].run_id);
  }, [persistedRuns, loadedRunId]);
  const { data: loadedRun } = useQuery({
    queryKey: ["backtestRun", loadedRunId],
    queryFn: () => api.loadBacktestRun(loadedRunId!),
    enabled: !!loadedRunId,
  });

  const chartRows = useMemo(() => [...(tickerData?.data ?? [])].reverse(), [tickerData]);

  // --- Meta-driven options (periods, families, asset classes) ---
  const PERIODS = meta?.periods ?? PERIODS_FALLBACK;
  const families = meta?.model_families ?? [];
  const nonBaselineFamilies = families.filter((f) => f.key !== "baseline");
  const assetClasses = meta?.asset_classes ?? [];
  const tickersByClass = useMemo(() => {
    const out: { key: string; label: string; tickers: string[] }[] = [];
    for (const ac of assetClasses) {
      const present = (tickers ?? []).filter((t) => t.asset_type === ac.key).map((t) => t.ticker);
      if (present.length) out.push({ key: ac.key, label: ac.label, tickers: present });
    }
    return out;
  }, [tickers, assetClasses]);
  const allTickerSyms = useMemo(() => (tickers ?? []).map((t) => t.ticker), [tickers]);

  // Dynamic result-table filter groups (Family list comes from meta).
  const filterGroups = useMemo(() => {
    const famTags = nonBaselineFamilies.map((f) => f.label);
    if (includeBaselines || famTags.length === 0) famTags.push("Baseline");
    return [
      { label: "Family", tags: famTags },
      { label: "Type", tags: ["Basic", "Enhanced"] },
      { label: "Variant", tags: ["TW", "No TW"] },
      { label: "Sentiment", tags: ["News", "No News"] },
      { label: "Stop-loss", tags: ["SL"] },
      { label: "Period", tags: PERIODS.map((p) => p.toUpperCase()) },
    ];
  }, [nonBaselineFamilies, includeBaselines, PERIODS]);

  // ------------------------------------------------------------------
  // Live progress polling
  // ------------------------------------------------------------------
  const [polling, setPolling] = useState(false);
  const { data: progress, refetch: refetchProgress } = useQuery({
    queryKey: ["backtestProgress"],
    queryFn: api.backtestProgress,
    staleTime: 0,
  });
  useEffect(() => {
    if (!polling) return;
    refetchProgress();
    const id = setInterval(() => refetchProgress(), 500);
    return () => clearInterval(id);
  }, [polling, refetchProgress]);
  const [nowTick, setNowTick] = useState(0);
  useEffect(() => {
    if (!polling) return;
    const id = setInterval(() => setNowTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [polling]);

  const runMut = useMutation({
    mutationFn: () => {
      const newsRequested = selNews !== "no";
      const fams = [...selFamilies];
      return api.backtest({
        tickers: [...selTickers],
        periods: [...selPeriods],
        days: parseInt(days) || 20,
        fee_pct: parseFloat(globalFee) || 0,
        stop_loss_pct: parseFloat(globalSL) || 0,
        sl_sweep: slSweep,
        buy_hold: buyHold,
        refresh_data: refreshData,
        // Family filter — omit (= all) when nothing or everything selected.
        ...(fams.length && fams.length !== nonBaselineFamilies.length
          ? { models: includeBaselines ? [...fams, "baseline"] : fams }
          : {}),
        include_baselines: includeBaselines,
        min_confidence: parseFloat(minConfidence) || 0,
        turnover_fees: turnoverFees,
        hold_days: parseInt(holdDays) || 1,
        ...(newsRequested
          ? {
              sentiment_method: sentimentMethod,
              news_lookback_days: parseInt(newsLookback) || 7,
              news_half_life_days: parseFloat(newsHalfLife) || 0,
            }
          : {}),
      });
    },
    onMutate: () => {
      setPolling(true);
      queryClient.invalidateQueries({ queryKey: ["backtestProgress"] });
    },
    onSuccess: () => {
      setSumTickers(new Set());
      setLoadedRunId(null); // prefer the fresh run over any loaded cached run
      queryClient.invalidateQueries({ queryKey: ["backtestRuns"] });
    },
    onSettled: () => setPolling(false),
  });

  // Prefer fresh run results, fall back to the loaded cached run.
  const results: Res[] = (runMut.data?.results as Res[] | undefined)
    ?? ((loadedRun?.response?.results as Res[] | undefined))
    ?? [];
  const fromCachedRun = !runMut.data && (loadedRun?.response?.results?.length ?? 0) > 0;

  // Ticker toggle helpers
  const toggleTicker = (t: string) => {
    const n = new Set(selTickers);
    if (n.has(t)) { if (n.size > 1) n.delete(t); } else n.add(t);
    setSelTickers(n);
    if (!n.has(chartTicker)) setChartTicker([...n][0]);
  };
  const setClass = (syms: string[]) => { setSelTickers(new Set(syms)); if (syms[0]) setChartTicker(syms[0]); };
  const setAllTickers = () => setSelTickers(new Set(allTickerSyms));

  const allResultTickers = useMemo(() => [...new Set(results.map((r) => String(r.ticker)))], [results]);

  const filteredResults = useMemo(() => {
    const activeByGroup: Map<string, string[]> = new Map();
    for (const g of filterGroups) {
      const active = g.tags.filter((t) => resFilters.has(t));
      if (active.length > 0) activeByGroup.set(g.label, active);
    }
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
  }, [results, resFilters, resSearch, allResultTickers, filterGroups]);

  const sortedResults = useMemo(() => {
    return [...filteredResults].sort((a, b) => {
      const va = (a[resSort.col] as number) ?? 0; const vb = (b[resSort.col] as number) ?? 0;
      return resSort.asc ? va - vb : vb - va;
    });
  }, [filteredResults, resSort]);

  const handleResSort = (col: string) => setResSort((p) => (p.col === col ? { col, asc: !p.asc } : { col, asc: false }));
  const toggleResFilter = (f: string) => { const n = new Set(resFilters); if (n.has(f)) n.delete(f); else n.add(f); setResFilters(n); };

  const gateActive = (parseFloat(minConfidence) || 0) > 0;
  const turnoverActive = turnoverFees || (parseInt(holdDays) || 1) > 1;

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
      const vals = pool.map((r) => ({ model: String(r.model), ticker: String(r.ticker), period: String(r.period), value: (r[m.key] as number) ?? -999 }));
      const bestVal = vals.reduce((best, v) => (v.value > best ? v.value : best), -Infinity);
      const epsilon = Math.abs(bestVal) * 0.001 || 0.0001;
      const ties = vals.filter((v) => Math.abs(v.value - bestVal) < epsilon);
      out[m.key] = { models: ties };
    }
    return out;
  }, [results, sumMetrics, sumTickers]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>Backtest</h2>

      {/* Tickers — grouped by every asset class from /api/meta */}
      <Panel title="Tickers">
        <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          {tickersByClass.map((cls) => (
            <div key={cls.key} style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: s.muted, minWidth: 80 }}>{cls.label}</span>
              <Chip label={`All ${cls.label}`} active={cls.tickers.every((t) => selTickers.has(t))} onClick={() => setClass(cls.tickers)} accent />
              {cls.tickers.map((t) => <Chip key={t} label={t} active={selTickers.has(t)} onClick={() => toggleTicker(t)} />)}
            </div>
          ))}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Chip label="All" active={selTickers.size === allTickerSyms.length && allTickerSyms.length > 0} onClick={setAllTickers} accent />
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

      {/* Cached runs picker */}
      {persistedRuns && persistedRuns.length > 0 && (
        <Panel
          title={`Saved runs (${persistedRuns.length})`}
          extra={fromCachedRun && loadedRun?.saved_at ? (
            <span style={{ fontSize: 11, color: s.muted }}>
              <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(148,163,184,0.15)", color: s.muted, fontWeight: 600, marginRight: 6 }}>CACHED</span>
              {fmtRel(loadedRun.saved_at)} · CSV: <code style={{ fontFamily: s.mono, color: s.accent }}>{loadedRun.results_dir}</code>
            </span>
          ) : undefined}
        >
          <div style={{ padding: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: s.muted }}>Load run:</span>
            <select
              value={loadedRunId ?? ""}
              onChange={(e) => { setLoadedRunId(e.target.value || null); runMut.reset(); }}
              style={{ ...selSt, fontSize: 12, maxWidth: 360 }}
            >
              <option value="">— don't load —</option>
              {persistedRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} · {r.result_count} results · {fmtShortDate(r.saved_at)}
                </option>
              ))}
            </select>
          </div>
        </Panel>
      )}

      {/* Builder */}
      <Panel title="Backtest Builder">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <SelRow label="Models">
            <Chip label="All" active={selFamilies.size === 0} onClick={() => setSelFamilies(new Set())} accent />
            {nonBaselineFamilies.map((f) => (
              <Chip
                key={f.key}
                label={f.available ? f.label : `${f.label} (n/a)`}
                active={selFamilies.has(f.key)}
                onClick={() => {
                  if (!f.available) return;
                  const n = new Set(selFamilies);
                  if (n.has(f.key)) n.delete(f.key); else n.add(f.key);
                  setSelFamilies(n);
                }}
              />
            ))}
            <span style={{ width: 12 }} />
            <Chk label="Baselines" checked={includeBaselines} onChange={setIncludeBaselines} />
            <HelpLink to="models/baselines" title="Baselines are dumb predictors (Always-Long, Random…) a real model must beat. Click for details." />
            <HelpLink to="models/model-families" title="What each model family is (k-NN, LinReg, LSTM, Prophet, Chronos-2, Kronos)." />
          </SelRow>
          <SelRow label="Periods">
            <Chip label="All" active={selPeriods.size === PERIODS.length} onClick={() => setSelPeriods(selPeriods.size === PERIODS.length ? new Set() : new Set(PERIODS))} accent />
            {PERIODS.map((p) => <Chip key={p} label={p.toUpperCase()} active={selPeriods.has(p)} onClick={() => { const n = new Set(selPeriods); if (n.has(p)) n.delete(p); else n.add(p); setSelPeriods(n); }} />)}
          </SelRow>
          <SelRow label="News">
            {(["no", "yes", "both"] as const).map((v) => <Chip key={v} label={v === "no" ? "Without" : v === "yes" ? "With" : "Both"} active={selNews === v} onClick={() => setSelNews(v)} accent={selNews === v} />)}
            <HelpLink to="models/news-sentiment" title="Models can read recent headlines (sentiment) and nudge their call. Look-ahead-safe in backtests." />
          </SelRow>
          {selNews !== "no" && (
            <>
              <SelRow label="Scorer">
                {(["vader", "finbert", "naive"] as const).map((m) => (
                  <Chip key={m} label={m === "vader" ? "VADER" : m === "finbert" ? "FinBERT" : "Naive"} active={sentimentMethod === m} onClick={() => setSentimentMethod(m)} accent={sentimentMethod === m} />
                ))}
              </SelRow>
              <SelRow label="News cfg">
                <NumField label="Lookback days" value={newsLookback} onChange={setNewsLookback} width={50} />
                <NumField label="Half-life days" value={newsHalfLife} onChange={setNewsHalfLife} width={50} />
                <span style={{ fontSize: 10, color: s.muted, alignSelf: "center" }}>0 = no decay</span>
              </SelRow>
            </>
          )}
          {/* Strategy / fee / risk knobs */}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Fee %" value={globalFee} onChange={setGlobalFee} width={60} />
              <HelpLink to="strategy/trading-fees" title="Cost per side (buy + sell = 2×). Piles up over a daily-trading backtest." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="SL %" value={globalSL} onChange={setGlobalSL} width={50} />
              <HelpLink to="strategy/stop-loss" title="Auto-close a losing trade at this % drop. A risk control, not an edge. 0 = off." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <Chk label={`SL sweep ${meta ? "(" + meta.sl_sweep.join("/") + ")" : ""}`} checked={slSweep} onChange={setSlSweep} />
              <HelpLink to="strategy/stop-loss-sweep" title="Run several stop-loss levels at once and compare. Overrides the single SL%." />
            </span>
            <NumField label="Days" value={days} onChange={setDays} width={50} />
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Min conf θ" value={minConfidence} onChange={setMinConfidence} width={50} />
              <HelpLink to="strategy/confidence-gate-min-confidence" title="Sit out days the model isn't confident about. Helps only if confidence is calibrated." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <Chk label="Turnover fees" checked={turnoverFees} onChange={setTurnoverFees} />
              <HelpLink to="strategy/turnover-fees" title="Charge fees only when the position changes, not every day. The realistic cost." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Hold days" value={holdDays} onChange={setHoldDays} width={50} />
              <HelpLink to="strategy/hold-days" title="Hold a position N days before re-reading the signal. Accuracy is unchanged." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <Chk label="B&H benchmark" checked={buyHold} onChange={setBuyHold} />
              <HelpLink to="strategy/buy-and-hold" title="The do-nothing benchmark. Beating it after fees is the real test." />
            </span>
            <Chk label="Update data" checked={refreshData} onChange={setRefreshData} />
          </div>
          <div style={{ fontSize: 10, color: s.muted }}>
            {gateActive && <span>Gate θ={minConfidence}: low-confidence days sit out. </span>}
            {turnoverActive && <span>Turnover-aware fees{(parseInt(holdDays) || 1) > 1 ? `, ${holdDays}-day holds` : ""}. </span>}
            {slSweep && <span>Stop-loss sweep overrides the single SL%. </span>}
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <Btn
              onClick={() => runMut.mutate()}
              loading={runMut.isPending}
              label={runMut.isPending ? "Running..." : `Run (${selTickers.size} × ${selPeriods.size || PERIODS.length})`}
            />
          </div>
        </div>
      </Panel>

      {runMut.isPending && <ProgressPanel p={progress} tick={nowTick} />}
      {runMut.isError && (
        <Panel title="Error"><div style={{ padding: 12, color: s.red, fontSize: 12 }}>{String((runMut.error as Error)?.message ?? runMut.error)}</div></Panel>
      )}

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
                const cols = [
                  "model", "ticker", "period", "total_return", "accuracy", "profit_factor",
                  "sharpe_ratio", "max_drawdown", "buy_hold_return", "fee_pct", "stop_loss_pct",
                  "min_confidence", "coverage", "turnover_count", "fees_paid",
                ];
                const h = cols.join(",") + "\n";
                const csv = sortedResults.map((r) => cols.map((c) => r[c] ?? "").join(",")).join("\n");
                const b = new Blob([h + csv], { type: "text/csv" });
                const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `backtest_${days}d.csv`; a.click();
              }} />
            </div>
            {filterGroups.map((g) => (
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
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 11, minWidth: 980 }}>
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
                {gateActive && <STh col="coverage" l="Cov." sort={resSort} onSort={handleResSort} />}
                {turnoverActive && <STh col="turnover_count" l="Turn." sort={resSort} onSort={handleResSort} />}
                <th style={rTh}>Sent</th>
                <th style={rTh}>Beat?</th>
              </tr></thead>
              <tbody>{sortedResults.map((r, i) => {
                if (r.error) return <tr key={i} style={{ borderBottom: `1px solid ${s.border}`, opacity: 0.4 }}><td style={{ padding: "4px 8px" }} colSpan={14}>{String(r.model)} — {String(r.error)}</td></tr>;
                const ret = (r.total_return as number) ?? 0; const bh = (r.buy_hold_return as number) ?? 0; const beat = ret > bh;
                const days_ = (r.days as { sentiment_score?: number }[] | undefined) ?? [];
                const sentiments = days_.map((d) => d.sentiment_score ?? 0);
                const meanSent = sentiments.length ? sentiments.reduce((a, b) => a + b, 0) / sentiments.length : 0;
                const activeDays = sentiments.filter((x) => x !== 0).length;
                const cov = (r.coverage as number) ?? 1;
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
                    {gateActive && <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{(cov * 100).toFixed(0)}%</td>}
                    {turnoverActive && <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{String(r.turnover_count ?? "—")}</td>}
                    <td title={`mean per-day sentiment · active days: ${activeDays}/${sentiments.length}`}
                      style={{ padding: "4px 8px", textAlign: "center", color: meanSent > 0 ? s.green : meanSent < 0 ? s.red : s.muted }}>
                      {activeDays === 0 ? "—" : (meanSent >= 0 ? "+" : "") + meanSent.toFixed(2)}
                    </td>
                    <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: beat ? s.green : s.red }}>{beat ? "✓" : "✗"}</td>
                  </tr>);
              })}</tbody>
            </table>
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
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: 1, alignItems: "center" }}>{children}</div>
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
function ProgressPanel({ p, tick }: { p: Record<string, unknown> | undefined; tick: number }) {
  void tick;
  const total = (p?.total_units as number) ?? 0;
  const done = (p?.completed_units as number) ?? 0;
  const fraction = total > 0 ? done / total : 0;
  const startedAt = p?.started_at as string | undefined;
  const elapsed = startedAt ? Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000)) : 0;
  const m = Math.floor(elapsed / 60); const sec = elapsed % 60;
  const elapsedStr = m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  const etaStr = (() => {
    if (done < 1 || elapsed < 1 || total <= done) return "—";
    const perUnit = elapsed / done; const remaining = Math.round(perUnit * (total - done));
    const mm = Math.floor(remaining / 60); const ss = remaining % 60;
    return mm > 0 ? `~${mm}m ${ss}s` : `~${ss}s`;
  })();
  return (
    <Panel title={`Running backtest — ${done}/${total || "?"}`}>
      <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", fontSize: 12 }}>
          <span style={{ color: s.muted }}>Ticker:</span>
          <strong style={{ color: s.text, fontFamily: s.mono }}>
            {String(p?.ticker ?? "—")} ({(p?.ticker_idx as number) ?? 0}/{(p?.ticker_total as number) ?? 0})
          </strong>
          <span style={{ color: s.muted }}>Period:</span>
          <strong style={{ color: s.text, fontFamily: s.mono }}>
            {String(p?.period ?? "—")} ({(p?.period_idx as number) ?? 0}/{(p?.period_total as number) ?? 0})
          </strong>
          <span style={{ color: s.muted, marginLeft: "auto" }}>
            Elapsed: <span style={{ color: s.text }}>{elapsedStr}</span>
            {" · ETA: "}<span style={{ color: s.text }}>{etaStr}</span>
            {" · Rows: "}<span style={{ color: s.accent }}>{(p?.results_so_far as number) ?? 0}</span>
          </span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: s.hover, overflow: "hidden", border: `1px solid ${s.border}` }}>
          <div style={{ width: `${fraction * 100}%`, height: "100%", background: s.accent, transition: "width 0.3s ease" }} />
        </div>
        <div style={{ fontSize: 10, color: s.muted, fontStyle: "italic" }}>
          Backend stage: <strong>{String(p?.stage ?? "preparing")}</strong>
        </div>
      </div>
    </Panel>
  );
}

function fmtRel(iso: string): string {
  try {
    const d = new Date(iso); const ms = Date.now() - d.getTime();
    if (isNaN(ms)) return iso;
    const sec = Math.floor(ms / 1000); if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60); if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60); if (hr < 24) return `${hr}h ago`;
    return `${Math.floor(hr / 24)}d ago`;
  } catch { return iso; }
}
function fmtShortDate(iso: string | undefined): string {
  if (!iso) return "";
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

const selSt: React.CSSProperties = { padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600, background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer" };
const rTh: React.CSSProperties = { padding: "5px 8px", fontSize: 10, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
