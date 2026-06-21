import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { OOSTickerRow, OOSSummary } from "../lib/api";
import { s, Panel, Btn, pct, HelpLink } from "../components/ui";

const PERIODS_FALLBACK = ["1mo", "1y", "2y", "5y", "max"];

/**
 * Out-of-sample harness tab.
 *
 * Runs the disciplined select-on-window-N → evaluate-on-disjoint-N+1 pipeline
 * (the honest alternative to "best per ticker"). Mirrors the Backtest tab's
 * shape: meta-driven config, live progress, persisted-run redisplay. Results
 * show each ticker's selection winner and how it actually did out-of-sample,
 * plus the headline selection-inflation gap.
 */
export default function OOS() {
  const [selTickers, setSelTickers] = useState<Set<string>>(new Set(["AAPL", "MSFT", "NVDA"]));
  const [selPeriods, setSelPeriods] = useState<Set<string>>(new Set(["1y", "2y", "5y"]));
  const [selFamilies, setSelFamilies] = useState<Set<string>>(new Set());
  const [includeBaselines, setIncludeBaselines] = useState(true);
  const [days, setDays] = useState("50");
  const [fee, setFee] = useState("0.05");
  const [stopLoss, setStopLoss] = useState("0");
  const [minConfidence, setMinConfidence] = useState("0");
  const [turnoverFees, setTurnoverFees] = useState(false);
  const [holdDays, setHoldDays] = useState("1");
  const [positionMode, setPositionMode] = useState(false);
  const [buyHold, setBuyHold] = useState(true);
  const [refreshData, setRefreshData] = useState(false);
  const [selNews, setSelNews] = useState<"no" | "yes">("no");
  const [sentimentMethod, setSentimentMethod] = useState<"vader" | "finbert" | "naive">("vader");

  const qc = useQueryClient();
  const { data: tickers } = useQuery({ queryKey: ["tickers"], queryFn: api.getTickers });
  const { data: meta } = useQuery({ queryKey: ["meta"], queryFn: api.getMeta, staleTime: 60_000 });

  const PERIODS = meta?.periods ?? PERIODS_FALLBACK;
  const families = meta?.model_families ?? [];
  const nonBaselineFamilies = families.filter((f) => f.key !== "baseline");
  const mainFamilies = nonBaselineFamilies.filter((f) => f.tier !== "educational");
  const eduFamilies = nonBaselineFamilies.filter((f) => f.tier === "educational");
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

  // Persisted runs
  const { data: runs } = useQuery({ queryKey: ["oosRuns"], queryFn: api.listOosRuns, staleTime: 5_000 });
  const [loadedRunId, setLoadedRunId] = useState<string | null>(null);
  useEffect(() => {
    if (loadedRunId !== null) return;
    if (runs && runs.length > 0) setLoadedRunId(runs[0].run_id);
  }, [runs, loadedRunId]);
  const { data: loadedRun } = useQuery({
    queryKey: ["oosRun", loadedRunId],
    queryFn: () => api.loadOosRun(loadedRunId!),
    enabled: !!loadedRunId,
  });

  // Live progress
  const [polling, setPolling] = useState(false);
  const { data: progress, refetch: refetchProgress } = useQuery({
    queryKey: ["oosProgress"], queryFn: api.oosProgress, staleTime: 0,
  });
  useEffect(() => {
    if (!polling) return;
    refetchProgress();
    const id = setInterval(() => refetchProgress(), 600);
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
      const fams = [...selFamilies];
      return api.oos({
        tickers: [...selTickers],
        periods: [...selPeriods],
        days: parseInt(days) || 50,
        fee_pct: parseFloat(fee) || 0,
        stop_loss_pct: parseFloat(stopLoss) || 0,
        buy_hold: buyHold,
        refresh_data: refreshData,
        ...(fams.length && fams.length !== nonBaselineFamilies.length
          ? { models: includeBaselines ? [...fams, "baseline"] : fams }
          : {}),
        include_baselines: includeBaselines,
        min_confidence: parseFloat(minConfidence) || 0,
        turnover_fees: turnoverFees,
        hold_days: parseInt(holdDays) || 1,
        position_mode: positionMode,
        ...(selNews === "yes" ? { sentiment_method: sentimentMethod } : {}),
      });
    },
    onMutate: () => { setPolling(true); qc.invalidateQueries({ queryKey: ["oosProgress"] }); },
    onSuccess: () => { setLoadedRunId(null); qc.invalidateQueries({ queryKey: ["oosRuns"] }); },
    onSettled: () => setPolling(false),
  });

  const rows: OOSTickerRow[] = (runMut.data?.rows as OOSTickerRow[] | undefined)
    ?? (loadedRun?.response?.rows as OOSTickerRow[] | undefined) ?? [];
  const summary: OOSSummary | undefined = runMut.data?.summary ?? loadedRun?.response?.summary;
  const fromCached = !runMut.data && (loadedRun?.response?.rows?.length ?? 0) > 0;
  const gateActive = (summary?.min_confidence ?? 0) > 0;

  const toggleTicker = (t: string) => {
    const n = new Set(selTickers);
    if (n.has(t)) { if (n.size > 1) n.delete(t); } else n.add(t);
    setSelTickers(n);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
          Out-of-Sample Harness
          <HelpLink to="oos/out-of-sample-testing" title="Picks the best setup on one window, scores it on a fresh disjoint window — the honest read. Click for the full explanation." />
        </h2>
        <span style={{ fontSize: 12, color: s.muted, maxWidth: 560 }}>
          Picks the best model+period on a selection window, then scores it on the next
          <em> disjoint</em> window — the honest beat-buy-and-hold rate, free of selection inflation.
        </span>
      </div>

      {/* Tickers */}
      <Panel title="Tickers">
        <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          {tickersByClass.map((cls) => (
            <div key={cls.key} style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: s.muted, minWidth: 80 }}>{cls.label}</span>
              <Chip label={`All ${cls.label}`} active={cls.tickers.every((t) => selTickers.has(t))} onClick={() => setSelTickers(new Set(cls.tickers))} accent />
              {cls.tickers.map((t) => <Chip key={t} label={t} active={selTickers.has(t)} onClick={() => toggleTicker(t)} />)}
            </div>
          ))}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Chip label="All" active={selTickers.size === allTickerSyms.length && allTickerSyms.length > 0} onClick={() => setSelTickers(new Set(allTickerSyms))} accent />
            <span style={{ fontSize: 11, color: s.muted }}>{selTickers.size} selected</span>
          </div>
        </div>
      </Panel>

      {/* Saved runs */}
      {runs && runs.length > 0 && (
        <Panel
          title={`Saved OOS runs (${runs.length})`}
          extra={fromCached && loadedRun?.saved_at ? (
            <span style={{ fontSize: 11, color: s.muted }}>
              <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(148,163,184,0.15)", color: s.muted, fontWeight: 600, marginRight: 6 }}>CACHED</span>
              CSV: <code style={{ fontFamily: s.mono, color: s.accent }}>{loadedRun.results_dir}</code>
            </span>
          ) : undefined}
        >
          <div style={{ padding: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: s.muted }}>Load run:</span>
            <select
              value={loadedRunId ?? ""}
              onChange={(e) => { setLoadedRunId(e.target.value || null); runMut.reset(); }}
              style={{ ...selSt, fontSize: 12, maxWidth: 420 }}
            >
              <option value="">— don't load —</option>
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.row_count} tickers</option>
              ))}
            </select>
          </div>
        </Panel>
      )}

      {/* Config */}
      <Panel title="OOS Configuration">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <SelRow label="Models">
            <Chip label="All" active={selFamilies.size === 0} onClick={() => setSelFamilies(new Set())} accent />
            {mainFamilies.map((f) => (
              <Chip key={f.key} label={(f.available ? f.label : `${f.label} (n/a)`) + (f.slow ? " ⏳" : "")} active={selFamilies.has(f.key)}
                onClick={() => { if (!f.available) return; const n = new Set(selFamilies); if (n.has(f.key)) n.delete(f.key); else n.add(f.key); setSelFamilies(n); }} />
            ))}
            <span style={{ width: 12 }} />
            <Chk label="Baselines as candidates" checked={includeBaselines} onChange={setIncludeBaselines} />
            <HelpLink to="models/baselines" title="Include the naive baselines as selection candidates — detects when a coin flip 'wins' OOS." />
          </SelRow>
          {eduFamilies.length > 0 && (
            <SelRow label="Simple">
              {eduFamilies.map((f) => (
                <Chip key={f.key} label={f.label} active={selFamilies.has(f.key)}
                  onClick={() => { const n = new Set(selFamilies); if (n.has(f.key)) n.delete(f.key); else n.add(f.key); setSelFamilies(n); }} />
              ))}
              <span style={{ fontSize: 10, color: s.muted, alignSelf: "center" }}>educational / illustrative</span>
            </SelRow>
          )}
          <SelRow label="Periods">
            <Chip label="All" active={selPeriods.size === PERIODS.length} onClick={() => setSelPeriods(selPeriods.size === PERIODS.length ? new Set() : new Set(PERIODS))} accent />
            {PERIODS.map((p) => <Chip key={p} label={p.toUpperCase()} active={selPeriods.has(p)} onClick={() => { const n = new Set(selPeriods); if (n.has(p)) n.delete(p); else n.add(p); setSelPeriods(n); }} />)}
          </SelRow>
          <SelRow label="News">
            {(["no", "yes"] as const).map((v) => <Chip key={v} label={v === "no" ? "Without" : "With"} active={selNews === v} onClick={() => setSelNews(v)} accent={selNews === v} />)}
            {selNews === "yes" && (["vader", "finbert", "naive"] as const).map((m) => (
              <Chip key={m} label={m === "vader" ? "VADER" : m === "finbert" ? "FinBERT" : "Naive"} active={sentimentMethod === m} onClick={() => setSentimentMethod(m)} accent={sentimentMethod === m} />
            ))}
          </SelRow>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <NumField label="Days/window" value={days} onChange={setDays} width={50} />
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Fee %" value={fee} onChange={setFee} width={50} />
              <HelpLink to="strategy/trading-fees" title="Cost per side (buy + sell = 2×)." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="SL %" value={stopLoss} onChange={setStopLoss} width={50} />
              <HelpLink to="strategy/stop-loss" title="Auto-close a losing trade at this % drop. Single value here (no sweep in OOS)." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Min conf θ" value={minConfidence} onChange={setMinConfidence} width={50} />
              <HelpLink to="strategy/confidence-gate-min-confidence" title="Sit out low-confidence days, applied to both windows." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <Chk label="Turnover fees" checked={turnoverFees} onChange={setTurnoverFees} />
              <HelpLink to="strategy/turnover-fees" title="Charge fees only on position changes — realistic cost." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <NumField label="Hold days" value={holdDays} onChange={setHoldDays} width={50} />
              <HelpLink to="strategy/hold-days" title="Hold N days before re-reading the signal." />
            </span>
            <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
              <Chk label="Position mode" checked={positionMode} onChange={setPositionMode} />
              <HelpLink to="strategy/position-mode" title="Compound same-direction holds into one trade (one round-trip fee per run)." />
            </span>
            <Chk label="B&H" checked={buyHold} onChange={setBuyHold} />
            <Chk label="Update data" checked={refreshData} onChange={setRefreshData} />
          </div>
          <div style={{ fontSize: 10, color: s.muted }}>
            Needs ≥ 2×days+20 rows per ticker (two disjoint windows). θ &amp; turnover apply to both windows; SL is single-valued (not swept) to avoid re-inflating selection.
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <Btn onClick={() => runMut.mutate()} loading={runMut.isPending} label={runMut.isPending ? "Running..." : `Run OOS (${selTickers.size} tickers)`} />
          </div>
        </div>
      </Panel>

      {runMut.isPending && <ProgressPanel p={progress} tick={nowTick} />}
      {runMut.isError && (
        <Panel title="Error"><div style={{ padding: 12, color: s.red, fontSize: 12 }}>{String((runMut.error as Error)?.message ?? runMut.error)}</div></Panel>
      )}

      {/* Summary */}
      {summary && summary.tickers > 0 && (
        <Panel title="Aggregate — the honest read">
          <div style={{ padding: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 10 }}>
            <Stat label="Tickers" value={String(summary.tickers)} />
            <Stat label="OOS beat-B&H rate" value={pct(summary.oos_beat_bh_rate * 100, false)} color={summary.oos_beat_bh_rate >= 0.5 ? s.green : s.red} />
            <Stat label="Median OOS return" value={pct(summary.median_oos_return * 100)} color={summary.median_oos_return >= 0 ? s.green : s.red} />
            <Stat label="Median OOS accuracy" value={summary.median_oos_accuracy.toFixed(3)} />
            <Stat label="Selection-inflation gap" value={pct(summary.in_sample_minus_oos_median * 100)} color={s.muted}
              hint="median(in-sample − OOS return). Large = the in-sample winners were mostly overfit." />
            {gateActive && <Stat label="Median OOS coverage" value={pct(summary.median_oos_coverage * 100, false)} />}
            {gateActive && <Stat label="Median Brier / ECE" value={`${summary.median_oos_brier.toFixed(3)} / ${summary.median_oos_ece.toFixed(3)}`} />}
            {gateActive && <Stat label="Significant (p<0.05)" value={`${summary.tickers_significant_p05}/${summary.tickers}`} />}
          </div>
        </Panel>
      )}

      {/* Per-ticker table */}
      {rows.length > 0 && (
        <Panel title={`Per-ticker (${rows.length})`}>
          <div style={{ maxHeight: 520, overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 11, minWidth: 900 }}>
              <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
                <th style={{ ...rTh, textAlign: "left" }}>Ticker</th>
                <th style={{ ...rTh, textAlign: "left" }}>Winner</th>
                <th style={rTh}>Per.</th>
                <th style={rTh}>In-sample</th>
                <th style={rTh}>OOS ret</th>
                <th style={rTh}>OOS acc</th>
                <th style={rTh}>OOS B&H</th>
                <th style={rTh}>Beat?</th>
                {gateActive && <th style={rTh}>Coverage</th>}
                {gateActive && <th style={rTh}>OOS p</th>}
              </tr></thead>
              <tbody>{rows.map((r) => {
                const beat = r.beats_bh_oos === 1;
                return (
                  <tr key={r.ticker} style={{ borderBottom: `1px solid ${s.border}` }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                    <td style={{ padding: "4px 8px", color: s.text }}>{r.ticker}</td>
                    <td style={{ padding: "4px 8px", color: s.text, whiteSpace: "nowrap" }}>{r.winner_model}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{r.winner_period}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{pct(r.in_sample_return * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: r.oos_return >= 0 ? s.green : s.red }}>{pct(r.oos_return * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center" }}>{pct(r.oos_accuracy * 100, false)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", color: r.oos_buy_hold >= 0 ? s.green : s.red }}>{pct(r.oos_buy_hold * 100)}</td>
                    <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: beat ? s.green : s.red }}>{beat ? "✓" : "✗"}</td>
                    {gateActive && <td style={{ padding: "4px 8px", textAlign: "center", color: s.muted }}>{(r.oos_coverage * 100).toFixed(0)}%</td>}
                    {gateActive && <td style={{ padding: "4px 8px", textAlign: "center", color: r.oos_binomial_p < 0.05 ? s.green : s.muted }}>{r.oos_binomial_p.toFixed(3)}</td>}
                  </tr>);
              })}</tbody>
            </table>
          </div>
          <div style={{ padding: "8px 12px", display: "flex", justifyContent: "flex-end" }}>
            <SmBtn label="Export CSV" color={s.muted} onClick={() => {
              const cols = ["ticker", "winner_model", "winner_period", "in_sample_return", "oos_return", "oos_accuracy", "oos_buy_hold", "beats_bh_oos", "oos_coverage", "oos_binomial_p"];
              const h = cols.join(",") + "\n";
              const csv = rows.map((r) => cols.map((c) => (r as unknown as Record<string, unknown>)[c] ?? "").join(",")).join("\n");
              const b = new Blob([h + csv], { type: "text/csv" });
              const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `oos_${days}d.csv`; a.click();
            }} />
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ---- Helpers (local copies, same style as Backtest) ---- */
function SelRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (<div style={{ display: "flex", gap: 8, alignItems: "flex-start", flexWrap: "wrap" }}>
    <span style={{ fontSize: 12, color: s.muted, minWidth: 55, paddingTop: 4 }}>{label}</span>
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: 1, alignItems: "center" }}>{children}</div>
  </div>);
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
function Stat({ label, value, color, hint }: { label: string; value: string; color?: string; hint?: string }) {
  return (
    <div title={hint} style={{ background: s.surface, border: `1px solid ${s.border}`, borderRadius: 8, padding: "10px 14px" }}>
      <div style={{ fontSize: 11, color: s.muted, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: color ?? s.text, fontFamily: s.mono }}>{value}</div>
    </div>
  );
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
  return (
    <Panel title={`Running OOS — ${done}/${total || "?"}`}>
      <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", fontSize: 12 }}>
          <span style={{ color: s.muted }}>Ticker:</span>
          <strong style={{ color: s.text, fontFamily: s.mono }}>{String(p?.ticker ?? "—")} ({(p?.ticker_idx as number) ?? 0}/{(p?.ticker_total as number) ?? 0})</strong>
          <span style={{ color: s.muted, marginLeft: "auto" }}>
            Elapsed: <span style={{ color: s.text }}>{elapsedStr}</span>
            {" · Rows: "}<span style={{ color: s.accent }}>{(p?.rows_so_far as number) ?? 0}</span>
          </span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: s.hover, overflow: "hidden", border: `1px solid ${s.border}` }}>
          <div style={{ width: `${fraction * 100}%`, height: "100%", background: s.accent, transition: "width 0.3s ease" }} />
        </div>
        <div style={{ fontSize: 10, color: s.muted, fontStyle: "italic" }}>Backend stage: <strong>{String(p?.stage ?? "preparing")}</strong></div>
      </div>
    </Panel>
  );
}

const selSt: React.CSSProperties = { padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600, background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer" };
const rTh: React.CSSProperties = { padding: "5px 8px", fontSize: 10, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
