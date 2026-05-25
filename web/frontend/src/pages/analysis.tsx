/**
 * Analysis.tsx — Research-oriented tab.
 *
 * Reads the ``results/`` tree produced by ``run_all.py`` and lets the
 * user inspect:
 *
 *   1. **Best models per ticker** — what landed in ``_summary.csv``.
 *
 *   2. **News vs no-news comparison** — for each per-ticker CSV, pair the
 *      "+ News" model variant with its no-news sibling and report the
 *      accuracy / return / Sharpe delta. Same logic as
 *      ``scripts/news_impact.py`` but evaluated in the browser so no extra
 *      backend round-trip is needed.
 *
 *   3. **Cross-run comparison** — pick two directories (e.g. a VADER run
 *      and a FinBERT run, or 20-day vs 50-day holdout) and diff their
 *      summary numbers per ticker.
 *
 * All numeric coercion happens client-side because the backend just hands
 * over the raw CSV rows as strings.
 */

import { useMemo, useState } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import { api } from "../lib/api";
import { s, Panel, pct } from "../components/ui";

// ----------------------------------------------------------------------
// Constants — kept in sync with scripts/news_impact.py PAIRS.
// ----------------------------------------------------------------------

/** Map "+ News" variant → its price-only sibling. */
const PAIRS: Record<string, string> = {
  "k-NN TW + News": "k-NN Time-Weighted",
  "k-NN Enh. TW + News": "k-NN Enh. TW",
  "LinReg TW + News": "LinReg Time-Weighted",
  "LinReg Enh. TW + News": "LinReg Enh. TW",
  "LSTM + News": "LSTM",
};

/** Metrics to diff. Each `higher` is true means "larger value is better".
 *  max_drawdown is negative; closer to 0 = better, so a "higher" value is
 *  literally larger (less negative) → higher_is_better = true. */
const METRICS: { key: string; label: string; higher: boolean }[] = [
  { key: "accuracy", label: "Accuracy", higher: true },
  { key: "total_return", label: "Return", higher: true },
  { key: "profit_factor", label: "PF", higher: true },
  { key: "max_drawdown", label: "Max DD", higher: true },
  { key: "sharpe_ratio", label: "Sharpe", higher: true },
  { key: "sortino_ratio", label: "Sortino", higher: true },
];

// ----------------------------------------------------------------------
// Pure helpers (mirror scripts/news_impact.py)
// ----------------------------------------------------------------------

type Row = Record<string, string>;

function num(v: string | undefined): number | null {
  if (v === undefined || v === "" || v === "nan" || v === "None") return null;
  const f = Number(v);
  return Number.isFinite(f) ? f : null;
}

interface Pair {
  ticker: string;
  period: string;
  model_family: string;
  // Per-metric base / news / delta. `null` means data missing.
  metrics: Record<string, { base: number | null; news: number | null; delta: number | null }>;
  // Convenience booleans for the three headline metrics
  return_news_wins: boolean | null;
  accuracy_news_wins: boolean | null;
  sharpe_news_wins: boolean | null;
}

function pairRows(rows: Row[]): Pair[] {
  // Group by (period, model_name) so we can pair newsName with baseName.
  const byPeriod = new Map<string, Map<string, Row>>();
  for (const r of rows) {
    const period = r.period ?? "";
    const model = r.model ?? "";
    if (!period || !model) continue;
    if (!byPeriod.has(period)) byPeriod.set(period, new Map());
    byPeriod.get(period)!.set(model, r);
  }

  const out: Pair[] = [];
  for (const [period, models] of byPeriod) {
    for (const [newsName, baseName] of Object.entries(PAIRS)) {
      const baseRow = models.get(baseName);
      const newsRow = models.get(newsName);
      if (!baseRow || !newsRow) continue;

      const metrics: Pair["metrics"] = {};
      for (const m of METRICS) {
        const b = num(baseRow[m.key]);
        const n = num(newsRow[m.key]);
        const delta = b === null || n === null ? null : m.higher ? n - b : b - n;
        metrics[m.key] = { base: b, news: n, delta };
      }
      const bw = (key: string) => {
        const b = metrics[key].base;
        const n = metrics[key].news;
        if (b === null || n === null) return null;
        if (n > b) return true;
        if (b > n) return false;
        return null;
      };
      out.push({
        ticker: baseRow.ticker ?? newsRow.ticker ?? "",
        period,
        model_family: baseName,
        metrics,
        return_news_wins: bw("total_return"),
        accuracy_news_wins: bw("accuracy"),
        sharpe_news_wins: bw("sharpe_ratio"),
      });
    }
  }
  return out;
}

function overallStats(pairs: Pair[]) {
  const winRate = (sel: (p: Pair) => boolean | null) => {
    const defined = pairs.filter((p) => sel(p) !== null);
    const wins = defined.filter((p) => sel(p) === true);
    return defined.length === 0
      ? { wins: 0, defined: 0, rate: null as number | null }
      : { wins: wins.length, defined: defined.length, rate: wins.length / defined.length };
  };
  const deltas = (key: string) =>
    pairs.map((p) => p.metrics[key]?.delta).filter((d): d is number => d !== null && d !== undefined);
  const median = (xs: number[]): number | null => {
    if (xs.length === 0) return null;
    const sorted = [...xs].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };
  const mean = (xs: number[]): number | null => (xs.length === 0 ? null : xs.reduce((a, b) => a + b, 0) / xs.length);
  return {
    total: pairs.length,
    return: winRate((p) => p.return_news_wins),
    accuracy: winRate((p) => p.accuracy_news_wins),
    sharpe: winRate((p) => p.sharpe_news_wins),
    return_median: median(deltas("total_return")),
    return_mean: mean(deltas("total_return")),
    accuracy_median: median(deltas("accuracy")),
  };
}

// ----------------------------------------------------------------------
// Component
// ----------------------------------------------------------------------

type View = "summary" | "news" | "compare";

export default function Analysis() {
  const { data: dirs, isLoading } = useQuery({
    queryKey: ["resultsDirs"],
    queryFn: api.listResultsDirs,
  });

  const [selDir, setSelDir] = useState<string | null>(null);
  const [view, setView] = useState<View>("summary");
  const [compareDir, setCompareDir] = useState<string | null>(null);

  // Auto-pick the most recently modified directory once they load.
  useMemo(() => {
    if (!selDir && dirs && dirs.length > 0) setSelDir(dirs[0].name);
  }, [dirs, selDir]);

  const selected = useMemo(() => dirs?.find((d) => d.name === selDir), [dirs, selDir]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>Analysis</h2>

      {/* Directory picker */}
      <Panel title="Pick a results directory">
        {isLoading ? (
          <div style={{ padding: 24, textAlign: "center", color: s.muted }}>Loading…</div>
        ) : !dirs?.length ? (
          <div style={{ padding: 24, textAlign: "center", color: s.muted, fontSize: 13 }}>
            No runs found under <code style={code}>results/</code>.
            <br />
            Generate some with <code style={code}>uv run python run_all.py --stocks --days 20 --buy-hold</code>.
          </div>
        ) : (
          <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 4 }}>
            {dirs.map((d) => (
              <button
                key={d.name}
                onClick={() => setSelDir(d.name)}
                style={{
                  textAlign: "left", padding: "8px 12px", borderRadius: 6, cursor: "pointer",
                  background: d.name === selDir ? "rgba(59,130,246,0.15)" : "transparent",
                  border: `1px solid ${d.name === selDir ? s.accent : s.border}`,
                  color: s.text, fontSize: 12, fontFamily: s.mono,
                  display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between",
                }}
              >
                <span>{d.name}</span>
                <span style={{ fontSize: 10, color: s.muted, display: "flex", gap: 10 }}>
                  <span>{d.csv_count} CSV{d.csv_count !== 1 ? "s" : ""}</span>
                  {d.has_news_impact && <span style={{ color: s.green }}>news-impact ✓</span>}
                  <span>{fmtDate(d.modified)}</span>
                </span>
              </button>
            ))}
          </div>
        )}
      </Panel>

      {selected && (
        <>
          {/* View switcher */}
          <div style={{ display: "flex", gap: 6 }}>
            <Tab label="Best models" active={view === "summary"} onClick={() => setView("summary")} />
            <Tab label="News vs no-news" active={view === "news"} onClick={() => setView("news")} />
            <Tab label="Compare runs" active={view === "compare"} onClick={() => setView("compare")} />
          </div>

          {view === "summary" && <SummaryView dirName={selected.name} hasSummary={selected.has_summary} />}
          {view === "news" && <NewsView dirName={selected.name} tickerCsvs={selected.ticker_csvs} />}
          {view === "compare" && (
            <CompareView
              dirs={dirs ?? []}
              dirA={selected.name}
              dirB={compareDir}
              setDirB={setCompareDir}
            />
          )}
        </>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------
// Sub-views
// ----------------------------------------------------------------------

function SummaryView({ dirName, hasSummary }: { dirName: string; hasSummary: boolean }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["resultCsv", dirName, "_summary"],
    queryFn: () => api.readResultCsv(dirName, "_summary"),
    enabled: hasSummary,
    retry: false,
  });

  if (!hasSummary) {
    return (
      <Panel title="Best models per ticker">
        <div style={{ padding: 24, color: s.muted, fontSize: 13, textAlign: "center" }}>
          No <code style={code}>_summary.csv</code> in this directory. Re-run with{" "}
          <code style={code}>run_all.py</code> to generate one.
        </div>
      </Panel>
    );
  }
  if (isLoading) return <Panel title="Best models per ticker"><Loading /></Panel>;
  if (error) return <Panel title="Best models per ticker"><Err msg={String(error)} /></Panel>;

  const rows = data?.rows ?? [];
  return (
    <Panel title={`Best models per ticker (${rows.length})`}>
      <div style={{ overflowX: "auto" }}>
        <table style={tbl}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${s.border}` }}>
              <th style={th}>Ticker</th>
              <th style={th}>Best model</th>
              <th style={th}>Period</th>
              <th style={th}>Return</th>
              <th style={th}>Accuracy</th>
              <th style={th}>PF</th>
              <th style={th}>Sharpe</th>
              <th style={th}>Max DD</th>
              <th style={th}>vs B&amp;H</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const ret = num(r.total_return) ?? 0;
              const bh = num(r.buy_hold_return) ?? 0;
              return (
                <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}
                >
                  <td style={td}>{r.ticker}</td>
                  <td style={td}>{r.model}</td>
                  <td style={tdMuted}>{r.period}</td>
                  <td style={{ ...td, textAlign: "center", color: ret >= 0 ? s.green : s.red }}>{pct(ret * 100)}</td>
                  <td style={{ ...tdMuted, textAlign: "center" }}>{pct((num(r.accuracy) ?? 0) * 100, false)}</td>
                  <td style={{ ...tdMuted, textAlign: "center" }}>{fmtPF(num(r.profit_factor))}</td>
                  <td style={{ ...tdMuted, textAlign: "center" }}>{(num(r.sharpe_ratio) ?? 0).toFixed(2)}</td>
                  <td style={{ ...td, textAlign: "center", color: s.red }}>{pct((num(r.max_drawdown) ?? 0) * 100)}</td>
                  <td style={{ ...td, textAlign: "center", color: ret > bh ? s.green : s.red, fontWeight: 700 }}>
                    {ret > bh ? "✓" : "✗"} {pct(bh * 100)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function NewsView({ dirName, tickerCsvs }: { dirName: string; tickerCsvs: string[] }) {
  // Fetch every per-ticker CSV in parallel.
  const queries = useQueries({
    queries: tickerCsvs.map((t) => ({
      queryKey: ["resultCsv", dirName, t],
      queryFn: () => api.readResultCsv(dirName, t),
    })),
  });

  const allLoaded = queries.every((q) => !q.isLoading);
  const pairs = useMemo(() => {
    if (!allLoaded) return [];
    const out: Pair[] = [];
    for (const q of queries) {
      const rows = (q.data?.rows as Row[] | undefined) ?? [];
      out.push(...pairRows(rows));
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allLoaded, queries.map((q) => q.dataUpdatedAt).join(",")]);

  const stats = useMemo(() => overallStats(pairs), [pairs]);

  if (!allLoaded) return <Panel title="News vs no-news"><Loading /></Panel>;
  if (pairs.length === 0) {
    return (
      <Panel title="News vs no-news">
        <div style={{ padding: 24, color: s.muted, fontSize: 13, textAlign: "center" }}>
          No paired "+ News" / no-news rows found in this run.
          <br />
          Make sure the run was produced with news enabled (the default).
        </div>
      </Panel>
    );
  }

  // Best / worst 5 by return delta
  const ranked = pairs
    .filter((p) => p.metrics.total_return.delta !== null)
    .sort((a, b) => (b.metrics.total_return.delta ?? 0) - (a.metrics.total_return.delta ?? 0));
  const top = ranked.slice(0, 5);
  const bottom = ranked.slice(-5).reverse();

  return (
    <>
      <Panel title="Overall news impact">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <Stat label="Pairs compared" value={String(stats.total)} />
            <Stat label="Return news-wins" value={fmtRate(stats.return)} hint={`${stats.return.wins}/${stats.return.defined}`} />
            <Stat label="Accuracy news-wins" value={fmtRate(stats.accuracy)} hint={`${stats.accuracy.wins}/${stats.accuracy.defined}`} />
            <Stat label="Sharpe news-wins" value={fmtRate(stats.sharpe)} hint={`${stats.sharpe.wins}/${stats.sharpe.defined}`} />
          </div>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <Stat label="Median Δreturn" value={stats.return_median !== null ? pct(stats.return_median * 100) : "—"} />
            <Stat label="Mean Δreturn" value={stats.return_mean !== null ? pct(stats.return_mean * 100) : "—"} />
            <Stat label="Median Δaccuracy" value={stats.accuracy_median !== null ? (stats.accuracy_median >= 0 ? "+" : "") + stats.accuracy_median.toFixed(4) : "—"} />
          </div>
        </div>
      </Panel>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <Panel title="Top 5 — news helped most">
          <DeltaTable pairs={top} />
        </Panel>
        <Panel title="Bottom 5 — news hurt most">
          <DeltaTable pairs={bottom} />
        </Panel>
      </div>

      <Panel title={`All paired comparisons (${pairs.length})`}>
        <div style={{ overflowX: "auto", maxHeight: 600 }}>
          <table style={tbl}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${s.border}` }}>
                <th style={th}>Ticker</th>
                <th style={th}>Period</th>
                <th style={th}>Model family</th>
                <th style={th}>Acc Δ</th>
                <th style={th}>Return Δ</th>
                <th style={th}>Sharpe Δ</th>
                <th style={th}>News wins?</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}
                >
                  <td style={td}>{p.ticker}</td>
                  <td style={tdMuted}>{p.period}</td>
                  <td style={td}>{p.model_family}</td>
                  <td style={deltaCell(p.metrics.accuracy.delta)}>{fmtDelta(p.metrics.accuracy.delta, 4)}</td>
                  <td style={deltaCell(p.metrics.total_return.delta)}>{fmtPctDelta(p.metrics.total_return.delta)}</td>
                  <td style={deltaCell(p.metrics.sharpe_ratio.delta)}>{fmtDelta(p.metrics.sharpe_ratio.delta, 2)}</td>
                  <td style={{ ...td, textAlign: "center", color: p.return_news_wins ? s.green : s.red }}>
                    {p.return_news_wins === null ? "—" : p.return_news_wins ? "✓" : "✗"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

function CompareView({ dirs, dirA, dirB, setDirB }: {
  dirs: { name: string; modified: string }[];
  dirA: string;
  dirB: string | null;
  setDirB: (d: string | null) => void;
}) {
  const otherDirs = dirs.filter((d) => d.name !== dirA);
  const { data: aSummary } = useQuery({
    queryKey: ["resultCsv", dirA, "_summary"],
    queryFn: () => api.readResultCsv(dirA, "_summary"),
    retry: false,
  });
  const { data: bSummary } = useQuery({
    queryKey: ["resultCsv", dirB, "_summary"],
    queryFn: () => api.readResultCsv(dirB!, "_summary"),
    enabled: !!dirB,
    retry: false,
  });

  // Diff by ticker — show the same ticker side-by-side from both runs
  const rowsA: Row[] = aSummary?.rows ?? [];
  const rowsB: Row[] = bSummary?.rows ?? [];
  const byTickerA = new Map(rowsA.map((r) => [r.ticker, r] as const));
  const byTickerB = new Map(rowsB.map((r) => [r.ticker, r] as const));
  const allTickers = Array.from(new Set([...byTickerA.keys(), ...byTickerB.keys()])).sort();

  return (
    <>
      <Panel title="Compare against another run">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 12, color: s.muted }}>
            <strong>A:</strong>{" "}<code style={code}>{dirA}</code>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 12, color: s.muted }}>B:</span>
            <select
              value={dirB ?? ""}
              onChange={(e) => setDirB(e.target.value || null)}
              style={{ padding: "6px 10px", borderRadius: 6, fontSize: 13, background: s.surface, color: s.text, border: `1px solid ${s.border}` }}
            >
              <option value="">— pick a run —</option>
              {otherDirs.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </div>
        </div>
      </Panel>

      {dirB && (
        <Panel title={`Δ Return per ticker — ${dirA} vs ${dirB}`}>
          <div style={{ overflowX: "auto" }}>
            <table style={tbl}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={th}>Ticker</th>
                  <th style={th}>A: model</th>
                  <th style={th}>A: return</th>
                  <th style={th}>B: model</th>
                  <th style={th}>B: return</th>
                  <th style={th}>Δ Return</th>
                  <th style={th}>Δ Sharpe</th>
                  <th style={th}>Better in</th>
                </tr>
              </thead>
              <tbody>
                {allTickers.map((t) => {
                  const a = byTickerA.get(t);
                  const b = byTickerB.get(t);
                  const aRet = num(a?.total_return);
                  const bRet = num(b?.total_return);
                  const dRet = aRet !== null && bRet !== null ? bRet - aRet : null;
                  const aSh = num(a?.sharpe_ratio);
                  const bSh = num(b?.sharpe_ratio);
                  const dSh = aSh !== null && bSh !== null ? bSh - aSh : null;
                  const winner = dRet === null ? "—" : dRet > 0 ? "B" : dRet < 0 ? "A" : "tie";
                  return (
                    <tr key={t} style={{ borderBottom: `1px solid ${s.border}` }}>
                      <td style={td}>{t}</td>
                      <td style={tdMuted}>{a?.model ?? "—"}</td>
                      <td style={{ ...td, textAlign: "center", color: (aRet ?? 0) >= 0 ? s.green : s.red }}>{aRet !== null ? pct(aRet * 100) : "—"}</td>
                      <td style={tdMuted}>{b?.model ?? "—"}</td>
                      <td style={{ ...td, textAlign: "center", color: (bRet ?? 0) >= 0 ? s.green : s.red }}>{bRet !== null ? pct(bRet * 100) : "—"}</td>
                      <td style={deltaCell(dRet)}>{fmtPctDelta(dRet)}</td>
                      <td style={deltaCell(dSh)}>{fmtDelta(dSh, 2)}</td>
                      <td style={{ ...td, textAlign: "center", color: winner === "B" ? s.green : winner === "A" ? s.accent : s.muted, fontWeight: 700 }}>{winner}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </>
  );
}

// ----------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------

function Tab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "8px 14px", borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: "pointer",
        border: `1px solid ${active ? s.accent : s.border}`,
        background: active ? "rgba(59,130,246,0.15)" : "transparent",
        color: active ? s.accent : s.muted,
      }}
    >{label}</button>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ fontSize: 11, color: s.muted, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: s.text, fontFamily: s.mono }}>{value}</div>
      {hint && <div style={{ fontSize: 10, color: s.muted }}>{hint}</div>}
    </div>
  );
}

function DeltaTable({ pairs }: { pairs: Pair[] }) {
  return (
    <div style={{ padding: 8 }}>
      <table style={{ ...tbl, fontSize: 11 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${s.border}` }}>
            <th style={th}>Ticker</th>
            <th style={th}>Period</th>
            <th style={th}>Model</th>
            <th style={th}>Δ Return</th>
            <th style={th}>Δ Acc</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((p, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${s.border}` }}>
              <td style={td}>{p.ticker}</td>
              <td style={tdMuted}>{p.period}</td>
              <td style={td}>{p.model_family}</td>
              <td style={deltaCell(p.metrics.total_return.delta)}>{fmtPctDelta(p.metrics.total_return.delta)}</td>
              <td style={deltaCell(p.metrics.accuracy.delta)}>{fmtDelta(p.metrics.accuracy.delta, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Loading() { return <div style={{ padding: 24, textAlign: "center", color: s.muted }}>Loading…</div>; }
function Err({ msg }: { msg: string }) { return <div style={{ padding: 24, color: s.red, fontSize: 12 }}>{msg}</div>; }

function fmtDate(iso: string): string {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}
function fmtRate(r: { rate: number | null }): string {
  return r.rate === null ? "—" : `${(r.rate * 100).toFixed(0)}%`;
}
function fmtPF(v: number | null): string {
  if (v === null) return "—";
  if (v >= 100) return "∞";
  return v.toFixed(2);
}
function fmtDelta(v: number | null, dp: number): string {
  if (v === null) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(dp);
}
function fmtPctDelta(v: number | null): string {
  if (v === null) return "—";
  return pct(v * 100);
}
function deltaCell(v: number | null): React.CSSProperties {
  const color = v === null || v === 0 ? s.muted : v > 0 ? s.green : s.red;
  return { ...td, textAlign: "center", color, fontWeight: 700 };
}

const tbl: React.CSSProperties = { width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: s.mono };
const th: React.CSSProperties = { padding: "8px 10px", fontSize: 11, fontWeight: 500, color: s.muted, textAlign: "left", borderBottom: `1px solid ${s.border}`, position: "sticky", top: 0, background: s.surface };
const td: React.CSSProperties = { padding: "6px 10px", color: s.text };
const tdMuted: React.CSSProperties = { padding: "6px 10px", color: s.muted };
const code: React.CSSProperties = { padding: "1px 5px", borderRadius: 3, background: s.hover, fontFamily: s.mono, fontSize: 11 };
