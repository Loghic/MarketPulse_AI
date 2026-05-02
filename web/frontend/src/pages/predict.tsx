import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import PriceChart from "../components/PriceChart";
import DataTable from "../components/DataTable";
import type { Column } from "../components/DataTable";
import { s, Panel, Btn, pct } from "../components/ui";

const PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const CHART_PERIODS = ["1mo", "1y", "2y", "5y", "max"];

const ALL_MODELS = [
  "k-NN", "k-NN (TW)", "k-NN Enhanced", "k-NN Enhanced (TW)",
  "LinReg", "LinReg (TW)", "LinReg Enhanced", "LinReg Enhanced (TW)",
  "LSTM",
];

const PRESETS: { label: string; items: { model: string; news: boolean }[] }[] = [
  { label: "All", items: ALL_MODELS.map((m) => ({ model: m, news: false })) },
  { label: "All + News", items: ALL_MODELS.map((m) => ({ model: m, news: true })) },
  { label: "k-NN family", items: ["k-NN", "k-NN (TW)", "k-NN Enhanced", "k-NN Enhanced (TW)"].map((m) => ({ model: m, news: false })) },
  { label: "k-NN + News", items: ["k-NN", "k-NN (TW)", "k-NN Enhanced", "k-NN Enhanced (TW)"].map((m) => ({ model: m, news: true })) },
  { label: "LinReg family", items: ["LinReg", "LinReg (TW)", "LinReg Enhanced", "LinReg Enhanced (TW)"].map((m) => ({ model: m, news: false })) },
  { label: "LinReg + News", items: ["LinReg", "LinReg (TW)", "LinReg Enhanced", "LinReg Enhanced (TW)"].map((m) => ({ model: m, news: true })) },
  { label: "LSTM", items: [{ model: "LSTM", news: false }] },
  { label: "LSTM + News", items: [{ model: "LSTM", news: true }] },
];

interface BuilderRow {
  model: string;
  period: string;
  news: boolean;
}

export default function Predict() {
  const [ticker, setTicker] = useState("AAPL");
  const [chartPeriod, setChartPeriod] = useState("1y");
  const [quickPeriod, setQuickPeriod] = useState("1y");
  const [showChart, setShowChart] = useState(true);
  const [refreshData, setRefreshData] = useState(true);
  const [rows, setRows] = useState<BuilderRow[]>([
    { model: "k-NN Enhanced", period: "1y", news: false },
    { model: "LinReg Enhanced (TW)", period: "1y", news: true },
    { model: "LSTM", period: "max", news: false },
  ]);
  const [histDate, setHistDate] = useState("");

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
  const chartRows = useMemo(() => [...(tickerData?.data ?? [])].reverse(), [tickerData]);

  const nextDay = predictInfo?.next_trading_day ?? "—";
  const isCrypto = ticker.includes("-USD");

  // Run predictions
  const runMut = useMutation({
    mutationFn: () =>
      fetch("/api/predict/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          items: rows.map((r) => ({ model: r.model, period: r.period, news: r.news })),
          refresh_data: refreshData,
        }),
      }).then((r) => r.json()),
  });

  // Historical
  const histMut = useMutation({
    mutationFn: () =>
      fetch(`/api/predict/historical?ticker=${ticker}&date=${histDate}&period=${quickPeriod}`,
        { method: "POST" }).then((r) => r.json()),
  });

  const predictions = runMut.data?.predictions ?? [];
  const consensus = runMut.data?.consensus;

  // Quick add preset
  const addPreset = (preset: typeof PRESETS[number]) => {
    const newRows = preset.items.map((item) => ({
      model: item.model, period: quickPeriod, news: item.news,
    }));
    setRows(newRows);
  };

  // Builder helpers
  const updateRow = (i: number, field: keyof BuilderRow, value: string | boolean) => {
    const next = [...rows];
    next[i] = { ...next[i], [field]: value };
    setRows(next);
  };
  const removeRow = (i: number) => setRows(rows.filter((_, j) => j !== i));
  const addRow = () => setRows([...rows, { model: "k-NN", period: quickPeriod, news: false }]);

  // Results table columns
  const cols: Column<Record<string, unknown>>[] = [
    { key: "model", label: "Model" },
    { key: "prediction", label: "Direction", align: "right",
      color: (v) => (v === "UP" ? s.green : v === "DOWN" ? s.red : s.muted) },
    { key: "confidence", label: "Conf.", align: "right",
      fmt: (v) => (v as number) > 0 ? pct((v as number) * 100, false) : "—" },
    { key: "period", label: "Period", align: "right" },
    { key: "sentiment_score", label: "Sent.", align: "right",
      fmt: (v) => { const n = v as number; return n === 0 ? "—" : (n > 0 ? "+" : "") + n.toFixed(2); },
      color: (v) => { const n = v as number; return n > 0 ? s.green : n < 0 ? s.red : s.muted; } },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap" }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Predictions</h2>
        <div style={{ fontSize: 13, color: s.muted }}>
          Target: <strong style={{ color: s.accent }}>{isCrypto ? "tomorrow (24/7)" : nextDay}</strong>
        </div>
      </div>

      {/* Ticker + chart toggle */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: s.muted }}>Ticker</span>
        <select value={ticker} onChange={(e) => setTicker(e.target.value)} style={selSt}>
          {tickers?.filter((t) => t.asset_type === "stock").map((t) =>
            <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}
          {tickers?.filter((t) => t.asset_type === "crypto").map((t) =>
            <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}
        </select>
        <Chk label="Show chart" checked={showChart} onChange={setShowChart} />
        {showChart && (
          <>
            <span style={{ fontSize: 12, color: s.muted, marginLeft: 8 }}>Chart:</span>
            <Pills values={CHART_PERIODS} selected={chartPeriod} onSelect={setChartPeriod} />
          </>
        )}
        <Chk label="Update data first" checked={refreshData} onChange={setRefreshData} />
      </div>

      {/* Chart (optional) */}
      {showChart && chartRows.length > 0 && (
        <Panel title={`${ticker} — ${chartPeriod.toUpperCase()}`}
          extra={consensus ? <ConsBadge c={consensus} /> : undefined}>
          <PriceChart rows={chartRows} height={280} />
        </Panel>
      )}

      {/* Builder */}
      <Panel title="Prediction Builder">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Quick add */}
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: s.muted, minWidth: 60 }}>Quick add:</span>
            {PRESETS.map((p) => (
              <button key={p.label} onClick={() => addPreset(p)} style={{
                padding: "3px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, cursor: "pointer",
                border: `1px solid ${s.border}`, background: "transparent", color: s.muted,
              }}>{p.label}</button>
            ))}
          </div>

          {/* Quick period (used by presets and new rows) */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ fontSize: 12, color: s.muted, minWidth: 60 }}>Period:</span>
            <Pills values={PERIODS} selected={quickPeriod} onSelect={setQuickPeriod} />
          </div>

          {/* Separator */}
          <div style={{ borderTop: `1px solid ${s.border}`, paddingTop: 10 }}>
            <span style={{ fontSize: 11, color: s.muted }}>
              Models to run ({rows.length}):
            </span>
          </div>

          {/* Builder rows */}
          {rows.map((r, i) => (
            <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
              <select value={r.model} onChange={(e) => updateRow(i, "model", e.target.value)}
                style={{ ...selSt, minWidth: 170, fontSize: 12 }}>
                {ALL_MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
              <select value={r.period} onChange={(e) => updateRow(i, "period", e.target.value)}
                style={{ ...selSt, minWidth: 70, fontSize: 12 }}>
                {PERIODS.map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
              </select>
              <Chk label="News" checked={r.news} onChange={(v) => updateRow(i, "news", v)} />
              <button onClick={() => removeRow(i)} style={{
                padding: "3px 7px", borderRadius: 4, border: `1px solid ${s.border}`,
                background: "transparent", color: s.red, cursor: "pointer", fontSize: 11,
              }}>✕</button>
            </div>
          ))}

          {/* Add + Run */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button onClick={addRow} style={{
              padding: "4px 10px", borderRadius: 4, border: `1px solid ${s.border}`,
              background: "transparent", color: s.accent, cursor: "pointer", fontSize: 12,
            }}>+ Add model</button>
            <div style={{ marginLeft: "auto" }}>
              <Btn onClick={() => runMut.mutate()} loading={runMut.isPending}
                label={runMut.isPending ? "Running..." : `Run (${rows.length} models)`} />
            </div>
          </div>
        </div>
      </Panel>

      {runMut.isPending && <Loading text="Running models..." />}

      {/* Consensus summary */}
      {consensus && consensus.total > 0 && (
        <Panel title="Consensus">
          <div style={{ padding: 16, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            <ConsBar up={consensus.up} down={consensus.down} />
            <span style={{ fontFamily: s.mono, fontSize: 24, fontWeight: 700,
              color: consensus.direction === "UP" ? s.green : consensus.direction === "DOWN" ? s.red : s.muted }}>
              {consensus.direction}
            </span>
            <span style={{ fontSize: 13, color: s.muted }}>
              {consensus.up}↑ / {consensus.down}↓ · {pct(consensus.agreement * 100, false)} agreement
            </span>
          </div>
        </Panel>
      )}

      {/* Results table */}
      {predictions.length > 0 && (
        <Panel title={`Results (${predictions.length})`}>
          <DataTable rows={predictions} columns={cols} defaultSort="confidence" defaultAsc={false}
            perPage={25} filterKeys={["model", "prediction", "period"]}
            exportFilename={`${ticker}_predictions.csv`} />
        </Panel>
      )}

      {/* News */}
      {predictions.some((p: Record<string, unknown>) => ((p.headlines as string[])?.length ?? 0) > 0) && (
        <Panel title="News Headlines">
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {[...new Set(predictions.flatMap((p: Record<string, unknown>) => (p.headlines as string[]) ?? []))].map((h, i) => (
              <div key={i} style={{ fontSize: 12, color: s.text, padding: "4px 0",
                borderBottom: `1px solid ${s.border}` }}>{h}</div>
            ))}
          </div>
        </Panel>
      )}

      {/* Historical */}
      <Panel title="Historical Prediction">
        <div style={{ padding: 16, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: s.muted }}>Predict as if today were:</span>
          <input type="date" value={histDate} onChange={(e) => setHistDate(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 6, fontSize: 13, background: "transparent",
              border: `1px solid ${s.border}`, color: s.text, colorScheme: "dark" }} />
          <span style={{ fontSize: 12, color: s.muted }}>Period:</span>
          <select value={quickPeriod} onChange={(e) => setQuickPeriod(e.target.value)} style={{ ...selSt, fontSize: 12 }}>
            {PERIODS.map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
          <Btn onClick={() => histMut.mutate()} loading={histMut.isPending} label="Run" secondary={!histDate} />
          {histMut.data?.error && <span style={{ fontSize: 12, color: s.red }}>{histMut.data.error}</span>}
        </div>
        {(histMut.data?.predictions?.length ?? 0) > 0 && (
          <div style={{ padding: "0 16px 16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 6 }}>
              {histMut.data.predictions.map((p: Record<string, unknown>, i: number) => (
                <div key={i} style={{ padding: "6px 10px", borderRadius: 5, border: `1px solid ${s.border}`,
                  display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, color: s.text }}>{String(p.model)}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, fontFamily: s.mono,
                    color: p.prediction === "UP" ? s.green : s.red }}>
                    {String(p.prediction)} {typeof p.confidence === "number" && p.confidence > 0 ? pct(p.confidence * 100, false) : ""}
                  </span>
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

function Pills({ values, selected, onSelect }: { values: string[]; selected: string; onSelect: (v: string) => void }) {
  return (
    <div style={{ display: "flex", gap: 3 }}>
      {values.map((v) => (
        <button key={v} onClick={() => onSelect(v)} style={{
          padding: "4px 10px", borderRadius: 5, fontSize: 11, fontWeight: 600, cursor: "pointer",
          border: `1px solid ${s.border}`,
          background: selected === v ? s.accent : "transparent",
          color: selected === v ? "#fff" : s.muted,
        }}>{v.toUpperCase()}</button>
      ))}
    </div>
  );
}

function Chk({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", fontSize: 12, color: s.muted }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ accentColor: s.accent }} />
      {label}
    </label>
  );
}

function ConsBadge({ c }: { c: { direction: string; up: number; down: number } }) {
  const color = c.direction === "UP" ? s.green : c.direction === "DOWN" ? s.red : s.muted;
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <span style={{ fontSize: 11, color: s.muted }}>{c.up}↑ {c.down}↓</span>
      <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700,
        background: color + "20", color }}>{c.direction}</span>
    </div>
  );
}

function ConsBar({ up, down }: { up: number; down: number }) {
  const t = up + down || 1;
  return (
    <div style={{ width: 160, height: 10, borderRadius: 5, overflow: "hidden", display: "flex" }}>
      <div style={{ width: `${(up / t) * 100}%`, background: s.green, transition: "width 0.3s" }} />
      <div style={{ flex: 1, background: s.red }} />
    </div>
  );
}

function Loading({ text }: { text: string }) {
  return <div style={{ textAlign: "center", padding: 32, color: s.muted }}>{text}</div>;
}

const selSt: React.CSSProperties = {
  padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600,
  background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer",
};
