import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { OHLCVRow, TickerInfo } from "../lib/api";
import PriceChart from "../components/PriceChart";
import DataTable from "../components/DataTable";
import type { Column } from "../components/DataTable";
import { s, Panel, Btn, usd, pct } from "../components/ui";

const PERIODS = ["1mo", "1y", "2y", "5y", "max", "custom"] as const;

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState<string>("1y");
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const qc = useQueryClient();

  const { data: tickers } = useQuery({
    queryKey: ["tickers"],
    queryFn: api.getTickers,
  });

  const { data: tickerData, isLoading, error } = useQuery({
    queryKey: ["tickerData", ticker, period],
    queryFn: () => api.getTickerData(ticker, period === "custom" ? "max" : period),
    enabled: !!ticker,
  });

  const refreshOne = useMutation({
    mutationFn: () => api.refresh([ticker]),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tickerData"] });
      qc.invalidateQueries({ queryKey: ["tickers"] });
    },
    onError: (err) => console.error("Refresh failed:", err),
  });

  const refreshAll = useMutation({
    mutationFn: () => api.refresh([]),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tickerData"] });
      qc.invalidateQueries({ queryKey: ["tickers"] });
    },
    onError: (err) => console.error("Refresh all failed:", err),
  });

  // Filter for custom period
  const rows = useMemo(() => {
    const raw = tickerData?.data ?? [];
    if (period !== "custom" || !customFrom) return raw;
    return raw.filter((r) => {
      const d = r.date.slice(0, 10);
      if (customFrom && d < customFrom) return false;
      if (customTo && d > customTo) return false;
      return true;
    });
  }, [tickerData, period, customFrom, customTo]);

  // Chart needs ascending order
  const chartRows = useMemo(() => [...rows].reverse(), [rows]);
  const currentTicker = tickers?.find((t: TickerInfo) => t.ticker === ticker);

  // Enrich rows with delta %
  const enrichedRows = useMemo(() => {
    return rows.map((row, i) => {
      const prev = rows[i + 1]; // rows are descending
      const delta = prev ? ((row.close - prev.close) / prev.close) * 100 : null;
      return { ...row, delta };
    });
  }, [rows]);

  // Table columns
  const ohlcvColumns: Column<typeof enrichedRows[number]>[] = [
    { key: "date", label: "Date", fmt: (v) => String(v).slice(0, 10) },
    { key: "open", label: "Open", align: "right", fmt: (v) => usd(v as number) },
    { key: "high", label: "High", align: "right", fmt: (v) => usd(v as number) },
    { key: "low", label: "Low", align: "right", fmt: (v) => usd(v as number) },
    { key: "close", label: "Close", align: "right", fmt: (v) => usd(v as number) },
    { key: "volume", label: "Volume", align: "right", fmt: (v) => Number(v).toLocaleString() },
    {
      key: "delta",
      label: "Δ%",
      align: "right",
      fmt: (v) => (v != null ? pct(v as number) : "—"),
      sortValue: (row) => row.delta ?? 0,
      color: (v) => {
        if (v == null) return s.muted;
        return (v as number) >= 0 ? s.green : s.red;
      },
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <TickerSelect tickers={tickers ?? []} value={ticker} onChange={setTicker} />
        <PeriodTabs value={period} onChange={setPeriod} />
        {period === "custom" && (
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <DateInput value={customFrom} onChange={setCustomFrom} />
            <span style={{ color: s.muted }}>→</span>
            <DateInput value={customTo} onChange={setCustomTo} />
          </div>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <Btn onClick={() => refreshOne.mutate()} loading={refreshOne.isPending} label={`Update ${ticker}`} />
          <Btn onClick={() => refreshAll.mutate()} loading={refreshAll.isPending} label="Update All" secondary />
        </div>
      </div>

      {/* Ticker info */}
      {currentTicker && (
        <div style={{ fontSize: 12, color: s.muted }}>
          {currentTicker.asset_type.toUpperCase()} · {currentTicker.rows.toLocaleString()} rows
          {currentTicker.last_date && ` · Last: ${currentTicker.last_date}`}
        </div>
      )}

      {/* Stats */}
      {rows.length > 1 && <StatsCards rows={rows} />}

      {/* Chart */}
      <Panel title={`${ticker} — ${period === "custom" ? `${customFrom || "?"} → ${customTo || "now"}` : period.toUpperCase()}`}>
        {isLoading ? (
          <div style={{ height: 350, display: "flex", alignItems: "center", justifyContent: "center", color: s.muted }}>
            Loading...
          </div>
        ) : error ? (
          <div style={{ padding: 24, color: s.red }}>
            Error: {(error as Error).message}
          </div>
        ) : (
          <PriceChart rows={chartRows} height={350} />
        )}
      </Panel>

      {/* OHLCV Table */}
      <Panel title={`OHLCV Data (${rows.length.toLocaleString()} rows)`}>
        <DataTable
          rows={enrichedRows}
          columns={ohlcvColumns}
          defaultSort="date"
          defaultAsc={false}
          perPage={30}
          filterKeys={["date"]}
          exportFilename={`${ticker}_${period}_ohlcv.csv`}
        />
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Local helper components                                             */
/* ------------------------------------------------------------------ */

function TickerSelect({ tickers, value, onChange }: { tickers: TickerInfo[]; value: string; onChange: (v: string) => void }) {
  const stocks = tickers.filter((t) => t.asset_type === "stock");
  const crypto = tickers.filter((t) => t.asset_type === "crypto");
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={{
      padding: "8px 12px", borderRadius: 6, fontSize: 14, fontWeight: 600,
      background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer", minWidth: 140,
    }}>
      {stocks.length > 0 && <optgroup label="Stocks">{stocks.map((t) => <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}</optgroup>}
      {crypto.length > 0 && <optgroup label="Crypto">{crypto.map((t) => <option key={t.ticker} value={t.ticker}>{t.ticker}</option>)}</optgroup>}
    </select>
  );
}

function PeriodTabs({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
      {PERIODS.map((p) => (
        <button key={p} onClick={() => onChange(p)} style={{
          padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
          border: `1px solid ${s.border}`,
          background: value === p ? s.accent : "transparent",
          color: value === p ? "#fff" : s.muted, cursor: "pointer",
        }}>
          {p === "custom" ? "Custom" : p.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function DateInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input type="date" value={value} onChange={(e) => onChange(e.target.value)} style={{
      padding: "5px 8px", borderRadius: 4, fontSize: 12, background: "transparent",
      border: `1px solid ${s.border}`, color: s.text, colorScheme: "dark",
    }} />
  );
}

function StatsCards({ rows }: { rows: OHLCVRow[] }) {
  const latest = rows[0];
  const prev = rows[1];
  if (!latest || !prev) return null;

  const change = latest.close - prev.close;
  const changePct = (change / prev.close) * 100;
  const isUp = change >= 0;
  const high = Math.max(...rows.slice(0, 252).map((r) => r.high));
  const low = Math.min(...rows.slice(0, 252).map((r) => r.low));
  const avgVol = Math.round(rows.slice(0, 20).reduce((sum, r) => sum + r.volume, 0) / Math.min(rows.length, 20));
  const totalReturn = ((latest.close - rows[rows.length - 1].close) / rows[rows.length - 1].close) * 100;

  const cards = [
    { label: "Last Close", value: usd(latest.close), color: s.text },
    { label: "Day Change", value: `${isUp ? "+" : ""}${change.toFixed(2)} (${pct(changePct)})`, color: isUp ? s.green : s.red },
    { label: "Period High", value: usd(high), color: s.text },
    { label: "Period Low", value: usd(low), color: s.text },
    { label: "Avg Vol (20d)", value: avgVol.toLocaleString(), color: s.text },
    { label: "Period Return", value: pct(totalReturn), color: totalReturn >= 0 ? s.green : s.red },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
      {cards.map((c) => (
        <div key={c.label} style={{ background: s.surface, border: `1px solid ${s.border}`, borderRadius: 8, padding: "10px 14px" }}>
          <div style={{ fontSize: 11, color: s.muted, marginBottom: 4 }}>{c.label}</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: c.color, fontFamily: s.mono }}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}
