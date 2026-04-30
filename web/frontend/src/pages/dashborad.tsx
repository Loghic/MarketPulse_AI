import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, OHLCVRow } from "../lib/api";

const PERIODS = ["1mo", "1y", "2y", "5y", "max"];

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState("1y");
  const queryClient = useQueryClient();

  const { data: tickers } = useQuery({
    queryKey: ["tickers"],
    queryFn: api.getTickers,
  });

  const { data: tickerData, isLoading } = useQuery({
    queryKey: ["tickerData", ticker, period],
    queryFn: () => api.getTickerData(ticker, period, 1000),
    enabled: !!ticker,
  });

  const refreshMutation = useMutation({
    mutationFn: (t: string[]) => api.refresh(t),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tickerData"] });
      queryClient.invalidateQueries({ queryKey: ["tickers"] });
    },
  });

  const rows = tickerData?.data ?? [];
  // Chart data needs ascending order
  const chartRows = [...rows].reverse();

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="px-3 py-2 rounded-md text-sm font-medium"
          style={{ background: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)" }}
        >
          {tickers?.map((t) => (
            <option key={t.ticker} value={t.ticker}>
              {t.ticker} ({t.asset_type})
            </option>
          ))}
        </select>

        <div className="flex gap-1">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className="px-3 py-1.5 rounded text-xs font-medium transition-colors"
              style={{
                background: period === p ? "var(--c-accent)" : "var(--c-surface)",
                color: period === p ? "#fff" : "var(--c-text-muted)",
                border: "1px solid var(--c-border)",
              }}
            >
              {p}
            </button>
          ))}
        </div>

        <div className="flex gap-2 ml-auto">
          <button
            onClick={() => refreshMutation.mutate([ticker])}
            disabled={refreshMutation.isPending}
            className="px-4 py-1.5 rounded text-xs font-medium transition-colors"
            style={{ background: "var(--c-accent)", color: "#fff" }}
          >
            {refreshMutation.isPending ? "Updating..." : `Update ${ticker}`}
          </button>
          <button
            onClick={() => refreshMutation.mutate([])}
            disabled={refreshMutation.isPending}
            className="px-4 py-1.5 rounded text-xs font-medium transition-colors"
            style={{ background: "var(--c-surface)", color: "var(--c-text-muted)", border: "1px solid var(--c-border)" }}
          >
            {refreshMutation.isPending ? "Updating..." : "Update All"}
          </button>
        </div>
      </div>

      {/* Price Chart */}
      <div
        className="rounded-lg p-4"
        style={{ background: "var(--c-surface)", border: "1px solid var(--c-border)" }}
      >
        <h2 className="text-sm font-semibold mb-3" style={{ color: "var(--c-text-muted)" }}>
          {ticker} — {period.toUpperCase()}
        </h2>
        {isLoading ? (
          <div className="h-80 flex items-center justify-center" style={{ color: "var(--c-text-muted)" }}>
            Loading chart...
          </div>
        ) : (
          <PriceChart rows={chartRows} ticker={ticker} />
        )}
      </div>

      {/* Stats cards */}
      {rows.length > 0 && <StatsCards rows={rows} />}

      {/* Data Table */}
      <div
        className="rounded-lg overflow-hidden"
        style={{ background: "var(--c-surface)", border: "1px solid var(--c-border)" }}
      >
        <div className="p-4 flex items-center justify-between" style={{ borderBottom: "1px solid var(--c-border)" }}>
          <h2 className="text-sm font-semibold" style={{ color: "var(--c-text-muted)" }}>
            OHLCV Data ({rows.length} rows)
          </h2>
        </div>
        <DataTable rows={rows} />
      </div>
    </div>
  );
}


function StatsCards({ rows }: { rows: OHLCVRow[] }) {
  const latest = rows[0];
  const prev = rows[1];
  if (!latest || !prev) return null;

  const change = latest.close - prev.close;
  const changePct = (change / prev.close) * 100;
  const isUp = change >= 0;

  const high = Math.max(...rows.map((r) => r.high));
  const low = Math.min(...rows.map((r) => r.low));
  const avgVol = Math.round(rows.reduce((s, r) => s + r.volume, 0) / rows.length);

  const cards = [
    { label: "Last Close", value: `$${latest.close.toFixed(2)}` },
    {
      label: "Change",
      value: `${isUp ? "+" : ""}${change.toFixed(2)} (${isUp ? "+" : ""}${changePct.toFixed(2)}%)`,
      color: isUp ? "var(--c-green)" : "var(--c-red)",
    },
    { label: "Period High", value: `$${high.toFixed(2)}` },
    { label: "Period Low", value: `$${low.toFixed(2)}` },
    { label: "Avg Volume", value: avgVol.toLocaleString() },
  ];

  return (
    <div className="grid grid-cols-5 gap-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className="rounded-lg p-3"
          style={{ background: "var(--c-surface)", border: "1px solid var(--c-border)" }}
        >
          <div className="text-xs" style={{ color: "var(--c-text-muted)" }}>{c.label}</div>
          <div className="text-lg font-semibold mt-1" style={{ color: c.color || "var(--c-text)", fontFamily: "var(--font-mono)" }}>
            {c.value}
          </div>
        </div>
      ))}
    </div>
  );
}


function PriceChart({ rows, ticker }: { rows: OHLCVRow[]; ticker: string }) {
  // Dynamic import would be cleaner, but for now inline
  // In production, use lazy loading for Plotly
  if (rows.length === 0) return <div>No data</div>;

  const dates = rows.map((r) => r.date);
  const closes = rows.map((r) => r.close);
  const volumes = rows.map((r) => r.volume);

  return (
    <div>
      <svg viewBox={`0 0 800 300`} className="w-full" style={{ fontFamily: "var(--font-mono)" }}>
        {/* Simple line chart — replaced with Plotly.js once npm installed */}
        <PriceLine dates={dates} values={closes} width={800} height={240} />
        <text x="10" y="290" fill="var(--c-text-muted)" fontSize="10">
          {dates[0]} → {dates[dates.length - 1]} | {rows.length} data points
        </text>
      </svg>
    </div>
  );
}


function PriceLine({ values, width, height }: {
  dates: string[];
  values: number[];
  width: number;
  height: number;
}) {
  if (values.length < 2) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pad = 20;

  const points = values
    .map((v, i) => {
      const x = pad + (i / (values.length - 1)) * (width - 2 * pad);
      const y = pad + (1 - (v - min) / range) * (height - 2 * pad);
      return `${x},${y}`;
    })
    .join(" ");

  const isUp = values[values.length - 1] >= values[0];
  const color = isUp ? "var(--c-green)" : "var(--c-red)";

  // Area fill
  const firstX = pad;
  const lastX = pad + ((values.length - 1) / (values.length - 1)) * (width - 2 * pad);
  const areaPoints = `${firstX},${height - pad} ${points} ${lastX},${height - pad}`;

  return (
    <g>
      <polygon points={areaPoints} fill={color} fillOpacity="0.08" />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
      {/* Price labels */}
      <text x={width - pad} y={pad - 5} textAnchor="end" fill="var(--c-text-muted)" fontSize="11">
        ${max.toFixed(2)}
      </text>
      <text x={width - pad} y={height - pad + 15} textAnchor="end" fill="var(--c-text-muted)" fontSize="11">
        ${min.toFixed(2)}
      </text>
    </g>
  );
}


function DataTable({ rows }: { rows: OHLCVRow[] }) {
  const [sortCol, setSortCol] = useState<keyof OHLCVRow>("date");
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(0);
  const perPage = 25;

  const sorted = [...rows].sort((a, b) => {
    const va = a[sortCol];
    const vb = b[sortCol];
    const cmp = va < vb ? -1 : va > vb ? 1 : 0;
    return sortAsc ? cmp : -cmp;
  });

  const paged = sorted.slice(page * perPage, (page + 1) * perPage);
  const totalPages = Math.ceil(sorted.length / perPage);

  const handleSort = (col: keyof OHLCVRow) => {
    if (col === sortCol) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(false);
    }
  };

  const cols: { key: keyof OHLCVRow; label: string; fmt: (v: number | string) => string }[] = [
    { key: "date", label: "Date", fmt: (v) => String(v) },
    { key: "open", label: "Open", fmt: (v) => `$${Number(v).toFixed(2)}` },
    { key: "high", label: "High", fmt: (v) => `$${Number(v).toFixed(2)}` },
    { key: "low", label: "Low", fmt: (v) => `$${Number(v).toFixed(2)}` },
    { key: "close", label: "Close", fmt: (v) => `$${Number(v).toFixed(2)}` },
    { key: "volume", label: "Volume", fmt: (v) => Number(v).toLocaleString() },
  ];

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" style={{ fontFamily: "var(--font-mono)" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--c-border)" }}>
              {cols.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="px-4 py-2 text-left cursor-pointer select-none"
                  style={{ color: "var(--c-text-muted)", fontWeight: 500 }}
                >
                  {col.label}
                  {sortCol === col.key && (
                    <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, i) => (
              <tr
                key={i}
                className="transition-colors"
                style={{ borderBottom: "1px solid var(--c-border)" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--c-surface-hover)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "")}
              >
                {cols.map((col) => (
                  <td key={col.key} className="px-4 py-2">
                    {col.fmt(row[col.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          className="flex items-center justify-between px-4 py-3"
          style={{ borderTop: "1px solid var(--c-border)" }}
        >
          <span className="text-xs" style={{ color: "var(--c-text-muted)" }}>
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="px-3 py-1 rounded text-xs"
              style={{
                background: "var(--c-surface)",
                color: page === 0 ? "var(--c-border)" : "var(--c-text-muted)",
                border: "1px solid var(--c-border)",
              }}
            >
              ← Prev
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 rounded text-xs"
              style={{
                background: "var(--c-surface)",
                color: page >= totalPages - 1 ? "var(--c-border)" : "var(--c-text-muted)",
                border: "1px solid var(--c-border)",
              }}
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
