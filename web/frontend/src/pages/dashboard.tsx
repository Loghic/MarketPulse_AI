import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { OHLCVRow, TickerInfo } from "../lib/api";

const PERIODS = ["1mo", "1y", "2y", "5y", "max"] as const;

const s = {
  surface: "#111827",
  border: "#1e293b",
  hover: "#1a2235",
  text: "#e2e8f0",
  muted: "#94a3b8",
  accent: "#3b82f6",
  green: "#22c55e",
  red: "#ef4444",
  mono: "'JetBrains Mono', monospace",
};

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState<string>("1y");
  const qc = useQueryClient();

  const { data: tickers } = useQuery({
    queryKey: ["tickers"],
    queryFn: api.getTickers,
  });

  const {
    data: tickerData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["tickerData", ticker, period],
    queryFn: () => api.getTickerData(ticker, period, 2000),
    enabled: !!ticker,
  });

  const refreshOne = useMutation({
    mutationFn: () => api.refresh([ticker]),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tickerData"] });
      qc.invalidateQueries({ queryKey: ["tickers"] });
    },
  });

  const refreshAll = useMutation({
    mutationFn: () => api.refresh([]),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tickerData"] });
      qc.invalidateQueries({ queryKey: ["tickers"] });
    },
  });

  const rows = tickerData?.data ?? [];
  const chartRows = useMemo(() => [...rows].reverse(), [rows]);
  const currentTicker = tickers?.find((t: TickerInfo) => t.ticker === ticker);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Controls bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <TickerSelect
          tickers={tickers ?? []}
          value={ticker}
          onChange={setTicker}
        />
        <PeriodTabs value={period} onChange={setPeriod} />
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <Btn
            onClick={() => refreshOne.mutate()}
            loading={refreshOne.isPending}
            label={`Update ${ticker}`}
          />
          <Btn
            onClick={() => refreshAll.mutate()}
            loading={refreshAll.isPending}
            label="Update All"
            secondary
          />
        </div>
      </div>

      {/* Ticker info bar */}
      {currentTicker && (
        <div style={{ fontSize: 12, color: s.muted }}>
          {currentTicker.asset_type.toUpperCase()} ·{" "}
          {currentTicker.rows.toLocaleString()} rows in DB
          {currentTicker.last_date && ` · Last: ${currentTicker.last_date}`}
        </div>
      )}

      {/* Stats cards */}
      {rows.length > 1 && <StatsCards rows={rows} />}

      {/* Chart */}
      <Panel title={`${ticker} — ${period.toUpperCase()}`}>
        {isLoading ? (
          <div
            style={{
              height: 350,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: s.muted,
            }}
          >
            Loading...
          </div>
        ) : error ? (
          <div style={{ padding: 24, color: s.red }}>
            Error: {(error as Error).message}. Try &quot;Update {ticker}&quot;
            first.
          </div>
        ) : (
          <PriceChart rows={chartRows} height={350} />
        )}
      </Panel>

      {/* OHLCV Table */}
      <Panel title={`OHLCV Data (${rows.length.toLocaleString()} rows)`}>
        <DataTable rows={rows} />
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reusable UI                                                         */
/* ------------------------------------------------------------------ */

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: s.surface,
        border: `1px solid ${s.border}`,
        borderRadius: 8,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "12px 16px",
          borderBottom: `1px solid ${s.border}`,
          fontSize: 13,
          fontWeight: 600,
          color: s.muted,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function TickerSelect({
  tickers,
  value,
  onChange,
}: {
  tickers: TickerInfo[];
  value: string;
  onChange: (v: string) => void;
}) {
  const stocks = tickers.filter((t) => t.asset_type === "stock");
  const crypto = tickers.filter((t) => t.asset_type === "crypto");
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: "8px 12px",
        borderRadius: 6,
        fontSize: 14,
        fontWeight: 600,
        background: s.surface,
        color: s.text,
        border: `1px solid ${s.border}`,
        cursor: "pointer",
        minWidth: 140,
      }}
    >
      {stocks.length > 0 && (
        <optgroup label="Stocks">
          {stocks.map((t) => (
            <option key={t.ticker} value={t.ticker}>
              {t.ticker}
            </option>
          ))}
        </optgroup>
      )}
      {crypto.length > 0 && (
        <optgroup label="Crypto">
          {crypto.map((t) => (
            <option key={t.ticker} value={t.ticker}>
              {t.ticker}
            </option>
          ))}
        </optgroup>
      )}
    </select>
  );
}

function PeriodTabs({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {PERIODS.map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          style={{
            padding: "6px 14px",
            borderRadius: 6,
            fontSize: 12,
            fontWeight: 600,
            border: `1px solid ${s.border}`,
            background: value === p ? s.accent : "transparent",
            color: value === p ? "#fff" : s.muted,
            cursor: "pointer",
          }}
        >
          {p.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function Btn({
  onClick,
  loading,
  label,
  secondary,
}: {
  onClick: () => void;
  loading: boolean;
  label: string;
  secondary?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      style={{
        padding: "6px 16px",
        borderRadius: 6,
        fontSize: 12,
        fontWeight: 600,
        border: secondary ? `1px solid ${s.border}` : "none",
        background: secondary ? "transparent" : s.accent,
        color: secondary ? s.muted : "#fff",
        cursor: loading ? "wait" : "pointer",
        opacity: loading ? 0.6 : 1,
      }}
    >
      {loading ? "Updating..." : label}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Stats Cards                                                         */
/* ------------------------------------------------------------------ */

function StatsCards({ rows }: { rows: OHLCVRow[] }) {
  const latest = rows[0];
  const prev = rows[1];
  if (!latest || !prev) return null;

  const change = latest.close - prev.close;
  const changePct = (change / prev.close) * 100;
  const isUp = change >= 0;

  const high = Math.max(...rows.slice(0, 252).map((r) => r.high));
  const low = Math.min(...rows.slice(0, 252).map((r) => r.low));
  const avgVol = Math.round(
    rows
      .slice(0, 20)
      .reduce((sum, r) => sum + r.volume, 0) / Math.min(rows.length, 20),
  );
  const totalReturn =
    ((latest.close - rows[rows.length - 1].close) /
      rows[rows.length - 1].close) *
    100;

  const cards = [
    { label: "Last Close", value: `$${latest.close.toFixed(2)}`, color: s.text },
    {
      label: "Day Change",
      value: `${isUp ? "+" : ""}${change.toFixed(2)} (${isUp ? "+" : ""}${changePct.toFixed(2)}%)`,
      color: isUp ? s.green : s.red,
    },
    { label: "Period High", value: `$${high.toFixed(2)}`, color: s.text },
    { label: "Period Low", value: `$${low.toFixed(2)}`, color: s.text },
    { label: "Avg Vol (20d)", value: avgVol.toLocaleString(), color: s.text },
    {
      label: "Period Return",
      value: `${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}%`,
      color: totalReturn >= 0 ? s.green : s.red,
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(6, 1fr)",
        gap: 10,
      }}
    >
      {cards.map((c) => (
        <div
          key={c.label}
          style={{
            background: s.surface,
            border: `1px solid ${s.border}`,
            borderRadius: 8,
            padding: "10px 14px",
          }}
        >
          <div style={{ fontSize: 11, color: s.muted, marginBottom: 4 }}>
            {c.label}
          </div>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: c.color,
              fontFamily: s.mono,
            }}
          >
            {c.value}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Price Chart (SVG — line + candle)                                    */
/* ------------------------------------------------------------------ */

function PriceChart({
  rows,
  height = 350,
}: {
  rows: OHLCVRow[];
  height?: number;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [chartType, setChartType] = useState<"line" | "candle">("line");

  if (rows.length < 2) {
    return (
      <div style={{ padding: 24, color: s.muted }}>Not enough data</div>
    );
  }

  const width = 780;
  const pad = { top: 20, right: 60, bottom: 40, left: 10 };
  const cw = width - pad.left - pad.right;
  const ch = height - pad.top - pad.bottom;

  const closes = rows.map((r) => r.close);
  const highs = rows.map((r) => r.high);
  const lows = rows.map((r) => r.low);
  const minP = Math.min(...lows);
  const maxP = Math.max(...highs);
  const range = maxP - minP || 1;

  const toX = (i: number) => pad.left + (i / (rows.length - 1)) * cw;
  const toY = (v: number) => pad.top + (1 - (v - minP) / range) * ch;

  const isUp = closes[closes.length - 1] >= closes[0];
  const lineColor = isUp ? s.green : s.red;
  const maxVol = Math.max(...rows.map((r) => r.volume));

  const gridCount = 5;
  const gridStep = range / gridCount;

  const hoverRow = hoverIdx !== null ? rows[hoverIdx] : null;

  return (
    <div style={{ padding: "8px 16px 16px" }}>
      {/* Controls + hover tooltip */}
      <div style={{ display: "flex", gap: 4, marginBottom: 8, alignItems: "center" }}>
        {(["line", "candle"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setChartType(t)}
            style={{
              padding: "3px 10px",
              borderRadius: 4,
              fontSize: 11,
              border: `1px solid ${s.border}`,
              background: chartType === t ? s.border : "transparent",
              color: chartType === t ? s.text : s.muted,
              cursor: "pointer",
            }}
          >
            {t === "line" ? "Line" : "Candle"}
          </button>
        ))}
        {hoverRow && (
          <div
            style={{
              marginLeft: "auto",
              fontSize: 11,
              color: s.muted,
              fontFamily: s.mono,
            }}
          >
            {hoverRow.date.slice(0, 10)} · O:{hoverRow.open.toFixed(2)} H:
            {hoverRow.high.toFixed(2)} L:{hoverRow.low.toFixed(2)} C:
            {hoverRow.close.toFixed(2)} V:{hoverRow.volume.toLocaleString()}
          </div>
        )}
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", cursor: "crosshair" }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * width;
          const idx = Math.round(
            ((x - pad.left) / cw) * (rows.length - 1),
          );
          if (idx >= 0 && idx < rows.length) setHoverIdx(idx);
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Grid lines + price labels */}
        {Array.from({ length: gridCount + 1 }, (_, i) => {
          const price = minP + i * gridStep;
          const y = toY(price);
          return (
            <g key={`g${i}`}>
              <line
                x1={pad.left} y1={y} x2={width - pad.right} y2={y}
                stroke={s.border} strokeWidth={0.5}
              />
              <text
                x={width - pad.right + 6} y={y + 4}
                fill={s.muted} fontSize={10} fontFamily={s.mono}
              >
                ${price.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Date labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
          const idx = Math.floor(pct * (rows.length - 1));
          const row = rows[idx];
          if (!row) return null;
          return (
            <text
              key={`d${pct}`} x={toX(idx)} y={height - 8}
              fill={s.muted} fontSize={10} fontFamily={s.mono} textAnchor="middle"
            >
              {row.date.slice(0, 10)}
            </text>
          );
        })}

        {/* Volume bars (background) */}
        {rows.map((r, i) => {
          const barW = Math.max(1, cw / rows.length - 1);
          const barH = (r.volume / maxVol) * 50;
          return (
            <rect
              key={`v${i}`}
              x={toX(i) - barW / 2} y={pad.top + ch - barH}
              width={barW} height={barH}
              fill={r.close >= r.open ? s.green : s.red} opacity={0.07}
            />
          );
        })}

        {/* Line chart */}
        {chartType === "line" && (
          <>
            <polygon
              points={`${toX(0)},${pad.top + ch} ${closes.map((c, i) => `${toX(i)},${toY(c)}`).join(" ")} ${toX(closes.length - 1)},${pad.top + ch}`}
              fill={lineColor} fillOpacity={0.06}
            />
            <polyline
              points={closes.map((c, i) => `${toX(i)},${toY(c)}`).join(" ")}
              fill="none" stroke={lineColor} strokeWidth={1.5}
            />
          </>
        )}

        {/* Candlestick chart */}
        {chartType === "candle" &&
          rows.map((r, i) => {
            const green = r.close >= r.open;
            const color = green ? s.green : s.red;
            const barW = Math.max(2, (cw / rows.length) * 0.7);
            const bodyTop = toY(Math.max(r.open, r.close));
            const bodyBot = toY(Math.min(r.open, r.close));
            return (
              <g key={`c${i}`}>
                <line
                  x1={toX(i)} y1={toY(r.high)} x2={toX(i)} y2={toY(r.low)}
                  stroke={color} strokeWidth={1}
                />
                <rect
                  x={toX(i) - barW / 2} y={bodyTop}
                  width={barW} height={Math.max(1, bodyBot - bodyTop)}
                  fill={color} fillOpacity={green ? 0.3 : 0.8}
                  stroke={color} strokeWidth={0.5}
                />
              </g>
            );
          })}

        {/* Hover crosshair */}
        {hoverIdx !== null && (
          <>
            <line
              x1={toX(hoverIdx)} y1={pad.top}
              x2={toX(hoverIdx)} y2={pad.top + ch}
              stroke={s.muted} strokeWidth={0.5} strokeDasharray="3,3"
            />
            <circle
              cx={toX(hoverIdx)} cy={toY(closes[hoverIdx])}
              r={3} fill={lineColor} stroke={s.surface} strokeWidth={1.5}
            />
          </>
        )}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sortable, Filterable, Paginated OHLCV Table                         */
/* ------------------------------------------------------------------ */

type SortKey = keyof OHLCVRow;

function DataTable({ rows }: { rows: OHLCVRow[] }) {
  const [sortCol, setSortCol] = useState<SortKey>("date");
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState("");
  const perPage = 30;

  const filtered = useMemo(() => {
    if (!filter) return rows;
    const q = filter.toLowerCase();
    return rows.filter((r) => r.date.toLowerCase().includes(q));
  }, [rows, filter]);

  const sorted = useMemo(
    () =>
      [...filtered].sort((a, b) => {
        const va = a[sortCol];
        const vb = b[sortCol];
        const cmp = va < vb ? -1 : va > vb ? 1 : 0;
        return sortAsc ? cmp : -cmp;
      }),
    [filtered, sortCol, sortAsc],
  );

  const totalPages = Math.ceil(sorted.length / perPage);
  const paged = sorted.slice(page * perPage, (page + 1) * perPage);

  const handleSort = (col: SortKey) => {
    if (col === sortCol) setSortAsc(!sortAsc);
    else {
      setSortCol(col);
      setSortAsc(col === "date" ? false : true);
    }
    setPage(0);
  };

  const cols: {
    key: SortKey;
    label: string;
    align: string;
    fmt: (v: number | string) => string;
  }[] = [
    {
      key: "date",
      label: "Date",
      align: "left",
      fmt: (v) => String(v).slice(0, 10),
    },
    {
      key: "open",
      label: "Open",
      align: "right",
      fmt: (v) => `$${Number(v).toFixed(2)}`,
    },
    {
      key: "high",
      label: "High",
      align: "right",
      fmt: (v) => `$${Number(v).toFixed(2)}`,
    },
    {
      key: "low",
      label: "Low",
      align: "right",
      fmt: (v) => `$${Number(v).toFixed(2)}`,
    },
    {
      key: "close",
      label: "Close",
      align: "right",
      fmt: (v) => `$${Number(v).toFixed(2)}`,
    },
    {
      key: "volume",
      label: "Volume",
      align: "right",
      fmt: (v) => Number(v).toLocaleString(),
    },
  ];

  const thStyle = (col: SortKey): React.CSSProperties => ({
    padding: "8px 14px",
    textAlign: cols.find((c) => c.key === col)?.align as "left" | "right",
    color: sortCol === col ? s.text : s.muted,
    fontWeight: 500,
    fontSize: 12,
    cursor: "pointer",
    userSelect: "none",
    borderBottom: `1px solid ${s.border}`,
    position: "sticky",
    top: 0,
    background: s.surface,
  });

  return (
    <div>
      {/* Filter */}
      <div
        style={{
          padding: "8px 14px",
          display: "flex",
          alignItems: "center",
          gap: 12,
          borderBottom: `1px solid ${s.border}`,
        }}
      >
        <input
          placeholder="Filter by date..."
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setPage(0);
          }}
          style={{
            padding: "5px 10px",
            borderRadius: 4,
            fontSize: 12,
            background: "transparent",
            border: `1px solid ${s.border}`,
            color: s.text,
            width: 180,
            outline: "none",
          }}
        />
        <span style={{ fontSize: 11, color: s.muted }}>
          {filtered.length === rows.length
            ? `${rows.length} rows`
            : `${filtered.length} / ${rows.length} rows`}
        </span>
      </div>

      {/* Scrollable table */}
      <div style={{ maxHeight: 500, overflowY: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: s.mono,
            fontSize: 13,
          }}
        >
          <thead>
            <tr>
              {cols.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={thStyle(col.key)}
                >
                  {col.label}
                  {sortCol === col.key && (
                    <span style={{ marginLeft: 4, opacity: 0.6 }}>
                      {sortAsc ? "↑" : "↓"}
                    </span>
                  )}
                </th>
              ))}
              <th style={{ ...thStyle("close"), cursor: "default" }}>Δ%</th>
            </tr>
          </thead>
          <tbody>
            {paged.map((row, i) => {
              const nextRow = sorted[page * perPage + i + 1];
              const dayChange = nextRow
                ? ((row.close - nextRow.close) / nextRow.close) * 100
                : 0;
              const up = dayChange >= 0;
              return (
                <tr
                  key={`${row.date}-${i}`}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = s.hover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "";
                  }}
                  style={{ borderBottom: `1px solid ${s.border}` }}
                >
                  {cols.map((col) => (
                    <td
                      key={col.key}
                      style={{
                        padding: "6px 14px",
                        textAlign: col.align as "left" | "right",
                      }}
                    >
                      {col.fmt(row[col.key])}
                    </td>
                  ))}
                  <td
                    style={{
                      padding: "6px 14px",
                      textAlign: "right",
                      color: up ? s.green : s.red,
                      fontSize: 12,
                    }}
                  >
                    {nextRow
                      ? `${up ? "+" : ""}${dayChange.toFixed(2)}%`
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          style={{
            padding: "10px 14px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderTop: `1px solid ${s.border}`,
          }}
        >
          <span style={{ fontSize: 12, color: s.muted }}>
            Page {page + 1} of {totalPages}
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            <PgBtn
              label="« First"
              onClick={() => setPage(0)}
              disabled={page === 0}
            />
            <PgBtn
              label="‹ Prev"
              onClick={() => setPage(page - 1)}
              disabled={page === 0}
            />
            <PgBtn
              label="Next ›"
              onClick={() => setPage(page + 1)}
              disabled={page >= totalPages - 1}
            />
            <PgBtn
              label="Last »"
              onClick={() => setPage(totalPages - 1)}
              disabled={page >= totalPages - 1}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function PgBtn({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "4px 10px",
        borderRadius: 4,
        fontSize: 11,
        border: `1px solid ${s.border}`,
        background: "transparent",
        color: disabled ? s.border : s.muted,
        cursor: disabled ? "default" : "pointer",
      }}
    >
      {label}
    </button>
  );
}
