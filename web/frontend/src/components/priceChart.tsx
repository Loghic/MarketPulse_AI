/**
 * PriceChart.tsx – Reusable zoomable SVG price chart.
 *
 * Features:
 *   - Line / Candlestick toggle
 *   - Scroll wheel zoom
 *   - Drag-select zoom
 *   - Navigation bar (pan scrollbar) at bottom
 *   - Hover crosshair + OHLCV tooltip
 *   - Volume bars background
 *   - Responsive (SVG viewBox scales)
 *
 * Usage:
 *   <PriceChart rows={ohlcvRows} height={350} />
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { s } from "./ui";

export interface ChartRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Props {
  rows: ChartRow[];
  height?: number;
}

export default function PriceChart({ rows, height = 350 }: Props) {
  const [chartType, setChartType] = useState<"line" | "candle">("line");
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [zoomStart, setZoomStart] = useState(0);
  const [zoomEnd, setZoomEnd] = useState(rows.length - 1);
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [dragEnd, setDragEnd] = useState<number | null>(null);

  // Nav bar drag state
  const [navDragging, setNavDragging] = useState(false);
  const [navDragAnchor, setNavDragAnchor] = useState(0);
  const navRef = useRef<SVGSVGElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Reset zoom on data change
  const len = rows.length;
  const prevLen = useRef(len);
  useEffect(() => {
    if (len !== prevLen.current) {
      prevLen.current = len;
      setZoomStart(0);
      setZoomEnd(len - 1);
    }
  }, [len]);

  if (rows.length < 2) {
    return <div style={{ padding: 24, color: s.muted }}>Not enough data</div>;
  }

  const visible = rows.slice(zoomStart, zoomEnd + 1);
  const isZoomed = zoomStart !== 0 || zoomEnd !== rows.length - 1;

  // Chart dimensions
  const width = 780;
  const navHeight = 40;
  const pad = { top: 20, right: 60, bottom: 40, left: 10 };
  const cw = width - pad.left - pad.right;
  const ch = height - pad.top - pad.bottom;

  const closes = visible.map((r) => r.close);
  const minP = Math.min(...visible.map((r) => r.low));
  const maxP = Math.max(...visible.map((r) => r.high));
  const range = maxP - minP || 1;
  const maxVol = Math.max(...visible.map((r) => r.volume));

  const toX = (i: number) => pad.left + (i / Math.max(visible.length - 1, 1)) * cw;
  const toY = (v: number) => pad.top + (1 - (v - minP) / range) * ch;
  const xToIdx = (x: number) => Math.round(((x - pad.left) / cw) * (visible.length - 1));

  const overallUp = closes[closes.length - 1] >= closes[0];
  const lineColor = overallUp ? s.green : s.red;

  const gridCount = 5;
  const gridStep = range / gridCount;

  // SVG coordinate from mouse event
  const getSvgX = (e: React.MouseEvent, ref: React.RefObject<SVGSVGElement | null>, w: number) => {
    if (!ref.current) return 0;
    const rect = ref.current.getBoundingClientRect();
    return ((e.clientX - rect.left) / rect.width) * w;
  };

  // Main chart mouse handlers
  const handleMouseMove = (e: React.MouseEvent) => {
    const x = getSvgX(e, svgRef, width);
    const idx = xToIdx(x);
    if (idx >= 0 && idx < visible.length) setHoverIdx(idx);
    if (dragStart !== null) {
      setDragEnd(Math.min(Math.max(idx, 0), visible.length - 1));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const x = getSvgX(e, svgRef, width);
    const idx = xToIdx(x);
    setDragStart(Math.min(Math.max(idx, 0), visible.length - 1));
    setDragEnd(null);
  };

  const handleMouseUp = () => {
    if (dragStart !== null && dragEnd !== null && Math.abs(dragEnd - dragStart) > 3) {
      const lo = Math.min(dragStart, dragEnd);
      const hi = Math.max(dragStart, dragEnd);
      setZoomStart(zoomStart + lo);
      setZoomEnd(zoomStart + hi);
    }
    setDragStart(null);
    setDragEnd(null);
  };

  // Wheel zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.2 : 0.8;
      const curLen = zoomEnd - zoomStart;
      const mid = (zoomStart + zoomEnd) / 2;
      const newLen = Math.max(10, Math.min(rows.length, curLen * factor));
      setZoomStart(Math.max(0, Math.round(mid - newLen / 2)));
      setZoomEnd(Math.min(rows.length - 1, Math.round(mid + newLen / 2)));
    },
    [zoomStart, zoomEnd, rows.length],
  );

  const zoomIn = () => handleWheel({ deltaY: -100, preventDefault: () => {} } as React.WheelEvent);
  const zoomOut = () => handleWheel({ deltaY: 100, preventDefault: () => {} } as React.WheelEvent);
  const resetZoom = () => { setZoomStart(0); setZoomEnd(rows.length - 1); };

  // Nav bar handlers
  const navW = width;
  const thumbLeft = (zoomStart / (rows.length - 1)) * cw + pad.left;
  const thumbRight = (zoomEnd / (rows.length - 1)) * cw + pad.left;
  const thumbWidth = Math.max(20, thumbRight - thumbLeft);

  const handleNavMouseDown = (e: React.MouseEvent) => {
    const x = getSvgX(e, navRef, navW);
    // Check if click is inside thumb
    if (x >= thumbLeft && x <= thumbLeft + thumbWidth) {
      setNavDragging(true);
      setNavDragAnchor(x - thumbLeft);
    } else {
      // Click outside thumb — center zoom there
      const pct = (x - pad.left) / cw;
      const idx = Math.round(pct * (rows.length - 1));
      const halfLen = Math.round((zoomEnd - zoomStart) / 2);
      setZoomStart(Math.max(0, idx - halfLen));
      setZoomEnd(Math.min(rows.length - 1, idx + halfLen));
    }
  };

  const handleNavMouseMove = (e: React.MouseEvent) => {
    if (!navDragging) return;
    const x = getSvgX(e, navRef, navW);
    const newLeft = x - navDragAnchor;
    const pct = (newLeft - pad.left) / cw;
    const newStart = Math.round(pct * (rows.length - 1));
    const len = zoomEnd - zoomStart;
    const clampedStart = Math.max(0, Math.min(rows.length - 1 - len, newStart));
    setZoomStart(clampedStart);
    setZoomEnd(clampedStart + len);
  };

  const handleNavMouseUp = () => setNavDragging(false);

  // Mini chart for nav bar (full data overview)
  const allCloses = rows.map((r) => r.close);
  const allMin = Math.min(...allCloses);
  const allMax = Math.max(...allCloses);
  const allRange = allMax - allMin || 1;

  const hoverRow = hoverIdx !== null ? visible[hoverIdx] : null;
  const selX1 = dragStart !== null ? toX(Math.min(dragStart, dragEnd ?? dragStart)) : 0;
  const selX2 = dragEnd !== null ? toX(Math.max(dragStart!, dragEnd)) : 0;

  return (
    <div style={{ padding: "8px 16px 8px" }}>
      {/* Toolbar */}
      <div style={{ display: "flex", gap: 4, marginBottom: 8, alignItems: "center", flexWrap: "wrap" }}>
        {(["line", "candle"] as const).map((t) => (
          <button key={t} onClick={() => setChartType(t)} style={{
            padding: "3px 10px", borderRadius: 4, fontSize: 11,
            border: `1px solid ${s.border}`,
            background: chartType === t ? s.border : "transparent",
            color: chartType === t ? s.text : s.muted, cursor: "pointer",
          }}>
            {t === "line" ? "Line" : "Candle"}
          </button>
        ))}
        <ZoomBtn label="+" onClick={zoomIn} />
        <ZoomBtn label="−" onClick={zoomOut} />
        {isZoomed && <ZoomBtn label="Reset" onClick={resetZoom} wide />}

        {isZoomed && (
          <span style={{ fontSize: 11, color: s.accent, fontFamily: s.mono }}>
            {visible[0]?.date.slice(0, 10)} → {visible[visible.length - 1]?.date.slice(0, 10)} ({visible.length}d)
          </span>
        )}

        {hoverRow && (
          <div style={{ marginLeft: "auto", fontSize: 11, color: s.muted, fontFamily: s.mono }}>
            {hoverRow.date.slice(0, 10)} O:{hoverRow.open.toFixed(2)} H:{hoverRow.high.toFixed(2)} L:{hoverRow.low.toFixed(2)} C:{hoverRow.close.toFixed(2)} V:{hoverRow.volume.toLocaleString()}
          </div>
        )}
      </div>

      {/* Main chart */}
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", cursor: dragStart !== null ? "col-resize" : "crosshair", touchAction: "none" }}
        onMouseMove={handleMouseMove} onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp} onMouseLeave={() => { setHoverIdx(null); handleMouseUp(); }}
        onWheel={handleWheel}
      >
        {/* Grid */}
        {Array.from({ length: gridCount + 1 }, (_, i) => {
          const price = minP + i * gridStep;
          const y = toY(price);
          return (
            <g key={`g${i}`}>
              <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke={s.border} strokeWidth={0.5} />
              <text x={width - pad.right + 6} y={y + 4} fill={s.muted} fontSize={10} fontFamily={s.mono}>${price.toFixed(2)}</text>
            </g>
          );
        })}

        {/* Date labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
          const idx = Math.floor(pct * (visible.length - 1));
          const row = visible[idx];
          if (!row) return null;
          return <text key={`d${pct}`} x={toX(idx)} y={height - 8} fill={s.muted} fontSize={10} fontFamily={s.mono} textAnchor="middle">{row.date.slice(0, 10)}</text>;
        })}

        {/* Volume */}
        {visible.map((r, i) => {
          const barW = Math.max(1, cw / visible.length - 1);
          const barH = (r.volume / maxVol) * 50;
          return <rect key={`v${i}`} x={toX(i) - barW / 2} y={pad.top + ch - barH} width={barW} height={barH}
            fill={r.close >= r.open ? s.green : s.red} opacity={0.07} />;
        })}

        {/* Line */}
        {chartType === "line" && (
          <>
            <polygon points={`${toX(0)},${pad.top + ch} ${closes.map((c, i) => `${toX(i)},${toY(c)}`).join(" ")} ${toX(closes.length - 1)},${pad.top + ch}`}
              fill={lineColor} fillOpacity={0.06} />
            <polyline points={closes.map((c, i) => `${toX(i)},${toY(c)}`).join(" ")}
              fill="none" stroke={lineColor} strokeWidth={1.5} />
          </>
        )}

        {/* Candles */}
        {chartType === "candle" && visible.map((r, i) => {
          const green = r.close >= r.open;
          const color = green ? s.green : s.red;
          const barW = Math.max(2, (cw / visible.length) * 0.7);
          const bodyTop = toY(Math.max(r.open, r.close));
          const bodyBot = toY(Math.min(r.open, r.close));
          return (
            <g key={`c${i}`}>
              <line x1={toX(i)} y1={toY(r.high)} x2={toX(i)} y2={toY(r.low)} stroke={color} strokeWidth={1} />
              <rect x={toX(i) - barW / 2} y={bodyTop} width={barW} height={Math.max(1, bodyBot - bodyTop)}
                fill={color} fillOpacity={green ? 0.3 : 0.8} stroke={color} strokeWidth={0.5} />
            </g>
          );
        })}

        {/* Drag selection */}
        {dragStart !== null && dragEnd !== null && (
          <rect x={selX1} y={pad.top} width={Math.abs(selX2 - selX1)} height={ch}
            fill={s.accent} fillOpacity={0.15} stroke={s.accent} strokeWidth={0.5} />
        )}

        {/* Hover crosshair */}
        {hoverIdx !== null && dragStart === null && (
          <>
            <line x1={toX(hoverIdx)} y1={pad.top} x2={toX(hoverIdx)} y2={pad.top + ch}
              stroke={s.muted} strokeWidth={0.5} strokeDasharray="3,3" />
            <circle cx={toX(hoverIdx)} cy={toY(closes[hoverIdx])} r={3}
              fill={lineColor} stroke={s.surface} strokeWidth={1.5} />
          </>
        )}
      </svg>

      {/* Navigation bar (pan scrollbar) */}
      <svg ref={navRef} viewBox={`0 0 ${navW} ${navHeight}`}
        style={{ width: "100%", cursor: navDragging ? "grabbing" : "pointer", marginTop: 4 }}
        onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove}
        onMouseUp={handleNavMouseUp} onMouseLeave={handleNavMouseUp}
      >
        {/* Background: mini line chart of full data */}
        <rect x={pad.left} y={0} width={cw} height={navHeight} fill={s.surface} stroke={s.border} strokeWidth={0.5} rx={3} />
        <polyline
          points={allCloses.map((c, i) => {
            const x = pad.left + (i / (allCloses.length - 1)) * cw;
            const y = 4 + (1 - (c - allMin) / allRange) * (navHeight - 8);
            return `${x},${y}`;
          }).join(" ")}
          fill="none" stroke={s.muted} strokeWidth={0.8} opacity={0.5}
        />

        {/* Dimmed area outside selection */}
        <rect x={pad.left} y={0} width={thumbLeft - pad.left} height={navHeight} fill="#000" opacity={0.3} />
        <rect x={thumbLeft + thumbWidth} y={0} width={cw - (thumbLeft + thumbWidth - pad.left)} height={navHeight} fill="#000" opacity={0.3} />

        {/* Thumb (draggable viewport) */}
        <rect x={thumbLeft} y={1} width={thumbWidth} height={navHeight - 2}
          fill={s.accent} fillOpacity={0.15} stroke={s.accent} strokeWidth={1} rx={3} />

        {/* Grab handles */}
        <line x1={thumbLeft + thumbWidth / 2 - 4} y1={navHeight / 2 - 4} x2={thumbLeft + thumbWidth / 2 - 4} y2={navHeight / 2 + 4} stroke={s.accent} strokeWidth={1.5} strokeLinecap="round" />
        <line x1={thumbLeft + thumbWidth / 2} y1={navHeight / 2 - 4} x2={thumbLeft + thumbWidth / 2} y2={navHeight / 2 + 4} stroke={s.accent} strokeWidth={1.5} strokeLinecap="round" />
        <line x1={thumbLeft + thumbWidth / 2 + 4} y1={navHeight / 2 - 4} x2={thumbLeft + thumbWidth / 2 + 4} y2={navHeight / 2 + 4} stroke={s.accent} strokeWidth={1.5} strokeLinecap="round" />
      </svg>

      <div style={{ fontSize: 10, color: s.muted, marginTop: 4, textAlign: "center" }}>
        Scroll to zoom · Drag chart to select range · Drag bar to pan
      </div>
    </div>
  );
}

function ZoomBtn({ label, onClick, wide }: { label: string; onClick: () => void; wide?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: "3px 8px", borderRadius: 4, fontSize: wide ? 11 : 13, fontWeight: 700,
      width: wide ? "auto" : 28, border: `1px solid ${s.border}`,
      background: "transparent", color: s.muted, cursor: "pointer",
    }}>
      {label}
    </button>
  );
}
