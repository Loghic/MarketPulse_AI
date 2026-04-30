/**
 * PriceChart.tsx – Reusable zoomable SVG price chart.
 *
 * Features: line/candle toggle, scroll zoom, drag-select zoom,
 * navigation bar (pan), hover crosshair + OHLCV tooltip, volume bars.
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
  const [navDragging, setNavDragging] = useState(false);
  const [navDragAnchor, setNavDragAnchor] = useState(0);
  const navRef = useRef<SVGSVGElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Reset zoom on data change
  const len = rows.length;
  useEffect(() => {
    setZoomStart(0);
    setZoomEnd(len - 1);
  }, [len]);

  // Wheel zoom via useEffect to avoid passive listener issue
  const zoomState = useRef({ zoomStart, zoomEnd });
  zoomState.current = { zoomStart, zoomEnd };

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const { zoomStart: zs, zoomEnd: ze } = zoomState.current;
      const factor = e.deltaY > 0 ? 1.2 : 0.8;
      const curLen = ze - zs;
      const mid = (zs + ze) / 2;
      const newLen = Math.max(10, Math.min(rows.length, curLen * factor));
      setZoomStart(Math.max(0, Math.round(mid - newLen / 2)));
      setZoomEnd(Math.min(rows.length - 1, Math.round(mid + newLen / 2)));
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [rows.length]);

  if (rows.length < 2) {
    return <div style={{ padding: 24, color: s.muted }}>Not enough data</div>;
  }

  const visible = rows.slice(zoomStart, zoomEnd + 1);
  const isZoomed = zoomStart !== 0 || zoomEnd !== rows.length - 1;

  const width = 780;
  const navHeight = 40;
  const pad = { top: 20, right: 60, bottom: 40, left: 10 };
  const cw = width - pad.left - pad.right;
  const ch = height - pad.top - pad.bottom;

  const closes = visible.map((r) => r.close);
  const minP = Math.min(...visible.map((r) => r.low));
  const maxP = Math.max(...visible.map((r) => r.high));
  const range = maxP - minP || 1;
  const maxVol = Math.max(...visible.map((r) => r.volume), 1);

  const n = Math.max(visible.length - 1, 1);
  const toX = (i: number) => pad.left + (i / n) * cw;
  const toY = (v: number) => pad.top + (1 - (v - minP) / range) * ch;
  const xToIdx = (x: number) => Math.round(((x - pad.left) / cw) * n);

  const overallUp = closes[closes.length - 1] >= closes[0];
  const lineColor = overallUp ? s.green : s.red;
  const gridCount = 5;
  const gridStep = range / gridCount;

  // SVG coordinate from mouse
  const getSvgX = (e: React.MouseEvent, ref: React.RefObject<SVGSVGElement | null>, w: number) => {
    if (!ref.current) return 0;
    const rect = ref.current.getBoundingClientRect();
    return ((e.clientX - rect.left) / rect.width) * w;
  };

  const clampIdx = (idx: number) => Math.min(Math.max(idx, 0), visible.length - 1);

  // Main chart mouse handlers
  const handleMouseMove = (e: React.MouseEvent) => {
    const x = getSvgX(e, svgRef, width);
    const idx = clampIdx(xToIdx(x));
    setHoverIdx(idx);
    if (dragStart !== null) setDragEnd(idx);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const x = getSvgX(e, svgRef, width);
    setDragStart(clampIdx(xToIdx(x)));
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

  const doZoom = useCallback((dir: number) => {
    const factor = dir > 0 ? 1.2 : 0.8;
    const curLen = zoomEnd - zoomStart;
    const mid = (zoomStart + zoomEnd) / 2;
    const newLen = Math.max(10, Math.min(rows.length, curLen * factor));
    setZoomStart(Math.max(0, Math.round(mid - newLen / 2)));
    setZoomEnd(Math.min(rows.length - 1, Math.round(mid + newLen / 2)));
  }, [zoomStart, zoomEnd, rows.length]);

  const resetZoom = () => { setZoomStart(0); setZoomEnd(rows.length - 1); };

  // Nav bar
  const totalN = Math.max(rows.length - 1, 1);
  const thumbLeft = (zoomStart / totalN) * cw + pad.left;
  const thumbRight = (zoomEnd / totalN) * cw + pad.left;
  const thumbWidth = Math.max(20, thumbRight - thumbLeft);

  const handleNavMouseDown = (e: React.MouseEvent) => {
    const x = getSvgX(e, navRef, width);
    if (x >= thumbLeft && x <= thumbLeft + thumbWidth) {
      setNavDragging(true);
      setNavDragAnchor(x - thumbLeft);
    } else {
      const pct = (x - pad.left) / cw;
      const idx = Math.round(pct * totalN);
      const halfLen = Math.round((zoomEnd - zoomStart) / 2);
      setZoomStart(Math.max(0, idx - halfLen));
      setZoomEnd(Math.min(rows.length - 1, idx + halfLen));
    }
  };

  const handleNavMouseMove = (e: React.MouseEvent) => {
    if (!navDragging) return;
    const x = getSvgX(e, navRef, width);
    const newLeft = x - navDragAnchor;
    const pct = (newLeft - pad.left) / cw;
    const newStart = Math.round(pct * totalN);
    const zLen = zoomEnd - zoomStart;
    const clamped = Math.max(0, Math.min(rows.length - 1 - zLen, newStart));
    setZoomStart(clamped);
    setZoomEnd(clamped + zLen);
  };

  const handleNavMouseUp = () => setNavDragging(false);

  // Mini chart for nav
  const allCloses = rows.map((r) => r.close);
  const allMin = Math.min(...allCloses);
  const allMax = Math.max(...allCloses);
  const allRange = allMax - allMin || 1;

  const hoverRow = hoverIdx !== null && hoverIdx < visible.length ? visible[hoverIdx] : null;
  const hoverClose = hoverIdx !== null && hoverIdx < closes.length ? closes[hoverIdx] : null;

  // Drag selection rect
  const selLo = dragStart !== null && dragEnd !== null ? Math.min(dragStart, dragEnd) : 0;
  const selHi = dragStart !== null && dragEnd !== null ? Math.max(dragStart, dragEnd) : 0;

  // Nav dimmed areas (clamped to non-negative)
  const dimLeftW = Math.max(0, thumbLeft - pad.left);
  const dimRightX = thumbLeft + thumbWidth;
  const dimRightW = Math.max(0, cw - (dimRightX - pad.left));

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
        <ZBtn label="+" onClick={() => doZoom(-1)} />
        <ZBtn label="−" onClick={() => doZoom(1)} />
        {isZoomed && <ZBtn label="Reset" onClick={resetZoom} wide />}

        {isZoomed && visible.length > 0 && (
          <span style={{ fontSize: 11, color: s.accent, fontFamily: s.mono }}>
            {visible[0].date.slice(0, 10)} → {visible[visible.length - 1].date.slice(0, 10)} ({visible.length}d)
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
          const barW = Math.max(1, (cw / visible.length) * 0.8);
          const barH = (r.volume / maxVol) * 50;
          return <rect key={`v${i}`} x={toX(i) - barW / 2} y={pad.top + ch - barH} width={barW} height={barH}
            fill={r.close >= r.open ? s.green : s.red} opacity={0.07} />;
        })}

        {/* Line */}
        {chartType === "line" && closes.length > 1 && (
          <>
            <polygon
              points={`${toX(0)},${pad.top + ch} ${closes.map((c, i) => `${toX(i)},${toY(c)}`).join(" ")} ${toX(closes.length - 1)},${pad.top + ch}`}
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
          <rect x={toX(selLo)} y={pad.top} width={Math.max(0, toX(selHi) - toX(selLo))} height={ch}
            fill={s.accent} fillOpacity={0.15} stroke={s.accent} strokeWidth={0.5} />
        )}

        {/* Hover crosshair */}
        {hoverIdx !== null && hoverClose !== null && dragStart === null && (
          <>
            <line x1={toX(hoverIdx)} y1={pad.top} x2={toX(hoverIdx)} y2={pad.top + ch}
              stroke={s.muted} strokeWidth={0.5} strokeDasharray="3,3" />
            <circle cx={toX(hoverIdx)} cy={toY(hoverClose)} r={3}
              fill={lineColor} stroke={s.surface} strokeWidth={1.5} />
          </>
        )}
      </svg>

      {/* Navigation bar */}
      <svg ref={navRef} viewBox={`0 0 ${width} ${navHeight}`}
        style={{ width: "100%", cursor: navDragging ? "grabbing" : "pointer", marginTop: 4 }}
        onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove}
        onMouseUp={handleNavMouseUp} onMouseLeave={handleNavMouseUp}
      >
        <rect x={pad.left} y={0} width={cw} height={navHeight} fill={s.surface} stroke={s.border} strokeWidth={0.5} rx={3} />
        <polyline
          points={allCloses.map((c, i) => {
            const x = pad.left + (i / Math.max(allCloses.length - 1, 1)) * cw;
            const y = 4 + (1 - (c - allMin) / allRange) * (navHeight - 8);
            return `${x},${y}`;
          }).join(" ")}
          fill="none" stroke={s.muted} strokeWidth={0.8} opacity={0.5}
        />
        {/* Dimmed outside selection */}
        {dimLeftW > 0 && <rect x={pad.left} y={0} width={dimLeftW} height={navHeight} fill="#000" opacity={0.3} />}
        {dimRightW > 0 && <rect x={dimRightX} y={0} width={dimRightW} height={navHeight} fill="#000" opacity={0.3} />}
        {/* Thumb */}
        <rect x={thumbLeft} y={1} width={thumbWidth} height={navHeight - 2}
          fill={s.accent} fillOpacity={0.15} stroke={s.accent} strokeWidth={1} rx={3} />
        {/* Grab handles */}
        {[thumbWidth / 2 - 4, thumbWidth / 2, thumbWidth / 2 + 4].map((dx) => (
          <line key={dx} x1={thumbLeft + dx} y1={navHeight / 2 - 4} x2={thumbLeft + dx} y2={navHeight / 2 + 4}
            stroke={s.accent} strokeWidth={1.5} strokeLinecap="round" />
        ))}
      </svg>

      <div style={{ fontSize: 10, color: s.muted, marginTop: 4, textAlign: "center" }}>
        Scroll to zoom · Drag chart to select range · Drag bar to pan
      </div>
    </div>
  );
}

function ZBtn({ label, onClick, wide }: { label: string; onClick: () => void; wide?: boolean }) {
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
