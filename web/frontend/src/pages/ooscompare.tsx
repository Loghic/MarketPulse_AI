import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { OOSResponse, OOSSummary } from "../lib/api";
import { s, Panel, pct } from "../components/ui";

/**
 * OOS comparison tab.
 *
 * Pick two persisted OOS runs (A vs B) and diff them: headline aggregate
 * metrics side-by-side, then a per-ticker OOS-return diff. Useful for "did
 * adding the confidence gate / turnover fees / a different model set change
 * the honest beat-B&H picture?" — run the harness once per setting, then
 * compare here.
 */
export default function OOSCompare() {
  const { data: runs } = useQuery({ queryKey: ["oosRuns"], queryFn: api.listOosRuns, staleTime: 5_000 });
  const [runA, setRunA] = useState<string | null>(null);
  const [runB, setRunB] = useState<string | null>(null);

  useEffect(() => {
    if (!runs || runs.length === 0) return;
    if (runA === null) setRunA(runs[0].run_id);
    if (runB === null && runs.length > 1) setRunB(runs[1].run_id);
  }, [runs, runA, runB]);

  const { data: a } = useQuery({ queryKey: ["oosRun", runA], queryFn: () => api.loadOosRun(runA!), enabled: !!runA });
  const { data: b } = useQuery({ queryKey: ["oosRun", runB], queryFn: () => api.loadOosRun(runB!), enabled: !!runB });

  if (!runs || runs.length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>OOS Comparison</h2>
        <Panel title="No OOS runs yet">
          <div style={{ padding: 16, color: s.muted, fontSize: 13 }}>
            Run the harness on the <strong>Out-of-Sample</strong> tab first. Each run is saved and
            shows up here for side-by-side comparison.
          </div>
        </Panel>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>OOS Comparison</h2>

      <Panel title="Pick two runs">
        <div style={{ padding: 12, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <RunPicker label="A" value={runA} onChange={setRunA} runs={runs} accent={s.accent} />
          <RunPicker label="B" value={runB} onChange={setRunB} runs={runs} accent={s.green} />
        </div>
      </Panel>

      {a && b && (
        <>
          <SummaryDiff a={a.response} b={b.response} aId={runA!} bId={runB!} />
          <PerTickerDiff a={a.response} b={b.response} />
        </>
      )}
      {(!a || !b) && <Panel title="Select two runs"><div style={{ padding: 16, color: s.muted, fontSize: 13 }}>Pick a run for both A and B.</div></Panel>}
    </div>
  );
}

function RunPicker({ label, value, onChange, runs, accent }: {
  label: string; value: string | null; onChange: (v: string | null) => void;
  runs: { run_id: string; row_count: number }[]; accent: string;
}) {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <span style={{ fontSize: 12, fontWeight: 700, color: accent }}>{label}</span>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}
        style={{ padding: "6px 10px", borderRadius: 6, fontSize: 12, background: s.surface, color: s.text, border: `1px solid ${s.border}`, maxWidth: 380 }}>
        <option value="">— pick —</option>
        {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.row_count} tickers</option>)}
      </select>
    </div>
  );
}

const SUMMARY_ROWS: { key: keyof OOSSummary; label: string; pctMode?: boolean; higherBetter: boolean }[] = [
  { key: "oos_beat_bh_rate", label: "OOS beat-B&H rate", pctMode: true, higherBetter: true },
  { key: "median_oos_return", label: "Median OOS return", pctMode: true, higherBetter: true },
  { key: "mean_oos_return", label: "Mean OOS return", pctMode: true, higherBetter: true },
  { key: "median_oos_accuracy", label: "Median OOS accuracy", higherBetter: true },
  { key: "in_sample_minus_oos_median", label: "Selection-inflation gap", pctMode: true, higherBetter: false },
  { key: "median_oos_coverage", label: "Median OOS coverage", pctMode: true, higherBetter: true },
];

function SummaryDiff({ a, b, aId, bId }: { a: OOSResponse; b: OOSResponse; aId: string; bId: string }) {
  const sa = a.summary; const sb = b.summary;
  const fmt = (v: number, pctMode?: boolean) => (pctMode ? pct(v * 100, false) : v.toFixed(3));
  return (
    <Panel title="Aggregate diff (A vs B)">
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 12, minWidth: 560 }}>
          <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
            <th style={{ ...rTh, textAlign: "left" }}>Metric</th>
            <th style={{ ...rTh, color: s.accent }} title={aId}>A</th>
            <th style={{ ...rTh, color: s.green }} title={bId}>B</th>
            <th style={rTh}>Δ (B−A)</th>
            <th style={rTh}>Better</th>
          </tr></thead>
          <tbody>
            <tr style={{ borderBottom: `1px solid ${s.border}` }}>
              <td style={{ padding: "5px 8px", color: s.muted }}>Tickers</td>
              <td style={{ padding: "5px 8px", textAlign: "center", color: s.text }}>{sa.tickers}</td>
              <td style={{ padding: "5px 8px", textAlign: "center", color: s.text }}>{sb.tickers}</td>
              <td style={{ padding: "5px 8px", textAlign: "center", color: s.muted }}>—</td>
              <td style={{ padding: "5px 8px", textAlign: "center", color: s.muted }}>—</td>
            </tr>
            {SUMMARY_ROWS.map((row) => {
              const va = (sa[row.key] as number) ?? 0;
              const vb = (sb[row.key] as number) ?? 0;
              const delta = vb - va;
              const eps = 1e-9;
              let better: "A" | "B" | "tie" = "tie";
              if (Math.abs(delta) > eps) better = (delta > 0) === row.higherBetter ? "B" : "A";
              const betterColor = better === "A" ? s.accent : better === "B" ? s.green : s.muted;
              return (
                <tr key={String(row.key)} style={{ borderBottom: `1px solid ${s.border}` }}>
                  <td style={{ padding: "5px 8px", color: s.muted }}>{row.label}</td>
                  <td style={{ padding: "5px 8px", textAlign: "center", color: s.text }}>{fmt(va, row.pctMode)}</td>
                  <td style={{ padding: "5px 8px", textAlign: "center", color: s.text }}>{fmt(vb, row.pctMode)}</td>
                  <td style={{ padding: "5px 8px", textAlign: "center", color: delta === 0 ? s.muted : (delta > 0 ? s.green : s.red) }}>
                    {row.pctMode ? pct(delta * 100) : (delta >= 0 ? "+" : "") + delta.toFixed(3)}
                  </td>
                  <td style={{ padding: "5px 8px", textAlign: "center", fontWeight: 700, color: betterColor }}>{better === "tie" ? "=" : better}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function PerTickerDiff({ a, b }: { a: OOSResponse; b: OOSResponse }) {
  const rows = useMemo(() => {
    const byTickerA = new Map(a.rows.map((r) => [r.ticker, r] as const));
    const byTickerB = new Map(b.rows.map((r) => [r.ticker, r] as const));
    const all = [...new Set([...byTickerA.keys(), ...byTickerB.keys()])].sort();
    return all.map((t) => {
      const ra = byTickerA.get(t); const rb = byTickerB.get(t);
      const retA = ra?.oos_return ?? null; const retB = rb?.oos_return ?? null;
      const delta = retA != null && retB != null ? retB - retA : null;
      return { ticker: t, ra, rb, retA, retB, delta };
    });
  }, [a, b]);

  return (
    <Panel title={`Per-ticker OOS return (${rows.length})`}>
      <div style={{ maxHeight: 480, overflow: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: s.mono, fontSize: 11, minWidth: 760 }}>
          <thead><tr style={{ borderBottom: `1px solid ${s.border}` }}>
            <th style={{ ...rTh, textAlign: "left" }}>Ticker</th>
            <th style={{ ...rTh, textAlign: "left", color: s.accent }}>A winner</th>
            <th style={{ ...rTh, color: s.accent }}>A OOS</th>
            <th style={{ ...rTh, textAlign: "left", color: s.green }}>B winner</th>
            <th style={{ ...rTh, color: s.green }}>B OOS</th>
            <th style={rTh}>Δ ret</th>
            <th style={rTh}>Better</th>
          </tr></thead>
          <tbody>{rows.map((r) => {
            const better = r.delta == null ? "—" : Math.abs(r.delta) < 1e-9 ? "=" : r.delta > 0 ? "B" : "A";
            const bc = better === "A" ? s.accent : better === "B" ? s.green : s.muted;
            return (
              <tr key={r.ticker} style={{ borderBottom: `1px solid ${s.border}` }}
                onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}>
                <td style={{ padding: "4px 8px", color: s.text }}>{r.ticker}</td>
                <td style={{ padding: "4px 8px", color: s.muted, whiteSpace: "nowrap" }}>{r.ra?.winner_model ?? "—"}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", color: r.retA == null ? s.muted : r.retA >= 0 ? s.green : s.red }}>{r.retA == null ? "—" : pct(r.retA * 100)}</td>
                <td style={{ padding: "4px 8px", color: s.muted, whiteSpace: "nowrap" }}>{r.rb?.winner_model ?? "—"}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", color: r.retB == null ? s.muted : r.retB >= 0 ? s.green : s.red }}>{r.retB == null ? "—" : pct(r.retB * 100)}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", color: r.delta == null ? s.muted : r.delta >= 0 ? s.green : s.red }}>{r.delta == null ? "—" : pct(r.delta * 100)}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", fontWeight: 700, color: bc }}>{better}</td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
    </Panel>
  );
}

const rTh: React.CSSProperties = { padding: "5px 8px", fontSize: 10, fontWeight: 500, color: s.muted, borderBottom: `1px solid ${s.border}`, textAlign: "center", position: "sticky", top: 0, background: s.surface };
