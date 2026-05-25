/**
 * Training.tsx — LSTM training tab.
 *
 * Shows the inventory of saved LSTM models (one per ticker × period × preset)
 * with the timestamp of when each was trained, plus a minimal form to kick
 * off a new training job. Polls the status endpoint until the job finishes.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type ModelInventoryItem } from "../lib/api";
import { s, Panel, Btn } from "../components/ui";

const PERIODS = ["1mo", "1y", "2y", "5y", "max"];
const PRESETS = ["quick", "standard", "cluster"];

/** Preset ordering for "best available" — matches the engine's auto-load. */
const PRESET_RANK: Record<string, number> = { cluster: 3, standard: 2, quick: 1 };

export default function Training() {
  const queryClient = useQueryClient();
  const { data: tickers } = useQuery({ queryKey: ["tickers"], queryFn: api.getTickers });
  const { data: models, isLoading } = useQuery({
    queryKey: ["lstmModels"],
    queryFn: api.getModels,
  });

  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState("1y");
  const [preset, setPreset] = useState("quick");
  const [activeKey, setActiveKey] = useState<string | null>(null);

  // Poll the training status endpoint while a job is active.
  const { data: status } = useQuery({
    queryKey: ["trainingStatus", activeKey],
    queryFn: () => api.trainingStatus(activeKey!),
    enabled: !!activeKey,
    refetchInterval: (q) => {
      const data = q.state.data as { status?: string } | undefined;
      return data?.status === "running" ? 2000 : false;
    },
  });

  // When a job finishes, refresh the model inventory so the new file appears.
  useEffect(() => {
    if (status && status.status !== "running" && status.status !== "not_found") {
      queryClient.invalidateQueries({ queryKey: ["lstmModels"] });
    }
  }, [status, queryClient]);

  const startMut = useMutation({
    mutationFn: () => api.startTraining({ ticker, period, preset }),
    onSuccess: () => setActiveKey(`${ticker.toUpperCase()}_${period}_${preset}`),
  });

  // The model that the predictor will actually pick up for (ticker, period):
  // best preset wins (cluster > standard > quick), same logic as the engine.
  const bestPerTickerPeriod = useMemo(() => {
    const map = new Map<string, ModelInventoryItem>();
    for (const m of models ?? []) {
      const key = `${m.ticker}|${m.period}`;
      const cur = map.get(key);
      if (!cur || (PRESET_RANK[m.preset] ?? 0) > (PRESET_RANK[cur.preset] ?? 0)) {
        map.set(key, m);
      }
    }
    return map;
  }, [models]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>LSTM Training</h2>

      {/* New training job */}
      <Panel title="Start training">
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            <label style={lbl}>
              Ticker
              <select value={ticker} onChange={(e) => setTicker(e.target.value)} style={sel}>
                {tickers?.filter((t) => t.asset_type === "stock").map((t) => (
                  <option key={t.ticker} value={t.ticker}>{t.ticker}</option>
                ))}
                {tickers?.filter((t) => t.asset_type === "crypto").map((t) => (
                  <option key={t.ticker} value={t.ticker}>{t.ticker}</option>
                ))}
              </select>
            </label>
            <label style={lbl}>
              Period
              <select value={period} onChange={(e) => setPeriod(e.target.value)} style={sel}>
                {PERIODS.map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
              </select>
            </label>
            <label style={lbl}>
              Preset
              <select value={preset} onChange={(e) => setPreset(e.target.value)} style={sel}>
                {PRESETS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            <Btn
              onClick={() => startMut.mutate()}
              loading={startMut.isPending || status?.status === "running"}
              label={status?.status === "running" ? "Training…" : "Start"}
            />
          </div>

          <div style={{ fontSize: 11, color: s.muted }}>
            Presets — <strong>quick</strong>: ~1-5 min · <strong>standard</strong>: ~5-15 min ·
            <strong> cluster</strong>: hours, GPU recommended. The predictor auto-picks the best
            available preset (cluster &gt; standard &gt; quick) at runtime.
          </div>

          {/* Live status of the most recent job */}
          {status && (
            <div style={{
              padding: 10, borderRadius: 6, border: `1px solid ${s.border}`,
              display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", fontSize: 12,
            }}>
              <span style={{
                padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700,
                background: statusColor(status.status) + "20", color: statusColor(status.status),
              }}>{status.status.toUpperCase()}</span>
              <span style={{ color: s.text }}>
                {status.ticker}_{status.period}_{status.preset}
              </span>
              {status.total_epochs > 0 && (
                <span style={{ color: s.muted }}>
                  epoch {status.epoch}/{status.total_epochs}
                </span>
              )}
              {status.val_accuracy > 0 && (
                <span style={{ color: s.muted }}>
                  val_acc = {(status.val_accuracy * 100).toFixed(1)}%
                </span>
              )}
              <span style={{ color: s.muted, fontStyle: "italic", flex: 1 }}>{status.message}</span>
            </div>
          )}
        </div>
      </Panel>

      {/* Inventory */}
      <Panel title={`Saved models (${models?.length ?? 0})`}>
        {isLoading ? (
          <div style={{ padding: 24, textAlign: "center", color: s.muted }}>Loading…</div>
        ) : !models?.length ? (
          <div style={{ padding: 24, textAlign: "center", color: s.muted, fontSize: 13 }}>
            No models trained yet. Pick a ticker + period + preset above and hit Start.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={tbl}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${s.border}` }}>
                  <th style={th}>Ticker</th>
                  <th style={th}>Period</th>
                  <th style={th}>Preset</th>
                  <th style={th}>Trained</th>
                  <th style={th}>Size</th>
                  <th style={th}>Used by predictor?</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const best = bestPerTickerPeriod.get(`${m.ticker}|${m.period}`);
                  const isBest = best?.filename === m.filename;
                  return (
                    <tr key={m.filename} style={{ borderBottom: `1px solid ${s.border}` }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}
                    >
                      <td style={td}>{m.ticker}</td>
                      <td style={tdMuted}>{m.period}</td>
                      <td style={tdMuted}>{m.preset}</td>
                      <td style={tdMuted}>{fmtDate(m.modified)}</td>
                      <td style={tdMuted}>{m.size_kb.toFixed(0)} KB</td>
                      <td style={{ ...td, color: isBest ? s.green : s.muted }}>
                        {isBest ? "✓ active" : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}

function statusColor(status: string): string {
  if (status === "complete") return s.green;
  if (status === "running") return s.accent;
  if (status === "error") return s.red;
  return s.muted;
}

function fmtDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

const lbl: React.CSSProperties = { display: "flex", flexDirection: "column", gap: 4, fontSize: 11, color: s.muted };
const sel: React.CSSProperties = { padding: "6px 10px", borderRadius: 6, fontSize: 13, fontWeight: 600, background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer" };
const tbl: React.CSSProperties = { width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: s.mono };
const th: React.CSSProperties = { padding: "8px 10px", fontSize: 11, fontWeight: 500, color: s.muted, textAlign: "left", borderBottom: `1px solid ${s.border}` };
const td: React.CSSProperties = { padding: "8px 10px", color: s.text };
const tdMuted: React.CSSProperties = { padding: "8px 10px", color: s.muted };
