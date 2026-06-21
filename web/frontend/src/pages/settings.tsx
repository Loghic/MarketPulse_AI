import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { UserSettings } from "../lib/api";
import { s, Panel } from "../components/ui";

export default function Settings() {
  const qc = useQueryClient();
  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const [form, setForm] = useState<UserSettings | null>(null);
  const [saved, setSaved] = useState(false);
  const [devOpen, setDevOpen] = useState(false);

  useEffect(() => {
    if (settings && !form) setForm(settings);
  }, [settings, form]);

  const saveMutation = useMutation({
    mutationFn: (data: UserSettings) => api.updateSettings(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  const resetMutation = useMutation({
    mutationFn: () => {
      const defaults: UserSettings = {
        default_period: "1y",
        default_fee_pct: 0.05,
        default_stop_loss_pct: 0.0,
        default_backtest_days: 20,
        knn_k: 5,
        knn_enhanced_k: 5,
        lstm_preferred_preset: "standard",
        lstm_fallback: true,
        log_mode: "gui",
      };
      return api.updateSettings(defaults);
    },
    onSuccess: (data) => {
      setForm(data);
      qc.invalidateQueries({ queryKey: ["settings"] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  if (isLoading || !form) {
    return <div style={{ color: s.muted, padding: 40 }}>Loading settings...</div>;
  }

  const update = (key: keyof UserSettings, value: string | number) => {
    setForm({ ...form, [key]: value });
    setSaved(false);
  };

  const hasChanges = JSON.stringify(form) !== JSON.stringify(settings);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 700 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Settings</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {saved && <span style={{ fontSize: 12, color: s.green }}>✓ Saved</span>}
          <button
            onClick={() => saveMutation.mutate(form)}
            disabled={!hasChanges || saveMutation.isPending}
            style={{
              padding: "8px 20px", borderRadius: 6, fontSize: 13, fontWeight: 600, border: "none",
              background: hasChanges ? s.accent : s.border,
              color: hasChanges ? "#fff" : s.muted,
              cursor: hasChanges ? "pointer" : "default",
            }}
          >
            {saveMutation.isPending ? "Saving..." : "Save"}
          </button>
          <button
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            style={{
              padding: "8px 16px", borderRadius: 6, fontSize: 13, fontWeight: 600,
              border: `1px solid ${s.border}`, background: "transparent", color: s.muted, cursor: "pointer",
            }}
          >
            Reset defaults
          </button>
        </div>
      </div>

      {/* Global Defaults */}
      <Panel title="Global Defaults">
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
          <Field label="Default Period" help="Used when no period is specified in Predict/Backtest">
            <Select value={form.default_period} onChange={(v) => update("default_period", v)}
              options={[
                { value: "1mo", label: "1 Month" },
                { value: "1y", label: "1 Year" },
                { value: "2y", label: "2 Years" },
                { value: "5y", label: "5 Years" },
                { value: "max", label: "Maximum" },
              ]}
            />
          </Field>

          <Field label="Trading Fee" help="Per-side fee % (round-trip = 2×). Stocks ~0.03%, Crypto ~0.15%">
            <NumInput value={form.default_fee_pct} onChange={(v) => update("default_fee_pct", v)}
              min={0} max={1} suffix="%" />
          </Field>

          <Field label="Stop-Loss" help="0 = disabled. Exits position if intraday drop exceeds this %">
            <NumInput value={form.default_stop_loss_pct} onChange={(v) => update("default_stop_loss_pct", v)}
              min={0} max={20} suffix="%" />
          </Field>

          <Field label="Backtest Days" help="Number of holdout days for walk-forward testing">
            <NumInput value={form.default_backtest_days} onChange={(v) => update("default_backtest_days", Math.round(v))}
              min={5} max={500} integer />
          </Field>
        </div>
      </Panel>

      {/* k-NN */}
      <Panel title="k-NN Model">
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
          <Field label="k-NN (naive) — k" help="Number of nearest neighbors. Lower = more sensitive to local patterns">
            <NumInput value={form.knn_k} onChange={(v) => update("knn_k", Math.round(v))}
              min={1} max={50} integer />
          </Field>

          <Field label="k-NN Enhanced — k" help="Same parameter for enhanced features (RSI, MACD, volume, volatility)">
            <NumInput value={form.knn_enhanced_k} onChange={(v) => update("knn_enhanced_k", Math.round(v))}
              min={1} max={50} integer />
          </Field>
        </div>
      </Panel>

      {/* LSTM Model Preference */}
      <Panel title="LSTM Neural Network">
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
          <Field label="Preferred Model" help="Which trained model to use for predictions. Higher = better but slower to train">
            <Select value={form.lstm_preferred_preset} onChange={(v) => update("lstm_preferred_preset", v)}
              options={[
                { value: "cluster", label: "Cluster (best)" },
                { value: "standard", label: "Standard" },
                { value: "quick", label: "Quick (fastest)" },
              ]}
            />
          </Field>

          <div style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "8px 12px",
            background: "rgba(59, 130, 246, 0.05)", borderRadius: 6, border: `1px solid ${s.border}` }}>
            <div style={{ fontSize: 12, color: s.muted, lineHeight: 1.5 }}>
              <strong style={{ color: s.text }}>Fallback:</strong> If the preferred model doesn't exist for a ticker,
              the system automatically tries lower presets (cluster → standard → quick).
              If no LSTM model exists at all, LSTM predictions are skipped.
            </div>
          </div>
        </div>
      </Panel>

      {/* Developer Settings (collapsible) */}
      <div style={{ background: s.surface, border: `1px solid ${s.border}`, borderRadius: 8, overflow: "hidden" }}>
        <button
          onClick={() => setDevOpen(!devOpen)}
          style={{
            width: "100%", padding: "12px 16px", border: "none", background: "transparent",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            cursor: "pointer", color: s.muted, fontSize: 13, fontWeight: 600,
          }}
        >
          Developer Settings
          <span style={{ fontSize: 16, transition: "transform 0.2s", transform: devOpen ? "rotate(180deg)" : "rotate(0)" }}>
            ▾
          </span>
        </button>

        {devOpen && (
          <div style={{ padding: "0 20px 20px", display: "flex", flexDirection: "column", gap: 20,
            borderTop: `1px solid ${s.border}`, paddingTop: 20 }}>
            <Field label="Log Mode" help="Controls backend logging verbosity">
              <Select value={form.log_mode} onChange={(v) => update("log_mode", v)}
                options={[
                  { value: "gui", label: "GUI (quiet)" },
                  { value: "cli", label: "CLI (verbose)" },
                ]}
              />
            </Field>

            <Panel title="Current Config (JSON)">
              <pre style={{
                padding: 16, fontSize: 12, fontFamily: s.mono, color: s.muted,
                overflow: "auto", maxHeight: 200, margin: 0,
              }}>
                {JSON.stringify(form, null, 2)}
              </pre>
            </Panel>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Form components                                                     */
/* ------------------------------------------------------------------ */

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: 180 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: s.text }}>{label}</div>
        {help && <div style={{ fontSize: 11, color: s.muted, marginTop: 2 }}>{help}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

/**
 * NumInput — text field that validates on blur.
 * User can type anything freely. On blur: clamps to min/max, shows error flash if invalid.
 */
function NumInput({ value, onChange, min, max, suffix, integer }: {
  value: number; onChange: (v: number) => void;
  min: number; max: number; suffix?: string; integer?: boolean;
}) {
  const format = (v: number) => integer ? String(Math.round(v)) : String(v);
  const [text, setText] = useState(format(value));
  const [error, setError] = useState(false);

  // Sync when value changes externally (e.g. reset defaults)
  useEffect(() => {
    setText(format(value));
  }, [value, integer]);

  const validate = () => {
    const num = Number(text);
    if (isNaN(num) || text.trim() === "") {
      setError(true);
      setText(format(min));
      onChange(min);
      setTimeout(() => setError(false), 1500);
      return;
    }
    const clamped = Math.min(max, Math.max(min, integer ? Math.round(num) : num));
    if (clamped !== num) {
      setError(true);
      setTimeout(() => setError(false), 1500);
    }
    setText(format(clamped));
    onChange(clamped);
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <input
        type="text"
        inputMode="decimal"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={validate}
        onKeyDown={(e) => { if (e.key === "Enter") validate(); }}
        style={{
          width: 72,
          padding: "6px 10px",
          borderRadius: 6,
          fontSize: 13,
          fontWeight: 600,
          fontFamily: s.mono,
          background: "transparent",
          border: `1px solid ${error ? s.red : s.border}`,
          color: error ? s.red : s.accent,
          textAlign: "right",
          outline: "none",
          transition: "border-color 0.2s, color 0.2s",
        }}
      />
      {suffix && <span style={{ fontSize: 12, color: s.muted }}>{suffix}</span>}
      <span style={{ fontSize: 10, color: s.muted }}>({min}–{max})</span>
    </div>
  );
}

function Select({ value, onChange, options }: {
  value: string; onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={{
      padding: "6px 12px", borderRadius: 6, fontSize: 13, fontWeight: 600,
      background: s.surface, color: s.text, border: `1px solid ${s.border}`, cursor: "pointer", minWidth: 140,
    }}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}
