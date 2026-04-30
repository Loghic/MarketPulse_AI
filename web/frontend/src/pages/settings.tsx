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
        lstm_preset: "standard",
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
    return (
      <div style={{ color: s.muted, padding: 40 }}>Loading settings...</div>
    );
  }

  const update = (key: keyof UserSettings, value: string | number) => {
    setForm({ ...form, [key]: value });
    setSaved(false);
  };

  const hasChanges = JSON.stringify(form) !== JSON.stringify(settings);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 20,
        maxWidth: 700,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Settings</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {saved && (
            <span style={{ fontSize: 12, color: s.green }}>✓ Saved</span>
          )}
          <SaveBtn
            onClick={() => saveMutation.mutate(form)}
            disabled={!hasChanges || saveMutation.isPending}
            loading={saveMutation.isPending}
            hasChanges={hasChanges}
          />
          <button
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            style={{
              padding: "8px 16px",
              borderRadius: 6,
              fontSize: 13,
              fontWeight: 600,
              border: `1px solid ${s.border}`,
              background: "transparent",
              color: s.muted,
              cursor: "pointer",
            }}
          >
            Reset defaults
          </button>
        </div>
      </div>

      {/* Global Defaults */}
      <Panel title="Global Defaults">
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
          <Field label="Default Period" help="Used when no period is specified">
            <Select
              value={form.default_period}
              onChange={(v) => update("default_period", v)}
              options={["1mo", "1y", "2y", "5y", "max"]}
            />
          </Field>

          <Field
            label="Trading Fee"
            help="Per-side fee % (round-trip = 2×)"
          >
            <NumField
              value={form.default_fee_pct}
              onChange={(v) => update("default_fee_pct", v)}
              min={0}
              max={1}
              step={0.01}
              suffix="%"
              decimals={2}
            />
          </Field>

          <Field
            label="Stop-Loss"
            help="0 = disabled. Exits if intraday drop exceeds this %"
          >
            <NumField
              value={form.default_stop_loss_pct}
              onChange={(v) => update("default_stop_loss_pct", v)}
              min={0}
              max={20}
              step={0.5}
              suffix="%"
              decimals={1}
            />
          </Field>

          <Field
            label="Backtest Days"
            help="Number of holdout days for walk-forward testing"
          >
            <NumField
              value={form.default_backtest_days}
              onChange={(v) => update("default_backtest_days", Math.round(v))}
              min={5}
              max={500}
              step={5}
              integer
            />
          </Field>
        </div>
      </Panel>

      {/* k-NN */}
      <Panel title="k-NN Model">
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
          <Field
            label="k-NN (naive) — k"
            help="Number of nearest neighbors. Lower = more sensitive"
          >
            <NumField
              value={form.knn_k}
              onChange={(v) => update("knn_k", Math.round(v))}
              min={1}
              max={50}
              step={1}
              integer
            />
          </Field>

          <Field
            label="k-NN Enhanced — k"
            help="For enhanced features (RSI, MACD, volume, volatility)"
          >
            <NumField
              value={form.knn_enhanced_k}
              onChange={(v) => update("knn_enhanced_k", Math.round(v))}
              min={1}
              max={50}
              step={1}
              integer
            />
          </Field>
        </div>
      </Panel>

      {/* LSTM */}
      <Panel title="LSTM Neural Network">
        <div style={{ padding: 20 }}>
          <Field
            label="Training Preset"
            help="Quick ~1-5min · Standard ~5-15min · Cluster hours (GPU)"
          >
            <Select
              value={form.lstm_preset}
              onChange={(v) => update("lstm_preset", v)}
              options={["quick", "standard", "cluster"]}
            />
          </Field>
        </div>
      </Panel>

      {/* Display */}
      <Panel title="Display">
        <div style={{ padding: 20 }}>
          <Field
            label="Log Mode"
            help="CLI = verbose + progress bars · GUI = warnings only"
          >
            <Select
              value={form.log_mode}
              onChange={(v) => update("log_mode", v)}
              options={["cli", "gui"]}
            />
          </Field>
        </div>
      </Panel>

      {/* JSON preview */}
      <Panel title="Current Values (JSON)">
        <pre
          style={{
            padding: 16,
            fontSize: 12,
            fontFamily: s.mono,
            color: s.muted,
            overflow: "auto",
            maxHeight: 200,
          }}
        >
          {JSON.stringify(form, null, 2)}
        </pre>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Form components                                                     */
/* ------------------------------------------------------------------ */

function Field({
  label,
  help,
  children,
}: {
  label: string;
  help?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        flexWrap: "wrap",
      }}
    >
      <div style={{ flex: 1, minWidth: 180 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: s.text }}>
          {label}
        </div>
        {help && (
          <div style={{ fontSize: 11, color: s.muted, marginTop: 2 }}>
            {help}
          </div>
        )}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

function NumField({
  value,
  onChange,
  min,
  max,
  step,
  suffix,
  decimals,
  integer,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  decimals?: number;
  integer?: boolean;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  const fmt = integer ? String(Math.round(value)) : value.toFixed(decimals ?? 2);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        style={{ width: 100, accentColor: s.accent }}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={fmt}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (!isNaN(v)) onChange(clamp(v));
        }}
        style={{
          width: 64,
          padding: "4px 8px",
          borderRadius: 4,
          fontSize: 13,
          fontWeight: 600,
          fontFamily: s.mono,
          background: "transparent",
          border: `1px solid ${s.border}`,
          color: s.accent,
          textAlign: "right",
          outline: "none",
        }}
      />
      {suffix && (
        <span style={{ fontSize: 12, color: s.muted, minWidth: 20 }}>
          {suffix}
        </span>
      )}
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: "6px 12px",
        borderRadius: 6,
        fontSize: 13,
        fontWeight: 600,
        background: s.surface,
        color: s.text,
        border: `1px solid ${s.border}`,
        cursor: "pointer",
        minWidth: 120,
      }}
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o.toUpperCase()}
        </option>
      ))}
    </select>
  );
}

function SaveBtn({
  onClick,
  disabled,
  loading,
  hasChanges,
}: {
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
  hasChanges: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 20px",
        borderRadius: 6,
        fontSize: 13,
        fontWeight: 600,
        border: "none",
        background: hasChanges ? s.accent : s.border,
        color: hasChanges ? "#fff" : s.muted,
        cursor: hasChanges ? "pointer" : "default",
      }}
    >
      {loading ? "Saving..." : "Save"}
    </button>
  );
}
