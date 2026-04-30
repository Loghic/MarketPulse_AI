/**
 * ui.tsx – Shared UI primitives reused across all pages.
 */

export const s = {
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

export function Panel({
  title,
  extra,
  children,
}: {
  title: string;
  extra?: React.ReactNode;
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
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        {title}
        {extra}
      </div>
      {children}
    </div>
  );
}

export function Btn({
  onClick,
  loading,
  label,
  secondary,
  small,
}: {
  onClick: () => void;
  loading?: boolean;
  label: string;
  secondary?: boolean;
  small?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      style={{
        padding: small ? "4px 10px" : "6px 16px",
        borderRadius: 6,
        fontSize: small ? 11 : 12,
        fontWeight: 600,
        border: secondary ? `1px solid ${s.border}` : "none",
        background: secondary ? "transparent" : s.accent,
        color: secondary ? s.muted : "#fff",
        cursor: loading ? "wait" : "pointer",
        opacity: loading ? 0.6 : 1,
      }}
    >
      {loading ? "..." : label}
    </button>
  );
}

export function LoadingBox({ height = 350 }: { height?: number }) {
  return (
    <div
      style={{
        height,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: s.muted,
      }}
    >
      Loading...
    </div>
  );
}

export function pct(v: number, plus = true): string {
  const str = v.toFixed(2) + "%";
  return plus && v >= 0 ? "+" + str : str;
}

export function usd(v: number): string {
  return "$" + v.toFixed(2);
}
