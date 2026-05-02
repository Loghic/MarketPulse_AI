import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, NavLink, Outlet } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Settings from "./pages/Settings";
import { s } from "./components/ui";
import "./app.css";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

const tabs = [
  { to: "/dashboard", label: "Dashboard", icon: "📊" },
  { to: "/predict", label: "Predict", icon: "🎯" },
  { to: "/backtest", label: "Backtest", icon: "📈" },
  { to: "/training", label: "Training", icon: "🧠" },
  { to: "/analysis", label: "Analysis", icon: "📝" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

function Layout() {
  return (
    <div style={{ minHeight: "100vh" }}>
      <header style={{
        borderBottom: `1px solid ${s.border}`,
        background: s.surface,
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "0 24px",
        height: 56,
        overflowX: "auto",
      }}>
        <h1 style={{ color: s.accent, fontSize: 18, fontWeight: 700, whiteSpace: "nowrap" }}>
          MarketPulse AI
        </h1>
        <nav style={{ display: "flex", gap: 4 }}>
          {tabs.map((t) => (
            <NavLink key={t.to} to={t.to} style={({ isActive }) => ({
              padding: "8px 12px", borderRadius: 6, fontSize: 14, fontWeight: 500,
              textDecoration: "none", whiteSpace: "nowrap",
              background: isActive ? s.accent : "transparent",
              color: isActive ? "#fff" : s.muted,
            })}>
              <span style={{ marginRight: 6 }}>{t.icon}</span>{t.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main style={{ padding: 24, maxWidth: 1400, margin: "0 auto" }}>
        <Outlet />
      </main>
    </div>
  );
}

function Stub({ title, icon }: { title: string; icon: string }) {
  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>{title}</h2>
      <div style={{
        background: s.surface, border: `1px solid ${s.border}`, borderRadius: 8,
        padding: 48, textAlign: "center", color: s.muted,
      }}>
        {icon} Coming next
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/backtest" element={<Stub title="Backtest" icon="📈" />} />
            <Route path="/training" element={<Stub title="Training" icon="🧠" />} />
            <Route path="/analysis" element={<Stub title="Analysis" icon="📝" />} />
            <Route path="/settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
