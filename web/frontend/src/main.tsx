import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, NavLink, Outlet } from "react-router-dom";
import "./app.css";

const qc = new QueryClient();

function Layout() {
  const tabs = [
    { to: "/dashboard", label: "📊 Dashboard" },
    { to: "/predict", label: "🎯 Predict" },
    { to: "/backtest", label: "📈 Backtest" },
    { to: "/training", label: "🧠 Training" },
    { to: "/analysis", label: "📝 Analysis" },
    { to: "/settings", label: "⚙ Settings" },
  ];

  return (
    <div style={{ minHeight: "100vh" }}>
      <header style={{
        borderBottom: "1px solid #1e293b",
        background: "#111827",
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "0 24px",
        height: 56,
      }}>
        <h1 style={{ color: "#3b82f6", fontSize: 18, fontWeight: 700 }}>
          MarketPulse AI
        </h1>
        <nav style={{ display: "flex", gap: 4 }}>
          {tabs.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              style={({ isActive }) => ({
                padding: "8px 12px",
                borderRadius: 6,
                fontSize: 14,
                fontWeight: 500,
                textDecoration: "none",
                background: isActive ? "#3b82f6" : "transparent",
                color: isActive ? "#fff" : "#94a3b8",
              })}
            >
              {t.label}
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
        background: "#111827",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: 48,
        textAlign: "center",
        color: "#94a3b8",
      }}>
        {icon} Coming next
      </div>
    </div>
  );
}

function Dashboard() {
  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>Dashboard</h2>
      <div style={{
        background: "#111827",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: 24,
        color: "#94a3b8",
      }}>
        📊 Dashboard with ticker selector, chart, and OHLCV table — wiring up next
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
            <Route path="/predict" element={<Stub title="Predictions" icon="🎯" />} />
            <Route path="/backtest" element={<Stub title="Backtest" icon="📈" />} />
            <Route path="/training" element={<Stub title="Training" icon="🧠" />} />
            <Route path="/analysis" element={<Stub title="Analysis" icon="📝" />} />
            <Route path="/settings" element={<Stub title="Settings" icon="⚙" />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>
);
