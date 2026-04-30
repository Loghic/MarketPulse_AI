import { NavLink, Outlet } from "react-router-dom";
import { clsx } from "clsx";

const tabs = [
  { to: "/dashboard", label: "Dashboard", icon: "📊" },
  { to: "/predict", label: "Predict", icon: "🎯" },
  { to: "/backtest", label: "Backtest", icon: "📈" },
  { to: "/training", label: "Training", icon: "🧠" },
  { to: "/analysis", label: "Analysis", icon: "📝" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header
        className="border-b flex items-center gap-6 px-6 h-14 shrink-0"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}
      >
        <h1
          className="text-lg font-bold tracking-tight whitespace-nowrap"
          style={{ color: "var(--c-accent)" }}
        >
          MarketPulse AI
        </h1>

        <nav className="flex gap-1 overflow-x-auto">
          {tabs.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              className={({ isActive }) =>
                clsx(
                  "px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap",
                  isActive
                    ? "text-white"
                    : "hover:text-white"
                )
              }
              style={({ isActive }) => ({
                background: isActive ? "var(--c-accent)" : "transparent",
                color: isActive ? "#fff" : "var(--c-text-muted)",
              })}
            >
              <span className="mr-1.5">{tab.icon}</span>
              {tab.label}
            </NavLink>
          ))}
        </nav>
      </header>

      {/* Page content */}
      <main className="flex-1 p-6 max-w-[1400px] w-full mx-auto">
        <Outlet />
      </main>
    </div>
  );
}
