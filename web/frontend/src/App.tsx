import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/dashboard";
import Predict from "./pages/predict";
import Backtest from "./pages/backtest";
import OOS from "./pages/oos";
import OOSCompare from "./pages/ooscompare";
import Training from "./pages/training";
import Analysis from "./pages/analysis";
import Settings from "./pages/settings";
import Help from "./pages/help";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/oos" element={<OOS />} />
          <Route path="/oos-compare" element={<OOSCompare />} />
          <Route path="/training" element={<Training />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/help" element={<Help />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
