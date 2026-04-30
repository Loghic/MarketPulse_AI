import { useState } from "react";

export default function Predict() {
  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold">Predictions</h2>
      <p style={{ color: "var(--c-text-muted)" }}>
        Next-day direction predictions. Select tickers, models, and period — then run.
      </p>

      {/* TODO: Implement prediction form + results table + consensus */}
      <div
        className="rounded-lg p-8 flex items-center justify-center"
        style={{ background: "var(--c-surface)", border: "1px solid var(--c-border)", minHeight: 300 }}
      >
        <span style={{ color: "var(--c-text-muted)" }}>
          🎯 Prediction interface — coming next
        </span>
      </div>
    </div>
  );
}
