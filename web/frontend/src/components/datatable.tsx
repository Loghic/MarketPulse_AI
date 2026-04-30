/**
 * DataTable.tsx – Generic sortable, filterable, paginated table.
 *
 * Features:
 *   - Click header to sort (entire dataset, not just current page)
 *   - Filter by text (searches all string columns)
 *   - Pagination with First/Prev/Next/Last
 *   - Computed columns (e.g. Δ% day change)
 *   - Export CSV button
 *   - Responsive (horizontal scroll on mobile)
 *
 * Usage:
 *   <DataTable
 *     rows={data}
 *     columns={[
 *       { key: "date", label: "Date" },
 *       { key: "close", label: "Close", align: "right", fmt: v => `$${v.toFixed(2)}` },
 *     ]}
 *     defaultSort="date"
 *     defaultAsc={false}
 *     perPage={30}
 *   />
 */

import { useState, useMemo } from "react";
import { s } from "./ui";

export interface Column<T> {
  key: string;
  label: string;
  align?: "left" | "right";
  fmt?: (value: unknown, row: T, idx: number, allRows: T[]) => string | React.ReactNode;
  sortValue?: (row: T, idx: number, allRows: T[]) => number | string;
  color?: (value: unknown, row: T) => string | undefined;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  defaultSort?: string;
  defaultAsc?: boolean;
  perPage?: number;
  filterKeys?: string[];
  exportFilename?: string;
}

export default function DataTable<T extends Record<string, unknown>>({
  rows,
  columns,
  defaultSort,
  defaultAsc = false,
  perPage = 30,
  filterKeys,
  exportFilename,
}: Props<T>) {
  const [sortCol, setSortCol] = useState(defaultSort ?? columns[0]?.key ?? "");
  const [sortAsc, setSortAsc] = useState(defaultAsc);
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState("");

  // Filter across all searchable columns
  const filtered = useMemo(() => {
    if (!filter) return rows;
    const q = filter.toLowerCase();
    const keys = filterKeys ?? columns.filter((c) => c.align !== "right").map((c) => c.key);
    return rows.filter((r) =>
      keys.some((k) => String(r[k] ?? "").toLowerCase().includes(q)),
    );
  }, [rows, filter, filterKeys, columns]);

  // Sort entire filtered dataset
  const sorted = useMemo(() => {
    const col = columns.find((c) => c.key === sortCol);
    return [...filtered].sort((a, b) => {
      let va: unknown;
      let vb: unknown;

      const idxA = filtered.indexOf(a);
      const idxB = filtered.indexOf(b);

      if (col?.sortValue) {
        va = col.sortValue(a, idxA, filtered);
        vb = col.sortValue(b, idxB, filtered);
      } else {
        va = a[sortCol];
        vb = b[sortCol];
      }

      // Handle nullish
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;

      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return sortAsc ? cmp : -cmp;
    });
  }, [filtered, sortCol, sortAsc, columns]);

  const totalPages = Math.ceil(sorted.length / perPage);
  const paged = sorted.slice(page * perPage, (page + 1) * perPage);

  const handleSort = (key: string) => {
    if (key === sortCol) setSortAsc(!sortAsc);
    else {
      setSortCol(key);
      setSortAsc(key === "date" ? false : true);
    }
    setPage(0);
  };

  // Export CSV
  const handleExport = () => {
    const headers = columns.map((c) => c.label).join(",");
    const csvRows = sorted.map((row, idx) =>
      columns
        .map((col) => {
          if (col.sortValue) return col.sortValue(row, idx, sorted);
          return row[col.key] ?? "";
        })
        .join(","),
    );
    const csv = [headers, ...csvRows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = exportFilename ?? "export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      {/* Toolbar */}
      <div
        style={{
          padding: "8px 14px",
          display: "flex",
          alignItems: "center",
          gap: 12,
          borderBottom: `1px solid ${s.border}`,
          flexWrap: "wrap",
        }}
      >
        <input
          placeholder="Filter..."
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setPage(0);
          }}
          style={{
            padding: "5px 10px",
            borderRadius: 4,
            fontSize: 12,
            background: "transparent",
            border: `1px solid ${s.border}`,
            color: s.text,
            width: 180,
            outline: "none",
          }}
        />
        <span style={{ fontSize: 11, color: s.muted }}>
          {filtered.length === rows.length
            ? `${rows.length} rows`
            : `${filtered.length} / ${rows.length} rows`}
        </span>
        {exportFilename && (
          <button
            onClick={handleExport}
            style={{
              marginLeft: "auto",
              padding: "4px 10px",
              borderRadius: 4,
              fontSize: 11,
              border: `1px solid ${s.border}`,
              background: "transparent",
              color: s.muted,
              cursor: "pointer",
            }}
          >
            Export CSV
          </button>
        )}
      </div>

      {/* Table with horizontal scroll on mobile */}
      <div style={{ maxHeight: 500, overflow: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: s.mono,
            fontSize: 13,
            minWidth: Math.max(600, columns.length * 100),
          }}
        >
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{
                    padding: "8px 14px",
                    textAlign: col.align ?? "left",
                    color: sortCol === col.key ? s.text : s.muted,
                    fontWeight: 500,
                    fontSize: 12,
                    cursor: "pointer",
                    userSelect: "none",
                    borderBottom: `1px solid ${s.border}`,
                    position: "sticky",
                    top: 0,
                    background: s.surface,
                    whiteSpace: "nowrap",
                  }}
                >
                  {col.label}
                  {sortCol === col.key && (
                    <span style={{ marginLeft: 4, opacity: 0.6 }}>
                      {sortAsc ? "↑" : "↓"}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, i) => {
              const globalIdx = page * perPage + i;
              return (
                <tr
                  key={i}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = s.hover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "";
                  }}
                  style={{ borderBottom: `1px solid ${s.border}` }}
                >
                  {columns.map((col) => {
                    const raw = row[col.key];
                    const display = col.fmt
                      ? col.fmt(raw, row, globalIdx, sorted)
                      : String(raw ?? "");
                    const color = col.color?.(raw, row);
                    return (
                      <td
                        key={col.key}
                        style={{
                          padding: "6px 14px",
                          textAlign: col.align ?? "left",
                          whiteSpace: "nowrap",
                          color: color ?? s.text,
                          fontSize: 13,
                        }}
                      >
                        {display}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          style={{
            padding: "10px 14px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderTop: `1px solid ${s.border}`,
            flexWrap: "wrap",
            gap: 8,
          }}
        >
          <span style={{ fontSize: 12, color: s.muted }}>
            Page {page + 1} of {totalPages}
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            <PgBtn label="«" onClick={() => setPage(0)} disabled={page === 0} />
            <PgBtn
              label="‹"
              onClick={() => setPage(page - 1)}
              disabled={page === 0}
            />
            <PgBtn
              label="›"
              onClick={() => setPage(page + 1)}
              disabled={page >= totalPages - 1}
            />
            <PgBtn
              label="»"
              onClick={() => setPage(totalPages - 1)}
              disabled={page >= totalPages - 1}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function PgBtn({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "4px 10px",
        borderRadius: 4,
        fontSize: 11,
        border: `1px solid ${s.border}`,
        background: "transparent",
        color: disabled ? s.border : s.muted,
        cursor: disabled ? "default" : "pointer",
      }}
    >
      {label}
    </button>
  );
}
