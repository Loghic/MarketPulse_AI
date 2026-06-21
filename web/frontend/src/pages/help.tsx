import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { s, Panel } from "../components/ui";

/**
 * Help / glossary tab.
 *
 * Fetches the end-user concept docs (web/docs/*.md) via /api/docs and renders
 * them with a tiny dependency-free markdown renderer. Features:
 *  - sidebar nav over all docs
 *  - full-text search across every doc (jumps to the matching section)
 *  - deep-linking via the URL hash: #<docSlug>/<sectionSlug>, e.g.
 *    #strategy/stop-loss — used by the "?" links on the Backtest / OOS tabs.
 *  - intra-doc markdown links like "strategy#stop-loss" are rewritten to that
 *    hash form so everything stays inside the Help tab.
 */

// GitHub-style heading slug: lowercase, drop non-word chars, spaces→hyphens.
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_]/g, "")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

// --- Minimal markdown → React renderer (headings, lists, p, inline). ---
// Deliberately small: our glossary uses #/##, -, **bold**, `code`, [links].

function renderInline(text: string, onLink: (href: string) => void): React.ReactNode[] {
  // Tokenise on links / bold / inline-code. Order matters.
  const out: React.ReactNode[] = [];
  let rest = text;
  let key = 0;
  const re = /(\[([^\]]+)\]\(([^)]+)\))|(\*\*([^*]+)\*\*)|(`([^`]+)`)/;
  while (rest.length) {
    const m = re.exec(rest);
    if (!m) { out.push(rest); break; }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    if (m[1]) {
      const label = m[2]; const href = m[3];
      out.push(
        <a key={key++} href="#" onClick={(e) => { e.preventDefault(); onLink(href); }}
          style={{ color: s.accent, textDecoration: "none", borderBottom: `1px dotted ${s.accent}` }}>
          {label}
        </a>,
      );
    } else if (m[4]) {
      out.push(<strong key={key++} style={{ color: s.text }}>{m[5]}</strong>);
    } else if (m[6]) {
      out.push(<code key={key++} style={{ fontFamily: s.mono, fontSize: "0.9em", background: s.hover, padding: "1px 5px", borderRadius: 4, color: s.accent }}>{m[7]}</code>);
    }
    rest = rest.slice(m.index + m[0].length);
    key++;
  }
  return out;
}

interface Block { type: "h1" | "h2" | "h3" | "p" | "ul"; text?: string; items?: string[]; slug?: string }

function parseMarkdown(md: string): Block[] {
  const lines = md.split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];
  const flushPara = () => { if (para.length) { blocks.push({ type: "p", text: para.join(" ") }); para = []; } };
  const flushList = () => { if (list.length) { blocks.push({ type: "ul", items: list }); list = []; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (/^###\s+/.test(line)) { flushPara(); flushList(); const t = line.replace(/^###\s+/, ""); blocks.push({ type: "h3", text: t, slug: slugify(t) }); }
    else if (/^##\s+/.test(line)) { flushPara(); flushList(); const t = line.replace(/^##\s+/, ""); blocks.push({ type: "h2", text: t, slug: slugify(t) }); }
    else if (/^#\s+/.test(line)) { flushPara(); flushList(); const t = line.replace(/^#\s+/, ""); blocks.push({ type: "h1", text: t, slug: slugify(t) }); }
    else if (/^[-*]\s+/.test(line)) { flushPara(); list.push(line.replace(/^[-*]\s+/, "")); }
    else if (line.trim() === "") { flushPara(); flushList(); }
    else { flushList(); para.push(line.trim()); }
  }
  flushPara(); flushList();
  return blocks;
}

export default function Help() {
  const navigate = useNavigate();
  const location = useLocation();
  const [search, setSearch] = useState("");

  const { data: docList } = useQuery({ queryKey: ["docList"], queryFn: api.listDocs, staleTime: 300_000 });

  // Parse the hash: #<docSlug>/<sectionSlug>
  const [hashDoc, hashSection] = useMemo(() => {
    const h = location.hash.replace(/^#/, "");
    const [d, sec] = h.split("/");
    return [d || null, sec || null];
  }, [location.hash]);

  const activeSlug = hashDoc ?? docList?.[0]?.slug ?? null;

  // Load every doc once (so search can scan all of them); cheap, all-local.
  const docQueries = useQueries({
    queries: (docList ?? []).map((d) => ({
      queryKey: ["doc", d.slug],
      queryFn: () => api.getDoc(d.slug),
      staleTime: 300_000,
    })),
  });
  const docsBySlug = useMemo(() => {
    const m = new Map<string, { slug: string; title: string; markdown: string }>();
    for (const q of docQueries) if (q.data) m.set(q.data.slug, q.data);
    return m;
  }, [docQueries]);

  const active = activeSlug ? docsBySlug.get(activeSlug) : undefined;
  const blocks = useMemo(() => (active ? parseMarkdown(active.markdown) : []), [active]);

  // Navigate helper: switch doc + section via the hash.
  const goto = (docSlug: string, section?: string) => {
    navigate(`/help#${docSlug}${section ? "/" + section : ""}`);
  };

  // Intra-doc link handler. Markdown hrefs come as "strategy#stop-loss",
  // "oos", or "metrics#calibration". Rewrite to our hash navigation.
  const onLink = (href: string) => {
    if (/^https?:\/\//.test(href)) { window.open(href, "_blank"); return; }
    const [doc, sec] = href.split("#");
    goto(doc || activeSlug || "", sec);
  };

  // Scroll to the hash section after the active doc renders.
  const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!hashSection || !active) return;
    const el = document.getElementById(hashSection);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [hashSection, active]);

  // --- Search: scan all docs for the query, list matching sections. ---
  const searchHits = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (q.length < 2) return [];
    const hits: { docSlug: string; docTitle: string; section: string; sectionSlug: string; snippet: string }[] = [];
    for (const doc of docsBySlug.values()) {
      const bks = parseMarkdown(doc.markdown);
      let curSection = doc.title; let curSlug = "";
      for (const b of bks) {
        if (b.type === "h2" || b.type === "h3" || b.type === "h1") { curSection = b.text!; curSlug = b.slug!; }
        const hay = (b.text ?? "") + " " + (b.items?.join(" ") ?? "");
        if (hay.toLowerCase().includes(q)) {
          const idx = hay.toLowerCase().indexOf(q);
          const snippet = hay.slice(Math.max(0, idx - 30), idx + 60);
          hits.push({ docSlug: doc.slug, docTitle: doc.title, section: curSection, sectionSlug: curSlug, snippet });
        }
      }
    }
    // de-dupe by doc+section
    const seen = new Set<string>();
    return hits.filter((h) => { const k = h.docSlug + h.sectionSlug; if (seen.has(k)) return false; seen.add(k); return true; }).slice(0, 30);
  }, [search, docsBySlug]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700 }}>Help &amp; Glossary</h2>
      <p style={{ fontSize: 13, color: s.muted, margin: 0, maxWidth: 720 }}>
        Plain-language explanations of everything in the app — models, stop-loss, the
        confidence gate, out-of-sample testing, and how to read every metric. Written
        for someone with no trading or machine-learning background.
      </p>

      {/* Search */}
      <input
        placeholder="Search the docs… (e.g. stop-loss, OOS, Sharpe)"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{ padding: "8px 12px", borderRadius: 6, fontSize: 13, background: s.surface, border: `1px solid ${s.border}`, color: s.text, outline: "none", maxWidth: 480 }}
      />

      {search.trim().length >= 2 && (
        <Panel title={`Search results (${searchHits.length})`}>
          <div style={{ padding: 8, display: "flex", flexDirection: "column" }}>
            {searchHits.length === 0 && <div style={{ padding: 8, color: s.muted, fontSize: 13 }}>No matches.</div>}
            {searchHits.map((h, i) => (
              <button key={i} onClick={() => { setSearch(""); goto(h.docSlug, h.sectionSlug || undefined); }}
                style={{ textAlign: "left", padding: "8px 10px", border: "none", background: "transparent", cursor: "pointer", borderBottom: `1px solid ${s.border}` }}
                onMouseEnter={(e) => { e.currentTarget.style.background = s.hover; }} onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: s.accent }}>{h.docTitle} › {h.section}</div>
                <div style={{ fontSize: 11, color: s.muted }}>…{h.snippet}…</div>
              </button>
            ))}
          </div>
        </Panel>
      )}

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* Sidebar */}
        <div style={{ width: 200, flexShrink: 0, display: "flex", flexDirection: "column", gap: 4 }}>
          {(docList ?? []).map((d) => (
            <button key={d.slug} onClick={() => goto(d.slug)}
              style={{ textAlign: "left", padding: "8px 12px", borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: "pointer",
                border: `1px solid ${activeSlug === d.slug ? s.accent : s.border}`,
                background: activeSlug === d.slug ? "rgba(59,130,246,0.15)" : "transparent",
                color: activeSlug === d.slug ? s.accent : s.muted }}>
              {d.title}
            </button>
          ))}
        </div>

        {/* Content */}
        <div ref={contentRef} style={{ flex: 1, minWidth: 320 }}>
          <Panel title={active?.title ?? "Loading…"}>
            <div style={{ padding: "8px 20px 20px", lineHeight: 1.6 }}>
              {blocks.map((b, i) => {
                if (b.type === "h1") return null; // page title already shown in the Panel header
                if (b.type === "h2") return <h3 key={i} id={b.slug} style={{ fontSize: 16, fontWeight: 700, color: s.text, marginTop: 22, marginBottom: 6, scrollMarginTop: 70 }}>{b.text}</h3>;
                if (b.type === "h3") return <h4 key={i} id={b.slug} style={{ fontSize: 14, fontWeight: 700, color: s.text, marginTop: 16, marginBottom: 4, scrollMarginTop: 70 }}>{b.text}</h4>;
                if (b.type === "ul") return (
                  <ul key={i} style={{ margin: "6px 0", paddingLeft: 20, color: s.muted, fontSize: 13 }}>
                    {b.items!.map((it, j) => <li key={j} style={{ marginBottom: 4 }}>{renderInline(it, onLink)}</li>)}
                  </ul>
                );
                return <p key={i} style={{ margin: "8px 0", color: s.muted, fontSize: 13 }}>{renderInline(b.text!, onLink)}</p>;
              })}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
