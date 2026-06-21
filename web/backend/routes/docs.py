"""
routes/docs.py – Serve the end-user concept docs to the web UI.

The markdown under ``web/docs/`` explains what every concept in the app means
(stop-loss, OOS, baselines, the metrics…) for someone with no trading / ML
background. The Help tab fetches the manifest, then the raw markdown per page,
and renders it client-side. This is intentionally separate from the developer
``docs/`` tree (architecture / how-to-run), which web users don't care about.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/docs", tags=["docs"])

# web/docs lives two levels up from this file: web/backend/routes/ -> web/docs/
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"

# Ordered manifest: (slug, title). Slug = filename without .md. Order drives the
# Help sidebar. Keeping it explicit (rather than globbing) gives a deliberate
# reading order and lets us title pages independently of filenames.
_MANIFEST: list[tuple[str, str]] = [
    ("getting-started", "Getting started"),
    ("models", "Models & baselines"),
    ("strategy", "Strategy, fees & risk"),
    ("metrics", "Reading the results"),
    ("oos", "Out-of-sample & honesty"),
]


@router.get("")
def list_docs() -> list[dict]:
    """List the available concept docs (only those present on disk), in order."""
    out = []
    for slug, title in _MANIFEST:
        if (DOCS_DIR / f"{slug}.md").exists():
            out.append({"slug": slug, "title": title})
    return out


@router.get("/{slug}")
def get_doc(slug: str) -> dict:
    """Return one doc's raw markdown by slug."""
    # Guard against path traversal — slug must be a bare manifest key.
    titles = {s: t for s, t in _MANIFEST}
    if slug not in titles:
        raise HTTPException(status_code=404, detail=f"Doc '{slug}' not found")
    path = DOCS_DIR / f"{slug}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Doc '{slug}' not found")
    return {"slug": slug, "title": titles[slug], "markdown": path.read_text(encoding="utf-8")}
