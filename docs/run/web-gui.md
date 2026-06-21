# Web GUI

```bash
# One-time install (after `uv pip install -e ".[web]"` — see setup.md)
cd web/frontend && npm install && cd ../..
chmod +x web/dev.sh

# Start both servers
./web/dev.sh

# Or manually in two terminals:
# Terminal 1: uv run uvicorn web.backend.app:app --reload --port 8000
# Terminal 2: cd web/frontend && npm run dev
```

* Frontend: <http://localhost:5173>
* Backend API: <http://localhost:8000>
* Swagger docs: <http://localhost:8000/docs>

See [docs/web.md](../web.md) for the full API surface and per-page
documentation.

End-user concept docs (what stop-loss / OOS / baselines / the metrics mean)
live as markdown in [`web/docs/`](../../web/docs/) and render inside the app
under the **Help** tab (served by `GET /api/docs`). Those are written for
someone with no trading/ML background; this `docs/` tree is the developer
companion.
