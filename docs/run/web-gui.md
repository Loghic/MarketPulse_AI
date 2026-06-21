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
