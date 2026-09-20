# AGENTS.md

## Cursor Cloud specific instructions

### Product

Single app in `conversor-video/`: **PitchView** — Node.js/Express video converter and football-analysis UI (Portuguese). SQLite (`data/app.db`) and uploads (`data/uploads/`) are created automatically; no separate DB or Docker services.

### Run the app

From repo root:

```bash
cd conversor-video
npm install   # first time or after dependency changes
npm start     # same as npm run dev — listens on PORT or 3000
```

Windows equivalent: `conversor-video/iniciar.bat` (runs `npm install` if needed, then `node server/index.js`).

Do **not** run `server.py` alongside the Node server — both default to port 3000; the Python server is legacy static-only.

### Verify locally

- UI: `http://localhost:3000` → redirects to `login.html` or `app.html` if session exists.
- API smoke (with session cookie): `POST /api/auth/register` or `POST /api/auth/login`, then `GET /api/auth/me`, `GET /api/dashboard/stats`.

### Lint / tests

No ESLint, Prettier, or test runner is configured in `package.json`. Validation is manual (API curls + browser) or adding tooling if requested.

### Optional external deps

- **FFmpeg.wasm** loads from jsDelivr CDN in the browser for in-client conversion; offline VMs can still use auth, history, and mock analysis APIs.
- Env: `PORT`, `SESSION_SECRET`, `ANALYSIS_PROVIDER` (default `mock`).

### Long-running server

Use a dedicated tmux session (e.g. `conversor-dev-server`) for `npm start` so the process survives backgrounding. COOP/COEP headers are set for FFmpeg.wasm SharedArrayBuffer support.
