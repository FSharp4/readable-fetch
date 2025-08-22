# Readable-Fetch

Methods for fetching markdown content from HTML URLs on the internet.
Uses Firefox's approach to do this.

## Dev setup (quickstart)

This repository uses the "src/" layout. To get a working development environment where imports and console scripts "just work", install the package in editable mode inside a virtual environment.

1. Create and activate a virtualenv

- Windows (cmd.exe)
  ```
  python -m venv .venv
  .venv\Scripts\activate
  ```

- macOS / Linux (bash/zsh)
  ```
  python -m venv .venv
  source .venv/bin/activate
  ```

2. Install in editable mode (dev deps optional)
```
pip install -e .
pip install -r <(python - <<'PY'
import tomllib, sys
with open("pyproject.toml","rb") as f:
    doc = tomllib.load(f)
deps = doc.get("dependency-groups", {}).get("dev", [])
if deps:
    print("\\n".join(deps))
PY
) 2>/dev/null || true
```
(Alternatively: `poetry install` if you prefer Poetry.)

Installing makes the package importable (no PYTHONPATH tweaks) and installs a console script `readable_fetch` that starts the MCP server.

3. Start the MCP server

- Using the installed console script (recommended after install)
```
readable_fetch
```

- Or using uvicorn (works after install)
```
uvicorn readable_fetch.mcp_server.server:app --reload
```

- Or run in-place without installing (quick, ephemeral)
  - Windows (cmd.exe):
    ```
    set PYTHONPATH=src && python scripts/run_mcp_server.py
    ```
  - macOS / Linux:
    ```
    PYTHONPATH=src python scripts/run_mcp_server.py
    ```

4. Endpoints

- GET /tools
  - Returns discovery metadata for available tools.

- POST /invoke
  - Body: `{"tool":"<tool_name>", "args": { ... } }`
  - Tools:
    - `fetch_and_markdownify` — deterministic readability + markdownify fallback. Args: `{"url": "https://..."}`.
    - `fetch_and_markdownify_gpt` — OpenAI LLM-first extraction (falls back to local conversion). Args:
      - `url` (required)
      - `api_key` (optional) — explicit OpenAI key
      - `model`, `chunk_chars`, `token_threshold`, `prefer_sharing_key`, `preclean` (optional)

Example curl:
```
curl http://127.0.0.1:8000/tools
curl -X POST http://127.0.0.1:8000/invoke -H "Content-Type: application/json" -d "{\"tool\":\"fetch_and_markdownify\",\"args\":{\"url\":\"https://example.com\"}}"
```

5. Tests

- The test suite includes `tests/conftest.py` which automatically inserts `src/` into `sys.path` so `pytest` works without manual PYTHONPATH.
```
pytest -q
```

6. OpenAI / GPT notes

- The GPT tool looks for keys in this order:
  - explicit `api_key` argument
  - `OPENAI_SHARING_KEY` environment variable (preferred)
  - `OPENAI_API_KEY` environment variable (fallback)

Be careful about costs when using real API keys.

7. CI / CI recommendation

- In CI, prefer installing the package before running tests:
  - `pip install -e .`
  - `pytest -q`

If you want, I can add a GitHub Actions workflow that performs the install and runs the tests.

---

If you'd like, I can:
- Add a short CLI wrapper to expose common flags (host/port/reload) for `readable_fetch`.
- Add a Dockerfile.
- Add a GitHub Actions workflow that installs the package and runs tests.
