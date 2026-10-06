# Py2Sharp Lens

A **learning tool** for a senior C# developer learning Python: paste Python on
the left, get idiomatic modern C# on the right, plus an explanation panel of
semantic differences and gotchas — each tied to a Python line number.

This is a learning lens, not a production transpiler. Readable C# and accurate
gotcha explanations matter more than compilable output.

## How it works (hybrid: deterministic + LLM)

1. **Parser/annotator (deterministic):** Python's `ast` module walks the code
   and detects constructs from `app/catalog.json` → findings
   `{line, end_line, construct_key}`.
2. **Concept catalog:** `app/catalog.json` is the core asset — 47 hand-editable
   constructs, each with an idiomatic C# equivalent, a gotcha note, and a
   severity (`info` | `gotcha` | `danger`).
3. **LLM translator (optional):** the model returns **plain C# only** (no JSON,
   no fences). Each translated block starts with a `// PY: <line>: <python>`
   marker; type assumptions are `// ASSUME:` lines. Small local models work.
4. **Deterministic merge:** notes always come from the annotator + catalog
   (never the model); `line_map` is parsed from the `// PY:` markers.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:app
```

Open http://localhost:8000. With no configuration you get **annotate-only
mode** (construct detection + notes, no C# translation) and a banner saying so.

## Choosing a provider

| `LLM_PROVIDER` | What it does | Needs |
|---|---|---|
| `none` (default) | Annotate-only: notes + banner, no LLM call ever | Nothing |
| `openai_compat` | Any OpenAI-compatible chat API | `LLM_BASE_URL`, `LLM_MODEL`, optional `LLM_API_KEY` |
| `anthropic` | Anthropic API (`pip install anthropic` — optional dep) | `ANTHROPIC_MODEL`, `ANTHROPIC_API_KEY` |
| `manual` | You are the transport: Copy prompt → your chat app → Paste result | Nothing |

Copy `.env.example` to `.env` and set what you need (`.env` is gitignored).

### Ollama (recommended local setup)

1. Install Ollama: https://ollama.com
2. `ollama pull qwen2.5-coder:7b`
3. In `.env` (or environment):
   ```
   LLM_PROVIDER=openai_compat
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_MODEL=qwen2.5-coder:7b
   ```
4. Restart uvicorn. The badge in the header shows the active provider.

The same `openai_compat` provider works with Groq, OpenRouter, and Gemini's
OpenAI-compatible endpoint — only the base URL, key, and model change.

### Manual mode (no API at all)

1. Set `LLM_PROVIDER=manual`.
2. Paste Python, click **Copy prompt** — the exact system+user prompt is copied.
3. Paste it into any chat app, get the C# reply.
4. Click **Paste result**, paste the reply back. Line mapping and notes are
   rebuilt deterministically from the `// PY:` markers and the annotator.

## Running with Docker

Annotate-only (no provider needed):

```bash
docker build -t py2sharp-lens .
docker run -p 8000:8000 py2sharp-lens
```

App + Ollama sidecar:

```bash
docker compose up
docker compose exec ollama ollama pull qwen2.5-coder:7b
```

Inside compose the app reaches Ollama at `http://ollama:11434/v1` (not
localhost — localhost is the app container itself), so set
`LLM_BASE_URL=http://ollama:11434/v1` and `LLM_PROVIDER=openai_compat` in
`.env`. The annotate-only banner is server-driven, so it works identically
in Docker: with no provider configured you see the banner and get notes only.

Health check: `GET /healthz` → `{"status": "ok"}`.

## Config reference

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `none` | `none` \| `openai_compat` \| `anthropic` \| `manual` |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | _(empty)_ | Model name for `openai_compat` |
| `LLM_API_KEY` | _(empty)_ | Optional; empty is fine for Ollama |
| `ANTHROPIC_MODEL` | _(empty)_ | Model name for `anthropic` |
| `ANTHROPIC_API_KEY` | _(empty)_ | Key for `anthropic` |

API keys are never logged and never leave the process environment — only
`has_api_key: bool` is exposed via `GET /api/status`. Input is limited to
20 KB (`413` beyond that). CORS allows only localhost origins.

## Project layout

```
app/
  main.py        FastAPI app: endpoints, cache, limits, CORS, static UI
  models.py      Pydantic request/response schemas
  annotator.py   ast.NodeVisitor -> findings [{line, end_line, construct_key}]
  translator.py  Prompts, provider abstraction, marker parsing, merge
  catalog.json   47 constructs: {title, csharp, note, severity}
web/
  index.html     Shell: header, toolbar, panes, explanation panel
  app.js         Vanilla JS: Monaco, translate, cross-highlight, manual mode
  styles.css     Two-pane layout, OS light/dark via prefers-color-scheme
vscode-extension/
  README.md      Phase 2 placeholder
tests/
  test_annotator.py   One test per catalog construct
  test_translator.py  Marker parsing, providers (mocked), merge
  test_api.py         Endpoint contracts incl. syntax-error + mapping notice
Dockerfile / docker-compose.yml / .dockerignore
requirements.txt / .env.example / .gitignore
```

### API endpoints

- `POST /api/translate` `{"python": code}` → full response (translate or annotate-only)
- `POST /api/annotate` `{"python": code}` → AST findings only, always works
- `POST /api/prompt` `{"python": code}` → the exact prompt as plain text (manual mode)
- `POST /api/import` `{"python": code, "reply": csharp}` → deterministic merge (manual mode)
- `GET /api/catalog` → catalog.json
- `GET /api/status` → `{provider, model, has_api_key, catalog_constructs}`
- `GET /healthz` → `{"status": "ok"}`
- `GET /` → the web UI

## Acceptance checklist

- [ ] Annotate-only works with zero setup and shows the banner.
- [ ] With Ollama running, a 30-line file translates with notes + line mapping.
- [ ] Manual mode round-trips: Copy prompt → paste reply → line_map and notes render.
- [ ] Invalid Python returns line/column errors with **no API call**; the C# pane is cleared and no stale output remains.
- [ ] `x = -7 // 2` yields the note: Python gives -4, C# int division gives -3.
- [ ] A marker-less model reply keeps the C# verbatim and shows the mapping notice.
- [ ] Clicking a Python line highlights the matching C# lines, and vice versa.
- [ ] `docker compose up` serves the UI with the annotate-only banner when no provider is configured.

## Notes for the Python learner (reading this codebase)

- Type hints are on every function — they are documentation the tooling checks.
- `# like C#'s ...` comments flag Pythonic idioms with their C# counterpart.
- `ast.NodeVisitor` ≈ Roslyn's `CSharpSyntaxWalker`: `visit_For` is `VisitForStatement`.
- `getattr(node, "end_lineno", None) or lineno` — Python's `or` returns the first truthy value (like C# `??` but for truthiness, not just null).
- `OrderedDict` + `popitem(last=False)` is the hand-rolled LRU; C# would use `MemoryCache`.
