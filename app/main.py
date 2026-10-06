"""Py2Sharp Lens backend: FastAPI app, endpoints, limits, CORS, static UI.

Run locally with:
    pip install -r requirements.txt
    uvicorn app.main:app
Then open http://localhost:8000.

Configuration is via environment variables (or a `.env` file, loaded below
with python-dotenv).  Never commit `.env` — it is in .gitignore.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()  # no-op if there is no .env file

from app import translator  # noqa: E402  (import after load_dotenv on purpose)
from app.annotator import annotate  # noqa: E402
from app.models import (  # noqa: E402
    AnnotateFinding,
    AnnotateResponse,
    ImportRequest,
    StatusResponse,
    TranslateRequest,
    TranslateResponse,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR.parent / "web"  # ../web relative to this file: works in Docker too
CATALOG = translator.load_catalog()

MAX_BYTES = 20 * 1024  # 20 KB input limit

app = FastAPI(title="Py2Sharp Lens")

# CORS restricted to localhost origins only (regex form).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _oversize(code: str) -> JSONResponse | None:
    """Return a 413 response if the input exceeds the limit, else None."""
    if len(code.encode("utf-8")) > MAX_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Input exceeds 20 KB"})
    return None


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe (used by Docker / uptime checks)."""
    return {"status": "ok"}


@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    """Translate Python -> C#.  Never calls the provider on syntax errors."""
    if (err := _oversize(req.python)) is not None:
        return err
    provider = translator.get_provider()  # construction only: no network here
    try:
        return translator.translate(req.python, provider=provider)
    except translator.PythonSyntaxError as exc:
        # Hardened contract: HTTP 200, csharp == "", everything empty
        # except errors.  The provider was never called.
        return translator.syntax_error_response(
            exc.line, exc.column, exc.message, provider.name
        )
    except translator.ProviderError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.post("/api/annotate")
def api_annotate(req: TranslateRequest):  # returns AnnotateResponse or JSONResponse
    """AST-only annotation: always works, no LLM involved."""
    if (err := _oversize(req.python)) is not None:
        return err
    try:
        findings = annotate(req.python)
    except SyntaxError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Syntax error at line {exc.lineno}, "
                f"column {exc.offset}: {exc.msg}"
            },
        )
    enriched = [
        AnnotateFinding(
            line=f["line"],
            end_line=f["end_line"],
            construct_key=f["construct_key"],
            title=CATALOG.get(f["construct_key"], {}).get("title", f["construct_key"]),
            csharp=CATALOG.get(f["construct_key"], {}).get("csharp", ""),
            note=CATALOG.get(f["construct_key"], {}).get("note", ""),
            severity=CATALOG.get(f["construct_key"], {}).get("severity", "info"),
        )
        for f in findings
    ]
    return AnnotateResponse(findings=enriched)


@app.post("/api/prompt", response_class=PlainTextResponse)
def api_prompt(req: TranslateRequest):
    """Manual mode: return the EXACT system+user prompt as plain text.

    Works regardless of provider (no provider is called).  The user copies
    this into their own chat app.
    """
    if (err := _oversize(req.python)) is not None:
        return err
    try:
        findings = annotate(req.python)
    except SyntaxError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Syntax error at line {exc.lineno}, "
                f"column {exc.offset}: {exc.msg}"
            },
        )
    system = translator.build_system_prompt()
    user = translator.build_user_prompt(req.python, findings)
    return PlainTextResponse(f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}\n")


@app.post("/api/import")
def api_import(req: ImportRequest):
    """Manual mode: merge a pasted-back model reply deterministically.

    Runs the syntax check on the Python, parses // PY: markers from the
    reply, and builds notes from the annotator + catalog.  Provider is
    reported as "manual".
    """
    if (err := _oversize(req.python)) is not None:
        return err
    if (err := _oversize(req.reply)) is not None:
        return err
    try:
        findings = annotate(req.python)
    except SyntaxError as exc:
        return translator.syntax_error_response(
            exc.lineno or 0, exc.offset or 0, f"{exc.msg}", "manual"
        )
    return translator.merge(req.python, req.reply, findings, CATALOG, "manual")


@app.get("/api/catalog")
def api_catalog() -> dict:
    """The hand-editable concept catalog."""
    return CATALOG


@app.get("/api/status")
def api_status() -> StatusResponse:
    """Provider/model info.  Exposes has_api_key as a BOOLEAN ONLY."""
    provider = translator.get_provider()
    return StatusResponse(
        provider=provider.name,
        model=translator.provider_model(provider),
        has_api_key=translator.has_api_key(),
        catalog_constructs=len(CATALOG),
    )


# Static UI last: Starlette matches routes in registration order, so the
# API routes above win and everything else falls through to web/.
# html=True serves web/index.html for GET /.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
