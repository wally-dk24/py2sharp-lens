"""Prompts, LLM provider abstraction, and the deterministic merge.

Architecture (hybrid, as the name says):
  1. The AST annotator finds constructs deterministically (no LLM).
  2. A provider turns (system prompt, user prompt) into plain C# text.
     Every provider obeys the SAME contract: return ONLY C# code, with
     `// PY: <line>: <python source>` marker comments and `// ASSUME:`
     lines.  This keeps small local models (Ollama) viable — no JSON
     required from the model.
  3. The merge is deterministic: notes come from the annotator + catalog,
     never from the model; line_map is parsed from the `// PY:` markers.

Provider selection is via the LLM_PROVIDER env var:
  "none" (default) -> annotate-only, no LLM call ever.
  "openai_compat"  -> any OpenAI-compatible chat API (Ollama, Groq,
                      OpenRouter, Gemini's OpenAI endpoint).
  "anthropic"      -> Anthropic API (optional dependency, lazy import).
  "manual"         -> no API: /api/prompt gives you the prompt text,
                      /api/import merges a pasted-back reply.

Security: API keys are read from env vars and NEVER logged.  Only a
boolean `has_api_key` is ever exposed (via /api/status).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from app.annotator import annotate
from app.models import LineMap, Note, SyntaxErrorItem, TranslateResponse

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_CATALOG: dict | None = None


def load_catalog() -> dict:
    """Load catalog.json once and cache it (it is read-only at runtime)."""
    global _CATALOG
    if _CATALOG is None:
        path = Path(__file__).resolve().parent / "catalog.json"
        _CATALOG = json.loads(path.read_text(encoding="utf-8"))
    return _CATALOG


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PythonSyntaxError(Exception):
    """Raised when the input is not valid Python (before any LLM call)."""

    def __init__(self, line: int, column: int, message: str) -> None:
        super().__init__(message)
        self.line = line
        self.column = column
        self.message = message


class ProviderError(Exception):
    """Raised when the configured LLM provider fails."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class LLMProvider:
    """Abstract provider: turn (system, user) prompts into plain C# text."""

    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class NoneProvider(LLMProvider):
    """Annotate-only: there is no model to call."""

    name = "none"

    def complete(self, system: str, user: str) -> str:
        raise ProviderError(
            "No LLM provider configured (LLM_PROVIDER=none). "
            "See README 'Choosing a provider'."
        )


class ManualProvider(LLMProvider):
    """Manual mode: the user is the transport (copy prompt / paste reply)."""

    name = "manual"

    def complete(self, system: str, user: str) -> str:
        raise ProviderError(
            "Manual mode: use POST /api/prompt to get the prompt text and "
            "POST /api/import to merge the reply you pasted back."
        )


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible chat-completions API.

    Primary target: Ollama at http://localhost:11434/v1 (no key needed).
    Also works with Groq, OpenRouter, and Gemini's OpenAI-compatible
    endpoint — only the base URL / key / model change.
    """

    name = "openai_compat"

    def __init__(self) -> None:
        # os.environ.get(key, default): like C#'s
        # Environment.GetEnvironmentVariable(key) ?? default.
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = os.environ.get("LLM_MODEL", "")
        # Ollama needs no key; the SDK requires *something*, so use a dummy.
        self._api_key = os.environ.get("LLM_API_KEY") or "not-needed"
        # Optional caps: keep output room for the answer. Reasoning models
        # (e.g. gpt-oss-20b) can burn the whole completion budget "thinking"
        # and return finish_reason=length with EMPTY content — a bigger
        # max_tokens and/or a low reasoning effort fixes that.
        # Parsed in complete() (not here) so a bad value surfaces as a
        # clean ProviderError (HTTP 502) instead of an import-time crash.
        self._max_tokens_raw = os.environ.get("LLM_MAX_TOKENS", "").strip()
        # Provider-specific (Pollinations honors "low"); plain OpenAI-style
        # servers may ignore or reject it — only set it if you need it.
        self._reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip() or None

    def _max_tokens(self) -> int | None:
        if not self._max_tokens_raw:
            return None
        try:
            return int(self._max_tokens_raw)
        except ValueError as exc:
            raise ProviderError(
                f"LLM_MAX_TOKENS must be an integer, got {self._max_tokens_raw!r}."
            ) from exc

    def complete(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI  # lazy: keeps `openai` out of import time
        except ImportError as exc:
            raise ProviderError(
                "The 'openai' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        client = OpenAI(
            base_url=self.base_url,
            api_key=self._api_key,
            timeout=60,  # seconds; local models can be slow on first tokens
            max_retries=1,  # one retry on transient failures, then give up
        )
        try:
            create_kwargs: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,  # low: we want faithful translation, not prose
            }
            if (max_tokens := self._max_tokens()) is not None:
                create_kwargs["max_tokens"] = max_tokens
            if self._reasoning_effort:
                create_kwargs["reasoning_effort"] = self._reasoning_effort
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            # Never leak the key: scrub it if it somehow appears in the message.
            msg = str(exc)
            if self._api_key and self._api_key != "not-needed":
                msg = msg.replace(self._api_key, "[redacted]")
            raise ProviderError(
                f"OpenAI-compatible provider error ({type(exc).__name__}): {msg}"
            ) from exc
        content = resp.choices[0].message.content or ""
        return content


class AnthropicProvider(LLMProvider):
    """Anthropic API. The `anthropic` package is OPTIONAL (lazy import)."""

    name = "anthropic"

    def __init__(self) -> None:
        self.model = os.environ.get("ANTHROPIC_MODEL", "")
        self._api_key = os.environ.get("ANTHROPIC_API_KEY") or ""

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic  # lazy: not in requirements.txt
        except ImportError as exc:
            raise ProviderError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install anthropic "
                "(it is an optional dependency)."
            ) from exc
        client = (
            anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        )
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.2,
            )
        except Exception as exc:
            msg = str(exc)
            if self._api_key:
                msg = msg.replace(self._api_key, "[redacted]")
            raise ProviderError(
                f"Anthropic provider error ({type(exc).__name__}): {msg}"
            ) from exc
        # Response blocks can be text or tool-use; keep just the text parts.
        return "\n".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )


def get_provider() -> LLMProvider:
    """Build the configured provider. Unknown values fall back to 'none'."""
    name = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    if name == "openai_compat":
        return OpenAICompatProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "manual":
        return ManualProvider()
    return NoneProvider()  # "none" and anything unrecognised -> safe default


def provider_model(provider: LLMProvider) -> str:
    """Model name to report for /api/status (never a key)."""
    return provider.model or ""


def has_api_key() -> bool:
    """Boolean only — the key value never leaves the process env."""
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """The instruction block sent to every provider. Plain C# out, no JSON."""
    return """You translate Python code into idiomatic modern C# (C# 12, .NET 8) \
so a senior C# developer can learn Python by comparison. This is a LEARNING \
tool, not a production transpiler: readable, idiomatic C# and honest notes \
matter more than compilable output.

Rules you MUST follow:
1. Use idiomatic modern C#: records, LINQ, pattern matching, `using` \
declarations, async/await, and nullable reference types where they fit.
2. Python is dynamically typed: infer the most likely C# types and emit ONE \
`// ASSUME:` comment line per assumption near the top, e.g. \
`// ASSUME: items is List<int>`.
3. Preserve the order and structure of the Python code so line mapping works.
4. Start EVERY translated block with a marker comment quoting the Python line \
being translated: `// PY: <line>: <exact python source line>`, e.g. \
`// PY: 4: x = -7 // 2`. One marker per Python statement/expression block.
5. RETURN ONLY C# CODE. No markdown fences, no JSON, no prose outside `//` \
comments. The reply must start with C# (or a `//` comment) and contain \
nothing else.
6. FLAG semantic differences in `// NOTE:` comments instead of silently \
"fixing" them: truthiness, integer vs float division, mutable default args, \
late-binding closures, duck typing, generators vs IEnumerable laziness, dict \
ordering, `is` vs `==`, string immutability and slicing, negative indexing, \
exceptions (try/except/else/finally), and `with` semantics.
7. For third-party libraries (numpy, pandas, requests), name the closest \
.NET equivalent and mark it "approximate" in a `// NOTE:`.
8. NEVER invent APIs. If there is no clean equivalent, say so in a \
`// NOTE:` comment.

Example reply shape:
// ASSUME: items is List<int>
// PY: 1: x = -7 // 2
var x = (int)Math.Floor((double)-7 / 2); // NOTE: Python // floors (-4); C# int / truncates (-3)
// PY: 2: squares = [i*i for i in range(5)]
var squares = Enumerable.Range(0, 5).Select(i => i * i).ToList();
"""


def build_user_prompt(python_code: str, findings: list) -> str:
    """User prompt: the code plus the AST findings so the model stays
    consistent with catalog.json (the deterministic source of truth)."""
    catalog = load_catalog()
    lines = ["Translate the following Python code to C#.", "", "```python", python_code, "```", ""]
    if findings:
        lines.append(
            "The static analyser already detected these constructs "
            "(stay consistent with their C# equivalents and notes):"
        )
        for f in findings:
            entry = catalog.get(f["construct_key"], {})
            title = entry.get("title", f["construct_key"])
            note = entry.get("note", "")
            lines.append(f"- Python line {f['line']}: {title}. {note}")
    else:
        lines.append("The static analyser found no catalogued constructs.")
    lines.append("")
    lines.append("Reply with ONLY the C# code per the system instructions.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Marker parsing (deterministic line mapping)
# ---------------------------------------------------------------------------

# `// PY: 4: x = -7 // 2` — tolerant of spacing; requires the colon after
# the line number, so `// PY: abc:` or `//PY no-colon` simply don't match
# and are ignored (they're "malformed markers").
PY_MARKER_RE = re.compile(r"^\s*//\s*PY:\s*(\d+)\s*:")
ASSUME_RE = re.compile(r"^\s*//\s*ASSUME:\s*(.+?)\s*$")

MAPPING_NOTICE = (
    "The model reply contained no usable // PY: markers, "
    "so line cross-highlighting is disabled."
)


@dataclass
class ParsedReply:
    line_map: list[LineMap] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    cleaned: str = ""
    markers_missing: bool = False


def parse_markers(csharp: str) -> ParsedReply:
    """Parse `// PY:` markers and `// ASSUME:` lines from a model reply.

    Defensive fence-stripping: any line starting with ``` is dropped, so a
    reply wrapped in ```csharp ... ``` still parses.
    Returns markers_missing=True when ZERO valid markers were found — the
    caller then disables cross-highlighting but KEEPS the C# verbatim.
    Malformed markers are ignored; valid ones are still used.
    """
    # Drop markdown fence lines defensively (```csharp, ```, ...).
    kept = [ln for ln in csharp.splitlines() if not ln.strip().startswith("```")]
    cleaned = "\n".join(kept)

    markers: list[tuple[int, int]] = []  # (csharp_line_no, python_line_no), 1-based
    assumptions: list[str] = []
    for i, ln in enumerate(kept, start=1):
        m = PY_MARKER_RE.match(ln)
        if m:
            markers.append((i, int(m.group(1))))
            continue
        a = ASSUME_RE.match(ln)
        if a:
            assumptions.append(a.group(1))

    if not markers:
        return ParsedReply(
            line_map=[], assumptions=assumptions, cleaned=cleaned, markers_missing=True
        )

    # Last non-blank line: trailing blank lines don't belong to any block.
    last_content = len(kept)
    while last_content > 0 and not kept[last_content - 1].strip():
        last_content -= 1

    line_map: list[LineMap] = []
    seen_py: set[int] = set()  # one entry per distinct Python line, first wins
    for idx, (cs_marker_line, py_line) in enumerate(markers):
        if py_line in seen_py:
            continue
        seen_py.add(py_line)
        cs_start = cs_marker_line + 1
        if idx + 1 < len(markers):
            cs_end = markers[idx + 1][0] - 1
        else:
            cs_end = last_content
        if cs_end < cs_start:
            cs_end = cs_start  # empty block: point at the line after the marker
        line_map.append(
            LineMap(py_start=py_line, py_end=py_line, cs_start=cs_start, cs_end=cs_end)
        )

    return ParsedReply(
        line_map=line_map,
        assumptions=assumptions,
        cleaned=cleaned,
        markers_missing=False,
    )


# ---------------------------------------------------------------------------
# Merge + translate
# ---------------------------------------------------------------------------


def _notes_for(findings: list, catalog: dict) -> list[Note]:
    """Notes ALWAYS come from the annotator + catalog — never from the model."""
    notes: list[Note] = []
    for f in findings:
        entry = catalog.get(f["construct_key"])
        if entry is None:
            # Defensive: annotator and catalog drifted. Keep the finding
            # visible rather than crashing.
            entry = {
                "title": f["construct_key"],
                "note": "Detected construct (no catalog entry).",
                "severity": "info",
            }
        notes.append(
            Note(
                py_line=f["line"],
                construct=f["construct_key"],
                severity=entry.get("severity", "info"),
                message=entry.get("note", ""),
            )
        )
    return notes


def merge(
    python_code: str,
    csharp_raw: str,
    findings: list,
    catalog: dict,
    provider_name: str,
) -> TranslateResponse:
    """Deterministic merge: model reply + annotator findings -> response."""
    parsed = parse_markers(csharp_raw)
    return TranslateResponse(
        csharp=parsed.cleaned,  # kept verbatim (minus fence lines); never dropped
        assumptions=parsed.assumptions,
        line_map=parsed.line_map,
        notes=_notes_for(findings, catalog),
        errors=[],
        provider=provider_name,
        banner=None,
        mapping_notice=MAPPING_NOTICE if parsed.markers_missing else None,
    )


NONE_BANNER = (
    "Annotate-only mode: no LLM provider configured. "
    "Set LLM_PROVIDER to translate (see README 'Choosing a provider')."
)
MANUAL_BANNER = (
    "Manual mode: use 'Copy prompt' to grab the prompt, translate it in your "
    "own chat app, then 'Paste result' to merge the reply."
)

# In-memory LRU cache: sha256(provider | model | code) -> TranslateResponse.
# Saves API cost on repeated translations.  (functools.lru_cache can't be
# used directly because provider objects aren't hashable.)
_CACHE: OrderedDict[str, TranslateResponse] = OrderedDict()
_CACHE_MAX = 128


def _cache_key(provider: LLMProvider, code: str) -> str:
    digest = hashlib.sha256(f"{provider.name}|{provider.model}|{code}".encode("utf-8"))
    return digest.hexdigest()


def clear_cache() -> None:
    """Test helper: empty the translation cache."""
    _CACHE.clear()


def _has_content(raw: str) -> bool:
    """True if a model reply contains any real content.

    Blank lines and markdown fence lines don't count — a reply that is
    only whitespace (or only fences) is treated as empty. `//` comment
    lines DO count: a markers-only reply is odd but still usable.
    """
    for ln in raw.splitlines():
        stripped = ln.strip()
        if stripped and not stripped.startswith("```"):
            return True
    return False


def translate(python_code: str, provider: LLMProvider | None = None) -> TranslateResponse:
    """Full pipeline: syntax check -> annotate -> (maybe) LLM -> merge.

    Raises PythonSyntaxError BEFORE any provider call, so invalid Python
    never costs an API call.  annotate-only providers ("none", "manual")
    return notes with an explanatory banner and no C#.
    """
    provider = provider or get_provider()

    # 1. Syntax check first: ast.parse raises SyntaxError with lineno/offset.
    try:
        findings = annotate(python_code)
    except SyntaxError as exc:
        raise PythonSyntaxError(
            line=exc.lineno or 0,
            column=exc.offset or 0,
            message=f"{exc.msg}",
        ) from exc

    catalog = load_catalog()

    # 2. Annotate-only modes: notes + banner, no C#, no provider call.
    if provider.name in ("none", "manual"):
        banner = NONE_BANNER if provider.name == "none" else MANUAL_BANNER
        return TranslateResponse(
            csharp="",
            assumptions=[],
            line_map=[],
            notes=_notes_for(findings, catalog),
            errors=[],
            provider=provider.name,
            banner=banner,
            mapping_notice=None,
        )

    # 3. Cached? (keyed by provider + model + code)
    key = _cache_key(provider, python_code)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    # 4. Call the model, then merge deterministically.
    system = build_system_prompt()
    user = build_user_prompt(python_code, findings)
    raw = provider.complete(system, user)
    if not _has_content(raw):
        # Free-tier models occasionally return an empty reply. The SDK's
        # max_retries only covers transport errors, not empty content,
        # so retry once here before giving up.
        raw = provider.complete(system, user)
    if not _has_content(raw):
        raise ProviderError(
            "The model returned an empty reply twice in a row. "
            "This happens occasionally on free tiers — please hit "
            "Translate again."
        )
    response = merge(python_code, raw, findings, catalog, provider.name)

    _CACHE[key] = response
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)  # evict least-recently-used
    return response


def syntax_error_response(
    line: int, column: int, message: str, provider_name: str
) -> TranslateResponse:
    """The hardened syntax-error contract: 200, csharp == "", everything
    empty except errors (+ the usual provider/banner fields)."""
    return TranslateResponse(
        csharp="",
        assumptions=[],
        line_map=[],
        notes=[],
        errors=[SyntaxErrorItem(line=line, column=column, message=message)],
        provider=provider_name,
        banner=None,
        mapping_notice=None,
    )
