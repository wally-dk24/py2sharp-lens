"""Translator tests.  Every provider is mocked/faked — no real LLM API is
ever called.  Covers the unified plain-C# contract: marker parsing, fence
stripping, ASSUME lines, the marker-less mapping notice, retry config,
and the syntax-error-before-provider ordering."""

import sys
import types

import pytest

from app import translator
from app.translator import (
    AnthropicProvider,
    LLMProvider,
    NoneProvider,
    OpenAICompatProvider,
    PythonSyntaxError,
    build_system_prompt,
    build_user_prompt,
    clear_cache,
    merge,
    parse_markers,
    translate,
)

CATALOG = translator.load_catalog()

CANNED = """// ASSUME: items is List<int>
// PY: 1: x = -7 // 2
var x = (int)Math.Floor((double)-7 / 2); // NOTE: floors, not truncates
// PY: 2: squares = [i*i for i in range(5)]
var squares = items.Select(i => i * i).ToList();
"""

FENCED = "```csharp\n" + CANNED + "```\n"

NO_MARKERS = """var x = 1;
var y = x + 2;
"""

MIXED_MARKERS = """// PY: abc:
var bad = 1;
//PY no-colon
var alsoBad = 2;
// PY: 3: ok = True
var ok = true;
"""


class FakeProvider(LLMProvider):
    """Recording fake: returns canned C#, counts complete() calls."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = CANNED) -> None:
        self.reply = reply
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return self.reply


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


# --- marker parsing --------------------------------------------------------


def test_parse_markers_line_map():
    parsed = parse_markers(CANNED)
    assert not parsed.markers_missing
    assert parsed.assumptions == ["items is List<int>"]
    assert [(m.py_start, m.py_end, m.cs_start, m.cs_end) for m in parsed.line_map] == [
        (1, 1, 3, 3),  # marker on line 2 -> block is line 3
        (2, 2, 5, 5),  # marker on line 4 -> block is line 5
    ]


def test_parse_markers_keeps_csharp_verbatim():
    parsed = parse_markers(CANNED)
    assert parsed.cleaned == CANNED.rstrip("\n")


def test_fence_stripping():
    parsed = parse_markers(FENCED)
    assert not parsed.markers_missing
    assert "```" not in parsed.cleaned
    assert len(parsed.line_map) == 2


def test_no_markers_mapping_notice():
    parsed = parse_markers(NO_MARKERS)
    assert parsed.markers_missing
    assert parsed.line_map == []
    assert parsed.cleaned == NO_MARKERS.rstrip("\n")  # C# kept verbatim


def test_mixed_malformed_and_valid_markers():
    parsed = parse_markers(MIXED_MARKERS)
    assert not parsed.markers_missing  # at least one valid marker exists
    assert [(m.py_start, m.cs_start, m.cs_end) for m in parsed.line_map] == [
        (3, 6, 6)
    ]


def test_duplicate_py_line_keeps_first():
    reply = "// PY: 1: a = 1\nvar a = 1;\n// PY: 1: a = 1\nvar a2 = 1;\n"
    parsed = parse_markers(reply)
    assert len(parsed.line_map) == 1
    assert parsed.line_map[0].py_start == 1


# --- merge -----------------------------------------------------------------


def test_merge_notes_come_from_catalog_not_model():
    findings = [
        {"line": 1, "end_line": 1, "construct_key": "FloorDiv"},
    ]
    resp = merge("x = -7 // 2", CANNED, findings, CATALOG, "fake")
    assert len(resp.notes) == 1
    note = resp.notes[0]
    assert note.py_line == 1
    assert note.construct == "FloorDiv"
    assert note.severity == "gotcha"
    assert "-7 // 2 == -4" in note.message  # the required -4 vs -3 note
    assert resp.mapping_notice is None


def test_merge_marker_less_reply():
    findings = [{"line": 1, "end_line": 1, "construct_key": "FloorDiv"}]
    resp = merge("x = -7 // 2", NO_MARKERS, findings, CATALOG, "fake")
    assert resp.csharp == NO_MARKERS.rstrip("\n")  # verbatim, never dropped
    assert resp.line_map == []
    assert resp.mapping_notice is not None
    assert "// PY:" in resp.mapping_notice
    # Notes still come from the annotator even without markers.
    assert len(resp.notes) == 1


# --- translate ---------------------------------------------------------------


def test_translate_full_pipeline_fake_provider():
    fake = FakeProvider()
    resp = translate("x = -7 // 2\nsquares = [i*i for i in range(5)]", provider=fake)
    assert fake.calls == 1
    assert resp.provider == "fake"
    assert resp.assumptions == ["items is List<int>"]
    assert len(resp.line_map) == 2
    assert any(n.construct == "FloorDiv" for n in resp.notes)
    assert any(n.construct == "ListComp" for n in resp.notes)
    assert resp.errors == []
    assert resp.banner is None


def test_syntax_error_before_provider_called():
    fake = FakeProvider()
    with pytest.raises(PythonSyntaxError) as exc_info:
        translate("def broken(:", provider=fake)
    assert fake.calls == 0  # provider must NOT be called
    assert exc_info.value.line == 1


def test_none_provider_annotate_only():
    resp = translate("x = -7 // 2", provider=NoneProvider())
    assert resp.provider == "none"
    assert resp.csharp == ""
    assert resp.banner  # clear annotate-only banner
    assert any(n.construct == "FloorDiv" for n in resp.notes)


def test_translate_cached_by_sha256():
    fake = FakeProvider()
    code = "unique_cache_probe_xyz = 1"
    r1 = translate(code, provider=fake)
    r2 = translate(code, provider=fake)
    assert fake.calls == 1  # second call served from cache
    assert r1.csharp == r2.csharp


def test_prompts_mention_plain_csharp_contract():
    system = build_system_prompt()
    assert "RETURN ONLY C#" in system
    assert "// PY:" in system
    assert "// ASSUME:" in system


def test_user_prompt_includes_code_and_findings():
    findings = [{"line": 1, "end_line": 1, "construct_key": "FloorDiv"}]
    user = build_user_prompt("x = -7 // 2", findings)
    assert "x = -7 // 2" in user
    assert "Floor division" in user  # catalog title for the finding


# --- provider construction ---------------------------------------------------


def test_openai_compat_provider_options(monkeypatch):
    """The SDK must be built with timeout=60 and max_retries=1 (one retry)."""
    seen = {}

    class FakeCompletions:
        def create(self, **kwargs):
            msg = types.SimpleNamespace(content="var x = 1;")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-coder:7b")

    provider = OpenAICompatProvider()
    assert provider.complete("sys", "usr") == "var x = 1;"
    assert seen["timeout"] == 60
    assert seen["max_retries"] == 1
    assert seen["base_url"] == "http://localhost:11434/v1"
    assert seen["api_key"] == "not-needed"  # Ollama needs no key
    assert "sk-" not in str(seen)  # no real key anywhere near here


def test_anthropic_provider_missing_package(monkeypatch):
    """anthropic is optional: a clear error, not an ImportError traceback."""
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    # Block the real import by making it unimportable.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = AnthropicProvider()
    with pytest.raises(translator.ProviderError, match="not installed"):
        provider.complete("sys", "usr")


def test_get_provider_defaults_to_none(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert translator.get_provider().name == "none"


def test_get_provider_unknown_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "definitely-not-a-provider")
    assert translator.get_provider().name == "none"
