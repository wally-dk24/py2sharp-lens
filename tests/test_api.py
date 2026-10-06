"""API tests via FastAPI's TestClient.  All providers are mocked — no real
LLM API is ever called.  Covers the endpoint contracts, the hardened
syntax-error response, the 20 KB limit, manual-mode round-trip, the
mapping notice, and /healthz."""

import pytest
from fastapi.testclient import TestClient

from app import translator
from app.main import app
from app.translator import LLMProvider

client = TestClient(app)


@pytest.fixture(autouse=True)
def _none_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


class RecordingProvider(LLMProvider):
    """Fake that records whether complete() was ever called."""

    name = "recording"
    model = "recording-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return "// PY: 1: x = 1\nvar x = 1;"


# --- health & meta -----------------------------------------------------------


def test_healthz():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_catalog_has_40_plus_entries():
    res = client.get("/api/catalog")
    assert res.status_code == 200
    assert len(res.json()) >= 40


def test_status_shape():
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "none"
    assert body["has_api_key"] is False  # boolean only: the key value never leaves env
    assert set(body.keys()) == {"provider", "model", "has_api_key", "catalog_constructs"}
    assert body["catalog_constructs"] >= 40


def test_index_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "Py2Sharp Lens" in res.text


# --- annotate -----------------------------------------------------------------


def test_annotate_floordiv_and_listcomp():
    res = client.post(
        "/api/annotate", json={"python": "x = -7 // 2\nsquares = [i*i for i in range(5)]"}
    )
    assert res.status_code == 200
    findings = {(f["construct_key"], f["line"]) for f in res.json()["findings"]}
    assert ("FloorDiv", 1) in findings
    assert ("ListComp", 2) in findings
    # Enriched with catalog content:
    floordiv = next(f for f in res.json()["findings"] if f["construct_key"] == "FloorDiv")
    assert floordiv["severity"] == "gotcha"
    assert "-7 // 2 == -4" in floordiv["note"]


def test_annotate_invalid_python_is_422():
    res = client.post("/api/annotate", json={"python": "def broken(:"})
    assert res.status_code == 422
    assert "line 1" in res.json()["detail"]


# --- translate -----------------------------------------------------------------


def test_translate_none_provider_banner_and_notes():
    res = client.post("/api/translate", json={"python": "x = -7 // 2"})
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "none"
    assert body["csharp"] == ""
    assert body["banner"]  # clear annotate-only banner
    assert any(n["construct"] == "FloorDiv" for n in body["notes"])
    assert body["errors"] == []
    assert body["mapping_notice"] is None


def test_syntax_error_contract_and_provider_not_called(monkeypatch):
    """Hardened contract: HTTP 200, csharp == "", everything empty except
    errors[{line, column, message}].  The provider must NOT be called."""
    rec = RecordingProvider()
    monkeypatch.setattr(translator, "get_provider", lambda: rec)

    res = client.post("/api/translate", json={"python": "def broken(:"})
    assert res.status_code == 200
    body = res.json()
    assert body["csharp"] == ""  # guaranteed empty string, never null/stale
    assert body["assumptions"] == []
    assert body["line_map"] == []
    assert body["notes"] == []
    assert len(body["errors"]) == 1
    err = body["errors"][0]
    assert err["line"] == 1
    assert err["column"] == 12
    assert isinstance(err["message"], str) and err["message"]
    assert body["provider"] == "recording"  # usual provider field still present
    assert rec.calls == 0  # provider.complete() never invoked


def test_oversize_input_is_413():
    res = client.post("/api/translate", json={"python": "x = 1\n" * 5000})  # ~30 KB
    assert res.status_code == 413
    assert res.json()["detail"] == "Input exceeds 20 KB"


# --- manual mode ---------------------------------------------------------------


def test_prompt_returns_text_with_code():
    code = "x = -7 // 2"
    res = client.post("/api/prompt", json={"python": code})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    text = res.text
    assert "=== SYSTEM ===" in text
    assert "=== USER ===" in text
    assert code in text
    assert "RETURN ONLY C#" in text


def test_import_round_trip_builds_line_map_and_notes():
    code = "x = -7 // 2\nsquares = [i*i for i in range(5)]"
    reply = (
        "// ASSUME: x is int\n"
        "// PY: 1: x = -7 // 2\n"
        "var x = (int)Math.Floor((double)-7 / 2);\n"
        "// PY: 2: squares = [i*i for i in range(5)]\n"
        "var squares = Enumerable.Range(0, 5).Select(i => i * i).ToList();"
    )
    res = client.post("/api/import", json={"python": code, "reply": reply})
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "manual"
    assert body["assumptions"] == ["x is int"]
    assert [(m["py_start"], m["cs_start"], m["cs_end"]) for m in body["line_map"]] == [
        (1, 3, 3),
        (2, 5, 5),
    ]
    assert any(n["construct"] == "FloorDiv" for n in body["notes"])
    assert body["mapping_notice"] is None
    assert body["errors"] == []


def test_import_marker_less_reply_keeps_csharp_and_notice():
    reply = "var x = 1;\nvar y = 2;"
    res = client.post("/api/import", json={"python": "x = 1", "reply": reply})
    assert res.status_code == 200
    body = res.json()
    assert body["csharp"] == reply  # kept verbatim, never dropped
    assert body["line_map"] == []
    assert body["mapping_notice"]  # cross-highlighting disabled notice


def test_import_invalid_python_returns_errors():
    res = client.post("/api/import", json={"python": "def broken(:", "reply": "x"})
    assert res.status_code == 200
    body = res.json()
    assert body["csharp"] == ""
    assert len(body["errors"]) == 1
    assert body["errors"][0]["line"] == 1
