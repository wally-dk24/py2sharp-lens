"""Pydantic request/response models for the Py2Sharp Lens API.

Pydantic is the standard FastAPI companion for this: declare the shape of
your JSON once as a class, and FastAPI validates requests and serialises
responses automatically.  A C# reader can think of these like C# records
used as DTOs, except validation runs for free.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    python: str = Field(description="Python source code to translate (max 20 KB)")


class ImportRequest(BaseModel):
    python: str = Field(description="The original Python source code")
    reply: str = Field(
        description="Model reply pasted back by the user (plain C# with // PY: markers)"
    )


class LineMap(BaseModel):
    """One Python line -> the C# lines that translate it (all 1-based, inclusive)."""

    py_start: int
    py_end: int
    cs_start: int
    cs_end: int


class Note(BaseModel):
    """One semantic gotcha, tied to a Python line number."""

    py_line: int
    construct: str  # catalog key, e.g. "FloorDiv"
    severity: str  # "info" | "gotcha" | "danger"
    message: str


class SyntaxErrorItem(BaseModel):
    line: int
    column: int
    message: str


class TranslateResponse(BaseModel):
    csharp: str = ""
    assumptions: list[str] = Field(default_factory=list)
    line_map: list[LineMap] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    errors: list[SyntaxErrorItem] = Field(default_factory=list)
    provider: str = "none"
    banner: str | None = None
    mapping_notice: str | None = None


class AnnotateFinding(BaseModel):
    line: int
    end_line: int
    construct_key: str
    title: str
    csharp: str
    note: str
    severity: str


class AnnotateResponse(BaseModel):
    findings: list[AnnotateFinding]


class StatusResponse(BaseModel):
    provider: str
    model: str
    has_api_key: bool
    catalog_constructs: int
