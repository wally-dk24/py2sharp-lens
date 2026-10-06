/* Py2Sharp Lens frontend — dependency-free vanilla JS + Monaco from CDN. */
"use strict";

const $ = (id) => document.getElementById(id);

let pyEditor = null;
let csEditor = null;
let lineMap = [];          // [{py_start, py_end, cs_start, cs_end}, ...]
let pyDecorations = [];    // decoration ids on the Python editor
let csDecorations = [];    // decoration ids on the C# editor

const DEFAULT_CODE = `x = -7 // 2
squares = [i * i for i in range(5)]


def greet(name):
    return f"Hello, {name}!" if name else "Hello, stranger!"


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
`;

// --- Monaco bootstrap -------------------------------------------------------
require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" } });
require(["vs/editor/editor.main"], init);

function init() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = dark ? "vs-dark" : "vs";
  const base = {
    theme,
    fontSize: 14,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    automaticLayout: true,
  };
  pyEditor = monaco.editor.create($("py-pane"), {
    ...base, value: DEFAULT_CODE, language: "python",
  });
  csEditor = monaco.editor.create($("cs-pane"), {
    ...base, value: "", language: "csharp", readOnly: true,
    placeholder: "C# translation appears here…",
  });

  // Ctrl+Enter (Cmd+Enter on macOS) translates.
  pyEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, doTranslate);

  // Click-to-highlight, both directions.
  pyEditor.onDidChangeCursorPosition((e) => highlightFromPy(e.position.lineNumber));
  csEditor.onDidChangeCursorPosition((e) => highlightFromCs(e.position.lineNumber));

  $("translate-btn").addEventListener("click", doTranslate);
  $("copy-prompt-btn").addEventListener("click", copyPrompt);
  $("paste-result-btn").addEventListener("click", () => { $("paste-modal").hidden = false; });
  $("paste-cancel-btn").addEventListener("click", () => { $("paste-modal").hidden = true; });
  $("paste-import-btn").addEventListener("click", importReply);
  $("mapping-notice-dismiss").addEventListener("click", () => { $("mapping-notice").hidden = true; });

  loadStatus();
}

// --- status / banner ---------------------------------------------------------
async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    $("provider-badge").textContent =
      `provider: ${s.provider}` + (s.model ? ` · ${s.model}` : "") +
      ` · catalog: ${s.catalog_constructs}`;
    if (s.provider === "none") {
      showBanner("Annotate-only mode: no LLM provider configured. Set LLM_PROVIDER to enable translation — see README “Choosing a provider”.");
    } else if (s.provider === "manual") {
      showBanner("Manual mode: use “Copy prompt”, translate in your own chat app, then “Paste result”.");
    }
  } catch (e) {
    setStatus("Could not reach the backend.");
  }
}

function showBanner(text) {
  const b = $("banner");
  b.textContent = text;
  b.hidden = false;
}

function setStatus(text) {
  $("status").textContent = text;
}

// --- translate ---------------------------------------------------------------
async function doTranslate() {
  const btn = $("translate-btn");
  btn.disabled = true;
  btn.textContent = "Translating…";
  setStatus("");
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ python: pyEditor.getValue() }),
    });
    if (res.status === 413) {
      showError("Input exceeds 20 KB. Please paste a smaller snippet.");
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || `Backend error (HTTP ${res.status}).`);
      return;
    }
    renderResult(await res.json());
  } catch (e) {
    showError("Could not reach the backend. Is uvicorn running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Translate";
  }
}

// --- rendering ----------------------------------------------------------------
function renderResult(data) {
  // Syntax-error contract: a non-empty errors array means the C# pane,
  // decorations, and notes MUST be cleared — no stale output may remain.
  if (data.errors && data.errors.length > 0) {
    clearAll();
    const e = data.errors[0];
    showError(`Syntax error at line ${e.line}, column ${e.column}: ${e.message}`);
    return;
  }
  hideError();
  lineMap = data.line_map || [];
  csEditor.setValue(data.csharp || "");
  clearDecorations();
  renderNotes(data.notes || []);
  renderMappingNotice(data.mapping_notice);
  const n = (data.notes || []).length;
  setStatus(
    data.provider === "none" || data.provider === "manual"
      ? `Annotated ${n} construct${n === 1 ? "" : "s"} (no translation — ${data.provider} mode).`
      : `Translated with ${n} note${n === 1 ? "" : "s"}.`
  );
}

/** Clear everything: C# editor, decorations, notes, notice. */
function clearAll() {
  csEditor.setValue("");
  lineMap = [];
  clearDecorations();
  $("notes-list").innerHTML = "";
  $("notes-count").textContent = "";
  $("mapping-notice").hidden = true;
}

function clearDecorations() {
  pyDecorations = pyEditor.deltaDecorations(pyDecorations, []);
  csDecorations = csEditor.deltaDecorations(csDecorations, []);
}

function showError(text) {
  const box = $("error-box");
  box.textContent = text;
  box.hidden = false;
}

function hideError() {
  $("error-box").hidden = true;
  $("error-box").textContent = "";
}

function renderNotes(notes) {
  const list = $("notes-list");
  list.innerHTML = "";
  $("notes-count").textContent = notes.length ? `(${notes.length})` : "";
  for (const n of notes) {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "note-head";
    const chip = document.createElement("span");
    chip.className = `chip ${n.severity}`;
    chip.textContent = n.severity;
    const line = document.createElement("span");
    line.className = "note-line";
    line.textContent = `py:${n.py_line}`;
    const title = document.createElement("span");
    title.className = "note-title";
    title.textContent = n.construct;
    head.append(chip, line, title);
    const msg = document.createElement("p");
    msg.className = "note-msg";
    msg.textContent = n.message;
    li.append(head, msg);
    list.appendChild(li);
  }
}

function renderMappingNotice(notice) {
  const box = $("mapping-notice");
  if (notice) {
    $("mapping-notice-text").textContent = notice;
    box.hidden = false;
  } else {
    box.hidden = true;
  }
}

// --- cross-highlighting -------------------------------------------------------
// Clicking a Python line highlights the mapped C# lines and vice versa.
// Disabled entirely when lineMap is empty (e.g. marker-less model reply).
function highlightFromPy(pyLine) {
  clearDecorations();
  if (!lineMap.length) return; // no mapping: clicks do nothing
  const ranges = lineMap
    .filter((m) => pyLine >= m.py_start && pyLine <= m.py_end)
    .map((m) => new monaco.Range(m.cs_start, 1, m.cs_end, 1));
  if (ranges.length) {
    csDecorations = csEditor.deltaDecorations(csDecorations,
      ranges.map((r) => ({ range: r, options: { isWholeLine: true, className: "py2sharp-highlight" } })));
  }
}

function highlightFromCs(csLine) {
  clearDecorations();
  if (!lineMap.length) return; // no mapping: clicks do nothing
  const ranges = lineMap
    .filter((m) => csLine >= m.cs_start && csLine <= m.cs_end)
    .map((m) => new monaco.Range(m.py_start, 1, m.py_end, 1));
  if (ranges.length) {
    pyDecorations = pyEditor.deltaDecorations(pyDecorations,
      ranges.map((r) => ({ range: r, options: { isWholeLine: true, className: "py2sharp-highlight" } })));
  }
}

// Inject the highlight style (Monaco decorations need a CSS class).
const style = document.createElement("style");
style.textContent = ".py2sharp-highlight { background: var(--highlight); }";
document.head.appendChild(style);

// --- manual mode ---------------------------------------------------------------
async function copyPrompt() {
  setStatus("Building prompt…");
  try {
    const res = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ python: pyEditor.getValue() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || "Could not build the prompt (is the Python valid?).");
      return;
    }
    await navigator.clipboard.writeText(await res.text());
    setStatus("Prompt copied — paste it into your chat app, then use “Paste result”.");
  } catch (e) {
    showError("Could not reach the backend.");
  }
}

async function importReply() {
  const reply = $("paste-textarea").value;
  $("paste-modal").hidden = true;
  setStatus("Merging reply…");
  try {
    const res = await fetch("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ python: pyEditor.getValue(), reply }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || `Backend error (HTTP ${res.status}).`);
      return;
    }
    renderResult(await res.json());
  } catch (e) {
    showError("Could not reach the backend.");
  }
}
