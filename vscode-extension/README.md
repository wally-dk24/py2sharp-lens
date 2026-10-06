# VS Code extension — Phase 2 (planned)

The extension will reuse this project's backend unchanged:

- Command **Py2Sharp: Translate Selection** (or the whole file if nothing is
  selected) → opens a side-by-side webview with the C# and the notes.
- Hover provider on Python files: hovering a known construct (list
  comprehension, `with`, `yield`, `//`, truthiness check, …) shows the C#
  equivalent and the note from `app/catalog.json` — no LLM call needed.
- Setting: backend URL (default `http://localhost:8000`).
