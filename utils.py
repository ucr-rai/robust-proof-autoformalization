"""Shared helpers for the edit-construction pipelines.

Cross-cutting utilities used by all three edit types (and by evaluate.py):
JSONL IO, context-anchored offset repair, robust LaTeX-JSON parsing, and the
Gemini model factory. Nothing here is specific to a single edit type —
type-specific logic lives in number_edit.py / symbol_edit.py / step_edit.py,
and Lean-specific handling lives in verify.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ── JSONL IO ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Gemini (google-genai SDK) ────────────────────────────────────────────────

def _safety_off():
    if types is None:
        return []
    return [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
        for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
    ]


_SAFETY_OFF = _safety_off()


class _GeminiModel:
    """Thin adapter exposing the legacy ``.generate_content(prompt).text`` API
    over the google-genai client, so call sites don't need to change."""

    def __init__(self, model_name: str, config=None):
        if genai is None or types is None:
            raise RuntimeError("Install google-genai to use Gemini-backed generation")
        self._model = model_name
        self._config = config or types.GenerateContentConfig(safety_settings=_SAFETY_OFF)

    def generate_content(self, prompt: str):
        return _client().models.generate_content(
            model=self._model, contents=prompt, config=self._config,
        )


_CLIENT = None


def _client():
    """Lazily create one shared google-genai client from GOOGLE_API_KEY."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            sys.exit("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def make_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Gemini model factory with safety filters disabled.

    Math/Lean content gets false-positively flagged by the default safety
    filter; all four HarmCategory thresholds are set to BLOCK_NONE for academic
    eval. Single source of truth for every Gemini-judge / labeling script.
    """
    return _GeminiModel(model_name)


# ── Robust LaTeX-JSON parsing ────────────────────────────────────────────────

def robust_json_loads(text: str):
    """Parse JSON text that contains unescaped LaTeX backslashes.

    ALWAYS doubles backslashes inside string values first, then parses, so
    \\frac stays \\frac instead of decoding to a form feed + "rac". Returns
    parsed dict/list, or None if parsing fails.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    fixed = _fix_backslashes_in_strings(text)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = json.loads(text)
        if "\x0c" not in repr(result) and "\x08" not in repr(result):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _fix_backslashes_in_strings(text: str) -> str:
    """Double all backslashes inside JSON string values (preserve \\")."""
    result = []
    in_string = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
            i += 1
        else:
            if ch == "\\":
                if i + 1 < n and text[i + 1] == '"':
                    result.append('\\')
                    result.append('"')
                    i += 2
                elif i + 1 < n and text[i + 1] == "\\":
                    result.append("\\\\")
                    result.append("\\\\")
                    i += 2
                else:
                    result.append("\\\\")
                    i += 1
            elif ch == '"':
                in_string = False
                result.append(ch)
                i += 1
            else:
                result.append(ch)
                i += 1

    return "".join(result)


# ── Context-anchored offset repair ───────────────────────────────────────────

def fix_offset_by_context(
    text: str,
    value: str,
    context: str,
    old_start: int = -1,
    old_end: int = -1,
) -> Optional[tuple]:
    """Relocate ``value`` inside ``text`` using ``context`` as a semantic anchor.

    Returns ``(new_start, new_end)`` with ``text[new_start:new_end] == value``,
    or ``None`` if no reliable match is found. Strategy, in order: keep a
    correct offset; use a unique occurrence; match ``context`` (exact, then
    unescaped, then whitespace-collapsed); finally distinctive-word overlap.
    """
    if not value or not text:
        return None

    if 0 <= old_start < old_end <= len(text) and text[old_start:old_end] == value:
        return old_start, old_end

    all_positions = []
    i = 0
    while True:
        i = text.find(value, i)
        if i < 0:
            break
        all_positions.append(i)
        i += 1
    if len(all_positions) == 1:
        p = all_positions[0]
        return p, p + len(value)
    if not all_positions:
        return None

    ctx = (context or "").strip()
    if ctx:
        ctx_pos = text.find(ctx)
        if ctx_pos >= 0:
            rel = ctx.find(value)
            if rel >= 0:
                return ctx_pos + rel, ctx_pos + rel + len(value)

        ctx_unesc = ctx.replace("\\\\", "\\")
        if ctx_unesc != ctx:
            ctx_pos = text.find(ctx_unesc)
            if ctx_pos >= 0:
                rel = ctx_unesc.find(value)
                if rel >= 0:
                    return ctx_pos + rel, ctx_pos + rel + len(value)

        def _collapse(s):
            return re.sub(r"\s+", " ", s)

        text_c = _collapse(text)
        ctx_c = _collapse(ctx_unesc)
        if ctx_c in text_c:
            c_pos = text_c.find(ctx_c)
            orig_pos, walked = 0, 0
            while orig_pos < len(text) and walked < c_pos:
                ch = text[orig_pos]
                orig_pos += 1
                if ch.isspace():
                    while orig_pos < len(text) and text[orig_pos].isspace():
                        orig_pos += 1
                    walked += 1
                else:
                    walked += 1
            window_end = min(len(text), orig_pos + len(ctx_unesc) + 20)
            rel = text.find(value, orig_pos, window_end)
            if rel >= 0:
                return rel, rel + len(value)

    ctx_words = set(re.findall(r"[A-Za-z][A-Za-z_]{2,}|\d+", ctx))
    ctx_words.discard(value)
    if not ctx_words:
        return None

    best_pos, best_score = -1, 0
    for pos in all_positions:
        window_start = max(0, pos - 60)
        window_end = min(len(text), pos + len(value) + 60)
        window = text[window_start:window_end]
        window_words = set(re.findall(r"[A-Za-z][A-Za-z_]{2,}|\d+", window))
        score = len(ctx_words & window_words)
        if score > best_score:
            best_score = score
            best_pos = pos
    if best_pos >= 0 and best_score >= 2:
        return best_pos, best_pos + len(value)

    return None
