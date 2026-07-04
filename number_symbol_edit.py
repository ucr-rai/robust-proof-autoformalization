"""Number- and symbol-edit dataset construction (shared pipeline, one module).

Number and symbol edits share the same four-stage shape:

    label       — Gemini lists editable candidates per problem
    fix_offsets — repair char offsets via context anchoring
    select      — filter + sha256-seeded pick one candidate per source
    build       — perturb + rewrite -> {config_name: HF Dataset} in memory

The stage *drivers* (`_label`, `_fix_offsets`, `_select`, `_build`) are generic;
each edit type supplies only its own prompt, perturbation, selection filter, and
record builder. Entry points ``run_number`` / ``run_symbol`` return
{config_name: Dataset} for build.py to push to HuggingFace (no local final data).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, List, Optional

from datasets import Dataset

from utils import (
    ensure_dir,
    fix_offset_by_context,
    load_jsonl,
    make_gemini_model,
    robust_json_loads,
    write_jsonl,
)

EDIT_TYPES = ("statement_edit", "proof_edit")
SOURCE_OF = {"statement_edit": "statement", "proof_edit": "proof"}


# ═══════════════════════════════════════════════════════════════════════════
# Shared stage drivers
# ═══════════════════════════════════════════════════════════════════════════

def _seeded_rng(problem_name: str, source: str) -> random.Random:
    digest = hashlib.sha256(f"{problem_name}:{source}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _label_one(model, problem: dict, prompt: str, result_key: str) -> dict:
    name = problem.get("name", "unknown")
    stmt = str(problem.get("informal_statement", "") or "")
    proof = str(problem.get("informal_proof", "") or "")
    filled = prompt % (stmt, proof)

    for attempt in range(3):
        try:
            response = model.generate_content(filled)
            result = robust_json_loads(response.text.strip())
            return {"problem_name": name, result_key: result.get(result_key) or []}
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {name}: {e}")
            time.sleep(2)

    return {"problem_name": name, result_key: [], "error": "Failed after 3 attempts"}


def _label(dataset, output, prompt, result_key, model_name, limit):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    problems = load_jsonl(dataset)
    if limit > 0:
        problems = problems[:limit]
    print(f"Loaded {len(problems)} problems from {dataset}")

    model = make_gemini_model(model_name)

    with open(output, "w") as fout:
        for i, problem in enumerate(problems):
            name = problem.get("name", f"problem_{i}")
            print(f"[{i+1}/{len(problems)}] Labeling {name}...")
            result = _label_one(model, problem, prompt, result_key)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            time.sleep(0.5)


def _fix_offsets(dataset, labels, output, cand_key, value_key):
    problems = {row["name"]: row for row in load_jsonl(dataset)}
    labeled = load_jsonl(labels)

    fixed_rows = []
    n_in = n_out = n_dropped = 0

    for row in labeled:
        p = problems.get(row.get("problem_name", ""))
        if p is None:
            fixed_rows.append(row)
            continue
        stmt = str(p.get("informal_statement", "") or "")
        proof = str(p.get("informal_proof", "") or "")

        kept = []
        for cand in row.get(cand_key, []) or []:
            n_in += 1
            source = cand.get("source", "")
            text = stmt if source == "statement" else proof if source == "proof" else ""
            if not text:
                n_dropped += 1
                continue
            value = str(cand.get(value_key, "")).strip()
            result = fix_offset_by_context(
                text, value, cand.get("context", ""),
                int(cand.get("char_offset_start", -1)),
                int(cand.get("char_offset_end", -1)),
            )
            if result is None:
                n_dropped += 1
                continue
            cand = dict(cand)
            cand["char_offset_start"], cand["char_offset_end"] = result
            kept.append(cand)
            n_out += 1

        new_row = dict(row)
        new_row[cand_key] = kept
        fixed_rows.append(new_row)

    write_jsonl(output, fixed_rows)
    print(f"=== Offset Fix ===  in:{n_in} out:{n_out} dropped:{n_dropped} -> {output}")


def _select(labels, output, dataset, cand_key, filter_fn, pick_fn):
    """filter_fn(candidates, problem_row_or_None) -> filtered; pick_fn(cands, name, source) -> dict|None."""
    rows = load_jsonl(labels)
    problems_by_name = {r["name"]: r for r in load_jsonl(dataset)} if dataset else {}

    out_rows = []
    n_stmt = n_proof = 0
    for row in rows:
        name = row["problem_name"]
        cands = filter_fn(row.get(cand_key, []) or [], problems_by_name.get(name))
        stmt_pick = pick_fn([c for c in cands if c.get("source") == "statement"], name, "statement")
        proof_pick = pick_fn([c for c in cands if c.get("source") == "proof"], name, "proof")
        out_rows.append({"problem_name": name, "statement_edit": stmt_pick, "proof_edit": proof_pick})
        n_stmt += bool(stmt_pick)
        n_proof += bool(proof_pick)

    write_jsonl(output, out_rows)
    print(f"=== Select ===  problems:{len(out_rows)} stmt:{n_stmt} proof:{n_proof} -> {output}")


def _build(dataset, labels, limit, configs, record_fn: Callable) -> Dict[str, Dataset]:
    problems = load_jsonl(dataset)
    labels_by_name = {row["problem_name"]: row for row in load_jsonl(labels)}
    if limit > 0:
        problems = problems[:limit]

    outputs = {et: [] for et in EDIT_TYPES}
    skipped = Counter()

    for problem in problems:
        label_record = labels_by_name.get(problem["name"])
        if label_record is None:
            skipped["missing_labels"] += 1
            continue
        for edit_type in EDIT_TYPES:
            pick = label_record.get(edit_type)
            if not pick:
                skipped[f"null_{edit_type}"] += 1
                continue
            edited = record_fn(problem, pick, edit_type)
            if edited is None:
                skipped[f"skip_{edit_type}"] += 1
                continue
            outputs[edit_type].append(edited)

    print(json.dumps({
        "input_size": len(problems),
        "output_sizes": {k: len(v) for k, v in outputs.items()},
        "skipped": dict(skipped),
    }, indent=2))
    return {configs[et]: Dataset.from_list(rows) for et, rows in outputs.items() if rows}


# ═══════════════════════════════════════════════════════════════════════════
# NUMBER edits
# ═══════════════════════════════════════════════════════════════════════════

NUMBER_CONFIGS = {"statement_edit": "number_edit_statement", "proof_edit": "number_edit_proof"}

NUMBER_LABEL_PROMPT = """\
You are a math annotation assistant. Given a math problem's informal statement
and informal proof, list EVERY numeric literal that is SAFELY EDITABLE for
perturbation. For each, specify whether it appears in the STATEMENT or the PROOF.

DEFINITION OF "safely editable":
- A concrete numeric literal in digit form: integer, decimal, percent, or simple
  fraction like "1/3". Words like "nine" or "twice" are NOT editable.
- Changing this number should yield a plausibly wrong but well-formed problem.

DO NOT label:
- Structural numbers: subscripts (the 1 in x_1), superscripts (the 2 in x^2),
  indices, dimensions.
- Numbers that are part of a formula DEFINITION rather than a value (the 1 in
  "Let f(x) = x + 1" when 1 is just a definitional term, not the intended value
  to perturb).
- Numbers inside LaTeX commands like \\frac, \\sqrt when they're formatting
  arguments rather than actual parameter values.

OUTPUT: one JSON object with exactly this shape:
{
  "problem_name": "...",
  "numbers": [
    {
      "value":              "<exact substring as it appears in text>",
      "source":             "statement" or "proof",
      "char_offset_start":  <int offset in the respective source text>,
      "char_offset_end":    <int offset in the respective source text>,
      "context":            "<~20 words around the number, as it appears>"
    }
  ]
}

Rules:
- The SAME number value may appear multiple times — list each occurrence
  separately with its own offsets and source.
- "numbers" may be empty if nothing is safely editable.
- Be precise with char_offset_start / char_offset_end so that
  text[start:end] == value. Offsets are within the respective source text
  (NOT the combined text).
- Only output the JSON object — no prose, no code fences.

--- INFORMAL STATEMENT ---
%s

--- INFORMAL PROOF ---
%s
"""


# ── Numeric-string classifiers + deterministic perturbation ──────────────────

def is_int_string(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", text))


def is_decimal_string(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+\.\d+", text))


def is_fraction_string(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+/-?\d+", text))


def is_percent_string(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%", text))


def _stable_rng(value: str, context: str = "") -> random.Random:
    digest = hashlib.sha256(f"{context}:{value}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _sign_preserving_direction(rng: random.Random, original, delta) -> int:
    """Pick +1 / -1 so the perturbation never crosses zero."""
    if original == 0:
        return 1
    if original > 0:
        if delta >= original:
            return 1
        return rng.choice([1, -1])
    if delta >= -original:
        return -1
    return rng.choice([1, -1])


def perturb_numeric_string(
    value: str,
    problem_name: str = "",
    source: str = "",
    role: str = "",
) -> Optional[str]:
    """Return a small deterministic perturbation of a numeric string.

    sha256-seeded by problem_name + source + role + value; magnitude-proportional;
    sign-preserving; never equal to the original and never 0 for nonzero inputs.
    """
    value = value.strip()
    context = f"{problem_name}:{source}:{role}"
    rng = _stable_rng(value, context)

    if is_percent_string(value):
        base = value[:-1]
        edited = perturb_numeric_string(base, problem_name, source, role)
        return None if edited is None else f"{edited}%"

    if is_int_string(value):
        n = int(value)
        abs_n = abs(n)
        if abs_n <= 10:
            delta = rng.randint(1, 3)
        elif abs_n <= 100:
            delta = rng.randint(1, 5)
        elif abs_n <= 1000:
            delta = rng.randint(2, 15)
        else:
            delta = rng.randint(1, max(1, abs_n // 20))
        direction = _sign_preserving_direction(rng, n, delta)
        return str(n + direction * delta)

    if is_decimal_string(value):
        try:
            dec = Decimal(value)
        except InvalidOperation:
            return None
        decimals = len(value.split(".")[1])
        ulp = Decimal(1).scaleb(-decimals)
        abs_dec = abs(dec)
        if abs_dec <= Decimal("1"):
            delta_units = rng.randint(1, 3)
        elif abs_dec <= Decimal("10"):
            delta_units = rng.randint(1, 5)
        else:
            delta_units = rng.randint(1, max(1, int(abs_dec / Decimal("5"))))
        direction = _sign_preserving_direction(
            rng, int(dec / ulp) if dec != 0 else 0, delta_units,
        )
        new_dec = dec + direction * delta_units * ulp
        return format(new_dec, f".{decimals}f")

    if is_fraction_string(value):
        num, den = value.split("/")
        try:
            num_i = int(num)
            den_i = int(den)
        except ValueError:
            return None
        if den_i == 0:
            return None
        abs_num = abs(num_i)
        if abs_num <= 10:
            delta = rng.randint(1, 3)
        else:
            delta = rng.randint(1, max(1, abs_num // 5))
        direction = _sign_preserving_direction(rng, num_i, delta)
        return f"{num_i + direction * delta}/{den_i}"

    return None


def latex_fraction_patterns(value: str) -> List[str]:
    """Regex patterns for LaTeX variants of a fraction like '1/3'."""
    if not is_fraction_string(value):
        return []
    parts = value.split("/")
    if len(parts) != 2:
        return []
    num, den = parts[0].strip(), parts[1].strip()
    return [
        rf"\\frac\{{{re.escape(num)}\}}\{{{re.escape(den)}\}}",
        rf"\\frac{re.escape(num)}{re.escape(den)}",
        rf"\\dfrac\{{{re.escape(num)}\}}\{{{re.escape(den)}\}}",
        rf"\\tfrac\{{{re.escape(num)}\}}\{{{re.escape(den)}\}}",
    ]


def replace_span(
    text: str, start: int, end: int, old_value: str, new_value: str,
) -> Optional[str]:
    """Replace a span; fall back to nearest exact / LaTeX-fraction match."""
    if 0 <= start < end <= len(text) and text[start:end] == old_value:
        return text[:start] + new_value + text[end:]

    matches = list(re.finditer(re.escape(old_value), text))
    if matches:
        best = min(matches, key=lambda m: abs(m.start() - start))
        return text[: best.start()] + new_value + text[best.end():]

    if is_fraction_string(new_value):
        new_num, new_den = new_value.split("/")
        for pattern in latex_fraction_patterns(old_value):
            m = re.search(pattern, text)
            if m:
                matched_text = m.group()
                if "\\dfrac" in matched_text:
                    replacement = f"\\dfrac{{{new_num}}}{{{new_den}}}"
                elif "\\tfrac" in matched_text:
                    replacement = f"\\tfrac{{{new_num}}}{{{new_den}}}"
                elif "{" in matched_text:
                    replacement = f"\\frac{{{new_num}}}{{{new_den}}}"
                else:
                    replacement = f"\\frac{new_num}{new_den}"
                return text[: m.start()] + replacement + text[m.end():]

    return None


# ── Number select filter + pick ──────────────────────────────────────────────

def _is_numeric_literal(value: str) -> bool:
    v = value.strip()
    return bool(
        re.fullmatch(r"-?\d+", v)
        or re.fullmatch(r"-?\d+\.\d+", v)
        or re.fullmatch(r"-?\d+\s*/\s*-?\d+", v)
        or re.fullmatch(r"-?\d+(?:\.\d+)?%", v)
    )


def _is_isolated_number(text: str, start: int, end: int) -> bool:
    """Reject candidates whose span is a fragment of a larger numeric token."""
    if start < 0 or end > len(text) or start >= end:
        return True
    prev_ch = text[start - 1] if start > 0 else ""
    next_ch = text[end] if end < len(text) else ""
    if prev_ch.isdigit() or prev_ch == ".":
        return False
    if next_ch.isdigit() or next_ch == ".":
        return False
    return True


def _number_filter(cands, problem_row):
    cands = [n for n in cands if _is_numeric_literal(str(n.get("value", "")))]
    if problem_row is None:
        return cands
    stmt_text = str(problem_row.get("informal_statement", "") or "")
    proof_text = str(problem_row.get("informal_proof", "") or "")
    kept = []
    for n in cands:
        text = stmt_text if n.get("source") == "statement" else proof_text
        s = int(n.get("char_offset_start", -1))
        e = int(n.get("char_offset_end", -1))
        if _is_isolated_number(text, s, e):
            kept.append(n)
    return kept


def _number_pick(cands: list, problem_name: str, source: str) -> Optional[dict]:
    if not cands:
        return None
    chosen = _seeded_rng(problem_name, source).choice(cands)
    return {
        "old_value": str(chosen.get("value", "")).strip(),
        "char_offset_start": int(chosen.get("char_offset_start", -1)),
        "char_offset_end": int(chosen.get("char_offset_end", -1)),
        "context": chosen.get("context", ""),
    }


def _number_record(problem: dict, pick: dict, edit_type: str) -> Optional[dict]:
    source = SOURCE_OF[edit_type]
    old_value = str(pick.get("old_value", "")).strip()
    start = int(pick.get("char_offset_start", -1))
    end = int(pick.get("char_offset_end", -1))
    if not old_value:
        return None

    new_value = perturb_numeric_string(old_value, problem_name=problem.get("name", ""), source=source)
    if new_value is None or new_value == old_value:
        return None

    if source == "statement":
        text = str(problem.get("informal_statement", ""))
        edited = replace_span(text, start, end, old_value, new_value)
        if edited is None:
            return None
        edited_statement, edited_proof = edited, str(problem.get("informal_proof", ""))
    else:
        text = str(problem.get("informal_proof", ""))
        edited = replace_span(text, start, end, old_value, new_value)
        if edited is None:
            return None
        edited_statement, edited_proof = str(problem.get("informal_statement", "")), edited

    record = dict(problem)
    record["original_informal_statement"] = str(problem.get("informal_statement", ""))
    record["original_informal_proof"] = str(problem.get("informal_proof", ""))
    record["informal_statement"] = edited_statement
    record["informal_proof"] = edited_proof
    record["number_edit_type"] = edit_type
    record["number_edit_source"] = source
    record["number_edit_old_value"] = old_value
    record["number_edit_new_value"] = new_value
    record["number_edit_context"] = pick.get("context", "")
    record["number_edit_char_offset_start"] = start
    record["number_edit_char_offset_end"] = end
    return record


def run_number(dataset, data_dir, model="gemini-2.5-flash", limit=0):
    data_dir = str(ensure_dir(data_dir))
    labeled = f"{data_dir}/labeled_numbers.jsonl"
    fixed = f"{data_dir}/labeled_numbers_fixed.jsonl"
    selected = f"{data_dir}/selected_numbers.jsonl"

    _label(dataset, labeled, NUMBER_LABEL_PROMPT, "numbers", model, limit)
    _fix_offsets(dataset, labeled, fixed, "numbers", "value")
    _select(fixed, selected, dataset, "numbers", _number_filter, _number_pick)
    return _build(dataset, selected, limit, NUMBER_CONFIGS, _number_record)


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL edits
# ═══════════════════════════════════════════════════════════════════════════

SYMBOL_CONFIGS = {"statement_edit": "symbol_edit_statement", "proof_edit": "symbol_edit_proof"}

# All target symbols and their unsound swaps (no = sign).
SYMBOL_SWAP: Dict[str, str] = {
    ">": "<", "<": ">",
    "\\geq": "\\leq", "\\ge": "\\le", "≥": "≤",
    "\\leq": "\\geq", "\\le": "\\ge", "≤": "≥",
    "\\gt": "\\lt", "\\lt": "\\gt",
    "+": "-", "-": "+",
    "\\times": "\\div", "×": "÷",
    "\\cdot": "\\div", "·": "÷",
    "\\div": "\\times", "÷": "×",
}

EXCLUDED = {"=", "≠", "\\neq"}
RELATIONS = {"<", ">", "\\le", "\\ge", "\\leq", "\\geq", "<=", ">=", "≤", "≥"}
_REL_PAT = re.compile(r"(\\leq|\\geq|\\le|\\ge|<=|>=|≤|≥|<|>)")

SYMBOL_LABEL_PROMPT = """\
You are a math annotation assistant. Given a math problem's informal statement
and informal proof, list EVERY occurrence of the following target symbols that
is SAFELY EDITABLE for perturbation. For each, specify whether it appears in
the STATEMENT or the PROOF.

TARGET SYMBOLS (LaTeX command forms are the same symbol written differently):
  relation: >  <  >=  <=    (or \\gt \\lt \\geq \\leq \\ge \\le)
  operator: +  -  ×  ·  ÷   (or \\times \\cdot \\div)

EXCLUDE: = ≠ \\neq, and any symbol not in the list above.

DEFINITION OF "safely editable":
- Changing the symbol should yield a plausibly wrong but well-formed claim or
  computation.

DO NOT label:
- Definitional: the + in "Let f(x) = x + 1" when 1 is just a definitional term.
- Structural: + in subscripts like x_{n+1}, - in superscripts like x^{2-k}.
- Inside \\frac, \\sqrt, or other LaTeX command arguments.

OUTPUT: one JSON object with exactly this shape:
{
  "problem_name": "...",
  "symbols": [
    {
      "symbol":             "<exact substring as it appears, e.g. \\\\leq or ≤>",
      "family":             "relation" or "operator",
      "source":             "statement" or "proof",
      "char_offset_start":  <int offset in the respective source text>,
      "char_offset_end":    <int offset in the respective source text>,
      "context":            "<~20 words around the symbol, as it appears>"
    }
  ]
}

Rules:
- List each occurrence SEPARATELY with its own offsets.
- "symbols" may be empty if nothing is safely editable.
- Be precise with char_offset_start / char_offset_end so that
  text[start:end] == symbol. For multi-char symbols like \\leq (4 chars), span
  is start to start+4. Offsets are within the respective source text.
- Only output the JSON object — no prose, no code fences.

--- INFORMAL STATEMENT ---
%s

--- INFORMAL PROOF ---
%s
"""


# ── Perturbation + replacement ───────────────────────────────────────────────

def perturb_symbol(symbol: str) -> Optional[str]:
    """Return the unsound replacement for a symbol. Deterministic, no randomness."""
    s = symbol.strip()
    new = SYMBOL_SWAP.get(s)
    if new is None:
        return None
    if new.startswith("\\") and not s.startswith("\\"):
        new = new + " "
    return new


def replace_symbol(
    text: str, offset_start: int, offset_end: int,
    old_symbol: str, new_symbol: str, context: str = "",
) -> Optional[str]:
    """Replace a symbol. Exact offset first, then context-guided fallback."""
    if 0 <= offset_start < len(text) and offset_end <= len(text):
        if text[offset_start:offset_end] == old_symbol:
            return text[:offset_start] + new_symbol + text[offset_end:]

    matches = list(re.finditer(re.escape(old_symbol), text))
    if not matches:
        return None

    def is_in_comment_marker(m):
        pos = m.start()
        if old_symbol == "-":
            if pos > 0 and text[pos - 1] == "/":
                return True
            if pos > 0 and text[pos - 1] == "-" and pos > 1 and text[pos - 2] == "/":
                return True
            if pos + 1 < len(text) and text[pos + 1] == "/":
                return True
            if pos + 1 < len(text) and text[pos + 1] == "-":
                return True
        return False

    matches = [m for m in matches if not is_in_comment_marker(m)]
    if not matches:
        return None

    if len(matches) == 1:
        m = matches[0]
        return text[:m.start()] + new_symbol + text[m.end():]

    if context:
        ctx_clean = context[:40].strip()
        for m in matches:
            window = text[max(0, m.start() - 30):min(len(text), m.end() + 30)]
            ctx_words = [w for w in ctx_clean.split() if len(w) > 3]
            if sum(1 for w in ctx_words if w in window) >= 2:
                return text[:m.start()] + new_symbol + text[m.end():]

    if offset_start >= 0:
        best = min(matches, key=lambda m: abs(m.start() - offset_start))
        return text[:best.start()] + new_symbol + text[best.end():]

    return None


# ── Symbol select filter + pick ──────────────────────────────────────────────

def _is_valid_target(symbol: str) -> bool:
    s = symbol.strip()
    return s not in EXCLUDED and s in SYMBOL_SWAP


def _is_in_relation_chain(text: str, start: int, end: int) -> bool:
    """True iff the candidate is part of a chain of 2+ relation operators."""
    if start < 0 or end <= start:
        return False
    L = start
    while L > 0 and text[L - 1] not in ".\n;" and start - L < 80:
        L -= 1
    R = end
    while R < len(text) and text[R] not in ".\n;" and R - end < 80:
        R += 1
    return len(_REL_PAT.findall(text[L:R])) >= 2


def _is_inside_script_group(text: str, start: int) -> bool:
    """True if position ``start`` is inside a ``_{...}`` or ``^{...}`` group."""
    if start <= 0:
        return False
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i] in "_^" and text[i + 1] == "{":
            depth = 1
            j = i + 2
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if i + 2 <= start < j:
                return True
            i = j + 1
        else:
            i += 1
    return False


def _symbol_filter(cands, problem_row):
    cands = [s for s in cands if _is_valid_target(str(s.get("symbol", "")))]
    if problem_row is None:
        return cands
    stmt_text = str(problem_row.get("informal_statement", "") or "")
    proof_text = str(problem_row.get("informal_proof", "") or "")
    kept = []
    for s in cands:
        text = stmt_text if s.get("source") == "statement" else proof_text
        sym = str(s.get("symbol", "")).strip()
        pos_start = int(s.get("char_offset_start", -1))
        pos_end = int(s.get("char_offset_end", -1))
        if _is_inside_script_group(text, pos_start):
            continue
        if sym in RELATIONS and _is_in_relation_chain(text, pos_start, pos_end):
            continue
        kept.append(s)
    return kept


def _symbol_pick(cands: list, problem_name: str, source: str) -> Optional[dict]:
    """Pick one candidate; prefer relation-family targets to avoid operator bias."""
    if not cands:
        return None
    rng = _seeded_rng(problem_name, source)
    rel_cands = [c for c in cands if str(c.get("symbol", "")).strip() in RELATIONS]
    pool = rel_cands if rel_cands else cands
    chosen = rng.choice(pool)
    return {
        "symbol": str(chosen.get("symbol", "")).strip(),
        "family": chosen.get("family", ""),
        "char_offset_start": int(chosen.get("char_offset_start", -1)),
        "char_offset_end": int(chosen.get("char_offset_end", -1)),
        "context": chosen.get("context", ""),
    }


def _symbol_record(problem: dict, pick: dict, edit_type: str) -> Optional[dict]:
    source = SOURCE_OF[edit_type]
    old_symbol = str(pick.get("symbol", "")).strip()
    start = int(pick.get("char_offset_start", -1))
    end = int(pick.get("char_offset_end", -1))
    context = pick.get("context", "")
    if not old_symbol:
        return None

    new_symbol = perturb_symbol(old_symbol)
    if new_symbol is None or new_symbol == old_symbol:
        return None

    orig_stmt = str(problem.get("informal_statement", ""))
    orig_proof = str(problem.get("informal_proof", ""))
    if source == "statement":
        edited = replace_symbol(orig_stmt, start, end, old_symbol, new_symbol, context)
        if edited is None:
            return None
        edited_statement, edited_proof = edited, orig_proof
    else:
        edited = replace_symbol(orig_proof, start, end, old_symbol, new_symbol, context)
        if edited is None:
            return None
        edited_statement, edited_proof = orig_stmt, edited

    if edited_statement == orig_stmt and edited_proof == orig_proof:
        return None

    record = dict(problem)
    record["original_informal_statement"] = orig_stmt
    record["original_informal_proof"] = orig_proof
    record["informal_statement"] = edited_statement
    record["informal_proof"] = edited_proof
    record["symbol_edit_type"] = edit_type
    record["symbol_edit_source"] = source
    record["symbol_edit_family"] = pick.get("family")
    record["symbol_edit_old_symbol"] = old_symbol
    record["symbol_edit_new_symbol"] = new_symbol
    record["symbol_edit_context"] = context
    record["symbol_edit_char_offset_start"] = start
    record["symbol_edit_char_offset_end"] = end
    return record


def run_symbol(dataset, data_dir, model="gemini-2.5-flash", limit=0):
    data_dir = str(ensure_dir(data_dir))
    labeled = f"{data_dir}/labeled_symbols.jsonl"
    fixed = f"{data_dir}/labeled_symbols_fixed.jsonl"
    selected = f"{data_dir}/selected_symbols.jsonl"

    _label(dataset, labeled, SYMBOL_LABEL_PROMPT, "symbols", model, limit)
    _fix_offsets(dataset, labeled, fixed, "symbols", "symbol")
    _select(fixed, selected, dataset, "symbols", _symbol_filter, _symbol_pick)
    return _build(dataset, selected, limit, SYMBOL_CONFIGS, _symbol_record)
