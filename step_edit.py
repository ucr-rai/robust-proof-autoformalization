"""Step-delete dataset construction (all stages in one module).

Stage 1 label  — Gemini segments the proof into deletable reasoning/outcome steps
Stage 2 select — tiered quality filter + sha256-seeded pick of one step per proof
Stage 3 build  — delete the step's reasoning, keep its outcome -> rows in memory

Entry point: ``run(dataset, data_dir, model, limit)`` returns {"step_delete": Dataset}
for build.py to push to HuggingFace (no local copy of the final data).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from datasets import Dataset

from utils import ensure_dir, load_jsonl, make_gemini_model, robust_json_loads


# ── Stage 1: label ───────────────────────────────────────────────────────────

LABEL_PROMPT = """\
You are annotating a natural-language math proof for a step-deletion benchmark.

Your task: split the proof into ordered steps. For each step, identify reasoning
(the justification) and outcome (the conclusion).

==========================================================================
RULES — read carefully before annotating
==========================================================================

R1 (HARD). REASONING must be SUBSTANTIVE.
    - Must be >= 30 characters.
    - Must contain at least one: math expression ($...$), a named theorem or
      technique (e.g. "by Cauchy-Schwarz", "by induction"), or a multi-step
      computation.
    - REJECT (set deletable=false) if reasoning is ONLY a transition word or
      phrase such as: "Thus", "So", "Hence", "Therefore", "Consequently",
      "We have that", "We see that", "We get", "Notice that", "Note that",
      "It follows", "As a result", "In the diagram", "WLOG", "Clearly",
      "Obviously", "We obtain", "This gives us", "That means", "Simplifying",
      "As before", "From this".

R2 (SOFT). OUTCOME should ideally be a readable standalone claim.
    - Preferred: starts with uppercase, ends with period or $, has English context.
    - Also acceptable: formula-only like "$r = 3.$" IF reasoning is substantive.
    - Set deletable=false only if outcome is empty, or < 10 characters, or is a
      single symbol/digit with no equation structure (e.g. just "5" or "$x$").

R3 (HARD). OUTCOME must NOT be an incomplete clause that requires its antecedent.
    - REJECT if outcome starts with a relative/subordinating fragment:
      "which ...", "thereby ...", "whereby ...", "whereas ..."
    - REJECT if outcome starts with an anaphoric pronoun whose referent is
      exclusively in reasoning: "this ...", "that ...", "these ...", "those ..."
      (as sentence subject).
    - "So $x=5$.", "Hence $r=3$.", "Therefore the answer is 42." are OK — these
      are standalone transitions, not incomplete clauses.

R4 (HARD). reasoning_text + outcome_text must reconstruct full_text.
    - full_text = reasoning_text + (whitespace/punctuation) + outcome_text.
    - If you cannot cleanly split, set deletable=false.

R6 (SOFT). Prefer proofs with >= 3 steps.
    - If the proof has only 1 step, set all deletable=false.
    - If the proof has 2 steps, you may still mark one deletable if R1-R4 pass.

Be CONSERVATIVE: if unsure whether R1-R4 are satisfied, set deletable=false.
We prefer fewer high-quality edits over many noisy ones.

==========================================================================
EXAMPLES
==========================================================================

GOOD (deletable=true):
  full_text: "By the AM-GM inequality, $\\frac{{a+b}}{{2}} \\geq \\sqrt{{ab}}$. Therefore $a + b \\geq 2\\sqrt{{ab}}$."
  reasoning_text: "By the AM-GM inequality, $\\frac{{a+b}}{{2}} \\geq \\sqrt{{ab}}$."
  outcome_text: "Therefore $a + b \\geq 2\\sqrt{{ab}}$."

GOOD (formula-only outcome, reasoning is substantive):
  full_text: "Substituting $x = 0$ and $y = 3$ into $r = \\sqrt{{x^2 + y^2}}$, we get $r = 3.$"
  reasoning_text: "Substituting $x = 0$ and $y = 3$ into $r = \\sqrt{{x^2 + y^2}}$,"
  outcome_text: "$r = 3.$"

BAD (vacuous reasoning — deletable=false):
  full_text: "Consequently, $S = -14$ and $P = -38$."
  reasoning_text: "Consequently,"
  outcome_text: "$S = -14$ and $P = -38$."

BAD (degenerate outcome — deletable=false):
  full_text: "Simplifying, $013$"
  reasoning_text: "Simplifying,"
  outcome_text: "$013$"

==========================================================================
OUTPUT FORMAT
==========================================================================

Return ONLY valid JSON, no prose, no code fences:

{{
  "problem_name": "...",
  "steps": [
    {{
      "step_idx": 0,
      "full_text": "exact verbatim text from the proof",
      "reasoning_text": "the justification / computation / derivation",
      "outcome_text": "the conclusion",
      "deletable": true,
      "is_last": false
    }},
    ...
  ]
}}

List ALL steps, not just deletable ones.
Do NOT add or remove escape characters — copy text EXACTLY from the input.

--- INFORMAL PROOF ---
{proof}
"""


def _label_one(model, problem: dict) -> dict:
    name = problem.get("name", "unknown")
    proof = str(problem.get("informal_proof", "") or "")
    prompt = LABEL_PROMPT.format(proof=proof)

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            result = robust_json_loads(response.text.strip())
            result["problem_name"] = name
            return result
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {name}: {e}")
            time.sleep(2)

    return {"problem_name": name, "steps": [], "error": "Failed after 3 attempts"}


def label(dataset, output, model="gemini-2.5-flash", limit=0):
    model = make_gemini_model(model)

    problems = load_jsonl(dataset)
    if limit > 0:
        problems = problems[:limit]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(problems):
            name = prob.get("name", f"row_{i}")
            print(f"  [{i+1}/{len(problems)}] {name}...", end=" ", flush=True)
            result = _label_one(model, prob)
            n_steps = len(result.get("steps", []))
            n_del = sum(1 for s in result.get("steps", []) if s.get("deletable"))
            print(f"{n_steps} steps, {n_del} deletable")
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            time.sleep(0.5)


# ── Stage 2: clean + select ──────────────────────────────────────────────────

DEFAULT_MIN_REASONING_CHARS = 30
DEFAULT_MIN_OUTCOME_CHARS = 10

VACUOUS_PHRASES = {
    "so", "thus", "hence", "therefore", "clearly", "obviously",
    "then", "next", "first", "second", "finally", "now", "here",
    "note", "notice", "observe", "also", "further", "moreover",
    "indeed", "similarly", "and", "but", "however", "wlog",
    "consequently", "trivially", "simplifying",
    "we have", "we have that", "we get", "we obtain", "we see",
    "we see that", "we note", "we note that", "we know",
    "we know that", "we conclude",
    "in the diagram", "by this", "by that", "this gives us",
    "that means", "as a result", "it follows", "it follows that",
    "so that", "such that", "as before", "as above", "from this",
    "from the above", "by the above",
}

SUBSTANTIVE_VERB_RE = re.compile(
    r"\b(substitut|expand|factor|comput|derive|appl|rewrite|square|multipl|"
    r"divid|integrat|differentiat|sum|simplif|cancel|invert|transpos|"
    r"normaliz|equate|solve|rearrang|combine|isolate|reduce|evaluat|"
    r"subtract|add|multiply|square root|take the|let|set|define|"
    r"plug|insert|convert|transform|express|manipulat|"
    r"by.*theorem|by.*inequality|by.*lemma|by.*definition|"
    r"using|since|because|as)\w*\b",
    re.I,
)

DANGLING_FRAGMENT_RE = re.compile(
    r"^\s*(which\b|that\s+is\b|thereby\b|whereas\b|whereby\b|"
    r"so\s+that\b|for\s+a\s+total\b|meaning\s+that\b|implying\s+that\b)",
    re.I,
)
ANAPHORIC_START_RE = re.compile(
    r"^\s*(this|that|these|those)\s+(is|are|was|were|gives|means|implies|shows|proves|yields)\b",
    re.I,
)

TIER_RANK = {"gold": 2, "silver": 1, "bronze": 0}


def classify_step(step: dict, min_reasoning: int, min_outcome: int, total_steps: int
                  ) -> tuple[str, str]:
    """Return (tier, reason). tier ∈ {gold, silver, bronze, reject}."""
    reasoning = (step.get("reasoning_text") or "").strip()
    outcome = (step.get("outcome_text") or "").strip()
    full_text = (step.get("full_text") or "").strip()

    if not reasoning or not outcome or not full_text:
        return "reject", "empty_field"

    norm_reasoning = re.sub(r"[\s.,;:!?]+", " ", reasoning.lower()).strip()
    if norm_reasoning in VACUOUS_PHRASES:
        return "reject", f"vacuous_phrase:{norm_reasoning[:30]}"
    if len(reasoning) < min_reasoning:
        has_math = "$" in reasoning or "\\(" in reasoning
        has_action = bool(SUBSTANTIVE_VERB_RE.search(reasoning))
        if not (has_math or has_action):
            return "reject", f"short_no_math_no_action:{len(reasoning)}"

    if DANGLING_FRAGMENT_RE.match(outcome):
        return "reject", "dangling_fragment"
    if ANAPHORIC_START_RE.match(outcome):
        return "reject", "anaphoric_start"

    full_ue = full_text.replace("\\\\", "\\")
    outcome_ue = outcome.replace("\\\\", "\\")
    outcome_in_full = (
        outcome in full_text
        or outcome_ue in full_ue
        or re.sub(r"\s+", " ", outcome_ue).strip() in re.sub(r"\s+", " ", full_ue).strip()
    )
    if not outcome_in_full:
        return "reject", "outcome_not_substring_of_full"

    norm_full = re.sub(r"\s+", " ", full_text.lower()).strip()
    norm_join = re.sub(r"\s+", " ", (reasoning + " " + outcome).lower()).strip()
    if norm_join not in norm_full and norm_full not in norm_join:
        if len(norm_join) > 10 and len(norm_full) > 10:
            ratio = SequenceMatcher(None, norm_join, norm_full).ratio()
            if ratio < 0.85:
                return "reject", f"split_inconsistent:ratio={ratio:.2f}"

    if len(outcome) < min_outcome:
        return "reject", f"outcome_too_short:{len(outcome)}"

    stripped = re.sub(r"\$[^$]*\$", "", outcome).strip()
    stripped = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", "", stripped).strip()
    stripped = re.sub(r"[\\\{\}_\^.,;:!?\s]", "", stripped)
    if not stripped and len(outcome) < 15 and "=" not in outcome and "<" not in outcome and ">" not in outcome:
        return "reject", "degenerate_outcome"

    starts_upper = outcome[0].isupper() or outcome[0] == "\\"
    has_english = len([w for w in re.sub(r"\$[^$]*\$", "", outcome).split()
                       if w.isalpha() and len(w) >= 2]) >= 2
    ends_properly = outcome.rstrip().endswith((".", "$", ")"))
    has_equation = any(c in outcome for c in "=<>≤≥≠∈∀∃")

    if starts_upper and has_english and ends_properly:
        tier = "gold"
    elif has_equation or (starts_upper and ends_properly):
        tier = "silver"
    else:
        tier = "bronze"

    if total_steps <= 2:
        tier = "bronze"

    return tier, ""


def _sort_key(step: dict, problem_name: str, tier_rank: int) -> tuple:
    """Quality-ranked sort key (descending); sha256 tiebreak last."""
    reasoning = (step.get("reasoning_text") or "").strip()
    outcome = (step.get("outcome_text") or "").strip()

    has_math = "$" in reasoning or "\\(" in reasoning
    has_action = bool(SUBSTANTIVE_VERB_RE.search(reasoning))
    reasoning_substance = 2 * int(has_math) + 2 * int(has_action)

    starts_upper = bool(outcome) and (outcome[0].isupper() or outcome[0] == "\\")
    has_english = len([w for w in re.sub(r"\$[^$]*\$", "", outcome).split()
                       if w.isalpha() and len(w) >= 2]) >= 2
    ends_properly = outcome.rstrip().endswith((".", "$", ")"))
    outcome_readability = int(starts_upper) + int(has_english) + int(ends_properly)

    digest = hashlib.sha256(
        f"{problem_name}:step_v2:{step.get('step_idx', -1)}".encode("utf-8")
    ).hexdigest()
    tiebreak = int(digest[:16], 16)

    return (tier_rank, reasoning_substance, outcome_readability, len(reasoning), tiebreak)


def select(labels, output, min_reasoning=DEFAULT_MIN_REASONING_CHARS,
           min_outcome=DEFAULT_MIN_OUTCOME_CHARS):
    rows = load_jsonl(labels)
    reject_reasons: Counter = Counter()
    tier_counts: Counter = Counter()
    n_selected = n_no_candidate = 0

    out_rows = []
    for row in rows:
        name = row.get("problem_name", "?")
        steps = row.get("steps", []) or []
        total_steps = len(steps)

        classified = []
        for step in steps:
            if not step.get("deletable"):
                continue
            tier, reason = classify_step(step, min_reasoning, min_outcome, total_steps)
            if tier == "reject":
                reject_reasons[reason] += 1
            else:
                tier_counts[tier] += 1
                classified.append((step, tier))

        if not classified:
            n_no_candidate += 1
            out_rows.append({"problem_name": name, "selected_step": None})
            continue

        classified.sort(key=lambda sc: _sort_key(sc[0], name, TIER_RANK[sc[1]]), reverse=True)
        picked, picked_tier = classified[0]
        n_selected += 1
        out_rows.append({"problem_name": name, "selected_step": picked, "selected_tier": picked_tier})

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== Select ===  problems:{len(rows)} selected:{n_selected} "
          f"no_candidate:{n_no_candidate} tiers:{dict(tier_counts)} -> {output}")


# ── Stage 3: build step-delete ───────────────────────────────────────────────

def _unescape_only(text: str) -> str:
    return text.replace("\\\\", "\\")


def _normalize_both_sides(text: str) -> str:
    return re.sub(r"\s+", " ", _unescape_only(text)).strip()


def _find_span(proof: str, full_text: str) -> tuple[int, int, str, str]:
    """Find full_text in proof. Returns (start, end, method, outcome_basis)."""
    idx = proof.find(full_text)
    if idx >= 0:
        return idx, idx + len(full_text), "exact", "raw"

    ft_ue = _unescape_only(full_text)
    idx = proof.find(ft_ue)
    if idx >= 0:
        return idx, idx + len(ft_ue), "unescape", "unescaped"

    proof_norm = _normalize_both_sides(proof)
    ft_norm = _normalize_both_sides(full_text)
    idx = proof_norm.find(ft_norm)
    if idx >= 0:
        return idx, idx + len(ft_norm), "normalized", "normalized"

    if len(full_text) >= 60:
        head, tail = full_text[:30], full_text[-30:]
        hi, ti = proof.find(head), proof.rfind(tail)
        if hi >= 0 and ti >= 0 and ti > hi:
            return hi, ti + len(tail), "fuzzy", "raw"
        head_ue, tail_ue = _unescape_only(head), _unescape_only(tail)
        hi, ti = proof.find(head_ue), proof.rfind(tail_ue)
        if hi >= 0 and ti >= 0 and ti > hi:
            return hi, ti + len(tail_ue), "fuzzy", "unescaped"

    return -1, -1, "not_found", ""


def _build_one(problem: dict, step: dict, tier: str) -> Optional[dict]:
    proof = problem.get("informal_proof", "") or ""
    full_text = (step.get("full_text") or "").strip()
    raw_outcome = (step.get("outcome_text") or "").strip()
    reasoning = (step.get("reasoning_text") or "").strip()

    if not full_text or not raw_outcome:
        return None

    full_ue = full_text.replace("\\\\", "\\")
    outcome_ue = raw_outcome.replace("\\\\", "\\")
    outcome_in_full = (
        raw_outcome in full_text
        or outcome_ue in full_ue
        or re.sub(r"\s+", " ", outcome_ue).strip() in re.sub(r"\s+", " ", full_ue).strip()
    )
    if not outcome_in_full:
        return None

    start, end, method, outcome_basis = _find_span(proof, full_text)
    if method == "not_found":
        return None

    if outcome_basis == "raw":
        inserted = raw_outcome
    elif outcome_basis == "unescaped":
        inserted = _unescape_only(raw_outcome)
    else:
        inserted = _normalize_both_sides(raw_outcome)

    edited_proof = proof[:start] + inserted + proof[end:]

    if edited_proof == proof:
        return None
    if edited_proof[start:start + len(inserted)] != inserted:
        return None
    if edited_proof[:start] != proof[:start]:
        return None
    if edited_proof[start + len(inserted):] != proof[end:]:
        return None
    if len(proof) == 0:
        return None
    shrinkage = 1.0 - len(edited_proof) / len(proof)
    if not (0.005 <= shrinkage <= 0.80):
        return None

    out = dict(problem)
    out["original_informal_proof"] = proof
    out["informal_proof"] = edited_proof
    out["step_edit_type"] = "step_delete"
    out["step_edit_target_step_idx"] = step.get("step_idx", -1)
    out["step_edit_target_full_text"] = full_text
    out["step_edit_target_outcome_text"] = raw_outcome
    out["step_edit_target_reasoning_text"] = reasoning
    out["step_edit_matched_by"] = method
    out["step_edit_match_confidence"] = {"exact": 1.0, "unescape": 0.9,
                                         "normalized": 0.7, "fuzzy": 0.5}[method]
    out["step_edit_is_last"] = bool(step.get("is_last", False))
    out["step_edit_tier"] = tier
    out["step_edit_span_start"] = start
    out["step_edit_span_end"] = end

    remaining = proof[:start] + proof[end:]
    reasoning_ue = reasoning.replace("\\\\", "\\")
    out["step_edit_reasoning_reappears_elsewhere"] = bool(
        reasoning and (reasoning in remaining or reasoning_ue in remaining)
    )
    return out


def build(dataset, labels):
    """Delete steps; return {"step_delete": HF Dataset} of the gold set (no files).

    Only the gold set (exact/unescape span match) is returned as benchmark data;
    the normalized/fuzzy "review" pool is a QA artifact and is not emitted.
    """
    problems = {r["name"]: r for r in load_jsonl(dataset)}
    label_rows = load_jsonl(labels)

    gold_rows, review_rows = [], []
    n_failed = 0

    for label_row in label_rows:
        name = label_row.get("problem_name")
        step = label_row.get("selected_step")
        tier = label_row.get("selected_tier", "bronze")
        if step is None or name not in problems:
            continue
        result = _build_one(problems[name], step, tier)
        if result is None:
            n_failed += 1
            continue
        if result["step_edit_matched_by"] in ("exact", "unescape"):
            gold_rows.append(result)
        else:
            review_rows.append(result)

    print(json.dumps({
        "input_problems": len(problems),
        "labeled_with_selection": sum(1 for r in label_rows if r.get("selected_step")),
        "gold_set": len(gold_rows),
        "review_pool_dropped": len(review_rows),
        "build_failed": n_failed,
        "gold_tiers": dict(Counter(r["step_edit_tier"] for r in gold_rows)),
        "gold_methods": dict(Counter(r["step_edit_matched_by"] for r in gold_rows)),
    }, indent=2))
    return {"step_delete": Dataset.from_list(gold_rows)} if gold_rows else {}


# ── Entry point ──────────────────────────────────────────────────────────────

def run(dataset, data_dir, model="gemini-2.5-flash", limit=0):
    data_dir = str(ensure_dir(data_dir))
    labeled = f"{data_dir}/labeled_steps.jsonl"
    selected = f"{data_dir}/selected_steps.jsonl"

    label(dataset, labeled, model=model, limit=limit)
    select(labeled, selected)
    return build(dataset, selected)
