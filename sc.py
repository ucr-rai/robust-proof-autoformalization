"""SC v3 — Decoupled SC evaluator.

Core principle: Statement equivalence and Proof equivalence are orthogonal.
A reliable SC evaluator must judge them INDEPENDENTLY, without using one
as evidence for the other.

Two LLM calls per sample:

  Call 1 (StmtSC):  NL stmt + Lean theorem signature only.
                    Proof body is HIDDEN. Judge propositional equivalence.

  Call 2 (ProofSC): NL proof + Lean proof body (tactics) only.
                    Theorem signature is HIDDEN. Judge reasoning-sequence
                    equivalence. Prompt explicitly tells the LLM to ignore
                    whether the tactics actually prove the theorem (that is
                    the type checker's job).

Cost: 2 Gemini calls per sample.
"""
from __future__ import annotations

import re
import time


# ─── Lean parsing helpers ──────────────────────────────────────────────────

_BY_MARKER_RE = re.compile(r":=\s*by\b")
_BLOCK_COMMENT_RE = re.compile(r"/-(?:-)?[\s\S]*?-/")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

_STMT_SCORE_RE = re.compile(r"<stmt_score>\s*(-?1)\s*</stmt_score>", re.IGNORECASE)
_PROOF_SCORE_RE = re.compile(r"<proof_score>\s*(-?1)\s*</proof_score>", re.IGNORECASE)
_STMT_JUST_RE = re.compile(r"<stmt_justification>([\s\S]*?)</stmt_justification>", re.IGNORECASE)
_PROOF_JUST_RE = re.compile(r"<proof_justification>([\s\S]*?)</proof_justification>", re.IGNORECASE)


def extract_theorem_parts(lean_code: str) -> tuple[str | None, str | None]:
    """Return (theorem_statement, proof_body), or (None, None) if extraction fails.

    ProofFlow-style outputs may contain helper lemmas before the final theorem.
    In that case, judge the last theorem/lemma/example declaration rather than
    the first helper lemma. Single-declaration model outputs are unchanged.
    """
    if not lean_code:
        return None, None

    decl_re = re.compile(r"(?m)^\s*(?:theorem|lemma|example)\s+\w*")
    decls = list(decl_re.finditer(lean_code))
    for i in range(len(decls) - 1, -1, -1):
        start = decls[i].start()
        end = decls[i + 1].start() if i + 1 < len(decls) else len(lean_code)
        block = lean_code[start:end].strip()
        m = _BY_MARKER_RE.search(block)
        if m:
            return block[:m.start()].rstrip(), block[m.end():].strip()

    thm_m = re.search(r"\b(theorem|lemma|example)\s+\w*[\s\S]*", lean_code)
    if thm_m:
        return lean_code.strip(), None
    return None, None


def detect_degenerate(proof_body: str) -> tuple[bool, str]:
    """Identify proofs that are placeholders, not real reasoning."""
    if not proof_body or not proof_body.strip():
        return True, "empty"
    body = proof_body.strip()
    body_no_comments = _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", body)).strip()
    if not body_no_comments:
        return True, "comments_only"
    if re.fullmatch(r"(sorry|admit)\s*", body_no_comments):
        return True, "single_sorry_or_admit"
    tactic_lines = [ln.strip() for ln in body_no_comments.split("\n") if ln.strip()]
    if all(re.fullmatch(r"(sorry|admit)\s*", ln) for ln in tactic_lines):
        return True, "all_sorry_or_admit"
    return False, ""


# ─── Gemini call wrapper + parsers ─────────────────────────────────────────

def _call_gemini_text(model, prompt: str, retries: int = 3) -> dict:
    """Call Gemini, return raw text + diagnostic. Retries on transient failures."""
    last_diag = ""
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt)
            try:
                if resp.candidates:
                    last_diag = f"finish_reason={resp.candidates[0].finish_reason}"
                if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                    pf = resp.prompt_feedback
                    if getattr(pf, "block_reason", None):
                        last_diag += f" prompt_block={pf.block_reason}"
            except Exception:
                pass
            text = (resp.text or "").strip() if hasattr(resp, "text") else ""
            if text:
                return {"text": text, "_diag": last_diag, "_parse_failed": False}
        except BaseException as e:
            last_diag = f"exception={type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    return {"text": "", "_diag": last_diag, "_parse_failed": True}


def _last_int_match(pat: re.Pattern, text: str) -> int | None:
    matches = list(pat.finditer(text or ""))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except Exception:
        return None


def _last_text_match(pat: re.Pattern, text: str) -> str:
    matches = list(pat.finditer(text or ""))
    return matches[-1].group(1).strip() if matches else ""


# ─── Prompts ───────────────────────────────────────────────────────────────

CALL_STMT_V3 = """\
You are an expert in formal mathematics and Lean 4.

Does the following Lean 4 theorem signature express the same mathematical \
proposition as the natural-language statement?

═══ Equivalence criteria ═══

Two statements are equivalent if they capture the same mathematical content, \
even if they differ in surface form: variable names, quantifier order (when \
commutative), notation (`\\le` vs `≤`, `\\sum` vs `∑`), or how the conclusion \
is phrased.

Use your mathematical understanding to judge whether the two statements \
describe the same problem.

═══ Note on inputs ═══

You are shown the Lean theorem signature (binders + goal type) only; the \
Lean proof body is hidden because it is judged separately.

═══ Output ═══

Provide a clear justification, then output:
   <stmt_justification>your reasoning here</stmt_justification>
   <stmt_score>1 or -1</stmt_score>

═══ Inputs ═══

Natural-language statement:
{nl_statement}

Lean 4 theorem signature:
{lean_theorem_signature}
"""


CALL_PROOF_V3 = """\
You are an expert in formal mathematics and Lean 4. Judge ONE thing:

Do the following two pieces of reasoning carry out the SAME LOGICAL STEPS?

═══ CRITICAL — Independence from validity ═══

You are shown a natural-language proof and a sequence of Lean 4 proof tactics. \
You are NOT shown the Lean theorem statement that these tactics target. This \
is intentional and important.

Your job is to compare the two REASONING SEQUENCES on their own merit:

  • Whether the Lean tactics actually prove the (hidden) Lean theorem is \
checked separately by Lean's type checker. You MUST IGNORE that question \
entirely. The tactics may or may not prove the theorem they are attached to \
— that is not your concern.

  • Even if the Lean tactics seem to refer to a different goal than the NL \
proof suggests, evaluate the reasoning sequence on its own terms.

═══ Equivalence criteria ═══

Two proofs are equivalent if they follow the SAME OVERALL STRATEGY and the \
substantive intermediate claims agree, modulo:
- wording and step ordering,
- level of abstraction (Lean tactics may IMPLEMENT abstract NL principles \
such as AM-GM, Cauchy-Schwarz, or trigonometric identities via automation \
tactics like `nlinarith`/`polyrith`/`positivity` — this counts as the SAME \
step, not a different one).

They are NOT equivalent if:
- they use a fundamentally different strategy or prove a different goal,
- the Lean proof body is empty, `sorry`, `admit`, or a degenerate placeholder,
- the substantive intermediate claims disagree.

═══ Output ═══

First, briefly map each substantive NL step to a corresponding Lean tactic \
(or note where they disagree). Then output:
   <proof_justification>your step-by-step correspondence here</proof_justification>
   <proof_score>1 or -1</proof_score>

═══ Inputs ═══

Natural-language proof:
{nl_proof}

Lean 4 proof tactics (theorem signature deliberately hidden):
{lean_proof_body}
"""


# ─── Per-call wrappers ─────────────────────────────────────────────────────

def _parse_stmt_response(text: str) -> dict:
    return {
        "stmt_score": _last_int_match(_STMT_SCORE_RE, text),
        "stmt_justification": _last_text_match(_STMT_JUST_RE, text),
    }


def _parse_proof_response(text: str) -> dict:
    return {
        "proof_score": _last_int_match(_PROOF_SCORE_RE, text),
        "proof_justification": _last_text_match(_PROOF_JUST_RE, text),
    }


def call_stmt_judge_v3(nl_statement: str, lean_theorem_signature: str, model) -> dict:
    prompt = CALL_STMT_V3.format(
        nl_statement=nl_statement or "(empty)",
        lean_theorem_signature=lean_theorem_signature or "(empty)",
    )
    raw = _call_gemini_text(model, prompt)
    parsed = _parse_stmt_response(raw["text"])
    parsed["_parse_failed"] = raw["_parse_failed"] or (parsed["stmt_score"] is None)
    parsed["_diag"] = raw["_diag"]
    parsed["_raw_response"] = raw["text"]
    return parsed


def call_proof_judge_v3(nl_proof: str, lean_proof_body: str, model) -> dict:
    prompt = CALL_PROOF_V3.format(
        nl_proof=nl_proof or "(empty)",
        lean_proof_body=lean_proof_body or "(empty)",
    )
    raw = _call_gemini_text(model, prompt)
    parsed = _parse_proof_response(raw["text"])
    parsed["_parse_failed"] = raw["_parse_failed"] or (parsed["proof_score"] is None)
    parsed["_diag"] = raw["_diag"]
    parsed["_raw_response"] = raw["text"]
    return parsed


# ─── Orchestrator ──────────────────────────────────────────────────────────

def _empty_result(reason: str) -> dict:
    return {
        "LLM_StmtSC?": "no",
        "LLM_ProofSC?": "no",
        "LLM_BothSC?": "no",
        "LLM_ValidProofSC?": "no",
        "LLM_FullyCorrect?": "no",
        "LLM_Semantics?": "no",
        "LLM_SC_details": {"version": "v3", "early_return_reason": reason},
    }


def run_sc_v3(informal_statement: str, informal_proof: str,
              generated_fl: str, tc_passes: bool, model) -> dict:
    """Decoupled SC: 2 calls, one per axis, each blind to the other axis."""
    thm_stmt, proof_body = extract_theorem_parts(generated_fl or "")
    if thm_stmt is None:
        return _empty_result("no_theorem")

    extractable_proof = proof_body is not None and bool((proof_body or "").strip())
    is_degen, degen_type = (False, "")
    if extractable_proof:
        is_degen, degen_type = detect_degenerate(proof_body)
    has_content = extractable_proof and not is_degen

    # Strip leading "tactics" / "lean4" markdown tags from the proof body
    # using the same convention as cleanse_lean (Bug-1 safety net).
    proof_body_clean = (proof_body or "").lstrip("﻿")
    while True:
        rest = proof_body_clean.lstrip("\n\r\t ")
        if not rest:
            break
        nl = rest.find("\n")
        first = rest[:nl] if nl != -1 else rest
        if first.strip().lower() in {"tactics", "lean", "lean4"}:
            proof_body_clean = rest[nl + 1:] if nl != -1 else ""
        else:
            break

    # Call 1: StmtSC — only theorem signature
    stmt_call = call_stmt_judge_v3(informal_statement, thm_stmt, model)

    # Call 2: ProofSC — only proof body (signature deliberately hidden).
    # If proof is degenerate, short-circuit to no without burning a call.
    if not has_content:
        proof_call = {
            "proof_score": -1,
            "proof_justification": f"degenerate proof body ({degen_type})",
            "_parse_failed": False,
            "_raw_response": "(skipped — degenerate)",
        }
    else:
        proof_call = call_proof_judge_v3(informal_proof, proof_body_clean, model)

    stmt_sc = (stmt_call.get("stmt_score") == 1)
    proof_sc = has_content and (proof_call.get("proof_score") == 1)
    both_sc = stmt_sc and proof_sc
    valid_proof_sc = bool(tc_passes) and proof_sc
    fully_correct = bool(tc_passes) and stmt_sc and proof_sc

    return {
        "LLM_StmtSC?": "yes" if stmt_sc else "no",
        "LLM_ProofSC?": "yes" if proof_sc else "no",
        "LLM_BothSC?": "yes" if both_sc else "no",
        "LLM_ValidProofSC?": "yes" if valid_proof_sc else "no",
        "LLM_FullyCorrect?": "yes" if fully_correct else "no",
        "LLM_Semantics?": "yes" if stmt_sc else "no",
        "LLM_SC_details": {
            "version": "v3_decoupled",
            "has_content": has_content,
            "degeneracy_type": degen_type or None,
            "stmt_call": {
                "stmt_score": stmt_call.get("stmt_score"),
                "_parse_failed": stmt_call.get("_parse_failed", False),
                "_raw_response": (stmt_call.get("_raw_response", "") or "")[:600],
            },
            "proof_call": {
                "proof_score": proof_call.get("proof_score"),
                "_parse_failed": proof_call.get("_parse_failed", False),
                "_raw_response": (proof_call.get("_raw_response", "") or "")[:600],
            },
        },
    }
