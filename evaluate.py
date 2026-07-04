"""Score inference output: type-check (TC) + statement/proof semantic checks.

Reads inference output (LLM_Output#k fields), runs a Lean type-check plus a
2-call decoupled semantic-consistency judge (StmtSC + ProofSC), and writes
per-sample results to a jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re as _re
import threading
import time

from datasets import load_dataset
from dotenv import load_dotenv

from sc import run_sc_v3 as run_sc
from utils import load_jsonl, make_gemini_model, robust_json_loads
from verify import extract_region, verify_lean

load_dotenv()  # GOOGLE_API_KEY / HF_TOKEN from .env


def _derive_path(inference_output, suffix):
    """Derive a sibling output path from the inference-output path.

    Replaces a trailing ``_output.jsonl`` (or just the extension) with ``suffix``
    so scored/summary files land next to the inference output.
    """
    p = str(inference_output)
    if p.endswith("_output.jsonl"):
        return p[: -len("_output.jsonl")] + suffix
    return str(Path(p).with_suffix("")) + suffix


def _sample_keys(row, sample_key):
    if sample_key == "auto":
        return sorted(
            [k for k in row.keys() if _re.fullmatch(r"LLM_Output#\d+", k)],
            key=lambda k: int(k.split("#")[1]),
        )
    return [sample_key]


def _batch_typecheck(pending, sample_key):
    """Type-check every sample of every pending row via the kimina Lean server.

    Returns ``{name#idx: (ok, error)}``. Empty / ``ERROR`` outputs short-circuit
    to a failed check without hitting the server. All real outputs go through a
    single batched ``verify_lean`` call.
    """
    tc = {}
    codes, ids = [], []
    for name, row in pending:
        for key in _sample_keys(row, sample_key):
            idx = key.split("#")[1]
            out = str(row.get(key, "") or "")
            cid = f"{name}#{idx}"
            if out and not out.startswith("ERROR"):
                codes.append(out)
                ids.append(cid)
            else:
                tc[cid] = (False, "(empty or parse-error output, skipped Lean)")
    if codes:
        print(f"[evaluate] type-checking {len(codes)} outputs via kimina-lean-server...")
        results = verify_lean(codes)
        for cid, res in zip(ids, results):
            if res.get("passed"):
                tc[cid] = (True, "")
            else:
                sys_err = res.get("system_error")
                err = (str(sys_err) if sys_err else str(res.get("errors") or ""))[:500]
                tc[cid] = (False, err)
    return tc


def run_tcsc(args):
    """Score inference output: type-check (TC, via kimina) + StmtSC + ProofSC."""
    do_tc = args.mode in ("full", "tc")
    do_sc = args.mode in ("full", "sc")

    output_path = Path(args.output or _derive_path(args.input, "_scored.jsonl"))
    rows = load_jsonl(Path(args.input))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[evaluate] → {len(rows)} rows (scored → {output_path})")

    print(f"[evaluate] mode={args.mode}  (do_tc={do_tc} do_sc={do_sc})")

    model = None
    if do_sc:
        model = make_gemini_model(args.gemini_model)
        print(f"[evaluate] Gemini: {args.gemini_model}")

    pending = [(r.get("name", f"row_{i}"), r) for i, r in enumerate(rows)]

    # Phase 1: batch type-check all pending samples through one server run.
    tc = _batch_typecheck(pending, args.sample_key) if do_tc else {}

    # Phase 2: per-sample StmtSC/ProofSC (Gemini) + assemble enriched rows.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        t_start = time.time()
        total = len(pending)
        for i, (name, row) in enumerate(pending):
            sample_keys = _sample_keys(row, args.sample_key)
            informal_stmt = row.get("informal_statement", "") or ""
            informal_proof = row.get("informal_proof", "") or ""

            enriched = dict(row)
            per_sample_summary = []
            for key in sample_keys:
                idx = key.split("#")[1]
                llm_output = str(row.get(key, "") or "")

                tc_ok, tc_err = tc.get(f"{name}#{idx}", (False, "")) if do_tc else (False, "")
                if do_tc:
                    enriched[f"LLM_Syntax?#{idx}"] = "yes" if tc_ok else "no"
                    enriched[f"LLM_SyntaxError#{idx}"] = tc_err if not tc_ok else ""

                if do_sc:
                    # If TC was skipped, tc_passes=False (only used by ValidProofSC/FullyCorrect aggregation).
                    sc = run_sc(
                        informal_statement=informal_stmt,
                        informal_proof=informal_proof,
                        generated_fl=llm_output,
                        tc_passes=tc_ok,
                        model=model,
                    )
                    for field, val in sc.items():
                        if field == "LLM_SC_details":
                            enriched[f"LLM_SC_details#{idx}"] = val
                        else:
                            enriched[f"{field}#{idx}"] = val

                bits = []
                if do_tc:
                    bits.append(f"TC={enriched.get(f'LLM_Syntax?#{idx}','?')}")
                if do_sc:
                    bits.append(f"StmtSC={enriched.get(f'LLM_StmtSC?#{idx}','?')}")
                    bits.append(f"ProofSC={enriched.get(f'LLM_ProofSC?#{idx}','?')}")
                per_sample_summary.append(f"#{idx}: " + " ".join(bits))

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()

            elapsed = time.time() - t_start
            processed = i + 1
            rate = processed / elapsed if elapsed > 0 else 0
            eta_min = (total - processed) / rate / 60 if rate > 0 else 0
            print(f"  [{processed}/{total}] {name} ({len(sample_keys)} samples)  "
                  f"{' | '.join(per_sample_summary[:2])}{' ...' if len(per_sample_summary) > 2 else ''}  "
                  f"(rate={rate:.3f}/s ETA={eta_min:.1f}min)",
                  flush=True)

    print("[evaluate] done")


# ═══════════════════════════════════════════════════════════════════════════
# Edit-faithfulness scoring (FR / RR / OUR)
#
# A perturbation-specific Gemini judge: given one edit (number / symbol / step)
# and the model's Lean output, decide whether the model carried the edit
# (faithful), silently reverted it (reverted), or neither (other). For number
# and symbol edits the judge sees ONLY the Lean region the edit targets
# (signature for statement edits, proof body for proof edits); step-delete
# judges the full output.
# ═══════════════════════════════════════════════════════════════════════════

NUMBER_EDIT_PROMPT = """\
You are an expert in mathematical formalization. Given a math problem (whose natural language has been edited by changing one number) and a fragment of the model-generated Lean 4 formalization, determine whether the model faithfully reflects the number edit.

The Lean fragment shown below is the ONLY part of the formalization you should consider. It is the region targeted by the edit ({source}). Do not infer or imagine any other Lean code. Base your judgment strictly on the fragment.

--- Edit Info ---
Edit type: {edit_type}
Original number: {old_value}
Edited number: {new_value}
Edit location: {source} (statement or proof)
Edit context: "{context}"

--- Edited informal statement ---
{informal_statement}

--- Edited informal proof ---
{informal_proof}

--- Model-generated Lean 4 fragment ({source} region only) ---
{generated_fl}

Tasks:

1. In the Lean fragment, find the number that semantically corresponds to "{new_value}" or "{old_value}" in the edit context.
   Note: Numbers in Lean may differ from natural language, e.g., 6.5 may appear as 13/2, and 65 may appear as (65 : R).

2. Determine the number at that position:
   - If it is {new_value} (or an equivalent form) -> "faithful"
   - If it is {old_value} (or an equivalent form) -> "reverted"
   - If the corresponding position cannot be found in the fragment, or it is some other value -> "other"

IMPORTANT — ground your judgment in evidence; do not hallucinate. Before returning "faithful" or "reverted", you MUST be able to point to the literal substring of the fragment where the value appears. If you cannot find such a substring, return value_in_fl="" and judgment="other". The context_in_fl field below is the proof — it must be copied verbatim from the fragment and must contain value_in_fl.

Return strict JSON only (no other text):
{{"found_in_fl": true/false,
  "value_in_fl": "copy the number from the fragment as-is (or empty string if not found)",
  "context_in_fl": "if value_in_fl is non-empty, copy the ~40-character window of the fragment surrounding the value, preserving exact characters; otherwise empty string",
  "judgment": "faithful"/"reverted"/"other",
  "reason": "one-sentence explanation"}}"""

SYMBOL_EDIT_PROMPT = """\
You are an expert in mathematical formalization. Given a math problem (whose natural language has been edited by changing one math symbol) and a fragment of the model-generated Lean 4 formalization, determine whether the model faithfully reflects the symbol edit.

The Lean fragment shown below is the ONLY part of the formalization you should consider. It is the region targeted by the edit ({source}). Do not infer or imagine any other Lean code. Base your judgment strictly on the fragment.

--- Edit Info ---
Edit type: {edit_type}
Original symbol: {old_symbol}
Edited symbol: {new_symbol}
Symbol family: {family}
Edit location: {source} (statement or proof)
Edit context: "{context}"

--- Edited informal statement ---
{informal_statement}

--- Edited informal proof ---
{informal_proof}

--- Model-generated Lean 4 fragment ({source} region only) ---
{generated_fl}

Tasks:

1. In the Lean fragment, find the math symbol that corresponds to the edited symbol in the edit context.
   Symbol mapping from LaTeX to Lean:
   - \\geq / >= -> Lean: >=
   - \\leq / <= -> Lean: <=
   - > -> Lean: >
   - < -> Lean: <
   - \\neq / != -> Lean: != or Ne
   - + -> Lean: +
   - - -> Lean: -
   - \\times / \\cdot -> Lean: *
   - \\div -> Lean: /
   Note: a > b in Lean may be written as b < a (equivalent flip). If the edit is > -> <, this counts as faithful.

2. Determine the symbol at that position:
   - If it is the Lean equivalent of {new_symbol} -> "faithful"
   - If it is the Lean equivalent of {old_symbol} -> "reverted"
   - If the corresponding position cannot be found in the fragment, or other cases -> "other"

IMPORTANT — ground your judgment in evidence; do not hallucinate. Before returning "faithful" or "reverted", you MUST be able to point to the literal substring of the fragment where the symbol appears. If you cannot find such a substring, return value_in_fl="" and judgment="other". The context_in_fl field below is the proof — it must be copied verbatim from the fragment and must contain value_in_fl.

Return strict JSON only (no other text):
{{"found_in_fl": true/false,
  "value_in_fl": "copy the symbol from the fragment as-is (or empty string if not found)",
  "context_in_fl": "if value_in_fl is non-empty, copy the ~40-character window of the fragment surrounding the symbol, preserving exact characters; otherwise empty string",
  "judgment": "faithful"/"reverted"/"other",
  "reason": "one-sentence explanation"}}"""

STEP_EDIT_PROMPT = """\
You are an expert in mathematical formalization. A math proof was edited by removing the reasoning from one proof step while keeping only the step's outcome/claim. The model then generated a Lean 4 formalization from the edited proof.

Determine whether the model faithfully reflects the missing reasoning.

--- Edit Info ---
Deleted reasoning: "{reasoning_text}"
Preserved outcome: "{outcome_text}"
Step was last step: {is_last}

--- Edited informal proof (reasoning removed) ---
{informal_proof}

--- Model-generated Lean 4 code ---
{generated_fl}

Tasks:

1. Find where the outcome claim ("{outcome_text}") is formalized in the Lean code. Look for a `have`, `show`, theorem conclusion, or equivalent statement that corresponds to this claim.

2. Determine how the model handles the proof of that claim:
   - If the model uses `sorry`, `admit`, leaves it incomplete, or provides a significantly simplified/placeholder proof -> the model is FAITHFUL (it reflected the missing reasoning)
   - If the model provides a full, substantive tactic proof for this claim (e.g., `by rw [...]; norm_num; linarith`) that reconstructs the deleted reasoning -> the model REVERTED (it fabricated the missing justification)
   - If the outcome claim is not present at all, or the output is empty/error -> OTHER

IMPORTANT — ground your judgment in evidence; do not hallucinate. Before returning "faithful" or "reverted", you MUST be able to point to the literal Lean code that supports your call: for "faithful" the placeholder/sorry/simplified token(s), for "reverted" the substantive tactic block(s) that reconstruct the deleted reasoning. If you cannot find such literal evidence, return judgment="other". The evidence_in_fl field is the proof of your judgment — it must be copied verbatim from the Lean code.

Return strict JSON only (no other text):
{{"outcome_present": true/false,
  "proof_method": "sorry"/"placeholder"/"simplified"/"full"/"absent" (pick exactly one),
  "evidence_in_fl": "if judgment is faithful or reverted, copy the relevant ~80-character snippet of Lean code that supports the verdict, preserving exact characters; otherwise empty string",
  "judgment": "faithful"/"reverted"/"other",
  "reason": "one-sentence explanation"}}"""


# Per-kind adapters. ``prompt_args`` returns the .format() kwargs; ``meta``
# returns the edit-identifying fields copied into each scored row; ``result_defaults``
# are the non-judgment judge fields (also used for empty/degenerate rows).
EDIT_SPECS = {
    "number": {
        "prompt": NUMBER_EDIT_PROMPT,
        "uses_region": True,
        "source_field": "number_edit_source",
        "edit_type_field": "number_edit_type",
        "result_defaults": {"found_in_fl": False, "value_in_fl": "", "context_in_fl": ""},
        "prompt_args": lambda row, frag: dict(
            edit_type=row.get("number_edit_type", ""),
            old_value=row.get("number_edit_old_value", ""),
            new_value=row.get("number_edit_new_value", ""),
            source=row.get("number_edit_source", ""),
            context=row.get("number_edit_context", ""),
            informal_statement=row.get("informal_statement", ""),
            informal_proof=row.get("informal_proof", ""),
            generated_fl=frag,
        ),
        "meta": lambda row, src: {
            "old_value": row.get("number_edit_old_value", ""),
            "new_value": row.get("number_edit_new_value", ""),
            "edit_source": src,
        },
    },
    "symbol": {
        "prompt": SYMBOL_EDIT_PROMPT,
        "uses_region": True,
        "source_field": "symbol_edit_source",
        "edit_type_field": "symbol_edit_type",
        "result_defaults": {"found_in_fl": False, "value_in_fl": "", "context_in_fl": ""},
        "prompt_args": lambda row, frag: dict(
            edit_type=row.get("symbol_edit_type", ""),
            old_symbol=row.get("symbol_edit_old_symbol", ""),
            new_symbol=row.get("symbol_edit_new_symbol", ""),
            family=row.get("symbol_edit_family", ""),
            source=row.get("symbol_edit_source", ""),
            context=row.get("symbol_edit_context", ""),
            informal_statement=row.get("informal_statement", ""),
            informal_proof=row.get("informal_proof", ""),
            generated_fl=frag,
        ),
        "meta": lambda row, src: {
            "old_symbol": row.get("symbol_edit_old_symbol", ""),
            "new_symbol": row.get("symbol_edit_new_symbol", ""),
            "family": row.get("symbol_edit_family", ""),
            "edit_source": src,
        },
    },
    "step": {
        "prompt": STEP_EDIT_PROMPT,
        "uses_region": False,
        "source_field": "",
        "edit_type_field": "",  # constant "step_delete"
        "result_defaults": {"outcome_present": False, "proof_method": "absent", "evidence_in_fl": ""},
        "prompt_args": lambda row, frag: dict(
            reasoning_text=row.get("step_edit_target_reasoning_text", ""),
            outcome_text=row.get("step_edit_target_outcome_text", ""),
            is_last=row.get("step_edit_is_last", False),
            informal_proof=row.get("informal_proof", ""),
            generated_fl=frag,
        ),
        "meta": lambda row, src: {
            "target_step_idx": row.get("step_edit_target_step_idx", -1),
            "outcome_text": row.get("step_edit_target_outcome_text", ""),
        },
    },
}


def _judge_edit(model, prompt, result_defaults):
    """Call the Gemini judge; return the parsed judgment dict (5 retries)."""
    for attempt in range(5):
        try:
            text = model.generate_content(prompt).text.strip()
            result = robust_json_loads(text) or {}
            judgment = result.get("judgment", "other")
            if judgment not in ("faithful", "reverted", "other"):
                judgment = "other"
            out = {k: result.get(k, d) for k, d in result_defaults.items()}
            out["judgment"] = judgment
            out["reason"] = result.get("reason", "")
            return out
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    out = dict(result_defaults)
    out["judgment"] = "other"
    out["reason"] = "scoring_failed_after_retries"
    return out


def _fr_rr_our(rows):
    counts = Counter(r["judgment"] for r in rows)
    total = len(rows)
    fr, rr, our = counts.get("faithful", 0), counts.get("reverted", 0), counts.get("other", 0)
    return {
        "count": total,
        "FR": round(fr / total, 4) if total else 0,
        "RR": round(rr / total, 4) if total else 0,
        "OUR": round(our / total, 4) if total else 0,
        "counts": {"faithful": fr, "reverted": rr, "other": our},
    }


def score_edit(kind, model, rows, output_path, scored_path):
    """Judge one edited dataset (rows in memory); write scored jsonl; return summary."""
    spec = EDIT_SPECS[kind]
    outputs = load_jsonl(Path(output_path))
    outputs_by_name = {o["name"]: o for o in outputs if o.get("name")}
    missing = [r.get("name", f"row_{i}") for i, r in enumerate(rows)
               if r.get("name") not in outputs_by_name]
    if missing:
        print(f"  WARNING: {len(missing)} input rows have no matching output "
              f"(first 3: {missing[:3]})")

    scored_path = Path(scored_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_rows = []

    pending = []
    for i, row in enumerate(rows):
        name = row.get("name", f"row_{i}")
        output = outputs_by_name.get(name, {})
        fl_text = str(output.get("LLM_Output#1", "") or "")
        src = row.get(spec["source_field"], "") if spec["source_field"] else ""
        if spec["uses_region"]:
            fragment = extract_region(fl_text, src) if fl_text and not fl_text.startswith("ERROR") else ""
        else:
            fragment = fl_text
        pending.append((name, row, fl_text, fragment, src))

    def score_task(task):
        name, row, fl_text, fragment, src = task
        if not fl_text or fl_text.startswith("ERROR"):
            result = dict(spec["result_defaults"])
            result["judgment"], result["reason"] = "other", "empty_or_error_output"
            region_status = "empty_output"
        elif spec["uses_region"] and not fragment:
            result = dict(spec["result_defaults"])
            result["judgment"], result["reason"] = "other", "degenerate_lean_no_target_region"
            region_status = "no_target_region"
        else:
            prompt = spec["prompt"].format(**spec["prompt_args"](row, fragment))
            result = _judge_edit(model, prompt, spec["result_defaults"])
            region_status = "judged"
        edit_type = row.get(spec["edit_type_field"], "") if spec["edit_type_field"] else "step_delete"
        entry = {"name": name, "edit_type": edit_type}
        entry.update(spec["meta"](row, src))
        if spec["uses_region"]:
            entry["region_status"] = region_status
            entry["region_chars"] = len(fragment)
        entry.update(result)
        return entry

    max_workers = int(os.environ.get("SCORE_WORKERS", "30"))
    file_lock = threading.Lock()
    print(f"  Scoring {kind} with {max_workers} workers on {len(pending)} pending rows")
    with open(scored_path, "w", encoding="utf-8") as fout:
        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(score_task, t) for t in pending]
                for done, fut in enumerate(as_completed(futures), 1):
                    entry = fut.result()
                    with file_lock:
                        scored_rows.append(entry)
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fout.flush()
                    if done % 50 == 0 or done == len(pending):
                        print(f"  [{done}/{len(pending)}] scored "
                              f"(latest: {entry['name']} -> {entry['judgment']})")

    by_type = defaultdict(list)
    for r in scored_rows:
        by_type[r.get("edit_type", kind)].append(r)
    summary = {et: _fr_rr_our(group) for et, group in by_type.items()}

    print(f"\n{'='*50}\n  FR/RR/OUR — {kind}\n{'='*50}")
    print(f"{'Type':<20} {'FR':>7} {'RR':>7} {'OUR':>7} {'N':>6}")
    print("-" * 50)
    for et, m in summary.items():
        print(f"{et:<20} {m['FR']:>6.1%} {m['RR']:>6.1%} {m['OUR']:>6.1%} {m['count']:>6}")
    return summary


def _load_edited(repo_id, config, split):
    """Load an edited dataset from the Hub as a list of row dicts."""
    print(f"[evaluate] loading {repo_id} :: {config} [{split}]")
    ds = load_dataset(repo_id, config, split=split)
    return [dict(r) for r in ds]


def _kind_from_config(config):
    """Infer the edit kind (number / symbol / step) from the HF config name."""
    for kind in ("number", "symbol", "step"):
        if kind in config:
            return kind
    raise ValueError(f"Cannot infer edit kind from config '{config}'")


def run_edit(args):
    """FR/RR/OUR edit-faithfulness judge (number / symbol / step)."""
    model = make_gemini_model(args.gemini_model)

    kind = _kind_from_config(args.config)
    scored_output = args.scored_output or _derive_path(args.output_jsonl, "_scored.jsonl")
    summary = args.summary or _derive_path(args.output_jsonl, "_summary.json")

    rows = _load_edited(args.repo_id, args.config, args.split)
    if args.limit > 0:
        rows = rows[:args.limit]
    summary_data = score_edit(kind, model, rows, args.output_jsonl, scored_output)

    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"[evaluate] summary -> {summary}")


def main():
    p = argparse.ArgumentParser(
        description="Score inference output: TC/SC (task=tcsc) or edit-faithfulness FR/RR/OUR (task=edit)")
    sub = p.add_subparsers(dest="task", required=True)

    t = sub.add_parser("tcsc", help="Type-check (TC) + StmtSC + ProofSC")
    t.add_argument("--input", required=True)
    t.add_argument("--output", default="",
                   help="Scored jsonl to write (default: <input with _output->_scored>)")
    t.add_argument("--gemini_model", default="gemini-2.5-flash")
    t.add_argument("--sample_key", default="auto")
    t.add_argument("--mode", choices=("full", "tc", "sc"), default="full",
                   help="full = TC + SC (default); tc = TC only (no Gemini, no SC fields); "
                        "sc = SC only (no Lean REPL, no TC fields)")
    t.add_argument("--limit", type=int, default=0,
                   help="Limit number of rows processed (0 for no limit)")
    t.set_defaults(func=run_tcsc)

    e = sub.add_parser("edit", help="FR/RR/OUR edit-faithfulness judge")
    e.add_argument("--repo-id", default="ucr-rai/RobustPABench", help="HF dataset repo")
    e.add_argument("--config", required=True, help="HF config, e.g. local_number_edit_statement")
    e.add_argument("--split", default="miniF2F", help="HF split to score")
    e.add_argument("--output_jsonl", required=True, help="Inference output (LLM_Output#1 per row)")
    e.add_argument("--scored_output", default="",
                   help="Per-sample scored jsonl to write "
                        "(default: <output_jsonl with _output->_scored>)")
    e.add_argument("--summary", default="",
                   help="Summary json to write (default: <output_jsonl with _output.jsonl->_summary.json>)")
    e.add_argument("--gemini_model", default="gemini-2.5-flash")
    e.add_argument("--limit", type=int, default=0,
                   help="Limit number of rows processed (0 for no limit)")
    e.set_defaults(func=run_edit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
