"""Global perturbation construction for RobustPABench.

Global perturbations are meaning-preserving rewrites of both the informal
statement and informal proof.  This module returns HuggingFace ``Dataset``
objects keyed by the Hub config names used by the paper:

    global_original
    global_gemini_rephrase
    global_gemini_step
    global_qwen3_rephrase
    global_qwen3_step

Intermediate JSONL files are written under ``data_dir`` so interrupted API runs
can resume.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable

from datasets import Dataset

from utils import load_jsonl, make_gemini_model, write_jsonl


FAITHFULNESS_RULES = """This is a MEANING-PRESERVING transformation. Your rewrite must be a faithful translation of the original; a reader must be able to recover the original's exact mathematical content from your version. Specifically:

- Do NOT change any numerical constant, variable name, function name, set, inequality direction, or quantifier.
- Do NOT change, weaken, strengthen, or generalize the hypotheses or the conclusion.
- Do NOT add new mathematical facts, lemmas, intermediate results, or justifications that were not in the original.
- Do NOT remove, merge, or reorder logical steps of the original argument.
- Do NOT fix or "clean up" the original. If the original is awkward, incomplete, or even wrong, preserve it faithfully; your job is translation, not editing.
- The rewritten version should be roughly the same length as the original (within plus/minus 30%).

Use LaTeX math notation exactly as in the original (e.g., $x^2$, \\frac{a}{b}, \\neq, \\notin, \\tan). Write every backslash literally as a single backslash; do NOT escape them, do NOT wrap anything in JSON, do NOT add code fences.

Output format: return EXACTLY these two tagged blocks and NOTHING else before or after. Your entire response must start with `<informal_statement>` and end with `</informal_proof>`.

<informal_statement>
... rewritten problem statement ...
</informal_statement>

<informal_proof>
... rewritten proof ...
</informal_proof>"""


PROMPTS = {
    "rephrase": f"""You are a mathematics professor. Rephrase the following problem statement and its proof into different wording while preserving the exact same mathematical meaning. Change only the phrasing, sentence structure, and word choice; keep every mathematical object, value, and logical step identical.

{FAITHFULNESS_RULES}
Original problem statement:
%STATEMENT%

Original proof:
%PROOF%""",
    "step": f"""You are a mathematics professor. Faithfully translate the following problem statement and its proof into a numbered step-by-step format. This is a pure formatting change: break the existing content into numbered steps with short bold step titles. Do NOT add explanations, details, motivation, or any content that is not already in the original; only reorganize what is there.

The proof should follow this structure:
1. **Step Title:** <content taken directly from the original>
2. **Step Title:** <content taken directly from the original>

The statement may use the same numbered-step format if it naturally decomposes, or remain as a single paragraph if it does not, but its mathematical content must stay identical.

{FAITHFULNESS_RULES}
Original problem statement:
%STATEMENT%

Original proof:
%PROOF%""",
}


_STMT_RE = re.compile(r"<informal_statement>(.*?)</informal_statement>", re.DOTALL)
_PROOF_RE = re.compile(r"<informal_proof>(.*?)</informal_proof>", re.DOTALL)


def _strip_statement_comment(statement: str) -> str:
    statement = (statement or "").strip()
    if statement.startswith("/--"):
        statement = statement[3:].strip()
    if statement.endswith("-/"):
        statement = statement[:-2].strip()
    return statement


def parse_response(text: str | None) -> dict | None:
    """Extract raw statement/proof text from tagged model output."""
    if not text:
        return None
    stmt_match = _STMT_RE.search(text)
    proof_match = _PROOF_RE.search(text)
    if not stmt_match or not proof_match:
        return None
    statement = stmt_match.group(1).strip()
    proof = proof_match.group(1).strip()
    if not statement or not proof:
        return None
    return {"informal_statement": statement, "informal_proof": proof}


def _call_gemini(prompt: str, model_name: str) -> dict | None:
    model = make_gemini_model(model_name)
    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            result = parse_response(getattr(response, "text", "") or "")
            if result:
                return result
            raise ValueError("Could not parse tagged response")
        except Exception as exc:  # noqa: BLE001 - keep API retry diagnostics concise
            print(f"  [Gemini] attempt {attempt + 1} failed: {exc}")
            time.sleep(5)
    return None


def _call_openai_compatible(prompt: str, *, model_name: str, label: str) -> dict | None:
    from openai import OpenAI

    api_key = os.environ.get("NRP_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NRP_LLM_KEY for the OpenAI-compatible rewrite model")
    base_url = os.environ.get("NRP_LLM_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            text = message.content or getattr(message, "reasoning", "") or ""
            result = parse_response(text.strip())
            if result:
                return result
            raise ValueError("Could not parse tagged response")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] attempt {attempt + 1} failed: {exc}")
            time.sleep(5)
    return None


def _rewrite_rows(
    rows: list[dict],
    *,
    output_path: Path,
    style: str,
    call_fn: Callable[[str], dict | None],
    limit: int,
    sleep_s: float,
) -> list[dict]:
    rows = rows[:limit] if limit else rows
    done: dict[str, dict] = {}
    if output_path.exists():
        done = {row["name"]: row for row in load_jsonl(output_path)}
        print(f"Resuming {output_path}: {len(done)}/{len(rows)} already done")

    prompt_template = PROMPTS[style]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            if row["name"] in done:
                continue

            proof = str(row.get("informal_proof", "") or "").strip()
            if not proof:
                out_row = dict(row)
                print(f"  [{idx}/{len(rows)}] {row['name']}: no proof, copied original")
            else:
                statement = _strip_statement_comment(row.get("informal_statement", ""))
                prompt = prompt_template.replace("%STATEMENT%", statement).replace("%PROOF%", proof)
                print(f"  [{idx}/{len(rows)}] {row['name']}...", end=" ", flush=True)
                result = call_fn(prompt)
                out_row = dict(row)
                if result:
                    out_row["informal_statement"] = f"/-- {result['informal_statement'].strip()}-/"
                    out_row["informal_proof"] = result["informal_proof"].strip()
                    print("OK")
                else:
                    print("FAILED, copied original")

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out.flush()
            if sleep_s:
                time.sleep(sleep_s)

    return load_jsonl(output_path)


def run(
    dataset: str,
    data_dir: str,
    *,
    style: str = "all",
    model: str = "all",
    gemini_model: str = "gemini-2.5-flash",
    qwen_model: str = "qwen3",
    limit: int = 0,
    sleep_s: float = 1.0,
) -> dict[str, Dataset]:
    """Build global perturbation datasets for one source JSONL."""
    rows = load_jsonl(dataset)
    if limit:
        rows = rows[:limit]
    data_dir_path = Path(data_dir)
    datasets: dict[str, Dataset] = {}

    styles = ["rephrase", "step"] if style == "all" else ([] if style == "original" else [style])
    models = ["gemini", "qwen3"] if model == "all" else [model]

    if style in ("all", "original"):
        original_path = data_dir_path / "global_original.jsonl"
        write_jsonl(original_path, rows)
        datasets["global_original"] = Dataset.from_list(rows)

    for model_key in models:
        for style_key in styles:
            config_name = f"global_{model_key}_{style_key}"
            output_path = data_dir_path / f"{config_name}.jsonl"
            if model_key == "gemini":
                call_fn = lambda prompt, m=gemini_model: _call_gemini(prompt, m)
            elif model_key == "qwen3":
                call_fn = lambda prompt, m=qwen_model: _call_openai_compatible(
                    prompt, model_name=m, label="qwen3"
                )
            else:
                raise ValueError(f"Unknown global rewrite model: {model_key}")

            rewritten = _rewrite_rows(
                rows,
                output_path=output_path,
                style=style_key,
                call_fn=call_fn,
                limit=limit,
                sleep_s=sleep_s,
            )
            datasets[config_name] = Dataset.from_list(rewritten)

    return datasets
