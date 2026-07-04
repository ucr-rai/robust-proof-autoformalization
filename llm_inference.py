"""Proof-autoformalization inference with any chat model via vLLM.

Loads a HuggingFace chat/prover model with an in-process vLLM engine (no
separate server needed) and autoformalizes the natural-language proofs of a
RobustPABench split into Lean 4. The dataset can come from the HuggingFace Hub
(``--repo_id``/``--config``/``--split``) or a local jsonl (``--dataset_path``).
Outputs are written as ``LLM_Output#k`` fields that ``evaluate.py`` consumes.

Inference is run in zero-shot mode.
"""

from tqdm import tqdm
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from vllm import LLM
from google.genai import types
from utils import _GeminiModel, _SAFETY_OFF

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run proof-autoformalization inference with any model via vLLM"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=max(torch.cuda.device_count(), 1),
        help="Number of GPUs to shard the model across (defaults to all visible GPUs)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Max model context length (defaults to the model's own setting)",
    )
    parser.add_argument(
        "--num_samples_per_task",
        type=int,
        default=1,
        help="Number of samples to generate per task",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gemini-3.1-pro",
        help="Model identifier: either a Gemini model (starting with 'gemini-') or a HuggingFace model via vLLM",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results/miniF2F",
        help="Directory to store inference results",
    )
    # Dataset source: either a HuggingFace Hub dataset (repo_id/config/split)
    # or a local jsonl (dataset_path). If dataset_path is given it takes priority.
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ucr-rai/RobustPABench",
        help="HF dataset repo to load the benchmark from",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="HF dataset config, e.g. number_edit_statement",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="miniF2F",
        help="HF dataset split to run inference on",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a local dataset jsonl (overrides --repo_id if set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run inference on the first N examples (0 = all)",
    )

    return parser.parse_args()


# ── Lean-code parser ────────────────────────────────────────────────────────

def _clean_lean_code(code):
    """Remove non-Lean content that may be captured by fallback strategies."""
    code = re.sub(r"</formal_(theorem|proof)>.*", "", code, flags=re.DOTALL)
    code = re.sub(r"^<formal_(theorem|proof)>\s*", "", code)
    code = re.sub(r"^```(lean4?)?\s*", "", code)
    code = re.sub(r"\s*```\s*$", "", code)
    lines = code.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("--") and re.match(
            r"^(This|Note|The above|Here|I |We |In |My |Above|Proof|QED|Q\.E\.D)",
            stripped
        ):
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def extract_fl_proof(text):
    """Extract Lean code with multiple fallback strategies."""
    if not text:
        return ""
    text = str(text)

    # Strip <think>...</think> blocks (thinking-mode models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strategy 1: ```lean4 ... ```
    matches = re.findall(r"```lean4\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Strategy 2: ``` ... ``` containing lean keywords
    matches2 = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    for m in reversed(matches2):
        if 'import' in m or 'theorem' in m:
            return m.strip()

    # Strategy 3: <formal_proof>/<formal_theorem> tag
    for tag in ("formal_proof", "formal_theorem"):
        start_delim = f"<{tag}>"
        end_delim = f"</{tag}>"
        if start_delim in text and end_delim in text:
            inner = text.split(start_delim)[-1].split(end_delim)[0].strip()
            if inner:
                inner_fenced = re.findall(r"```(?:lean4)?\s*(.*?)```", inner, re.DOTALL)
                if inner_fenced:
                    return inner_fenced[-1].strip()
                return _clean_lean_code(inner)

    # Strategy 4: import Mathlib
    if "import Mathlib" in text:
        return _clean_lean_code(text[text.index("import Mathlib"):])

    # Strategy 5: import Aesop
    if "import Aesop" in text:
        return _clean_lean_code(text[text.index("import Aesop"):])

    # Strategy 6: theorem keyword
    if re.search(r'\btheorem\s+\w+', text):
        match = re.search(r'(theorem\s+\w+.*)', text, re.DOTALL)
        if match:
            return _clean_lean_code(match.group(1))

    return ""


# ── Dataset loading + result scaffolding ────────────────────────────────────

def _load_dataset_rows(args):
    """Load the benchmark rows from a local jsonl or the HuggingFace Hub."""
    if args.dataset_path:
        print(f"[inference] loading local dataset {args.dataset_path}")
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    print(f"[inference] loading {args.repo_id} :: {args.config} [{args.split}]")
    ds = load_dataset(args.repo_id, args.config, split=args.split)
    return [dict(r) for r in ds]


def prepare_evaluation(rows, eval_path, run_name, num_samples):
    eval_filename = f"{run_name}_output.jsonl"
    eval_file_path = os.path.join(eval_path, eval_filename)

    for data_instance in rows:
        data_instance["INFERENCE_DONE"] = "no"
        for i in range(1, num_samples + 1):
            for col_base in [
                "LLM_Output#",
                "LLM_Syntax?#",
                "LLM_SyntaxError#",
                "LLM_Semantics?#",
                "LLM_SemanticsError#",
            ]:
                data_instance[f"{col_base}{i}"] = ""

    with open(eval_file_path, "w", encoding="utf-8") as f:
        for data_instance in rows:
            f.write(json.dumps(data_instance) + "\n")

    print(f"Initialized output file at {eval_file_path}")
    return rows, eval_file_path


def save_jsonl(list_of_dicts, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for data_instance in list_of_dicts:
            f.write(json.dumps(data_instance) + "\n")


# ── Prompt construction ─────────────────────────────────────────────────────


def wrap_prompt_in_query(informal_statement, informal_proof):
    """Build user prompt for zero-shot autoformalization."""
    return f'''
    You task is to take as input an informal proof in natural language and autoformalize it in Lean 4 with a header.
    Think step-by-step and ensure that the output formal theorem is compilabile with Lean 4 (version 4.15.0).

    Here is the **actual** informal proof in natural language:
    <informal_statement>
    {informal_statement}
    </informal_statement>

    <informal_proof>
    {informal_proof}
    </informal_proof>

    Now first think step-by-step for the actual output and autoformalize it in Lean 4 with a header. Importantly, enclose the final formal proof in Lean 4 inside the following tags:

    <formal_proof>
    ```lean4
    (Provide your entire Lean 4 proof with header here)
    ```
    </formal_proof>
    '''


# ── Main inference loop ─────────────────────────────────────────────────────

def inference_on_dataset(args):
    if args.model_id.startswith("https://huggingface.co/"):
        args.model_id = args.model_id[len("https://huggingface.co/"):]

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    # Organize outputs under a per-model directory, named by the dataset source
    # so different configs don't overwrite each other.
    model_name = args.model_id.split("/")[-1].replace(".", "-")
    if args.dataset_path:
        source_tag = os.path.splitext(os.path.basename(args.dataset_path))[0]
    else:
        source_tag = args.config or args.split
    run_dir = os.path.join(args.eval_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, f"{source_tag}_LOG.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")

    rows = _load_dataset_rows(args)
    list_of_data_dicts, final_save_path = prepare_evaluation(
        rows,
        run_dir,
        run_name=source_tag,
        num_samples=args.num_samples_per_task,
    )

    system_prompt = (
        "You are an expert in mathematics. Your task is to convert informal, "
        "natural-language proofs into correct Lean 4 formalizations."
    )
    is_gemini = args.model_id.startswith("gemini-")

    if is_gemini:
        gemini_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            safety_settings=_SAFETY_OFF,
            temperature=0.6,
            top_p=0.95,
            max_output_tokens=8192,
        )
        model = _GeminiModel(args.model_id, config=gemini_config)
    else:
        llm = LLM(
            model=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        # Start from the model's own recommended sampling params (temperature,
        # top_p, top_k, ... from its generation_config.json) so thinking-mode
        # models don't loop to max_tokens; only override n and max_tokens.
        sampling_params = llm.get_default_sampling_params()
        sampling_params.n = args.num_samples_per_task
        sampling_params.max_tokens = 12000

    if args.limit > 0:
        list_of_data_dicts = list_of_data_dicts[:args.limit]

    data_num = 0
    for data_item in tqdm(list_of_data_dicts):
        data_num += 1
        if data_item["INFERENCE_DONE"] == "yes":
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"==================NUM{data_num}==================\n\n")
            continue

        informal_statement = str(data_item["informal_statement"]).strip()
        informal_proof = str(data_item["informal_proof"]).strip()

        input_text = wrap_prompt_in_query(informal_statement, informal_proof)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"==================NUM{data_num}==================\n\n")
            f.write("informal_statement: \n" + informal_statement + "\n")
            f.write("informal_proof: \n" + informal_proof + "\n")

        if is_gemini:

            @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
            def call_gemini(trial_idx):
                response = model.generate_content(input_text)
                if not response.text:
                    raise ValueError("No output returned")
                return response.text

            with ThreadPoolExecutor(max_workers=min(args.num_samples_per_task, 8)) as executor:
                model_outs = list(executor.map(call_gemini, range(args.num_samples_per_task)))
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
            outputs = llm.chat(messages, sampling_params, use_tqdm=False)
            model_outs = [o.text for o in outputs[0].outputs]

        assert len(model_outs) == args.num_samples_per_task

        responses = []
        for model_out_idx, model_out in enumerate(model_outs):
            if model_out is not None:
                parsed_model_out = extract_fl_proof(model_out)
                if len(parsed_model_out.strip()) == 0:
                    parsed_model_out = "ERROR [No text within tags]"
            else:
                parsed_model_out = "ERROR [No output returned]"
            responses.append(parsed_model_out)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Generated output# {model_out_idx}/{args.num_samples_per_task}:\n")
                f.write(f"rawModelOut: {model_out}\n")
                f.write(f"parsedModelOut: {parsed_model_out}\n")

        for trial_num in range(1, args.num_samples_per_task + 1):
            data_item[f"LLM_Output#{trial_num}"] = responses[trial_num - 1]
        data_item["INFERENCE_DONE"] = "yes"
        save_jsonl(list_of_data_dicts, final_save_path)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("Wrote all LLM outputs!!\n")
            f.write("=====================================\n\n")

    print("Inference and saving complete.")


if __name__ == "__main__":
    inference_on_dataset(parse_args())
