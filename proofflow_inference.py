"""Run ProofFlow autoformalization on a RobustPABench split.

This is an inference-backend adapter that mirrors ``llm_inference.py``: it reads
rows with ``informal_statement`` + ``informal_proof`` (from the HuggingFace Hub
via ``--config``/``--split``, or a local jsonl via ``--dataset_path``) and writes
the same result shape, namely ``LLM_Output#1..N`` plus ``INFERENCE_DONE``, to
``<eval_dir>/<model>/<config>_output.jsonl``.

The downstream TC/SC and FR/RR/OUR scripts can therefore consume ProofFlow
outputs without learning a new file format. ProofFlow-specific diagnostics are
stored in ``ProofFlow_Summary#i``, ``ProofFlow_StatePath#i``, and
``ProofFlow_Error#i``.

ProofFlow itself is not on PyPI: clone it into this repo
(``git clone https://github.com/Huawei-AI4Math/ProofFlow``) so the default
``--proofflow_home`` (``./ProofFlow``) resolves, or point ``--proofflow_home``
(or ``PROOFFLOW_HOME``) at a checkout elsewhere.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_PROOFFLOW_HOME = ROOT / "ProofFlow"
# ProofFlow can verify against a local Lean project instead of a running server.
# Reuse the mathlib4 that kimina-lean-server/setup.sh already builds.
DEFAULT_LEAN_PROJECT = ROOT / "kimina-lean-server" / "mathlib4"
_THREAD_LOCAL = threading.local()

load_dotenv()


def _ensure_proofflow_on_path(proofflow_home: Path) -> None:
    if not proofflow_home.exists():
        raise FileNotFoundError(
            f"ProofFlow home not found: {proofflow_home}. "
            "Pass --proofflow_home or set PROOFFLOW_HOME."
        )
    if str(proofflow_home) not in sys.path:
        sys.path.insert(0, str(proofflow_home))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_jsonl_locked(rows: list[dict[str, Any]], path: Path,
                      row_indices: list[int] | None = None) -> None:
    """Safely update a shared output file from multiple shard workers."""
    with file_lock(path):
        if row_indices is not None and path.exists():
            latest = load_jsonl(path)
            if len(latest) != len(rows):
                raise ValueError(
                    f"Cannot merge shard output: {path} has {len(latest)} rows, "
                    f"current dataset has {len(rows)} rows"
                )
            for idx in row_indices:
                latest[idx] = rows[idx]
            save_jsonl(latest, path)
        else:
            save_jsonl(rows, path)


def load_source_rows(args) -> list[dict[str, Any]]:
    """Load benchmark rows from a local jsonl or the HuggingFace Hub."""
    if args.dataset_path:
        print(f"[proofflow] loading local dataset {args.dataset_path}")
        return load_jsonl(Path(args.dataset_path))
    print(f"[proofflow] loading {args.repo_id} :: {args.config} [{args.split}]")
    ds = load_dataset(args.repo_id, args.config, split=args.split)
    return [dict(r) for r in ds]


def init_output_rows(source_rows: list[dict[str, Any]], output_path: Path,
                     num_samples: int, resume: bool) -> list[dict[str, Any]]:
    with file_lock(output_path):
        if resume and output_path.exists():
            return load_jsonl(output_path)

        rows = source_rows
        for row in rows:
            row["INFERENCE_DONE"] = "no"
            for i in range(1, num_samples + 1):
                row[f"LLM_Output#{i}"] = ""
                row[f"ProofFlow_Summary#{i}"] = {}
                row[f"ProofFlow_StatePath#{i}"] = ""
                row[f"ProofFlow_Error#{i}"] = ""
        save_jsonl(rows, output_path)
        return rows


def reset_proofflow_outputs(row: dict[str, Any], num_samples: int) -> None:
    row["INFERENCE_DONE"] = "no"
    for i in range(1, num_samples + 1):
        row[f"LLM_Output#{i}"] = ""
        row[f"ProofFlow_Summary#{i}"] = {}
        row[f"ProofFlow_StatePath#{i}"] = ""
        row[f"ProofFlow_Error#{i}"] = ""


def make_model_manager(model: str, base_url: str, api_key: str,
                       prompt_path: Path | None,
                       max_new_tokens: int | None = None,
                       temperature: float | None = None,
                       top_p: float | None = None):
    from proofflow import LLMManager

    model_info = {"model": model}
    if max_new_tokens is not None:
        model_info["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        model_info["temperature"] = temperature
    if top_p is not None:
        model_info["top_p"] = top_p

    if not base_url:
        return LLMManager(
            model_info=model_info,
            system_prompt_path=str(prompt_path) if prompt_path else None,
        )

    model_info.update({
        "api_key": api_key,
        "base_url": base_url,
    })
    return LLMManager(
        model_info=model_info,
        system_prompt_path=str(prompt_path) if prompt_path else None,
    )


def make_lean_server(args):
    from proofflow import LeanServer

    if args.light_lean:
        os.environ["PROOFFLOW_LIGHT_LEAN"] = "1"

    if args.lean_server:
        return LeanServer(api_url=args.lean_server)
    if args.lean_project:
        if not Path(args.lean_project).exists():
            raise FileNotFoundError(
                f"Lean project not found: {args.lean_project}. Run "
                "`bash setup.sh` in kimina-lean-server to build mathlib4, or pass "
                "--lean_project / --lean_server."
            )
        return LeanServer(project_path=args.lean_project)
    raise ValueError("Pass --lean_server or --lean_project. For smoke tests, "
                     "use --light_lean with a tiny local lake project.")


def make_runtime(args, prompt_dir: Path):
    """Create per-worker ProofFlow clients.

    LLMManager wraps an httpx/OpenAI client and LeanServer may hold a persistent
    REPL process, so we keep one runtime per worker thread rather than sharing
    those mutable objects across concurrent examples.
    """
    managers = (
        make_model_manager(args.graph_model, args.graph_base_url, args.graph_api_key,
                           prompt_dir / "proof_graph.md",
                           args.graph_max_new_tokens, args.graph_temperature, args.graph_top_p),
        make_model_manager(args.formalize_model, args.formalize_base_url, args.formalize_api_key,
                           prompt_dir / args.formalize_prompt,
                           args.formalize_max_new_tokens, args.formalize_temperature, args.formalize_top_p),
        make_model_manager(args.solver_model, args.solver_base_url, args.solver_api_key,
                           prompt_dir / args.solver_prompt,
                           args.solver_max_new_tokens, args.solver_temperature, args.solver_top_p),
    )
    return managers, make_lean_server(args)


def get_thread_runtime(args, prompt_dir: Path):
    runtime = getattr(_THREAD_LOCAL, "proofflow_runtime", None)
    if runtime is None:
        runtime = make_runtime(args, prompt_dir)
        _THREAD_LOCAL.proofflow_runtime = runtime
    return runtime


def get_verified_lean_code(pf) -> str:
    """Export only Lean blocks that ProofFlow actually verified.

    ProofFlow's upstream `get_lean_code()` falls back to statement-only
    formalizations when a proof is unavailable. That is useful for visualization,
    but it can leak `sorry` blocks into our downstream TC/SC evaluation. For model
    comparison, treat the output as successful only when the final theorem node
    has a verified solved proof.
    """
    from proofflow.proof_graph import TheoremStatement
    from proofflow.utils import remove_imports

    header = "" if os.getenv("PROOFFLOW_LIGHT_LEAN") == "1" else (
        "import Mathlib\n"
        "import Aesop\n"
        "set_option maxHeartbeats 0\n"
        "open BigOperators Real Nat Topology Rat Filter"
    )
    blocks = []
    has_verified_final = False

    for item in pf.proof_items or []:
        solved = getattr(item, "solved_lemma", None) or {}
        if not solved.get("lean_verify"):
            continue
        lean_code = solved.get("lean_code")
        if not lean_code:
            continue
        cleaned = remove_imports(lean_code)
        if cleaned:
            blocks.append(cleaned)
        if isinstance(item, TheoremStatement):
            has_verified_final = True

    if not blocks or not has_verified_final:
        return "ERROR [ProofFlow produced no verified final theorem]"
    return (header + "\n\n" if header else "") + "\n\n".join(blocks)


def run_one_proofflow(nl_text: str, args, managers, lean_server, state_path: Path):
    from proofflow import ProofFlow

    graph_model, formalize_model, solver_model = managers
    pf = ProofFlow(
        lean_server=lean_server,
        graph_model_manager=graph_model,
        formalize_model_manager=formalize_model,
        solver_model_manager=solver_model,
        verbose=args.verbose,
    )
    pf.autoformalize_series(
        nl_text,
        graph_builder_retries=args.graph_retries,
        formalizer_retries=args.formalizer_retries,
        prover_retries=args.prover_retries,
        follow_dag=not args.no_follow_dag,
        previous_context=not args.no_previous_context,
        supply_proof=not args.no_supply_proof,
    )
    lean_code = get_verified_lean_code(pf)
    summary = pf.summary(verbose=False, pass_at=args.pass_at)
    if args.save_state:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pf.save(str(state_path))
    return lean_code, summary


def build_nl_text(row: dict[str, Any]) -> str:
    stmt = str(row.get("informal_statement", "") or "").strip()
    proof = str(row.get("informal_proof", "") or "").strip()
    return f"Theorem: {stmt}\n\nProof: {proof}\n"


def run_selected_row(row_idx: int, row: dict[str, Any], args, prompt_dir: Path,
                     state_dir: Path) -> tuple[int, dict[str, Any], list[str]]:
    """Run all samples for one dataset row and return the updated row."""
    managers, lean_server = get_thread_runtime(args, prompt_dir)
    name = row.get("name", f"row_{row_idx}")
    log_lines = [f"\n--- [{row_idx}] {name} ---\n"]
    nl_text = build_nl_text(row)

    for sample_idx in range(1, args.num_samples_per_task + 1):
        state_path = state_dir / f"{name}__sample{sample_idx}.pickle"
        try:
            lean_code, summary = run_one_proofflow(
                nl_text, args, managers, lean_server, state_path
            )
            row[f"LLM_Output#{sample_idx}"] = lean_code or "ERROR [ProofFlow produced empty Lean code]"
            row[f"ProofFlow_Summary#{sample_idx}"] = summary
            row[f"ProofFlow_StatePath#{sample_idx}"] = str(state_path) if args.save_state else ""
            row[f"ProofFlow_Error#{sample_idx}"] = ""
            log_lines.append(f"sample#{sample_idx}: OK {summary}\n")
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"
            row[f"LLM_Output#{sample_idx}"] = f"ERROR [ProofFlow failed: {err}]"
            row[f"ProofFlow_Summary#{sample_idx}"] = {}
            row[f"ProofFlow_StatePath#{sample_idx}"] = ""
            row[f"ProofFlow_Error#{sample_idx}"] = err
            log_lines.append(f"sample#{sample_idx}: ERROR {err}\n")
            log_lines.append(traceback.format_exc() + "\n")

    row["INFERENCE_DONE"] = "yes"
    return row_idx, row, log_lines


def parse_args():
    p = argparse.ArgumentParser(description="Run ProofFlow inference on RobustPABench")
    p.add_argument("--repo_id", default="ucr-rai/RobustPABench", help="HF dataset repo")
    p.add_argument("--config", default=None, help="HF config, e.g. local_number_edit_statement")
    p.add_argument("--split", default="miniF2F", help="HF split to run inference on")
    p.add_argument("--dataset_path", default=None,
                   help="Local dataset jsonl (overrides --repo_id/--config if set)")
    p.add_argument("--eval_dir", default="results/miniF2F", help="Directory to store results")
    p.add_argument("--model_name", default="ProofFlow-Gemini",
                   help="Name used for the per-model output subdirectory")
    p.add_argument("--num_samples_per_task", type=int, default=1)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", "--no-resume", dest="resume", action="store_false")
    p.add_argument("--force_rerun", "--force-rerun", action="store_true",
                   default=os.getenv("PROOFFLOW_FORCE_RERUN", "0") == "1",
                   help="rerun the selected shard rows even if an output file already marks them done")

    p.add_argument("--proofflow_home", default=os.getenv("PROOFFLOW_HOME", str(DEFAULT_PROOFFLOW_HOME)))
    p.add_argument("--openai_base_url",
                   default=os.getenv("PROOFFLOW_OPENAI_BASE_URL",
                                     "https://generativelanguage.googleapis.com/v1beta/openai/"))
    p.add_argument("--api_key", default=os.getenv("PROOFFLOW_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
    p.add_argument("--graph_model", default=os.getenv("PROOFFLOW_GRAPH_MODEL", "gemini-2.5-pro"))
    p.add_argument("--formalize_model", default=os.getenv("PROOFFLOW_FORMALIZE_MODEL", "gemini-2.5-flash"))
    p.add_argument("--solver_model", default=os.getenv("PROOFFLOW_SOLVER_MODEL", "gemini-2.5-flash"))
    p.add_argument("--formalize_prompt",
                   default=os.getenv("PROOFFLOW_FORMALIZE_PROMPT", "lemma_formalizer.md"))
    p.add_argument("--solver_prompt",
                   default=os.getenv("PROOFFLOW_SOLVER_PROMPT", "lemma_prover.md"))
    p.add_argument("--graph_base_url", default=os.getenv("PROOFFLOW_GRAPH_BASE_URL", ""))
    p.add_argument("--formalize_base_url", default=os.getenv("PROOFFLOW_FORMALIZE_BASE_URL", ""))
    p.add_argument("--solver_base_url", default=os.getenv("PROOFFLOW_SOLVER_BASE_URL", ""))
    p.add_argument("--graph_api_key", default=os.getenv("PROOFFLOW_GRAPH_API_KEY", ""))
    p.add_argument("--formalize_api_key", default=os.getenv("PROOFFLOW_FORMALIZE_API_KEY", ""))
    p.add_argument("--solver_api_key", default=os.getenv("PROOFFLOW_SOLVER_API_KEY", ""))
    p.add_argument("--graph_max_new_tokens", type=int,
                   default=int(os.getenv("PROOFFLOW_GRAPH_MAX_NEW_TOKENS",
                                         os.getenv("PROOFFLOW_MAX_NEW_TOKENS", "16384"))))
    p.add_argument("--formalize_max_new_tokens", type=int,
                   default=int(os.getenv("PROOFFLOW_FORMALIZE_MAX_NEW_TOKENS", "16384")))
    p.add_argument("--solver_max_new_tokens", type=int,
                   default=int(os.getenv("PROOFFLOW_SOLVER_MAX_NEW_TOKENS", "16384")))
    p.add_argument("--graph_temperature", type=float,
                   default=float(os.getenv("PROOFFLOW_GRAPH_TEMPERATURE",
                                           os.getenv("PROOFFLOW_TEMPERATURE", "0.9"))))
    p.add_argument("--formalize_temperature", type=float,
                   default=float(os.getenv("PROOFFLOW_FORMALIZE_TEMPERATURE",
                                           os.getenv("PROOFFLOW_TEMPERATURE", "0.9"))))
    p.add_argument("--solver_temperature", type=float,
                   default=float(os.getenv("PROOFFLOW_SOLVER_TEMPERATURE",
                                           os.getenv("PROOFFLOW_TEMPERATURE", "0.9"))))
    p.add_argument("--graph_top_p", type=float,
                   default=float(os.getenv("PROOFFLOW_GRAPH_TOP_P",
                                           os.getenv("PROOFFLOW_TOP_P", "0.95"))))
    p.add_argument("--formalize_top_p", type=float,
                   default=float(os.getenv("PROOFFLOW_FORMALIZE_TOP_P",
                                           os.getenv("PROOFFLOW_TOP_P", "0.95"))))
    p.add_argument("--solver_top_p", type=float,
                   default=float(os.getenv("PROOFFLOW_SOLVER_TOP_P",
                                           os.getenv("PROOFFLOW_TOP_P", "0.95"))))

    p.add_argument("--lean_server", default=os.getenv("LEAN_SERVER_URL", ""),
                   help="URL of a running Lean server (takes priority over --lean_project)")
    p.add_argument("--lean_project",
                   default=os.getenv("PROOFFLOW_LEAN_PROJECT", str(DEFAULT_LEAN_PROJECT)),
                   help="Local Lean project with Mathlib built "
                        "(default: kimina-lean-server/mathlib4)")
    p.add_argument("--light_lean", action="store_true",
                   help="Smoke-test mode: strip Mathlib/Aesop imports. Not for paper runs.")

    p.add_argument("--graph_retries", type=int,
                   default=int(os.getenv("PROOFFLOW_GRAPH_RETRIES", "3")))
    p.add_argument("--formalizer_retries", type=int,
                   default=int(os.getenv("PROOFFLOW_FORMALIZER_RETRIES", "5")))
    p.add_argument("--prover_retries", type=int,
                   default=int(os.getenv("PROOFFLOW_PROVER_RETRIES", "5")))
    p.add_argument("--pass_at", type=int,
                   default=int(os.getenv("PROOFFLOW_PASS_AT", "5")))
    p.add_argument("--no_follow_dag", action="store_true")
    p.add_argument("--no_previous_context", action="store_true")
    p.add_argument("--no_supply_proof", action="store_true")

    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save_every", type=int,
                   default=int(os.getenv("PROOFFLOW_SAVE_EVERY", "1")))
    p.add_argument("--workers", type=int,
                   default=int(os.getenv("PROOFFLOW_EXAMPLE_WORKERS", "1")),
                   help="number of examples to process concurrently in this process")
    p.add_argument("--save_state", action="store_true",
                   default=os.getenv("PROOFFLOW_SAVE_STATE", "1") == "1")
    p.add_argument("--no_save_state", "--no-save-state",
                   dest="save_state", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.graph_base_url = args.graph_base_url or args.openai_base_url
    args.formalize_base_url = args.formalize_base_url or args.openai_base_url
    args.solver_base_url = args.solver_base_url or args.openai_base_url
    args.graph_api_key = args.graph_api_key or args.api_key
    args.formalize_api_key = args.formalize_api_key or args.api_key
    args.solver_api_key = args.solver_api_key or args.api_key or "EMPTY"
    if (args.graph_base_url and not args.graph_api_key) or (args.formalize_base_url and not args.formalize_api_key):
        sys.exit("Set GOOGLE_API_KEY/PROOFFLOW_API_KEY, or pass stage-specific *_api_key values")

    proofflow_home = Path(args.proofflow_home).resolve()
    _ensure_proofflow_on_path(proofflow_home)

    prompt_dir = proofflow_home / "prompts"

    # Organize outputs like llm_inference.py: <eval_dir>/<model>/<source>_*.
    if args.dataset_path:
        source_tag = Path(args.dataset_path).stem
    else:
        source_tag = args.config or args.split
    run_dir = Path(args.eval_dir) / args.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{source_tag}_output.jsonl"
    log_path = run_dir / f"{source_tag}_LOG.txt"
    state_dir = run_dir / "proofflow_states" / source_tag

    source_rows = load_source_rows(args)
    rows = init_output_rows(source_rows, output_path, args.num_samples_per_task, args.resume)
    selected = list(enumerate(rows))
    if args.limit > 0:
        selected = selected[:args.limit]
    if args.force_rerun:
        for _, row in selected:
            reset_proofflow_outputs(row, args.num_samples_per_task)

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== ProofFlow inference start {time.ctime()} ===\n")
        log.write(f"source={source_tag} output={output_path}\n")
        log.write(f"rows={len(selected)}\n")
        log.write(f"resume={args.resume} force_rerun={args.force_rerun}\n")
        log.write(f"workers={args.workers}\n")
        log.write(f"models graph={args.graph_model} formalize={args.formalize_model} solver={args.solver_model}\n")
        log.write(f"base_urls graph={args.graph_base_url} formalize={args.formalize_base_url} solver={args.solver_base_url}\n")

    pending = [
        (row_idx, row) for row_idx, row in selected
        if row.get("INFERENCE_DONE") != "yes"
    ]
    if args.workers <= 1:
        for row_idx, row in pending:
            row_idx, updated_row, log_lines = run_selected_row(
                row_idx, row, args, prompt_dir, state_dir
            )
            rows[row_idx] = updated_row
            with open(log_path, "a", encoding="utf-8") as log:
                log.writelines(log_lines)
            save_jsonl_locked(rows, output_path, [row_idx])
    else:
        log_lock = threading.Lock()
        max_workers = min(args.workers, len(pending)) if pending else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_selected_row, row_idx, row, args, prompt_dir, state_dir
                ): row_idx
                for row_idx, row in pending
            }
            for future in as_completed(futures):
                row_idx = futures[future]
                try:
                    row_idx, updated_row, log_lines = future.result()
                except BaseException as e:
                    err = f"{type(e).__name__}: {e}"
                    rows[row_idx]["LLM_Output#1"] = f"ERROR [ProofFlow worker failed: {err}]"
                    rows[row_idx]["ProofFlow_Error#1"] = err
                    rows[row_idx]["INFERENCE_DONE"] = "yes"
                    log_lines = [
                        f"\n--- [{row_idx}] {rows[row_idx].get('name', f'row_{row_idx}')} ---\n",
                        f"worker: ERROR {err}\n",
                        traceback.format_exc() + "\n",
                    ]
                else:
                    rows[row_idx] = updated_row
                with log_lock:
                    with open(log_path, "a", encoding="utf-8") as log:
                        log.writelines(log_lines)
                save_jsonl_locked(rows, output_path, [row_idx])

    save_jsonl_locked(rows, output_path, [idx for idx, _ in selected])
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== ProofFlow inference done {time.ctime()} ===\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
