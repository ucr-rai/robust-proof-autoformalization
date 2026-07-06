"""Build the MATH-500 source split used by RobustPABench.

The local edit builders expect the same row shape as ``data/minif2f.jsonl``:
``name``, ``id``, ``split``, ``informal_statement``, ``informal_proof``,
``formal_statement``, and ``formal_proof``.  MATH-500 does not ship Lean
signatures, so the formal statement is intentionally empty and the proof is a
placeholder.

Usage:
    python math500.py --output data/math500.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_REPO_ID = "HuggingFaceH4/MATH-500"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT = Path("data/math500.jsonl")


def _name_from_unique_id(unique_id: str, fallback_idx: int) -> str:
    """Convert ``test/precalculus/807.json`` to ``test_precalculus_807``."""
    if not unique_id:
        return f"math500_{fallback_idx:04d}"
    stem = unique_id.rsplit(".", 1)[0]
    return stem.replace("/", "_").replace("-", "_")


def _answer_clause(answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return ""
    if answer.startswith("$") and answer.endswith("$"):
        return f"Show that it is {answer}."
    return f"Show that it is ${answer}$."


def build_math500_rows(repo_id: str = DEFAULT_REPO_ID, split: str = DEFAULT_SPLIT) -> list[dict]:
    """Return MATH-500 rows in the RobustPABench source schema."""
    ds = load_dataset(repo_id, split=split)
    rows = []
    for i, row in enumerate(ds):
        name = _name_from_unique_id(row.get("unique_id", ""), i)
        problem = str(row.get("problem", "")).strip()
        clause = _answer_clause(row.get("answer", ""))
        statement_body = problem if not clause else f"{problem}\n{clause}"
        rows.append(
            {
                "name": name,
                "id": name,
                "split": "test",
                "informal_statement": f"/-- {statement_body}-/",
                "informal_proof": str(row.get("solution", "")).strip(),
                "formal_statement": "",
                "formal_proof": "sorry",
            }
        )
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_math500_dataset(path: str | Path = DEFAULT_OUTPUT, *, repo_id: str = DEFAULT_REPO_ID,
                           split: str = DEFAULT_SPLIT) -> Path:
    """Create ``path`` if missing and return it."""
    path = Path(path)
    if not path.exists():
        rows = build_math500_rows(repo_id=repo_id, split=split)
        write_jsonl(path, rows)
        print(f"Wrote {len(rows)} MATH-500 rows -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--hf-split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rows = build_math500_rows(repo_id=args.repo_id, split=args.hf_split)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} MATH-500 rows -> {args.output}")


if __name__ == "__main__":
    main()
