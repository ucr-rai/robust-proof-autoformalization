import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from global_perturbation import run as run_global
from math500 import ensure_math500_dataset
from number_symbol_edit import run_number, run_symbol
from step_edit import run as run_step

load_dotenv()  # GOOGLE_API_KEY / HF_TOKEN from .env

# miniF2F source problems (name / informal_statement / informal_proof / formal_*)
# from ProofBridge — see README.md.
DATASETS = {
    "miniF2F": "./data/minif2f.jsonl",
    "MATH500": "./data/math500.jsonl",
}
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "RobustPABench")

def push(datasets, split, *, prefix=""):
    """Push each {config_name: Dataset} as a private config/split of RobustPABench.

    Local perturbations are namespaced with a ``local_`` config prefix to match
    the layout on the Hub (global perturbations use ``global_``).
    """
    for config_name, ds in datasets.items():
        config_name = f"{prefix}{config_name}"
        ds.push_to_hub(HF_DATASET_NAME, config_name=config_name, split=split, private=True)
    print(f"Pushed {[prefix + c for c in datasets]} to {HF_DATASET_NAME} (private)")


def resolve_dataset(split, explicit_dataset=None):
    if explicit_dataset:
        return explicit_dataset
    if split not in DATASETS:
        raise ValueError(f"Unknown split {split!r}; pass --dataset explicitly")
    path = Path(DATASETS[split])
    if split == "MATH500":
        ensure_math500_dataset(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=("global", "number", "symbol", "step"))
    parser.add_argument("--dataset", default=None,
                        help="Source JSONL. Defaults to data/minif2f.jsonl or data/math500.jsonl by --split.")
    parser.add_argument("--data-dir", default="./data/output",
                        help="Local dir for intermediate label/select/rewrite results (default: ./data/output)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("--split", default="miniF2F", choices=tuple(DATASETS),
                        help="HF split name for the push.")
    parser.add_argument("--style", choices=("original", "rephrase", "step", "all"), default="all",
                        help="Global perturbation style (only for --type global).")
    parser.add_argument("--variant-model", choices=("gemini", "qwen3", "all"), default="all",
                        help="Global rewrite model (only for --type global).")
    parser.add_argument("--qwen-model", default="qwen3",
                        help="OpenAI-compatible model id for Qwen global rewrites.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between global rewrite API calls.")
    parser.add_argument("--no-push", action="store_true",
                        help="Build local intermediate files but do not push to HuggingFace.")
    args = parser.parse_args()

    dataset = resolve_dataset(args.split, args.dataset)
    data_dir = args.data_dir

    if args.type == "global":
        datasets = run_global(
            dataset=dataset,
            data_dir=data_dir,
            style=args.style,
            model=args.variant_model,
            gemini_model=args.model,
            qwen_model=args.qwen_model,
            limit=args.limit,
            sleep_s=args.sleep,
        )
        prefix = ""
    elif args.type == "number":
        datasets = run_number(dataset=dataset, data_dir=data_dir, model=args.model, limit=args.limit)
        prefix = "local_"
    elif args.type == "symbol":
        datasets = run_symbol(dataset=dataset, data_dir=data_dir, model=args.model, limit=args.limit)
        prefix = "local_"
    else:
        datasets = run_step(dataset=dataset, data_dir=data_dir, model=args.model, limit=args.limit)
        prefix = "local_"

    if args.no_push:
        print(f"Built {list(datasets)} under {data_dir}; --no-push set, skipping Hub push.")
    else:
        push(datasets, args.split, prefix=prefix)


if __name__ == "__main__":
    main()
