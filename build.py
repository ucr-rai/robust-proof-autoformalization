import argparse

from dotenv import load_dotenv

from number_symbol_edit import run_number, run_symbol
from step_edit import run as run_step

load_dotenv()  # GOOGLE_API_KEY / HF_TOKEN from .env

# miniF2F source problems (name / informal_statement / informal_proof / formal_*)
# from ProofBridge — see README.md.
DEFAULT_DATASET = "./data/minif2f.jsonl"
HF_DATASET_NAME = "RobustPABench"

def push(datasets, split):
    """Push each {config_name: Dataset} as a private config/split of RobustPABench.

    Local perturbations are namespaced with a ``local_`` config prefix to match
    the layout on the Hub (global perturbations use ``global_``).
    """
    for config_name, ds in datasets.items():
        config_name = f"local_{config_name}"
        ds.push_to_hub(HF_DATASET_NAME, config_name=config_name, split=split, private=True)
    print(f"Pushed {['local_' + c for c in datasets]} to {HF_DATASET_NAME} (private)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=("number", "symbol", "step"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", default="./data/output",
                        help="Local dir for intermediate label/select results (default: ./data/output)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("--split", default="miniF2F", help="HF split name for the push.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.type == "number":
        datasets = run_number(dataset=args.dataset, data_dir=data_dir, model=args.model, limit=args.limit)
    elif args.type == "symbol":
        datasets = run_symbol(dataset=args.dataset, data_dir=data_dir, model=args.model, limit=args.limit)
    else:
        datasets = run_step(dataset=args.dataset, data_dir=data_dir, model=args.model, limit=args.limit)

    push(datasets, args.split)


if __name__ == "__main__":
    main()
