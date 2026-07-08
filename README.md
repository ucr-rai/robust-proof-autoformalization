# RobustPABench: Evaluating the Robustness of Proof Autoformalization in Lean 4

[![arXiv](https://img.shields.io/badge/arXiv-2606.14867-b31b1b.svg)](https://arxiv.org/pdf/2606.14867)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RobustPABench-yellow)](https://huggingface.co/datasets/ucr-rai/RobustPABench)

Code for paper: [Evaluating the Robustness of Proof Autoformalization in Lean 4](https://arxiv.org/abs/2606.14867), by Zhengtao Gui, Sheng Yang, and Zhouxing Shi.
In the 3rd AI for Math Workshop at ICML 2026.

This work evaluates the robustness of proof autoformalization in [Lean 4](https://lean-lang.org/).
Proof autoformalization aims to translate a mathematical informal proof written in natural language into a formal proof in a formal language such as Lean 4.
A robust proof autoformalizer must remain faithful even for informal proofs that diverge from idealized ones, and we present the first study on it.

We formulate two categories of perturbations and evaluate robustness under each:
* A global perturbation paraphrases the informal proof in a different style, under which the formalization should remain consistent;
* A local perturbation alters a value, symbol, or proof step, possibly in a counterfactual way, and a robust formalization should faithfully reflect the perturbation rather than reverting to the original one or inferring a different one on its own.

We built a benchmark, [RobustPABench](https://huggingface.co/datasets/ucr-rai/RobustPABench), based on [miniF2F](https://github.com/openai/miniF2F) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and using the perturbations we proposed, which reveals the weakness of existing proof autoformalizers such as [ProofBridge](https://github.com/PrithwishJana/ProofBridge) and [ProofFlow](https://arxiv.org/abs/2510.15981).


Citation:
```bibtex
@article{gui2026evaluating,
  title={Evaluating the Robustness of Proof Autoformalization in Lean 4},
  author={Gui, Zhengtao and Yang, Sheng and Shi, Zhouxing},
  journal={arXiv preprint arXiv:2606.14867},
  year={2026}
}
```

## Setup

* Python Dependencies:
  * First, install **PyTorch** and **vLLM** compatible with your system (OS, hardware, and CUDA version).
  * Then, install the remaining python packages:
    ```bash
    pip install -r requirements.txt
    ```
* Provide a Gemini API key in the environment variable `GOOGLE_API_KEY`.
* Login to HuggingFace by `huggingface-cli login` or provide a token in the environment variable `HF_TOKEN`.
* Set up the Lean server, included as the [`kimina-lean-server`](https://github.com/ucr-rai/kimina-lean-server) submodule:

  ```bash
  git submodule update --init --recursive
  cd kimina-lean-server
  bash setup.sh
  ```

## Evaluation

To run evaluation with the benchmark ([RobustPABench](https://huggingface.co/datasets/ucr-rai/RobustPABench)) we have constructed, run inference with a model and then run scoring.

### Inference

We provide two inference backends. Both load a benchmark split from the
[RobustPABench](https://huggingface.co/datasets/ucr-rai/RobustPABench) Hub repo
(`--config`/`--split`) and write results to
`<eval_dir>/<model>/<config>_output.jsonl`, which [`evaluate.py`](evaluate.py)
then scores.

<details>
<summary>See the available <code>--config</code> (perturbation type) and <code>--split</code> values.</summary>

| Perturbation | `--config` | Available `--split` |
| --- | --- | --- |
| Original (unperturbed) | `global_original` | `miniF2F`, `MATH500` |
| Global (Gemini rephrase) | `global_gemini_rephrase` | `miniF2F`, `MATH500` |
| Global (Gemini step) | `global_gemini_step` | `miniF2F`, `MATH500` |
| Global (Qwen3 rephrase) | `global_qwen3_rephrase` | `miniF2F`, `MATH500` |
| Global (Qwen3 step) | `global_qwen3_step` | `miniF2F`, `MATH500` |
| Local (number, statement) | `local_number_edit_statement` | `miniF2F`, `MATH500` |
| Local (number, proof) | `local_number_edit_proof` | `miniF2F`, `MATH500` |
| Local (symbol, statement) | `local_symbol_edit_statement` | `miniF2F`, `MATH500` |
| Local (symbol, proof) | `local_symbol_edit_proof` | `miniF2F`, `MATH500` |
| Local (step delete) | `local_step_delete` | `miniF2F`, `MATH500` |

</details>

#### Single model

To use a HuggingFace model:
```bash
python llm_inference.py \
    --model_id <hf_model_id> \
    --config <perturbation_config> \
    --split <miniF2F|MATH500> \
    --eval_dir <output_dir>
```

For example:
```bash
python llm_inference.py \
    --model_id ucr-rai/ProofBridge-SFT-only \
    --config local_number_edit_statement \
    --split miniF2F \
    --eval_dir results/miniF2F
```

#### ProofFlow

[ProofFlow](https://github.com/Huawei-AI4Math/ProofFlow) is a pipelined proof autoformalizer.

First clone it into this repo and install its dependencies:
```bash
git clone https://github.com/Huawei-AI4Math/ProofFlow
pip install -r ProofFlow/requirements.txt
```

To run inference with ProofFlow:
```bash
python proofflow_inference.py \
    --config <perturbation_config> \
    --split <miniF2F|MATH500> \
    --eval_dir <output_dir>
```

For example:
```bash
python proofflow_inference.py \
    --config local_number_edit_statement \
    --split miniF2F \
    --eval_dir results/miniF2F
```

### Scoring

Type-check the generated Lean and run the semantic-consistency judge:

```bash
python evaluate.py tcsc \
    --input results/miniF2F/ProofBridge-SFT-only/local_number_edit_statement_output.jsonl
```

Score edit-faithfulness (FR / RR / OUR) for a perturbation type:

```bash
python evaluate.py edit \
    --config local_number_edit_statement \
    --split miniF2F \
    --output_jsonl results/miniF2F/ProofBridge-SFT-only/local_number_edit_statement_output.jsonl
```

Scored/summary files are written next to the input (`_output` → `_scored` / `_summary`).

## Constructing the benchmark

By default, the generated dataset will be pushed to a HuggingFace repo
`<user>/RobustPABench` with different splits that correspond to the perturbation types.

### Data Source

* miniF2F: We use [miniF2F-Test-PF](https://github.com/PrithwishJana/ProofBridge/blob/bb0f247a62077a273058daa966472f0f109c92f2/datasets_validation/minif2f/dataset.jsonl) processed by [ProofBridge](https://arxiv.org/pdf/2510.15681), which includes NL proofs. The dataset has been attached at [`data/minif2f.jsonl`](data/minif2f.jsonl).
* MATH-500: We use [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500). To build the local source JSONL in the same schema as miniF2F, run:
  ```bash
  python math500.py --output data/math500.jsonl
  ```
  `build.py --split MATH500` will also create this file automatically if it is missing.

### Pipeline for global perturbations

To construct the original split plus the four meaning-preserving global variants:
```bash
python build.py --type global --split miniF2F --style all --variant-model all
python build.py --type global --split MATH500 --style all --variant-model all
```

`--style original` creates only `global_original` and does not call any model API.
`--variant-model gemini` uses Gemini through `GOOGLE_API_KEY`; `--variant-model qwen3`
uses an OpenAI-compatible endpoint through `NRP_LLM_KEY` and optional
`NRP_LLM_BASE_URL`.

### Pipeline for local perturbations

To construct local perturbations, run:
```bash
python build.py --type number --split miniF2F
python build.py --type symbol --split miniF2F
python build.py --type step   --split miniF2F
python build.py --type number --split MATH500
python build.py --type symbol --split MATH500
python build.py --type step   --split MATH500
```

For dry runs that only write local intermediate JSONL files, pass `--no-push`.
