# LLM Bias Pruning Reproduction

This workspace now contains a runnable reproduction harness for the experiment in [arXiv:2502.07771v1](https://arxiv.org/html/2502.07771v1).

The paper studies pruning-based mitigation of racial bias in Llama-3-8B-Instruct using:

* prompt-specific pruning on the paper's 10 purchase variations
* within-context leave-one-out pruning across those purchase variations
* cross-context pruning from Activities, Services, and Finance back onto Purchase prompts
* bias measured with Standardized Mean Difference
* utility measured with inlier ratio, plus an optional Earth Mover's Distance check

The main implementation is in [reproduce_experiment.py](reproduce_experiment.py). It includes the paper's name lists, prompt templates, scoring logic for neurons and attention heads, pruning masks, and evaluation loops.

It now also supports a smaller Llama-compatible smoke model, optional plot export, and a notebook workflow.

## Run

1. Install dependencies from [requirements.txt](requirements.txt).
2. Run a dry pass to confirm the prompt inventory:

```bash
python reproduce_experiment.py --dry-run
```

3. Run the full pipeline with access to the target model:

```bash
python reproduce_experiment.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --output-json results.json
```

4. For a fast local validation run, use the smaller built-in smoke-test model:

```bash
python reproduce_experiment.py --smoke-test --output-json smoke_results.json --plot-dir plots
```

If you do not have access to that model or enough GPU memory, the script still serves as a faithful reproduction scaffold that can be pointed at a smaller causal LM for smoke tests.

## Notes

* The paper's appendix lists the final prompt sets used for analysis, which are encoded directly in the script.
* Exact published numbers depend on the same model checkpoint, inference settings, and hardware used by the authors.
* The repository currently focuses on the experiment harness rather than the paper's plotting code.
* The smoke-test mode uses a much smaller model so you can verify the pipeline structure without needing the full 8B checkpoint.

## Kaggle GPU

Use this if you want to run the reproduction in a Kaggle notebook with a GPU accelerator.

1. Create a Kaggle Notebook and turn on GPU in the notebook settings. A T4 or L4 is enough for the smoke test; the full `Meta-Llama-3-8B-Instruct` run will usually need a much larger GPU than the free Kaggle default.
2. Upload this repository as a Kaggle dataset or copy the files into the notebook session so that `reproduce_experiment.py`, `requirements.txt`, and `reproduction_walkthrough.ipynb` are available under `/kaggle/working`.
3. Install dependencies in a notebook cell:

```python
!pip install -q -r /kaggle/working/requirements.txt
```

4. If you want the full model, add your Hugging Face token as a Kaggle secret and log in before loading the checkpoint:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

5. Run the smoke test first to confirm the pipeline and GPU path:

```python
!python /kaggle/working/reproduce_experiment.py --smoke-test --output-json /kaggle/working/smoke_results.json --plot-dir /kaggle/working/plots
```

If that is still too slow, run a quick validation pass:

```python
!python /kaggle/working/reproduce_experiment.py --smoke-test --limit-names 4 --limit-purchase-prompts 2 --skip-cross-context --max-new-tokens 12 --output-json /kaggle/working/smoke_fast.json --plot-dir /kaggle/working/plots_fast
```

6. If you have enough GPU memory and access to the model, run the full checkpoint:

```python
!python /kaggle/working/reproduce_experiment.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --output-json /kaggle/working/results.json --plot-dir /kaggle/working/plots
```

7. Download artifacts from `/kaggle/working/plots` and the JSON output files when the run finishes.

If the full model does not fit on the Kaggle GPU, keep the smoke-test run and use it to verify that the pruning, evaluation, and plotting flow is working end to end.