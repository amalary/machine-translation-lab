# Machine Translation Lab

A notebook-based learning project for English-to-French machine translation with Hugging Face Transformers, OPUS Books, and T5-small. It covers inspecting parallel text, tokenization, a small fine-tuning experiment, translation evaluation, and pretrained inference.

This is an exploratory lab, not a deployed translation service. The notebooks contain implemented experiments alongside TODO comments and learning notes. The small training subsets and hand-written examples demonstrate the workflow; they do not establish production translation quality.

## What's in the project

| Notebook | What it does |
|---|---|
| [01 Dataset Exploration](notebooks/01_dataset_exploration.ipynb) | Loads English/French book text, creates a train/test split, and checks empty pairs, duplicates, and overlap. |
| [02 Tokenization](notebooks/02_tokenization.ipynb) | Prepares source and target tokens with the T5 translation prefix and inspects padding, truncation, attention masks, and labels. |
| [03 Model Evaluation](notebooks/03_model_evaluation.ipynb) | Configures a small CPU fine-tuning run, compares validation metrics before and after training, and explores translation errors and retail examples. |
| [04 Model Inference](notebooks/04_model_inference.ipynb) | Generates translations with pretrained T5-small and compares sampling, greedy decoding, and beam search. |

The dataset is `Helsinki-NLP/opus_books`, configuration `en-fr`. The starting checkpoint is `google-t5/t5-small`.

Other project files:

- `requirements.txt`: pinned dependencies for Windows x64 and Python 3.12, including a CPU-only PyTorch wheel.
- `tests/test_environment.py`: environment, notebook schema, and kernel smoke tests.
- `.vscode/`: project interpreter and editor settings.
- `src/`: a placeholder for future reusable Python utilities; the experiments currently live in notebooks.
- `data/` and `outputs/`: local inputs and generated results. Their contents are ignored by Git apart from placeholder files.

## Set up locally

The checked-in environment configuration targets **Windows x64 with Python 3.12**. The requirements include a platform-specific PyTorch wheel, so they are not a portable installation recipe for other operating systems or Python versions.

Clone the repository and run these commands from its root in PowerShell. Python 3.12 and `uv` must already be installed:

```powershell
py -3.12 -m venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name machine-translation-lab --display-name 'Python (machine-translation-lab)'
.\.venv\Scripts\python.exe -m pip check
```

A short checkout path can help avoid Windows path-length errors during installation. The commands use the environment's Python directly, so activating the virtual environment is optional.

Open the project in VS Code and select `.venv\Scripts\python.exe` as the Python interpreter. In each notebook, select **Python (machine-translation-lab)** or the project `.venv` kernel. Alternatively, start JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Re-register the kernel if you move the project or recreate the environment; its kernelspec records an absolute interpreter path.

## Run the lab

Start with notebook 1 and work through the notebooks in order. Each notebook has its own variables, so run its setup and preparation cells before later sections. Dataset, tokenizer, and model-loading cells may download assets on their first run. The notebooks use the normal Hugging Face cache under the user profile rather than storing those downloads in the repository.

**Notebook 3 contains an active `trainer.train()` call.** Running all of its cells starts training, evaluates the model, and saves a checkpoint. Some notebook instructions still describe TODO-only cells or commented-out training; those notes are out of date. Inspect the code before running an entire notebook.

The training experiment uses a seeded 80/10/10 train/validation/test split, with up to 128 training examples and 32 validation examples for one CPU epoch. It reserves the test split for later assessment and saves the model and tokenizer under `outputs/t5-small-en-fr/final`.

Notebook 4 loads the original pretrained checkpoint by default. To inspect your fine-tuned model, change its `model_name` to the saved checkpoint folder. Its sampling example can produce different outputs between runs; the greedy and beam-search comparison disables sampling.

## Evaluation and limitations

The lab uses SacreBLEU together with source/reference/prediction tables and manual review. It also includes small retail examples to inspect quantities, units, brands, and policy meaning, with illustrative latency and review checks.

These experiments have several limits:

- Book prose differs from retail and other application domains. A book-domain score does not establish quality on those domains.
- A tiny fine-tuning run demonstrates training and evaluation mechanics; it is not evidence of a broadly improved model.
- BLEU and automatic text checks do not replace bilingual review. Valid alternative translations can score poorly, and meaning errors can pass simple checks.
- The hand-written examples and local timings are exploratory, not a representative benchmark or load test.
- Notebook consistency checks verify the experiment's bookkeeping, not deployment readiness.

## Verify the environment

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -s
```

The tests check package imports, tensor operations, device detection, the API signatures used by the lab, a tiny randomly initialized T5 model, local SacreBLEU, notebook schemas, VS Code interpreter configuration, and a live Jupyter kernel. They do not download pretrained weights or datasets, train a model, execute the full notebooks, or measure translation quality.

If imports or kernel startup fail, check that both the terminal and notebook use the project environment. The pinned PyTorch build uses CPU, so CUDA being unavailable is expected.
