# Machine Translation Lab

A local, beginner-friendly English-to-French learning workspace for your Walmart Translation Platform ML Scientist interview on **Friday, October 16, 2026**. You implement the exercises. The setup contains no completed translation pipeline and starts no training.

## Open in VS Code

From PowerShell:

```powershell
cd 'C:\Users\admin\Documents\Codex\2026-10-12\files-pasted-by-the-user-task\outputs\machine-translation-lab'
code .
```

The project uses **Python 3.12.10** in `.venv`. Python 3.11 was not installed; the existing 3.12 installation was used instead of changing global Python installations. Microsoft Python and Jupyter extensions were already installed.

`.vscode/settings.json` sets the project interpreter. If VS Code shows another interpreter, run **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`. The notebook kernel is a separate selection: choose **Python (machine-translation-lab)**, or **Python Environments > .venv (Python 3.12.10)** in the notebook's top-right kernel picker.

## Terminal and notebooks

Open **Terminal > New Terminal**, then activate if VS Code has not done so:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pytest tests -q -s
```

You can also avoid activation entirely:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -s
```

Open **`notebooks/01_dataset_exploration.ipynb` first**. Select the project kernel, add your own code to a TODO cell, and execute with **Shift+Enter**. Restart the kernel and use **Run All** to check your completed work. Initially every exercise cell is a comment, so running all cells performs no tutorial work.

For the browser Jupyter interface:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## Tutorial alignment

Primary reference: [Hugging Face translation tutorial](https://huggingface.co/docs/transformers/tasks/translation), checked October 12, 2026.

| Notebook | Focus | Tutorial connection |
|---|---|---|
| 01 Dataset Exploration | Pairs, splits, data quality | OPUS Books loading |
| 02 Tokenization | Source/target tokens, padding, truncation | Preprocessing and collation |
| 03 Evaluation | References, SacreBLEU, error analysis | Evaluation |
| 04 Model Inference | Pretrained baseline, generation, decoding | Inference APIs |

Use dataset **`Helsinki-NLP/opus_books`**, configuration **`en-fr`**, and checkpoint **`google-t5/t5-small`**. Dataset and pretrained weights are intentionally deferred until you implement loading. Public resources do not require a Hugging Face login.

Compatibility notes for the installed stack:

- Target tokenization uses `text_target`; generation uses `AutoModelForSeq2SeqLM.generate`.
- The current trainer interface uses `processing_class` and training arguments use `eval_strategy`.
- For a sequence-to-sequence collator, pass an actual model object or omit `model`; a checkpoint name is not a loaded model.
- The inference exercise uses pretrained T5-small. The tutorial's later inference section assumes you already fine-tuned a checkpoint, so its example output is not a baseline quality guarantee.
- Evaluate's SacreBLEU wrapper groups references by example. Direct SacreBLEU groups them by reference stream. Inspect the chosen API before scoring.
- Optional fine-tuning is deferred. For future CPU experiments, keep `fp16=False`, `bf16=False`, `push_to_hub=False`, and start with a small batch. Do not copy the tutorial's GPU and upload settings unchanged.

## Compute and caching

The environment uses the official **CPU build of PyTorch**. Windows detects an **NVIDIA RTX 500 Ada Generation Laptop GPU (4 GB)** with driver 596.58, but CUDA execution is not enabled or claimed verified in this CPU environment. CPU is sufficient for the initial small inference exercises. No CUDA toolkit or driver was changed. Consult [PyTorch's installation guide](https://pytorch.org/get-started/locally/) if you choose GPU setup later.

Hugging Face uses its normal user cache, `%USERPROFILE%\.cache\huggingface`. This shorter location avoids Windows path-length failures in Evaluate's hashed metric filenames. Do not set `HF_HOME` to this deeply nested project folder. Only small repository metadata, tokenizer/config files, and the SacreBLEU module were fetched during setup.

`data/` holds your local inputs; `outputs/` holds your future results. Their contents, `.venv`, and notebook checkpoints are ignored by Git. Keep credentials out of notebooks.

## Verification and reproducibility

`tests/test_environment.py` checks all requested imports, actual tensor operations, CPU/CUDA detection, the installed tutorial API signatures, a tiny randomly initialized T5 backend, local SacreBLEU, all four notebook schemas, interpreter configuration, and a live kernel process. It downloads no data or model weights and does not train. The tiny synthetic T5 check establishes backend functionality only.

`requirements.txt` pins the successfully installed packages, including transitive dependencies, for **Windows x64 / Python 3.12**. The CPU wheel is pinned to its official URL. To recreate this environment at a new location:

```powershell
py -3.12 -m venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name machine-translation-lab --display-name 'Python (machine-translation-lab)'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q -s
```

The existing `uv` installer was used to complete installation after pip encountered Windows' path-length limit in a JupyterLab bundled asset. The requirements file is also pip-compatible when used at a shorter path. No system path-length or execution-policy setting was changed.

The dedicated kernelspec contains an absolute interpreter path. Re-register it after moving or recreating the project; use a different kernel name if keeping multiple copies. Recreating this environment does not require editing any global Python installation.

## Troubleshooting

- **Wrong packages or missing imports:** check `sys.executable` in the notebook and terminal. Select the project interpreter and kernel, then restart the notebook kernel.
- **Activation blocked by PowerShell policy:** use `.\.venv\Scripts\python.exe` directly, or choose a Command Prompt terminal and run `.venv\Scripts\activate.bat`. No system execution-policy change is needed.
- **Kernel missing or stale:** run the `ipykernel install` command above, reload the VS Code window, and select it again.
- **CUDA reports false:** this is expected for the installed CPU wheel, even though NVIDIA hardware is present.
- **Model/dataset downloads fail:** check connectivity and disk space. Avoid login or tokens for these public resources unless an actual access error requires them. A sliced dataset load can still download a whole shard; streaming can limit initial data transfer.
- **Tutorial code differs:** use the compatibility notes and smoke tests before changing versions. Preserve the pinned file and create a separate environment for experiments with upgrades.
- **T5 predictions disappoint:** verify the checkpoint, prefix, inputs, and decoding. The baseline has not been fine-tuned on your task data.

## Suggested learning sequence

- **Monday, October 12:** notebook 1; inspect pairs, splits, leakage, and domain mismatch.
- **Tuesday, October 13:** notebook 2; explain source/target tokens, masks, padding, and labels.
- **Wednesday, October 14:** notebook 3; assess metric limits, retail-specific errors, latency, and quality tradeoffs.
- **Thursday, October 15:** notebook 4; record a small baseline and compare generation settings. Prepare a short explanation of the complete pipeline.
- **Friday, October 16:** review your own observations before the interview. Fine-tuning can wait until you decide it is useful.
