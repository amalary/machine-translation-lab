"""Offline setup checks: no pretrained downloads, translation exercises, or training."""
import importlib
import inspect
import json
from pathlib import Path
import sys

import nbformat
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
KERNEL = 'machine-translation-lab'


@pytest.mark.parametrize('module', [
    'torch', 'transformers', 'datasets', 'evaluate', 'sacrebleu',
    'sentencepiece', 'accelerate', 'numpy', 'jupyter', 'jupyterlab',
    'notebook', 'ipykernel', 'pytest',
])
def test_required_import(module):
    importlib.import_module(module)


def test_project_interpreter():
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    settings = json.loads((ROOT / '.vscode' / 'settings.json').read_text())
    configured = settings['python.defaultInterpreterPath'].replace('${workspaceFolder}', str(ROOT))
    assert Path(configured).resolve() == PYTHON.resolve()


def test_torch_initializes_and_detects_device():
    import torch
    assert torch.equal(torch.eye(2) @ torch.ones(2), torch.ones(2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.ones(2, device=device)
    assert tensor.sum().item() == 2
    print(f'Compute device: {device}; torch={torch.__version__}; CUDA runtime={torch.version.cuda}')


def test_tutorial_api_compatibility():
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        PreTrainedTokenizerBase, Seq2SeqTrainer, Seq2SeqTrainingArguments,
        T5Config, T5ForConditionalGeneration,
    )
    import evaluate
    assert callable(load_dataset) and callable(evaluate.load)
    assert callable(Dataset.train_test_split)
    assert callable(AutoTokenizer.from_pretrained)
    assert callable(AutoModelForSeq2SeqLM.from_pretrained)
    assert 'text_target' in inspect.signature(PreTrainedTokenizerBase.__call__).parameters
    assert 'processing_class' in inspect.signature(Seq2SeqTrainer.__init__).parameters
    assert 'eval_strategy' in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    assert callable(DataCollatorForSeq2Seq)
    # Tiny randomly initialized architecture verifies the T5 backend, not translation quality.
    import torch
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
        num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
        eos_token_id=1, pad_token_id=0,
    )).eval()
    with torch.inference_mode():
        result = model.generate(torch.tensor([[2, 3, 1]]), max_new_tokens=2)
    assert result.ndim == 2 and result.shape[0] == 1


def test_sacrebleu_local_backend():
    from sacrebleu.metrics import BLEU
    text = ['This is a sufficiently long identical reference sentence.']
    assert BLEU().corpus_score(text, [text]).score == pytest.approx(100)


def test_notebooks_valid():
    expected = [
        '01_dataset_exploration.ipynb', '02_tokenization.ipynb',
        '03_model_evaluation.ipynb', '04_model_inference.ipynb',
    ]
    assert sorted(p.name for p in (ROOT / 'notebooks').glob('*.ipynb')) == expected
    for name in expected:
        notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata.kernelspec.name == KERNEL
        assert any(cell.cell_type == 'code' for cell in notebook.cells)


def test_kernel_points_to_project_and_starts():
    spec = KernelSpecManager().get_kernel_spec(KERNEL)
    assert Path(spec.argv[0]).resolve() == PYTHON.resolve()
    manager = KernelManager(kernel_name=KERNEL)
    manager.start_kernel(cwd=str(ROOT))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        reply = client.execute_interactive(
            'import sys; from pathlib import Path; '
            f'assert Path(sys.executable).resolve() == Path({str(PYTHON)!r}).resolve(); '
            'import torch; assert torch.ones(1).item() == 1',
            timeout=60,
        )
        assert reply['content']['status'] == 'ok', reply['content']
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
