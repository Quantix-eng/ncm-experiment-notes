"""Isolated dependencies; existing NumPy/PyTorch load before optional HF stack."""
import numpy as np
import torch
import os
from pathlib import Path
import sys
import json
import hashlib
import time

D = Path(__file__).resolve().parent
ROOT = D.parent.parent
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(D/'deps'))
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = D/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(name):
    return json.loads((D/name).read_text())


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261025)


def model():
    return GPT2LMHeadModel.from_pretrained(str(ROOT/'artifacts/gpt2'), local_files_only=True,
                                         attn_implementation='eager', dtype=torch.float32).eval()


def check_freeze():
    for path, expected in read('results/protocol_frozen.json')['files'].items():
        assert sha(Path(path)) == expected, path


def logprob(logits):
    value = np.asarray(logits, dtype=np.float64)
    maximum = value.max(axis=-1, keepdims=True)
    return value-maximum-np.log(np.exp(value-maximum).sum(axis=-1, keepdims=True))


def bootstrap(values, seed):
    values = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(len(values), size=(20000, len(values)))
    estimates = values[idx].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                ci_adjusted=np.quantile(estimates, [.00625, .99375]).tolist(), n_families=len(values))
