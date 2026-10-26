"""CPU-only, isolated dependencies for the Pythia final-reader control."""
from pathlib import Path
import numpy as np
import torch
import os
import sys
import hashlib
import json

D = Path(__file__).resolve().parents[1]
BASE = D.parents[1]
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(BASE/'benchmark_multimodel_ar/deps'))
sys.path.insert(0, str(D/'deps_compatible'))
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import transformers

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def write(p,x):
    p=D/p; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def check_freeze():
    f=json.loads((D/'results/protocol_frozen.json').read_text())
    for p,h in f['files'].items(): assert sha(p)==h,p

def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def load_model():
    m=GPTNeoXForCausalLM.from_pretrained(D/'weights',local_files_only=True,
        attn_implementation='eager',dtype=torch.float32).eval()
    assert m.config.use_parallel_residual and m.config.num_hidden_layers==12
    return m
