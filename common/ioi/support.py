"""Isolated GPT-2 stack; no writes outside this new campaign."""
import numpy as np
import torch
import os, sys, json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
D = Path(__file__).resolve().parent
ROOT = D.parents[2]
DEPS = D.parent.parent/'benchmark_multimodel_ar/deps'
sys.path.insert(0, str(DEPS))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write(p, x):
    p = D/p; p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(x, ensure_ascii=False, allow_nan=False, indent=2)+'\n')

def read(p): return json.loads((D/p).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def setup():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261025)

def load():
    net = GPT2LMHeadModel.from_pretrained(str(ROOT/'artifacts/gpt2'),
        local_files_only=True, attn_implementation='eager', dtype=torch.float32).eval()
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT/'artifacts/gpt2'), local_files_only=True)
    return net, tok

def metric(logits, choices, answer):
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(); lse = m+np.log(np.exp(x-m).sum())
    right, wrong = choices[answer], choices[1-answer]
    return dict(nll=float(lse-x[right]), margin=float(x[right]-x[wrong]),
        correct_candidate=bool(x[right]>x[wrong]), correct_vocab=bool(x.argmax()==right),
        centred_logit=float(x[right]-x.mean()))

def freeze(stage):
    paths = list(D.glob('*.py'))+list(D.glob('PROTOCOLE*.md'))+list((D/'data').glob('*.json'))
    paths += list((ROOT/'artifacts/gpt2').glob('*'))
    paths += [DEPS/'transformers/models/gpt2/modeling_gpt2.py']
    files = {str(p.resolve()):sha(p) for p in paths if p.is_file()}
    write('results/freeze_'+stage+'.json', dict(time_utc=now(), stage=stage, files=files,
        status='local freeze, not independent preregistration'))

def check(stage):
    for p,h in read('results/freeze_'+stage+'.json')['files'].items():
        assert sha(p)==h, p
