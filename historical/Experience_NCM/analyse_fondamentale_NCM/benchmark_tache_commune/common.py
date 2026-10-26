import json
import hashlib
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

D=Path(__file__).resolve().parent
ROOT=D.parent.parent
Q=json.loads((D/'config.json').read_text())
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def write(name,value):
    p=D/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(name):return json.loads((D/name).read_text())


def check():
    for p,h in read('results/protocol_frozen.json')['files'].items():assert sha(p)==h,p


def examples(base,b,n):
    return [8,9+b,11+n]+list(base),list(base[b:])+list(base[:b])


def fixed_queries(batch):
    return torch.tensor([14,13,13,13]).repeat(batch,1)


def logits_loss(logits,target):
    v=np.asarray(logits,dtype=np.float64)
    m=v.max(axis=-1,keepdims=True)
    lp=v-m-np.log(np.exp(v-m).sum(axis=-1,keepdims=True))
    return -np.take_along_axis(lp,np.asarray(target)[...,None],axis=-1)[...,0]


def label(exact,g,w,u):
    if not all(exact):return 'non_identifiable_native_anchor_failure'
    if g<=.1:return 'non_identifiable_weak_native_preference_span'
    if abs(w/g)>.2:return 'non_additive_interaction_not_a_mechanism_class'
    if not 0<=u/g<=1:return 'non_monotone_non_identifiable'
    if u/g>=.8:return 'input_dominant_response_preference_in_this_test'
    if u/g<=.2:return 'state_dominant_response_preference_in_this_test'
    return 'two_sensitivities_mixed_or_unmodeled'


def ci(values,seed):
    v=np.asarray(values);ix=np.random.default_rng(seed).integers(len(v),size=(20000,len(v)))
    b=v[ix].mean(axis=1)
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),
                ci_adjusted=np.quantile(b,[.00625,.99375]).tolist(),n_families=len(v))
