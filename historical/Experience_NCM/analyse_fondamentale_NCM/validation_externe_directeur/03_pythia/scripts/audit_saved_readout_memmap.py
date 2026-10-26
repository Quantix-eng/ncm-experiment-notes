"""Post-hoc resource-limited implementation of the frozen FP64 reader audit; no model import."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse
import json
import struct
import hashlib
import datetime
import time
import math
import numpy as np

D=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],required=True)
a=parser.parse_args();start=time.perf_counter()
freeze=json.loads((D/'results/protocol_frozen.json').read_text())
for name,expected in freeze['files'].items():
    digest=hashlib.sha256()
    with Path(name).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):digest.update(chunk)
    assert digest.hexdigest()==expected,name
path=D/'weights/model.safetensors'
with path.open('rb') as f:
    header_bytes=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_bytes))
def tensor(name):
    v=header[name];assert v['dtype']=='F16'
    return np.memmap(path,dtype='<f2',mode='r',offset=8+header_bytes+v['data_offsets'][0],shape=tuple(v['shape']))
wu=tensor('embed_out.weight');gamma=tensor('gpt_neox.final_layer_norm.weight').astype(np.float64)
beta=tensor('gpt_neox.final_layer_norm.bias').astype(np.float64)
states=np.load(D/f'results/{a.split}_states.npz')['states']
rows=json.loads((D/f'results/{a.split}_observations.json').read_text())
summary=json.loads((D/f'results/{a.split}_summary.json').read_text())
assert np.isfinite(states).all()
contrast_checks=0
for row in rows:
    loss=row['loss'];assert all(math.isfinite(v) for v in loss)
    for key,value in dict(D=loss[1]-loss[0],R=loss[1]-loss[2],I=loss[1]+loss[3]-loss[2]-loss[0],S=loss[4]-loss[2]).items():
        assert abs(value-row[key])<1e-12;contrast_checks+=1
    assert row['continuation_R']==row['continuation_I']==row['continuation_S']==0
assert [sum(row['correct'][i] for row in rows) for i in range(5)]==summary['metrics']['correct_by_condition']
selected=min(32,len(rows));normalized=[];targets=[];expected=[]
for i in range(selected):
    r0,rd,s0,sd,so=states[i].astype(np.float64)
    for r,s in [(r0,s0),(rd,s0),(rd,sd),(r0,sd),(rd,so)]:
        normalized.append((r-r.mean())/s*gamma+beta)
        targets.append(rows[i]['target'])
    expected.extend(rows[i]['loss'])
x=np.stack(normalized);targets=np.array(targets);expected=np.array(expected)
assert np.isfinite(x).all()
maximum=np.full(len(x),-np.inf);exp_sum=np.zeros(len(x));target_logits=np.full(len(x),np.nan)
for begin in range(0,len(wu),1024):
    weight=wu[begin:begin+1024].astype(np.float64)
    z=x@weight.T
    assert np.isfinite(z).all()
    local=z.max(axis=1);new_maximum=np.maximum(maximum,local)
    exp_sum=exp_sum*np.exp(maximum-new_maximum)+np.exp(z-new_maximum[:,None]).sum(axis=1)
    maximum=new_maximum
    chosen=np.flatnonzero((targets>=begin)&(targets<begin+len(weight)))
    target_logits[chosen]=z[chosen,targets[chosen]-begin]
loss=maximum+np.log(exp_sum)-target_logits
assert np.isfinite(loss).all()
error=float(np.max(np.abs(loss-expected)))
r_error=max(abs((loss[5*i+1]-loss[5*i+2])-rows[i]['R']) for i in range(selected))
passed=bool(error<2e-4 and r_error<2e-4)
out=dict(split=a.split,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    audit_status='passed' if passed else 'failed',passed=passed,
    added_after_test_results=True,caveat='additional software implementation of the same frozen FP64 formula, first32 records and tolerance; not independent scientific validation',
    scalar_loss_checks=len(x),scalar_contrast_checks=contrast_checks,
    max_absolute_loss_error=error,max_reader_R_error=float(r_error),unchanged_tolerance=2e-4,
    all_condition_counts_verified=True,all_stored_states_finite=True,seal_intact=True,
    model_loaded=False,full_network_forward_executed=False,unembedding_rows_per_chunk=1024,
    estimated_peak_weight_chunk_bytes=1024*768*8,elapsed_seconds=time.perf_counter()-start)
(D/f'results/{a.split}_memmap_verification.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2));assert passed,out
