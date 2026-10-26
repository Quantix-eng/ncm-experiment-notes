"""Recompute only DE/self-repair from saved states and memory-mapped official weights; no model import."""
from pathlib import Path
import json
import struct
import math
import hashlib
import datetime
import time
import numpy as np

D=Path(__file__).resolve().parents[1];started=time.perf_counter()
freeze=json.loads((D/'results/protocol_frozen.json').read_text())
for path,h in freeze['files'].items():
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):digest.update(chunk)
    assert digest.hexdigest()==h,path
weights=D/'weights/model.safetensors'
with weights.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
def tensor(name):
    v=header[name];assert v['dtype']=='F16'
    return np.memmap(weights,mode='r',dtype='<f2',offset=8+n+v['data_offsets'][0],shape=tuple(v['shape']))
wo=tensor('gpt_neox.layers.11.attention.dense.weight').astype(np.float64)
gamma=tensor('gpt_neox.final_layer_norm.weight').astype(np.float64)
wu=tensor('embed_out.weight');mean=np.zeros(wu.shape[1],np.float64)
for begin in range(0,len(wu),1024): mean+=wu[begin:begin+1024].astype(np.float64).sum(axis=0)
mean/=len(wu)
old=json.loads((D/'archives/v1_1_before_fp64_measurement/results/validation_observations.json').read_text())
rows=json.loads((D/'results/validation_observations.json').read_text())
q=np.load(D/'results/validation_native_cache.npz')['q'];states=np.load(D/'results/validation_states.npz')['states']
changes=[]
for i,row in enumerate(rows):
    h=row['head'];source=row['sequence'];donor=row['donor'];target=row['target'];s0=float(states[i,2,0])
    dq=q[donor,h*64:(h+1)*64].astype(np.float64)-q[source,h*64:(h+1)*64].astype(np.float64)
    dh=np.ascontiguousarray(wo[:,h*64:(h+1)*64])@dq
    de=float(np.dot((dh-dh.mean())/s0*gamma,wu[target].astype(np.float64)-mean))
    assert math.isfinite(de)
    changes.append(de-old[i]['delta_DE']);row['delta_DE']=de
    row['self_repair']=row['centered_target_logit'][2]-row['centered_target_logit'][0]-de
    for key in ['loss','correct','centered_target_logit','D','R','I','S','scale_ratio']:
        assert row[key]==old[i][key],key
summary=json.loads((D/'results/validation_summary.json').read_text());nseq=summary['n_sequences']
def update(summary_rows,output):
    for key in ['delta_DE','self_repair']:
        values=np.array([np.mean([r[key] for r in summary_rows if r['sequence']==i]) for i in range(nseq)])
        rng=np.random.default_rng(20261025);samples=values[rng.integers(nseq,size=(10000,nseq))].mean(axis=1)
        output[key]=dict(mean=float(values.mean()),descriptive_ci95=np.quantile(samples,[.025,.975]).tolist())
    output['positive_self_repair']=sum(r['self_repair']>0 for r in summary_rows)
update(rows,summary['metrics'])
for h in [2,10]: update([r for r in rows if r['head']==h],summary['by_head'][str(h)])
identity=max(abs(r['self_repair']-r['reader_logit_effect']) for r in rows)
assert identity<2e-4,identity
summary['technical']['max_self_repair_reader_identity_error']=identity
summary['measurement_amendment']='1.2; direct-effect/self-repair only recomputed without model forward'
audit=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_loaded=False,
    network_forward_executed=False,n_measurements=len(rows),max_abs_DE_change=float(np.max(np.abs(changes))),
    mean_signed_DE_change=float(np.mean(changes)),all_other_observations_exactly_unchanged=True,
    max_self_repair_reader_identity_error=identity,elapsed_seconds=time.perf_counter()-started)
for path,x in [('results/validation_observations.json',rows),('results/validation_summary.json',summary),('results/de_fp64_amendment_audit.json',audit)]:
    (D/path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(audit,indent=2))
