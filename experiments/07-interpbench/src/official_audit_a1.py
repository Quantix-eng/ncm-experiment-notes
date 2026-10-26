"""A1: official TransformerLens development parity and crossed replay only."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
sys.path.insert(0,str(HERE.parents[1]/'benchmark_multimodel_ar/deps'))
sys.path.insert(0,str(HERE/'deps'))
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pilot import CASES,encode,mean_loss
from verify import forward as numpy_forward

torch.set_num_threads(1)


def build(case,dtype):
    cfg=CASES[str(case)]['transformer_cfg'].copy()
    cfg.update(device='cpu',dtype=dtype,init_weights=False)
    model=HookedTransformer(HookedTransformerConfig(**cfg)).eval()
    weights=torch.load(HERE/f'sources/models/{case}/ll_model.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True)
    return model,weights


def tail_names(cut):
    return [f'blocks.{cut}.hook_mlp_out']+[name for layer in range(cut+1,2)
                                         for name in [f'blocks.{layer}.hook_attn_out',f'blocks.{layer}.hook_mlp_out']]


def fixed_hooks(cache,cut):
    return [(name,lambda value,hook,kept=cache[name]:kept) for name in tail_names(cut)]


def head_hook(cache,cut):
    name=f'blocks.{cut}.attn.hook_result'
    def patch(value,hook):
        result=value.clone();result[:,:,0]=cache[name][:,:,0];return result
    return (name,patch)


@torch.inference_mode()
def run():
    start=time.perf_counter()
    seal=json.loads((HERE/'results/amendment_a1_frozen.json').read_text())
    for p,h in seal['sha256'].items():
        path=Path(p) if Path(p).is_absolute() else HERE/p
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,p
    data=json.loads((HERE/'data/dev.json').read_text());old=np.load(HERE/'results/development_logits.npz')
    native=[];obs=[];arrays={};max64=0.;max32=0.;argmax32=0;native_replay=0
    for case in [8,33]:
        rows=[r for r in data if r['case']==case]
        model64,weights=build(case,torch.float64)
        for row in rows:
            ids=encode(row);official=model64(ids).numpy()[0];reference=numpy_forward(ids.numpy()[0],weights)
            err=float(np.max(np.abs(official-reference)));max64=max(max64,err)
            assert np.allclose(official,reference,atol=1e-8,rtol=1e-10),(row['id'],err)
        del model64
        model,_=build(case,torch.float32)
        states=[model.run_with_cache(encode(r)) for r in rows]
        for i,row in enumerate(rows):
            ids=encode(row);z,c=states[i];loss,exact=mean_loss(z,row)
            native.append({'id':row['id'],'case':case,'loss':loss,'exact':exact})
            assert np.array_equal(z.numpy().argmax(-1),old[row['id']+'-native'].argmax(-1))
            max32=max(max32,float(np.max(np.abs(z.numpy()-old[row['id']+'-native']))));argmax32+=1
            arrays[row['id']+'-native']=z.numpy()
            for cut in [0,1]:
                d1=states[(i+1)%128][1];d2=states[(i+2)%128][1]
                with model.hooks(fwd_hooks=[head_hook(d1,cut)]):
                    y11,adapt=model.run_with_cache(ids)
                with model.hooks(fwd_hooks=[head_hook(d2,cut)]):
                    _,wrong=model.run_with_cache(ids)
                y00=model.run_with_hooks(ids,fwd_hooks=fixed_hooks(c,cut))
                assert torch.equal(y00,z),'official native replay';native_replay+=1
                ys={'Y00':y00,'Y10':model.run_with_hooks(ids,fwd_hooks=[head_hook(d1,cut)]+fixed_hooks(c,cut)),
                    'Y11':y11,'Y01':model.run_with_hooks(ids,fwd_hooks=fixed_hooks(adapt,cut)),
                    'Y1wrong':model.run_with_hooks(ids,fwd_hooks=[head_hook(d1,cut)]+fixed_hooks(wrong,cut))}
                sham=model.run_with_hooks(ids,fwd_hooks=[head_hook(c,cut)]+fixed_hooks(c,cut))
                assert torch.equal(sham,z),'official sham'
                record={'id':row['id'],'case':case,'cut':cut,'head':0,'conditions':{}}
                for name,value in ys.items():
                    l,e=mean_loss(value,row);record['conditions'][name]={'loss':l,'exact':e}
                    a=value.numpy();b=old[row['id']+f'-L{cut}H0-'+name]
                    assert np.array_equal(a.argmax(-1),b.argmax(-1)),(row['id'],cut,name)
                    max32=max(max32,float(np.max(np.abs(a-b))));argmax32+=1
                    arrays[row['id']+f'-L{cut}H0-'+name]=a
                l={k:v['loss'] for k,v in record['conditions'].items()}
                record['contrasts']=dict(D=l['Y10']-l['Y00'],R=l['Y10']-l['Y11'],I=l['Y10']+l['Y01']-l['Y11']-l['Y00'],S=l['Y1wrong']-l['Y11'])
                obs.append(record)
        del model,states
    np.savez_compressed(HERE/'results/official_development_logits_a1.npz',**arrays)
    out={'status':'passed_official_backend_development_audit','native':native,'interventions':obs,
         'official_fp64_vs_numpy_max_abs':max64,'fp32_vs_archived_max_abs':max32,
         'fp32_argmax_comparisons':argmax32,'official_native_replays':native_replay,'official_shams':native_replay,
         'initial_failed_audit_preserved':True,'reserved_tasks_executed':[],
         'elapsed_seconds':time.perf_counter()-start,
         'versions':{n:importlib.metadata.version(n) for n in ['transformer-lens','datasets','pyarrow','accelerate','einops','torch','numpy','transformers']}}
    (HERE/'results/official_audit_a1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['native','interventions']},indent=2))


if __name__=='__main__':run()
