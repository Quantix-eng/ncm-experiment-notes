"""Native finite-stage residual reconstruction and reversible code interventions."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from features import CELL_POS

LEVELS=[1,2,4,6,9,13,18,26,37,50,67,89]

@torch.inference_mode()
def reconstruct(model,codes,level):
    assert 1<=level<=12 and codes.shape[1]==sum(LEVELS[:level]), 'Reject future codes'
    vq=model.vqvae.vq
    z=torch.zeros((len(codes),89,256),dtype=torch.float32)
    offset=0
    for k,length in enumerate(LEVELS[:level]):
        e=vq.embeddings[k](codes[:,offset:offset+length])
        if k<11:e=vq._up_to_full(e,k,89)
        z.add_(vq._apply_quant_resi(e,k));offset+=length
    return z

@torch.inference_mode()
def native_features(model,codes):
    assert codes.shape[1]==116
    z8=reconstruct(model,codes[:,:79],8)
    z9=reconstruct(model,codes,9)
    delta=z9-z8
    s=z9[:,CELL_POS].numpy().copy();d=delta[:,CELL_POS].numpy().copy()
    a,b=np.linalg.norm(s.astype(float),axis=2),np.linalg.norm(d.astype(float),axis=2)
    cosine=(s.astype(float)*d).sum(2)/np.maximum(a*b,1e-12)
    amplitude=np.stack([a/16,b/16,cosine],axis=2)
    return s,d,amplitude,z9

@torch.inference_mode()
def forced_prediction(model,prefix,positions,values):
    """Force one level8 code per example, no changes to weights or public path."""
    seen=[]
    def force(module,args,output):
        seen.append(True)
        result=output.clone()
        rows=torch.arange(len(prefix));positions_t=torch.as_tensor(positions);values_t=torch.as_tensor(values)
        result[rows,positions_t,values_t]=result.amax((1,2))+100
        return result
    h=model.net.lm_heads[7].register_forward_hook(force)
    try:prediction=model.predict(prefix)
    finally:h.remove()
    assert len(seen)==1
    assert np.array_equal(prediction.codes[:,53:79].numpy()[np.arange(len(prefix)),positions],values)
    assert all(not b.attn.caching and b.attn.cached_k is None for b in model.net.transformer.blocks)
    return prediction

@torch.inference_mode()
def clean_prediction(model,prefix):
    captured=[]
    h=model.net.lm_heads[7].register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
    try:p=model.predict(prefix)
    finally:h.remove()
    assert len(captured)==1
    return p,captured[0]

@torch.inference_mode()
def choose_interventions(model,codes,logits,groups):
    positions=np.array([np.random.default_rng(np.random.SeedSequence([20261203,int(g)])).integers(26)for g in groups])
    original=codes[:,53:79].numpy()[np.arange(len(codes)),positions]
    ranked=torch.argsort(logits,dim=-1,descending=True,stable=True).numpy()[np.arange(len(codes)),positions]
    assert np.array_equal(ranked[:,0],original)
    plausible=ranked[:,1];other=[];norms=[]
    vq=model.vqvae.vq
    for j in range(len(codes)):
        base=codes[j:j+1,53:79]
        alternatives=base.repeat(16,1);alternatives[torch.arange(16),positions[j]]=torch.arange(16)
        e=vq.embeddings[7](alternatives);e=vq._up_to_full(e,7,89);r=vq._apply_quant_resi(e,7)
        n=(r-r[int(original[j])]).double().flatten(1).norm(dim=1).numpy()
        valid=[k for k in range(16)if k not in (original[j],plausible[j])]
        chosen=min(valid,key=lambda k:(abs(n[k]-n[plausible[j]]),k))
        other.append(chosen);norms.append([n[plausible[j]],n[chosen]])
    return positions,original,plausible,np.array(other),np.array(norms)
