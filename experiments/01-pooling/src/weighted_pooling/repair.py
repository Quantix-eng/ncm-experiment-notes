"""Frozen-size pooling replacements and an explicit auditable residual codec."""
from base import *
import math,contextlib
import torch.nn.functional as F

def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def statehash(module):
    h=hashlib.sha256()
    for key,v in sorted(module.state_dict().items()):h.update(key.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def pool(x,name):
    """Only length-89 -> length-9; weights fixed before seeing any test input."""
    assert x.ndim==3 and x.shape[1:]==(89,256),'Sudoku-only fixed-size repair'
    base=F.interpolate(x.transpose(1,2),size=9,mode='area').transpose(1,2)
    if name=='native':return base
    kind,amount=name.rsplit('_',1);alpha=float(amount)
    if kind=='weighted':
        moments=[]
        for j in range(9):
            lo=math.floor(j*89/9);hi=math.ceil((j+1)*89/9);t=torch.linspace(-1,1,hi-lo,dtype=x.dtype,device=x.device)
            moments.append((x[:,lo:hi]*t[None,:,None]).mean(1))
        return base+alpha*torch.stack(moments,1)
    if kind=='shift':
        shifted=torch.cat([x[:,1:],x[:,-1:]],1);u=F.interpolate(shifted.transpose(1,2),size=9,mode='area').transpose(1,2)
        return base+alpha*(u-base)
    if kind=='max':
        u=F.adaptive_max_pool1d(x.transpose(1,2),9).transpose(1,2)
        return base+alpha*(u-base)
    if kind=='fixed_bias':
        pattern=np.random.default_rng(Q['seed']).normal(size=(9,x.shape[-1]));pattern/=np.sqrt(np.mean(pattern**2))
        return base+alpha*torch.as_tensor(pattern,dtype=x.dtype,device=x.device)[None]
    raise ValueError(name)

@contextlib.contextmanager
def patched(vq,name,counter=None):
    original=vq._down_for_quant;had='_down_for_quant' in vq.__dict__;old=vq.__dict__.get('_down_for_quant')
    def replacement(z,k,length):
        if k==4:
            assert int(length)==9
            if counter is not None:counter.append(k)
            return pool(z,name)
        return original(z,k,length)
    vq._down_for_quant=replacement
    try:yield
    finally:
        if had:vq._down_for_quant=old
        else:delattr(vq,'_down_for_quant')

@torch.inference_mode()
def encode(model,solution,name,noise=None):
    v=model.vqvae;vq=v.vq;xx=torch.as_tensor(solution,dtype=torch.long);z=v._encode_pre_quant(xx)
    if noise is not None:z=z+torch.as_tensor(noise,dtype=z.dtype)
    raw=v.encoder.wte(xx)
    if v.encoder.wpe is not None:raw=raw+v.encoder.wpe(torch.arange(89))[None]
    rest=z.clone();zhat=torch.zeros_like(z);codes=[];capture={}
    for k,l in enumerate(vq.levels):
        u=(pool(rest,name) if k==4 else vq._down_for_quant(rest,k,l)) if k<11 else rest
        idx=vq._distances(u.reshape(-1,256),vq.embeddings[k].weight).argmin(1).reshape(len(xx),l)
        contribution=vq.embeddings[k](idx)
        if k<11:contribution=vq._up_to_full(contribution,k,89)
        contribution=vq._apply_quant_resi(contribution,k);codes.append(idx)
        if k==4:capture=dict(u5=u.numpy(),native_u5=pool(rest,'native').numpy(),raw5=pool(raw,name).numpy(),raw_native5=pool(raw,'native').numpy())
        rest=rest-contribution;zhat=zhat+contribution
    logits=v.decoder(zhat);return dict(codes=torch.cat(codes,1).numpy(),ids=logits.argmax(-1).numpy(),**capture)

def occupancy(c):
    counts=np.stack([np.bincount(c[:,j],minlength=16) for j in range(9)]);p=counts/counts.sum(1,keepdims=True);m=counts.sum(0)/counts.sum()
    entropy=lambda a:float(-(a[a>0]*np.log(a[a>0])).sum())
    return dict(counts=counts,codes_used=int((counts.sum(0)>0).sum()),marginal_entropy=entropy(m),positional_entropy=float(np.mean([entropy(v) for v in p])),
      unique_sequences=len(np.unique(c,axis=0)))

def run(model,solutions,name,noise_sigma=0.,noise_seed=None,indices=None):
    parts={}
    for lo in range(0,len(solutions),Q['batch_size']):
        batch=solutions[lo:lo+Q['batch_size']];noise=None
        if noise_seed is not None:
            ids=np.arange(lo,lo+len(batch)) if indices is None else indices[lo:lo+len(batch)]
            noise=np.stack([np.random.default_rng(noise_seed+int(i)*100003).normal(0,noise_sigma,size=(89,256)) for i in ids]).astype(np.float32)
        out=encode(model,batch,name,noise=noise)
        for key,value in out.items():parts.setdefault(key,[]).append(value)
    return {key:np.concatenate(value) for key,value in parts.items()}

def summary(out,solutions,native=None):
    exact=np.all(out['ids']==solutions,axis=1);result=dict(n=len(exact),exact=int(exact.sum()),exact_rate=float(exact.mean()),
        exact_wilson95=ci_rate(exact.sum(),len(exact)),token_errors=int((out['ids']!=solutions).sum()),cell_errors=int((out['ids'][:,CELL]!=solutions[:,CELL]).sum()),
        occupancy=occupancy(out['codes'][:,13:22]),input_variance=float(out['u5'].astype(float).var(0).mean()),raw_variance=float(out['raw5'].astype(float).var(0).mean()),
        input_change_rms=float(np.sqrt(np.mean((out['u5'].astype(float)-out['native_u5'])**2))))
    if native is not None:
        before=np.all(native['ids']==solutions,axis=1);result.update(introduced=int((before&~exact).sum()),rescued=int((~before&exact).sum()),
            changed_grids=int(np.any(out['ids']!=native['ids'],axis=1).sum()),changed_codes=int((out['codes']!=native['codes']).sum()),
            changed_codes5=int((out['codes'][:,13:22]!=native['codes'][:,13:22]).sum()),
            delta_exact=paired_mean_ci(exact.astype(float)-before))
    return result
