"""Native unconditional LM1B generation and codec controls; raw readout histories."""
from base import *
sys.path.insert(0,str(D/'sources'))
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from models.common import precompute_rope_cos_sin
from omegaconf import OmegaConf
from assets import filehash
import gc
import ctypes
def release_unused_memory():
    gc.collect()
    if sys.platform=='darwin':
        lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib');fn=lib.malloc_zone_pressure_relief;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_size_t;fn(None,0)

def load():
    cfg=OmegaConf.load(D/'artifacts/lm1b-small/config.yaml')
    with torch.device('meta'):vq=VQVAE(**cfg.tokenizer.vqvae).eval().requires_grad_(False)
    old=ROOT/'generalisation_architecturale';sd=torch.load(old/'artifacts/lm1b/tokenizer.pt',weights_only=True,mmap=True,map_location='cpu')['model'];vq.load_state_dict(sd,strict=True,assign=True);del sd
    with torch.device('meta'):net=NextConceptTransformer(vq,cfg.model).eval().requires_grad_(False)
    ck=torch.load(D/'artifacts/lm1b-small/generator.pt',weights_only=True,mmap=True,map_location='cpu');net.load_state_dict(ck['model'],strict=True,assign=True)
    # Shared-head views keep Python references to biases outside the registered tree.
    # Rebuild this small component with ordinary copying so every view sees the
    # published biases, rather than stale pre-assignment Parameter references.
    heads=type(net.lm_heads)(torch.nn.Linear(int(cfg.model.n_embd),int(cfg.model.vocab_size[0]),bias=False),len(cfg.model.vocab_size),per_level_bias=True)
    heads.load_state_dict({k[len('lm_heads.'):]:v for k,v in ck['model'].items() if k.startswith('lm_heads.')},strict=True)
    net.lm_heads=heads.eval().requires_grad_(False)
    assert all(net.lm_heads[i].bias.data_ptr()==net.lm_heads.biases[i].data_ptr() for i in range(len(net.lm_heads)))
    assert all(torch.equal(v,ck['model'][k]) for k,v in net.state_dict().items())
    # Rebuild only deterministic nonpersistent buffers on CPU, exactly as __init__.
    P=int(net.prefix_len);head_dim=int(cfg.model.n_embd)//int(cfg.model.n_head);levels=net.levels[net._lvl_first:]
    pos=torch.cat([torch.arange(P),*[torch.arange(int(l)) for l in levels]])
    net.rope_cos,net.rope_sin=precompute_rope_cos_sin(pos,head_dim,net.rope_base)
    for k,l in enumerate(levels):
        co,si=precompute_rope_cos_sin(torch.cat([torch.arange(P),torch.arange(int(l))]),head_dim,net.rope_base);setattr(net,f'rope_cos_gen_{k}',co);setattr(net,f'rope_sin_gen_{k}',si)
    net.flex_lvl_full=torch.cat([torch.full((P,),-1,dtype=torch.long),torch.cat([torch.full((int(l),),i,dtype=torch.long) for i,l in enumerate(levels)])])
    assert not any(t.is_meta for t in list(net.parameters())+list(net.buffers()))
    inference=ck['inference'];write(D/'results/lm1b_checkpoint_structure.json',dict(checkpoint_keys=list(ck),inference={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in inference.items() if torch.is_tensor(v)},n_generator_parameters=sum(p.numel() for p in net.parameters()),all_state_tensors_equal_checkpoint=True,shared_head_views_bound_to_loaded_biases=True))
    return vq,net,cfg,inference

@torch.inference_mode()
def decode_history(vq,codes):
    z=torch.zeros(len(codes),128,vq.embed_dim);outputs=[];offset=0;snapshots=[]
    for k,l in enumerate(vq.vq.levels):
        c=codes[:,offset:offset+l];r=vq.vq.embeddings[k](c)
        if k<len(vq.vq.levels)-1:r=vq.vq._up_to_full(r,k,128)
        z=z+vq.vq._apply_quant_resi(r,k);outputs.append(vq.decoder(z).argmax(-1).numpy().copy());snapshots.append(z.clone());offset+=l
    native=vq.vq.idx_to_zhats(codes,return_zlevels=False)[0][-1];assert torch.equal(native,z);assert len({t.data_ptr() for t in snapshots})==12
    return np.stack(outputs,1),z

def sample(net,unigram,seed):
    torch.manual_seed(int(seed));l0=torch.multinomial(unigram.float().cpu(),len(net.levels[:1]),replacement=True).view(1,int(net.levels[0]));prefix=torch.tensor([[Q['text_prefix']]])
    return net.generate(l0,prefix,temperature=Q['text_temperature'],top_k=Q['text_top_k'],top_p=Q['text_top_p'],cfg_scale=0.)

def official_readout(logits,cfg,inference):
    # Exact CPU port of eval_shard.pick_tokens, kept separate from raw histories.
    g=cfg.generation;freq=float(g.frequency_penalty);mu=float(g.frequency_deficit);adj=[float(x) for x in str(g.adjacency_penalty).split(',')];lg=logits.double();B,T,V=lg.shape;allow=inference['token_unigram'][:V].double()[None]*T;cnt=torch.zeros(B,V,dtype=torch.float64);hist=[];out=torch.empty(B,T,dtype=torch.long)
    for t in range(T):
        step=lg[:,t].clone();step-=freq*(cnt-allow).clamp_min(0.);step+=mu*((T-t)/T)*(allow-cnt).clamp_min(0.)
        for j,h in enumerate(reversed(hist[-int(adj[1]):])):step.scatter_add_(1,h[:,None],torch.full((B,1),-adj[0]*adj[2]**j,dtype=torch.float64))
        p=step.argmax(-1);out[:,t]=p;hist.append(p);cnt.scatter_add_(1,p[:,None],torch.ones(B,1,dtype=torch.float64))
    return out.numpy()

def ratio(n,d):
    n=np.asarray(n);d=np.asarray(d);rng=np.random.default_rng(Q['seed']);ix=rng.integers(0,len(n),(Q['bootstrap'],len(n)));den=d[ix].sum(1);b=n[ix].sum(1)[den>0]/den[den>0]
    return dict(numerator=int(n.sum()),denominator=int(d.sum()),rate=float(n.sum()/d.sum()) if d.sum() else None,ci95=np.quantile(b,[.025,.975]) if len(b) else None)
def describe(ids,anchor,mask):
    good=ids==anchor[:,None];result={};levels=[]
    for k in range(1,12):
        adopted=~good[:,k-1]&good[:,k]&mask;laterbad=(~good[:,k+1:]).any(1) if k<11 else np.zeros_like(mask)
        levels.append(dict(arrival_level=k+1,changed=int(((ids[:,k-1]!=ids[:,k])&mask).sum()),adoption_then_abandonment=ratio((adopted&laterbad).sum(1),adopted.sum(1))))
    result['transitions']=levels;result['agreement_by_level']=[float(good[:,k][mask].mean()) for k in range(12)];result['changed_after9_rate']=float(((ids[:,9:]!=ids[:,8,None]).any(1)&mask).sum()/mask.sum());result['n_positions']=int(mask.sum());result['n_trajectories']=len(ids)
    return result

@torch.inference_mode()
def main():
    vq,net,cfg,inference=load();saved=D/'data/text_trajectories';saved.mkdir(exist_ok=True);allcodes=[];allids=[];allofficial=[];replays=[]
    for i in range(Q['text_n']):
        seed=Q['text_seed_base']+i;path=saved/f'{i:04}.npz'
        if path.exists():
            row=np.load(path);codes=row['codes'];ids=row['ids'];final=row['official_final']
        else:
            c=sample(net,inference['level0_unigram'],seed);ids,z=decode_history(vq,c);codes=c.numpy();final=official_readout(vq.decoder(z),cfg,inference);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,codes=codes,ids=ids,official_final=final,seed=seed);tmp.replace(path)
        if i<2:
            replay=sample(net,inference['level0_unigram'],seed);assert np.array_equal(replay.numpy(),codes);checkids,_=decode_history(vq,replay);assert np.array_equal(checkids,ids);replays.append(dict(index=i,codes_equal=True,all_readouts_equal=True))
        allcodes.append(codes);allids.append(ids);allofficial.append(final)
        if (i+1)%8==0:release_unused_memory();log('lm1b_generation',done=i+1,total=Q['text_n'])
    ids=np.concatenate(allids);codes=np.concatenate(allcodes);official=np.concatenate(allofficial);mask=np.ones_like(ids[:,-1],bool);content=(ids[:,-1]>103)&(ids[:,-1]<30522)
    report=dict(all_positions=describe(ids,ids[:,-1],mask),final_content_positions=describe(ids,ids[:,-1],content),replays=replays,levels=list(vq.vq.levels),seed_range=[Q['text_seed_base'],Q['text_seed_base']+Q['text_n']-1],sampling='native eager, official level0 unigram and configured temperature/top-k/top-p; independently seeded B=1 trajectories',anchor_warning='Final raw token is retrospective, not ground truth. Last-step agreement is 1 by construction; abandonment at arrival 11/12 is structurally zero.',raw_vs_official_final_disagreement=float((ids[:,-1]!=official).mean()))
    np.savez_compressed(D/'results/lm1b_generation.npz',codes=codes,decoded=ids,official_final=official,seeds=np.arange(Q['text_seed_base'],Q['text_seed_base']+Q['text_n']));write(D/'results/lm1b_generation.json',report)
    # Oracle reconstruction control, distinct from unconditioned generated sequences.
    del net;gc.collect()
    x=np.load(ROOT/'generalisation_architecturale/data/lm1b_selected.npz')['tokens'][:Q['text_oracle_control_n']];out=[];cc=[];oracle_parity=[]
    for i in range(0,len(x),4):
        c=vq.encode(torch.tensor(x[i:i+4]));h,_=decode_history(vq,c);out.append(h);cc.append(c.numpy())
        if i==0:
            for j in range(len(c)):
                c1=vq.encode(torch.tensor(x[j:j+1]));h1,_=decode_history(vq,c1);oracle_parity.append(dict(codes_equal=bool(torch.equal(c1,c[j:j+1])),readouts_equal=bool(np.array_equal(h1,h[j:j+1]))));assert all(oracle_parity[-1].values())
        if (i+4)%16==0:log('lm1b_oracle',done=min(i+4,len(x)),total=len(x))
    h=np.concatenate(out);np.savez_compressed(D/'results/lm1b_oracle.npz',reference=x,codes=np.concatenate(cc),decoded=h);write(D/'results/lm1b_oracle.json',dict(reference_nonpad=describe(h,x,x!=0),final_anchor_nonpad=describe(h,h[:,-1],x!=0),batch_size=4,cross_batch_parity=oracle_parity,note='Reused validation rows; true reconstruction tokens are available here only. Different inputs from generated unconditional sequences.'))
    log('lm1b_text_complete',generation9=report['all_positions']['transitions'][7])
if __name__=='__main__':main()
