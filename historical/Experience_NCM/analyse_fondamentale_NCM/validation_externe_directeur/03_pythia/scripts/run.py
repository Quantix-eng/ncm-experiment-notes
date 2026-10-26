"""Final-layer resampling: separate a changing reader from an empty neural continuation."""
from common import *
import argparse
import time
import datetime

parser=argparse.ArgumentParser(); parser.add_argument('--split',choices=['validation','test'],required=True)
args=parser.parse_args(); check_freeze(); setup()
if args.split=='test':
    gate=json.loads((D/'results/validation_summary.json').read_text())
    assert gate['technical_pass'], 'stop before test: failed development technical gate'
started=time.perf_counter(); m=load_model()
rec=json.loads((D/'data/sequences.json').read_text())[args.split]['records']; n=len(rec)
last=m.gpt_neox.layers[11]; captures={}
def grab(name):
    def hook(mod,inputs,output): captures[name]=output.detach().clone()
    return hook
def pre(name):
    def hook(mod,inputs): captures[name]=inputs[0].detach().clone()
    return hook
hooks=[last.register_forward_pre_hook(pre('h')),last.mlp.register_forward_hook(grab('mlp')),
    last.attention.dense.register_forward_pre_hook(pre('q')),
    last.attention.dense.register_forward_hook(grab('attention')),
    m.gpt_neox.final_layer_norm.register_forward_pre_hook(pre('residual'))]
native=[]; errors=[]
with torch.inference_mode():
    for i,r in enumerate(rec):
        out=m(torch.tensor([r['tokens'][:32]]),use_cache=False,logits_to_keep=1)
        c={k:v[0,-1].clone() for k,v in captures.items()}
        c['logits']=out.logits[0,-1].clone()
        assert all(bool(torch.isfinite(v).all()) for v in c.values()),('nonfinite native',i)
        native.append(c)
        if (i+1)%64==0: print(args.split,'native',i+1,'elapsed',round(time.perf_counter()-started,2),flush=True)
for h in hooks: h.remove()
ln=m.gpt_neox.final_layer_norm
unembedding_mean=m.embed_out.weight.detach().double().mean(dim=0)
head_projection_fp64={h:last.attention.dense.weight[:,h*64:(h+1)*64].detach().double().contiguous() for h in [2,10]}
def scale(r): return torch.sqrt(torch.var(r,unbiased=False)+ln.eps)
def read(r,s):
    # Preserve the actual native LayerNorm kernel exactly when the denominator is native.
    # Algebraically: (center(r)/s_r*gamma+beta) + center(r)/s_r*gamma*(s_r/s-1).
    normalized=ln(r)
    return m.embed_out(normalized+(normalized-ln.bias)*(scale(r)/s-1))
def nll(z,y): return float(torch.logsumexp(z.double(),dim=-1)-z[y].double())
def centered(z,y): return float(z[y].double()-z.double().mean())
records=[]; stored=[]; max_native=0.; max_replay=0.; max_residual=0.; max_mlp=0.; max_identity=0.
with torch.inference_mode():
    for i,source in enumerate(rec):
        c=native[i]; r0=c['residual']; s0=scale(r0); y=source['tokens'][32]
        z0=read(r0,s0)
        max_native=max(max_native,float((z0-c['logits']).abs().max()))
        max_residual=max(max_residual,float(((c['mlp']+c['attention'])+c['h']-r0).abs().max()))
        for head in [2,10]:
            rs=[]; ss=[]; diffs=[]
            for j in source['donors']:
                q=c['q'].clone(); q[head*64:(head+1)*64]=native[j]['q'][head*64:(head+1)*64]
                a=last.attention.dense(q); r=(c['mlp']+a)+c['h']
                rs.append(r);ss.append(scale(r))
                # Published direct-effect difference: projected head delta, clean LN denominator.
                dq=native[j]['q'][head*64:(head+1)*64].double()-c['q'][head*64:(head+1)*64].double()
                dh=torch.nn.functional.linear(dq,head_projection_fp64[head],None)
                assert bool(torch.isfinite(dh).all()),('nonfinite direct effect',i,head,j)
                diffs.append(dh)
            for d,(r,s) in enumerate(zip(rs,ss)):
                other=1-d
                zs=[z0,read(r,s0),read(r,s),read(r0,s),read(r,ss[other])]
                assert all(bool(torch.isfinite(z).all()) for z in zs),('nonfinite crossed readout',i,head,d)
                losses=[nll(z,y) for z in zs]
                correct=[int(z.argmax()==y) for z in zs]
                target=[centered(z,y) for z in zs]
                dh=diffs[d]; delta_de=float(torch.dot((dh-dh.mean())/s0.double()*ln.weight.double(),
                      m.embed_out.weight[y].double()-unembedding_mean))
                sr=target[2]-target[0]-delta_de
                reader_shift=target[2]-target[1]
                max_identity=max(max_identity,abs(sr-reader_shift))
                records.append(dict(sequence=i,head=head,donor=source['donors'][d],target=y,loss=losses,
                    correct=correct,centered_target_logit=target,delta_DE=delta_de,self_repair=sr,
                    reader_logit_effect=reader_shift,scale_ratio=float(s0/s),
                    D=losses[1]-losses[0],R=losses[1]-losses[2],
                    I=losses[1]+losses[3]-losses[2]-losses[0],S=losses[4]-losses[2],
                    continuation_R=0.0,continuation_I=0.0,continuation_S=0.0))
                stored.append(torch.stack([r0,r,s0.expand_as(r0),s.expand_as(r0),ss[other].expand_as(r0)]).numpy())
            # Full patched forward for fixed preannounced small technical audit, not favorable cases.
            if i<4:
                donor=source['donors'][0]
                audited={}
                def patch(mod,inputs):
                    q=inputs[0].clone();q[0,-1,head*64:(head+1)*64]=native[donor]['q'][head*64:(head+1)*64]
                    return (q,)
                def log_mlp(mod,inputs,output): audited['mlp']=output[0,-1].clone()
                hk=last.attention.dense.register_forward_pre_hook(patch)
                hm=last.mlp.register_forward_hook(log_mlp)
                out=m(torch.tensor([source['tokens'][:32]]),use_cache=False,logits_to_keep=1).logits[0,-1]
                hk.remove();hm.remove()
                max_replay=max(max_replay,float((out-read(rs[0],ss[0])).abs().max()))
                max_mlp=max(max_mlp,float((audited['mlp']-c['mlp']).abs().max()))
        if (i+1)%64==0: print(args.split,'crossed',i+1,'elapsed',round(time.perf_counter()-started,2),flush=True)

def summary(rows):
    out=dict(n=len(rows))
    for metric in ['D','R','I','S','delta_DE','self_repair','reader_logit_effect','scale_ratio']:
        byseq=np.array([np.mean([x[metric] for x in rows if x['sequence']==i]) for i in range(n)])
        rng=np.random.default_rng(20261025); estimates=byseq[rng.integers(n,size=(10000,n))].mean(axis=1)
        out[metric]=dict(mean=float(byseq.mean()),descriptive_ci95=np.quantile(estimates,[.025,.975]).tolist())
    out['correct_by_condition']=np.array([x['correct'] for x in rows]).sum(axis=0).tolist()
    out['scale_ratio_greater_one']=sum(x['scale_ratio']>1 for x in rows)
    out['positive_reader_recovery']=sum(x['R']>0 for x in rows)
    out['positive_self_repair']=sum(x['self_repair']>0 for x in rows)
    return out
technical=dict(max_native_logit_error=max_native,max_patched_replay_logit_error=max_replay,
    max_native_residual_error=max_residual,max_final_mlp_change=max_mlp,
    max_self_repair_reader_identity_error=max_identity,full_forward_replays=8)
passed=max_native<2e-4 and max_replay<2e-4 and max_residual<1e-5 and max_mlp==0 and max_identity<2e-4
result=dict(split=args.split,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter()-started,n_sequences=n,n_interventions=len(records),
    native_correct_sequences=sum(int(native[i]['logits'].argmax()==rec[i]['tokens'][32]) for i in range(n)),
    technical_pass=passed,technical=technical,metrics=summary(records),
    by_head={str(h):summary([r for r in records if r['head']==h]) for h in [2,10]},
    conclusion='reader-only structural control; no adaptive downstream attention/MLP exists at this boundary',
    parameters=sum(p.numel() for p in m.parameters()))
write(f'results/{args.split}_observations.json',records)
np.savez_compressed(D/f'results/{args.split}_states.npz',states=np.stack(stored))
np.savez_compressed(D/f'results/{args.split}_native_cache.npz',
    **{k:np.stack([c[k].numpy() for c in native]) for k in native[0]})
write(f'results/{args.split}_summary.json',result)
check_freeze();print(json.dumps(result,indent=2));assert passed,technical
