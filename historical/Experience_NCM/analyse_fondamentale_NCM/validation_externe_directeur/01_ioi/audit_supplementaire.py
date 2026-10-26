"""Post-hoc software audit, no new effects or scientific decision thresholds.

Compare saved effects to independent full official forwards, including blockers
and the E intervention boundary; check DE and frozen-reader linear identities.
"""
from support import *
from runtime import Engine
import argparse

def run(stage,suffix):
    setup();check(stage);net,tok=load();engine=Engine(net)
    rows=read('data/selected.json')[stage]
    obs=read('results/'+stage+suffix+'_observations.json')
    states=np.load(D/'results'/f'{stage}{suffix}_states.npz')
    l,h=(9,9) if not suffix else ((9,6) if suffix=='_L9H6' else (10,0))
    backups=[p for p in [(10,1),(10,10),(10,6),(10,2),(11,9),(11,2)] if p[0]>l]
    controls=[p for p in [(10,0),(10,3),(10,4),(10,5),(11,0),(11,1)] if p[0]>l]
    full_checks=0;linear_checks=0;de_error=0.;zero_error=0.;allocation_error=0.;rounding_max=0.
    official_max=0.;finite_count=0
    with torch.inference_mode():
        mean=net.lm_head.weight.double().mean(0)
        ln=net.transformer.ln_f
        gamma=ln.weight.double().numpy();beta=ln.bias.double().numpy()
        for i in range(4):
            row=rows[i];c=engine.native(row['tokens'][0])
            for k,offset in enumerate((37,73)):
                j=(i+offset)%len(rows);dc=engine.native(rows[j]['tokens'][0])
                raw=dc['heads',l][0,-1,h*64:(h+1)*64].clone()
                actual=states['crossed'][2*i+k].astype(np.float64)
                direction=(net.lm_head.weight[row['choices'][0]].double()-mean).numpy()
                r0=actual[0];sigma=np.sqrt(np.mean((r0-r0.mean())**2)+ln.eps)
                ideal=np.einsum('i,ij->j',
                    raw.double().numpy()-c['heads',l][0,-1,h*64:(h+1)*64].double().numpy(),
                    net.transformer.h[l].attn.c_proj.weight[h*64:(h+1)*64].double().numpy(),optimize=False)
                de=float(np.dot(gamma*(ideal-ideal.mean())/sigma,direction))
                stored=obs['crossed'][2*i+k]
                de_error=max(de_error,abs(de-stored['delta_DE_original_LN']))
                assert abs(de-stored['delta_DE_original_LN'])<1e-9
                def f0(r): return float(np.dot(gamma*(r-r.mean())/sigma+beta,direction))
                rounding=actual[1]-actual[0]-ideal
                predicted=float(np.dot(gamma*(rounding-rounding.mean())/sigma,direction))
                frozen_sr=f0(actual[1])-f0(actual[0])-de
                zero_error=max(zero_error,abs(frozen_sr-predicted));rounding_max=max(rounding_max,abs(frozen_sr))
                assert abs(frozen_sr-predicted)<1e-8
                residual_effect=f0(actual[2])-f0(actual[0])-de
                reader_change=(stored['values'][2]['centred_logit']-f0(actual[2]))-(stored['values'][0]['centred_logit']-f0(actual[0]))
                err=abs(stored['self_repair']-residual_effect-reader_change)
                allocation_error=max(allocation_error,err);assert err<1e-8;linear_checks+=3
                for condition,blocked in [(2,[]),(5,backups),(6,controls)]:
                    handles=[]
                    def target_hook(module,args):
                        u=args[0].clone();u[0,-1,h*64:(h+1)*64]=raw;return (u,)
                    handles.append(net.transformer.h[l].attn.c_proj.register_forward_pre_hook(target_hook))
                    for layer in sorted({a for a,b in blocked}):
                        selected=[b for a,b in blocked if a==layer]
                        def block_hook(module,args,layer=layer,selected=selected):
                            u=args[0].clone()
                            for head in selected:u[0,-1,head*64:(head+1)*64]=c['heads',layer][0,-1,head*64:(head+1)*64]
                            return (u,)
                        handles.append(net.transformer.h[layer].attn.c_proj.register_forward_pre_hook(block_hook))
                    try:official=net(torch.tensor([row['tokens'][0]]),use_cache=False).logits[0,-1]
                    finally:
                        for handle in handles:handle.remove()
                    expected=engine.decode(c,torch.tensor(states['crossed'][2*i+k,condition]))
                    err=float((official-expected).abs().max());official_max=max(official_max,err)
                    assert torch.equal(official,expected),('block full replay',i,k,condition,err)
                    finite_count+=int(torch.isfinite(official).sum());full_checks+=1
            # Full-forward access check: inject the saved donor boundary in place
            # at ln_2 input, which is also the residual operand of the MLP add.
            for other in (1,2):
                cs=[c,engine.native(row['tokens'][other])]
                zs=[engine.boundary(cc,l,h) for cc in cs]
                for recipient,donor in [(0,1),(1,0)]:
                    def boundary_hook(module,args,donor=donor):
                        args[0][0,-1]=zs[donor]
                    handle=net.transformer.h[l].ln_2.register_forward_pre_hook(boundary_hook)
                    try:official=net(torch.tensor([row['tokens'][0 if recipient==0 else other]]),use_cache=False).logits[0,-1]
                    finally:handle.remove()
                    index=next(index for index,r in enumerate(obs['E']) if r['family']==row['id'] and
                        r['other']==other and r['recipient']==recipient and r['donor']==donor and r['policy']=='regenerated')
                    expected=engine.decode(cs[recipient],torch.tensor(states['E'][index]))
                    err=float((official-expected).abs().max());official_max=max(official_max,err)
                    assert torch.equal(official,expected),('E full replay',i,other,recipient,donor,err)
                    full_checks+=1
        engine.close()
    result=dict(status='passed',stage=stage,head=suffix or 'L9H9',posthoc_software_audit=True,
        scientific_decision_rules_changed=False,full_official_forward_checks=full_checks,
        max_official_logit_error=official_max,independent_linear_checks=linear_checks,
        max_delta_DE_error=de_error,max_frozen_SR_minus_rounding_error=zero_error,
        max_frozen_SR_rounding_magnitude=rounding_max,max_reader_allocation_error=allocation_error,
        finite_official_logits_checked=finite_count,limitation='internal audit, four fixed families, no external replication')
    write('results/'+stage+suffix+'_supplementary_audit.json',result);check(stage);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='test');p.add_argument('--suffix',default='');a=p.parse_args()
    run(a.split,a.suffix)
