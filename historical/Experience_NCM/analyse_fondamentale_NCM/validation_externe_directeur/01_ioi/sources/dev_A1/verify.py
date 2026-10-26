"""Independent scalar arithmetic and official hooked-forward checks on stored sources."""
from support import *
from runtime import Engine
import argparse,math

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');a=p.parse_args();stage=a.split
    setup();check(stage);obs=read('results/'+stage+'_observations.json')
    rows=read('data/selected.json')[stage];data=np.load(D/'results'/f'{stage}_states.npz')
    net,tok=load();engine=Engine(net);checks=0;max_replay=0.;direct_hook_errors=[]
    with torch.inference_mode():
        for i in (0,1,2,3):
            row=rows[i];c=engine.native(row['tokens'][0]);native=engine.boundary(c,9,9)
            for k,offset in enumerate((37,73)):
                donor=rows[(i+offset)%len(rows)];dc=engine.native(donor['tokens'][0])
                raw=dc['heads',9][0,-1,9*64:10*64]
                z=engine.boundary(c,9,9,raw);tail,final,yy=engine.continuation(c,9,z)
                def hook(module,args):
                    u=args[0].clone();u[0,-1,9*64:10*64]=raw;return (u,)
                h=net.transformer.h[9].attn.c_proj.register_forward_pre_hook(hook)
                try:official=net(torch.tensor([row['tokens'][0]]),use_cache=False).logits[0,-1]
                finally:h.remove()
                delta=float((official-yy).abs().max());direct_hook_errors.append(delta)
                assert torch.equal(official,yy),'hooked full forward mismatch'
                index=2*i+k
                stored=torch.tensor(data['crossed'][index,2]);assert torch.equal(stored,final)
                for cond in range(7):
                    logits=engine.decode(c,torch.tensor(data['crossed'][index,cond])).double().numpy()
                    expected=obs['crossed'][index]['values'][cond]
                    m=float(max(logits));logsum=m+math.log(math.fsum(math.exp(float(t)-m) for t in logits))
                    loss=logsum-float(logits[row['choices'][0]])
                    error=abs(loss-expected['nll']);max_replay=max(max_replay,error)
                    assert error<1e-10
                    assert bool(np.argmax(logits)==row['choices'][0])==expected['correct_vocab'];checks+=1
                for tail_index,state_index,start in ((0,0,native),(0,1,z),(1,2,z),(1,3,native),(2,4,z),(3,5,z),(4,6,z)):
                    recomposed=start.clone()
                    for ti,t in enumerate(data['tails'][index,tail_index]):
                        t=torch.tensor(t);recomposed=t+recomposed if ti%2 else recomposed+t
                    assert torch.equal(recomposed,torch.tensor(data['crossed'][index,state_index]));checks+=1
        engine.close()
    # Recompute family damage and gate by scalar loops rather than analysis vectorization.
    ds=[]
    for i in range(len(rows)):
        vals=[]
        for k in range(2):
            o=obs['crossed'][2*i+k]['values'];vals.append(o[1]['nll']-o[0]['nll'])
        ds.append(sum(vals)/2)
    assert abs(sum(ds)/len(ds)-obs['gate']['mean_damage'])<1e-12
    assert (sum(ds)/len(ds)>=.05 and sum(d>.01 for d in ds)/len(ds)>=.25)==obs['gate']['passed']
    write('results/'+stage+'_verification.json',dict(passed=True,scalar_metric_and_recomposition_checks=checks,
        hooked_full_forward_examples=len(direct_hook_errors),hooked_max_errors=direct_hook_errors,
        max_scalar_nll_error=max_replay,gate_recomputed=True,finished_utc=now(),
        limitation='internal software audit, not independent scientific replication'))
    check(stage);print('verification passed',checks)
