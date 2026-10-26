from common import *


def analyse():
    check();out=[];folder=D/'results/evaluation'
    rows=[r for r in read('data/families.json')['rows'] if r['split']=='evaluation']
    for architecture in Q['architectures']:
        for seed in Q['seeds']:
            n=np.load(folder/f'{architecture}-{seed}-native.npz');refs=n['refs'];pred=n['logits'].argmax(-1)
            O=dict(first_exact_by_variant=(pred[:,:,-1]==refs[:,:,0]).sum(axis=0).tolist(),n_per_variant=40,
                   sequence_exact_by_variant=(n['answers']==refs).all(axis=-1).sum(axis=0).tolist(),
                   first_exact_by_readout=(pred==refs[:,:,0,None]).sum(axis=(0,1)).tolist())
            for cut in Q['cuts']:
                c=np.load(folder/f'{architecture}-{seed}-cut{cut}-C.npz')
                logits=c['logits'];t=np.broadcast_to(refs[:,:2,0,None,None],(40,2,4,5))
                losses=logits_loss(logits,t);correct=logits.argmax(-1)==t
                contrasts=np.stack((losses[...,1]-losses[...,0],losses[...,1]-losses[...,2],
                         losses[...,1]+losses[...,3]-losses[...,2]-losses[...,0],losses[...,4]-losses[...,2]),axis=-1).mean(axis=(1,2))
                C={k:ci(contrasts[:,j],seed*100+cut*10+j) for j,k in enumerate(('D','R','I','S'))}
                if all(C[k]['ci_adjusted'][0]>.01 for k in ('D','R','I','S')):decision='useful_specific_recovery_supported_at_fixed_margin'
                elif all(abs(v)<.01 for v in C['D']['ci_adjusted']):decision='mean_frozen_damage_equivalent_within_prespecified_margin'
                else:decision='insufficient_or_not_confirmed'
                C.update(decision=decision,secondary_descriptive=cut!=2,quality={key:dict(n=320,exact=int(correct[...,j].sum()),
                   losses_vs_native=int((correct[...,0]&~correct[...,j]).sum()),gains_vs_native=int((~correct[...,0]&correct[...,j]).sum()))
                       for j,key in enumerate(('Y00','Y10','Y11','Y01','Y1wrong'))},per_family_contrasts=contrasts.tolist())
                e=np.load(folder/f'{architecture}-{seed}-cut{cut}-E.npz');el=e['logits'];ep=el.argmax(-1);E={}
                for track_index,track in enumerate(('retained','whole')):
                    h=np.asarray([el[f,track_index,0,...,refs[f,1,0]].astype(np.float64)-el[f,track_index,0,...,refs[f,0,0]].astype(np.float64) for f in range(40)])
                    hreg=h[...,2];aa,ab,ba,bb=hreg[:,0,0],hreg[:,0,1],hreg[:,1,0],hreg[:,1,1]
                    u=.5*((ba-aa)+(bb-ab));v=.5*((ab-aa)+(bb-ba));w=bb-ba-ab+aa;g=bb-aa
                    counts={};details=[]
                    for f,row in enumerate(rows):
                        exact=(pred[f,:2,-1]==refs[f,:2,0]).tolist();name=label(exact,g[f],w[f],u[f]);counts[name]=counts.get(name,0)+1
                        details.append(dict(source_id=row['source_id'],label=name,u=float(u[f]),v=float(v[f]),w=float(w[f]),g=float(g[f]),
                                   input_ratio=float(u[f]/g[f]) if abs(g[f])>1e-12 else None,native_exact=exact))
                    quality={}
                    for comparison,branches in enumerate(((0,1),(0,2))):
                        policies={}
                        for mode in ('recipient_native','donor_native','regenerated'):
                            rc=dc=other=gains=lost=0
                            for f in range(40):
                                for r in range(2):
                                    d=1-r;p={'recipient_native':r,'donor_native':d,'regenerated':2}[mode]
                                    actual=ep[f,track_index,comparison,r,d,p];target=refs[f,branches[r],0];donor=refs[f,branches[d],0]
                                    ok=actual==target;baseline=ep[f,track_index,comparison,r,d,r]==target
                                    rc+=int(ok);dc+=int(actual==donor);other+=int(actual not in (target,donor))
                                    gains+=int(ok and not baseline);lost+=int(baseline and not ok)
                            policies[mode]=dict(n=80,recipient_exact=rc,donor_exact=dc,other=other,
                                         gains_vs_recipient_native=gains,losses_vs_recipient_native=lost)
                        quality[('answer_change','answer_preserve')[comparison]]=policies
                    invariant=[]
                    for f in range(40):
                        for r in range(2):
                            baseline=el[f,track_index,1,r,r,2].astype(np.float64)
                            swapped=el[f,track_index,1,r,1-r,2].astype(np.float64)
                            def prob(v):
                                p=np.exp(v-v.max());return p/p.sum()
                            invariant.append(dict(source_id=rows[f]['source_id'],recipient=r,
                                nll_difference=float(logits_loss(swapped,refs[f,0,0])-logits_loss(baseline,refs[f,0,0])),
                                probability_TV=float(.5*abs(prob(swapped)-prob(baseline)).sum()),
                                argmax_changed=bool(swapped.argmax()!=baseline.argmax())))
                    E[track]=dict(labels=counts,per_family=details,u=ci(u,seed+1000+cut),v=ci(v,seed+2000+cut),
                                  w=ci(w,seed+3000+cut),quality=quality,answer_preserving=invariant)
                out.append(dict(architecture=architecture,seed=seed,cut=cut,primary=cut==2,O=O,C=C,E=E))
    write('results/summary.json',dict(status='complete',units='40 source orbits per training; seeds reported separately',
             calibration=read('results/calibration.json'),cells=out))
    check();print('Scored 12 architecture/seed/cut cells without pooling seeds.')


if __name__=='__main__':analyse()
