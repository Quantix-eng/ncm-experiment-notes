from model import *
import math
import itertools


def scalar_loss(v,t):
    m=float(max(v));return m+math.log(math.fsum(math.exp(float(a)-m) for a in v))-float(v[t])


def explicit(net,h,cut):
    h=h.clone();residuals=[]
    for block in net.blocks[cut:]:
        v=F.layer_norm(h,(48,),block.ln1.weight,block.ln1.bias,block.ln1.eps)
        q,k,value=F.linear(v,block.qkv.weight,block.qkv.bias).reshape(1,11,3,4,12).permute(2,0,3,1,4).unbind(0)
        weights=((q@k.transpose(-1,-2))/(12**.5)).masked_fill(~net.mask,float('-inf')).softmax(-1)
        a=F.linear((weights@value).transpose(1,2).reshape(1,11,48),block.proj.weight,block.proj.bias)
        h=h+a
        v=F.layer_norm(h,(48,),block.ln2.weight,block.ln2.bias,block.ln2.eps)
        m=F.linear(F.gelu(F.linear(v,block.fc1.weight,block.fc1.bias)),block.fc2.weight,block.fc2.bias)
        h=h+m;residuals.extend((a,m))
    return h,torch.stack(residuals)


def canonical(v):
    return min(tuple(np.roll(w,j).tolist()) for w in (np.array(v),np.array(v)[::-1]) for j in range(4))


@torch.inference_mode()
def verify():
    check()
    for p,h in read('results/calibration_frozen.json')['files'].items():assert sha(p)==h,p
    rows=read('data/families.json')['rows'];groups=[canonical(r['canonical']) for r in rows]
    assert len(rows)==len(set(groups))==210
    assert [sum(r['split']==s for r in rows) for s in ('train','development','evaluation')]==[140,30,40]
    all_members=[tuple(v) for r in rows for v in r['orbit']]
    assert len(set(all_members))==1680
    train_groups={canonical(r['canonical']) for r in rows if r['split']=='train'}
    data=np.load(D/'data/train.npz')
    for x,y in zip(data['x'],data['y']):
        assert canonical(x[-4:]) in train_groups
        b=int(x[1])-9;assert np.array_equal(y,np.roll(x[-4:],-b))
    training=read('results/training.json')['models']
    assert len(training)==6
    for seed in Q['seeds']:
        pair=[r for r in training if r['seed']==seed]
        assert len({r['initial_parameter_sha256'] for r in pair})==1
        assert len({r['batch_indices_sha256'] for r in pair})==1
        assert len({r['parameters'] for r in pair})==1
        assert all(len(r['all_losses'])==1200 and np.isfinite(r['all_losses']).all() for r in pair)
    calibration=read('results/calibration.json')
    for record in calibration['per_model']:
        arch,seed=record['architecture'],record['seed']
        d=np.load(D/f'results/development-{arch}-{seed}.npz')
        baseline=np.array([scalar_loss(v,int(t)) for v,t in zip(d['native_logits'].reshape(-1,8),d['refs'].reshape(-1))]).reshape(30,2)
        acc=(d['native_logits'].argmax(-1)==d['refs']).mean()
        assert float(acc)==record['native_first_accuracy']
        for i,dose in enumerate(record['doses']):
            targets=np.broadcast_to(d['refs'][...,None],(30,2,4))
            l=np.array([scalar_loss(v,int(t)) for v,t in zip(d['frozen_logits'][i].reshape(-1,8),targets.reshape(-1))]).reshape(30,2,4)
            drop=acc-(d['frozen_logits'][i].argmax(-1)==d['refs'][...,None]).mean()
            assert abs(float((l-baseline[...,None]).mean())-dose['damage_nll'])<1e-12
            assert abs(float(drop)-dose['accuracy_drop'])<1e-12
    for i,candidate in enumerate(calibration['all_candidates']):
        for arch in Q['architectures']:
            values=[r['doses'][i] for r in calibration['per_model'] if r['architecture']==arch]
            for key in ('damage_nll','accuracy_drop'):
                assert abs(candidate['architectures'][arch][key]-np.mean([v[key] for v in values]))<1e-12
        assert candidate['admissible']==all(.1<=v['accuracy_drop']<=.4 and v['damage_nll']>=.05 for v in candidate['architectures'].values())
    expected=next((v['rho'] for v in calibration['all_candidates'] if v['admissible']),3.)
    assert calibration['selected_rho']==expected
    folder=D/'results/evaluation';counts=dict(decoder_recompositions=0,explicit_regenerations=0,scalar_losses=0,native_replays=0,blocked_path_cases=0)
    evaluation=[r for r in rows if r['split']=='evaluation']
    summary=read('results/summary.json')
    for cell in summary['cells']:
        architecture,seed,cut=cell['architecture'],cell['seed'],cell['cut']
        net=load(architecture,seed)
        n=np.load(folder/f'{architecture}-{seed}-native.npz')
        c=np.load(folder/f'{architecture}-{seed}-cut{cut}-C.npz');e=np.load(folder/f'{architecture}-{seed}-cut{cut}-E.npz')
        states=n['states'];tails=n['tails'];refs=n['refs'];pred=n['logits'].argmax(-1)
        cl,cd,ct,cf=[c[k] for k in ('logits','deltas','tails','finals')]
        el,et,ef=[e[k] for k in ('logits','tails','finals')]
        def decode(h):return F.linear(F.layer_norm(h,(48,),net.norm.weight,net.norm.bias,net.norm.eps),net.head.weight,net.head.bias)[0,7]
        def add(h,t):
            h=h.clone()
            for delta in t:h=h+delta
            return h
        scalar=np.zeros((40,2,4,5))
        for f in range(40):
            for anchor in range(2):
                h=torch.from_numpy(states[f,anchor,cut].copy());t0=torch.from_numpy(tails[f,anchor,2*cut:].copy())
                for q in range(4):
                    delta=torch.from_numpy(cd[f,anchor,q].copy())
                    assert torch.count_nonzero(delta[:,:7])==0
                    other_half=24 if q<2 else 0
                    assert torch.count_nonzero(delta[:,:,other_half:other_half+24])==0
                    assert np.array_equal(cd[f,anchor,q^1],-cd[f,anchor,q])
                    assert math.isclose(float(delta.double().norm()),expected*float(h[:,7:].double().norm()),rel_tol=2e-6)
                    adapted=torch.from_numpy(ct[f,anchor,q].copy());wrong=torch.from_numpy(ct[f,anchor,q^2].copy())
                    for j,(z,t) in enumerate(zip((h,h+delta,h+delta,h,h+delta),(t0,t0,adapted,adapted,wrong))):
                        final=add(z,t)
                        assert np.array_equal(final.numpy(),cf[f,anchor,q,j])
                        assert np.array_equal(decode(final).numpy(),cl[f,anchor,q,j])
                        scalar[f,anchor,q,j]=scalar_loss(cl[f,anchor,q,j],int(refs[f,anchor,0]))
                        counts['decoder_recompositions']+=1;counts['scalar_losses']+=1
                    if f<2:
                        final,tt=explicit(net,h+delta,cut)
                        assert torch.equal(tt,adapted) and np.array_equal(final.numpy(),cf[f,anchor,q,2])
                        counts['explicit_regenerations']+=1
            if f<2:
                for a in range(3):
                    prefix=torch.tensor([evaluation[f]['inputs'][a]])
                    logits=net(torch.cat((prefix,fixed_queries(1)),dim=1))
                    assert np.array_equal(logits[0,0].numpy(),n['logits'][f,a,-1])
                    assert np.array_equal(net.generate(prefix)[0].numpy(),n['answers'][f,a])
                    counts['native_replays']+=1
            for track_index,track in enumerate(('retained','whole')):
                for comparison,branches in enumerate(((0,1),(0,2))):
                    for r,d in itertools.product(range(2),repeat=2):
                        h=torch.from_numpy(states[f,branches[d],cut].copy())
                        if track=='retained':h[:,:7]=torch.from_numpy(states[f,branches[r],cut,:,:7].copy())
                        ts=[torch.from_numpy(tails[f,a,2*cut:].copy()) for a in branches]+[torch.from_numpy(et[f,track_index,comparison,r,d].copy())]
                        for policy,t in enumerate(ts):
                            final=add(h,t)
                            assert np.array_equal(final.numpy(),ef[f,track_index,comparison,r,d,policy])
                            assert np.array_equal(decode(final).numpy(),el[f,track_index,comparison,r,d,policy])
                            counts['decoder_recompositions']+=1
                        if track=='whole':
                            assert np.array_equal(el[f,track_index,comparison,r,d,2],n['logits'][f,branches[d],-1])
                            counts['blocked_path_cases']+=1
                        if f<2:
                            final,tt=explicit(net,h,cut)
                            assert torch.equal(tt,ts[2]) and np.array_equal(final.numpy(),ef[f,track_index,comparison,r,d,2])
                            counts['explicit_regenerations']+=1
        contrasts=np.stack((scalar[...,1]-scalar[...,0],scalar[...,1]-scalar[...,2],scalar[...,1]+scalar[...,3]-scalar[...,2]-scalar[...,0],scalar[...,4]-scalar[...,2]),axis=-1).mean(axis=(1,2))
        assert np.allclose(contrasts,cell['C']['per_family_contrasts'],atol=1e-12,rtol=0)
        correct=cl.argmax(-1)==refs[:,:2,0,None,None]
        for j,key in enumerate(('Y00','Y10','Y11','Y01','Y1wrong')):
            assert cell['C']['quality'][key]==dict(n=320,exact=int(correct[...,j].sum()),
                losses_vs_native=int((correct[...,0]&~correct[...,j]).sum()),gains_vs_native=int((~correct[...,0]&correct[...,j]).sum()))
        for j,key in enumerate(('D','R','I','S')):
            indices=np.random.default_rng(seed*100+cut*10+j).integers(40,size=(20000,40))
            b=contrasts[:,j][indices].mean(axis=1)
            assert np.allclose(np.quantile(b,[.00625,.99375]),cell['C'][key]['ci_adjusted'],atol=1e-12,rtol=0)
        for track_index,track in enumerate(('retained','whole')):
            for f,row in enumerate(cell['E'][track]['per_family']):
                h=el[f,track_index,0,:,:,2,refs[f,1,0]].astype(float)-el[f,track_index,0,:,:,2,refs[f,0,0]].astype(float)
                aa,ab,ba,bb=h[0,0],h[0,1],h[1,0],h[1,1]
                u=((ba-aa)+(bb-ab))/2;v=((ab-aa)+(bb-ba))/2;w=bb-ba-ab+aa;g=bb-aa
                assert all(abs(row[k]-value)<1e-12 for k,value in (('u',u),('v',v),('w',w),('g',g)))
                if track=='whole':assert u==w==0
                exact=(pred[f,:2,-1]==refs[f,:2,0]).all()
                if not exact:name='non_identifiable_native_anchor_failure'
                elif g<=.1:name='non_identifiable_weak_native_preference_span'
                elif abs(w/g)>.2:name='non_additive_interaction_not_a_mechanism_class'
                elif not 0<=u/g<=1:name='non_monotone_non_identifiable'
                elif u/g>=.8:name='input_dominant_response_preference_in_this_test'
                elif u/g<=.2:name='state_dominant_response_preference_in_this_test'
                else:name='two_sensitivities_mixed_or_unmodeled'
                assert row['label']==name
            for comparison,branches in enumerate(((0,1),(0,2))):
                for mode,reported in cell['E'][track]['quality'][('answer_change','answer_preserve')[comparison]].items():
                    rc=dc=other=gains=lost=0
                    for f in range(40):
                        for r in range(2):
                            d=1-r;policy=r if mode=='recipient_native' else d if mode=='donor_native' else 2
                            actual=int(el[f,track_index,comparison,r,d,policy].argmax())
                            target=int(refs[f,branches[r],0]);donor=int(refs[f,branches[d],0])
                            baseline=int(el[f,track_index,comparison,r,d,r].argmax())==target
                            ok=actual==target;rc+=int(ok);dc+=int(actual==donor);other+=int(actual not in (target,donor))
                            gains+=int(ok and not baseline);lost+=int(baseline and not ok)
                    assert reported==dict(n=80,recipient_exact=rc,donor_exact=dc,other=other,gains_vs_recipient_native=gains,losses_vs_recipient_native=lost)
        assert cell['O']['first_exact_by_variant']==(pred[:,:,-1]==refs[:,:,0]).sum(axis=0).tolist()
        assert cell['O']['sequence_exact_by_variant']==(n['answers']==refs).all(axis=-1).sum(axis=0).tolist()
    historical={}
    for directory in ('benchmark_multimodel_ar','article_benchmark_coordination','benchmark_ncm_reel'):
        base=D.parent/directory;manifest=json.loads((base/'results/final_manifest.json').read_text())
        for p,h in manifest['files'].items():assert sha(base/p)==h,(directory,p)
        historical[directory]=len(manifest['files'])
    check()
    write('results/verification.json',dict(status='passed',counts=counts,historical_integrity=historical,
          scope='Same-agent internal software audit, not independent scientific replication',
          files_frozen_before_training=True,calibration_fixed_before_evaluation=True))
    print(json.dumps(read('results/verification.json'),indent=2))


if __name__=='__main__':verify()
