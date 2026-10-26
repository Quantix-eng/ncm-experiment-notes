"""Development gate first. Test is rejected unless separately frozen and dev passes."""
from support import *
from runtime import Engine
import argparse

BACKUPS=[(10,1),(10,10),(10,6),(10,2),(11,9),(11,2)]
CONTROLS=[(10,0),(10,3),(10,4),(10,5),(11,0),(11,1)]

def evaluate(split,l=9,h=9):
    setup();check(split);started=time.perf_counter()
    rows=read('data/selected.json')[split];net,tok=load();engine=Engine(net)
    suffix='' if (l,h)==(9,9) else f'_L{l}H{h}'
    native=[];bank=[];native_z=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            vals=[];zs=[];heads=[]
            for tokens,answer in zip(row['tokens'],row['answer_keys']):
                c=engine.native(tokens)
                z=engine.boundary(c,l,h);tail,result,logits=engine.continuation(c,l,z)
                assert torch.equal(tail,engine.tail_native(c,l))
                assert torch.equal(logits,c['logits']),'sham'
                vals.append(metric(logits.numpy(),row['choices'],answer));zs.append(result.numpy())
                heads.append(c['heads',l][0,-1,h*64:(h+1)*64].clone())
            native.append(vals);native_z.append(zs);bank.append(heads[0])
            if (i+1)%32==0:print(f'{split} native {i+1}/{len(rows)}',flush=True)
        observations=[];state_records=[];tail_records=[];e_records=[];e_states=[]
        for i,row in enumerate(rows):
            c=engine.native(row['tokens'][0]);z0=engine.boundary(c,l,h);t0=engine.tail_native(c,l)
            donors=[(i+37)%len(rows),(i+73)%len(rows)]
            zs=[engine.boundary(c,l,h,bank[j]) for j in donors]
            adapted=[engine.continuation(c,l,z) for z in zs]
            for k,(j,z) in enumerate(zip(donors,zs)):
                t1,result,y11=adapted[k]
                states=[engine.replay(z0,t0),engine.replay(z,t0),result,
                        engine.replay(z0,t1),engine.replay(z,adapted[1-k][0])]
                logits=[engine.decode(c,s) for s in states]
                assert torch.equal(logits[0],c['logits']);assert torch.equal(logits[2],y11)
                backup=engine.continuation(c,l,z,[(a,b) for a,b in BACKUPS if a>l])
                control=engine.continuation(c,l,z,[(a,b) for a,b in CONTROLS if a>l])
                states.extend([backup[1],control[1]])
                values=[metric(q.numpy(),row['choices'],0) for q in logits+[backup[2],control[2]]]
                # Measurement only: FP64 projection avoids a reproduced local FP32 GEMV NaN.
                delta=(bank[j].double()-c['heads',l][0,-1,h*64:(h+1)*64].double())@net.transformer.h[l].attn.c_proj.weight[h*64:(h+1)*64].double()
                de=engine.direct_effect(c,delta,row['choices'][0])
                sr=values[2]['centred_logit']-values[0]['centred_logit']-de
                assert np.isfinite([de,sr]).all(),('nonfinite effect',i,k,de,sr)
                assert all(np.isfinite([v['nll'],v['margin'],v['centred_logit']]).all() for v in values),('nonfinite crossed',i,k,values)
                observations.append(dict(family=row['id'],template=row['template'],donor=rows[j]['id'],
                    wrong_donor=rows[donors[1-k]]['id'],values=values,delta_DE_original_LN=de,self_repair=sr))
                state_records.append(torch.stack(states).numpy())
                tail_records.append(torch.stack([t0,t1,adapted[1-k][0],backup[0],control[0]]).numpy())
            # E: A/B and A/Aplus; held native tails aligned explicitly.
            for other in (1,2):
                cs=[c,engine.native(row['tokens'][other])]
                boundaries=[engine.boundary(q,l,h) for q in cs]
                tails=[engine.tail_native(q,l) for q in cs]
                for recipient in range(2):
                    for donor in range(2):
                        for policy in ('fixed_left','fixed_right','regenerated'):
                            if policy=='regenerated':
                                tt,zz,yy=engine.continuation(cs[recipient],l,boundaries[donor])
                            else:
                                tt=tails[0 if policy=='fixed_left' else 1]
                                zz=engine.replay(boundaries[donor],tt);yy=engine.decode(cs[recipient],zz)
                            ans=0 if recipient==0 else row['answer_keys'][other]
                            assert bool(torch.isfinite(yy).all()),('nonfinite E logits',i,other,recipient,donor,policy)
                            e_records.append(dict(family=row['id'],other=other,recipient=recipient,donor=donor,
                                policy=policy,values=metric(yy.numpy(),row['choices'],ans),
                                preference_B_minus_A=float(yy[row['choices'][1]]-yy[row['choices'][0]])))
                            e_states.append(zz.numpy())
            if (i+1)%16==0:print(f'{split} interventions {i+1}/{len(rows)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
        engine.close()
    losses=np.array([[o['values'][k]['nll'] for k in range(7)] for o in observations])
    Dv=(losses[:,1]-losses[:,0]).reshape(len(rows),2).mean(1)
    gate=dict(mean_damage=float(Dv.mean()),fraction_damage_over_001=float((Dv>.01).mean()),
        passed=bool(Dv.mean()>=.05 and (Dv>.01).mean()>=.25))
    np.savez_compressed(D/'results'/f'{split}{suffix}_states.npz',native=np.array(native_z),
        crossed=np.array(state_records),tails=np.array(tail_records),E=np.array(e_states))
    write('results/'+split+suffix+'_observations.json',dict(native=native,crossed=observations,E=e_records,
        conditions=['Y00','Y10','Y11','Y01','Y1wrong','block_backups','block_controls'],
        gate=gate,elapsed_seconds=time.perf_counter()-started,n_families=len(rows),finished_utc=now()))
    check(split)
    print(json.dumps(gate),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--layer',type=int,default=9);p.add_argument('--head',type=int,default=9)
    args=p.parse_args()
    if args.split=='test':
        assert read('results/dev_observations.json')['gate']['passed'],'Development gate failed'
        assert (D/'PROTOCOLE_TEST.md').exists(),'No sealed test protocol'
    assert (args.layer,args.head) in [(9,9),(9,6),(10,0)]
    assert args.split=='test' or (args.layer,args.head)==(9,9),'Development primary only'
    evaluate(args.split,args.layer,args.head)
