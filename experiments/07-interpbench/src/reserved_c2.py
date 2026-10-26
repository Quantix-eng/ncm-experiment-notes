"""Fixed invariance predictions, sealed before counterfactual falsification.

No mechanism-class labels. Oracle references and learned future outputs remain
separate; compiler auxiliaries without semantic masks are outside the domain.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,argparse
import numpy as np
import torch
from official_audit_a1 import build
from pilot import vocabulary,CASES
from craft_development_b1 import patch_hook
from craft_oracle_c2 import CraftOracle

D=Path(__file__).resolve().parent
TASKS=[13,18,19,21]
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):(D/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads((D/p).read_text())
def tokens(row):
    mapping={v:i for i,v in enumerate(vocabulary(row['case']))}
    return torch.tensor([[mapping['BOS']]+[mapping[t] for t in row['tokens']]])
def all_rows():return read('data/reserved_inputs.json')
def key(v):return f"L{v[0]}_{v[1]}_{v[2]}"
def prepare():
    plan={};sites={};aux={}
    for case in TASKS:
        oracle=CraftOracle(case);schema=oracle.schema();cfg=CASES[str(case)]['transformer_cfg']
        complete={(r['layer'],r['kind'],r['head']) for r in schema}
        selected=[dict(variable=[r['layer'],r['kind'],r['head']],type='aligned')
                  for r in schema if r['all_labels_found']]
        aux[str(case)]=[r for r in schema if not r['all_labels_found']]
        for layer in range(cfg['n_layers']):
            absent=[h for h in range(cfg['n_heads']) if (layer,'attn',h) not in complete]
            if absent:selected.append(dict(variable=[layer,'attn',absent[0]],type='unaligned'))
            if (layer,'mlp',None) not in complete:
                selected.append(dict(variable=[layer,'mlp',None],type='unaligned'))
        assert len({tuple(r['variable']) for r in selected})==len(selected)
        sites[str(case)]=selected
        subset=[r for r in all_rows() if r['case']==case]
        for row in subset:
            group=[r for r in subset if len(r['tokens'])==len(row['tokens'])]
            assert len(group)>=4,(case,len(row['tokens']),len(group))
            i=next(i for i,r in enumerate(group) if r['id']==row['id'])
            plan[row['id']]=[group[(i+k)%len(group)]['id'] for k in (1,2,3)]
    write('data/c2_plan.json',dict(sites=sites,excluded_auxiliaries=aux,donors=plan))
    files=list(D.glob('*c2.py'))+[D/'PROTOCOLE_C2.md',D/'data/c2_plan.json',D/'data/reserved_inputs.json',
        D/'official_audit_a1.py',D/'pilot.py',D/'craft_development_b1.py',D/'craft_oracle_b1.py']
    files+=list((D/'sources/history_audit').glob('*'))
    for seal in ['results/craft_b1_frozen.json','results/amendment_a1_frozen.json']:
        for p,h in read(seal)['sha256'].items():
            p=Path(p) if Path(p).is_absolute() else D/p
            assert sha(p)==h,p;files.append(p)
    write('results/c2_frozen.json',dict(timestamp_utc=stamp(),scope='reserved C2 before neural outputs',
        files={str(p.resolve()):sha(p) for p in files if p.is_file()},
        sites=sites,auxiliary_counts={c:len(a) for c,a in aux.items()}))
    print('C2 sealed; sites', {c:len(a) for c,a in sites.items()},flush=True)

def check():
    for p,h in read('results/c2_frozen.json')['files'].items():assert sha(Path(p))==h,p

def tail_names(model,variable):
    layer,kind,_=variable
    names=[f'blocks.{layer}.hook_mlp_out'] if kind=='attn' else []
    return names+[n for j in range(layer+1,model.cfg.n_layers)
                  for n in (f'blocks.{j}.hook_attn_out',f'blocks.{j}.hook_mlp_out')]
def frozen(cache,names):
    return [(name,lambda value,hook,kept=cache[name]:kept) for name in names]
def cut_name(v):return f"blocks.{v[0]}.hook_resid_{'mid' if v[1]=='attn' else 'post'}"
def arrays(logits,n):
    assert bool(torch.isfinite(logits).all())
    return logits[0,1:n+1].argmax(-1).numpy().tolist()
def loss(logits,targets):
    indices=[i for i,t in enumerate(targets) if t is not None]
    if not indices:return None
    z=logits[0,torch.tensor(indices)+1].double();t=torch.tensor([targets[i] for i in indices])
    return float((torch.logsumexp(z,dim=-1)-z[torch.arange(len(t)),t]).mean())
def predict(probe_matches):
    # Only Boolean matches of this query's authorized probe outputs enter here.
    return dict(O_prudent=0.,O_naive=1.,C_without_frozen=float(np.mean([p[2] for p in probe_matches])),
        C_without_mismatch=float(np.mean([p[:4] for p in probe_matches])),
        C_full=float(np.mean(probe_matches)))

@torch.inference_mode()
def probes():
    check();torch.set_num_threads(1);start=time.perf_counter();plan=read('data/c2_plan.json')
    native_rows=[];observations=[];predictions=[];saved={};shams=0
    for case in TASKS:
        oracle=CraftOracle(case);model,_=build(case,torch.float32)
        subset=[r for r in all_rows() if r['case']==case];mapping={r['id']:r for r in subset}
        native={r['id']:model.run_with_cache(tokens(r)) for r in subset}
        for row in subset:
            values=oracle.semantic(row['tokens']);targets=oracle.target_ids(values)
            cscores,_=oracle.native(row['tokens']);mask=np.array([t is not None for t in targets])
            expected=np.array([0 if t is None else t for t in targets])
            craft_ok=bool(np.array_equal(cscores[1:].argmax(-1)[mask],expected[mask]))
            assert craft_ok,('native Craft/RASP',row['id'])
            z,cache=native[row['id']];actual=arrays(z,len(targets));saved[row['id']+'-native']=z.numpy()
            native_rows.append(dict(id=row['id'],case=case,targets=targets,learned_prediction=actual,
                exact_defined=bool(np.array_equal(np.array(actual)[mask],expected[mask])),defined=int(mask.sum())))
            for spec in plan['sites'][str(case)]:
                variable=tuple(spec['variable']);names=tail_names(model,variable);matches=[];probe_records=[]
                y00=model.run_with_hooks(tokens(row),fwd_hooks=frozen(cache,names))
                assert torch.equal(y00,z),'native tail replay'
                sham=model.run_with_hooks(tokens(row),fwd_hooks=[patch_hook(variable,cache)]+frozen(cache,names))
                assert torch.equal(sham,z),'native head/MLP sham';shams+=1
                ds=plan['donors'][row['id']]
                adapted=[]
                for donor in ds[:2]:
                    with model.hooks(fwd_hooks=[patch_hook(variable,native[donor][1])]):
                        yy,tt=model.run_with_cache(tokens(row))
                    adapted.append((yy,tt))
                qid=row['id']+'-'+key(variable)
                for k,donor in enumerate(ds[:2]):
                    outputs=[y00,model.run_with_hooks(tokens(row),fwd_hooks=[patch_hook(variable,native[donor][1])]+frozen(cache,names)),
                        adapted[k][0],model.run_with_hooks(tokens(row),fwd_hooks=frozen(adapted[k][1],names)),
                        model.run_with_hooks(tokens(row),fwd_hooks=[patch_hook(variable,native[donor][1])]+frozen(adapted[1-k][1],names))]
                    cats=[arrays(v,len(targets)) for v in outputs]
                    matches.append([v==actual for v in cats]);ls=[loss(v,targets) for v in outputs]
                    contrasts=None if ls[0] is None else dict(D=ls[1]-ls[0],R=ls[1]-ls[2],I=ls[1]+ls[3]-ls[2]-ls[0],S=ls[4]-ls[2])
                    probe_records.append(dict(donor=donor,argmax=cats,nll_on_native_defined=ls,contrasts=contrasts))
                    for j,v in enumerate(outputs):saved[qid+f'-probe{k}-Y{j}']=v.numpy()
                observations.append(dict(query=qid,source=row['id'],case=case,variable=list(variable),type=spec['type'],probes=probe_records))
                predictions.append(dict(query=qid,source=row['id'],case=case,variable=list(variable),type=spec['type'],
                    future_donor=ds[2],prediction=actual,confidence=predict(matches)))
        print('C2 probes task',case,'sources',len(subset),'elapsed',round(time.perf_counter()-start,2),flush=True)
        del model,native
    write('results/c2_native.json',native_rows);write('results/c2_probes.json',observations)
    write('results/c2_predictions.json',predictions)
    np.savez_compressed(D/'results/c2_probe_logits.npz',**saved)
    write('results/c2_predictions_frozen.json',dict(timestamp_utc=stamp(),future_outputs_computed=False,
        n_predictions=len(predictions),native_replays_and_shams=shams,elapsed_seconds=time.perf_counter()-start,
        files={p:sha(D/p) for p in ['results/c2_predictions.json','results/c2_native.json','results/c2_probes.json','data/c2_plan.json']}))
    check()

@torch.inference_mode()
def falsify():
    check();torch.set_num_threads(1);start=time.perf_counter()
    for p,h in read('results/c2_predictions_frozen.json')['files'].items():assert sha(D/p)==h,p
    predictions=read('results/c2_predictions.json');plan=read('data/c2_plan.json');rows=[];Es=[];saved={}
    for case in TASKS:
        oracle=CraftOracle(case);model,_=build(case,torch.float32)
        subset=[r for r in all_rows() if r['case']==case];mapping={r['id']:r for r in subset}
        native={r['id']:model.run_with_cache(tokens(r)) for r in subset}
        craft={r['id']:oracle.native(r['tokens']) for r in subset};e_seen=set()
        for pred in [p for p in predictions if p['case']==case]:
            row=mapping[pred['source']];donor=mapping[pred['future_donor']];v=tuple(pred['variable'])
            semantic=oracle.semantic(row['tokens'],v,donor['tokens']);targets=oracle.target_ids(semantic)
            mask=np.array([t is not None for t in targets]);expected=np.array([0 if t is None else t for t in targets])
            patch=None if v not in craft[donor['id']][1] else (v,craft[donor['id']][1][v])
            cscores,_=oracle.native(row['tokens'],patch)
            oracle_ok=bool(np.array_equal(cscores[1:].argmax(-1)[mask],expected[mask]))
            yy=model.run_with_hooks(tokens(row),fwd_hooks=[patch_hook(v,native[donor['id']][1])])
            actual=arrays(yy,len(targets));model_ok=bool(np.array_equal(np.array(actual)[mask],expected[mask]))
            candidate=np.array(pred['prediction']);valid=bool(mask.any()) and oracle_ok
            rows.append(dict(query=pred['query'],case=case,type=pred['type'],variable=pred['variable'],
                source=row['id'],donor=donor['id'],oracle_targets=targets,defined_mask=mask.tolist(),
                craft_RASP_agreement=oracle_ok,scorable_oracle=valid,learned_oracle_agreement=model_ok,
                learned_prediction=actual,confidence=pred['confidence'],
                prediction_contradicted_by_network=pred['prediction']!=actual,
                prediction_contradicted_by_oracle=None if not valid else not bool(np.array_equal(candidate[mask],expected[mask])),
                changed_oracle_defined=None if not valid else bool(np.any(expected[mask]!=np.array(oracle.target_ids(oracle.semantic(row['tokens'])),dtype=object)[mask]))))
            saved[pred['query']+'-future']=yy.numpy()
            # E is a distinct whole-memory access check, never used by predict().
            ekey=(row['id'],donor['id'],v[0],v[1])
            if ekey not in e_seen:
                e_seen.add(ekey);pair=[row,donor];boundary=cut_name(v);names=tail_names(model,v)
                for recipient in (0,1):
                    for state_donor in (0,1):
                        for policy in ('fixed_left','fixed_right','regenerated'):
                            dc=native[pair[state_donor]['id']][1]
                            def swap(value,hook,kept=dc[boundary]):return kept
                            hooks=[(boundary,swap)]
                            if policy!='regenerated':hooks+=frozen(native[pair[0 if policy=='fixed_left' else 1]['id']][1],names)
                            out=model.run_with_hooks(tokens(pair[recipient]),fwd_hooks=hooks)
                            result=dict(source=row['id'],case=case,layer=v[0],cut_kind=v[1],recipient=recipient,
                                state_donor=state_donor,policy=policy,
                                donor_native_logit_max_error=float((out-native[pair[state_donor]['id']][0]).abs().max()))
                            if policy=='regenerated':assert torch.equal(out,native[pair[state_donor]['id']][0]),('E blocked path',result)
                            assert bool(torch.isfinite(out).all());Es.append(result)
        print('C2 falsification task',case,'elapsed',round(time.perf_counter()-start,2),flush=True)
        del model,native,craft
    write('results/c2_falsification.json',rows);write('results/c2_E.json',Es)
    np.savez_compressed(D/'results/c2_future_logits.npz',**saved)
    write('results/c2_execution.json',dict(finished_utc=stamp(),n_queries=len(rows),n_E_conditions=len(Es),
        future_elapsed_seconds=time.perf_counter()-start,
        oracle_disagreements=sum(not r['craft_RASP_agreement'] for r in rows),
        empty_oracle_masks=sum(not any(r['defined_mask']) for r in rows),
        limitations='semantic subset; test-time probes; internal methods and implementation; no human external replication'))
    check()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','probes','falsify'],required=True);a=p.parse_args()
    {'prepare':prepare,'probes':probes,'falsify':falsify}[a.phase]()
