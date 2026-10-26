"""Train templates, validation-only selection and held-out joint level bypass."""
from base import *
from runtime import Engine,parity

def describe(codes,lengths):
    out=[];start=0
    for k,l in enumerate(lengths):
        c=codes[:,start:start+l];counts=np.stack([np.bincount(c[:,j],minlength=16) for j in range(l)])
        p=counts/counts.sum(1,keepdims=True);m=counts.sum(0)/counts.sum();entropy=lambda v:float(-(v[v>0]*np.log(v[v>0])).sum())
        out.append(dict(level=k+1,length=l,template=counts.argmax(1),modal_position_accuracy=float(p.max(1).mean()),
          codes_used=int((counts.sum(0)>0).sum()),marginal_entropy=entropy(m),positional_entropy=float(np.mean([entropy(v) for v in p])),
          unique_code_sequences=len(np.unique(c,axis=0)),counts=counts))
        start+=l
    return out

def compare(ids,codes,native,nativecodes,ref,prefix,calls):
    before=np.all(native==ref,axis=1);after=np.all(ids==ref,axis=1);status=before!=after
    return dict(metrics=evaluate(ids,ref,prefix),changed_grids=int(np.any(ids!=native,axis=1).sum()),changed_codes=int((codes!=nativecodes).sum()),
       status_changes=int(status.sum()),introduced=int((before&~after).sum()),rescued=int((~before&after).sum()),
       delta_success=paired_mean_ci(after.astype(float)-before),status_change_wilson95=ci_rate(status.sum(),len(status)),generator_calls=calls)

@torch.inference_mode()
def generate(model,prefix,templates,fixed=(),zero=()):
    ids=[];codes=[];calls=[]
    for lo in range(0,len(prefix),16):
        with Engine(model,prefix[lo:lo+16]) as e:
            z=torch.zeros(e.b,89,256);cc=[]
            for k in range(12):
                z,c,_,_=e.step(k,z,fixed_code=templates[k] if k in fixed else None,omit_contribution=k in zero);cc.append(c)
            ids.append(model.vqvae.decoder(z).argmax(-1).numpy());codes.append(torch.cat(cc,1).numpy());calls.append(e.calls)
    assert len(set(calls))==1
    return np.concatenate(ids),np.concatenate(codes),calls[0]

@torch.inference_mode()
def main(resume=False):
    parity();cache=np.load(ROOT/'results/hard.npz');tr=np.array(SPLIT['train']);va=np.array(SPLIT['validation'])[:Q['sudoku_screen_n']];te=np.array(SPLIT['test'])
    lengths=[1,2,4,6,9,13,18,26,37,50,67,89];audit={}
    for d in ['hard','easy','medium']:
        a=np.load(ROOT/f'results/{d}.npz');audit[d]=dict(generated_train=describe(a['codes'][tr],lengths),encoded_train=describe(a['encoded'][tr],lengths))
    write(D/'results/all_levels_audit.json',audit)
    templates=[r['template'] for r in audit['hard']['generated_train']];screen={};arrays={};model=SudokuModel(ROOT/'artifacts/sudoku/hard')
    for k in ([] if resume else range(12)):
        for kind in ['fixed','zero']:
            ids,codes,calls=generate(model,cache['prefix'][va],templates,fixed=[k] if kind=='fixed' else [],zero=[k] if kind=='zero' else [])
            name=f'{kind}_{k+1}';screen[name]=compare(ids,codes,cache['decoded'][va,-1],cache['codes'][va],cache['reference'][va],cache['prefix'][va],calls)
            arrays[name+'_ids']=ids;arrays[name+'_codes']=codes
            write(D/'results/sudoku_screen.json',screen);log('screen_level',condition=name,**screen[name])
    if resume:
        screen=json.loads((D/'results/sudoku_screen.json').read_text());assert len(screen)==24 and (D/'results/sudoku_screen.npz').exists()
    else:np.savez_compressed(D/'results/sudoku_screen.npz',indices=va,**arrays)
    rank=sorted(range(12),key=lambda k:(screen[f'fixed_{k+1}']['status_changes'],screen[f'fixed_{k+1}']['changed_grids'],k))
    selected=rank[:3];selection=dict(levels=[k+1 for k in selected],validation_indices=va,ranking=[k+1 for k in rank],
       criterion=Q['sudoku_selection'],screen_sha256=sha(D/'results/sudoku_screen.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if resume:
        frozen=json.loads((D/'results/sudoku_selection_frozen.json').read_text());assert frozen['levels']==selection['levels'] and frozen['screen_sha256']==selection['screen_sha256'];selection=frozen
    else:write(D/'results/sudoku_selection_frozen.json',selection)
    log('selected_levels',levels=selection['levels'],ranking=selection['ranking'],freeze_utc=selection['utc'])
    conditions={f'single_{k+1}':[k] for k in selected};conditions['joint_two']=selected[:2];conditions['joint_three']=selected
    results={};arrays={}
    for corpus in ['test','external']:
        if corpus=='test':c=cache;ix=te
        else:c=np.load(ROOT/'results/external.npz');ix=np.arange(len(c['prefix']))
        results[corpus]={};arrays[corpus+'_indices']=ix
        for name,fixed in conditions.items():
            ids,codes,calls=generate(model,c['prefix'][ix],templates,fixed=fixed)
            result=compare(ids,codes,c['decoded'][ix,-1],c['codes'][ix],c['reference'][ix],c['prefix'][ix],calls)
            results[corpus][name]=dict(fixed_levels=[k+1 for k in fixed],**result);arrays[corpus+'_'+name+'_ids']=ids;arrays[corpus+'_'+name+'_codes']=codes
            write(D/'results/sudoku_test.json',results);log('sudoku_test',corpus=corpus,condition=name,**result)
    np.savez_compressed(D/'results/sudoku_test.npz',**arrays);log('sudoku_levels_complete')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args();main(resume=args.resume)
