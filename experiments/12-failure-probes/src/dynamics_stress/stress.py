"""Compact stress follow-up: calibrate DAMAGE only on old training, then new test."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'dynamics'))
from native import reconstruct
from run_pilot import ROOT,SudokuModel,load_boards,file_hash,state_hash,assert_prediction_equal,write_json,verify_assets
from features import CELL_POS,observed_counts
from confirmation_corpus import backtracking,exact_cover_count,known_transform_key
BASE=ROOT/'audit/metacognition/native-stress-v1';DESIGN=HERE/'design-v1.json'

def stamp():return datetime.now(timezone.utc).isoformat()
def fingerprint():
    files=list(HERE.glob('*.py'))+[DESIGN]+[HERE.parent/'dynamics/native.py']
    files+=list(HERE.parent.glob('*.py'))+list((ROOT/'src/ncm_mac').rglob('*.py'))+[ROOT/'assets.lock.json']
    return {str(p.relative_to(ROOT)):file_hash(p)for p in files}

def selection(codes,groups,count):
    pos=[];values=[]
    for row,g in enumerate(groups):
        rng=np.random.default_rng(np.random.SeedSequence([20261210,int(g)]))
        positions=rng.permutation(26);delta=rng.integers(1,16,26)
        original=codes[row,53:79].numpy()
        pos.append(positions[:count]);values.append((original[positions[:count]]+delta[:count])%16)
    return np.array(pos),np.array(values)

@torch.inference_mode()
def force(model,prefix,positions,values):
    def hook(module,args,output):
        result=output.clone();rows=torch.arange(len(prefix))[:,None]
        result[rows,torch.from_numpy(positions),torch.from_numpy(values)]=result.amax((1,2))[:,None]+100
        return result
    h=model.net.lm_heads[7].register_forward_hook(hook)
    try:prediction=model.predict(prefix)
    finally:h.remove()
    assert not model.net.lm_heads[7]._forward_hooks
    return prediction

def strict(ids,puzzles,gold):
    _,_,_,_,valid=observed_counts(ids,puzzles)
    return valid&((ids[:,CELL_POS]==gold).all(1))

def prefix(model,puzzles):
    rows=np.full((len(puzzles),89),10,dtype=np.int64);rows[:,CELL_POS]=puzzles
    return model.prefix(np.pad(rows,((0,0),(1,1)),constant_values=11).tolist())

def bootstrap(x,seed=20261212,coverage=.99):
    rng=np.random.default_rng(seed);a=(1-coverage)/2
    return np.quantile([rng.choice(x,len(x),replace=True).mean()for _ in range(10000)],[a,1-a]).tolist()

def exact_p(win,lose):
    n=win+lose
    if n==0:return 1.
    log_terms=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)for k in range(min(win,lose)+1)]
    top=max(log_terms)
    return min(1.,2*math.exp(top)*sum(math.exp(v-top)for v in log_terms))

def pilot():
    BASE.mkdir(exist_ok=False);start=time.perf_counter();design=json.loads(DESIGN.read_text());model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    source=ROOT/'audit/metacognition/train-v0.2';manifest=json.loads((source/'report.json').read_text())
    refs=load_boards(ROOT/'data/sudoku/hard/solutions.pickle');pieces=[]
    for entry in manifest['batches'][:16]:
        p=source/entry['file'];assert file_hash(p)==entry['sha256']
        with np.load(p,allow_pickle=False)as trace:
            codes=torch.from_numpy(trace['codes'].astype(np.int64));groups=trace['sample_indices']
            puzzles=trace['prefix'][:,1:-1][:,CELL_POS];gold=np.array([refs[g]for g in groups])[:,CELL_POS]
            cleanz=reconstruct(model,codes,12);cleanids=model.vqvae.decoder(cleanz).argmax(-1).numpy()
            assert np.array_equal(cleanids,trace['decoded_ids'][:,-1])
            out=dict(group=groups,clean=strict(cleanids,puzzles,gold))
            for count in design['calibration']['counts']:
                positions,values=selection(codes,groups,count);fixed=codes.clone()
                fixed[torch.arange(len(groups))[:,None],53+torch.from_numpy(positions)]=torch.from_numpy(values)
                ids=model.vqvae.decoder(reconstruct(model,fixed,12)).argmax(-1).numpy()
                out[str(count)]=strict(ids,puzzles,gold)
            pieces.append(out)
    arrays={k:np.concatenate([r[k]for r in pieces])for k in pieces[0]};summary={}
    for count in design['calibration']['counts']:
        damage=arrays['clean'].astype(int)-arrays[str(count)].astype(int)
        summary[str(count)]=dict(clean_solved=int(arrays['clean'].sum()),fixed_solved=int(arrays[str(count)].sum()),resolution_loss=float(damage.mean()))
    eligible=[c for c in design['calibration']['counts']if summary[str(c)]['resolution_loss']>=.1]
    dose=min(eligible)if eligible else None
    assert state_hash(model)==before
    np.savez_compressed(BASE/'dose-pilot.npz',**arrays)
    write_json(BASE/'pilot.json',dict(status='passed'if dose else 'stop_insufficient_damage',utc=stamp(),source_files=fingerprint(),
        n=128,summary=summary,selected_count=dose,model_state_sha256_before_and_after=before,
        arrays_sha256=file_hash(BASE/'dose-pilot.npz'),seconds=time.perf_counter()-start,
        selection_uses_only_fixed_tail_damage=True,adaptive_outcomes_not_computed=True))
    print(json.dumps(dict(selected_count=dose,summary=summary)),flush=True)

def generate():
    pilot=json.loads((BASE/'pilot.json').read_text());assert pilot['status']=='passed'
    destination=BASE/'external';destination.mkdir(exist_ok=False);config=json.loads(DESIGN.read_text())['external'];start=time.perf_counter()
    official=load_boards(ROOT/'data/sudoku/hard/solutions.pickle');seen={known_transform_key([s[i]for i in CELL_POS])for s in official}
    for path in ['correction-v0.1/external/corpus.npz','native-dynamics-v1/external/corpus.npz']:
        with np.load(ROOT/'audit/metacognition'/path,allow_pickle=False)as z:seen.update(known_transform_key(s.tolist())for s in z['solutions'])
    oldpilot=json.loads((ROOT/'audit/metacognition/confirmation-generator-pilot/puzzles.json').read_text());seen.update(known_transform_key(r['solution'])for r in oldpilot)
    rng=random.Random(config['seed']);puzzles=[];solutions=[];rejected=[];attempts=0
    while len(puzzles)<config['n']:
        attempts+=1
        if attempts>config['attempt_cap']:raise RuntimeError('generation cap')
        solution=backtracking([0]*81,limit=1,rng=rng)[0];key=known_transform_key(solution)
        if key in seen:rejected.append(dict(attempt=attempts,reason='known_transform'));continue
        puzzle=solution.copy();order=list(range(81));rng.shuffle(order)
        for i in order:
            old,puzzle[i]=puzzle[i],0
            if len(backtracking(puzzle))!=1:puzzle[i]=old
            if sum(bool(v)for v in puzzle)==30:break
        if sum(bool(v)for v in puzzle)!=30:rejected.append(dict(attempt=attempts,reason='clues'));continue
        assert exact_cover_count(puzzle)==1 and backtracking(puzzle,limit=1)[0]==solution
        seen.add(key);puzzles.append(puzzle);solutions.append(solution)
        if len(puzzles)%200==0:print(f'stress corpus {len(puzzles)}/1000',flush=True)
    np.savez_compressed(destination/'corpus.npz',puzzles=np.array(puzzles,dtype='uint8'),solutions=np.array(solutions,dtype='uint8'),groups=np.arange(config['groups_start'],config['groups_start']+config['n']))
    write_json(destination/'manifest.json',dict(status='complete',utc=stamp(),n=1000,attempts=attempts,rejections=rejected,
        corpus_sha256=file_hash(destination/'corpus.npz'),independent_exact_cover_uniqueness=True,no_ncm_inference=True,seconds=time.perf_counter()-start))
    dose=pilot['selected_count'];loss=pilot['summary'][str(dose)]['resolution_loss']
    scenarios=[]
    for discordance in [.1,.3,.5]:
        se=math.sqrt((discordance-.03**2)/1000);normal=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
        scenarios.append(dict(discordance=discordance,effect=.03,n=1000,power=normal(.03/se-2.5758293)+normal(-.03/se-2.5758293)))
    write_json(BASE/'lock.json',dict(status='locked_before_external_inference',utc=stamp(),source_files=fingerprint(),
        selected_count=dose,pilot_sha256=file_hash(BASE/'pilot.json'),corpus_manifest_sha256=file_hash(destination/'manifest.json'),
        corpus_sha256=file_hash(destination/'corpus.npz'),n=1000,primary='adaptive minus fixed tail verified resolution',
        power_sensitivity=scenarios,training_fixed_tail_resolution_loss=loss,extension_or_optional_stopping=False))

def locked():
    record=json.loads((BASE/'lock.json').read_text());assert record['source_files']==fingerprint()
    assert file_hash(BASE/'pilot.json')==record['pilot_sha256']
    assert file_hash(BASE/'external/corpus.npz')==record['corpus_sha256']
    assert file_hash(BASE/'external/manifest.json')==record['corpus_manifest_sha256']
    return record

@torch.inference_mode()
def run():
    lock=locked();destination=BASE/'external/run';destination.mkdir(exist_ok=False);start=time.perf_counter()
    corpus=np.load(BASE/'external/corpus.npz',allow_pickle=False);model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    pieces=[];records=[];dose=lock['selected_count']
    for offset in range(0,1000,8):
        puzzles=corpus['puzzles'][offset:offset+8];gold=corpus['solutions'][offset:offset+8];groups=corpus['groups'][offset:offset+8]
        p=prefix(model,puzzles);clean=model.predict(p)
        if offset%200==0:assert_prediction_equal(model.predict(p,optimized=False),clean)
        positions,values=selection(clean.codes,groups,dose)
        original=clean.codes.numpy()[np.arange(len(groups))[:,None],53+positions]
        assert_prediction_equal(clean,force(model,p,positions,original))
        active=force(model,p,positions,values);fixed=clean.codes.clone();fixed[torch.arange(len(groups))[:,None],53+torch.from_numpy(positions)]=torch.from_numpy(values)
        assert torch.equal(active.codes[:,:79],fixed[:,:79])and torch.equal(active.codes[:,:53],clean.codes[:,:53])
        assert ((active.codes[:,53:79]!=clean.codes[:,53:79]).sum(1)==dose).all()
        fixedz=reconstruct(model,fixed,12);fixedids=model.vqvae.decoder(fixedz).argmax(-1).numpy()
        e=(reconstruct(model,fixed[:,:79],8)-reconstruct(model,clean.codes[:,:79],8)).double().flatten(1)
        r=(active.reconstruction.double()-fixedz.double()).flatten(1);en=e.norm(dim=1)
        blank=puzzles==0
        out=dict(group=groups,clean_ids=clean.ids.numpy(),active_ids=active.ids.numpy(),fixed_ids=fixedids,
            clean_success=strict(clean.ids.numpy(),puzzles,gold),active_success=strict(active.ids.numpy(),puzzles,gold),fixed_success=strict(fixedids,puzzles,gold),
            positions=positions,replacements=values,compensation=(-(e*r).sum(1)/en.square()).numpy(),
            relative_final_norm=((active.reconstruction.double()-clean.reconstruction.double()).flatten(1).norm(dim=1)/en).numpy(),
            initial_norm=en.numpy(),future_changed=(active.codes[:,79:]!=clean.codes[:,79:]).sum(1).numpy(),
            active_blank_accuracy=(((active.ids.numpy()[:,CELL_POS]==gold)&blank).sum(1)/51),
            fixed_blank_accuracy=(((fixedids[:,CELL_POS]==gold)&blank).sum(1)/51))
        if offset==0:np.savez_compressed(destination/'latent-check.npz',initial_effect=e.numpy(),future_response=r.numpy(),active_final=active.reconstruction.numpy(),fixed_final=fixedz.numpy(),clean_final=clean.reconstruction.numpy())
        pieces.append(out);records.append(dict(groups=groups.tolist(),sham_bitwise_equal=True,exactly_selected_codes_changed=True,past_unchanged=True))
        if offset%160==0:print(f'stress intervention {offset+len(groups)}/1000',flush=True)
    arrays={k:np.concatenate([r[k]for r in pieces])for k in pieces[0]};np.savez_compressed(destination/'outcomes.npz',**arrays)
    assert state_hash(model)==before;locked()
    delta=arrays['active_success'].astype(int)-arrays['fixed_success'].astype(int);win=int((delta==1).sum());lose=int((delta==-1).sum())
    clean=arrays['clean_success'];induced=clean&~arrays['fixed_success']
    acc=arrays['active_blank_accuracy']-arrays['fixed_blank_accuracy']
    report=dict(status='complete',utc=stamp(),lock_sha256=file_hash(BASE/'lock.json'),n=1000,dose=dose,
        source_files=fingerprint(),model_state_sha256_before_and_after=before,seconds=time.perf_counter()-start,batches=records,
        artifacts={p.name:file_hash(p)for p in destination.glob('*.npz')},
        clean_resolutions=int(clean.sum()),fixed_resolutions=int(arrays['fixed_success'].sum()),active_resolutions=int(arrays['active_success'].sum()),
        adaptive_only=win,fixed_only=lose,primary_resolution_gain=float(delta.mean()),primary_interval_99=bootstrap(delta),exact_paired_p=exact_p(win,lose),
        clean_solved_induced_failures=int(induced.sum()),adaptively_recovered_induced_failures=int((induced&arrays['active_success']).sum()),
        lost_clean_success=int((clean&~arrays['active_success']).sum()),recovered_clean_failure=int((~clean&arrays['active_success']).sum()),
        blank_accuracy_gain=float(acc.mean()),blank_accuracy_interval_95=bootstrap(acc,20261213,.95),
        compensation_mean=float(arrays['compensation'].mean()),compensation_interval_95=bootstrap(arrays['compensation'],20261214,.95),
        relative_final_norm_mean=float(arrays['relative_final_norm'].mean()),mean_future_changed=float(arrays['future_changed'].mean()),
        identical_fixed_and_clean_grid=int((arrays['fixed_ids']==arrays['clean_ids']).all(1).sum()),
        identical_active_and_clean_grid=int((arrays['active_ids']==arrays['clean_ids']).all(1).sum()),
        caveat='New exploratory follow-up with preselected training damage dose and reserved external corpus. Strong synthetic intervention, no second-order reliability or native metacognition proof.')
    write_json(destination/'report.json',report)
    print(json.dumps({k:v for k,v in report.items()if k not in ['batches','source_files']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','generate','run']);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261210)
    {'pilot':pilot,'generate':generate,'run':run}[args.stage]()

if __name__=='__main__':main()
