"""Local native-dynamics study. Each numerical stage refuses output overwrite."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from native import LEVELS,reconstruct,native_features,forced_prediction,clean_prediction,choose_interventions
from run_pilot import ROOT,SudokuModel,file_hash,load_boards,state_hash,write_json,assert_prediction_equal,verify_assets
from features import CELL_POS,prompt_cells,observed_counts
from confirmation_corpus import backtracking,exact_cover_count,known_transform_key
from correction.cell_features import visible_features
from correction.learning import CellPCA,fit_logistic,grouped_metrics
from probes import Standardizer
from study_v03 import ObservableScaler
from trace import instrumented_prediction

BASE=ROOT/'audit/metacognition/native-dynamics-v1'
OLD=ROOT/'audit/metacognition/correction-v0.1/training'
PROTOCOL=HERE/'protocol-v1.json'
MODELS=['visible','static','dynamic','noise','shuffle']

def fingerprint():
    files=list(HERE.glob('*.py'))+[PROTOCOL]
    files +=[HERE.parent/n for n in ['confirmation_corpus.py','run_pilot.py','design.py','features.py','probes.py','study_v03.py','trace.py']]
    files +=[HERE.parent/'correction'/n for n in ['cell_features.py','learning.py']]
    files+=list((ROOT/'src/ncm_mac').rglob('*.py'))+[ROOT/'assets.lock.json']
    return {str(p.relative_to(ROOT)):file_hash(p)for p in sorted(files)}

def stamp():return datetime.now(timezone.utc).isoformat()
def hashes(folder):return {p.name:file_hash(p)for p in sorted(folder.glob('*.npy'))}
def check_arrays(folder):
    report=json.loads((folder/'report.json').read_text())
    assert report['status']=='complete'
    for n,h in report['arrays'].items():assert file_hash(folder/n)==h
    return report

def allocate(folder,n):
    shapes=dict(visible=((n*51,388),np.float64),static=((n*51,256),np.float32),delta=((n*51,256),np.float32),
        amplitude=((n*51,3),np.float64),group=((n*51,),np.int64),cell=((n*51,),np.uint8),
        current_error=((n*51,),bool),persistent=((n*51,),bool))
    return {k:np.lib.format.open_memmap(folder/f'{k}.npy',mode='w+',shape=shape,dtype=dtype)for k,(shape,dtype)in shapes.items()}

def load_data(folder):return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False)for p in folder.glob('*.npy')}

def technical():
    BASE.mkdir(exist_ok=False)
    start=time.perf_counter();verify_assets(ROOT,'hard',include_data=True)
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    puzzles=load_boards(ROOT/'data/sudoku/hard/puzzles.pickle');records=[]
    for offset in range(0,32,8):
        groups=split['pilot_indices'][offset:offset+8]
        prefix=model.prefix([puzzles[g]for g in groups]);clean,logits=clean_prediction(model,prefix)
        assert_prediction_equal(clean,model.predict(prefix,optimized=False))
        z=reconstruct(model,clean.codes,12);assert torch.equal(z,clean.reconstruction)
        s,d,a,z9=native_features(model,clean.codes[:,:116]);assert np.isfinite(a).all()
        snapshots,=model.vqvae.vq.idx_to_zhats(clean.codes,return_zlevels=False)
        assert torch.equal(z9,snapshots[8])
        positions,original,plausible,other,norms=choose_interventions(model,clean.codes,logits,groups)
        assert_prediction_equal(clean,forced_prediction(model,prefix,positions,original))
        perturbed=forced_prediction(model,prefix,positions,plausible)
        assert torch.equal(clean.codes[:,:53],perturbed.codes[:,:53])
        mid=clean.codes[:,53:79]!=perturbed.codes[:,53:79]
        assert (mid.sum(1)==1).all()
        fixed=clean.codes.clone();fixed[torch.arange(len(groups)),53+torch.from_numpy(positions)]=torch.from_numpy(plausible)
        e=(reconstruct(model,fixed[:,:79],8)-reconstruct(model,clean.codes[:,:79],8)).double().flatten(1).norm(dim=1)
        assert np.allclose(e.numpy(),norms[:,0],rtol=2e-6,atol=2e-6)
        try:native_features(model,clean.codes)
        except AssertionError:pass
        else:raise AssertionError('Future codes accepted')
        assert not model.net.lm_heads[7]._forward_hooks
        records.append(dict(groups=groups,sham_bitwise_equal=True,native_latents_bitwise_equal=True,
                            only_one_level8_code_forced=True,past_unchanged=True,future_codes_rejected=True))
    assert before==state_hash(model)
    write_json(BASE/'technical.json',dict(status='passed',source_files=fingerprint(),protocol_sha256=file_hash(PROTOCOL),
        utc=stamp(),model_state_sha256_before_and_after=before,batches=records,seconds=time.perf_counter()-start,
        free_disk=shutil.disk_usage(ROOT).free))
    print('technical gates passed: 32 old training puzzles',flush=True)

def extract_training():
    assert json.loads((BASE/'technical.json').read_text())['status']=='passed'
    dest=BASE/'training';dest.mkdir(exist_ok=False);start=time.perf_counter()
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    oldreport=json.loads((OLD/'report.json').read_text())
    for n,e in oldreport['arrays'].items():assert file_hash(OLD/n)==e['sha256']
    oldgroup=np.load(OLD/'group.npy',allow_pickle=False);oldcell=np.load(OLD/'cell.npy',allow_pickle=False)
    static=np.lib.format.open_memmap(dest/'static.npy',mode='w+',dtype='float32',shape=(51000,256))
    delta=np.lib.format.open_memmap(dest/'delta.npy',mode='w+',dtype='float32',shape=(51000,256))
    amp=np.lib.format.open_memmap(dest/'amplitude.npy',mode='w+',dtype='float64',shape=(51000,3))
    source=ROOT/'audit/metacognition/train-v0.2';report=json.loads((source/'report.json').read_text());row=0
    for j,entry in enumerate(report['batches']):
        path=source/entry['file'];assert file_hash(path)==entry['sha256']
        with np.load(path,allow_pickle=False)as trace:
            s,d,a,z9=native_features(model,torch.from_numpy(trace['codes'][:,:116].astype(np.int64)))
            logits=model.vqvae.decoder(z9)
            assert np.array_equal(logits.double().softmax(-1).numpy(),trace['decoder_probs'][:,8])
            blank=prompt_cells(trace['prefix'])==0;n=int(blank.sum())
            g=np.broadcast_to(trace['sample_indices'][:,None],blank.shape)[blank]
            c=np.broadcast_to(np.arange(81),blank.shape)[blank]
            assert np.array_equal(g,oldgroup[row:row+n])and np.array_equal(c,oldcell[row:row+n])
            static[row:row+n]=s[blank];delta[row:row+n]=d[blank];amp[row:row+n]=a[blank];row+=n
        if j%25==0:print(f'training latent replay {j+1}/{len(report["batches"])}',flush=True)
    assert row==51000 and before==state_hash(model)
    for a in [static,delta,amp]:a.flush()
    write_json(dest/'report.json',dict(status='complete',utc=stamp(),source_files=fingerprint(),
        n=1000,arrays=hashes(dest),old_training_report_sha256=file_hash(OLD/'report.json'),
        model_state_sha256_before_and_after=before,decoder_probabilities_bitwise_equal=True,seconds=time.perf_counter()-start))

def training_data():
    check_arrays(BASE/'training')
    data={k:np.load(OLD/f'{k}.npy',mmap_mode='r',allow_pickle=False)for k in ['visible','group','cell','current_error','persistent']}
    data.update(load_data(BASE/'training'));return data

def controls(data):
    group=data['group'];cell=data['cell'];noise=np.empty((len(group),16));shuffle=np.arange(len(group))
    for g in np.unique(group):
        rows=np.flatnonzero(group==g);assert len(rows)==51
        noise[rows]=np.random.default_rng(np.random.SeedSequence([20261205,int(g)])).standard_normal((81,16))[cell[rows]]
        shuffle[rows]=np.random.default_rng(np.random.SeedSequence([20261206,int(g)])).permutation(rows)
    return noise,shuffle

def prepare(data,train_groups,valid_groups):
    ta=np.isin(data['group'],train_groups);va=np.isin(data['group'],valid_groups)
    assert not (ta&va).any();t=np.flatnonzero(ta&data['current_error']);v=np.flatnonzero(va&data['current_error'])
    ns,sh=controls(data)
    base=ObservableScaler().fit(data['visible'][t]);am=ObservableScaler().fit(data['amplitude'][t])
    sp=CellPCA().fit(data['static'][ta]);dp=CellPCA().fit(data['delta'][ta]);nz=Standardizer().fit(ns[t])
    arrays={}
    for pre,obj in [('base',base),('amplitude',am),('noise',nz)]:
        arrays.update({f'{pre}_{k}':getattr(obj,k)for k in ['mean','keep','scale']})
    for pre,obj in [('static',sp),('delta',dp)]:arrays.update({pre+'_'+k:value for k,value in obj.arrays().items()})
    out={}
    for name in MODELS:
        pair=[]
        for rows in [t,v]:
            x=base.transform(data['visible'][rows])
            if name!='visible':x=np.column_stack([x,am.transform(data['amplitude'][rows]),sp.transform(data['static'][rows])])
            if name=='dynamic':x=np.column_stack([x,dp.transform(data['delta'][rows])])
            if name=='shuffle':x=np.column_stack([x,dp.transform(data['delta'][sh[rows]])])
            if name=='noise':x=np.column_stack([x,nz.transform(ns[rows])])
            pair.append(x)
        out[name]=pair
    return out,t,v,arrays

def transform(saved,data,name,rows):
    def scale(prefix,x):
        keep=saved[prefix+'_keep'];return (x[:,keep]-saved[prefix+'_mean'][keep])/saved[prefix+'_scale']
    def pca(prefix,x):
        h=scale(prefix+'_hidden',x.astype(float))
        return (h@saved[prefix+'_pca_axes'].T-saved[prefix+'_pca_mean'])/saved[prefix+'_pca_scale']
    x=scale('base',data['visible'][rows])
    if name!='visible':x=np.column_stack([x,scale('amplitude',data['amplitude'][rows]),pca('static',data['static'][rows])])
    if name in ['dynamic','shuffle']:
        source=rows if name=='dynamic'else controls(data)[1][rows]
        x=np.column_stack([x,pca('delta',data['delta'][source])])
    if name=='noise':x=np.column_stack([x,scale('noise',controls(data)[0][rows])])
    return x

def fit():
    dest=BASE/'models';dest.mkdir(exist_ok=False);data=training_data();config=json.loads(PROTOCOL.read_text())
    protocol=config['learning'];groups=np.unique(data['group'])
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    fold=np.array([split['train_cv_fold'][str(g)]for g in groups]);values={m:{str(s):[]for s in protocol['lambda_grid']}for m in MODELS}
    oof={m:np.full(len(data['group']),np.nan)for m in MODELS};byfold=[];start=time.perf_counter()
    for k in range(5):
        matrices,t,v,_=prepare(data,groups[fold!=k],groups[fold==k])
        for name,(a,b)in matrices.items():
            for strength in protocol['lambda_grid']:
                fitted=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
                score=b@fitted['weight']+fitted['intercept']
                metric=grouped_metrics(data['persistent'][v],score,data['group'][v])
                values[name][str(strength)].append(metric['within_puzzle_auc'])
            print(f'native CV fold{k+1}/5 {name}',flush=True)
    matrices,t,_,arrays=prepare(data,groups,np.array([],dtype=int));selection={}
    for name,(a,_)in matrices.items():
        means={s:float(np.mean(v))for s,v in values[name].items()};best=max(means.values())
        strength=max(float(s)for s in means if means[s]>=best-1e-8)
        fitted=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
        np.savez_compressed(dest/f'{name}.npz',**arrays,weight=fitted['weight'],intercept=np.array(fitted['intercept']))
        selection[name]=dict(lambda_value=strength,cv_mean_within_auc=means,iterations=fitted['iterations'],
            max_abs_gradient=fitted['max_abs_gradient'],dimension=len(fitted['weight']))
        with np.load(dest/f'{name}.npz',allow_pickle=False)as saved:
            replay=transform(saved,data,name,t)
            assert np.allclose(replay,a,rtol=1e-12,atol=1e-12)
    # Paired endpoint precision sensitivity, analytic and explicitly conditional; no external outcomes.
    z=2.241402727604947
    power=[]
    normal=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
    for sd in [.05,.1,.2]:
        se=sd/math.sqrt(290);effect=.02
        power.append(dict(endpoint='dynamic within-puzzle AUC',assumed_mixed_puzzles=290,paired_sd=sd,
            interval_halfwidth=z*se,positive_detection_power=normal(effect/se-z)+normal(-effect/se-z)))
    for discordance in [.05,.1,.2]:
        effect=.03;se=math.sqrt((discordance-effect**2)/1500)
        power.append(dict(endpoint='paired causal resolution',discordance=discordance,n=1500,
            interval_halfwidth=z*se,positive_detection_power=normal(effect/se-z)+normal(-effect/se-z)))
    write_json(BASE/'fit.json',dict(status='complete',utc=stamp(),source_files=fingerprint(),selection=selection,
        power_sensitivity=power,seconds=time.perf_counter()-start,training_report_sha256=file_hash(BASE/'training/report.json'),
        caveat='CV selection performance is development, not an independent evaluation. Power scenarios omit training variance and unknown external prevalence.'))
    print(json.dumps(selection),flush=True)

def generate():
    dest=BASE/'external';dest.mkdir(exist_ok=False);config=json.loads(PROTOCOL.read_text())['external'];start=time.perf_counter()
    official=load_boards(ROOT/'data/sudoku/hard/solutions.pickle')
    seen={known_transform_key([s[i]for i in CELL_POS])for s in official}
    previous=np.load(ROOT/'audit/metacognition/correction-v0.1/external/corpus.npz',allow_pickle=False)
    seen.update(known_transform_key(s.tolist())for s in previous['solutions'])
    pilot=json.loads((ROOT/'audit/metacognition/confirmation-generator-pilot/puzzles.json').read_text())
    seen.update(known_transform_key(row['solution'])for row in pilot)
    rng=random.Random(config['seed']);puzzles=[];solutions=[];rejected=[];attempts=0
    while len(puzzles)<config['n']:
        attempts+=1
        if attempts>config['attempt_cap']:raise RuntimeError('Generation cap reached')
        solution=backtracking([0]*81,limit=1,rng=rng)[0];key=known_transform_key(solution)
        if key in seen:rejected.append(dict(attempt=attempts,reason='known_transform'));continue
        puzzle=solution.copy();order=list(range(81));rng.shuffle(order)
        for c in order:
            old,puzzle[c]=puzzle[c],0
            if len(backtracking(puzzle))!=1:puzzle[c]=old
            if sum(bool(v)for v in puzzle)==30:break
        if sum(bool(v)for v in puzzle)!=30:rejected.append(dict(attempt=attempts,reason='clues'));continue
        assert exact_cover_count(puzzle)==1 and backtracking(puzzle,limit=1)[0]==solution
        seen.add(key);puzzles.append(puzzle);solutions.append(solution)
        if len(puzzles)%200==0:print(f'new corpus {len(puzzles)}/{config["n"]}',flush=True)
    np.savez_compressed(dest/'corpus.npz',puzzles=np.array(puzzles,dtype=np.uint8),solutions=np.array(solutions,dtype=np.uint8),
        groups=np.arange(config['groups_start'],config['groups_start']+config['n']))
    write_json(dest/'manifest.json',dict(status='complete',utc=stamp(),n=config['n'],seed=config['seed'],attempts=attempts,
        rejections=rejected,corpus_sha256=file_hash(dest/'corpus.npz'),seconds=time.perf_counter()-start,
        uniqueness_independent_exact_cover=True,no_ncm_inference=True,protocol_sha256=file_hash(PROTOCOL)))

def lock():
    assert not (BASE/'lock.json').exists()
    fitted=json.loads((BASE/'fit.json').read_text());assert fitted['source_files']==fingerprint()
    corpus=json.loads((BASE/'external/manifest.json').read_text())
    assert corpus['corpus_sha256']==file_hash(BASE/'external/corpus.npz')
    write_json(BASE/'lock.json',dict(status='locked_before_external_inference',utc=stamp(),source_files=fingerprint(),
        corpus_manifest_sha256=file_hash(BASE/'external/manifest.json'),fit_sha256=file_hash(BASE/'fit.json'),
        models={p.name:file_hash(p)for p in (BASE/'models').glob('*.npz')},
        protocol_sha256=file_hash(PROTOCOL),n=2000,causal_n=1500,two_primary_endpoints=True))

def locked():
    record=json.loads((BASE/'lock.json').read_text());assert record['source_files']==fingerprint()
    assert record['corpus_manifest_sha256']==file_hash(BASE/'external/manifest.json')
    corpus=json.loads((BASE/'external/manifest.json').read_text());assert corpus['corpus_sha256']==file_hash(BASE/'external/corpus.npz')
    assert record['fit_sha256']==file_hash(BASE/'fit.json')
    for name,h in record['models'].items():assert file_hash(BASE/'models'/name)==h
    return record

def prefixes(model,puzzles):
    rows=np.full((len(puzzles),89),10,dtype=np.int64);rows[:,CELL_POS]=puzzles
    return model.prefix(np.pad(rows,((0,0),(1,1)),constant_values=11).tolist())

def external_extract():
    locked();dest=BASE/'external/traces';dest.mkdir(exist_ok=False);start=time.perf_counter()
    corpus=np.load(BASE/'external/corpus.npz',allow_pickle=False);n=len(corpus['groups']);arrays=allocate(dest,n)
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model);codes=[];records=[]
    for offset in range(0,n,8):
        puzzles=corpus['puzzles'][offset:offset+8];gold=corpus['solutions'][offset:offset+8];groups=corpus['groups'][offset:offset+8]
        prefix=prefixes(model,puzzles);reference=model.predict(prefix)
        prediction,trace,raw,_=instrumented_prediction(model,prefix);assert_prediction_equal(reference,prediction)
        if offset%400==0:assert_prediction_equal(model.predict(prefix,optimized=False),prediction)
        s,d,a,z9=native_features(model,trace_to_codes(trace))
        assert torch.equal(z9,raw['reconstruction'][8]);logits=raw['decoder_logits'][8].numpy()
        visible=visible_features(prefix.numpy(),trace,logits)
        blank=puzzles==0;current=trace['decoded_ids'][:,8,CELL_POS];final=trace['decoded_ids'][:,11,CELL_POS]
        block=dict(visible=visible[blank],static=s[blank],delta=d[blank],amplitude=a[blank],
            group=np.broadcast_to(groups[:,None],blank.shape)[blank],cell=np.broadcast_to(np.arange(81),blank.shape)[blank],
            current_error=(current!=gold)[blank],persistent=(final!=gold)[blank])
        for k,value in block.items():arrays[k][offset*51:(offset+len(puzzles))*51]=value
        codes.append(prediction.codes.numpy().copy())
        records.append(dict(groups=groups.tolist(),exact_final_cells=int((final==gold).all(1).sum()),bitwise_parity=True))
        if offset%160==0:print(f'new clean NCM {offset+len(puzzles)}/{n}',flush=True)
    for a in arrays.values():a.flush()
    np.save(dest/'codes.npy',np.concatenate(codes))
    assert before==state_hash(model);locked()
    write_json(dest/'report.json',dict(status='complete',utc=stamp(),n=n,source_files=fingerprint(),arrays=hashes(dest),batches=records,
        model_state_sha256_before_and_after=before,lock_sha256=file_hash(BASE/'lock.json'),seconds=time.perf_counter()-start))

def trace_to_codes(trace):return torch.from_numpy(trace['codes'][:,:116].astype(np.int64))

def bootstrap_mean(values,seed,coverage=.975):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean()for _ in range(10000)])
    alpha=(1-coverage)/2
    return np.quantile(means,[alpha,1-alpha]).tolist()

def evaluate():
    locked();check_arrays(BASE/'external/traces');dest=BASE/'external/analysis';dest.mkdir(exist_ok=False)
    data=load_data(BASE/'external/traces');rows=np.flatnonzero(data['current_error']);g=data['group'][rows];y=data['persistent'][rows]
    scores={};results={}
    for name in MODELS:
        with np.load(BASE/f'models/{name}.npz',allow_pickle=False)as saved:
            x=transform(saved,data,name,rows);scores[name]=x@saved['weight']+saved['intercept']
        results[name]=grouped_metrics(y,scores[name],g)
    pairs={m:{r['group']:r['auc']for r in results[m]['per_group']}for m in MODELS};keys=sorted(pairs['visible']);assert keys
    contrasts={}
    for m,b in [('dynamic','static'),('static','visible'),('dynamic','visible'),('noise','static'),('shuffle','static')]:
        delta=np.array([pairs[m][k]-pairs[b][k]for k in keys])
        contrasts[m+'-'+b]=dict(delta=float(delta.mean()),interval_97_5=bootstrap_mean(delta,20261202),mixed_puzzles=len(keys),
            paired_sd=float(delta.std(ddof=1)))
    np.savez_compressed(dest/'predictions.npz',group=g,persistent=y,cell=data['cell'][rows],rows=rows,**scores)
    write_json(dest/'report.json',dict(status='complete',utc=stamp(),lock_sha256=file_hash(BASE/'lock.json'),results=results,
        contrasts=contrasts,n=2000,current_errors=len(rows),persistent_errors=int(y.sum()),mixed_groups=len(keys),
        predictions_sha256=file_hash(dest/'predictions.npz'),
        primary='dynamic-static',interval='97.5 percent paired group bootstrap conditional on fits; two primary endpoints Bonferroni',
        caveat='Oracle current-error cohort; external generated distribution, no full isomorphism/pretraining audit; no native monitoring inference'))
    print(json.dumps(dict(contrasts=contrasts,within_auc={k:v['within_puzzle_auc']for k,v in results.items()})),flush=True)

def resolution(ids,puzzles,gold):
    invalid,sep,clue,pairs,valid=observed_counts(ids,puzzles)
    strict=valid&((ids[:,CELL_POS]==gold).all(1))
    return strict

def causal():
    locked();dest=BASE/'causal';dest.mkdir(exist_ok=False);start=time.perf_counter()
    corpus=np.load(BASE/'external/corpus.npz',allow_pickle=False);oldcodes=np.load(BASE/'external/traces/codes.npy',allow_pickle=False)
    n=json.loads(PROTOCOL.read_text())['causal']['n'];model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    pieces={};records=[]
    for offset in range(0,n,8):
        puzzles=corpus['puzzles'][offset:min(offset+8,n)];gold=corpus['solutions'][offset:min(offset+8,n)];groups=corpus['groups'][offset:min(offset+8,n)]
        prefix=prefixes(model,puzzles);clean,logits=clean_prediction(model,prefix)
        assert np.array_equal(clean.codes.numpy(),oldcodes[offset:offset+len(groups)])
        positions,original,plausible,other,norms=choose_interventions(model,clean.codes,logits,groups)
        sham=forced_prediction(model,prefix,positions,original);assert_prediction_equal(clean,sham)
        z8=reconstruct(model,clean.codes[:,:79],8);clean_final=clean.reconstruction
        block=dict(group=groups,position=positions,original=original,runner_up=plausible,matched=other,
            candidate_norms=norms,clean_success=resolution(clean.ids.numpy(),puzzles,gold))
        blank=puzzles==0;cleanacc=((clean.ids.numpy()[:,CELL_POS]==gold)&blank).sum(1)/51
        block['clean_blank_accuracy']=cleanacc
        for name,values in [('runner_up',plausible),('matched',other)]:
            active=forced_prediction(model,prefix,positions,values)
            fixed_codes=clean.codes.clone();fixed_codes[torch.arange(len(groups)),53+torch.from_numpy(positions)]=torch.from_numpy(values)
            fixedz=reconstruct(model,fixed_codes,12);fixed_logits=model.vqvae.decoder(fixedz);fixedids=fixed_logits.argmax(-1).numpy()
            activeids=active.ids.numpy()
            expected=fixed_codes[:,:79];assert torch.equal(active.codes[:,:79],expected)
            e=(reconstruct(model,expected,8)-z8).double().flatten(1)
            response=(active.reconstruction.double()-fixedz.double()).flatten(1)
            ef=(active.reconstruction.double()-clean_final.double()).flatten(1)
            if offset==0:
                np.savez_compressed(dest/f'latent-check-{name}.npz',
                    initial_effect=e.numpy(),future_response=response.numpy(),
                    active_final=active.reconstruction.numpy(),fixed_final=fixedz.numpy(),clean_final=clean_final.numpy())
            en=e.norm(dim=1);rn=response.norm(dim=1)
            assert (en>0).all()
            block.update({name+'_active_success':resolution(activeids,puzzles,gold),name+'_fixed_success':resolution(fixedids,puzzles,gold),
                name+'_active_blank_accuracy':((activeids[:,CELL_POS]==gold)&blank).sum(1)/51,
                name+'_fixed_blank_accuracy':((fixedids[:,CELL_POS]==gold)&blank).sum(1)/51,
                name+'_compensation':(-(e*response).sum(1)/en.square()).numpy(),
                name+'_cosine':(-(e*response).sum(1)/(en*rn).clamp_min(1e-12)).numpy(),
                name+'_relative_final_latent_norm':(ef.norm(dim=1)/en).numpy(),
                name+'_future_codes_changed':(active.codes[:,79:]!=clean.codes[:,79:]).sum(1).numpy(),
                name+'_active_ids':activeids,name+'_fixed_ids':fixedids,
                name+'_forced_probability':logits.double().softmax(-1).numpy()[np.arange(len(groups)),positions,values]})
        for k,v in block.items():pieces.setdefault(k,[]).append(v)
        records.append(dict(groups=groups.tolist(),sham_bitwise_equal=True,past_equal=True,exactly_one_forced_code=True))
        if offset%160==0:print(f'paired interventions {offset+len(groups)}/{n}',flush=True)
    arrays={k:np.concatenate(v)for k,v in pieces.items()};np.savez_compressed(dest/'outcomes.npz',**arrays)
    assert before==state_hash(model);locked()
    result={}
    for name in ['runner_up','matched']:
        active=arrays[name+'_active_success'].astype(int);fixed=arrays[name+'_fixed_success'].astype(int);delta=active-fixed
        acc=arrays[name+'_active_blank_accuracy']-arrays[name+'_fixed_blank_accuracy']
        result[name]=dict(active_resolutions=int(active.sum()),fixed_resolutions=int(fixed.sum()),
            adaptive_only=int((delta==1).sum()),fixed_only=int((delta==-1).sum()),
            resolution_gain=float(delta.mean()),interval_97_5=bootstrap_mean(delta,20261204),
            blank_accuracy_gain=float(acc.mean()),blank_accuracy_interval_95=bootstrap_mean(acc,20261207,.95),
            compensation_mean=float(arrays[name+'_compensation'].mean()),
            compensation_interval_95=bootstrap_mean(arrays[name+'_compensation'],20261208,.95),
            cosine_mean=float(arrays[name+'_cosine'].mean()),relative_final_latent_norm_mean=float(arrays[name+'_relative_final_latent_norm'].mean()),
            mean_future_codes_changed=float(arrays[name+'_future_codes_changed'].mean()),
            recovered_from_clean_failure=int(((active==1)&~arrays['clean_success']).sum()),
            lost_clean_success=int(((active==0)&arrays['clean_success']).sum()),
            forced_code_probability_mean=float(arrays[name+'_forced_probability'].mean()))
    mismatch=np.abs(arrays['candidate_norms'][:,1]/arrays['candidate_norms'][:,0]-1)
    write_json(dest/'report.json',dict(status='complete',utc=stamp(),n=n,source_files=fingerprint(),
        outcomes_sha256=file_hash(dest/'outcomes.npz'),lock_sha256=file_hash(BASE/'lock.json'),
        latent_checks={p.name:file_hash(p)for p in dest.glob('latent-check-*.npz')},
        model_state_sha256_before_and_after=before,batches=records,seconds=time.perf_counter()-start,
        clean_resolutions=int(arrays['clean_success'].sum()),primary='runner_up adaptive minus fixed tail verified resolution',results=result,
        norm_match_relative_mismatch_quantiles=np.quantile(mismatch,[0,.5,.9,1]).tolist(),
        caveat='Fixed-tail counterfactual is not an alternative deployed solver. Positive adaptation tests response to code interventions, not second-order error monitoring.'))
    print(json.dumps(dict(clean_resolutions=int(arrays['clean_success'].sum()),results=result)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['technical','training','fit','generate','lock','extract','evaluate','causal']);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261201)
    {'technical':technical,'training':extract_training,'fit':fit,'generate':generate,'lock':lock,'extract':external_extract,'evaluate':evaluate,'causal':causal}[args.stage]()

if __name__=='__main__':main()
