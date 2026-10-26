"""Feasibility gates before any independent early-control evaluation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys,time,random,shutil,resource
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from run_pilot import ROOT,SudokuModel,file_hash,state_hash,load_boards,write_json,verify_assets,assert_prediction_equal
from engine import begin,finish,rows,pack,code_features
from features import prompt_cells,observed_counts,CELL_POS
from probes import HiddenPCA,fit_logistic,auc
from study_v03 import ObservableScaler

BASE=ROOT/'audit/metacognition/early-control-v1';DESIGN=HERE/'design-v1.json'
NAMES=['codes','hidden','codes_hidden','summary_archived']

def stamp():return datetime.now(timezone.utc).isoformat()
def fingerprint():
    files=list(HERE.glob('*.py'))+[DESIGN]+list((ROOT/'src/ncm_mac').rglob('*.py'))
    files+=[HERE.parent/n for n in ['features.py','probes.py','study_v03.py','run_pilot.py','design.py','train_probes.py']]
    return {str(p.relative_to(ROOT)):file_hash(p)for p in sorted(files)}

def technical():
    BASE.mkdir(exist_ok=False);start=time.perf_counter();verify_assets(ROOT,'hard',include_data=True)
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text());puzzles=load_boards(ROOT/'data/sudoku/hard/puzzles.pickle')
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model);states=[];references=[];indices=split['pilot_indices'];source_files=fingerprint()
    for offset in range(0,32,8):
        prefix=model.prefix([puzzles[g]for g in indices[offset:offset+8]]);reference=model.predict(prefix)
        state,feature=begin(model,prefix,'codes_hidden');out=finish(model,state);assert_prediction_equal(reference,out)
        # Independent archived feature extraction is not needed to verify h: compare full passive reference observer.
        from trace import instrumented_prediction
        ref,trace,raw,_=instrumented_prediction(model,prefix);assert_prediction_equal(reference,ref)
        np.testing.assert_array_equal(feature['hidden'],trace['hidden_summary'][:,8])
        np.testing.assert_array_equal(feature['codes'],code_features(prefix.numpy(),trace['codes'][:,:116],trace['code_probs'][:,:116]))
        state2,features2=begin(model,prefix,'summary_archived')
        from features import build_features
        np.testing.assert_array_equal(features2['summary'],build_features(prefix.numpy(),trace,9,list(model.net.levels))['summary'])
        assert_prediction_equal(reference,finish(model,state2));states.extend(rows(state));references.append(reference)
    order=np.random.default_rng(20261220).permutation(32)
    for offset in range(0,32,8):
        chosen=order[offset:offset+8];out=finish(model,pack([states[i]for i in chosen]))
        for j,i in enumerate(chosen):
            ref=references[i//8]
            for field in ['codes','reconstruction','logits','ids']:assert torch.equal(getattr(out,field)[j],getattr(ref,field)[i%8])
    chosen=[i for i in range(32)if i%3!=0]
    for offset in range(0,len(chosen),8):
        selected=chosen[offset:offset+8];actual=len(selected);selected+=selected[:1]*(8-actual)
        out=finish(model,pack([states[i]for i in selected]))
        for j,i in enumerate(selected[:actual]):assert torch.equal(out.codes[j],references[i//8].codes[i%8])and torch.equal(out.logits[j],references[i//8].logits[i%8])
    assert before==state_hash(model);assert all(not b.attn.caching and b.attn.cached_k is None for b in model.net.transformer.blocks)
    write_json(BASE/'technical.json',dict(status='passed',utc=stamp(),source_files=source_files,model_state_sha256_before_and_after=before,
        n=32,past_and_future_features_parity=True,retained_rebatched_and_padded_outputs_bitwise_equal=True,
        no_retained_state_requires_no_tail='checked separately below',hooks_and_caches_cleared=True,seconds=time.perf_counter()-start))
    print('32-puzzle pause/resume, regrouping, padding and features: exact parity',flush=True)

def data():
    c=ROOT/'audit/metacognition/train-features-v0.2'
    return dict(indices=np.load(c/'indices.npy'),failure=np.load(c/'failure.npy'),hidden=np.load(c/'hidden_raw.npy',mmap_mode='r'))

def fit():
    assert json.loads((BASE/'technical.json').read_text())['status']=='passed'
    destination=BASE/'models';destination.mkdir(exist_ok=False);start=time.perf_counter();d=data()
    source=ROOT/'audit/metacognition/train-v0.2';report=json.loads((source/'report.json').read_text());parts=[];ordered=[]
    for item in report['batches']:
        assert file_hash(source/item['file'])==item['sha256']
        with np.load(source/item['file'],allow_pickle=False)as z:
            parts.append(code_features(z['prefix'],z['codes'][:,:116],z['code_probs'][:,:116]));ordered.extend(z['sample_indices'].tolist())
    codes=np.concatenate(parts);assert ordered==d['indices'].tolist()
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text());fold=np.array([split['train_cv_fold'][str(g)]for g in d['indices']])
    settings=json.loads((HERE.parent/'protocol-v0.2.json').read_text())['fit']['optimizer_settings']
    scores={name:np.full(1000,np.nan)for name in NAMES}
    with np.load(ROOT/'audit/metacognition/development-v0.3/oof-progress.npz',allow_pickle=False)as archive:
        assert np.array_equal(archive['indices'],d['indices']);scores['summary_archived']=archive['summary'].copy()
    diagnostics=[]
    for k in range(5):
        t=np.flatnonzero(fold!=k);v=np.flatnonzero(fold==k)
        base=ObservableScaler().fit(codes[t]);pca=HiddenPCA().fit(np.asarray(d['hidden'][t]))
        ct,cv=base.transform(codes[t]),base.transform(codes[v]);ht,hv=pca.transform(d['hidden'][t]),pca.transform(d['hidden'][v])
        arrays=dict(base_mean=base.mean,base_keep=base.keep,base_scale=base.scale,hidden_mean=pca.standardizer.mean,
            hidden_keep=pca.standardizer.keep,hidden_scale=pca.standardizer.scale,pca_axes=pca.axes,
            pca_score_mean=pca.score_scaler.mean,pca_score_scale=pca.score_scaler.scale)
        for name,a,b in [('codes',ct,cv),('hidden',ht,hv),('codes_hidden',np.column_stack([ct,ht]),np.column_stack([cv,hv]))]:
            fitted=fit_logistic(a,d['failure'][t],1.,settings);scores[name][v]=b@fitted['weight']+fitted['intercept']
            np.savez_compressed(destination/f'fold{k}-{name}.npz',**arrays,weight=fitted['weight'],intercept=np.array(fitted['intercept']))
            diagnostics.append(dict(fold=k,model=name,iterations=fitted['iterations'],max_abs_gradient=fitted['max_abs_gradient']))
        print('cheap score crossfit',k+1,'/5',flush=True)
    np.savez_compressed(BASE/'development-scores.npz',indices=d['indices'],failure=d['failure'],fold=fold,**scores)
    policies={};frontiers={}
    for name,score in scores.items():
        assert np.isfinite(score).all();frontiers[name]={}
        for allowed in [0,3,10]:
            thresholds=np.r_[np.inf,np.unique(score)[::-1]];best=None
            for threshold in thresholds:
                reject=score>threshold;lost=int(((d['failure']==0)&reject).sum())
                if lost<=allowed and (best is None or int(reject.sum())>best['rejected']):
                    best=dict(threshold=float(threshold) if np.isfinite(threshold) else None,rejected=int(reject.sum()),lost_resolutions=lost,
                        true_failures_rejected=int((reject&(d['failure']==1)).sum()))
            frontiers[name][str(allowed)]=best
        policies[name]=frontiers[name]['3']
    write_json(BASE/'development.json',dict(status='complete',utc=stamp(),source_files=fingerprint(),n=1000,scope='development only, threshold selection optimistic',
        policies=policies,frontiers=frontiers,auc={n:auc(d['failure'],s)for n,s in scores.items()},diagnostics=diagnostics,
        scores_sha256=file_hash(BASE/'development-scores.npz'),models={p.name:file_hash(p)for p in destination.glob('*.npz')},seconds=time.perf_counter()-start,
        archived_summary_source_sha256=file_hash(ROOT/'audit/metacognition/development-v0.3/oof-progress.npz')))
    print(json.dumps(dict(policies=policies,frontiers=frontiers)),flush=True)

class Scorer:
    def __init__(self,name):
        self.name=name;self.models=[]
        for k in range(5):
            path=(ROOT/f'audit/metacognition/development-v0.3/outer-{k}-summary.npz')if name=='summary_archived'else BASE/f'models/fold{k}-{name}.npz'
            with np.load(path,allow_pickle=False)as z:self.models.append({n:z[n].copy()for n in z.files})
    def __call__(self,features,fold):
        out=np.empty(len(fold))
        for k in np.unique(fold):
            m=self.models[k];r=np.flatnonzero(fold==k)
            if self.name=='summary_archived':
                x=features['summary'][r];x=(x[:,m['base_keep']]-m['base_mean'][m['base_keep']])/m['base_scale']
            else:
                pieces=[]
                if self.name!='hidden':
                    x=features['codes'][r];pieces.append((x[:,m['base_keep']]-m['base_mean'][m['base_keep']])/m['base_scale'])
                if self.name!='codes':
                    h=features['hidden'][r];h=(h[:,m['hidden_keep']]-m['hidden_mean'][m['hidden_keep']])/m['hidden_scale']
                    pieces.append((h@m['pca_axes'].T-m['pca_score_mean'])/m['pca_score_scale'])
                x=np.column_stack(pieces)
            out[r]=x@m['weight']+m['intercept']
        return out

def verified(ids,prefix):return observed_counts(ids,prompt_cells(prefix))[4]

@torch.inference_mode()
def run_policy(model,prefix,fold,mode,scorer=None,threshold=None,scripted=None):
    queue=[];retained=[];answers={};all_scores=[];t0=time.perf_counter();stage1=scoretime=packtime=tailtime=verifytime=0.;tail_batches=0
    for offset in range(0,len(prefix),8):
        t=time.perf_counter();state,f=begin(model,prefix[offset:offset+8],mode if mode in NAMES else 'none');stage1+=time.perf_counter()-t
        t=time.perf_counter()
        score=scorer(f,fold[offset:offset+8])if scorer else np.zeros(8)
        keep=score<=threshold if scorer and threshold is not None else np.ones(8,dtype=bool)
        if scripted is not None:keep=scripted[offset:offset+8]
        all_scores.extend(score.tolist());scoretime+=time.perf_counter()-t
        t=time.perf_counter();rowstates=rows(state)
        for j in np.flatnonzero(keep):queue.append((offset+int(j),rowstates[j]))
        packtime+=time.perf_counter()-t
        while len(queue)>=8:
            batch,queue=queue[:8],queue[8:];t=time.perf_counter();packed=pack([v for _,v in batch]);packtime+=time.perf_counter()-t
            t=time.perf_counter();out=finish(model,packed);tailtime+=time.perf_counter()-t;tail_batches+=1
            t=time.perf_counter();valid=verified(out.ids.numpy(),packed['prefix'].numpy());verifytime+=time.perf_counter()-t
            for j,(i,_)in enumerate(batch):answers[i]=(out.codes[j].numpy().copy(),out.ids[j].numpy().copy(),bool(valid[j]))
    if queue:
        n=len(queue);t=time.perf_counter();packed=pack([v for _,v in queue]+[queue[0][1]]*(8-n));packtime+=time.perf_counter()-t
        t=time.perf_counter();out=finish(model,packed);tailtime+=time.perf_counter()-t;tail_batches+=1
        t=time.perf_counter();valid=verified(out.ids.numpy(),packed['prefix'].numpy());verifytime+=time.perf_counter()-t
        for j,(i,_)in enumerate(queue):answers[i]=(out.codes[j].numpy().copy(),out.ids[j].numpy().copy(),bool(valid[j]))
    return answers,np.array(all_scores),dict(total_seconds=time.perf_counter()-t0,first9_seconds=stage1,score_seconds=scoretime,
        pack_seconds=packtime,tail_seconds=tailtime,verifier_seconds=verifytime,tail_batches=tail_batches,retained=len(answers),
        rejected=len(prefix)-len(answers),final_padding=tail_batches*8-len(answers))


def profile():
    dev=json.loads((BASE/'development.json').read_text());dest=BASE/'profiling';dest.mkdir(exist_ok=False)
    with np.load(BASE/'development-scores.npz',allow_pickle=False)as z:indices=z['indices'].copy();fold=z['fold'].copy();expected={n:z[n].copy()for n in NAMES};failure=z['failure'].copy()
    puzzles=load_boards(ROOT/'data/sudoku/hard/puzzles.pickle');model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    prefix=model.prefix([puzzles[g]for g in indices]);model.predict(prefix[:8]);records=[];rng=random.Random(20261221)
    empty,_,empty_timing=run_policy(model,prefix[:8],fold[:8],'none',scripted=np.zeros(8,dtype=bool))
    assert not empty and empty_timing['tail_batches']==0 and empty_timing['tail_seconds']==0
    frozen_sources=fingerprint()
    # Measure primitive first9 and remaining stages, and score costs on identical batches.
    for repeat in range(5):
        names=['none',*NAMES];rng.shuffle(names)
        for name in names:
            scorer=Scorer(name)if name in NAMES else None
            for offset in range(0,128,8):
                t=time.perf_counter();state,features=begin(model,prefix[offset:offset+8],name);first=time.perf_counter()-t
                t=time.perf_counter()
                if scorer:
                    s=scorer(features,fold[offset:offset+8]);assert np.allclose(s,expected[name][offset:offset+8],rtol=1e-11,atol=1e-10)
                scoring=time.perf_counter()-t
                t=time.perf_counter();out=finish(model,state);tail=time.perf_counter()-t
                t=time.perf_counter();verified(out.ids.numpy(),prefix[offset:offset+8].numpy());vtime=time.perf_counter()-t
                records.append(dict(repeat=repeat,mode=name,offset=offset,first9=first,score=scoring,tail=tail,verify=vtime,total=first+scoring+tail+vtime))
            print('stage profiling',repeat+1,'/5',name,flush=True)
    # Optimistic analytic screening ignores all score, staging and packing overhead.
    plain=[r for r in records if r['mode']=='none'];shares=[r['tail']/(r['first9']+r['tail']+r['verify'])for r in plain]
    share=float(np.median(shares));conservative_share=float(max(shares))
    screening={}
    for name in NAMES:
        optimistic=dev['frontiers'][name]['10']['rejected']/1000
        screening[name]=dict(optimistic_reject_fraction=optimistic,zero_overhead_saving_median_tail_share=optimistic*share,
            zero_overhead_saving_max_observed_tail_share=optimistic*conservative_share,
            distribution_free_ceiling_if_all_remaining_cost_avoidable=optimistic)
    # Run real policies once initially, then repeat3 times only when internal possibility survives the optimistic gate.
    promising=any(screening[name]['zero_overhead_saving_max_observed_tail_share']>=.05 for name in ['hidden','codes_hidden'])
    repeats=3 if promising else 1;runtime=[];saved_predictions={}
    fixed_ids=[];fixed_codes=[];fixed_t=[]
    for repeat in range(repeats):
        modes=['fixed','keep_all','random',*NAMES];rng.shuffle(modes)
        for name in modes:
            if name=='fixed':
                t=time.perf_counter();ids=[];codes=[]
                for offset in range(0,1000,8):
                    out=model.predict(prefix[offset:offset+8]);verified(out.ids.numpy(),prefix[offset:offset+8].numpy())
                    ids.append(out.ids.numpy().copy());codes.append(out.codes.numpy().copy())
                duration=time.perf_counter()-t;fixed_t.append(duration);fixed_ids=np.concatenate(ids);fixed_codes=np.concatenate(codes)
                runtime.append(dict(repeat=repeat,mode=name,total_seconds=duration,tail_batches=125,rejected=0,retained=1000))
            else:
                scorer=Scorer(name)if name in NAMES else None
                threshold=dev['policies'][name]['threshold']if scorer else None
                scripted=None
                if name=='random':
                    scripted=np.ones(1000,dtype=bool);scripted[np.random.default_rng(20261222).permutation(1000)[:dev['policies']['codes_hidden']['rejected']]]=False
                answers,scores,timing=run_policy(model,prefix,fold,name,scorer,threshold,scripted)
                if scorer:assert np.allclose(scores,expected[name],rtol=1e-11,atol=1e-10)
                # Compare to stored native clean codes independent of condition order, no references in the policy.
                source=ROOT/'audit/metacognition/train-v0.2';manifest=json.loads((source/'report.json').read_text());clean=[];cleanids=[]
                for entry in manifest['batches']:
                    with np.load(source/entry['file'],allow_pickle=False)as z:clean.append(z['codes']);cleanids.append(z['decoded_ids'][:,-1])
                clean=np.concatenate(clean);cleanids=np.concatenate(cleanids)
                for i,(c,ids,valid)in answers.items():assert np.array_equal(c,clean[i])and np.array_equal(ids,cleanids[i])
                retained=np.array(sorted(answers),dtype=int);lost=int(((failure==0)&~np.isin(np.arange(1000),retained)).sum())
                if name!='random':assert lost==(dev['policies'][name]['lost_resolutions']if scorer else 0)
                np.savez_compressed(dest/f'retention-{repeat}-{name}.npz',indices=indices,retained=np.isin(np.arange(1000),retained),scores=scores,verified=np.array([answers[i][2]if i in answers else False for i in range(1000)]))
                runtime.append(dict(repeat=repeat,mode=name,**timing,lost_resolutions=lost,
                    verified_solved=int(sum(v[2]for v in answers.values())),retained_trajectory_parity=True))
            print('real policy',repeat+1,'/',repeats,name,flush=True)
    assert before==state_hash(model)
    assert frozen_sources==fingerprint()
    write_json(dest/'report.json',dict(status='complete',utc=stamp(),source_files=fingerprint(),scope='training-only feasibility; no independent test',
        keep_none_technical_timing=empty_timing,stage_records=records,stage_tail_share_median=share,stage_tail_share_max_observed=conservative_share,
        screening=screening,promising_under_optimistic_gate=promising,runtime=runtime,policy_timing_repeats_completed=repeats,
        model_state_sha256_before_and_after=before,development_sha256=file_hash(BASE/'development.json'),
        decision='proceed_to_sample_size_design_only'if promising else 'stop_no_new_corpus',
        caveat='Gate is practical screening of specified observers, not a statistical impossibility theorem or a metacognition finding. Conditional timing repetition rule specified before profiling.'))
    print(json.dumps(dict(screening=screening,decision='proceed'if promising else 'stop_no_new_corpus',runtime=runtime)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['technical','fit','profile']);args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(20261220)
    {'technical':technical,'fit':fit,'profile':profile}[args.stage]()

if __name__=='__main__':main()
