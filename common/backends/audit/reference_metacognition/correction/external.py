"""Fixed new corpus, locked cell probes, then one external evaluation."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from confirmation_corpus import backtracking,exact_cover_count,known_transform_key
from features import CELL_POS,prompt_cells
from probes import stratified_folds
from run_pilot import ROOT,SudokuModel,assert_prediction_equal,file_hash,load_boards,state_hash,verify_assets,write_json
from trace import instrumented_prediction
from correction.cell_features import visible_features
from correction.extract_training import local_readout
from correction.learning import controls,fit_logistic,grouped_metrics,prepare
from correction.verify_development import transform,shuffled_rows,check_scores

BASE=ROOT/'audit/metacognition/correction-v0.1'
EXTERNAL=BASE/'external'


def fingerprint():
    files=list(HERE.glob('*.py'))+[HERE/'protocol-v0.1.json',HERE/'external-design-v0.1.json']
    files +=[HERE.parent/name for name in ['confirmation_corpus.py','features.py','probes.py','study_v03.py','trace.py']]
    files +=list((ROOT/'src/ncm_mac').rglob('*.py'))+[ROOT/'assets.lock.json']
    return {str(p.relative_to(ROOT)):file_hash(p)for p in files}


def generate():
    EXTERNAL.mkdir(exist_ok=False)
    config=json.loads((HERE/'external-design-v0.1.json').read_text())
    official=load_boards(ROOT/'data/sudoku/hard/solutions.pickle')
    seen={known_transform_key([s[i]for i in CELL_POS])for s in official}
    pilot=json.loads((ROOT/'audit/metacognition/confirmation-generator-pilot/puzzles.json').read_text())
    seen.update(known_transform_key(row['solution'])for row in pilot)
    rng=random.Random(config['generation_seed']);puzzles=[];solutions=[];rejected=[];attempts=0;start=time.perf_counter()
    while len(puzzles)<config['n']:
        attempts+=1
        if attempts>config['attempt_cap']:raise RuntimeError('Preset generation cap reached')
        solution=backtracking([0]*81,limit=1,rng=rng)[0]
        key=known_transform_key(solution)
        if key in seen:rejected.append(dict(attempt=attempts,reason='known_transform'));continue
        puzzle=solution.copy();order=list(range(81));rng.shuffle(order)
        for cell in order:
            old,puzzle[cell]=puzzle[cell],0
            if len(backtracking(puzzle))!=1:puzzle[cell]=old
            if sum(bool(v)for v in puzzle)==30:break
        if sum(bool(v)for v in puzzle)!=30:
            rejected.append(dict(attempt=attempts,reason='above_30_clues'));continue
        assert exact_cover_count(puzzle)==1
        assert backtracking(puzzle,limit=1)[0]==solution
        seen.add(key);puzzles.append(puzzle);solutions.append(solution)
        if len(puzzles)%100==0:print(f'new unique corpus {len(puzzles)}/1000',flush=True)
    np.savez_compressed(EXTERNAL/'corpus.npz',puzzles=np.array(puzzles,dtype=np.uint8),solutions=np.array(solutions,dtype=np.uint8),groups=np.arange(100000,100000+len(puzzles)))
    write_json(EXTERNAL/'corpus-manifest.json',dict(status='complete',n=len(puzzles),attempts=attempts,rejections=rejected,
        config_sha256=file_hash(HERE/'external-design-v0.1.json'),corpus_sha256=file_hash(EXTERNAL/'corpus.npz'),
        independent_exact_cover_uniqueness_verified=True,no_ncm_scoring=True,wall_seconds=time.perf_counter()-start,
        caveat='Generated external distribution; exclude known digit/dihedral transforms, not all isomorphisms or pretraining contamination'))


def fit():
    verification=json.loads((BASE/'development/verification.json').read_text());assert verification['status']=='passed'
    corpus=json.loads((EXTERNAL/'corpus-manifest.json').read_text())
    assert corpus['status']=='complete'and file_hash(EXTERNAL/'corpus.npz')==corpus['corpus_sha256']
    destination=EXTERNAL/'models';destination.mkdir(exist_ok=False)
    protocol=json.loads((HERE/'protocol-v0.1.json').read_text())
    data={k:np.load(BASE/f'training/{k}.npy',mmap_mode='r',allow_pickle=False)for k in
        ['visible','hidden','group','cell','current_error','persistent']}
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text());groups=np.unique(data['group'])
    assert groups.tolist()==split['partition_indices']['train']
    fold=np.array([split['train_cv_fold'][str(g)]for g in groups]);noise,shuffle=controls(data)
    values={m:{s:[]for s in protocol['lambda_grid']}for m in protocol['models']}
    for k in range(5):
        matrices,t,v=prepare(data,groups[fold!=k],groups[fold==k],noise,shuffle)
        for name,(a,b,_)in matrices.items():
            for strength in protocol['lambda_grid']:
                fitted=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
                metric=grouped_metrics(data['persistent'][v],b@fitted['weight']+fitted['intercept'],data['group'][v])
                values[name][strength].append(metric['within_puzzle_auc'])
        print(f'final correction CV {k+1}/5',flush=True)
    # Empty validation is allowed by preprocessing; final fit uses every old TRAINING group.
    matrices,t,_=prepare(data,groups,np.array([],dtype=int),noise,shuffle)
    selection={}
    for name,(a,_,preprocessing)in matrices.items():
        means={s:float(np.mean(v))for s,v in values[name].items()}
        strength=max(s for s in means if means[s]>=max(means.values())-1e-8)
        fitted=fit_logistic(a,data['persistent'][t],data['group'][t],strength,protocol['optimizer'])
        np.savez_compressed(destination/f'{name}.npz',weight=fitted['weight'],intercept=np.array(fitted['intercept']),**preprocessing)
        selection[name]=dict(strength=strength,cv_mean_within_auc=means,iterations=fitted['iterations'],max_abs_gradient=fitted['max_abs_gradient'])
    write_json(EXTERNAL/'lock.json',dict(status='locked_before_external_inference',models=protocol['models'],selection=selection,
        source_files=fingerprint(),corpus_manifest_sha256=file_hash(EXTERNAL/'corpus-manifest.json'),
        model_files={p.name:file_hash(p)for p in destination.glob('*.npz')},n=1000,
        primary='macro within-puzzle AUROC difference visible_hidden-visible among mixed current-error groups',
        bootstrap='5000 paired mixed-group resamples seed20261105',sample_size_fixed_no_extension=True,
        caveat='External-distribution exploratory evaluation; no power guarantee; prior H1 calibration/test not reused'))


def locked():
    lock=json.loads((EXTERNAL/'lock.json').read_text());assert lock['source_files']==fingerprint()
    assert lock['corpus_manifest_sha256']==file_hash(EXTERNAL/'corpus-manifest.json')
    corpus=json.loads((EXTERNAL/'corpus-manifest.json').read_text())
    assert corpus['corpus_sha256']==file_hash(EXTERNAL/'corpus.npz')
    for name,digest in lock['model_files'].items():assert file_hash(EXTERNAL/'models'/name)==digest
    return lock


def extract():
    locked();destination=EXTERNAL/'traces';destination.mkdir(exist_ok=False)
    verify_assets(ROOT,'hard',include_data=True)
    corpus=np.load(EXTERNAL/'corpus.npz',allow_pickle=False)
    def tokens(cells):
        result=np.full(89,10,dtype=np.int64);result[CELL_POS]=cells;return [11,*result.tolist(),11]
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model)
    parts={};records=[];start=time.perf_counter()
    for offset in range(0,1000,8):
        puzzles=corpus['puzzles'][offset:offset+8];gold=corpus['solutions'][offset:offset+8]
        groups=corpus['groups'][offset:offset+8]
        prefix=model.prefix([tokens(c)for c in puzzles])
        reference=model.predict(prefix);prediction,trace,raw,_=instrumented_prediction(model,prefix)
        assert_prediction_equal(reference,prediction)
        eager=offset%80==0 or offset+8>=1000
        if eager:assert_prediction_equal(model.predict(prefix,optimized=False),prediction)
        hidden,logits=local_readout(model,torch.from_numpy(trace['codes'][:,:116].astype(np.int64)))
        assert np.array_equal(torch.from_numpy(logits).double().softmax(-1).numpy(),trace['decoder_probs'][:,8])
        visible=visible_features(prefix.numpy(),trace,logits);blank=puzzles==0
        current=trace['decoded_ids'][:,8,CELL_POS];final=trace['decoded_ids'][:,11,CELL_POS]
        block=dict(visible=visible[blank],hidden=hidden[:,CELL_POS][blank],
            group=np.broadcast_to(groups[:,None],blank.shape)[blank],cell=np.broadcast_to(np.arange(81),blank.shape)[blank],
            current_error=(current!=gold)[blank],persistent=(final!=gold)[blank])
        for key,value in block.items():parts.setdefault(key,[]).append(value)
        records.append(dict(groups=groups.tolist(),cached_bitwise_equal=True,eager_bitwise_checked=eager,
                            final_successes=int((final==gold).all(1).sum())))
        del reference,prediction,trace,raw
        if offset%80==0:print(f'external NCM {offset+len(puzzles)}/1000',flush=True)
    assert state_hash(model)==before;locked()
    for key,value in parts.items():np.save(destination/f'{key}.npy',np.concatenate(value))
    write_json(destination/'report.json',dict(status='complete',lock_sha256=file_hash(EXTERNAL/'lock.json'),
        model_state_sha256_before_and_after=before,batches=records,wall_seconds=time.perf_counter()-start,
        arrays={p.name:file_hash(p)for p in destination.glob('*.npy')}))


def evaluate():
    lock=locked();output=EXTERNAL/'analysis';output.mkdir(exist_ok=False)
    extraction=json.loads((EXTERNAL/'traces/report.json').read_text());assert extraction['status']=='complete'
    for name,digest in extraction['arrays'].items():assert file_hash(EXTERNAL/'traces'/name)==digest
    data={k:np.load(EXTERNAL/f'traces/{k}.npy',mmap_mode='r',allow_pickle=False)for k in
        ['visible','hidden','group','cell','current_error','persistent']}
    shuffle=shuffled_rows(data['group']);rows=np.flatnonzero(data['current_error'])
    scores={}
    for name in lock['models']:
        with np.load(EXTERNAL/f'models/{name}.npz',allow_pickle=False)as model:
            x=transform(model,data,name,rows,shuffle)
            scores[name]=x@model['weight']+model['intercept']
    y,g=data['persistent'][rows],data['group'][rows]
    result={m:grouped_metrics(y,s,g)for m,s in scores.items()}
    a={r['group']:r['auc']for r in result['visible']['per_group']};b={r['group']:r['auc']for r in result['visible_hidden']['per_group']}
    assert len(a)>0,'No mixed groups: report inconclusive, do not extend corpus'
    paired=np.array([b[k]-a[k]for k in sorted(a)]);rng=np.random.default_rng(20261105)
    deltas=np.array([rng.choice(paired,len(paired),replace=True).mean()for _ in range(5000)])
    np.savez_compressed(output/'predictions.npz',group=g,cell=data['cell'][rows],persistent=y,rows=rows,**scores)
    report=dict(status='complete',models=lock['models'],lock_sha256=file_hash(EXTERNAL/'lock.json'),n_groups=1000,
        current_errors=len(rows),persistent_errors=int(y.sum()),mixed_groups=len(a),results=result,
        primary_delta_within_auc=float(paired.mean()),paired_group_bootstrap_95=np.quantile(deltas,[.025,.975]).tolist(),
        corpus_distribution='randomized backtracking plus greedy clue removal, different from official corpus',
        caveat='Oracle conditional error cohort; fixed-predictor group bootstrap, exploratory external evaluation, not native metacognition proof')
    check_scores(report,dict(group=g,persistent=y,**scores))
    report['artifacts']={p.name:file_hash(p)for p in output.glob('*.npz')}
    write_json(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],within_auc={m:r['within_puzzle_auc']for m,r in result.items()},
        delta=report['primary_delta_within_auc'],interval=report['paired_group_bootstrap_95'],mixed_groups=len(a))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['generate','fit','extract','evaluate'])
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(20261101)
    {'generate':generate,'fit':fit,'extract':extract,'evaluate':evaluate}[args.stage]()

if __name__=='__main__':main()
