"""Verify fixed external corpus, groups, hashes and saved probe predictions."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_pilot import ROOT,file_hash,write_json
from correction.external import locked
from correction.verify_development import transform,shuffled_rows,check_scores


def main():
    base=ROOT/'audit/metacognition/correction-v0.1/external'
    lock=locked();extraction=json.loads((base/'traces/report.json').read_text())
    report=json.loads((base/'analysis/report.json').read_text())
    assert report['status']==extraction['status']=='complete'
    assert report['lock_sha256']==extraction['lock_sha256']==file_hash(base/'lock.json')
    for name,digest in extraction['arrays'].items():assert file_hash(base/'traces'/name)==digest
    for name,digest in report['artifacts'].items():assert file_hash(base/'analysis'/name)==digest
    data={k:np.load(base/f'traces/{k}.npy',mmap_mode='r',allow_pickle=False)for k in
        ['visible','hidden','group','cell','current_error','persistent']}
    corpus=np.load(base/'corpus.npz',allow_pickle=False)
    assert np.unique(data['group']).tolist()==corpus['groups'].tolist()
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    assert not set(data['group'])&set(sum(split['partition_indices'].values(),[]))
    assert (np.count_nonzero(corpus['puzzles'],axis=1)==30).all()
    shuffled=shuffled_rows(data['group']);maximum=0.
    with np.load(base/'analysis/predictions.npz',allow_pickle=False)as scores:
        rows=scores['rows'];assert np.array_equal(rows,np.flatnonzero(data['current_error']))
        assert np.array_equal(scores['group'],data['group'][rows])
        assert np.array_equal(scores['persistent'],data['persistent'][rows])
        for model in lock['models']:
            with np.load(base/f'models/{model}.npz',allow_pickle=False)as saved:
                x=transform(saved,data,model,rows,shuffled)
                replay=x@saved['weight']+saved['intercept']
            maximum=max(maximum,float(np.max(abs(replay-scores[model]))))
            assert np.allclose(replay,scores[model],rtol=1e-12,atol=1e-10)
        check_scores(report,scores)
    write_json(base/'analysis/verification.json',dict(status='passed',report_sha256=file_hash(base/'analysis/report.json'),
        max_abs_replay_error=maximum,bootstrap_recomputed=True,group_disjointness_verified=True,fitting=False,model_execution=False))
    print('External correction probe analysis independently verified')

if __name__=='__main__':main()
