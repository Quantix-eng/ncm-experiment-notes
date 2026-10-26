"""Post-hoc training-only decoder ablations; no change to generator or fitted probes."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/metacognition'))
sys.path.insert(0,str(ROOT/'experiments/metacognition/dynamics'))
from native import LEVELS
from run_pilot import SudokuModel,load_boards,file_hash,state_hash,write_json
from features import CELL_POS,observed_counts

@torch.inference_mode()
def main():
    torch.set_num_threads(4);dest=ROOT/'audit/metacognition/native-stress-v1/codec-training.json';assert not dest.exists()
    start=time.perf_counter();source=ROOT/'audit/metacognition/train-v0.2';report=json.loads((source/'report.json').read_text())
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');before=state_hash(model);refs=load_boards(ROOT/'data/sudoku/hard/solutions.pickle')
    allrows=[];norms=[]
    for entry in report['batches'][:16]:
        assert file_hash(source/entry['file'])==entry['sha256']
        with np.load(source/entry['file'],allow_pickle=False)as trace:
            codes=torch.from_numpy(trace['codes'].astype(np.int64));prefix=trace['prefix'];puzzles=prefix[:,1:-1][:,CELL_POS]
            gold=np.array([refs[g]for g in trace['sample_indices']])[:,CELL_POS]
            contributions=[];offset=0
            for k,n in enumerate(LEVELS):
                e=model.vqvae.vq.embeddings[k](codes[:,offset:offset+n]);offset+=n
                if k<11:e=model.vqvae.vq._up_to_full(e,k,89)
                contributions.append(model.vqvae.vq._apply_quant_resi(e,k))
            full=torch.zeros_like(contributions[0])
            for c in contributions:full.add_(c)
            variants=dict(full=full,single12=contributions[11],without12=sum(contributions[:11],torch.zeros_like(full)),
                without8=sum([c for k,c in enumerate(contributions)if k!=7],torch.zeros_like(full)),
                suffix9=sum(contributions[8:],torch.zeros_like(full)),zero=torch.zeros_like(full))
            expected=trace['decoded_ids'][:,-1]
            for name,z in variants.items():
                ids=model.vqvae.decoder(z).argmax(-1).numpy();_,_,_,_,valid=observed_counts(ids,puzzles)
                if name=='full':assert np.array_equal(ids,expected)
                for i,g in enumerate(trace['sample_indices']):allrows.append(dict(group=int(g),variant=name,
                    solved=bool(valid[i]and np.array_equal(ids[i,CELL_POS],gold[i])),
                    cell_accuracy=float((ids[i,CELL_POS]==gold[i]).mean()),identical_clean=bool(np.array_equal(ids[i],expected[i]))))
            norms.extend(np.stack([c.double().flatten(1).norm(dim=1).numpy()for c in contributions],axis=1).tolist())
    assert before==state_hash(model);summary={}
    for name in variants:
        r=[r for r in allrows if r['variant']==name]
        summary[name]=dict(n=len(r),solved=sum(v['solved']for v in r),identical_clean=sum(v['identical_clean']for v in r),cell_accuracy=float(np.mean([v['cell_accuracy']for v in r])))
    write_json(dest,dict(status='complete',scope='posthoc_training_only_not_external_evaluation',n=128,summary=summary,
        level_contribution_mean_norm=np.mean(norms,axis=0).tolist(),rows=allrows,seconds=time.perf_counter()-start,
        model_state_sha256_before_and_after=before,source_sha256=file_hash(Path(__file__)),
        caveat='Decoded generated codes with omitted contributions; does not test whether generator needs earlier levels to produce final codes'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
