"""Paired throughput study and exact per-batch fingerprints against 254bb49."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from ncm_mac.sudoku import SudokuModel,load_boards,metrics,verify_assets
from ncm_mac.parallel import SudokuPool,prediction_fingerprints


def main():
    for path in ['src/ncm_mac/cached.py','src/ncm_mac/sudoku.py','src/ncm_mac/_vendor/models/common.py',
                 'src/ncm_mac/_vendor/models/next_concept.py','src/ncm_mac/_vendor/models/quant.py',
                 'src/ncm_mac/_vendor/models/vae.py','src/ncm_mac/_vendor/models/vqvae.py']:
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show','254bb49:'+path],cwd=ROOT),path
    torch.set_num_threads(4)
    report=dict(reference='254bb49',batch_size=8,workers=4,worker_threads=1,
                parent_threads=4,repeats=3,tasks={})
    for difficulty in ['easy','medium','hard']:
        verify_assets(ROOT,difficulty,include_data=True)
        dd=ROOT/'data/sudoku'/difficulty;checkpoint=ROOT/'artifacts/sudoku'/difficulty
        all_boards,all_refs=load_boards(dd/'puzzles.pickle'),load_boards(dd/'solutions.pickle')
        indices=[i*1999//127 for i in range(128)]
        boards,refs=[all_boards[i] for i in indices],[all_refs[i] for i in indices]
        batches=[boards[i:i+8] for i in range(0,len(boards),8)]
        model=SudokuModel(checkpoint)
        expected=[prediction_fingerprints(model.predict(model.prefix(b))) for b in batches]
        started=time.perf_counter()
        with SudokuPool(checkpoint,workers=4,threads=1) as pool:
            warm=list(pool.map([batches[0]]*8))
            startup=time.perf_counter()-started
            timings=dict(reference=[],parallel=[])
            worker_memory={}
            for rep in range(3):
                for mode in (['reference','parallel'] if rep%2==0 else ['parallel','reference']):
                    started=time.perf_counter()
                    if mode=='reference':
                        original=[model.predict(model.prefix(b)).ids.tolist() for b in batches]
                    else:
                        actual=list(pool.map(batches,audit=True))
                    timings[mode].append(time.perf_counter()-started)
                assert [r.fingerprints for r in actual]==expected,'Exact tensor parity failed'
                assert [r.ids for r in actual]==original,'Answer ordering failed'
                for r in actual:worker_memory[r.worker_pid]=max(worker_memory.get(r.worker_pid,0),r.peak_rss_bytes)
            # Cover an incomplete last batch and interleaving after full batches.
            partial=[boards[:8],boards[8:9]]
            tail=list(pool.map(partial,audit=True))
            assert [r.fingerprints for r in tail]==[prediction_fingerprints(model.predict(model.prefix(b))) for b in partial]
        predictions=[row for r in actual for row in r.ids]
        baseline=json.loads(subprocess.check_output(['git','show','254bb49:audit/sudoku-parity.json'],cwd=ROOT))
        sha=hashlib.sha256(json.dumps(predictions,separators=(',',':')).encode()).hexdigest()
        assert sha==baseline['tasks'][difficulty]['predictions_sha256'],'Committed CPU8 answers differ'
        report['tasks'][difficulty]=dict(indices=indices,exact_codes_reconstruction_logits=True,tail_batch_equal=True,
            predictions_sha256=sha,metrics=metrics(predictions,refs,boards),startup_seconds=startup,
            warmed_worker_pids=sorted({r.worker_pid for r in warm}),seconds=timings,
            speedup=statistics.median(timings['reference'])/statistics.median(timings['parallel']),
            per_worker_peak_rss_bytes=worker_memory)
        (ROOT/'audit/cpu-parallel-validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(difficulty,report['tasks'][difficulty]['speedup'],timings,flush=True)
        del model,pool,actual,tail
        gc.collect()


if __name__=='__main__':main()
