"""Preserve batch boundaries and exact FP32 outputs while distributing batches."""
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from ncm_mac.sudoku import SudokuModel,load_boards

MODEL=None


def init_worker(checkpoint,threads):
    global MODEL
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    MODEL=SudokuModel(checkpoint)


def fingerprint(prediction):
    return [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
            for x in [prediction.codes,prediction.reconstruction,prediction.logits]]


def solve(boards):
    result=MODEL.predict(MODEL.prefix(boards))
    return dict(hashes=fingerprint(result),ids=result.ids.tolist(),
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    torch.set_num_threads(4)
    checkpoint=str(ROOT/'artifacts/sudoku/easy')
    model=SudokuModel(checkpoint)
    boards=load_boards(ROOT/'data/sudoku/easy/puzzles.pickle')[:64]
    batches=[boards[i:i+8] for i in range(0,len(boards),8)]
    reference=[fingerprint(model.predict(model.prefix(b))) for b in batches]
    results=[]
    samples=[]
    for _ in range(3):
        start=time.perf_counter()
        for batch in batches:model.predict(model.prefix(batch))
        samples.append(time.perf_counter()-start)
    results.append(dict(workers=1,threads=4,seconds=samples,median=statistics.median(samples),parity=True))
    for workers,threads in [(2,2),(4,1)]:
        started=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
              initializer=init_worker,initargs=(checkpoint,threads)) as pool:
            list(pool.map(solve,[batches[0]]*workers))
            startup=time.perf_counter()-started
            times=[]
            for _ in range(3):
                start=time.perf_counter();outputs=list(pool.map(solve,batches));times.append(time.perf_counter()-start)
                if [x['hashes'] for x in outputs]!=reference:
                    raise RuntimeError('Parallel scheduling changes exact outputs')
            row=dict(workers=workers,threads=threads,startup_seconds=startup,seconds=times,
                     median=statistics.median(times),parity=True,max_child_rss=max(x['rss'] for x in outputs))
            results.append(row);print(json.dumps(row),flush=True)
            (ROOT/'audit/cpu-parallel-screen.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
