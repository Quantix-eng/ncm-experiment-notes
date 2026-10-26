"""Check whether graph freezing pays for itself after compilation and warmup."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from ncm_mac.sudoku import SudokuModel,load_boards
from cpu_candidates import InferenceGraph


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    model=SudokuModel(ROOT/'artifacts/sudoku/easy')
    boards=load_boards(ROOT/'data/sudoku/easy/puzzles.pickle')
    eager=InferenceGraph(model).eval()
    p=model.prefix(boards[:8]);start=time.perf_counter()
    frozen=torch.jit.freeze(torch.jit.trace(eager,p,check_inputs=[(model.prefix(boards[8:16]),)]))
    report=dict(compilation_seconds=time.perf_counter()-start,benchmarks={})
    for batch in [1,8]:
        prefix=model.prefix(boards[:batch])
        a,b=eager(prefix),frozen(prefix)
        if not all(torch.equal(x,y) for x,y in zip(a,b)):raise RuntimeError('Trace changes output')
        for _ in range(2):eager(prefix);frozen(prefix)
        times=dict(eager=[],frozen=[])
        for i in range(7):
            for name,run in ([('eager',eager),('frozen',frozen)] if i%2==0 else [('frozen',frozen),('eager',eager)]):
                start=time.perf_counter();run(prefix);times[name].append(time.perf_counter()-start)
        report['benchmarks'][str(batch)]=dict(seconds=times,speedup=statistics.median(times['eager'])/statistics.median(times['frozen']))
        (ROOT/'audit/cpu-graph-paired.json').write_text(json.dumps(report,indent=2)+'\n')
        print(batch,report['benchmarks'][str(batch)],flush=True)


if __name__=='__main__':main()
