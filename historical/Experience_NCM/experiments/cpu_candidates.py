"""Screen exact-FP32 CPU candidates; production code remains unchanged."""
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from ncm_mac.sudoku import SudokuModel, load_boards
from ncm_mac.cached import generate_cached


class InferenceGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vqvae=model.vqvae
        self.net=model.net

    def forward(self,prefix):
        codes,z=generate_cached(self.net,prefix)
        return codes,z,self.vqvae.decoder(z)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    model=SudokuModel(ROOT/'artifacts/sudoku/easy')
    boards=load_boards(ROOT/'data/sudoku/easy/puzzles.pickle')
    prefix=model.prefix(boards[:8])
    eager=InferenceGraph(model).eval()
    baseline=eager(prefix)
    results=[]
    def measure(name,run):
        out=run(prefix)
        equal=[torch.equal(x,y) for x,y in zip(baseline,out)]
        samples=[]
        for _ in range(5):
            start=time.perf_counter();run(prefix);samples.append(time.perf_counter()-start)
        row=dict(candidate=name,equal=equal,logit_max_abs=(baseline[2]-out[2]).abs().max().item(),
                 seconds=samples,median=statistics.median(samples))
        results.append(row);print(json.dumps(row),flush=True)
        (ROOT/'audit/cpu-candidates.json').write_text(json.dumps(results,indent=2)+'\n')
    measure('baseline',eager)
    weights=[]
    for module in eager.modules():
        if isinstance(module,torch.nn.Linear):
            weights.append((module,module.weight.data))
            module.weight.data=module.weight.data.t().contiguous().t()
    measure('column_major_weights',eager)
    for module,weight in weights:module.weight.data=weight
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        traced=torch.jit.trace(eager,prefix,check_inputs=[(model.prefix(boards[8:16]),)],strict=True)
        frozen=torch.jit.freeze(traced)
    (ROOT/'audit/cpu-trace-warnings.json').write_text(json.dumps(dict(
        compile_seconds=time.perf_counter()-started,warnings=[str(x.message) for x in caught]),indent=2)+'\n')
    measure('torchscript_frozen',frozen)


if __name__=='__main__':main()
