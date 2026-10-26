"""Only pre-step-9 states enter predictors; future observations are separate labels."""
from base import *
from runtime import Engine

@torch.inference_mode()
def main():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard')
    for name in ['official','external']:
        p,s=boards('hard') if name=='official' else external_boards();chunks={};cached=np.load(ROOT/('results/hard.npz' if name=='official' else 'results/external.npz'))
        start=time.perf_counter()
        for lo in range(0,len(p),16):
            ix=slice(lo,lo+16)
            with Engine(model,p[ix]) as engine:r=engine.run(capture=True)
            assert np.array_equal(r['codes'].numpy(),cached['codes'][ix])
            assert np.array_equal(r['decoded8'].numpy(),cached['decoded'][ix,7])
            assert np.array_equal(r['decoded9'].numpy(),cached['decoded'][ix,8])
            for key in ['z8','h8','prob8','decoded8','decoded9','final_ids','codes']:
                v=r[key].numpy()
                if key in ['z8','prob8']:v=v[:,CELL]
                chunks.setdefault(key,[]).append(v.copy())
            if lo%128==0:log('extract_revision',corpus=name,done=min(lo+16,len(p)),total=len(p),seconds=time.perf_counter()-start)
        out={key:np.concatenate(v) for key,v in chunks.items()};out.update(prefix=p,reference=s)
        np.savez_compressed(D/f'results/revision_{name}.npz',**out)
        log('extract_revision_complete',corpus=name,seconds=time.perf_counter()-start)

if __name__=='__main__':main()
