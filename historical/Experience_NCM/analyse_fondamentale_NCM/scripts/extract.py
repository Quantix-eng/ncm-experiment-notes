"""Fresh inference, independent of all previous project results."""
from common import *
import argparse

def extract(d):
    start=time.perf_counter(); p,s=boards(d); model=SudokuModel(ROOT/f'artifacts/sudoku/{d}')
    t=trace(model,p[:16]); original=model.predict(torch.tensor(p[:16]),optimized=False)
    parity=dict(codes_equal=bool(np.array_equal(t['codes'],original.codes.numpy())),
        outputs_equal=bool(np.array_equal(t['decoded'][:,-1],original.ids.numpy())))
    assert all(parity.values())
    allout={}; enc=[]; recon=[]
    for lo in range(0,len(p),P['batch_size']):
        hi=min(len(p),lo+P['batch_size'])
        tr=trace(model,p[lo:hi],local=(d=='hard'))
        for key,val in tr.items(): allout.setdefault(key,[]).append(val)
        with torch.inference_mode():
            idx=model.vqvae.encode(torch.tensor(s[lo:hi])); snapshots,=model.vqvae.vq.idx_to_zhats(idx,return_zlevels=False)
            enc.append(idx.numpy()); recon.append(np.stack([model.vqvae.decoder(z).argmax(-1).numpy() for z in snapshots],1))
        if lo%128==0: progress('extract',checkpoint=d,done=hi,total=len(p),seconds=time.perf_counter()-start)
    allout={key:np.concatenate(val) for key,val in allout.items()}
    allout.update(prefix=p,reference=s,encoded=np.concatenate(enc),reconstructed=np.concatenate(recon))
    np.savez_compressed(ROOT/f'results/{d}.npz',**allout)
    met=evaluate(allout['decoded'][:,-1],s,p); met['solved_wilson95']=ci_rate(met['solved'],len(p))
    write(ROOT/f'results/reproduction_{d}.json',dict(metrics=met,parity=parity,seconds=time.perf_counter()-start,levels=list(model.net.levels)))
    progress('extract_complete',checkpoint=d,**met,seconds=time.perf_counter()-start)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('difficulty',choices=['hard','easy','medium']); args=a.parse_args()
    import fcntl
    lock=(ROOT/f'.cache/extract_{args.difficulty}.lock').open('w')
    waited=False
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        waited=True; print('Waiting for concurrent extraction of',args.difficulty,flush=True); fcntl.flock(lock,fcntl.LOCK_EX)
    if waited and (ROOT/f'results/{args.difficulty}.npz').exists():
        print('Concurrent extraction saved its arrays; no duplicate inference (check summary/recovery log)',flush=True)
    else: extract(args.difficulty)
