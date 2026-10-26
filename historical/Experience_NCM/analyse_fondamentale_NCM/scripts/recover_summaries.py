"""Recover only JSON summaries after Hydra ListConfig serialization failure.

The already completed inference arrays are preserved and independently replayed
on the parity sample. This script does not change any predictions or features.
"""
from common import *
import zipfile
start=time.perf_counter();done=set()
while len(done)<3:
    for d in ['hard','easy','medium']:
        if d in done:continue
        target=ROOT/f'results/reproduction_{d}.json'
        if target.exists():done.add(d);continue
        logs=[ROOT/f'logs/extract_{d}.log',ROOT/f'logs/extract_{d}_parallel.log']
        log=next((f for f in logs if f.exists() and 'TypeError: ListConfig' in f.read_text()),None)
        if log is None:continue
        path=ROOT/f'results/{d}.npz';a=np.load(path)
        p,s=boards(d); assert np.array_equal(a['prefix'],p) and np.array_equal(a['reference'],s)
        model=SudokuModel(ROOT/f'artifacts/sudoku/{d}'); prefix=torch.tensor(p[:16]); original=model.predict(prefix,optimized=False);cached=model.predict(prefix)
        parity=dict(codes_equal=bool(np.array_equal(a['codes'][:16],original.codes.numpy()) and torch.equal(original.codes,cached.codes)),
                    outputs_equal=bool(np.array_equal(a['decoded'][:16,-1],original.ids.numpy()) and torch.equal(original.ids,cached.ids)))
        assert all(parity.values())
        met=evaluate(a['decoded'][:,-1],s,p);met['solved_wilson95']=ci_rate(met['solved'],len(p))
        old=[]
        for line in log.read_text().splitlines():
            try:
                obj=json.loads(line)
                if obj.get('stage')=='extract':old.append(obj)
            except json.JSONDecodeError:pass
        write(target,dict(metrics=met,parity=parity,levels=list(model.net.levels),
            last_progress_seconds=old[-1]['seconds'],timing_note='Last recorded extraction progress, not total runtime; summary recovered after JSON-only ListConfig serialization failure',
            recovery=dict(source_npz_sha256=sha(path),parity_replayed=True,log=str(log.relative_to(ROOT)))))
        progress('summary_recovered',checkpoint=d,**met);done.add(d)
    if len(done)<3:time.sleep(5)
progress('summary_recovery_complete',seconds=time.perf_counter()-start)
