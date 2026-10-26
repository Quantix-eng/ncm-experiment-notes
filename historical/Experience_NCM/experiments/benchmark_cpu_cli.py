"""Fresh-process end-to-end control, with no parent PyTorch/model residency."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    results={}
    for workers in [1,4]:
        output=ROOT/f'audit/cpu-cli-w{workers}.json'
        command=[sys.executable,'-m','ncm_mac.cli','--root',str(ROOT),'--difficulty','easy',
                 '--n','513','--workers',str(workers),'--output',str(output)]
        started=time.perf_counter()
        with (ROOT/f'audit/cpu-cli-w{workers}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[str(workers)]=dict(wall_seconds=time.perf_counter()-started,report=json.loads(output.read_text()))
        print(workers,results[str(workers)]['wall_seconds'],flush=True)
    assert results['1']['report']['predictions_sha256']==results['4']['report']['predictions_sha256']
    results['answers_equal']=True
    results['wall_speedup']=results['1']['wall_seconds']/results['4']['wall_seconds']
    results['note']='One fresh invocation per configuration; includes import, asset checks, load, warmup, inference and shutdown'
    (ROOT/'audit/cpu-cli-control.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
