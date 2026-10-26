"""Schedule independent stages as their prerequisites finish; no model API calls."""
from pathlib import Path
import subprocess,os,time,json,sys
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ['PYTHONPYCACHEPREFIX']=str(ROOT/'.cache/pycache')
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
python=str(ROOT/'.venv/bin/python'); running={}; complete=set(); handles=[]
def exists(*names):return all((ROOT/('results/'+n)).exists() for n in names)
jobs={
 'primary':(['scripts/analyze.py','--stage','primary'],lambda:exists('reproduction_hard.json')),
 'causal':(['scripts/experiments.py','--stage','causal'],lambda:'primary' in complete),
 'supplementary':(['scripts/analyze.py','--stage','supplementary'],lambda:'primary' in complete and exists('reproduction_easy.json','reproduction_medium.json')),
 'scores':(['scripts/experiments.py','--stage','scores'],lambda:'supplementary' in complete and exists('official_uniqueness.json','cross_models.json','functional_substitutions.json','symmetry_traces.npz','external.json') and len(json.loads((ROOT/'results/official_uniqueness.json').read_text()))==3),
 'report':(['scripts/report.py'],lambda:all(k in complete for k in ['primary','causal','supplementary','scores']) and exists('upstream_parity.json')),
 'verify':(['scripts/verify.py'],lambda:'report' in complete)
}
while len(complete)<len(jobs):
    for name,proc in list(running.items()):
        code=proc.poll()
        if code is not None:
            if code:print('FAILED',name,code,flush=True);sys.exit(code)
            complete.add(name);del running[name];print('COMPLETE',name,flush=True)
    for name,(args,ready) in jobs.items():
        if name not in running and name not in complete and ready():
            log=(ROOT/f'logs/{name}.log').open('w');handles.append(log)
            running[name]=subprocess.Popen([python,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
            print('START',name,flush=True)
    time.sleep(5)
print('ALL_COMPLETE',flush=True)
