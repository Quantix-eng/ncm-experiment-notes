"""Wait for independent extraction jobs, then run the dependent pipeline."""
import os,sys,time,subprocess
from pathlib import Path
D=Path(__file__).resolve().parents[1];os.chdir(D)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ['PYTHONPYCACHEPREFIX']=str(D/'.cache/pycache');os.environ['MPLCONFIGDIR']=str(D/'.cache/matplotlib')
python=str(D.parent/'.venv/bin/python')
def waitfor(names):
    while not all((D/'results'/n).exists() for n in names):time.sleep(5)
def run(script,*extra):
    print('START',script,flush=True)
    tag=script+('_conditional' if '--conditional' in extra else '')
    with (D/'logs'/f'{tag}.log').open('w') as log:
        r=subprocess.run([python,'-u',f'scripts/{script}.py',*extra],stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:print('FAILED',script,r.returncode,flush=True);sys.exit(r.returncode)
    print('COMPLETE',script,flush=True)
waitfor(['revision_official.npz','revision_external.npz']);run('probes');run('conditional');run('steer_revision');run('steer_revision','--conditional')
waitfor(['level5_diagnostic.json','level5_interventions.npz','level5_interventions.json','pooling_algebra.json','kmeans_centers.npz','dropout_control.json','dropout_control.npz']);run('report');run('verify')
print('ALL_COMPLETE',flush=True)
