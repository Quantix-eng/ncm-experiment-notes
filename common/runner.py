"""Local entry points and isolated workspaces for the relocated scientific code.

The compatibility layout exists only under an experiment's ignored runs directory.
Source files themselves live in src/ and common/, not in an archive tree.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
LAYOUT = REPO / 'common/layout.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def initialize(folder, config, name, assets):
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Use a simple run name, without path separators.')
    run = folder / 'runs' / name
    layout = read(LAYOUT)
    fingerprints = [(row['path'],sha(REPO/row['path'])) for row in layout['files']]
    fingerprints += [('runner',sha(Path(__file__))),('configuration',sha(folder/'experiment.json')),('note',sha(folder/'README.md'))]
    source_fingerprint=hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
    if run.exists():
        marker = read(run/'run.json')
        if marker['experiment'] != config['id']:
            raise ValueError('Wrong experiment workspace.')
        current = sha(LAYOUT)
        if marker['layout_sha256'] != current or marker.get('source_fingerprint') != source_fingerprint or marker['assets'] != str(assets):
            raise ValueError('Sources changed: use a new run name.')
        return run
    run.mkdir(parents=True)
    work = run/'workspace'
    layout = read(LAYOUT)
    copied=[]
    for row in layout['files']:
        relative = row['runtime_path']
        # Do not seed this experiment's outcome files into a new experiment.
        is_outcome = '/results/' in relative or relative.startswith('audit/')
        seed = relative.endswith(('discovery_tables.json', 'dev_observations.json'))
        if row['owner']==config['id'] and is_outcome and not seed:
            continue
        # The constructed-family runner must generate its own families.
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/row['path'],target)
        copied.append({'path':relative,'sha256':sha(target)})
    # Old scripts fingerprint protocol prose by its former filename. For new
    # runs, seal the current English note instead; never reuse an old seal.
    for relative in layout['runtime_documents']:
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('# Relocated execution note\n\nThis is a new local execution of relocated source code. It is not the historical protocol freeze.\n\n'+(folder/'README.md').read_text())
    for base in [work, work/'analyse_fondamentale_NCM']:
        for group in ['artifacts','data']:
            source=assets/group
            if source.exists():
                for p in source.rglob('*'):
                    if not p.is_file():continue
                    dst=base/group/p.relative_to(source)
                    if not dst.exists():
                        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(p.resolve())
        for dirname in ['results','logs','figures','.cache']:
            (base/dirname).mkdir(parents=True,exist_ok=True)
    lock=assets/'assets.lock.json'
    if lock.exists():shutil.copyfile(lock,work/'assets.lock.json')
    if (assets/'pythia-160m').exists():
        dst=work/'analyse_fondamentale_NCM/validation_externe_directeur/03_pythia/weights'
        if not dst.exists():dst.symlink_to((assets/'pythia-160m').resolve(),target_is_directory=True)
    # Ensure every study may create new results without pre-existing outcomes.
    for row in layout['files']:
        if row['owner']==config['id']:
            p=work/row['runtime_path']; (p.parent/'results').mkdir(exist_ok=True,parents=True) if p.suffix=='.py' and p.parent.name!='scripts' else None
    marker={'experiment':config['id'],'layout_sha256':sha(LAYOUT),'source_fingerprint':source_fingerprint,'assets':str(assets),
            'scope':'relocated execution, new local seals; not the historical freeze',
            'seeded_files':copied}
    (run/'run.json').write_text(json.dumps(marker,indent=2)+'\n')
    return run


def main(folder):
    config=read(folder/'experiment.json')
    parser=argparse.ArgumentParser(description=config['title'])
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Recalculate local recorded-data checks; no model execution.')
    sub.add_parser('stages',help='List real source files, stage names and prerequisites.')
    p=sub.add_parser('run',help='Execute a scientific stage in an isolated workspace.')
    p.add_argument('stage',choices=config['stages'])
    p.add_argument('--name',default='trial-01')
    p.add_argument('--assets',type=Path,default=REPO/'assets')
    p.add_argument('--plan',action='store_true',help='Show the source and arguments without execution.')
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[],help='Additional arguments for the scientific script.')
    args=parser.parse_args()
    if args.command=='check':
        subprocess.run([sys.executable,str(folder/'src/check_recorded.py')],cwd=folder,check=True)
        return
    if args.command=='stages':
        print('Inputs:',config['requirements'])
        for name,step in config['stages'].items():
            print(name,':',step['source'],*step['args'])
            print(' ',step['note'])
        return
    step=config['stages'][args.stage]
    if args.plan:
        print(json.dumps(step,indent=2));return
    run=initialize(folder,config,args.name,args.assets.resolve())
    row=next(x for x in read(LAYOUT)['files'] if x['path']==step['source'])
    source=REPO/row['path'] if step.get('direct') else run/'workspace'/row['runtime_path']
    extra=list(args.extra)
    if step.get('asset_argument'):extra.extend([step['asset_argument'],str(args.assets.resolve())])
    if step.get('output_argument'):extra.extend([step['output_argument'],str(run/'checkpoint.json')])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(run/'cache/matplotlib'))
    (run/'cache/matplotlib').mkdir(parents=True,exist_ok=True)
    if step.get('function'):
        code='import sys,runpy;sys.path.insert(0,sys.argv[1]);n=runpy.run_path(sys.argv[2]);n[sys.argv[3]](*sys.argv[4:])'
        command=[sys.executable,'-B','-c',code,str(source.parent),str(source),step['function'],*step['args'],*extra]
    else:
        command=[sys.executable,'-B',str(source),*step['args'],*extra]
    print('Executing:',step['source'],flush=True)
    print('Workspace:',run,flush=True)
    result=subprocess.run(command,cwd=source.parent,env=env)
    record={'stage':args.stage,'source':step['source'],'exit_code':result.returncode,
            'scope':'new relocated execution; recorded snapshots are unchanged','arguments':command}
    with (run/'execution.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if result.returncode:
        print('Stage failed. Check its input/dependency requirements; this is not a successful reproduction.',file=sys.stderr)
    raise SystemExit(result.returncode)
