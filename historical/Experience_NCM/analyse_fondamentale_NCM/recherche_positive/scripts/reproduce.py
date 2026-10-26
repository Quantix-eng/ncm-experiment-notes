"""Computationally reproduce the historical campaign in a new sibling folder.

Uses the preserved pre-editorial sources so the delivered numerical evidence and
its editorial integrity baseline are never overwritten. This is not a new test
of generalization or an independent scientific replication.
"""
import argparse,io,json,shutil,subprocess,zipfile
from pathlib import Path

def main():
    study=Path(__file__).resolve().parents[1];root=study.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',default=str(root/'reproduction_alphabet'))
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();dest=Path(args.destination).resolve()
    if dest.parent!=root or dest.exists():parser.error('Destination must be a new direct child of analyse_fondamentale_NCM')
    baseline=json.loads((study/'results/editorial_revision_baseline.json').read_text())
    import hashlib
    backup=study/baseline['historical_backup']
    assert hashlib.sha256(backup.read_bytes()).hexdigest()==baseline['historical_backup_sha256']
    with zipfile.ZipFile(backup) as history:
        raw=history.read('sources_recherche_positive.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as sources:
        assert sources.testzip() is None
        members=[m for m in sources.infolist() if not m.filename.startswith('parent/')]
        assert all(not Path(m.filename).is_absolute() and '..' not in Path(m.filename).parts for m in members)
        dest.mkdir();sources.extractall(dest,members=members)
    # References and frozen development traces required by the historical runner.
    for name in ['data/fresh.json','results/discovery_pilot.npz']:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(study/name,target)
    checked=[]
    for name,expected in baseline['immutable_files'].items():
        if name.startswith('scripts/') or name in ['data/fresh.json','results/discovery_pilot.npz']:
            assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected,name
            checked.append(name)
    (dest/'logs').mkdir(exist_ok=True);(dest/'figures').mkdir(exist_ok=True)
    (dest/'REPRODUCTION_NOTICE.md').write_text('''# Computational reproduction of the historical campaign

This folder reruns the preserved pre-editorial scripts on the same frozen inputs. It is not independent confirmation. Timing may vary. Regenerated reports use the older wording; the critically revised interpretation remains in ../recherche_positive/RAPPORT.md and PAPER.md. Do not interpret the historical wording as a new scientific endorsement. The delivered study and its numerical evidence remain unchanged.
''')
    print(json.dumps({'prepared':str(dest),'historical_sources':True,'source_and_input_hashes_verified':checked,'new_independent_test':False}),flush=True)
    if not args.prepare_only:subprocess.run(['sh','run.sh'],cwd=dest,check=True)

if __name__=='__main__':main()
