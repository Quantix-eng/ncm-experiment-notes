"""Preserve the pre-revision publication and check the scope of editorial changes.

This audit checks unchanged files and reporting requirements. It is not an
independent scientific review or retrospective proof against p-hacking.
"""
from base import *
import zipfile

BASELINE=D/'results/editorial_revision_baseline.json'

def snapshot():
    assert not BASELINE.exists(),'Do not replace the pre-revision baseline'
    manifest=json.loads((D/'results/final_manifest.json').read_text())
    assert all(sha(D/p)==h for p,h in manifest.items()),'Existing manifest mismatch'
    scientific_scripts={'base.py','benchmark.py','discovery.py','experiment.py','freeze.py','policy.py','runtime.py'}
    mutable_results={'results/verification.json','results/final_manifest.json','results/progress.json','results/pdf_inspection.json'}
    immutable={p:h for p,h in manifest.items() if p.startswith('data/') or (p.startswith('results/') and p not in mutable_results) or (p.startswith('scripts/') and Path(p).name in scientific_scripts) or p in ['protocol.json','PROTOCOLE_GELE.md','FEUILLE_DE_ROUTE.md','sources/previous_manifests.json','sources/environment.json']}
    backup=D/'sources/avant_revision_editoriale_20261024.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
        names=[p for p in manifest if p.startswith('scripts/') or p in ['RAPPORT.md','PAPER.md','PAPER.tex','PAPER.pdf','README.md','JOURNAL.md','sources_recherche_positive.zip','results/verification.json']]
        for p in names:z.write(D/p,p)
        z.write(D/'results/final_manifest.json','results/final_manifest.json')
    write(BASELINE,dict(purpose='Editorial revision following critical review; not a new experimental freeze or external preregistration',date='2026-10-24',previous_manifest_sha256=sha(D/'results/final_manifest.json'),historical_backup=str(backup.relative_to(D)),historical_backup_sha256=sha(backup),immutable_files=immutable))
    print(json.dumps({'snapshot':'created','immutable_files':len(immutable),'backup':str(backup.relative_to(D))}))

def audit():
    b=json.loads(BASELINE.read_text());mismatch=[p for p,h in b['immutable_files'].items() if sha(D/p)!=h]
    assert not mismatch,mismatch
    assert sha(D/b['historical_backup'])==b['historical_backup_sha256']
    requirements={
        'RAPPORT.md':['ni prérégistration indépendante','scripts expérimentaux','30 %','50,25 %','4 096','géométrique','nouveauté','237','trois donnant un autre chiffre'],
        'PAPER.md':['not an independent preregistration','experimental scripts','50.25%','4,096','237','three producing a different digit','novelty','geometry'],
        'README.md':['nouveauté','réplication indépendante'],
    }
    checked=[]
    for name,phrases in requirements.items():
        text=(D/name).read_text()
        for phrase in phrases:
            assert phrase in text,(name,phrase)
            checked.append(name+': '+phrase)
    a=np.load(D/'results/writing_interventions.npz');n=len(a['cell']);tp=CELL[a['cell']];outputs=a['patched'][np.arange(n)[:,None],np.arange(9)[None,:],tp[:,None]];target=np.arange(1,10)[None,:];old=a['native_digit'][:,None];eligible=a['eligible']
    failures=dict(target_hits=int(((outputs==target)&eligible).sum()),unchanged_failures=int(((outputs!=target)&(outputs==old)&eligible).sum()),different_digit_failures=int(((outputs!=target)&(outputs!=old)&eligible).sum()))
    assert failures==dict(target_hits=784,unchanged_failures=237,different_digit_failures=3)
    result=dict(status='passed',scope='File integrity and reporting consistency only; not independent scientific validation',date='2026-10-24',immutable_files_verified=len(b['immutable_files']),all_original_numerical_artifacts_and_frozen_rules_unchanged=True,historical_backup=b['historical_backup'],reporting_requirements=checked,new_experimental_data_generated=False,post_hoc_descriptive_failure_counts=failures)
    write(D/'results/editorial_revision_audit.json',result)
    return result

if __name__=='__main__':
    if '--snapshot' in sys.argv:snapshot()
    else:print(json.dumps(audit()))
