"""Check the research note against measurements and archive all executed local code."""
import json,hashlib,zipfile
from pathlib import Path
D=Path(__file__).resolve().parents[1];ROOT=D.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def main():
    a=json.loads((D/'results/audit.json').read_text());v=json.loads((D/'results/verification.json').read_text());text=(D/'ANALYSE_ET_PISTES.md').read_text();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    check('audit_passed',v['status']=='passed')
    hard=a['hard']['oracle_codec']['all_cells'];check('note_oracle_counts',hard['corrected9']==12717 and hard['reverted_later']==3176 and '3 176/12 717' in text)
    g=a['hard']['generator']['all_cells'];check('note_generator_counts',g['corrected9']==12333 and g['reverted_later']==3088 and '3 088/12 333' in text)
    check('note_oracle_final_accuracy',a['hard']['oracle_codec']['exact_grids_by_level']==[0]*11+[600]);check('note_replay_scope','16 premières solutions' in text and '35 contrôles' in text and v['n_checks']==35)
    check('proposals_marked_unexecuted','ne sont pas exécutées dans cette note' in text)
    prior=json.loads((ROOT/'dynamique_temporelle/sources/previous_manifests.json').read_text())
    for name,h in prior.items():check('previous_manifest_unchanged:'+name,sha(ROOT/name)==h)
    temporal=ROOT/'dynamique_temporelle';previous=json.loads((temporal/'results/final_manifest.json').read_text())
    for name in ['RAPPORT.md','PAPER.md','PAPER.pdf','scripts/publish.py','scripts/verify.py']:check('temporal_unchanged:'+name,sha(temporal/name)==previous[name])
    parents=['scripts/common.py','scripts/extract.py','protocol.json','requirements.txt','sources/requirements-lock.txt','results/splits.json','artifacts/sudoku/hard/config.yaml']
    with zipfile.ZipFile(D/'sources_analyse.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((D/'scripts').glob('*.py')):z.write(p,str(p.relative_to(D)))
        for name in ['ANALYSE_ET_PISTES.md','JOURNAL.md','README.md']:z.write(D/name,name)
        for name in parents:z.write(ROOT/name,'parent/'+name)
        for p in sorted((ROOT/'vendor/ncm_mac').rglob('*')):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.toml','.txt'] and '__pycache__' not in str(p):z.write(p,'parent/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(D/'sources_analyse.zip') as z:
        check('source_archive_crc',z.testzip() is None)
        for p in sorted((D/'scripts').glob('*.py')):check('source_archive_current:'+p.name,z.read('scripts/'+p.name)==p.read_bytes())
    write(D/'results/publication_checks.json',dict(status='passed',n_checks=len(checks),checks=checks,previous_campaign_manifests_unchanged=prior,temporal_manifest_sha256=sha(temporal/'results/final_manifest.json')))
    paths=[p for p in D.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name!='final_manifest.json']
    write(D/'results/final_manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(paths)})
    print(json.dumps(dict(status='passed',publication_checks=len(checks),audit_checks=v['n_checks'],files=len(paths))))
if __name__=='__main__':main()
