"""Archive the early-control decision and documentation hashes without inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'audit/metacognition/early-control-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=json.loads((BASE/'verification.json').read_text());assert audit['status']=='passed'
    assert audit['source_sha256']==sha(ROOT/'scripts/verify_early_control.py')
    for rel,h in audit['evidence_sha256'].items():assert sha(ROOT/rel)==h,rel
    docs=['RAPPORT_CONTROLE_PRECOCE.md','SYNTHESE_METACOGNITION.md','PLAN_RECHERCHE_METACOGNITION.md','FEUILLE_DE_ROUTE.md','experiments/metacognition/early_control/README.md']
    destination=BASE/'campaign-summary.json'
    for rel in docs:
        p=ROOT/rel
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            q=p.parent/target.strip('<>').split('#')[0]
            assert q.exists()or q==destination,(rel,target)
    out=dict(status='closed',utc=datetime.now(timezone.utc).isoformat(),decision=audit['decision'],independent_confirmation_started=False,
        verification_sha256=sha(BASE/'verification.json'),documents_sha256={rel:sha(ROOT/rel)for rel in docs},
        closure_source_sha256=sha(Path(__file__)),scope='Exploratory training-only feasibility; no new corpus, calibrated product or native metacognition claim.')
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
