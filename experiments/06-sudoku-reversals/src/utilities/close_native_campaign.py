"""Link existing results and documentation with hashes. No inference or fitting."""
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'audit/metacognition/native-dynamics-v1'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(p.read_text())

def main():
    prediction=read(BASE/'external/analysis/report.json');causal=read(BASE/'causal/report.json');verify=read(BASE/'verification.json')
    assert verify['status']=='passed'
    assert verify['report_sha256']==digest(BASE/'external/analysis/report.json')
    assert verify['causal_report_sha256']==digest(BASE/'causal/report.json')
    assert verify['source_sha256']==digest(ROOT/'scripts/verify_native_dynamics_bounded.py')
    lock=read(BASE/'lock.json')
    for p,h in lock['source_files'].items():assert digest(ROOT/p)==h,p
    stress=ROOT/'audit/metacognition/native-stress-v1';pilot=read(stress/'pilot.json')
    assert pilot['status']=='stop_insufficient_damage'and pilot['selected_count']is None
    assert not (stress/'external').exists()
    assert read(stress/'verification.json')['status']=='passed'
    assert read(BASE/'technical-replication-final-source/technical.json')['status']=='passed'
    old=read(ROOT/'audit/metacognition/consolidation-v1.json');assert old['status']=='passed'
    for p,h in old['document_sha256'].items():assert digest(ROOT/p)==h,p
    docs=['RAPPORT_DYNAMIQUE_NATIVE.md','SYNTHESE_METACOGNITION.md','PLAN_RECHERCHE_METACOGNITION.md',
        'FEUILLE_DE_ROUTE.md','ANALYSE_CRITIQUE_SYNTHESE.md','experiments/metacognition/dynamics/README.md',
        'experiments/metacognition/dynamics_stress/README.md']
    destination=BASE/'campaign-summary.json'
    for name in docs:
        p=ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.strip('<>').split('#')[0];resolved=p.parent/target
            assert resolved.exists()or resolved==destination,(name,target)
    evidence=[BASE/p for p in ['lock.json','technical.json','technical-replication-final-source/technical.json',
        'training/report.json','fit.json','external/manifest.json','external/traces/report.json',
        'external/analysis/report.json','causal/report.json','causal/descriptive-diagnostics.json','verification.json',
        'verification-attempt-01.json','resource-observations.json','storage-compression.json']]
    evidence+=[stress/p for p in ['pilot.json','codec-training.json','verification.json']]
    evidence+=[ROOT/'audit/metacognition/consolidation-v1.json']
    sources=[ROOT/p for p in ['scripts/verify_native_dynamics_bounded.py','scripts/summarize_native_dynamics_diagnostics.py',
        'scripts/diagnose_native_codec.py','scripts/close_native_campaign.py','tests/test_native_dynamics.py']]
    record=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),scope='local native study and stopped training-only stress follow-up',
        environment=dict(python=sys.version,torch=version('torch'),numpy=version('numpy'),platform=platform.platform(),
            inference='CPU FP32,4 threads,batch8'),
        units=dict(new_unique_puzzles=2000,causal_subset_puzzles=1500,training_stress_puzzles=128,new_external_stress_puzzles=0),
        primary_prediction=prediction['contrasts']['dynamic-static'],primary_causal=causal['results']['runner_up'],
        stress_status='stop_insufficient_damage_before_new_corpus',tests=dict(command='.venv/bin/python -m unittest discover -s tests',last_observed_passed=29),
        numerical_replay_status='passed',previous_consolidation_status='passed',
        evidence_sha256={str(p.relative_to(ROOT)):digest(p)for p in evidence},
        documentation_sha256={p:digest(ROOT/p)for p in docs},additional_source_sha256={str(p.relative_to(ROOT)):digest(p)for p in sources},
        free_disk_bytes_at_close=shutil.disk_usage(ROOT).free,
        conclusion='No useful improvement from selected dynamic readout; directional compensation under weak code perturbations, functional benefit not established; native metacognition not demonstrated.',
        caveats=['fixed-predictor intervals','oracle current-error cohort','generated distribution','weak functional stress',
            'training-only posthoc dose/decoder diagnostics','no native second-order monitoring identification','system resource pressure'])
    destination.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(status='complete',evidence_files=len(evidence),documents=len(docs),external_puzzles=2000,new_stress_corpus=False)))

if __name__=='__main__':main()
