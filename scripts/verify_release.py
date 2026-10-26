"""Check the modular package and execute its twenty recorded-data programs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path)
    parser.add_argument('--write-report',type=Path)
    args=parser.parse_args()
    layout=json.loads((ROOT/'common/layout.json').read_text())
    original_checks=0
    for row in layout['files']:
        assert digest(ROOT/row['path'])==row['sha256'],row['path']
        if args.original_root and not row.get('new'):
            assert digest(args.original_root/row['runtime_path'])==row.get('original_sha256',row['sha256']),row['runtime_path']
            original_checks+=1
    documents=[ROOT/'README.md',ROOT/'NOTICE.md',ROOT/'common/README.md',*(ROOT/'docs').glob('*.md'),*(ROOT/'experiments').glob('README.md'),*(ROOT/'experiments').glob('*/README.md'),*(ROOT/'experiments').glob('*/inputs/README.md'),*(ROOT/'experiments').glob('*/recorded/README.md')]
    links=0
    for p in documents:
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in raw or raw.startswith(('#','mailto:')):continue
            assert (p.parent/unquote(raw.split('#')[0])).exists(),f'{p.relative_to(ROOT)}: {raw}'
            links+=1
    sources=0
    for top in ['common','experiments','scripts']:
        for p in (ROOT/top).rglob('*.py'):
            if 'runs' in p.relative_to(ROOT).parts:continue
            ast.parse(p.read_text(),filename=str(p));sources+=1
    results=[]
    entries=json.loads((ROOT/'experiments/catalog.json').read_text())
    assert len(entries)==20 and {e['original_experiment']for e in entries}==set(range(1,21))
    mapped={x['path']for x in layout['files']}
    for e in entries:
        folder=(ROOT/e['path']).parent
        for name in ['src','inputs','recorded','run.py','experiment.json']:assert (folder/name).exists()
        config=json.loads((folder/'experiment.json').read_text())
        assert config['stages']
        for step in config['stages'].values():assert step['source'] in mapped
        execution=subprocess.run([sys.executable,str(folder/'run.py'),'check'],cwd=folder,capture_output=True,text=True)
        assert execution.returncode==0,execution.stderr
        result=json.loads(execution.stdout);assert result['passed']
        results.append({'experiment':folder.name,'passed':True,'scope':result['scope']})
    report={'passed':True,'source_files':sources,'mapped_files':len(layout['files']),'curated_links':links,
            'original_files_checked':original_checks,'recorded_data_checks':results,
            'scope':'packaging and recorded-data checks; full inference validation is listed separately in provenance/execution-checks.json'}
    if args.write_report:args.write_report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
