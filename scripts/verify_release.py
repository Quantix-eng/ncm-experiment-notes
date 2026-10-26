"""Check copied-file hashes, curated links and Python syntax without importing runners."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-originals', action='store_true', help='also compare originals, if available on this machine')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'provenance/historical-manifest.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path']
        assert p.is_file() and digest(p)==row['sha256'], row['path']
        if args.check_originals:
            original=Path(manifest['source_root_at_export'])/row['source']
            assert digest(original)==row['sha256'], 'Original changed: '+row['source']
    docs=[ROOT/'README.md',ROOT/'NOTICE.md']+list((ROOT/'docs').glob('*.md'))+list((ROOT/'experiments').rglob('*.md'))
    checked_links=0
    for p in docs:
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in raw or raw.startswith(('#','mailto:')): continue
            target=unquote(raw.split('#')[0])
            assert (p.parent/target).exists(), f'{p.relative_to(ROOT)}: {raw}'
            checked_links+=1
    for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    catalog=json.loads((ROOT/'experiments/catalog.json').read_text())
    assert len(catalog)==20 and {e['original_experiment'] for e in catalog}==set(range(1,21))
    print(json.dumps({'passed':True,'historical_files':len(manifest['files']),'curated_links':checked_links,
                      'experiments':len(catalog),'originals_checked':args.check_originals,
                      'scope':'packaging integrity and syntax, not scientific replication'},indent=2))


if __name__=='__main__':main()
