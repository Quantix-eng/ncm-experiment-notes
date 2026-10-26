"""Download pinned official Sudoku files to a separate directory."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'historical/Experience_NCM/assets.lock.json'


def matches(path,entry):
    h=hashlib.sha256() if entry['sha256'] else hashlib.sha1()
    if not entry['sha256']:h.update(b'blob '+str(entry['bytes']).encode()+b'\0')
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return path.stat().st_size==entry['bytes'] and h.hexdigest()==(entry['sha256'] or entry['git_oid'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path,default=ROOT/'runs/sudoku')
    ap.add_argument('--difficulty',choices=['easy','medium','hard'],default='hard');a=ap.parse_args()
    dest=a.destination.resolve()
    if dest.is_relative_to(ROOT/'historical'):raise ValueError('Choose a destination outside historical/')
    selected=[e for e in json.loads(LOCK.read_text()) if e['path'].startswith(f'sudoku/{a.difficulty}/')]
    for e in selected:
        p=dest/('artifacts' if e['kind']=='model' else 'data')/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            if matches(p,e):print('Verified existing',e['path']);continue
            raise FileExistsError('Existing file differs; choose a fresh destination: '+str(p))
        prefix='datasets/' if e['kind']=='dataset' else ''
        url=f"https://huggingface.co/{prefix}{e['repo']}/resolve/{e['revision']}/{e['path']}"
        part=p.with_suffix(p.suffix+'.part')
        with urllib.request.urlopen(url,timeout=60) as src,part.open('xb') as out:shutil.copyfileobj(src,out)
        if not matches(part,e):raise ValueError('Download failed integrity check: '+str(part))
        part.replace(p);print('Downloaded and verified',e['path'])
    lock=dest/'assets.lock.json'
    if lock.exists() and lock.read_bytes()!=LOCK.read_bytes():raise FileExistsError(str(lock))
    if not lock.exists():shutil.copyfile(LOCK,lock)


if __name__=='__main__':main()
