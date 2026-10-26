"""Pin official assets and confine all downloads/copies to this study."""
from base import *
import urllib.request,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed

def get(url):
    with urllib.request.urlopen(url,timeout=60) as f:return f.read()

def download(url,target,size):
    """Validated byte ranges, a single output file, resumable completed ranges."""
    temp=target.with_suffix(target.suffix+'.part');state=temp.with_suffix('.ranges.json')
    done=json.loads(state.read_text()) if state.exists() else {'prefix':temp.stat().st_size if temp.exists() else 0,'ranges':[]}
    chunk=8*1024*1024;ranges=[(start,min(size-1,start+chunk-1)) for start in range(done['prefix'],size,chunk)]
    completed={tuple(v) for v in done['ranges']};pending=[v for v in ranges if v not in completed]
    fd=os.open(temp,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size);write(state,done)
    def fetch(pair):
        start,stop=pair
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{stop}'})
                with urllib.request.urlopen(req,timeout=120) as src:
                    assert src.status==206 and src.headers.get('Content-Range')==f'bytes {start}-{stop}/{size}'
                    at=start
                    while True:
                        data=src.read(1024*1024)
                        if not data:break
                        wrote=os.pwrite(fd,data,at);assert wrote==len(data);at+=wrote
                    assert at==stop+1,(pair,at)
                return pair
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for future in as_completed([pool.submit(fetch,pair) for pair in pending]):
                completed.add(future.result());done['ranges']=sorted(completed);write(state,done)
                log('download_range',asset=target.name,completed_bytes=done['prefix']+sum(b-a+1 for a,b in completed),total_bytes=size)
    finally:os.close(fd)
    temp.replace(target);state.unlink()

def main():
    if not (D/'results/protocol_frozen.json').exists():freeze()
    manifest={}
    for kind,repo in [('model','hazyresearch/ncm-checkpoints'),('dataset','hazyresearch/ncm-datasets')]:
        endpoint='models' if kind=='model' else 'datasets'
        meta=json.loads(get(f'https://huggingface.co/api/{endpoint}/{repo}'));commit=meta['sha']
        tree=json.loads(get(f'https://huggingface.co/api/{endpoint}/{repo}/tree/{commit}?recursive=true&expand=false'))
        write(D/f'sources/{kind}_metadata.json',dict(metadata=meta,tree=tree));by={r['path']:r for r in tree}
        names=['tinystories/tokenizer.pt','tinystories/config.yaml','lm1b/small/config.yaml','lm1b/tokenizer.pt'] if kind=='model' else ['lm1b/valid_bert_30522.bin','lm1b/valid_bert_30522_offsets.npy','tinystories/valid_bpe_8192.bin']
        for name in names:
            target=D/('artifacts' if kind=='model' else 'data')/name;target.parent.mkdir(parents=True,exist_ok=True)
            base='https://huggingface.co/' + ('datasets/' if kind=='dataset' else '')+repo
            url=f'{base}/resolve/{commit}/{name}';entry=by[name];expected=entry.get('lfs',{}).get('oid')
            if not target.exists():
                if name.startswith('tinystories/') and kind=='model':shutil.copyfile(ROOT.parent/'artifacts'/name,target)
                else:
                    checkfree=shutil.disk_usage(D).free-entry['size'];assert checkfree>300_000_000,(name,checkfree)
                    if entry['size']>4*1024*1024:download(url,target,entry['size'])
                    else:target.write_bytes(get(url))
            digest=sha(target);assert target.stat().st_size==entry['size']
            if expected:assert digest==expected,(name,digest,expected)
            manifest[str(target.relative_to(D))]=dict(url=url,commit=commit,sha256=digest,bytes=target.stat().st_size,official_lfs_sha256=expected)
            write(D/'results/assets_manifest.json',manifest);log('asset_ready',name=name,bytes=target.stat().st_size)
    write(D/'sources/previous_manifests.json',dict(initial=sha(ROOT/'results/final_manifest.json'),mechanistic=sha(ROOT/'directions_mecanistes/results/final_manifest.json')))

if __name__=='__main__':main()
