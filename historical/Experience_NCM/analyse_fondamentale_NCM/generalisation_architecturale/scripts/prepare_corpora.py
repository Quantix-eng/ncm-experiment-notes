"""Prepare validation rows independently of the larger checkpoint transfer."""
from base import *
from prepare import get,download

def main():
    repo='hazyresearch/ncm-datasets';meta=json.loads(get(f'https://huggingface.co/api/datasets/{repo}'));commit=meta['sha']
    tree=json.loads(get(f'https://huggingface.co/api/datasets/{repo}/tree/{commit}?recursive=true&expand=false'));by={v['path']:v for v in tree}
    write(D/'sources/dataset_metadata.json',dict(metadata=meta,tree=tree));manifest={}
    for name in ['tinystories/valid_bpe_8192.pickle','tinystories/valid_bpe_8192.bin','lm1b/valid_bert_30522.bin','lm1b/valid_bert_30522_offsets.npy']:
        target=D/'data'/name;target.parent.mkdir(exist_ok=True,parents=True);entry=by[name];url=f'https://huggingface.co/datasets/{repo}/resolve/{commit}/{name}'
        if not target.exists():
            if entry['size']>4*1024*1024:download(url,target,entry['size'])
            else:target.write_bytes(get(url))
        expected=entry.get('lfs',{}).get('oid');digest=sha(target);assert target.stat().st_size==entry['size']
        if expected:assert digest==expected
        manifest[str(target.relative_to(D))]=dict(url=url,commit=commit,sha256=digest,official_lfs_sha256=expected,bytes=entry['size']);write(D/'results/text_corpus_manifest.json',manifest)
        log('corpus_ready',name=name)

if __name__=='__main__':main()
