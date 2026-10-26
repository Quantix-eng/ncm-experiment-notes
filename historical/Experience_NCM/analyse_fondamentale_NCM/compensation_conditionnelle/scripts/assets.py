"""Fetch only the pinned LM1B small generator; all other weights stay read-only."""
from base import *
from transfer import download
import socket
# This machine's IPv6 route stalls in SYN_SENT; restrict this process to IPv4.
_addresses=socket.getaddrinfo
socket.getaddrinfo=lambda host,port,family=0,type=0,proto=0,flags=0:_addresses(host,port,socket.AF_INET,type,proto,flags)
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    m=json.loads((D/'sources/model_metadata.json').read_text());entry=next(x for x in m['tree'] if x['path']=='lm1b/small/generator.pt');commit=m['metadata']['sha'];url=f'https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/{commit}/lm1b/small/generator.pt';target=D/'artifacts/lm1b-small/generator.pt'
    if not target.exists():download(url,target,entry['size'])
    h=filehash(target);assert h==entry['lfs']['oid'];write(D/'results/lm1b_asset.json',dict(url=url,commit=commit,bytes=entry['size'],sha256=h,tokenizer_path='../generalisation_architecturale/artifacts/lm1b/tokenizer.pt'));log('lm1b_generator_ready',bytes=entry['size'])
if __name__=='__main__':main()
