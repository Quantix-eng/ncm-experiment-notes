"""Frozen official text tokenizers: occupancy, geometry and controlled permutations."""
from base import *
sys.path.insert(0,str(D/'sources'))
from models.vqvae import VQVAE
from omegaconf import OmegaConf
from prepare import download,get
import pickle,io,gc
import torch.nn.functional as F

class Plain(pickle.Unpickler):
    def find_class(self,module,name):raise pickle.UnpicklingError('Only plain data allowed')

def load_model(task):
    path=D/'artifacts'/task/('small/config.yaml' if task=='lm1b' else 'config.yaml')
    config=OmegaConf.load(path)
    with torch.device('meta'):model=VQVAE(**config.tokenizer.vqvae).eval().requires_grad_(False)
    sd=torch.load(D/f'artifacts/{task}/tokenizer.pt',map_location='cpu',weights_only=True,mmap=True)['model']
    model.load_state_dict(sd,strict=True,assign=True);return model,config

def statehash(model):
    h=hashlib.sha256()
    for key,v in sorted(model.state_dict().items()):h.update(key.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def inputs(task):
    pad=0 if task=='lm1b' else 1
    if task=='lm1b':
        stream=np.memmap(D/'data/lm1b/valid_bert_30522.bin',dtype='<u2',mode='r');offsets=np.load(D/'data/lm1b/valid_bert_30522_offsets.npy')
        assert offsets[0]==0 and offsets[-1]==len(stream) and np.all(np.diff(offsets)>0)
        source=[np.asarray(stream[int(a):int(b)],dtype=np.int64) for a,b in zip(offsets[:-1],offsets[1:])]
    else:
        path=D/'data/tinystories/valid_bpe_8192.pickle'
        if not path.exists():
            meta=json.loads((D/'sources/dataset_metadata.json').read_text());entry=next(v for v in meta['tree'] if v['path']=='tinystories/valid_bpe_8192.pickle');commit=meta['metadata']['sha']
            download(f'https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/{commit}/tinystories/valid_bpe_8192.pickle',path,entry['size'])
            expected=entry.get('lfs',{}).get('oid')
            if expected:assert sha(path)==expected
        source=Plain(io.BytesIO(path.read_bytes())).load();assert isinstance(source,list) and all(isinstance(v,list) and all(type(i)==int for i in v) for v in source)
    rows=np.full((len(source),128),pad,dtype=np.int64)
    for i,r in enumerate(source):rows[i,:min(128,len(r))]=r[:128]
    vocab=30528 if task=='lm1b' else 8192;assert rows.min()>=0 and rows.max()<vocab
    lengths=(rows!=pad).sum(1);eligible=np.flatnonzero(lengths>0);long=np.flatnonzero(lengths>=96);rng=np.random.default_rng(Q['seed'])
    random=rng.choice(eligible,min(Q['text_random_n'],len(eligible)),replace=False);chosenlong=rng.choice(long,min(Q['text_long_n'],len(long)),replace=False)
    assert len(chosenlong)>0
    selected=np.r_[random,chosenlong];x=rows[selected];mask=x!=pad
    out=dict(task=task,n_source_rows=len(source),n_eligible=len(eligible),n_long_eligible=len(long),pad_id=pad,vocab_size=vocab,
      source_length_quantiles=np.quantile(lengths,[0,.1,.5,.9,1]),random_indices=random,long_indices=chosenlong,stratum_overlap=len(set(random)&set(chosenlong)),
      random_pad_fraction=float((x[:len(random)]==pad).mean()),long_pad_fraction=float((x[len(random):]==pad).mean()),
      n_duplicate_random_windows=len(random)-len(np.unique(x[:len(random)],axis=0)),
      construction='Official row boundaries; first 128 tokens, right pad if shorter. Nonpadding includes special tokens other than PAD.')
    np.savez_compressed(D/f'data/{task}_selected.npz',tokens=x,mask=mask,source_indices=selected,stratum=np.r_[np.zeros(len(random),int),np.ones(len(chosenlong),int)])
    write(D/f'results/{task}_data.json',out);return x,mask,out

def entropy(p):
    p=np.asarray(p,dtype=float);return float(-(p[p>0]*np.log(p[p>0])).sum())

def occupancy(codes,vocab):
    counts=np.stack([np.bincount(codes[:,j],minlength=vocab) for j in range(codes.shape[1])]);p=counts.sum(0)/counts.sum()
    return dict(counts=counts.sum(0),used=int((p>0).sum()),marginal_entropy=entropy(p),perplexity=float(np.exp(entropy(p))),
      positional_entropy=float(np.mean([entropy(v/v.sum()) for v in counts])),modal_fraction=float(p.max()),
      modal_position_fraction=float(counts.max(1).sum()/counts.sum()),unique_code_sequences=len(np.unique(codes,axis=0)),
      empirical_constant_level=bool(np.all(codes==codes[0:1])),
      coverage=[dict(n=min(n,len(codes)),used=len(np.unique(codes[:min(n,len(codes))]))) for n in [16,64,128,256]])

class Moments:
    def __init__(self):self.n=0;self.sum=None;self.sq=None
    def add(self,x):
        x=x.double().numpy();s=x.sum(0);sq=(x*x).sum(0)
        if self.sum is None:self.sum=np.zeros_like(s);self.sq=np.zeros_like(s)
        self.n+=len(x);self.sum+=s;self.sq+=sq
    def result(self):return dict(inter_sequence_variance=float(np.maximum(self.sq/self.n-(self.sum/self.n)**2,0).mean()),rms=float(np.sqrt((self.sq/self.n).mean())))

@torch.inference_mode()
def audit(task,model,x,mask,data):
    vq=model.vq;levels=list(vq.levels);dim=model.embed_dim;beforehash=statehash(model);saved={};report={};parities=[]
    for stratum,ix in [('random',np.arange(len(data['random_indices']))),('long',np.arange(len(data['random_indices']),len(x)))]:
        moments=[dict(residual=Moments(),input=Moments(),raw=Moments()) for _ in levels];cc=[[] for _ in levels];quant=[0. for _ in levels];nb=[0 for _ in levels];mse_before=np.zeros(len(levels));mse_after=np.zeros(len(levels));ids=[];loss=0.;correct=0;total=0;explicit=[]
        batch_size=1 if task=='lm1b' else Q['batch_size']
        for lo in range(0,len(ix),batch_size):
            batch=ix[lo:lo+batch_size];xx=torch.tensor(x[batch]);z=model._encode_pre_quant(xx);rest=z.clone();zhat=torch.zeros_like(z);raw=model.encoder.wte(xx)
            if model.encoder.wpe is not None:raw=raw+model.encoder.wpe(torch.arange(128))[None]
            batchcodes=[]
            for k,l in enumerate(levels):
                u=vq._down_for_quant(rest,k,l) if k<len(levels)-1 else rest
                flat=u.reshape(-1,dim);dist=vq._distances(flat,vq.embeddings[k].weight);code=dist.argmin(-1).reshape(len(xx),l);cc[k].append(code.numpy());batchcodes.append(code)
                moments[k]['residual'].add(rest);moments[k]['input'].add(u);moments[k]['raw'].add(F.interpolate(raw.transpose(1,2),size=int(l),mode='area').transpose(1,2))
                quant[k]+=float(dist.min(-1).values.sum()/dim);nb[k]+=len(flat);mse_before[k]+=float(rest.double().square().sum())
                c=vq.embeddings[k](code)
                if k<len(levels)-1:c=vq._up_to_full(c,k,128)
                c=vq._apply_quant_resi(c,k);rest=rest-c;zhat=zhat+c;mse_after[k]+=float(rest.double().square().sum())
                if stratum=='random' and lo==0:
                    # Direct differences, not the same expanded-distance formula.
                    f=flat[:4].double();book=vq.embeddings[k].weight.double();direct=torch.cat([((f[:,None]-w[None])**2).sum(-1) for w in book.split(512)],dim=1)
                    explicit.append(dict(level=k+1,samples=len(f),assignment_disagreements=int((direct.argmin(-1)!=dist[:4].argmin(-1)).sum())))
            allcodes=torch.cat(batchcodes,1)
            if stratum=='random' and lo<8:
                native=model.encode(xx);parities.append(bool(torch.equal(native,allcodes)));assert torch.equal(native,allcodes)
            logits=model.decoder(zhat);pred=logits.argmax(-1);ids.append(pred.numpy());mm=torch.tensor(mask[batch]);correct+=int(((pred==xx)&mm).sum());total+=int(mm.sum());loss+=float(F.cross_entropy(logits[mm],xx[mm],reduction='sum'))
            if lo%32==0:log('text_batch',task=task,stratum=stratum,done=lo+len(batch),total=len(ix))
        lev=[]
        for k,l in enumerate(levels):
            codes=np.concatenate(cc[k]);saved[stratum+f'_codes_{k+1}']=codes
            mo={name:v.result() for name,v in moments[k].items()};r=mo['residual']['inter_sequence_variance'];u=mo['input']['inter_sequence_variance']
            lev.append(dict(level=k+1,length=int(l),vocab=int(vq.vqvae_vocab_sizes[k]),occupancy=occupancy(codes,int(vq.vqvae_vocab_sizes[k])),geometry=mo,
               pooling_variance_ratio=u/max(r,1e-30),quantization_mse=quant[k]/nb[k],residual_mse_before=mse_before[k]/(len(ix)*128*dim),residual_mse_after=mse_after[k]/(len(ix)*128*dim)))
        saved[stratum+'_decoded']=np.concatenate(ids);report[stratum]=dict(n=len(ix),levels=lev,reconstruction_nonpad_accuracy=correct/total,reconstruction_nonpad_crossentropy=loss/total,n_nonpad_tokens=total)
        if stratum=='random':report['distance_checks']=explicit
    c1=torch.cat([model.encode(torch.tensor(row[None])) for row in x[:8]]).numpy();c4=torch.cat([model.encode(torch.tensor(x[i:i+4])) for i in range(0,8,4)]).numpy()
    report['cross_batch_code_disagreements']=int((c1!=c4).sum());report['batch_size']=1 if task=='lm1b' else Q['batch_size']
    report['native_encode_parity']=all(parities);report['native_encode_parity_batches']=len(parities);report['state_before']=beforehash;report['state_after']=statehash(model);assert report['state_after']==beforehash
    np.savez_compressed(D/f'results/{task}_audit.npz',**saved);write(D/f'results/{task}_audit.json',report);return report

@torch.inference_mode()
def controls(task,model,x,data):
    n=Q['text_controls_n'];start=len(data['random_indices']);original=x[start:start+n].copy();rng=np.random.default_rng(Q['seed']);within=original.copy();global_=original.copy()
    for b in range(len(original)):
        for j in range(0,128,4):within[b,j:j+4]=rng.permutation(original[b,j:j+4])
        global_[b]=rng.permutation(original[b])
    variants=dict(original=original,within_four=within,global_permutation=global_);outputs={};arrays={}
    for name,rows in variants.items():
        codes=[];pool=[];rawpool=[];first=[]
        for lo in range(0,len(rows),4):
            xx=torch.tensor(rows[lo:lo+4]);z=model._encode_pre_quant(xx);raw=model.encoder.wte(xx)
            if model.encoder.wpe is not None:raw+=model.encoder.wpe(torch.arange(128))[None]
            codes.append(model.encode(xx).numpy());pool.append(F.interpolate(z.transpose(1,2),size=32,mode='area').transpose(1,2).numpy());rawpool.append(F.interpolate(raw.transpose(1,2),size=32,mode='area').transpose(1,2).numpy())
            first.append(model.vq._down_for_quant(z,0,1).numpy())
        arrays[name+'_tokens']=rows;arrays[name+'_codes']=np.concatenate(codes);arrays[name+'_norm_pool32']=np.concatenate(pool);arrays[name+'_raw_pool32']=np.concatenate(rawpool);arrays[name+'_first_input']=np.concatenate(first)
    offset=np.r_[0,np.cumsum(model.vq.levels)]
    for name in variants:
        c=arrays[name+'_codes'];native=arrays['original_codes'];out={}
        for key in ['raw_pool32','norm_pool32','first_input']:
            base=arrays['original_'+key].astype(float);new=arrays[name+'_'+key].astype(float);out[key+'_relative_l2']=float(np.linalg.norm(new-base)/max(np.linalg.norm(base),1e-30));out[key+'_max_abs']=float(np.max(np.abs(new-base)))
        out['per_level_code_disagreement']=[float((c[:,int(a):int(b)]!=native[:,int(a):int(b)]).mean()) for a,b in zip(offset[:-1],offset[1:])]
        outputs[name]=out
    np.savez_compressed(D/f'results/{task}_controls.npz',**arrays);write(D/f'results/{task}_controls.json',outputs)

def main(tasks=('tinystories','lm1b')):
    for task in tasks:
        deadline=time.monotonic()+3600
        required=[D/f'artifacts/{task}/tokenizer.pt']+([D/'data/tinystories/valid_bpe_8192.pickle'] if task=='tinystories' else [D/'data/lm1b/valid_bert_30522.bin',D/'data/lm1b/valid_bert_30522_offsets.npy'])
        while not all(path.exists() for path in required):
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint transfer not completed: '+task)
            time.sleep(5)
        x,mask,data=inputs(task);model,config=load_model(task);report=audit(task,model,x,mask,data);controls(task,model,x,data)
        write(D/f'results/{task}_architecture.json',dict(enc_positional=model.encoder.wpe is not None,encoder_blocks=len(model.encoder.blocks),encoder_conv_blocks=len(model.encoder.conv_blocks),
            learned_l1_sampling=bool(model.vq.learned_l1_sampling),learned_l1_cascade=bool(model.vq.learned_l1_cascade),learned_all_sampling=bool(model.vq.learned_all_sampling),levels=list(model.vq.levels),embed_dim=model.embed_dim))
        log('text_task_complete',task=task,dead_random=[r['level'] for r in report['random']['levels'] if r['occupancy']['empirical_constant_level']]);del model;gc.collect()
    log('text_audit_complete')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['tinystories','lm1b']);args=parser.parse_args();main(tasks=(args.task,) if args.task else ('tinystories','lm1b'))
