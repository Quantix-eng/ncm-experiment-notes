"""Explicit cached NCM runner with exact checkpoints after each code contribution.

Arithmetic and masks follow the copied generate_cached implementation. The runner
is separately compared to the native path before mechanistic interpretation.
"""
from base import *
import torch.nn.functional as F

class Engine:
    def __init__(self,model,prefix):
        self.model=model;self.net=model.net;self.vq=model.vqvae.vq
        self.prefix=torch.as_tensor(prefix,dtype=torch.long);self.b,self.p=self.prefix.shape
        self.positions=self.net.transformer.wle(self.net.lvl_1L)
        self.offsets=np.r_[0,np.cumsum(list(self.net.levels))]
        self.kv=None;self.calls=0
    def __enter__(self):
        n=self.net;n._set_kv_caching(True)
        x=n.transformer.wte(self.model.vqvae.encoder.wte(self.prefix));mask=torch.zeros(1,1,self.p,self.p)
        rope=(n.rope_cos_gen_0,n.rope_sin_gen_0)
        for block in n.transformer.blocks:x=block(x,attn_mask=mask,rope_cos_sin=rope)
        self.dtype=x.dtype;self.kv=[(block.attn.cached_k,block.attn.cached_v) for block in n.transformer.blocks]
        return self
    def __exit__(self,*exc):
        self.net._set_kv_caching(False)
        assert not any(b.attn.caching or b.attn.cached_k is not None for b in self.net.transformer.blocks)
    def contribution(self,k,codes):
        e=self.vq.embeddings[k](codes)
        if k<11:e=self.vq._up_to_full(e,k,89)
        return self.vq._apply_quant_resi(e,k)
    def step(self,k,z,fixed_code=None,omit_contribution=False):
        n=self.net;l=int(n.levels[k]);h=None;logits=None
        if fixed_code is None:
            x=n.build_l0_query_input(self.b,self.dtype) if k==0 else F.interpolate(z.transpose(1,2),size=(l,),mode='area').transpose(1,2)
            x=n.transformer.wte(x)+self.positions[:,int(self.offsets[k]):int(self.offsets[k+1])]
            mask=torch.zeros(1,1,l,self.p+l);rope=(getattr(n,f'rope_cos_gen_{k}'),getattr(n,f'rope_sin_gen_{k}'))
            for block,(pk,pv) in zip(n.transformer.blocks,self.kv):
                block.attn.cached_k,block.attn.cached_v=pk,pv;x=block(x,attn_mask=mask,rope_cos_sin=rope)
            h=x;logits=n.lm_heads[k](n.transformer.ln_f(x));code=logits.argmax(-1);self.calls+=1
        else:code=torch.full((self.b,l),int(fixed_code),dtype=torch.long)
        if not omit_contribution:z=z+self.contribution(k,code)
        return z,code,h,logits
    def run(self,fixed5=False,zero5=False,capture=False):
        z=torch.zeros(self.b,89,256);codes=[];out={}
        for k in range(12):
            z,c,h,l=self.step(k,z,fixed_code=11 if fixed5 and k==4 else None,omit_contribution=zero5 and k==4);codes.append(c)
            if capture and k==7:
                out['z8']=z.detach().clone();out['h8']=torch.cat([h.mean(1),h.std(1,correction=0)],-1)
                dl=self.model.vqvae.decoder(z);out['prob8']=dl.softmax(-1);out['decoded8']=dl.argmax(-1)
            if capture and k==8:
                out['z9']=z.detach().clone();dl=self.model.vqvae.decoder(z);out['prob9']=dl.softmax(-1);out['decoded9']=dl.argmax(-1)
        out.update(codes=torch.cat(codes,1),zfinal=z,final_logits=self.model.vqvae.decoder(z),calls=self.calls)
        out['final_ids']=out['final_logits'].argmax(-1)
        return out
    def first8(self):
        z=torch.zeros(self.b,89,256);codes=[]
        for k in range(8):z,c,_,_=self.step(k,z);codes.append(c)
        return z,codes
    def branch(self,z8,codes8,delta=None):
        z=z8.clone() if delta is None else z8+delta;codes=list(codes8);out={}
        for k in range(8,12):
            z,c,_,_=self.step(k,z);codes.append(c)
            if k==8:
                out['z9']=z.clone();out['codes9']=c;dl=self.model.vqvae.decoder(z)
                out['prob9']=dl.softmax(-1);out['decoded9']=dl.argmax(-1)
        out['final_ids']=self.model.vqvae.decoder(z).argmax(-1);out['codes']=torch.cat(codes,1)
        return out

@torch.inference_mode()
def parity():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');p,_=boards('hard');res=[]
    for ix in [np.arange(16),np.array(SPLIT['test'])[:16]]:
        native=model.predict(torch.tensor(p[ix]));
        with Engine(model,p[ix]) as engine:r=engine.run(capture=True)
        snap,=model.vqvae.vq.idx_to_zhats(native.codes,return_zlevels=False)
        check=dict(codes=bool(torch.equal(native.codes,r['codes'])),logits=bool(torch.equal(native.logits,r['final_logits'])),
          z8=bool(torch.equal(snap[7],r['z8'])),z9=bool(torch.equal(snap[8],r['z9'])),calls=r['calls'])
        assert all(check[k] for k in ['codes','logits','z8','z9']),check
        res.append(check)
    write(D/'results/runtime_parity.json',dict(status='passed',samples=res));log('runtime_parity',samples=res)

if __name__=='__main__':parity()
