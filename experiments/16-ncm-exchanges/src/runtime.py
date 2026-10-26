"""Explicit FP32 adapter. Prefix-only KV references are reset at every level."""
from support import np, torch, Q
import torch.nn.functional as F


class Engine:
    def __init__(self,model,prefix):
        self.model = model
        self.net = model.net
        self.vq = model.vqvae.vq
        self.prefix = torch.as_tensor(prefix,dtype=torch.long)
        self.b,self.p = self.prefix.shape
        self.offsets = np.r_[0,np.cumsum(self.net.levels)]

    def __enter__(self):
        n = self.net
        n._set_kv_caching(True)
        self.positions = n.transformer.wle(n.lvl_1L)
        x = n.transformer.wte(self.model.vqvae.encoder.wte(self.prefix))
        mask = torch.zeros(1,1,self.p,self.p)
        rope = n.rope_cos_gen_0,n.rope_sin_gen_0
        for block in n.transformer.blocks:
            x = block(x,attn_mask=mask,rope_cos_sin=rope)
        self.dtype = x.dtype
        self.kv = [(block.attn.cached_k,block.attn.cached_v) for block in n.transformer.blocks]
        return self

    def __exit__(self,*exc):
        self.net._set_kv_caching(False)
        assert all(not block.attn.caching and block.attn.cached_k is None for block in self.net.transformer.blocks)

    def contribution(self,k,c):
        emb = self.vq.embeddings[k](c)
        if k < len(self.net.levels)-1:
            emb = self.vq._up_to_full(emb,k,89)
        return self.vq._apply_quant_resi(emb,k)

    def step(self,k,z):
        n = self.net
        length = int(n.levels[k])
        query = n.build_l0_query_input(self.b,self.dtype) if k == 0 else F.interpolate(z.transpose(1,2),size=(length,),mode='area').transpose(1,2)
        x = n.transformer.wte(query)+self.positions[:,int(self.offsets[k]):int(self.offsets[k+1])]
        mask = torch.zeros(1,1,length,self.p+length)
        rope = getattr(n,f'rope_cos_gen_{k}'),getattr(n,f'rope_sin_gen_{k}')
        for block,(pk,pv) in zip(n.transformer.blocks,self.kv):
            block.attn.cached_k,block.attn.cached_v = pk,pv
            x = block(x,attn_mask=mask,rope_cos_sin=rope)
        codes = n.lm_heads[k](n.transformer.ln_f(x)).argmax(-1)
        residual = self.contribution(k,codes)
        return z+residual,codes,residual

    def native(self):
        z = torch.zeros(self.b,89,256)
        codes,snapshots,residuals,logits = [],[],[],[]
        for k in range(12):
            z,c,r = self.step(k,z)
            codes.append(c)
            snapshots.append(z.clone())
            residuals.append(r.clone())
            logits.append(self.model.vqvae.decoder(z))
        return dict(codes=torch.cat(codes,1),z=torch.stack(snapshots,1),
                    residuals=torch.stack(residuals,1),logits=torch.stack(logits,1))

    def continuation(self,start):
        z = start.clone()
        codes,residuals = [],[]
        for k in range(Q['cut_level_one_based'],12):
            z,c,r = self.step(k,z)
            codes.append(c)
            residuals.append(r.clone())
        return dict(z=z,codes=torch.cat(codes,1),residuals=torch.stack(residuals,1),
                    logits=self.model.vqvae.decoder(z))


def append(start,residuals):
    z = start.clone()
    for k in range(residuals.shape[1]):
        z = z+residuals[:,k]
    return z


def perturbations(z,family,mask):
    rng = np.random.default_rng(Q['seed']+1009*family)
    cells = rng.choice(np.flatnonzero(mask),2,replace=False)
    units = rng.normal(size=(2,256))
    units /= np.linalg.norm(units,axis=1,keepdims=True)
    desired = Q['perturbation_local_rms_multiplier']*float(np.linalg.norm(z.numpy().astype(np.float64)))/np.sqrt(89)
    delta = torch.zeros(4,89,256)
    for j in range(2):
        delta[2*j,int(cells[j])+int(cells[j])//9] = torch.tensor(units[j]*desired,dtype=torch.float32)
        delta[2*j+1] = -delta[2*j]
    actual = delta.double().flatten(1).norm(dim=1).numpy()
    assert np.allclose(actual,desired,rtol=2e-6,atol=0)
    return delta,cells,desired
