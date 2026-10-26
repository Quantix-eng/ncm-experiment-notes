"""Head-output resampling, exact native replay, explicit tail and head blockers."""
from support import *

class Engine:
    def __init__(self, net):
        self.net=net; self.saved={}; self.handles=[]; self.record=False
        for l,b in enumerate(net.transformer.h):
            def pre(m,args,l=l):
                if self.record:self.saved[('input',l)]=args[0].detach().clone()
            def proj(m,args,l=l):
                if self.record:self.saved[('heads',l)]=args[0].detach().clone()
            def attn(m,args,out,l=l):
                if self.record:self.saved[('attn',l)]=out[0].detach().clone()
            def mlp(m,args,out,l=l):
                if self.record:self.saved[('mlp',l)]=out.detach().clone()
            def post(m,args,out,l=l):
                if self.record:self.saved[('out',l)]=out[0].detach().clone()
            self.handles += [b.register_forward_pre_hook(pre),b.attn.c_proj.register_forward_pre_hook(proj),
                b.attn.register_forward_hook(attn),b.mlp.register_forward_hook(mlp),b.register_forward_hook(post)]

    def decode(self,c,z):
        full=c['out',11].clone(); full[0,-1]=z
        return self.net.lm_head(self.net.transformer.ln_f(full))[0,-1].detach().clone()

    def native(self,tokens):
        self.record=True; self.saved={}
        logits=self.net(torch.tensor([tokens]),use_cache=False).logits[0,-1].detach().clone()
        self.record=False
        c={k:v.clone() for k,v in self.saved.items()}; c['logits']=logits
        assert torch.equal(self.decode(c,c['out',11][0,-1]),logits)
        return c

    def boundary(self,c,l,h,donor=None):
        b=self.net.transformer.h[l]
        if donor is None:
            return (c['attn',l]+c['input',l])[0,-1].clone()
        raw=c['heads',l].clone(); raw[0,-1,h*64:(h+1)*64]=donor
        attention=b.attn.resid_dropout(b.attn.c_proj(raw))
        return (attention+c['input',l])[0,-1].clone()

    @staticmethod
    def replay(z,tail):
        for i,t in enumerate(tail):
            # First is same-layer MLP; remaining attention then MLP, official add order.
            z=t+z if i%2==1 else z+t
        return z

    def tail_native(self,c,l):
        return torch.stack([c['mlp',l][0,-1]]+[c[k,j][0,-1]
            for j in range(l+1,12) for k in ('attn','mlp')])

    def continuation(self,c,l,z,blocked=()):
        bylayer={j:[h for a,h in blocked if a==j] for j in range(l+1,12)}
        full=c['input',l]+c['attn',l]; full=full.clone();full[0,-1]=z
        b=self.net.transformer.h[l]; m=b.mlp(b.ln_2(full)); result=(full+m)[0,-1].clone()
        tail=[m[0,-1].clone()]
        for j in range(l+1,12):
            b=self.net.transformer.h[j]; full=c['input',j].clone();full[0,-1]=result
            handle=None
            if bylayer[j]:
                def replace(module,args,j=j):
                    raw=args[0].clone()
                    for h in bylayer[j]:raw[0,-1,h*64:(h+1)*64]=c['heads',j][0,-1,h*64:(h+1)*64]
                    return (raw,)
                handle=b.attn.c_proj.register_forward_pre_hook(replace)
            try:a=b.attn(b.ln_1(full),use_cache=False)[0]
            finally:
                if handle is not None:handle.remove()
            mid=a+full; m=b.mlp(b.ln_2(mid)); out=mid+m
            assert torch.equal(out[0,:-1],c['out',j][0,:-1]),'past changed'
            tail += [a[0,-1].clone(),m[0,-1].clone()];result=out[0,-1].clone()
        tail=torch.stack(tail)
        assert torch.equal(self.replay(z,tail),result)
        return tail,result,self.decode(c,result)

    def projected_change(self,c,dc,l,h):
        raw=dc['heads',l][0,-1,h*64:(h+1)*64].double()-c['heads',l][0,-1,h*64:(h+1)*64].double()
        return raw@self.net.transformer.h[l].attn.c_proj.weight[h*64:(h+1)*64].double()

    def direct_effect(self,c,delta,answer):
        z=c['out',11][0,-1].double();delta=delta.double()
        ln=self.net.transformer.ln_f
        scale=torch.sqrt(((z-z.mean())**2).mean()+ln.eps)
        if not hasattr(self,'unembed_mean'):
            self.unembed_mean=self.net.lm_head.weight.double().mean(0)
        direction=(self.net.lm_head.weight[answer].double()-self.unembed_mean)
        return float((((delta-delta.mean())/scale)*ln.weight.double())@direction)

    def close(self):
        for h in self.handles:h.remove()
