from common import *


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(48);self.qkv=nn.Linear(48,144);self.proj=nn.Linear(48,48)
        self.ln2=nn.LayerNorm(48);self.fc1=nn.Linear(48,96);self.fc2=nn.Linear(96,48)

    def attention(self,h,mask):
        b,l,d=h.shape
        q,k,v=self.qkv(self.ln1(h)).reshape(b,l,3,4,12).permute(2,0,3,1,4).unbind(0)
        p=((q@k.transpose(-1,-2))/(12**.5)).masked_fill(~mask,float('-inf')).softmax(-1)
        return self.proj((p@v).transpose(1,2).reshape(b,l,d))

    def forward(self,h,mask):
        a=self.attention(h,mask);h=h+a
        m=self.fc2(F.gelu(self.fc1(self.ln2(h))));h=h+m
        return h,a,m


class Net(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=architecture
        self.embedding=nn.Embedding(15,48);self.positions=nn.Embedding(11,48)
        self.blocks=nn.ModuleList([Block() for _ in range(4)])
        self.norm=nn.LayerNorm(48);self.head=nn.Linear(48,8)
        mask=torch.ones(11,11,dtype=torch.bool).tril()
        if architecture=='parallel':mask[7:,7:]=True
        self.register_buffer('mask',mask[None,None])
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,std=.02)
                if isinstance(module,nn.Linear):nn.init.zeros_(module.bias)

    def read(self,h):return self.head(self.norm(h))[:,7:]

    def forward(self,tokens,trace=False):
        h=self.embedding(tokens)+self.positions(torch.arange(11))
        states=[h];residuals=[]
        for block in self.blocks:
            h,a,m=block(h,self.mask)
            if trace:states.append(h);residuals.extend((a,m))
        logits=self.read(h)
        if trace:return logits,torch.stack(states),torch.stack(residuals)
        return logits

    def generate(self,prefix):
        queries=fixed_queries(len(prefix));answer=[]
        if self.architecture=='parallel':return self(torch.cat((prefix,queries),dim=1)).argmax(-1)
        for j in range(4):
            token=self(torch.cat((prefix,queries),dim=1))[:,j].argmax(-1)
            answer.append(token)
            if j<3:queries[:,j+1]=token
        return torch.stack(answer,dim=1)


def load(architecture,seed):
    net=Net(architecture)
    net.load_state_dict(torch.load(D/f'checkpoints/{architecture}-{seed}.pt',weights_only=True))
    return net.eval()


def parameter_hash(net):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
