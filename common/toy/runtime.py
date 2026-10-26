from model import *


def decode(net,h):return net.read(h)[0,0]


def frozen(h,tail):
    out=h.clone()
    for r in tail:out=out+r
    return out


def continuation(net,h,cut):
    out=h.clone();tail=[]
    for block in net.blocks[cut:]:
        out,a,m=block(out,net.mask);tail.extend((a,m))
    tail=torch.stack(tail)
    assert torch.equal(frozen(h,tail),out)
    return out,tail


def native(net,row):
    states=[];tails=[];logits=[];answers=[]
    for prefix in row['inputs']:
        x=torch.tensor([prefix]);tokens=torch.cat((x,fixed_queries(1)),dim=1)
        final,h,r=net(tokens,trace=True)
        readings=torch.stack([decode(net,v) for v in h])
        assert torch.equal(readings[-1],final[0,0])
        for cut in Q['cuts']:
            z,t=continuation(net,h[cut],cut)
            assert torch.equal(z,h[-1]) and torch.equal(t,r[2*cut:])
        states.append(h);tails.append(r);logits.append(readings);answers.append(net.generate(x)[0])
    return dict(states=torch.stack(states),tails=torch.stack(tails),logits=torch.stack(logits),
                answers=torch.stack(answers))


def directions(h,source_index,anchor,partition):
    seed=Q['direction_seed']+(0 if partition=='development' else 100000)+100*source_index+anchor
    rng=np.random.default_rng(seed);values=[]
    for support in range(2):
        v=np.zeros(h.shape,dtype=np.float64)
        v[:,7:,support*24:(support+1)*24]=rng.normal(size=(1,4,24))
        v*=float(h[:,7:].double().norm())/np.linalg.norm(v)
        delta=torch.tensor(v,dtype=torch.float32)
        values.extend((delta,-delta))
    return values


def crossed(net,n,source_index,anchor,cut,rho,partition):
    h=n['states'][anchor,cut];t0=n['tails'][anchor,2*cut:]
    ds=[rho*d for d in directions(h,source_index,anchor,partition)]
    adapted=[continuation(net,h+d,cut)[1] for d in ds]
    outputs=[];finals=[]
    for j,d in enumerate(ds):
        starts=(h,h+d,h+d,h,h+d)
        tails=(t0,t0,adapted[j],adapted[j],adapted[j^2])
        values=[frozen(z,t) for z,t in zip(starts,tails)]
        finals.append(torch.stack(values));outputs.append(torch.stack([decode(net,z) for z in values]))
    return dict(logits=torch.stack(outputs),finals=torch.stack(finals),
                tails=torch.stack(adapted),deltas=torch.stack(ds))


def initial(n,cut,recipient,donor,track):
    h=n['states'][donor,cut].clone()
    if track=='retained':h[:,:7]=n['states'][recipient,cut,:,:7]
    return h


def exchange(net,n,cut,comparison,track):
    branches=(0,1) if comparison==0 else (0,2)
    results=[];states=[];adapted=[]
    for recipient in range(2):
        rr=[];ss=[];tt=[]
        for donor in range(2):
            h=initial(n,cut,branches[recipient],branches[donor],track)
            reg,tail=continuation(net,h,cut)
            policies=(n['tails'][branches[0],2*cut:],n['tails'][branches[1],2*cut:],tail)
            finals=[frozen(h,t) for t in policies]
            rr.append(torch.stack([decode(net,z) for z in finals]));ss.append(torch.stack(finals));tt.append(tail)
            if recipient==donor:assert torch.equal(rr[-1][2],n['logits'][branches[donor],-1])
            if track=='whole':assert torch.equal(rr[-1][2],n['logits'][branches[donor],-1])
        results.append(torch.stack(rr));states.append(torch.stack(ss));adapted.append(torch.stack(tt))
    return dict(logits=torch.stack(results),finals=torch.stack(states),tails=torch.stack(adapted))
