from model import *
import math


def train():
    check();assert not (D/'checkpoints').exists(),'Refusing to replace training'
    (D/'checkpoints').mkdir()
    data=np.load(D/'data/train.npz');x=torch.tensor(data['x']);y=torch.tensor(data['y'])
    records=[];start=time.monotonic()
    for architecture in Q['architectures']:
        for seed in Q['seeds']:
            torch.manual_seed(seed);net=Net(architecture).train()
            initial=parameter_hash(net);rng=np.random.default_rng(seed)
            opt=torch.optim.AdamW(net.parameters(),lr=.002,betas=(.9,.99),eps=1e-8,weight_decay=.01)
            losses=[];batch_hash=hashlib.sha256()
            for step in range(Q['steps']):
                indices=rng.integers(len(x),size=Q['batch_size']);batch_hash.update(indices.tobytes())
                bx,by=x[indices],y[indices]
                queries=torch.cat((torch.full((len(bx),1),14),by[:,:3]),dim=1) if architecture=='ar' else fixed_queries(len(bx))
                pred=net(torch.cat((bx,queries),dim=1))
                loss=F.cross_entropy(pred.reshape(-1,8),by.reshape(-1))
                scale=(step+1)/50 if step<50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/(Q['steps']-50)))
                opt.param_groups[0]['lr']=.002*scale
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
                losses.append(float(loss.detach()))
                if (step+1)%200==0:print(architecture,seed,'step',step+1,'loss',losses[-1],flush=True)
            path=D/f'checkpoints/{architecture}-{seed}.pt';torch.save(net.state_dict(),path)
            records.append(dict(architecture=architecture,seed=seed,initial_parameter_sha256=initial,
                 final_parameter_sha256=parameter_hash(net),checkpoint_sha256=sha(path),
                 batch_indices_sha256=batch_hash.hexdigest(),parameters=sum(p.numel() for p in net.parameters()),
                 first_loss=losses[0],last_loss=losses[-1],all_losses=losses))
            write('results/training.json',dict(status='partial',models=records,elapsed_seconds=time.monotonic()-start))
    for seed in Q['seeds']:
        pair=[r for r in records if r['seed']==seed]
        assert len({r['initial_parameter_sha256'] for r in pair})==1
        assert len({r['batch_indices_sha256'] for r in pair})==1
    write('results/training.json',dict(status='complete',models=records,elapsed_seconds=time.monotonic()-start))
    check()


if __name__=='__main__':train()
