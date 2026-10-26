"""Group-weighted correction probes and cell-local PCA, no held-out fitting."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import Standardizer, auc, metrics
from study_v03 import ObservableScaler


def puzzle_weights(groups):
    _,inverse,count=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1/count[inverse]
    return weights/weights.sum()


def grouped_metrics(y,score,group):
    per_group=[]
    for g in np.unique(group):
        rows=group==g
        if y[rows].min()!=y[rows].max():
            per_group.append(dict(group=int(g),n=int(rows.sum()),persistent=int(y[rows].sum()),auc=auc(y[rows],score[rows])))
    result=metrics(y,score)
    w=puzzle_weights(group)
    probability=1/(1+np.exp(-np.clip(score,-700,700)))
    result.update(within_puzzle_auc=float(np.mean([r['auc']for r in per_group])) if per_group else None,
        mixed_groups=len(per_group),weighted_log_loss=float(np.sum(w*(np.logaddexp(0,score)-y*score))),
        weighted_brier=float(np.sum(w*(probability-y)**2)),per_group=per_group)
    return result


class CellPCA:
    def fit(self,x):
        self.scaler=Standardizer().fit(x.astype(np.float64))
        z=torch.from_numpy(self.scaler.transform(x.astype(np.float64)))
        _,vectors=torch.linalg.eigh(z.T@z/len(z))
        self.axes=vectors[:,-16:].flip(1).T.numpy().copy()
        for axis in self.axes:
            if axis[np.argmax(abs(axis))]<0:axis*=-1
        self.score_scaler=Standardizer().fit(z.numpy()@self.axes.T)
        assert self.score_scaler.keep.all()
        return self
    def transform(self,x):
        return self.score_scaler.transform(self.scaler.transform(x.astype(np.float64))@self.axes.T)
    def arrays(self):
        return dict(hidden_mean=self.scaler.mean,hidden_keep=self.scaler.keep,hidden_scale=self.scaler.scale,
            pca_axes=self.axes,pca_mean=self.score_scaler.mean,pca_scale=self.score_scaler.scale)


def fit_logistic(x,y,group,strength,settings):
    tx=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float64));ty=torch.from_numpy(y.astype(np.float64))
    weight=torch.from_numpy(puzzle_weights(group))
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    b=torch.tensor(float(torch.logit((weight*ty).sum())),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,b],**settings)
    def closure():
        optimizer.zero_grad()
        loss=(weight*F.binary_cross_entropy_with_logits(tx@w+b,ty,reduction='none')).sum()+.5*strength*w.square().sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction objective')
        loss.backward();return loss
    optimizer.step(closure);loss=closure()
    iterations=int(optimizer.state[w].get('n_iter',0))
    if iterations>=settings['max_iter']:raise RuntimeError('Preset LBFGS cap reached, no automatic tuning')
    return dict(weight=w.detach().numpy(),intercept=float(b.detach()),iterations=iterations,
        max_abs_gradient=float(torch.cat([w.grad,b.grad.reshape(1)]).abs().max()),objective=float(loss.detach()))


def controls(data):
    noise=np.empty((len(data['group']),16),dtype=np.float64)
    shuffle=np.arange(len(noise))
    for g in np.unique(data['group']):
        rows=np.flatnonzero(data['group']==g)
        assert len(rows)==51
        noise[rows]=np.random.default_rng(np.random.SeedSequence([20261104,int(g)])).standard_normal((81,16))[data['cell'][rows]]
        shuffle[rows]=np.random.default_rng(np.random.SeedSequence([20261103,int(g)])).permutation(rows)
    return noise,shuffle


def prepare(data,train_groups,valid_groups,noise,shuffle):
    t_all=np.isin(data['group'],train_groups);v_all=np.isin(data['group'],valid_groups)
    assert not (t_all&v_all).any()
    t=np.flatnonzero(t_all&data['current_error']);v=np.flatnonzero(v_all&data['current_error'])
    scaler=ObservableScaler().fit(np.asarray(data['visible'][t]))
    a,b=scaler.transform(data['visible'][t]),scaler.transform(data['visible'][v])
    pca=CellPCA().fit(data['hidden'][t_all])
    ht,hv=pca.transform(data['hidden'][t]),pca.transform(data['hidden'][v])
    sh_t,sh_v=pca.transform(data['hidden'][shuffle[t]]),pca.transform(data['hidden'][shuffle[v]])
    ns=Standardizer().fit(noise[t])
    base=dict(base_mean=scaler.mean,base_keep=scaler.keep,base_scale=scaler.scale)
    out=dict(visible=(a,b,base),visible_hidden=(np.column_stack([a,ht]),np.column_stack([b,hv]),dict(**base,**pca.arrays())),
        visible_noise=(np.column_stack([a,ns.transform(noise[t])]),np.column_stack([b,ns.transform(noise[v])]),
                       dict(**base,noise_mean=ns.mean,noise_keep=ns.keep,noise_scale=ns.scale)),
        visible_shuffle=(np.column_stack([a,sh_t]),np.column_stack([b,sh_v]),dict(**base,**pca.arrays())))
    return out,t,v
