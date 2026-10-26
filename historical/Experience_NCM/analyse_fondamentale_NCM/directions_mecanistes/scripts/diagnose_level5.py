"""Frozen tokenizer geometry and controlled corruptions; no NCM retraining."""
from base import *
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch.nn.functional as F
threadpool_limits(1)

def variance(x):
    x=np.asarray(x,dtype=float)
    return dict(rms=float(np.sqrt(np.mean(x*x))),inter_puzzle_variance=float(x.var(axis=0).mean()),
        inter_puzzle_rms=float(np.sqrt(x.var(axis=0).mean())),position_mean_variance=float(x.mean(0).var(axis=0).mean()))

def utilization(code):
    c=np.bincount(np.asarray(code).ravel(),minlength=16);p=c/c.sum();h=-(p[p>0]*np.log(p[p>0])).sum()
    return dict(counts=c,used=int((c>0).sum()),entropy=float(h),perplexity=float(np.exp(h)))

@torch.inference_mode()
def diagnose(model,solutions,full=True):
    vq=model.vqvae.vq;z=model.vqvae._encode_pre_quant(torch.tensor(solutions));rest=z.clone();codes=[];levels=[];capture={}
    for k,length in enumerate(vq.levels):
        u=vq._down_for_quant(rest,k,length) if k<11 else rest
        flat=u.reshape(-1,256);dist=vq._distances(flat,vq.embeddings[k].weight);idx=dist.argmin(1).reshape(len(z),length)
        codes.append(idx);emb=vq.embeddings[k](idx)
        if k<11:emb=vq._up_to_full(emb,k,89)
        contribution=vq._apply_quant_resi(emb,k);after=rest-contribution
        if full:
            us=variance(u.numpy());rs=variance(rest.numpy());cs=variance(contribution.numpy());sorted_d=dist.sort(1).values
            selected=np.random.default_rng(Q['seed']+k).choice(len(flat),min(256,len(flat)),replace=False)
            explicit=((flat[selected].double()[:,None]-vq.embeddings[k].weight.double()[None])**2).sum(-1)
            fast=dist[selected].argmin(1);direct=explicit.argmin(1);norm=vq.embeddings[k].weight.norm(dim=1).numpy();ema=getattr(vq,f'ema_cluster_size_{k}').numpy()
            levels.append(dict(level=k+1,length=int(length),input=us,residual_before=rs,contribution=cs,
              pooling_inter_variance_ratio=us['inter_puzzle_variance']/max(rs['inter_puzzle_variance'],1e-30),
              input_mse_to_selected=float(dist.gather(1,idx.reshape(-1,1)).mean()/256),
              mse_residual_before=float(rest.square().mean()),mse_residual_after=float(after.square().mean()),
              normalized_assignment_margin_quantiles=np.quantile(((sorted_d[:,1]-sorted_d[:,0])/(flat.square().sum(1)+1e-12)).numpy(),[0,.1,.5,.9,1]),
              explicit_distance_samples=len(selected),direct_float64_assignment_disagreements=int((fast!=direct).sum()),
              max_distance_relative_difference=float(((dist[selected].double()-explicit).abs()/(explicit.abs()+1e-12)).max()),
              embedding_norms=norm,ema_cluster_sizes=ema,ema_active=int((ema>1e-8).sum()),
              codes=utilization(idx.numpy())))
        if k==4:
            capture=dict(z=z.numpy(),rest_before5=rest.numpy().copy(),u5=u.numpy().copy(),codes5=idx.numpy().copy())
            if not full:break
        rest=after
    if full:
        encoded=model.vqvae.encode(torch.tensor(solutions));assert torch.equal(torch.cat(codes,1),encoded)
    return levels,capture

@torch.inference_mode()
def main():
    train=np.array(SPLIT['train'])[:Q['tokenizer_train_n']];test=np.array(SPLIT['test'])[:Q['tokenizer_test_n']];report={};cache={}
    for d in ['hard','easy','medium']:
        model=SudokuModel(ROOT/f'artifacts/sudoku/{d}');_,s=boards(d)
        lev,tr=diagnose(model,s[train]);_,te=diagnose(model,s[test]);x=tr['u5'].reshape(-1,256).astype(float);xx=te['u5'].reshape(-1,256).astype(float)
        km=KMeans(n_clusters=16,n_init=5,max_iter=200,random_state=Q['seed']).fit(x)
        centers=km.cluster_centers_;labels=km.predict(xx);mse_km=float(((xx-centers[labels])**2).mean())
        book=model.vqvae.vq.embeddings[4].weight.numpy().astype(float);dists=((xx[:,None]-book[None])**2).sum(-1);oldlab=dists.argmin(1);mse_old=float(dists.min(1).mean()/256)
        cf={}
        for length in [8,9,10,18]:
            pooled=F.interpolate(torch.tensor(tr['rest_before5']).transpose(1,2),size=length,mode='area').transpose(1,2).numpy()
            nearest=((pooled.reshape(-1,256).astype(float)[:,None]-book[None])**2).sum(-1).argmin(1)
            cf[str(length)]=dict(variance=variance(pooled),codes=utilization(nearest))
        rows=tr['rest_before5'][:,CELL].reshape(len(train),9,9,256).mean(2)
        cf['exact_digit_rows']=dict(variance=variance(rows),note='Excludes separators; diagnostic averaging, not the native adaptive pooling.')
        report[d]=dict(train_indices=train,test_indices=test,levels=lev,test_level5_codes=utilization(oldlab),
           quantizer_refit=dict(test_mse_original=mse_old,test_mse_kmeans=mse_km,relative_mse_reduction=1-mse_km/max(mse_old,1e-30),
             test_usage=utilization(labels),training='Only a local k-means probe; decoder/generator unchanged, no reconstructed-quality claim.'),
           pooling_counterfactuals=cf,restart_threshold=float(model.vqvae.vq.restart_threshold))
        cache.update({d+'_'+key:value for key,value in tr.items()})
        log('tokenizer_diagnostic',checkpoint=d,used5=lev[4]['codes']['used'],variance5=lev[4]['input']['inter_puzzle_variance'],refit=report[d]['quantizer_refit'])
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');_,s=boards('hard');cells=s[train[:Q['controls_n']]][:,CELL];rng=np.random.default_rng(Q['seed']);variants={'original':cells.copy()}
    digits=[];rowperm=[];globalperm=[]
    for c in cells:
        mapping=np.r_[0,rng.permutation(np.arange(1,10))];digits.append(mapping[c])
        rowperm.append(np.stack([rng.permutation(row) for row in c.reshape(9,9)]).ravel());globalperm.append(rng.permutation(c))
    variants.update(valid_digit_permutation=np.array(digits),within_rows_invalid=np.array(rowperm),global_shuffle_invalid=np.array(globalperm),iid_digits_invalid=rng.integers(1,10,size=cells.shape))
    controls={}
    for name,c in variants.items():
        _,state=diagnose(model,tokens(c),full=False)
        raw9=F.interpolate(torch.tensor(state['z']).transpose(1,2),size=9,mode='area').transpose(1,2).numpy()
        controls[name]=dict(codes5=utilization(state['codes5']),u5=variance(state['u5']),raw_encoder_pooled9=variance(raw9))
        cache[name+'_u5']=state['u5'];log('pooling_control',condition=name,used5=controls[name]['codes5']['used'])
    report['controls_hard']=controls
    write(D/'results/level5_diagnostic.json',report);np.savez_compressed(D/'results/level5_geometry.npz',**cache)
    log('level5_diagnostic_complete')

if __name__=='__main__':main()
