"""Post-hoc exploratory test: remove a constant level's residual contribution."""
from common import *

@torch.inference_mode()
def main():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard'); a=np.load(ROOT/'results/hard.npz');split=json.loads((ROOT/'results/splits.json').read_text())
    ix=np.array(split['test'])[:128];p=a['prefix'][ix];s=a['reference'][ix];original=a['decoded'][ix,-1]
    vq=model.vqvae.vq; apply=vq._apply_quant_resi; conditions={};outputs={};norms=[]
    for name in ['sham','remove_level5_residual']:
        def intervene(h,k):
            result=apply(h,k)
            if name=='remove_level5_residual' and k==4:
                norms.append(float(result.square().mean().sqrt()));return torch.zeros_like(result)
            return result
        vq._apply_quant_resi=intervene
        pred=[]
        try:
            for lo in range(0,len(p),16):pred.append(model.predict(torch.tensor(p[lo:lo+16])).ids.numpy())
        finally:vq._apply_quant_resi=apply
        pred=np.concatenate(pred);outputs[name]=pred; y=np.all(pred==s,axis=1);base=np.all(original==s,axis=1);diff=y.astype(float)-base
        rng=np.random.default_rng(P['seed']); boot=diff[rng.integers(0,len(diff),(P['bootstrap'],len(diff)))].mean(1)
        conditions[name]=dict(metrics=evaluate(pred,s,p),changed_boards=int(np.any(pred!=original,axis=1).sum()),
           introduced_errors=int((base&~y).sum()),rescued=int((~base&y).sum()),
           success_status_changes=int((base!=y).sum()),success_status_change_wilson95=ci_rate(int((base!=y).sum()),len(base)),delta_solved=float(diff.mean()),delta_ci95=np.quantile(boot,[.025,.975]))
        progress('constant_level_ablation',condition=name,**conditions[name]['metrics'])
    assert np.array_equal(outputs['sham'],original)
    np.savez_compressed(ROOT/'results/constant_level_predictions.npz',indices=ix,reference=s,baseline=original,**outputs)
    write(ROOT/'results/constant_level.json',dict(status='exploratory_posthoc',n=128,level=5,conditions=conditions,residual_rms_per_batch=norms,
      intervention='Replace the full residual contribution of VQ level 5 with zero before generation of finer levels; regenerate all downstream codes',
      caveat='Selected after observing constant VQ codes; tests contribution of a common learned offset, not sample-specific information or metacognitive control.'))

if __name__=='__main__':main()
