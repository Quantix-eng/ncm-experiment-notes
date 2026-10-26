"""Post-hoc numerical incident isolation; not a scientific effect test."""
from support import *
from runtime import Engine
if __name__=='__main__':
    setup();net,tok=load();engine=Engine(net);rows=read('data/selected.json')['dev']
    with torch.inference_mode():
        c=engine.native(rows[0]['tokens'][0]);d=engine.native(rows[37]['tokens'][0])
        a=d['heads',9][0,-1,576:640].clone();b=c['heads',9][0,-1,576:640].clone()
        w=net.transformer.h[9].attn.c_proj.weight[576:640].clone()
        raw=a-b;bad=raw@w;corrected=(a.double()-b.double())@w.double()
        an,bn,wn=a.numpy().astype(np.float64),b.numpy().astype(np.float64),w.numpy().astype(np.float64)
        reference=np.einsum('i,ij->j',an-bn,wn,optimize=False)
        scalar=np.array([sum((float(an[i])-float(bn[i]))*float(wn[i,j]) for i in range(64)) for j in range(768)])
        error=float(np.max(np.abs(corrected.numpy()-reference)))
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(w).all()
        assert bool(torch.isfinite(corrected).all()) and error<1e-12
        np.savez_compressed(D/'results/projection_incident.npz',donor=a.numpy(),recipient=b.numpy(),
            weight=w.numpy(),fp32=bad.numpy(),fp64=corrected.numpy(),numpy_reference=reference,scalar=scalar)
        write('results/projection_incident.json',dict(time_utc=now(),inputs_finite=True,
            n_nan_fp32=int(torch.isnan(bad).sum()),n_nonfinite_fp64=int((~torch.isfinite(corrected)).sum()),
            fp64_vs_numpy_max_error=error,fp64_vs_scalar_max_error=float(np.max(np.abs(corrected.numpy()-scalar))),
            interpretation='local numerical defect in FP32 vector/matrix measurement; network uses full official projection',
            scientific_thresholds_changed=False))
        print(read('results/projection_incident.json'))
